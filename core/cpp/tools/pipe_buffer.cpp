#include "binary_stdio.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::size_t buffer_bytes = 64U * 1024U * 1024U;
    std::size_t chunk_bytes = 1U * 1024U * 1024U;
    std::size_t prefill_bytes = 0;
    std::string diagnostics_path = "-";
    std::string input_path = "-";
    std::string output_path = "-";
};

struct Chunk {
    std::vector<char> bytes;
};

struct SharedState {
    std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    std::deque<Chunk> queue;
    std::size_t buffered_bytes = 0;
    std::size_t max_buffered_bytes = 0;
    std::size_t prefill_released_bytes = 0;
    bool reader_done = false;
    bool reader_blocked_full = false;
    bool prefill_released = false;
    bool stop = false;
    std::string error;
    std::string prefill_release_reason = "not-released";
};

struct Counters {
    std::uint64_t bytes_in = 0;
    std::uint64_t bytes_out = 0;
    std::uint64_t chunks_in = 0;
    std::uint64_t chunks_out = 0;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " [--buffer-bytes N] [--chunk-bytes N] [--prefill-bytes N]"
        << " [--diagnostics-out path|-] [input|-] [output|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

std::istream& input_stream(
    const std::string& path,
    std::unique_ptr<std::ifstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cin;
    }
    file_holder = std::make_unique<std::ifstream>(path, std::ios::binary);
    if (!*file_holder) {
        throw std::runtime_error("failed to open input: " + path);
    }
    return *file_holder;
}

std::ostream& output_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cout;
    }
    file_holder = std::make_unique<std::ofstream>(path, std::ios::binary);
    if (!*file_holder) {
        throw std::runtime_error("failed to open output: " + path);
    }
    return *file_holder;
}

std::ostream& diagnostics_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cerr;
    }
    file_holder = std::make_unique<std::ofstream>(path, std::ios::out | std::ios::trunc);
    if (!*file_holder) {
        throw std::runtime_error("failed to open diagnostics output: " + path);
    }
    return *file_holder;
}

std::string json_escape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() + 8);
    for (const char ch : text) {
        switch (ch) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += ch;
            break;
        }
    }
    return escaped;
}

void set_error(SharedState& state, std::string message) {
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.error.empty()) {
            state.error = std::move(message);
        }
        state.stop = true;
        state.reader_done = true;
    }
    state.not_empty.notify_all();
    state.not_full.notify_all();
}

void reader_thread(
    std::istream& input,
    const Options& options,
    SharedState& state,
    Counters& counters) {
    try {
        std::vector<char> scratch(options.chunk_bytes);
        while (true) {
            input.read(scratch.data(), static_cast<std::streamsize>(scratch.size()));
            const auto got = input.gcount();
            if (got <= 0) {
                if (input.bad()) {
                    throw std::runtime_error("failed to read input stream");
                }
                break;
            }

            auto chunk_size = static_cast<std::size_t>(got);
            Chunk chunk;
            chunk.bytes.assign(scratch.begin(), scratch.begin() + got);

            {
                std::unique_lock<std::mutex> lock(state.mutex);
                while (!state.stop
                       && state.buffered_bytes + chunk_size > options.buffer_bytes) {
                    state.reader_blocked_full = true;
                    state.not_empty.notify_one();
                    state.not_full.wait(lock, [&]() {
                        return state.stop
                            || state.buffered_bytes + chunk_size <= options.buffer_bytes;
                    });
                }
                state.reader_blocked_full = false;
                if (state.stop) {
                    break;
                }
                state.buffered_bytes += chunk_size;
                if (state.buffered_bytes > state.max_buffered_bytes) {
                    state.max_buffered_bytes = state.buffered_bytes;
                }
                state.queue.push_back(std::move(chunk));
                counters.bytes_in += chunk_size;
                ++counters.chunks_in;
            }
            state.not_empty.notify_one();

            if (static_cast<std::size_t>(got) < scratch.size()) {
                if (input.bad()) {
                    throw std::runtime_error("failed to read input stream");
                }
                break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.reader_done = true;
            state.reader_blocked_full = false;
        }
        state.not_empty.notify_all();
    } catch (const std::exception& exc) {
        set_error(state, exc.what());
    }
}

void release_prefill_locked(const Options& options, SharedState& state) {
    if (state.prefill_released) {
        return;
    }
    state.prefill_released = true;
    state.prefill_released_bytes = state.buffered_bytes;
    if (options.prefill_bytes == 0) {
        state.prefill_release_reason = "disabled";
    } else if (state.buffered_bytes >= options.prefill_bytes) {
        state.prefill_release_reason = "threshold";
    } else if (state.reader_done) {
        state.prefill_release_reason = "eof";
    } else if (state.reader_blocked_full) {
        state.prefill_release_reason = "capacity";
    } else if (state.stop) {
        state.prefill_release_reason = "stop";
    } else {
        state.prefill_release_reason = "unknown";
    }
}

void writer_thread(
    std::ostream& output,
    const Options& options,
    SharedState& state,
    Counters& counters) {
    try {
        while (true) {
            Chunk chunk;
            {
                std::unique_lock<std::mutex> lock(state.mutex);
                if (!state.prefill_released) {
                    state.not_empty.wait(lock, [&]() {
                        return state.stop
                            || state.reader_done
                            || state.reader_blocked_full
                            || options.prefill_bytes == 0
                            || state.buffered_bytes >= options.prefill_bytes;
                    });
                    release_prefill_locked(options, state);
                }
                state.not_empty.wait(lock, [&]() {
                    return state.stop || !state.queue.empty() || state.reader_done;
                });
                if (state.queue.empty()) {
                    break;
                }
                chunk = std::move(state.queue.front());
                state.queue.pop_front();
                state.buffered_bytes -= chunk.bytes.size();
            }
            state.not_full.notify_one();

            output.write(chunk.bytes.data(), static_cast<std::streamsize>(chunk.bytes.size()));
            if (!output) {
                throw std::runtime_error("failed to write output stream");
            }
            counters.bytes_out += chunk.bytes.size();
            ++counters.chunks_out;
        }
    } catch (const std::exception& exc) {
        set_error(state, exc.what());
    }
}

void write_report(
    std::ostream& output,
    const Options& options,
    const SharedState& state,
    const Counters& counters,
    double duration_s) {
    const auto verdict = state.error.empty() ? "ok" : "error";
    output << std::setprecision(9)
           << "{"
           << "\"schema\":\"dtmb.pipe_buffer.v1\","
           << "\"verdict\":\"" << verdict << "\","
           << "\"input\":\"" << json_escape(options.input_path) << "\","
           << "\"output\":\"" << json_escape(options.output_path) << "\","
           << "\"buffer_bytes\":" << options.buffer_bytes << ","
           << "\"chunk_bytes\":" << options.chunk_bytes << ","
           << "\"prefill_bytes\":" << options.prefill_bytes << ","
           << "\"prefill_released\":" << (state.prefill_released ? "true" : "false") << ","
           << "\"prefill_released_bytes\":" << state.prefill_released_bytes << ","
           << "\"prefill_reached\":"
           << (state.prefill_released_bytes >= options.prefill_bytes ? "true" : "false") << ","
           << "\"prefill_release_reason\":\""
           << json_escape(state.prefill_release_reason) << "\","
           << "\"bytes_in\":" << counters.bytes_in << ","
           << "\"bytes_out\":" << counters.bytes_out << ","
           << "\"chunks_in\":" << counters.chunks_in << ","
           << "\"chunks_out\":" << counters.chunks_out << ","
           << "\"max_buffered_bytes\":" << state.max_buffered_bytes << ","
           << "\"duration_s\":" << duration_s;
    if (!state.error.empty()) {
        output << ",\"error\":\"" << json_escape(state.error) << "\"";
    }
    output << "}\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    bool input_set = false;
    bool output_set = false;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--buffer-bytes") {
            if (++index >= argc) {
                throw std::invalid_argument("--buffer-bytes requires a value");
            }
            options.buffer_bytes = parse_size(argv[index], "buffer size");
        } else if (arg == "--chunk-bytes") {
            if (++index >= argc) {
                throw std::invalid_argument("--chunk-bytes requires a value");
            }
            options.chunk_bytes = parse_size(argv[index], "chunk size");
        } else if (arg == "--prefill-bytes") {
            if (++index >= argc) {
                throw std::invalid_argument("--prefill-bytes requires a value");
            }
            options.prefill_bytes = parse_size(argv[index], "prefill size");
        } else if (arg == "--diagnostics-out") {
            if (++index >= argc) {
                throw std::invalid_argument("--diagnostics-out requires a value");
            }
            options.diagnostics_path = argv[index];
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (!input_set) {
            options.input_path = arg;
            input_set = true;
        } else if (!output_set) {
            options.output_path = arg;
            output_set = true;
        } else {
            throw std::invalid_argument("too many positional arguments");
        }
    }
    if (options.buffer_bytes == 0) {
        throw std::invalid_argument("--buffer-bytes must be non-zero");
    }
    if (options.chunk_bytes == 0) {
        throw std::invalid_argument("--chunk-bytes must be non-zero");
    }
    if (options.chunk_bytes > options.buffer_bytes) {
        throw std::invalid_argument("--chunk-bytes must be <= --buffer-bytes");
    }
    if (options.prefill_bytes > options.buffer_bytes) {
        throw std::invalid_argument("--prefill-bytes must be <= --buffer-bytes");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        dtmb::tools::configure_binary_stdio(options.input_path == "-", options.output_path == "-");

        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        std::unique_ptr<std::ofstream> diagnostics_file;
        auto& input = input_stream(options.input_path, input_file);
        auto& output = output_stream(options.output_path, output_file);

        SharedState state;
        Counters counters;
        const auto start = std::chrono::steady_clock::now();
        std::thread reader(reader_thread, std::ref(input), std::cref(options), std::ref(state), std::ref(counters));
        std::thread writer(writer_thread, std::ref(output), std::cref(options), std::ref(state), std::ref(counters));
        reader.join();
        writer.join();
        const auto finish = std::chrono::steady_clock::now();
        const auto duration_s = std::chrono::duration<double>(finish - start).count();

        output.flush();
        if (!output) {
            set_error(state, "failed to flush output stream");
        }

        auto& diagnostics = diagnostics_stream(options.diagnostics_path, diagnostics_file);
        write_report(diagnostics, options, state, counters, duration_s);
        return state.error.empty() ? 0 : 1;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_pipe_buffer: " << exc.what() << '\n';
        usage(argc > 0 ? argv[0] : "dtmb_core_pipe_buffer");
        return 2;
    }
}
