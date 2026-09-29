#include "dtmb/core.hpp"

#include "binary_stdio.hpp"
#include "nr_stream.hpp"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFloatsPerSymbol = 2;

void usage(const char* program) {
    std::cerr
        << "usage: " << program << " [--qam 4qam-nr|4qam|16qam|32qam|64qam]"
        << " [--workers N] [--min-parallel-symbols N] [--chunk-symbols N]"
        << " [--noise-variance X] [--soft-demod-method max-log|log-sum-exp]"
        << " [input.cf32|-] [output.llr.f32|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

float parse_float(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stof(text, &parsed);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

dtmb::core::QamSoftDemapMethod parse_soft_demap_method(const std::string& text) {
    if (text == "max-log" || text == "maxlog") {
        return dtmb::core::QamSoftDemapMethod::max_log;
    }
    if (text == "log-sum-exp" || text == "logsumexp" || text == "exact") {
        return dtmb::core::QamSoftDemapMethod::log_sum_exp;
    }
    throw std::invalid_argument("invalid soft demod method: " + text);
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

void write_all(std::ostream& output, std::span<const float> values) {
    const auto bytes = static_cast<std::streamsize>(values.size() * sizeof(float));
    output.write(reinterpret_cast<const char*>(values.data()), bytes);
    if (!output) {
        throw std::runtime_error("failed to write output LLR stream");
    }
}

}  // namespace

int main(int argc, char** argv) {
    dtmb::core::QamSoftDemapOptions options{};
    auto qam_mode = dtmb::core::QamMode::qam64;
    std::size_t chunk_symbols = 1U << 16U;
    std::string input_path = "-";
    std::string output_path = "-";
    std::vector<std::string> positional;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--qam") {
                if (++index >= argc) throw std::invalid_argument("missing QAM mode");
                qam_mode = dtmb::core::parse_qam_mode(argv[index]);
            } else if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                options.requested_workers = parse_size(argv[index], "worker count");
            } else if (arg == "--min-parallel-symbols") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                options.min_parallel_symbols = parse_size(argv[index], "parallel threshold");
            } else if (arg == "--chunk-symbols") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                chunk_symbols = parse_size(argv[index], "chunk size");
            } else if (arg == "--noise-variance") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                options.noise_variance = parse_float(argv[index], "noise variance");
            } else if (arg == "--soft-demod-method") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                options.method = parse_soft_demap_method(argv[index]);
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                positional.push_back(arg);
            }
        }

        if (positional.size() > 2 || chunk_symbols == 0) {
            usage(argv[0]);
            return 2;
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }
        if (positional.size() == 2) {
            output_path = positional[1];
        }

        dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);

        if (qam_mode == dtmb::core::QamMode::qam4_nr)
            return dtmb::tools::run_nr_demapper(input, output, options, chunk_symbols);

        std::vector<float> input_chunk(chunk_symbols * kFloatsPerSymbol);
        std::vector<float> output_chunk(chunk_symbols * dtmb::core::qam_definition(qam_mode).bits_per_symbol);
        const auto input_chunk_bytes = static_cast<std::streamsize>(
            input_chunk.size() * sizeof(float));

        while (true) {
            input.read(reinterpret_cast<char*>(input_chunk.data()), input_chunk_bytes);
            const auto bytes_read = input.gcount();
            if (bytes_read == 0) {
                break;
            }
            if ((bytes_read % static_cast<std::streamsize>(sizeof(float) * kFloatsPerSymbol)) != 0) {
                throw std::runtime_error("CF32 input byte count is not a whole number of symbols");
            }

            const auto symbols_read = static_cast<std::size_t>(bytes_read)
                / (sizeof(float) * kFloatsPerSymbol);
            const auto input_values = std::span<const float>(
                input_chunk.data(),
                symbols_read * kFloatsPerSymbol);
            auto output_values = std::span<float>(
                output_chunk.data(),
                symbols_read * dtmb::core::qam_definition(qam_mode).bits_per_symbol);
            dtmb::core::qam_soft_demodulate_cf32(input_values, output_values, qam_mode, options);
            write_all(output, output_values);

            if (bytes_read < input_chunk_bytes) {
                break;
            }
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_qam64_softdemap: " << exc.what() << '\n';
        return 1;
    }
}
