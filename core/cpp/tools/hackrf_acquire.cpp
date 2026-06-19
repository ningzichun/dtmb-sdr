// dtmb_core_hackrf_acquire: Bounded HackRF CI8 acquisition to stdout or disk
//
// Decouples HackRF sample acquisition from downstream DSP processing by using
// a bounded ring buffer. When the buffer fills (consumer too slow), drops
// samples rather than blocking the HackRF transfer callback.
//
// Usage:
//   dtmb_core_hackrf_acquire --frequency Hz --sample-rate sps \
//       [--bandwidth Hz] [--duration s | --samples N] \
//       [--amp 0|1] [--lna-gain dB] [--vga-gain dB] \
//       [--serial SERIAL] [--output path.ci8 | -]
//
// Outputs:
//   stdout or --output file: raw CI8 bytes (int8 I, int8 Q)
//   stderr: NDJSON telemetry and final stats
//
// Features:
//   - Bounded buffer (default 64 MB) prevents HackRF callback blocking
//   - Drops samples when consumer falls behind (reports dropped_bytes)
//   - Can write to file or stdout (for piping to native processing chain)
//   - Real-time throughput monitoring
//   - Graceful shutdown on SIGINT/SIGTERM

#include "dtmb/core.hpp"
#include "binary_stdio.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef DTMB_CORE_HAVE_HACKRF
#include <hackrf.h>
#endif

namespace {

volatile std::sig_atomic_t stop_signal = 0;

void handle_stop_signal(int signal_number) {
    stop_signal = signal_number;
}

bool stop_requested() noexcept {
    return stop_signal != 0;
}

std::string_view signal_stop_reason() noexcept {
    if (stop_signal == SIGINT) {
        return "sigint";
    }
#ifdef SIGTERM
    if (stop_signal == SIGTERM) {
        return "sigterm";
    }
#endif
    return "signal";
}

class StopSignalSession {
public:
    StopSignalSession() {
        stop_signal = 0;
        previous_sigint_ = std::signal(SIGINT, handle_stop_signal);
#ifdef SIGTERM
        previous_sigterm_ = std::signal(SIGTERM, handle_stop_signal);
#endif
    }

    StopSignalSession(const StopSignalSession&) = delete;
    StopSignalSession& operator=(const StopSignalSession&) = delete;

    ~StopSignalSession() {
        if (previous_sigint_ != SIG_ERR) {
            (void) std::signal(SIGINT, previous_sigint_);
        }
#ifdef SIGTERM
        if (previous_sigterm_ != SIG_ERR) {
            (void) std::signal(SIGTERM, previous_sigterm_);
        }
#endif
    }

private:
    using SignalHandler = void (*)(int);
    SignalHandler previous_sigint_ = SIG_ERR;
#ifdef SIGTERM
    SignalHandler previous_sigterm_ = SIG_ERR;
#endif
};

struct Options {
    std::uint64_t frequency_hz = 0;
    std::size_t sample_rate_sps = dtmb::core::kDtmbSymbolRateSps;
    std::size_t bandwidth_hz = 0;
    double duration_s = 0.0;
    std::size_t max_samples = 0;
    int amp_enable = 0;
    std::uint32_t lna_gain = 16;
    std::uint32_t vga_gain = 16;
    std::string serial;
    std::string output_path = "-";
    std::size_t max_queued_bytes = 64U * 1024U * 1024U;  // 64 MB
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program << "\n"
        << "  --frequency HZ\n"
        << "  --sample-rate SPS (default: " << dtmb::core::kDtmbSymbolRateSps << ")\n"
        << "  [--bandwidth HZ]\n"
        << "  [--duration SECONDS | --samples N]\n"
        << "  [--amp 0|1] (default: 0)\n"
        << "  [--lna-gain DB] (default: 16)\n"
        << "  [--vga-gain DB] (default: 16)\n"
        << "  [--serial SERIAL]\n"
        << "  [--output PATH|-] (default: stdout)\n"
        << "  [--max-queued-bytes N] (default: 67108864)\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size() || value == 0) {
        throw std::invalid_argument(std::string(field) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t parse_u64(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

double parse_double(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stod(text, &parsed);
    if (parsed != text.size() || value <= 0.0) {
        throw std::invalid_argument(std::string(field) + " must be positive");
    }
    return value;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto need_value = [&](const char* field) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + field);
            }
            return argv[index];
        };

        if (arg == "--frequency") {
            options.frequency_hz = parse_u64(need_value("frequency"), "frequency");
        } else if (arg == "--sample-rate") {
            options.sample_rate_sps = parse_size(need_value("sample-rate"), "sample-rate");
        } else if (arg == "--bandwidth") {
            options.bandwidth_hz = parse_size(need_value("bandwidth"), "bandwidth");
        } else if (arg == "--duration") {
            options.duration_s = parse_double(need_value("duration"), "duration");
        } else if (arg == "--samples") {
            options.max_samples = parse_size(need_value("samples"), "samples");
        } else if (arg == "--amp") {
            options.amp_enable = std::stoi(need_value("amp"));
        } else if (arg == "--lna-gain") {
            options.lna_gain = static_cast<std::uint32_t>(std::stoul(need_value("lna-gain")));
        } else if (arg == "--vga-gain") {
            options.vga_gain = static_cast<std::uint32_t>(std::stoul(need_value("vga-gain")));
        } else if (arg == "--serial") {
            options.serial = need_value("serial");
        } else if (arg == "--output") {
            options.output_path = need_value("output");
        } else if (arg == "--max-queued-bytes") {
            options.max_queued_bytes = parse_size(need_value("max-queued-bytes"), "max-queued-bytes");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.frequency_hz == 0) {
        throw std::invalid_argument("--frequency is required");
    }

    if (options.duration_s > 0.0 && options.max_samples == 0) {
        options.max_samples = static_cast<std::size_t>(
            options.duration_s * static_cast<double>(options.sample_rate_sps));
    }

    return options;
}

#ifdef DTMB_CORE_HAVE_HACKRF

struct HackrfRxQueue {
    using Clock = std::chrono::steady_clock;

    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::vector<std::int8_t>> chunks;
    std::size_t queued_bytes = 0;
    std::size_t max_observed_queued_bytes = 0;
    std::atomic<std::size_t> dropped_bytes = 0;
    std::size_t max_queued_bytes = 64U * 1024U * 1024U;
    std::atomic<bool> stop_requested = false;
    std::atomic<std::size_t> samples_seen = 0;
    Clock::time_point capture_start{};
    Clock::time_point capture_finish{};
    bool capture_started = false;
    bool capture_finished = false;
    bool finished = false;
    std::size_t max_samples = 0;
};

void check_hackrf_result(int result, const char* call_name) {
    if (result == HACKRF_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << "HackRF error in " << call_name << ": "
            << hackrf_error_name(static_cast<hackrf_error>(result))
            << " (" << result << ")";
    throw std::runtime_error(message.str());
}

int hackrf_rx_callback(hackrf_transfer* transfer) {
    auto* queue = static_cast<HackrfRxQueue*>(transfer->rx_ctx);
    if (queue == nullptr || queue->stop_requested.load(std::memory_order_relaxed)) {
        return 1;
    }

    const auto callback_time = HackrfRxQueue::Clock::now();

    auto bytes_to_copy = static_cast<std::size_t>(std::max(transfer->valid_length, 0));
    bytes_to_copy -= bytes_to_copy % 2U;
    if (bytes_to_copy == 0) {
        return 0;
    }

    bool should_stop = false;
    if (queue->max_samples != 0) {
        const auto transfer_samples = bytes_to_copy / 2U;
        const auto prior_samples = queue->samples_seen.fetch_add(
            transfer_samples,
            std::memory_order_relaxed);
        if (prior_samples >= queue->max_samples) {
            bytes_to_copy = 0;
            should_stop = true;
        } else {
            const auto remaining_samples = queue->max_samples - prior_samples;
            if (transfer_samples >= remaining_samples) {
                bytes_to_copy = remaining_samples * 2U;
                should_stop = true;
            }
        }
    }

    {
        std::unique_lock<std::mutex> lock(queue->mutex);
        if (!queue->capture_started) {
            queue->capture_started = true;
            queue->capture_start = callback_time;
        }
        if (queue->queued_bytes + bytes_to_copy > queue->max_queued_bytes) {
            queue->dropped_bytes.fetch_add(bytes_to_copy, std::memory_order_relaxed);
        } else {
            std::vector<std::int8_t> chunk(bytes_to_copy);
            std::memcpy(chunk.data(), transfer->buffer, bytes_to_copy);
            queue->chunks.push_back(std::move(chunk));
            queue->queued_bytes += bytes_to_copy;
            queue->max_observed_queued_bytes =
                std::max(queue->max_observed_queued_bytes, queue->queued_bytes);
            queue->condition.notify_one();
        }
        if (should_stop && !queue->capture_finished) {
            queue->capture_finished = true;
            queue->capture_finish = callback_time;
        }
    }

    if (should_stop) {
        {
            std::lock_guard<std::mutex> lock(queue->mutex);
            queue->finished = true;
        }
        queue->condition.notify_all();
        return 1;
    }

    return 0;
}

class HackrfLibrarySession {
public:
    HackrfLibrarySession() {
        check_hackrf_result(hackrf_init(), "hackrf_init");
    }

    HackrfLibrarySession(const HackrfLibrarySession&) = delete;
    HackrfLibrarySession& operator=(const HackrfLibrarySession&) = delete;

    ~HackrfLibrarySession() {
        (void) hackrf_exit();
    }
};

class HackrfDeviceHandle {
public:
    explicit HackrfDeviceHandle(const std::string& serial) {
        int result = HACKRF_SUCCESS;
        if (serial.empty()) {
            result = hackrf_open(&device_);
        } else {
            result = hackrf_open_by_serial(serial.c_str(), &device_);
        }
        check_hackrf_result(result, serial.empty() ? "hackrf_open" : "hackrf_open_by_serial");
    }

    HackrfDeviceHandle(const HackrfDeviceHandle&) = delete;
    HackrfDeviceHandle& operator=(const HackrfDeviceHandle&) = delete;

    ~HackrfDeviceHandle() {
        if (device_ != nullptr) {
            (void) hackrf_close(device_);
        }
    }

    [[nodiscard]] hackrf_device* get() const noexcept {
        return device_;
    }

private:
    hackrf_device* device_ = nullptr;
};

void run_hackrf_acquire(const Options& options) {
    HackrfLibrarySession library;
    HackrfDeviceHandle device(options.serial);

    check_hackrf_result(
        hackrf_set_sample_rate(device.get(), static_cast<double>(options.sample_rate_sps)),
        "hackrf_set_sample_rate");

    if (options.bandwidth_hz != 0) {
        check_hackrf_result(
            hackrf_set_baseband_filter_bandwidth(
                device.get(),
                static_cast<std::uint32_t>(options.bandwidth_hz)),
            "hackrf_set_baseband_filter_bandwidth");
    }

    check_hackrf_result(
        hackrf_set_freq(device.get(), options.frequency_hz),
        "hackrf_set_freq");

    check_hackrf_result(
        hackrf_set_amp_enable(device.get(), static_cast<std::uint8_t>(options.amp_enable)),
        "hackrf_set_amp_enable");

    check_hackrf_result(
        hackrf_set_lna_gain(device.get(), options.lna_gain),
        "hackrf_set_lna_gain");

    check_hackrf_result(
        hackrf_set_vga_gain(device.get(), options.vga_gain),
        "hackrf_set_vga_gain");

    HackrfRxQueue queue;
    queue.max_queued_bytes = options.max_queued_bytes;
    queue.max_samples = options.max_samples;

    std::unique_ptr<std::ofstream> output_file;
    std::ostream* output = &std::cout;
    if (options.output_path != "-") {
        output_file = std::make_unique<std::ofstream>(options.output_path, std::ios::binary);
        if (!*output_file) {
            throw std::runtime_error("failed to open output: " + options.output_path);
        }
        output = output_file.get();
    } else {
        dtmb::tools::configure_binary_stdio(false, true);
    }

    std::cerr << "{\"status\":\"starting\""
              << ",\"frequency_hz\":" << options.frequency_hz
              << ",\"sample_rate_sps\":" << options.sample_rate_sps
              << ",\"bandwidth_hz\":" << options.bandwidth_hz
              << ",\"max_samples\":" << options.max_samples
              << ",\"amp\":" << options.amp_enable
              << ",\"lna_gain\":" << options.lna_gain
              << ",\"vga_gain\":" << options.vga_gain
              << ",\"max_queued_bytes\":" << options.max_queued_bytes
              << "}\n";

    check_hackrf_result(
        hackrf_start_rx(device.get(), hackrf_rx_callback, &queue),
        "hackrf_start_rx");

    std::size_t total_bytes_written = 0;
    std::size_t final_dropped_bytes = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_report_time = start_time;
    bool capture_complete_reported = false;

    bool stopped = false;
    try {
        while (!stop_requested()) {
            std::vector<std::int8_t> chunk;
            {
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.condition.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return !queue.chunks.empty()
                        || queue.finished
                        || queue.stop_requested.load(std::memory_order_relaxed);
                });

                if (stop_requested()) {
                    queue.stop_requested.store(true, std::memory_order_relaxed);
                    final_dropped_bytes = queue.dropped_bytes.load(std::memory_order_acquire);
                    break;
                }

                if (queue.chunks.empty()) {
                    if (queue.finished) {
                        final_dropped_bytes = queue.dropped_bytes.load(std::memory_order_acquire);
                        break;
                    }
                    continue;
                }

                chunk = std::move(queue.chunks.front());
                queue.queued_bytes -= chunk.size();
                queue.chunks.pop_front();
            }

            output->write(reinterpret_cast<const char*>(chunk.data()),
                         static_cast<std::streamsize>(chunk.size()));
            if (!*output) {
                throw std::runtime_error("write failed");
            }

            total_bytes_written += chunk.size();

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double>(now - last_report_time).count();
            std::size_t current_queued_bytes = 0;
            std::size_t max_observed_queued_bytes = 0;
            bool capture_started = false;
            bool capture_finished = false;
            double capture_elapsed_s = 0.0;
            {
                std::lock_guard<std::mutex> lock(queue.mutex);
                current_queued_bytes = queue.queued_bytes;
                max_observed_queued_bytes = queue.max_observed_queued_bytes;
                capture_started = queue.capture_started;
                capture_finished = queue.capture_finished;
                if (capture_started) {
                    const auto capture_end =
                        capture_finished ? queue.capture_finish : now;
                    capture_elapsed_s =
                        std::chrono::duration<double>(capture_end - queue.capture_start).count();
                }
            }
            const auto samples_seen = queue.samples_seen.load(std::memory_order_relaxed);
            if (capture_finished && !capture_complete_reported) {
                std::cerr << "{\"status\":\"capture_complete\""
                          << ",\"capture_elapsed_s\":" << std::fixed
                          << std::setprecision(3) << capture_elapsed_s
                          << ",\"samples_seen\":" << samples_seen
                          << ",\"expected_samples\":" << options.max_samples
                          << ",\"max_observed_queued_bytes\":" << max_observed_queued_bytes
                          << ",\"dropped_bytes\":"
                          << queue.dropped_bytes.load(std::memory_order_relaxed)
                          << "}\n";
                capture_complete_reported = true;
            }
            if (elapsed >= 1.0) {
                auto total_elapsed = std::chrono::duration<double>(now - start_time).count();
                auto throughput_mbps = (total_bytes_written / total_elapsed) / (1024.0 * 1024.0);
                std::cerr << "{\"status\":\"running\""
                          << ",\"elapsed_s\":" << std::fixed << std::setprecision(2) << total_elapsed
                          << ",\"capture_elapsed_s\":" << std::setprecision(3) << capture_elapsed_s
                          << ",\"samples_seen\":" << samples_seen
                          << ",\"queued_bytes\":" << current_queued_bytes
                          << ",\"max_observed_queued_bytes\":" << max_observed_queued_bytes
                          << ",\"bytes_written\":" << total_bytes_written
                          << ",\"throughput_mbps\":" << std::setprecision(1) << throughput_mbps
                          << ",\"dropped_bytes\":" << queue.dropped_bytes.load(std::memory_order_relaxed)
                          << "}\n";
                last_report_time = now;
            }
        }
    } catch (...) {
        queue.stop_requested.store(true, std::memory_order_relaxed);
        final_dropped_bytes = queue.dropped_bytes.load(std::memory_order_acquire);
        check_hackrf_result(hackrf_stop_rx(device.get()), "hackrf_stop_rx");
        stopped = true;
        throw;
    }

    if (!stopped) {
        check_hackrf_result(hackrf_stop_rx(device.get()), "hackrf_stop_rx");
    }

    // Capture final dropped bytes count from queue
    final_dropped_bytes = queue.dropped_bytes.load();

    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    auto throughput_mbps = (total_bytes_written / total_elapsed) / (1024.0 * 1024.0);
    std::size_t max_observed_queued_bytes = 0;
    bool capture_started = false;
    bool capture_finished = false;
    double capture_elapsed_s = 0.0;
    {
        std::lock_guard<std::mutex> lock(queue.mutex);
        max_observed_queued_bytes = queue.max_observed_queued_bytes;
        capture_started = queue.capture_started;
        capture_finished = queue.capture_finished;
        if (capture_started) {
            const auto capture_end =
                capture_finished ? queue.capture_finish : end_time;
            capture_elapsed_s =
                std::chrono::duration<double>(capture_end - queue.capture_start).count();
        }
    }
    const auto samples_seen = queue.samples_seen.load();
    const auto expected_capture_s = options.sample_rate_sps != 0
        ? static_cast<double>(samples_seen) / static_cast<double>(options.sample_rate_sps)
        : 0.0;
    const auto capture_realtime_ratio = capture_elapsed_s > 0.0
        ? expected_capture_s / capture_elapsed_s
        : 0.0;

    const auto stop_reason = stop_requested()
        ? signal_stop_reason()
        : (options.max_samples != 0 && samples_seen >= options.max_samples)
            ? std::string_view("max_samples")
            : std::string_view("source_complete");

    std::cerr << "{\"status\":\"complete\""
              << ",\"stop_reason\":\"" << stop_reason << "\""
              << ",\"total_elapsed_s\":" << std::fixed << std::setprecision(3) << total_elapsed
              << ",\"capture_elapsed_s\":" << std::setprecision(3) << capture_elapsed_s
              << ",\"expected_capture_s\":" << std::setprecision(3) << expected_capture_s
              << ",\"capture_realtime_ratio\":" << std::setprecision(6)
              << capture_realtime_ratio
              << ",\"samples_seen\":" << samples_seen
              << ",\"max_observed_queued_bytes\":" << max_observed_queued_bytes
              << ",\"bytes_written\":" << total_bytes_written
              << ",\"samples_written\":" << (total_bytes_written / 2)
              << ",\"throughput_mbps\":" << std::setprecision(2) << throughput_mbps
              << ",\"dropped_bytes\":" << final_dropped_bytes
              << ",\"dropped_samples\":" << (final_dropped_bytes / 2)
              << "}\n";

    if (final_dropped_bytes != 0) {
        std::cerr << "WARNING: dropped " << final_dropped_bytes
                  << " bytes (" << (final_dropped_bytes / 2) << " samples) "
                  << "due to consumer falling behind\n";
    }
}

#else

void run_hackrf_acquire(const Options&) {
    throw std::runtime_error(
        "direct HackRF acquisition not available; rebuild with DTMB_CORE_ENABLE_HACKRF=ON and libusb");
}

#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        StopSignalSession signal_session;
        run_hackrf_acquire(options);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
