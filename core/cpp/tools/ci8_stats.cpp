#include "dtmb/core.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " [--workers N] [--min-parallel-samples N] [--chunk-samples N]"
        << " [--passthrough] [capture.ci8|-]\n";
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

void write_all(std::ostream& output, std::span<const std::int8_t> bytes) {
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("failed to write CI8 passthrough stream");
    }
}

void print_stats(std::ostream& output, const dtmb::core::Ci8PowerStats& stats) {
    output << std::setprecision(10)
           << "sample_count=" << stats.sample_count << '\n'
           << "mean_i2q2=" << stats.mean_i2q2 << '\n'
           << "rms_iq=" << stats.rms_iq << '\n'
           << "clip_count_i=" << stats.clip_count_i << '\n'
           << "clip_count_q=" << stats.clip_count_q << '\n'
           << "worker_count=" << stats.worker_count << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    dtmb::core::Ci8PowerStatsOptions options{};
    std::size_t chunk_samples = 1U << 20U;
    bool passthrough = false;
    std::string input_path = "-";
    bool input_path_set = false;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                options.requested_workers = parse_size(argv[index], "worker count");
            } else if (arg == "--min-parallel-samples") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                options.min_parallel_samples = parse_size(argv[index], "parallel threshold");
            } else if (arg == "--chunk-samples") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                chunk_samples = parse_size(argv[index], "chunk size");
            } else if (arg == "--passthrough") {
                passthrough = true;
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else if (!input_path_set) {
                input_path = arg;
                input_path_set = true;
            } else {
                usage(argv[0]);
                return 2;
            }
        }

        if (chunk_samples == 0) {
            usage(argv[0]);
            return 2;
        }

        dtmb::tools::configure_binary_stdio(input_path == "-", passthrough);
        std::unique_ptr<std::ifstream> input_file;
        auto& input = input_stream(input_path, input_file);
        std::vector<std::int8_t> chunk(chunk_samples * 2);
        const auto chunk_bytes = static_cast<std::streamsize>(chunk.size());
        std::size_t total_samples = 0;
        long double total_i2q2 = 0.0L;
        std::size_t total_clip_i = 0;
        std::size_t total_clip_q = 0;
        std::size_t max_workers = 0;

        while (true) {
            input.read(reinterpret_cast<char*>(chunk.data()), chunk_bytes);
            const auto bytes_read = input.gcount();
            if (bytes_read == 0) {
                break;
            }
            if ((bytes_read % 2) != 0) {
                throw std::runtime_error("CI8 input byte count is not a whole number of samples");
            }

            const auto values = std::span<const std::int8_t>(
                chunk.data(),
                static_cast<std::size_t>(bytes_read));
            if (passthrough) {
                write_all(std::cout, values);
            }
            const auto chunk_stats = dtmb::core::ci8_power_stats(values, options);
            total_samples += chunk_stats.sample_count;
            total_i2q2 += static_cast<long double>(chunk_stats.mean_i2q2)
                * static_cast<long double>(chunk_stats.sample_count);
            total_clip_i += chunk_stats.clip_count_i;
            total_clip_q += chunk_stats.clip_count_q;
            max_workers = std::max(max_workers, chunk_stats.worker_count);

            if (bytes_read < chunk_bytes) {
                break;
            }
        }

        dtmb::core::Ci8PowerStats stats;
        stats.sample_count = total_samples;
        stats.clip_count_i = total_clip_i;
        stats.clip_count_q = total_clip_q;
        stats.worker_count = max_workers;
        if (total_samples != 0) {
            stats.mean_i2q2 = static_cast<float>(
                total_i2q2 / static_cast<long double>(total_samples));
            stats.rms_iq = std::sqrt(stats.mean_i2q2);
        }
        print_stats(passthrough ? std::cerr : std::cout, stats);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_ci8_stats: " << exc.what() << '\n';
        return 1;
    }
}
