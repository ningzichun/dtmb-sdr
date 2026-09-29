#include "dtmb/core.hpp"
#include "dtmb/iq.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " [--input-rate N] [--output-rate N]"
        << " [--input-format ci8|cs8|cu8|ci16|sc16|cf32] [--output-format ci8|cf32]"
        << " [--srrc-span N] [--roll-off X] [--output-scale X]"
        << " [--workers N] [--min-parallel-output-samples N]"
        << " [--chunk-samples N] [input.ci8|-] [output.ci8|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size() || value == 0) {
        throw std::invalid_argument(std::string(field) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::size_t parse_nonnegative_size(const std::string& text, const char* field) {
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

std::size_t read_bytes(std::istream& input, std::span<std::int8_t> buffer) {
    input.read(
        reinterpret_cast<char*>(buffer.data()),
        static_cast<std::streamsize>(buffer.size()));
    return static_cast<std::size_t>(input.gcount());
}

void write_ci8(
    std::ostream& output,
    std::span<const float> interleaved_cf32,
    float output_scale,
    std::size_t& clip_count) {
    std::vector<std::int8_t> ci8(interleaved_cf32.size());
    for (std::size_t value = 0; value < interleaved_cf32.size(); ++value) {
        const auto scaled = std::nearbyint(interleaved_cf32[value] * output_scale);
        clip_count += scaled < -128.0F || scaled > 127.0F ? 1U : 0U;
        ci8[value] = static_cast<std::int8_t>(std::clamp(scaled, -128.0F, 127.0F));
    }
    output.write(
        reinterpret_cast<const char*>(ci8.data()),
        static_cast<std::streamsize>(ci8.size()));
    if (!output) {
        throw std::runtime_error("failed to write resampled CI8 stream");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t input_rate = 20'000'000;
    std::size_t output_rate = dtmb::core::kDtmbSymbolRateSps;
    std::size_t srrc_span = 8;
    std::size_t chunk_samples = 1U << 20U;
    std::size_t requested_workers = 0;
    std::size_t min_parallel_output_samples = 16'384;
    float roll_off = 0.05F;
    float output_scale = 1.0F / std::sqrt(45.0F);
    bool scale_explicit = false;
    auto input_format = dtmb::core::IqFormat::ci8;
    std::string output_format = "ci8";
    std::string input_path = "-";
    std::string output_path = "-";
    std::vector<std::string> positional;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--input-format") {
                if (++index >= argc) throw std::invalid_argument("missing input format");
                input_format = dtmb::core::parse_iq_format(argv[index]);
            } else if (arg == "--output-format") {
                if (++index >= argc) throw std::invalid_argument("missing output format");
                output_format = argv[index];
                if (output_format != "ci8" && output_format != "cf32")
                    throw std::invalid_argument("output format must be ci8 or cf32");
            } else if (arg == "--input-rate") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                input_rate = parse_size(argv[index], "input rate");
            } else if (arg == "--output-rate") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                output_rate = parse_size(argv[index], "output rate");
            } else if (arg == "--srrc-span") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                srrc_span = parse_size(argv[index], "SRRC span");
            } else if (arg == "--roll-off") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                roll_off = parse_float(argv[index], "roll off");
            } else if (arg == "--output-scale") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                output_scale = parse_float(argv[index], "output scale");
                scale_explicit = true;
            } else if (arg == "--chunk-samples") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                chunk_samples = parse_size(argv[index], "chunk sample count");
            } else if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                requested_workers = parse_nonnegative_size(argv[index], "worker count");
            } else if (arg == "--min-parallel-output-samples") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                min_parallel_output_samples = parse_size(
                    argv[index],
                    "parallel output threshold");
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                positional.push_back(arg);
            }
        }
        if (positional.size() > 2 || !(roll_off > 0.0F && roll_off <= 1.0F)
            || !std::isfinite(output_scale) || output_scale <= 0.0F
            || input_rate > std::numeric_limits<std::uint32_t>::max()
            || output_rate > std::numeric_limits<std::uint32_t>::max()) {
            usage(argv[0]);
            return 2;
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }
        if (positional.size() == 2) {
            output_path = positional[1];
        }

        const auto common = std::gcd(input_rate, output_rate);
        const auto up = output_rate / common;
        const auto down = input_rate / common;
        if (input_format != dtmb::core::IqFormat::ci8 || output_format == "cf32"
            || std::max(up, down) > 4096) {
            dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
            std::unique_ptr<std::ifstream> input_file;
            std::unique_ptr<std::ofstream> output_file;
            auto& input = input_stream(input_path, input_file);
            auto& output = output_stream(output_path, output_file);
            dtmb::core::IqResampler converter(static_cast<std::uint32_t>(input_rate),
                static_cast<std::uint32_t>(output_rate), srrc_span, roll_off);
            std::vector<float> values(chunk_samples * 2), converted;
            std::size_t clips = 0, samples = 0, produced = 0;
            const float gain = scale_explicit ? output_scale : 1.0F;
            auto write = [&] {
                produced += converted.size() / 2;
                if (output_format == "cf32") {
                    for (auto& v : converted) v *= gain / 128.0F;
                    output.write(reinterpret_cast<const char*>(converted.data()),
                        static_cast<std::streamsize>(converted.size() * sizeof(float)));
                    if (!output) throw std::runtime_error("failed to write CF32 output");
                } else write_ci8(output, converted, gain, clips);
            };
            while (auto count = dtmb::core::read_iq(input, input_format, values)) {
                samples += count / 2;
                converted.clear();
                if (input_rate == output_rate) converted.assign(values.begin(), values.begin() + count);
                else converter.process(std::span(values).first(count), converted);
                write();
            }
            converted.clear();
            if (input_rate != output_rate) converter.finish(converted);
            write();
            output.flush();
            std::cerr << "input_rate_sps=" << input_rate << '\n'
                      << "output_rate_sps=" << output_rate << '\n'
                      << "input_samples=" << samples << '\n'
                      << "output_samples=" << produced << '\n'
                      << "output_format=" << output_format << '\n'
                      << "output_clip_values=" << clips << '\n'
                      << "trailing_input_bytes=0\n";
            return 0;
        }
        const auto taps = dtmb::core::square_root_raised_cosine_taps(
            srrc_span,
            std::max(up, down),
            roll_off);
        dtmb::core::RationalResamplerCf32 resampler(
            up,
            down,
            taps,
            requested_workers,
            min_parallel_output_samples);

        dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);

        std::vector<std::int8_t> input_ci8(chunk_samples * 2);
        std::vector<float> input_cf32(chunk_samples * 2);
        std::vector<float> output_cf32;
        output_cf32.reserve((chunk_samples * up / down + 64) * 2);
        std::size_t trailing_bytes = 0;
        std::size_t clip_count = 0;
        while (true) {
            const auto bytes = read_bytes(input, input_ci8);
            if (bytes == 0) {
                break;
            }
            const auto usable_bytes = bytes - bytes % 2;
            trailing_bytes += bytes - usable_bytes;
            for (std::size_t value = 0; value < usable_bytes; ++value) {
                input_cf32[value] = static_cast<float>(input_ci8[value]);
            }
            output_cf32.clear();
            resampler.process(
                std::span<const float>(input_cf32.data(), usable_bytes),
                output_cf32);
            write_ci8(output, output_cf32, output_scale, clip_count);
            if (bytes != input_ci8.size()) {
                break;
            }
        }
        output_cf32.clear();
        resampler.finish(output_cf32);
        write_ci8(output, output_cf32, output_scale, clip_count);

        std::cerr << "input_rate_sps=" << input_rate << '\n'
                  << "output_rate_sps=" << output_rate << '\n'
                  << "up_factor=" << up << '\n'
                  << "down_factor=" << down << '\n'
                  << "srrc_span_symbols=" << srrc_span << '\n'
                  << "srrc_roll_off=" << roll_off << '\n'
                  << "prototype_taps=" << taps.size() << '\n'
                  << "worker_count=" << resampler.max_worker_count() << '\n'
                  << "output_scale=" << output_scale << '\n'
                  << "input_samples=" << resampler.processed_input_samples() << '\n'
                  << "output_samples=" << resampler.produced_output_samples() << '\n'
                  << "output_clip_values=" << clip_count << '\n'
                  << "trailing_input_bytes=" << trailing_bytes << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_ci8_resample: " << exc.what() << '\n';
        return 1;
    }
}
