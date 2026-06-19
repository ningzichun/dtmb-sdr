#include "dtmb/core.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

struct FrameWork {
    FrameWork()
        : header_ci8(dtmb::core::kPn945HeaderSymbols * 2),
          body_ci8(dtmb::core::kC3780FrameBodySymbols * 2),
          header_cf32(dtmb::core::kPn945HeaderSymbols * 2),
          body_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          spectrum_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          logical_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          data_cf32(dtmb::core::kC3780DataSymbols * 2),
          llr_f32(dtmb::core::kC3780DataSymbols * 6) {}

    std::vector<std::int8_t> header_ci8;
    std::vector<std::int8_t> body_ci8;
    std::vector<float> header_cf32;
    std::vector<float> body_cf32;
    std::vector<float> spectrum_cf32;
    std::vector<float> logical_cf32;
    std::vector<float> data_cf32;
    std::vector<float> llr_f32;
    std::size_t sample_start = 0;
};

constexpr std::size_t kLlrsPerSymbol = 6;

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " [--phase-offset N|--auto-sync]"
        << " [--sync-frames N] [--acquisition-frames N]"
        << " [--sync-hit-threshold X]"
        << " [--frequency-shift-hz X]"
        << " [--pn-channel-taps N]"
        << " [--noise-variance X]"
        << " [--workers N]"
        << " [--max-frames N]"
        << " [input.ci8|-] [output.llr.f32|-]\n";
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

// __CONTINUE_HERE__

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

void write_all(std::ostream& output, std::span<const float> values) {
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write LLR stream");
    }
}

void convert_ci8_to_cf32(
    std::span<const std::int8_t> input,
    std::span<float> output,
    std::size_t sample_start,
    float frequency_shift_hz) {
    constexpr double sample_rate = 7'560'000.0;
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    const auto radians_per_sample =
        two_pi * static_cast<double>(frequency_shift_hz) / sample_rate;
    auto rotation = std::polar(
        1.0,
        radians_per_sample * static_cast<double>(sample_start));
    const auto rotation_step = std::polar(1.0, radians_per_sample);
    for (std::size_t sample = 0; sample < input.size() / 2; ++sample) {
        const auto value = std::complex<double>{
            static_cast<float>(input[sample * 2]),
            static_cast<float>(input[sample * 2 + 1]),
        } / 128.0;
        const auto shifted = value * rotation;
        output[sample * 2] = shifted.real();
        output[sample * 2 + 1] = shifted.imag();
        rotation *= rotation_step;
    }
}

// __CONTINUE_HERE__

bool read_frame(
    std::istream& input,
    FrameWork& frame,
    std::size_t& trailing_ci8_bytes,
    std::size_t& next_sample_start) {
    frame.sample_start = next_sample_start;
    const auto header_bytes = read_bytes(input, frame.header_ci8);
    if (header_bytes == 0) {
        return false;
    }
    if (header_bytes != frame.header_ci8.size()) {
        trailing_ci8_bytes = header_bytes;
        return false;
    }
    const auto body_bytes = read_bytes(input, frame.body_ci8);
    if (body_bytes != frame.body_ci8.size()) {
        trailing_ci8_bytes = frame.header_ci8.size() + body_bytes;
        return false;
    }
    next_sample_start += dtmb::core::kPn945FrameSymbols;
    return true;
}

dtmb::core::Pn945EqualizeResult process_pn_frame(
    FrameWork& frame,
    FrameWork& next_frame,
    std::size_t pn_channel_taps,
    float frequency_shift_hz,
    float noise_variance) {
    convert_ci8_to_cf32(
        frame.header_ci8,
        frame.header_cf32,
        frame.sample_start,
        frequency_shift_hz);
    convert_ci8_to_cf32(
        frame.body_ci8,
        frame.body_cf32,
        frame.sample_start + dtmb::core::kPn945HeaderSymbols,
        frequency_shift_hz);
    convert_ci8_to_cf32(
        next_frame.header_ci8,
        next_frame.header_cf32,
        next_frame.sample_start,
        frequency_shift_hz);
    const auto result = dtmb::core::pn945_equalize_c3780_frame_cf32(
        frame.header_cf32,
        frame.body_cf32,
        next_frame.header_cf32,
        frame.spectrum_cf32,
        dtmb::core::Pn945EqualizeOptions{
            pn_channel_taps,
            1.0e-3F,
            1.0e-6F,
            noise_variance,
        });
    dtmb::core::c3780_deinterleave_spectrum_cf32(
        frame.spectrum_cf32,
        frame.logical_cf32);
    return result;
}

// __CONTINUE_HERE__

}  // namespace

int main(int argc, char** argv) {
    std::size_t phase_offset = 0;
    std::size_t max_frames = 0;
    std::size_t sync_frames = 300;
    std::size_t acquisition_frames = 16;
    std::size_t pn_channel_taps = 8;
    float frequency_shift_hz = 0.0F;
    float sync_hit_threshold = 0.35F;
    float noise_variance = 1.0F;
    std::size_t requested_workers = 0;
    bool auto_sync = false;
    bool phase_offset_set = false;
    std::string input_path = "-";
    std::string output_path = "-";
    std::vector<std::string> positional;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--phase-offset") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                phase_offset = parse_size(argv[index], "phase offset");
                phase_offset_set = true;
            } else if (arg == "--auto-sync") {
                auto_sync = true;
            } else if (arg == "--sync-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                sync_frames = parse_size(argv[index], "sync frame count");
            } else if (arg == "--acquisition-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                acquisition_frames = parse_size(argv[index], "acquisition frame count");
            } else if (arg == "--sync-hit-threshold") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                sync_hit_threshold = parse_float(argv[index], "sync hit threshold");
// __CONTINUE_HERE__

            } else if (arg == "--frequency-shift-hz") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                frequency_shift_hz = parse_float(argv[index], "frequency shift");
            } else if (arg == "--pn-channel-taps") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_channel_taps = parse_size(argv[index], "PN channel tap count");
            } else if (arg == "--noise-variance") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                noise_variance = parse_float(argv[index], "noise variance");
            } else if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                requested_workers = parse_size(argv[index], "worker count");
            } else if (arg == "--max-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                max_frames = parse_size(argv[index], "max frames");
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                positional.push_back(arg);
            }
        }
        if (positional.size() > 2 || (auto_sync && phase_offset_set)
            || pn_channel_taps == 0 || pn_channel_taps > 511
            || sync_frames < 2 || acquisition_frames == 0
            || sync_hit_threshold < 0.0F || sync_hit_threshold > 1.0F) {
            usage(argv[0]);
            return 2;
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }
        if (positional.size() == 2) {
            output_path = positional[1];
        }

// __CONTINUE_HERE__

        dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);

        // Auto-sync: acquire PN945 frame phase
        float automatic_frequency_shift_hz = 0.0F;
        if (auto_sync) {
            const auto startup_samples = (sync_frames + 1) * dtmb::core::kPn945FrameSymbols;
            std::vector<std::int8_t> startup_ci8(startup_samples * 2);
            const auto startup_bytes = read_bytes(input, startup_ci8);
            startup_ci8.resize(startup_bytes);
            if ((startup_ci8.size() % 2) != 0) {
                throw std::runtime_error("auto-sync startup buffer ended with incomplete sample");
            }
            if (startup_ci8.size() / 2 < dtmb::core::kPn945FrameSymbols * 2) {
                throw std::runtime_error("auto-sync requires at least two PN945 frames");
            }

            std::vector<float> startup_cf32(startup_ci8.size());
            convert_ci8_to_cf32(startup_ci8, startup_cf32, 0, 0.0F);
            const auto acquisition = dtmb::core::acquire_pn945_cf32(
                startup_cf32,
                dtmb::core::Pn945AcquisitionOptions{
                    acquisition_frames,
                    sync_hit_threshold,
                    requested_workers,
                });
            if (acquisition.hit_count < 2) {
                throw std::runtime_error(
                    "auto-sync did not find repeated PN945 train above threshold");
            }
            phase_offset = acquisition.phase_offset;
            if (acquisition.coarse_cfo_valid) {
                automatic_frequency_shift_hz = -acquisition.coarse_cfo_hz;
            }

            std::cerr << "{\"event\":\"acquisition\",\"phase_offset\":" << phase_offset
                      << ",\"hit_count\":" << acquisition.hit_count
                      << ",\"max_metric\":" << acquisition.max_metric
                      << ",\"coarse_cfo_hz\":" << acquisition.coarse_cfo_hz
                      << ",\"coarse_cfo_valid\":" << (acquisition.coarse_cfo_valid ? "true" : "false")
                      << "}\n";
        }
        frequency_shift_hz += automatic_frequency_shift_hz;

// __CONTINUE_HERE__

        // Discard phase offset
        const auto phase_bytes = phase_offset * 2;
        std::vector<std::int8_t> discard_buffer(std::min<std::size_t>(phase_bytes, 1U << 16U));
        std::size_t discarded = 0;
        while (discarded < phase_bytes) {
            const auto wanted = std::min(discard_buffer.size(), phase_bytes - discarded);
            const auto count = read_bytes(input, std::span<std::int8_t>(discard_buffer.data(), wanted));
            discarded += count;
            if (count != wanted) {
                break;
            }
        }

        // Process frames
        std::size_t frame_count = 0;
        std::size_t trailing_ci8_bytes = 0;
        std::size_t next_sample_start = phase_offset;
        std::size_t first_pn_phase = 0;
        std::size_t last_pn_phase = 0;

        FrameWork current;
        FrameWork next;
        auto have_current = read_frame(input, current, trailing_ci8_bytes, next_sample_start);
        auto have_next = have_current && read_frame(input, next, trailing_ci8_bytes, next_sample_start);

        dtmb::core::QamSoftDemapOptions demap_options{};
        demap_options.noise_variance = noise_variance;
        demap_options.requested_workers = requested_workers;

        while (have_current && have_next && (max_frames == 0 || frame_count < max_frames)) {
            const auto result = process_pn_frame(
                current,
                next,
                pn_channel_taps,
                frequency_shift_hz,
                -1.0F);
            if (frame_count == 0) {
                first_pn_phase = result.pn_phase;
            }
            last_pn_phase = result.pn_phase;

            // Extract data symbols (skip system info)
            const auto data = std::span<const float>(
                current.logical_cf32.data() + dtmb::core::kC3780SystemInfoSymbols * 2,
                dtmb::core::kC3780DataSymbols * 2);
            dtmb::core::qam64_normalize_cf32(data, current.data_cf32);

            // Soft demap to LLR
            dtmb::core::qam64_soft_demodulate_cf32(current.data_cf32, current.llr_f32, demap_options);
            write_all(output, current.llr_f32);
            ++frame_count;

            std::swap(current, next);
            have_next = read_frame(input, next, trailing_ci8_bytes, next_sample_start);
        }

// __CONTINUE_HERE__

        // Telemetry output
        std::cerr << "{\"event\":\"complete\""
                  << ",\"frames\":" << frame_count
                  << ",\"output_symbols\":" << frame_count * dtmb::core::kC3780DataSymbols
                  << ",\"output_llrs\":" << frame_count * dtmb::core::kC3780DataSymbols * kLlrsPerSymbol
                  << ",\"trailing_ci8_bytes\":" << trailing_ci8_bytes
                  << ",\"phase_offset\":" << phase_offset
                  << ",\"frequency_shift_hz\":" << frequency_shift_hz
                  << ",\"pn_channel_taps\":" << pn_channel_taps
                  << ",\"noise_variance\":" << noise_variance
                  << ",\"first_pn_phase\":" << first_pn_phase
                  << ",\"last_pn_phase\":" << last_pn_phase
                  << "}\n";

        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "{\"event\":\"error\",\"message\":\"" << exc.what() << "\"}\n";
        return 1;
    }
}

