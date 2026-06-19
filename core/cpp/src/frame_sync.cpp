#include "dtmb/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dtmb::core {
namespace {

using Complex = std::complex<double>;

struct PhaseScore {
    std::size_t phase_offset = 0;
    float mean_metric = 0.0F;
    float max_metric = 0.0F;
    std::size_t hit_count = 0;
    std::size_t observed_frames = 0;
};

[[nodiscard]] Complex sample_at(
    std::span<const float> symbols,
    std::size_t sample) noexcept {
    return Complex{symbols[sample * 2], symbols[sample * 2 + 1]};
}

[[nodiscard]] float repeated_window_similarity(
    std::span<const float> symbols,
    std::size_t start,
    std::size_t first_offset,
    std::size_t second_offset,
    std::size_t length) {
    auto numerator = Complex{};
    double first_power = 0.0;
    double second_power = 0.0;
    for (std::size_t sample = 0; sample < length; ++sample) {
        const auto first = sample_at(symbols, start + first_offset + sample);
        const auto second = sample_at(symbols, start + second_offset + sample);
        numerator += std::conj(first) * second;
        first_power += std::norm(first);
        second_power += std::norm(second);
    }
    const auto denominator = std::sqrt(std::max(first_power * second_power, 1.0e-30));
    return static_cast<float>(std::abs(numerator) / denominator);
}

[[nodiscard]] float cyclic_extension_metric(
    std::span<const float> symbols,
    std::size_t start) {
    const auto prefix = repeated_window_similarity(symbols, start, 0, 511, 217);
    const auto suffix = repeated_window_similarity(symbols, start, 217, 728, 217);
    return (prefix + suffix) * 0.5F;
}

[[nodiscard]] PhaseScore score_phase(
    std::span<const float> symbols,
    std::size_t phase_offset,
    const Pn945AcquisitionOptions& options) {
    const auto sample_count = symbols.size() / 2;
    std::vector<float> values;
    values.reserve(options.max_frames);
    for (auto start = phase_offset;
         start + kPn945HeaderSymbols <= sample_count
         && values.size() < options.max_frames;
         start += kPn945FrameSymbols) {
        values.push_back(cyclic_extension_metric(symbols, start));
    }

    PhaseScore result;
    result.phase_offset = phase_offset;
    result.observed_frames = values.size();
    if (values.empty()) {
        return result;
    }

    result.max_metric = *std::max_element(values.begin(), values.end());
    double hit_sum = 0.0;
    for (const auto value : values) {
        if (value >= options.hit_threshold) {
            hit_sum += value;
            ++result.hit_count;
        }
    }
    if (result.hit_count != 0) {
        result.mean_metric = static_cast<float>(
            hit_sum / static_cast<double>(result.hit_count));
        return result;
    }

    std::sort(values.begin(), values.end(), std::greater<>());
    const auto strongest_count = std::min<std::size_t>(8, values.size());
    double strongest_sum = 0.0;
    for (std::size_t index = 0; index < strongest_count; ++index) {
        strongest_sum += values[index];
    }
    result.mean_metric = static_cast<float>(
        strongest_sum / static_cast<double>(strongest_count));
    return result;
}

[[nodiscard]] bool score_is_better(
    const PhaseScore& candidate,
    const PhaseScore& current) noexcept {
    if (candidate.hit_count != current.hit_count) {
        return candidate.hit_count > current.hit_count;
    }
    if (candidate.mean_metric != current.mean_metric) {
        return candidate.mean_metric > current.mean_metric;
    }
    if (candidate.max_metric != current.max_metric) {
        return candidate.max_metric > current.max_metric;
    }
    return candidate.phase_offset < current.phase_offset;
}

[[nodiscard]] std::size_t choose_worker_count(
    std::size_t work_count,
    std::size_t requested_workers) noexcept {
    if (work_count == 0) {
        return 0;
    }
    auto worker_count = requested_workers;
    if (worker_count == 0) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    return std::clamp<std::size_t>(worker_count, 1, work_count);
}

[[nodiscard]] std::pair<float, bool> estimate_coarse_cfo(
    std::span<const float> symbols,
    std::size_t phase_offset,
    std::size_t max_frames) {
    const auto sample_count = symbols.size() / 2;
    auto accumulator = Complex{};
    std::size_t used_frames = 0;
    for (auto start = phase_offset;
         start + kPn945HeaderSymbols <= sample_count && used_frames < max_frames;
         start += kPn945FrameSymbols) {
        for (const auto [first_offset, second_offset, length] :
             {std::array<std::size_t, 3>{0, 511, 217},
              std::array<std::size_t, 3>{217, 728, 217}}) {
            for (std::size_t sample = 0; sample < length; ++sample) {
                const auto first = sample_at(symbols, start + first_offset + sample);
                const auto second = sample_at(symbols, start + second_offset + sample);
                accumulator += std::conj(first) * second;
            }
        }
        ++used_frames;
    }
    if (used_frames == 0 || std::abs(accumulator) == 0.0) {
        return {0.0F, false};
    }
    const auto phase = std::arg(accumulator);
    const auto cfo = phase * static_cast<double>(kDtmbSymbolRateSps)
        / (2.0 * std::numbers::pi_v<double> * 511.0);
    return {static_cast<float>(cfo), true};
}

}  // namespace

Pn945AcquisitionResult acquire_pn945_cf32(
    std::span<const float> interleaved_symbols,
    Pn945AcquisitionOptions options) {
    if ((interleaved_symbols.size() % 2) != 0) {
        throw std::invalid_argument("PN945 acquisition input must contain CF32 pairs");
    }
    if (options.max_frames == 0) {
        throw std::invalid_argument("PN945 acquisition max_frames must be positive");
    }
    if (options.hit_threshold < 0.0F || options.hit_threshold > 1.0F) {
        throw std::invalid_argument("PN945 acquisition hit_threshold must be in [0, 1]");
    }

    const auto sample_count = interleaved_symbols.size() / 2;
    if (sample_count < kPn945HeaderSymbols) {
        throw std::invalid_argument("PN945 acquisition requires at least one header");
    }
    const auto phase_count = std::min(
        kPn945FrameSymbols,
        sample_count - kPn945HeaderSymbols + 1);
    const auto worker_count = choose_worker_count(phase_count, options.requested_workers);
    std::vector<PhaseScore> scores(phase_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::size_t phase = worker; phase < phase_count; phase += worker_count) {
                scores[phase] = score_phase(interleaved_symbols, phase, options);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    auto best = scores.front();
    for (std::size_t phase = 1; phase < scores.size(); ++phase) {
        if (score_is_better(scores[phase], best)) {
            best = scores[phase];
        }
    }
    const auto [coarse_cfo_hz, coarse_cfo_valid] = estimate_coarse_cfo(
        interleaved_symbols,
        best.phase_offset,
        options.max_frames);
    return Pn945AcquisitionResult{
        best.phase_offset,
        best.mean_metric,
        best.max_metric,
        best.hit_count,
        best.observed_frames,
        worker_count,
        coarse_cfo_hz,
        coarse_cfo_valid,
    };
}

}  // namespace dtmb::core
