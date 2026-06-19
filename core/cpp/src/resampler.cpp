#include "dtmb/core.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dtmb::core {
namespace {

[[nodiscard]] std::int64_t floor_div(std::int64_t numerator, std::int64_t denominator) {
    auto quotient = numerator / denominator;
    const auto remainder = numerator % denominator;
    if (remainder < 0) {
        --quotient;
    }
    return quotient;
}

[[nodiscard]] std::int64_t ceil_div(std::int64_t numerator, std::int64_t denominator) {
    return -floor_div(-numerator, denominator);
}

[[nodiscard]] double srrc_value(double time_symbols, double roll_off) {
    if (time_symbols == 0.0) {
        return 1.0 - roll_off + 4.0 * roll_off / std::numbers::pi_v<double>;
    }
    if (std::abs(std::abs(4.0 * roll_off * time_symbols) - 1.0) < 1.0e-12) {
        const auto scale = roll_off / std::sqrt(2.0);
        return scale * (
            (1.0 + 2.0 / std::numbers::pi_v<double>)
                * std::sin(std::numbers::pi_v<double> / (4.0 * roll_off))
            + (1.0 - 2.0 / std::numbers::pi_v<double>)
                * std::cos(std::numbers::pi_v<double> / (4.0 * roll_off)));
    }
    const auto numerator =
        std::sin(std::numbers::pi_v<double> * time_symbols * (1.0 - roll_off))
        + 4.0 * roll_off * time_symbols
            * std::cos(std::numbers::pi_v<double> * time_symbols * (1.0 + roll_off));
    const auto denominator = std::numbers::pi_v<double> * time_symbols
        * (1.0 - std::pow(4.0 * roll_off * time_symbols, 2.0));
    return numerator / denominator;
}

[[nodiscard]] std::size_t checked_time_index(
    std::size_t output_sample,
    std::size_t pre_remove_output_samples,
    std::size_t down_factor) {
    if (output_sample > std::numeric_limits<std::size_t>::max()
            - pre_remove_output_samples
        || output_sample + pre_remove_output_samples
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
                / down_factor) {
        throw std::overflow_error("rational resampler time index overflow");
    }
    return (output_sample + pre_remove_output_samples) * down_factor;
}

}  // namespace

std::vector<float> square_root_raised_cosine_taps(
    std::size_t one_sided_symbol_span,
    std::size_t samples_per_symbol,
    float roll_off) {
    if (one_sided_symbol_span == 0 || samples_per_symbol == 0) {
        throw std::invalid_argument("SRRC span and samples_per_symbol must be positive");
    }
    if (!(roll_off > 0.0F && roll_off <= 1.0F)) {
        throw std::invalid_argument("SRRC roll_off must be in (0, 1]");
    }
    if (one_sided_symbol_span
        > (std::numeric_limits<std::size_t>::max() - 1) / (2 * samples_per_symbol)) {
        throw std::overflow_error("SRRC tap count overflow");
    }

    const auto tap_count = 2 * one_sided_symbol_span * samples_per_symbol + 1;
    const auto center = static_cast<double>(tap_count - 1) / 2.0;
    std::vector<double> response(tap_count);
    double energy = 0.0;
    for (std::size_t tap = 0; tap < tap_count; ++tap) {
        const auto time_symbols =
            (static_cast<double>(tap) - center) / static_cast<double>(samples_per_symbol);
        response[tap] = srrc_value(time_symbols, static_cast<double>(roll_off));
        energy += response[tap] * response[tap];
    }
    const auto normalization = std::sqrt(energy);
    std::vector<float> taps(tap_count);
    for (std::size_t tap = 0; tap < tap_count; ++tap) {
        taps[tap] = static_cast<float>(response[tap] / normalization);
    }
    return taps;
}

RationalResamplerCf32::RationalResamplerCf32(
    std::size_t up_factor,
    std::size_t down_factor,
    std::span<const float> prototype_taps,
    std::size_t requested_workers,
    std::size_t min_parallel_output_samples)
    : up_factor_(up_factor),
      down_factor_(down_factor),
      prototype_tap_count_(prototype_taps.size()),
      pre_remove_output_samples_(0),
      padded_filter_tap_count_(0),
      requested_workers_(requested_workers),
      min_parallel_output_samples_(min_parallel_output_samples) {
    if (up_factor_ == 0 || down_factor_ == 0) {
        throw std::invalid_argument("rational resampler factors must be positive");
    }
    if (prototype_taps.empty() || (prototype_taps.size() % 2) == 0) {
        throw std::invalid_argument("rational resampler prototype must have odd length");
    }
    if (min_parallel_output_samples_ == 0) {
        throw std::invalid_argument("rational resampler parallel threshold must be positive");
    }
    const auto half_length = (prototype_taps.size() - 1) / 2;
    const auto pre_pad = down_factor_ - half_length % down_factor_;
    pre_remove_output_samples_ = (half_length + pre_pad) / down_factor_;
    std::vector<float> filter(pre_pad + prototype_taps.size(), 0.0F);
    for (std::size_t tap = 0; tap < prototype_taps.size(); ++tap) {
        filter[pre_pad + tap] = prototype_taps[tap] * static_cast<float>(up_factor_);
    }
    padded_filter_tap_count_ = filter.size();
    phase_filters_.resize(up_factor_);
    for (std::size_t phase = 0; phase < up_factor_; ++phase) {
        auto& phase_filter = phase_filters_[phase];
        phase_filter.reserve((filter.size() + up_factor_ - 1 - phase) / up_factor_);
        for (std::size_t tap = phase; tap < filter.size(); tap += up_factor_) {
            phase_filter.push_back(filter[tap]);
        }
    }
}

void RationalResamplerCf32::reset() {
    history_.clear();
    history_base_sample_ = 0;
    processed_input_samples_ = 0;
    produced_output_samples_ = 0;
    max_worker_count_ = 0;
    finished_ = false;
}

void RationalResamplerCf32::process(
    std::span<const float> interleaved_input,
    std::vector<float>& interleaved_output) {
    if (finished_) {
        throw std::logic_error("rational resampler cannot process after finish");
    }
    if ((interleaved_input.size() % 2) != 0) {
        throw std::invalid_argument("rational resampler input must contain CF32 pairs");
    }
    const auto input_samples = interleaved_input.size() / 2;
    if (processed_input_samples_
        > std::numeric_limits<std::size_t>::max() - input_samples) {
        throw std::overflow_error("rational resampler input count overflow");
    }
    history_.insert(history_.end(), interleaved_input.begin(), interleaved_input.end());
    processed_input_samples_ += input_samples;
    emit_available(false, interleaved_output);
    compact_history();
}

void RationalResamplerCf32::finish(std::vector<float>& interleaved_output) {
    if (finished_) {
        return;
    }
    finished_ = true;
    emit_available(true, interleaved_output);
    history_.clear();
    history_base_sample_ = processed_input_samples_;
}

std::size_t RationalResamplerCf32::up_factor() const noexcept {
    return up_factor_;
}

std::size_t RationalResamplerCf32::down_factor() const noexcept {
    return down_factor_;
}

std::size_t RationalResamplerCf32::prototype_tap_count() const noexcept {
    return prototype_tap_count_;
}

std::size_t RationalResamplerCf32::processed_input_samples() const noexcept {
    return processed_input_samples_;
}

std::size_t RationalResamplerCf32::produced_output_samples() const noexcept {
    return produced_output_samples_;
}

std::size_t RationalResamplerCf32::max_worker_count() const noexcept {
    return max_worker_count_;
}

void RationalResamplerCf32::emit_available(
    bool finishing,
    std::vector<float>& interleaved_output) {
    if ((interleaved_output.size() % 2) != 0) {
        throw std::invalid_argument("rational resampler output must contain CF32 pairs");
    }
    if (processed_input_samples_
        > std::numeric_limits<std::size_t>::max() / up_factor_) {
        throw std::overflow_error("rational resampler output count overflow");
    }
    const auto numerator = processed_input_samples_ * up_factor_;
    const auto zero_padded_output_samples = numerator / down_factor_
        + (numerator % down_factor_ != 0 ? 1U : 0U);
    const auto target_output_samples = finishing
        ? zero_padded_output_samples
        : zero_padded_output_samples > pre_remove_output_samples_
            ? zero_padded_output_samples - pre_remove_output_samples_
            : 0U;
    if (target_output_samples <= produced_output_samples_) {
        return;
    }

    const auto output_count = target_output_samples - produced_output_samples_;
    const auto destination_sample_offset = interleaved_output.size() / 2;
    if (output_count > (std::numeric_limits<std::size_t>::max()
            - interleaved_output.size()) / 2) {
        throw std::overflow_error("rational resampler output vector overflow");
    }
    interleaved_output.resize(interleaved_output.size() + output_count * 2);

    auto worker_count = requested_workers_;
    if (worker_count == 0) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (output_count < min_parallel_output_samples_) {
        worker_count = 1;
    }
    worker_count = std::clamp<std::size_t>(worker_count, 1, output_count);
    max_worker_count_ = std::max(max_worker_count_, worker_count);
    if (worker_count == 1) {
        emit_range(
            produced_output_samples_,
            target_output_samples,
            destination_sample_offset,
            interleaved_output);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const auto first = produced_output_samples_ + output_count * worker / worker_count;
            const auto last =
                produced_output_samples_ + output_count * (worker + 1) / worker_count;
            const auto destination =
                destination_sample_offset + first - produced_output_samples_;
            workers.emplace_back([this, first, last, destination, &interleaved_output]() {
                emit_range(first, last, destination, interleaved_output);
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }
    produced_output_samples_ = target_output_samples;
}

void RationalResamplerCf32::emit_range(
    std::size_t first_output_sample,
    std::size_t last_output_sample,
    std::size_t destination_sample_offset,
    std::span<float> interleaved_output) const {
    for (auto output_sample = first_output_sample;
         output_sample < last_output_sample;
         ++output_sample) {
        const auto time = checked_time_index(
            output_sample,
            pre_remove_output_samples_,
            down_factor_);
        const auto signed_time = static_cast<std::int64_t>(time);
        const auto maximum_input = floor_div(
            signed_time,
            static_cast<std::int64_t>(up_factor_));
        const auto& phase_filter = phase_filters_[time % up_factor_];
        const auto minimum_input = phase_filter.empty()
            ? maximum_input + 1
            : maximum_input - static_cast<std::int64_t>(phase_filter.size() - 1);

        float value_real = 0.0F;
        float value_imag = 0.0F;
        const auto first_input = std::max<std::int64_t>(
            minimum_input,
            static_cast<std::int64_t>(history_base_sample_));
        const auto last_input = std::min<std::int64_t>(
            maximum_input,
            static_cast<std::int64_t>(processed_input_samples_) - 1);
        for (auto input = first_input; input <= last_input; ++input) {
            const auto filter_index =
                static_cast<std::size_t>(maximum_input - input);
            const auto history_index =
                static_cast<std::size_t>(input) - history_base_sample_;
            const auto coefficient = phase_filter[filter_index];
            value_real += history_[history_index * 2] * coefficient;
            value_imag += history_[history_index * 2 + 1] * coefficient;
        }
        const auto destination = destination_sample_offset
            + output_sample - first_output_sample;
        interleaved_output[destination * 2] = value_real;
        interleaved_output[destination * 2 + 1] = value_imag;
    }
}

void RationalResamplerCf32::compact_history() {
    if (history_.empty()) {
        return;
    }
    const auto time = checked_time_index(
        produced_output_samples_,
        pre_remove_output_samples_,
        down_factor_);
    const auto minimum_input = ceil_div(
        static_cast<std::int64_t>(time)
            - static_cast<std::int64_t>(padded_filter_tap_count_ - 1),
        static_cast<std::int64_t>(up_factor_));
    const auto keep_from = std::clamp<std::int64_t>(
        minimum_input,
        static_cast<std::int64_t>(history_base_sample_),
        static_cast<std::int64_t>(processed_input_samples_));
    const auto drop_samples =
        static_cast<std::size_t>(keep_from) - history_base_sample_;
    if (drop_samples == 0) {
        return;
    }
    history_.erase(
        history_.begin(),
        history_.begin() + static_cast<std::ptrdiff_t>(drop_samples * 2));
    history_base_sample_ += drop_samples;
}

}  // namespace dtmb::core
