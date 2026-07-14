#include "dtmb/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace dtmb::core {
namespace {

using Complex = std::complex<float>;

constexpr std::size_t kPn945CoreSymbols = 511;
constexpr std::size_t kPn945PrefixSymbols = 217;
constexpr std::string_view kPn945Seed = "111110111";
constexpr std::array<std::size_t, 4> kPn945RecurrenceTaps{0, 1, 2, 7};

[[nodiscard]] std::array<std::uint8_t, kPn945HeaderSymbols> pn945_phase0_bits() {
    std::array<std::uint8_t, kPn945HeaderSymbols> bits{};
    for (std::size_t index = 0; index < kPn945Seed.size(); ++index) {
        bits[index] = static_cast<std::uint8_t>(kPn945Seed[index] - '0');
    }
    for (std::size_t index = kPn945Seed.size(); index < bits.size(); ++index) {
        std::uint8_t value = 0;
        const auto window = index - kPn945Seed.size();
        for (const auto tap : kPn945RecurrenceTaps) {
            value ^= bits[window + tap];
        }
        bits[index] = value;
    }
    return bits;
}

[[nodiscard]] const std::array<std::uint8_t, kPn945CoreSymbols>& pn945_core_bits() {
    static const auto core = [] {
        const auto phase0 = pn945_phase0_bits();
        std::array<std::uint8_t, kPn945CoreSymbols> result{};
        std::copy_n(
            phase0.begin() + kPn945PrefixSymbols,
            kPn945CoreSymbols,
            result.begin());
        return result;
    }();
    return core;
}

[[nodiscard]] const std::array<std::int8_t, kPn945CoreSymbols>& pn945_core_chips() {
    static const auto chips = [] {
        const auto& bits = pn945_core_bits();
        std::array<std::int8_t, kPn945CoreSymbols> result{};
        for (std::size_t index = 0; index < kPn945CoreSymbols; ++index) {
            result[index] = bits[index] == 0 ? 1 : -1;
        }
        return result;
    }();
    return chips;
}

[[nodiscard]] Complex pn945_core_symbol(std::size_t phase, std::size_t core_index) {
    const auto chip = static_cast<float>(
        pn945_core_chips()[(core_index + phase) % kPn945CoreSymbols]);
    return Complex{chip, chip};
}

[[nodiscard]] Complex pn945_header_symbol(std::size_t phase, std::size_t header_index) {
    if (header_index < kPn945PrefixSymbols) {
        return pn945_core_symbol(
            phase,
            kPn945CoreSymbols - kPn945PrefixSymbols + header_index);
    }
    if (header_index < kPn945PrefixSymbols + kPn945CoreSymbols) {
        return pn945_core_symbol(phase, header_index - kPn945PrefixSymbols);
    }
    return pn945_core_symbol(
        phase,
        header_index - kPn945PrefixSymbols - kPn945CoreSymbols);
}

[[nodiscard]] const std::vector<float>& pn945_core_reference_fft(std::size_t phase) {
    static const auto cache = [] {
        std::array<std::vector<float>, kPn945CoreSymbols> result{};
        for (std::size_t cached_phase = 0; cached_phase < kPn945CoreSymbols; ++cached_phase) {
            std::vector<float> reference(kPn945CoreSymbols * 2);
            for (std::size_t index = 0; index < kPn945CoreSymbols; ++index) {
                const auto value = pn945_core_symbol(cached_phase, index);
                reference[index * 2] = value.real();
                reference[index * 2 + 1] = value.imag();
            }
            result[cached_phase].resize(kPn945CoreSymbols * 2);
            mixed_radix_fft_forward_cf32(reference, result[cached_phase]);
        }
        return result;
    }();
    return cache[phase % kPn945CoreSymbols];
}

struct PnPhaseProjection {
    std::size_t phase = 0;
    Complex projection{};
};

struct PnHeaderCoreAccumulator {
    std::array<float, kPn945CoreSymbols> real{};
    std::array<float, kPn945CoreSymbols> imag{};
};

[[nodiscard]] PnHeaderCoreAccumulator accumulate_pn945_header_by_core_index(
    std::span<const float> interleaved_header) {
    PnHeaderCoreAccumulator accum{};
    for (std::size_t core_index = 0; core_index < kPn945CoreSymbols; ++core_index) {
        const auto main_sample = kPn945PrefixSymbols + core_index;
        accum.real[core_index] = interleaved_header[main_sample * 2];
        accum.imag[core_index] = interleaved_header[main_sample * 2 + 1];
    }
    for (std::size_t core_index = 0; core_index < kPn945PrefixSymbols; ++core_index) {
        const auto post_sample = kPn945PrefixSymbols + kPn945CoreSymbols + core_index;
        accum.real[core_index] += interleaved_header[post_sample * 2];
        accum.imag[core_index] += interleaved_header[post_sample * 2 + 1];
    }
    for (std::size_t core_index = kPn945CoreSymbols - kPn945PrefixSymbols;
         core_index < kPn945CoreSymbols;
         ++core_index) {
        const auto prefix_sample = core_index - (kPn945CoreSymbols - kPn945PrefixSymbols);
        accum.real[core_index] += interleaved_header[prefix_sample * 2];
        accum.imag[core_index] += interleaved_header[prefix_sample * 2 + 1];
    }
    return accum;
}

[[nodiscard]] PnPhaseProjection pn945_bipolar_phase_projection(
    std::span<const float> interleaved_header) {
    const auto accum = accumulate_pn945_header_by_core_index(interleaved_header);
    const auto& chips = pn945_core_chips();
    std::size_t best_phase = 0;
    auto best_projection = Complex{};
    float best_power = -1.0F;
    for (std::size_t phase = 0; phase < kPn945CoreSymbols; ++phase) {
        float projection_real = 0.0F;
        float projection_imag = 0.0F;
        auto chip_index = phase;
        for (std::size_t core_index = 0; core_index < kPn945CoreSymbols; ++core_index) {
            if (chips[chip_index] > 0) {
                projection_real += accum.real[core_index];
                projection_imag += accum.imag[core_index];
            } else {
                projection_real -= accum.real[core_index];
                projection_imag -= accum.imag[core_index];
            }
            ++chip_index;
            if (chip_index == kPn945CoreSymbols) {
                chip_index = 0;
            }
        }
        const auto projection = Complex{projection_real, projection_imag};
        const auto power = std::norm(projection);
        if (power > best_power) {
            best_power = power;
            best_phase = phase;
            best_projection = projection;
        }
    }
    return PnPhaseProjection{best_phase, best_projection};
}

[[nodiscard]] std::size_t choose_residual_cfo_worker_count(
    std::size_t frame_count,
    std::size_t requested_workers) noexcept {
    if (frame_count == 0) {
        return 0;
    }
    auto worker_count = requested_workers;
    if (worker_count == 0) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    return std::clamp<std::size_t>(worker_count, 1, frame_count);
}

void inverse_fft_cf32(
    std::span<const float> interleaved_frequency_bins,
    std::span<float> interleaved_time_samples) {
    std::vector<float> conjugated(interleaved_frequency_bins.size());
    for (std::size_t index = 0; index < interleaved_frequency_bins.size() / 2; ++index) {
        conjugated[index * 2] = interleaved_frequency_bins[index * 2];
        conjugated[index * 2 + 1] = -interleaved_frequency_bins[index * 2 + 1];
    }
    mixed_radix_fft_forward_cf32(conjugated, interleaved_time_samples);
    const auto size = static_cast<float>(interleaved_frequency_bins.size() / 2);
    for (std::size_t index = 0; index < interleaved_time_samples.size() / 2; ++index) {
        interleaved_time_samples[index * 2] /= size;
        interleaved_time_samples[index * 2 + 1] =
            -interleaved_time_samples[index * 2 + 1] / size;
    }
}

[[nodiscard]] float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0F;
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if ((values.size() % 2) != 0) {
        return *middle;
    }
    const auto lower = std::max_element(values.begin(), middle);
    return (*lower + *middle) * 0.5F;
}

void repair_low_energy_pn_dc_response(
    std::span<float> response_fft,
    std::span<const float> reference_fft) {
    std::vector<float> reference_power;
    reference_power.reserve(kPn945CoreSymbols - 1);
    for (std::size_t bin = 1; bin < kPn945CoreSymbols; ++bin) {
        reference_power.push_back(std::norm(Complex{
            reference_fft[bin * 2],
            reference_fft[bin * 2 + 1],
        }));
    }
    const auto typical_reference_power = median(std::move(reference_power));
    const auto dc_reference_power = std::norm(Complex{
        reference_fft[0],
        reference_fft[1],
    });
    if (typical_reference_power <= 0.0F
        || dc_reference_power > typical_reference_power * 0.01F) {
        return;
    }

    std::vector<float> provisional_taps(response_fft.size());
    inverse_fft_cf32(response_fft, provisional_taps);
    float peak = 0.0F;
    for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
        peak = std::max(peak, std::abs(Complex{
            provisional_taps[tap * 2],
            provisional_taps[tap * 2 + 1],
        }));
    }
    if (peak <= 0.0F) {
        return;
    }
    std::size_t last_significant = 0;
    for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
        if (std::abs(Complex{
                provisional_taps[tap * 2],
                provisional_taps[tap * 2 + 1],
            }) >= peak * 0.05F) {
            last_significant = tap;
        }
    }
    if (last_significant > 64) {
        return;
    }
    const auto interpolated = (
        Complex{response_fft[2], response_fft[3]}
        + Complex{
            response_fft[(kPn945CoreSymbols - 1) * 2],
            response_fft[(kPn945CoreSymbols - 1) * 2 + 1],
        }) * 0.5F;
    const auto current = Complex{response_fft[0], response_fft[1]};
    const auto scale = std::max(std::abs(interpolated), 1.0e-12F);
    if (std::abs(current - interpolated) <= scale * 0.25F) {
        return;
    }
    response_fft[0] = interpolated.real();
    response_fft[1] = interpolated.imag();
}

[[nodiscard]] std::vector<Complex> estimate_channel_taps(
    std::span<const float> header,
    std::size_t phase,
    std::size_t channel_taps,
    float regularization,
    Pn945HeaderObservation observation = Pn945HeaderObservation::core_only,
    std::size_t safe_prefix_skip = kPn945PrefixSymbols) {
    std::vector<float> observed(kPn945CoreSymbols * 2);
    for (std::size_t index = 0; index < kPn945CoreSymbols; ++index) {
        const auto main_sample = kPn945PrefixSymbols + index;
        auto value = Complex{
            header[main_sample * 2],
            header[main_sample * 2 + 1],
        };
        std::size_t observation_count = 1;
        if (observation != Pn945HeaderObservation::core_only
            && index < kPn945PrefixSymbols) {
            const auto postfix_sample =
                kPn945PrefixSymbols + kPn945CoreSymbols + index;
            value += Complex{
                header[postfix_sample * 2],
                header[postfix_sample * 2 + 1],
            };
            ++observation_count;
        }
        constexpr auto prefix_core_start = kPn945CoreSymbols - kPn945PrefixSymbols;
        if (observation == Pn945HeaderObservation::core_cyclic_safe_average
            && index >= prefix_core_start + safe_prefix_skip) {
            const auto prefix_sample = index - prefix_core_start;
            value += Complex{
                header[prefix_sample * 2],
                header[prefix_sample * 2 + 1],
            };
            ++observation_count;
        }
        value /= static_cast<float>(observation_count);
        observed[index * 2] = value.real();
        observed[index * 2 + 1] = value.imag();
    }

    std::vector<float> observed_fft(observed.size());
    const auto& reference_fft = pn945_core_reference_fft(phase);
    std::vector<float> response_fft(reference_fft.size());
    std::vector<float> taps_cf32(reference_fft.size());
    mixed_radix_fft_forward_cf32(observed, observed_fft);
    for (std::size_t bin = 0; bin < kPn945CoreSymbols; ++bin) {
        const auto obs = Complex{observed_fft[bin * 2], observed_fft[bin * 2 + 1]};
        const auto ref = Complex{reference_fft[bin * 2], reference_fft[bin * 2 + 1]};
        const auto response = obs * std::conj(ref) / (std::norm(ref) + regularization);
        response_fft[bin * 2] = response.real();
        response_fft[bin * 2 + 1] = response.imag();
    }
    repair_low_energy_pn_dc_response(response_fft, reference_fft);
    inverse_fft_cf32(response_fft, taps_cf32);

    std::vector<Complex> taps(
        std::min(channel_taps, kPn945CoreSymbols),
        Complex{0.0F, 0.0F});
    for (std::size_t tap = 0; tap < taps.size(); ++tap) {
        taps[tap] = Complex{taps_cf32[tap * 2], taps_cf32[tap * 2 + 1]};
    }
    return taps;
}

[[nodiscard]] std::vector<Complex> rotate_left(
    std::span<const Complex> values,
    std::size_t shift) {
    std::vector<Complex> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[(index + shift) % values.size()];
    }
    return result;
}

[[nodiscard]] std::vector<bool> rotate_left(
    const std::vector<bool>& values,
    std::size_t shift) {
    std::vector<bool> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[(index + shift) % values.size()];
    }
    return result;
}

[[nodiscard]] std::pair<std::size_t, std::size_t> longest_circular_false_run(
    const std::vector<bool>& mask) {
    if (mask.empty()) {
        return {0, 0};
    }
    if (std::all_of(mask.begin(), mask.end(), [](bool value) { return value; })) {
        return {0, 0};
    }
    if (std::none_of(mask.begin(), mask.end(), [](bool value) { return value; })) {
        return {0, mask.size()};
    }
    std::size_t best_start = 0;
    std::size_t best_length = 0;
    std::size_t run_start = 0;
    bool in_run = false;
    for (std::size_t index = 0; index < mask.size() * 2; ++index) {
        if (!mask[index % mask.size()]) {
            if (!in_run) {
                run_start = index;
                in_run = true;
            }
            const auto length = index - run_start + 1;
            if (run_start < mask.size() && length > best_length) {
                best_start = run_start;
                best_length = length;
            }
        } else {
            in_run = false;
        }
    }
    return {best_start % mask.size(), std::min(best_length, mask.size())};
}

void dilate_linear_mask(std::vector<bool>& mask, std::size_t guard_taps) {
    if (guard_taps == 0) {
        return;
    }
    const auto source = mask;
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (!source[index]) {
            continue;
        }
        const auto start = index > guard_taps ? index - guard_taps : 0;
        const auto stop = std::min(source.size(), index + guard_taps + 1);
        std::fill(mask.begin() + static_cast<std::ptrdiff_t>(start),
                  mask.begin() + static_cast<std::ptrdiff_t>(stop),
                  true);
    }
}

[[nodiscard]] Complex bounded_wideband_response_scale(
    Complex anchor,
    Complex reference,
    float per_frame_noise_tap_power) {
    auto scale = Complex{1.0F, 0.0F};
    const auto anchor_power = std::norm(anchor);
    if (std::abs(reference) > 0.0F
        && anchor_power > 4.0F * per_frame_noise_tap_power) {
        scale = anchor / reference;
        const auto magnitude = std::abs(scale);
        if (magnitude > 4.0F || magnitude < 0.25F) {
            scale /= magnitude;
        }
    }
    return scale;
}

[[nodiscard]] Complex bounded_wideband_response_scale_from_sums(
    Complex cross_sum,
    float reference_power_sum,
    float observed_power_sum,
    std::size_t tap_count,
    float per_frame_noise_tap_power) {
    auto scale = Complex{1.0F, 0.0F};
    if (reference_power_sum > 0.0F
        && tap_count > 0
        && observed_power_sum
            > 4.0F * per_frame_noise_tap_power * static_cast<float>(tap_count)) {
        scale = cross_sum / reference_power_sum;
        const auto magnitude = std::abs(scale);
        if (magnitude > 4.0F || magnitude < 0.25F) {
            scale /= magnitude;
        }
    }
    return scale;
}

[[nodiscard]] Complex wideband_response_scale_for_rotated_taps(
    std::span<const Complex> rotated_taps,
    const Pn945WidebandChannelModel& model,
    Pn945WidebandScaleEstimator estimator) {
    const auto template_size = model.template_taps.size() / 2;
    if (template_size == 0 || model.dominant_tap_index >= template_size) {
        throw std::invalid_argument("wideband PN945 model has invalid template taps");
    }
    if (estimator == Pn945WidebandScaleEstimator::dominant_tap) {
        const auto anchor = rotated_taps[model.dominant_tap_index];
        const auto reference = Complex{
            model.template_taps[model.dominant_tap_index * 2],
            model.template_taps[model.dominant_tap_index * 2 + 1],
        };
        return bounded_wideband_response_scale(
            anchor,
            reference,
            model.per_frame_noise_tap_power);
    }

    Complex cross_sum{0.0F, 0.0F};
    float reference_power_sum = 0.0F;
    float observed_power_sum = 0.0F;
    std::size_t tap_count = 0;
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        const auto reference = Complex{
            model.template_taps[tap * 2],
            model.template_taps[tap * 2 + 1],
        };
        const auto reference_power = std::norm(reference);
        if (reference_power <= 0.0F) {
            continue;
        }
        const auto observed = rotated_taps[tap];
        cross_sum += observed * std::conj(reference);
        reference_power_sum += reference_power;
        observed_power_sum += std::norm(observed);
        ++tap_count;
    }
    return bounded_wideband_response_scale_from_sums(
        cross_sum,
        reference_power_sum,
        observed_power_sum,
        tap_count,
        model.per_frame_noise_tap_power);
}

[[nodiscard]] bool uses_masked_frame_taps(Pn945WidebandScaleEstimator estimator) noexcept {
    return estimator == Pn945WidebandScaleEstimator::masked_frame_taps;
}

[[nodiscard]] std::vector<Complex> masked_frame_taps_from_rotated_taps(
    std::span<const Complex> rotated_taps,
    const Pn945WidebandChannelModel& model) {
    const auto template_size = model.template_taps.size() / 2;
    if (template_size == 0
        || model.template_tap_mask.size() != template_size
        || rotated_taps.size() < template_size) {
        throw std::invalid_argument("wideband PN945 frame-tap model has invalid mask");
    }
    std::vector<Complex> taps(template_size);
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        if (model.template_tap_mask[tap] != 0U) {
            taps[tap] = rotated_taps[tap];
        }
    }
    return taps;
}

[[nodiscard]] std::vector<Complex> masked_frame_taps_for_model_frame(
    const Pn945WidebandChannelModel& model,
    std::size_t model_frame_index) {
    const auto template_size = model.template_taps.size() / 2;
    if (template_size == 0
        || (model_frame_index + 1) * template_size * 2 > model.frame_template_taps.size()) {
        throw std::invalid_argument("wideband PN945 cached frame taps are missing");
    }
    std::vector<Complex> taps(template_size);
    const auto base = model_frame_index * template_size * 2;
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        taps[tap] = Complex{
            model.frame_template_taps[base + tap * 2],
            model.frame_template_taps[base + tap * 2 + 1],
        };
    }
    return taps;
}

[[nodiscard]] std::vector<Complex> scaled_template_taps(
    const Pn945WidebandChannelModel& model,
    Complex scale) {
    const auto template_size = model.template_taps.size() / 2;
    std::vector<Complex> result(template_size);
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        result[tap] = Complex{
            model.template_taps[tap * 2],
            model.template_taps[tap * 2 + 1],
        } * scale;
    }
    return result;
}

[[nodiscard]] std::vector<Complex> instantiate_wideband_taps(
    std::span<const float> header,
    const Pn945WidebandChannelModel& model,
    float regularization,
    std::size_t& pn_phase,
    Complex& scale) {
    const auto header_phase = pn945_detect_phase_cf32(header);
    const auto raw_taps = estimate_channel_taps(
        header,
        header_phase,
        kPn945CoreSymbols,
        regularization,
        model.header_observation);
    const auto rotated = rotate_left(raw_taps, model.rotation_symbols);
    if (uses_masked_frame_taps(model.scale_estimator)) {
        scale = Complex{1.0F, 0.0F};
        auto result = masked_frame_taps_from_rotated_taps(rotated, model);
        pn_phase = (header_phase + kPn945CoreSymbols - model.rotation_symbols)
            % kPn945CoreSymbols;
        return result;
    }
    scale = wideband_response_scale_for_rotated_taps(
        rotated,
        model,
        model.scale_estimator);
    auto result = scaled_template_taps(model, scale);
    pn_phase = (header_phase + kPn945CoreSymbols - model.rotation_symbols)
        % kPn945CoreSymbols;
    return result;
}

[[nodiscard]] std::size_t wideband_pn_phase_only(
    std::span<const float> header,
    const Pn945WidebandChannelModel& model) {
    const auto header_phase = pn945_detect_phase_cf32(header);
    return (header_phase + kPn945CoreSymbols - model.rotation_symbols)
        % kPn945CoreSymbols;
}

[[nodiscard]] Complex convolved_pn_header_sample(
    std::size_t phase,
    std::span<const Complex> taps,
    std::size_t output_index) {
    if (taps.empty()) {
        return {};
    }
    auto sum = Complex{};
    const auto max_tap = std::min(output_index, taps.size() - 1);
    for (std::size_t tap_count = max_tap + 1; tap_count > 0; --tap_count) {
        const auto tap = tap_count - 1;
        const auto header_index = output_index - tap;
        if (header_index >= kPn945HeaderSymbols) {
            continue;
        }
        sum += pn945_header_symbol(phase, header_index) * taps[tap];
    }
    return sum;
}

[[nodiscard]] std::vector<Complex> midpoint_taps(
    std::span<const Complex> current_taps,
    std::span<const Complex> next_taps) {
    if (current_taps.size() != next_taps.size()) {
        throw std::invalid_argument("PN945 midpoint tap spans must match");
    }
    std::vector<Complex> result(current_taps.size());
    for (std::size_t tap = 0; tap < result.size(); ++tap) {
        result[tap] = 0.5F * (current_taps[tap] + next_taps[tap]);
    }
    return result;
}

void restore_and_equalize_transition(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    std::size_t phase,
    std::size_t next_phase,
    std::span<const Complex> current_taps,
    std::span<const Complex> next_taps,
    std::span<const Complex> body_taps,
    float response_floor,
    float noise_variance,
    int response_window_offset,
    bool mmse_unbias,
    float mmse_unbias_gain_floor,
    std::span<float> interleaved_channel_power) {
    if (current_taps.empty() || next_taps.empty() || body_taps.empty()) {
        throw std::invalid_argument("PN945 transition equalizer taps must not be empty");
    }
    if (!interleaved_channel_power.empty()
        && interleaved_channel_power.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("PN945 channel-power output span is too small");
    }
    if (mmse_unbias
        && (!std::isfinite(mmse_unbias_gain_floor)
            || mmse_unbias_gain_floor <= 0.0F
            || mmse_unbias_gain_floor > 1.0F)) {
        throw std::invalid_argument("PN945 MMSE unbias gain floor must be in (0,1]");
    }
    thread_local std::vector<float> restored;
    restored.assign(interleaved_time_body.begin(), interleaved_time_body.end());
    const auto tail_length = std::min<std::size_t>(
        std::max(current_taps.size(), next_taps.size()) - 1,
        kC3780FrameBodySymbols);
    for (std::size_t sample = 0; sample < tail_length; ++sample) {
        auto body = Complex{restored[sample * 2], restored[sample * 2 + 1]};
        body -= convolved_pn_header_sample(
            phase,
            current_taps,
            kPn945HeaderSymbols + sample);
        const auto observed_next = Complex{
            interleaved_next_header[sample * 2],
            interleaved_next_header[sample * 2 + 1],
        };
        body += observed_next
            - convolved_pn_header_sample(next_phase, next_taps, sample);
        restored[sample * 2] = body.real();
        restored[sample * 2 + 1] = body.imag();
    }

    thread_local std::vector<float> padded_taps;
    thread_local std::vector<float> response;
    padded_taps.assign(kC3780FrameBodySymbols * 2, 0.0F);
    response.resize(kC3780FrameBodySymbols * 2);
    mixed_radix_fft_forward_cf32(restored, interleaved_equalized_spectrum);
    for (std::size_t tap = 0; tap < body_taps.size(); ++tap) {
        padded_taps[tap * 2] = body_taps[tap].real();
        padded_taps[tap * 2 + 1] = body_taps[tap].imag();
    }
    mixed_radix_fft_forward_cf32(padded_taps, response);
    for (std::size_t bin = 0; bin < kC3780FrameBodySymbols; ++bin) {
        const auto value = Complex{
            interleaved_equalized_spectrum[bin * 2],
            interleaved_equalized_spectrum[bin * 2 + 1],
        };
        auto channel = Complex{response[bin * 2], response[bin * 2 + 1]};
        if (response_window_offset != 0) {
            const auto angle = 2.0F * std::numbers::pi_v<float>
                * static_cast<float>(response_window_offset)
                * static_cast<float>(bin)
                / static_cast<float>(kC3780FrameBodySymbols);
            channel *= Complex{std::cos(angle), std::sin(angle)};
        }
        const auto channel_power = std::norm(channel);
        if (!interleaved_channel_power.empty()) {
            interleaved_channel_power[bin * 2] = channel_power;
            interleaved_channel_power[bin * 2 + 1] = 0.0F;
        }
        Complex equalized;
        if (noise_variance >= 0.0F) {
            equalized = value * std::conj(channel)
                / (channel_power + noise_variance);
            if (mmse_unbias) {
                const auto gain = channel_power / (channel_power + noise_variance);
                equalized /= std::max(gain, mmse_unbias_gain_floor);
            }
        } else {
            equalized = std::abs(channel) >= response_floor
                ? value / channel
                : value;
        }
        interleaved_equalized_spectrum[bin * 2] = equalized.real();
        interleaved_equalized_spectrum[bin * 2 + 1] = equalized.imag();
    }
}

void restore_and_equalize(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    std::size_t phase,
    std::size_t next_phase,
    std::span<const Complex> taps,
    float response_floor,
    float noise_variance,
    int response_window_offset,
    bool mmse_unbias,
    float mmse_unbias_gain_floor,
    std::span<float> interleaved_channel_power) {
    restore_and_equalize_transition(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        taps,
        taps,
        response_floor,
        noise_variance,
        response_window_offset,
        mmse_unbias,
        mmse_unbias_gain_floor,
        interleaved_channel_power);
}

void restore_and_equalize_with_template_response(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    std::size_t phase,
    std::size_t next_phase,
    std::span<const Complex> taps,
    std::span<const float> template_response_fft,
    Complex response_scale,
    float response_floor,
    float noise_variance,
    int response_window_offset,
    bool mmse_unbias,
    float mmse_unbias_gain_floor,
    std::span<float> interleaved_channel_power) {
    if (template_response_fft.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("wideband PN945 template response FFT is too small");
    }
    if (!interleaved_channel_power.empty()
        && interleaved_channel_power.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("PN945 channel-power output span is too small");
    }
    if (mmse_unbias
        && (!std::isfinite(mmse_unbias_gain_floor)
            || mmse_unbias_gain_floor <= 0.0F
            || mmse_unbias_gain_floor > 1.0F)) {
        throw std::invalid_argument("PN945 MMSE unbias gain floor must be in (0,1]");
    }
    thread_local std::vector<float> restored;
    restored.assign(interleaved_time_body.begin(), interleaved_time_body.end());
    const auto tail_length = std::min<std::size_t>(
        taps.size() - 1,
        kC3780FrameBodySymbols);
    for (std::size_t sample = 0; sample < tail_length; ++sample) {
        auto body = Complex{restored[sample * 2], restored[sample * 2 + 1]};
        body -= convolved_pn_header_sample(phase, taps, kPn945HeaderSymbols + sample);
        const auto observed_next = Complex{
            interleaved_next_header[sample * 2],
            interleaved_next_header[sample * 2 + 1],
        };
        body += observed_next - convolved_pn_header_sample(next_phase, taps, sample);
        restored[sample * 2] = body.real();
        restored[sample * 2 + 1] = body.imag();
    }

    mixed_radix_fft_forward_cf32(restored, interleaved_equalized_spectrum);
    for (std::size_t bin = 0; bin < kC3780FrameBodySymbols; ++bin) {
        const auto value = Complex{
            interleaved_equalized_spectrum[bin * 2],
            interleaved_equalized_spectrum[bin * 2 + 1],
        };
        auto channel = Complex{
            template_response_fft[bin * 2],
            template_response_fft[bin * 2 + 1],
        } * response_scale;
        if (response_window_offset != 0) {
            const auto angle = 2.0F * std::numbers::pi_v<float>
                * static_cast<float>(response_window_offset)
                * static_cast<float>(bin)
                / static_cast<float>(kC3780FrameBodySymbols);
            channel *= Complex{std::cos(angle), std::sin(angle)};
        }
        const auto channel_power = std::norm(channel);
        if (!interleaved_channel_power.empty()) {
            interleaved_channel_power[bin * 2] = channel_power;
            interleaved_channel_power[bin * 2 + 1] = 0.0F;
        }
        Complex equalized;
        if (noise_variance >= 0.0F) {
            equalized = value * std::conj(channel)
                / (channel_power + noise_variance);
            if (mmse_unbias) {
                const auto gain = channel_power / (channel_power + noise_variance);
                equalized /= std::max(gain, mmse_unbias_gain_floor);
            }
        } else {
            equalized = std::abs(channel) >= response_floor
                ? value / channel
                : value;
        }
        interleaved_equalized_spectrum[bin * 2] = equalized.real();
        interleaved_equalized_spectrum[bin * 2 + 1] = equalized.imag();
    }
}

}  // namespace

std::size_t pn945_detect_phase_cf32(std::span<const float> interleaved_header) {
    if (interleaved_header.size() != kPn945HeaderSymbols * 2) {
        throw std::invalid_argument("PN945 header must contain exactly 945 CF32 samples");
    }
    return pn945_bipolar_phase_projection(interleaved_header).phase;
}

Pn945ResidualCfoResult estimate_pn945_residual_cfo_cf32(
    std::span<const float> interleaved_symbols,
    std::size_t phase_offset,
    Pn945ResidualCfoOptions options) {
    if ((interleaved_symbols.size() % 2) != 0) {
        throw std::invalid_argument("PN945 residual-CFO input must contain CF32 pairs");
    }
    if (options.max_frames < 2) {
        throw std::invalid_argument("PN945 residual-CFO max_frames must be at least 2");
    }
    if (options.min_fit_r_squared < 0.0F || options.min_fit_r_squared > 1.0F) {
        throw std::invalid_argument("PN945 residual-CFO min_fit_r_squared must be in [0, 1]");
    }

    const auto sample_count = interleaved_symbols.size() / 2;
    if (phase_offset + kPn945HeaderSymbols > sample_count) {
        return {};
    }
    const auto available_frames =
        1 + (sample_count - phase_offset - kPn945HeaderSymbols) / kPn945FrameSymbols;
    const auto frame_count = std::min(options.max_frames, available_frames);
    if (frame_count < 2) {
        return {};
    }

    std::vector<PnPhaseProjection> projections(frame_count);
    const auto worker_count = choose_residual_cfo_worker_count(
        frame_count,
        options.requested_workers);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::size_t frame = worker; frame < frame_count; frame += worker_count) {
                const auto start = phase_offset + frame * kPn945FrameSymbols;
                const auto header = std::span<const float>(
                    interleaved_symbols.data() + start * 2,
                    kPn945HeaderSymbols * 2);
                projections[frame] = pn945_bipolar_phase_projection(header);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    std::vector<double> phases;
    std::vector<double> weights;
    std::vector<double> frame_indices;
    phases.reserve(frame_count);
    weights.reserve(frame_count);
    frame_indices.reserve(frame_count);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const auto weight = std::abs(projections[frame].projection);
        if (weight <= 0.0F) {
            continue;
        }
        auto phase = static_cast<double>(std::arg(projections[frame].projection));
        if (!phases.empty()) {
            while (phase - phases.back() > std::numbers::pi_v<double>) {
                phase -= 2.0 * std::numbers::pi_v<double>;
            }
            while (phase - phases.back() < -std::numbers::pi_v<double>) {
                phase += 2.0 * std::numbers::pi_v<double>;
            }
        }
        phases.push_back(phase);
        weights.push_back(weight);
        frame_indices.push_back(static_cast<double>(frame));
    }
    if (phases.size() < 2) {
        return Pn945ResidualCfoResult{
            0.0F,
            0.0F,
            phases.size(),
            worker_count,
            false,
        };
    }

    const auto max_weight = *std::max_element(weights.begin(), weights.end());
    if (max_weight > 0.0) {
        for (auto& weight : weights) {
            weight /= max_weight;
        }
    } else {
        std::fill(weights.begin(), weights.end(), 1.0);
    }
    double weight_sum = 0.0;
    double weighted_index_sum = 0.0;
    double weighted_phase_sum = 0.0;
    for (std::size_t index = 0; index < phases.size(); ++index) {
        weight_sum += weights[index];
        weighted_index_sum += weights[index] * frame_indices[index];
        weighted_phase_sum += weights[index] * phases[index];
    }
    const auto mean_index = weighted_index_sum / weight_sum;
    const auto mean_phase = weighted_phase_sum / weight_sum;
    double covariance = 0.0;
    double index_variance = 0.0;
    double phase_variance = 0.0;
    for (std::size_t index = 0; index < phases.size(); ++index) {
        const auto index_delta = frame_indices[index] - mean_index;
        const auto phase_delta = phases[index] - mean_phase;
        covariance += weights[index] * index_delta * phase_delta;
        index_variance += weights[index] * index_delta * index_delta;
        phase_variance += weights[index] * phase_delta * phase_delta;
    }
    if (index_variance <= 0.0) {
        return Pn945ResidualCfoResult{
            0.0F,
            0.0F,
            phases.size(),
            worker_count,
            false,
        };
    }
    const auto slope = covariance / index_variance;
    auto r_squared = 1.0;
    if (phase_variance > 0.0) {
        const auto residual_variance = phase_variance - slope * covariance;
        r_squared = 1.0 - residual_variance / phase_variance;
    }
    const auto cfo_hz = slope * static_cast<double>(kDtmbSymbolRateSps)
        / (2.0 * std::numbers::pi_v<double> * static_cast<double>(kPn945FrameSymbols));
    return Pn945ResidualCfoResult{
        static_cast<float>(cfo_hz),
        static_cast<float>(r_squared),
        phases.size(),
        worker_count,
        r_squared >= options.min_fit_r_squared,
    };
}

Pn945EqualizeResult pn945_equalize_c3780_frame_cf32(
    std::span<const float> interleaved_header,
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    Pn945EqualizeOptions options) {
    if (interleaved_header.size() != kPn945HeaderSymbols * 2
        || interleaved_next_header.size() != kPn945HeaderSymbols * 2) {
        throw std::invalid_argument("PN945 headers must contain exactly 945 CF32 samples");
    }
    if (interleaved_time_body.size() != kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("C=3780 body must contain exactly 3780 CF32 samples");
    }
    if (interleaved_equalized_spectrum.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("equalized C=3780 spectrum output span is too small");
    }
    if (options.channel_taps == 0 || options.channel_taps > kPn945CoreSymbols) {
        throw std::invalid_argument("PN945 channel taps must be in 1..511");
    }
    if (options.regularization <= 0.0F || options.response_floor <= 0.0F) {
        throw std::invalid_argument("PN945 regularization and response floor must be positive");
    }

    const auto phase = pn945_detect_phase_cf32(interleaved_header);
    const auto next_phase = pn945_detect_phase_cf32(interleaved_next_header);
    const auto taps = estimate_channel_taps(
        interleaved_header,
        phase,
        options.channel_taps,
        options.regularization);

    restore_and_equalize(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        options.response_floor,
        options.noise_variance,
        0,
        options.mmse_unbias,
        options.mmse_unbias_gain_floor,
        options.interleaved_channel_power);
    return Pn945EqualizeResult{phase, next_phase};
}

Pn945WidebandChannelModel build_pn945_wideband_channel_model_cf32(
    std::span<const float> interleaved_headers,
    Pn945WidebandModelOptions options) {
    const auto header_stride = kPn945HeaderSymbols * 2;
    if (interleaved_headers.empty()
        || (interleaved_headers.size() % header_stride) != 0) {
        throw std::invalid_argument(
            "wideband PN945 model needs one or more complete interleaved headers");
    }
    if (options.threshold_factor <= 0.0F
        || options.max_span_symbols == 0
        || options.max_span_symbols > kPn945CoreSymbols
        || options.min_relative_power <= 0.0F
        || options.regularization <= 0.0F) {
        throw std::invalid_argument("invalid wideband PN945 model options");
    }
    const auto frame_count = interleaved_headers.size() / header_stride;
    std::vector<std::size_t> phases(frame_count);
    std::vector<std::vector<Complex>> rows;
    std::vector<std::vector<Complex>> structure_rows;
    rows.reserve(frame_count);
    structure_rows.reserve(frame_count);
    std::array<std::size_t, kPn945CoreSymbols> phase_counts{};
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const auto header = interleaved_headers.subspan(frame * header_stride, header_stride);
        phases[frame] = pn945_detect_phase_cf32(header);
        ++phase_counts[phases[frame]];
        rows.push_back(estimate_channel_taps(
            header,
            phases[frame],
            kPn945CoreSymbols,
            options.regularization,
            options.header_observation));
        if (options.header_observation == Pn945HeaderObservation::core_only) {
            structure_rows.push_back(rows.back());
        } else {
            structure_rows.push_back(estimate_channel_taps(
                header,
                phases[frame],
                kPn945CoreSymbols,
                options.regularization,
                Pn945HeaderObservation::core_only));
        }
    }
    const auto base_phase = static_cast<std::size_t>(
        std::distance(
            phase_counts.begin(),
            std::max_element(phase_counts.begin(), phase_counts.end())));

    std::vector<float> raw_power(kPn945CoreSymbols, 0.0F);
    for (const auto& row : structure_rows) {
        for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
            raw_power[tap] += std::norm(row[tap]) / static_cast<float>(frame_count);
        }
    }
    const auto dominant_raw = static_cast<std::size_t>(
        std::distance(raw_power.begin(), std::max_element(raw_power.begin(), raw_power.end())));
    std::vector<Complex> mean_taps(kPn945CoreSymbols, Complex{0.0F, 0.0F});
    for (const auto& row : structure_rows) {
        const auto anchor = row[dominant_raw];
        const auto rotation = std::abs(anchor) > 0.0F
            ? std::conj(anchor) / std::abs(anchor)
            : Complex{1.0F, 0.0F};
        for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
            mean_taps[tap] += row[tap] * rotation / static_cast<float>(frame_count);
        }
    }

    std::vector<float> averaged_power(kPn945CoreSymbols);
    for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
        averaged_power[tap] = std::norm(mean_taps[tap]);
    }
    auto noise_power = median(averaged_power);
    const auto peak_power = *std::max_element(averaged_power.begin(), averaged_power.end());
    auto make_mask = [&](float noise) {
        const auto threshold = std::max(
            options.threshold_factor * options.threshold_factor * std::max(noise, 0.0F),
            peak_power * options.min_relative_power);
        std::vector<bool> result(kPn945CoreSymbols);
        for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
            result[tap] = averaged_power[tap] >= threshold;
        }
        return result;
    };
    auto mask = make_mask(noise_power);
    std::vector<float> provisional_noise;
    for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
        if (!mask[tap]) {
            provisional_noise.push_back(averaged_power[tap]);
        }
    }
    const auto refined_noise = median(std::move(provisional_noise));
    if (refined_noise > 0.0F) {
        noise_power = refined_noise;
        mask = make_mask(noise_power);
    }
    if (std::none_of(mask.begin(), mask.end(), [](bool value) { return value; })) {
        mask[dominant_raw] = true;
    }

    std::vector<float> per_frame_noise_samples;
    for (const auto& row : structure_rows) {
        for (std::size_t tap = 0; tap < kPn945CoreSymbols; ++tap) {
            if (!mask[tap]) {
                per_frame_noise_samples.push_back(std::norm(row[tap]));
            }
        }
    }
    const auto per_frame_noise = median(std::move(per_frame_noise_samples));
    const auto [gap_start, gap_length] = longest_circular_false_run(mask);
    const auto rotation = (gap_start + gap_length) % kPn945CoreSymbols;
    auto rotated_taps = rotate_left(mean_taps, rotation);
    auto rotated_mask = rotate_left(mask, rotation);
    dilate_linear_mask(rotated_mask, options.guard_taps);
    std::size_t span = 1;
    for (std::size_t tap = 0; tap < rotated_mask.size(); ++tap) {
        if (rotated_mask[tap]) {
            span = tap + 1;
        }
    }
    span = std::min(span, options.max_span_symbols);
    if (options.header_observation == Pn945HeaderObservation::core_cyclic_safe_average) {
        const auto safe_prefix_skip = std::min(span - 1, kPn945PrefixSymbols);
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            const auto header = interleaved_headers.subspan(frame * header_stride, header_stride);
            rows[frame] = estimate_channel_taps(
                header,
                phases[frame],
                kPn945CoreSymbols,
                options.regularization,
                options.header_observation,
                safe_prefix_skip);
        }
    }
    std::vector<float> template_taps(span * 2, 0.0F);
    std::vector<std::uint8_t> template_tap_mask(span, 0U);
    std::size_t significant_taps = 0;
    float kept_energy = 0.0F;
    float dropped_energy = 0.0F;
    for (std::size_t tap = 0; tap < rotated_mask.size(); ++tap) {
        if (!rotated_mask[tap]) {
            continue;
        }
        if (tap < span) {
            template_taps[tap * 2] = rotated_taps[tap].real();
            template_taps[tap * 2 + 1] = rotated_taps[tap].imag();
            template_tap_mask[tap] = 1U;
            kept_energy += std::norm(rotated_taps[tap]);
            ++significant_taps;
        } else {
            dropped_energy += std::norm(rotated_taps[tap]);
        }
    }
    std::size_t dominant_tap = 0;
    float dominant_power = -1.0F;
    for (std::size_t tap = 0; tap < span; ++tap) {
        const auto power = std::norm(Complex{
            template_taps[tap * 2],
            template_taps[tap * 2 + 1],
        });
        if (power > dominant_power) {
            dominant_power = power;
            dominant_tap = tap;
        }
    }
    std::vector<float> padded_taps(kC3780FrameBodySymbols * 2, 0.0F);
    std::copy(template_taps.begin(), template_taps.end(), padded_taps.begin());
    std::vector<float> template_response_fft(kC3780FrameBodySymbols * 2);
    mixed_radix_fft_forward_cf32(padded_taps, template_response_fft);

    std::vector<std::size_t> frame_pn_phases(frame_count);
    std::vector<float> frame_response_scales(frame_count * 2);
    std::vector<float> frame_template_taps;
    if (uses_masked_frame_taps(options.scale_estimator)) {
        frame_template_taps.assign(frame_count * span * 2, 0.0F);
    }
    const auto reference = Complex{
        template_taps[dominant_tap * 2],
        template_taps[dominant_tap * 2 + 1],
    };
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        frame_pn_phases[frame] =
            (phases[frame] + kPn945CoreSymbols - rotation) % kPn945CoreSymbols;
        Complex scale{1.0F, 0.0F};
        if (uses_masked_frame_taps(options.scale_estimator)) {
            Complex cross_sum{0.0F, 0.0F};
            float reference_power_sum = 0.0F;
            float observed_power_sum = 0.0F;
            std::size_t tap_count = 0;
            for (std::size_t tap = 0; tap < span; ++tap) {
                if (template_tap_mask[tap] == 0U) {
                    continue;
                }
                const auto observed =
                    rows[frame][(tap + rotation) % kPn945CoreSymbols];
                const auto base = (frame * span + tap) * 2;
                frame_template_taps[base] = observed.real();
                frame_template_taps[base + 1] = observed.imag();
                const auto reference = Complex{
                    template_taps[tap * 2],
                    template_taps[tap * 2 + 1],
                };
                const auto reference_power = std::norm(reference);
                if (reference_power <= 0.0F) {
                    continue;
                }
                cross_sum += observed * std::conj(reference);
                reference_power_sum += reference_power;
                observed_power_sum += std::norm(observed);
                ++tap_count;
            }
            scale = bounded_wideband_response_scale_from_sums(
                cross_sum,
                reference_power_sum,
                observed_power_sum,
                tap_count,
                per_frame_noise);
        } else if (options.scale_estimator
            == Pn945WidebandScaleEstimator::least_squares_template) {
            Complex cross_sum{0.0F, 0.0F};
            float reference_power_sum = 0.0F;
            float observed_power_sum = 0.0F;
            std::size_t tap_count = 0;
            for (std::size_t tap = 0; tap < span; ++tap) {
                const auto template_tap = Complex{
                    template_taps[tap * 2],
                    template_taps[tap * 2 + 1],
                };
                const auto template_power = std::norm(template_tap);
                if (template_power <= 0.0F) {
                    continue;
                }
                const auto observed =
                    rows[frame][(tap + rotation) % kPn945CoreSymbols];
                cross_sum += observed * std::conj(template_tap);
                reference_power_sum += template_power;
                observed_power_sum += std::norm(observed);
                ++tap_count;
            }
            scale = bounded_wideband_response_scale_from_sums(
                cross_sum,
                reference_power_sum,
                observed_power_sum,
                tap_count,
                per_frame_noise);
        } else {
            const auto anchor =
                rows[frame][(dominant_tap + rotation) % kPn945CoreSymbols];
            scale = bounded_wideband_response_scale(
                anchor,
                reference,
                per_frame_noise);
        }
        frame_response_scales[frame * 2] = scale.real();
        frame_response_scales[frame * 2 + 1] = scale.imag();
    }
    return Pn945WidebandChannelModel{
        base_phase,
        (base_phase + kPn945CoreSymbols - rotation) % kPn945CoreSymbols,
        rotation,
        std::move(template_taps),
        std::move(template_response_fft),
        std::move(template_tap_mask),
        std::move(frame_pn_phases),
        std::move(frame_response_scales),
        std::move(frame_template_taps),
        significant_taps,
        dominant_tap,
        frame_count,
        static_cast<float>(phase_counts[base_phase]) / static_cast<float>(frame_count),
        per_frame_noise,
        per_frame_noise * static_cast<float>(kC3780FrameBodySymbols),
        (kept_energy + dropped_energy) > 0.0F
            ? dropped_energy / (kept_energy + dropped_energy)
            : 0.0F,
        options.scale_estimator,
        options.header_observation,
    };
}

Pn945EqualizeResult pn945_equalize_c3780_frame_wideband_cf32(
    std::span<const float> interleaved_header,
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    const Pn945WidebandChannelModel& model,
    Pn945EqualizeOptions options) {
    if (interleaved_header.size() != kPn945HeaderSymbols * 2
        || interleaved_next_header.size() != kPn945HeaderSymbols * 2) {
        throw std::invalid_argument("PN945 headers must contain exactly 945 CF32 samples");
    }
    if (interleaved_time_body.size() != kC3780FrameBodySymbols * 2
        || interleaved_equalized_spectrum.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("wideband PN945 equalizer needs one complete C=3780 body");
    }
    if (options.regularization <= 0.0F || options.response_floor <= 0.0F) {
        throw std::invalid_argument("PN945 regularization and response floor must be positive");
    }
    std::size_t phase = 0;
    Complex response_scale{1.0F, 0.0F};
    const auto taps = instantiate_wideband_taps(
        interleaved_header,
        model,
        options.regularization,
        phase,
        response_scale);
    std::size_t next_phase = 0;
    Complex next_response_scale{1.0F, 0.0F};
    std::vector<Complex> next_taps;
    if (options.interpolate_body_channel) {
        next_taps = instantiate_wideband_taps(
            interleaved_next_header,
            model,
            options.regularization,
            next_phase,
            next_response_scale);
    } else {
        next_phase = wideband_pn_phase_only(interleaved_next_header, model);
    }
    auto signed_rotation = static_cast<int>(model.rotation_symbols);
    if (signed_rotation > static_cast<int>(kPn945CoreSymbols / 2)) {
        signed_rotation -= static_cast<int>(kPn945CoreSymbols);
    }
    if (options.interpolate_body_channel) {
        const auto body_taps = midpoint_taps(taps, next_taps);
        restore_and_equalize_transition(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            next_taps,
            body_taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power);
        return Pn945EqualizeResult{phase, next_phase};
    }
    if (uses_masked_frame_taps(model.scale_estimator)) {
        restore_and_equalize(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power);
        return Pn945EqualizeResult{phase, next_phase};
    }
    restore_and_equalize_with_template_response(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        model.template_response_fft,
        response_scale,
        options.response_floor,
        options.noise_variance,
        -signed_rotation + options.response_window_offset_adjust,
        options.mmse_unbias,
        options.mmse_unbias_gain_floor,
        options.interleaved_channel_power);
    return Pn945EqualizeResult{phase, next_phase};
}

Pn945EqualizeResult pn945_equalize_c3780_frame_wideband_cached_cf32(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    const Pn945WidebandChannelModel& model,
    std::size_t model_frame_index,
    Pn945EqualizeOptions options) {
    if (interleaved_next_header.size() != kPn945HeaderSymbols * 2) {
        throw std::invalid_argument("PN945 next header must contain exactly 945 CF32 samples");
    }
    if (interleaved_time_body.size() != kC3780FrameBodySymbols * 2
        || interleaved_equalized_spectrum.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("wideband PN945 cached equalizer needs one C=3780 body");
    }
    if (options.response_floor <= 0.0F) {
        throw std::invalid_argument("PN945 response floor must be positive");
    }
    if (model.template_taps.empty() || (model.template_taps.size() % 2) != 0) {
        throw std::invalid_argument("wideband PN945 cached model has invalid template taps");
    }
    if (model_frame_index + 1 >= model.frame_pn_phases.size()
        || (model_frame_index + 1) * 2 + 1 >= model.frame_response_scales.size()) {
        throw std::invalid_argument("wideband PN945 cached frame index is outside model state");
    }
    const auto phase = model.frame_pn_phases[model_frame_index];
    const auto next_phase = model.frame_pn_phases[model_frame_index + 1];
    const auto response_scale = Complex{
        model.frame_response_scales[model_frame_index * 2],
        model.frame_response_scales[model_frame_index * 2 + 1],
    };
    const auto taps = uses_masked_frame_taps(model.scale_estimator)
        ? masked_frame_taps_for_model_frame(model, model_frame_index)
        : scaled_template_taps(model, response_scale);
    auto signed_rotation = static_cast<int>(model.rotation_symbols);
    if (signed_rotation > static_cast<int>(kPn945CoreSymbols / 2)) {
        signed_rotation -= static_cast<int>(kPn945CoreSymbols);
    }
    if (options.interpolate_body_channel) {
        const auto next_response_scale = Complex{
            model.frame_response_scales[(model_frame_index + 1) * 2],
            model.frame_response_scales[(model_frame_index + 1) * 2 + 1],
        };
        const auto next_taps = uses_masked_frame_taps(model.scale_estimator)
            ? masked_frame_taps_for_model_frame(model, model_frame_index + 1)
            : scaled_template_taps(model, next_response_scale);
        const auto body_taps = midpoint_taps(taps, next_taps);
        restore_and_equalize_transition(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            next_taps,
            body_taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power);
        return Pn945EqualizeResult{phase, next_phase};
    }
    if (uses_masked_frame_taps(model.scale_estimator)) {
        restore_and_equalize(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power);
        return Pn945EqualizeResult{phase, next_phase};
    }
    restore_and_equalize_with_template_response(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        model.template_response_fft,
        response_scale,
        options.response_floor,
        options.noise_variance,
        -signed_rotation + options.response_window_offset_adjust,
        options.mmse_unbias,
        options.mmse_unbias_gain_floor,
        options.interleaved_channel_power);
    return Pn945EqualizeResult{phase, next_phase};
}

}  // namespace dtmb::core
