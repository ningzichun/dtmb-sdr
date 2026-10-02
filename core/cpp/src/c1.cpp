#include "dtmb/c1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <string_view>

namespace dtmb::core {
namespace {

using Complex = std::complex<double>;
constexpr std::size_t kTapRadius = 74;
constexpr std::size_t kFftSymbols = 8192;
constexpr double kSymbolRate = 7560000.0;
constexpr std::array<std::string_view, 11> kInfoVectors{
    "01111000110010000010111011010101", "01110111110001110010000111011010",
    "00100010100100100111010010001111", "01001011111110110001110111100110",
    "00010001101000010100011110111100", "01111000001101110010111000101010",
    "00101101100111010111101110000000", "01110111001110000010000100100101",
    "00100010011011010111010001110000", "01000100000010110001001000010110",
    "00010001010111100100011101000011",
};

Complex sample_at(std::span<const float> samples, std::size_t index) {
    return {samples[index * 2], samples[index * 2 + 1]};
}

void shift_samples(std::span<const float> input, std::span<float> output,
                   std::size_t sample_origin, float shift_hz) {
    for (std::size_t index = 0; index < input.size() / 2; ++index) {
        const auto phase = 2.0 * std::numbers::pi * shift_hz
            * static_cast<double>(sample_origin + index) / kSymbolRate;
        const auto value = sample_at(input, index) * std::polar(1.0, phase);
        output[index * 2] = static_cast<float>(value.real());
        output[index * 2 + 1] = static_cast<float>(value.imag());
    }
}

float fit_error(std::span<const float> header, std::span<const float> taps) {
    const auto chips = pn_reference_chips(PnMode::pn595);
    double residual_power = 0.0, observed_power = 0.0;
    for (std::size_t row = kTapRadius; row < 595 - kTapRadius; ++row) {
        Complex predicted{};
        for (std::size_t column = 0; column < 2 * kTapRadius + 1; ++column) {
            const auto reference_index = row + kTapRadius - column;
            predicted += sample_at(taps, column)
                * Complex{static_cast<double>(chips[reference_index]),
                          static_cast<double>(chips[reference_index])};
        }
        const auto observed = sample_at(header, row);
        residual_power += std::norm(predicted - observed);
        observed_power += std::norm(observed);
    }
    return static_cast<float>(residual_power / std::max(observed_power, 1e-20));
}

double info_metric(std::span<const float> observed, std::span<const float> reference) {
    Complex correlation{};
    double power = 0.0;
    for (std::size_t index = 0; index < 36; ++index) {
        const auto value = sample_at(observed, index);
        correlation += value * std::conj(sample_at(reference, index));
        power += std::norm(value);
    }
    correlation /= std::sqrt(std::max(power * 72.0, 1e-20));
    return std::abs(std::arg(correlation)) <= std::numbers::pi / 4.0
        ? std::abs(correlation) : correlation.real();
}

double fourfold_metric(std::span<const float> data) {
    Complex fourth_sum{};
    double fourth_power = 0.0;
    for (std::size_t index = 0; index < data.size() / 2; ++index) {
        const auto squared = sample_at(data, index) * sample_at(data, index);
        fourth_sum += squared * squared;
        fourth_power += std::norm(squared);
    }
    return std::abs(fourth_sum) / std::max(fourth_power, 1e-20);
}

}

std::vector<float> c1_system_info_reference_cf32(std::size_t index) {
    if (index < 3 || index > 24) throw std::invalid_argument("C=1 SI index must be 3..24");
    const auto bits = kInfoVectors[(index - 3) / 2];
    std::vector<float> reference(72);
    for (std::size_t symbol = 0; symbol < 36; ++symbol) {
        auto bit = symbol < 4 ? '0' : bits[symbol - 4];
        if (symbol >= 4 && index % 2 == 0) bit = bit == '0' ? '1' : '0';
        reference[symbol * 2] = reference[symbol * 2 + 1] = bit == '0' ? -1.0F : 1.0F;
    }
    return reference;
}

float c1_equalize_pn595_cf32(std::span<const float> input, std::span<float> equalized) {
    if (input.size() != kC1Pn595ContextSymbols * 2 || equalized.size() != input.size()) {
        throw std::invalid_argument("C=1 equalization needs one PN595 frame and a following header");
    }
    if (!std::all_of(input.begin(), input.end(), [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("C=1 input contains nonfinite samples");
    }
    const auto taps = pn595_linear_channel_cf32(input.first(595 * 2));
    std::vector<float> impulse(kFftSymbols * 2), response(impulse.size());
    std::vector<float> time(impulse.size()), spectrum(impulse.size());
    std::copy(input.begin(), input.end(), time.begin());
    for (std::size_t column = 0; column < 2 * kTapRadius + 1; ++column) {
        const auto delay = static_cast<int>(column) - static_cast<int>(kTapRadius);
        const auto position = static_cast<std::size_t>((delay + static_cast<int>(kFftSymbols))
                                                       % static_cast<int>(kFftSymbols));
        impulse[position * 2] = taps[column * 2];
        impulse[position * 2 + 1] = taps[column * 2 + 1];
    }
    mixed_radix_fft_forward_cf32(impulse, response);
    mixed_radix_fft_forward_cf32(time, spectrum);
    double mean_power = 0.0;
    for (std::size_t bin = 0; bin < kFftSymbols; ++bin) mean_power += std::norm(sample_at(response, bin));
    mean_power /= kFftSymbols;
    if (mean_power <= 1e-20) throw std::runtime_error("C=1 PN channel has zero power");
    for (std::size_t bin = 0; bin < kFftSymbols; ++bin) {
        const auto channel = sample_at(response, bin);
        const auto value = sample_at(spectrum, bin) * std::conj(channel)
            / std::max(std::norm(channel) + mean_power * 0.01, 1e-20);
        spectrum[bin * 2] = static_cast<float>(value.real());
        spectrum[bin * 2 + 1] = static_cast<float>(-value.imag());
    }
    mixed_radix_fft_forward_cf32(spectrum, time);
    for (std::size_t index = 0; index < equalized.size() / 2; ++index) {
        equalized[index * 2] = time[index * 2] / static_cast<float>(kFftSymbols);
        equalized[index * 2 + 1] = -time[index * 2 + 1] / static_cast<float>(kFftSymbols);
    }
    return fit_error(input.first(595 * 2), taps);
}

C1AcquisitionResult acquire_c1_pn595_cf32(
    std::span<const float> samples, std::size_t phase_offset,
    float frequency_shift_hz, std::size_t observations, float min_metric, float min_margin) {
    if (samples.size() % 2 != 0 || phase_offset >= 4375
        || !std::isfinite(frequency_shift_hz) || observations < 2 || observations > 256
        || !std::isfinite(min_metric) || min_metric < 0.0F || min_metric > 1.0F
        || !std::isfinite(min_margin) || min_margin < 0.0F || min_margin > 1.0F) {
        throw std::invalid_argument("invalid C=1 acquisition settings");
    }
    C1AcquisitionResult result;
    result.frequency_shift_hz = frequency_shift_hz;
    const auto required = phase_offset + (observations - 1) * 4375 + kC1Pn595ContextSymbols;
    if (samples.size() / 2 < required) return result;
    std::array<double, 5> errors{};
    std::vector<float> header(595 * 2);
    const auto fit_frames = std::min(observations, std::size_t{16});
    for (int alias = -2; alias <= 2; ++alias) {
        const auto shift = frequency_shift_hz + static_cast<float>(alias * kSymbolRate / 4375.0);
        for (std::size_t frame = 0; frame < fit_frames; ++frame) {
            const auto origin = phase_offset + frame * 4375;
            shift_samples(samples.subspan(origin * 2, header.size()), header, origin, shift);
            const auto taps = pn595_linear_channel_cf32(header);
            errors[static_cast<std::size_t>(alias + 2)] += fit_error(header, taps) / fit_frames;
        }
    }
    auto chosen = static_cast<std::size_t>(std::min_element(errors.begin(), errors.end()) - errors.begin());
    if (chosen != 2 && errors[chosen] > errors[2] * 0.75) chosen = 2;
    result.cfo_alias = static_cast<int>(chosen) - 2;
    result.frequency_shift_hz += static_cast<float>(result.cfo_alias * kSymbolRate / 4375.0);
    result.pn_fit_error = static_cast<float>(errors[chosen]);
    std::array<std::vector<float>, 22> references;
    for (std::size_t index = 3; index <= 24; ++index) references[index - 3] = c1_system_info_reference_cf32(index);
    std::array<std::array<double, 22>, 5> scores{};
    std::array<double, 5> fourth{};
    std::vector<float> shifted(kC1Pn595ContextSymbols * 2), equalized(shifted.size());
    for (std::size_t frame = 0; frame < observations; ++frame) {
        const auto origin = phase_offset + frame * 4375;
        shift_samples(samples.subspan(origin * 2, shifted.size()), shifted, origin, result.frequency_shift_hz);
        c1_equalize_pn595_cf32(shifted, equalized);
        for (int offset = -2; offset <= 2; ++offset) {
            const auto body = static_cast<std::size_t>(595 + offset);
            const auto info = std::span<const float>(equalized).subspan(body * 2, 72);
            const auto slot = static_cast<std::size_t>(offset + 2);
            for (std::size_t index = 0; index < references.size(); ++index) {
                scores[slot][index] += info_metric(info, references[index]) / observations;
            }
            fourth[slot] += fourfold_metric(std::span<const float>(equalized).subspan((body + 36) * 2, 3744 * 2)) / observations;
        }
    }
    double best = -std::numeric_limits<double>::infinity();
    std::size_t best_slot = 0, best_index = 0;
    for (std::size_t slot = 0; slot < scores.size(); ++slot) {
        for (std::size_t index = 0; index < references.size(); ++index) {
            if (scores[slot][index] > best) {
                best = scores[slot][index]; best_slot = slot; best_index = index;
            }
        }
    }
    double runner_up = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < references.size(); ++index) {
        if (index != best_index) runner_up = std::max(runner_up, scores[best_slot][index]);
    }
    result.observations = observations;
    result.system_info_index = best_index + 3;
    result.body_offset = static_cast<int>(best_slot) - 2;
    result.metric = static_cast<float>(best);
    result.margin = static_cast<float>(best - runner_up);
    result.data_fourfold_metric = static_cast<float>(fourth[best_slot]);
    result.locked = result.system_info_index >= 5 && result.system_info_index <= 10
        && best >= min_metric && result.margin >= min_margin && fourth[best_slot] >= 0.5;
    return result;
}

}
