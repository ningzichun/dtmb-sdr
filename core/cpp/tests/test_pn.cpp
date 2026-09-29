#ifdef NDEBUG
#undef NDEBUG
#endif
#include "dtmb/core.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

using namespace dtmb::core;
using Complex = std::complex<float>;

namespace {

void check_vector(PnMode mode, std::string_view hex) {
    const auto& pn = pn_definition(mode);
    const auto chips = pn_reference_chips(mode);
    std::vector<float> symbols(pn.header_symbols * 2);
    pn_header_symbols_cf32(mode, symbols);
    for (std::size_t n = 0; n < pn.header_symbols; ++n) {
        const char digit = hex[n / 4];
        const int value = digit <= '9' ? digit - '0' : digit - 'A' + 10;
        const float expected = (value >> (3 - n % 4)) & 1 ? -1.0F : 1.0F;
        assert(chips[n] == expected);
        assert(symbols[n * 2] == expected && symbols[n * 2 + 1] == expected);
    }
}

std::vector<float> header(PnMode mode, std::size_t phase) {
    std::vector<float> result(pn_definition(mode).header_symbols * 2);
    pn_header_symbols_cf32(mode, result, phase);
    return result;
}

void check_phases_and_correlation(PnMode mode) {
    const auto& pn = pn_definition(mode);
    for (std::size_t phase = 0; phase < pn.phase_count(); ++phase) {
        auto symbols = header(mode, phase);
        std::vector<std::int8_t> ci8(symbols.size());
        for (std::size_t n = 0; n < symbols.size(); n += 2) {
            const auto value = Complex{symbols[n], symbols[n + 1]} * std::polar(23.0F, 0.4F);
            symbols[n] = value.real(); symbols[n + 1] = value.imag();
            ci8[n] = static_cast<std::int8_t>(std::round(value.real()));
            ci8[n + 1] = static_cast<std::int8_t>(std::round(value.imag()));
        }
        assert(pn_detect_phase_cf32(mode, symbols) == phase);
        assert(pn_known_phase_metric_ci8(mode, ci8, phase) > 0.999F);
        assert(pn_header_metric_ci8(mode, ci8) > 0.999F);
        if (pn.cyclic_extension()) {
            assert(pn_known_phase_metric_ci8(mode, ci8, (phase + 31) % pn.core_symbols) < 0.05F);
        } else {
            // A shifted PN595 sequence is not another member of a cyclic family.
            std::rotate(ci8.begin(), ci8.begin() + 14, ci8.end());
            assert(pn_known_phase_metric_ci8(mode, ci8) < 0.1F);
            assert(pn_header_metric_ci8(mode, ci8) < 0.3F);
        }
    }
}

void check_acquisition_and_cfo(PnMode mode) {
    const auto& pn = pn_definition(mode);
    constexpr std::size_t leading = 23, frames = 10;
    std::mt19937 rng(341);
    std::uniform_real_distribution<float> noise(-0.25F, 0.25F);
    std::vector<float> symbols((leading + frames * pn.frame_symbols()) * 2);
    for (auto& value : symbols) value = noise(rng);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const auto reference = header(mode, pn_phase_for_frame(mode, frame));
        std::copy(reference.begin(), reference.end(), symbols.begin() + (leading + frame * pn.frame_symbols()) * 2);
    }
    for (const float cfo : {25.0F, 1200.0F}) {
        auto shifted = symbols;
        for (std::size_t n = 0; n < shifted.size() / 2; ++n) {
            const auto angle = 2.0 * std::numbers::pi * cfo * static_cast<double>(n) / kDtmbSymbolRateSps;
            const auto value = std::complex<double>{symbols[n * 2], symbols[n * 2 + 1]} * std::polar(1.0, angle);
            shifted[n * 2] = static_cast<float>(value.real());
            shifted[n * 2 + 1] = static_cast<float>(value.imag());
        }
        const auto result = acquire_pn_cf32(mode, shifted, {frames, 0.35F, 3});
        assert(result.phase_offset == leading && result.hit_count == frames);
        assert(result.coarse_cfo_valid && std::abs(result.coarse_cfo_hz - cfo) < 0.01F);
        if (cfo == 25.0F) {
            const auto residual = estimate_pn_residual_cfo_cf32(mode, shifted, leading, {frames, 0.5F, 3});
            assert(residual.valid && residual.used_frames == frames);
            assert(std::abs(residual.cfo_hz - cfo) < 0.01F);
        }
    }
}

void check_schedule(PnMode mode) {
    const auto& pn = pn_definition(mode);
    constexpr std::size_t origin = 29;
    std::vector<std::size_t> observed(pn.frames_per_superframe);
    for (std::size_t n = 0; n < observed.size(); ++n) {
        observed[n] = (pn_phase_for_frame(mode, n + origin) + pn.core_symbols - 3) % pn.core_symbols;
    }
    const auto fit = fit_pn_phase_schedule(mode, observed);
    assert(fit.valid && fit.superframe_index == origin && fit.phase_bias == -3);
    assert(fit.inliers == pn.frames_per_superframe);
    observed.resize(observed.size() - 1);
    assert(!fit_pn_phase_schedule(mode, observed).valid);
}

void check_equalizer(PnMode mode, std::size_t window_offset) {
    const auto& pn = pn_definition(mode);
    constexpr std::size_t frames = 5;
    std::vector<float> transmitted(frames * pn.frame_symbols() * 2);
    std::vector<std::vector<float>> bodies;
    std::vector<std::size_t> phases;
    std::mt19937 rng(908);
    std::uniform_real_distribution<float> data(-0.2F, 0.2F);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        phases.push_back(pn_phase_for_frame(mode, frame));
        const auto reference = header(mode, phases.back());
        const auto start = frame * pn.frame_symbols() * 2;
        std::copy(reference.begin(), reference.end(), transmitted.begin() + start);
        bodies.emplace_back(kC3780FrameBodySymbols * 2);
        for (auto& value : bodies.back()) value = data(rng);
        std::copy(bodies.back().begin(), bodies.back().end(), transmitted.begin() + start + pn.header_symbols * 2);
    }
    std::array<Complex, 14> taps{};
    taps[0] = {1, 0}; taps[4] = {0.17F, -0.06F}; taps[13] = {-0.1F, 0.09F};
    std::vector<float> received(transmitted.size() + 2 * taps.size());
    for (std::size_t n = 0; n < transmitted.size() / 2; ++n) {
        for (std::size_t t = 0; t < taps.size(); ++t) {
            const auto value = Complex{transmitted[n * 2], transmitted[n * 2 + 1]} * taps[t];
            received[(n + t) * 2] += value.real(); received[(n + t) * 2 + 1] += value.imag();
        }
    }
    std::vector<float> headers;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const auto first = (frame * pn.frame_symbols() + window_offset) * 2;
        headers.insert(headers.end(), received.begin() + first, received.begin() + first + pn.header_symbols * 2);
    }
    PnWidebandModelOptions model_options;
    model_options.regularization = 1.0e-6F;
    model_options.expected_phases = phases;
    model_options.max_span_symbols = pn.default_channel_span();
    model_options.scale_estimator = PnWidebandScaleEstimator::masked_frame_taps;
    if (pn.cyclic_extension()) model_options.header_observation = PnHeaderObservation::core_cyclic_safe_average;
    const auto model = build_pn_wideband_channel_model_cf32(mode, headers, model_options);
    assert(model.significant_taps >= 3 && model.truncated_energy_fraction < 1e-5F);
    for (std::size_t frame = 1; frame + 1 < frames; ++frame) {
        const auto first = (frame * pn.frame_symbols() + window_offset) * 2;
        const auto body = std::span<const float>(received).subspan(first + pn.header_symbols * 2, kC3780FrameBodySymbols * 2);
        const auto next_header = std::span<const float>(received).subspan(first + pn.frame_symbols() * 2, pn.header_symbols * 2);
        std::vector<float> output(kC3780FrameBodySymbols * 2), expected(output.size());
        (void)pn_equalize_c3780_frame_wideband_cached_cf32(mode, body, next_header, output, model, frame,
                                                         {14, 1e-6F, 1e-6F, -1.0F});
        mixed_radix_fft_forward_cf32(bodies[frame], expected);
        double error = 0, power = 0;
        for (std::size_t n = 0; n < output.size(); ++n) {
            error += std::pow(output[n] - expected[n], 2); power += std::pow(expected[n], 2);
        }
        assert(std::sqrt(error / power) < 0.03);
    }
}

}  // namespace

int main() {
    const auto release = version();
    assert(release.major == 0 && release.minor == 4 && release.patch == 1);
    assert(std::string_view(build_info()) == "dtmb-core-cpp 0.4.1");
    // PN420/PN945: GB 20600 appendices D/E phase-zero vectors.
    check_vector(PnMode::pn420,
        "B0A5E9FEA1CF0D9A3DC7407C4A22D5C8C938109BCCEFCB2B69063"
        "58AA60BAFB7614BD3FD439E1B347B8E80F89445AB91927021379");
    check_vector(PnMode::pn945,
        "FB946DFF3259AD7E9A7C9CDC081A82531A292661E8F1233A432F2D8055B"
        "ABDE0EBA16959060BE1BD4B9EDAA88E44D953B15C5DA03FB3F1884F38EEF"
        "AC28708B1F728DBFE64B35AFD34F939B8103504A634524CC3D1E24674865"
        "E5B00AB757BC1D742D2B20C17C37A973DB5511C89B2A762B8BB407F678");
    // PN595: section 4.6.2.2, x^10+x^3+1, state 0000000001,
    // first 595 output chips. The last hexadecimal bit is padding.
    check_vector(PnMode::pn595,
        "004934D7CC7C8EFC380FFC713B2BBD47A5417FAABD0E9196B3D633F2A9994FA708D914DEEAE67773"
        "A9D07B70C4A59A22D2E98B0292FBC63761E4E5886FE71A94212DF5C5C87DAA2F673E0");
    for (auto mode : {PnMode::pn420, PnMode::pn595, PnMode::pn945}) {
        const auto& pn = pn_definition(mode);
        assert(pn.header_symbols == pn.core_symbols + pn.prefix_symbols + pn.suffix_symbols);
        assert(pn.frame_symbols() * pn.frames_per_superframe == 945000);
        assert(pn.header_to_body_power_ratio == (mode == PnMode::pn595 ? 1.0F : 2.0F));
        assert(parse_pn_mode(pn_mode_name(mode)) == mode);
        check_phases_and_correlation(mode);
        check_acquisition_and_cfo(mode);
        check_equalizer(mode, 0);
        check_equalizer(mode, 7);
        if (pn.cyclic_extension()) check_schedule(mode);
    }
    // Independent Table 2 PN420 seeds at the turn and superframe boundary.
    for (auto [index, seed] : {std::pair{0U, 0xB0}, {1U, 0x61}, {2U, 0xD8},
                              {111U, 0x9A}, {112U, 0x83}, {113U, 0x9A}, {224U, 0xB0}, {225U, 0xB0}}) {
        const auto symbols = header(PnMode::pn420, pn_phase_for_frame(PnMode::pn420, index));
        int actual = 0;
        for (int bit = 0; bit < 8; ++bit) actual = (actual << 1) | (symbols[bit * 2] < 0);
        assert(actual == seed);
    }
    std::cout << "PN standard vectors, acquisition, CFO, schedules and multipath equalization passed\n";
}
