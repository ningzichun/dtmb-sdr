#include "dtmb/core.hpp"
#include "dtmb/core_c.h"
#include "dtmb/live_view.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <span>
#include <string_view>
#include <vector>

namespace {

void test_version() {
    const auto version = dtmb::core::version();
    assert(version.major == dtmb_core_version_major());
    assert(version.minor == dtmb_core_version_minor());
    assert(version.patch == dtmb_core_version_patch());
    assert(version.abi_major == dtmb_core_abi_version_major());
    assert(version.abi_minor == dtmb_core_abi_version_minor());
}

void test_known_ci8_stats_scalar() {
    const std::vector<std::int8_t> samples{3, 4, INT8_MIN, INT8_MAX, 0, 0};
    const auto stats = dtmb::core::ci8_power_stats(samples);

    assert(stats.sample_count == 3);
    assert(stats.mean_i2q2 == static_cast<float>((25.0 + 32513.0) / 3.0));
    assert(std::abs(stats.rms_iq - std::sqrt(stats.mean_i2q2)) < 1.0e-6F);
    assert(stats.clip_count_i == 1);
    assert(stats.clip_count_q == 1);
    assert(stats.worker_count == 1);
}

void test_parallel_matches_scalar() {
    std::vector<std::int8_t> samples;
    samples.reserve(20000);
    for (int index = 0; index < 5000; ++index) {
        samples.push_back(static_cast<std::int8_t>((index % 251) - 125));
        samples.push_back(static_cast<std::int8_t>(((index * 7) % 251) - 125));
    }

    const auto scalar = dtmb::core::ci8_power_stats(
        samples,
        dtmb::core::Ci8PowerStatsOptions{1, 1});
    const auto parallel = dtmb::core::ci8_power_stats(
        samples,
        dtmb::core::Ci8PowerStatsOptions{4, 1});

    assert(parallel.worker_count > 1);
    assert(parallel.sample_count == scalar.sample_count);
    assert(parallel.mean_i2q2 == scalar.mean_i2q2);
    assert(parallel.rms_iq == scalar.rms_iq);
    assert(parallel.clip_count_i == scalar.clip_count_i);
    assert(parallel.clip_count_q == scalar.clip_count_q);
}

void test_rejects_odd_byte_count() {
    const std::vector<std::int8_t> samples{1, 2, 3};
    bool threw = false;
    try {
        (void)dtmb::core::ci8_power_stats(samples);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_c_abi_validation_and_output() {
    const std::vector<std::int8_t> samples{3, 4};
    DtmbCoreCi8PowerStats stats{};

    assert(dtmb_core_ci8_power_stats(samples.data(), samples.size(), nullptr)
        == DTMB_CORE_STATUS_INVALID_ARGUMENT);
    assert(dtmb_core_ci8_power_stats(nullptr, 2, &stats)
        == DTMB_CORE_STATUS_INVALID_ARGUMENT);
    assert(dtmb_core_ci8_power_stats(samples.data(), 1, &stats)
        == DTMB_CORE_STATUS_INVALID_LENGTH);
    assert(dtmb_core_ci8_power_stats(nullptr, 0, &stats)
        == DTMB_CORE_STATUS_OK);
    assert(stats.sample_count == 0);

    assert(dtmb_core_ci8_power_stats(samples.data(), samples.size(), &stats)
        == DTMB_CORE_STATUS_OK);
    assert(stats.sample_count == 1);
    assert(stats.mean_i2q2 == 25.0F);
    assert(stats.rms_iq == 5.0F);
}

void test_qam64_soft_demodulate_known_symbol() {
    const std::vector<float> symbols{-7.0F, 7.0F};
    std::vector<float> llr(6, 0.0F);

    dtmb::core::qam64_soft_demodulate_cf32(
        symbols,
        llr,
        dtmb::core::QamSoftDemapOptions{1.0F, 1, 1});

    assert(llr[0] > 0.0F);
    assert(llr[1] > 0.0F);
    assert(llr[2] > 0.0F);
    assert(llr[3] > 0.0F);
    assert(llr[4] > 0.0F);
    assert(llr[5] < 0.0F);
}

void test_qam64_soft_demodulate_parallel_matches_scalar() {
    std::vector<float> symbols;
    symbols.reserve(4096);
    for (int index = 0; index < 2048; ++index) {
        symbols.push_back(static_cast<float>((index % 17) - 8) * 0.5F);
        symbols.push_back(static_cast<float>(((index * 5) % 19) - 9) * 0.5F);
    }

    std::vector<float> scalar((symbols.size() / 2) * 6, 0.0F);
    std::vector<float> parallel(scalar.size(), 0.0F);
    dtmb::core::qam64_soft_demodulate_cf32(
        symbols,
        scalar,
        dtmb::core::QamSoftDemapOptions{0.75F, 1, 1});
    dtmb::core::qam64_soft_demodulate_cf32(
        symbols,
        parallel,
        dtmb::core::QamSoftDemapOptions{0.75F, 4, 1});

    assert(parallel == scalar);
}

void test_qam64_c_abi_validation_and_output() {
    const std::vector<float> symbols{-7.0F, 7.0F};
    std::vector<float> llr(6, 0.0F);

    assert(dtmb_core_qam64_soft_demodulate_cf32(nullptr, 1, 1.0F, llr.data())
        == DTMB_CORE_STATUS_INVALID_ARGUMENT);
    assert(dtmb_core_qam64_soft_demodulate_cf32(symbols.data(), 1, 0.0F, llr.data())
        == DTMB_CORE_STATUS_INVALID_ARGUMENT);
    assert(dtmb_core_qam64_soft_demodulate_cf32(symbols.data(), 1, 1.0F, nullptr)
        == DTMB_CORE_STATUS_INVALID_ARGUMENT);
    assert(dtmb_core_qam64_soft_demodulate_cf32(symbols.data(), 1, 1.0F, llr.data())
        == DTMB_CORE_STATUS_OK);
    assert(llr[0] > 0.0F);
}

void test_qam64_normalize_removes_complex_gain() {
    const std::vector<float> symbols{
        -7.0F, 7.0F,
        -3.0F, -1.0F,
        5.0F, 3.0F,
        1.0F, -5.0F,
    };
    std::vector<float> impaired(symbols.size());
    const auto gain = std::complex<float>{2.5F, -0.75F};
    for (std::size_t index = 0; index < symbols.size() / 2; ++index) {
        const auto value = std::complex<float>{symbols[index * 2], symbols[index * 2 + 1]}
            * gain;
        impaired[index * 2] = value.real();
        impaired[index * 2 + 1] = value.imag();
    }
    dtmb::core::qam64_normalize_cf32(impaired, impaired);
    for (std::size_t index = 0; index < symbols.size(); ++index) {
        assert(std::abs(impaired[index] - symbols[index]) < 1.0e-5F);
    }
}

void test_mixed_radix_fft_known_tone() {
    constexpr std::size_t size = 30;
    constexpr std::size_t tone_bin = 7;
    std::vector<float> time(size * 2);
    std::vector<float> spectrum(size * 2);
    for (std::size_t sample = 0; sample < size; ++sample) {
        const auto angle = 2.0F * std::numbers::pi_v<float>
            * static_cast<float>(tone_bin * sample) / static_cast<float>(size);
        time[sample * 2] = std::cos(angle);
        time[sample * 2 + 1] = std::sin(angle);
    }
    dtmb::core::mixed_radix_fft_forward_cf32(time, spectrum);
    for (std::size_t bin = 0; bin < size; ++bin) {
        const auto magnitude = std::hypot(spectrum[bin * 2], spectrum[bin * 2 + 1]);
        if (bin == tone_bin) {
            assert(std::abs(magnitude - static_cast<float>(size)) < 1.0e-4F);
        } else {
            assert(magnitude < 1.0e-4F);
        }
    }
}

constexpr std::array<std::size_t, dtmb::core::kC3780SystemInfoSymbols>
kC3780SystemInfoPositionsForTest{
    0, 140, 279, 419, 420, 560, 699, 839, 840, 980, 1119, 1259,
    1260, 1400, 1539, 1679, 1680, 1820, 1959, 2099, 2100, 2240,
    2379, 2519, 2520, 2660, 2799, 2939, 2940, 3080, 3219, 3359,
    3360, 3500, 3639, 3779,
};

std::size_t c3780_logical_to_physical_for_test(std::size_t inserted_logical) {
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                for (std::size_t l = 0; l < 2; ++l) {
                    for (std::size_t m = 0; m < 2; ++m) {
                        for (std::size_t n = 0; n < 5; ++n) {
                            for (std::size_t o = 0; o < 7; ++o) {
                                const auto logical =
                                    i * 1260 + j * 420 + k * 140 + l * 70
                                    + m * 35 + n * 7 + o;
                                if (logical == inserted_logical) {
                                    return o * 540 + n * 108 + m * 54 + l * 27
                                        + k * 9 + j * 3 + i;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert(false);
    return 0;
}

bool c3780_is_system_info_position_for_test(std::size_t inserted_logical) {
    return std::find(
        kC3780SystemInfoPositionsForTest.begin(),
        kC3780SystemInfoPositionsForTest.end(),
        inserted_logical)
        != kC3780SystemInfoPositionsForTest.end();
}

void test_c3780_deinterleave_spectrum_preserves_tagged_carrier_provenance() {
    std::array<std::size_t, dtmb::core::kC3780FrameBodySymbols> physical_to_logical{};
    for (std::size_t inserted_logical = 0;
         inserted_logical < dtmb::core::kC3780FrameBodySymbols;
         ++inserted_logical) {
        physical_to_logical[c3780_logical_to_physical_for_test(inserted_logical)] =
            inserted_logical;
    }

    std::vector<float> physical(dtmb::core::kC3780FrameBodySymbols * 2);
    for (std::size_t physical_bin = 0;
         physical_bin < dtmb::core::kC3780FrameBodySymbols;
         ++physical_bin) {
        physical[physical_bin * 2] = static_cast<float>(physical_bin) + 0.25F;
        physical[physical_bin * 2 + 1] =
            static_cast<float>(physical_to_logical[physical_bin]) + 0.5F;
    }

    std::vector<float> output(dtmb::core::kC3780FrameBodySymbols * 2);
    dtmb::core::c3780_deinterleave_spectrum_cf32(physical, output);

    for (std::size_t si_index = 0;
         si_index < kC3780SystemInfoPositionsForTest.size();
         ++si_index) {
        const auto inserted_logical = kC3780SystemInfoPositionsForTest[si_index];
        const auto physical_bin = c3780_logical_to_physical_for_test(inserted_logical);
        assert(output[si_index * 2] == static_cast<float>(physical_bin) + 0.25F);
        assert(output[si_index * 2 + 1] == static_cast<float>(inserted_logical) + 0.5F);
    }

    std::size_t data_index = 0;
    for (std::size_t inserted_logical = 0;
         inserted_logical < dtmb::core::kC3780FrameBodySymbols;
         ++inserted_logical) {
        if (c3780_is_system_info_position_for_test(inserted_logical)) {
            continue;
        }
        const auto output_index = dtmb::core::kC3780SystemInfoSymbols + data_index;
        const auto physical_bin = c3780_logical_to_physical_for_test(inserted_logical);
        assert(output[output_index * 2] == static_cast<float>(physical_bin) + 0.25F);
        assert(output[output_index * 2 + 1] == static_cast<float>(inserted_logical) + 0.5F);
        ++data_index;
    }
    assert(data_index == dtmb::core::kC3780DataSymbols);
}

std::array<std::uint8_t, dtmb::core::kPn945HeaderSymbols> make_pn945_phase0_bits() {
    constexpr std::string_view seed = "111110111";
    constexpr std::array<std::size_t, 4> recurrence_taps{0, 1, 2, 7};
    std::array<std::uint8_t, dtmb::core::kPn945HeaderSymbols> bits{};
    for (std::size_t index = 0; index < seed.size(); ++index) {
        bits[index] = static_cast<std::uint8_t>(seed[index] - '0');
    }
    for (std::size_t index = seed.size(); index < bits.size(); ++index) {
        const auto window = index - seed.size();
        for (const auto tap : recurrence_taps) {
            bits[index] ^= bits[window + tap];
        }
    }
    return bits;
}

std::vector<float> make_pn945_header(std::size_t phase, float amplitude = 0.25F) {
    const auto bits = make_pn945_phase0_bits();
    std::vector<float> header(dtmb::core::kPn945HeaderSymbols * 2);
    auto core_index_for_header = [](std::size_t header_index) {
        constexpr auto prefix = std::size_t{217};
        constexpr auto core = std::size_t{511};
        if (header_index < prefix) {
            return core - prefix + header_index;
        }
        if (header_index < prefix + core) {
            return header_index - prefix;
        }
        return header_index - prefix - core;
    };
    for (std::size_t sample = 0; sample < dtmb::core::kPn945HeaderSymbols; ++sample) {
        const auto core_index =
            (core_index_for_header(sample) + phase) % std::size_t{511};
        const auto bit = bits[std::size_t{217} + core_index];
        const auto chip = bit == 0 ? amplitude : -amplitude;
        header[sample * 2] = chip;
        header[sample * 2 + 1] = chip;
    }
    return header;
}

std::vector<float> make_pn945_stream(
    std::size_t leading_samples,
    std::size_t frame_count,
    float cfo_hz) {
    const auto header = make_pn945_header(0);
    const auto sample_count = leading_samples + frame_count * dtmb::core::kPn945FrameSymbols;
    std::vector<float> stream(sample_count * 2);
    std::uint32_t random_state = 0x12345678U;
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const auto frame_start = leading_samples + frame * dtmb::core::kPn945FrameSymbols;
        for (std::size_t sample = 0; sample < dtmb::core::kPn945HeaderSymbols; ++sample) {
            stream[(frame_start + sample) * 2] = header[sample * 2];
            stream[(frame_start + sample) * 2 + 1] = header[sample * 2 + 1];
        }
        for (std::size_t sample = dtmb::core::kPn945HeaderSymbols;
             sample < dtmb::core::kPn945FrameSymbols;
             ++sample) {
            random_state = random_state * 1664525U + 1013904223U;
            stream[(frame_start + sample) * 2] =
                static_cast<float>(static_cast<int>((random_state >> 24U) & 0xffU) - 128)
                / 64.0F;
            random_state = random_state * 1664525U + 1013904223U;
            stream[(frame_start + sample) * 2 + 1] =
                static_cast<float>(static_cast<int>((random_state >> 24U) & 0xffU) - 128)
                / 64.0F;
        }
    }
    const auto radians_per_sample = 2.0 * std::numbers::pi_v<double>
        * static_cast<double>(cfo_hz)
        / static_cast<double>(dtmb::core::kDtmbSymbolRateSps);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const auto value = std::complex<double>{stream[sample * 2], stream[sample * 2 + 1]};
        const auto rotated = value * std::polar(
            1.0,
            radians_per_sample * static_cast<double>(sample));
        stream[sample * 2] = static_cast<float>(rotated.real());
        stream[sample * 2 + 1] = static_cast<float>(rotated.imag());
    }
    return stream;
}

void test_pn945_detect_phase_tracks_shifted_headers() {
    for (const auto phase : std::array<std::size_t, 6>{0, 1, 17, 217, 294, 510}) {
        const auto header = make_pn945_header(phase);
        assert(dtmb::core::pn945_detect_phase_cf32(header) == phase);
    }
}

void test_pn945_acquisition_finds_unaligned_train_and_coarse_cfo() {
    constexpr std::size_t leading_samples = 123;
    constexpr float cfo_hz = -173.25F;
    const auto stream = make_pn945_stream(leading_samples, 20, cfo_hz);

    const auto result = dtmb::core::acquire_pn945_cf32(
        stream,
        dtmb::core::Pn945AcquisitionOptions{16, 0.35F, 4});

    assert(result.phase_offset == leading_samples);
    assert(result.hit_count == 16);
    assert(result.observed_frames == 16);
    assert(result.mean_metric > 0.999F);
    assert(result.coarse_cfo_valid);
    assert(std::abs(result.coarse_cfo_hz - cfo_hz) < 0.1F);
    assert(result.worker_count == 4);
}

void test_pn945_residual_cfo_resolves_five_hz() {
    constexpr std::size_t leading_samples = 37;
    constexpr float cfo_hz = 5.0F;
    const auto stream = make_pn945_stream(leading_samples, 100, cfo_hz);

    const auto result = dtmb::core::estimate_pn945_residual_cfo_cf32(
        stream,
        leading_samples,
        dtmb::core::Pn945ResidualCfoOptions{100, 0.5F, 4});

    assert(result.valid);
    assert(result.used_frames == 100);
    assert(result.fit_r_squared > 0.999F);
    assert(std::abs(result.cfo_hz - cfo_hz) < 0.1F);
    assert(result.worker_count == 4);
}

void test_symbol_deinterleaver_chunked_matches_source_trace() {
    const auto mode = dtmb::core::SymbolInterleaverMode::mode1;
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const std::size_t phase = 17;
    const std::size_t payload_symbols = 257;
    const auto symbol_count = spec.full_stream_latency_symbols() + payload_symbols;
    std::vector<float> input(symbol_count * 2);
    std::vector<float> output(symbol_count * 2, 0.0F);

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        input[symbol * 2] = static_cast<float>(symbol);
        input[symbol * 2 + 1] = -static_cast<float>(symbol);
    }

    dtmb::core::SymbolDeinterleaverCf32 deinterleaver(mode, phase);
    const std::size_t first_chunk = 1003;
    deinterleaver.process(
        std::span<const float>(input.data(), first_chunk * 2),
        std::span<float>(output.data(), first_chunk * 2));
    deinterleaver.process(
        std::span<const float>(input.data() + first_chunk * 2, input.size() - first_chunk * 2),
        std::span<float>(output.data() + first_chunk * 2, output.size() - first_chunk * 2));

    assert(deinterleaver.processed_symbols() == symbol_count);
    assert(deinterleaver.latency_symbols() == spec.full_stream_latency_symbols());
    for (std::size_t output_symbol = spec.full_stream_latency_symbols();
         output_symbol < symbol_count;
         ++output_symbol) {
        const auto start = output_symbol % spec.branch_count;
        const auto branch = (start + phase) % spec.branch_count;
        const auto branch_delay = (spec.branch_count - 1 - branch) * spec.delay_step;
        const auto branch_offset = output_symbol / spec.branch_count;
        const auto source_symbol = start + spec.branch_count * (branch_offset - branch_delay);
        assert(output[output_symbol * 2] == static_cast<float>(source_symbol));
        assert(output[output_symbol * 2 + 1] == -static_cast<float>(source_symbol));
    }
}

void test_symbol_deinterleaver_reset() {
    dtmb::core::SymbolDeinterleaverCf32 deinterleaver(
        dtmb::core::SymbolInterleaverMode::mode2,
        0,
        2.0F,
        -3.0F);
    const std::vector<float> input{10.0F, 20.0F};
    std::vector<float> output(2);

    deinterleaver.process(input, output);
    assert(output[0] == 2.0F);
    assert(output[1] == -3.0F);
    deinterleaver.reset();
    deinterleaver.process(input, output);
    assert(output[0] == 2.0F);
    assert(output[1] == -3.0F);
}

void test_sparse_ldpc_min_sum_decodes_single_parity_check() {
    const auto graph = dtmb::core::make_ldpc_sparse_graph({{0, 1, 2}}, 3);
    const std::vector<float> llr{0.1F, -8.0F, -8.0F};
    std::vector<std::uint8_t> bits(3);

    const auto result = dtmb::core::ldpc_decode_min_sum_sparse(
        llr,
        graph,
        bits,
        dtmb::core::LdpcDecodeOptions{8, 0.75F});

    assert(result.converged);
    assert(result.syndrome_weight == 0);
    assert(bits == std::vector<std::uint8_t>({0, 1, 1}));
}

void test_sparse_ldpc_layered_min_sum_decodes_single_parity_check() {
    const auto graph = dtmb::core::make_ldpc_sparse_graph({{0, 1, 2}}, 3);
    const std::vector<float> llr{0.1F, -8.0F, -8.0F};
    std::vector<std::uint8_t> bits(3);

    const auto result = dtmb::core::ldpc_decode_layered_min_sum_sparse(
        llr,
        graph,
        bits,
        dtmb::core::LdpcDecodeOptions{8, 0.75F});

    assert(result.converged);
    assert(result.syndrome_weight == 0);
    assert(bits == std::vector<std::uint8_t>({0, 1, 1}));
}

void test_live_view_scene_reducer_composes_controls_and_draw_packet() {
    dtmb::core::LiveViewSceneState state;

    const auto channel = dtmb::core::apply_live_view_scene_input_json(
        state,
        R"({"schema":"dtmb.live_view.control.v1","event":"control","action":"channel.select","channel_id":"rthk-522","center_frequency_hz":522000000})");
    assert(channel.kind == dtmb::core::LiveViewSceneInputKind::control);
    assert(channel.action == "channel.select");
    assert(state.revision == 1);
    assert(state.channel_id == "rthk-522");
    assert(state.center_frequency_hz == 522000000);

    const auto draw = dtmb::core::apply_live_view_scene_input_json(
        state,
        R"({"schema":"dtmb.live_view.draw_packet.v1","event":"draw","verdict":"ok","report_index":4,"draw":[]})");
    assert(draw.kind == dtmb::core::LiveViewSceneInputKind::draw_packet);
    assert(state.revision == 2);
    assert(state.latest_draw_verdict == "ok");

    const auto panel = dtmb::core::apply_live_view_scene_input_json(
        state,
        R"({"schema":"dtmb.live_view.control.v1","event":"control","action":"debug_panel.toggle","panel_id":"constellation"})");
    assert(panel.kind == dtmb::core::LiveViewSceneInputKind::control);
    assert(state.visible_debug_panels == std::vector<std::string>(
        {"acquisition", "constellation", "spectrum"}));

    const auto packet = dtmb::core::live_view_scene_packet_json(state, panel);
    assert(packet.find(R"("schema":"dtmb.live_view.scene.v1")") != std::string::npos);
    assert(packet.find(R"("revision":3)") != std::string::npos);
    assert(packet.find(R"("action":"debug_panel.toggle")") != std::string::npos);
    assert(packet.find(R"("draw_packet":{"schema":"dtmb.live_view.draw_packet.v1")")
        != std::string::npos);
}

void test_live_view_scene_reducer_rejects_control_without_partial_update() {
    dtmb::core::LiveViewSceneState state;
    state.channel_id = "original";
    state.center_frequency_hz = 482000000;

    bool threw = false;
    try {
        (void)dtmb::core::apply_live_view_scene_input_json(
            state,
            R"({"schema":"dtmb.live_view.control.v1","event":"control","action":"channel.select","channel_id":"invalid","center_frequency_hz":1.5})");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert(state.revision == 0);
    assert(state.channel_id == "original");
    assert(state.center_frequency_hz == 482000000);
}

void test_live_view_scene_terminal_verdict_retains_worst_draw_packet() {
    dtmb::core::LiveViewSceneState degraded_state;
    (void)dtmb::core::apply_live_view_scene_input_json(
        degraded_state,
        R"({"schema":"dtmb.live_view.draw_packet.v1","event":"draw","verdict":"degraded","draw":[]})");
    (void)dtmb::core::apply_live_view_scene_input_json(
        degraded_state,
        R"({"schema":"dtmb.live_view.draw_packet.v1","event":"draw","verdict":"ok","draw":[]})");
    assert(degraded_state.latest_draw_verdict == "ok");
    assert(degraded_state.worst_draw_verdict == "degraded");

    dtmb::core::LiveViewSceneStreamSummary degraded_summary;
    degraded_summary.line_count = 2;
    degraded_summary.draw_packet_count = 2;
    degraded_summary.scene_packet_count = 2;
    const auto degraded_end =
        dtmb::core::live_view_scene_stream_end_packet_json(degraded_summary, degraded_state);
    assert(degraded_end.find(R"("verdict":"degraded")") != std::string::npos);

    dtmb::core::LiveViewSceneState error_state;
    (void)dtmb::core::apply_live_view_scene_input_json(
        error_state,
        R"({"schema":"dtmb.live_view.draw_packet.v1","event":"error","verdict":"error","draw":[]})");
    (void)dtmb::core::apply_live_view_scene_input_json(
        error_state,
        R"({"schema":"dtmb.live_view.draw_packet.v1","event":"draw","verdict":"ok","draw":[]})");
    assert(error_state.latest_draw_verdict == "ok");
    assert(error_state.worst_draw_verdict == "error");

    dtmb::core::LiveViewSceneStreamSummary error_summary;
    error_summary.line_count = 2;
    error_summary.draw_packet_count = 2;
    error_summary.scene_packet_count = 2;
    const auto error_end =
        dtmb::core::live_view_scene_stream_end_packet_json(error_summary, error_state);
    assert(error_end.find(R"("verdict":"error")") != std::string::npos);
}

void test_live_view_scene_terminal_verdict_preserves_clean_and_malformed_recovery() {
    dtmb::core::LiveViewSceneState state;
    (void)dtmb::core::apply_live_view_scene_input_json(
        state,
        R"({"schema":"dtmb.live_view.draw_packet.v1","event":"draw","verdict":"ok","draw":[]})");

    dtmb::core::LiveViewSceneStreamSummary clean_summary;
    clean_summary.line_count = 1;
    clean_summary.draw_packet_count = 1;
    clean_summary.scene_packet_count = 1;
    const auto clean_end =
        dtmb::core::live_view_scene_stream_end_packet_json(clean_summary, state);
    assert(clean_end.find(R"("verdict":"ok")") != std::string::npos);

    bool threw = false;
    try {
        (void)dtmb::core::apply_live_view_scene_input_json(
            state,
            R"({"schema":"dtmb.live_view.draw_packet.v1","event":"draw","verdict":)");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    (void)dtmb::core::apply_live_view_scene_input_json(
        state,
        R"({"schema":"dtmb.live_view.draw_packet.v1","event":"draw","verdict":"ok","draw":[]})");
    assert(state.latest_draw_verdict == "ok");
    assert(state.worst_draw_verdict == "ok");

    auto recovered_summary = clean_summary;
    recovered_summary.line_count = 3;
    recovered_summary.draw_packet_count = 2;
    recovered_summary.scene_packet_count = 2;
    recovered_summary.error_packet_count = 1;
    const auto recovered_end =
        dtmb::core::live_view_scene_stream_end_packet_json(recovered_summary, state);
    assert(recovered_end.find(R"("verdict":"degraded")") != std::string::npos);
}

}  // namespace

int main() {
    test_version();
    test_known_ci8_stats_scalar();
    test_parallel_matches_scalar();
    test_rejects_odd_byte_count();
    test_c_abi_validation_and_output();
    test_qam64_soft_demodulate_known_symbol();
    test_qam64_soft_demodulate_parallel_matches_scalar();
    test_qam64_c_abi_validation_and_output();
    test_qam64_normalize_removes_complex_gain();
    test_mixed_radix_fft_known_tone();
    test_c3780_deinterleave_spectrum_preserves_tagged_carrier_provenance();
    test_pn945_detect_phase_tracks_shifted_headers();
    test_pn945_acquisition_finds_unaligned_train_and_coarse_cfo();
    test_pn945_residual_cfo_resolves_five_hz();
    test_symbol_deinterleaver_chunked_matches_source_trace();
    test_symbol_deinterleaver_reset();
    test_sparse_ldpc_min_sum_decodes_single_parity_check();
    test_sparse_ldpc_layered_min_sum_decodes_single_parity_check();
    test_live_view_scene_reducer_composes_controls_and_draw_packet();
    test_live_view_scene_reducer_rejects_control_without_partial_update();
    test_live_view_scene_terminal_verdict_retains_worst_draw_packet();
    test_live_view_scene_terminal_verdict_preserves_clean_and_malformed_recovery();
    std::cout << "dtmb_core_tests: ok\n";
    return 0;
}
