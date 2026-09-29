// Assertions must also execute in release builds.
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
#include <stdexcept>
#include <vector>

using namespace dtmb::core;

namespace {
struct Figure4Point { std::string_view bits; float real; float imag; };
// Independent transcription of Figure 4, reading the printed labels b0 first.
constexpr std::array<Figure4Point, 32> figure4{{
    {"10111", -4.5F, 7.5F}, {"10011", -1.5F, 7.5F}, {"11011", 1.5F, 7.5F}, {"11111", 4.5F, 7.5F},
    {"10010", -7.5F, 4.5F}, {"00111", -4.5F, 4.5F}, {"00011", -1.5F, 4.5F}, {"01011", 1.5F, 4.5F},
    {"01111", 4.5F, 4.5F}, {"11010", 7.5F, 4.5F}, {"10110", -7.5F, 1.5F}, {"00110", -4.5F, 1.5F},
    {"00010", -1.5F, 1.5F}, {"01010", 1.5F, 1.5F}, {"01110", 4.5F, 1.5F}, {"11110", 7.5F, 1.5F},
    {"10100", -7.5F, -1.5F}, {"00100", -4.5F, -1.5F}, {"00000", -1.5F, -1.5F}, {"01000", 1.5F, -1.5F},
    {"01100", 4.5F, -1.5F}, {"11100", 7.5F, -1.5F}, {"10000", -7.5F, -4.5F}, {"00101", -4.5F, -4.5F},
    {"00001", -1.5F, -4.5F}, {"01001", 1.5F, -4.5F}, {"01101", 4.5F, -4.5F}, {"11000", 7.5F, -4.5F},
    {"10101", -4.5F, -7.5F}, {"10001", -1.5F, -7.5F}, {"11001", 1.5F, -7.5F}, {"11101", 4.5F, -7.5F},
}};

void check_32qam() {
    const auto& definition = qam_definition(parse_qam_mode("32qam"));
    assert(definition.bits_per_symbol == 5 && definition.average_power == 45.0F);
    assert(definition.signal_frames_per_fec_group() == 2);
    assert(definition.codewords_per_fec_group() == 5);
    std::vector<float> symbols;
    for (const auto& point : figure4) {
        symbols.push_back(point.real); symbols.push_back(point.imag);
    }
    for (auto method : {QamSoftDemapMethod::max_log, QamSoftDemapMethod::log_sum_exp}) {
        QamSoftDemapOptions options;
        options.method = method;
        options.noise_variance = 1.75F;
        options.requested_workers = 3;
        options.min_parallel_symbols = 1;
        std::vector<float> llr(32 * 5);
        qam_soft_demodulate_cf32(symbols, llr, QamMode::qam32, options);
        for (std::size_t n = 0; n < figure4.size(); ++n) {
            for (std::size_t bit = 0; bit < 5; ++bit) {
                assert((llr[n * 5 + bit] < 0) == (figure4[n].bits[bit] == '1'));
            }
        }
        options.requested_workers = 1;
        std::vector<float> scalar(llr.size());
        qam_soft_demodulate_cf32(symbols, scalar, QamMode::qam32, options);
        assert(scalar == llr);
        // Off-grid points and missing corners require a joint I/Q likelihood.
        for (const auto& point : std::array<QamPoint, 5>{{{0, 0}, {7.5F, 7.5F}, {-9, 8}, {2.2F, -3.1F}, {-0.3F, 4.8F}}}) {
            const std::array<float, 2> input{point.real, point.imag};
            std::array<float, 5> actual{};
            qam_soft_demodulate_cf32(input, actual, QamMode::qam32, options);
            for (std::size_t bit = 0; bit < 5; ++bit) {
                double m0 = 1e30, m1 = 1e30, s0 = 0, s1 = 0;
                for (const auto& candidate : figure4) {
                    const double di = double(point.real) - candidate.real;
                    const double dq = double(point.imag) - candidate.imag;
                    const double d = (di * di + dq * dq) / options.noise_variance;
                    if (candidate.bits[bit] == '0') { m0 = std::min(m0, d); s0 += std::exp(-d); }
                    else { m1 = std::min(m1, d); s1 += std::exp(-d); }
                }
                const double expected = method == QamSoftDemapMethod::max_log ? m1 - m0 : std::log(s0 / s1);
                assert(std::abs(actual[bit] - expected) < 2e-5);
            }
        }
    }
    std::vector<float> corners(kC3780DataSymbols * 2, 7.5F);
    assert(c3780_qam_frame_mse_cf32(corners, QamMode::qam32) == 9.0);
}
}  // namespace

int main() {
    check_32qam();
    const auto& qam4 = qam_definition(parse_qam_mode("4qam"));
    assert(qam4.bits_per_symbol == 2 && qam4.average_power == 40.5F);
    assert(qam4.signal_frames_per_fec_group() == 1 && qam4.codewords_per_fec_group() == 1);
    constexpr std::array<float, 8> four_points{-4.5F, -4.5F, 4.5F, -4.5F,
                                                 -4.5F, 4.5F, 4.5F, 4.5F};
    for (auto method : {QamSoftDemapMethod::max_log, QamSoftDemapMethod::log_sum_exp}) {
        QamSoftDemapOptions options;
        options.method = method;
        std::array<float, 8> llr{};
        qam_soft_demodulate_cf32(four_points, llr, QamMode::qam4, options);
        for (std::size_t n = 0; n < 4; ++n) {
            assert((llr[n * 2] < 0) == (four_points[n * 2] > 0));
            assert((llr[n * 2 + 1] < 0) == (four_points[n * 2 + 1] > 0));
        }
        assert(qam4.nearest_level(0.0F) == -4.5F);
    }
    const auto& qam = qam_definition(QamMode::qam16);
    assert(qam.bits_per_symbol == 4 && qam.average_power == 40.0F);
    // Independent ascending-level Gray labels, with b0 transmitted first.
    constexpr std::array<float, 4> levels{-6, -2, 2, 6};
    constexpr std::array<std::array<int, 2>, 4> bits{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    std::vector<float> symbols;
    for (auto i : levels) for (auto q : levels) {
        symbols.push_back(i); symbols.push_back(q);
    }
    for (auto method : {QamSoftDemapMethod::max_log, QamSoftDemapMethod::log_sum_exp}) {
        QamSoftDemapOptions options;
        options.noise_variance = 0.75F;
        options.method = method;
        options.requested_workers = 3;
        options.min_parallel_symbols = 1;
        std::vector<float> llr(16 * 4);
        qam_soft_demodulate_cf32(symbols, llr, QamMode::qam16, options);
        for (std::size_t i = 0; i < 4; ++i) for (std::size_t q = 0; q < 4; ++q) {
            const auto n = (i * 4 + q) * 4;
            for (std::size_t bit = 0; bit < 2; ++bit) {
                assert((llr[n + bit] < 0) == bool(bits[i][bit]));
                assert((llr[n + 2 + bit] < 0) == bool(bits[q][bit]));
                if (method == QamSoftDemapMethod::max_log) {
                    assert(std::abs(llr[n + bit]) >= 16.0F / 0.75F - 1e-5F);
                }
            }
            assert(qam.nearest_level(levels[i] + 0.1F) == levels[i]);
        }
        options.requested_workers = 1;
        std::vector<float> scalar(llr.size());
        qam_soft_demodulate_cf32(symbols, scalar, QamMode::qam16, options);
        assert(scalar == llr);
    }
    // At the I-axis origin, sign-plane LLR is zero; the inner-plane is bit 1.
    std::array<float, 2> midpoint{0, 0};
    std::array<float, 4> midpoint_llr{};
    qam_soft_demodulate_cf32(midpoint, midpoint_llr, QamMode::qam16);
    assert(midpoint_llr[0] == -32 && midpoint_llr[1] == 0);
    assert(midpoint_llr[2] == -32 && midpoint_llr[3] == 0);

    for (auto mode : {QamMode::qam4, QamMode::qam16, QamMode::qam32, QamMode::qam64}) {
        const auto& definition = qam_definition(mode);
        std::vector<float> points;
        if (!definition.points_by_label.empty()) {
            for (auto p : definition.points_by_label) {
                points.push_back(p.real); points.push_back(p.imag);
            }
        } else {
            for (auto i : definition.levels) for (auto q : definition.levels) {
                points.push_back(i); points.push_back(q);
            }
        }
        const auto gain = std::polar(0.37F, 0.07F);
        auto received = points;
        auto amplitude = points;
        for (std::size_t n = 0; n < points.size(); n += 2) {
            auto value = std::complex<float>(points[n], points[n + 1]) * gain;
            received[n] = value.real(); received[n + 1] = value.imag();
            amplitude[n] *= 0.37F; amplitude[n + 1] *= 0.37F;
        }
        auto normalized = received;
        qam_normalize_cf32(received, normalized, mode);
        qam_normalize_amplitude_cf32(amplitude, amplitude, mode);
        for (std::size_t n = 0; n < points.size(); ++n) {
            assert(std::abs(normalized[n] - points[n]) < 4e-5F);
            assert(std::abs(amplitude[n] - points[n]) < 4e-5F);
        }
        std::vector<float> frame(kC3780DataSymbols * 2);
        for (std::size_t n = 0; n < frame.size(); ++n) frame[n] = points[n % points.size()];
        assert(c3780_qam_frame_mse_cf32(frame, mode) == 0);
        for (auto& value : frame) value += 0.25F;
        assert(std::abs(c3780_qam_frame_mse_cf32(frame, mode) - 0.125) < 1e-12);
        if (mode == QamMode::qam64) {
            std::vector<float> legacy(received.size());
            qam64_normalize_cf32(received, legacy);
            assert(legacy == normalized);
            std::vector<float> llr(points.size() / 2 * 6), old(llr.size());
            qam_soft_demodulate_cf32(received, llr, mode);
            qam64_soft_demodulate_cf32(received, old);
            assert(llr == old);
        }
    }
    bool rejected = false;
    try { qam_soft_demodulate_cf32(symbols, midpoint_llr, QamMode::qam16); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { (void)parse_qam_mode("128qam"); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "QAM constellation, LLR, normalization, and compatibility checks passed\n";
}
