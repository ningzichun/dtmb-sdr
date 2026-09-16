#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dtmb::core {

struct Version {
    std::uint32_t major;
    std::uint32_t minor;
    std::uint32_t patch;
    std::uint32_t abi_major;
    std::uint32_t abi_minor;
};

struct Ci8PowerStats {
    std::size_t sample_count = 0;
    float mean_i2q2 = 0.0F;
    float rms_iq = 0.0F;
    std::size_t clip_count_i = 0;
    std::size_t clip_count_q = 0;
    std::size_t worker_count = 0;
};

struct Ci8PowerStatsOptions {
    std::size_t requested_workers = 0;
    std::size_t min_parallel_samples = 1U << 20U;
};

enum class QamMode { qam16 = 16, qam32 = 32, qam64 = 64 };

enum class PnMode { pn420 = 420, pn595 = 595, pn945 = 945 };

struct PnDefinition {
    PnMode mode;
    std::size_t header_symbols;
    // For PN595 this is the complete fixed reference, not a cyclic core.
    std::size_t core_symbols;
    std::size_t prefix_symbols;
    std::size_t suffix_symbols;
    std::size_t frames_per_superframe;
    float header_to_body_power_ratio;
    std::string_view seed;
    std::array<std::size_t, 4> recurrence_taps;
    std::size_t recurrence_tap_count;

    [[nodiscard]] constexpr bool cyclic_extension() const noexcept { return prefix_symbols != 0; }
    [[nodiscard]] constexpr std::size_t frame_symbols() const noexcept { return header_symbols + 3780; }
    [[nodiscard]] constexpr std::size_t phase_count() const noexcept {
        return cyclic_extension() ? core_symbols : 1;
    }
    [[nodiscard]] constexpr std::size_t default_channel_span() const noexcept {
        return cyclic_extension() ? prefix_symbols : (header_symbols + 1) / 4;
    }
};

// GB 20600-2006 sections 4.6.2.1--3. Recurrences use the appendix bit order.
inline constexpr PnDefinition kPn420Definition{
    PnMode::pn420, 420, 255, 82, 83, 225, 2.0F, "10110000", {0, 2, 3, 7}, 4};
inline constexpr PnDefinition kPn595Definition{
    PnMode::pn595, 595, 595, 0, 0, 216, 1.0F, "0000000001", {0, 7, 0, 0}, 2};
inline constexpr PnDefinition kPn945Definition{
    PnMode::pn945, 945, 511, 217, 217, 200, 2.0F, "111110111", {0, 1, 2, 7}, 4};

[[nodiscard]] constexpr const PnDefinition& pn_definition(PnMode mode) {
    switch (mode) {
    case PnMode::pn420: return kPn420Definition;
    case PnMode::pn595: return kPn595Definition;
    case PnMode::pn945: return kPn945Definition;
    }
    throw std::invalid_argument("PN mode must be pn420, pn595 or pn945");
}

[[nodiscard]] PnMode parse_pn_mode(std::string_view name);
[[nodiscard]] const char* pn_mode_name(PnMode mode);
// Unscaled phase-zero reference chips, 0 -> +1 and 1 -> -1.
[[nodiscard]] std::span<const std::int8_t> pn_reference_chips(PnMode mode);
void pn_header_symbols_cf32(PnMode mode, std::span<float> output, std::size_t phase = 0);

struct QamPoint {
    float real;
    float imag;
};

struct QamDefinition {
    QamMode mode;
    std::size_t bits_per_symbol;
    float average_power;
    std::span<const float> levels;
    // Cross constellations are indexed by the transmitted b0-first label.
    std::span<const QamPoint> points_by_label = {};

    [[nodiscard]] std::size_t axis_bits() const {
        if (bits_per_symbol % 2 != 0) {
            throw std::invalid_argument("cross QAM does not have independent bit axes");
        }
        return bits_per_symbol / 2;
    }
    [[nodiscard]] std::size_t signal_frames_per_fec_group() const noexcept {
        return bits_per_symbol % 2 == 0 ? 1 : 2;
    }
    [[nodiscard]] std::size_t codewords_per_fec_group() const noexcept {
        return bits_per_symbol * signal_frames_per_fec_group() / 2;
    }
    // GB 20600 reflected Gray label, transmitted least-significant bit first.
    [[nodiscard]] static std::size_t axis_label(std::size_t level) noexcept {
        return level ^ (level >> 1U);
    }
    [[nodiscard]] std::size_t nearest_level_index(float value) const noexcept;
    [[nodiscard]] float nearest_level(float value) const noexcept {
        return levels[nearest_level_index(value)];
    }
    [[nodiscard]] QamPoint nearest_point(float real, float imag) const noexcept;
};

[[nodiscard]] const QamDefinition& qam_definition(QamMode mode);
[[nodiscard]] QamMode parse_qam_mode(std::string_view name);
[[nodiscard]] const char* qam_mode_name(QamMode mode);

enum class QamSoftDemapMethod {
    max_log,
    log_sum_exp,
};

struct QamSoftDemapOptions {
    float noise_variance = 1.0F;
    std::size_t requested_workers = 0;
    std::size_t min_parallel_symbols = 1U << 14U;
    QamSoftDemapMethod method = QamSoftDemapMethod::max_log;
};

struct C3780Qam64IntegerTimingOptions {
    int max_delta_samples = 2;
    double min_mse_improvement = 0.25;
};

struct C3780Qam64IntegerTimingResult {
    // Positive delta removes exp(+j*2*pi*physical_bin*delta/3780).
    int delta_samples = 0;
    double baseline_mse = 0.0;
    double corrected_mse = 0.0;
    double relative_improvement = 0.0;
};

struct Pn945EqualizeOptions {
    std::size_t channel_taps = 8;
    float regularization = 1.0e-3F;
    float response_floor = 1.0e-6F;
    float noise_variance = -1.0F;
    int response_window_offset_adjust = 0;
    bool interpolate_body_channel = false;
    bool mmse_unbias = false;
    float mmse_unbias_gain_floor = 0.25F;
    std::span<float> interleaved_channel_power{};
};

struct Pn945EqualizeResult {
    std::size_t pn_phase = 0;
    std::size_t next_pn_phase = 0;
};

enum class Pn945WidebandScaleEstimator {
    dominant_tap,
    least_squares_template,
    masked_frame_taps,
};

enum class Pn945HeaderObservation {
    core_only,
    core_postfix_average,
    core_cyclic_safe_average,
};

struct Pn945WidebandModelOptions {
    float threshold_factor = 3.0F;
    std::size_t guard_taps = 2;
    std::size_t max_span_symbols = 217;
    float min_relative_power = 1.0e-5F;
    float regularization = 1.0e-3F;
    Pn945WidebandScaleEstimator scale_estimator =
        Pn945WidebandScaleEstimator::dominant_tap;
    Pn945HeaderObservation header_observation =
        Pn945HeaderObservation::core_only;
    // Optional trusted transmitted phase for each header, in source order.
    // Empty retains independent phase detection. This span is used only by
    // build_pn945_wideband_channel_model_cf32, not retained in the model.
    std::span<const std::size_t> expected_phases;
};

struct Pn945WidebandChannelModel {
    std::size_t base_pn_phase = 0;
    std::size_t pn_phase = 0;
    std::size_t rotation_symbols = 0;
    std::vector<float> template_taps;
    std::vector<float> template_response_fft;
    std::vector<std::uint8_t> template_tap_mask;
    std::vector<std::size_t> frame_pn_phases;
    std::vector<float> frame_response_scales;
    std::vector<float> frame_template_taps;
    std::size_t significant_taps = 0;
    std::size_t dominant_tap_index = 0;
    std::size_t frame_count = 0;
    float phase_agreement = 0.0F;
    float per_frame_noise_tap_power = 0.0F;
    float noise_variance = 0.0F;
    float truncated_energy_fraction = 0.0F;
    Pn945WidebandScaleEstimator scale_estimator =
        Pn945WidebandScaleEstimator::dominant_tap;
    Pn945HeaderObservation header_observation =
        Pn945HeaderObservation::core_only;
    // Cached observations are needed to restore data leaked into the current
    // header by channel taps before the selected sample origin.
    std::vector<float> frame_headers_cf32;
};

struct Pn945AcquisitionOptions {
    std::size_t max_frames = 16;
    float hit_threshold = 0.35F;
    std::size_t requested_workers = 0;
};

struct Pn945ScheduleAlignment {
    bool valid = false;
    std::size_t observations = 0;
    std::size_t superframe_index = 0;
    int phase_bias = 0;
    double mean_squared_error = 0.0;
    double runner_up_error = 0.0;
    std::size_t inliers = 0;
};

// Fits only the first complete 200-frame PN945 superframe observation window.
// The phase bias is a header-window displacement, not decoded-payload state.
[[nodiscard]] Pn945ScheduleAlignment fit_pn945_phase_schedule(
    std::span<const std::size_t> observed_phases);

[[nodiscard]] std::size_t pn945_phase_for_frame(std::size_t superframe_index) noexcept;

// Normalized coherent correlation of the 511 PN core chips at a known phase.
// Independent of constant complex gain; the header must contain 945 CI8 samples.
[[nodiscard]] float pn945_known_phase_metric_ci8(
    std::span<const std::int8_t> interleaved_header, std::size_t phase);

struct Pn945AcquisitionResult {
    std::size_t phase_offset = 0;
    float mean_metric = 0.0F;
    float max_metric = 0.0F;
    std::size_t hit_count = 0;
    std::size_t observed_frames = 0;
    std::size_t worker_count = 0;
    float coarse_cfo_hz = 0.0F;
    bool coarse_cfo_valid = false;
};

struct Pn945ResidualCfoOptions {
    std::size_t max_frames = 300;
    float min_fit_r_squared = 0.5F;
    std::size_t requested_workers = 0;
};

struct Pn945ResidualCfoResult {
    float cfo_hz = 0.0F;
    float fit_r_squared = 0.0F;
    std::size_t used_frames = 0;
    std::size_t worker_count = 0;
    bool valid = false;
};

inline constexpr std::size_t kC3780FrameBodySymbols = 3780;
inline constexpr std::size_t kC3780SystemInfoSymbols = 36;
inline constexpr std::size_t kC3780DataSymbols = 3744;
inline constexpr std::size_t kPn945HeaderSymbols = kPn945Definition.header_symbols;
inline constexpr std::size_t kPn945FrameSymbols = kPn945Definition.frame_symbols();
inline constexpr std::size_t kDtmbSymbolRateSps = 7'560'000;

struct LdpcSparseGraph {
    std::size_t variable_count = 0;
    std::vector<std::size_t> check_offsets;
    std::vector<std::size_t> edge_variables;

    [[nodiscard]] std::size_t check_count() const noexcept;
    [[nodiscard]] std::size_t edge_count() const noexcept;
};

struct LdpcDecodeOptions {
    std::size_t max_iterations = 50;
    float attenuation = 0.75F;
};

struct LdpcDecodeResult {
    std::size_t iterations = 0;
    bool converged = false;
    std::size_t syndrome_weight = 0;
};

enum class LdpcStreamBitOrder {
    identity,
    reverse_each_byte,
    reverse_each_codeword,
};

struct LdpcHardCandidateScoreOptions {
    std::size_t bit_offset = 0;
    std::size_t max_codewords = 0;
    std::size_t requested_workers = 0;
    LdpcStreamBitOrder stream_bit_order = LdpcStreamBitOrder::identity;
};

struct LdpcHardCandidateScore {
    std::size_t bit_offset = 0;
    std::size_t codewords = 0;
    std::size_t scored_bits = 0;
    std::size_t unused_bits = 0;
    std::size_t clean_rows = 0;
    std::size_t worker_count = 0;
    double mean_syndrome_ratio = 0.0;
    double min_syndrome_ratio = 0.0;
    double max_syndrome_ratio = 0.0;
    std::size_t zero_syndrome_codewords = 0;
    std::vector<std::size_t> syndrome_weights;
};

struct DtmbBchDecodeStats {
    std::size_t block_count = 0;
    std::size_t corrected_errors = 0;
    std::size_t unclean_blocks = 0;
    std::vector<std::uint8_t> block_clean;
    std::vector<std::size_t> block_corrected_errors;
};

enum class SymbolInterleaverMode {
    mode1,
    mode2,
};

struct SymbolInterleaverSpec {
    std::size_t branch_count;
    std::size_t delay_step;

    [[nodiscard]] std::size_t max_branch_delay() const noexcept;
    [[nodiscard]] std::size_t full_stream_latency_symbols() const noexcept;
};

class SymbolDeinterleaverCf32 {
public:
    explicit SymbolDeinterleaverCf32(
        SymbolInterleaverMode mode,
        std::size_t phase = 0,
        float fill_real = 0.0F,
        float fill_imag = 0.0F);

    void reset();
    void process(
        std::span<const float> interleaved_symbols,
        std::span<float> output_symbols);

    [[nodiscard]] SymbolInterleaverSpec spec() const noexcept;
    [[nodiscard]] std::size_t phase() const noexcept;
    [[nodiscard]] std::size_t processed_symbols() const noexcept;
    [[nodiscard]] std::size_t latency_symbols() const noexcept;

private:
    SymbolInterleaverSpec spec_;
    std::size_t phase_;
    float fill_real_;
    float fill_imag_;
    std::size_t processed_symbols_ = 0;
    std::vector<std::size_t> branch_offsets_;
    std::vector<std::size_t> branch_lengths_;
    std::vector<std::size_t> branch_positions_;
    std::vector<float> delay_line_;
};

class RationalResamplerCf32 {
public:
    RationalResamplerCf32(
        std::size_t up_factor,
        std::size_t down_factor,
        std::span<const float> prototype_taps,
        std::size_t requested_workers = 1,
        std::size_t min_parallel_output_samples = 16'384);

    void reset();
    void process(
        std::span<const float> interleaved_input,
        std::vector<float>& interleaved_output);
    void finish(std::vector<float>& interleaved_output);

    [[nodiscard]] std::size_t up_factor() const noexcept;
    [[nodiscard]] std::size_t down_factor() const noexcept;
    [[nodiscard]] std::size_t prototype_tap_count() const noexcept;
    [[nodiscard]] std::size_t processed_input_samples() const noexcept;
    [[nodiscard]] std::size_t produced_output_samples() const noexcept;
    [[nodiscard]] std::size_t max_worker_count() const noexcept;

private:
    void emit_available(bool finishing, std::vector<float>& interleaved_output);
    void emit_range(
        std::size_t first_output_sample,
        std::size_t last_output_sample,
        std::size_t destination_sample_offset,
        std::span<float> interleaved_output) const;
    void compact_history();

    std::size_t up_factor_;
    std::size_t down_factor_;
    std::size_t prototype_tap_count_;
    std::size_t pre_remove_output_samples_;
    std::size_t padded_filter_tap_count_;
    std::vector<std::vector<float>> phase_filters_;
    std::size_t requested_workers_;
    std::size_t min_parallel_output_samples_;
    std::vector<float> history_;
    std::size_t history_base_sample_ = 0;
    std::size_t processed_input_samples_ = 0;
    std::size_t produced_output_samples_ = 0;
    std::size_t max_worker_count_ = 0;
    bool finished_ = false;
};

[[nodiscard]] Version version() noexcept;
[[nodiscard]] const char* build_info() noexcept;
[[nodiscard]] SymbolInterleaverSpec symbol_interleaver_spec(SymbolInterleaverMode mode) noexcept;
[[nodiscard]] std::vector<float> square_root_raised_cosine_taps(
    std::size_t one_sided_symbol_span,
    std::size_t samples_per_symbol,
    float roll_off = 0.05F);
[[nodiscard]] LdpcSparseGraph make_ldpc_sparse_graph(
    const std::vector<std::vector<std::size_t>>& check_variables,
    std::size_t variable_count);
[[nodiscard]] std::size_t ldpc_syndrome_weight(
    std::span<const std::uint8_t> bits,
    const LdpcSparseGraph& graph);
// The clean-check graph variables index one transmitted hard-bit codeword.
// Candidate stream ordering is applied after bit_offset.
[[nodiscard]] LdpcHardCandidateScore ldpc_score_hard_bit_candidate(
    std::span<const std::uint8_t> input_bits,
    const LdpcSparseGraph& clean_check_graph,
    LdpcHardCandidateScoreOptions options = {});
[[nodiscard]] LdpcDecodeResult ldpc_decode_min_sum_sparse(
    std::span<const float> llr,
    const LdpcSparseGraph& graph,
    std::span<std::uint8_t> output_bits,
    LdpcDecodeOptions options = {});

[[nodiscard]] LdpcDecodeResult ldpc_decode_layered_min_sum_sparse(
    std::span<const float> llr,
    const LdpcSparseGraph& graph,
    std::span<std::uint8_t> output_bits,
    LdpcDecodeOptions options = {});
[[nodiscard]] DtmbBchDecodeStats dtmb_bch_descramble_message_bits(
    std::span<const std::uint8_t> ldpc_message_bits,
    std::span<std::uint8_t> output_bytes,
    bool correct = true,
    std::size_t scrambler_skip_bits = 0);
// Periodic resets are measured in transport bits. 32QAM resets every 20 BCH
// payload blocks (15040 bits), including within codeword 2 of a five-word group.
[[nodiscard]] DtmbBchDecodeStats dtmb_bch_descramble_message_bits(
    std::span<const std::uint8_t> ldpc_message_bits,
    std::span<std::uint8_t> output_bytes,
    bool correct,
    std::size_t scrambler_skip_bits,
    std::size_t scrambler_reset_bits);

struct Qam32FrameAlignment {
    std::size_t phase = 0;
    std::array<std::size_t, 2> clean_codewords{};
};

// A two-signal-frame prefix contains two complete codewords at either possible
// packing phase. Lock only when one phase passes LDPC and BCH on both words.
// This is acquisition only: every emitted FEC group still passes its own gates.
[[nodiscard]] Qam32FrameAlignment qam32_frame_alignment(
    std::span<const float> transmitted_llr,
    const LdpcSparseGraph& rate3_graph,
    LdpcDecodeOptions options = {});

[[nodiscard]] Ci8PowerStats ci8_power_stats(
    std::span<const std::int8_t> interleaved_iq,
    Ci8PowerStatsOptions options = {});

// Per-symbol output is b0-first; a positive LLR means bit zero.
void qam_soft_demodulate_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_llr,
    QamMode mode,
    QamSoftDemapOptions options = {});

void qam_normalize_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols,
    QamMode mode);

void qam_normalize_amplitude_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols,
    QamMode mode);

[[nodiscard]] double c3780_qam_frame_mse_cf32(
    std::span<const float> interleaved_data_symbols, QamMode mode);

// Compatibility entry points preserve the existing 64QAM API and ABI.
void qam64_soft_demodulate_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_llr,
    QamSoftDemapOptions options = {});

void qam64_normalize_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols);

void qam64_normalize_amplitude_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols);

// Operates on one already normalized, frequency-deinterleaved data frame,
// before the convolutional symbol deinterleaver. Uses only that frame's
// distance to the 64QAM alphabet, with no decoder or payload observations.
// A rejected candidate leaves the input byte-identical. The caller opts in.
[[nodiscard]] C3780Qam64IntegerTimingResult c3780_qam64_integer_timing_correct_cf32(
    std::span<float> interleaved_data_symbols,
    C3780Qam64IntegerTimingOptions options = {});

// Sliced residual used for current-frame confidence, not a true SNR estimate.
[[nodiscard]] double c3780_qam64_frame_mse_cf32(
    std::span<const float> interleaved_data_symbols);

void mixed_radix_fft_forward_cf32(
    std::span<const float> interleaved_time_samples,
    std::span<float> interleaved_frequency_bins);

void c3780_extract_frame_symbols_cf32(
    std::span<const float> interleaved_time_body,
    std::span<float> interleaved_logical_symbols);

void c3780_deinterleave_spectrum_cf32(
    std::span<const float> interleaved_physical_spectrum,
    std::span<float> interleaved_logical_symbols);

void c3780_extract_data_symbols_cf32(
    std::span<const float> interleaved_time_body,
    std::span<float> interleaved_data_symbols,
    bool normalize_qam64 = true);

void c3780_extract_data_symbols_cf32(
    std::span<const float> interleaved_time_body,
    std::span<float> interleaved_data_symbols,
    QamMode mode);

[[nodiscard]] Pn945AcquisitionResult acquire_pn945_cf32(
    std::span<const float> interleaved_symbols,
    Pn945AcquisitionOptions options = {});

[[nodiscard]] std::size_t pn945_detect_phase_cf32(
    std::span<const float> interleaved_header);

[[nodiscard]] Pn945ResidualCfoResult estimate_pn945_residual_cfo_cf32(
    std::span<const float> interleaved_symbols,
    std::size_t phase_offset,
    Pn945ResidualCfoOptions options = {});

[[nodiscard]] Pn945EqualizeResult pn945_equalize_c3780_frame_cf32(
    std::span<const float> interleaved_header,
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    Pn945EqualizeOptions options = {});

[[nodiscard]] Pn945WidebandChannelModel build_pn945_wideband_channel_model_cf32(
    std::span<const float> interleaved_headers,
    Pn945WidebandModelOptions options = {});

[[nodiscard]] Pn945EqualizeResult pn945_equalize_c3780_frame_wideband_cf32(
    std::span<const float> interleaved_header,
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    const Pn945WidebandChannelModel& model,
    Pn945EqualizeOptions options = {});

[[nodiscard]] Pn945EqualizeResult pn945_equalize_c3780_frame_wideband_cached_cf32(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    const Pn945WidebandChannelModel& model,
    std::size_t model_frame_index,
    Pn945EqualizeOptions options = {});

// Generic names share the established option/result layouts. The PN945 entry
// points above remain available for source and binary compatibility.
using PnAcquisitionOptions = Pn945AcquisitionOptions;
using PnAcquisitionResult = Pn945AcquisitionResult;
using PnResidualCfoOptions = Pn945ResidualCfoOptions;
using PnResidualCfoResult = Pn945ResidualCfoResult;
using PnScheduleAlignment = Pn945ScheduleAlignment;
using PnEqualizeOptions = Pn945EqualizeOptions;
using PnEqualizeResult = Pn945EqualizeResult;
using PnWidebandScaleEstimator = Pn945WidebandScaleEstimator;
using PnHeaderObservation = Pn945HeaderObservation;
using PnWidebandModelOptions = Pn945WidebandModelOptions;
using PnWidebandChannelModel = Pn945WidebandChannelModel;

[[nodiscard]] PnAcquisitionResult acquire_pn_cf32(
    PnMode mode, std::span<const float> symbols, PnAcquisitionOptions options = {});
// Cyclic-extension similarity or fixed-sequence direct correlation, by mode.
[[nodiscard]] float pn_header_metric_ci8(PnMode mode, std::span<const std::int8_t> header);
// PN595 uses direct correlation of its fixed reference; it has only phase zero.
[[nodiscard]] std::size_t pn_detect_phase_cf32(PnMode mode, std::span<const float> header);
[[nodiscard]] float pn_known_phase_metric_ci8(
    PnMode mode, std::span<const std::int8_t> header, std::size_t phase = 0);
[[nodiscard]] std::size_t pn_phase_for_frame(PnMode mode, std::size_t superframe_index);
// Only cyclic modes carry a PN phase schedule; PN595 cannot identify a superframe origin.
[[nodiscard]] PnScheduleAlignment fit_pn_phase_schedule(
    PnMode mode, std::span<const std::size_t> observed_phases);
[[nodiscard]] PnResidualCfoResult estimate_pn_residual_cfo_cf32(
    PnMode mode, std::span<const float> symbols, std::size_t phase_offset,
    PnResidualCfoOptions options = {});
[[nodiscard]] PnEqualizeResult pn_equalize_c3780_frame_cf32(
    PnMode mode, std::span<const float> header, std::span<const float> body,
    std::span<const float> next_header, std::span<float> spectrum,
    PnEqualizeOptions options = {});
[[nodiscard]] PnWidebandChannelModel build_pn_wideband_channel_model_cf32(
    PnMode mode, std::span<const float> headers, PnWidebandModelOptions options = {});
[[nodiscard]] PnEqualizeResult pn_equalize_c3780_frame_wideband_cf32(
    PnMode mode, std::span<const float> header, std::span<const float> body,
    std::span<const float> next_header, std::span<float> spectrum,
    const PnWidebandChannelModel& model, PnEqualizeOptions options = {});
[[nodiscard]] PnEqualizeResult pn_equalize_c3780_frame_wideband_cached_cf32(
    PnMode mode, std::span<const float> body, std::span<const float> next_header,
    std::span<float> spectrum, const PnWidebandChannelModel& model,
    std::size_t model_frame_index, PnEqualizeOptions options = {});

}  // namespace dtmb::core
