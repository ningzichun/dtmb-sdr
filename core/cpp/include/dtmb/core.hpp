#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
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

struct QamSoftDemapOptions {
    float noise_variance = 1.0F;
    std::size_t requested_workers = 0;
    std::size_t min_parallel_symbols = 1U << 14U;
};

struct Pn945EqualizeOptions {
    std::size_t channel_taps = 8;
    float regularization = 1.0e-3F;
    float response_floor = 1.0e-6F;
    float noise_variance = -1.0F;
};

struct Pn945EqualizeResult {
    std::size_t pn_phase = 0;
    std::size_t next_pn_phase = 0;
};

struct Pn945WidebandModelOptions {
    float threshold_factor = 3.0F;
    std::size_t guard_taps = 2;
    std::size_t max_span_symbols = 217;
    float min_relative_power = 1.0e-5F;
    float regularization = 1.0e-3F;
};

struct Pn945WidebandChannelModel {
    std::size_t base_pn_phase = 0;
    std::size_t pn_phase = 0;
    std::size_t rotation_symbols = 0;
    std::vector<float> template_taps;
    std::vector<float> template_response_fft;
    std::vector<std::size_t> frame_pn_phases;
    std::vector<float> frame_response_scales;
    std::size_t significant_taps = 0;
    std::size_t dominant_tap_index = 0;
    std::size_t frame_count = 0;
    float phase_agreement = 0.0F;
    float per_frame_noise_tap_power = 0.0F;
    float noise_variance = 0.0F;
    float truncated_energy_fraction = 0.0F;
};

struct Pn945AcquisitionOptions {
    std::size_t max_frames = 16;
    float hit_threshold = 0.35F;
    std::size_t requested_workers = 0;
};

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
inline constexpr std::size_t kPn945HeaderSymbols = 945;
inline constexpr std::size_t kPn945FrameSymbols = 4725;
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
    bool correct = true);

[[nodiscard]] Ci8PowerStats ci8_power_stats(
    std::span<const std::int8_t> interleaved_iq,
    Ci8PowerStatsOptions options = {});

void qam64_soft_demodulate_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_llr,
    QamSoftDemapOptions options = {});

void qam64_normalize_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols);

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

}  // namespace dtmb::core
