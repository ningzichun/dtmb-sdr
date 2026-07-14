#include "dtmb/core.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

enum class Normalization {
    system_info,
    qam64,
    qam64_amplitude,
    none,
};

enum class Equalizer {
    flat,
    pn,
};

enum class PnEstimator {
    compact,
    wideband,
};

struct PnMmse {
    bool automatic = false;
    float noise_variance = -1.0F;
};

struct SourceCarrierResidualRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

using SourceCarrierChannelRule = SourceCarrierResidualRule;

constexpr std::size_t kPn945CoreSymbols = 511;

constexpr std::array<std::size_t, dtmb::core::kC3780SystemInfoSymbols>
    kC3780SystemInfoPositions{
        0, 140, 279, 419, 420, 560, 699, 839, 840, 980, 1119, 1259,
        1260, 1400, 1539, 1679, 1680, 1820, 1959, 2099, 2100, 2240,
        2379, 2519, 2520, 2660, 2799, 2939, 2940, 3080, 3219, 3359,
        3360, 3500, 3639, 3779,
    };

struct DataDecisionDirectedOptions {
    bool enabled = false;
    float max_relative_error = 0.55F;
    std::size_t min_reliable_carriers = dtmb::core::kC3780DataSymbols / 2;
    float max_axis_inner_fraction = 0.60F;
    float max_hard_bit_bias = 0.0F;
};

enum class DataDecisionDirectedRejectReason {
    none,
    reliable_carriers,
    axis_inner_fraction,
    hard_bit_bias,
};

struct DataDecisionDirectedStats {
    mutable std::mutex mutex;
    std::size_t observed_frames = 0;
    std::size_t refined_frames = 0;
    std::size_t rejected_frames = 0;
    std::size_t reliable_carrier_rejected_frames = 0;
    std::size_t axis_inner_rejected_frames = 0;
    std::size_t hard_bit_bias_rejected_frames = 0;
    std::size_t min_reliable_carriers = std::numeric_limits<std::size_t>::max();
    std::size_t max_reliable_carriers = 0;
    std::uint64_t reliable_carrier_sum = 0;

    void observe(
        std::size_t reliable_carriers,
        bool refined,
        DataDecisionDirectedRejectReason reason =
            DataDecisionDirectedRejectReason::none) {
        const auto lock = std::lock_guard<std::mutex>{mutex};
        ++observed_frames;
        if (refined) {
            ++refined_frames;
        } else {
            ++rejected_frames;
            if (reason == DataDecisionDirectedRejectReason::reliable_carriers) {
                ++reliable_carrier_rejected_frames;
            } else if (reason
                == DataDecisionDirectedRejectReason::axis_inner_fraction) {
                ++axis_inner_rejected_frames;
            } else if (reason == DataDecisionDirectedRejectReason::hard_bit_bias) {
                ++hard_bit_bias_rejected_frames;
            }
        }
        min_reliable_carriers = std::min(min_reliable_carriers, reliable_carriers);
        max_reliable_carriers = std::max(max_reliable_carriers, reliable_carriers);
        reliable_carrier_sum += reliable_carriers;
    }

    [[nodiscard]] double mean_reliable_carriers() const noexcept {
        const auto lock = std::lock_guard<std::mutex>{mutex};
        return observed_frames == 0
            ? 0.0
            : static_cast<double>(reliable_carrier_sum)
                / static_cast<double>(observed_frames);
    }
};

struct Qam64ResidualPoint {
    float observed_real = 0.0F;
    float observed_imag = 0.0F;
    float decision_real = 0.0F;
    float decision_imag = 0.0F;
    std::size_t decision_real_index = 0;
    std::size_t decision_imag_index = 0;
    float residual_real = 0.0F;
    float residual_imag = 0.0F;
    float residual_abs = 0.0F;
    float decision_abs = 0.0F;
    float observed_abs = 0.0F;
};

Qam64ResidualPoint qam64_residual_point(float observed_real, float observed_imag);

struct ComplexGainEstimate {
    std::complex<float> gain{1.0F, 0.0F};
    float coherence = 0.0F;
    bool valid = false;
};

struct FrameNormalizationStats {
    ComplexGainEstimate qam_gain;
    ComplexGainEstimate system_info_gain;
    float qam_residual_ratio = 0.0F;
    float qam_minus_system_info_phase_rad = 0.0F;
    float qam_minus_system_info_phase_mod_pi_over_2_rad = 0.0F;
    int qam_minus_system_info_quadrant = 0;
};

struct FrameWork {
    FrameWork()
        : header_ci8(dtmb::core::kPn945HeaderSymbols * 2),
          body_ci8(dtmb::core::kC3780FrameBodySymbols * 2),
          header_cf32(dtmb::core::kPn945HeaderSymbols * 2),
          body_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          spectrum_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          logical_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          data_cf32(dtmb::core::kC3780DataSymbols * 2),
          channel_power_spectrum_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          channel_power_logical_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          csi_weights(dtmb::core::kC3780DataSymbols) {}

    std::vector<std::int8_t> header_ci8;
    std::vector<std::int8_t> body_ci8;
    std::vector<float> header_cf32;
    std::vector<float> body_cf32;
    std::vector<float> spectrum_cf32;
    std::vector<float> logical_cf32;
    std::vector<float> data_cf32;
    std::vector<float> channel_power_spectrum_cf32;
    std::vector<float> channel_power_logical_cf32;
    std::vector<float> csi_weights;
    FrameNormalizationStats normalization_stats;
    std::size_t sample_start = 0;
};

void prepare_csi_weights(FrameWork& frame);

struct TimingTrackerStats {
    bool enabled = false;
    bool trajectory_enabled = false;
    std::size_t search_radius = 0;
    float hit_threshold = 0.0F;
    std::size_t frames = 0;
    std::size_t corrections = 0;
    std::size_t low_metric_fallbacks = 0;
    std::int64_t min_offset = 0;
    std::int64_t max_offset = 0;
    double mean_abs_offset = 0.0;
    float min_metric = 0.0F;
    float last_metric = 0.0F;
    float min_best_metric = 0.0F;
    float last_best_metric = 0.0F;
    std::size_t trajectory_interval_frames = 0;
    std::size_t trajectory_fit_points = 0;
    std::size_t trajectory_seed_points = 0;
    std::size_t trajectory_frame_index_offset = 0;
    double trajectory_offset_origin_samples = 0.0;
    std::size_t trajectory_reacquisitions = 0;
    std::size_t trajectory_accepted_points = 0;
    std::size_t trajectory_low_metric_fallbacks = 0;
    std::size_t trajectory_innovation_rejections = 0;
    bool trajectory_local_search_enabled = false;
    std::size_t trajectory_local_searches = 0;
    std::size_t trajectory_local_hits = 0;
    std::size_t trajectory_local_corrections = 0;
    std::size_t trajectory_local_low_metric_fallbacks = 0;
    std::size_t trajectory_local_innovation_rejections = 0;
    std::size_t trajectory_local_improvement_rejections = 0;
    std::int64_t trajectory_local_last_delta = 0;
    std::int64_t trajectory_local_max_abs_delta = 0;
    float trajectory_local_last_metric = 0.0F;
    float trajectory_local_best_metric = 0.0F;
    float trajectory_local_last_improvement = 0.0F;
    float trajectory_local_max_improvement = 0.0F;
    std::size_t trajectory_shadow_searches = 0;
    std::size_t trajectory_shadow_hits = 0;
    std::size_t trajectory_shadow_reanchors = 0;
    std::size_t trajectory_scheduled_slips = 0;
    std::int64_t trajectory_min_offset = 0;
    std::int64_t trajectory_max_offset = 0;
    double trajectory_slope_samples_per_frame = 0.0;
    double trajectory_last_observed_offset = 0.0;
    double trajectory_last_innovation = 0.0;
    double trajectory_max_abs_innovation = 0.0;
    std::int64_t trajectory_shadow_last_delta = 0;
    std::int64_t trajectory_shadow_max_abs_delta = 0;
    float trajectory_shadow_last_metric = 0.0F;
    float trajectory_shadow_best_metric = 0.0F;
    float trajectory_min_metric = 0.0F;
    float trajectory_last_metric = 0.0F;
};

struct TimingTrajectorySeedPoint {
    double frame_index = 0.0;
    double offset_samples = 0.0;
};

struct TimingTrajectorySeedState {
    std::vector<TimingTrajectorySeedPoint> points;
    std::optional<std::size_t> frame_index_offset;
    std::optional<double> offset_origin_samples;
};

struct TimingLocalSearchScopedMinImprovementRule {
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float min_improvement = 0.0F;
};

struct TimingLocalSearchTransientRange {
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct TimingShadowResult {
    bool available = false;
    bool hit = false;
    std::size_t selected_start = 0;
    std::int64_t delta_samples = 0;
    float metric = 0.0F;
};

struct TimingFrameEvent {
    std::size_t frame_index = 0;
    std::string_view mode;
    std::string_view event;
    std::size_t nominal_start = 0;
    std::size_t predicted_start = 0;
    std::size_t selected_start = 0;
    std::size_t expected_contiguous_start = 0;
    std::int64_t offset_samples = 0;
    std::int64_t local_offset_samples = 0;
    float metric = 0.0F;
    float best_metric = 0.0F;
    float predicted_metric = 0.0F;
    float improvement = 0.0F;
    double innovation_samples = 0.0;
    bool low_metric_fallback = false;
    bool innovation_rejected = false;
    bool local_correction = false;
    bool scheduled_slip = false;
    std::int64_t local_delta_samples = 0;
    bool shadow_available = false;
    bool shadow_hit = false;
    std::int64_t shadow_delta_samples = 0;
    float shadow_metric = 0.0F;
};

class TimingDiagnostics {
public:
    explicit TimingDiagnostics(std::string path)
        : path_(std::move(path)) {
        if (path_.empty()) {
            return;
        }
        output_ = std::make_unique<std::ofstream>(path_, std::ios::binary);
        if (!*output_) {
            throw std::runtime_error(
                "failed to open timing diagnostics: " + path_);
        }
        *output_
            << "frame_index,mode,event,nominal_start,predicted_start,"
            << "selected_start,expected_contiguous_start,offset_samples,"
            << "local_offset_samples,metric,best_metric,predicted_metric,"
            << "improvement,innovation_samples,low_metric_fallback,"
            << "innovation_rejected,local_correction,scheduled_slip,"
            << "local_delta_samples,shadow_available,shadow_hit,"
            << "shadow_delta_samples,shadow_metric\n";
    }

    TimingDiagnostics(const TimingDiagnostics&) = delete;
    TimingDiagnostics& operator=(const TimingDiagnostics&) = delete;

    ~TimingDiagnostics() {
        if (output_ && !closed_) {
            output_->flush();
        }
    }

    [[nodiscard]] bool enabled() const noexcept {
        return static_cast<bool>(output_);
    }

    void write(const TimingFrameEvent& event) {
        if (!output_) {
            return;
        }
        auto& output = *output_;
        output << event.frame_index << ','
               << event.mode << ','
               << event.event << ','
               << event.nominal_start << ','
               << event.predicted_start << ','
               << event.selected_start << ','
               << event.expected_contiguous_start << ','
               << event.offset_samples << ','
               << event.local_offset_samples << ','
               << event.metric << ','
               << event.best_metric << ','
               << event.predicted_metric << ','
               << event.improvement << ','
               << event.innovation_samples << ','
               << (event.low_metric_fallback ? 1 : 0) << ','
               << (event.innovation_rejected ? 1 : 0) << ','
               << (event.local_correction ? 1 : 0) << ','
               << (event.scheduled_slip ? 1 : 0) << ','
               << event.local_delta_samples << ','
               << (event.shadow_available ? 1 : 0) << ','
               << (event.shadow_hit ? 1 : 0) << ','
               << event.shadow_delta_samples << ','
               << event.shadow_metric << '\n';
        if (!output) {
            throw std::runtime_error(
                "failed to write timing diagnostics: " + path_);
        }
    }

    void close() {
        if (!output_ || closed_) {
            return;
        }
        output_->flush();
        if (!*output_) {
            throw std::runtime_error(
                "failed to close timing diagnostics: " + path_);
        }
        closed_ = true;
    }

private:
    std::string path_;
    std::unique_ptr<std::ofstream> output_;
    bool closed_ = false;
};

struct WidebandModelStats {
    std::size_t count = 0;
    std::size_t span_min = 0;
    std::size_t span_max = 0;
    double span_sum = 0.0;
    std::size_t significant_min = 0;
    std::size_t significant_max = 0;
    double significant_sum = 0.0;
    std::size_t rotation_first = 0;
    std::size_t rotation_last = 0;
    std::size_t rotation_min = 0;
    std::size_t rotation_max = 0;
    float noise_variance_min = 0.0F;
    float noise_variance_max = 0.0F;
    double noise_variance_sum = 0.0;

    void observe(const dtmb::core::Pn945WidebandChannelModel& model) {
        const auto span = model.template_taps.size() / 2;
        const auto significant = model.significant_taps;
        const auto rotation = model.rotation_symbols;
        const auto noise = model.noise_variance;
        if (count == 0) {
            span_min = span_max = span;
            significant_min = significant_max = significant;
            rotation_first = rotation_last = rotation_min = rotation_max = rotation;
            noise_variance_min = noise_variance_max = noise;
        } else {
            span_min = std::min(span_min, span);
            span_max = std::max(span_max, span);
            significant_min = std::min(significant_min, significant);
            significant_max = std::max(significant_max, significant);
            rotation_min = std::min(rotation_min, rotation);
            rotation_max = std::max(rotation_max, rotation);
            rotation_last = rotation;
            noise_variance_min = std::min(noise_variance_min, noise);
            noise_variance_max = std::max(noise_variance_max, noise);
        }
        span_sum += static_cast<double>(span);
        significant_sum += static_cast<double>(significant);
        noise_variance_sum += static_cast<double>(noise);
        ++count;
    }

    [[nodiscard]] double mean_span() const noexcept {
        return count == 0 ? 0.0 : span_sum / static_cast<double>(count);
    }

    [[nodiscard]] double mean_significant() const noexcept {
        return count == 0 ? 0.0 : significant_sum / static_cast<double>(count);
    }

    [[nodiscard]] double mean_noise_variance() const noexcept {
        return count == 0 ? 0.0 : noise_variance_sum / static_cast<double>(count);
    }
};

constexpr std::array<std::string_view, 32> kOddSystemInfoVectors{
    // GB 20600-2006 Appendix G odd system-information vectors. Even vectors
    // are their bitwise complements; the four C=3780 prefix bits remain 1111.
    "00011110101011100100100010110011",
    "01111000110010000010111011010101",
    "01110111110001110010000111011010",
    "00100010100100100111010010001111",
    "01001011111110110001110111100110",
    "00010001101000010100011110111100",
    "01111000001101110010111000101010",
    "00101101100111010111101110000000",
    "01110111001110000010000100100101",
    "00100010011011010111010001110000",
    "01000100000010110001001000010110",
    "00010001010111100100011101000011",
    "00101101011000100111101101111111",
    "01000100111101000001001011101001",
    "01001011000001000001110100011001",
    "00011110010100010100100001001100",
    "01111000110010001101000100101010",
    "00101101100111011000010001111111",
    "01001011111110111110001000011001",
    "00011110101011101011011101001100",
    "01110111110001111101111000100101",
    "00100010100100101000101101110000",
    "01000100111101001110110100010110",
    "00010001101000011011100001000011",
    "01111000001101111101000111010101",
    "00101101011000101000010010000000",
    "01001011000001001110001011100110",
    "00011110010100011011011110110011",
    "01110111001110001101111011011010",
    "00100010011011011000101110001111",
    "01000100000010111110110111101001",
    "00010001010111101011100010111100",
};

constexpr std::array<float, 8> kQam64Levels{-7.0F, -5.0F, -3.0F, -1.0F, 1.0F, 3.0F, 5.0F, 7.0F};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " [--phase-offset N|--auto-sync] [--max-frames N]"
        << " [--sync-frames N] [--acquisition-frames N]"
        << " [--sync-hit-threshold X] [--auto-phase-adjustment N]"
        << " [--timing-search-radius N] [--timing-search-threshold X]"
        << " [--timing-trajectory-interval-frames N]"
        << " [--timing-trajectory-fit-points N]"
        << " [--timing-trajectory-max-innovation-samples X]"
        << " [--timing-trajectory-seed PATH]"
        << " [--timing-trajectory-frame-index-offset N]"
        << " [--timing-trajectory-offset-origin-samples X]"
        << " [--timing-trajectory-state-out PATH]"
        << " [--timing-trajectory-state-out-frame N]"
        << " [--timing-trajectory-local-search]"
        << " [--timing-trajectory-local-search-min-improvement X]"
        << " [--timing-trajectory-local-search-scoped-min-improvement FIRST:LAST:X]"
        << " [--timing-trajectory-local-search-transient]"
        << " [--timing-trajectory-local-search-transient-range FIRST:LAST]"
        << " [--timing-diagnostics PATH]"
        << " [--no-residual-cfo]"
        << " [--frequency-shift-hz X]"
        << " [--workers N] [--batch-frames N]"
        << " [--equalizer flat|pn] [--pn-estimator compact|wideband]"
        << " [--pn-channel-taps N] [--pn-wideband-block-frames N]"
        << " [--pn-wideband-header-observation core|core-postfix|core-cyclic-safe]"
        << " [--pn-wideband-scale-estimator dominant|least-squares|masked-frame-taps]"
        << " [--pn-wideband-max-span-symbols N]"
        << " [--pn-wideband-response-window-offset-adjust N]"
        << " [--pn-wideband-body-channel current|midpoint]"
        << " [--pn-csi-demap] [--pn-csi-weights-out PATH]"
        << " [--pn-wideband-diagnostics PATH]"
        << " [--pn-wideband-frame-diagnostics PATH]"
        << " [--frame-residual-diagnostics PATH]"
        << " [--normalization-diagnostics PATH]"
        << " [--source-carrier-residual CARRIER:FIRST:LAST]"
        << " [--source-carrier-residual-diagnostics-out PATH]"
        << " [--source-carrier-channel CARRIER:FIRST:LAST]"
        << " [--source-carrier-channel-diagnostics-out PATH]"
        << " [--pn-mmse off|auto|X]"
        << " [--remove-dc]"
        << " [--normalization system-info|qam64|qam64-amplitude|none]"
        << " [--system-info-index N|auto]"
        << " [--system-info-auto-observation-frames N]"
        << " [--system-info-auto-min-metric X]"
        << " [--system-info-auto-min-margin X]"
        << " [--data-dd-refine]"
        << " [--data-dd-max-relative-error X]"
        << " [--data-dd-min-reliable-carriers N]"
        << " [--data-dd-max-axis-inner-fraction X]"
        << " [--data-dd-max-hard-bit-bias X]"
        << " [input.ci8|-] [output.cf32|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

std::int64_t parse_signed_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoll(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

float parse_float(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stof(text, &parsed);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

double parse_double(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

TimingTrajectorySeedState load_timing_trajectory_seed(
    const std::string& path) {
    if (path.empty()) {
        return {};
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open timing trajectory seed: " + path);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::invalid_argument("timing trajectory seed is empty: " + path);
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    auto state = TimingTrajectorySeedState{};
    constexpr auto frame_offset_prefix = std::string_view("frame_index_offset,");
    if (line.starts_with(frame_offset_prefix)) {
        state.frame_index_offset = parse_size(
            line.substr(frame_offset_prefix.size()),
            "timing trajectory seed frame index offset");
        if (!std::getline(input, line)) {
            throw std::invalid_argument(
                "timing trajectory seed is missing offset origin: " + path);
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        constexpr auto offset_origin_prefix =
            std::string_view("offset_origin_samples,");
        if (!line.starts_with(offset_origin_prefix)) {
            throw std::invalid_argument(
                "timing trajectory seed is missing offset_origin_samples: " + path);
        }
        state.offset_origin_samples = parse_double(
            line.substr(offset_origin_prefix.size()),
            "timing trajectory seed offset origin");
        if (!std::getline(input, line)) {
            throw std::invalid_argument(
                "timing trajectory seed is missing point header: " + path);
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
    }
    if (line != "frame_index,offset_samples") {
        throw std::invalid_argument(
            "timing trajectory seed header must be frame_index,offset_samples: "
            + path);
    }

    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        const auto comma = line.find(',');
        if (comma == std::string::npos || line.find(',', comma + 1U) != std::string::npos) {
            throw std::invalid_argument(
                "invalid timing trajectory seed row " + std::to_string(line_number)
                + ": " + line);
        }
        const auto frame_index = parse_double(
            line.substr(0, comma),
            "timing trajectory seed frame index");
        const auto offset_samples = parse_double(
            line.substr(comma + 1U),
            "timing trajectory seed offset");
        if (!state.points.empty()
            && frame_index <= state.points.back().frame_index) {
            throw std::invalid_argument(
                "timing trajectory seed frame indices must be strictly increasing");
        }
        state.points.push_back(TimingTrajectorySeedPoint{frame_index, offset_samples});
    }
    if (state.points.empty()) {
        throw std::invalid_argument("timing trajectory seed has no points: " + path);
    }
    return state;
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

[[nodiscard]] std::complex<double> ci8_sample_at(
    std::span<const std::int8_t> samples,
    std::size_t sample) noexcept {
    return {
        static_cast<double>(samples[sample * 2]),
        static_cast<double>(samples[sample * 2 + 1]),
    };
}

[[nodiscard]] float ci8_repeated_window_similarity(
    std::span<const std::int8_t> samples,
    std::size_t start,
    std::size_t first_offset,
    std::size_t second_offset,
    std::size_t length) {
    auto numerator = std::complex<double>{};
    double first_power = 0.0;
    double second_power = 0.0;
    for (std::size_t sample = 0; sample < length; ++sample) {
        const auto first = ci8_sample_at(samples, start + first_offset + sample);
        const auto second = ci8_sample_at(samples, start + second_offset + sample);
        numerator += std::conj(first) * second;
        first_power += std::norm(first);
        second_power += std::norm(second);
    }
    const auto denominator = std::sqrt(std::max(first_power * second_power, 1.0e-30));
    return static_cast<float>(std::abs(numerator) / denominator);
}

[[nodiscard]] float pn945_ci8_cyclic_extension_metric(
    std::span<const std::int8_t> samples,
    std::size_t start) {
    const auto prefix = ci8_repeated_window_similarity(samples, start, 0, 511, 217);
    const auto suffix = ci8_repeated_window_similarity(samples, start, 217, 728, 217);
    return (prefix + suffix) * 0.5F;
}

class ReplayInput {
public:
    explicit ReplayInput(std::istream& input, std::vector<std::int8_t> prefix = {})
        : input_(input),
          prefix_(std::move(prefix)) {}

    std::size_t read(std::span<std::int8_t> buffer) {
        std::size_t copied = 0;
        if (prefix_position_ < prefix_.size()) {
            copied = std::min(buffer.size(), prefix_.size() - prefix_position_);
            std::copy_n(prefix_.data() + prefix_position_, copied, buffer.data());
            prefix_position_ += copied;
        }
        if (copied < buffer.size()) {
            copied += read_bytes(input_, buffer.subspan(copied));
        }
        return copied;
    }

private:
    std::istream& input_;
    std::vector<std::int8_t> prefix_;
    std::size_t prefix_position_ = 0;
};

class TrackingFrameReader {
public:
    TrackingFrameReader(
        ReplayInput& input,
        std::size_t phase_offset,
        std::size_t search_radius,
        float hit_threshold,
        std::size_t trajectory_interval_frames,
        std::size_t trajectory_fit_points,
        double trajectory_max_innovation_samples,
        std::vector<TimingTrajectorySeedPoint> trajectory_seed_points,
        std::size_t trajectory_frame_index_offset,
        double trajectory_offset_origin_samples,
        std::string trajectory_state_out_path,
        std::size_t trajectory_state_out_frame,
        bool trajectory_local_search,
        float trajectory_local_search_min_improvement,
        std::vector<TimingLocalSearchScopedMinImprovementRule>
            trajectory_local_search_scoped_min_improvements,
        bool trajectory_local_search_transient,
        std::vector<TimingLocalSearchTransientRange>
            trajectory_local_search_transient_ranges,
        TimingDiagnostics* timing_diagnostics)
        : input_(input),
          buffer_start_sample_(phase_offset),
          next_expected_sample_(phase_offset),
          trajectory_origin_sample_(phase_offset),
          search_radius_(search_radius),
          hit_threshold_(hit_threshold),
          trajectory_interval_frames_(trajectory_interval_frames),
          trajectory_fit_points_(trajectory_fit_points),
          trajectory_max_innovation_samples_(trajectory_max_innovation_samples),
          trajectory_frame_index_offset_(trajectory_frame_index_offset),
          trajectory_offset_origin_samples_(trajectory_offset_origin_samples),
          trajectory_state_out_path_(std::move(trajectory_state_out_path)),
          trajectory_state_out_frame_(trajectory_state_out_frame),
          trajectory_local_search_(trajectory_local_search),
          trajectory_local_search_min_improvement_(
              trajectory_local_search_min_improvement),
          trajectory_local_search_scoped_min_improvements_(
              std::move(trajectory_local_search_scoped_min_improvements)),
          trajectory_local_search_transient_(trajectory_local_search_transient),
          trajectory_local_search_transient_ranges_(
              std::move(trajectory_local_search_transient_ranges)),
          timing_diagnostics_(timing_diagnostics) {
        stats_.enabled = search_radius_ != 0 && trajectory_interval_frames_ == 0;
        stats_.trajectory_enabled = trajectory_interval_frames_ != 0;
        stats_.search_radius = search_radius_;
        stats_.hit_threshold = hit_threshold_;
        stats_.trajectory_interval_frames = trajectory_interval_frames_;
        stats_.trajectory_fit_points = trajectory_fit_points_;
        stats_.trajectory_seed_points = trajectory_seed_points.size();
        stats_.trajectory_frame_index_offset = trajectory_frame_index_offset_;
        stats_.trajectory_offset_origin_samples = trajectory_offset_origin_samples_;
        stats_.trajectory_local_search_enabled = trajectory_local_search_;
        trajectory_seeded_restart_ = !trajectory_seed_points.empty();
        for (const auto& point : trajectory_seed_points) {
            trajectory_points_.push_back(TrajectoryPoint{
                point.frame_index,
                point.offset_samples,
            });
        }
        while (trajectory_points_.size() > trajectory_fit_points_) {
            trajectory_points_.pop_front();
        }
        fit_trajectory();
        stats_.trajectory_slope_samples_per_frame =
            trajectory_slope_samples_per_frame_;
    }

    bool read(FrameWork& frame, std::size_t& trailing_ci8_bytes) {
        if (search_radius_ == 0) {
            return read_untracked(frame, trailing_ci8_bytes);
        }
        if (trajectory_interval_frames_ != 0) {
            return read_trajectory(frame, trailing_ci8_bytes);
        }
        const auto search_min = next_expected_sample_ > search_radius_
            ? next_expected_sample_ - search_radius_
            : 0;
        const auto bounded_min = std::max(search_min, buffer_start_sample_);
        const auto search_max = next_expected_sample_ + search_radius_;
        ensure_until(search_max + dtmb::core::kPn945FrameSymbols);

        const auto available_end_sample = buffer_start_sample_ + buffer_.size() / 2;
        if (bounded_min + dtmb::core::kPn945FrameSymbols > available_end_sample) {
            trailing_ci8_bytes = buffer_.size();
            return false;
        }
        const auto bounded_max = std::min(
            search_max,
            available_end_sample - dtmb::core::kPn945FrameSymbols);
        const auto expected_candidate = std::clamp(
            next_expected_sample_,
            bounded_min,
            bounded_max);

        auto selected_start = expected_candidate;
        auto selected_metric = metric_at(expected_candidate);
        for (auto candidate = bounded_min; candidate <= bounded_max; ++candidate) {
            const auto metric = metric_at(candidate);
            if (metric > selected_metric) {
                selected_start = candidate;
                selected_metric = metric;
            }
        }
        const auto best_metric = selected_metric;
        auto low_metric_fallback = false;
        if (selected_metric < hit_threshold_) {
            selected_start = expected_candidate;
            selected_metric = metric_at(expected_candidate);
            low_metric_fallback = true;
        }

        copy_frame(selected_start, frame);
        if (timing_diagnostics_ != nullptr && timing_diagnostics_->enabled()) {
            const auto offset = static_cast<std::int64_t>(selected_start)
                - static_cast<std::int64_t>(next_expected_sample_);
            timing_diagnostics_->write(TimingFrameEvent{
                stats_.frames,
                "tracker",
                low_metric_fallback
                    ? std::string_view("tracker_low_metric_fallback")
                    : offset != 0
                        ? std::string_view("tracker_correction")
                        : std::string_view("tracker"),
                next_expected_sample_,
                expected_candidate,
                selected_start,
                next_expected_sample_,
                offset,
                0,
                selected_metric,
                best_metric,
                metric_at(expected_candidate),
                0.0F,
                0.0,
                low_metric_fallback,
                false,
                offset != 0,
                offset != 0,
                offset,
                false,
                false,
                0,
                0.0F,
            });
        }
        record_selection(selected_start, selected_metric, best_metric, low_metric_fallback);
        next_expected_sample_ = selected_start + dtmb::core::kPn945FrameSymbols;
        discard_before(next_expected_sample_ > search_radius_
                           ? next_expected_sample_ - search_radius_
                           : 0);
        return true;
    }

    [[nodiscard]] const TimingTrackerStats& stats() const noexcept {
        return stats_;
    }

private:
    struct TrajectoryPoint {
        double frame_index = 0.0;
        double offset_samples = 0.0;
    };

    [[nodiscard]] std::size_t trajectory_nominal_start(std::size_t frame_index) const {
        if (frame_index
            > (std::numeric_limits<std::size_t>::max() - trajectory_origin_sample_)
                / dtmb::core::kPn945FrameSymbols) {
            throw std::overflow_error("timing trajectory frame index overflow");
        }
        return trajectory_origin_sample_
            + frame_index * dtmb::core::kPn945FrameSymbols;
    }

    [[nodiscard]] std::size_t apply_trajectory_offset(
        std::size_t nominal_start,
        double offset_samples) const {
        const auto rounded = static_cast<std::int64_t>(std::llround(offset_samples));
        if (rounded < 0) {
            const auto magnitude = static_cast<std::uint64_t>(-(rounded + 1)) + 1;
            return magnitude > nominal_start
                ? 0
                : nominal_start - static_cast<std::size_t>(magnitude);
        }
        const auto positive = static_cast<std::size_t>(rounded);
        if (positive > std::numeric_limits<std::size_t>::max() - nominal_start) {
            throw std::overflow_error("timing trajectory sample offset overflow");
        }
        return nominal_start + positive;
    }

    [[nodiscard]] double trajectory_offset_at(std::size_t frame_index) const noexcept {
        return trajectory_intercept_samples_
            + trajectory_slope_samples_per_frame_
                * (static_cast<double>(trajectory_frame_index_offset_)
                   + static_cast<double>(frame_index))
            - trajectory_offset_origin_samples_;
    }

    void fit_trajectory() {
        if (trajectory_points_.empty()) {
            trajectory_intercept_samples_ = 0.0;
            trajectory_slope_samples_per_frame_ = 0.0;
            return;
        }
        if (trajectory_points_.size() == 1) {
            trajectory_intercept_samples_ = trajectory_points_.front().offset_samples;
            trajectory_slope_samples_per_frame_ = 0.0;
            return;
        }

        double mean_frame = 0.0;
        double mean_offset = 0.0;
        for (const auto& point : trajectory_points_) {
            mean_frame += point.frame_index;
            mean_offset += point.offset_samples;
        }
        mean_frame /= static_cast<double>(trajectory_points_.size());
        mean_offset /= static_cast<double>(trajectory_points_.size());

        double covariance = 0.0;
        double frame_variance = 0.0;
        for (const auto& point : trajectory_points_) {
            const auto frame_delta = point.frame_index - mean_frame;
            covariance += frame_delta * (point.offset_samples - mean_offset);
            frame_variance += frame_delta * frame_delta;
        }
        trajectory_slope_samples_per_frame_ =
            frame_variance == 0.0 ? 0.0 : covariance / frame_variance;
        trajectory_intercept_samples_ =
            mean_offset - trajectory_slope_samples_per_frame_ * mean_frame;
    }

    void add_trajectory_point(std::size_t frame_index, std::int64_t offset_samples) {
        trajectory_points_.push_back(TrajectoryPoint{
            static_cast<double>(trajectory_frame_index_offset_)
                + static_cast<double>(frame_index),
            static_cast<double>(offset_samples) + trajectory_offset_origin_samples_,
        });
        while (trajectory_points_.size() > trajectory_fit_points_) {
            trajectory_points_.pop_front();
        }
        fit_trajectory();
        stats_.trajectory_accepted_points += 1;
        stats_.trajectory_slope_samples_per_frame = trajectory_slope_samples_per_frame_;
        stats_.trajectory_last_observed_offset = static_cast<double>(offset_samples);
    }

    void reset_trajectory_at(std::size_t frame_index, std::size_t selected_start) {
        const auto nominal_start = trajectory_nominal_start(frame_index);
        const auto observed_offset = static_cast<std::int64_t>(selected_start)
            - static_cast<std::int64_t>(nominal_start);
        trajectory_points_.clear();
        trajectory_local_offset_samples_ = 0;
        add_trajectory_point(frame_index, observed_offset);
    }

    void record_trajectory_selection(std::size_t selected_start) {
        const auto nominal_start = trajectory_nominal_start(trajectory_frame_index_);
        const auto offset = static_cast<std::int64_t>(selected_start)
            - static_cast<std::int64_t>(nominal_start);
        if (trajectory_frame_index_ == 0) {
            stats_.trajectory_min_offset = offset;
            stats_.trajectory_max_offset = offset;
        } else {
            stats_.trajectory_min_offset = std::min(stats_.trajectory_min_offset, offset);
            stats_.trajectory_max_offset = std::max(stats_.trajectory_max_offset, offset);
            const auto expected_contiguous = previous_trajectory_start_
                + dtmb::core::kPn945FrameSymbols;
            stats_.trajectory_scheduled_slips += selected_start != expected_contiguous ? 1U : 0U;
        }
        previous_trajectory_start_ = selected_start;
    }

    TimingShadowResult record_trajectory_shadow_search(std::size_t predicted_start) {
        constexpr auto half_frame = dtmb::core::kPn945FrameSymbols / 2;
        const auto search_min = predicted_start > half_frame
            ? predicted_start - half_frame
            : 0;
        const auto bounded_min = std::max(search_min, buffer_start_sample_);
        const auto search_max = predicted_start + half_frame;
        ensure_until(search_max + dtmb::core::kPn945FrameSymbols);

        const auto available_end_sample = buffer_start_sample_ + buffer_.size() / 2;
        if (bounded_min + dtmb::core::kPn945FrameSymbols > available_end_sample) {
            return {};
        }
        const auto bounded_max = std::min(
            search_max,
            available_end_sample - dtmb::core::kPn945FrameSymbols);
        auto best_start = std::clamp(predicted_start, bounded_min, bounded_max);
        auto best_metric = metric_at(best_start);
        for (auto candidate = bounded_min; candidate <= bounded_max; ++candidate) {
            const auto metric = metric_at(candidate);
            if (metric > best_metric) {
                best_start = candidate;
                best_metric = metric;
            }
        }

        const auto delta = static_cast<std::int64_t>(best_start)
            - static_cast<std::int64_t>(predicted_start);
        ++stats_.trajectory_shadow_searches;
        stats_.trajectory_shadow_hits += best_metric >= hit_threshold_ ? 1U : 0U;
        stats_.trajectory_shadow_last_delta = delta;
        stats_.trajectory_shadow_max_abs_delta = std::max(
            stats_.trajectory_shadow_max_abs_delta,
            std::abs(delta));
        stats_.trajectory_shadow_last_metric = best_metric;
        stats_.trajectory_shadow_best_metric = std::max(
            stats_.trajectory_shadow_best_metric,
            best_metric);
        return TimingShadowResult{
            true,
            best_metric >= hit_threshold_,
            best_start,
            delta,
            best_metric,
        };
    }

    bool read_trajectory(FrameWork& frame, std::size_t& trailing_ci8_bytes) {
        const auto nominal_start = trajectory_nominal_start(trajectory_frame_index_);
        auto selected_start = apply_trajectory_offset(
            nominal_start,
            trajectory_offset_at(trajectory_frame_index_));
        selected_start = apply_trajectory_offset(
            selected_start,
            static_cast<double>(trajectory_local_offset_samples_));
        const auto seeded_restart = trajectory_seeded_restart_
            && trajectory_frame_index_ == 0;
        if (seeded_restart) {
            selected_start = nominal_start;
        }
        const auto predicted_start = selected_start;
        auto event_name = seeded_restart
            ? std::string_view("trajectory_seeded_restart")
            : std::string_view("trajectory_predicted");
        auto metric = 0.0F;
        auto best_metric_for_event = 0.0F;
        auto predicted_metric = 0.0F;
        auto improvement = 0.0F;
        auto innovation = 0.0;
        auto low_metric_fallback = false;
        auto innovation_rejected = false;
        auto local_correction = false;
        auto local_delta = std::int64_t{0};
        auto shadow = TimingShadowResult{};
        const auto reacquire_phase =
            (trajectory_frame_index_offset_ % trajectory_interval_frames_
             + trajectory_frame_index_ % trajectory_interval_frames_)
            % trajectory_interval_frames_;
        const auto reacquire = !seeded_restart && reacquire_phase == 0;
        if (reacquire) {
            const auto search_min = selected_start > search_radius_
                ? selected_start - search_radius_
                : 0;
            const auto bounded_min = std::max(search_min, buffer_start_sample_);
            const auto search_max = selected_start + search_radius_;
            ensure_until(search_max + dtmb::core::kPn945FrameSymbols);

            const auto available_end_sample = buffer_start_sample_ + buffer_.size() / 2;
            if (bounded_min + dtmb::core::kPn945FrameSymbols > available_end_sample) {
                trailing_ci8_bytes = buffer_.size();
                return false;
            }
            const auto bounded_max = std::min(
                search_max,
                available_end_sample - dtmb::core::kPn945FrameSymbols);
            auto best_start = std::clamp(selected_start, bounded_min, bounded_max);
            predicted_metric = metric_at(best_start);
            auto best_metric = predicted_metric;
            for (auto candidate = bounded_min; candidate <= bounded_max; ++candidate) {
                const auto metric = metric_at(candidate);
                if (metric > best_metric) {
                    best_start = candidate;
                    best_metric = metric;
                }
            }
            metric = best_metric;
            best_metric_for_event = best_metric;
            event_name = std::string_view("trajectory_reacquire");
            ++stats_.trajectory_reacquisitions;
            if (stats_.trajectory_reacquisitions == 1) {
                stats_.trajectory_min_metric = best_metric;
            } else {
                stats_.trajectory_min_metric =
                    std::min(stats_.trajectory_min_metric, best_metric);
            }
            stats_.trajectory_last_metric = best_metric;
            if (best_metric >= hit_threshold_) {
                const auto observed_offset = static_cast<std::int64_t>(best_start)
                    - static_cast<std::int64_t>(nominal_start);
                const auto predicted_offset =
                    trajectory_offset_at(trajectory_frame_index_);
                const auto expected_offset = predicted_offset
                    + static_cast<double>(trajectory_local_offset_samples_);
                innovation =
                    static_cast<double>(observed_offset) - expected_offset;
                stats_.trajectory_last_innovation = innovation;
                stats_.trajectory_max_abs_innovation = std::max(
                    stats_.trajectory_max_abs_innovation,
                    std::abs(innovation));
                if (trajectory_points_.size() >= 2
                    && std::abs(innovation) > trajectory_max_innovation_samples_) {
                    ++stats_.trajectory_innovation_rejections;
                    innovation_rejected = true;
                    event_name =
                        std::string_view("trajectory_reacquire_innovation_rejected");
                } else {
                    add_trajectory_point(trajectory_frame_index_, observed_offset);
                    selected_start = apply_trajectory_offset(
                        nominal_start,
                        trajectory_offset_at(trajectory_frame_index_));
                    trajectory_local_offset_samples_ = 0;
                    event_name = std::string_view("trajectory_reacquire_accepted");
                }
            } else {
                ++stats_.trajectory_low_metric_fallbacks;
                low_metric_fallback = true;
                event_name = std::string_view("trajectory_reacquire_low_metric");
                shadow = record_trajectory_shadow_search(selected_start);
                if (shadow.hit) {
                    selected_start = shadow.selected_start;
                    metric = shadow.metric;
                    best_metric_for_event = shadow.metric;
                    reset_trajectory_at(trajectory_frame_index_, selected_start);
                    ++stats_.trajectory_shadow_reanchors;
                    event_name = std::string_view("trajectory_shadow_reanchor");
                }
            }
        } else if (trajectory_local_search_) {
            const auto predicted_start = selected_start;
            const auto search_min = predicted_start > search_radius_
                ? predicted_start - search_radius_
                : 0;
            const auto bounded_min = std::max(search_min, buffer_start_sample_);
            const auto search_max = predicted_start + search_radius_;
            ensure_until(search_max + dtmb::core::kPn945FrameSymbols);

            const auto available_end_sample = buffer_start_sample_ + buffer_.size() / 2;
            if (bounded_min + dtmb::core::kPn945FrameSymbols > available_end_sample) {
                trailing_ci8_bytes = buffer_.size();
                return false;
            }
            const auto bounded_max = std::min(
                search_max,
                available_end_sample - dtmb::core::kPn945FrameSymbols);
            auto best_start = std::clamp(predicted_start, bounded_min, bounded_max);
            auto best_metric = metric_at(best_start);
            predicted_metric = best_metric;
            for (auto candidate = bounded_min; candidate <= bounded_max; ++candidate) {
                const auto metric = metric_at(candidate);
                if (metric > best_metric) {
                    best_start = candidate;
                    best_metric = metric;
                }
            }

            const auto delta = static_cast<std::int64_t>(best_start)
                - static_cast<std::int64_t>(predicted_start);
            improvement = best_metric - predicted_metric;
            local_delta = delta;
            metric = best_metric;
            best_metric_for_event = best_metric;
            event_name = std::string_view("trajectory_local_hit");
            ++stats_.trajectory_local_searches;
            stats_.trajectory_local_last_delta = delta;
            stats_.trajectory_local_max_abs_delta = std::max(
                stats_.trajectory_local_max_abs_delta,
                std::abs(delta));
            stats_.trajectory_local_last_metric = best_metric;
            stats_.trajectory_local_best_metric = std::max(
                stats_.trajectory_local_best_metric,
                best_metric);
            stats_.trajectory_local_last_improvement = improvement;
            stats_.trajectory_local_max_improvement = std::max(
                stats_.trajectory_local_max_improvement,
                improvement);
            if (best_metric < hit_threshold_) {
                ++stats_.trajectory_local_low_metric_fallbacks;
                low_metric_fallback = true;
                event_name = std::string_view("trajectory_local_low_metric");
            } else if (
                std::abs(static_cast<double>(delta))
                > trajectory_max_innovation_samples_) {
                ++stats_.trajectory_local_innovation_rejections;
                innovation_rejected = true;
                event_name =
                    std::string_view("trajectory_local_innovation_rejected");
            } else if (
                delta != 0
                && improvement
                    < trajectory_local_search_min_improvement_for(
                        trajectory_frame_index_)) {
                ++stats_.trajectory_local_improvement_rejections;
                event_name =
                    std::string_view("trajectory_local_improvement_rejected");
            } else {
                ++stats_.trajectory_local_hits;
                if (best_start != predicted_start) {
                    ++stats_.trajectory_local_corrections;
                    const auto transient_correction =
                        trajectory_local_search_transient_for(trajectory_frame_index_);
                    if (!transient_correction) {
                        trajectory_local_offset_samples_ += delta;
                    }
                    local_correction = true;
                    event_name = transient_correction
                        ? std::string_view("trajectory_local_transient_correction")
                        : std::string_view("trajectory_local_correction");
                }
                selected_start = best_start;
            }
        }

        ensure_until(selected_start + dtmb::core::kPn945FrameSymbols);
        const auto available_end_sample = buffer_start_sample_ + buffer_.size() / 2;
        if (selected_start < buffer_start_sample_
            || selected_start + dtmb::core::kPn945FrameSymbols > available_end_sample) {
            trailing_ci8_bytes = buffer_.size();
            return false;
        }
        copy_frame(selected_start, frame);
        if (timing_diagnostics_ != nullptr && timing_diagnostics_->enabled()) {
            const auto expected_contiguous = trajectory_frame_index_ == 0
                ? selected_start
                : previous_trajectory_start_ + dtmb::core::kPn945FrameSymbols;
            const auto scheduled_slip = trajectory_frame_index_ != 0
                && selected_start != expected_contiguous;
            timing_diagnostics_->write(TimingFrameEvent{
                trajectory_frame_index_,
                "trajectory",
                event_name,
                nominal_start,
                predicted_start,
                selected_start,
                expected_contiguous,
                static_cast<std::int64_t>(selected_start)
                    - static_cast<std::int64_t>(nominal_start),
                trajectory_local_offset_samples_,
                metric,
                best_metric_for_event,
                predicted_metric,
                improvement,
                innovation,
                low_metric_fallback,
                innovation_rejected,
                local_correction,
                scheduled_slip,
                local_delta,
                shadow.available,
                shadow.hit,
                shadow.delta_samples,
                shadow.metric,
            });
        }
        record_trajectory_selection(selected_start);
        if (!trajectory_state_out_path_.empty()
            && trajectory_frame_index_ == trajectory_state_out_frame_) {
            write_trajectory_state(selected_start, nominal_start);
        }
        ++trajectory_frame_index_;

        const auto next_nominal = trajectory_nominal_start(trajectory_frame_index_);
        const auto next_scheduled = apply_trajectory_offset(
            next_nominal,
            trajectory_offset_at(trajectory_frame_index_));
        constexpr auto retained_history = dtmb::core::kPn945FrameSymbols / 2;
        discard_before(next_scheduled > retained_history
                           ? next_scheduled - retained_history
                           : 0);
        return true;
    }

    void write_trajectory_state(
        std::size_t selected_start,
        std::size_t nominal_start) {
        std::ofstream output(trajectory_state_out_path_, std::ios::binary);
        if (!output) {
            throw std::runtime_error(
                "failed to open timing trajectory state output: "
                + trajectory_state_out_path_);
        }
        const auto global_frame_index = trajectory_frame_index_offset_
            + trajectory_frame_index_;
        const auto local_selected_offset = static_cast<std::int64_t>(selected_start)
            - static_cast<std::int64_t>(nominal_start);
        const auto global_offset_origin =
            static_cast<double>(local_selected_offset)
            + trajectory_offset_origin_samples_;
        output << "frame_index_offset," << global_frame_index << '\n'
               << "offset_origin_samples," << global_offset_origin << '\n'
               << "frame_index,offset_samples\n";
        for (const auto& point : trajectory_points_) {
            output << point.frame_index << ',' << point.offset_samples << '\n';
        }
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to write timing trajectory state output: "
                + trajectory_state_out_path_);
        }
    }

    bool read_untracked(FrameWork& frame, std::size_t& trailing_ci8_bytes) {
        frame.sample_start = next_expected_sample_;
        const auto header_bytes = input_.read(frame.header_ci8);
        if (header_bytes == 0) {
            return false;
        }
        if (header_bytes != frame.header_ci8.size()) {
            trailing_ci8_bytes = header_bytes;
            return false;
        }
        const auto body_bytes = input_.read(frame.body_ci8);
        if (body_bytes != frame.body_ci8.size()) {
            trailing_ci8_bytes = frame.header_ci8.size() + body_bytes;
            return false;
        }
        next_expected_sample_ += dtmb::core::kPn945FrameSymbols;
        return true;
    }

    void ensure_until(std::size_t sample_end) {
        if (sample_end <= buffer_start_sample_) {
            return;
        }
        auto required_bytes = (sample_end - buffer_start_sample_) * 2;
        while (!input_ended_ && buffer_.size() < required_bytes) {
            std::vector<std::int8_t> chunk(
                std::max<std::size_t>(required_bytes - buffer_.size(), 1U << 16U));
            const auto count = input_.read(chunk);
            if (count == 0) {
                input_ended_ = true;
                break;
            }
            buffer_.insert(buffer_.end(), chunk.begin(), chunk.begin() + count);
            required_bytes = (sample_end - buffer_start_sample_) * 2;
        }
    }

    [[nodiscard]] float metric_at(std::size_t candidate_start) const {
        const auto relative_start = candidate_start - buffer_start_sample_;
        return pn945_ci8_cyclic_extension_metric(buffer_, relative_start);
    }

    [[nodiscard]] float trajectory_local_search_min_improvement_for(
        std::size_t frame_index) const noexcept {
        for (const auto& rule : trajectory_local_search_scoped_min_improvements_) {
            if (frame_index >= rule.first_frame && frame_index <= rule.last_frame) {
                return rule.min_improvement;
            }
        }
        return trajectory_local_search_min_improvement_;
    }

    [[nodiscard]] bool trajectory_local_search_transient_for(
        std::size_t frame_index) const noexcept {
        if (trajectory_local_search_transient_) {
            return true;
        }
        for (const auto& range : trajectory_local_search_transient_ranges_) {
            if (frame_index >= range.first_frame && frame_index <= range.last_frame) {
                return true;
            }
        }
        return false;
    }

    void copy_frame(std::size_t sample_start, FrameWork& frame) const {
        const auto offset = (sample_start - buffer_start_sample_) * 2;
        frame.sample_start = sample_start;
        std::copy_n(buffer_.data() + offset, frame.header_ci8.size(), frame.header_ci8.data());
        std::copy_n(
            buffer_.data() + offset + frame.header_ci8.size(),
            frame.body_ci8.size(),
            frame.body_ci8.data());
    }

    void record_selection(
        std::size_t selected_start,
        float metric,
        float best_metric,
        bool low_metric_fallback) {
        const auto offset = static_cast<std::int64_t>(selected_start)
            - static_cast<std::int64_t>(next_expected_sample_);
        if (stats_.frames == 0) {
            stats_.min_offset = offset;
            stats_.max_offset = offset;
            stats_.min_metric = metric;
            stats_.min_best_metric = best_metric;
        } else {
            stats_.min_offset = std::min(stats_.min_offset, offset);
            stats_.max_offset = std::max(stats_.max_offset, offset);
            stats_.min_metric = std::min(stats_.min_metric, metric);
            stats_.min_best_metric = std::min(stats_.min_best_metric, best_metric);
        }
        stats_.corrections += offset != 0 ? 1U : 0U;
        stats_.low_metric_fallbacks += low_metric_fallback ? 1U : 0U;
        stats_.mean_abs_offset +=
            (static_cast<double>(std::abs(offset)) - stats_.mean_abs_offset)
            / static_cast<double>(stats_.frames + 1U);
        stats_.last_metric = metric;
        stats_.last_best_metric = best_metric;
        ++stats_.frames;
    }

    void discard_before(std::size_t sample) {
        if (sample <= buffer_start_sample_) {
            return;
        }
        const auto available_samples = buffer_.size() / 2;
        const auto discard_samples = std::min(sample - buffer_start_sample_, available_samples);
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(discard_samples * 2));
        buffer_start_sample_ += discard_samples;
    }

    ReplayInput& input_;
    std::vector<std::int8_t> buffer_;
    std::size_t buffer_start_sample_ = 0;
    std::size_t next_expected_sample_ = 0;
    std::size_t trajectory_origin_sample_ = 0;
    std::size_t search_radius_ = 0;
    float hit_threshold_ = 0.0F;
    std::size_t trajectory_interval_frames_ = 0;
    std::size_t trajectory_fit_points_ = 0;
    double trajectory_max_innovation_samples_ = 0.0;
    std::size_t trajectory_frame_index_offset_ = 0;
    double trajectory_offset_origin_samples_ = 0.0;
    std::string trajectory_state_out_path_;
    std::size_t trajectory_state_out_frame_ = 0;
    bool trajectory_local_search_ = false;
    bool trajectory_local_search_transient_ = false;
    std::vector<TimingLocalSearchTransientRange>
        trajectory_local_search_transient_ranges_;
    float trajectory_local_search_min_improvement_ = 0.0F;
    std::vector<TimingLocalSearchScopedMinImprovementRule>
        trajectory_local_search_scoped_min_improvements_;
    std::int64_t trajectory_local_offset_samples_ = 0;
    std::size_t trajectory_frame_index_ = 0;
    std::size_t previous_trajectory_start_ = 0;
    std::deque<TrajectoryPoint> trajectory_points_;
    double trajectory_intercept_samples_ = 0.0;
    double trajectory_slope_samples_per_frame_ = 0.0;
    bool trajectory_seeded_restart_ = false;
    bool input_ended_ = false;
    TimingTrackerStats stats_{};
    TimingDiagnostics* timing_diagnostics_ = nullptr;
};

std::size_t discard_bytes(ReplayInput& input, std::size_t byte_count) {
    std::vector<std::int8_t> buffer(std::min<std::size_t>(byte_count, 1U << 16U));
    std::size_t discarded = 0;
    while (discarded < byte_count) {
        const auto wanted = std::min(buffer.size(), byte_count - discarded);
        const auto count = input.read(std::span<std::int8_t>(buffer.data(), wanted));
        discarded += count;
        if (count != wanted) {
            break;
        }
    }
    return discarded;
}

Normalization parse_normalization(const std::string& text) {
    if (text == "system-info") {
        return Normalization::system_info;
    }
    if (text == "qam64") {
        return Normalization::qam64;
    }
    if (text == "qam64-amplitude" || text == "qam64-amp") {
        return Normalization::qam64_amplitude;
    }
    if (text == "none") {
        return Normalization::none;
    }
    throw std::invalid_argument(
        "normalization must be system-info, qam64, qam64-amplitude, or none");
}

Equalizer parse_equalizer(const std::string& text) {
    if (text == "flat") {
        return Equalizer::flat;
    }
    if (text == "pn") {
        return Equalizer::pn;
    }
    throw std::invalid_argument("equalizer must be flat or pn");
}

PnEstimator parse_pn_estimator(const std::string& text) {
    if (text == "compact") {
        return PnEstimator::compact;
    }
    if (text == "wideband") {
        return PnEstimator::wideband;
    }
    throw std::invalid_argument("PN estimator must be compact or wideband");
}

dtmb::core::Pn945WidebandScaleEstimator parse_pn_wideband_scale_estimator(
    const std::string& text) {
    if (text == "dominant" || text == "dominant-tap") {
        return dtmb::core::Pn945WidebandScaleEstimator::dominant_tap;
    }
    if (text == "least-squares" || text == "ls") {
        return dtmb::core::Pn945WidebandScaleEstimator::least_squares_template;
    }
    if (text == "masked-frame-taps" || text == "frame-taps" || text == "masked-frame") {
        return dtmb::core::Pn945WidebandScaleEstimator::masked_frame_taps;
    }
    throw std::invalid_argument(
        "PN wideband scale estimator must be dominant, least-squares, or masked-frame-taps");
}

const char* pn_wideband_scale_estimator_name(
    dtmb::core::Pn945WidebandScaleEstimator estimator) noexcept {
    switch (estimator) {
    case dtmb::core::Pn945WidebandScaleEstimator::dominant_tap:
        return "dominant";
    case dtmb::core::Pn945WidebandScaleEstimator::least_squares_template:
        return "least-squares";
    case dtmb::core::Pn945WidebandScaleEstimator::masked_frame_taps:
        return "masked-frame-taps";
    }
    return "unknown";
}

dtmb::core::Pn945HeaderObservation parse_pn_wideband_header_observation(
    const std::string& text) {
    if (text == "core" || text == "core-only") {
        return dtmb::core::Pn945HeaderObservation::core_only;
    }
    if (text == "core-postfix" || text == "postfix-average") {
        return dtmb::core::Pn945HeaderObservation::core_postfix_average;
    }
    if (text == "core-cyclic-safe" || text == "cyclic-safe-average") {
        return dtmb::core::Pn945HeaderObservation::core_cyclic_safe_average;
    }
    throw std::invalid_argument(
        "PN wideband header observation must be core, core-postfix, or core-cyclic-safe");
}

const char* pn_wideband_header_observation_name(
    dtmb::core::Pn945HeaderObservation observation) noexcept {
    switch (observation) {
    case dtmb::core::Pn945HeaderObservation::core_only:
        return "core";
    case dtmb::core::Pn945HeaderObservation::core_postfix_average:
        return "core-postfix";
    case dtmb::core::Pn945HeaderObservation::core_cyclic_safe_average:
        return "core-cyclic-safe";
    }
    return "unknown";
}

PnMmse parse_pn_mmse(const std::string& text) {
    if (text == "off" || text == "none") {
        return {};
    }
    if (text == "auto") {
        return PnMmse{true, -1.0F};
    }
    const auto value = parse_float(text, "PN MMSE noise variance");
    if (value < 0.0F) {
        throw std::invalid_argument("PN MMSE noise variance must be non-negative");
    }
    return PnMmse{false, value};
}

bool parse_pn_wideband_body_channel_midpoint(const std::string& text) {
    if (text == "current") {
        return false;
    }
    if (text == "midpoint") {
        return true;
    }
    throw std::invalid_argument(
        "PN wideband body channel must be current or midpoint");
}

TimingLocalSearchScopedMinImprovementRule
parse_timing_local_search_scoped_min_improvement_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument(
            "invalid timing local-search scoped min-improvement rule: " + text);
    }
    TimingLocalSearchScopedMinImprovementRule rule;
    rule.first_frame = parse_size(parts[0], "timing local-search scoped first frame");
    rule.last_frame = parse_size(parts[1], "timing local-search scoped last frame");
    rule.min_improvement =
        parse_float(parts[2], "timing local-search scoped minimum improvement");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "timing local-search scoped end is before start: " + text);
    }
    if (!std::isfinite(rule.min_improvement) || rule.min_improvement < 0.0F) {
        throw std::invalid_argument(
            "timing local-search scoped minimum improvement must be non-negative: "
            + text);
    }
    return rule;
}

TimingLocalSearchTransientRange
parse_timing_local_search_transient_range(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 2) {
        throw std::invalid_argument(
            "invalid timing local-search transient range: " + text);
    }
    TimingLocalSearchTransientRange range;
    range.first_frame = parse_size(parts[0], "timing local-search transient first frame");
    range.last_frame = parse_size(parts[1], "timing local-search transient last frame");
    if (range.last_frame < range.first_frame) {
        throw std::invalid_argument(
            "timing local-search transient range end is before start: " + text);
    }
    return range;
}

SourceCarrierResidualRule parse_source_carrier_residual_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument(
            "invalid source carrier residual rule: " + text);
    }
    SourceCarrierResidualRule rule;
    rule.carrier = parse_size(parts[0], "source carrier residual carrier");
    rule.first_frame = parse_size(parts[1], "source carrier residual first frame");
    rule.last_frame = parse_size(parts[2], "source carrier residual last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier residual end is before start: " + text);
    }
    return rule;
}

SourceCarrierChannelRule parse_source_carrier_channel_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument(
            "invalid source carrier channel rule: " + text);
    }
    SourceCarrierChannelRule rule;
    rule.carrier = parse_size(parts[0], "source carrier channel carrier");
    rule.first_frame = parse_size(parts[1], "source carrier channel first frame");
    rule.last_frame = parse_size(parts[2], "source carrier channel last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier channel end is before start: " + text);
    }
    return rule;
}

std::vector<float> system_info_reference(std::size_t index) {
    if (index < 1 || index > 64) {
        throw std::invalid_argument("system information index must be 1..64");
    }
    const auto bits = kOddSystemInfoVectors[(index - 1) / 2];
    const auto complement = (index % 2) == 0;
    std::vector<float> reference(dtmb::core::kC3780SystemInfoSymbols * 2);
    for (std::size_t symbol = 0; symbol < dtmb::core::kC3780SystemInfoSymbols; ++symbol) {
        auto bit = symbol < 4 ? '1' : bits[symbol - 4];
        if (symbol >= 4 && complement) {
            bit = bit == '0' ? '1' : '0';
        }
        const auto chip = bit == '0' ? 1.0F : -1.0F;
        reference[symbol * 2] = chip;
        reference[symbol * 2 + 1] = chip;
    }
    return reference;
}

ComplexGainEstimate estimate_system_info_gain(
    std::span<const float> logical_symbols,
    std::span<const float> reference) {
    using Complex = std::complex<float>;
    auto numerator = Complex{0.0F, 0.0F};
    double denominator = 0.0;
    double observed_power = 0.0;
    for (std::size_t symbol = 0; symbol < dtmb::core::kC3780SystemInfoSymbols; ++symbol) {
        const auto observed = Complex{
            logical_symbols[symbol * 2],
            logical_symbols[symbol * 2 + 1],
        };
        const auto expected = Complex{
            reference[symbol * 2],
            reference[symbol * 2 + 1],
        };
        numerator += std::conj(expected) * observed;
        denominator += std::norm(expected);
        observed_power += std::norm(observed);
    }
    if (denominator <= 1.0e-12) {
        throw std::runtime_error("system-information reference has zero power");
    }
    const auto gain = numerator / static_cast<float>(denominator);
    const auto coherence_denominator = std::sqrt(denominator * observed_power);
    return ComplexGainEstimate{
        gain,
        coherence_denominator > 1.0e-12
            ? static_cast<float>(std::abs(numerator) / coherence_denominator)
            : 0.0F,
        std::abs(gain) > 1.0e-12F,
    };
}

float system_info_match_metric(
    std::span<const float> logical_symbols,
    std::span<const float> reference) {
    using Complex = std::complex<float>;
    auto numerator = Complex{0.0F, 0.0F};
    double reference_power = 0.0;
    double observed_power = 0.0;
    for (std::size_t symbol = 0; symbol < dtmb::core::kC3780SystemInfoSymbols; ++symbol) {
        const auto observed = Complex{
            logical_symbols[symbol * 2],
            logical_symbols[symbol * 2 + 1],
        };
        const auto expected = Complex{
            reference[symbol * 2],
            reference[symbol * 2 + 1],
        };
        numerator += std::conj(expected) * observed;
        reference_power += std::norm(expected);
        observed_power += std::norm(observed);
    }
    const auto denominator = std::sqrt(reference_power * observed_power);
    if (denominator <= 1.0e-12) {
        return 0.0F;
    }
    const auto correlation = numerator / static_cast<float>(denominator);
    // The PN equalizer leaves a capture-dependent common phase on the sparse
    // system-information carriers.  Fit that one nuisance phase instead of
    // penalizing otherwise correct references whose correlation is outside a
    // fixed +/-45 degree sector.  The common C=3780 prefix remains part of
    // every reference, so polarity/complement hypotheses are still distinct.
    return std::abs(correlation);
}

struct SystemInfoAutoSelection {
    std::size_t index = 0;
    std::size_t runner_up_index = 0;
    std::size_t observations = 0;
    float metric = 0.0F;
    float runner_up_metric = 0.0F;
    float margin = 0.0F;
    bool complete = false;
    bool locked = false;
};

class SystemInfoAutoSelector {
public:
    SystemInfoAutoSelector(
        std::size_t observation_frames,
        float min_metric,
        float min_margin)
        : observation_frames_(observation_frames),
          min_metric_(min_metric),
          min_margin_(min_margin) {
        for (std::size_t index = kFirstCandidate; index <= kLastCandidate; ++index) {
            references_[index - 1] = system_info_reference(index);
        }
    }

    void observe(std::span<const float> logical_symbols) {
        if (selection_.complete) {
            return;
        }
        for (std::size_t index = kFirstCandidate; index <= kLastCandidate; ++index) {
            metric_sums_[index - 1] += system_info_match_metric(
                logical_symbols,
                references_[index - 1]);
        }
        ++selection_.observations;
        if (selection_.observations >= observation_frames_) {
            finalize();
        }
    }

    [[nodiscard]] const SystemInfoAutoSelection& selection() const noexcept {
        return selection_;
    }

private:
    static constexpr std::size_t kFirstCandidate = 3;
    static constexpr std::size_t kLastCandidate = 24;

    void finalize() {
        auto best_index = std::size_t{0};
        auto second_index = std::size_t{0};
        auto best_metric = -std::numeric_limits<double>::infinity();
        auto second_metric = -std::numeric_limits<double>::infinity();
        for (std::size_t index = kFirstCandidate; index <= kLastCandidate; ++index) {
            const auto mean = metric_sums_[index - 1]
                / static_cast<double>(selection_.observations);
            if (mean > best_metric) {
                second_metric = best_metric;
                second_index = best_index;
                best_metric = mean;
                best_index = index;
            } else if (mean > second_metric) {
                second_metric = mean;
                second_index = index;
            }
        }
        selection_.index = best_index;
        selection_.runner_up_index = second_index;
        selection_.metric = static_cast<float>(best_metric);
        selection_.runner_up_metric = static_cast<float>(second_metric);
        selection_.margin = static_cast<float>(best_metric - second_metric);
        selection_.complete = true;
        selection_.locked = best_index != 0
            && best_metric >= static_cast<double>(min_metric_)
            && best_metric - second_metric >= static_cast<double>(min_margin_);
    }

    std::size_t observation_frames_ = 0;
    float min_metric_ = 0.0F;
    float min_margin_ = 0.0F;
    std::array<std::vector<float>, 64> references_{};
    std::array<double, 64> metric_sums_{};
    SystemInfoAutoSelection selection_{};
};

void normalize_data_from_system_info(
    std::span<const float> logical_symbols,
    std::span<const float> reference,
    std::span<float> output_data) {
    using Complex = std::complex<float>;
    const auto estimate = estimate_system_info_gain(logical_symbols, reference);
    const auto gain = estimate.gain;
    if (std::abs(gain) <= 1.0e-12F) {
        throw std::runtime_error("system-information gain estimate is zero");
    }
    for (std::size_t symbol = 0; symbol < dtmb::core::kC3780DataSymbols; ++symbol) {
        const auto logical_offset = (dtmb::core::kC3780SystemInfoSymbols + symbol) * 2;
        const auto normalized = Complex{
            logical_symbols[logical_offset],
            logical_symbols[logical_offset + 1],
        } / gain;
        output_data[symbol * 2] = normalized.real();
        output_data[symbol * 2 + 1] = normalized.imag();
    }
}

ComplexGainEstimate estimate_applied_gain(
    std::span<const float> raw_symbols,
    std::span<const float> normalized_symbols) {
    using Complex = std::complex<float>;
    if (raw_symbols.size() != normalized_symbols.size()
        || (raw_symbols.size() % 2U) != 0U) {
        throw std::invalid_argument("gain diagnostic CF32 spans must have equal size");
    }
    auto numerator = Complex{0.0F, 0.0F};
    double denominator = 0.0;
    for (std::size_t symbol = 0; symbol < raw_symbols.size() / 2U; ++symbol) {
        const auto raw = Complex{
            raw_symbols[symbol * 2U],
            raw_symbols[symbol * 2U + 1U],
        };
        const auto normalized = Complex{
            normalized_symbols[symbol * 2U],
            normalized_symbols[symbol * 2U + 1U],
        };
        numerator += std::conj(normalized) * raw;
        denominator += std::norm(normalized);
    }
    if (denominator <= 1.0e-12) {
        return {};
    }
    const auto gain = numerator / static_cast<float>(denominator);
    return ComplexGainEstimate{gain, 1.0F, std::abs(gain) > 1.0e-12F};
}

void update_frame_normalization_stats(
    FrameWork& frame,
    std::span<const float> raw_data,
    std::span<const float> system_info_reference) {
    frame.normalization_stats = {};
    frame.normalization_stats.system_info_gain = estimate_system_info_gain(
        frame.logical_cf32,
        system_info_reference);
    frame.normalization_stats.qam_gain = estimate_applied_gain(
        raw_data,
        frame.data_cf32);

    double residual_ratio_sum = 0.0;
    for (std::size_t carrier = 0; carrier < dtmb::core::kC3780DataSymbols; ++carrier) {
        const auto point = qam64_residual_point(
            frame.data_cf32[carrier * 2U],
            frame.data_cf32[carrier * 2U + 1U]);
        residual_ratio_sum += point.decision_abs > 0.0F
            ? point.residual_abs / point.decision_abs
            : 0.0F;
    }
    frame.normalization_stats.qam_residual_ratio = static_cast<float>(
        residual_ratio_sum / static_cast<double>(dtmb::core::kC3780DataSymbols));

    if (!frame.normalization_stats.qam_gain.valid
        || !frame.normalization_stats.system_info_gain.valid) {
        return;
    }
    const auto qam_phase = std::arg(frame.normalization_stats.qam_gain.gain);
    const auto system_info_phase = std::arg(
        frame.normalization_stats.system_info_gain.gain);
    const auto difference = qam_phase - system_info_phase;
    constexpr auto half_pi = std::numbers::pi_v<float> / 2.0F;
    const auto quadrant = static_cast<int>(std::llround(difference / half_pi));
    frame.normalization_stats.qam_minus_system_info_phase_rad = difference;
    frame.normalization_stats.qam_minus_system_info_phase_mod_pi_over_2_rad =
        std::remainder(difference, half_pi);
    frame.normalization_stats.qam_minus_system_info_quadrant =
        ((quadrant % 4) + 4) % 4;
}

void convert_ci8_to_cf32(
    std::span<const std::int8_t> input,
    std::span<float> output,
    std::size_t sample_start,
    float frequency_shift_hz,
    std::complex<float> dc_offset = {});

void extract_data_symbols(
    FrameWork& frame,
    Normalization normalization,
    std::span<const float> system_info_reference,
    const DataDecisionDirectedOptions& dd_options,
    DataDecisionDirectedStats& dd_stats);

void process_frame(
    FrameWork& frame,
    Normalization normalization,
    std::span<const float> system_info_reference,
    float frequency_shift_hz,
    const DataDecisionDirectedOptions& dd_options,
    DataDecisionDirectedStats& dd_stats) {
    convert_ci8_to_cf32(
        frame.body_ci8,
        frame.body_cf32,
        frame.sample_start + dtmb::core::kPn945HeaderSymbols,
        frequency_shift_hz);
    dtmb::core::c3780_extract_frame_symbols_cf32(frame.body_cf32, frame.logical_cf32);
    extract_data_symbols(
        frame,
        normalization,
        system_info_reference,
        dd_options,
        dd_stats);
}

void convert_ci8_to_cf32(
    std::span<const std::int8_t> input,
    std::span<float> output,
    std::size_t sample_start,
    float frequency_shift_hz,
    std::complex<float> dc_offset) {
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
            static_cast<float>(input[sample * 2]) - dc_offset.real(),
            static_cast<float>(input[sample * 2 + 1]) - dc_offset.imag(),
        } / 128.0;
        const auto shifted = value * rotation;
        output[sample * 2] = shifted.real();
        output[sample * 2 + 1] = shifted.imag();
        rotation *= rotation_step;
    }
}

dtmb::core::Pn945EqualizeResult process_pn_frame(
    FrameWork& frame,
    FrameWork& next_frame,
    Normalization normalization,
    std::span<const float> system_info_reference,
    std::size_t pn_channel_taps,
    float frequency_shift_hz,
    float noise_variance,
    const DataDecisionDirectedOptions& dd_options,
    DataDecisionDirectedStats& dd_stats) {
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
    extract_data_symbols(
        frame,
        normalization,
        system_info_reference,
        dd_options,
        dd_stats);
    return result;
}

dtmb::core::Pn945EqualizeResult process_wideband_pn_frame(
    FrameWork& frame,
    FrameWork& next_frame,
    Normalization normalization,
    std::span<const float> system_info_reference,
    const dtmb::core::Pn945WidebandChannelModel& model,
    std::size_t model_frame_index,
    float frequency_shift_hz,
    float noise_variance,
    std::complex<float> dc_offset,
    int response_window_offset_adjust,
    bool interpolate_body_channel,
    bool csi_demap,
    const DataDecisionDirectedOptions& dd_options,
    DataDecisionDirectedStats& dd_stats) {
    convert_ci8_to_cf32(
        frame.body_ci8,
        frame.body_cf32,
        frame.sample_start + dtmb::core::kPn945HeaderSymbols,
        frequency_shift_hz,
        dc_offset);
    const auto result = dtmb::core::pn945_equalize_c3780_frame_wideband_cached_cf32(
        frame.body_cf32,
        next_frame.header_cf32,
        frame.spectrum_cf32,
        model,
        model_frame_index,
        dtmb::core::Pn945EqualizeOptions{
            model.template_taps.size() / 2,
            1.0e-3F,
            1.0e-6F,
            noise_variance,
            response_window_offset_adjust,
            interpolate_body_channel,
            csi_demap,
            0.25F,
            csi_demap
                ? std::span<float>(frame.channel_power_spectrum_cf32)
                : std::span<float>{},
        });
    dtmb::core::c3780_deinterleave_spectrum_cf32(
        frame.spectrum_cf32,
        frame.logical_cf32);
    extract_data_symbols(
        frame,
        normalization,
        system_info_reference,
        dd_options,
        dd_stats);
    if (csi_demap) {
        prepare_csi_weights(frame);
    }
    return result;
}

dtmb::core::Pn945WidebandChannelModel build_wideband_model(
    std::deque<FrameWork>& frames,
    float frequency_shift_hz,
    std::complex<float> dc_offset,
    dtmb::core::Pn945WidebandScaleEstimator scale_estimator,
    dtmb::core::Pn945HeaderObservation header_observation,
    std::size_t max_span_symbols) {
    std::vector<float> headers;
    headers.reserve(frames.size() * dtmb::core::kPn945HeaderSymbols * 2);
    for (auto& frame : frames) {
        convert_ci8_to_cf32(
            frame.header_ci8,
            frame.header_cf32,
            frame.sample_start,
            frequency_shift_hz,
            dc_offset);
        headers.insert(headers.end(), frame.header_cf32.begin(), frame.header_cf32.end());
    }
    auto options = dtmb::core::Pn945WidebandModelOptions{};
    options.scale_estimator = scale_estimator;
    options.header_observation = header_observation;
    options.max_span_symbols = max_span_symbols;
    return dtmb::core::build_pn945_wideband_channel_model_cf32(headers, options);
}

std::complex<float> estimate_block_dc_offset(const std::deque<FrameWork>& frames) {
    double sum_i = 0.0;
    double sum_q = 0.0;
    std::size_t samples = 0;
    for (const auto& frame : frames) {
        for (const auto* values : {&frame.header_ci8, &frame.body_ci8}) {
            for (std::size_t sample = 0; sample < values->size() / 2; ++sample) {
                sum_i += (*values)[sample * 2];
                sum_q += (*values)[sample * 2 + 1];
            }
            samples += values->size() / 2;
        }
    }
    if (samples == 0) {
        return {};
    }
    return {
        static_cast<float>(sum_i / static_cast<double>(samples)),
        static_cast<float>(sum_q / static_cast<double>(samples)),
    };
}

bool read_frame(
    ReplayInput& input,
    FrameWork& frame,
    std::size_t& trailing_ci8_bytes,
    std::size_t& next_sample_start) {
    frame.sample_start = next_sample_start;
    const auto header_bytes = input.read(frame.header_ci8);
    if (header_bytes == 0) {
        return false;
    }
    if (header_bytes != frame.header_ci8.size()) {
        trailing_ci8_bytes = header_bytes;
        return false;
    }
    const auto body_bytes = input.read(frame.body_ci8);
    if (body_bytes != frame.body_ci8.size()) {
        trailing_ci8_bytes = frame.header_ci8.size() + body_bytes;
        return false;
    }
    next_sample_start += dtmb::core::kPn945HeaderSymbols
        + dtmb::core::kC3780FrameBodySymbols;
    return true;
}

void write_all(std::ostream& output, std::span<const float> values) {
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write C=3780 data-symbol stream");
    }
}

void prepare_csi_weights(FrameWork& frame) {
    dtmb::core::c3780_deinterleave_spectrum_cf32(
        frame.channel_power_spectrum_cf32,
        frame.channel_power_logical_cf32);
    const auto power = std::span<const float>(
        frame.channel_power_logical_cf32.data()
            + dtmb::core::kC3780SystemInfoSymbols * 2,
        dtmb::core::kC3780DataSymbols * 2);
    std::vector<float> positive;
    positive.reserve(dtmb::core::kC3780DataSymbols);
    for (std::size_t carrier = 0; carrier < dtmb::core::kC3780DataSymbols; ++carrier) {
        const auto value = power[carrier * 2];
        if (std::isfinite(value) && value > 0.0F) {
            positive.push_back(value);
        }
    }
    auto scale = 1.0F;
    if (!positive.empty()) {
        const auto middle = positive.begin()
            + static_cast<std::ptrdiff_t>(positive.size() / 2);
        std::nth_element(positive.begin(), middle, positive.end());
        scale = std::max(*middle, 1.0e-12F);
    }
    for (std::size_t carrier = 0; carrier < dtmb::core::kC3780DataSymbols; ++carrier) {
        const auto value = power[carrier * 2];
        frame.csi_weights[carrier] = std::isfinite(value) && value > 0.0F
            ? std::clamp(value / scale, 0.0F, 4.0F)
            : 0.0F;
    }
}

std::array<std::uint8_t, 3> qam64_axis_bits(std::size_t level_index) {
    const auto gray = level_index ^ (level_index >> 1U);
    return {
        static_cast<std::uint8_t>(gray & 0x01U),
        static_cast<std::uint8_t>((gray >> 1U) & 0x01U),
        static_cast<std::uint8_t>((gray >> 2U) & 0x01U),
    };
}

void extract_data_symbols(
    FrameWork& frame,
    Normalization normalization,
    std::span<const float> system_info_reference,
    const DataDecisionDirectedOptions& dd_options,
    DataDecisionDirectedStats& dd_stats) {
    const auto data = std::span<const float>(
        frame.logical_cf32.data() + dtmb::core::kC3780SystemInfoSymbols * 2,
        dtmb::core::kC3780DataSymbols * 2);
    if (normalization == Normalization::system_info) {
        normalize_data_from_system_info(
            frame.logical_cf32,
            system_info_reference,
            frame.data_cf32);
        update_frame_normalization_stats(frame, data, system_info_reference);
        return;
    }
    if (normalization == Normalization::none) {
        std::copy(data.begin(), data.end(), frame.data_cf32.begin());
        update_frame_normalization_stats(frame, data, system_info_reference);
        return;
    }
    if (normalization == Normalization::qam64_amplitude) {
        dtmb::core::qam64_normalize_amplitude_cf32(data, frame.data_cf32);
        update_frame_normalization_stats(frame, data, system_info_reference);
        return;
    }
    dtmb::core::qam64_normalize_cf32(data, frame.data_cf32);
    update_frame_normalization_stats(frame, data, system_info_reference);
    if (!dd_options.enabled) {
        return;
    }

    constexpr auto qam64_rms = float{6.48074069840786F};
    std::vector<std::size_t> reliable_indices;
    std::vector<std::complex<float>> reliable_channels;
    reliable_indices.reserve(dtmb::core::kC3780DataSymbols);
    reliable_channels.reserve(dtmb::core::kC3780DataSymbols);
    std::array<std::size_t, 6> hard_one_counts{};
    std::size_t real_inner_count = 0;
    std::size_t imag_inner_count = 0;
    for (std::size_t carrier = 0; carrier < dtmb::core::kC3780DataSymbols; ++carrier) {
        const auto point = qam64_residual_point(
            frame.data_cf32[carrier * 2],
            frame.data_cf32[carrier * 2 + 1]);
        if (std::abs(point.decision_real) == 1.0F) {
            ++real_inner_count;
        }
        if (std::abs(point.decision_imag) == 1.0F) {
            ++imag_inner_count;
        }
        const auto real_bits = qam64_axis_bits(point.decision_real_index);
        const auto imag_bits = qam64_axis_bits(point.decision_imag_index);
        for (std::size_t bit = 0; bit < real_bits.size(); ++bit) {
            hard_one_counts[bit] += real_bits[bit];
            hard_one_counts[bit + real_bits.size()] += imag_bits[bit];
        }
        if ((point.residual_abs / qam64_rms) > dd_options.max_relative_error
            || point.decision_abs <= 0.0F) {
            continue;
        }
        const auto observed = std::complex<float>{
            frame.data_cf32[carrier * 2],
            frame.data_cf32[carrier * 2 + 1],
        };
        const auto decision = std::complex<float>{
            point.decision_real,
            point.decision_imag,
        };
        reliable_indices.push_back(carrier);
        reliable_channels.push_back(observed / decision);
    }
    if (dd_options.max_axis_inner_fraction > 0.0F) {
        const auto denominator =
            static_cast<float>(dtmb::core::kC3780DataSymbols);
        const auto inner_fraction = std::max(
            static_cast<float>(real_inner_count) / denominator,
            static_cast<float>(imag_inner_count) / denominator);
        if (inner_fraction > dd_options.max_axis_inner_fraction) {
            dd_stats.observe(
                reliable_indices.size(),
                false,
                DataDecisionDirectedRejectReason::axis_inner_fraction);
            return;
        }
    }
    if (dd_options.max_hard_bit_bias > 0.0F) {
        float max_bias = 0.0F;
        const auto denominator =
            static_cast<float>(dtmb::core::kC3780DataSymbols);
        for (const auto count : hard_one_counts) {
            const auto balance = static_cast<float>(count) / denominator;
            max_bias = std::max(max_bias, std::abs(balance - 0.5F));
        }
        if (max_bias > dd_options.max_hard_bit_bias) {
            dd_stats.observe(
                reliable_indices.size(),
                false,
                DataDecisionDirectedRejectReason::hard_bit_bias);
            return;
        }
    }
    if (reliable_indices.size() < dd_options.min_reliable_carriers) {
        dd_stats.observe(
            reliable_indices.size(),
            false,
            DataDecisionDirectedRejectReason::reliable_carriers);
        return;
    }

    std::size_t next = 0;
    for (std::size_t carrier = 0; carrier < dtmb::core::kC3780DataSymbols; ++carrier) {
        while (next + 1 < reliable_indices.size()
               && reliable_indices[next + 1] < carrier) {
            ++next;
        }
        auto channel = reliable_channels[next];
        if (carrier <= reliable_indices.front()) {
            channel = reliable_channels.front();
        } else if (carrier >= reliable_indices.back()) {
            channel = reliable_channels.back();
        } else if (next + 1 < reliable_indices.size()) {
            const auto left_index = reliable_indices[next];
            const auto right_index = reliable_indices[next + 1];
            const auto span = static_cast<float>(right_index - left_index);
            const auto fraction = span > 0.0F
                ? static_cast<float>(carrier - left_index) / span
                : 0.0F;
            channel = reliable_channels[next] * (1.0F - fraction)
                + reliable_channels[next + 1] * fraction;
        }
        if (std::abs(channel) <= 1.0e-6F) {
            continue;
        }
        const auto observed = std::complex<float>{
            frame.data_cf32[carrier * 2],
            frame.data_cf32[carrier * 2 + 1],
        };
        const auto corrected = observed / channel;
        frame.data_cf32[carrier * 2] = corrected.real();
        frame.data_cf32[carrier * 2 + 1] = corrected.imag();
    }
    dd_stats.observe(reliable_indices.size(), true);
}

std::size_t nearest_qam64_level_index(float value) {
    std::size_t best = 0;
    auto best_distance = std::abs(value - kQam64Levels[best]);
    for (std::size_t index = 1; index < kQam64Levels.size(); ++index) {
        const auto distance = std::abs(value - kQam64Levels[index]);
        if (distance < best_distance) {
            best = index;
            best_distance = distance;
        }
    }
    return best;
}

float nearest_qam64_level(float value) {
    return kQam64Levels[nearest_qam64_level_index(value)];
}

Qam64ResidualPoint qam64_residual_point(float observed_real, float observed_imag) {
    Qam64ResidualPoint point;
    point.observed_real = observed_real;
    point.observed_imag = observed_imag;
    point.decision_real_index = nearest_qam64_level_index(observed_real);
    point.decision_imag_index = nearest_qam64_level_index(observed_imag);
    point.decision_real = kQam64Levels[point.decision_real_index];
    point.decision_imag = kQam64Levels[point.decision_imag_index];
    point.residual_real = observed_real - point.decision_real;
    point.residual_imag = observed_imag - point.decision_imag;
    point.residual_abs = std::hypot(point.residual_real, point.residual_imag);
    point.decision_abs = std::hypot(point.decision_real, point.decision_imag);
    point.observed_abs = std::hypot(observed_real, observed_imag);
    return point;
}

struct FrameResidualStats {
    std::size_t count = 0;
    double residual_abs_sum = 0.0;
    double residual_abs_sq_sum = 0.0;
    float residual_abs_max = 0.0F;
    double residual_ratio_sum = 0.0;
    double observed_abs_sum = 0.0;

    void observe(const Qam64ResidualPoint& point) {
        residual_abs_sum += point.residual_abs;
        residual_abs_sq_sum +=
            static_cast<double>(point.residual_abs) * point.residual_abs;
        residual_abs_max = std::max(residual_abs_max, point.residual_abs);
        residual_ratio_sum += point.decision_abs > 0.0F
            ? point.residual_abs / point.decision_abs
            : 0.0F;
        observed_abs_sum += point.observed_abs;
        ++count;
    }

    [[nodiscard]] double mean_residual_abs() const noexcept {
        return count == 0 ? 0.0 : residual_abs_sum / static_cast<double>(count);
    }

    [[nodiscard]] double rms_residual_abs() const noexcept {
        return count == 0
            ? 0.0
            : std::sqrt(residual_abs_sq_sum / static_cast<double>(count));
    }

    [[nodiscard]] double mean_residual_ratio() const noexcept {
        return count == 0 ? 0.0 : residual_ratio_sum / static_cast<double>(count);
    }

    [[nodiscard]] double mean_observed_abs() const noexcept {
        return count == 0 ? 0.0 : observed_abs_sum / static_cast<double>(count);
    }
};

void write_frame_residual_stats(std::ostream& output, const FrameResidualStats& stats) {
    output << stats.mean_residual_abs() << ','
           << stats.rms_residual_abs() << ','
           << stats.residual_abs_max << ','
           << stats.mean_residual_ratio() << ','
           << stats.mean_observed_abs();
}

bool is_c3780_system_info_position(std::size_t inserted_logical) {
    return std::find(
        kC3780SystemInfoPositions.begin(),
        kC3780SystemInfoPositions.end(),
        inserted_logical)
        != kC3780SystemInfoPositions.end();
}

std::size_t data_carrier_to_inserted_logical(std::size_t carrier) {
    std::size_t data_index = 0;
    for (std::size_t inserted_logical = 0;
         inserted_logical < dtmb::core::kC3780FrameBodySymbols;
         ++inserted_logical) {
        if (is_c3780_system_info_position(inserted_logical)) {
            continue;
        }
        if (data_index == carrier) {
            return inserted_logical;
        }
        ++data_index;
    }
    throw std::logic_error("invalid C3780 data carrier index");
}

std::size_t c3780_logical_to_physical(std::size_t inserted_logical) {
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
    throw std::logic_error("invalid C3780 logical carrier index");
}

std::size_t data_carrier_to_physical_bin(std::size_t carrier) {
    static const auto physical_bins = [] {
        std::array<std::size_t, dtmb::core::kC3780DataSymbols> bins{};
        std::size_t data_carrier = 0;
        for (std::size_t inserted_logical = 0;
             inserted_logical < dtmb::core::kC3780FrameBodySymbols;
             ++inserted_logical) {
            if (is_c3780_system_info_position(inserted_logical)) {
                continue;
            }
            bins[data_carrier++] = c3780_logical_to_physical(inserted_logical);
        }
        if (data_carrier != bins.size()) {
            throw std::logic_error("invalid C3780 data-carrier mapping size");
        }
        return bins;
    }();
    if (carrier >= physical_bins.size()) {
        throw std::logic_error("invalid C3780 data carrier index");
    }
    return physical_bins[carrier];
}

int wideband_response_window_offset(
    const dtmb::core::Pn945WidebandChannelModel& model,
    int response_window_offset_adjust) {
    auto signed_rotation = static_cast<int>(model.rotation_symbols);
    if (signed_rotation > static_cast<int>(kPn945CoreSymbols / 2U)) {
        signed_rotation -= static_cast<int>(kPn945CoreSymbols);
    }
    return -signed_rotation + response_window_offset_adjust;
}

struct PnTapTransitionStats {
    float phase_rad = 0.0F;
    float coherence = 0.0F;
    float next_to_current_power = 0.0F;
    float sampled_carrier_shape_change_rms = 0.0F;
    float sampled_carrier_shape_change_max = 0.0F;
    std::array<float, 3> slot_sampled_carrier_shape_change_rms{};
    std::array<float, 3> slot_sampled_carrier_shape_change_max{};
};

PnTapTransitionStats pn_tap_transition_stats(
    const dtmb::core::Pn945WidebandChannelModel& model,
    std::size_t model_frame_index,
    float noise_variance) {
    const auto span = model.template_taps.size() / 2U;
    if (span == 0U) {
        throw std::runtime_error("PN tap transition diagnostics require model taps");
    }

    const auto masked_frame_taps = model.scale_estimator
        == dtmb::core::Pn945WidebandScaleEstimator::masked_frame_taps;
    const auto frame_tap = [&](std::size_t frame, std::size_t tap) {
        if (masked_frame_taps) {
            const auto offset = (frame * span + tap) * 2U;
            if (offset + 1U >= model.frame_template_taps.size()) {
                throw std::runtime_error(
                    "PN tap transition diagnostics are outside masked frame taps");
            }
            return std::complex<float>{
                model.frame_template_taps[offset],
                model.frame_template_taps[offset + 1U],
            };
        }
        const auto scale_offset = frame * 2U;
        if (scale_offset + 1U >= model.frame_response_scales.size()) {
            throw std::runtime_error(
                "PN tap transition diagnostics are outside response scales");
        }
        const auto scale = std::complex<float>{
            model.frame_response_scales[scale_offset],
            model.frame_response_scales[scale_offset + 1U],
        };
        return scale * std::complex<float>{
            model.template_taps[tap * 2U],
            model.template_taps[tap * 2U + 1U],
        };
    };

    std::complex<float> cross{};
    float current_power = 0.0F;
    float next_power = 0.0F;
    for (std::size_t tap = 0; tap < span; ++tap) {
        const auto current = frame_tap(model_frame_index, tap);
        const auto next = frame_tap(model_frame_index + 1U, tap);
        cross += next * std::conj(current);
        current_power += std::norm(current);
        next_power += std::norm(next);
    }
    const auto denominator = std::sqrt(current_power * next_power);
    const auto alignment = current_power > 0.0F
        ? cross / current_power
        : std::complex<float>{1.0F, 0.0F};

    constexpr auto carriers_per_slot = dtmb::core::kC3780DataSymbols / 3U;
    constexpr std::size_t samples_per_slot = 32U;
    constexpr auto sampled_carrier_count = samples_per_slot * 3U;
    static_assert(carriers_per_slot * 3U == dtmb::core::kC3780DataSymbols);
    struct CarrierSample {
        std::size_t slot = 0;
        std::complex<float> step{1.0F, 0.0F};
    };
    static const auto carrier_samples = [] {
        std::array<CarrierSample, sampled_carrier_count> samples{};
        std::size_t output = 0;
        for (std::size_t slot = 0; slot < 3U; ++slot) {
            for (std::size_t sample = 0; sample < samples_per_slot; ++sample) {
                const auto within_slot = static_cast<std::size_t>(std::llround(
                    static_cast<double>(sample) * (carriers_per_slot - 1U)
                    / static_cast<double>(samples_per_slot - 1U)));
                const auto carrier = slot * carriers_per_slot + within_slot;
                const auto physical_bin = data_carrier_to_physical_bin(carrier);
                const auto angle = -2.0F * std::numbers::pi_v<float>
                    * static_cast<float>(physical_bin)
                    / static_cast<float>(dtmb::core::kC3780FrameBodySymbols);
                samples[output++] = CarrierSample{
                    slot,
                    std::complex<float>{std::cos(angle), std::sin(angle)},
                };
            }
        }
        return samples;
    }();

    double carrier_shape_change_sq_sum = 0.0;
    std::array<double, 3> slot_shape_change_sq_sum{};
    auto carrier_shape_change_max = 0.0F;
    std::array<float, 3> slot_shape_change_max{};
    const auto regularization = std::max(noise_variance, 0.0F);
    for (const auto& sample : carrier_samples) {
        auto current = std::complex<float>{0.0F, 0.0F};
        auto next = std::complex<float>{0.0F, 0.0F};
        auto weight = std::complex<float>{1.0F, 0.0F};
        for (std::size_t tap = 0; tap < span; ++tap) {
            current += frame_tap(model_frame_index, tap) * weight;
            next += frame_tap(model_frame_index + 1U, tap) * weight;
            weight *= sample.step;
        }
        const auto scale = std::sqrt(std::norm(current) + regularization);
        const auto change = scale > 0.0F
            ? std::abs(next - alignment * current) / scale
            : 0.0F;
        carrier_shape_change_sq_sum += static_cast<double>(change) * change;
        slot_shape_change_sq_sum[sample.slot] +=
            static_cast<double>(change) * change;
        carrier_shape_change_max = std::max(carrier_shape_change_max, change);
        slot_shape_change_max[sample.slot] = std::max(
            slot_shape_change_max[sample.slot],
            change);
    }

    std::array<float, 3> slot_shape_change_rms{};
    for (std::size_t slot = 0; slot < slot_shape_change_rms.size(); ++slot) {
        slot_shape_change_rms[slot] = std::sqrt(
            slot_shape_change_sq_sum[slot]
            / static_cast<double>(samples_per_slot));
    }
    return PnTapTransitionStats{
        std::atan2(cross.imag(), cross.real()),
        denominator > 0.0F ? std::abs(cross) / denominator : 0.0F,
        current_power > 0.0F ? next_power / current_power : 0.0F,
        static_cast<float>(std::sqrt(
            carrier_shape_change_sq_sum
            / static_cast<double>(sampled_carrier_count))),
        carrier_shape_change_max,
        slot_shape_change_rms,
        slot_shape_change_max,
    };
}

class FrameResidualDiagnostics {
public:
    explicit FrameResidualDiagnostics(std::string path)
        : path_(std::move(path)) {
        if (path_.empty()) {
            return;
        }
        output_ = std::make_unique<std::ofstream>(path_, std::ios::binary);
        if (!*output_) {
            throw std::runtime_error(
                "failed to open frame residual diagnostics: " + path_);
        }
        auto& output = *output_;
        output
            << "source_frame,residual_abs_mean,residual_abs_rms,"
            << "residual_abs_max,residual_abs_over_decision_abs_mean,"
            << "observed_abs_mean";
        for (std::size_t slot = 0; slot < 3; ++slot) {
            output << ",slot" << slot << "_residual_abs_mean"
                   << ",slot" << slot << "_residual_abs_rms"
                   << ",slot" << slot << "_residual_abs_max"
                   << ",slot" << slot << "_residual_abs_over_decision_abs_mean"
                   << ",slot" << slot << "_observed_abs_mean";
        }
        output << '\n';
    }

    FrameResidualDiagnostics(const FrameResidualDiagnostics&) = delete;
    FrameResidualDiagnostics& operator=(const FrameResidualDiagnostics&) = delete;

    ~FrameResidualDiagnostics() {
        if (output_ && !closed_) {
            output_->flush();
        }
    }

    void observe(std::size_t source_frame, std::span<const float> data_symbols) {
        if (!output_) {
            return;
        }
        constexpr auto symbols_per_slot = dtmb::core::kC3780DataSymbols / 3U;
        static_assert(symbols_per_slot * 3U == dtmb::core::kC3780DataSymbols);
        FrameResidualStats all;
        std::array<FrameResidualStats, 3> slots;
        for (std::size_t carrier = 0; carrier < dtmb::core::kC3780DataSymbols; ++carrier) {
            const auto offset = carrier * 2;
            const auto point = qam64_residual_point(
                data_symbols[offset],
                data_symbols[offset + 1]);
            all.observe(point);
            slots[carrier / symbols_per_slot].observe(point);
        }
        auto& output = *output_;
        output << source_frame << ',';
        write_frame_residual_stats(output, all);
        for (const auto& slot : slots) {
            output << ',';
            write_frame_residual_stats(output, slot);
        }
        output << '\n';
        if (!output) {
            throw std::runtime_error(
                "failed to write frame residual diagnostics: " + path_);
        }
    }

    void close() {
        if (!output_ || closed_) {
            return;
        }
        output_->flush();
        if (!*output_) {
            throw std::runtime_error(
                "failed to close frame residual diagnostics: " + path_);
        }
        closed_ = true;
    }

private:
    std::string path_;
    std::unique_ptr<std::ofstream> output_;
    bool closed_ = false;
};

std::string_view normalization_name(Normalization normalization) noexcept {
    switch (normalization) {
    case Normalization::system_info:
        return "system-info";
    case Normalization::qam64:
        return "qam64";
    case Normalization::qam64_amplitude:
        return "qam64-amplitude";
    case Normalization::none:
        return "none";
    }
    return "unknown";
}

class FrameNormalizationDiagnostics {
public:
    FrameNormalizationDiagnostics(std::string path, Normalization normalization)
        : path_(std::move(path)), normalization_(normalization) {
        if (path_.empty()) {
            return;
        }
        output_ = std::make_unique<std::ofstream>(path_, std::ios::binary);
        if (!*output_) {
            throw std::runtime_error(
                "failed to open frame normalization diagnostics: " + path_);
        }
        *output_
            << "source_frame,normalization,qam_gain_valid,qam_gain_real,"
            << "qam_gain_imag,qam_gain_abs,qam_gain_phase_rad,"
            << "system_info_gain_valid,system_info_gain_real,"
            << "system_info_gain_imag,system_info_gain_abs,"
            << "system_info_gain_phase_rad,system_info_coherence,"
            << "qam_minus_system_info_phase_rad,"
            << "qam_minus_system_info_phase_mod_pi_over_2_rad,"
            << "qam_minus_system_info_quadrant,qam_residual_ratio,"
            << "pn_metadata_valid,pn_phase,next_pn_phase,model_pn_phase,"
            << "model_next_pn_phase,model_rotation_symbols\n";
    }

    FrameNormalizationDiagnostics(const FrameNormalizationDiagnostics&) = delete;
    FrameNormalizationDiagnostics& operator=(
        const FrameNormalizationDiagnostics&) = delete;

    ~FrameNormalizationDiagnostics() {
        if (output_ && !closed_) {
            output_->flush();
        }
    }

    void observe(
        std::size_t source_frame,
        const FrameNormalizationStats& stats,
        bool pn_metadata_valid,
        std::size_t pn_phase,
        std::size_t next_pn_phase,
        std::size_t model_pn_phase,
        std::size_t model_next_pn_phase,
        std::size_t model_rotation_symbols) {
        if (!output_) {
            return;
        }
        const auto& qam = stats.qam_gain;
        const auto& system_info = stats.system_info_gain;
        *output_
            << source_frame << ','
            << normalization_name(normalization_) << ','
            << (qam.valid ? 1 : 0) << ','
            << qam.gain.real() << ','
            << qam.gain.imag() << ','
            << std::abs(qam.gain) << ','
            << std::arg(qam.gain) << ','
            << (system_info.valid ? 1 : 0) << ','
            << system_info.gain.real() << ','
            << system_info.gain.imag() << ','
            << std::abs(system_info.gain) << ','
            << std::arg(system_info.gain) << ','
            << system_info.coherence << ','
            << stats.qam_minus_system_info_phase_rad << ','
            << stats.qam_minus_system_info_phase_mod_pi_over_2_rad << ','
            << stats.qam_minus_system_info_quadrant << ','
            << stats.qam_residual_ratio << ','
            << (pn_metadata_valid ? 1 : 0) << ','
            << pn_phase << ','
            << next_pn_phase << ','
            << model_pn_phase << ','
            << model_next_pn_phase << ','
            << model_rotation_symbols << '\n';
        if (!*output_) {
            throw std::runtime_error(
                "failed to write frame normalization diagnostics: " + path_);
        }
    }

    void close() {
        if (!output_ || closed_) {
            return;
        }
        output_->flush();
        if (!*output_) {
            throw std::runtime_error(
                "failed to close frame normalization diagnostics: " + path_);
        }
        closed_ = true;
    }

private:
    std::string path_;
    Normalization normalization_;
    std::unique_ptr<std::ofstream> output_;
    bool closed_ = false;
};

class SourceCarrierResidualDiagnostics {
public:
    SourceCarrierResidualDiagnostics(
        std::string path,
        std::vector<SourceCarrierResidualRule> rules)
        : path_(std::move(path)),
          rules_(std::move(rules)) {
        if (path_.empty()) {
            return;
        }
        output_ = std::make_unique<std::ofstream>(path_, std::ios::binary);
        if (!*output_) {
            throw std::runtime_error(
                "failed to open source carrier residual diagnostics: " + path_);
        }
        auto& output = *output_;
        output << "{\"schema\":\"dtmb.c3780_source_carrier_residuals.v1\"";
        output << ",\"rules\":[";
        for (std::size_t index = 0; index < rules_.size(); ++index) {
            if (index != 0) {
                output << ',';
            }
            output << "{\"carrier\":" << rules_[index].carrier
                   << ",\"first_frame\":" << rules_[index].first_frame
                   << ",\"last_frame\":" << rules_[index].last_frame << '}';
        }
        output << "],\"rows\":[";
    }

    SourceCarrierResidualDiagnostics(const SourceCarrierResidualDiagnostics&) = delete;
    SourceCarrierResidualDiagnostics& operator=(const SourceCarrierResidualDiagnostics&) = delete;

    ~SourceCarrierResidualDiagnostics() {
        if (output_ && !closed_) {
            *output_ << "]}";
        }
    }

    void observe(std::size_t source_frame, std::span<const float> data_symbols) {
        if (!output_ || rules_.empty()) {
            return;
        }
        auto& output = *output_;
        for (std::size_t rule_index = 0; rule_index < rules_.size(); ++rule_index) {
            const auto& rule = rules_[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto offset = rule.carrier * 2;
            const auto point = qam64_residual_point(
                data_symbols[offset],
                data_symbols[offset + 1]);
            if (wrote_row_) {
                output << ',';
            }
            wrote_row_ = true;
            output << "{\"rule_index\":" << rule_index
                   << ",\"source_frame\":" << source_frame
                   << ",\"carrier\":" << rule.carrier
                   << ",\"observed_real\":" << point.observed_real
                   << ",\"observed_imag\":" << point.observed_imag
                   << ",\"decision_real\":" << point.decision_real
                   << ",\"decision_imag\":" << point.decision_imag
                   << ",\"residual_real\":" << point.residual_real
                   << ",\"residual_imag\":" << point.residual_imag
                   << ",\"residual_abs\":" << point.residual_abs
                   << ",\"residual_abs_over_decision_abs\":"
                   << (point.decision_abs > 0.0F
                           ? point.residual_abs / point.decision_abs
                           : 0.0F)
                   << ",\"observed_abs\":" << point.observed_abs << '}';
        }
        if (!output) {
            throw std::runtime_error(
                "failed to write source carrier residual diagnostics");
        }
    }

    void close() {
        if (!output_ || closed_) {
            return;
        }
        *output_ << "]}";
        if (!*output_) {
            throw std::runtime_error(
                "failed to close source carrier residual diagnostics");
        }
        closed_ = true;
    }

private:
    std::string path_;
    std::vector<SourceCarrierResidualRule> rules_;
    std::unique_ptr<std::ofstream> output_;
    bool wrote_row_ = false;
    bool closed_ = false;
};

class SourceCarrierChannelDiagnostics {
public:
    SourceCarrierChannelDiagnostics(
        std::string path,
        std::vector<SourceCarrierChannelRule> rules)
        : path_(std::move(path)),
          rules_(std::move(rules)) {
        if (path_.empty()) {
            return;
        }
        output_ = std::make_unique<std::ofstream>(path_, std::ios::binary);
        if (!*output_) {
            throw std::runtime_error(
                "failed to open source carrier channel diagnostics: " + path_);
        }
        auto& output = *output_;
        output << "{\"schema\":\"dtmb.c3780_source_carrier_channel.v1\"";
        output << ",\"rules\":[";
        for (std::size_t index = 0; index < rules_.size(); ++index) {
            if (index != 0) {
                output << ',';
            }
            output << "{\"carrier\":" << rules_[index].carrier
                   << ",\"first_frame\":" << rules_[index].first_frame
                   << ",\"last_frame\":" << rules_[index].last_frame << '}';
        }
        output << "],\"rows\":[";
    }

    SourceCarrierChannelDiagnostics(const SourceCarrierChannelDiagnostics&) = delete;
    SourceCarrierChannelDiagnostics& operator=(
        const SourceCarrierChannelDiagnostics&) = delete;

    ~SourceCarrierChannelDiagnostics() {
        if (output_ && !closed_) {
            *output_ << "]}";
        }
    }

    void observe(
        std::size_t source_frame,
        std::size_t model_index,
        std::size_t model_frame_index,
        const dtmb::core::Pn945WidebandChannelModel& model,
        float noise_variance,
        float response_floor,
        int response_window_offset_adjust) {
        if (!output_ || rules_.empty()) {
            return;
        }
        if (model.template_response_fft.size()
            < dtmb::core::kC3780FrameBodySymbols * 2) {
            throw std::runtime_error(
                "source carrier channel diagnostics require template response FFT");
        }
        if (model_frame_index * 2 + 1 >= model.frame_response_scales.size()) {
            throw std::runtime_error(
                "source carrier channel diagnostics model frame index is outside response scales");
        }
        bool matched_frame = false;
        for (const auto& rule : rules_) {
            if (source_frame >= rule.first_frame && source_frame <= rule.last_frame) {
                matched_frame = true;
                break;
            }
        }
        if (!matched_frame) {
            return;
        }

        const auto response_scale = std::complex<float>{
            model.frame_response_scales[model_frame_index * 2],
            model.frame_response_scales[model_frame_index * 2 + 1],
        };
        const auto response_window_offset =
            wideband_response_window_offset(model, response_window_offset_adjust);
        const auto masked_frame_taps = model.scale_estimator
            == dtmb::core::Pn945WidebandScaleEstimator::masked_frame_taps;
        std::vector<float> masked_frame_response_fft;
        if (masked_frame_taps) {
            const auto span = model.template_taps.size() / 2;
            if (span == 0
                || (model_frame_index + 1) * span * 2
                    > model.frame_template_taps.size()) {
                throw std::runtime_error(
                    "source carrier channel diagnostics missing masked frame taps");
            }
            std::vector<float> padded_taps(
                dtmb::core::kC3780FrameBodySymbols * 2,
                0.0F);
            const auto base = model_frame_index * span * 2;
            std::copy_n(
                model.frame_template_taps.begin() + static_cast<std::ptrdiff_t>(base),
                span * 2,
                padded_taps.begin());
            masked_frame_response_fft.resize(dtmb::core::kC3780FrameBodySymbols * 2);
            dtmb::core::mixed_radix_fft_forward_cf32(
                padded_taps,
                masked_frame_response_fft);
        }
        auto& output = *output_;
        for (std::size_t rule_index = 0; rule_index < rules_.size(); ++rule_index) {
            const auto& rule = rules_[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto inserted_logical = data_carrier_to_inserted_logical(rule.carrier);
            const auto physical_bin = c3780_logical_to_physical(inserted_logical);
            auto template_response = std::complex<float>{
                model.template_response_fft[physical_bin * 2],
                model.template_response_fft[physical_bin * 2 + 1],
            };
            auto frame_response = masked_frame_taps
                ? std::complex<float>{
                    masked_frame_response_fft[physical_bin * 2],
                    masked_frame_response_fft[physical_bin * 2 + 1],
                }
                : template_response * response_scale;
            auto channel = frame_response;
            if (response_window_offset != 0) {
                const auto angle = 2.0F * std::numbers::pi_v<float>
                    * static_cast<float>(response_window_offset)
                    * static_cast<float>(physical_bin)
                    / static_cast<float>(dtmb::core::kC3780FrameBodySymbols);
                channel *= std::complex<float>{std::cos(angle), std::sin(angle)};
            }

            std::complex<float> equalizer_coefficient{};
            double mmse_denominator = 0.0;
            if (noise_variance >= 0.0F) {
                mmse_denominator =
                    static_cast<double>(std::norm(channel)) + noise_variance;
                equalizer_coefficient =
                    std::conj(channel) / static_cast<float>(mmse_denominator);
            } else if (std::abs(channel) >= response_floor) {
                equalizer_coefficient = 1.0F / channel;
            } else {
                equalizer_coefficient = std::complex<float>{1.0F, 0.0F};
            }

            if (wrote_row_) {
                output << ',';
            }
            wrote_row_ = true;
            output << "{\"rule_index\":" << rule_index
                   << ",\"source_frame\":" << source_frame
                   << ",\"model_index\":" << model_index
                   << ",\"model_frame_index\":" << model_frame_index
                   << ",\"carrier\":" << rule.carrier
                   << ",\"inserted_logical\":" << inserted_logical
                   << ",\"physical_bin\":" << physical_bin
                   << ",\"masked_frame_taps\":"
                   << (masked_frame_taps ? "true" : "false")
                   << ",\"response_window_offset\":" << response_window_offset
                   << ",\"noise_variance\":" << noise_variance
                   << ",\"response_floor\":" << response_floor
                   << ",\"template_response_real\":" << template_response.real()
                   << ",\"template_response_imag\":" << template_response.imag()
                   << ",\"template_response_abs\":" << std::abs(template_response)
                   << ",\"template_response_phase_rad\":"
                   << std::atan2(
                          template_response.imag(),
                          template_response.real())
                   << ",\"response_scale_real\":" << response_scale.real()
                   << ",\"response_scale_imag\":" << response_scale.imag()
                   << ",\"response_scale_abs\":" << std::abs(response_scale)
                   << ",\"response_scale_phase_rad\":"
                   << std::atan2(response_scale.imag(), response_scale.real())
                   << ",\"frame_response_real\":" << frame_response.real()
                   << ",\"frame_response_imag\":" << frame_response.imag()
                   << ",\"frame_response_abs\":" << std::abs(frame_response)
                   << ",\"frame_response_phase_rad\":"
                   << std::atan2(frame_response.imag(), frame_response.real())
                   << ",\"channel_real\":" << channel.real()
                   << ",\"channel_imag\":" << channel.imag()
                   << ",\"channel_abs\":" << std::abs(channel)
                   << ",\"channel_phase_rad\":"
                   << std::atan2(channel.imag(), channel.real())
                   << ",\"equalizer_coefficient_real\":"
                   << equalizer_coefficient.real()
                   << ",\"equalizer_coefficient_imag\":"
                   << equalizer_coefficient.imag()
                   << ",\"equalizer_coefficient_abs\":"
                   << std::abs(equalizer_coefficient)
                   << ",\"equalizer_coefficient_phase_rad\":"
                   << std::atan2(
                          equalizer_coefficient.imag(),
                          equalizer_coefficient.real())
                   << ",\"mmse_denominator\":";
            if (noise_variance >= 0.0F) {
                output << mmse_denominator;
            } else {
                output << "null";
            }
            output << '}';
        }
        if (!output) {
            throw std::runtime_error(
                "failed to write source carrier channel diagnostics");
        }
    }

    void close() {
        if (!output_ || closed_) {
            return;
        }
        *output_ << "]}";
        if (!*output_) {
            throw std::runtime_error(
                "failed to close source carrier channel diagnostics");
        }
        closed_ = true;
    }

private:
    std::string path_;
    std::vector<SourceCarrierChannelRule> rules_;
    std::unique_ptr<std::ofstream> output_;
    bool wrote_row_ = false;
    bool closed_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    std::size_t phase_offset = 0;
    std::size_t max_frames = 0;
    std::size_t sync_frames = 300;
    std::size_t acquisition_frames = 16;
    std::size_t requested_workers = 0;
    std::size_t batch_frames = 0;
    std::size_t pn_channel_taps = 8;
    std::size_t pn_wideband_block_frames = 16;
    std::size_t timing_search_radius = 0;
    std::size_t timing_trajectory_interval_frames = 0;
    std::size_t timing_trajectory_fit_points = 9;
    float timing_trajectory_max_innovation_samples = 2.0F;
    std::string timing_trajectory_seed_path;
    std::size_t timing_trajectory_frame_index_offset = 0;
    double timing_trajectory_offset_origin_samples = 0.0;
    std::string timing_trajectory_state_out_path;
    std::size_t timing_trajectory_state_out_frame = 0;
    bool timing_trajectory_local_search = false;
    float timing_trajectory_local_search_min_improvement = 0.0F;
    std::vector<TimingLocalSearchScopedMinImprovementRule>
        timing_trajectory_local_search_scoped_min_improvements;
    bool timing_trajectory_local_search_transient = false;
    std::vector<TimingLocalSearchTransientRange>
        timing_trajectory_local_search_transient_ranges;
    std::int64_t auto_phase_adjustment = 0;
    float frequency_shift_hz = 0.0F;
    float sync_hit_threshold = 0.35F;
    float timing_search_threshold = 0.45F;
    std::size_t system_info_index = 23;
    bool system_info_auto = false;
    std::size_t system_info_auto_observation_frames = 32;
    float system_info_auto_min_metric = 0.75F;
    float system_info_auto_min_margin = 0.03F;
    auto normalization = Normalization::system_info;
    auto equalizer = Equalizer::flat;
    auto pn_estimator = PnEstimator::compact;
    auto pn_wideband_scale_estimator =
        dtmb::core::Pn945WidebandScaleEstimator::dominant_tap;
    auto pn_wideband_header_observation =
        dtmb::core::Pn945HeaderObservation::core_only;
    std::size_t pn_wideband_max_span_symbols =
        dtmb::core::Pn945WidebandModelOptions{}.max_span_symbols;
    int pn_wideband_response_window_offset_adjust = 0;
    bool pn_wideband_body_channel_midpoint = false;
    bool pn_csi_demap = false;
    auto pn_mmse = PnMmse{};
    bool remove_dc = false;
    bool auto_sync = false;
    bool estimate_residual_cfo = true;
    bool phase_offset_set = false;
    auto data_dd_options = DataDecisionDirectedOptions{};
    auto data_dd_stats = DataDecisionDirectedStats{};
    std::string wideband_diagnostics_path;
    std::string wideband_frame_diagnostics_path;
    std::string pn_csi_weights_path;
    std::string timing_diagnostics_path;
    std::string frame_residual_diagnostics_path;
    std::string normalization_diagnostics_path;
    std::string source_carrier_residual_diagnostics_path;
    std::vector<SourceCarrierResidualRule> source_carrier_residual_rules;
    std::string source_carrier_channel_diagnostics_path;
    std::vector<SourceCarrierChannelRule> source_carrier_channel_rules;
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
            } else if (arg == "--auto-phase-adjustment") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto_phase_adjustment = parse_signed_size(
                    argv[index],
                    "auto phase adjustment");
            } else if (arg == "--timing-search-radius") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_search_radius = parse_size(argv[index], "timing search radius");
            } else if (arg == "--timing-search-threshold") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_search_threshold = parse_float(argv[index], "timing search threshold");
            } else if (arg == "--timing-trajectory-interval-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_interval_frames = parse_size(
                    argv[index],
                    "timing trajectory interval frame count");
            } else if (arg == "--timing-trajectory-fit-points") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_fit_points = parse_size(
                    argv[index],
                    "timing trajectory fit point count");
            } else if (arg == "--timing-trajectory-max-innovation-samples") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_max_innovation_samples = parse_float(
                    argv[index],
                    "timing trajectory maximum innovation");
            } else if (arg == "--timing-trajectory-seed") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_seed_path = argv[index];
            } else if (arg == "--timing-trajectory-frame-index-offset") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_frame_index_offset = parse_size(
                    argv[index],
                    "timing trajectory frame index offset");
            } else if (arg == "--timing-trajectory-offset-origin-samples") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_offset_origin_samples = parse_double(
                    argv[index],
                    "timing trajectory offset origin");
            } else if (arg == "--timing-trajectory-state-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_state_out_path = argv[index];
            } else if (arg == "--timing-trajectory-state-out-frame") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_state_out_frame = parse_size(
                    argv[index],
                    "timing trajectory state output frame");
            } else if (arg == "--timing-trajectory-local-search") {
                timing_trajectory_local_search = true;
            } else if (arg == "--timing-trajectory-local-search-transient") {
                timing_trajectory_local_search_transient = true;
            } else if (arg == "--timing-trajectory-local-search-transient-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_local_search_transient_ranges.push_back(
                    parse_timing_local_search_transient_range(argv[index]));
            } else if (arg == "--timing-trajectory-local-search-min-improvement") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_local_search_min_improvement = parse_float(
                    argv[index],
                    "timing trajectory local-search minimum improvement");
            } else if (
                arg == "--timing-trajectory-local-search-scoped-min-improvement") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                const auto rule =
                    parse_timing_local_search_scoped_min_improvement_rule(argv[index]);
                for (const auto& existing :
                     timing_trajectory_local_search_scoped_min_improvements) {
                    const auto disjoint = rule.last_frame < existing.first_frame
                        || existing.last_frame < rule.first_frame;
                    if (!disjoint) {
                        throw std::invalid_argument(
                            std::string(
                                "overlapping timing local-search scoped "
                                "min-improvement rule: ")
                            + argv[index]);
                    }
                }
                timing_trajectory_local_search_scoped_min_improvements.push_back(rule);
            } else if (arg == "--timing-diagnostics") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_diagnostics_path = argv[index];
            } else if (arg == "--no-residual-cfo") {
                estimate_residual_cfo = false;
            } else if (arg == "--max-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                max_frames = parse_size(argv[index], "max frames");
            } else if (arg == "--frequency-shift-hz") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                frequency_shift_hz = parse_float(argv[index], "frequency shift");
            } else if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                requested_workers = parse_size(argv[index], "worker count");
            } else if (arg == "--batch-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                batch_frames = parse_size(argv[index], "batch frame count");
            } else if (arg == "--equalizer") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                equalizer = parse_equalizer(argv[index]);
            } else if (arg == "--pn-channel-taps") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_channel_taps = parse_size(argv[index], "PN channel tap count");
            } else if (arg == "--pn-estimator") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_estimator = parse_pn_estimator(argv[index]);
            } else if (arg == "--pn-wideband-block-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_wideband_block_frames = parse_size(
                    argv[index],
                    "PN wideband block frame count");
            } else if (arg == "--pn-wideband-header-observation") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_wideband_header_observation =
                    parse_pn_wideband_header_observation(argv[index]);
            } else if (arg == "--pn-wideband-scale-estimator") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_wideband_scale_estimator =
                    parse_pn_wideband_scale_estimator(argv[index]);
            } else if (arg == "--pn-wideband-max-span-symbols") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_wideband_max_span_symbols = parse_size(
                    argv[index],
                    "PN wideband maximum span");
            } else if (arg == "--pn-wideband-response-window-offset-adjust") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_wideband_response_window_offset_adjust = static_cast<int>(
                    parse_signed_size(
                        argv[index],
                        "PN wideband response-window offset adjustment"));
            } else if (arg == "--pn-wideband-body-channel") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_wideband_body_channel_midpoint =
                    parse_pn_wideband_body_channel_midpoint(argv[index]);
            } else if (arg == "--pn-csi-demap") {
                pn_csi_demap = true;
            } else if (arg == "--pn-csi-weights-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_csi_weights_path = argv[index];
            } else if (arg == "--pn-wideband-diagnostics") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                wideband_diagnostics_path = argv[index];
            } else if (arg == "--pn-wideband-frame-diagnostics") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                wideband_frame_diagnostics_path = argv[index];
            } else if (arg == "--frame-residual-diagnostics") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                frame_residual_diagnostics_path = argv[index];
            } else if (arg == "--normalization-diagnostics") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                normalization_diagnostics_path = argv[index];
            } else if (arg == "--source-carrier-residual") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                source_carrier_residual_rules.push_back(
                    parse_source_carrier_residual_rule(argv[index]));
            } else if (arg == "--source-carrier-residual-diagnostics-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                source_carrier_residual_diagnostics_path = argv[index];
            } else if (arg == "--source-carrier-channel") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                source_carrier_channel_rules.push_back(
                    parse_source_carrier_channel_rule(argv[index]));
            } else if (arg == "--source-carrier-channel-diagnostics-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                source_carrier_channel_diagnostics_path = argv[index];
            } else if (arg == "--pn-mmse") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                pn_mmse = parse_pn_mmse(argv[index]);
            } else if (arg == "--remove-dc") {
                remove_dc = true;
            } else if (arg == "--normalization") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                normalization = parse_normalization(argv[index]);
            } else if (arg == "--data-dd-refine") {
                data_dd_options.enabled = true;
            } else if (arg == "--data-dd-max-relative-error") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                data_dd_options.max_relative_error = parse_float(
                    argv[index],
                    "data DD maximum relative error");
            } else if (arg == "--data-dd-min-reliable-carriers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                data_dd_options.min_reliable_carriers = parse_size(
                    argv[index],
                    "data DD minimum reliable carrier count");
            } else if (arg == "--data-dd-max-axis-inner-fraction") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                data_dd_options.max_axis_inner_fraction = parse_float(
                    argv[index],
                    "data DD maximum axis-inner fraction");
            } else if (arg == "--data-dd-max-hard-bit-bias") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                data_dd_options.max_hard_bit_bias = parse_float(
                    argv[index],
                    "data DD maximum hard-bit bias");
            } else if (arg == "--system-info-index") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                const auto value = std::string_view{argv[index]};
                if (value == "auto") {
                    system_info_auto = true;
                } else {
                    system_info_index = parse_size(argv[index], "system information index");
                    system_info_auto = false;
                }
            } else if (arg == "--system-info-auto-observation-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                system_info_auto_observation_frames = parse_size(
                    argv[index],
                    "system information auto observation frames");
            } else if (arg == "--system-info-auto-min-metric") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                system_info_auto_min_metric = parse_float(
                    argv[index],
                    "system information auto minimum metric");
            } else if (arg == "--system-info-auto-min-margin") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                system_info_auto_min_margin = parse_float(
                    argv[index],
                    "system information auto minimum margin");
            } else if (arg == "--no-normalize") {
                normalization = Normalization::none;
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                positional.push_back(arg);
            }
        }
        if (positional.size() > 2 || pn_channel_taps == 0 || pn_channel_taps > 511
            || pn_wideband_block_frames == 0
            || pn_wideband_max_span_symbols == 0
            || pn_wideband_max_span_symbols > 511
            || timing_search_radius > dtmb::core::kPn945HeaderSymbols
            || timing_trajectory_fit_points > 256
            || !std::isfinite(timing_trajectory_max_innovation_samples)
            || timing_trajectory_max_innovation_samples < 0.0F
            || timing_trajectory_max_innovation_samples
                > static_cast<float>(dtmb::core::kPn945HeaderSymbols)
            || !std::isfinite(timing_trajectory_local_search_min_improvement)
            || timing_trajectory_local_search_min_improvement < 0.0F
            || sync_frames < 2
            || acquisition_frames == 0
            || sync_hit_threshold < 0.0F
            || sync_hit_threshold > 1.0F
            || timing_search_threshold < 0.0F
            || timing_search_threshold > 1.0F
            || auto_phase_adjustment
                <= -static_cast<std::int64_t>(dtmb::core::kPn945FrameSymbols)
            || auto_phase_adjustment
                >= static_cast<std::int64_t>(dtmb::core::kPn945FrameSymbols)
            || sync_frames
                > std::numeric_limits<std::size_t>::max()
                    / (dtmb::core::kPn945FrameSymbols * 2)
                    - 1
            || !std::isfinite(data_dd_options.max_axis_inner_fraction)
            || data_dd_options.max_axis_inner_fraction < 0.0F
            || data_dd_options.max_axis_inner_fraction > 1.0F
            || !std::isfinite(data_dd_options.max_hard_bit_bias)
            || data_dd_options.max_hard_bit_bias < 0.0F
            || system_info_auto_observation_frames == 0
            || !std::isfinite(system_info_auto_min_metric)
            || system_info_auto_min_metric < 0.0F
            || system_info_auto_min_metric > 1.0F
            || !std::isfinite(system_info_auto_min_margin)
            || system_info_auto_min_margin < 0.0F
            || system_info_auto_min_margin > 1.0F
            || phase_offset > std::numeric_limits<std::size_t>::max() / 2) {
            usage(argv[0]);
            return 2;
        }
        if (auto_sync && phase_offset_set) {
            throw std::invalid_argument("--auto-sync and --phase-offset are mutually exclusive");
        }
        if (!auto_sync && auto_phase_adjustment != 0) {
            throw std::invalid_argument(
                "--auto-phase-adjustment requires --auto-sync");
        }
        if (timing_trajectory_interval_frames != 0 && timing_search_radius == 0) {
            throw std::invalid_argument(
                "--timing-trajectory-interval-frames requires --timing-search-radius");
        }
        if (timing_trajectory_interval_frames != 0 && timing_trajectory_fit_points < 2) {
            throw std::invalid_argument(
                "--timing-trajectory-fit-points must be at least 2");
        }
        if (!timing_trajectory_seed_path.empty()
            && timing_trajectory_interval_frames == 0) {
            throw std::invalid_argument(
                "--timing-trajectory-seed requires "
                "--timing-trajectory-interval-frames");
        }
        if (timing_trajectory_frame_index_offset != 0
            && timing_trajectory_interval_frames == 0) {
            throw std::invalid_argument(
                "--timing-trajectory-frame-index-offset requires "
                "--timing-trajectory-interval-frames");
        }
        if (timing_trajectory_offset_origin_samples != 0.0
            && timing_trajectory_interval_frames == 0) {
            throw std::invalid_argument(
                "--timing-trajectory-offset-origin-samples requires "
                "--timing-trajectory-interval-frames");
        }
        if (timing_trajectory_offset_origin_samples != 0.0
            && timing_trajectory_seed_path.empty()) {
            throw std::invalid_argument(
                "--timing-trajectory-offset-origin-samples requires "
                "--timing-trajectory-seed");
        }
        if (!timing_trajectory_state_out_path.empty()
            && timing_trajectory_interval_frames == 0) {
            throw std::invalid_argument(
                "--timing-trajectory-state-out requires "
                "--timing-trajectory-interval-frames");
        }
        if (timing_trajectory_state_out_frame != 0
            && timing_trajectory_state_out_path.empty()) {
            throw std::invalid_argument(
                "--timing-trajectory-state-out-frame requires "
                "--timing-trajectory-state-out");
        }
        if (timing_trajectory_local_search && timing_trajectory_interval_frames == 0) {
            throw std::invalid_argument(
                "--timing-trajectory-local-search requires "
                "--timing-trajectory-interval-frames");
        }
        if (!timing_trajectory_local_search_scoped_min_improvements.empty()
            && !timing_trajectory_local_search) {
            throw std::invalid_argument(
                "--timing-trajectory-local-search-scoped-min-improvement requires "
                "--timing-trajectory-local-search");
        }
        if (timing_trajectory_local_search_min_improvement != 0.0F
            && !timing_trajectory_local_search) {
            throw std::invalid_argument(
                "--timing-trajectory-local-search-min-improvement requires "
                "--timing-trajectory-local-search");
        }
        if (timing_trajectory_local_search_transient
            && !timing_trajectory_local_search) {
            throw std::invalid_argument(
                "--timing-trajectory-local-search-transient requires "
                "--timing-trajectory-local-search");
        }
        if (!timing_trajectory_local_search_transient_ranges.empty()
            && !timing_trajectory_local_search) {
            throw std::invalid_argument(
                "--timing-trajectory-local-search-transient-range requires "
                "--timing-trajectory-local-search");
        }
        if (pn_mmse.automatic && pn_estimator != PnEstimator::wideband) {
            throw std::invalid_argument("--pn-mmse auto requires --pn-estimator wideband");
        }
        if (data_dd_options.enabled && normalization != Normalization::qam64) {
            throw std::invalid_argument("--data-dd-refine requires --normalization qam64");
        }
        if (data_dd_options.max_relative_error <= 0.0F) {
            throw std::invalid_argument("--data-dd-max-relative-error must be positive");
        }
        if (data_dd_options.min_reliable_carriers > dtmb::core::kC3780DataSymbols) {
            throw std::invalid_argument(
                "--data-dd-min-reliable-carriers exceeds C3780 data carrier count");
        }
        for (const auto& rule : source_carrier_residual_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier residual carrier is outside the C3780 data span");
            }
        }
        if (!source_carrier_residual_rules.empty()
            && source_carrier_residual_diagnostics_path.empty()) {
            throw std::invalid_argument(
                "--source-carrier-residual requires --source-carrier-residual-diagnostics-out");
        }
        for (const auto& rule : source_carrier_channel_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier channel carrier is outside the C3780 data span");
            }
        }
        if (!source_carrier_channel_rules.empty()
            && source_carrier_channel_diagnostics_path.empty()) {
            throw std::invalid_argument(
                "--source-carrier-channel requires --source-carrier-channel-diagnostics-out");
        }
        if (!source_carrier_channel_rules.empty()
            && (equalizer != Equalizer::pn || pn_estimator != PnEstimator::wideband)) {
            throw std::invalid_argument(
                "--source-carrier-channel requires --equalizer pn --pn-estimator wideband");
        }
        if (!wideband_frame_diagnostics_path.empty()
            && (equalizer != Equalizer::pn || pn_estimator != PnEstimator::wideband)) {
            throw std::invalid_argument(
                "--pn-wideband-frame-diagnostics requires --equalizer pn --pn-estimator wideband");
        }
        if (pn_csi_demap
            && (equalizer != Equalizer::pn || pn_estimator != PnEstimator::wideband)) {
            throw std::invalid_argument(
                "--pn-csi-demap requires --equalizer pn --pn-estimator wideband");
        }
        if (pn_csi_demap && pn_csi_weights_path.empty()) {
            throw std::invalid_argument(
                "--pn-csi-demap requires --pn-csi-weights-out");
        }
        if (!pn_csi_demap && !pn_csi_weights_path.empty()) {
            throw std::invalid_argument(
                "--pn-csi-weights-out requires --pn-csi-demap");
        }
        if (pn_csi_demap && normalization != Normalization::qam64) {
            throw std::invalid_argument(
                "--pn-csi-demap requires --normalization qam64");
        }
        if (pn_csi_demap
            && !pn_mmse.automatic
            && pn_mmse.noise_variance <= 0.0F) {
            throw std::invalid_argument("--pn-csi-demap requires PN MMSE equalization");
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }
        if (positional.size() == 2) {
            output_path = positional[1];
        }

        dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);
        std::ofstream pn_csi_weights;
        if (!pn_csi_weights_path.empty()) {
            pn_csi_weights.open(pn_csi_weights_path, std::ios::binary);
            if (!pn_csi_weights) {
                throw std::runtime_error(
                    "failed to open PN CSI weights output: " + pn_csi_weights_path);
            }
        }
        std::ofstream wideband_diagnostics;
        if (!wideband_diagnostics_path.empty()) {
            wideband_diagnostics.open(wideband_diagnostics_path, std::ios::binary);
            if (!wideband_diagnostics) {
                throw std::runtime_error(
                    "failed to open PN wideband diagnostics: "
                    + wideband_diagnostics_path);
            }
            wideband_diagnostics
                << "model_index,first_output_frame,available_frames,output_frames,"
                << "model_frame_count,span_symbols,significant_taps,"
                << "rotation_symbols,base_pn_phase,pn_phase,dominant_tap_index,"
                << "phase_agreement,per_frame_noise_tap_power,noise_variance,"
                << "truncated_energy_fraction,dc_offset_i,dc_offset_q\n";
        }
        std::ofstream wideband_frame_diagnostics;
        if (!wideband_frame_diagnostics_path.empty()) {
            wideband_frame_diagnostics.open(
                wideband_frame_diagnostics_path,
                std::ios::binary);
            if (!wideband_frame_diagnostics) {
                throw std::runtime_error(
                    "failed to open PN wideband frame diagnostics: "
                    + wideband_frame_diagnostics_path);
            }
            wideband_frame_diagnostics
                << "source_frame,model_index,model_frame_index,"
                << "sample_start,next_sample_start,pn_phase,next_pn_phase,"
                << "model_pn_phase,model_next_pn_phase,"
                << "response_scale_real,response_scale_imag,"
                << "response_scale_abs,response_scale_phase_rad,"
                << "next_tap_phase_drift_rad,next_tap_coherence,"
                << "next_to_current_tap_power,"
                << "next_sampled_carrier_shape_change_rms,"
                << "next_sampled_carrier_shape_change_max,"
                << "slot0_next_sampled_carrier_shape_change_rms,"
                << "slot0_next_sampled_carrier_shape_change_max,"
                << "slot1_next_sampled_carrier_shape_change_rms,"
                << "slot1_next_sampled_carrier_shape_change_max,"
                << "slot2_next_sampled_carrier_shape_change_rms,"
                << "slot2_next_sampled_carrier_shape_change_max,"
                << "model_rotation_symbols,model_dominant_tap_index,"
                << "model_span_symbols,model_significant_taps,"
                << "model_phase_agreement,model_noise_variance,"
                << "model_truncated_energy_fraction,dc_offset_i,dc_offset_q\n";
        }
        SourceCarrierResidualDiagnostics source_carrier_residuals(
            source_carrier_residual_diagnostics_path,
            source_carrier_residual_rules);
        SourceCarrierChannelDiagnostics source_carrier_channels(
            source_carrier_channel_diagnostics_path,
            source_carrier_channel_rules);
        TimingDiagnostics timing_diagnostics(timing_diagnostics_path);
        FrameResidualDiagnostics frame_residuals(frame_residual_diagnostics_path);
        FrameNormalizationDiagnostics normalization_diagnostics(
            normalization_diagnostics_path,
            normalization);

        auto worker_count = requested_workers;
        if (worker_count == 0) {
            worker_count = std::thread::hardware_concurrency();
        }
        worker_count = std::max<std::size_t>(worker_count, 1);
        if (batch_frames == 0) {
            batch_frames = worker_count;
        }
        if (batch_frames == 0) {
            throw std::invalid_argument("batch frame count must be positive");
        }
        worker_count = std::min(worker_count, batch_frames);

        const auto manual_frequency_shift_hz = frequency_shift_hz;
        std::vector<std::int8_t> startup_ci8;
        std::size_t startup_buffered_samples = 0;
        auto acquisition = dtmb::core::Pn945AcquisitionResult{};
        auto residual_cfo = dtmb::core::Pn945ResidualCfoResult{};
        float automatic_frequency_shift_hz = 0.0F;
        if (auto_sync) {
            const auto startup_samples = (sync_frames + 1) * dtmb::core::kPn945FrameSymbols;
            startup_ci8.resize(startup_samples * 2);
            const auto startup_bytes = read_bytes(input, startup_ci8);
            startup_ci8.resize(startup_bytes);
            if ((startup_ci8.size() % 2) != 0) {
                throw std::runtime_error("auto-sync startup buffer ended with an incomplete CI8 sample");
            }
            if (startup_ci8.size() / 2 < dtmb::core::kPn945FrameSymbols * 2) {
                throw std::runtime_error("auto-sync requires at least two PN945 frames");
            }
            startup_buffered_samples = startup_ci8.size() / 2;
            std::vector<float> startup_cf32(startup_ci8.size());
            convert_ci8_to_cf32(startup_ci8, startup_cf32, 0, 0.0F);
            acquisition = dtmb::core::acquire_pn945_cf32(
                startup_cf32,
                dtmb::core::Pn945AcquisitionOptions{
                    acquisition_frames,
                    sync_hit_threshold,
                    requested_workers,
                });
            if (acquisition.hit_count < 2) {
                throw std::runtime_error(
                    "auto-sync did not find a repeated PN945 train above the hit threshold");
            }
            phase_offset = acquisition.phase_offset;
            const auto adjusted_phase =
                static_cast<std::int64_t>(phase_offset) + auto_phase_adjustment;
            phase_offset = static_cast<std::size_t>(
                (adjusted_phase
                    + static_cast<std::int64_t>(dtmb::core::kPn945FrameSymbols))
                % static_cast<std::int64_t>(dtmb::core::kPn945FrameSymbols));
            if (acquisition.coarse_cfo_valid) {
                automatic_frequency_shift_hz = -acquisition.coarse_cfo_hz;
            }
            if (estimate_residual_cfo) {
                convert_ci8_to_cf32(
                    startup_ci8,
                    startup_cf32,
                    0,
                    automatic_frequency_shift_hz);
                residual_cfo = dtmb::core::estimate_pn945_residual_cfo_cf32(
                    startup_cf32,
                    phase_offset,
                    dtmb::core::Pn945ResidualCfoOptions{
                        sync_frames,
                        0.5F,
                        requested_workers,
                    });
                if (residual_cfo.valid) {
                    automatic_frequency_shift_hz -= residual_cfo.cfo_hz;
                }
            }
            frequency_shift_hz += automatic_frequency_shift_hz;
        }

        ReplayInput replay_input(input, std::move(startup_ci8));
        const auto phase_bytes = phase_offset * 2;
        const auto discarded_phase_bytes = discard_bytes(replay_input, phase_bytes);
        const auto timing_trajectory_seed = load_timing_trajectory_seed(
            timing_trajectory_seed_path);
        if (timing_trajectory_seed.frame_index_offset.has_value()) {
            if (timing_trajectory_frame_index_offset != 0
                && timing_trajectory_frame_index_offset
                    != *timing_trajectory_seed.frame_index_offset) {
                throw std::invalid_argument(
                    "timing trajectory seed frame-index metadata conflicts with CLI");
            }
            timing_trajectory_frame_index_offset =
                *timing_trajectory_seed.frame_index_offset;
        }
        if (system_info_auto && normalization == Normalization::system_info) {
            throw std::invalid_argument(
                "--system-info-index auto requires qam64, qam64-amplitude, or no normalization");
        }
        if (timing_trajectory_seed.offset_origin_samples.has_value()) {
            if (timing_trajectory_offset_origin_samples != 0.0
                && timing_trajectory_offset_origin_samples
                    != *timing_trajectory_seed.offset_origin_samples) {
                throw std::invalid_argument(
                    "timing trajectory seed offset-origin metadata conflicts with CLI");
            }
            timing_trajectory_offset_origin_samples =
                *timing_trajectory_seed.offset_origin_samples;
        }
        TrackingFrameReader frame_reader(
            replay_input,
            phase_offset,
            timing_search_radius,
            timing_search_threshold,
            timing_trajectory_interval_frames,
            timing_trajectory_fit_points,
            timing_trajectory_max_innovation_samples,
            timing_trajectory_seed.points,
            timing_trajectory_frame_index_offset,
            timing_trajectory_offset_origin_samples,
            timing_trajectory_state_out_path,
            timing_trajectory_state_out_frame,
            timing_trajectory_local_search,
            timing_trajectory_local_search_min_improvement,
            timing_trajectory_local_search_scoped_min_improvements,
            timing_trajectory_local_search_transient,
            timing_trajectory_local_search_transient_ranges,
            &timing_diagnostics);
        auto reference_index = system_info_index;
        auto reference = system_info_reference(reference_index);
        auto system_info_selector = SystemInfoAutoSelector{
            system_info_auto_observation_frames,
            system_info_auto_min_metric,
            system_info_auto_min_margin,
        };
        const auto observe_system_information = [&](const FrameWork& frame) {
            if (!system_info_auto) {
                return;
            }
            system_info_selector.observe(frame.logical_cf32);
            const auto& selection = system_info_selector.selection();
            if (selection.locked && selection.index != reference_index) {
                reference_index = selection.index;
                reference = system_info_reference(reference_index);
            }
        };

        std::size_t frame_count = 0;
        std::size_t input_frame_count = 0;
        std::size_t trailing_ci8_bytes = 0;
        std::size_t first_pn_phase = 0;
        std::size_t last_pn_phase = 0;
        std::size_t wideband_model_count = 0;
        std::size_t wideband_span_symbols = 0;
        std::size_t wideband_significant_taps = 0;
        std::size_t wideband_rotation_symbols = 0;
        float wideband_auto_noise_variance = 0.0F;
        std::complex<float> wideband_last_dc_offset{};
        auto wideband_model_stats = WidebandModelStats{};
        std::size_t next_sample_start = phase_offset;
        if (discarded_phase_bytes == phase_bytes
            && equalizer == Equalizer::pn
            && pn_estimator == PnEstimator::compact) {
            FrameWork current;
            FrameWork next;
            auto have_current = timing_search_radius == 0
                ? read_frame(
                    replay_input,
                    current,
                    trailing_ci8_bytes,
                    next_sample_start)
                : frame_reader.read(current, trailing_ci8_bytes);
            input_frame_count += have_current ? 1U : 0U;
            auto have_next = have_current && (timing_search_radius == 0
                ? read_frame(
                    replay_input,
                    next,
                    trailing_ci8_bytes,
                    next_sample_start)
                : frame_reader.read(next, trailing_ci8_bytes));
            input_frame_count += have_next ? 1U : 0U;
            while (have_current && have_next
                   && (max_frames == 0 || frame_count < max_frames)) {
                const auto result = process_pn_frame(
                    current,
                    next,
                    normalization,
                    reference,
                    pn_channel_taps,
                    frequency_shift_hz,
                    pn_mmse.noise_variance,
                    data_dd_options,
                    data_dd_stats);
                if (frame_count == 0) {
                    first_pn_phase = result.pn_phase;
                }
                last_pn_phase = result.pn_phase;
                frame_residuals.observe(frame_count, current.data_cf32);
                observe_system_information(current);
                normalization_diagnostics.observe(
                    frame_count,
                    current.normalization_stats,
                    true,
                    result.pn_phase,
                    result.next_pn_phase,
                    result.pn_phase,
                    result.next_pn_phase,
                    0U);
                source_carrier_residuals.observe(frame_count, current.data_cf32);
                write_all(output, current.data_cf32);
                ++frame_count;
                std::swap(current, next);
                have_next = timing_search_radius == 0
                    ? read_frame(
                        replay_input,
                        next,
                        trailing_ci8_bytes,
                        next_sample_start)
                    : frame_reader.read(next, trailing_ci8_bytes);
                input_frame_count += have_next ? 1U : 0U;
            }
        } else if (discarded_phase_bytes == phase_bytes
                   && equalizer == Equalizer::pn
                   && pn_estimator == PnEstimator::wideband) {
            std::deque<FrameWork> frames;
            bool input_ended = false;
            while (!input_ended && (max_frames == 0 || frame_count < max_frames)) {
                while (frames.size() < pn_wideband_block_frames + 1) {
                    frames.emplace_back();
                    const auto frame_ok = timing_search_radius == 0
                        ? read_frame(
                            replay_input,
                            frames.back(),
                            trailing_ci8_bytes,
                            next_sample_start)
                        : frame_reader.read(frames.back(), trailing_ci8_bytes);
                    if (!frame_ok) {
                        frames.pop_back();
                        input_ended = true;
                        break;
                    }
                    ++input_frame_count;
                }
                if (frames.size() < 2) {
                    break;
                }
                const auto dc_offset = remove_dc
                    ? estimate_block_dc_offset(frames)
                    : std::complex<float>{};
                const auto model = build_wideband_model(
                    frames,
                    frequency_shift_hz,
                    dc_offset,
                    pn_wideband_scale_estimator,
                    pn_wideband_header_observation,
                    pn_wideband_max_span_symbols);
                ++wideband_model_count;
                const auto model_index = wideband_model_count - 1;
                wideband_model_stats.observe(model);
                wideband_span_symbols = model.template_taps.size() / 2;
                wideband_significant_taps = model.significant_taps;
                wideband_rotation_symbols = model.rotation_symbols;
                wideband_auto_noise_variance = model.noise_variance;
                wideband_last_dc_offset = dc_offset;
                const auto available = frames.size() - 1;
                const auto remaining = max_frames == 0
                    ? available
                    : std::min(available, max_frames - frame_count);
                const auto output_count = std::min(pn_wideband_block_frames, remaining);
                if (wideband_diagnostics.is_open()) {
                    wideband_diagnostics
                        << model_index << ','
                        << frame_count << ','
                        << available << ','
                        << output_count << ','
                        << model.frame_count << ','
                        << wideband_span_symbols << ','
                        << wideband_significant_taps << ','
                        << model.rotation_symbols << ','
                        << model.base_pn_phase << ','
                        << model.pn_phase << ','
                        << model.dominant_tap_index << ','
                        << model.phase_agreement << ','
                        << model.per_frame_noise_tap_power << ','
                        << model.noise_variance << ','
                        << model.truncated_energy_fraction << ','
                        << dc_offset.real() << ','
                        << dc_offset.imag() << '\n';
                    if (!wideband_diagnostics) {
                        throw std::runtime_error(
                            "failed to write PN wideband diagnostics");
                    }
                }
                const auto noise_variance = pn_mmse.automatic
                    ? model.noise_variance
                    : pn_mmse.noise_variance;
                const auto active_workers = std::min(worker_count, output_count);
                std::vector<std::thread> workers;
                std::vector<std::exception_ptr> worker_errors(active_workers);
                std::vector<dtmb::core::Pn945EqualizeResult> results(output_count);
                workers.reserve(active_workers);
                for (std::size_t worker = 0; worker < active_workers; ++worker) {
                    workers.emplace_back([&, worker] {
                        try {
                            for (std::size_t frame = worker;
                                 frame < output_count;
                                 frame += active_workers) {
                                results[frame] = process_wideband_pn_frame(
                                    frames[frame],
                                    frames[frame + 1],
                                    normalization,
                                    reference,
                                    model,
                                    frame,
                                    frequency_shift_hz,
                                    noise_variance,
                                    dc_offset,
                                    pn_wideband_response_window_offset_adjust,
                                    pn_wideband_body_channel_midpoint,
                                    pn_csi_demap,
                                    data_dd_options,
                                    data_dd_stats);
                            }
                        } catch (...) {
                            worker_errors[worker] = std::current_exception();
                        }
                    });
                }
                for (auto& worker : workers) {
                    worker.join();
                }
                for (const auto& error : worker_errors) {
                    if (error) {
                        std::rethrow_exception(error);
                    }
                }
                const auto output_frame_base = frame_count;
                for (std::size_t frame = 0; frame < output_count; ++frame) {
                    const auto& result = results[frame];
                    if (frame_count == 0) {
                        first_pn_phase = result.pn_phase;
                    }
                    last_pn_phase = result.pn_phase;
                    const auto output_frame_index = output_frame_base + frame;
                    if (wideband_frame_diagnostics.is_open()) {
                        const auto response_scale = std::complex<float>{
                            model.frame_response_scales[frame * 2],
                            model.frame_response_scales[frame * 2 + 1],
                        };
                        const auto transition =
                            pn_tap_transition_stats(
                                model,
                                frame,
                                noise_variance);
                        wideband_frame_diagnostics
                            << output_frame_index << ','
                            << model_index << ','
                            << frame << ','
                            << frames[frame].sample_start << ','
                            << frames[frame + 1].sample_start << ','
                            << result.pn_phase << ','
                            << result.next_pn_phase << ','
                            << model.frame_pn_phases[frame] << ','
                            << model.frame_pn_phases[frame + 1] << ','
                            << response_scale.real() << ','
                            << response_scale.imag() << ','
                            << std::abs(response_scale) << ','
                            << std::atan2(
                                   response_scale.imag(),
                                   response_scale.real()) << ','
                            << transition.phase_rad << ','
                            << transition.coherence << ','
                            << transition.next_to_current_power << ','
                            << transition.sampled_carrier_shape_change_rms << ','
                            << transition.sampled_carrier_shape_change_max << ','
                            << transition.slot_sampled_carrier_shape_change_rms[0] << ','
                            << transition.slot_sampled_carrier_shape_change_max[0] << ','
                            << transition.slot_sampled_carrier_shape_change_rms[1] << ','
                            << transition.slot_sampled_carrier_shape_change_max[1] << ','
                            << transition.slot_sampled_carrier_shape_change_rms[2] << ','
                            << transition.slot_sampled_carrier_shape_change_max[2] << ','
                            << model.rotation_symbols << ','
                            << model.dominant_tap_index << ','
                            << (model.template_taps.size() / 2) << ','
                            << model.significant_taps << ','
                            << model.phase_agreement << ','
                            << model.noise_variance << ','
                            << model.truncated_energy_fraction << ','
                            << dc_offset.real() << ','
                            << dc_offset.imag() << '\n';
                        if (!wideband_frame_diagnostics) {
                            throw std::runtime_error(
                                "failed to write PN wideband frame diagnostics");
                        }
                    }
                    frame_residuals.observe(
                        output_frame_index,
                        frames[frame].data_cf32);
                    observe_system_information(frames[frame]);
                    normalization_diagnostics.observe(
                        output_frame_index,
                        frames[frame].normalization_stats,
                        true,
                        result.pn_phase,
                        result.next_pn_phase,
                        model.frame_pn_phases[frame],
                        model.frame_pn_phases[frame + 1U],
                        model.rotation_symbols);
                    source_carrier_residuals.observe(
                        output_frame_index,
                        frames[frame].data_cf32);
                    source_carrier_channels.observe(
                        output_frame_index,
                        model_index,
                        frame,
                        model,
                        noise_variance,
                        1.0e-3F,
                        pn_wideband_response_window_offset_adjust);
                    write_all(output, frames[frame].data_cf32);
                    if (pn_csi_weights.is_open()) {
                        write_all(pn_csi_weights, frames[frame].csi_weights);
                    }
                    ++frame_count;
                }
                if (output_count < pn_wideband_block_frames
                    || (max_frames != 0 && frame_count >= max_frames)) {
                    break;
                }
                for (std::size_t frame = 0; frame < output_count; ++frame) {
                    frames.pop_front();
                }
            }
        } else if (discarded_phase_bytes == phase_bytes) {
            std::vector<FrameWork> frame_batch(batch_frames);
            bool input_ended = false;
            while (!input_ended && (max_frames == 0 || frame_count < max_frames)) {
                std::size_t loaded_frames = 0;
                while (loaded_frames < batch_frames
                       && (max_frames == 0 || frame_count + loaded_frames < max_frames)) {
                    const auto frame_ok = timing_search_radius == 0
                        ? read_frame(
                            replay_input,
                            frame_batch[loaded_frames],
                            trailing_ci8_bytes,
                            next_sample_start)
                        : frame_reader.read(frame_batch[loaded_frames], trailing_ci8_bytes);
                    if (!frame_ok) {
                        input_ended = true;
                        break;
                    }
                    ++loaded_frames;
                    ++input_frame_count;
                }
                if (loaded_frames == 0) {
                    break;
                }

                const auto active_workers = std::min(worker_count, loaded_frames);
                std::vector<std::thread> workers;
                std::vector<std::exception_ptr> worker_errors(active_workers);
                workers.reserve(active_workers);
                for (std::size_t worker = 0; worker < active_workers; ++worker) {
                    workers.emplace_back([&, worker] {
                        try {
                            for (std::size_t frame = worker;
                                 frame < loaded_frames;
                                 frame += active_workers) {
                                process_frame(
                                    frame_batch[frame],
                                    normalization,
                                    reference,
                                    frequency_shift_hz,
                                    data_dd_options,
                                    data_dd_stats);
                            }
                        } catch (...) {
                            worker_errors[worker] = std::current_exception();
                        }
                    });
                }
                for (auto& worker : workers) {
                    worker.join();
                }
                for (const auto& error : worker_errors) {
                    if (error) {
                        std::rethrow_exception(error);
                    }
                }
                for (std::size_t frame = 0; frame < loaded_frames; ++frame) {
                    frame_residuals.observe(
                        frame_count + frame,
                        frame_batch[frame].data_cf32);
                    observe_system_information(frame_batch[frame]);
                    normalization_diagnostics.observe(
                        frame_count + frame,
                        frame_batch[frame].normalization_stats,
                        false,
                        0U,
                        0U,
                        0U,
                        0U,
                        0U);
                    source_carrier_residuals.observe(
                        frame_count + frame,
                        frame_batch[frame].data_cf32);
                    write_all(output, frame_batch[frame].data_cf32);
                }
                frame_count += loaded_frames;
            }
        }
        source_carrier_residuals.close();
        source_carrier_channels.close();
        timing_diagnostics.close();
        frame_residuals.close();
        normalization_diagnostics.close();

        std::cerr << "front_end=" << (auto_sync ? "auto" : "pinned") << "_pn945_"
                  << (equalizer == Equalizer::pn ? "pn_equalized" : "flat")
                  << "_c3780\n"
                  << "auto_sync=" << (auto_sync ? "true" : "false") << '\n'
                  << "sync_frames=" << sync_frames << '\n'
                  << "acquisition_frames=" << acquisition_frames << '\n'
                  << "sync_hit_threshold=" << sync_hit_threshold << '\n'
                  << "startup_buffered_samples=" << startup_buffered_samples << '\n'
                  << "acquisition_mean_metric=" << acquisition.mean_metric << '\n'
                  << "acquisition_max_metric=" << acquisition.max_metric << '\n'
                  << "acquisition_hit_count=" << acquisition.hit_count << '\n'
                  << "acquisition_observed_frames=" << acquisition.observed_frames << '\n'
                  << "acquisition_worker_count=" << acquisition.worker_count << '\n'
                  << "auto_phase_adjustment=" << auto_phase_adjustment << '\n'
                  << "timing_search_radius=" << timing_search_radius << '\n'
                  << "timing_search_threshold=" << timing_search_threshold << '\n'
                  << "timing_trajectory_interval_frames="
                  << timing_trajectory_interval_frames << '\n'
                  << "timing_trajectory_fit_points="
                  << timing_trajectory_fit_points << '\n'
                  << "timing_trajectory_max_innovation_samples="
                  << timing_trajectory_max_innovation_samples << '\n'
                  << "timing_trajectory_seed="
                  << (timing_trajectory_seed_path.empty()
                          ? "none"
                          : timing_trajectory_seed_path)
                  << '\n'
                  << "timing_trajectory_frame_index_offset="
                  << timing_trajectory_frame_index_offset << '\n'
                  << "timing_trajectory_offset_origin_samples="
                  << timing_trajectory_offset_origin_samples << '\n'
                  << "timing_trajectory_state_out="
                  << (timing_trajectory_state_out_path.empty()
                          ? "none"
                          : timing_trajectory_state_out_path)
                  << '\n'
                  << "timing_trajectory_state_out_frame="
                  << timing_trajectory_state_out_frame << '\n'
                  << "timing_trajectory_local_search="
                  << (timing_trajectory_local_search ? "true" : "false") << '\n'
                  << "timing_trajectory_local_search_transient="
                  << (timing_trajectory_local_search_transient ? "true" : "false") << '\n'
                  << "timing_trajectory_local_search_transient_ranges="
                  << timing_trajectory_local_search_transient_ranges.size() << '\n'
                  << "timing_trajectory_local_search_min_improvement="
                  << timing_trajectory_local_search_min_improvement << '\n'
                  << "timing_trajectory_local_search_scoped_min_improvement_rules="
                  << timing_trajectory_local_search_scoped_min_improvements.size()
                  << '\n'
                  << "timing_diagnostics="
                  << (timing_diagnostics_path.empty() ? "none" : timing_diagnostics_path)
                  << '\n'
                  << "coarse_cfo_hz=" << acquisition.coarse_cfo_hz << '\n'
                  << "coarse_cfo_valid="
                  << (acquisition.coarse_cfo_valid ? "true" : "false") << '\n'
                  << "residual_cfo_enabled="
                  << (auto_sync && estimate_residual_cfo ? "true" : "false") << '\n'
                  << "residual_cfo_hz=" << residual_cfo.cfo_hz << '\n'
                  << "residual_cfo_valid=" << (residual_cfo.valid ? "true" : "false")
                  << '\n'
                  << "residual_cfo_fit_r_squared=" << residual_cfo.fit_r_squared << '\n'
                  << "residual_cfo_used_frames=" << residual_cfo.used_frames << '\n'
                  << "residual_cfo_worker_count=" << residual_cfo.worker_count << '\n'
                  << "manual_frequency_shift_hz=" << manual_frequency_shift_hz << '\n'
                  << "automatic_frequency_shift_hz=" << automatic_frequency_shift_hz << '\n'
                  << "phase_offset_samples=" << phase_offset << '\n'
                  << "discarded_phase_samples=" << discarded_phase_bytes / 2 << '\n'
                  << "input_frames=" << input_frame_count << '\n'
                  << "frames=" << frame_count << '\n'
                  << "output_symbols=" << frame_count * dtmb::core::kC3780DataSymbols << '\n'
                  << "trailing_ci8_bytes=" << trailing_ci8_bytes << '\n'
                  << "worker_count=" << worker_count << '\n'
                  << "batch_frames=" << batch_frames << '\n'
                  << "equalizer=" << (equalizer == Equalizer::pn ? "pn" : "flat") << '\n'
                  << "pn_estimator="
                  << (pn_estimator == PnEstimator::wideband ? "wideband" : "compact")
                  << '\n'
                  << "pn_channel_taps=" << pn_channel_taps << '\n'
                  << "pn_wideband_block_frames=" << pn_wideband_block_frames << '\n'
                  << "pn_wideband_header_observation="
                  << pn_wideband_header_observation_name(
                         pn_wideband_header_observation)
                  << '\n'
                  << "pn_wideband_scale_estimator="
                  << pn_wideband_scale_estimator_name(pn_wideband_scale_estimator)
                  << '\n'
                  << "pn_wideband_max_span_symbols="
                  << pn_wideband_max_span_symbols << '\n'
                  << "pn_wideband_response_window_offset_adjust="
                  << pn_wideband_response_window_offset_adjust << '\n'
                  << "pn_wideband_body_channel="
                  << (pn_wideband_body_channel_midpoint ? "midpoint" : "current")
                  << '\n'
                  << "pn_wideband_model_count=" << wideband_model_count << '\n'
                  << "pn_wideband_span_min_symbols=" << wideband_model_stats.span_min << '\n'
                  << "pn_wideband_span_max_symbols=" << wideband_model_stats.span_max << '\n'
                  << "pn_wideband_span_mean_symbols="
                  << wideband_model_stats.mean_span() << '\n'
                  << "pn_wideband_significant_taps_min="
                  << wideband_model_stats.significant_min << '\n'
                  << "pn_wideband_significant_taps_max="
                  << wideband_model_stats.significant_max << '\n'
                  << "pn_wideband_significant_taps_mean="
                  << wideband_model_stats.mean_significant() << '\n'
                  << "pn_wideband_rotation_first_symbols="
                  << wideband_model_stats.rotation_first << '\n'
                  << "pn_wideband_rotation_last_symbols="
                  << wideband_model_stats.rotation_last << '\n'
                  << "pn_wideband_rotation_min_symbols="
                  << wideband_model_stats.rotation_min << '\n'
                  << "pn_wideband_rotation_max_symbols="
                  << wideband_model_stats.rotation_max << '\n'
                  << "pn_wideband_noise_variance_min="
                  << wideband_model_stats.noise_variance_min << '\n'
                  << "pn_wideband_noise_variance_max="
                  << wideband_model_stats.noise_variance_max << '\n'
                  << "pn_wideband_noise_variance_mean="
                  << wideband_model_stats.mean_noise_variance() << '\n'
                  << "pn_wideband_last_span_symbols=" << wideband_span_symbols << '\n'
                  << "pn_wideband_last_significant_taps=" << wideband_significant_taps << '\n'
                  << "pn_wideband_last_rotation_symbols=" << wideband_rotation_symbols << '\n'
                  << "pn_wideband_last_auto_noise_variance="
                  << wideband_auto_noise_variance << '\n'
                  << "pn_wideband_last_dc_offset_i=" << wideband_last_dc_offset.real() << '\n'
                  << "pn_wideband_last_dc_offset_q=" << wideband_last_dc_offset.imag() << '\n'
                  << "pn_wideband_diagnostics="
                  << (wideband_diagnostics_path.empty() ? "none" : wideband_diagnostics_path)
                  << '\n'
                  << "pn_wideband_frame_diagnostics="
                  << (wideband_frame_diagnostics_path.empty()
                          ? "none"
                          : wideband_frame_diagnostics_path)
                  << '\n'
                  << "pn_csi_demap=" << (pn_csi_demap ? "true" : "false") << '\n'
                  << "pn_csi_weights="
                  << (pn_csi_weights_path.empty() ? "none" : pn_csi_weights_path)
                  << '\n'
                  << "frame_residual_diagnostics="
                  << (frame_residual_diagnostics_path.empty()
                          ? "none"
                          : frame_residual_diagnostics_path)
                  << '\n'
                  << "normalization_diagnostics="
                  << (normalization_diagnostics_path.empty()
                          ? "none"
                          : normalization_diagnostics_path)
                  << '\n'
                  << "source_carrier_residual_rule_count="
                  << source_carrier_residual_rules.size() << '\n'
                  << "source_carrier_residual_diagnostics="
                  << (source_carrier_residual_diagnostics_path.empty()
                          ? "none"
                          : source_carrier_residual_diagnostics_path)
                  << '\n'
                  << "source_carrier_channel_rule_count="
                  << source_carrier_channel_rules.size() << '\n'
                  << "source_carrier_channel_diagnostics="
                  << (source_carrier_channel_diagnostics_path.empty()
                          ? "none"
                          : source_carrier_channel_diagnostics_path)
                  << '\n'
                  << "data_dd_refine_enabled="
                  << (data_dd_options.enabled ? "true" : "false") << '\n'
                  << "data_dd_max_relative_error="
                  << data_dd_options.max_relative_error << '\n'
                  << "data_dd_min_reliable_carriers="
                  << data_dd_options.min_reliable_carriers << '\n'
                  << "data_dd_max_axis_inner_fraction="
                  << data_dd_options.max_axis_inner_fraction << '\n'
                  << "data_dd_max_hard_bit_bias="
                  << data_dd_options.max_hard_bit_bias << '\n'
                  << "data_dd_observed_frames=" << data_dd_stats.observed_frames << '\n'
                  << "data_dd_refined_frames=" << data_dd_stats.refined_frames << '\n'
                  << "data_dd_rejected_frames=" << data_dd_stats.rejected_frames << '\n'
                  << "data_dd_reliable_carrier_rejected_frames="
                  << data_dd_stats.reliable_carrier_rejected_frames << '\n'
                  << "data_dd_axis_inner_rejected_frames="
                  << data_dd_stats.axis_inner_rejected_frames << '\n'
                  << "data_dd_hard_bit_bias_rejected_frames="
                  << data_dd_stats.hard_bit_bias_rejected_frames << '\n'
                  << "data_dd_reliable_carriers_min="
                  << (data_dd_stats.observed_frames == 0
                          ? 0
                          : data_dd_stats.min_reliable_carriers)
                  << '\n'
                  << "data_dd_reliable_carriers_max="
                  << data_dd_stats.max_reliable_carriers << '\n'
                  << "data_dd_reliable_carriers_mean="
                  << data_dd_stats.mean_reliable_carriers() << '\n'
                  << "remove_dc=" << (remove_dc ? "true" : "false") << '\n'
                  << "pn_mmse="
                  << (pn_mmse.automatic
                          ? "auto"
                          : pn_mmse.noise_variance >= 0.0F
                              ? std::to_string(pn_mmse.noise_variance)
                              : "off")
                  << '\n'
                  << "frequency_shift_hz=" << frequency_shift_hz << '\n'
                  << "first_pn_phase=" << first_pn_phase << '\n'
                  << "last_pn_phase=" << last_pn_phase << '\n'
                  << "normalization="
                  << (normalization == Normalization::system_info
                          ? "system-info"
                          : normalization == Normalization::qam64
                              ? "qam64"
                              : normalization == Normalization::qam64_amplitude
                                  ? "qam64-amplitude"
                                  : "none")
                  << '\n'
                  << "system_info_index="
                  << (system_info_auto
                          ? system_info_selector.selection().index
                          : system_info_index)
                  << '\n'
                  << "system_info_auto="
                  << (system_info_auto ? "true" : "false") << '\n'
                  << "system_info_auto_complete="
                  << (system_info_selector.selection().complete ? "true" : "false")
                  << '\n'
                  << "system_info_auto_locked="
                  << (system_info_selector.selection().locked ? "true" : "false")
                  << '\n'
                  << "system_info_auto_observations="
                  << system_info_selector.selection().observations << '\n'
                  << "system_info_auto_metric="
                  << system_info_selector.selection().metric << '\n'
                  << "system_info_auto_runner_up_index="
                  << system_info_selector.selection().runner_up_index << '\n'
                  << "system_info_auto_runner_up_metric="
                  << system_info_selector.selection().runner_up_metric << '\n'
                  << "system_info_auto_margin="
                  << system_info_selector.selection().margin << '\n';
        const auto timing_stats = frame_reader.stats();
        std::cerr << "timing_tracker_enabled="
                  << (timing_stats.enabled ? "true" : "false") << '\n'
                  << "timing_tracker_frames=" << timing_stats.frames << '\n'
                  << "timing_tracker_corrections=" << timing_stats.corrections << '\n'
                  << "timing_tracker_low_metric_fallbacks="
                  << timing_stats.low_metric_fallbacks << '\n'
                  << "timing_tracker_min_offset_samples=" << timing_stats.min_offset << '\n'
                  << "timing_tracker_max_offset_samples=" << timing_stats.max_offset << '\n'
                  << "timing_tracker_mean_abs_offset_samples="
                  << timing_stats.mean_abs_offset << '\n'
                  << "timing_tracker_min_metric=" << timing_stats.min_metric << '\n'
                  << "timing_tracker_last_metric=" << timing_stats.last_metric << '\n'
                  << "timing_tracker_min_best_metric="
                  << timing_stats.min_best_metric << '\n'
                  << "timing_tracker_last_best_metric="
                  << timing_stats.last_best_metric << '\n'
                  << "timing_trajectory_enabled="
                  << (timing_stats.trajectory_enabled ? "true" : "false") << '\n'
                  << "timing_trajectory_reacquisitions="
                  << timing_stats.trajectory_reacquisitions << '\n'
                  << "timing_trajectory_accepted_points="
                  << timing_stats.trajectory_accepted_points << '\n'
                  << "timing_trajectory_seed_points="
                  << timing_stats.trajectory_seed_points << '\n'
                  << "timing_trajectory_low_metric_fallbacks="
                  << timing_stats.trajectory_low_metric_fallbacks << '\n'
                  << "timing_trajectory_innovation_rejections="
                  << timing_stats.trajectory_innovation_rejections << '\n'
                  << "timing_trajectory_local_searches="
                  << timing_stats.trajectory_local_searches << '\n'
                  << "timing_trajectory_local_hits="
                  << timing_stats.trajectory_local_hits << '\n'
                  << "timing_trajectory_local_corrections="
                  << timing_stats.trajectory_local_corrections << '\n'
                  << "timing_trajectory_local_low_metric_fallbacks="
                  << timing_stats.trajectory_local_low_metric_fallbacks << '\n'
                  << "timing_trajectory_local_innovation_rejections="
                  << timing_stats.trajectory_local_innovation_rejections << '\n'
                  << "timing_trajectory_local_improvement_rejections="
                  << timing_stats.trajectory_local_improvement_rejections << '\n'
                  << "timing_trajectory_local_last_delta_samples="
                  << timing_stats.trajectory_local_last_delta << '\n'
                  << "timing_trajectory_local_max_abs_delta_samples="
                  << timing_stats.trajectory_local_max_abs_delta << '\n'
                  << "timing_trajectory_local_last_metric="
                  << timing_stats.trajectory_local_last_metric << '\n'
                  << "timing_trajectory_local_best_metric="
                  << timing_stats.trajectory_local_best_metric << '\n'
                  << "timing_trajectory_local_last_improvement="
                  << timing_stats.trajectory_local_last_improvement << '\n'
                  << "timing_trajectory_local_max_improvement="
                  << timing_stats.trajectory_local_max_improvement << '\n'
                  << "timing_trajectory_local_scoped_min_improvement_rules="
                  << timing_trajectory_local_search_scoped_min_improvements.size()
                  << '\n'
                  << "timing_trajectory_local_transient_ranges="
                  << timing_trajectory_local_search_transient_ranges.size() << '\n'
                  << "timing_trajectory_shadow_searches="
                  << timing_stats.trajectory_shadow_searches << '\n'
                  << "timing_trajectory_shadow_hits="
                  << timing_stats.trajectory_shadow_hits << '\n'
                  << "timing_trajectory_shadow_reanchors="
                  << timing_stats.trajectory_shadow_reanchors << '\n'
                  << "timing_trajectory_shadow_last_delta_samples="
                  << timing_stats.trajectory_shadow_last_delta << '\n'
                  << "timing_trajectory_shadow_max_abs_delta_samples="
                  << timing_stats.trajectory_shadow_max_abs_delta << '\n'
                  << "timing_trajectory_shadow_last_metric="
                  << timing_stats.trajectory_shadow_last_metric << '\n'
                  << "timing_trajectory_shadow_best_metric="
                  << timing_stats.trajectory_shadow_best_metric << '\n'
                  << "timing_trajectory_scheduled_slips="
                  << timing_stats.trajectory_scheduled_slips << '\n'
                  << "timing_trajectory_min_offset_samples="
                  << timing_stats.trajectory_min_offset << '\n'
                  << "timing_trajectory_max_offset_samples="
                  << timing_stats.trajectory_max_offset << '\n'
                  << "timing_trajectory_slope_samples_per_frame="
                  << timing_stats.trajectory_slope_samples_per_frame << '\n'
                  << "timing_trajectory_last_observed_offset_samples="
                  << timing_stats.trajectory_last_observed_offset << '\n'
                  << "timing_trajectory_last_innovation_samples="
                  << timing_stats.trajectory_last_innovation << '\n'
                  << "timing_trajectory_max_abs_innovation_samples="
                  << timing_stats.trajectory_max_abs_innovation << '\n'
                  << "timing_trajectory_min_metric="
                  << timing_stats.trajectory_min_metric << '\n'
                  << "timing_trajectory_last_metric="
                  << timing_stats.trajectory_last_metric << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_c3780_extract: " << exc.what() << '\n';
        return 1;
    }
}
