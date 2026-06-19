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
#include <numbers>
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

struct Qam64ResidualPoint {
    float observed_real = 0.0F;
    float observed_imag = 0.0F;
    float decision_real = 0.0F;
    float decision_imag = 0.0F;
    float residual_real = 0.0F;
    float residual_imag = 0.0F;
    float residual_abs = 0.0F;
    float decision_abs = 0.0F;
    float observed_abs = 0.0F;
};

struct FrameWork {
    FrameWork()
        : header_ci8(dtmb::core::kPn945HeaderSymbols * 2),
          body_ci8(dtmb::core::kC3780FrameBodySymbols * 2),
          header_cf32(dtmb::core::kPn945HeaderSymbols * 2),
          body_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          spectrum_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          logical_cf32(dtmb::core::kC3780FrameBodySymbols * 2),
          data_cf32(dtmb::core::kC3780DataSymbols * 2) {}

    std::vector<std::int8_t> header_ci8;
    std::vector<std::int8_t> body_ci8;
    std::vector<float> header_cf32;
    std::vector<float> body_cf32;
    std::vector<float> spectrum_cf32;
    std::vector<float> logical_cf32;
    std::vector<float> data_cf32;
    std::size_t sample_start = 0;
};

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

struct TimingShadowResult {
    bool available = false;
    bool hit = false;
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
        << " [--timing-trajectory-local-search]"
        << " [--timing-trajectory-local-search-min-improvement X]"
        << " [--timing-diagnostics PATH]"
        << " [--no-residual-cfo]"
        << " [--frequency-shift-hz X]"
        << " [--workers N] [--batch-frames N]"
        << " [--equalizer flat|pn] [--pn-estimator compact|wideband]"
        << " [--pn-channel-taps N] [--pn-wideband-block-frames N]"
        << " [--pn-wideband-diagnostics PATH]"
        << " [--frame-residual-diagnostics PATH]"
        << " [--source-carrier-residual CARRIER:FIRST:LAST]"
        << " [--source-carrier-residual-diagnostics-out PATH]"
        << " [--pn-mmse off|auto|X]"
        << " [--remove-dc]"
        << " [--normalization system-info|qam64|none] [--system-info-index N]"
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
        bool trajectory_local_search,
        float trajectory_local_search_min_improvement,
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
          trajectory_local_search_(trajectory_local_search),
          trajectory_local_search_min_improvement_(
              trajectory_local_search_min_improvement),
          timing_diagnostics_(timing_diagnostics) {
        stats_.enabled = search_radius_ != 0 && trajectory_interval_frames_ == 0;
        stats_.trajectory_enabled = trajectory_interval_frames_ != 0;
        stats_.search_radius = search_radius_;
        stats_.hit_threshold = hit_threshold_;
        stats_.trajectory_interval_frames = trajectory_interval_frames_;
        stats_.trajectory_fit_points = trajectory_fit_points_;
        stats_.trajectory_local_search_enabled = trajectory_local_search_;
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
            + trajectory_slope_samples_per_frame_ * static_cast<double>(frame_index);
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
            static_cast<double>(frame_index),
            static_cast<double>(offset_samples),
        });
        while (trajectory_points_.size() > trajectory_fit_points_) {
            trajectory_points_.pop_front();
        }
        fit_trajectory();
        stats_.trajectory_accepted_points += 1;
        stats_.trajectory_slope_samples_per_frame = trajectory_slope_samples_per_frame_;
        stats_.trajectory_last_observed_offset = static_cast<double>(offset_samples);
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
        const auto predicted_start = selected_start;
        auto event_name = std::string_view("trajectory_predicted");
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
        const auto reacquire =
            (trajectory_frame_index_ % trajectory_interval_frames_) == 0;
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
                && improvement < trajectory_local_search_min_improvement_) {
                ++stats_.trajectory_local_improvement_rejections;
                event_name =
                    std::string_view("trajectory_local_improvement_rejected");
            } else {
                ++stats_.trajectory_local_hits;
                if (best_start != predicted_start) {
                    ++stats_.trajectory_local_corrections;
                    trajectory_local_offset_samples_ += delta;
                    local_correction = true;
                    event_name = std::string_view("trajectory_local_correction");
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
    bool trajectory_local_search_ = false;
    float trajectory_local_search_min_improvement_ = 0.0F;
    std::int64_t trajectory_local_offset_samples_ = 0;
    std::size_t trajectory_frame_index_ = 0;
    std::size_t previous_trajectory_start_ = 0;
    std::deque<TrajectoryPoint> trajectory_points_;
    double trajectory_intercept_samples_ = 0.0;
    double trajectory_slope_samples_per_frame_ = 0.0;
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
    if (text == "none") {
        return Normalization::none;
    }
    throw std::invalid_argument("normalization must be system-info, qam64, or none");
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

void normalize_data_from_system_info(
    std::span<const float> logical_symbols,
    std::span<const float> reference,
    std::span<float> output_data) {
    using Complex = std::complex<float>;
    auto numerator = Complex{0.0F, 0.0F};
    double denominator = 0.0;
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
    }
    if (denominator <= 1.0e-12) {
        throw std::runtime_error("system-information reference has zero power");
    }
    const auto gain = numerator / static_cast<float>(denominator);
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

void convert_ci8_to_cf32(
    std::span<const std::int8_t> input,
    std::span<float> output,
    std::size_t sample_start,
    float frequency_shift_hz,
    std::complex<float> dc_offset = {});

void process_frame(
    FrameWork& frame,
    Normalization normalization,
    std::span<const float> system_info_reference,
    float frequency_shift_hz) {
    convert_ci8_to_cf32(
        frame.body_ci8,
        frame.body_cf32,
        frame.sample_start + dtmb::core::kPn945HeaderSymbols,
        frequency_shift_hz);
    dtmb::core::c3780_extract_frame_symbols_cf32(frame.body_cf32, frame.logical_cf32);
    const auto data = std::span<const float>(
        frame.logical_cf32.data() + dtmb::core::kC3780SystemInfoSymbols * 2,
        dtmb::core::kC3780DataSymbols * 2);
    if (normalization == Normalization::system_info) {
        normalize_data_from_system_info(
            frame.logical_cf32,
            system_info_reference,
            frame.data_cf32);
    } else if (normalization == Normalization::qam64) {
        dtmb::core::qam64_normalize_cf32(data, frame.data_cf32);
    } else {
        std::copy(data.begin(), data.end(), frame.data_cf32.begin());
    }
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
    float noise_variance) {
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
    const auto data = std::span<const float>(
        frame.logical_cf32.data() + dtmb::core::kC3780SystemInfoSymbols * 2,
        dtmb::core::kC3780DataSymbols * 2);
    if (normalization == Normalization::system_info) {
        normalize_data_from_system_info(
            frame.logical_cf32,
            system_info_reference,
            frame.data_cf32);
    } else if (normalization == Normalization::qam64) {
        dtmb::core::qam64_normalize_cf32(data, frame.data_cf32);
    } else {
        std::copy(data.begin(), data.end(), frame.data_cf32.begin());
    }
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
    std::complex<float> dc_offset) {
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
        });
    dtmb::core::c3780_deinterleave_spectrum_cf32(
        frame.spectrum_cf32,
        frame.logical_cf32);
    const auto data = std::span<const float>(
        frame.logical_cf32.data() + dtmb::core::kC3780SystemInfoSymbols * 2,
        dtmb::core::kC3780DataSymbols * 2);
    if (normalization == Normalization::system_info) {
        normalize_data_from_system_info(
            frame.logical_cf32,
            system_info_reference,
            frame.data_cf32);
    } else if (normalization == Normalization::qam64) {
        dtmb::core::qam64_normalize_cf32(data, frame.data_cf32);
    } else {
        std::copy(data.begin(), data.end(), frame.data_cf32.begin());
    }
    return result;
}

dtmb::core::Pn945WidebandChannelModel build_wideband_model(
    std::deque<FrameWork>& frames,
    float frequency_shift_hz,
    std::complex<float> dc_offset) {
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
    return dtmb::core::build_pn945_wideband_channel_model_cf32(headers);
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

float nearest_qam64_level(float value) {
    auto best = kQam64Levels.front();
    auto best_distance = std::abs(value - best);
    for (const auto level : kQam64Levels) {
        const auto distance = std::abs(value - level);
        if (distance < best_distance) {
            best = level;
            best_distance = distance;
        }
    }
    return best;
}

Qam64ResidualPoint qam64_residual_point(float observed_real, float observed_imag) {
    Qam64ResidualPoint point;
    point.observed_real = observed_real;
    point.observed_imag = observed_imag;
    point.decision_real = nearest_qam64_level(observed_real);
    point.decision_imag = nearest_qam64_level(observed_imag);
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
    bool timing_trajectory_local_search = false;
    float timing_trajectory_local_search_min_improvement = 0.0F;
    std::int64_t auto_phase_adjustment = 0;
    float frequency_shift_hz = 0.0F;
    float sync_hit_threshold = 0.35F;
    float timing_search_threshold = 0.45F;
    std::size_t system_info_index = 23;
    auto normalization = Normalization::system_info;
    auto equalizer = Equalizer::flat;
    auto pn_estimator = PnEstimator::compact;
    auto pn_mmse = PnMmse{};
    bool remove_dc = false;
    bool auto_sync = false;
    bool estimate_residual_cfo = true;
    bool phase_offset_set = false;
    std::string wideband_diagnostics_path;
    std::string timing_diagnostics_path;
    std::string frame_residual_diagnostics_path;
    std::string source_carrier_residual_diagnostics_path;
    std::vector<SourceCarrierResidualRule> source_carrier_residual_rules;
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
            } else if (arg == "--timing-trajectory-local-search") {
                timing_trajectory_local_search = true;
            } else if (arg == "--timing-trajectory-local-search-min-improvement") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                timing_trajectory_local_search_min_improvement = parse_float(
                    argv[index],
                    "timing trajectory local-search minimum improvement");
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
            } else if (arg == "--pn-wideband-diagnostics") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                wideband_diagnostics_path = argv[index];
            } else if (arg == "--frame-residual-diagnostics") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                frame_residual_diagnostics_path = argv[index];
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
            } else if (arg == "--system-info-index") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                system_info_index = parse_size(argv[index], "system information index");
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
        if (timing_trajectory_local_search && timing_trajectory_interval_frames == 0) {
            throw std::invalid_argument(
                "--timing-trajectory-local-search requires "
                "--timing-trajectory-interval-frames");
        }
        if (timing_trajectory_local_search_min_improvement != 0.0F
            && !timing_trajectory_local_search) {
            throw std::invalid_argument(
                "--timing-trajectory-local-search-min-improvement requires "
                "--timing-trajectory-local-search");
        }
        if (pn_mmse.automatic && pn_estimator != PnEstimator::wideband) {
            throw std::invalid_argument("--pn-mmse auto requires --pn-estimator wideband");
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
        SourceCarrierResidualDiagnostics source_carrier_residuals(
            source_carrier_residual_diagnostics_path,
            source_carrier_residual_rules);
        TimingDiagnostics timing_diagnostics(timing_diagnostics_path);
        FrameResidualDiagnostics frame_residuals(frame_residual_diagnostics_path);

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
        TrackingFrameReader frame_reader(
            replay_input,
            phase_offset,
            timing_search_radius,
            timing_search_threshold,
            timing_trajectory_interval_frames,
            timing_trajectory_fit_points,
            timing_trajectory_max_innovation_samples,
            timing_trajectory_local_search,
            timing_trajectory_local_search_min_improvement,
            &timing_diagnostics);
        const auto reference = system_info_reference(system_info_index);

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
                    pn_mmse.noise_variance);
                if (frame_count == 0) {
                    first_pn_phase = result.pn_phase;
                }
                last_pn_phase = result.pn_phase;
                frame_residuals.observe(frame_count, current.data_cf32);
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
                    dc_offset);
                ++wideband_model_count;
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
                        << (wideband_model_count - 1) << ','
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
                                    dc_offset);
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
                    frame_residuals.observe(
                        output_frame_index,
                        frames[frame].data_cf32);
                    source_carrier_residuals.observe(
                        output_frame_index,
                        frames[frame].data_cf32);
                    write_all(output, frames[frame].data_cf32);
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
                                    frequency_shift_hz);
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
                    source_carrier_residuals.observe(
                        frame_count + frame,
                        frame_batch[frame].data_cf32);
                    write_all(output, frame_batch[frame].data_cf32);
                }
                frame_count += loaded_frames;
            }
        }
        source_carrier_residuals.close();
        timing_diagnostics.close();
        frame_residuals.close();

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
                  << "timing_trajectory_local_search="
                  << (timing_trajectory_local_search ? "true" : "false") << '\n'
                  << "timing_trajectory_local_search_min_improvement="
                  << timing_trajectory_local_search_min_improvement << '\n'
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
                  << "frame_residual_diagnostics="
                  << (frame_residual_diagnostics_path.empty()
                          ? "none"
                          : frame_residual_diagnostics_path)
                  << '\n'
                  << "source_carrier_residual_rule_count="
                  << source_carrier_residual_rules.size() << '\n'
                  << "source_carrier_residual_diagnostics="
                  << (source_carrier_residual_diagnostics_path.empty()
                          ? "none"
                          : source_carrier_residual_diagnostics_path)
                  << '\n'
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
                          : normalization == Normalization::qam64 ? "qam64" : "none")
                  << '\n'
                  << "system_info_index=" << system_info_index << '\n';
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
                  << "timing_trajectory_shadow_searches="
                  << timing_stats.trajectory_shadow_searches << '\n'
                  << "timing_trajectory_shadow_hits="
                  << timing_stats.trajectory_shadow_hits << '\n'
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
