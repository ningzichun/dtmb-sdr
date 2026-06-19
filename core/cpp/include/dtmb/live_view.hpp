#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dtmb::core {

struct LiveMonitorIqTelemetry {
    double rms_dbfs = 0.0;
    std::uint64_t clip_count_i = 0;
    std::uint64_t clip_count_q = 0;
    double clip_ratio = 0.0;
};

struct LiveMonitorSpectrumTelemetry {
    std::size_t fft_size = 0;
    double start_hz = 0.0;
    double bin_width_hz = 0.0;
    double peak_frequency_hz = 0.0;
    double peak_dbfs = 0.0;
    std::vector<double> power_dbfs;
};

struct LiveMonitorPn945Telemetry {
    bool available = false;
    std::size_t phase_offset_symbols = 0;
    double mean_metric = 0.0;
    double max_metric = 0.0;
    std::uint64_t hit_count = 0;
    std::uint64_t observed_frames = 0;
    double coarse_cfo_hz = 0.0;
    bool coarse_cfo_valid = false;
    std::string reason;
};

struct LiveMonitorTelemetry {
    std::string schema;
    std::uint64_t report_index = 0;
    double elapsed_ms = 0.0;
    std::uint64_t input_samples = 0;
    std::uint64_t sample_rate_sps = 0;
    double sample_rate_msps = 0.0;
    std::uint64_t center_frequency_hz = 0;
    double center_frequency_mhz = 0.0;
    std::uint64_t bandwidth_hz = 0;
    double bandwidth_mhz = 0.0;
    LiveMonitorIqTelemetry iq;
    LiveMonitorSpectrumTelemetry spectrum;
    LiveMonitorPn945Telemetry pn945;
};

struct LiveViewOptions {
    double width = 960.0;
    double height = 540.0;
    double min_dbfs = -100.0;
    double max_dbfs = 0.0;
};

struct LiveViewStreamSummary {
    std::uint64_t line_count = 0;
    std::uint64_t telemetry_count = 0;
    std::uint64_t draw_packet_count = 0;
    std::uint64_t error_packet_count = 0;
    std::uint64_t ignored_line_count = 0;
    bool input_complete = true;
};

enum class LiveViewSceneInputKind {
    ignored,
    draw_packet,
    control,
};

struct LiveViewSceneApplyResult {
    LiveViewSceneInputKind kind = LiveViewSceneInputKind::ignored;
    std::string source_schema;
    std::string action;
};

struct LiveViewSceneState {
    std::uint64_t revision = 0;
    std::string channel_id;
    std::uint64_t center_frequency_hz = 0;
    std::string track_id;
    bool menu_open = false;
    std::vector<std::string> visible_debug_panels{"acquisition", "spectrum"};
    std::string latest_draw_packet_json;
    std::string latest_draw_verdict;
    std::string worst_draw_verdict;
};

struct LiveViewSceneStreamSummary {
    std::uint64_t line_count = 0;
    std::uint64_t control_count = 0;
    std::uint64_t draw_packet_count = 0;
    std::uint64_t scene_packet_count = 0;
    std::uint64_t error_packet_count = 0;
    std::uint64_t ignored_line_count = 0;
    bool input_complete = true;
};

void validate_live_view_options(const LiveViewOptions& options);

[[nodiscard]] std::optional<LiveMonitorTelemetry> parse_live_monitor_telemetry_json(
    std::string_view json_text);

[[nodiscard]] std::string live_monitor_draw_packet_json(
    const LiveMonitorTelemetry& telemetry,
    LiveViewOptions options = {});

[[nodiscard]] std::string live_view_error_packet_json(
    std::string_view message,
    std::uint64_t line_index = 0,
    LiveViewOptions options = {});

[[nodiscard]] std::string live_view_stream_end_packet_json(
    const LiveViewStreamSummary& summary);

[[nodiscard]] LiveViewSceneApplyResult apply_live_view_scene_input_json(
    LiveViewSceneState& state,
    std::string_view json_text);

[[nodiscard]] std::string live_view_scene_packet_json(
    const LiveViewSceneState& state,
    const LiveViewSceneApplyResult& cause);

[[nodiscard]] std::string live_view_scene_error_packet_json(
    std::string_view message,
    std::uint64_t line_index,
    const LiveViewSceneState& state);

[[nodiscard]] std::string live_view_scene_stream_end_packet_json(
    const LiveViewSceneStreamSummary& summary,
    const LiveViewSceneState& state);

}  // namespace dtmb::core
