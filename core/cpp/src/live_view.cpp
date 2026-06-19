#include "dtmb/live_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dtmb::core {
namespace {

struct JsonValue {
    enum class Type {
        null_value,
        bool_value,
        number_value,
        string_value,
        array_value,
        object_value,
    };

    Type type = Type::null_value;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    [[nodiscard]] JsonValue parse() {
        skip_whitespace();
        auto value = parse_value();
        skip_whitespace();
        if (position_ != text_.size()) {
            throw std::invalid_argument("unexpected trailing JSON data");
        }
        return value;
    }

private:
    [[nodiscard]] JsonValue parse_value() {
        skip_whitespace();
        if (position_ >= text_.size()) {
            throw std::invalid_argument("unexpected end of JSON input");
        }

        const auto ch = text_[position_];
        if (ch == '{') {
            return parse_object();
        }
        if (ch == '[') {
            return parse_array();
        }
        if (ch == '"') {
            JsonValue value;
            value.type = JsonValue::Type::string_value;
            value.string = parse_string();
            return value;
        }
        if (ch == '-' || (ch >= '0' && ch <= '9')) {
            return parse_number();
        }
        if (match_literal("true")) {
            JsonValue value;
            value.type = JsonValue::Type::bool_value;
            value.boolean = true;
            return value;
        }
        if (match_literal("false")) {
            JsonValue value;
            value.type = JsonValue::Type::bool_value;
            value.boolean = false;
            return value;
        }
        if (match_literal("null")) {
            JsonValue value;
            value.type = JsonValue::Type::null_value;
            return value;
        }
        throw std::invalid_argument("invalid JSON value");
    }

    [[nodiscard]] JsonValue parse_object() {
        JsonValue value;
        value.type = JsonValue::Type::object_value;
        expect('{');
        skip_whitespace();
        if (consume('}')) {
            return value;
        }

        while (true) {
            skip_whitespace();
            if (position_ >= text_.size() || text_[position_] != '"') {
                throw std::invalid_argument("expected JSON object key");
            }
            auto key = parse_string();
            skip_whitespace();
            expect(':');
            value.object.emplace_back(std::move(key), parse_value());
            skip_whitespace();
            if (consume('}')) {
                return value;
            }
            expect(',');
        }
    }

    [[nodiscard]] JsonValue parse_array() {
        JsonValue value;
        value.type = JsonValue::Type::array_value;
        expect('[');
        skip_whitespace();
        if (consume(']')) {
            return value;
        }

        while (true) {
            value.array.push_back(parse_value());
            skip_whitespace();
            if (consume(']')) {
                return value;
            }
            expect(',');
        }
    }

    [[nodiscard]] std::string parse_string() {
        expect('"');
        std::string output;
        while (position_ < text_.size()) {
            const char ch = text_[position_++];
            if (ch == '"') {
                return output;
            }
            if (static_cast<unsigned char>(ch) < 0x20U) {
                throw std::invalid_argument("unescaped control character in JSON string");
            }
            if (ch != '\\') {
                output.push_back(ch);
                continue;
            }
            if (position_ >= text_.size()) {
                throw std::invalid_argument("unterminated JSON string escape");
            }
            const char escaped = text_[position_++];
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                output.push_back(escaped);
                break;
            case 'b':
                output.push_back('\b');
                break;
            case 'f':
                output.push_back('\f');
                break;
            case 'n':
                output.push_back('\n');
                break;
            case 'r':
                output.push_back('\r');
                break;
            case 't':
                output.push_back('\t');
                break;
            case 'u':
                output.push_back(parse_ascii_unicode_escape());
                break;
            default:
                throw std::invalid_argument("invalid JSON string escape");
            }
        }
        throw std::invalid_argument("unterminated JSON string");
    }

    [[nodiscard]] char parse_ascii_unicode_escape() {
        if (position_ + 4U > text_.size()) {
            throw std::invalid_argument("short JSON unicode escape");
        }
        unsigned int codepoint = 0;
        for (std::size_t count = 0; count < 4U; ++count) {
            const char ch = text_[position_++];
            codepoint <<= 4U;
            if (ch >= '0' && ch <= '9') {
                codepoint += static_cast<unsigned int>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                codepoint += static_cast<unsigned int>(ch - 'a' + 10);
            } else if (ch >= 'A' && ch <= 'F') {
                codepoint += static_cast<unsigned int>(ch - 'A' + 10);
            } else {
                throw std::invalid_argument("invalid JSON unicode escape");
            }
        }
        return codepoint <= 0x7FU ? static_cast<char>(codepoint) : '?';
    }

    [[nodiscard]] JsonValue parse_number() {
        const auto start = position_;
        if (text_[position_] == '-') {
            ++position_;
        }
        consume_digits();
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            consume_digits();
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            consume_digits();
        }

        JsonValue value;
        value.type = JsonValue::Type::number_value;
        const std::string number_text(text_.substr(start, position_ - start));
        std::istringstream parser(number_text);
        parser.imbue(std::locale::classic());
        parser >> value.number;
        char extra = '\0';
        if (parser.fail() || (parser >> extra)) {
            throw std::invalid_argument("invalid JSON number");
        }
        return value;
    }

    void consume_digits() {
        const auto start = position_;
        while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
            ++position_;
        }
        if (position_ == start) {
            throw std::invalid_argument("expected JSON number digit");
        }
    }

    void skip_whitespace() {
        while (position_ < text_.size()) {
            const char ch = text_[position_];
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
                return;
            }
            ++position_;
        }
    }

    [[nodiscard]] bool consume(char expected) {
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            throw std::invalid_argument("unexpected JSON token");
        }
    }

    [[nodiscard]] bool match_literal(std::string_view literal) {
        if (text_.substr(position_, literal.size()) == literal) {
            position_ += literal.size();
            return true;
        }
        return false;
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

[[nodiscard]] bool is_blank(std::string_view text) {
    return std::all_of(text.begin(), text.end(), [](char ch) {
        return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
    });
}

[[nodiscard]] const JsonValue* field(const JsonValue& value, std::string_view key) {
    if (value.type != JsonValue::Type::object_value) {
        return nullptr;
    }
    for (const auto& item : value.object) {
        if (item.first == key) {
            return &item.second;
        }
    }
    return nullptr;
}

[[nodiscard]] std::optional<std::string> string_field(
    const JsonValue& value,
    std::string_view key) {
    const auto* item = field(value, key);
    if (item == nullptr || item->type != JsonValue::Type::string_value) {
        return std::nullopt;
    }
    return item->string;
}

[[nodiscard]] std::optional<double> number_field(
    const JsonValue& value,
    std::string_view key) {
    const auto* item = field(value, key);
    if (item == nullptr || item->type != JsonValue::Type::number_value) {
        return std::nullopt;
    }
    return item->number;
}

[[nodiscard]] std::optional<bool> bool_field(
    const JsonValue& value,
    std::string_view key) {
    const auto* item = field(value, key);
    if (item == nullptr || item->type != JsonValue::Type::bool_value) {
        return std::nullopt;
    }
    return item->boolean;
}

[[nodiscard]] std::string required_string_field(
    const JsonValue& value,
    std::string_view key) {
    const auto parsed = string_field(value, key);
    if (!parsed.has_value() || parsed->empty()) {
        throw std::invalid_argument("missing or invalid string field: " + std::string(key));
    }
    return *parsed;
}

[[nodiscard]] bool required_bool_field(
    const JsonValue& value,
    std::string_view key) {
    const auto parsed = bool_field(value, key);
    if (!parsed.has_value()) {
        throw std::invalid_argument("missing or invalid boolean field: " + std::string(key));
    }
    return *parsed;
}

[[nodiscard]] std::uint64_t unsigned_field(
    const JsonValue& value,
    std::string_view key,
    std::uint64_t fallback = 0) {
    const auto parsed = number_field(value, key);
    if (!parsed.has_value() || !std::isfinite(*parsed) || *parsed < 0.0) {
        return fallback;
    }
    constexpr auto max_exact_integer = 9007199254740992.0;
    if (*parsed > max_exact_integer) {
        return fallback;
    }
    return static_cast<std::uint64_t>(*parsed);
}

[[nodiscard]] std::optional<std::uint64_t> optional_unsigned_field(
    const JsonValue& value,
    std::string_view key) {
    const auto* item = field(value, key);
    if (item == nullptr) {
        return std::nullopt;
    }
    if (item->type != JsonValue::Type::number_value
        || !std::isfinite(item->number)
        || item->number < 0.0
        || std::floor(item->number) != item->number) {
        throw std::invalid_argument("invalid unsigned integer field: " + std::string(key));
    }
    constexpr auto max_exact_integer = 9007199254740992.0;
    if (item->number > max_exact_integer) {
        throw std::invalid_argument("unsigned integer field is too large: " + std::string(key));
    }
    return static_cast<std::uint64_t>(item->number);
}

[[nodiscard]] double numeric_field(
    const JsonValue& value,
    std::string_view key,
    double fallback = 0.0) {
    const auto parsed = number_field(value, key);
    return parsed.has_value() && std::isfinite(*parsed) ? *parsed : fallback;
}

void append_json_string(std::ostream& output, std::string_view value) {
    output << '"';
    for (const char ch : value) {
        switch (ch) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20U) {
                output << "\\u"
                       << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(static_cast<unsigned char>(ch))
                       << std::dec << std::setfill(' ');
            } else {
                output << ch;
            }
        }
    }
    output << '"';
}

void append_string_array(std::ostream& output, const std::vector<std::string>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        append_json_string(output, values[index]);
    }
    output << ']';
}

void append_fixed(std::ostream& output, double value, int precision = 6) {
    if (!std::isfinite(value)) {
        output << "null";
        return;
    }
    output << std::fixed << std::setprecision(precision) << value;
}

void append_rect(std::ostream& output, double x, double y, double width, double height) {
    output << '[';
    append_fixed(output, x);
    output << ',';
    append_fixed(output, y);
    output << ',';
    append_fixed(output, width);
    output << ',';
    append_fixed(output, height);
    output << ']';
}

void append_point(std::ostream& output, double x, double y) {
    output << '[';
    append_fixed(output, x);
    output << ',';
    append_fixed(output, y);
    output << ']';
}

void append_rgba(std::ostream& output, double r, double g, double b, double a) {
    output << '[';
    append_fixed(output, r);
    output << ',';
    append_fixed(output, g);
    output << ',';
    append_fixed(output, b);
    output << ',';
    append_fixed(output, a);
    output << ']';
}

void append_panel(
    std::ostream& output,
    std::string_view id,
    std::string_view title,
    double x,
    double y,
    double width,
    double height) {
    output << "{\"id\":";
    append_json_string(output, id);
    output << ",\"title\":";
    append_json_string(output, title);
    output << ",\"rect\":";
    append_rect(output, x, y, width, height);
    output << '}';
}

void append_rect_draw(
    std::ostream& output,
    std::string_view id,
    std::string_view panel,
    double r,
    double g,
    double b) {
    output << "{\"op\":\"rect\",\"id\":";
    append_json_string(output, id);
    output << ",\"panel\":";
    append_json_string(output, panel);
    output << ",\"space\":\"panel-normalized\",\"primitive\":\"TRIANGLE_STRIP\""
           << ",\"shader\":\"solid_color_gles2\"";
    output << ",\"rect\":";
    append_rect(output, 0.0, 0.0, 1.0, 1.0);
    output << ",\"rgba\":";
    append_rgba(output, r, g, b, 1.0);
    output << '}';
}

void append_text_draw(
    std::ostream& output,
    std::string_view id,
    std::string_view panel,
    double x,
    double y,
    std::string_view text,
    double size = 14.0) {
    output << "{\"op\":\"text\",\"id\":";
    append_json_string(output, id);
    output << ",\"panel\":";
    append_json_string(output, panel);
    output << ",\"space\":\"panel-normalized\",\"renderer\":\"host-text\"";
    output << ",\"anchor\":";
    append_point(output, x, y);
    output << ",\"size\":";
    append_fixed(output, size);
    output << ",\"rgba\":";
    append_rgba(output, 0.88, 0.92, 0.96, 1.0);
    output << ",\"text\":";
    append_json_string(output, text);
    output << '}';
}

[[nodiscard]] double clipped_unit(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return std::clamp(value, 0.0, 1.0);
}

void append_spectrum_polyline(
    std::ostream& output,
    const LiveMonitorSpectrumTelemetry& spectrum,
    const LiveViewOptions& options) {
    output << "{\"op\":\"polyline\",\"id\":\"spectrum.power\",\"panel\":\"spectrum\""
           << ",\"space\":\"panel-normalized\",\"primitive\":\"LINE_STRIP\""
           << ",\"shader\":\"polyline_gles2\",\"rgba\":";
    append_rgba(output, 0.14, 0.62, 0.92, 1.0);
    output << ",\"width\":";
    append_fixed(output, 2.0);
    output << ",\"vertices\":[";

    const auto count = spectrum.power_dbfs.size();
    const auto range = options.max_dbfs - options.min_dbfs;
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) {
            output << ',';
        }
        const auto x = count <= 1U
            ? 0.0
            : static_cast<double>(index) / static_cast<double>(count - 1U);
        const auto normalized = range > 0.0
            ? clipped_unit((spectrum.power_dbfs[index] - options.min_dbfs) / range)
            : 0.0;
        const auto y = 1.0 - normalized;
        output << '[';
        append_fixed(output, x);
        output << ',';
        append_fixed(output, y);
        output << ']';
    }
    output << "]}";
}

[[nodiscard]] double mhz_from_hz(std::uint64_t hz, double fallback_mhz) {
    return hz != 0 ? static_cast<double>(hz) / 1.0e6 : fallback_mhz;
}

[[nodiscard]] std::string rf_label(const LiveMonitorTelemetry& telemetry) {
    std::ostringstream label;
    label.imbue(std::locale::classic());
    label << std::fixed << std::setprecision(3)
          << "RF " << mhz_from_hz(telemetry.center_frequency_hz, telemetry.center_frequency_mhz)
          << " MHz, " << (telemetry.sample_rate_msps != 0.0
              ? telemetry.sample_rate_msps
              : static_cast<double>(telemetry.sample_rate_sps) / 1.0e6)
          << " Msps";
    if (telemetry.bandwidth_hz != 0 || telemetry.bandwidth_mhz != 0.0) {
        label << ", BW " << mhz_from_hz(telemetry.bandwidth_hz, telemetry.bandwidth_mhz)
              << " MHz";
    }
    return label.str();
}

[[nodiscard]] std::string spectrum_label(const LiveMonitorTelemetry& telemetry) {
    std::ostringstream label;
    label.imbue(std::locale::classic());
    label << std::fixed << std::setprecision(3)
          << "Spectrum peak "
          << (telemetry.spectrum.peak_frequency_hz / 1.0e6)
          << " MHz @ " << telemetry.spectrum.peak_dbfs << " dBFS";
    return label.str();
}

[[nodiscard]] std::string iq_label(const LiveMonitorTelemetry& telemetry) {
    std::ostringstream label;
    label.imbue(std::locale::classic());
    label << std::fixed << std::setprecision(3)
          << "IQ RMS " << telemetry.iq.rms_dbfs << " dBFS, clip ratio "
          << std::setprecision(6) << telemetry.iq.clip_ratio;
    return label.str();
}

[[nodiscard]] std::string pn_label(const LiveMonitorTelemetry& telemetry) {
    std::ostringstream label;
    label.imbue(std::locale::classic());
    if (!telemetry.pn945.available) {
        label << "PN945 unavailable";
        if (!telemetry.pn945.reason.empty()) {
            label << ": " << telemetry.pn945.reason;
        }
        return label.str();
    }

    label << std::fixed << std::setprecision(6)
          << "PN945 mean " << telemetry.pn945.mean_metric
          << ", max " << telemetry.pn945.max_metric
          << ", hits " << telemetry.pn945.hit_count << '/'
          << telemetry.pn945.observed_frames
          << ", phase " << telemetry.pn945.phase_offset_symbols;
    if (telemetry.pn945.coarse_cfo_valid) {
        label << std::setprecision(3) << ", CFO " << telemetry.pn945.coarse_cfo_hz << " Hz";
    }
    return label.str();
}

constexpr std::string_view kSolidVertexShader = R"(attribute vec2 a_position;
uniform vec2 u_panel_origin;
uniform vec2 u_panel_size;
void main() {
  vec2 p = u_panel_origin + a_position * u_panel_size;
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
)";

constexpr std::string_view kSolidFragmentShader = R"(precision mediump float;
uniform vec4 u_color;
void main() {
  gl_FragColor = u_color;
}
)";

void append_shaders(std::ostream& output) {
    output << "\"shaders\":{\"solid_color_gles2\":{\"language\":\"GLES2/WebGL1\""
           << ",\"primitive_modes\":[\"TRIANGLE_STRIP\"]"
           << ",\"attributes\":[\"a_position\"]"
           << ",\"uniforms\":[\"u_panel_origin\",\"u_panel_size\",\"u_color\"]"
           << ",\"vertex\":";
    append_json_string(output, kSolidVertexShader);
    output << ",\"fragment\":";
    append_json_string(output, kSolidFragmentShader);
    output << "},\"polyline_gles2\":{\"language\":\"GLES2/WebGL1\""
           << ",\"primitive_modes\":[\"LINE_STRIP\"]"
           << ",\"attributes\":[\"a_position\"]"
           << ",\"uniforms\":[\"u_panel_origin\",\"u_panel_size\",\"u_color\"]"
           << ",\"vertex\":";
    append_json_string(output, kSolidVertexShader);
    output << ",\"fragment\":";
    append_json_string(output, kSolidFragmentShader);
    output << "}}";
}

void append_render_contract(std::ostream& output) {
    output << "\"render_contract\":{\"api\":\"GL/WebGL\""
           << ",\"version\":1"
           << ",\"viewport_space\":\"normalized top-left\""
           << ",\"panel_space\":\"normalized top-left\""
           << ",\"blend\":\"source-over\""
           << ",\"operations\":{\"rect\":{\"primitive\":\"TRIANGLE_STRIP\""
           << ",\"shader\":\"solid_color_gles2\"}"
           << ",\"polyline\":{\"primitive\":\"LINE_STRIP\""
           << ",\"shader\":\"polyline_gles2\"}"
           << ",\"text\":{\"renderer\":\"host-text\"}}}";
}

void append_layout_and_panels(std::ostream& output, const LiveViewOptions& options) {
    output << "\"layout\":{\"width\":";
    append_fixed(output, options.width);
    output << ",\"height\":";
    append_fixed(output, options.height);
    output << ",\"coordinate_system\":\"normalized top-left panels\"}";
    output << ",\"panels\":[";
    append_panel(output, "spectrum", "Spectrum", 0.0, 0.0, 1.0, 0.68);
    output << ',';
    append_panel(output, "acquisition", "Acquisition", 0.0, 0.68, 1.0, 0.32);
    output << ']';
}

void set_debug_panel_visibility(
    LiveViewSceneState& state,
    std::string_view panel_id,
    bool visible) {
    const auto match = std::find(
        state.visible_debug_panels.begin(),
        state.visible_debug_panels.end(),
        panel_id);
    if (visible && match == state.visible_debug_panels.end()) {
        state.visible_debug_panels.emplace_back(panel_id);
        std::sort(state.visible_debug_panels.begin(), state.visible_debug_panels.end());
    } else if (!visible && match != state.visible_debug_panels.end()) {
        state.visible_debug_panels.erase(match);
    }
}

[[nodiscard]] std::string_view scene_kind_name(LiveViewSceneInputKind kind) {
    switch (kind) {
    case LiveViewSceneInputKind::ignored:
        return "ignored";
    case LiveViewSceneInputKind::draw_packet:
        return "draw_packet";
    case LiveViewSceneInputKind::control:
        return "control";
    }
    return "ignored";
}

[[nodiscard]] std::string_view scene_verdict(const LiveViewSceneState& state) {
    if (state.latest_draw_packet_json.empty()) {
        return "degraded";
    }
    if (state.latest_draw_verdict == "ok"
        || state.latest_draw_verdict == "degraded"
        || state.latest_draw_verdict == "error") {
        return state.latest_draw_verdict;
    }
    return "degraded";
}

[[nodiscard]] int draw_verdict_severity(std::string_view verdict) {
    if (verdict == "error") {
        return 2;
    }
    if (verdict == "degraded") {
        return 1;
    }
    if (verdict == "ok") {
        return 0;
    }
    return -1;
}

void retain_worst_draw_verdict(LiveViewSceneState& state, std::string_view verdict) {
    if (draw_verdict_severity(verdict) > draw_verdict_severity(state.worst_draw_verdict)) {
        state.worst_draw_verdict = verdict;
    }
}

[[nodiscard]] std::string_view terminal_draw_verdict(const LiveViewSceneState& state) {
    return draw_verdict_severity(state.worst_draw_verdict) >= 0
        ? std::string_view(state.worst_draw_verdict)
        : scene_verdict(state);
}

}  // namespace

void validate_live_view_options(const LiveViewOptions& options) {
    if (!std::isfinite(options.width) || options.width <= 0.0) {
        throw std::invalid_argument("live view width must be finite and positive");
    }
    if (!std::isfinite(options.height) || options.height <= 0.0) {
        throw std::invalid_argument("live view height must be finite and positive");
    }
    if (!std::isfinite(options.min_dbfs) || !std::isfinite(options.max_dbfs)) {
        throw std::invalid_argument("live view dBFS bounds must be finite");
    }
    if (!(options.max_dbfs > options.min_dbfs)) {
        throw std::invalid_argument("live view maximum dBFS must be greater than minimum dBFS");
    }
}

std::optional<LiveMonitorTelemetry> parse_live_monitor_telemetry_json(
    std::string_view json_text) {
    if (is_blank(json_text)) {
        return std::nullopt;
    }

    auto root = JsonParser(json_text).parse();
    const auto schema = string_field(root, "schema");
    if (!schema.has_value() || *schema != "dtmb.live_monitor.v1") {
        return std::nullopt;
    }

    LiveMonitorTelemetry telemetry;
    telemetry.schema = *schema;
    telemetry.report_index = unsigned_field(root, "report_index");
    telemetry.elapsed_ms = numeric_field(root, "elapsed_ms");
    telemetry.input_samples = unsigned_field(root, "input_samples");
    telemetry.sample_rate_sps = unsigned_field(root, "sample_rate_sps");
    telemetry.sample_rate_msps = numeric_field(root, "sample_rate_msps");
    telemetry.center_frequency_hz = unsigned_field(root, "center_frequency_hz");
    telemetry.center_frequency_mhz = numeric_field(root, "center_frequency_mhz");
    telemetry.bandwidth_hz = unsigned_field(root, "bandwidth_hz");
    telemetry.bandwidth_mhz = numeric_field(root, "bandwidth_mhz");

    if (const auto* iq = field(root, "iq"); iq != nullptr) {
        telemetry.iq.rms_dbfs = numeric_field(*iq, "rms_dbfs");
        telemetry.iq.clip_count_i = unsigned_field(*iq, "clip_count_i");
        telemetry.iq.clip_count_q = unsigned_field(*iq, "clip_count_q");
        telemetry.iq.clip_ratio = numeric_field(*iq, "clip_ratio");
    }

    if (const auto* spectrum = field(root, "spectrum"); spectrum != nullptr) {
        telemetry.spectrum.fft_size = static_cast<std::size_t>(
            unsigned_field(*spectrum, "fft_size"));
        telemetry.spectrum.start_hz = numeric_field(*spectrum, "start_hz");
        telemetry.spectrum.bin_width_hz = numeric_field(*spectrum, "bin_width_hz");
        telemetry.spectrum.peak_frequency_hz = numeric_field(
            *spectrum,
            "peak_frequency_hz");
        telemetry.spectrum.peak_dbfs = numeric_field(*spectrum, "peak_dbfs");
        if (const auto* power = field(*spectrum, "power_dbfs");
            power != nullptr && power->type == JsonValue::Type::array_value) {
            telemetry.spectrum.power_dbfs.reserve(power->array.size());
            for (const auto& item : power->array) {
                if (item.type == JsonValue::Type::number_value && std::isfinite(item.number)) {
                    telemetry.spectrum.power_dbfs.push_back(item.number);
                }
            }
        }
    }

    if (const auto* pn = field(root, "pn945"); pn != nullptr) {
        telemetry.pn945.available = bool_field(*pn, "available").value_or(false);
        telemetry.pn945.phase_offset_symbols = static_cast<std::size_t>(
            unsigned_field(*pn, "phase_offset_symbols"));
        telemetry.pn945.mean_metric = numeric_field(*pn, "mean_metric");
        telemetry.pn945.max_metric = numeric_field(*pn, "max_metric");
        telemetry.pn945.hit_count = unsigned_field(*pn, "hit_count");
        telemetry.pn945.observed_frames = unsigned_field(*pn, "observed_frames");
        telemetry.pn945.coarse_cfo_hz = numeric_field(*pn, "coarse_cfo_hz");
        telemetry.pn945.coarse_cfo_valid = bool_field(*pn, "coarse_cfo_valid").value_or(false);
        telemetry.pn945.reason = string_field(*pn, "reason").value_or("");
    }

    return telemetry;
}

std::string live_monitor_draw_packet_json(
    const LiveMonitorTelemetry& telemetry,
    LiveViewOptions options) {
    validate_live_view_options(options);

    const bool has_spectrum = !telemetry.spectrum.power_dbfs.empty();
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"schema\":\"dtmb.live_view.draw_packet.v1\""
           << ",\"event\":\"draw\""
           << ",\"source_schema\":\"dtmb.live_monitor.v1\""
           << ",\"verdict\":\"" << (has_spectrum ? "ok" : "degraded") << '"'
           << ",\"report_index\":" << telemetry.report_index
           << ",\"elapsed_ms\":";
    append_fixed(output, telemetry.elapsed_ms, 3);
    output << ',';
    append_layout_and_panels(output, options);
    output << ",\"scales\":{\"spectrum_dbfs\":[";
    append_fixed(output, options.min_dbfs);
    output << ',';
    append_fixed(output, options.max_dbfs);
    output << "]}";
    output << ',';
    append_render_contract(output);
    output << ",\"draw\":[";
    append_rect_draw(output, "spectrum.background", "spectrum", 0.035, 0.043, 0.052);
    output << ',';
    append_rect_draw(output, "acquisition.background", "acquisition", 0.055, 0.063, 0.072);
    output << ',';
    append_spectrum_polyline(output, telemetry.spectrum, options);
    output << ',';
    append_text_draw(output, "label.rf", "acquisition", 0.030, 0.210, rf_label(telemetry));
    output << ',';
    append_text_draw(
        output,
        "label.spectrum",
        "acquisition",
        0.030,
        0.440,
        has_spectrum ? spectrum_label(telemetry) : "Spectrum unavailable");
    output << ',';
    append_text_draw(output, "label.iq", "acquisition", 0.030, 0.670, iq_label(telemetry));
    output << ',';
    append_text_draw(output, "label.pn945", "acquisition", 0.500, 0.440, pn_label(telemetry));
    output << "],";
    append_shaders(output);
    output << "}";
    return output.str();
}

std::string live_view_error_packet_json(
    std::string_view message,
    std::uint64_t line_index,
    LiveViewOptions options) {
    validate_live_view_options(options);

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"schema\":\"dtmb.live_view.draw_packet.v1\""
           << ",\"event\":\"error\""
           << ",\"source_schema\":\"dtmb.live_monitor.v1\""
           << ",\"verdict\":\"error\""
           << ",\"line_index\":" << line_index << ',';
    append_layout_and_panels(output, options);
    output << ',';
    append_render_contract(output);
    output << ",\"draw\":[";
    append_rect_draw(output, "acquisition.background", "acquisition", 0.055, 0.063, 0.072);
    output << ',';
    append_text_draw(output, "label.error", "acquisition", 0.030, 0.300, message);
    output << "],\"errors\":[";
    append_json_string(output, message);
    output << "],";
    append_shaders(output);
    output << "}";
    return output.str();
}

std::string live_view_stream_end_packet_json(const LiveViewStreamSummary& summary) {
    const std::string_view verdict = !summary.input_complete || summary.draw_packet_count == 0
        ? "error"
        : summary.error_packet_count != 0 ? "degraded" : "ok";

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"schema\":\"dtmb.live_view.stream_end.v1\""
           << ",\"event\":\"end\""
           << ",\"source_schema\":\"dtmb.live_monitor.v1\""
           << ",\"verdict\":\"" << verdict << '"'
           << ",\"input_complete\":" << (summary.input_complete ? "true" : "false")
           << ",\"line_count\":" << summary.line_count
           << ",\"telemetry_count\":" << summary.telemetry_count
           << ",\"draw_packet_count\":" << summary.draw_packet_count
           << ",\"error_packet_count\":" << summary.error_packet_count
           << ",\"ignored_line_count\":" << summary.ignored_line_count
           << '}';
    return output.str();
}

LiveViewSceneApplyResult apply_live_view_scene_input_json(
    LiveViewSceneState& state,
    std::string_view json_text) {
    if (is_blank(json_text)) {
        return {};
    }

    const auto root = JsonParser(json_text).parse();
    const auto schema = string_field(root, "schema");
    if (!schema.has_value()) {
        return {};
    }

    if (*schema == "dtmb.live_view.draw_packet.v1") {
        const auto event = required_string_field(root, "event");
        if (event != "draw" && event != "error") {
            throw std::invalid_argument("unsupported draw-packet event: " + event);
        }
        const auto verdict = required_string_field(root, "verdict");
        if (verdict != "ok" && verdict != "degraded" && verdict != "error") {
            throw std::invalid_argument("unsupported draw-packet verdict: " + verdict);
        }
        const auto* draw = field(root, "draw");
        if (draw == nullptr || draw->type != JsonValue::Type::array_value) {
            throw std::invalid_argument("missing or invalid draw-packet draw array");
        }
        state.latest_draw_packet_json = std::string(json_text);
        state.latest_draw_verdict = verdict;
        retain_worst_draw_verdict(state, verdict);
        ++state.revision;
        return {LiveViewSceneInputKind::draw_packet, *schema, event};
    }

    if (*schema != "dtmb.live_view.control.v1") {
        return {};
    }

    auto next = state;
    const auto event = required_string_field(root, "event");
    if (event != "control") {
        throw std::invalid_argument("unsupported live-view control event: " + event);
    }
    const auto action = required_string_field(root, "action");
    if (action == "channel.select") {
        next.channel_id = required_string_field(root, "channel_id");
        next.center_frequency_hz = optional_unsigned_field(root, "center_frequency_hz").value_or(0);
    } else if (action == "track.select") {
        next.track_id = required_string_field(root, "track_id");
    } else if (action == "menu.set") {
        next.menu_open = required_bool_field(root, "open");
    } else if (action == "menu.toggle") {
        next.menu_open = !next.menu_open;
    } else if (action == "debug_panel.set") {
        set_debug_panel_visibility(
            next,
            required_string_field(root, "panel_id"),
            required_bool_field(root, "visible"));
    } else if (action == "debug_panel.toggle") {
        const auto panel_id = required_string_field(root, "panel_id");
        const auto visible = std::find(
            next.visible_debug_panels.begin(),
            next.visible_debug_panels.end(),
            panel_id) == next.visible_debug_panels.end();
        set_debug_panel_visibility(next, panel_id, visible);
    } else {
        throw std::invalid_argument("unsupported live-view control action: " + action);
    }

    ++next.revision;
    state = std::move(next);
    return {LiveViewSceneInputKind::control, *schema, action};
}

std::string live_view_scene_packet_json(
    const LiveViewSceneState& state,
    const LiveViewSceneApplyResult& cause) {
    if (cause.kind == LiveViewSceneInputKind::ignored) {
        throw std::invalid_argument("cannot emit a scene packet for ignored input");
    }

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"schema\":\"dtmb.live_view.scene.v1\""
           << ",\"event\":\"scene\""
           << ",\"verdict\":\"" << scene_verdict(state) << '"'
           << ",\"revision\":" << state.revision
           << ",\"cause\":{\"kind\":";
    append_json_string(output, scene_kind_name(cause.kind));
    output << ",\"schema\":";
    append_json_string(output, cause.source_schema);
    if (!cause.action.empty()) {
        output << ",\"action\":";
        append_json_string(output, cause.action);
    }
    output << "},\"channel\":{\"id\":";
    append_json_string(output, state.channel_id);
    output << ",\"center_frequency_hz\":" << state.center_frequency_hz
           << "},\"track\":{\"id\":";
    append_json_string(output, state.track_id);
    output << "},\"menu\":{\"open\":" << (state.menu_open ? "true" : "false")
           << "},\"debug_panels\":{\"visible\":";
    append_string_array(output, state.visible_debug_panels);
    output << "},\"draw_packet\":";
    if (state.latest_draw_packet_json.empty()) {
        output << "null";
    } else {
        output << state.latest_draw_packet_json;
    }
    output << '}';
    return output.str();
}

std::string live_view_scene_error_packet_json(
    std::string_view message,
    std::uint64_t line_index,
    const LiveViewSceneState& state) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"schema\":\"dtmb.live_view.scene_error.v1\""
           << ",\"event\":\"error\""
           << ",\"verdict\":\"error\""
           << ",\"line_index\":" << line_index
           << ",\"revision\":" << state.revision
           << ",\"message\":";
    append_json_string(output, message);
    output << '}';
    return output.str();
}

std::string live_view_scene_stream_end_packet_json(
    const LiveViewSceneStreamSummary& summary,
    const LiveViewSceneState& state) {
    const auto draw_verdict = terminal_draw_verdict(state);
    const std::string_view verdict = !summary.input_complete
            || summary.scene_packet_count == 0
            || draw_verdict == "error"
        ? "error"
        : summary.error_packet_count != 0
                || summary.draw_packet_count == 0
                || draw_verdict == "degraded"
            ? "degraded"
            : "ok";

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"schema\":\"dtmb.live_view.scene_end.v1\""
           << ",\"event\":\"end\""
           << ",\"verdict\":\"" << verdict << '"'
           << ",\"input_complete\":" << (summary.input_complete ? "true" : "false")
           << ",\"line_count\":" << summary.line_count
           << ",\"control_count\":" << summary.control_count
           << ",\"draw_packet_count\":" << summary.draw_packet_count
           << ",\"scene_packet_count\":" << summary.scene_packet_count
           << ",\"error_packet_count\":" << summary.error_packet_count
           << ",\"ignored_line_count\":" << summary.ignored_line_count
           << ",\"final_revision\":" << state.revision
           << ",\"has_draw_packet\":"
           << (state.latest_draw_packet_json.empty() ? "false" : "true")
           << '}';
    return output.str();
}

}  // namespace dtmb::core
