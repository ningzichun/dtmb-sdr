#include "live_view_host_renderer.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dtmb::tools::live_view_host {
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
    std::map<std::string, JsonValue, std::less<>> object;
};

class JsonParser {
  public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    JsonValue parse() {
        skip_space();
        auto value = parse_value();
        skip_space();
        if (position_ != text_.size()) {
            fail("trailing content");
        }
        return value;
    }

  private:
    [[noreturn]] void fail(std::string_view message) const {
        throw std::invalid_argument("JSON " + std::string(message) +
                                    " at byte " + std::to_string(position_));
    }

    void skip_space() {
        while (position_ < text_.size()) {
            const char ch = text_[position_];
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
                break;
            }
            ++position_;
        }
    }

    bool consume(char expected) {
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    JsonValue parse_value() {
        if (position_ >= text_.size()) {
            fail("expected value");
        }
        switch (text_[position_]) {
        case 'n':
            return parse_literal("null", JsonValue{});
        case 't': {
            JsonValue value;
            value.type = JsonValue::Type::bool_value;
            value.boolean = true;
            return parse_literal("true", std::move(value));
        }
        case 'f': {
            JsonValue value;
            value.type = JsonValue::Type::bool_value;
            return parse_literal("false", std::move(value));
        }
        case '"': {
            JsonValue value;
            value.type = JsonValue::Type::string_value;
            value.string = parse_string();
            return value;
        }
        case '[':
            return parse_array();
        case '{':
            return parse_object();
        default:
            return parse_number();
        }
    }

    JsonValue parse_literal(std::string_view literal, JsonValue value) {
        if (text_.substr(position_, literal.size()) != literal) {
            fail("invalid literal");
        }
        position_ += literal.size();
        return value;
    }

    static int hex_digit(char ch) {
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        if (ch >= 'a' && ch <= 'f') {
            return ch - 'a' + 10;
        }
        if (ch >= 'A' && ch <= 'F') {
            return ch - 'A' + 10;
        }
        return -1;
    }

    void append_utf8(std::string &output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string output;
        while (position_ < text_.size()) {
            const char ch = text_[position_++];
            if (ch == '"') {
                return output;
            }
            if (static_cast<unsigned char>(ch) < 0x20U) {
                fail("control character in string");
            }
            if (ch != '\\') {
                output.push_back(ch);
                continue;
            }
            if (position_ >= text_.size()) {
                fail("unterminated escape");
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
            case 'u': {
                if (position_ + 4 > text_.size()) {
                    fail("short unicode escape");
                }
                std::uint32_t codepoint = 0;
                for (int index = 0; index < 4; ++index) {
                    const int digit = hex_digit(text_[position_++]);
                    if (digit < 0) {
                        fail("invalid unicode escape");
                    }
                    codepoint =
                        (codepoint << 4U) | static_cast<std::uint32_t>(digit);
                }
                append_utf8(output, codepoint);
                break;
            }
            default:
                fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue value;
        value.type = JsonValue::Type::array_value;
        skip_space();
        if (consume(']')) {
            return value;
        }
        while (true) {
            skip_space();
            value.array.push_back(parse_value());
            skip_space();
            if (consume(']')) {
                return value;
            }
            expect(',');
        }
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue value;
        value.type = JsonValue::Type::object_value;
        skip_space();
        if (consume('}')) {
            return value;
        }
        while (true) {
            skip_space();
            if (position_ >= text_.size() || text_[position_] != '"') {
                fail("expected object key");
            }
            auto key = parse_string();
            skip_space();
            expect(':');
            skip_space();
            auto [_, inserted] =
                value.object.emplace(std::move(key), parse_value());
            if (!inserted) {
                fail("duplicate object key");
            }
            skip_space();
            if (consume('}')) {
                return value;
            }
            expect(',');
        }
    }

    JsonValue parse_number() {
        const std::size_t start = position_;
        if (consume('-') && position_ >= text_.size()) {
            fail("invalid number");
        }
        if (consume('0')) {
            if (position_ < text_.size() && text_[position_] >= '0' &&
                text_[position_] <= '9') {
                fail("invalid leading zero");
            }
        } else {
            if (position_ >= text_.size() || text_[position_] < '1' ||
                text_[position_] > '9') {
                fail("invalid number");
            }
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
        }
        if (consume('.')) {
            if (position_ >= text_.size() || text_[position_] < '0' ||
                text_[position_] > '9') {
                fail("invalid fraction");
            }
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
        }
        if (position_ < text_.size() &&
            (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            if (position_ >= text_.size() || text_[position_] < '0' ||
                text_[position_] > '9') {
                fail("invalid exponent");
            }
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
        }

        const auto token = text_.substr(start, position_ - start);
        double number = 0.0;
        const auto [end, error] =
            std::from_chars(token.data(), token.data() + token.size(), number,
                            std::chars_format::general);
        if (error != std::errc{} || end != token.data() + token.size() ||
            !std::isfinite(number)) {
            fail("invalid finite number");
        }
        JsonValue value;
        value.type = JsonValue::Type::number_value;
        value.number = number;
        return value;
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

const JsonValue *field(const JsonValue &object, std::string_view name) {
    if (object.type != JsonValue::Type::object_value) {
        return nullptr;
    }
    const auto match = object.object.find(name);
    return match == object.object.end() ? nullptr : &match->second;
}

const JsonValue &required_field(const JsonValue &object,
                                std::string_view name) {
    const auto *value = field(object, name);
    if (value == nullptr) {
        throw std::invalid_argument("missing field: " + std::string(name));
    }
    return *value;
}

const JsonValue &required_object(const JsonValue &object,
                                 std::string_view name) {
    const auto &value = required_field(object, name);
    if (value.type != JsonValue::Type::object_value) {
        throw std::invalid_argument("invalid object field: " +
                                    std::string(name));
    }
    return value;
}

const JsonValue &required_array(const JsonValue &object,
                                std::string_view name) {
    const auto &value = required_field(object, name);
    if (value.type != JsonValue::Type::array_value) {
        throw std::invalid_argument("invalid array field: " +
                                    std::string(name));
    }
    return value;
}

std::string required_string(const JsonValue &object, std::string_view name) {
    const auto &value = required_field(object, name);
    if (value.type != JsonValue::Type::string_value) {
        throw std::invalid_argument("invalid string field: " +
                                    std::string(name));
    }
    return value.string;
}

double required_number(const JsonValue &object, std::string_view name) {
    const auto &value = required_field(object, name);
    if (value.type != JsonValue::Type::number_value ||
        !std::isfinite(value.number)) {
        throw std::invalid_argument("invalid numeric field: " +
                                    std::string(name));
    }
    return value.number;
}

bool required_bool(const JsonValue &object, std::string_view name) {
    const auto &value = required_field(object, name);
    if (value.type != JsonValue::Type::bool_value) {
        throw std::invalid_argument("invalid boolean field: " +
                                    std::string(name));
    }
    return value.boolean;
}

std::uint64_t required_unsigned(const JsonValue &object,
                                std::string_view name) {
    const double number = required_number(object, name);
    constexpr double kMaximumExactJsonInteger = 9007199254740991.0;
    if (number < 0.0 || number > kMaximumExactJsonInteger ||
        std::floor(number) != number) {
        throw std::invalid_argument("invalid unsigned integer field: " +
                                    std::string(name));
    }
    return static_cast<std::uint64_t>(number);
}

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Rect {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

struct Color {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    double a = 0.0;
};

constexpr double kMaximumNormalizedMagnitude = 1.0e6;
constexpr double kMaximumStrokePixels = 64.0;
constexpr double kMaximumTextPixels = 4096.0;
constexpr std::size_t kMaximumPanels = 4096U;
constexpr std::size_t kMaximumDrawOperations = 65536U;
constexpr std::size_t kMaximumPolylineVertices = 65536U;
constexpr std::size_t kMaximumTextBytes = 16384U;

void require_bounded_geometry(double value, std::string_view name) {
    if (!std::isfinite(value) ||
        std::abs(value) > kMaximumNormalizedMagnitude) {
        throw std::invalid_argument("out-of-range geometry in " +
                                    std::string(name));
    }
}

std::vector<double> number_array(const JsonValue &object, std::string_view name,
                                 std::size_t expected_size) {
    const auto &values = required_array(object, name).array;
    if (values.size() != expected_size) {
        throw std::invalid_argument("invalid " + std::string(name) +
                                    " element count");
    }
    std::vector<double> result;
    result.reserve(values.size());
    for (const auto &value : values) {
        if (value.type != JsonValue::Type::number_value ||
            !std::isfinite(value.number)) {
            throw std::invalid_argument("invalid numeric value in " +
                                        std::string(name));
        }
        result.push_back(value.number);
    }
    return result;
}

Rect rect_field(const JsonValue &object, std::string_view name) {
    const auto values = number_array(object, name, 4);
    if (values[2] < 0.0 || values[3] < 0.0) {
        throw std::invalid_argument("negative extent in " + std::string(name));
    }
    for (const double value : values) {
        require_bounded_geometry(value, name);
    }
    return {values[0], values[1], values[2], values[3]};
}

Point point_field(const JsonValue &object, std::string_view name) {
    const auto values = number_array(object, name, 2);
    for (const double value : values) {
        require_bounded_geometry(value, name);
    }
    return {values[0], values[1]};
}

Point point_value(const JsonValue &value, std::string_view name) {
    if (value.type != JsonValue::Type::array_value ||
        value.array.size() != 2U ||
        value.array[0].type != JsonValue::Type::number_value ||
        value.array[1].type != JsonValue::Type::number_value) {
        throw std::invalid_argument("invalid " + std::string(name));
    }
    const Point point{value.array[0].number, value.array[1].number};
    require_bounded_geometry(point.x, name);
    require_bounded_geometry(point.y, name);
    return point;
}

void require_normalized_panel_rect(const Rect &rect) {
    if (rect.x < 0.0 || rect.y < 0.0 || rect.width > 1.0 ||
        rect.height > 1.0 || rect.x > 1.0 - rect.width ||
        rect.y > 1.0 - rect.height) {
        throw std::invalid_argument(
            "draw-packet panel rect must fit normalized viewport");
    }
}

Color color_field(const JsonValue &object) {
    const auto values = number_array(object, "rgba", 4);
    return {
        std::clamp(values[0], 0.0, 1.0),
        std::clamp(values[1], 0.0, 1.0),
        std::clamp(values[2], 0.0, 1.0),
        std::clamp(values[3], 0.0, 1.0),
    };
}

struct Frame {
    std::uint64_t revision = 0;
    std::string verdict;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgb;
};

class Raster {
  public:
    Raster(std::uint32_t width, std::uint32_t height)
        : width_(width), height_(height),
          rgb_(static_cast<std::size_t>(width) * height * 3U, 0U) {}

    void blend(int x, int y, Color color) {
        if (x < 0 || y < 0 || x >= static_cast<int>(width_) ||
            y >= static_cast<int>(height_)) {
            return;
        }
        const auto offset = (static_cast<std::size_t>(y) * width_ +
                             static_cast<std::size_t>(x)) *
                            3U;
        const auto blend_channel = [&](double source,
                                       std::uint8_t destination) {
            const double value =
                source * color.a +
                (static_cast<double>(destination) / 255.0) * (1.0 - color.a);
            return static_cast<std::uint8_t>(
                std::clamp(std::lround(value * 255.0), 0L, 255L));
        };
        rgb_[offset] = blend_channel(color.r, rgb_[offset]);
        rgb_[offset + 1] = blend_channel(color.g, rgb_[offset + 1]);
        rgb_[offset + 2] = blend_channel(color.b, rgb_[offset + 2]);
    }

    void fill(Rect rect, Color color) {
        const double right = rect.x + rect.width;
        const double bottom = rect.y + rect.height;
        if (!std::isfinite(rect.x) || !std::isfinite(rect.y) ||
            !std::isfinite(rect.width) || !std::isfinite(rect.height) ||
            !std::isfinite(right) || !std::isfinite(bottom) ||
            rect.width < 0.0 || rect.height < 0.0) {
            throw std::invalid_argument("invalid rectangle raster geometry");
        }
        const double clipped_left =
            std::clamp(rect.x, 0.0, static_cast<double>(width_));
        const double clipped_top =
            std::clamp(rect.y, 0.0, static_cast<double>(height_));
        const double clipped_right =
            std::clamp(right, 0.0, static_cast<double>(width_));
        const double clipped_bottom =
            std::clamp(bottom, 0.0, static_cast<double>(height_));
        if (clipped_right <= clipped_left || clipped_bottom <= clipped_top) {
            return;
        }
        const int x0 = static_cast<int>(std::floor(clipped_left));
        const int y0 = static_cast<int>(std::floor(clipped_top));
        const int x1 = static_cast<int>(std::ceil(clipped_right));
        const int y1 = static_cast<int>(std::ceil(clipped_bottom));
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                blend(x, y, color);
            }
        }
    }

    void line(Point from, Point to, double width, Color color) {
        if (!std::isfinite(from.x) || !std::isfinite(from.y) ||
            !std::isfinite(to.x) || !std::isfinite(to.y) ||
            !std::isfinite(width) || !(width > 0.0) ||
            width > kMaximumStrokePixels) {
            throw std::invalid_argument("invalid polyline raster geometry");
        }
        const int radius =
            std::max(0, static_cast<int>(std::ceil(width * 0.5)) - 1);
        const double minimum = -static_cast<double>(radius);
        const double maximum_x =
            static_cast<double>(width_ - 1U) + static_cast<double>(radius);
        const double maximum_y =
            static_cast<double>(height_ - 1U) + static_cast<double>(radius);
        const double dx_double = to.x - from.x;
        const double dy_double = to.y - from.y;
        if (!std::isfinite(dx_double) || !std::isfinite(dy_double)) {
            throw std::invalid_argument("invalid polyline raster geometry");
        }
        double start = 0.0;
        double end = 1.0;
        const auto clip = [&](double direction, double distance) {
            if (direction == 0.0) {
                return distance >= 0.0;
            }
            const double ratio = distance / direction;
            if (direction < 0.0) {
                if (ratio > end) {
                    return false;
                }
                start = std::max(start, ratio);
            } else {
                if (ratio < start) {
                    return false;
                }
                end = std::min(end, ratio);
            }
            return true;
        };
        if (!clip(-dx_double, from.x - minimum) ||
            !clip(dx_double, maximum_x - from.x) ||
            !clip(-dy_double, from.y - minimum) ||
            !clip(dy_double, maximum_y - from.y)) {
            return;
        }
        const Point clipped_from{
            from.x + start * dx_double,
            from.y + start * dy_double,
        };
        const Point clipped_to{
            from.x + end * dx_double,
            from.y + end * dy_double,
        };
        if (!std::isfinite(clipped_from.x) || !std::isfinite(clipped_from.y) ||
            !std::isfinite(clipped_to.x) || !std::isfinite(clipped_to.y)) {
            throw std::invalid_argument("invalid clipped polyline geometry");
        }
        int x0 = static_cast<int>(std::lround(clipped_from.x));
        int y0 = static_cast<int>(std::lround(clipped_from.y));
        const int x1 = static_cast<int>(std::lround(clipped_to.x));
        const int y1 = static_cast<int>(std::lround(clipped_to.y));
        const int dx = std::abs(x1 - x0);
        const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true) {
            for (int py = y0 - radius; py <= y0 + radius; ++py) {
                for (int px = x0 - radius; px <= x0 + radius; ++px) {
                    blend(px, py, color);
                }
            }
            if (x0 == x1 && y0 == y1) {
                break;
            }
            const int twice = 2 * error;
            if (twice >= dy) {
                error += dy;
                x0 += sx;
            }
            if (twice <= dx) {
                error += dx;
                y0 += sy;
            }
        }
    }

    const std::vector<std::uint8_t> &pixels() const { return rgb_; }

  private:
    std::uint32_t width_;
    std::uint32_t height_;
    std::vector<std::uint8_t> rgb_;
};

using Glyph = std::array<std::uint8_t, 7>;

Glyph glyph(char raw) {
    const char ch =
        raw >= 'a' && raw <= 'z' ? static_cast<char>(raw - 'a' + 'A') : raw;
    switch (ch) {
    case 'A':
        return {14, 17, 17, 31, 17, 17, 17};
    case 'B':
        return {30, 17, 17, 30, 17, 17, 30};
    case 'C':
        return {14, 17, 16, 16, 16, 17, 14};
    case 'D':
        return {30, 17, 17, 17, 17, 17, 30};
    case 'E':
        return {31, 16, 16, 30, 16, 16, 31};
    case 'F':
        return {31, 16, 16, 30, 16, 16, 16};
    case 'G':
        return {14, 17, 16, 23, 17, 17, 15};
    case 'H':
        return {17, 17, 17, 31, 17, 17, 17};
    case 'I':
        return {31, 4, 4, 4, 4, 4, 31};
    case 'J':
        return {7, 2, 2, 2, 18, 18, 12};
    case 'K':
        return {17, 18, 20, 24, 20, 18, 17};
    case 'L':
        return {16, 16, 16, 16, 16, 16, 31};
    case 'M':
        return {17, 27, 21, 21, 17, 17, 17};
    case 'N':
        return {17, 25, 21, 19, 17, 17, 17};
    case 'O':
        return {14, 17, 17, 17, 17, 17, 14};
    case 'P':
        return {30, 17, 17, 30, 16, 16, 16};
    case 'Q':
        return {14, 17, 17, 17, 21, 18, 13};
    case 'R':
        return {30, 17, 17, 30, 20, 18, 17};
    case 'S':
        return {15, 16, 16, 14, 1, 1, 30};
    case 'T':
        return {31, 4, 4, 4, 4, 4, 4};
    case 'U':
        return {17, 17, 17, 17, 17, 17, 14};
    case 'V':
        return {17, 17, 17, 17, 17, 10, 4};
    case 'W':
        return {17, 17, 17, 21, 21, 21, 10};
    case 'X':
        return {17, 17, 10, 4, 10, 17, 17};
    case 'Y':
        return {17, 17, 10, 4, 4, 4, 4};
    case 'Z':
        return {31, 1, 2, 4, 8, 16, 31};
    case '0':
        return {14, 17, 19, 21, 25, 17, 14};
    case '1':
        return {4, 12, 4, 4, 4, 4, 14};
    case '2':
        return {14, 17, 1, 2, 4, 8, 31};
    case '3':
        return {30, 1, 1, 14, 1, 1, 30};
    case '4':
        return {2, 6, 10, 18, 31, 2, 2};
    case '5':
        return {31, 16, 16, 30, 1, 1, 30};
    case '6':
        return {14, 16, 16, 30, 17, 17, 14};
    case '7':
        return {31, 1, 2, 4, 8, 8, 8};
    case '8':
        return {14, 17, 17, 14, 17, 17, 14};
    case '9':
        return {14, 17, 17, 15, 1, 1, 14};
    case '.':
        return {0, 0, 0, 0, 0, 6, 6};
    case ',':
        return {0, 0, 0, 0, 6, 6, 4};
    case ':':
        return {0, 6, 6, 0, 6, 6, 0};
    case '-':
        return {0, 0, 0, 31, 0, 0, 0};
    case '/':
        return {1, 2, 2, 4, 8, 8, 16};
    case '(':
        return {2, 4, 8, 8, 8, 4, 2};
    case ')':
        return {8, 4, 2, 2, 2, 4, 8};
    case '+':
        return {0, 4, 4, 31, 4, 4, 0};
    case '=':
        return {0, 31, 0, 31, 0, 0, 0};
    case '_':
        return {0, 0, 0, 0, 0, 0, 31};
    case ' ':
        return {};
    default:
        return {31, 17, 21, 21, 21, 17, 31};
    }
}

void draw_text(Raster &raster, Point anchor, double size, Color color,
               std::string_view text) {
    if (!std::isfinite(anchor.x) || !std::isfinite(anchor.y) ||
        !std::isfinite(size) || !(size > 0.0) ||
        size > kMaximumTextPixels || text.size() > kMaximumTextBytes) {
        throw std::invalid_argument("invalid text raster geometry");
    }
    const double scale = std::max(1.0, std::round(size / 7.0));
    double cursor_x = anchor.x;
    const double origin_y = anchor.y - 7.0 * scale;
    for (const unsigned char byte : text) {
        if (byte == '\n') {
            cursor_x = anchor.x;
            continue;
        }
        const auto rows = glyph(static_cast<char>(byte));
        for (int row = 0; row < 7; ++row) {
            for (int column = 0; column < 5; ++column) {
                if ((rows[static_cast<std::size_t>(row)] &
                     (1U << (4 - column))) == 0U) {
                    continue;
                }
                raster.fill({cursor_x + static_cast<double>(column) * scale,
                             origin_y + static_cast<double>(row) * scale,
                             scale,
                             scale},
                            color);
            }
        }
        cursor_x += 6.0 * scale;
        if (!std::isfinite(cursor_x)) {
            throw std::invalid_argument("invalid text raster geometry");
        }
    }
}

void draw_host_chrome(Raster &raster, std::uint32_t width, std::uint32_t height,
                      std::string_view channel_id, std::string_view track_id,
                      bool menu_open) {
    const Color chrome_background{0.02, 0.025, 0.035, 0.88};
    const Color chrome_foreground{0.9, 0.94, 0.98, 1.0};
    if (!channel_id.empty() || !track_id.empty()) {
        const double bar_height = std::min(28.0, static_cast<double>(height));
        raster.fill({0.0, 0.0, static_cast<double>(width), bar_height},
                    chrome_background);
        std::string label;
        if (!channel_id.empty()) {
            label = "CHANNEL " + std::string(channel_id);
        }
        if (!track_id.empty()) {
            if (!label.empty()) {
                label += "  ";
            }
            label += "TRACK " + std::string(track_id);
        }
        const double text_size =
            std::min(14.0, std::max(7.0, bar_height - 5.0));
        draw_text(raster, {6.0, std::max(7.0, bar_height - 5.0)}, text_size,
                  chrome_foreground, label);
    }
    if (menu_open) {
        const double menu_width =
            std::min(420.0, static_cast<double>(width) * 0.62);
        const double menu_height =
            std::min(220.0, static_cast<double>(height) * 0.5);
        const Rect menu{
            (static_cast<double>(width) - menu_width) * 0.5,
            (static_cast<double>(height) - menu_height) * 0.5,
            menu_width,
            menu_height,
        };
        raster.fill(menu, {0.025, 0.03, 0.045, 0.94});
        draw_text(raster,
                  {menu.x + 12.0, menu.y + std::min(24.0, menu.height - 3.0)},
                  std::min(14.0, std::max(7.0, menu.height - 3.0)),
                  chrome_foreground, "MENU");
    }
}

Rect panel_pixels(Rect panel, std::uint32_t width, std::uint32_t height) {
    return {
        panel.x * width,
        panel.y * height,
        panel.width * width,
        panel.height * height,
    };
}

Point panel_point(Point point, Rect panel) {
    return {
        panel.x + point.x * panel.width,
        panel.y + point.y * panel.height,
    };
}

Rect panel_rect(Rect rect, Rect panel) {
    return {
        panel.x + rect.x * panel.width,
        panel.y + rect.y * panel.height,
        rect.width * panel.width,
        rect.height * panel.height,
    };
}

void require_string_value(const JsonValue &object, std::string_view name,
                          std::string_view expected) {
    const auto actual = required_string(object, name);
    if (actual != expected) {
        throw std::invalid_argument("unsupported " + std::string(name) + ": " +
                                    actual);
    }
}

int severity(std::string_view verdict);
void retain_worst(std::string &worst, std::string_view verdict);

Frame render_scene(const JsonValue &scene, const Options &options) {
    require_string_value(scene, "event", "scene");
    auto frame_verdict = required_string(scene, "verdict");
    if (severity(frame_verdict) < 0) {
        throw std::invalid_argument("unsupported scene verdict: " +
                                    frame_verdict);
    }
    const auto revision = required_unsigned(scene, "revision");
    const auto channel_id =
        required_string(required_object(scene, "channel"), "id");
    const auto track_id =
        required_string(required_object(scene, "track"), "id");
    const bool menu_open =
        required_bool(required_object(scene, "menu"), "open");

    std::set<std::string, std::less<>> visible_panels;
    const auto &visible =
        required_array(required_object(scene, "debug_panels"), "visible");
    if (visible.array.size() > kMaximumPanels) {
        throw std::invalid_argument("too many visible debug panels");
    }
    for (const auto &panel : visible.array) {
        if (panel.type != JsonValue::Type::string_value) {
            throw std::invalid_argument("invalid debug-panel visibility value");
        }
        visible_panels.insert(panel.string);
    }

    const auto &draw_packet = required_field(scene, "draw_packet");
    if (draw_packet.type == JsonValue::Type::null_value) {
        throw std::invalid_argument("scene has no draw packet");
    }
    if (draw_packet.type != JsonValue::Type::object_value) {
        throw std::invalid_argument("invalid scene draw packet");
    }
    require_string_value(draw_packet, "schema",
                         "dtmb.live_view.draw_packet.v1");
    const auto draw_event = required_string(draw_packet, "event");
    if (draw_event != "draw" && draw_event != "error") {
        throw std::invalid_argument("unsupported draw-packet event: " +
                                    draw_event);
    }
    const auto draw_verdict = required_string(draw_packet, "verdict");
    if (severity(draw_verdict) < 0) {
        throw std::invalid_argument("unsupported draw-packet verdict: " +
                                    draw_verdict);
    }
    retain_worst(frame_verdict, draw_verdict);
    const auto &contract = required_object(draw_packet, "render_contract");
    require_string_value(contract, "api", "GL/WebGL");
    if (required_unsigned(contract, "version") != 1U) {
        throw std::invalid_argument("unsupported render-contract version");
    }
    require_string_value(contract, "viewport_space", "normalized top-left");
    require_string_value(contract, "panel_space", "normalized top-left");
    require_string_value(contract, "blend", "source-over");

    const auto &layout = required_object(draw_packet, "layout");
    const double layout_width = required_number(layout, "width");
    const double layout_height = required_number(layout, "height");
    if (!(layout_width > 0.0) || !(layout_height > 0.0) ||
        layout_width > 16384.0 || layout_height > 16384.0) {
        throw std::invalid_argument(
            "draw-packet layout dimensions must be in [1, 16384]");
    }
    require_string_value(layout, "coordinate_system",
                         "normalized top-left panels");
    const auto dimension = [](double value, std::uint32_t override_value,
                              std::string_view name) {
        const double selected =
            override_value == 0 ? value : static_cast<double>(override_value);
        if (!(selected >= 1.0) || selected > 16384.0) {
            throw std::invalid_argument("invalid output " + std::string(name));
        }
        return static_cast<std::uint32_t>(std::lround(selected));
    };
    const auto width = dimension(layout_width, options.width, "width");
    const auto height = dimension(layout_height, options.height, "height");
    constexpr std::uint64_t kMaximumPixels = 64U * 1024U * 1024U;
    if (static_cast<std::uint64_t>(width) * height > kMaximumPixels) {
        throw std::invalid_argument("host-renderer frame exceeds pixel limit");
    }

    std::map<std::string, Rect, std::less<>> panels;
    const auto &panel_values = required_array(draw_packet, "panels").array;
    if (panel_values.size() > kMaximumPanels) {
        throw std::invalid_argument("too many draw-packet panels");
    }
    for (const auto &panel : panel_values) {
        if (panel.type != JsonValue::Type::object_value) {
            throw std::invalid_argument("invalid draw-packet panel");
        }
        const auto id = required_string(panel, "id");
        const auto panel_rect_normalized = rect_field(panel, "rect");
        require_normalized_panel_rect(panel_rect_normalized);
        const auto [_, inserted] = panels.emplace(
            id, panel_pixels(panel_rect_normalized, width, height));
        if (!inserted) {
            throw std::invalid_argument("duplicate draw-packet panel: " + id);
        }
    }

    Raster raster(width, height);
    const double x_scale = static_cast<double>(width) / layout_width;
    const double y_scale = static_cast<double>(height) / layout_height;
    const double pixel_scale = std::min(x_scale, y_scale);
    const auto &draw_operations = required_array(draw_packet, "draw").array;
    if (draw_operations.size() > kMaximumDrawOperations) {
        throw std::invalid_argument("too many draw operations");
    }
    for (const auto &operation : draw_operations) {
        if (operation.type != JsonValue::Type::object_value) {
            throw std::invalid_argument("invalid draw operation");
        }
        const auto panel_id = required_string(operation, "panel");
        if (!visible_panels.contains(panel_id)) {
            continue;
        }
        const auto panel_match = panels.find(panel_id);
        if (panel_match == panels.end()) {
            throw std::invalid_argument(
                "draw operation references unknown panel: " + panel_id);
        }
        require_string_value(operation, "space", "panel-normalized");
        const auto op = required_string(operation, "op");
        const auto color = color_field(operation);
        if (op == "rect") {
            require_string_value(operation, "primitive", "TRIANGLE_STRIP");
            require_string_value(operation, "shader", "solid_color_gles2");
            raster.fill(
                panel_rect(rect_field(operation, "rect"), panel_match->second),
                color);
        } else if (op == "polyline") {
            require_string_value(operation, "primitive", "LINE_STRIP");
            require_string_value(operation, "shader", "polyline_gles2");
            const double contract_line_width =
                required_number(operation, "width");
            require_bounded_geometry(contract_line_width, "polyline width");
            const double line_width = contract_line_width * pixel_scale;
            if (!std::isfinite(line_width) || !(line_width > 0.0) ||
                line_width > kMaximumStrokePixels) {
                throw std::invalid_argument(
                    "polyline pixel width must be in (0, 64]");
            }
            const auto &vertices = required_array(operation, "vertices").array;
            if (vertices.size() > kMaximumPolylineVertices) {
                throw std::invalid_argument("too many polyline vertices");
            }
            std::optional<Point> previous;
            for (const auto &vertex : vertices) {
                const auto current =
                    panel_point(point_value(vertex, "polyline vertex"),
                                panel_match->second);
                if (previous.has_value()) {
                    raster.line(*previous, current, line_width, color);
                }
                previous = current;
            }
        } else if (op == "text") {
            require_string_value(operation, "renderer", "host-text");
            const double contract_size = required_number(operation, "size");
            require_bounded_geometry(contract_size, "text size");
            const double size = contract_size * pixel_scale;
            if (!std::isfinite(size) || !(size > 0.0) ||
                size > kMaximumTextPixels) {
                throw std::invalid_argument(
                    "text pixel size must be in (0, 4096]");
            }
            draw_text(raster,
                      panel_point(point_field(operation, "anchor"),
                                  panel_match->second),
                      size, color, required_string(operation, "text"));
        } else {
            throw std::invalid_argument("unsupported draw operation: " + op);
        }
    }
    draw_host_chrome(raster, width, height, channel_id, track_id, menu_open);

    return {revision, frame_verdict, width, height, raster.pixels()};
}

void append_json_string(std::ostream &output, std::string_view text) {
    output << '"';
    for (const unsigned char ch : text) {
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
            if (ch < 0x20U) {
                output << "\\u00" << std::hex << std::setw(2)
                       << std::setfill('0') << static_cast<int>(ch) << std::dec
                       << std::setfill(' ');
            } else {
                output << static_cast<char>(ch);
            }
        }
    }
    output << '"';
}

void emit_diagnostic(std::ostream &output, std::string_view packet) {
    output << packet << '\n';
    output.flush();
    if (!output) {
        throw std::ios_base::failure(
            "failed to write host-renderer diagnostics");
    }
}

void emit_error(std::ostream &output, std::uint64_t line_index,
                std::string_view message) {
    std::ostringstream packet;
    packet.imbue(std::locale::classic());
    packet << "{\"schema\":\"dtmb.live_view.host_error.v1\""
           << ",\"event\":\"error\",\"verdict\":\"error\""
           << ",\"line_index\":" << line_index << ",\"message\":";
    append_json_string(packet, message);
    packet << '}';
    emit_diagnostic(output, packet.str());
}

void emit_frame(std::ostream &output, const Frame &frame,
                std::uint64_t frame_index) {
    std::ostringstream packet;
    packet.imbue(std::locale::classic());
    packet << "{\"schema\":\"dtmb.live_view.host_frame.v1\""
           << ",\"event\":\"frame\",\"verdict\":\"" << frame.verdict << '"'
           << ",\"frame_index\":" << frame_index
           << ",\"revision\":" << frame.revision << ",\"format\":\"ppm-p6\""
           << ",\"width\":" << frame.width << ",\"height\":" << frame.height
           << ",\"rgb_bytes\":" << frame.rgb.size() << '}';
    emit_diagnostic(output, packet.str());
}

int severity(std::string_view verdict) {
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

void retain_worst(std::string &worst, std::string_view verdict) {
    if (severity(verdict) > severity(worst)) {
        worst = verdict;
    }
}

struct Summary {
    std::uint64_t line_count = 0;
    std::uint64_t scene_count = 0;
    std::uint64_t frame_count = 0;
    std::uint64_t error_packet_count = 0;
    std::uint64_t ignored_line_count = 0;
    std::uint64_t no_draw_packet_count = 0;
    bool scene_end_seen = false;
    bool input_complete = false;
    std::string worst_upstream_verdict;
};

std::string terminal_verdict(const Summary &summary) {
    if (!summary.input_complete || summary.frame_count == 0 ||
        summary.worst_upstream_verdict == "error") {
        return "error";
    }
    if (summary.error_packet_count != 0 ||
        summary.worst_upstream_verdict == "degraded") {
        return "degraded";
    }
    return "ok";
}

int terminal_status(const Summary &summary) {
    return severity(terminal_verdict(summary));
}

void emit_end(std::ostream &output, const Summary &summary) {
    std::ostringstream packet;
    packet.imbue(std::locale::classic());
    packet << "{\"schema\":\"dtmb.live_view.host_end.v1\""
           << ",\"event\":\"end\",\"verdict\":\"" << terminal_verdict(summary)
           << '"' << ",\"input_complete\":"
           << (summary.input_complete ? "true" : "false")
           << ",\"line_count\":" << summary.line_count
           << ",\"scene_count\":" << summary.scene_count
           << ",\"frame_count\":" << summary.frame_count
           << ",\"error_packet_count\":" << summary.error_packet_count
           << ",\"ignored_line_count\":" << summary.ignored_line_count
           << ",\"no_draw_packet_count\":" << summary.no_draw_packet_count
           << '}';
    emit_diagnostic(output, packet.str());
}

void write_ppm(std::ostream &output, const Frame &frame) {
    output << "P6\n" << frame.width << ' ' << frame.height << "\n255\n";
    output.write(reinterpret_cast<const char *>(frame.rgb.data()),
                 static_cast<std::streamsize>(frame.rgb.size()));
    output.flush();
    if (!output) {
        throw std::ios_base::failure(
            "failed to write host-renderer frame output");
    }
}

bool is_blank(std::string_view line) {
    return std::all_of(line.begin(), line.end(), [](unsigned char ch) {
        return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
    });
}

} // namespace

int run(std::istream &input, std::ostream &frame_output,
        std::ostream &diagnostics, const Options &options) {
    if (options.width > 16384U || options.height > 16384U) {
        throw std::invalid_argument(
            "host-renderer dimensions must not exceed 16384");
    }

    Summary summary;
    std::string line;
    auto fail = [&](std::string_view message) {
        emit_error(diagnostics, summary.line_count, message);
        ++summary.error_packet_count;
    };

    while (std::getline(input, line)) {
        ++summary.line_count;
        if (is_blank(line)) {
            ++summary.ignored_line_count;
            continue;
        }
        try {
            const auto root = JsonParser(line).parse();
            if (root.type != JsonValue::Type::object_value) {
                throw std::invalid_argument(
                    "root JSON value must be an object");
            }
            const auto *schema_value = field(root, "schema");
            if (summary.scene_end_seen) {
                summary.input_complete = false;
                if (schema_value != nullptr &&
                    schema_value->type == JsonValue::Type::string_value &&
                    schema_value->string == "dtmb.live_view.scene_end.v1") {
                    throw std::invalid_argument("duplicate scene_end packet");
                }
                throw std::invalid_argument("packet received after scene_end");
            }
            if (schema_value == nullptr ||
                schema_value->type != JsonValue::Type::string_value) {
                ++summary.ignored_line_count;
                continue;
            }
            const auto &schema = schema_value->string;
            if (schema == "dtmb.live_view.scene.v1") {
                ++summary.scene_count;
                require_string_value(root, "event", "scene");
                const auto scene_verdict = required_string(root, "verdict");
                if (severity(scene_verdict) < 0) {
                    throw std::invalid_argument("unsupported scene verdict: " +
                                                scene_verdict);
                }
                retain_worst(summary.worst_upstream_verdict, scene_verdict);
                (void)required_unsigned(root, "revision");
                (void)required_array(required_object(root, "debug_panels"),
                                     "visible");
                const auto *draw_packet = field(root, "draw_packet");
                if (draw_packet != nullptr &&
                    draw_packet->type == JsonValue::Type::null_value) {
                    ++summary.no_draw_packet_count;
                    continue;
                }
                const auto frame = render_scene(root, options);
                write_ppm(frame_output, frame);
                ++summary.frame_count;
                retain_worst(summary.worst_upstream_verdict, frame.verdict);
                emit_frame(diagnostics, frame, summary.frame_count);
            } else if (schema == "dtmb.live_view.scene_error.v1") {
                require_string_value(root, "event", "error");
                const auto verdict = required_string(root, "verdict");
                if (severity(verdict) < 0) {
                    throw std::invalid_argument(
                        "unsupported scene-error verdict: " + verdict);
                }
                retain_worst(summary.worst_upstream_verdict, verdict);
                fail("upstream scene error: " +
                     required_string(root, "message"));
                if (options.fail_fast) {
                    summary.input_complete = false;
                    emit_end(diagnostics, summary);
                    return terminal_status(summary);
                }
            } else if (schema == "dtmb.live_view.scene_end.v1") {
                require_string_value(root, "event", "end");
                const auto verdict = required_string(root, "verdict");
                if (severity(verdict) < 0) {
                    throw std::invalid_argument(
                        "unsupported scene-end verdict: " + verdict);
                }
                const bool input_complete =
                    required_bool(root, "input_complete");
                summary.scene_end_seen = true;
                summary.input_complete = input_complete;
                retain_worst(summary.worst_upstream_verdict, verdict);
            } else {
                ++summary.ignored_line_count;
            }
        } catch (const std::ios_base::failure &) {
            throw;
        } catch (const std::exception &exc) {
            if (summary.scene_end_seen) {
                summary.input_complete = false;
            }
            fail(exc.what());
            if (options.fail_fast) {
                summary.input_complete = false;
                emit_end(diagnostics, summary);
                return terminal_status(summary);
            }
        }
    }

    if (input.bad()) {
        summary.input_complete = false;
        fail("input read failed");
        emit_end(diagnostics, summary);
        return terminal_status(summary);
    }
    if (!summary.scene_end_seen) {
        summary.input_complete = false;
        fail("missing dtmb.live_view.scene_end.v1 packet");
    }
    if (summary.scene_count == 0) {
        fail("no dtmb.live_view.scene.v1 packets found");
    }
    emit_end(diagnostics, summary);
    return terminal_status(summary);
}

} // namespace dtmb::tools::live_view_host
