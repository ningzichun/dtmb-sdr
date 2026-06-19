#include "dtmb/live_view.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <string_view>
#include <utility>

namespace {

struct Options {
    dtmb::core::LiveViewOptions view;
    bool latest_only = false;
    bool fail_fast = false;
    std::string input_path = "-";
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " [--width PX] [--height PX] [--min-dbfs DB] [--max-dbfs DB]"
        << " [--latest] [--fail-fast] [monitor.ndjson|-]\n";
}

double parse_double(const std::string& text, const char* field) {
    std::istringstream parser(text);
    parser.imbue(std::locale::classic());
    double value = 0.0;
    parser >> value;
    char extra = '\0';
    if (parser.fail() || (parser >> extra)) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

void emit_packet(std::string_view packet) {
    std::cout << packet << '\n';
    std::cout.flush();
    if (!std::cout) {
        throw std::runtime_error("failed to write draw-packet output");
    }
}

std::istream& input_stream(
    const std::string& path,
    std::unique_ptr<std::ifstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cin;
    }
    file_holder = std::make_unique<std::ifstream>(path);
    if (!*file_holder) {
        throw std::runtime_error("failed to open input: " + path);
    }
    return *file_holder;
}

Options parse_args(int argc, char** argv) {
    Options options;
    bool input_path_set = false;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto need_value = [&](const char* field) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + field);
            }
            return argv[index];
        };

        if (arg == "--width") {
            options.view.width = parse_double(need_value("--width"), "width");
        } else if (arg == "--height") {
            options.view.height = parse_double(need_value("--height"), "height");
        } else if (arg == "--min-dbfs") {
            options.view.min_dbfs = parse_double(need_value("--min-dbfs"), "minimum dBFS");
        } else if (arg == "--max-dbfs") {
            options.view.max_dbfs = parse_double(need_value("--max-dbfs"), "maximum dBFS");
        } else if (arg == "--latest") {
            options.latest_only = true;
        } else if (arg == "--fail-fast") {
            options.fail_fast = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (!input_path_set) {
            options.input_path = arg;
            input_path_set = true;
        } else {
            throw std::invalid_argument("too many positional inputs");
        }
    }

    dtmb::core::validate_live_view_options(options.view);
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        std::unique_ptr<std::ifstream> input_file;
        auto& input = input_stream(options.input_path, input_file);

        std::optional<dtmb::core::LiveMonitorTelemetry> latest;
        std::string line;
        dtmb::core::LiveViewStreamSummary summary;

        auto emit_error = [&](std::string_view message) {
            emit_packet(dtmb::core::live_view_error_packet_json(
                message,
                summary.line_count,
                options.view));
            ++summary.error_packet_count;
        };
        auto emit_end = [&]() {
            emit_packet(dtmb::core::live_view_stream_end_packet_json(summary));
        };

        while (std::getline(input, line)) {
            ++summary.line_count;
            std::optional<dtmb::core::LiveMonitorTelemetry> telemetry;
            try {
                telemetry = dtmb::core::parse_live_monitor_telemetry_json(line);
            } catch (const std::exception& exc) {
                emit_error(exc.what());
                if (options.fail_fast) {
                    summary.input_complete = false;
                    emit_end();
                    return 1;
                }
                continue;
            }
            if (!telemetry.has_value()) {
                ++summary.ignored_line_count;
                continue;
            }
            ++summary.telemetry_count;
            if (options.latest_only) {
                latest = std::move(telemetry);
                continue;
            }
            emit_packet(dtmb::core::live_monitor_draw_packet_json(*telemetry, options.view));
            ++summary.draw_packet_count;
        }

        if (input.bad()) {
            summary.input_complete = false;
            emit_error("input read failed");
            emit_end();
            return 1;
        }

        if (options.latest_only && latest.has_value()) {
            emit_packet(dtmb::core::live_monitor_draw_packet_json(*latest, options.view));
            ++summary.draw_packet_count;
        }

        const bool found_telemetry = summary.telemetry_count != 0;
        if (!found_telemetry) {
            emit_error("no dtmb.live_monitor.v1 telemetry found");
        }
        emit_end();
        return found_telemetry ? 0 : 1;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_live_view_draw: " << exc.what() << '\n';
        return 2;
    }
}
