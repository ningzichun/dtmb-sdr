#include "dtmb/live_view.hpp"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    bool fail_fast = false;
    std::string input_path = "-";
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " [--fail-fast] [draw-and-control.ndjson|-]\n";
}

void emit_packet(std::string_view packet) {
    std::cout << packet << '\n';
    std::cout.flush();
    if (!std::cout) {
        throw std::runtime_error("failed to write scene-packet output");
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
        if (arg == "--fail-fast") {
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
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        std::unique_ptr<std::ifstream> input_file;
        auto& input = input_stream(options.input_path, input_file);

        dtmb::core::LiveViewSceneState state;
        dtmb::core::LiveViewSceneStreamSummary summary;
        std::string line;

        auto emit_error = [&](std::string_view message) {
            emit_packet(dtmb::core::live_view_scene_error_packet_json(
                message,
                summary.line_count,
                state));
            ++summary.error_packet_count;
        };
        auto emit_end = [&]() {
            emit_packet(dtmb::core::live_view_scene_stream_end_packet_json(summary, state));
        };

        while (std::getline(input, line)) {
            ++summary.line_count;
            dtmb::core::LiveViewSceneApplyResult result;
            try {
                result = dtmb::core::apply_live_view_scene_input_json(state, line);
            } catch (const std::exception& exc) {
                emit_error(exc.what());
                if (options.fail_fast) {
                    summary.input_complete = false;
                    emit_end();
                    return 1;
                }
                continue;
            }

            if (result.kind == dtmb::core::LiveViewSceneInputKind::ignored) {
                ++summary.ignored_line_count;
                continue;
            }
            if (result.kind == dtmb::core::LiveViewSceneInputKind::draw_packet) {
                ++summary.draw_packet_count;
            } else {
                ++summary.control_count;
            }
            emit_packet(dtmb::core::live_view_scene_packet_json(state, result));
            ++summary.scene_packet_count;
        }

        if (input.bad()) {
            summary.input_complete = false;
            emit_error("input read failed");
            emit_end();
            return 1;
        }

        const bool found_scene_input = summary.scene_packet_count != 0;
        if (!found_scene_input) {
            emit_error("no dtmb.live_view draw packets or controls found");
        }
        emit_end();
        return found_scene_input ? 0 : 1;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_live_view_scene: " << exc.what() << '\n';
        return 2;
    }
}
