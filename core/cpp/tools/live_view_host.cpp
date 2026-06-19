#include "binary_stdio.hpp"
#include "live_view_host_renderer.hpp"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using dtmb::tools::live_view_host::Options;

void usage(const char *program) {
    std::cerr << "usage: " << program
              << " [--width PX] [--height PX] [--fail-fast] [scene.ndjson|-]\n"
              << "writes concatenated binary PPM (P6) frames to stdout and "
                 "NDJSON diagnostics"
              << " to stderr\n";
}

std::uint32_t parse_dimension(const std::string &text, const char *field) {
    std::size_t used = 0;
    unsigned long value = 0;
    try {
        value = std::stoul(text, &used, 10);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " +
                                    text);
    }
    if (used != text.size() || value == 0 || value > 16384) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " +
                                    text);
    }
    return static_cast<std::uint32_t>(value);
}

struct ParsedArgs {
    Options options;
    std::string input_path = "-";
};

ParsedArgs parse_args(int argc, char **argv) {
    ParsedArgs parsed;
    bool input_path_set = false;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto need_value = [&](const char *field) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string("missing value for ") +
                                            field);
            }
            return argv[index];
        };

        if (arg == "--width") {
            parsed.options.width =
                parse_dimension(need_value("--width"), "width");
        } else if (arg == "--height") {
            parsed.options.height =
                parse_dimension(need_value("--height"), "height");
        } else if (arg == "--fail-fast") {
            parsed.options.fail_fast = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (!input_path_set) {
            parsed.input_path = arg;
            input_path_set = true;
        } else {
            throw std::invalid_argument("too many positional inputs");
        }
    }
    return parsed;
}

std::istream &input_stream(const std::string &path,
                           std::unique_ptr<std::ifstream> &file_holder) {
    if (path.empty() || path == "-") {
        return std::cin;
    }
    file_holder = std::make_unique<std::ifstream>(path);
    if (!*file_holder) {
        throw std::runtime_error("failed to open input: " + path);
    }
    return *file_holder;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto parsed = parse_args(argc, argv);
        dtmb::tools::configure_binary_stdio(false, true);
        std::unique_ptr<std::ifstream> input_file;
        auto &input = input_stream(parsed.input_path, input_file);
        return dtmb::tools::live_view_host::run(input, std::cout, std::cerr,
                                                parsed.options);
    } catch (const std::exception &exc) {
        std::cerr << "dtmb_core_live_view_host: " << exc.what() << '\n';
        return 2;
    }
}
