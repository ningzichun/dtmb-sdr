#include "ldpc_h_gate.hpp"

#include "binary_stdio.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    dtmb::tools::ldpc_h_gate::Options gate;
    std::string alist_path;
    std::string input_path = "-";
    std::string diagnostics_path;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --fec-rate 1|2|3 --alist PATH"
        << " [--window-codewords N] [--window-step-codewords N]"
        << " [--window-threshold RATIO] [--read-chunk-bytes N]"
        << " [--diagnostics-format ndjson|json] [--diagnostics-out PATH]"
        << " [--output-mode passthrough|shortlist]"
        << " [--fec-frame-codewords N --input-fec-frame-aligned]"
        << " [input.llr.f32|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size() || value == 0) {
        throw std::invalid_argument(std::string(field) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

double parse_ratio(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(field) + " must be finite and within 0..1");
    }
    return value;
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto need_value = [&](const char* field) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + field);
            }
            return argv[index];
        };
        if (arg == "--fec-rate") {
            options.gate.fec_rate = parse_size(need_value("--fec-rate"), "fec rate");
        } else if (arg == "--alist") {
            options.alist_path = need_value("--alist");
        } else if (arg == "--window-codewords") {
            options.gate.window_codewords = parse_size(
                need_value("--window-codewords"),
                "window codewords");
        } else if (arg == "--window-step-codewords") {
            options.gate.window_step_codewords = parse_size(
                need_value("--window-step-codewords"),
                "window step codewords");
        } else if (arg == "--window-threshold") {
            options.gate.threshold = parse_ratio(
                need_value("--window-threshold"),
                "window threshold");
        } else if (arg == "--read-chunk-bytes") {
            options.gate.read_chunk_bytes = parse_size(
                need_value("--read-chunk-bytes"),
                "read chunk bytes");
        } else if (arg == "--diagnostics-format") {
            const auto value = need_value("--diagnostics-format");
            if (value == "ndjson") {
                options.gate.diagnostics_format =
                    dtmb::tools::ldpc_h_gate::DiagnosticsFormat::ndjson;
            } else if (value == "json") {
                options.gate.diagnostics_format =
                    dtmb::tools::ldpc_h_gate::DiagnosticsFormat::json;
            } else {
                throw std::invalid_argument("diagnostics format must be ndjson or json");
            }
        } else if (arg == "--diagnostics-out") {
            options.diagnostics_path = need_value("--diagnostics-out");
        } else if (arg == "--output-mode") {
            const auto value = need_value("--output-mode");
            if (value == "passthrough") {
                options.gate.output_mode =
                    dtmb::tools::ldpc_h_gate::OutputMode::passthrough;
            } else if (value == "shortlist") {
                options.gate.output_mode =
                    dtmb::tools::ldpc_h_gate::OutputMode::shortlist;
            } else {
                throw std::invalid_argument("output mode must be passthrough or shortlist");
            }
        } else if (arg == "--fec-frame-codewords") {
            options.gate.fec_frame_codewords = parse_size(
                need_value("--fec-frame-codewords"),
                "FEC frame codewords");
        } else if (arg == "--input-fec-frame-aligned") {
            options.gate.input_fec_frame_aligned = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            positional.push_back(arg);
        }
    }

    if (options.gate.fec_rate == 0 || options.alist_path.empty()) {
        throw std::invalid_argument("--fec-rate and --alist are required");
    }
    if (positional.size() > 1) {
        throw std::invalid_argument("too many positional inputs");
    }
    if (!positional.empty()) {
        options.input_path = positional.front();
    }
    return options;
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

std::ostream& diagnostics_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cerr;
    }
    file_holder = std::make_unique<std::ofstream>(path);
    if (!*file_holder) {
        throw std::runtime_error("failed to open diagnostics output: " + path);
    }
    return *file_holder;
}

}  // namespace

int main(int argc, char** argv) {
    std::ostream* diagnostics = &std::cerr;
    try {
        std::cout.imbue(std::locale::classic());
        std::cerr.imbue(std::locale::classic());
        const auto options = parse_args(argc, argv);
        dtmb::tools::configure_binary_stdio(options.input_path == "-", true);

        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> diagnostics_file;
        auto& input = input_stream(options.input_path, input_file);
        auto& diagnostics_output = diagnostics_stream(
            options.diagnostics_path,
            diagnostics_file);
        diagnostics = &diagnostics_output;
        diagnostics_output.imbue(std::locale::classic());

        const auto graph = dtmb::tools::ldpc_h_gate::load_clean_dtmb_graph(
            options.alist_path,
            options.gate.fec_rate);
        (void)dtmb::tools::ldpc_h_gate::run(
            input,
            std::cout,
            diagnostics_output,
            graph,
            options.gate);
        return 0;
    } catch (const std::exception& exc) {
        dtmb::tools::ldpc_h_gate::write_error_diagnostic(*diagnostics, exc.what());
        return 1;
    }
}
