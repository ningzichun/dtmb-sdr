#include "binary_stdio.hpp"
#include "dtmb/core.hpp"
#include "ldpc_h_gate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class SelectionMode {
    average,
    codeword,
    frame,
    weighted_average,
};

struct Options {
    std::size_t fec_rate = 0;
    std::string alist_path;
    std::size_t codewords_per_frame = 0;
    SelectionMode selection_mode = SelectionMode::codeword;
    std::string diagnostics_path;
    std::string output_path;
    std::vector<std::string> input_paths;
};

struct InputState {
    std::string path;
    std::ifstream stream;
    std::vector<char> frame_bytes;
    std::vector<std::size_t> syndrome_weights;
    bool complete = false;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --fec-rate 1|2|3 --alist PATH --codewords-per-frame N"
        << " [--selection average|codeword|frame|weighted-average]"
        << " [--diagnostics-out PATH]"
        << " OUTPUT INPUT...\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

SelectionMode parse_selection_mode(const std::string& text) {
    if (text == "average") {
        return SelectionMode::average;
    }
    if (text == "codeword") {
        return SelectionMode::codeword;
    }
    if (text == "frame") {
        return SelectionMode::frame;
    }
    if (text == "weighted-average") {
        return SelectionMode::weighted_average;
    }
    throw std::invalid_argument(
        "--selection must be average, codeword, frame, or weighted-average");
}

const char* selection_mode_name(SelectionMode mode) {
    switch (mode) {
    case SelectionMode::average:
        return "average";
    case SelectionMode::codeword:
        return "codeword";
    case SelectionMode::frame:
        return "frame";
    case SelectionMode::weighted_average:
        return "weighted-average";
    }
    return "unknown";
}

Options parse_args(int argc, char** argv) {
    Options options;
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto need_value = [&](const char* name) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[index];
        };
        if (arg == "--fec-rate") {
            options.fec_rate = parse_size(need_value("--fec-rate"), "fec rate");
        } else if (arg == "--alist") {
            options.alist_path = need_value("--alist");
        } else if (arg == "--codewords-per-frame") {
            options.codewords_per_frame =
                parse_size(need_value("--codewords-per-frame"), "codewords per frame");
        } else if (arg == "--selection") {
            options.selection_mode = parse_selection_mode(need_value("--selection"));
        } else if (arg == "--diagnostics-out") {
            options.diagnostics_path = need_value("--diagnostics-out");
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (!arg.empty() && arg.front() == '-' && arg != "-") {
            throw std::invalid_argument("unknown argument: " + arg);
        } else {
            positional.push_back(arg);
        }
    }
    if (options.fec_rate == 0 || options.alist_path.empty()
        || options.codewords_per_frame == 0 || positional.size() < 3) {
        usage(argv[0]);
        throw std::invalid_argument(
            "fec rate, alist, codewords per frame, output, and at least two inputs are required");
    }
    options.output_path = positional.front();
    options.input_paths.assign(positional.begin() + 1, positional.end());
    return options;
}

std::ostream& output_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& holder) {
    if (path == "-") {
        dtmb::tools::configure_binary_stdio(false, true);
        return std::cout;
    }
    holder = std::make_unique<std::ofstream>(path, std::ios::binary);
    if (!*holder) {
        throw std::runtime_error("failed to open output: " + path);
    }
    return *holder;
}

std::ostream& diagnostics_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& holder) {
    if (path.empty()) {
        return std::clog;
    }
    holder = std::make_unique<std::ofstream>(path);
    if (!*holder) {
        throw std::runtime_error("failed to open diagnostics: " + path);
    }
    return *holder;
}

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const auto ch : value) {
        if (ch == '"' || ch == '\\') {
            output << '\\' << ch;
        } else if (ch == '\n') {
            output << "\\n";
        } else {
            output << ch;
        }
    }
    output << '"';
}

std::vector<std::uint8_t> hard_bits_from_llr_bytes(std::span<const char> bytes) {
    if ((bytes.size() % sizeof(float)) != 0) {
        throw std::runtime_error("LLR byte count must be float-aligned");
    }
    std::vector<std::uint8_t> bits(bytes.size() / sizeof(float));
    for (std::size_t index = 0; index < bits.size(); ++index) {
        float value = 0.0F;
        std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
        if (!std::isfinite(value)) {
            throw std::runtime_error("LLR input contains non-finite values");
        }
        bits[index] = value < 0.0F ? 1U : 0U;
    }
    return bits;
}

std::size_t score_codeword(
    std::span<const char> codeword_bytes,
    const dtmb::core::LdpcSparseGraph& clean_graph) {
    const auto bits = hard_bits_from_llr_bytes(codeword_bytes);
    return dtmb::core::ldpc_syndrome_weight(bits, clean_graph);
}

void average_codeword(
    std::span<const InputState> inputs,
    std::size_t codeword,
    std::size_t codeword_llrs,
    std::size_t codeword_bytes,
    std::span<char> output_bytes) {
    const auto offset = codeword * codeword_bytes;
    for (std::size_t llr_index = 0; llr_index < codeword_llrs; ++llr_index) {
        double sum = 0.0;
        for (const auto& input : inputs) {
            float value = 0.0F;
            std::memcpy(
                &value,
                input.frame_bytes.data() + static_cast<std::ptrdiff_t>(
                    offset + llr_index * sizeof(float)),
                sizeof(float));
            if (!std::isfinite(value)) {
                throw std::runtime_error("LLR input contains non-finite values");
            }
            sum += static_cast<double>(value);
        }
        const auto averaged =
            static_cast<float>(sum / static_cast<double>(inputs.size()));
        std::memcpy(
            output_bytes.data() + static_cast<std::ptrdiff_t>(llr_index * sizeof(float)),
            &averaged,
            sizeof(float));
    }
}

void weighted_average_codeword(
    std::span<const InputState> inputs,
    std::size_t codeword,
    std::size_t codeword_llrs,
    std::size_t codeword_bytes,
    std::span<char> output_bytes) {
    const auto offset = codeword * codeword_bytes;
    std::vector<double> weights(inputs.size(), 0.0);
    std::size_t zero_syndrome_inputs = 0;
    for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const auto syndrome = inputs[input_index].syndrome_weights[codeword];
        if (syndrome == 0) {
            ++zero_syndrome_inputs;
        }
    }
    double weight_sum = 0.0;
    for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const auto syndrome = inputs[input_index].syndrome_weights[codeword];
        if (zero_syndrome_inputs != 0) {
            weights[input_index] = syndrome == 0
                ? 1.0 / static_cast<double>(zero_syndrome_inputs)
                : 0.0;
        } else {
            weights[input_index] = 1.0 / static_cast<double>(syndrome);
        }
        weight_sum += weights[input_index];
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        throw std::runtime_error("invalid weighted-average LLR weights");
    }

    for (std::size_t llr_index = 0; llr_index < codeword_llrs; ++llr_index) {
        double sum = 0.0;
        for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
            float value = 0.0F;
            std::memcpy(
                &value,
                inputs[input_index].frame_bytes.data() + static_cast<std::ptrdiff_t>(
                    offset + llr_index * sizeof(float)),
                sizeof(float));
            if (!std::isfinite(value)) {
                throw std::runtime_error("LLR input contains non-finite values");
            }
            sum += weights[input_index] * static_cast<double>(value);
        }
        const auto averaged = static_cast<float>(sum / weight_sum);
        std::memcpy(
            output_bytes.data() + static_cast<std::ptrdiff_t>(llr_index * sizeof(float)),
            &averaged,
            sizeof(float));
    }
}

bool read_frame(InputState& input, std::size_t frame_bytes) {
    input.stream.read(input.frame_bytes.data(), static_cast<std::streamsize>(frame_bytes));
    const auto count = static_cast<std::size_t>(input.stream.gcount());
    if (count == 0) {
        input.complete = false;
        return false;
    }
    if (count != frame_bytes) {
        throw std::runtime_error("partial LLR frame in input: " + input.path);
    }
    input.complete = true;
    return true;
}

void emit_diagnostic(
    std::ostream& output,
    std::size_t frame_index,
    const Options& options,
    std::span<const InputState> inputs,
    std::span<const std::size_t> selected_inputs) {
    output << "{\"schema\":\"dtmb.llr_frame_select.v1\""
           << ",\"frame_index\":" << frame_index
           << ",\"selection_mode\":\"" << selection_mode_name(options.selection_mode) << "\""
           << ",\"selected_inputs\":[";
    for (std::size_t index = 0; index < selected_inputs.size(); ++index) {
        if (index) {
            output << ',';
        }
        output << selected_inputs[index];
    }
    output << "],\"input_syndrome_weights\":[";
    for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        if (input_index) {
            output << ',';
        }
        output << '[';
        for (std::size_t codeword = 0;
             codeword < inputs[input_index].syndrome_weights.size();
             ++codeword) {
            if (codeword) {
                output << ',';
            }
            output << inputs[input_index].syndrome_weights[codeword];
        }
        output << ']';
    }
    output << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto clean_graph =
            dtmb::tools::ldpc_h_gate::load_clean_dtmb_graph(
                options.alist_path,
                options.fec_rate);
        const auto codeword_llrs = clean_graph.variable_count;
        const auto codeword_bytes = codeword_llrs * sizeof(float);
        const auto frame_bytes = codeword_bytes * options.codewords_per_frame;

        std::vector<InputState> inputs;
        inputs.reserve(options.input_paths.size());
        for (const auto& path : options.input_paths) {
            InputState state;
            state.path = path;
            state.stream.open(path, std::ios::binary);
            if (!state.stream) {
                throw std::runtime_error("failed to open input: " + path);
            }
            state.frame_bytes.assign(frame_bytes, 0);
            state.syndrome_weights.assign(options.codewords_per_frame, 0);
            inputs.push_back(std::move(state));
        }

        std::unique_ptr<std::ofstream> output_holder;
        auto& output = output_stream(options.output_path, output_holder);
        std::unique_ptr<std::ofstream> diagnostics_holder;
        auto& diagnostics = diagnostics_stream(options.diagnostics_path, diagnostics_holder);

        std::vector<std::size_t> selected_codewords_per_input(inputs.size(), 0);
        std::vector<std::size_t> selected_frames_per_input(inputs.size(), 0);
        std::vector<std::size_t> selected_inputs(options.codewords_per_frame, 0);
        std::vector<char> averaged_codeword_bytes(codeword_bytes, 0);
        std::size_t frames = 0;
        std::size_t codewords = 0;
        std::size_t averaged_codewords = 0;
        std::uint64_t selected_syndrome_weight = 0;

        while (true) {
            std::size_t complete_inputs = 0;
            for (auto& input : inputs) {
                complete_inputs += read_frame(input, frame_bytes) ? 1U : 0U;
            }
            if (complete_inputs == 0) {
                break;
            }
            if (complete_inputs != inputs.size()) {
                throw std::runtime_error("LLR inputs ended at different frame counts");
            }

            ++frames;
            for (auto& input : inputs) {
                for (std::size_t codeword = 0; codeword < options.codewords_per_frame;
                     ++codeword) {
                    const auto offset = codeword * codeword_bytes;
                    input.syndrome_weights[codeword] = score_codeword(
                        std::span<const char>(
                            input.frame_bytes.data() + static_cast<std::ptrdiff_t>(offset),
                            codeword_bytes),
                        clean_graph);
                }
            }

            if (options.selection_mode == SelectionMode::frame) {
                std::size_t best_input = 0;
                std::size_t best_weight = std::numeric_limits<std::size_t>::max();
                for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
                    const auto total = std::accumulate(
                        inputs[input_index].syndrome_weights.begin(),
                        inputs[input_index].syndrome_weights.end(),
                        std::size_t{0});
                    if (total < best_weight) {
                        best_weight = total;
                        best_input = input_index;
                    }
                }
                std::fill(selected_inputs.begin(), selected_inputs.end(), best_input);
                ++selected_frames_per_input[best_input];
            } else {
                for (std::size_t codeword = 0; codeword < options.codewords_per_frame;
                     ++codeword) {
                    std::size_t best_input = 0;
                    std::size_t best_weight = std::numeric_limits<std::size_t>::max();
                    for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
                        const auto weight = inputs[input_index].syndrome_weights[codeword];
                        if (weight < best_weight) {
                            best_weight = weight;
                            best_input = input_index;
                        }
                    }
                    selected_inputs[codeword] = best_input;
                }
                if (std::all_of(
                        selected_inputs.begin(),
                        selected_inputs.end(),
                        [&](std::size_t value) {
                            return value == selected_inputs.front();
                        })) {
                    ++selected_frames_per_input[selected_inputs.front()];
                }
            }

            for (std::size_t codeword = 0; codeword < options.codewords_per_frame;
                 ++codeword) {
                const auto offset = codeword * codeword_bytes;
                if (options.selection_mode == SelectionMode::average) {
                    average_codeword(
                        inputs,
                        codeword,
                        codeword_llrs,
                        codeword_bytes,
                        averaged_codeword_bytes);
                    output.write(
                        averaged_codeword_bytes.data(),
                        static_cast<std::streamsize>(codeword_bytes));
                    ++averaged_codewords;
                } else if (options.selection_mode == SelectionMode::weighted_average) {
                    weighted_average_codeword(
                        inputs,
                        codeword,
                        codeword_llrs,
                        codeword_bytes,
                        averaged_codeword_bytes);
                    output.write(
                        averaged_codeword_bytes.data(),
                        static_cast<std::streamsize>(codeword_bytes));
                    ++averaged_codewords;
                } else {
                    const auto input_index = selected_inputs[codeword];
                    output.write(
                        inputs[input_index].frame_bytes.data()
                            + static_cast<std::ptrdiff_t>(offset),
                        static_cast<std::streamsize>(codeword_bytes));
                    ++selected_codewords_per_input[input_index];
                    selected_syndrome_weight += inputs[input_index].syndrome_weights[codeword];
                }
                if (!output) {
                    throw std::runtime_error("failed while writing selected LLR output");
                }
                ++codewords;
            }

            if (!options.diagnostics_path.empty()) {
                emit_diagnostic(diagnostics, frames, options, inputs, selected_inputs);
            }
        }

        std::cerr << "schema=dtmb.llr_frame_select.v1\n"
                  << "frames=" << frames << '\n'
                  << "codewords=" << codewords << '\n'
                  << "input_count=" << inputs.size() << '\n'
                  << "fec_rate=" << options.fec_rate << '\n'
                  << "codewords_per_frame=" << options.codewords_per_frame << '\n'
                  << "selection_mode=" << selection_mode_name(options.selection_mode) << '\n'
                  << "selected_syndrome_weight=" << selected_syndrome_weight << '\n'
                  << "averaged_codewords=" << averaged_codewords << '\n';
        for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index) {
            std::cerr << "input_" << input_index << "_selected_codewords="
                      << selected_codewords_per_input[input_index] << '\n'
                      << "input_" << input_index << "_selected_whole_frames="
                      << selected_frames_per_input[input_index] << '\n';
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
