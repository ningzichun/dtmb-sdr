#include "dtmb/core.hpp"
#include "ldpc_h_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using dtmb::tools::ldpc_h_gate::DiagnosticsFormat;
using dtmb::tools::ldpc_h_gate::Options;
using dtmb::tools::ldpc_h_gate::OutputMode;

void check(bool condition, std::string_view message = "LDPC H gate test check failed") {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::string llr_bytes_from_bits(const std::vector<std::uint8_t>& bits) {
    std::string bytes(bits.size() * sizeof(float), '\0');
    for (std::size_t index = 0; index < bits.size(); ++index) {
        const float value = bits[index] == 0 ? 2.0F : -2.0F;
        std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
    }
    return bytes;
}

std::vector<std::uint8_t> deterministic_bits(std::size_t count) {
    std::vector<std::uint8_t> bits(count);
    std::uint32_t state = 0x6d2b79f5U;
    for (auto& bit : bits) {
        state = state * 1664525U + 1013904223U;
        bit = static_cast<std::uint8_t>((state >> 31U) & 1U);
    }
    return bits;
}

dtmb::core::LdpcSparseGraph small_graph() {
    return dtmb::core::make_ldpc_sparse_graph(
        {
            {0, 1, 2},
            {2, 3, 4},
            {4, 5, 6},
            {0, 6, 7},
        },
        8);
}

std::vector<std::string> lines(const std::string& text) {
    std::vector<std::string> result;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        result.push_back(line);
    }
    return result;
}

double json_number(const std::string& line, std::string_view key) {
    const auto marker = std::string{"\""} + std::string{key} + "\":";
    const auto start = line.find(marker);
    check(start != std::string::npos);
    std::size_t parsed = 0;
    const auto value = std::stod(line.substr(start + marker.size()), &parsed);
    check(parsed != 0);
    return value;
}

std::size_t json_size(const std::string& line, std::string_view key) {
    return static_cast<std::size_t>(json_number(line, key));
}

void test_dtmb_rate2_windows_match_core_h_scorer() {
    const auto graph = dtmb::tools::ldpc_h_gate::load_clean_dtmb_graph(
        DTMB_LDPC_RATE2_ALIST,
        2);
    constexpr std::size_t codewords = 30;
    const auto bits = deterministic_bits(codewords * graph.variable_count);
    const auto expected = dtmb::core::ldpc_score_hard_bit_candidate(
        bits,
        graph,
        dtmb::core::LdpcHardCandidateScoreOptions{
            0,
            0,
            1,
            dtmb::core::LdpcStreamBitOrder::identity,
        });
    const auto bytes = llr_bytes_from_bits(bits);

    std::istringstream input(bytes);
    std::ostringstream output;
    std::ostringstream diagnostics;
    Options options;
    options.fec_rate = 2;
    options.read_chunk_bytes = 7;
    const auto summary = dtmb::tools::ldpc_h_gate::run(
        input,
        output,
        diagnostics,
        graph,
        options);

    check(output.str() == bytes);
    check(summary.complete_codewords == codewords);
    check(summary.scored_codewords == codewords);
    check(summary.scored_windows == 3);
    check(summary.pass_windows == 0);
    check(summary.gate_verdict == "shortlist");
    check(summary.max_buffered_bytes <= summary.memory_bound_bytes);

    const auto diagnostic_lines = lines(diagnostics.str());
    check(diagnostic_lines.size() == summary.scored_windows + 1);
    for (std::size_t row = 0; row < summary.scored_windows; ++row) {
        const auto start = row * options.window_step_codewords;
        const auto end = start + options.window_codewords;
        std::size_t total = 0;
        auto minimum = std::numeric_limits<std::size_t>::max();
        std::size_t maximum = 0;
        std::size_t zero_count = 0;
        for (std::size_t index = start; index < end; ++index) {
            const auto weight = expected.syndrome_weights[index];
            total += weight;
            minimum = std::min(minimum, weight);
            maximum = std::max(maximum, weight);
            zero_count += weight == 0 ? 1U : 0U;
        }
        check(json_size(diagnostic_lines[row], "start_codeword") == start);
        check(json_size(diagnostic_lines[row], "end_codeword") == end);
        check(json_size(diagnostic_lines[row], "zero_syndrome_codewords") == zero_count);
        check(std::abs(
            json_number(diagnostic_lines[row], "mean_syndrome_ratio")
            - static_cast<double>(total)
                / static_cast<double>(options.window_codewords * graph.check_count()))
            < 1.0e-12);
        check(std::abs(
            json_number(diagnostic_lines[row], "min_syndrome_ratio")
            - static_cast<double>(minimum) / static_cast<double>(graph.check_count()))
            < 1.0e-12);
        check(std::abs(
            json_number(diagnostic_lines[row], "max_syndrome_ratio")
            - static_cast<double>(maximum) / static_cast<double>(graph.check_count()))
            < 1.0e-12);
    }
    check(diagnostic_lines.back().find(R"("event":"terminal")") != std::string::npos);
}

void test_arbitrary_chunk_boundaries_preserve_bytes_and_results() {
    const auto graph = small_graph();
    const auto bits = deterministic_bits(40 * graph.variable_count);
    const auto bytes = llr_bytes_from_bits(bits);
    std::optional<dtmb::tools::ldpc_h_gate::Summary> reference;
    for (const auto chunk_bytes : std::array<std::size_t, 8>{1, 2, 3, 4, 5, 7, 31, 4096}) {
        std::istringstream input(bytes);
        std::ostringstream output;
        std::ostringstream diagnostics;
        Options options;
        options.window_codewords = 4;
        options.window_step_codewords = 2;
        options.threshold = 0.5;
        options.read_chunk_bytes = chunk_bytes;
        const auto summary = dtmb::tools::ldpc_h_gate::run(
            input,
            output,
            diagnostics,
            graph,
            options);
        check(output.str() == bytes);
        check(summary.max_buffered_bytes <= summary.memory_bound_bytes);
        if (!reference) {
            reference = summary;
        } else {
            check(summary.complete_codewords == reference->complete_codewords);
            check(summary.scored_windows == reference->scored_windows);
            check(summary.pass_windows == reference->pass_windows);
            check(summary.best_window->start_codeword
                == reference->best_window->start_codeword);
            check(summary.best_window->mean_syndrome_ratio
                == reference->best_window->mean_syndrome_ratio);
        }
    }
}

void test_dirty_short_and_misaligned_inputs_are_nonfatal() {
    const auto graph = small_graph();
    auto bytes = llr_bytes_from_bits(deterministic_bits(10));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(bytes.data() + 3 * sizeof(float), &nan, sizeof(float));
    bytes.push_back('\x12');
    bytes.push_back('\x34');

    std::istringstream input(bytes);
    std::ostringstream output;
    std::ostringstream diagnostics;
    Options options;
    options.window_codewords = 2;
    options.window_step_codewords = 1;
    options.read_chunk_bytes = 3;
    const auto summary = dtmb::tools::ldpc_h_gate::run(
        input,
        output,
        diagnostics,
        graph,
        options);

    check(output.str() == bytes);
    check(summary.complete_codewords == 1);
    check(summary.dirty_codewords == 1);
    check(summary.nonfinite_llrs == 1);
    check(summary.trailing_float_bytes == 2);
    check(summary.trailing_codeword_llrs == 2);
    check(summary.status == "degraded");
    check(summary.verdict == "dirty_input");
    check(diagnostics.str().find(R"("input_byte_aligned":false)") != std::string::npos);

    std::istringstream short_input(std::string{});
    std::ostringstream short_output;
    std::ostringstream short_diagnostics;
    const auto short_summary = dtmb::tools::ldpc_h_gate::run(
        short_input,
        short_output,
        short_diagnostics,
        graph,
        options);
    check(short_summary.verdict == "short_input");
    check(short_summary.gate_verdict == "no_complete_windows");
}

void test_terminal_json_is_single_bounded_summary() {
    const auto graph = small_graph();
    const auto bytes = llr_bytes_from_bits(deterministic_bits(100 * graph.variable_count));
    std::istringstream input(bytes);
    std::ostringstream output;
    std::ostringstream diagnostics;
    Options options;
    options.window_codewords = 4;
    options.window_step_codewords = 1;
    options.diagnostics_format = DiagnosticsFormat::json;
    const auto summary = dtmb::tools::ldpc_h_gate::run(
        input,
        output,
        diagnostics,
        graph,
        options);
    check(lines(diagnostics.str()).size() == 1);
    check(summary.scored_windows == 97);
    check(summary.max_buffered_bytes <= summary.memory_bound_bytes);
}

void test_shortlist_requires_and_preserves_fec_frame_alignment() {
    const auto graph = small_graph();
    const auto bytes = llr_bytes_from_bits(deterministic_bits(10 * graph.variable_count));

    Options invalid;
    invalid.window_codewords = 4;
    invalid.window_step_codewords = 2;
    invalid.threshold = 1.0;
    invalid.output_mode = OutputMode::shortlist;
    bool threw = false;
    try {
        std::istringstream input(bytes);
        std::ostringstream output;
        std::ostringstream diagnostics;
        (void)dtmb::tools::ldpc_h_gate::run(input, output, diagnostics, graph, invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw);

    Options options = invalid;
    options.fec_frame_codewords = 2;
    options.input_fec_frame_aligned = true;
    std::istringstream input(bytes);
    std::ostringstream output;
    std::ostringstream diagnostics;
    const auto summary = dtmb::tools::ldpc_h_gate::run(
        input,
        output,
        diagnostics,
        graph,
        options);

    std::string expected;
    const auto codeword_bytes = graph.variable_count * sizeof(float);
    for (std::size_t start = 0; start + options.window_codewords <= 10;
         start += options.window_step_codewords) {
        expected.append(
            bytes,
            start * codeword_bytes,
            options.window_codewords * codeword_bytes);
    }
    check(summary.pass_windows == 4);
    check(output.str() == expected);
    check(summary.output_bytes == expected.size());
    check(summary.shortlist_bytes == expected.size());
    check(summary.max_buffered_bytes <= summary.memory_bound_bytes);
}

}  // namespace

int main() {
    test_dtmb_rate2_windows_match_core_h_scorer();
    test_arbitrary_chunk_boundaries_preserve_bytes_and_results();
    test_dirty_short_and_misaligned_inputs_are_nonfatal();
    test_terminal_json_is_single_bounded_summary();
    test_shortlist_requires_and_preserves_fec_frame_alignment();
    return 0;
}
