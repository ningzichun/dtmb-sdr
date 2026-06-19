#pragma once

#include "dtmb/core.hpp"

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>

namespace dtmb::tools::ldpc_h_gate {

enum class DiagnosticsFormat {
    ndjson,
    json,
};

enum class OutputMode {
    passthrough,
    shortlist,
};

struct Options {
    std::size_t fec_rate = 0;
    std::size_t window_codewords = 24;
    std::size_t window_step_codewords = 3;
    double threshold = 0.44;
    std::size_t read_chunk_bytes = 64U * 1024U;
    DiagnosticsFormat diagnostics_format = DiagnosticsFormat::ndjson;
    OutputMode output_mode = OutputMode::passthrough;
    std::size_t fec_frame_codewords = 0;
    bool input_fec_frame_aligned = false;
};

struct WindowScore {
    std::size_t start_codeword = 0;
    std::size_t end_codeword = 0;
    double mean_syndrome_ratio = 0.0;
    double min_syndrome_ratio = 0.0;
    double max_syndrome_ratio = 0.0;
    std::size_t zero_syndrome_codewords = 0;
    std::string verdict;
};

struct Summary {
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t input_llrs = 0;
    std::size_t complete_codewords = 0;
    std::size_t scored_codewords = 0;
    std::size_t dirty_codewords = 0;
    std::size_t nonfinite_llrs = 0;
    std::size_t trailing_float_bytes = 0;
    std::size_t trailing_codeword_llrs = 0;
    std::size_t eligible_window_positions = 0;
    std::size_t scored_windows = 0;
    std::size_t dirty_windows = 0;
    std::size_t pass_windows = 0;
    std::size_t shortlist_bytes = 0;
    std::size_t max_buffered_bytes = 0;
    std::size_t memory_bound_bytes = 0;
    std::optional<WindowScore> best_window;
    std::string status;
    std::string gate_verdict;
    std::string verdict;
};

[[nodiscard]] dtmb::core::LdpcSparseGraph load_clean_dtmb_graph(
    const std::string& alist_path,
    std::size_t fec_rate);

[[nodiscard]] Summary run(
    std::istream& input,
    std::ostream& output,
    std::ostream& diagnostics,
    const dtmb::core::LdpcSparseGraph& clean_check_graph,
    const Options& options = {});

void write_error_diagnostic(
    std::ostream& diagnostics,
    const std::string& message);

}  // namespace dtmb::tools::ldpc_h_gate
