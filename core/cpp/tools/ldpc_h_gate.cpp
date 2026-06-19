#include "ldpc_h_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dtmb::tools::ldpc_h_gate {
namespace {

constexpr std::size_t kFloatBytes = sizeof(float);

struct DtmbLdpcProfile {
    std::size_t c = 0;
    std::size_t message_bits = 0;
    std::size_t transmitted_bits = 7488;
    std::size_t erased_parity_bits = 5;
    std::size_t circulant_size = 127;

    [[nodiscard]] std::size_t full_parity_bits() const noexcept {
        return c * circulant_size;
    }

    [[nodiscard]] std::size_t parity_bits() const noexcept {
        return full_parity_bits() - erased_parity_bits;
    }

    [[nodiscard]] std::size_t full_codeword_bits() const noexcept {
        return full_parity_bits() + message_bits;
    }
};

struct CodewordRecord {
    std::optional<std::size_t> syndrome_weight;
    std::vector<char> llr_bytes;
};

DtmbLdpcProfile profile_for_rate(std::size_t rate) {
    switch (rate) {
    case 1:
        return DtmbLdpcProfile{35, 3048};
    case 2:
        return DtmbLdpcProfile{23, 4572};
    case 3:
        return DtmbLdpcProfile{11, 6096};
    default:
        throw std::invalid_argument("fec rate must be 1, 2, or 3");
    }
}

dtmb::core::LdpcSparseGraph read_alist(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open alist: " + path);
    }

    std::size_t variable_count = 0;
    std::size_t check_count = 0;
    std::size_t max_variable_degree = 0;
    std::size_t max_check_degree = 0;
    if (!(input >> variable_count >> check_count
          >> max_variable_degree >> max_check_degree)) {
        throw std::runtime_error("failed to read alist header: " + path);
    }

    std::vector<std::size_t> variable_degrees(variable_count);
    std::vector<std::size_t> check_degrees(check_count);
    for (auto& degree : variable_degrees) {
        input >> degree;
    }
    for (auto& degree : check_degrees) {
        input >> degree;
    }

    std::size_t ignored = 0;
    for (std::size_t variable = 0; variable < variable_count; ++variable) {
        for (std::size_t edge = 0; edge < max_variable_degree; ++edge) {
            input >> ignored;
        }
    }

    std::vector<std::vector<std::size_t>> check_variables(check_count);
    for (std::size_t check = 0; check < check_count; ++check) {
        check_variables[check].reserve(check_degrees[check]);
        for (std::size_t edge = 0; edge < max_check_degree; ++edge) {
            std::size_t one_based_variable = 0;
            input >> one_based_variable;
            if (edge < check_degrees[check]) {
                if (one_based_variable == 0 || one_based_variable > variable_count) {
                    throw std::runtime_error("alist check variable is out of range");
                }
                check_variables[check].push_back(one_based_variable - 1U);
            }
        }
    }
    if (!input) {
        throw std::runtime_error("failed to read complete alist: " + path);
    }
    return dtmb::core::make_ldpc_sparse_graph(check_variables, variable_count);
}

std::size_t transmitted_index_for_full_variable(
    std::size_t variable,
    const DtmbLdpcProfile& profile) {
    if (variable < profile.erased_parity_bits) {
        throw std::logic_error("erased parity variables are not transmitted");
    }
    if (variable < profile.full_parity_bits()) {
        return variable - profile.erased_parity_bits;
    }
    return profile.parity_bits() + (variable - profile.full_parity_bits());
}

dtmb::core::LdpcSparseGraph clean_transmitted_check_graph(
    const dtmb::core::LdpcSparseGraph& graph,
    const DtmbLdpcProfile& profile) {
    if (graph.variable_count != profile.full_codeword_bits()) {
        throw std::runtime_error("alist variable count does not match DTMB LDPC profile");
    }

    std::vector<std::vector<std::size_t>> clean;
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        bool touches_erased = false;
        std::vector<std::size_t> variables;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1U];
             ++edge) {
            const auto variable = graph.edge_variables[edge];
            if (variable < profile.erased_parity_bits) {
                touches_erased = true;
                break;
            }
            variables.push_back(transmitted_index_for_full_variable(variable, profile));
        }
        if (!touches_erased && !variables.empty()) {
            clean.push_back(std::move(variables));
        }
    }
    if (clean.empty()) {
        throw std::runtime_error("no clean Appendix-B rows for this profile");
    }
    return dtmb::core::make_ldpc_sparse_graph(clean, profile.transmitted_bits);
}

void validate_graph(const dtmb::core::LdpcSparseGraph& graph) {
    if (graph.variable_count == 0 || graph.check_count() == 0) {
        throw std::invalid_argument("rolling H gate requires a non-empty clean-check graph");
    }
    std::vector<std::uint8_t> zero_bits(graph.variable_count, 0U);
    (void)dtmb::core::ldpc_syndrome_weight(zero_bits, graph);
}

void validate_options(
    const Options& options,
    const dtmb::core::LdpcSparseGraph& graph) {
    if (options.window_codewords == 0) {
        throw std::invalid_argument("window codewords must be positive");
    }
    if (options.window_step_codewords == 0) {
        throw std::invalid_argument("window step codewords must be positive");
    }
    if (!std::isfinite(options.threshold)
        || options.threshold < 0.0
        || options.threshold > 1.0) {
        throw std::invalid_argument("window threshold must be finite and within 0..1");
    }
    if (options.read_chunk_bytes == 0
        || options.read_chunk_bytes > static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::invalid_argument("read chunk bytes must fit a positive streamsize");
    }
    if (graph.variable_count > std::numeric_limits<std::size_t>::max() / kFloatBytes) {
        throw std::invalid_argument("codeword byte count overflows size_t");
    }
    if (options.output_mode == OutputMode::shortlist) {
        if (!options.input_fec_frame_aligned || options.fec_frame_codewords == 0) {
            throw std::invalid_argument(
                "shortlist mode requires --input-fec-frame-aligned and "
                "--fec-frame-codewords");
        }
        if ((options.window_codewords % options.fec_frame_codewords) != 0
            || (options.window_step_codewords % options.fec_frame_codewords) != 0) {
            throw std::invalid_argument(
                "shortlist window and step must be whole FEC frames");
        }
    }
}

std::size_t score_identity_codeword(
    std::span<const std::uint8_t> bits,
    const dtmb::core::LdpcSparseGraph& graph) {
    std::size_t weight = 0;
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        std::uint8_t parity = 0;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1U];
             ++edge) {
            parity ^= bits[graph.edge_variables[edge]];
        }
        weight += parity;
    }
    return weight;
}

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const auto ch : value) {
        if (ch == '"' || ch == '\\') {
            output << '\\' << ch;
        } else if (ch == '\n') {
            output << "\\n";
        } else if (ch == '\r') {
            output << "\\r";
        } else if (ch == '\t') {
            output << "\\t";
        } else {
            output << ch;
        }
    }
    output << '"';
}

const char* output_mode_name(OutputMode mode) {
    return mode == OutputMode::passthrough ? "passthrough" : "shortlist";
}

void write_window_object(
    std::ostream& output,
    const WindowScore& score,
    bool include_envelope) {
    output << std::setprecision(12) << '{';
    if (include_envelope) {
        output << "\"schema\":\"dtmb.ldpc_h_gate.v1\",\"event\":\"window\",";
    }
    output << "\"end_codeword\":" << score.end_codeword
           << ",\"max_syndrome_ratio\":" << score.max_syndrome_ratio
           << ",\"mean_syndrome_ratio\":" << score.mean_syndrome_ratio
           << ",\"min_syndrome_ratio\":" << score.min_syndrome_ratio
           << ",\"start_codeword\":" << score.start_codeword
           << ",\"verdict\":";
    write_json_string(output, score.verdict);
    output << ",\"zero_syndrome_codewords\":" << score.zero_syndrome_codewords
           << '}';
}

void emit_window(std::ostream& diagnostics, const WindowScore& score) {
    write_window_object(diagnostics, score, true);
    diagnostics << '\n';
    if (!diagnostics) {
        throw std::runtime_error("failed to write rolling H diagnostics");
    }
}

void write_terminal(
    std::ostream& diagnostics,
    const Summary& summary,
    const Options& options,
    const dtmb::core::LdpcSparseGraph& graph) {
    diagnostics << std::setprecision(12)
                << "{\"schema\":\"dtmb.ldpc_h_gate.v1\",\"event\":\"terminal\""
                << ",\"terminal\":true,\"verdict\":";
    write_json_string(diagnostics, summary.verdict);
    diagnostics << ",\"status\":";
    write_json_string(diagnostics, summary.status);
    diagnostics << ",\"gate_verdict\":";
    write_json_string(diagnostics, summary.gate_verdict);
    diagnostics << ",\"input_format\":\"llr-f32\""
                << ",\"output_mode\":";
    write_json_string(diagnostics, output_mode_name(options.output_mode));
    diagnostics << ",\"fec_rate_index\":" << options.fec_rate
                << ",\"codeword_bits\":" << graph.variable_count
                << ",\"clean_rows\":" << graph.check_count()
                << ",\"window_codewords\":" << options.window_codewords
                << ",\"window_step_codewords\":" << options.window_step_codewords
                << ",\"window_threshold\":" << options.threshold
                << ",\"input_bytes\":" << summary.input_bytes
                << ",\"output_bytes\":" << summary.output_bytes
                << ",\"input_llrs\":" << summary.input_llrs
                << ",\"complete_codewords\":" << summary.complete_codewords
                << ",\"scored_codewords\":" << summary.scored_codewords
                << ",\"dirty_codewords\":" << summary.dirty_codewords
                << ",\"nonfinite_llrs\":" << summary.nonfinite_llrs
                << ",\"trailing_float_bytes\":" << summary.trailing_float_bytes
                << ",\"trailing_codeword_llrs\":" << summary.trailing_codeword_llrs
                << ",\"eligible_window_positions\":" << summary.eligible_window_positions
                << ",\"scored_windows\":" << summary.scored_windows
                << ",\"dirty_windows\":" << summary.dirty_windows
                << ",\"pass_windows\":" << summary.pass_windows
                << ",\"shortlist_bytes\":" << summary.shortlist_bytes
                << ",\"max_buffered_bytes\":" << summary.max_buffered_bytes
                << ",\"memory_bound_bytes\":" << summary.memory_bound_bytes
                << ",\"input_byte_aligned\":"
                << (summary.trailing_float_bytes == 0 ? "true" : "false")
                << ",\"input_codeword_aligned\":"
                << (summary.trailing_float_bytes == 0
                        && summary.trailing_codeword_llrs == 0
                    ? "true"
                    : "false")
                << ",\"best_window\":";
    if (summary.best_window) {
        write_window_object(diagnostics, *summary.best_window, false);
    } else {
        diagnostics << "null";
    }
    diagnostics << "}\n";
    if (!diagnostics) {
        throw std::runtime_error("failed to write rolling H terminal diagnostic");
    }
}

std::size_t buffered_bytes(
    std::size_t read_chunk_bytes,
    std::size_t pending_float_bytes,
    std::size_t partial_codeword_bits,
    std::size_t partial_codeword_llr_bytes,
    const std::deque<CodewordRecord>& window) {
    auto total = read_chunk_bytes
        + pending_float_bytes
        + partial_codeword_bits
        + partial_codeword_llr_bytes;
    for (const auto& record : window) {
        total += sizeof(record.syndrome_weight) + record.llr_bytes.size();
    }
    return total;
}

std::size_t memory_bound_bytes(
    const Options& options,
    std::size_t codeword_bits) {
    const auto codeword_bytes = codeword_bits * kFloatBytes;
    const auto shortlist_window_bytes = options.output_mode == OutputMode::shortlist
        ? options.window_codewords * codeword_bytes
        : std::size_t{0};
    return options.read_chunk_bytes
        + kFloatBytes
        + codeword_bits
        + (options.output_mode == OutputMode::shortlist ? codeword_bytes : 0U)
        + options.window_codewords * sizeof(std::optional<std::size_t>)
        + shortlist_window_bytes;
}

void write_all(std::ostream& output, std::span<const char> bytes) {
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("failed to write LLR output stream");
    }
}

WindowScore score_window(
    const std::deque<CodewordRecord>& records,
    std::size_t start_codeword,
    std::size_t clean_rows,
    double threshold) {
    std::size_t total = 0;
    auto minimum = std::numeric_limits<std::size_t>::max();
    std::size_t maximum = 0;
    std::size_t zero_count = 0;
    for (const auto& record : records) {
        const auto weight = *record.syndrome_weight;
        total += weight;
        minimum = std::min(minimum, weight);
        maximum = std::max(maximum, weight);
        zero_count += weight == 0 ? 1U : 0U;
    }

    WindowScore score;
    score.start_codeword = start_codeword;
    score.end_codeword = start_codeword + records.size();
    score.mean_syndrome_ratio = static_cast<double>(total)
        / static_cast<double>(records.size() * clean_rows);
    score.min_syndrome_ratio = static_cast<double>(minimum)
        / static_cast<double>(clean_rows);
    score.max_syndrome_ratio = static_cast<double>(maximum)
        / static_cast<double>(clean_rows);
    score.zero_syndrome_codewords = zero_count;
    score.verdict = score.mean_syndrome_ratio <= threshold ? "pass" : "shortlist";
    return score;
}

void finalize_summary(Summary& summary) {
    summary.status = summary.nonfinite_llrs != 0
            || summary.trailing_float_bytes != 0
            || summary.trailing_codeword_llrs != 0
        ? "degraded"
        : "ok";
    summary.gate_verdict = summary.scored_windows == 0
        ? "no_complete_windows"
        : summary.pass_windows != 0 ? "pass" : "shortlist";

    if (summary.nonfinite_llrs != 0) {
        summary.verdict = "dirty_input";
    } else if (summary.trailing_float_bytes != 0) {
        summary.verdict = "misaligned_input_bytes";
    } else if (summary.complete_codewords == 0) {
        summary.verdict = "short_input";
    } else if (summary.trailing_codeword_llrs != 0) {
        summary.verdict = "misaligned_input_codeword";
    } else {
        summary.verdict = summary.gate_verdict;
    }
}

}  // namespace

dtmb::core::LdpcSparseGraph load_clean_dtmb_graph(
    const std::string& alist_path,
    std::size_t fec_rate) {
    const auto profile = profile_for_rate(fec_rate);
    return clean_transmitted_check_graph(read_alist(alist_path), profile);
}

Summary run(
    std::istream& input,
    std::ostream& output,
    std::ostream& diagnostics,
    const dtmb::core::LdpcSparseGraph& clean_check_graph,
    const Options& options) {
    validate_graph(clean_check_graph);
    validate_options(options, clean_check_graph);

    Summary summary;
    summary.memory_bound_bytes = memory_bound_bytes(options, clean_check_graph.variable_count);
    std::vector<char> input_chunk(options.read_chunk_bytes);
    std::array<char, kFloatBytes> pending_float{};
    std::size_t pending_float_bytes = 0;
    std::vector<std::uint8_t> codeword_bits;
    codeword_bits.reserve(clean_check_graph.variable_count);
    std::vector<char> codeword_llr_bytes;
    if (options.output_mode == OutputMode::shortlist) {
        codeword_llr_bytes.reserve(clean_check_graph.variable_count * kFloatBytes);
    }
    bool codeword_dirty = false;
    std::deque<CodewordRecord> rolling_window;

    const auto update_buffered = [&] {
        summary.max_buffered_bytes = std::max(
            summary.max_buffered_bytes,
            buffered_bytes(
                input_chunk.size(),
                pending_float_bytes,
                codeword_bits.size(),
                codeword_llr_bytes.size(),
                rolling_window));
    };

    const auto complete_codeword = [&] {
        ++summary.complete_codewords;
        CodewordRecord record;
        if (codeword_dirty) {
            ++summary.dirty_codewords;
        } else {
            record.syndrome_weight = score_identity_codeword(
                codeword_bits,
                clean_check_graph);
            ++summary.scored_codewords;
        }
        if (options.output_mode == OutputMode::shortlist) {
            record.llr_bytes = std::move(codeword_llr_bytes);
        }
        if (rolling_window.size() == options.window_codewords) {
            rolling_window.pop_front();
        }
        rolling_window.push_back(std::move(record));

        const auto start_codeword = summary.complete_codewords >= options.window_codewords
            ? summary.complete_codewords - options.window_codewords
            : std::size_t{0};
        if (rolling_window.size() == options.window_codewords
            && (start_codeword % options.window_step_codewords) == 0) {
            ++summary.eligible_window_positions;
            const auto clean = std::all_of(
                rolling_window.begin(),
                rolling_window.end(),
                [](const CodewordRecord& value) {
                    return value.syndrome_weight.has_value();
                });
            if (!clean) {
                ++summary.dirty_windows;
            } else {
                auto score = score_window(
                    rolling_window,
                    start_codeword,
                    clean_check_graph.check_count(),
                    options.threshold);
                ++summary.scored_windows;
                if (score.verdict == "pass") {
                    ++summary.pass_windows;
                    if (options.output_mode == OutputMode::shortlist) {
                        for (const auto& value : rolling_window) {
                            write_all(output, value.llr_bytes);
                            summary.output_bytes += value.llr_bytes.size();
                            summary.shortlist_bytes += value.llr_bytes.size();
                        }
                    }
                }
                if (!summary.best_window
                    || score.mean_syndrome_ratio
                        < summary.best_window->mean_syndrome_ratio) {
                    summary.best_window = score;
                }
                if (options.diagnostics_format == DiagnosticsFormat::ndjson) {
                    emit_window(diagnostics, score);
                }
            }
        }

        codeword_bits.clear();
        codeword_dirty = false;
        if (options.output_mode == OutputMode::shortlist) {
            codeword_llr_bytes.clear();
            codeword_llr_bytes.reserve(clean_check_graph.variable_count * kFloatBytes);
        }
        update_buffered();
    };

    update_buffered();
    while (true) {
        input.read(input_chunk.data(), static_cast<std::streamsize>(input_chunk.size()));
        const auto bytes_read = static_cast<std::size_t>(input.gcount());
        if (bytes_read == 0) {
            if (input.bad()) {
                throw std::runtime_error("failed to read LLR input stream");
            }
            break;
        }

        summary.input_bytes += bytes_read;
        const auto bytes = std::span<const char>(input_chunk.data(), bytes_read);
        if (options.output_mode == OutputMode::passthrough) {
            write_all(output, bytes);
            summary.output_bytes += bytes.size();
        }

        for (const auto byte : bytes) {
            pending_float[pending_float_bytes++] = byte;
            if (pending_float_bytes != kFloatBytes) {
                continue;
            }

            float llr = 0.0F;
            std::memcpy(&llr, pending_float.data(), kFloatBytes);
            ++summary.input_llrs;
            if (!std::isfinite(llr)) {
                ++summary.nonfinite_llrs;
                codeword_dirty = true;
            }
            codeword_bits.push_back(llr < 0.0F ? 1U : 0U);
            if (options.output_mode == OutputMode::shortlist) {
                codeword_llr_bytes.insert(
                    codeword_llr_bytes.end(),
                    pending_float.begin(),
                    pending_float.end());
            }
            pending_float_bytes = 0;
            if (codeword_bits.size() == clean_check_graph.variable_count) {
                complete_codeword();
            }
        }
        update_buffered();

        if (bytes_read < input_chunk.size()) {
            if (input.bad()) {
                throw std::runtime_error("failed to read LLR input stream");
            }
            break;
        }
    }

    summary.trailing_float_bytes = pending_float_bytes;
    summary.trailing_codeword_llrs = codeword_bits.size();
    update_buffered();
    finalize_summary(summary);
    write_terminal(diagnostics, summary, options, clean_check_graph);
    output.flush();
    diagnostics.flush();
    if (!output) {
        throw std::runtime_error("failed to flush LLR output stream");
    }
    if (!diagnostics) {
        throw std::runtime_error("failed to flush rolling H diagnostics");
    }
    return summary;
}

void write_error_diagnostic(
    std::ostream& diagnostics,
    const std::string& message) {
    diagnostics << "{\"schema\":\"dtmb.ldpc_h_gate.v1\",\"event\":\"terminal\","
                   "\"terminal\":true,\"verdict\":\"error\","
                   "\"status\":\"error\",\"error\":";
    write_json_string(diagnostics, message);
    diagnostics << "}\n";
}

}  // namespace dtmb::tools::ldpc_h_gate
