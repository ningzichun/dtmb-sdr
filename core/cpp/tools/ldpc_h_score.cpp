#include "dtmb/core.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct DtmbLdpcProfile {
    std::size_t fec_rate = 0;
    std::size_t c = 0;
    std::size_t k = 0;
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

struct CandidateScore {
    std::size_t bit_shift = 0;
    std::string bit_plane_permutation;
    std::string stream_bit_order;
    std::size_t codewords = 0;
    std::size_t scored_bits = 0;
    std::size_t unused_bits = 0;
    std::size_t clean_rows = 0;
    std::size_t worker_count = 0;
    double mean_syndrome_ratio = 0.0;
    double min_syndrome_ratio = 0.0;
    double max_syndrome_ratio = 0.0;
    std::size_t zero_syndrome_codewords = 0;
    std::vector<std::size_t> sample_syndrome_weights;
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

struct WindowReport {
    std::size_t window_codewords = 0;
    std::size_t window_step_codewords = 0;
    std::optional<double> threshold;
    std::vector<WindowScore> rows;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --fec-rate 1|2|3 --alist PATH"
        << " [--input-format hard-u8|llr-f32] [--max-codewords N]"
        << " [--bit-shifts CSV] [--bits-per-symbol N]"
        << " [--bit-plane-permutations CSV]"
        << " [--stream-bit-orders CSV] [--workers N] [--top N]"
        << " [--window-codewords N] [--window-step-codewords N]"
        << " [--window-threshold RATIO]"
        << " [--json-out PATH]"
        << " [input|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

double parse_ratio(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

std::vector<std::string> split_csv_strings(const std::string& text) {
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), token.end());
        if (!token.empty()) {
            values.push_back(token);
        }
    }
    return values;
}

std::vector<std::size_t> split_csv_sizes(const std::string& text, const char* field) {
    std::vector<std::size_t> values;
    for (const auto& token : split_csv_strings(text)) {
        values.push_back(parse_size(token, field));
    }
    if (values.empty()) {
        throw std::invalid_argument(std::string("empty ") + field + " list");
    }
    return values;
}

DtmbLdpcProfile profile_for_rate(std::size_t rate) {
    switch (rate) {
    case 1:
        return DtmbLdpcProfile{1, 35, 24, 3048};
    case 2:
        return DtmbLdpcProfile{2, 23, 36, 4572};
    case 3:
        return DtmbLdpcProfile{3, 11, 48, 6096};
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
                check_variables[check].push_back(one_based_variable - 1);
            }
        }
    }
    if (!input) {
        throw std::runtime_error("failed to read complete alist: " + path);
    }
    return dtmb::core::make_ldpc_sparse_graph(check_variables, variable_count);
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

std::ostream& report_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cout;
    }
    file_holder = std::make_unique<std::ofstream>(path);
    if (!*file_holder) {
        throw std::runtime_error("failed to open JSON output: " + path);
    }
    return *file_holder;
}

std::vector<std::uint8_t> read_all_bytes(std::istream& input) {
    auto bytes = std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
    if (input.bad()) {
        throw std::runtime_error("failed while reading input");
    }
    return bytes;
}

std::vector<std::uint8_t> input_bits_from_bytes(
    const std::vector<std::uint8_t>& bytes,
    const std::string& input_format) {
    if (input_format == "hard-u8") {
        for (const auto value : bytes) {
            if (value > 1U) {
                throw std::runtime_error("hard-u8 input values must be 0 or 1");
            }
        }
        return bytes;
    }
    if (input_format != "llr-f32") {
        throw std::invalid_argument("input format must be hard-u8 or llr-f32");
    }
    if ((bytes.size() % sizeof(float)) != 0) {
        throw std::runtime_error("llr-f32 input byte count must be a multiple of 4");
    }
    std::vector<std::uint8_t> bits(bytes.size() / sizeof(float));
    for (std::size_t index = 0; index < bits.size(); ++index) {
        float value = 0.0F;
        std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
        if (!std::isfinite(value)) {
            throw std::runtime_error("llr-f32 input values must be finite");
        }
        bits[index] = value < 0.0F ? 1U : 0U;
    }
    return bits;
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
             edge < graph.check_offsets[check + 1];
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

std::vector<std::uint8_t> apply_bit_plane_permutation(
    std::vector<std::uint8_t> bits,
    const std::string& name,
    std::size_t bits_per_symbol) {
    if (name == "identity") {
        return bits;
    }
    if (bits_per_symbol == 0) {
        throw std::runtime_error("bit-plane permutation requires a positive bits-per-symbol value");
    }
    std::vector<std::size_t> permutation;
    permutation.reserve(bits_per_symbol);
    for (std::size_t index = 0; index < bits_per_symbol; ++index) {
        permutation.push_back(index);
    }
    if (name == "bit_reversal") {
        std::reverse(permutation.begin(), permutation.end());
    } else if (name == "iq_swap") {
        if ((bits_per_symbol % 2) != 0) {
            throw std::runtime_error("iq_swap requires an even bits-per-symbol value");
        }
        const auto half = bits_per_symbol / 2;
        for (std::size_t index = 0; index < half; ++index) {
            permutation[index] = index + half;
            permutation[index + half] = index;
        }
    } else {
        throw std::invalid_argument("unknown bit-plane permutation: " + name);
    }

    auto output = bits;
    for (std::size_t symbol = 0; symbol < bits.size() / bits_per_symbol; ++symbol) {
        const auto base = symbol * bits_per_symbol;
        for (std::size_t bit = 0; bit < bits_per_symbol; ++bit) {
            output[base + bit] = bits[base + permutation[bit]];
        }
    }
    return output;
}

dtmb::core::LdpcStreamBitOrder stream_bit_order_from_name(const std::string& name) {
    if (name == "identity") {
        return dtmb::core::LdpcStreamBitOrder::identity;
    }
    if (name == "reverse_each_byte") {
        return dtmb::core::LdpcStreamBitOrder::reverse_each_byte;
    }
    if (name == "reverse_each_codeword") {
        return dtmb::core::LdpcStreamBitOrder::reverse_each_codeword;
    }
    throw std::invalid_argument("unknown stream bit order: " + name);
}

CandidateScore score_candidate(
    std::span<const std::uint8_t> input,
    const dtmb::core::LdpcSparseGraph& clean_check_graph,
    std::size_t bit_shift,
    std::size_t max_codewords,
    const std::string& bit_plane_permutation,
    const std::string& stream_bit_order,
    std::size_t workers) {
    const auto core_score = dtmb::core::ldpc_score_hard_bit_candidate(
        input,
        clean_check_graph,
        dtmb::core::LdpcHardCandidateScoreOptions{
            bit_shift,
            max_codewords,
            workers,
            stream_bit_order_from_name(stream_bit_order),
        });
    CandidateScore score;
    score.bit_shift = bit_shift;
    score.bit_plane_permutation = bit_plane_permutation;
    score.stream_bit_order = stream_bit_order;
    score.codewords = core_score.codewords;
    score.scored_bits = core_score.scored_bits;
    score.unused_bits = core_score.unused_bits;
    score.clean_rows = core_score.clean_rows;
    score.worker_count = core_score.worker_count;
    score.mean_syndrome_ratio = core_score.mean_syndrome_ratio;
    score.min_syndrome_ratio = core_score.min_syndrome_ratio;
    score.max_syndrome_ratio = core_score.max_syndrome_ratio;
    score.zero_syndrome_codewords = core_score.zero_syndrome_codewords;
    const auto sample_count = std::min<std::size_t>(16, core_score.syndrome_weights.size());
    score.sample_syndrome_weights.assign(
        core_score.syndrome_weights.begin(),
        core_score.syndrome_weights.begin() + static_cast<std::ptrdiff_t>(sample_count));
    return score;
}

WindowReport score_identity_windows(
    std::span<const std::uint8_t> input,
    const dtmb::core::LdpcSparseGraph& clean_check_graph,
    std::size_t max_codewords,
    std::size_t workers,
    std::size_t window_codewords,
    std::size_t window_step_codewords,
    std::optional<double> threshold) {
    const auto identity_score = dtmb::core::ldpc_score_hard_bit_candidate(
        input,
        clean_check_graph,
        dtmb::core::LdpcHardCandidateScoreOptions{
            0,
            max_codewords,
            workers,
            dtmb::core::LdpcStreamBitOrder::identity,
        });

    WindowReport report;
    report.window_codewords = window_codewords;
    report.window_step_codewords = window_step_codewords;
    report.threshold = threshold;
    for (std::size_t start = 0; start < identity_score.syndrome_weights.size();) {
        if (window_codewords > identity_score.syndrome_weights.size() - start) {
            break;
        }
        const auto first = identity_score.syndrome_weights.begin()
            + static_cast<std::ptrdiff_t>(start);
        const auto last = first + static_cast<std::ptrdiff_t>(window_codewords);
        const auto minmax = std::minmax_element(first, last);
        const auto total = std::accumulate(first, last, std::size_t{0});

        WindowScore row;
        row.start_codeword = start;
        row.end_codeword = start + window_codewords;
        row.mean_syndrome_ratio = static_cast<double>(total)
            / static_cast<double>(window_codewords * identity_score.clean_rows);
        row.min_syndrome_ratio = static_cast<double>(*minmax.first)
            / static_cast<double>(identity_score.clean_rows);
        row.max_syndrome_ratio = static_cast<double>(*minmax.second)
            / static_cast<double>(identity_score.clean_rows);
        row.zero_syndrome_codewords = static_cast<std::size_t>(
            std::count(first, last, std::size_t{0}));
        row.verdict = threshold && row.mean_syndrome_ratio <= *threshold
            ? "pass"
            : "shortlist";
        report.rows.push_back(std::move(row));

        if (window_step_codewords > identity_score.syndrome_weights.size() - start) {
            break;
        }
        start += window_step_codewords;
    }
    return report;
}

bool candidate_sort_key(const CandidateScore& lhs, const CandidateScore& rhs) {
    if (lhs.mean_syndrome_ratio != rhs.mean_syndrome_ratio) {
        return lhs.mean_syndrome_ratio < rhs.mean_syndrome_ratio;
    }
    if (lhs.min_syndrome_ratio != rhs.min_syndrome_ratio) {
        return lhs.min_syndrome_ratio < rhs.min_syndrome_ratio;
    }
    if (lhs.zero_syndrome_codewords != rhs.zero_syndrome_codewords) {
        return lhs.zero_syndrome_codewords > rhs.zero_syndrome_codewords;
    }
    if (lhs.bit_shift != rhs.bit_shift) {
        return lhs.bit_shift < rhs.bit_shift;
    }
    if (lhs.bit_plane_permutation != rhs.bit_plane_permutation) {
        return lhs.bit_plane_permutation < rhs.bit_plane_permutation;
    }
    return lhs.stream_bit_order < rhs.stream_bit_order;
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

void write_string_array(std::ostream& output, const std::vector<std::string>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) {
            output << ", ";
        }
        write_json_string(output, values[index]);
    }
    output << ']';
}

void write_size_array(std::ostream& output, const std::vector<std::size_t>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) {
            output << ", ";
        }
        output << values[index];
    }
    output << ']';
}

void write_candidate(std::ostream& output, const CandidateScore& score, std::string indent) {
    output << indent << "{\n";
    output << indent << "  \"bit_shift\": " << score.bit_shift << ",\n";
    output << indent << "  \"bit_plane_permutation\": ";
    write_json_string(output, score.bit_plane_permutation);
    output << ",\n";
    output << indent << "  \"clean_rows\": " << score.clean_rows << ",\n";
    output << indent << "  \"codewords\": " << score.codewords << ",\n";
    output << indent << "  \"max_syndrome_ratio\": " << score.max_syndrome_ratio << ",\n";
    output << indent << "  \"mean_syndrome_ratio\": " << score.mean_syndrome_ratio << ",\n";
    output << indent << "  \"min_syndrome_ratio\": " << score.min_syndrome_ratio << ",\n";
    output << indent << "  \"sample_syndrome_weights\": ";
    write_size_array(output, score.sample_syndrome_weights);
    output << ",\n";
    output << indent << "  \"scored_bits\": " << score.scored_bits << ",\n";
    output << indent << "  \"stream_bit_order\": ";
    write_json_string(output, score.stream_bit_order);
    output << ",\n";
    output << indent << "  \"unused_bits\": " << score.unused_bits << ",\n";
    output << indent << "  \"worker_count\": " << score.worker_count << ",\n";
    output << indent << "  \"zero_syndrome_codewords\": " << score.zero_syndrome_codewords << "\n";
    output << indent << "}";
}

void write_window(std::ostream& output, const WindowScore& score, std::string indent) {
    output << indent << "{\n";
    output << indent << "  \"end_codeword\": " << score.end_codeword << ",\n";
    output << indent << "  \"max_syndrome_ratio\": " << score.max_syndrome_ratio << ",\n";
    output << indent << "  \"mean_syndrome_ratio\": " << score.mean_syndrome_ratio << ",\n";
    output << indent << "  \"min_syndrome_ratio\": " << score.min_syndrome_ratio << ",\n";
    output << indent << "  \"start_codeword\": " << score.start_codeword << ",\n";
    output << indent << "  \"verdict\": ";
    write_json_string(output, score.verdict);
    output << ",\n";
    output << indent << "  \"zero_syndrome_codewords\": "
           << score.zero_syndrome_codewords << "\n";
    output << indent << "}";
}

std::string verdict_for(const std::vector<CandidateScore>& candidates) {
    if (candidates.empty()) {
        return "no_complete_codewords";
    }
    if (candidates.front().mean_syndrome_ratio <= 0.05) {
        return "pre_ldpc_bits_match_ldpc_codeword_convention";
    }
    if (candidates.front().mean_syndrome_ratio <= 0.30) {
        return "partial_syndrome_drop_investigate_best_convention";
    }
    return "hard_parity_uninformative_before_soft_decode";
}

std::string window_verdict_for(const WindowReport& report) {
    if (report.rows.empty()) {
        return "no_complete_windows";
    }
    if (std::any_of(report.rows.begin(), report.rows.end(), [](const WindowScore& row) {
            return row.verdict == "pass";
        })) {
        return "pass";
    }
    return "shortlist";
}

void write_report(
    std::ostream& output,
    const DtmbLdpcProfile& profile,
    std::size_t input_bits,
    const std::string& input_format,
    std::size_t bits_per_symbol,
    std::size_t workers,
    const std::vector<std::size_t>& bit_shifts,
    const std::vector<std::string>& bit_plane_permutations,
    const std::vector<std::string>& stream_bit_orders,
    const std::vector<CandidateScore>& candidates,
    const std::optional<WindowReport>& window_report,
    std::size_t top) {
    output << std::setprecision(12);
    output << "{\n";
    output << "  \"schema\": \"dtmb.ldpc_h_score.v1\",\n";
    output << "  \"verdict\": ";
    write_json_string(output, verdict_for(candidates));
    output << ",\n";
    output << "  \"bit_plane_permutations\": ";
    write_string_array(output, bit_plane_permutations);
    output << ",\n";
    output << "  \"bit_shifts\": ";
    write_size_array(output, bit_shifts);
    output << ",\n";
    output << "  \"bits_per_symbol\": " << bits_per_symbol << ",\n";
    output << "  \"candidate_count\": " << candidates.size() << ",\n";
    output << "  \"codeword_bits\": " << profile.transmitted_bits << ",\n";
    output << "  \"fec_rate_index\": " << profile.fec_rate << ",\n";
    output << "  \"full_codeword_bits\": " << profile.full_codeword_bits() << ",\n";
    output << "  \"input_bits\": " << input_bits << ",\n";
    output << "  \"input_format\": ";
    write_json_string(output, input_format);
    output << ",\n";
    output << "  \"stream_bit_orders\": ";
    write_string_array(output, stream_bit_orders);
    output << ",\n";
    output << "  \"requested_workers\": " << workers << ",\n";
    output << "  \"workers\": "
           << (candidates.empty() ? 0 : candidates.front().worker_count) << ",\n";
    if (window_report) {
        output << "  \"window_candidate\": \"identity\",\n";
        output << "  \"window_codewords\": " << window_report->window_codewords << ",\n";
        output << "  \"window_count\": " << window_report->rows.size() << ",\n";
        output << "  \"window_step_codewords\": "
               << window_report->window_step_codewords << ",\n";
        output << "  \"window_threshold\": ";
        if (window_report->threshold) {
            output << *window_report->threshold;
        } else {
            output << "null";
        }
        output << ",\n";
        output << "  \"window_verdict\": ";
        write_json_string(output, window_verdict_for(*window_report));
        output << ",\n";
        output << "  \"windows\": [\n";
        for (std::size_t index = 0; index < window_report->rows.size(); ++index) {
            if (index) {
                output << ",\n";
            }
            write_window(output, window_report->rows[index], "    ");
        }
        output << "\n  ],\n";
    }
    output << "  \"best\": ";
    if (candidates.empty()) {
        output << "null";
    } else {
        write_candidate(output, candidates.front(), "  ");
    }
    output << ",\n";
    output << "  \"top\": [\n";
    const auto top_count = std::min(top, candidates.size());
    for (std::size_t index = 0; index < top_count; ++index) {
        if (index) {
            output << ",\n";
        }
        write_candidate(output, candidates[index], "    ");
    }
    output << "\n  ]\n";
    output << "}\n";
}

void write_error_report(std::ostream& output, const std::string& message) {
    output << "{\n"
           << "  \"schema\": \"dtmb.ldpc_h_score.v1\",\n"
           << "  \"verdict\": \"invalid_input\",\n"
           << "  \"error\": ";
    write_json_string(output, message);
    output << "\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t fec_rate = 0;
    std::string alist_path;
    std::string input_format = "hard-u8";
    std::string input_path = "-";
    std::size_t max_codewords = 32;
    bool max_codewords_set = false;
    std::vector<std::size_t> bit_shifts{0};
    std::size_t bits_per_symbol = 6;
    std::vector<std::string> bit_plane_permutations{"identity"};
    std::vector<std::string> stream_bit_orders{"identity"};
    std::size_t workers = 1;
    std::size_t top = 20;
    std::size_t window_codewords = 0;
    std::size_t window_step_codewords = 0;
    bool window_codewords_set = false;
    bool window_step_codewords_set = false;
    std::optional<double> window_threshold;
    std::string json_out_path;
    std::vector<std::string> positional;

    try {
        std::cout.imbue(std::locale::classic());
        std::cerr.imbue(std::locale::classic());
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--fec-rate") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                fec_rate = parse_size(argv[index], "fec rate");
            } else if (arg == "--alist") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                alist_path = argv[index];
            } else if (arg == "--input-format") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                input_format = argv[index];
            } else if (arg == "--max-codewords") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                max_codewords = parse_size(argv[index], "max codewords");
                max_codewords_set = true;
            } else if (arg == "--bit-shifts") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                bit_shifts = split_csv_sizes(argv[index], "bit shift");
            } else if (arg == "--bits-per-symbol") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                bits_per_symbol = parse_size(argv[index], "bits per symbol");
            } else if (arg == "--bit-plane-permutations") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                bit_plane_permutations = split_csv_strings(argv[index]);
            } else if (arg == "--stream-bit-orders") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                stream_bit_orders = split_csv_strings(argv[index]);
            } else if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                workers = parse_size(argv[index], "worker count");
            } else if (arg == "--top") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                top = parse_size(argv[index], "top count");
            } else if (arg == "--window-codewords") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                window_codewords = parse_size(argv[index], "window codewords");
                window_codewords_set = true;
            } else if (arg == "--window-step-codewords") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                window_step_codewords = parse_size(argv[index], "window step codewords");
                window_step_codewords_set = true;
            } else if (arg == "--window-threshold") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                window_threshold = parse_ratio(argv[index], "window threshold");
            } else if (arg == "--json-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                json_out_path = argv[index];
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                positional.push_back(arg);
            }
        }

        if (fec_rate == 0 || alist_path.empty() || positional.size() > 1
            || bits_per_symbol == 0 || bit_plane_permutations.empty()
            || stream_bit_orders.empty()) {
            usage(argv[0]);
            return 2;
        }
        if (!window_codewords_set && (window_step_codewords_set || window_threshold)) {
            throw std::invalid_argument(
                "window step and threshold require --window-codewords");
        }
        if (window_codewords_set && window_codewords == 0) {
            throw std::invalid_argument("window codewords must be positive");
        }
        if (window_step_codewords_set && window_step_codewords == 0) {
            throw std::invalid_argument("window step codewords must be positive");
        }
        if (window_codewords_set && !window_step_codewords_set) {
            window_step_codewords = window_codewords;
        }
        if (window_codewords_set && !max_codewords_set) {
            max_codewords = 0;
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }

        const auto profile = profile_for_rate(fec_rate);
        const auto graph = read_alist(alist_path);
        const auto clean_check_graph = clean_transmitted_check_graph(graph, profile);

        dtmb::tools::configure_binary_stdio(input_path == "-", false);
        std::unique_ptr<std::ifstream> input_file;
        auto& input = input_stream(input_path, input_file);
        const auto input_bytes = read_all_bytes(input);
        const auto input_bits = input_bits_from_bytes(input_bytes, input_format);

        std::vector<CandidateScore> candidates;
        for (const auto& permutation : bit_plane_permutations) {
            const auto permuted_bits = apply_bit_plane_permutation(
                input_bits,
                permutation,
                bits_per_symbol);
            for (const auto shift : bit_shifts) {
                for (const auto& order : stream_bit_orders) {
                    auto score = score_candidate(
                        permuted_bits,
                        clean_check_graph,
                        shift,
                        max_codewords,
                        permutation,
                        order,
                        workers);
                    if (score.codewords > 0) {
                        candidates.push_back(std::move(score));
                    }
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), candidate_sort_key);
        std::optional<WindowReport> window_report;
        if (window_codewords_set) {
            window_report = score_identity_windows(
                input_bits,
                clean_check_graph,
                max_codewords,
                workers,
                window_codewords,
                window_step_codewords,
                window_threshold);
        }
        std::unique_ptr<std::ofstream> report_file;
        auto& report = report_stream(json_out_path, report_file);
        write_report(
            report,
            profile,
            input_bits.size(),
            input_format,
            bits_per_symbol,
            workers,
            bit_shifts,
            bit_plane_permutations,
            stream_bit_orders,
            candidates,
            window_report,
            top);
        return 0;
    } catch (const std::exception& exc) {
        bool wrote_error_report = false;
        if (!json_out_path.empty() && json_out_path != "-") {
            std::ofstream report(json_out_path);
            if (report) {
                report.imbue(std::locale::classic());
                write_error_report(report, exc.what());
                wrote_error_report = true;
            }
        }
        if (!wrote_error_report) {
            write_error_report(std::cout, exc.what());
        }
        std::cerr << "dtmb_core_ldpc_h_score: " << exc.what() << '\n';
        return 1;
    }
}
