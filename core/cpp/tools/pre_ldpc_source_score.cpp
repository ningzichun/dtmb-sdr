#include "dtmb/core.hpp"

#include "binary_stdio.hpp"
#include "ldpc_h_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kDefaultReadChunkBytes = 64U * 1024U;
constexpr std::size_t kDefaultSourceFrameGroupCap = 8192U;
constexpr std::int64_t kOverflowGroupId = -1;

enum class InterleaverMode {
    none,
    mode1,
    mode2,
};

struct CliOptions {
    std::size_t fec_rate = 0;
    std::string alist_path;
    std::string input_path = "-";
    std::string diagnostics_path;
    std::size_t read_chunk_bytes = kDefaultReadChunkBytes;
    std::size_t bits_per_symbol = 6;
    std::size_t top = 20;
    double weak_llr_threshold = 1.0;
    std::size_t interleaver_phase = 0;
    std::size_t source_frame_group_cap = kDefaultSourceFrameGroupCap;
    std::size_t codewords_per_frame = 0;
    InterleaverMode interleaver_mode = InterleaverMode::mode2;
    std::vector<std::pair<std::size_t, std::size_t>> fec_frame_ranges;
    std::vector<std::size_t> tracked_ldpc_variables;
    bool passthrough = false;
};

struct SourceTags {
    std::int64_t frame = 0;
    std::int64_t carrier = 0;
    std::int64_t branch = -1;
    std::int64_t qam_plane = 0;
};

struct GroupStats {
    std::int64_t id = 0;
    std::size_t bit_count = 0;
    std::size_t weak_llr_count = 0;
    std::size_t zero_llr_count = 0;
    std::size_t hard_one_count = 0;
    std::size_t hard_h_observed_bit_count = 0;
    std::size_t hard_h_failed_bit_count = 0;
    std::uint64_t hard_h_observed_check_edges = 0;
    std::uint64_t hard_h_failed_check_edges = 0;
    double sum_abs_llr = 0.0;
    double sum_abs_llr2 = 0.0;
    double min_abs_llr = std::numeric_limits<double>::infinity();
    double max_abs_llr = 0.0;
};

struct GroupRow {
    std::int64_t id = 0;
    std::size_t bit_count = 0;
    std::size_t hard_h_observed_bit_count = 0;
    std::size_t hard_h_failed_bit_count = 0;
    std::uint64_t hard_h_observed_check_edges = 0;
    std::uint64_t hard_h_failed_check_edges = 0;
    double llr_min_abs = 0.0;
    double llr_mean_abs = 0.0;
    double llr_rms = 0.0;
    double llr_max_abs = 0.0;
    double weak_llr_fraction = 0.0;
    double zero_llr_fraction = 0.0;
    double hard_one_ratio = 0.0;
    std::optional<double> hard_h_failed_check_edge_ratio;
    std::optional<double> hard_h_failed_bit_fraction;
};

struct Distribution {
    std::size_t count = 0;
    std::optional<double> min;
    std::optional<double> mean;
    std::optional<double> median;
    std::optional<double> p90;
    std::optional<double> max;
};

struct DimensionReport {
    std::string name;
    std::size_t group_count = 0;
    bool capped = false;
    std::size_t group_cap = 0;
    std::size_t overflow_bit_count = 0;
    std::vector<GroupRow> groups;
};

struct DenseAccumulator {
    explicit DenseAccumulator(std::size_t size, std::int64_t first_id = 0) {
        groups.reserve(size);
        for (std::size_t index = 0; index < size; ++index) {
            GroupStats stats;
            stats.id = first_id + static_cast<std::int64_t>(index);
            groups.push_back(stats);
        }
    }

    GroupStats& at(std::int64_t id) {
        const auto index = static_cast<std::size_t>(id - groups.front().id);
        if (index >= groups.size()) {
            throw std::logic_error("dense source-score group id is out of range");
        }
        return groups[index];
    }

    std::vector<GroupStats> groups;
};

struct SparseAccumulator {
    explicit SparseAccumulator(std::size_t cap) : group_cap(cap) {
        overflow.id = kOverflowGroupId;
    }

    GroupStats& at(std::int64_t id) {
        const auto found = index_by_id.find(id);
        if (found != index_by_id.end()) {
            return groups[found->second];
        }
        if (groups.size() >= group_cap) {
            capped = true;
            return overflow;
        }
        const auto index = groups.size();
        index_by_id.emplace(id, index);
        GroupStats stats;
        stats.id = id;
        groups.push_back(stats);
        return groups.back();
    }

    std::size_t group_cap = 0;
    bool capped = false;
    std::unordered_map<std::int64_t, std::size_t> index_by_id;
    std::vector<GroupStats> groups;
    GroupStats overflow;
};

struct Summary {
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t input_llrs = 0;
    std::size_t finite_llrs = 0;
    std::size_t nonfinite_llrs = 0;
    std::size_t complete_codewords = 0;
    std::size_t included_codewords = 0;
    std::size_t skipped_codewords = 0;
    std::size_t scored_codewords = 0;
    std::size_t dirty_codewords = 0;
    std::size_t trailing_float_bytes = 0;
    std::size_t trailing_codeword_llrs = 0;
    std::size_t clean_rows_per_codeword = 0;
    std::uint64_t observed_check_edges = 0;
    std::uint64_t failed_check_edges = 0;
    std::size_t zero_syndrome_codewords = 0;
    std::size_t min_syndrome_weight = std::numeric_limits<std::size_t>::max();
    std::size_t max_syndrome_weight = 0;
    std::uint64_t total_syndrome_weight = 0;
    std::string status;
    std::string verdict;
};

struct Accumulators {
    DenseAccumulator qam_plane;
    DenseAccumulator codeword_slot;
    DenseAccumulator ldpc_variable;
    DenseAccumulator ldpc_variable_mod_127;
    DenseAccumulator source_carrier;
    DenseAccumulator source_branch;
    SparseAccumulator source_frame;

    Accumulators(
        std::size_t bits_per_symbol,
        std::size_t codeword_slot_count,
        std::size_t variable_count,
        std::size_t branch_groups,
        std::int64_t first_branch_id,
        std::size_t frame_group_cap)
        : qam_plane(bits_per_symbol),
          codeword_slot(codeword_slot_count),
          ldpc_variable(variable_count),
          ldpc_variable_mod_127(127),
          source_carrier(dtmb::core::kC3780DataSymbols),
          source_branch(branch_groups, first_branch_id),
          source_frame(frame_group_cap) {}
};

struct TrackedVariableAccumulators {
    TrackedVariableAccumulators(
        std::size_t variable_id,
        std::size_t bits_per_symbol,
        std::size_t codeword_slot_count,
        std::size_t branch_groups,
        std::int64_t first_branch_id,
        std::size_t frame_group_cap)
        : variable(variable_id),
          codeword_slot(codeword_slot_count),
          qam_plane(bits_per_symbol),
          source_carrier(dtmb::core::kC3780DataSymbols),
          source_branch(branch_groups, first_branch_id),
          source_frame(frame_group_cap) {
        summary.id = static_cast<std::int64_t>(variable_id);
    }

    std::size_t variable = 0;
    GroupStats summary;
    DenseAccumulator codeword_slot;
    DenseAccumulator qam_plane;
    DenseAccumulator source_carrier;
    DenseAccumulator source_branch;
    SparseAccumulator source_frame;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --fec-rate 1|2|3 --alist PATH"
        << " [--interleaver-mode none|mode1|mode2] [--interleaver-phase N]"
        << " [--bits-per-symbol N] [--weak-llr-threshold X]"
        << " [--codewords-per-frame N --fec-frame-range FIRST:LAST]"
        << " [--track-ldpc-variable N]"
        << " [--source-frame-group-cap N] [--top N]"
        << " [--read-chunk-bytes N] [--passthrough]"
        << " [--diagnostics-out PATH] [input.llr.f32|-]\n";
}

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const char ch : value) {
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

std::size_t parse_size(const std::string& text, const char* field, bool allow_zero = false) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size() || (!allow_zero && value == 0)) {
        throw std::invalid_argument(std::string(field) + " must be "
            + (allow_zero ? "non-negative" : "positive"));
    }
    return static_cast<std::size_t>(value);
}

double parse_nonnegative_double(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(std::string(field) + " must be finite and non-negative");
    }
    return value;
}

std::pair<std::size_t, std::size_t> parse_frame_range(const std::string& text) {
    const auto separator = text.find(':');
    if (separator == std::string::npos) {
        throw std::invalid_argument("FEC frame range must be FIRST:LAST");
    }
    const auto first = parse_size(text.substr(0, separator), "FEC frame range first");
    const auto last = parse_size(text.substr(separator + 1), "FEC frame range last");
    if (last < first) {
        throw std::invalid_argument("FEC frame range last must be >= first");
    }
    return {first, last};
}

InterleaverMode parse_interleaver_mode(const std::string& text) {
    if (text == "none") {
        return InterleaverMode::none;
    }
    if (text == "mode1") {
        return InterleaverMode::mode1;
    }
    if (text == "mode2") {
        return InterleaverMode::mode2;
    }
    throw std::invalid_argument("interleaver mode must be none, mode1, or mode2");
}

const char* interleaver_mode_name(InterleaverMode mode) {
    switch (mode) {
    case InterleaverMode::none:
        return "none";
    case InterleaverMode::mode1:
        return "mode1";
    case InterleaverMode::mode2:
        return "mode2";
    }
    return "unknown";
}

dtmb::core::SymbolInterleaverSpec interleaver_spec(InterleaverMode mode) {
    if (mode == InterleaverMode::mode1) {
        return dtmb::core::symbol_interleaver_spec(dtmb::core::SymbolInterleaverMode::mode1);
    }
    if (mode == InterleaverMode::mode2) {
        return dtmb::core::symbol_interleaver_spec(dtmb::core::SymbolInterleaverMode::mode2);
    }
    return dtmb::core::SymbolInterleaverSpec{1, 0};
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
            options.fec_rate = parse_size(need_value("--fec-rate"), "fec rate");
        } else if (arg == "--alist") {
            options.alist_path = need_value("--alist");
        } else if (arg == "--interleaver-mode") {
            options.interleaver_mode = parse_interleaver_mode(
                need_value("--interleaver-mode"));
        } else if (arg == "--interleaver-phase") {
            options.interleaver_phase = parse_size(
                need_value("--interleaver-phase"),
                "interleaver phase",
                true);
        } else if (arg == "--bits-per-symbol") {
            options.bits_per_symbol = parse_size(
                need_value("--bits-per-symbol"),
                "bits per symbol");
        } else if (arg == "--weak-llr-threshold") {
            options.weak_llr_threshold = parse_nonnegative_double(
                need_value("--weak-llr-threshold"),
                "weak LLR threshold");
        } else if (arg == "--source-frame-group-cap") {
            options.source_frame_group_cap = parse_size(
                need_value("--source-frame-group-cap"),
                "source frame group cap");
        } else if (arg == "--codewords-per-frame") {
            options.codewords_per_frame = parse_size(
                need_value("--codewords-per-frame"),
                "codewords per frame");
        } else if (arg == "--fec-frame-range") {
            options.fec_frame_ranges.push_back(
                parse_frame_range(need_value("--fec-frame-range")));
        } else if (arg == "--track-ldpc-variable") {
            options.tracked_ldpc_variables.push_back(parse_size(
                need_value("--track-ldpc-variable"),
                "tracked LDPC variable",
                true));
        } else if (arg == "--top") {
            options.top = parse_size(need_value("--top"), "top", true);
        } else if (arg == "--read-chunk-bytes") {
            options.read_chunk_bytes = parse_size(
                need_value("--read-chunk-bytes"),
                "read chunk bytes");
        } else if (arg == "--passthrough") {
            options.passthrough = true;
        } else if (arg == "--diagnostics-out") {
            options.diagnostics_path = need_value("--diagnostics-out");
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            positional.push_back(arg);
        }
    }
    if (options.fec_rate == 0 || options.alist_path.empty()) {
        throw std::invalid_argument("--fec-rate and --alist are required");
    }
    if (!options.fec_frame_ranges.empty() && options.codewords_per_frame == 0) {
        throw std::invalid_argument(
            "--fec-frame-range requires --codewords-per-frame");
    }
    if (options.read_chunk_bytes > static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::invalid_argument("read chunk bytes must fit streamsize");
    }
    const auto spec = interleaver_spec(options.interleaver_mode);
    if (options.interleaver_phase >= spec.branch_count) {
        throw std::invalid_argument("interleaver phase is outside the selected mode");
    }
    if (positional.size() > 1) {
        throw std::invalid_argument("too many positional inputs");
    }
    if (!positional.empty()) {
        options.input_path = positional.front();
    }
    return options;
}

bool codeword_in_selected_frame_ranges(
    std::size_t zero_based_codeword,
    const CliOptions& options) {
    if (options.fec_frame_ranges.empty()) {
        return true;
    }
    const auto frame_index =
        zero_based_codeword / options.codewords_per_frame + std::size_t{1};
    return std::any_of(
        options.fec_frame_ranges.begin(),
        options.fec_frame_ranges.end(),
        [frame_index](const auto& range) {
            return frame_index >= range.first && frame_index <= range.second;
        });
}

std::size_t codeword_slot_for(std::size_t zero_based_codeword, const CliOptions& options) {
    if (options.codewords_per_frame == 0) {
        return 0;
    }
    return zero_based_codeword % options.codewords_per_frame;
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

std::optional<double> optional_ratio(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        return std::nullopt;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

std::optional<double> optional_ratio(std::size_t numerator, std::size_t denominator) {
    if (denominator == 0) {
        return std::nullopt;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

void observe_llr(
    GroupStats& stats,
    float llr,
    std::uint8_t hard_bit,
    double weak_llr_threshold) {
    const auto absolute = static_cast<double>(std::abs(llr));
    ++stats.bit_count;
    stats.sum_abs_llr += absolute;
    stats.sum_abs_llr2 += absolute * absolute;
    stats.min_abs_llr = std::min(stats.min_abs_llr, absolute);
    stats.max_abs_llr = std::max(stats.max_abs_llr, absolute);
    stats.weak_llr_count += absolute < weak_llr_threshold ? 1U : 0U;
    stats.zero_llr_count += absolute == 0.0 ? 1U : 0U;
    stats.hard_one_count += hard_bit != 0 ? 1U : 0U;
}

void observe_hard_h(
    GroupStats& stats,
    std::uint16_t observed_edges,
    std::uint16_t failed_edges) {
    if (observed_edges == 0) {
        return;
    }
    ++stats.hard_h_observed_bit_count;
    stats.hard_h_failed_bit_count += failed_edges != 0 ? 1U : 0U;
    stats.hard_h_observed_check_edges += observed_edges;
    stats.hard_h_failed_check_edges += failed_edges;
}

void observe_all_llr(
    Accumulators& accumulators,
    const SourceTags& tags,
    float llr,
    std::uint8_t hard_bit,
    double weak_llr_threshold) {
    observe_llr(
        accumulators.qam_plane.at(tags.qam_plane),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        accumulators.source_carrier.at(tags.carrier),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        accumulators.source_branch.at(tags.branch),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        accumulators.source_frame.at(tags.frame),
        llr,
        hard_bit,
        weak_llr_threshold);
}

void observe_position_llr(
    Accumulators& accumulators,
    std::size_t variable,
    std::size_t codeword_slot,
    float llr,
    std::uint8_t hard_bit,
    double weak_llr_threshold) {
    observe_llr(
        accumulators.codeword_slot.at(static_cast<std::int64_t>(codeword_slot)),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        accumulators.ldpc_variable.at(static_cast<std::int64_t>(variable)),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        accumulators.ldpc_variable_mod_127.at(static_cast<std::int64_t>(variable % 127U)),
        llr,
        hard_bit,
        weak_llr_threshold);
}

void observe_all_hard_h(
    Accumulators& accumulators,
    const SourceTags& tags,
    std::uint16_t observed_edges,
    std::uint16_t failed_edges) {
    observe_hard_h(
        accumulators.qam_plane.at(tags.qam_plane),
        observed_edges,
        failed_edges);
    observe_hard_h(
        accumulators.source_carrier.at(tags.carrier),
        observed_edges,
        failed_edges);
    observe_hard_h(
        accumulators.source_branch.at(tags.branch),
        observed_edges,
        failed_edges);
    observe_hard_h(
        accumulators.source_frame.at(tags.frame),
        observed_edges,
        failed_edges);
}

void observe_position_hard_h(
    Accumulators& accumulators,
    std::size_t variable,
    std::size_t codeword_slot,
    std::uint16_t observed_edges,
    std::uint16_t failed_edges) {
    observe_hard_h(
        accumulators.codeword_slot.at(static_cast<std::int64_t>(codeword_slot)),
        observed_edges,
        failed_edges);
    observe_hard_h(
        accumulators.ldpc_variable.at(static_cast<std::int64_t>(variable)),
        observed_edges,
        failed_edges);
    observe_hard_h(
        accumulators.ldpc_variable_mod_127.at(static_cast<std::int64_t>(variable % 127U)),
        observed_edges,
        failed_edges);
}

void observe_tracked_llr(
    TrackedVariableAccumulators& tracked,
    const SourceTags& tags,
    std::size_t codeword_slot,
    float llr,
    std::uint8_t hard_bit,
    double weak_llr_threshold) {
    observe_llr(tracked.summary, llr, hard_bit, weak_llr_threshold);
    observe_llr(
        tracked.codeword_slot.at(static_cast<std::int64_t>(codeword_slot)),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        tracked.qam_plane.at(tags.qam_plane),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        tracked.source_carrier.at(tags.carrier),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        tracked.source_branch.at(tags.branch),
        llr,
        hard_bit,
        weak_llr_threshold);
    observe_llr(
        tracked.source_frame.at(tags.frame),
        llr,
        hard_bit,
        weak_llr_threshold);
}

void observe_tracked_hard_h(
    TrackedVariableAccumulators& tracked,
    const SourceTags& tags,
    std::size_t codeword_slot,
    std::uint16_t observed_edges,
    std::uint16_t failed_edges) {
    observe_hard_h(tracked.summary, observed_edges, failed_edges);
    observe_hard_h(
        tracked.codeword_slot.at(static_cast<std::int64_t>(codeword_slot)),
        observed_edges,
        failed_edges);
    observe_hard_h(tracked.qam_plane.at(tags.qam_plane), observed_edges, failed_edges);
    observe_hard_h(
        tracked.source_carrier.at(tags.carrier),
        observed_edges,
        failed_edges);
    observe_hard_h(
        tracked.source_branch.at(tags.branch),
        observed_edges,
        failed_edges);
    observe_hard_h(
        tracked.source_frame.at(tags.frame),
        observed_edges,
        failed_edges);
}

SourceTags tags_for_llr_index(std::size_t llr_index, const CliOptions& options) {
    const auto symbol_index = llr_index / options.bits_per_symbol;
    const auto qam_plane = llr_index % options.bits_per_symbol;

    std::size_t source_symbol = symbol_index;
    std::int64_t branch = -1;
    if (options.interleaver_mode != InterleaverMode::none) {
        const auto spec = interleaver_spec(options.interleaver_mode);
        const auto latency = spec.full_stream_latency_symbols();
        if (symbol_index > std::numeric_limits<std::size_t>::max() - latency) {
            throw std::overflow_error("source-symbol index overflow");
        }
        const auto output_index = symbol_index + latency;
        const auto start = output_index % spec.branch_count;
        const auto physical_branch = (start + options.interleaver_phase) % spec.branch_count;
        const auto branch_delay = (spec.branch_count - 1U - physical_branch)
            * spec.delay_step;
        const auto branch_offset = (output_index - start) / spec.branch_count;
        if (branch_offset < branch_delay) {
            throw std::runtime_error("deinterleaver source index is negative after latency discard");
        }
        const auto source_offset = branch_offset - branch_delay;
        if (source_offset > (
                std::numeric_limits<std::size_t>::max() - start) / spec.branch_count) {
            throw std::overflow_error("source-symbol index overflow");
        }
        source_symbol = start + spec.branch_count * source_offset;
        branch = static_cast<std::int64_t>(
            (source_symbol % spec.branch_count + options.interleaver_phase)
            % spec.branch_count);
    }

    return SourceTags{
        static_cast<std::int64_t>(source_symbol / dtmb::core::kC3780DataSymbols),
        static_cast<std::int64_t>(source_symbol % dtmb::core::kC3780DataSymbols),
        branch,
        static_cast<std::int64_t>(qam_plane),
    };
}

std::vector<std::size_t> unique_tracked_variables(
    std::vector<std::size_t> variables,
    std::size_t variable_count) {
    std::sort(variables.begin(), variables.end());
    variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
    for (const auto variable : variables) {
        if (variable >= variable_count) {
            throw std::invalid_argument("tracked LDPC variable is outside the codeword");
        }
    }
    return variables;
}

std::vector<std::uint16_t> total_edges_by_variable(
    const dtmb::core::LdpcSparseGraph& graph) {
    if (graph.variable_count > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("LDPC variable count exceeds uint16_t guard");
    }
    std::vector<std::uint16_t> total(graph.variable_count, 0);
    for (const auto variable : graph.edge_variables) {
        if (total[variable] == std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("LDPC check-edge count exceeds uint16_t");
        }
        ++total[variable];
    }
    return total;
}

std::size_t score_codeword_failed_edges(
    std::span<const std::uint8_t> bits,
    const dtmb::core::LdpcSparseGraph& graph,
    std::span<std::uint16_t> failed_edges) {
    std::fill(failed_edges.begin(), failed_edges.end(), std::uint16_t{0});
    std::size_t syndrome_weight = 0;
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        std::uint8_t parity = 0;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1U];
             ++edge) {
            parity ^= bits[graph.edge_variables[edge]];
        }
        if (parity == 0) {
            continue;
        }
        ++syndrome_weight;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1U];
             ++edge) {
            const auto variable = graph.edge_variables[edge];
            if (failed_edges[variable] == std::numeric_limits<std::uint16_t>::max()) {
                throw std::runtime_error("failed check-edge count exceeds uint16_t");
            }
            ++failed_edges[variable];
        }
    }
    return syndrome_weight;
}

GroupRow make_row(const GroupStats& stats) {
    GroupRow row;
    row.id = stats.id;
    row.bit_count = stats.bit_count;
    row.hard_h_observed_bit_count = stats.hard_h_observed_bit_count;
    row.hard_h_failed_bit_count = stats.hard_h_failed_bit_count;
    row.hard_h_observed_check_edges = stats.hard_h_observed_check_edges;
    row.hard_h_failed_check_edges = stats.hard_h_failed_check_edges;
    if (stats.bit_count != 0) {
        row.llr_min_abs = std::isfinite(stats.min_abs_llr) ? stats.min_abs_llr : 0.0;
        row.llr_mean_abs = stats.sum_abs_llr / static_cast<double>(stats.bit_count);
        row.llr_rms = std::sqrt(stats.sum_abs_llr2 / static_cast<double>(stats.bit_count));
        row.llr_max_abs = stats.max_abs_llr;
        row.weak_llr_fraction = static_cast<double>(stats.weak_llr_count)
            / static_cast<double>(stats.bit_count);
        row.zero_llr_fraction = static_cast<double>(stats.zero_llr_count)
            / static_cast<double>(stats.bit_count);
        row.hard_one_ratio = static_cast<double>(stats.hard_one_count)
            / static_cast<double>(stats.bit_count);
    }
    row.hard_h_failed_check_edge_ratio = optional_ratio(
        stats.hard_h_failed_check_edges,
        stats.hard_h_observed_check_edges);
    row.hard_h_failed_bit_fraction = optional_ratio(
        stats.hard_h_failed_bit_count,
        stats.hard_h_observed_bit_count);
    return row;
}

std::vector<GroupRow> rows_from_groups(const std::vector<GroupStats>& groups) {
    std::vector<GroupRow> rows;
    rows.reserve(groups.size());
    for (const auto& group : groups) {
        if (group.bit_count != 0 || group.hard_h_observed_check_edges != 0) {
            rows.push_back(make_row(group));
        }
    }
    std::sort(rows.begin(), rows.end(), [](const GroupRow& lhs, const GroupRow& rhs) {
        return lhs.id < rhs.id;
    });
    return rows;
}

std::vector<GroupRow> rows_from_sparse(const SparseAccumulator& source_frame) {
    auto rows = rows_from_groups(source_frame.groups);
    if (source_frame.overflow.bit_count != 0
        || source_frame.overflow.hard_h_observed_check_edges != 0) {
        rows.push_back(make_row(source_frame.overflow));
    }
    std::sort(rows.begin(), rows.end(), [](const GroupRow& lhs, const GroupRow& rhs) {
        return lhs.id < rhs.id;
    });
    return rows;
}

Distribution distribution(std::vector<double> values) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [](double value) {
            return !std::isfinite(value);
        }),
        values.end());
    Distribution result;
    result.count = values.size();
    if (values.empty()) {
        return result;
    }
    std::sort(values.begin(), values.end());
    const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
    const auto quantile = [&](double q) -> double {
        if (values.size() == 1) {
            return values.front();
        }
        const auto position = q * static_cast<double>(values.size() - 1U);
        const auto low = static_cast<std::size_t>(std::floor(position));
        const auto high = static_cast<std::size_t>(std::ceil(position));
        const auto fraction = position - static_cast<double>(low);
        return values[low] * (1.0 - fraction) + values[high] * fraction;
    };
    result.min = values.front();
    result.mean = sum / static_cast<double>(values.size());
    result.median = quantile(0.5);
    result.p90 = quantile(0.9);
    result.max = values.back();
    return result;
}

DimensionReport make_dense_report(
    std::string name,
    const DenseAccumulator& accumulator) {
    DimensionReport report;
    report.name = std::move(name);
    report.groups = rows_from_groups(accumulator.groups);
    report.group_count = report.groups.size();
    return report;
}

DimensionReport make_source_frame_report(const SparseAccumulator& accumulator) {
    DimensionReport report;
    report.name = "source_frame";
    report.capped = accumulator.capped;
    report.group_cap = accumulator.group_cap;
    report.overflow_bit_count = accumulator.overflow.bit_count;
    report.groups = rows_from_sparse(accumulator);
    report.group_count = report.groups.size();
    return report;
}

std::vector<GroupRow> top_rows(
    std::vector<GroupRow> rows,
    std::size_t top,
    const std::string& mode) {
    if (mode == "worst_hard_h") {
        rows.erase(
            std::remove_if(rows.begin(), rows.end(), [](const GroupRow& row) {
                return !row.hard_h_failed_check_edge_ratio.has_value();
            }),
            rows.end());
        std::sort(rows.begin(), rows.end(), [](const GroupRow& lhs, const GroupRow& rhs) {
            const auto lhs_h = *lhs.hard_h_failed_check_edge_ratio;
            const auto rhs_h = *rhs.hard_h_failed_check_edge_ratio;
            if (lhs_h != rhs_h) {
                return lhs_h > rhs_h;
            }
            if (lhs.llr_mean_abs != rhs.llr_mean_abs) {
                return lhs.llr_mean_abs < rhs.llr_mean_abs;
            }
            return lhs.id < rhs.id;
        });
    } else if (mode == "weakest_llr") {
        std::sort(rows.begin(), rows.end(), [](const GroupRow& lhs, const GroupRow& rhs) {
            if (lhs.weak_llr_fraction != rhs.weak_llr_fraction) {
                return lhs.weak_llr_fraction > rhs.weak_llr_fraction;
            }
            if (lhs.llr_mean_abs != rhs.llr_mean_abs) {
                return lhs.llr_mean_abs < rhs.llr_mean_abs;
            }
            return lhs.id < rhs.id;
        });
    } else if (mode == "strongest_llr") {
        std::sort(rows.begin(), rows.end(), [](const GroupRow& lhs, const GroupRow& rhs) {
            if (lhs.llr_mean_abs != rhs.llr_mean_abs) {
                return lhs.llr_mean_abs > rhs.llr_mean_abs;
            }
            if (lhs.weak_llr_fraction != rhs.weak_llr_fraction) {
                return lhs.weak_llr_fraction < rhs.weak_llr_fraction;
            }
            return lhs.id < rhs.id;
        });
    }
    if (rows.size() > top) {
        rows.resize(top);
    }
    return rows;
}

void write_optional(std::ostream& output, const std::optional<double>& value) {
    if (value) {
        output << *value;
    } else {
        output << "null";
    }
}

void write_distribution(std::ostream& output, const Distribution& distribution) {
    output << "{\"count\":" << distribution.count
           << ",\"max\":";
    write_optional(output, distribution.max);
    output << ",\"mean\":";
    write_optional(output, distribution.mean);
    output << ",\"median\":";
    write_optional(output, distribution.median);
    output << ",\"min\":";
    write_optional(output, distribution.min);
    output << ",\"p90\":";
    write_optional(output, distribution.p90);
    output << '}';
}

void write_row(std::ostream& output, const GroupRow& row) {
    output << "{\"hard_h_failed_bit_fraction\":";
    write_optional(output, row.hard_h_failed_bit_fraction);
    output << ",\"hard_h_failed_bit_count\":" << row.hard_h_failed_bit_count
           << ",\"hard_h_failed_check_edge_ratio\":";
    write_optional(output, row.hard_h_failed_check_edge_ratio);
    output << ",\"hard_h_failed_check_edges\":" << row.hard_h_failed_check_edges
           << ",\"hard_h_observed_bit_count\":" << row.hard_h_observed_bit_count
           << ",\"hard_h_observed_check_edges\":" << row.hard_h_observed_check_edges
           << ",\"hard_one_ratio\":" << row.hard_one_ratio
           << ",\"id\":" << row.id
           << ",\"bit_count\":" << row.bit_count
           << ",\"llr_max_abs\":" << row.llr_max_abs
           << ",\"llr_mean_abs\":" << row.llr_mean_abs
           << ",\"llr_min_abs\":" << row.llr_min_abs
           << ",\"llr_rms\":" << row.llr_rms
           << ",\"weak_llr_fraction\":" << row.weak_llr_fraction
           << ",\"zero_llr_fraction\":" << row.zero_llr_fraction
           << '}';
}

void write_rows(std::ostream& output, const std::vector<GroupRow>& rows) {
    output << '[';
    for (std::size_t index = 0; index < rows.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        write_row(output, rows[index]);
    }
    output << ']';
}

void write_dimension(
    std::ostream& output,
    const DimensionReport& report,
    std::size_t top) {
    std::vector<double> llr_means;
    std::vector<double> weak_fractions;
    std::vector<double> hard_h_ratios;
    llr_means.reserve(report.groups.size());
    weak_fractions.reserve(report.groups.size());
    hard_h_ratios.reserve(report.groups.size());
    for (const auto& row : report.groups) {
        llr_means.push_back(row.llr_mean_abs);
        weak_fractions.push_back(row.weak_llr_fraction);
        if (row.hard_h_failed_check_edge_ratio) {
            hard_h_ratios.push_back(*row.hard_h_failed_check_edge_ratio);
        }
    }

    output << "{\"capped\":" << (report.capped ? "true" : "false")
           << ",\"group_cap\":" << report.group_cap
           << ",\"group_count\":" << report.group_count
           << ",\"groups\":";
    write_rows(output, report.groups);
    output << ",\"overflow_bit_count\":" << report.overflow_bit_count
           << ",\"strongest_llr\":";
    write_rows(output, top_rows(report.groups, top, "strongest_llr"));
    output << ",\"summary\":{\"hard_h_failed_check_edge_ratio\":";
    write_distribution(output, distribution(std::move(hard_h_ratios)));
    output << ",\"llr_mean_abs\":";
    write_distribution(output, distribution(std::move(llr_means)));
    output << ",\"weak_llr_fraction\":";
    write_distribution(output, distribution(std::move(weak_fractions)));
    output << "},\"weakest_llr\":";
    write_rows(output, top_rows(report.groups, top, "weakest_llr"));
    output << ",\"worst_hard_h\":";
    write_rows(output, top_rows(report.groups, top, "worst_hard_h"));
    output << '}';
}

void write_tracked_variable(
    std::ostream& output,
    const TrackedVariableAccumulators& tracked,
    std::size_t top) {
    output << "{\"variable\":" << tracked.variable
           << ",\"summary\":";
    write_row(output, make_row(tracked.summary));
    output << ",\"dimensions\":{\"codeword_slot\":";
    write_dimension(
        output,
        make_dense_report("codeword_slot", tracked.codeword_slot),
        top);
    output << ",\"qam_plane\":";
    write_dimension(
        output,
        make_dense_report("qam_plane", tracked.qam_plane),
        top);
    output << ",\"source_branch\":";
    write_dimension(
        output,
        make_dense_report("source_branch", tracked.source_branch),
        top);
    output << ",\"source_carrier\":";
    write_dimension(
        output,
        make_dense_report("source_carrier", tracked.source_carrier),
        top);
    output << ",\"source_frame\":";
    write_dimension(
        output,
        make_source_frame_report(tracked.source_frame),
        top);
    output << "}}";
}

void write_tracked_variables(
    std::ostream& output,
    const std::vector<TrackedVariableAccumulators>& tracked_variables,
    std::size_t top) {
    output << '[';
    for (std::size_t index = 0; index < tracked_variables.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        write_tracked_variable(output, tracked_variables[index], top);
    }
    output << ']';
}

void finalize_summary(Summary& summary) {
    summary.status = summary.nonfinite_llrs != 0
            || summary.trailing_float_bytes != 0
            || summary.trailing_codeword_llrs != 0
        ? "degraded"
        : "ok";
    if (summary.nonfinite_llrs != 0) {
        summary.verdict = "dirty_input";
    } else if (summary.trailing_float_bytes != 0) {
        summary.verdict = "misaligned_input_bytes";
    } else if (summary.complete_codewords == 0) {
        summary.verdict = "short_input";
    } else if (summary.trailing_codeword_llrs != 0) {
        summary.verdict = "misaligned_input_codeword";
    } else {
        summary.verdict = "quality_summarized";
    }
    if (summary.scored_codewords == 0) {
        summary.min_syndrome_weight = 0;
    }
}

void write_terminal_report(
    std::ostream& output,
    const CliOptions& options,
    const Summary& summary,
    const Accumulators& accumulators,
    const std::vector<TrackedVariableAccumulators>& tracked_variables,
    const dtmb::core::LdpcSparseGraph& graph) {
    const auto hard_h_mean = summary.scored_codewords == 0
        ? std::optional<double>{}
        : std::optional<double>{
            static_cast<double>(summary.total_syndrome_weight)
            / static_cast<double>(summary.scored_codewords * graph.check_count())};
    const auto hard_h_min = summary.scored_codewords == 0
        ? std::optional<double>{}
        : std::optional<double>{
            static_cast<double>(summary.min_syndrome_weight)
            / static_cast<double>(graph.check_count())};
    const auto hard_h_max = summary.scored_codewords == 0
        ? std::optional<double>{}
        : std::optional<double>{
            static_cast<double>(summary.max_syndrome_weight)
            / static_cast<double>(graph.check_count())};

    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\"schema\":\"dtmb.pre_ldpc_source_score.v1\""
           << ",\"stage\":\"pre_ldpc_source_score\""
           << ",\"ok\":" << (summary.scored_codewords != 0 ? "true" : "false")
           << ",\"verdict\":";
    write_json_string(output, summary.verdict);
    output << ",\"status\":";
    write_json_string(output, summary.status);
    output << ",\"input_format\":\"llr-f32\""
           << ",\"output_mode\":";
    write_json_string(output, options.passthrough ? "passthrough" : "diagnostics_only");
    output << ",\"metadata\":{\"bits_per_symbol\":" << options.bits_per_symbol
           << ",\"clean_rows_per_codeword\":" << graph.check_count()
           << ",\"codeword_bits\":" << graph.variable_count
           << ",\"fec_rate_index\":" << options.fec_rate
           << ",\"interleaver_mode\":";
    write_json_string(output, interleaver_mode_name(options.interleaver_mode));
    output << ",\"interleaver_phase\":" << options.interleaver_phase
           << ",\"source_frame_group_cap\":" << options.source_frame_group_cap
           << ",\"codewords_per_frame\":" << options.codewords_per_frame
           << ",\"fec_frame_ranges\":[";
    for (std::size_t index = 0; index < options.fec_frame_ranges.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << "{\"first\":" << options.fec_frame_ranges[index].first
               << ",\"last\":" << options.fec_frame_ranges[index].second << '}';
    }
    output << ']'
           << ",\"tag_contract\":";
    write_json_string(
        output,
        "post-deinterleaver LLR bits traced to receiver-input data symbols using "
        "the configured convolutional-deinterleaver schedule");
    output << ",\"top\":" << options.top
           << ",\"tracked_ldpc_variables\":[";
    for (std::size_t index = 0; index < tracked_variables.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << tracked_variables[index].variable;
    }
    output << ']'
           << ",\"weak_llr_threshold\":" << options.weak_llr_threshold
           << "},\"summary\":{\"complete_codewords\":" << summary.complete_codewords
           << ",\"dirty_codewords\":" << summary.dirty_codewords
           << ",\"finite_llrs\":" << summary.finite_llrs
           << ",\"included_codewords\":" << summary.included_codewords
           << ",\"input_byte_aligned\":"
           << (summary.trailing_float_bytes == 0 ? "true" : "false")
           << ",\"input_bytes\":" << summary.input_bytes
           << ",\"input_codeword_aligned\":"
           << (summary.trailing_float_bytes == 0
                   && summary.trailing_codeword_llrs == 0
               ? "true"
               : "false")
           << ",\"input_llrs\":" << summary.input_llrs
           << ",\"nonfinite_llrs\":" << summary.nonfinite_llrs
           << ",\"output_bytes\":" << summary.output_bytes
           << ",\"scored_codewords\":" << summary.scored_codewords
           << ",\"skipped_codewords\":" << summary.skipped_codewords
           << ",\"trailing_codeword_llrs\":" << summary.trailing_codeword_llrs
           << ",\"trailing_float_bytes\":" << summary.trailing_float_bytes
           << "},\"hard_h\":{\"attribution\":";
    write_json_string(output, "failed_clean_check_edges_to_participating_source_tags");
    output << ",\"clean_rows_per_codeword\":" << graph.check_count()
           << ",\"codewords\":" << summary.scored_codewords
           << ",\"failed_check_edge_ratio\":";
    write_optional(output, optional_ratio(summary.failed_check_edges, summary.observed_check_edges));
    output << ",\"failed_check_edges\":" << summary.failed_check_edges
           << ",\"max_syndrome_ratio\":";
    write_optional(output, hard_h_max);
    output << ",\"mean_syndrome_ratio\":";
    write_optional(output, hard_h_mean);
    output << ",\"min_syndrome_ratio\":";
    write_optional(output, hard_h_min);
    output << ",\"observed_check_edges\":" << summary.observed_check_edges
           << ",\"unused_bits\":" << summary.trailing_codeword_llrs
           << ",\"zero_syndrome_codewords\":" << summary.zero_syndrome_codewords
           << "},\"dimensions\":{\"codeword_slot\":";
    write_dimension(
        output,
        make_dense_report("codeword_slot", accumulators.codeword_slot),
        options.top);
    output << ",\"ldpc_variable\":";
    write_dimension(
        output,
        make_dense_report("ldpc_variable", accumulators.ldpc_variable),
        options.top);
    output << ",\"ldpc_variable_mod_127\":";
    write_dimension(
        output,
        make_dense_report("ldpc_variable_mod_127", accumulators.ldpc_variable_mod_127),
        options.top);
    output << ",\"qam_plane\":";
    write_dimension(output, make_dense_report("qam_plane", accumulators.qam_plane), options.top);
    output << ",\"source_branch\":";
    write_dimension(output, make_dense_report("source_branch", accumulators.source_branch), options.top);
    output << ",\"source_carrier\":";
    write_dimension(output, make_dense_report("source_carrier", accumulators.source_carrier), options.top);
    output << ",\"source_frame\":";
    write_dimension(output, make_source_frame_report(accumulators.source_frame), options.top);
    output << "},\"tracked_ldpc_variables\":";
    write_tracked_variables(output, tracked_variables, options.top);
    output << "}\n";
}

void write_error_diagnostic(std::ostream& diagnostics, const std::string& message) {
    diagnostics << "{\"schema\":\"dtmb.pre_ldpc_source_score.v1\","
                   "\"stage\":\"pre_ldpc_source_score\","
                   "\"ok\":false,\"verdict\":\"error\",\"status\":\"error\","
                   "\"error\":";
    write_json_string(diagnostics, message);
    diagnostics << "}\n";
}

void write_all(std::ostream& output, std::span<const char> bytes) {
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("failed to write passthrough output");
    }
}

Summary run(
    std::istream& input,
    std::ostream& binary_output,
    std::ostream& diagnostics,
    const dtmb::core::LdpcSparseGraph& graph,
    const CliOptions& options) {
    const auto total_edges = total_edges_by_variable(graph);
    const auto total_edges_per_codeword = std::accumulate(
        total_edges.begin(),
        total_edges.end(),
        std::uint64_t{0});
    const auto branch_groups = options.interleaver_mode == InterleaverMode::none
        ? std::size_t{1}
        : interleaver_spec(options.interleaver_mode).branch_count;
    const auto first_branch_id = options.interleaver_mode == InterleaverMode::none
        ? std::int64_t{-1}
        : std::int64_t{0};
    Accumulators accumulators(
        options.bits_per_symbol,
        std::max<std::size_t>(options.codewords_per_frame, 1U),
        graph.variable_count,
        branch_groups,
        first_branch_id,
        options.source_frame_group_cap);
    std::vector<TrackedVariableAccumulators> tracked_variables;
    const auto tracked_variable_ids = unique_tracked_variables(
        options.tracked_ldpc_variables,
        graph.variable_count);
    tracked_variables.reserve(tracked_variable_ids.size());
    for (const auto variable : tracked_variable_ids) {
        tracked_variables.emplace_back(
            variable,
            options.bits_per_symbol,
            std::max<std::size_t>(options.codewords_per_frame, 1U),
            branch_groups,
            first_branch_id,
            options.source_frame_group_cap);
    }

    Summary summary;
    summary.clean_rows_per_codeword = graph.check_count();
    std::vector<char> input_chunk(options.read_chunk_bytes);
    std::array<char, kFloatBytes> pending_float{};
    std::size_t pending_float_bytes = 0;
    std::vector<std::uint8_t> codeword_bits;
    std::vector<SourceTags> codeword_tags;
    std::vector<float> codeword_llrs;
    std::vector<std::uint16_t> failed_edges(graph.variable_count);
    codeword_bits.reserve(graph.variable_count);
    codeword_tags.reserve(graph.variable_count);
    codeword_llrs.reserve(graph.variable_count);
    bool codeword_dirty = false;
    const bool range_filter_enabled = !options.fec_frame_ranges.empty();

    const auto complete_codeword = [&] {
        const auto codeword_index = summary.complete_codewords;
        ++summary.complete_codewords;
        const auto include_codeword = codeword_in_selected_frame_ranges(
            codeword_index,
            options);
        if (!include_codeword) {
            ++summary.skipped_codewords;
            codeword_bits.clear();
            codeword_tags.clear();
            codeword_llrs.clear();
            codeword_dirty = false;
            return;
        }
        ++summary.included_codewords;
        const auto codeword_slot = codeword_slot_for(codeword_index, options);
        for (auto& tracked : tracked_variables) {
            const auto variable = tracked.variable;
            if (variable < codeword_llrs.size() && std::isfinite(codeword_llrs[variable])) {
                observe_tracked_llr(
                    tracked,
                    codeword_tags[variable],
                    codeword_slot,
                    codeword_llrs[variable],
                    codeword_bits[variable],
                    options.weak_llr_threshold);
            }
        }
        if (range_filter_enabled) {
            for (std::size_t bit = 0; bit < codeword_llrs.size(); ++bit) {
                if (!std::isfinite(codeword_llrs[bit])) {
                    continue;
                }
                observe_all_llr(
                    accumulators,
                    codeword_tags[bit],
                    codeword_llrs[bit],
                    codeword_bits[bit],
                    options.weak_llr_threshold);
                observe_position_llr(
                    accumulators,
                    bit,
                    codeword_slot,
                    codeword_llrs[bit],
                    codeword_bits[bit],
                    options.weak_llr_threshold);
            }
        }
        if (codeword_dirty) {
            ++summary.dirty_codewords;
        } else {
            const auto syndrome_weight = score_codeword_failed_edges(
                codeword_bits,
                graph,
                failed_edges);
            ++summary.scored_codewords;
            summary.total_syndrome_weight += syndrome_weight;
            summary.zero_syndrome_codewords += syndrome_weight == 0 ? 1U : 0U;
            summary.min_syndrome_weight = std::min(
                summary.min_syndrome_weight,
                syndrome_weight);
            summary.max_syndrome_weight = std::max(
                summary.max_syndrome_weight,
                syndrome_weight);
            summary.observed_check_edges += total_edges_per_codeword;
            summary.failed_check_edges += std::accumulate(
                failed_edges.begin(),
                failed_edges.end(),
                std::uint64_t{0});
            for (std::size_t bit = 0; bit < graph.variable_count; ++bit) {
                observe_all_hard_h(
                    accumulators,
                    codeword_tags[bit],
                    total_edges[bit],
                    failed_edges[bit]);
                observe_position_hard_h(
                    accumulators,
                    bit,
                    codeword_slot,
                    total_edges[bit],
                    failed_edges[bit]);
            }
            for (auto& tracked : tracked_variables) {
                const auto variable = tracked.variable;
                observe_tracked_hard_h(
                    tracked,
                    codeword_tags[variable],
                    codeword_slot,
                    total_edges[variable],
                    failed_edges[variable]);
            }
        }
        codeword_bits.clear();
        codeword_tags.clear();
        codeword_llrs.clear();
        codeword_dirty = false;
    };

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
        if (options.passthrough) {
            write_all(binary_output, bytes);
            summary.output_bytes += bytes.size();
        }
        for (const auto byte : bytes) {
            pending_float[pending_float_bytes++] = byte;
            if (pending_float_bytes != kFloatBytes) {
                continue;
            }

            float llr = 0.0F;
            std::memcpy(&llr, pending_float.data(), kFloatBytes);
            const auto llr_index = summary.input_llrs;
            ++summary.input_llrs;
            pending_float_bytes = 0;
            const auto tags = tags_for_llr_index(llr_index, options);
            std::uint8_t hard_bit = 0;
            if (std::isfinite(llr)) {
                ++summary.finite_llrs;
                hard_bit = llr < 0.0F ? 1U : 0U;
                if (!range_filter_enabled) {
                    const auto variable = codeword_bits.size();
                    const auto codeword_slot =
                        codeword_slot_for(summary.complete_codewords, options);
                    observe_all_llr(
                        accumulators,
                        tags,
                        llr,
                        hard_bit,
                        options.weak_llr_threshold);
                    observe_position_llr(
                        accumulators,
                        variable,
                        codeword_slot,
                        llr,
                        hard_bit,
                        options.weak_llr_threshold);
                }
            } else {
                ++summary.nonfinite_llrs;
                codeword_dirty = true;
            }
            codeword_bits.push_back(hard_bit);
            codeword_tags.push_back(tags);
            codeword_llrs.push_back(llr);
            if (codeword_bits.size() == graph.variable_count) {
                complete_codeword();
            }
        }
        if (bytes_read < input_chunk.size()) {
            if (input.bad()) {
                throw std::runtime_error("failed to read LLR input stream");
            }
            break;
        }
    }

    summary.trailing_float_bytes = pending_float_bytes;
    summary.trailing_codeword_llrs = codeword_bits.size();
    finalize_summary(summary);
    write_terminal_report(
        diagnostics,
        options,
        summary,
        accumulators,
        tracked_variables,
        graph);
    if (options.passthrough) {
        binary_output.flush();
        if (!binary_output) {
            throw std::runtime_error("failed to flush passthrough output");
        }
    }
    diagnostics.flush();
    if (!diagnostics) {
        throw std::runtime_error("failed to flush diagnostics");
    }
    return summary;
}

}  // namespace

int main(int argc, char** argv) {
    std::ostream* diagnostics = &std::cerr;
    std::unique_ptr<std::ifstream> input_file;
    std::unique_ptr<std::ofstream> diagnostics_file;
    try {
        std::cout.imbue(std::locale::classic());
        std::cerr.imbue(std::locale::classic());
        const auto options = parse_args(argc, argv);
        dtmb::tools::configure_binary_stdio(options.input_path == "-", options.passthrough);

        auto& input = input_stream(options.input_path, input_file);
        auto& diagnostics_output = diagnostics_stream(
            options.diagnostics_path,
            diagnostics_file);
        diagnostics = &diagnostics_output;
        diagnostics_output.imbue(std::locale::classic());

        const auto graph = dtmb::tools::ldpc_h_gate::load_clean_dtmb_graph(
            options.alist_path,
            options.fec_rate);
        (void)run(input, std::cout, diagnostics_output, graph, options);
        return 0;
    } catch (const std::exception& exc) {
        write_error_diagnostic(*diagnostics, exc.what());
        diagnostics->flush();
        return 1;
    }
}
