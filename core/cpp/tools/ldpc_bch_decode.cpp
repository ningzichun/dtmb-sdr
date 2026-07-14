#include "dtmb/core.hpp"

#include "binary_stdio.hpp"
#include "ldpc_cuda_backend.hpp"
#include "ldpc_h_gate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct DtmbLdpcProfile {
    std::size_t fec_rate = 0;
    std::size_t check_blocks = 0;
    std::size_t message_bits = 0;
    std::size_t transmitted_bits = 7488;
    std::size_t erased_parity_bits = 5;
    std::size_t circulant_size = 127;

    [[nodiscard]] std::size_t full_parity_bits() const noexcept {
        return check_blocks * circulant_size;
    }

    [[nodiscard]] std::size_t full_codeword_bits() const noexcept {
        return full_parity_bits() + message_bits;
    }

    [[nodiscard]] std::size_t output_bytes_per_codeword() const noexcept {
        return (message_bits / 762) * 94;
    }
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --fec-rate 1|2|3 --alist PATH [--codewords-per-frame N]"
        << " [--workers N] [--max-iterations N] [--retry-max-iterations N]"
        << " [--attenuation X] [--llr-clip X]"
        << " [--ldpc-accel cpu|cuda]"
        << " [--retry-llr-clip X] [--retry-weak-erase-fraction X]"
        << " [--retry-llr-scale X]"
        << " [--cuda-retry-clips]"
        << " [--retry-attenuation X]"
        << " [--retry-unsatisfied-erase-count N] [--retry-layered]"
        << " [--retry-final-failed-erase-count N]"
        << " [--retry-final-failed-flip-count N]"
        << " [--retry-final-failed-greedy-flip-count N]"
        << " [--retry-final-failed-syndrome-solve-count N]"
        << " [--retry-final-failed-syndrome-solve-minimize-cost-passes N]"
        << " [--retry-final-failed-syndrome-solve-nullspace-search N]"
        << " [--retry-final-failed-syndrome-solve-nullspace-depth N]"
        << " [--retry-final-failed-syndrome-solve-ts-continuity-headers]"
        << " [--retry-final-failed-syndrome-solve-ts-continuity-header-templates]"
        << " [--retry-final-failed-syndrome-solve-ts-header-template FEC:SLOT:PACKET:PID:PUSI:AFC:CC]"
        << " [--retry-require-ts-packet-syntax]"
        << " [--retry-require-ts-packet-continuity]"
        << " [--retry-qam-plane N]"
        << " [--retry-variable-range FIRST:LAST]"
        << " [--retry-branch-window FIRST:LAST]"
        << " [--retry-branch-window-group FIRST:LAST,FIRST:LAST]"
        << " [--retry-branch-window-sweep FIRST:LAST:MIN_WIDTH:MAX_WIDTH]"
        << " [--retry-branch-window-frame-range FIRST:LAST]"
        << " [--retry-branch-window-for-frame-range FEC_FIRST:FEC_LAST:BRANCH_FIRST:BRANCH_LAST]"
        << " [--retry-mode2-source-frame-window FIRST:LAST]"
        << " [--retry-mode2-source-frame-window-for-frame-range FEC_FIRST:FEC_LAST:SOURCE_FIRST:SOURCE_LAST]"
        << " [--select-mode2-source-frame-window FIRST:LAST]"
        << " [--select-mode2-source-frame-window-for-frame-range FEC_FIRST:FEC_LAST:SOURCE_FIRST:SOURCE_LAST]"
        << " [--decode-batch-frames N]"
        << " [--early-syndrome-reject-ratio off|X]"
        << " [--clean-frames-only] [--fail-on-unclean-frame] [--mark-discontinuities]"
        << " [--emit-clean-codewords]"
        << " [--emit-bch-clean-codewords]"
        << " [--emit-bch-clean-packets]"
        << " [--insert-discontinuity-packets]"
        << " [--frame-index-offset N]"
        << " [--hard-h-gate-window-codewords N"
        << " --hard-h-gate-step-codewords N --hard-h-gate-threshold R]"
        << " [--hard-h-gate-fill-max-gap-frames N]"
        << " [--force-frame-range FIRST:LAST]"
        << " [--omit-frame-range FIRST:LAST]"
        << " [--llr-scale-codeword FIRST:LAST:SLOT:SCALE]"
        << " [--llr-scale-variable FIRST:LAST:VARIABLE:SCALE]"
        << " [--llr-scale-variable-mod FIRST:LAST:MOD:REMAINDER:SCALE]"
        << " [--emit-forced-codewords]"
        << " [--frame-diagnostics-out PATH]"
        << " [--failed-variable-diagnostics-top N]"
        << " [--packet-provenance-out PATH]"
        << " [input.llr.f32|-] [output.ts|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

float parse_float(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stof(text, &parsed);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

double parse_double(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stod(text, &parsed);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return value;
}

struct FrameRange {
    std::size_t first = 0;
    std::size_t last = 0;
};

struct CodewordLlrScaleRule {
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t codeword = 0;
    float scale = 1.0F;
};

struct VariableLlrScaleRule {
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t variable = 0;
    float scale = 1.0F;
};

struct VariableModLlrScaleRule {
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t modulus = 0;
    std::size_t remainder = 0;
    float scale = 1.0F;
};

struct BranchWindow {
    std::size_t first = 0;
    std::size_t last = 0;
};

struct BranchWindowRule {
    FrameRange fec_frame_range;
    BranchWindow branch_window;
};

struct SourceFrameWindowRule {
    FrameRange fec_frame_range;
    FrameRange source_frame_range;
};

struct VariableRange {
    std::size_t first = 0;
    std::size_t last = 0;
};

struct TsHeaderTemplateRule {
    std::size_t fec_frame = 0;
    std::size_t codeword = 0;
    std::size_t packet = 0;
    std::uint16_t pid = 0;
    std::uint8_t payload_unit_start = 0;
    std::uint8_t adaptation_control = 0;
    std::uint8_t continuity_counter = 0;
};

constexpr std::size_t kQam64LlrsPerSymbol = 6U;
constexpr std::size_t kSymbolInterleaverBranches = 52U;
constexpr std::size_t kNoRetryBranchWindow = kSymbolInterleaverBranches;
constexpr std::size_t kNoRetryQamPlane = kQam64LlrsPerSymbol;
constexpr std::size_t kNoRetryVariable = std::numeric_limits<std::size_t>::max();

enum class LdpcAccel {
    cpu,
    cuda,
};

LdpcAccel parse_ldpc_accel(const std::string& text) {
    if (text == "cpu") {
        return LdpcAccel::cpu;
    }
    if (text == "cuda") {
        return LdpcAccel::cuda;
    }
    throw std::invalid_argument("LDPC accelerator must be cpu or cuda");
}

const char* ldpc_accel_name(LdpcAccel accel) noexcept {
    switch (accel) {
    case LdpcAccel::cpu:
        return "cpu";
    case LdpcAccel::cuda:
        return "cuda";
    }
    return "unknown";
}

FrameRange parse_frame_range(const std::string& text) {
    const auto separator = text.find(':');
    if (separator == std::string::npos) {
        throw std::invalid_argument("invalid frame range, expected FIRST:LAST: " + text);
    }
    const auto first = parse_size(text.substr(0, separator), "frame range first");
    const auto last = parse_size(text.substr(separator + 1), "frame range last");
    if (first == 0 || last < first) {
        throw std::invalid_argument("invalid frame range, expected 1-based FIRST:LAST");
    }
    return FrameRange{first, last};
}

CodewordLlrScaleRule parse_codeword_llr_scale_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(':', start);
        parts.push_back(text.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (parts.size() != 4) {
        throw std::invalid_argument(
            "invalid codeword LLR scale rule, expected FIRST:LAST:SLOT:SCALE: " + text);
    }
    CodewordLlrScaleRule rule;
    rule.first_frame = parse_size(parts[0], "codeword LLR scale first frame");
    rule.last_frame = parse_size(parts[1], "codeword LLR scale last frame");
    rule.codeword = parse_size(parts[2], "codeword LLR scale slot");
    rule.scale = parse_float(parts[3], "codeword LLR scale");
    if (rule.first_frame == 0 || rule.last_frame < rule.first_frame
        || !std::isfinite(rule.scale) || rule.scale < 0.0F) {
        throw std::invalid_argument(
            "invalid codeword LLR scale rule, expected 1-based range and non-negative scale");
    }
    return rule;
}

VariableLlrScaleRule parse_variable_llr_scale_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(':', start);
        parts.push_back(text.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (parts.size() != 4) {
        throw std::invalid_argument(
            "invalid variable LLR scale rule, expected FIRST:LAST:VARIABLE:SCALE: "
            + text);
    }
    VariableLlrScaleRule rule;
    rule.first_frame = parse_size(parts[0], "variable LLR scale first frame");
    rule.last_frame = parse_size(parts[1], "variable LLR scale last frame");
    rule.variable = parse_size(parts[2], "variable LLR scale variable");
    rule.scale = parse_float(parts[3], "variable LLR scale");
    if (rule.first_frame == 0 || rule.last_frame < rule.first_frame
        || !std::isfinite(rule.scale) || rule.scale < 0.0F) {
        throw std::invalid_argument(
            "invalid variable LLR scale rule, expected 1-based range and non-negative scale");
    }
    return rule;
}

VariableModLlrScaleRule parse_variable_mod_llr_scale_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(':', start);
        parts.push_back(text.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (parts.size() != 5) {
        throw std::invalid_argument(
            "invalid variable-mod LLR scale rule, expected FIRST:LAST:MOD:REMAINDER:SCALE: "
            + text);
    }
    VariableModLlrScaleRule rule;
    rule.first_frame = parse_size(parts[0], "variable-mod LLR scale first frame");
    rule.last_frame = parse_size(parts[1], "variable-mod LLR scale last frame");
    rule.modulus = parse_size(parts[2], "variable-mod LLR scale modulus");
    rule.remainder = parse_size(parts[3], "variable-mod LLR scale remainder");
    rule.scale = parse_float(parts[4], "variable-mod LLR scale");
    if (rule.first_frame == 0 || rule.last_frame < rule.first_frame
        || rule.modulus == 0 || rule.remainder >= rule.modulus
        || !std::isfinite(rule.scale) || rule.scale < 0.0F) {
        throw std::invalid_argument(
            "invalid variable-mod LLR scale rule, expected 1-based range, positive modulus, "
            "remainder < modulus, and non-negative scale");
    }
    return rule;
}

BranchWindow parse_branch_window(const std::string& text) {
    const auto separator = text.find(':');
    if (separator == std::string::npos) {
        throw std::invalid_argument(
            "invalid branch retry window, expected FIRST:LAST: " + text);
    }
    BranchWindow window;
    window.first = parse_size(text.substr(0, separator), "branch retry window first");
    window.last = parse_size(text.substr(separator + 1), "branch retry window last");
    if (window.last < window.first) {
        throw std::invalid_argument(
            "invalid branch retry window, expected FIRST <= LAST");
    }
    return window;
}

FrameRange parse_source_frame_range(const std::string& text) {
    const auto separator = text.find(':');
    if (separator == std::string::npos) {
        throw std::invalid_argument(
            "invalid source-frame range, expected FIRST:LAST: " + text);
    }
    const auto first = parse_size(
        text.substr(0, separator),
        "source-frame range first");
    const auto last = parse_size(
        text.substr(separator + 1),
        "source-frame range last");
    if (last < first) {
        throw std::invalid_argument(
            "invalid source-frame range, expected zero-based FIRST:LAST");
    }
    return FrameRange{first, last};
}

std::vector<BranchWindow> parse_branch_window_group(const std::string& text) {
    std::vector<BranchWindow> group;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(',', start);
        const auto part = text.substr(start, separator - start);
        if (part.empty()) {
            throw std::invalid_argument(
                "invalid branch retry window group, expected comma-separated FIRST:LAST windows: "
                + text);
        }
        group.push_back(parse_branch_window(part));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (group.empty()) {
        throw std::invalid_argument(
            "invalid branch retry window group, expected at least one window");
    }
    return group;
}

BranchWindowRule parse_branch_window_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(':', start);
        parts.push_back(text.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (parts.size() != 4) {
        throw std::invalid_argument(
            "invalid branch retry rule, expected "
            "FEC_FIRST:FEC_LAST:BRANCH_FIRST:BRANCH_LAST: "
            + text);
    }
    BranchWindowRule rule;
    rule.fec_frame_range.first = parse_size(parts[0], "branch retry FEC first");
    rule.fec_frame_range.last = parse_size(parts[1], "branch retry FEC last");
    rule.branch_window.first = parse_size(parts[2], "branch retry first branch");
    rule.branch_window.last = parse_size(parts[3], "branch retry last branch");
    if (rule.fec_frame_range.first == 0
        || rule.fec_frame_range.last < rule.fec_frame_range.first
        || rule.branch_window.last < rule.branch_window.first) {
        throw std::invalid_argument(
            "invalid branch retry rule, expected 1-based FEC range and FIRST <= LAST branch range");
    }
    return rule;
}

VariableRange parse_variable_range(const std::string& text) {
    const auto separator = text.find(':');
    if (separator == std::string::npos) {
        throw std::invalid_argument(
            "invalid variable retry range, expected FIRST:LAST: " + text);
    }
    VariableRange range;
    range.first = parse_size(text.substr(0, separator), "variable retry range first");
    range.last = parse_size(text.substr(separator + 1), "variable retry range last");
    if (range.last < range.first) {
        throw std::invalid_argument(
            "invalid variable retry range, expected FIRST <= LAST");
    }
    return range;
}

SourceFrameWindowRule parse_source_frame_window_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(':', start);
        parts.push_back(text.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (parts.size() != 4) {
        throw std::invalid_argument(
            "invalid source-frame retry rule, expected "
            "FEC_FIRST:FEC_LAST:SOURCE_FIRST:SOURCE_LAST: "
            + text);
    }
    SourceFrameWindowRule rule;
    rule.fec_frame_range.first = parse_size(parts[0], "source retry FEC first");
    rule.fec_frame_range.last = parse_size(parts[1], "source retry FEC last");
    rule.source_frame_range.first = parse_size(parts[2], "source retry source first");
    rule.source_frame_range.last = parse_size(parts[3], "source retry source last");
    if (rule.fec_frame_range.first == 0
        || rule.fec_frame_range.last < rule.fec_frame_range.first
        || rule.source_frame_range.last < rule.source_frame_range.first) {
        throw std::invalid_argument(
            "invalid source-frame retry rule, expected 1-based FEC range and zero-based source range");
    }
    return rule;
}

TsHeaderTemplateRule parse_ts_header_template_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(':', start);
        parts.push_back(text.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (parts.size() != 7U) {
        throw std::invalid_argument(
            "invalid TS header template, expected FEC:SLOT:PACKET:PID:PUSI:AFC:CC: "
            + text);
    }
    TsHeaderTemplateRule rule;
    rule.fec_frame = parse_size(parts[0], "TS header template FEC frame");
    rule.codeword = parse_size(parts[1], "TS header template codeword slot");
    rule.packet = parse_size(parts[2], "TS header template packet index");
    const auto pid = parse_size(parts[3], "TS header template PID");
    const auto pusi = parse_size(parts[4], "TS header template PUSI");
    const auto afc = parse_size(parts[5], "TS header template adaptation control");
    const auto cc = parse_size(parts[6], "TS header template continuity counter");
    if (pid > 0x1FFFU) {
        throw std::invalid_argument("TS header template PID must be within 0..8191");
    }
    if (pusi > 1U) {
        throw std::invalid_argument("TS header template PUSI must be 0 or 1");
    }
    if (afc == 0U || afc > 3U) {
        throw std::invalid_argument("TS header template adaptation control must be 1..3");
    }
    if (cc > 15U) {
        throw std::invalid_argument("TS header template continuity counter must be 0..15");
    }
    rule.pid = static_cast<std::uint16_t>(pid);
    rule.payload_unit_start = static_cast<std::uint8_t>(pusi);
    rule.adaptation_control = static_cast<std::uint8_t>(afc);
    rule.continuity_counter = static_cast<std::uint8_t>(cc);
    return rule;
}

std::vector<BranchWindow> parse_branch_window_sweep(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto separator = text.find(':', start);
        parts.push_back(text.substr(start, separator - start));
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    if (parts.size() != 4) {
        throw std::invalid_argument(
            "invalid branch retry sweep, expected FIRST:LAST:MIN_WIDTH:MAX_WIDTH: "
            + text);
    }
    const auto first = parse_size(parts[0], "branch retry sweep first");
    const auto last = parse_size(parts[1], "branch retry sweep last");
    const auto min_width = parse_size(parts[2], "branch retry sweep min width");
    const auto max_width = parse_size(parts[3], "branch retry sweep max width");
    if (last < first || min_width == 0U || max_width < min_width) {
        throw std::invalid_argument(
            "invalid branch retry sweep bounds");
    }
    std::vector<BranchWindow> windows;
    for (std::size_t window_first = first; window_first <= last; ++window_first) {
        for (std::size_t width = min_width; width <= max_width; ++width) {
            const auto window_last = window_first + width - 1U;
            if (window_last > last) {
                break;
            }
            windows.push_back(BranchWindow{window_first, window_last});
        }
    }
    return windows;
}

bool frame_in_ranges(std::size_t frame_index, std::span<const FrameRange> ranges) {
    return std::any_of(
        ranges.begin(),
        ranges.end(),
        [frame_index](const FrameRange& range) {
            return frame_index >= range.first && frame_index <= range.last;
        });
}

std::vector<BranchWindow> active_branch_windows(
    std::size_t frame_index,
    std::span<const BranchWindow> global_branch_windows,
    std::span<const FrameRange> global_branch_window_frame_ranges,
    std::span<const BranchWindowRule> branch_window_rules) {
    std::vector<BranchWindow> active;
    const auto global_allowed =
        !global_branch_windows.empty()
        && (global_branch_window_frame_ranges.empty()
            || frame_in_ranges(frame_index, global_branch_window_frame_ranges));
    if (global_allowed) {
        active.insert(
            active.end(),
            global_branch_windows.begin(),
            global_branch_windows.end());
    }
    for (const auto& rule : branch_window_rules) {
        if (frame_index < rule.fec_frame_range.first
            || frame_index > rule.fec_frame_range.last) {
            continue;
        }
        active.push_back(rule.branch_window);
    }
    return active;
}

std::vector<std::vector<BranchWindow>> active_branch_window_groups(
    std::size_t frame_index,
    std::span<const std::vector<BranchWindow>> global_branch_window_groups,
    std::span<const FrameRange> global_branch_window_frame_ranges) {
    std::vector<std::vector<BranchWindow>> active;
    const auto global_active =
        !global_branch_window_groups.empty()
        && (global_branch_window_frame_ranges.empty()
            || frame_in_ranges(frame_index, global_branch_window_frame_ranges));
    if (global_active) {
        active.insert(
            active.end(),
            global_branch_window_groups.begin(),
            global_branch_window_groups.end());
    }
    return active;
}

std::vector<BranchWindow> mode2_source_frame_branch_windows(
    std::size_t frame_index,
    std::span<const FrameRange> source_frame_windows) {
    constexpr std::size_t kMode2SourceFrameStridePerBranch = 10U;
    std::vector<BranchWindow> branch_windows;
    for (const auto& source_range : source_frame_windows) {
        std::optional<std::size_t> first_branch;
        std::size_t last_branch = 0;
        for (std::size_t branch = 0; branch < kSymbolInterleaverBranches; ++branch) {
            const auto source_frame =
                frame_index - 1U + kMode2SourceFrameStridePerBranch * branch;
            if (source_frame < source_range.first || source_frame > source_range.last) {
                continue;
            }
            if (!first_branch.has_value()) {
                first_branch = branch;
            }
            last_branch = branch;
        }
        if (first_branch.has_value()) {
            branch_windows.push_back(BranchWindow{*first_branch, last_branch});
        }
    }
    return branch_windows;
}

std::vector<FrameRange> active_mode2_source_frame_windows(
    std::size_t frame_index,
    std::span<const FrameRange> global_source_frame_windows,
    std::span<const SourceFrameWindowRule> source_frame_window_rules) {
    std::vector<FrameRange> active(
        global_source_frame_windows.begin(),
        global_source_frame_windows.end());
    for (const auto& rule : source_frame_window_rules) {
        if (frame_index < rule.fec_frame_range.first
            || frame_index > rule.fec_frame_range.last) {
            continue;
        }
        active.push_back(rule.source_frame_range);
    }
    return active;
}

std::vector<BranchWindow> add_merged_branch_window_candidates(
    std::vector<BranchWindow> branch_windows) {
    if (branch_windows.size() < 2U) {
        return branch_windows;
    }

    auto ordered = branch_windows;
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const BranchWindow& left, const BranchWindow& right) {
            if (left.first != right.first) {
                return left.first < right.first;
            }
            return left.last < right.last;
        });

    std::vector<BranchWindow> merged;
    for (std::size_t first_index = 0; first_index < ordered.size(); ++first_index) {
        BranchWindow current = ordered[first_index];
        bool merged_multiple = false;
        for (std::size_t last_index = first_index + 1U;
             last_index < ordered.size();
             ++last_index) {
            const auto& next = ordered[last_index];
            if (next.first > current.last + 1U) {
                break;
            }
            current.last = std::max(current.last, next.last);
            merged_multiple = true;
            if (merged_multiple) {
                merged.push_back(current);
            }
        }
    }

    for (const auto& candidate : merged) {
        const auto duplicate = std::any_of(
            branch_windows.begin(),
            branch_windows.end(),
            [&candidate](const BranchWindow& existing) {
                return existing.first == candidate.first
                    && existing.last == candidate.last;
            });
        if (!duplicate) {
            branch_windows.push_back(candidate);
        }
    }
    return branch_windows;
}

DtmbLdpcProfile profile_for_rate(std::size_t rate) {
    switch (rate) {
    case 1:
        return DtmbLdpcProfile{1, 35, 3048};
    case 2:
        return DtmbLdpcProfile{2, 23, 4572};
    case 3:
        return DtmbLdpcProfile{3, 11, 6096};
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

std::ostream& output_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cout;
    }
    file_holder = std::make_unique<std::ofstream>(path, std::ios::binary);
    if (!*file_holder) {
        throw std::runtime_error("failed to open output: " + path);
    }
    return *file_holder;
}

void write_all(std::ostream& output, std::span<const std::uint8_t> values) {
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size()));
    if (!output) {
        throw std::runtime_error("failed to write decoded transport-byte stream");
    }
}

constexpr std::uint8_t kTsSyncByte = 0x47U;
constexpr std::size_t kTsPacketSize = 188;
constexpr std::uint16_t kTsNullPid = 0x1FFFU;

[[nodiscard]] bool ts_packet_syntax_valid(std::span<const std::uint8_t> packet) {
    if (packet.size() < kTsPacketSize || packet[0] != kTsSyncByte) {
        return false;
    }
    if ((packet[1] & 0x80U) != 0U) {
        return false;
    }
    if ((packet[3] & 0xC0U) != 0U) {
        return false;
    }
    const auto adaptation_control = static_cast<std::uint8_t>((packet[3] >> 4U) & 0x03U);
    if (adaptation_control == 0U) {
        return false;
    }
    if (adaptation_control == 2U || adaptation_control == 3U) {
        const auto adaptation_length = packet[4];
        if (5U + adaptation_length > kTsPacketSize) {
            return false;
        }
        if (adaptation_length >= 1U
            && (packet[5] & 0x10U) != 0U
            && adaptation_length < 7U) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool ts_packets_syntax_valid(std::span<const std::uint8_t> bytes) {
    if (bytes.empty() || (bytes.size() % kTsPacketSize) != 0U) {
        return false;
    }
    for (std::size_t offset = 0; offset < bytes.size(); offset += kTsPacketSize) {
        if (!ts_packet_syntax_valid(bytes.subspan(offset, kTsPacketSize))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::uint16_t ts_pid(std::span<const std::uint8_t> packet) {
    return static_cast<std::uint16_t>(((packet[1] & 0x1FU) << 8U) | packet[2]);
}

[[nodiscard]] bool ts_has_payload(std::span<const std::uint8_t> packet) {
    if (packet.size() < kTsPacketSize || packet[0] != kTsSyncByte) {
        return false;
    }
    const auto adaptation_control = static_cast<std::uint8_t>((packet[3] >> 4U) & 0x03U);
    return adaptation_control == 1U || adaptation_control == 3U;
}

[[nodiscard]] bool ts_payload_unit_start(std::span<const std::uint8_t> packet) {
    return packet.size() >= kTsPacketSize
        && packet[0] == kTsSyncByte
        && (packet[1] & 0x40U) != 0U;
}

[[nodiscard]] std::uint8_t ts_continuity_counter(std::span<const std::uint8_t> packet) {
    if (packet.size() < kTsPacketSize || packet[0] != kTsSyncByte) {
        return 0U;
    }
    return static_cast<std::uint8_t>(packet[3] & 0x0FU);
}

[[nodiscard]] bool ts_discontinuity_indicator(std::span<const std::uint8_t> packet) {
    if (packet.size() < kTsPacketSize || packet[0] != kTsSyncByte) {
        return false;
    }
    const auto adaptation_control = static_cast<std::uint8_t>((packet[3] >> 4U) & 0x03U);
    if (adaptation_control != 2U && adaptation_control != 3U) {
        return false;
    }
    const auto adaptation_length = packet[4];
    return adaptation_length > 0U
        && 5U + adaptation_length <= kTsPacketSize
        && (packet[5] & 0x80U) != 0U;
}

[[nodiscard]] bool mark_packet_discontinuity(std::span<std::uint8_t> packet) {
    if (packet.size() < kTsPacketSize || packet[0] != kTsSyncByte) {
        return false;
    }
    const auto pid = ts_pid(packet);
    if (pid == kTsNullPid) {
        return false;
    }
    const auto adaptation_control = static_cast<std::uint8_t>((packet[3] >> 4U) & 0x03U);
    if (adaptation_control != 2U && adaptation_control != 3U) {
        return false;
    }
    const auto adaptation_length = packet[4];
    if (adaptation_length == 0U || 5U + adaptation_length > kTsPacketSize) {
        return false;
    }
    packet[5] |= 0x80U;
    return true;
}

[[nodiscard]] std::size_t mark_frame_discontinuities_for_pending_pids(
    std::span<std::uint8_t> frame_bytes,
    const std::unordered_set<std::uint16_t>& pending_pids,
    std::unordered_set<std::uint16_t>& signaled_pids) {
    std::size_t marked = 0;
    for (std::size_t offset = 0; offset + kTsPacketSize <= frame_bytes.size();
         offset += kTsPacketSize) {
        auto packet = frame_bytes.subspan(offset, kTsPacketSize);
        if (packet[0] != kTsSyncByte || !ts_has_payload(packet)) {
            continue;
        }
        const auto pid = ts_pid(packet);
        if (pid == kTsNullPid || !pending_pids.contains(pid) || signaled_pids.contains(pid)) {
            continue;
        }
        if (mark_packet_discontinuity(packet)) {
            signaled_pids.insert(pid);
            ++marked;
        }
    }
    return marked;
}

[[nodiscard]] std::vector<std::array<std::uint8_t, kTsPacketSize>>
make_discontinuity_packets_for_pending_pids(
    std::span<const std::uint8_t> frame_bytes,
    const std::unordered_set<std::uint16_t>& pending_pids,
    std::unordered_set<std::uint16_t>& signaled_pids) {
    std::vector<std::array<std::uint8_t, kTsPacketSize>> packets;
    for (std::size_t offset = 0; offset + kTsPacketSize <= frame_bytes.size();
         offset += kTsPacketSize) {
        const auto packet = frame_bytes.subspan(offset, kTsPacketSize);
        if (packet[0] != kTsSyncByte || !ts_has_payload(packet)) {
            continue;
        }
        const auto pid = ts_pid(packet);
        if (pid == kTsNullPid || !pending_pids.contains(pid) || signaled_pids.contains(pid)) {
            continue;
        }
        signaled_pids.insert(pid);
        std::array<std::uint8_t, kTsPacketSize> discontinuity{};
        discontinuity.fill(0xFFU);
        discontinuity[0] = kTsSyncByte;
        discontinuity[1] = static_cast<std::uint8_t>((pid >> 8U) & 0x1FU);
        discontinuity[2] = static_cast<std::uint8_t>(pid & 0xFFU);
        discontinuity[3] = static_cast<std::uint8_t>(0x20U | (packet[3] & 0x0FU));
        discontinuity[4] = static_cast<std::uint8_t>(kTsPacketSize - 5U);
        discontinuity[5] = 0x80U;
        packets.push_back(discontinuity);
    }
    return packets;
}

void collect_payload_pids(
    std::span<const std::uint8_t> frame_bytes,
    std::unordered_set<std::uint16_t>& output_pids) {
    for (std::size_t offset = 0; offset + kTsPacketSize <= frame_bytes.size();
         offset += kTsPacketSize) {
        const auto packet = frame_bytes.subspan(offset, kTsPacketSize);
        if (packet[0] != kTsSyncByte || !ts_has_payload(packet)) {
            continue;
        }
        const auto pid = ts_pid(packet);
        if (pid != kTsNullPid) {
            output_pids.insert(pid);
        }
    }
}

void write_json_array(
    std::ostream& output,
    const char* name,
    std::span<const std::size_t> values) {
    output << ",\"" << name << "\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

void write_nullable_size(std::ostream& output, std::optional<std::size_t> value) {
    if (value.has_value()) {
        output << *value;
    } else {
        output << "null";
    }
}

void write_packet_provenance_row(
    std::ostream* packet_provenance,
    std::size_t output_packet_index,
    const char* kind,
    std::size_t frame_index,
    std::optional<std::size_t> codeword_slot,
    std::optional<std::size_t> packet_in_codeword,
    std::optional<std::size_t> packets_per_codeword,
    std::span<const std::uint8_t> packet) {
    if (packet_provenance == nullptr) {
        return;
    }
    const auto pid = ts_pid(packet);
    *packet_provenance
        << "{\"schema\":\"dtmb.ldpc_bch_packet_provenance.v1\""
        << ",\"output_packet_index\":" << output_packet_index
        << ",\"kind\":\"" << kind << "\""
        << ",\"frame_index\":" << frame_index
        << ",\"codeword_slot\":";
    write_nullable_size(*packet_provenance, codeword_slot);
    *packet_provenance << ",\"packet_in_codeword\":";
    write_nullable_size(*packet_provenance, packet_in_codeword);
    *packet_provenance << ",\"packets_per_codeword\":";
    write_nullable_size(*packet_provenance, packets_per_codeword);
    *packet_provenance
        << ",\"pid\":" << pid
        << ",\"pid_hex\":\"0x" << std::hex << std::nouppercase
        << std::setw(4) << std::setfill('0') << pid
        << std::dec << std::setfill(' ') << "\""
        << ",\"payload_unit_start\":"
        << (ts_payload_unit_start(packet) ? "true" : "false")
        << ",\"has_payload\":" << (ts_has_payload(packet) ? "true" : "false")
        << ",\"continuity_counter\":"
        << static_cast<unsigned>(ts_continuity_counter(packet))
        << ",\"discontinuity_indicator\":"
        << (ts_discontinuity_indicator(packet) ? "true" : "false")
        << "}\n";
    if (!*packet_provenance) {
        throw std::runtime_error("failed to write packet provenance row");
    }
}

void write_json_array(
    std::ostream& output,
    const char* name,
    std::span<const std::uint8_t> values) {
    output << ",\"" << name << "\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << static_cast<unsigned>(values[index]);
    }
    output << ']';
}

void write_json_array(
    std::ostream& output,
    const char* name,
    std::span<const double> values) {
    output << ",\"" << name << "\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const char ch : value) {
        switch (ch) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        default:
            output << ch;
            break;
        }
    }
    output << '"';
}

void write_json_array(
    std::ostream& output,
    const char* name,
    std::span<const std::string> values) {
    output << ",\"" << name << "\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        write_json_string(output, values[index]);
    }
    output << ']';
}

std::string branch_window_group_label(std::span<const BranchWindow> windows) {
    std::string label;
    for (std::size_t index = 0; index < windows.size(); ++index) {
        if (index != 0) {
            label += ',';
        }
        label += std::to_string(windows[index].first);
        label += ':';
        label += std::to_string(windows[index].last);
    }
    return label;
}

struct FailedVariableTopRow {
    std::size_t full_variable_index = 0;
    std::optional<std::size_t> transmitted_bit_index;
    const char* section = "unknown";
    std::size_t section_bit_index = 0;
    std::optional<std::size_t> qam_plane;
    std::uint8_t decoded_bit = 0;
    std::size_t check_degree = 0;
    std::size_t satisfied_check_count = 0;
    std::size_t unsatisfied_check_count = 0;
    long long flip_syndrome_delta = 0;
};

struct FailedVariableSummary {
    std::size_t clean_check_count = 0;
    std::size_t unsatisfied_clean_check_count = 0;
    std::size_t full_hard_one_count = 0;
    std::size_t transmitted_hard_one_count = 0;
    std::array<std::size_t, 6> qam_plane_involvement_counts{};
    std::size_t erased_parity_involvement_count = 0;
    std::size_t transmitted_parity_involvement_count = 0;
    std::size_t message_involvement_count = 0;
    std::vector<FailedVariableTopRow> top_variables;
};

[[nodiscard]] const char* ldpc_variable_section(
    std::size_t full_variable_index,
    const DtmbLdpcProfile& profile) {
    if (full_variable_index < profile.erased_parity_bits) {
        return "erased_parity";
    }
    if (full_variable_index < profile.full_parity_bits()) {
        return "transmitted_parity";
    }
    return "message";
}

[[nodiscard]] std::size_t ldpc_section_bit_index(
    std::size_t full_variable_index,
    const DtmbLdpcProfile& profile) {
    if (full_variable_index < profile.erased_parity_bits) {
        return full_variable_index;
    }
    if (full_variable_index < profile.full_parity_bits()) {
        return full_variable_index - profile.erased_parity_bits;
    }
    return full_variable_index - profile.full_parity_bits();
}

[[nodiscard]] std::optional<std::size_t> transmitted_bit_index_for_full_variable(
    std::size_t full_variable_index,
    const DtmbLdpcProfile& profile) {
    if (full_variable_index < profile.erased_parity_bits) {
        return std::nullopt;
    }
    return full_variable_index - profile.erased_parity_bits;
}

[[nodiscard]] FailedVariableSummary summarize_failed_variables(
    std::span<const std::uint8_t> full_bits,
    const dtmb::core::LdpcSparseGraph& graph,
    const DtmbLdpcProfile& profile,
    std::size_t top_count) {
    FailedVariableSummary summary;
    summary.clean_check_count = graph.check_count();
    for (std::size_t variable = 0; variable < full_bits.size(); ++variable) {
        if ((full_bits[variable] & 0x01U) == 0U) {
            continue;
        }
        ++summary.full_hard_one_count;
        if (variable >= profile.erased_parity_bits) {
            ++summary.transmitted_hard_one_count;
        }
    }

    std::vector<std::size_t> variable_counts(graph.variable_count, 0U);
    std::vector<std::size_t> variable_degrees(graph.variable_count, 0U);
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        std::uint8_t parity = 0;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            const auto variable = graph.edge_variables[edge];
            ++variable_degrees[variable];
            parity ^= static_cast<std::uint8_t>(full_bits[variable] & 0x01U);
        }
        if (parity == 0U) {
            continue;
        }
        ++summary.unsatisfied_clean_check_count;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            ++variable_counts[graph.edge_variables[edge]];
        }
    }

    summary.top_variables.reserve(std::min(top_count, graph.variable_count));
    for (std::size_t variable = 0; variable < variable_counts.size(); ++variable) {
        const auto count = variable_counts[variable];
        if (count == 0U) {
            continue;
        }
        const auto section = ldpc_variable_section(variable, profile);
        if (std::string(section) == "erased_parity") {
            summary.erased_parity_involvement_count += count;
        } else if (std::string(section) == "transmitted_parity") {
            summary.transmitted_parity_involvement_count += count;
        } else {
            summary.message_involvement_count += count;
        }
        const auto transmitted_index =
            transmitted_bit_index_for_full_variable(variable, profile);
        std::optional<std::size_t> qam_plane;
        if (transmitted_index.has_value()) {
            qam_plane = *transmitted_index % summary.qam_plane_involvement_counts.size();
            summary.qam_plane_involvement_counts[*qam_plane] += count;
        }
        const auto degree = variable_degrees[variable];
        const auto satisfied = degree >= count ? degree - count : std::size_t{0};
        summary.top_variables.push_back(
            FailedVariableTopRow{
                variable,
                transmitted_index,
                section,
                ldpc_section_bit_index(variable, profile),
                qam_plane,
                static_cast<std::uint8_t>(full_bits[variable] & 0x01U),
                degree,
                satisfied,
                count,
                static_cast<long long>(satisfied) - static_cast<long long>(count)});
    }
    std::sort(
        summary.top_variables.begin(),
        summary.top_variables.end(),
        [](const FailedVariableTopRow& left, const FailedVariableTopRow& right) {
            if (left.unsatisfied_check_count != right.unsatisfied_check_count) {
                return left.unsatisfied_check_count > right.unsatisfied_check_count;
            }
            return left.full_variable_index < right.full_variable_index;
        });
    if (summary.top_variables.size() > top_count) {
        summary.top_variables.resize(top_count);
    }
    return summary;
}

void write_failed_variable_summary_json(
    std::ostream& output,
    const FailedVariableSummary& summary,
    std::size_t fec_rate) {
    output
        << "{\"available\":true"
        << ",\"basis\":\"final_ldpc_decoder_bits_full_appendix_b_syndrome\""
        << ",\"fec_rate_index\":" << fec_rate
        << ",\"clean_check_count\":" << summary.clean_check_count
        << ",\"unsatisfied_clean_check_count\":"
        << summary.unsatisfied_clean_check_count
        << ",\"syndrome_ratio\":"
        << (summary.clean_check_count == 0
                ? 0.0
                : static_cast<double>(summary.unsatisfied_clean_check_count)
                    / static_cast<double>(summary.clean_check_count))
        << ",\"full_hard_one_count\":" << summary.full_hard_one_count
        << ",\"transmitted_hard_one_count\":"
        << summary.transmitted_hard_one_count
        << ",\"qam_plane_involvement_counts\":{";
    for (std::size_t plane = 0;
         plane < summary.qam_plane_involvement_counts.size();
         ++plane) {
        if (plane != 0U) {
            output << ',';
        }
        output << "\"" << plane << "\":"
               << summary.qam_plane_involvement_counts[plane];
    }
    output
        << "},\"section_involvement_counts\":{"
        << "\"erased_parity\":" << summary.erased_parity_involvement_count
        << ",\"transmitted_parity\":"
        << summary.transmitted_parity_involvement_count
        << ",\"message\":" << summary.message_involvement_count
        << "},\"top_variables\":[";
    for (std::size_t index = 0; index < summary.top_variables.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        const auto& row = summary.top_variables[index];
        output
            << "{\"full_variable_index\":" << row.full_variable_index
            << ",\"transmitted_bit_index\":";
        write_nullable_size(output, row.transmitted_bit_index);
        output
            << ",\"section\":\"" << row.section << "\""
            << ",\"section_bit_index\":" << row.section_bit_index
            << ",\"qam_plane\":";
        write_nullable_size(output, row.qam_plane);
        output
            << ",\"decoded_bit\":" << static_cast<unsigned>(row.decoded_bit)
            << ",\"check_degree\":" << row.check_degree
            << ",\"satisfied_check_count\":" << row.satisfied_check_count
            << ",\"would_reduce_syndrome_if_flipped\":"
            << (row.flip_syndrome_delta < 0 ? "true" : "false")
            << ",\"unsatisfied_check_count\":"
            << row.unsatisfied_check_count
            << ",\"flip_syndrome_delta\":" << row.flip_syndrome_delta
            << '}';
    }
    output << "]}";
}

void erase_final_failed_variables(
    std::span<float> llr,
    std::span<const std::uint8_t> decoded_bits,
    std::size_t erase_count,
    const dtmb::core::LdpcSparseGraph& graph,
    const DtmbLdpcProfile& profile) {
    if (erase_count == 0U) {
        return;
    }
    if (llr.size() != graph.variable_count || decoded_bits.size() != graph.variable_count) {
        throw std::runtime_error("LDPC final-failed retry spans do not match graph variable count");
    }
    const auto summary = summarize_failed_variables(decoded_bits, graph, profile, erase_count);
    for (const auto& row : summary.top_variables) {
        if (row.full_variable_index < llr.size()) {
            llr[row.full_variable_index] = 0.0F;
        }
    }
}

struct TsContinuityContext {
    std::unordered_map<std::uint16_t, std::uint8_t> last_payload_continuity_counter;
    std::unordered_set<std::uint16_t> seen_payload_pids;
};

struct BitValueConstraint {
    std::size_t variable = 0;
    std::uint8_t value = 0;
};

constexpr std::size_t kBchMessageBits = 752U;
constexpr std::size_t kBchCodeBits = 762U;
constexpr std::array<std::uint8_t, 15> kDtmbScramblerInitialState{
    1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
};

void advance_dtmb_scrambler(std::array<std::uint8_t, 15>& scrambler) {
    const auto next_scrambler_bit =
        static_cast<std::uint8_t>(scrambler[13] ^ scrambler[14]);
    for (std::size_t index = scrambler.size() - 1; index > 0; --index) {
        scrambler[index] = scrambler[index - 1];
    }
    scrambler[0] = next_scrambler_bit;
}

[[nodiscard]] std::vector<std::uint8_t> dtmb_scrambler_bits(
    std::size_t skip_bits,
    std::size_t bit_count) {
    auto scrambler = kDtmbScramblerInitialState;
    for (std::size_t bit = 0; bit < skip_bits; ++bit) {
        advance_dtmb_scrambler(scrambler);
    }
    std::vector<std::uint8_t> bits(bit_count, 0U);
    for (std::size_t bit = 0; bit < bit_count; ++bit) {
        bits[bit] = static_cast<std::uint8_t>(scrambler[13] ^ scrambler[14]);
        advance_dtmb_scrambler(scrambler);
    }
    return bits;
}

[[nodiscard]] bool ts_packets_fit_continuity_context(
    std::span<const std::uint8_t> bytes,
    const TsContinuityContext& context) {
    if (bytes.empty() || bytes.size() % kTsPacketSize != 0U) {
        return false;
    }

    auto candidate_counters = context.last_payload_continuity_counter;
    auto candidate_seen = context.seen_payload_pids;
    for (std::size_t offset = 0; offset < bytes.size(); offset += kTsPacketSize) {
        const auto packet = bytes.subspan(offset, kTsPacketSize);
        if (!ts_packet_syntax_valid(packet)) {
            return false;
        }
        if (!ts_has_payload(packet)) {
            continue;
        }

        const auto pid = ts_pid(packet);
        if (pid == kTsNullPid) {
            continue;
        }
        if (!context.seen_payload_pids.empty() && !candidate_seen.contains(pid)) {
            return false;
        }

        const auto continuity_counter = ts_continuity_counter(packet);
        const auto previous = candidate_counters.find(pid);
        if (previous != candidate_counters.end()
            && !ts_discontinuity_indicator(packet)) {
            const auto expected = static_cast<std::uint8_t>(
                (previous->second + 1U) & 0x0FU);
            if (continuity_counter != expected) {
                return false;
            }
        }
        candidate_seen.insert(pid);
        candidate_counters[pid] = continuity_counter;
    }
    return true;
}

void note_ts_packets_for_continuity_context(
    std::span<const std::uint8_t> bytes,
    TsContinuityContext& context) {
    if (bytes.empty() || bytes.size() % kTsPacketSize != 0U) {
        return;
    }
    for (std::size_t offset = 0; offset < bytes.size(); offset += kTsPacketSize) {
        const auto packet = bytes.subspan(offset, kTsPacketSize);
        if (!ts_packet_syntax_valid(packet) || !ts_has_payload(packet)) {
            continue;
        }
        const auto pid = ts_pid(packet);
        if (pid == kTsNullPid) {
            continue;
        }
        context.seen_payload_pids.insert(pid);
        context.last_payload_continuity_counter[pid] =
            ts_continuity_counter(packet);
    }
}

[[nodiscard]] std::vector<BitValueConstraint> ts_continuity_header_constraints(
    std::span<const std::uint8_t> bytes,
    const TsContinuityContext& context,
    std::size_t codeword,
    std::size_t codeword_transport_bytes,
    const DtmbLdpcProfile& profile) {
    std::vector<BitValueConstraint> constraints;
    if (bytes.empty() || bytes.size() % kTsPacketSize != 0U) {
        return constraints;
    }

    auto candidate_counters = context.last_payload_continuity_counter;
    auto candidate_seen = context.seen_payload_pids;
    const auto scrambler_skip_bits = codeword * codeword_transport_bytes * 8U;
    const auto scrambler = dtmb_scrambler_bits(scrambler_skip_bits, bytes.size() * 8U);
    const auto add_byte_constraints = [&](
        std::size_t byte_offset,
        std::uint8_t value) {
        for (std::size_t bit = 0; bit < 8U; ++bit) {
            const auto transport_bit = byte_offset * 8U + bit;
            const auto bch_block = transport_bit / kBchMessageBits;
            const auto bit_in_block = transport_bit % kBchMessageBits;
            const auto variable = profile.full_parity_bits()
                + bch_block * kBchCodeBits
                + bit_in_block;
            const auto transport_value = static_cast<std::uint8_t>(
                (value >> (7U - bit)) & 0x01U);
            constraints.push_back(
                BitValueConstraint{
                    variable,
                    static_cast<std::uint8_t>(
                        transport_value ^ scrambler[transport_bit])});
        }
    };

    for (std::size_t offset = 0; offset + kTsPacketSize <= bytes.size();
         offset += kTsPacketSize) {
        const auto packet = bytes.subspan(offset, kTsPacketSize);
        if (!ts_packet_syntax_valid(packet)) {
            continue;
        }
        std::array<std::uint8_t, 4> header{
            packet[0],
            packet[1],
            packet[2],
            packet[3],
        };
        if (ts_has_payload(packet)) {
            const auto pid = ts_pid(packet);
            if (pid != kTsNullPid
                && (context.seen_payload_pids.empty() || candidate_seen.contains(pid))) {
                const auto previous = candidate_counters.find(pid);
                if (previous != candidate_counters.end()
                    && !ts_discontinuity_indicator(packet)) {
                    header[3] = static_cast<std::uint8_t>(
                        (header[3] & 0xF0U)
                        | ((previous->second + 1U) & 0x0FU));
                }
                candidate_seen.insert(pid);
                candidate_counters[pid] = static_cast<std::uint8_t>(header[3] & 0x0FU);
            }
        }
        for (std::size_t header_byte = 0; header_byte < header.size(); ++header_byte) {
            add_byte_constraints(offset + header_byte, header[header_byte]);
        }
    }
    return constraints;
}

[[nodiscard]] std::size_t byte_hamming_distance(
    std::uint8_t left,
    std::uint8_t right) {
    auto value = static_cast<unsigned>(left ^ right);
    std::size_t count = 0;
    while (value != 0U) {
        count += static_cast<std::size_t>(value & 0x01U);
        value >>= 1U;
    }
    return count;
}

[[nodiscard]] std::size_t header_hamming_distance(
    const std::array<std::uint8_t, 4>& left,
    const std::array<std::uint8_t, 4>& right) {
    std::size_t distance = 0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        distance += byte_hamming_distance(left[index], right[index]);
    }
    return distance;
}

[[nodiscard]] std::vector<BitValueConstraint>
ts_continuity_header_template_constraints(
    std::span<const std::uint8_t> bytes,
    const TsContinuityContext& context,
    std::size_t codeword,
    std::size_t codeword_transport_bytes,
    const DtmbLdpcProfile& profile) {
    std::vector<BitValueConstraint> constraints;
    if (bytes.empty() || bytes.size() % kTsPacketSize != 0U
        || context.seen_payload_pids.empty()) {
        return constraints;
    }

    std::vector<std::uint16_t> seen_pids(
        context.seen_payload_pids.begin(),
        context.seen_payload_pids.end());
    std::sort(seen_pids.begin(), seen_pids.end());

    auto candidate_counters = context.last_payload_continuity_counter;
    auto candidate_seen = context.seen_payload_pids;
    const auto scrambler_skip_bits = codeword * codeword_transport_bytes * 8U;
    const auto scrambler = dtmb_scrambler_bits(scrambler_skip_bits, bytes.size() * 8U);
    const auto add_byte_constraints = [&](
        std::size_t byte_offset,
        std::uint8_t value) {
        for (std::size_t bit = 0; bit < 8U; ++bit) {
            const auto transport_bit = byte_offset * 8U + bit;
            const auto bch_block = transport_bit / kBchMessageBits;
            const auto bit_in_block = transport_bit % kBchMessageBits;
            const auto variable = profile.full_parity_bits()
                + bch_block * kBchCodeBits
                + bit_in_block;
            const auto transport_value = static_cast<std::uint8_t>(
                (value >> (7U - bit)) & 0x01U);
            constraints.push_back(
                BitValueConstraint{
                    variable,
                    static_cast<std::uint8_t>(
                        transport_value ^ scrambler[transport_bit])});
        }
    };

    for (std::size_t offset = 0; offset + kTsPacketSize <= bytes.size();
         offset += kTsPacketSize) {
        const auto packet = bytes.subspan(offset, kTsPacketSize);
        if (!ts_packet_syntax_valid(packet)) {
            return {};
        }

        std::array<std::uint8_t, 4> original{
            packet[0],
            packet[1],
            packet[2],
            packet[3],
        };
        auto header = original;
        header[0] = kTsSyncByte;
        if (ts_has_payload(packet) && ts_pid(packet) != kTsNullPid) {
            std::optional<std::array<std::uint8_t, 4>> best_header;
            std::uint16_t best_pid = 0U;
            std::uint8_t best_counter = 0U;
            std::size_t best_score = std::numeric_limits<std::size_t>::max();
            for (const auto pid : seen_pids) {
                if (!candidate_seen.contains(pid)) {
                    continue;
                }
                const auto previous = candidate_counters.find(pid);
                const auto counter = previous != candidate_counters.end()
                    ? static_cast<std::uint8_t>((previous->second + 1U) & 0x0FU)
                    : static_cast<std::uint8_t>(original[3] & 0x0FU);
                for (std::uint8_t pusi = 0; pusi <= 1U; ++pusi) {
                    auto candidate = original;
                    candidate[0] = kTsSyncByte;
                    candidate[1] = static_cast<std::uint8_t>(
                        (original[1] & 0x20U)
                        | (pusi << 6U)
                        | ((pid >> 8U) & 0x1FU));
                    candidate[2] = static_cast<std::uint8_t>(pid & 0xFFU);
                    candidate[3] = static_cast<std::uint8_t>(
                        (original[3] & 0xF0U) | counter);
                    const auto pusi_penalty =
                        pusi == static_cast<std::uint8_t>((original[1] >> 6U) & 0x01U)
                            ? std::size_t{0}
                            : std::size_t{1};
                    const auto score =
                        header_hamming_distance(original, candidate) + pusi_penalty;
                    if (score < best_score
                        || (score == best_score && pid < best_pid)) {
                        best_header = candidate;
                        best_pid = pid;
                        best_counter = counter;
                        best_score = score;
                    }
                }
            }
            if (!best_header.has_value()) {
                return {};
            }
            header = *best_header;
            candidate_seen.insert(best_pid);
            candidate_counters[best_pid] = best_counter;
        }

        for (std::size_t header_byte = 0; header_byte < header.size(); ++header_byte) {
            add_byte_constraints(offset + header_byte, header[header_byte]);
        }
    }
    return constraints;
}

[[nodiscard]] std::vector<BitValueConstraint> ts_explicit_header_template_constraints(
    std::size_t fec_frame,
    std::size_t codeword_slot,
    std::span<const TsHeaderTemplateRule> rules,
    std::size_t codeword,
    std::size_t codeword_transport_bytes,
    const DtmbLdpcProfile& profile) {
    std::vector<BitValueConstraint> constraints;
    if (rules.empty()) {
        return constraints;
    }

    const auto scrambler_skip_bits = codeword * codeword_transport_bytes * 8U;
    const auto scrambler =
        dtmb_scrambler_bits(scrambler_skip_bits, codeword_transport_bytes * 8U);
    const auto add_byte_constraints = [&](
        std::size_t byte_offset,
        std::uint8_t value) {
        for (std::size_t bit = 0; bit < 8U; ++bit) {
            const auto transport_bit = byte_offset * 8U + bit;
            const auto bch_block = transport_bit / kBchMessageBits;
            const auto bit_in_block = transport_bit % kBchMessageBits;
            const auto variable = profile.full_parity_bits()
                + bch_block * kBchCodeBits
                + bit_in_block;
            const auto transport_value = static_cast<std::uint8_t>(
                (value >> (7U - bit)) & 0x01U);
            constraints.push_back(
                BitValueConstraint{
                    variable,
                    static_cast<std::uint8_t>(
                        transport_value ^ scrambler[transport_bit])});
        }
    };

    for (const auto& rule : rules) {
        if (rule.fec_frame != fec_frame || rule.codeword != codeword_slot) {
            continue;
        }
        const auto byte_offset = rule.packet * kTsPacketSize;
        if (byte_offset + 4U > codeword_transport_bytes) {
            return {};
        }
        const std::array<std::uint8_t, 4> header{
            kTsSyncByte,
            static_cast<std::uint8_t>(
                (rule.payload_unit_start << 6U) | ((rule.pid >> 8U) & 0x1FU)),
            static_cast<std::uint8_t>(rule.pid & 0xFFU),
            static_cast<std::uint8_t>(
                (rule.adaptation_control << 4U) | rule.continuity_counter),
        };
        for (std::size_t header_byte = 0; header_byte < header.size(); ++header_byte) {
            add_byte_constraints(byte_offset + header_byte, header[header_byte]);
        }
    }
    return constraints;
}

void flip_final_failed_variables(
    std::span<float> llr,
    std::span<const std::uint8_t> decoded_bits,
    std::size_t flip_count,
    const dtmb::core::LdpcSparseGraph& graph,
    const DtmbLdpcProfile& profile) {
    if (flip_count == 0U) {
        return;
    }
    if (llr.size() != graph.variable_count || decoded_bits.size() != graph.variable_count) {
        throw std::runtime_error(
            "LDPC final-failed flip retry spans do not match graph variable count");
    }
    const auto summary = summarize_failed_variables(
        decoded_bits,
        graph,
        profile,
        graph.variable_count);
    std::size_t flipped = 0;
    for (const auto& row : summary.top_variables) {
        if (row.flip_syndrome_delta >= 0) {
            continue;
        }
        if (row.full_variable_index < llr.size()) {
            llr[row.full_variable_index] = -llr[row.full_variable_index];
            ++flipped;
        }
        if (flipped >= flip_count) {
            break;
        }
    }
}

struct GreedyFinalFailedFlipResult {
    std::size_t requested_flips = 0;
    std::size_t applied_flips = 0;
    std::size_t final_syndrome_weight = 0;
};

GreedyFinalFailedFlipResult greedy_flip_final_failed_variables(
    std::span<std::uint8_t> decoded_bits,
    std::size_t max_flips,
    const dtmb::core::LdpcSparseGraph& graph,
    const DtmbLdpcProfile& profile) {
    GreedyFinalFailedFlipResult result;
    result.requested_flips = max_flips;
    result.final_syndrome_weight = dtmb::core::ldpc_syndrome_weight(decoded_bits, graph);
    if (max_flips == 0U || result.final_syndrome_weight == 0U) {
        return result;
    }
    if (decoded_bits.size() != graph.variable_count) {
        throw std::runtime_error(
            "LDPC final-failed greedy flip retry span does not match graph variable count");
    }
    for (; result.applied_flips < max_flips; ++result.applied_flips) {
        const auto summary = summarize_failed_variables(
            decoded_bits,
            graph,
            profile,
            graph.variable_count);
        result.final_syndrome_weight = summary.unsatisfied_clean_check_count;
        if (result.final_syndrome_weight == 0U) {
            break;
        }
        auto chosen = summary.top_variables.end();
        for (auto it = summary.top_variables.begin();
             it != summary.top_variables.end();
             ++it) {
            if (it->flip_syndrome_delta < 0) {
                chosen = it;
                break;
            }
        }
        if (chosen == summary.top_variables.end()
            || chosen->full_variable_index >= decoded_bits.size()) {
            break;
        }
        decoded_bits[chosen->full_variable_index] ^=
            static_cast<std::uint8_t>(0x01U);
    }
    result.final_syndrome_weight = dtmb::core::ldpc_syndrome_weight(
        decoded_bits,
        graph);
    return result;
}

struct SyndromeSolveCandidate {
    std::size_t variable = 0;
    std::size_t unsatisfied_check_count = 0;
    long long flip_syndrome_delta = 0;
    float abs_llr = 0.0F;
};

struct FinalFailedSyndromeSolveResult {
    std::size_t requested_candidates = 0;
    std::size_t considered_candidates = 0;
    std::size_t applied_flips = 0;
    std::size_t checked_solutions = 0;
    std::size_t final_syndrome_weight = 0;
    bool solved = false;
};

constexpr std::size_t kBitsPerWord = 64U;

[[nodiscard]] std::size_t bit_word_count(std::size_t bits) {
    return (bits + kBitsPerWord - 1U) / kBitsPerWord;
}

void set_bit(std::span<std::uint64_t> words, std::size_t bit) {
    words[bit / kBitsPerWord] |= std::uint64_t{1}
        << static_cast<unsigned>(bit % kBitsPerWord);
}

[[nodiscard]] bool test_bit(std::span<const std::uint64_t> words, std::size_t bit) {
    return (words[bit / kBitsPerWord]
            & (std::uint64_t{1} << static_cast<unsigned>(bit % kBitsPerWord)))
        != 0U;
}

void xor_bits(std::span<std::uint64_t> left, std::span<const std::uint64_t> right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("bit-vector XOR size mismatch");
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        left[index] ^= right[index];
    }
}

[[nodiscard]] std::optional<std::size_t> first_set_bit(std::span<const std::uint64_t> words) {
    for (std::size_t word_index = 0; word_index < words.size(); ++word_index) {
        const auto word = words[word_index];
        if (word == 0U) {
            continue;
        }
#if defined(__GNUC__) || defined(__clang__)
        return word_index * kBitsPerWord
            + static_cast<std::size_t>(__builtin_ctzll(word));
#else
        for (std::size_t bit = 0; bit < kBitsPerWord; ++bit) {
            if ((word & (std::uint64_t{1} << bit)) != 0U) {
                return word_index * kBitsPerWord + bit;
            }
        }
#endif
    }
    return std::nullopt;
}

FinalFailedSyndromeSolveResult solve_final_failed_syndrome(
    std::span<std::uint8_t> decoded_bits,
    std::span<const float> llr,
    std::size_t candidate_count,
    std::size_t minimize_cost_passes,
    std::size_t nullspace_search_count,
    std::size_t nullspace_search_depth,
    std::span<const BitValueConstraint> bit_constraints,
    const std::function<bool(std::span<const std::uint8_t>)>& accept_solution,
    const dtmb::core::LdpcSparseGraph& graph,
    const DtmbLdpcProfile& profile) {
    FinalFailedSyndromeSolveResult result;
    result.requested_candidates = candidate_count;
    result.final_syndrome_weight = dtmb::core::ldpc_syndrome_weight(decoded_bits, graph);
    if (candidate_count == 0U) {
        result.solved = result.final_syndrome_weight == 0U
            && (!accept_solution || accept_solution(decoded_bits));
        return result;
    }
    if (decoded_bits.size() != graph.variable_count || llr.size() != graph.variable_count) {
        throw std::runtime_error(
            "LDPC final-failed syndrome solve spans do not match graph variable count");
    }
    if (result.final_syndrome_weight == 0U
        && (!accept_solution || accept_solution(decoded_bits))) {
        result.solved = true;
        return result;
    }

    std::vector<int> forced_value_by_variable(graph.variable_count, -1);
    std::vector<BitValueConstraint> unique_constraints;
    unique_constraints.reserve(bit_constraints.size());
    for (const auto& constraint : bit_constraints) {
        if (constraint.variable >= graph.variable_count || constraint.value > 1U) {
            return result;
        }
        auto& forced = forced_value_by_variable[constraint.variable];
        if (forced >= 0 && forced != static_cast<int>(constraint.value)) {
            return result;
        }
        if (forced < 0) {
            forced = static_cast<int>(constraint.value);
            unique_constraints.push_back(constraint);
        }
    }

    const auto row_count = graph.check_count() + unique_constraints.size();
    const auto row_words = bit_word_count(row_count);
    std::vector<std::uint64_t> syndrome(row_words, 0U);
    std::vector<std::size_t> unsatisfied_counts(graph.variable_count, 0U);
    std::vector<std::size_t> variable_degrees(graph.variable_count, 0U);
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        std::uint8_t parity = 0U;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            const auto variable = graph.edge_variables[edge];
            ++variable_degrees[variable];
            parity ^= static_cast<std::uint8_t>(decoded_bits[variable] & 0x01U);
        }
        if (parity == 0U) {
            continue;
        }
        set_bit(std::span<std::uint64_t>(syndrome), check);
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            ++unsatisfied_counts[graph.edge_variables[edge]];
        }
    }

    std::vector<SyndromeSolveCandidate> candidates;
    candidates.reserve(graph.variable_count);
    for (std::size_t variable = 0; variable < graph.variable_count; ++variable) {
        const auto unsatisfied = unsatisfied_counts[variable];
        if (unsatisfied == 0U && variable >= profile.erased_parity_bits) {
            continue;
        }
        const auto satisfied = variable_degrees[variable] >= unsatisfied
            ? variable_degrees[variable] - unsatisfied
            : std::size_t{0};
        candidates.push_back(
            SyndromeSolveCandidate{
                variable,
                unsatisfied,
                static_cast<long long>(satisfied) - static_cast<long long>(unsatisfied),
                std::abs(llr[variable])});
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const SyndromeSolveCandidate& left, const SyndromeSolveCandidate& right) {
            const auto left_score =
                static_cast<double>(left.abs_llr)
                / static_cast<double>(left.unsatisfied_check_count + 1U);
            const auto right_score =
                static_cast<double>(right.abs_llr)
                / static_cast<double>(right.unsatisfied_check_count + 1U);
            if (left_score != right_score) {
                return left_score < right_score;
            }
            if (left.flip_syndrome_delta != right.flip_syndrome_delta) {
                return left.flip_syndrome_delta < right.flip_syndrome_delta;
            }
            if (left.unsatisfied_check_count != right.unsatisfied_check_count) {
                return left.unsatisfied_check_count > right.unsatisfied_check_count;
            }
            return left.variable < right.variable;
        });
    if (candidates.size() > candidate_count) {
        candidates.resize(candidate_count);
    }
    std::vector<std::uint8_t> candidate_present(graph.variable_count, 0U);
    for (const auto& candidate : candidates) {
        candidate_present[candidate.variable] = 1U;
    }
    for (const auto& constraint : unique_constraints) {
        if (candidate_present[constraint.variable] != 0U) {
            continue;
        }
        const auto variable = constraint.variable;
        const auto unsatisfied = unsatisfied_counts[variable];
        const auto satisfied = variable_degrees[variable] >= unsatisfied
            ? variable_degrees[variable] - unsatisfied
            : std::size_t{0};
        candidates.push_back(
            SyndromeSolveCandidate{
                variable,
                unsatisfied,
                static_cast<long long>(satisfied) - static_cast<long long>(unsatisfied),
                std::abs(llr[variable])});
        candidate_present[variable] = 1U;
    }
    result.considered_candidates = candidates.size();
    if (candidates.empty()) {
        return result;
    }

    std::vector<std::size_t> candidate_by_variable(
        graph.variable_count,
        std::numeric_limits<std::size_t>::max());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        candidate_by_variable[candidates[index].variable] = index;
    }

    std::vector<std::vector<std::uint64_t>> columns(
        candidates.size(),
        std::vector<std::uint64_t>(row_words, 0U));
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            const auto candidate = candidate_by_variable[graph.edge_variables[edge]];
            if (candidate == std::numeric_limits<std::size_t>::max()) {
                continue;
            }
            set_bit(std::span<std::uint64_t>(columns[candidate]), check);
        }
    }
    for (std::size_t constraint_index = 0;
         constraint_index < unique_constraints.size();
         ++constraint_index) {
        const auto& constraint = unique_constraints[constraint_index];
        const auto candidate = candidate_by_variable[constraint.variable];
        if (candidate == std::numeric_limits<std::size_t>::max()) {
            continue;
        }
        const auto row = graph.check_count() + constraint_index;
        set_bit(std::span<std::uint64_t>(columns[candidate]), row);
        const auto target_flip = static_cast<std::uint8_t>(
            (decoded_bits[constraint.variable] & 0x01U) ^ constraint.value);
        if (target_flip != 0U) {
            set_bit(std::span<std::uint64_t>(syndrome), row);
        }
    }

    const auto combo_words = bit_word_count(candidates.size());
    std::vector<std::vector<std::uint64_t>> basis_vectors(row_count);
    std::vector<std::vector<std::uint64_t>> basis_combos(row_count);
    std::vector<std::vector<std::uint64_t>> nullspace_combos;
    for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
        auto vector = columns[candidate];
        std::vector<std::uint64_t> combo(combo_words, 0U);
        set_bit(std::span<std::uint64_t>(combo), candidate);
        while (true) {
            const auto pivot = first_set_bit(std::span<const std::uint64_t>(vector));
            if (!pivot.has_value() || *pivot >= row_count) {
                nullspace_combos.push_back(std::move(combo));
                break;
            }
            if (basis_vectors[*pivot].empty()) {
                basis_vectors[*pivot] = std::move(vector);
                basis_combos[*pivot] = std::move(combo);
                break;
            }
            xor_bits(
                std::span<std::uint64_t>(vector),
                std::span<const std::uint64_t>(basis_vectors[*pivot]));
            xor_bits(
                std::span<std::uint64_t>(combo),
                std::span<const std::uint64_t>(basis_combos[*pivot]));
        }
    }

    const auto original_bits = std::vector<std::uint8_t>(
        decoded_bits.begin(),
        decoded_bits.end());

    auto target = syndrome;
    std::vector<std::uint64_t> solution(combo_words, 0U);
    while (true) {
        const auto pivot = first_set_bit(std::span<const std::uint64_t>(target));
        if (!pivot.has_value() || *pivot >= row_count) {
            break;
        }
        if (basis_vectors[*pivot].empty()) {
            return result;
        }
        xor_bits(
            std::span<std::uint64_t>(target),
            std::span<const std::uint64_t>(basis_vectors[*pivot]));
        xor_bits(
            std::span<std::uint64_t>(solution),
            std::span<const std::uint64_t>(basis_combos[*pivot]));
    }

    for (std::size_t pass = 0;
         pass < minimize_cost_passes && !nullspace_combos.empty();
         ++pass) {
        auto changed = false;
        for (const auto& combo : nullspace_combos) {
            double delta = 0.0;
            for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
                if (!test_bit(std::span<const std::uint64_t>(combo), candidate)) {
                    continue;
                }
                const auto cost = static_cast<double>(candidates[candidate].abs_llr);
                delta += test_bit(std::span<const std::uint64_t>(solution), candidate)
                    ? -cost
                    : cost;
            }
            if (delta >= -1.0e-6) {
                continue;
            }
            xor_bits(
                std::span<std::uint64_t>(solution),
                std::span<const std::uint64_t>(combo));
            changed = true;
        }
        if (!changed) {
            break;
        }
    }

    const auto solution_cost_delta = [&](
        std::span<const std::uint64_t> combo) {
        double delta = 0.0;
        for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
            if (!test_bit(combo, candidate)) {
                continue;
            }
            const auto cost = static_cast<double>(candidates[candidate].abs_llr);
            delta += test_bit(std::span<const std::uint64_t>(solution), candidate)
                ? -cost
                : cost;
        }
        return delta;
    };
    const auto apply_solution = [&](
        std::span<const std::uint64_t> combo,
        std::vector<std::uint8_t>& candidate_bits) {
        candidate_bits = original_bits;
        std::size_t applied_flips = 0;
        for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
            if (!test_bit(combo, candidate)) {
                continue;
            }
            candidate_bits[candidates[candidate].variable] ^=
                static_cast<std::uint8_t>(0x01U);
            ++applied_flips;
        }
        return applied_flips;
    };
    std::vector<std::uint8_t> candidate_bits;
    const auto try_solution = [&](
        std::span<const std::uint64_t> combo) {
        ++result.checked_solutions;
        const auto applied_flips = apply_solution(combo, candidate_bits);
        if (accept_solution && !accept_solution(candidate_bits)) {
            return false;
        }
        std::copy(candidate_bits.begin(), candidate_bits.end(), decoded_bits.begin());
        result.applied_flips = applied_flips;
        result.final_syndrome_weight =
            dtmb::core::ldpc_syndrome_weight(decoded_bits, graph);
        result.solved = result.final_syndrome_weight == 0U;
        return result.solved;
    };

    if (try_solution(std::span<const std::uint64_t>(solution))) {
        return result;
    }
    if (!accept_solution || nullspace_search_count <= result.checked_solutions
        || nullspace_combos.empty()) {
        std::copy(original_bits.begin(), original_bits.end(), decoded_bits.begin());
        result.final_syndrome_weight =
            dtmb::core::ldpc_syndrome_weight(decoded_bits, graph);
        result.solved = false;
        result.applied_flips = 0;
        return result;
    }

    std::vector<double> nullspace_cost_delta(nullspace_combos.size());
    for (std::size_t index = 0; index < nullspace_combos.size(); ++index) {
        nullspace_cost_delta[index] = solution_cost_delta(
            std::span<const std::uint64_t>(nullspace_combos[index]));
    }
    std::vector<std::size_t> ordered_nullspace(nullspace_combos.size());
    std::iota(ordered_nullspace.begin(), ordered_nullspace.end(), 0U);
    std::sort(
        ordered_nullspace.begin(),
        ordered_nullspace.end(),
        [&](std::size_t left, std::size_t right) {
            const auto left_delta = nullspace_cost_delta[left];
            const auto right_delta = nullspace_cost_delta[right];
            if (left_delta != right_delta) {
                return left_delta < right_delta;
            }
            return left < right;
        });

    std::vector<std::uint64_t> alternate(combo_words, 0U);
    for (const auto index : ordered_nullspace) {
        alternate = solution;
        xor_bits(
            std::span<std::uint64_t>(alternate),
            std::span<const std::uint64_t>(nullspace_combos[index]));
        if (try_solution(std::span<const std::uint64_t>(alternate))) {
            return result;
        }
        if (result.checked_solutions >= nullspace_search_count) {
            break;
        }
    }
    for (std::size_t left_order = 0;
         nullspace_search_depth >= 2U
         && left_order < ordered_nullspace.size()
         && result.checked_solutions < nullspace_search_count;
         ++left_order) {
        for (std::size_t right_order = left_order + 1U;
             right_order < ordered_nullspace.size()
             && result.checked_solutions < nullspace_search_count;
             ++right_order) {
            alternate = solution;
            xor_bits(
                std::span<std::uint64_t>(alternate),
                std::span<const std::uint64_t>(
                    nullspace_combos[ordered_nullspace[left_order]]));
            xor_bits(
                std::span<std::uint64_t>(alternate),
                std::span<const std::uint64_t>(
                    nullspace_combos[ordered_nullspace[right_order]]));
            if (try_solution(std::span<const std::uint64_t>(alternate))) {
                return result;
            }
        }
    }
    for (std::size_t left_order = 0;
         nullspace_search_depth >= 3U
         && left_order < ordered_nullspace.size()
         && result.checked_solutions < nullspace_search_count;
         ++left_order) {
        for (std::size_t middle_order = left_order + 1U;
             middle_order < ordered_nullspace.size()
             && result.checked_solutions < nullspace_search_count;
             ++middle_order) {
            for (std::size_t right_order = middle_order + 1U;
                 right_order < ordered_nullspace.size()
                 && result.checked_solutions < nullspace_search_count;
                 ++right_order) {
                alternate = solution;
                xor_bits(
                    std::span<std::uint64_t>(alternate),
                    std::span<const std::uint64_t>(
                        nullspace_combos[ordered_nullspace[left_order]]));
                xor_bits(
                    std::span<std::uint64_t>(alternate),
                    std::span<const std::uint64_t>(
                        nullspace_combos[ordered_nullspace[middle_order]]));
                xor_bits(
                    std::span<std::uint64_t>(alternate),
                    std::span<const std::uint64_t>(
                        nullspace_combos[ordered_nullspace[right_order]]));
                if (try_solution(std::span<const std::uint64_t>(alternate))) {
                    return result;
                }
            }
        }
    }

    std::copy(original_bits.begin(), original_bits.end(), decoded_bits.begin());
    result.final_syndrome_weight =
        dtmb::core::ldpc_syndrome_weight(decoded_bits, graph);
    result.solved = false;
    result.applied_flips = 0;
    return result;
}

struct BufferedFecFrame {
    std::vector<float> transmitted_llr;
    std::size_t frame_index = 0;
    std::size_t clean_syndrome_weight = 0;
    bool selected = false;
    bool force_selected = false;
    bool force_omitted = false;
    bool mode2_source_frame_selected = false;
    bool hard_h_gap_filled = false;
};

struct PendingFecFrame {
    std::vector<float> transmitted_llr;
    std::size_t frame_index = 0;
    std::size_t clean_syndrome_weight = 0;
    bool selected = false;
    bool force_selected = false;
    bool force_omitted = false;
    bool mode2_source_frame_selected = false;
    bool hard_h_gap_filled = false;
};

struct DecodedFecFrame {
    std::vector<float> full_llr;
    std::vector<std::uint8_t> baseline_decoded_bits;
    std::vector<std::uint8_t> decoded_bits;
    std::vector<std::uint8_t> message_bits;
    std::vector<std::uint8_t> transport_bytes;
    std::vector<dtmb::core::LdpcDecodeResult> results;
    std::vector<std::size_t> initial_syndrome_weights;
    std::vector<std::uint8_t> early_rejected;
    std::vector<std::uint8_t> clean_codewords;
    std::vector<std::uint8_t> bch_clean_codewords;
    std::vector<std::size_t> retry_attempts;
    std::vector<std::uint8_t> retry_selected;
    std::vector<std::uint8_t> retry_layered;
    std::vector<double> retry_llr_scale;
    std::vector<double> retry_llr_clip;
    std::vector<double> retry_weak_erase_fraction;
    std::vector<double> retry_attenuation;
    std::vector<std::size_t> retry_unsatisfied_erase_count;
    std::vector<std::size_t> retry_final_failed_erase_count;
    std::vector<std::size_t> retry_final_failed_flip_count;
    std::vector<std::size_t> retry_final_failed_greedy_flip_count;
    std::vector<std::size_t> retry_final_failed_syndrome_solve_count;
    std::vector<std::size_t> retry_qam_plane;
    std::vector<std::size_t> retry_variable_range_first;
    std::vector<std::size_t> retry_variable_range_last;
    std::vector<std::size_t> retry_branch_window_first;
    std::vector<std::size_t> retry_branch_window_last;
    std::vector<std::string> retry_branch_window_group;
    std::vector<std::size_t> retry_bch_corrected_errors;
    std::vector<std::uint8_t> retry_ts_continuity_rejected;
    dtmb::core::DtmbBchDecodeStats bch;
    bool frame_clean = false;
};

struct UnsatisfiedVariableCandidate {
    std::size_t variable = 0;
    std::size_t unsatisfied_check_count = 0;
    float abs_llr = 0.0F;
};

void erase_initial_unsatisfied_variables(
    std::span<float> llr,
    std::size_t first_transmitted_variable,
    std::size_t erase_count,
    const dtmb::core::LdpcSparseGraph& graph) {
    if (erase_count == 0U) {
        return;
    }
    if (llr.size() != graph.variable_count) {
        throw std::runtime_error("LDPC retry LLR size does not match graph variable count");
    }

    std::vector<std::uint8_t> hard_bits(llr.size(), 0U);
    std::transform(
        llr.begin(),
        llr.end(),
        hard_bits.begin(),
        [](float value) { return value < 0.0F ? 1U : 0U; });

    std::vector<std::size_t> unsatisfied_counts(llr.size(), 0U);
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        std::uint8_t parity = 0U;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            parity ^=
                static_cast<std::uint8_t>(hard_bits[graph.edge_variables[edge]] & 0x01U);
        }
        if (parity == 0U) {
            continue;
        }
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            ++unsatisfied_counts[graph.edge_variables[edge]];
        }
    }

    std::vector<UnsatisfiedVariableCandidate> candidates;
    candidates.reserve(llr.size() - std::min(first_transmitted_variable, llr.size()));
    for (std::size_t variable = first_transmitted_variable; variable < llr.size(); ++variable) {
        const auto count = unsatisfied_counts[variable];
        const auto abs_llr = std::abs(llr[variable]);
        if (count == 0U || abs_llr == 0.0F) {
            continue;
        }
        candidates.push_back(
            UnsatisfiedVariableCandidate{variable, count, abs_llr});
    }
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const UnsatisfiedVariableCandidate& left,
           const UnsatisfiedVariableCandidate& right) {
            if (left.unsatisfied_check_count != right.unsatisfied_check_count) {
                return left.unsatisfied_check_count > right.unsatisfied_check_count;
            }
            if (left.abs_llr != right.abs_llr) {
                return left.abs_llr > right.abs_llr;
            }
            return left.variable < right.variable;
        });
    const auto selected = std::min(erase_count, candidates.size());
    for (std::size_t index = 0; index < selected; ++index) {
        llr[candidates[index].variable] = 0.0F;
    }
}

void write_final_failed_variable_summaries(
    std::ostream& output,
    const DecodedFecFrame* decoded_frame,
    const dtmb::core::LdpcSparseGraph& graph,
    const DtmbLdpcProfile& profile,
    std::size_t codewords_per_frame,
    std::size_t top_count) {
    output << ",\"codeword_final_failed_variable_summary\":[";
    for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
        if (codeword != 0U) {
            output << ',';
        }
        if (decoded_frame == nullptr) {
            output << "{\"available\":false,\"reason\":\"not_soft_decoded\"}";
            continue;
        }
        if (codeword >= decoded_frame->results.size()) {
            output << "{\"available\":false,\"reason\":\"codeword_result_missing\"}";
            continue;
        }
        if (decoded_frame->results[codeword].converged) {
            output << "{\"available\":false,\"reason\":\"codeword_converged\"}";
            continue;
        }
        const auto full_offset = codeword * profile.full_codeword_bits();
        if (decoded_frame->decoded_bits.size()
            < full_offset + profile.full_codeword_bits()) {
            output << "{\"available\":false,\"reason\":\"decoded_bits_missing\"}";
            continue;
        }
        const auto full_bits = std::span<const std::uint8_t>(
            decoded_frame->decoded_bits.data() + full_offset,
            profile.full_codeword_bits());
        write_failed_variable_summary_json(
            output,
            summarize_failed_variables(full_bits, graph, profile, top_count),
            profile.fec_rate);
    }
    output << ']';
}

void condition_retry_llr(
    std::span<const float> input,
    std::span<float> output,
    std::size_t first_transmitted_variable,
    float llr_scale,
    float clip,
    float erase_fraction,
    std::size_t unsatisfied_erase_count,
    std::optional<VariableRange> variable_range,
    std::span<const BranchWindow> branch_windows,
    std::size_t qam_plane,
    const dtmb::core::LdpcSparseGraph& graph) {
    std::copy(input.begin(), input.end(), output.begin());
    auto transmitted = output.subspan(first_transmitted_variable);
    if (llr_scale != 1.0F) {
        for (auto& llr : transmitted) {
            llr *= llr_scale;
        }
    }
    const auto has_qam_plane = qam_plane < kQam64LlrsPerSymbol;
    if (variable_range.has_value() || !branch_windows.empty() || has_qam_plane) {
        for (std::size_t variable = 0; variable < transmitted.size(); ++variable) {
            const auto symbol = variable / kQam64LlrsPerSymbol;
            const auto branch = symbol % kSymbolInterleaverBranches;
            const auto variable_plane = variable % kQam64LlrsPerSymbol;
            const auto variable_matches =
                !variable_range.has_value()
                || (variable >= variable_range->first && variable <= variable_range->last);
            const auto branch_matches =
                branch_windows.empty()
                || std::any_of(
                    branch_windows.begin(),
                    branch_windows.end(),
                    [branch](const BranchWindow& window) {
                        return branch >= window.first && branch <= window.last;
                    });
            const auto plane_matches = !has_qam_plane || variable_plane == qam_plane;
            if (variable_matches && branch_matches && plane_matches) {
                transmitted[variable] = 0.0F;
            }
        }
    }
    if (clip > 0.0F) {
        for (auto& llr : transmitted) {
            llr = std::clamp(llr, -clip, clip);
        }
    }
    const auto erase_count = static_cast<std::size_t>(
        std::floor(static_cast<double>(transmitted.size()) * erase_fraction));
    if (erase_count != 0U) {
        std::vector<std::size_t> indexes(transmitted.size());
        std::iota(indexes.begin(), indexes.end(), 0U);
        std::nth_element(
            indexes.begin(),
            indexes.begin() + static_cast<std::ptrdiff_t>(erase_count),
            indexes.end(),
            [&](std::size_t left, std::size_t right) {
                return std::abs(transmitted[left]) < std::abs(transmitted[right]);
            });
        for (std::size_t index = 0; index < erase_count; ++index) {
            transmitted[indexes[index]] = 0.0F;
        }
    }
    erase_initial_unsatisfied_variables(
        output,
        first_transmitted_variable,
        unsatisfied_erase_count,
        graph);
}

bool has_transmitted_llr_evidence(
    std::span<const float> llr,
    std::size_t first_transmitted_variable) {
    if (first_transmitted_variable >= llr.size()) {
        return false;
    }
    const auto transmitted = llr.subspan(first_transmitted_variable);
    return std::any_of(
        transmitted.begin(),
        transmitted.end(),
        [](float value) {
            return value != 0.0F;
        });
}

[[nodiscard]] std::vector<float> retry_axis_with_baseline(
    std::span<const float> requested,
    float baseline) {
    std::vector<float> axis;
    axis.reserve(requested.size() + 1U);
    axis.push_back(baseline);
    for (const auto value : requested) {
        const auto duplicate = std::any_of(
            axis.begin(),
            axis.end(),
            [value](float existing) {
                return existing == value;
            });
        if (!duplicate) {
            axis.push_back(value);
        }
    }
    return axis;
}

[[nodiscard]] std::vector<std::size_t> retry_axis_with_baseline(
    std::span<const std::size_t> requested,
    std::size_t baseline) {
    std::vector<std::size_t> axis;
    axis.reserve(requested.size() + 1U);
    axis.push_back(baseline);
    for (const auto value : requested) {
        const auto duplicate = std::any_of(
            axis.begin(),
            axis.end(),
            [value](std::size_t existing) {
                return existing == value;
            });
        if (!duplicate) {
            axis.push_back(value);
        }
    }
    return axis;
}

[[nodiscard]] std::size_t clean_syndrome_weight(
    std::span<const float> transmitted_llr,
    std::size_t transmitted_bits,
    const dtmb::core::LdpcSparseGraph& clean_graph) {
    std::vector<std::uint8_t> hard_bits(transmitted_bits);
    std::size_t weight = 0;
    for (std::size_t offset = 0; offset < transmitted_llr.size(); offset += transmitted_bits) {
        const auto codeword = transmitted_llr.subspan(offset, transmitted_bits);
        std::transform(
            codeword.begin(),
            codeword.end(),
            hard_bits.begin(),
            [](float llr) { return llr < 0.0F ? 1U : 0U; });
        weight += dtmb::core::ldpc_syndrome_weight(hard_bits, clean_graph);
    }
    return weight;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t fec_rate = 0;
    std::size_t codewords_per_frame = 3;
    std::size_t requested_workers = 0;
    std::string alist_path;
    std::string input_path = "-";
    std::string output_path = "-";
    std::string frame_diagnostics_path;
    std::string packet_provenance_path;
    std::vector<std::string> positional;
    dtmb::core::LdpcDecodeOptions decode_options{};
    std::optional<std::size_t> retry_max_iterations;
    std::optional<float> primary_llr_clip;
    auto ldpc_accel = LdpcAccel::cpu;
    std::size_t decode_batch_frames = 16;
    bool clean_frames_only = false;
    bool fail_on_unclean_frame = false;
    bool emit_clean_codewords = false;
    bool emit_bch_clean_codewords = false;
    bool emit_bch_clean_packets = false;
    bool emit_forced_codewords = false;
    bool mark_discontinuities = false;
    bool insert_discontinuity_packets = false;
    std::size_t frame_index_offset = 0;
    float early_syndrome_reject_ratio = -1.0F;
    std::optional<std::size_t> hard_h_gate_window_codewords;
    std::optional<std::size_t> hard_h_gate_step_codewords;
    std::optional<double> hard_h_gate_threshold;
    std::size_t hard_h_gate_fill_max_gap_frames = 0;
    std::vector<FrameRange> force_frame_ranges;
    std::vector<FrameRange> omit_frame_ranges;
    std::vector<CodewordLlrScaleRule> codeword_llr_scale_rules;
    std::vector<VariableLlrScaleRule> variable_llr_scale_rules;
    std::vector<VariableModLlrScaleRule> variable_mod_llr_scale_rules;
    std::vector<float> retry_llr_clips;
    bool cuda_retry_clips = false;
    std::vector<float> retry_llr_scales;
    std::vector<float> retry_weak_erase_fractions;
    std::vector<float> retry_attenuations;
    std::vector<std::size_t> retry_unsatisfied_erase_counts;
    std::vector<std::size_t> retry_final_failed_erase_counts;
    std::vector<std::size_t> retry_final_failed_flip_counts;
    std::vector<std::size_t> retry_final_failed_greedy_flip_counts;
    std::vector<std::size_t> retry_final_failed_syndrome_solve_counts;
    std::size_t retry_final_failed_syndrome_solve_minimize_cost_passes = 0;
    std::size_t retry_final_failed_syndrome_solve_nullspace_search_count = 0;
    std::size_t retry_final_failed_syndrome_solve_nullspace_depth = 2;
    bool retry_final_failed_syndrome_solve_ts_continuity_headers = false;
    bool retry_final_failed_syndrome_solve_ts_continuity_header_templates = false;
    std::vector<TsHeaderTemplateRule> retry_final_failed_syndrome_solve_ts_header_template_rules;
    std::vector<std::size_t> retry_qam_planes;
    std::vector<VariableRange> retry_variable_ranges;
    std::vector<BranchWindow> retry_branch_windows;
    std::vector<std::vector<BranchWindow>> retry_branch_window_groups;
    std::vector<FrameRange> retry_branch_window_frame_ranges;
    std::vector<BranchWindowRule> retry_branch_window_rules;
    std::vector<FrameRange> retry_mode2_source_frame_windows;
    std::vector<SourceFrameWindowRule> retry_mode2_source_frame_window_rules;
    std::vector<FrameRange> select_mode2_source_frame_windows;
    std::vector<SourceFrameWindowRule> select_mode2_source_frame_window_rules;
    bool retry_layered = false;
    bool retry_require_ts_packet_syntax = false;
    bool retry_require_ts_packet_continuity = false;
    std::size_t failed_variable_diagnostics_top = 0;

    try {
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
            } else if (arg == "--codewords-per-frame") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                codewords_per_frame = parse_size(argv[index], "codewords per frame");
            } else if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                requested_workers = parse_size(argv[index], "worker count");
            } else if (arg == "--max-iterations") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                decode_options.max_iterations = parse_size(argv[index], "max iterations");
            } else if (arg == "--retry-max-iterations") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_max_iterations =
                    parse_size(argv[index], "retry max iterations");
            } else if (arg == "--attenuation") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                decode_options.attenuation = parse_float(argv[index], "attenuation");
            } else if (arg == "--llr-clip") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                primary_llr_clip = parse_float(argv[index], "LLR clip");
            } else if (arg == "--ldpc-accel") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                ldpc_accel = parse_ldpc_accel(argv[index]);
            } else if (arg == "--retry-llr-clip") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_llr_clips.push_back(parse_float(argv[index], "retry LLR clip"));
            } else if (arg == "--retry-llr-scale") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_llr_scales.push_back(parse_float(argv[index], "retry LLR scale"));
            } else if (arg == "--retry-weak-erase-fraction") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_weak_erase_fractions.push_back(
                    parse_float(argv[index], "retry weak erase fraction"));
            } else if (arg == "--retry-attenuation") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_attenuations.push_back(
                    parse_float(argv[index], "retry attenuation"));
            } else if (arg == "--retry-unsatisfied-erase-count") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_unsatisfied_erase_counts.push_back(
                    parse_size(argv[index], "retry unsatisfied erase count"));
            } else if (arg == "--retry-final-failed-erase-count") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_erase_counts.push_back(
                    parse_size(argv[index], "retry final failed erase count"));
            } else if (arg == "--retry-final-failed-flip-count") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_flip_counts.push_back(
                    parse_size(argv[index], "retry final failed flip count"));
            } else if (arg == "--retry-final-failed-greedy-flip-count") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_greedy_flip_counts.push_back(
                    parse_size(argv[index], "retry final failed greedy flip count"));
            } else if (arg == "--retry-final-failed-syndrome-solve-count") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_syndrome_solve_counts.push_back(
                    parse_size(argv[index], "retry final failed syndrome solve count"));
            } else if (
                arg == "--retry-final-failed-syndrome-solve-minimize-cost-passes") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_syndrome_solve_minimize_cost_passes =
                    parse_size(
                        argv[index],
                        "retry final failed syndrome solve minimize cost passes");
            } else if (
                arg == "--retry-final-failed-syndrome-solve-nullspace-search") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_syndrome_solve_nullspace_search_count =
                    parse_size(
                        argv[index],
                        "retry final failed syndrome solve nullspace search count");
            } else if (
                arg == "--retry-final-failed-syndrome-solve-nullspace-depth") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_syndrome_solve_nullspace_depth =
                    parse_size(
                        argv[index],
                        "retry final failed syndrome solve nullspace depth");
            } else if (
                arg == "--retry-final-failed-syndrome-solve-ts-continuity-headers") {
                retry_final_failed_syndrome_solve_ts_continuity_headers = true;
            } else if (
                arg == "--retry-final-failed-syndrome-solve-ts-continuity-header-templates") {
                retry_final_failed_syndrome_solve_ts_continuity_header_templates = true;
            } else if (
                arg == "--retry-final-failed-syndrome-solve-ts-header-template") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_final_failed_syndrome_solve_ts_header_template_rules.push_back(
                    parse_ts_header_template_rule(argv[index]));
            } else if (arg == "--retry-qam-plane") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_qam_planes.push_back(parse_size(argv[index], "retry QAM plane"));
            } else if (arg == "--retry-variable-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_variable_ranges.push_back(parse_variable_range(argv[index]));
            } else if (arg == "--retry-branch-window") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_branch_windows.push_back(parse_branch_window(argv[index]));
            } else if (arg == "--retry-branch-window-group") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_branch_window_groups.push_back(parse_branch_window_group(argv[index]));
            } else if (arg == "--retry-branch-window-sweep") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto windows = parse_branch_window_sweep(argv[index]);
                retry_branch_windows.insert(
                    retry_branch_windows.end(),
                    windows.begin(),
                    windows.end());
            } else if (arg == "--retry-branch-window-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_branch_window_frame_ranges.push_back(
                    parse_frame_range(argv[index]));
            } else if (arg == "--retry-branch-window-for-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_branch_window_rules.push_back(
                    parse_branch_window_rule(argv[index]));
            } else if (arg == "--retry-mode2-source-frame-window") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_mode2_source_frame_windows.push_back(
                    parse_source_frame_range(argv[index]));
            } else if (arg == "--retry-mode2-source-frame-window-for-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_mode2_source_frame_window_rules.push_back(
                    parse_source_frame_window_rule(argv[index]));
            } else if (arg == "--select-mode2-source-frame-window") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                select_mode2_source_frame_windows.push_back(
                    parse_source_frame_range(argv[index]));
            } else if (arg == "--select-mode2-source-frame-window-for-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                select_mode2_source_frame_window_rules.push_back(
                    parse_source_frame_window_rule(argv[index]));
            } else if (arg == "--retry-layered") {
                retry_layered = true;
            } else if (arg == "--cuda-retry-clips") {
                cuda_retry_clips = true;
            } else if (arg == "--retry-require-ts-packet-syntax") {
                retry_require_ts_packet_syntax = true;
            } else if (arg == "--retry-require-ts-packet-continuity") {
                retry_require_ts_packet_continuity = true;
                retry_require_ts_packet_syntax = true;
            } else if (arg == "--decode-batch-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                decode_batch_frames = parse_size(argv[index], "decode batch frames");
            } else if (arg == "--early-syndrome-reject-ratio") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                const std::string value = argv[index];
                early_syndrome_reject_ratio =
                    value == "off" || value == "none"
                        ? -1.0F
                        : parse_float(value, "early syndrome reject ratio");
            } else if (arg == "--clean-frames-only") {
                clean_frames_only = true;
            } else if (arg == "--fail-on-unclean-frame") {
                fail_on_unclean_frame = true;
            } else if (arg == "--emit-clean-codewords") {
                emit_clean_codewords = true;
            } else if (arg == "--emit-bch-clean-codewords") {
                emit_bch_clean_codewords = true;
            } else if (arg == "--emit-bch-clean-packets") {
                emit_bch_clean_packets = true;
            } else if (arg == "--emit-forced-codewords") {
                emit_forced_codewords = true;
            } else if (arg == "--mark-discontinuities") {
                mark_discontinuities = true;
            } else if (arg == "--insert-discontinuity-packets") {
                insert_discontinuity_packets = true;
            } else if (arg == "--frame-index-offset") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                frame_index_offset = parse_size(argv[index], "frame index offset");
            } else if (arg == "--hard-h-gate-window-codewords") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                hard_h_gate_window_codewords =
                    parse_size(argv[index], "hard H gate window codewords");
            } else if (arg == "--hard-h-gate-step-codewords") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                hard_h_gate_step_codewords =
                    parse_size(argv[index], "hard H gate step codewords");
            } else if (arg == "--hard-h-gate-threshold") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                hard_h_gate_threshold =
                    parse_double(argv[index], "hard H gate threshold");
            } else if (arg == "--hard-h-gate-fill-max-gap-frames") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                hard_h_gate_fill_max_gap_frames =
                    parse_size(argv[index], "hard H gate fill max gap frames");
            } else if (arg == "--force-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                force_frame_ranges.push_back(parse_frame_range(argv[index]));
            } else if (arg == "--omit-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                omit_frame_ranges.push_back(parse_frame_range(argv[index]));
            } else if (arg == "--llr-scale-codeword") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                codeword_llr_scale_rules.push_back(
                    parse_codeword_llr_scale_rule(argv[index]));
            } else if (arg == "--llr-scale-variable") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                variable_llr_scale_rules.push_back(
                    parse_variable_llr_scale_rule(argv[index]));
            } else if (arg == "--llr-scale-variable-mod") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                variable_mod_llr_scale_rules.push_back(
                    parse_variable_mod_llr_scale_rule(argv[index]));
            } else if (arg == "--frame-diagnostics-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                frame_diagnostics_path = argv[index];
            } else if (arg == "--failed-variable-diagnostics-top") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                failed_variable_diagnostics_top =
                    parse_size(argv[index], "failed variable diagnostics top count");
            } else if (arg == "--packet-provenance-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                packet_provenance_path = argv[index];
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                positional.push_back(arg);
            }
        }
        if (fec_rate == 0 || alist_path.empty() || codewords_per_frame == 0 || positional.size() > 2) {
            usage(argv[0]);
            return 2;
        }
        if (decode_options.max_iterations == 0) {
            throw std::invalid_argument("max iterations must be positive");
        }
        if (retry_max_iterations.has_value() && *retry_max_iterations == 0) {
            throw std::invalid_argument("retry max iterations must be positive");
        }
        if (primary_llr_clip.has_value()
            && (!std::isfinite(*primary_llr_clip) || *primary_llr_clip <= 0.0F)) {
            throw std::invalid_argument("LLR clip must be positive and finite");
        }
        if ((early_syndrome_reject_ratio < 0.0F
             && early_syndrome_reject_ratio != -1.0F)
            || early_syndrome_reject_ratio > 1.0F) {
            usage(argv[0]);
            return 2;
        }
        for (const auto clip : retry_llr_clips) {
            if (!std::isfinite(clip) || clip <= 0.0F) {
                throw std::invalid_argument(
                    "retry LLR clips must be positive and finite");
            }
        }
        for (const auto scale : retry_llr_scales) {
            if (!std::isfinite(scale) || scale < 0.0F) {
                throw std::invalid_argument(
                    "retry LLR scales must be non-negative and finite");
            }
        }
        for (const auto fraction : retry_weak_erase_fractions) {
            if (!std::isfinite(fraction) || fraction <= 0.0F || fraction >= 1.0F) {
                throw std::invalid_argument(
                    "retry weak erase fractions must be finite and within 0..1");
            }
        }
        for (const auto attenuation : retry_attenuations) {
            if (!std::isfinite(attenuation) || attenuation <= 0.0F) {
                throw std::invalid_argument(
                    "retry attenuations must be positive and finite");
            }
        }
        for (const auto count : retry_unsatisfied_erase_counts) {
            if (count == 0U) {
                throw std::invalid_argument(
                    "retry unsatisfied erase counts must be positive");
            }
        }
        for (const auto count : retry_final_failed_erase_counts) {
            if (count == 0U) {
                throw std::invalid_argument(
                    "retry final failed erase counts must be positive");
            }
        }
        for (const auto count : retry_final_failed_flip_counts) {
            if (count == 0U) {
                throw std::invalid_argument(
                    "retry final failed flip counts must be positive");
            }
        }
        for (const auto count : retry_final_failed_greedy_flip_counts) {
            if (count == 0U) {
                throw std::invalid_argument(
                    "retry final failed greedy flip counts must be positive");
            }
        }
        for (const auto count : retry_final_failed_syndrome_solve_counts) {
            if (count == 0U) {
                throw std::invalid_argument(
                    "retry final failed syndrome solve counts must be positive");
            }
        }
        if (retry_final_failed_syndrome_solve_nullspace_depth == 0U
            || retry_final_failed_syndrome_solve_nullspace_depth > 3U) {
            throw std::invalid_argument(
                "retry final failed syndrome solve nullspace depth must be 1..3");
        }
        if (retry_final_failed_syndrome_solve_ts_continuity_headers
            && !retry_require_ts_packet_continuity) {
            throw std::invalid_argument(
                "retry TS continuity header constraints require "
                "--retry-require-ts-packet-continuity");
        }
        if (retry_final_failed_syndrome_solve_ts_continuity_header_templates
            && !retry_require_ts_packet_continuity) {
            throw std::invalid_argument(
                "retry TS continuity header template constraints require "
                "--retry-require-ts-packet-continuity");
        }
        if (retry_final_failed_syndrome_solve_ts_continuity_headers
            && retry_final_failed_syndrome_solve_ts_continuity_header_templates) {
            throw std::invalid_argument(
                "retry TS continuity header constraints and template constraints "
                "are mutually exclusive");
        }
        for (const auto plane : retry_qam_planes) {
            if (plane >= kQam64LlrsPerSymbol) {
                throw std::invalid_argument(
                    "retry QAM planes must be within 0..5");
            }
        }
        for (const auto& range : retry_variable_ranges) {
            if (range.last >= profile_for_rate(fec_rate).transmitted_bits) {
                throw std::invalid_argument(
                    "retry variable ranges must be within transmitted codeword bits");
            }
        }
        for (const auto& window : retry_branch_windows) {
            if (window.last >= kSymbolInterleaverBranches) {
                throw std::invalid_argument(
                    "retry branch windows must be within 0..51");
            }
        }
        for (const auto& group : retry_branch_window_groups) {
            if (group.empty()) {
                throw std::invalid_argument(
                    "retry branch window groups must contain at least one window");
            }
            for (const auto& window : group) {
                if (window.last >= kSymbolInterleaverBranches) {
                    throw std::invalid_argument(
                        "retry branch window group ranges must be within 0..51");
                }
            }
        }
        if (retry_branch_windows.empty()
            && retry_branch_window_groups.empty()
            && !retry_branch_window_frame_ranges.empty()) {
            throw std::invalid_argument(
                "retry branch window frame ranges require --retry-branch-window "
                "or --retry-branch-window-group");
        }
        for (const auto& rule : retry_branch_window_rules) {
            if (rule.branch_window.last >= kSymbolInterleaverBranches) {
                throw std::invalid_argument(
                    "retry branch window rule branch ranges must be within 0..51");
            }
        }
        if (emit_clean_codewords && !clean_frames_only) {
            throw std::invalid_argument(
                "emit-clean-codewords requires --clean-frames-only");
        }
        if (emit_bch_clean_codewords && !emit_clean_codewords) {
            throw std::invalid_argument(
                "emit-bch-clean-codewords requires --emit-clean-codewords");
        }
        if (emit_bch_clean_packets && !emit_clean_codewords) {
            throw std::invalid_argument(
                "emit-bch-clean-packets requires --emit-clean-codewords");
        }
        if (emit_forced_codewords && !emit_clean_codewords) {
            throw std::invalid_argument(
                "emit-forced-codewords requires --emit-clean-codewords");
        }
        if (emit_forced_codewords && force_frame_ranges.empty()) {
            throw std::invalid_argument(
                "emit-forced-codewords requires --force-frame-range");
        }
        const auto hard_h_gate_flag_count =
            static_cast<unsigned>(hard_h_gate_window_codewords.has_value())
            + static_cast<unsigned>(hard_h_gate_step_codewords.has_value())
            + static_cast<unsigned>(hard_h_gate_threshold.has_value());
        const auto hard_h_gate_enabled = hard_h_gate_flag_count == 3U;
        if (hard_h_gate_flag_count != 0U && !hard_h_gate_enabled) {
            throw std::invalid_argument("hard H gate options must be supplied together");
        }
        if (hard_h_gate_fill_max_gap_frames != 0 && !hard_h_gate_enabled) {
            throw std::invalid_argument(
                "hard H gate gap fill requires hard H gate options");
        }
        if (hard_h_gate_enabled) {
            if (*hard_h_gate_window_codewords == 0
                || (*hard_h_gate_window_codewords % codewords_per_frame) != 0
                || *hard_h_gate_step_codewords == 0
                || (*hard_h_gate_step_codewords % codewords_per_frame) != 0) {
                throw std::invalid_argument(
                    "hard H gate window and step must be positive whole FEC frames");
            }
            if (*hard_h_gate_step_codewords != codewords_per_frame) {
                throw std::invalid_argument(
                    "hard H gate v1 requires step codewords equal codewords per frame");
            }
            if (!std::isfinite(*hard_h_gate_threshold)
                || *hard_h_gate_threshold < 0.0
                || *hard_h_gate_threshold > 1.0) {
                throw std::invalid_argument(
                    "hard H gate threshold must be finite and within 0..1");
            }
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }
        if (positional.size() == 2) {
            output_path = positional[1];
        }
        for (const auto& rule : codeword_llr_scale_rules) {
            if (rule.codeword >= codewords_per_frame) {
                throw std::invalid_argument(
                    "codeword LLR scale slot is outside codewords-per-frame");
            }
        }

        dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
        const auto profile = profile_for_rate(fec_rate);
        const auto retry_enabled =
            !retry_llr_clips.empty()
            || !retry_llr_scales.empty()
            || !retry_weak_erase_fractions.empty()
            || !retry_attenuations.empty()
            || !retry_unsatisfied_erase_counts.empty()
            || !retry_final_failed_erase_counts.empty()
            || !retry_final_failed_flip_counts.empty()
            || !retry_final_failed_greedy_flip_counts.empty()
            || !retry_final_failed_syndrome_solve_counts.empty()
            || !retry_final_failed_syndrome_solve_ts_header_template_rules.empty()
            || !retry_qam_planes.empty()
            || !retry_variable_ranges.empty()
            || !retry_branch_windows.empty()
            || !retry_branch_window_groups.empty()
            || !retry_branch_window_rules.empty()
            || !retry_mode2_source_frame_windows.empty()
            || !retry_mode2_source_frame_window_rules.empty()
            || retry_layered;
        const auto retry_clip_axis = retry_axis_with_baseline(
            std::span<const float>(retry_llr_clips),
            0.0F);
        const auto retry_llr_scale_axis = retry_axis_with_baseline(
            std::span<const float>(retry_llr_scales),
            1.0F);
        const auto retry_erase_axis = retry_axis_with_baseline(
            std::span<const float>(retry_weak_erase_fractions),
            0.0F);
        const auto retry_attenuation_axis = retry_axis_with_baseline(
            std::span<const float>(retry_attenuations),
            decode_options.attenuation);
        auto retry_decode_options = decode_options;
        if (retry_max_iterations.has_value()) {
            retry_decode_options.max_iterations = *retry_max_iterations;
        }
        if (fail_on_unclean_frame && !clean_frames_only) {
            throw std::invalid_argument(
                "fail-on-unclean-frame requires --clean-frames-only");
        }
        const auto retry_unsatisfied_erase_axis = retry_axis_with_baseline(
            std::span<const std::size_t>(retry_unsatisfied_erase_counts),
            0U);
        const auto retry_final_failed_erase_axis = retry_axis_with_baseline(
            std::span<const std::size_t>(retry_final_failed_erase_counts),
            0U);
        const auto retry_final_failed_flip_axis = retry_axis_with_baseline(
            std::span<const std::size_t>(retry_final_failed_flip_counts),
            0U);
        const auto retry_final_failed_greedy_flip_axis = retry_axis_with_baseline(
            std::span<const std::size_t>(retry_final_failed_greedy_flip_counts),
            0U);
        const auto retry_final_failed_syndrome_solve_axis = retry_axis_with_baseline(
            std::span<const std::size_t>(retry_final_failed_syndrome_solve_counts),
            0U);
        const auto retry_qam_plane_axis = retry_axis_with_baseline(
            std::span<const std::size_t>(retry_qam_planes),
            kNoRetryQamPlane);
        for (const auto& rule : variable_llr_scale_rules) {
            if (rule.variable >= profile.transmitted_bits) {
                throw std::invalid_argument(
                    "variable LLR scale variable is outside transmitted codeword");
            }
        }
        for (const auto& rule : variable_mod_llr_scale_rules) {
            if (rule.remainder >= profile.transmitted_bits) {
                throw std::invalid_argument(
                    "variable-mod LLR scale remainder is outside transmitted codeword");
            }
        }
        const auto graph = read_alist(alist_path);
        if (graph.variable_count != profile.full_codeword_bits()) {
            throw std::runtime_error("alist variable count does not match selected DTMB FEC rate");
        }
        std::optional<dtmb::core::LdpcSparseGraph> hard_h_gate_graph;
        if (hard_h_gate_enabled) {
            hard_h_gate_graph =
                dtmb::tools::ldpc_h_gate::load_clean_dtmb_graph(alist_path, fec_rate);
            if (hard_h_gate_graph->variable_count != profile.transmitted_bits) {
                throw std::runtime_error(
                    "hard H gate graph variable count does not match transmitted codeword");
            }
        }

        if (decode_batch_frames == 0) {
            usage(argv[0]);
            return 2;
        }
        if (ldpc_accel == LdpcAccel::cuda && !dtmb::tools::ldpc_cuda::backend_compiled()) {
            throw std::runtime_error(
                "--ldpc-accel cuda requested, but this binary was built without "
                "DTMB_CORE_ENABLE_CUDA_LDPC=ON");
        }
        if (cuda_retry_clips) {
            if (ldpc_accel != LdpcAccel::cuda) {
                throw std::invalid_argument(
                    "--cuda-retry-clips requires --ldpc-accel cuda");
            }
            if (retry_llr_clips.empty()) {
                throw std::invalid_argument(
                    "--cuda-retry-clips requires at least one --retry-llr-clip");
            }
            if (!retry_llr_scales.empty()
                || !retry_weak_erase_fractions.empty()
                || !retry_attenuations.empty()
                || !retry_unsatisfied_erase_counts.empty()
                || !retry_final_failed_erase_counts.empty()
                || !retry_final_failed_flip_counts.empty()
                || !retry_final_failed_greedy_flip_counts.empty()
                || !retry_final_failed_syndrome_solve_counts.empty()
                || !retry_qam_planes.empty()
                || !retry_variable_ranges.empty()
                || !retry_branch_windows.empty()
                || !retry_branch_window_groups.empty()
                || !retry_branch_window_rules.empty()
                || !retry_mode2_source_frame_windows.empty()
                || !retry_mode2_source_frame_window_rules.empty()
                || retry_layered
                || retry_require_ts_packet_continuity) {
                throw std::invalid_argument(
                    "--cuda-retry-clips currently supports flooding clip retries only");
            }
        }

        auto worker_count = requested_workers;
        if (worker_count == 0) {
            worker_count = std::thread::hardware_concurrency();
        }
        worker_count = std::max<std::size_t>(worker_count, 1);

        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        std::unique_ptr<std::ofstream> frame_diagnostics_file;
        std::unique_ptr<std::ofstream> packet_provenance_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);
        std::ostream* frame_diagnostics = nullptr;
        std::ostream* packet_provenance = nullptr;
        if (!frame_diagnostics_path.empty()) {
            frame_diagnostics_file =
                std::make_unique<std::ofstream>(frame_diagnostics_path);
            if (!*frame_diagnostics_file) {
                throw std::runtime_error(
                    "failed to open frame diagnostics output: "
                    + frame_diagnostics_path);
            }
            frame_diagnostics = frame_diagnostics_file.get();
        }
        if (!packet_provenance_path.empty()) {
            packet_provenance_file =
                std::make_unique<std::ofstream>(packet_provenance_path);
            if (!*packet_provenance_file) {
                throw std::runtime_error(
                    "failed to open packet provenance output: "
                    + packet_provenance_path);
            }
            packet_provenance = packet_provenance_file.get();
        }

        const auto frame_llr_count = codewords_per_frame * profile.transmitted_bits;
        std::vector<float> transmitted_llr(frame_llr_count);
        const auto frame_transport_bytes =
            codewords_per_frame * profile.output_bytes_per_codeword();
        const auto codeword_transport_bytes = profile.output_bytes_per_codeword();
        if ((codeword_transport_bytes % kTsPacketSize) != 0) {
            throw std::runtime_error(
                "DTMB codeword transport bytes must align to MPEG-TS packets");
        }
        const auto packets_per_codeword = codeword_transport_bytes / kTsPacketSize;
        for (const auto& rule : retry_final_failed_syndrome_solve_ts_header_template_rules) {
            if (rule.codeword >= codewords_per_frame) {
                throw std::invalid_argument(
                    "TS header template codeword slot is outside codewords-per-frame");
            }
            if (rule.packet >= packets_per_codeword) {
                throw std::invalid_argument(
                    "TS header template packet index is outside packets-per-codeword");
            }
        }
        const auto frame_bytes = static_cast<std::streamsize>(frame_llr_count * sizeof(float));

        std::size_t frame_count = 0;
        std::size_t codeword_count = 0;
        std::size_t converged_count = 0;
        std::size_t iteration_count = 0;
        std::size_t initial_syndrome_weight = 0;
        std::size_t early_rejected_count = 0;
        std::size_t final_syndrome_weight = 0;
        std::size_t bch_corrected_errors = 0;
        std::size_t bch_unclean_blocks = 0;
        std::size_t clean_frame_count = 0;
        std::size_t emitted_frame_count = 0;
        std::size_t partial_emitted_frame_count = 0;
        std::size_t omitted_frame_count = 0;
        std::size_t clean_codeword_count = 0;
        std::size_t bch_clean_payload_codeword_count = 0;
        std::size_t bch_clean_payload_packet_count = 0;
        std::size_t bch_clean_packet_salvage_emitted_packet_count = 0;
        std::size_t emitted_codeword_count = 0;
        std::size_t omitted_codeword_count = 0;
        std::size_t output_byte_count = 0;
        std::size_t output_packet_count = 0;
        std::size_t discontinuity_marked_packets = 0;
        std::size_t discontinuity_inserted_packets = 0;
        std::size_t continuity_guard_pending_pids = 0;
        std::size_t hard_h_gate_scored_windows = 0;
        std::size_t hard_h_gate_pass_windows = 0;
        std::size_t hard_h_gate_selected_frames = 0;
        std::size_t forced_selected_frames = 0;
        std::size_t mode2_source_frame_selected_frames = 0;
        std::size_t hard_h_gate_gap_filled_frames = 0;
        std::size_t hard_h_gate_skipped_frames = 0;
        std::size_t soft_decoded_frames = 0;
        std::size_t retry_attempt_count = 0;
        std::size_t retry_recovered_codeword_count = 0;
        std::size_t layered_retry_recovered_codeword_count = 0;
        std::size_t qam_plane_retry_recovered_codeword_count = 0;
        std::size_t final_failed_flip_retry_recovered_codeword_count = 0;
        std::size_t final_failed_greedy_flip_retry_recovered_codeword_count = 0;
        std::size_t final_failed_syndrome_solve_retry_recovered_codeword_count = 0;
        std::size_t branch_window_retry_recovered_codeword_count = 0;
        std::size_t branch_window_group_retry_recovered_codeword_count = 0;
        std::size_t retry_ts_continuity_rejected_codeword_count = 0;
        std::size_t retry_ts_continuity_context_recovered_codeword_count = 0;
        std::size_t cuda_ldpc_codewords = 0;
        std::size_t cuda_ldpc_cpu_fallback_codewords = 0;
        double cuda_ldpc_h2d_ms = 0.0;
        double cuda_ldpc_kernel_ms = 0.0;
        double cuda_ldpc_d2h_ms = 0.0;
        double cuda_ldpc_total_ms = 0.0;
        std::size_t cuda_retry_codewords = 0;
        double cuda_retry_h2d_ms = 0.0;
        double cuda_retry_kernel_ms = 0.0;
        double cuda_retry_d2h_ms = 0.0;
        double cuda_retry_total_ms = 0.0;
        double batch_prepare_ms = 0.0;
        double batch_baseline_ms = 0.0;
        double batch_snapshot_ms = 0.0;
        double batch_retry_ms = 0.0;
        double batch_worker_ms = 0.0;
        double batch_finalize_ms = 0.0;
        double batch_total_ms = 0.0;
        std::unordered_map<std::uint16_t, std::uint8_t> last_payload_continuity_counter;
        std::unordered_set<std::uint16_t> seen_payload_pids;
        std::unordered_set<std::uint16_t> pending_discontinuity_pids;
        TsContinuityContext retry_ts_continuity_context;
        const bool has_select_mode2_source_frame_selection =
            !select_mode2_source_frame_windows.empty()
            || !select_mode2_source_frame_window_rules.empty();

        const auto note_transport_gap = [&] {
            if (!mark_discontinuities && !insert_discontinuity_packets) {
                return;
            }
            pending_discontinuity_pids.insert(
                seen_payload_pids.begin(),
                seen_payload_pids.end());
        };
        const auto note_transport_gap_for_packet = [&](
            std::span<const std::uint8_t> packet) {
            if (!mark_discontinuities && !insert_discontinuity_packets) {
                return;
            }
            if (packet.size() < kTsPacketSize || packet[0] != kTsSyncByte
                || !ts_has_payload(packet)) {
                return;
            }
            const auto pid = ts_pid(packet);
            if (pid != kTsNullPid) {
                pending_discontinuity_pids.insert(pid);
            }
        };
        const auto note_transport_gap_for_packet_or_unknown = [&](
            std::span<const std::uint8_t> packet) {
            if (!ts_packet_syntax_valid(packet)) {
                note_transport_gap();
                return;
            }
            note_transport_gap_for_packet(packet);
        };
        const auto note_transport_gap_for_packets_or_unknown = [&](
            std::span<const std::uint8_t> bytes) {
            if (bytes.empty() || bytes.size() % kTsPacketSize != 0U) {
                note_transport_gap();
                return;
            }
            for (std::size_t offset = 0; offset < bytes.size();
                 offset += kTsPacketSize) {
                if (!ts_packet_syntax_valid(bytes.subspan(offset, kTsPacketSize))) {
                    note_transport_gap();
                    return;
                }
            }
            for (std::size_t offset = 0; offset < bytes.size();
                 offset += kTsPacketSize) {
                note_transport_gap_for_packet(bytes.subspan(offset, kTsPacketSize));
            }
        };
        const auto note_continuity_guard_gaps = [&](
            std::span<const std::uint8_t> bytes) {
            if (!mark_discontinuities && !insert_discontinuity_packets) {
                return;
            }
            if (bytes.empty() || bytes.size() % kTsPacketSize != 0U) {
                return;
            }
            auto candidate_counters = last_payload_continuity_counter;
            for (std::size_t offset = 0; offset < bytes.size();
                 offset += kTsPacketSize) {
                const auto packet = bytes.subspan(offset, kTsPacketSize);
                if (!ts_packet_syntax_valid(packet) || !ts_has_payload(packet)) {
                    continue;
                }
                const auto pid = ts_pid(packet);
                if (pid == kTsNullPid) {
                    continue;
                }
                const auto continuity_counter = ts_continuity_counter(packet);
                const auto previous = candidate_counters.find(pid);
                if (previous != candidate_counters.end()
                    && !ts_discontinuity_indicator(packet)) {
                    const auto expected = static_cast<std::uint8_t>(
                        (previous->second + 1U) & 0x0FU);
                    if (continuity_counter != expected) {
                        if (pending_discontinuity_pids.insert(pid).second) {
                            ++continuity_guard_pending_pids;
                        }
                    }
                }
                candidate_counters[pid] = continuity_counter;
            }
        };

        const auto note_output_transport = [&](
            std::span<const std::uint8_t> bytes,
            const char* kind,
            std::size_t frame_index,
            std::optional<std::size_t> codeword_slot,
            std::size_t packet_in_codeword_base = 0,
            std::optional<std::size_t> packets_per_codeword = std::nullopt) {
            const auto packet_count = bytes.size() / kTsPacketSize;
            for (std::size_t packet_index = 0; packet_index < packet_count; ++packet_index) {
                const auto packet = bytes.subspan(
                    packet_index * kTsPacketSize,
                    kTsPacketSize);
                const auto packet_in_codeword =
                    codeword_slot.has_value()
                        ? std::optional<std::size_t>{
                              packet_in_codeword_base + packet_index}
                        : std::optional<std::size_t>{};
                write_packet_provenance_row(
                    packet_provenance,
                    output_packet_count + packet_index,
                    kind,
                    frame_index,
                    codeword_slot,
                    packet_in_codeword,
                    packets_per_codeword,
                    packet);
            }
            output_packet_count += packet_count;
        };

        const auto signal_pending_pid_discontinuities = [&](
            std::span<std::uint8_t> bytes,
            std::size_t frame_index,
            std::optional<std::size_t> codeword_slot) {
            std::unordered_set<std::uint16_t> signaled_pids;
            std::size_t inserted = 0;
            std::size_t marked = 0;
            if (!pending_discontinuity_pids.empty() && insert_discontinuity_packets) {
                const auto packets = make_discontinuity_packets_for_pending_pids(
                    bytes,
                    pending_discontinuity_pids,
                    signaled_pids);
                for (const auto& packet : packets) {
                    write_packet_provenance_row(
                        packet_provenance,
                        output_packet_count,
                        "inserted_discontinuity",
                        frame_index,
                        codeword_slot,
                        std::optional<std::size_t>{},
                        std::optional<std::size_t>{},
                        packet);
                    write_all(output, packet);
                    ++output_packet_count;
                }
                inserted = packets.size();
                output_byte_count += inserted * kTsPacketSize;
            }
            if (!pending_discontinuity_pids.empty() && mark_discontinuities) {
                marked = mark_frame_discontinuities_for_pending_pids(
                    bytes,
                    pending_discontinuity_pids,
                    signaled_pids);
            }
            for (const auto pid : signaled_pids) {
                pending_discontinuity_pids.erase(pid);
            }
            return std::pair<std::size_t, std::size_t>{marked, inserted};
        };

        const auto note_emitted_transport = [&](std::span<const std::uint8_t> bytes) {
            collect_payload_pids(bytes, seen_payload_pids);
            for (std::size_t offset = 0; offset + kTsPacketSize <= bytes.size();
                 offset += kTsPacketSize) {
                const auto packet = bytes.subspan(offset, kTsPacketSize);
                if (!ts_packet_syntax_valid(packet) || !ts_has_payload(packet)) {
                    continue;
                }
                const auto pid = ts_pid(packet);
                if (pid != kTsNullPid) {
                    last_payload_continuity_counter[pid] =
                        ts_continuity_counter(packet);
                }
            }
        };

        const auto omit_frame = [&] {
            ++omitted_frame_count;
            note_transport_gap();
        };

        std::deque<PendingFecFrame> pending_frames;

        const auto write_frame_diagnostics = [&](
            const PendingFecFrame& pending,
            const DecodedFecFrame* decoded_frame,
            bool emitted,
            std::size_t frame_discontinuity_marked_packets,
            std::size_t frame_discontinuity_inserted_packets,
            std::size_t frame_emitted_packets,
            std::size_t frame_omitted_packets,
            std::size_t frame_bch_clean_packet_salvage_emitted_packets) {
            if (frame_diagnostics == nullptr) {
                return;
            }
            std::size_t frame_converged_codewords = 0;
            std::size_t frame_ldpc_iterations = 0;
            std::size_t frame_final_syndrome_weight = 0;
            std::size_t frame_initial_syndrome_weight = 0;
            std::size_t frame_early_rejected_codewords = 0;
            std::size_t frame_bch_corrected_errors = 0;
            std::size_t frame_bch_unclean_blocks = 0;
            std::size_t frame_clean_codewords = 0;
            std::size_t frame_emitted_codewords = 0;
            std::size_t frame_omitted_codewords = codewords_per_frame;
            std::vector<std::uint8_t> codeword_converged(codewords_per_frame, 0U);
            std::vector<std::size_t> codeword_ldpc_iterations(codewords_per_frame, 0U);
            std::vector<std::size_t> codeword_initial_syndrome_weight(
                codewords_per_frame,
                0U);
            std::vector<std::size_t> codeword_final_syndrome_weight(
                codewords_per_frame,
                0U);
            std::vector<std::uint8_t> codeword_early_rejected(codewords_per_frame, 0U);
            std::vector<std::uint8_t> codeword_bch_clean(codewords_per_frame, 0U);
            std::vector<std::size_t> codeword_bch_corrected_errors(
                codewords_per_frame,
                0U);
            std::vector<std::size_t> codeword_bch_unclean_blocks(
                codewords_per_frame,
                0U);
            std::vector<std::uint8_t> codeword_clean(codewords_per_frame, 0U);
            std::vector<std::uint8_t> codeword_emitted(codewords_per_frame, 0U);
            std::vector<std::uint8_t> codeword_omitted(codewords_per_frame, 1U);
            const auto llr_weak_threshold = 1.0F;
            std::size_t llr_sample_count = pending.transmitted_llr.size();
            double llr_abs_sum = 0.0;
            float llr_abs_min = 0.0F;
            float llr_abs_max = 0.0F;
            std::size_t llr_weak_count = 0;
            std::size_t llr_hard_one_count = 0;
            std::vector<double> codeword_llr_abs_sum(codewords_per_frame, 0.0);
            std::vector<std::size_t> codeword_llr_sample_count(codewords_per_frame, 0U);
            std::vector<std::size_t> codeword_weak_llr_count(codewords_per_frame, 0U);
            std::vector<std::size_t> codeword_hard_one_count(codewords_per_frame, 0U);
            std::vector<std::size_t> codeword_retry_attempts(codewords_per_frame, 0U);
            std::vector<std::uint8_t> codeword_retry_selected(codewords_per_frame, 0U);
            std::vector<std::uint8_t> codeword_retry_layered(codewords_per_frame, 0U);
            std::vector<double> codeword_retry_llr_scale(codewords_per_frame, 0.0);
            std::vector<double> codeword_retry_llr_clip(codewords_per_frame, 0.0);
            std::vector<double> codeword_retry_weak_erase_fraction(
                codewords_per_frame,
                0.0);
            std::vector<double> codeword_retry_attenuation(codewords_per_frame, 0.0);
            std::vector<std::size_t> codeword_retry_unsatisfied_erase_count(
                codewords_per_frame,
                0U);
            std::vector<std::size_t> codeword_retry_final_failed_erase_count(
                codewords_per_frame,
                0U);
            std::vector<std::size_t> codeword_retry_final_failed_flip_count(
                codewords_per_frame,
                0U);
            std::vector<std::size_t> codeword_retry_final_failed_greedy_flip_count(
                codewords_per_frame,
                0U);
            std::vector<std::size_t> codeword_retry_final_failed_syndrome_solve_count(
                codewords_per_frame,
                0U);
            std::vector<std::size_t> codeword_retry_qam_plane(
                codewords_per_frame,
                kNoRetryQamPlane);
            std::vector<std::size_t> codeword_retry_variable_range_first(
                codewords_per_frame,
                kNoRetryVariable);
            std::vector<std::size_t> codeword_retry_variable_range_last(
                codewords_per_frame,
                kNoRetryVariable);
            std::vector<std::size_t> codeword_retry_branch_window_first(
                codewords_per_frame,
                kNoRetryBranchWindow);
            std::vector<std::size_t> codeword_retry_branch_window_last(
                codewords_per_frame,
                kNoRetryBranchWindow);
            std::vector<std::string> codeword_retry_branch_window_group(
                codewords_per_frame,
                std::string{});
            std::vector<std::size_t> codeword_retry_bch_corrected_errors(
                codewords_per_frame,
                0U);
            if (!pending.transmitted_llr.empty()) {
                llr_abs_min = std::numeric_limits<float>::infinity();
                for (std::size_t index = 0; index < pending.transmitted_llr.size(); ++index) {
                    const auto llr = pending.transmitted_llr[index];
                    const auto abs_llr = std::abs(llr);
                    llr_abs_sum += static_cast<double>(abs_llr);
                    llr_abs_min = std::min(llr_abs_min, abs_llr);
                    llr_abs_max = std::max(llr_abs_max, abs_llr);
                    llr_weak_count += abs_llr < llr_weak_threshold ? 1U : 0U;
                    llr_hard_one_count += llr < 0.0F ? 1U : 0U;
                    const auto codeword = index / profile.transmitted_bits;
                    if (codeword < codewords_per_frame) {
                        codeword_llr_abs_sum[codeword] += static_cast<double>(abs_llr);
                        ++codeword_llr_sample_count[codeword];
                        codeword_weak_llr_count[codeword] +=
                            abs_llr < llr_weak_threshold ? 1U : 0U;
                        codeword_hard_one_count[codeword] += llr < 0.0F ? 1U : 0U;
                    }
                }
            }
            bool clean = false;
            if (decoded_frame != nullptr) {
                clean = decoded_frame->frame_clean;
                frame_bch_corrected_errors = decoded_frame->bch.corrected_errors;
                frame_bch_unclean_blocks = decoded_frame->bch.unclean_blocks;
                for (const auto value : decoded_frame->clean_codewords) {
                    frame_clean_codewords += value != 0U ? 1U : 0U;
                }
                for (const auto& result : decoded_frame->results) {
                    frame_converged_codewords += result.converged ? 1U : 0U;
                    frame_ldpc_iterations += result.iterations;
                    frame_final_syndrome_weight += result.syndrome_weight;
                }
                for (std::size_t codeword = 0;
                     codeword < codewords_per_frame
                     && codeword < decoded_frame->results.size();
                     ++codeword) {
                    const auto& result = decoded_frame->results[codeword];
                    codeword_converged[codeword] = result.converged ? 1U : 0U;
                    codeword_ldpc_iterations[codeword] = result.iterations;
                    codeword_final_syndrome_weight[codeword] = result.syndrome_weight;
                }
                for (std::size_t codeword = 0;
                     codeword < codewords_per_frame
                     && codeword < decoded_frame->initial_syndrome_weights.size();
                     ++codeword) {
                    const auto weight = decoded_frame->initial_syndrome_weights[codeword];
                    frame_initial_syndrome_weight += weight;
                    codeword_initial_syndrome_weight[codeword] = weight;
                }
                for (std::size_t codeword = 0;
                     codeword < codewords_per_frame
                     && codeword < decoded_frame->early_rejected.size();
                     ++codeword) {
                    const auto rejected = decoded_frame->early_rejected[codeword];
                    frame_early_rejected_codewords += rejected != 0U ? 1U : 0U;
                    codeword_early_rejected[codeword] = rejected != 0U ? 1U : 0U;
                }
                codeword_retry_attempts = decoded_frame->retry_attempts;
                codeword_retry_selected = decoded_frame->retry_selected;
                codeword_retry_layered = decoded_frame->retry_layered;
                codeword_retry_llr_scale = decoded_frame->retry_llr_scale;
                codeword_retry_llr_clip = decoded_frame->retry_llr_clip;
                codeword_retry_weak_erase_fraction =
                    decoded_frame->retry_weak_erase_fraction;
                codeword_retry_attenuation = decoded_frame->retry_attenuation;
                codeword_retry_unsatisfied_erase_count =
                    decoded_frame->retry_unsatisfied_erase_count;
                codeword_retry_final_failed_erase_count =
                    decoded_frame->retry_final_failed_erase_count;
                codeword_retry_final_failed_flip_count =
                    decoded_frame->retry_final_failed_flip_count;
                codeword_retry_final_failed_greedy_flip_count =
                    decoded_frame->retry_final_failed_greedy_flip_count;
                codeword_retry_final_failed_syndrome_solve_count =
                    decoded_frame->retry_final_failed_syndrome_solve_count;
                codeword_retry_qam_plane = decoded_frame->retry_qam_plane;
                codeword_retry_variable_range_first =
                    decoded_frame->retry_variable_range_first;
                codeword_retry_variable_range_last =
                    decoded_frame->retry_variable_range_last;
                codeword_retry_branch_window_first =
                    decoded_frame->retry_branch_window_first;
                codeword_retry_branch_window_last =
                    decoded_frame->retry_branch_window_last;
                codeword_retry_branch_window_group =
                    decoded_frame->retry_branch_window_group;
                codeword_retry_bch_corrected_errors =
                    decoded_frame->retry_bch_corrected_errors;
                const auto bch_blocks_per_codeword =
                    codewords_per_frame == 0
                        ? std::size_t{0}
                        : decoded_frame->bch.block_count / codewords_per_frame;
                if (bch_blocks_per_codeword * codewords_per_frame
                    == decoded_frame->bch.block_count) {
                    for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                        bool bch_clean = true;
                        const auto first_block = codeword * bch_blocks_per_codeword;
                        const auto last_block = first_block + bch_blocks_per_codeword;
                        for (std::size_t block = first_block; block < last_block; ++block) {
                            const auto block_clean =
                                block < decoded_frame->bch.block_clean.size()
                                && decoded_frame->bch.block_clean[block] != 0U;
                            bch_clean = bch_clean && block_clean;
                            codeword_bch_unclean_blocks[codeword] += block_clean ? 0U : 1U;
                            if (block < decoded_frame->bch.block_corrected_errors.size()) {
                                codeword_bch_corrected_errors[codeword] +=
                                    decoded_frame->bch.block_corrected_errors[block];
                            }
                        }
                        codeword_bch_clean[codeword] = bch_clean ? 1U : 0U;
                    }
                }
                for (std::size_t codeword = 0;
                     codeword < codewords_per_frame
                     && codeword < decoded_frame->clean_codewords.size();
                     ++codeword) {
                    codeword_clean[codeword] =
                        decoded_frame->clean_codewords[codeword] != 0U ? 1U : 0U;
                    const auto force_emit_codeword =
                        pending.force_selected && emit_forced_codewords;
                    const auto transport_offset = codeword * codeword_transport_bytes;
                    const auto codeword_bytes =
                        decoded_frame->transport_bytes.size()
                                >= transport_offset + codeword_transport_bytes
                            ? std::span<const std::uint8_t>(
                                  decoded_frame->transport_bytes.data() + transport_offset,
                                  codeword_transport_bytes)
                            : std::span<const std::uint8_t>();
                    const auto bch_emit_codeword =
                        emit_bch_clean_codewords
                        && codeword_bch_clean[codeword] != 0U
                        && ts_packets_syntax_valid(codeword_bytes);
                    const auto codeword_was_emitted =
                        emitted
                        && (!emit_clean_codewords
                            || codeword_clean[codeword] != 0U
                            || bch_emit_codeword
                            || force_emit_codeword);
                    codeword_emitted[codeword] = codeword_was_emitted ? 1U : 0U;
                    codeword_omitted[codeword] = codeword_was_emitted ? 0U : 1U;
                    frame_emitted_codewords += codeword_was_emitted ? 1U : 0U;
                }
                frame_omitted_codewords = codewords_per_frame - frame_emitted_codewords;
            }
            *frame_diagnostics
                << "{\"schema\":\"dtmb.ldpc_bch_frame.v1\""
                << ",\"frame_index\":" << pending.frame_index
                << ",\"hard_h_selected\":" << (pending.selected ? "true" : "false")
                << ",\"force_selected\":" << (pending.force_selected ? "true" : "false")
                << ",\"force_omitted\":" << (pending.force_omitted ? "true" : "false")
                << ",\"mode2_source_frame_selected\":"
                << (pending.mode2_source_frame_selected ? "true" : "false")
                << ",\"hard_h_gap_filled\":"
                << (pending.hard_h_gap_filled ? "true" : "false")
                << ",\"hard_h_clean_syndrome_weight\":"
                << pending.clean_syndrome_weight
                << ",\"llr_sample_count\":" << llr_sample_count
                << ",\"llr_abs_sum\":" << llr_abs_sum
                << ",\"llr_abs_mean\":"
                << (llr_sample_count == 0
                        ? 0.0
                        : llr_abs_sum / static_cast<double>(llr_sample_count))
                << ",\"llr_abs_min\":" << llr_abs_min
                << ",\"llr_abs_max\":" << llr_abs_max
                << ",\"llr_weak_threshold\":" << llr_weak_threshold
                << ",\"llr_weak_count\":" << llr_weak_count
                << ",\"llr_hard_one_count\":" << llr_hard_one_count
                << ",\"soft_decoded\":" << (decoded_frame != nullptr ? "true" : "false")
                << ",\"clean\":" << (clean ? "true" : "false")
                << ",\"emitted\":" << (emitted ? "true" : "false")
                << ",\"converged_codewords\":" << frame_converged_codewords
                << ",\"ldpc_iterations\":" << frame_ldpc_iterations
                << ",\"initial_syndrome_scored\":"
                << (early_syndrome_reject_ratio >= 0.0F ? "true" : "false")
                << ",\"initial_syndrome_weight\":" << frame_initial_syndrome_weight
                << ",\"early_rejected_codewords\":" << frame_early_rejected_codewords
                << ",\"final_syndrome_weight\":" << frame_final_syndrome_weight
                << ",\"bch_corrected_errors\":" << frame_bch_corrected_errors
                << ",\"bch_unclean_blocks\":" << frame_bch_unclean_blocks
                << ",\"clean_codewords\":" << frame_clean_codewords
                << ",\"emitted_codewords\":" << frame_emitted_codewords
                << ",\"omitted_codewords\":" << frame_omitted_codewords
                << ",\"discontinuity_marked_packets\":"
                << frame_discontinuity_marked_packets
                << ",\"discontinuity_inserted_packets\":"
                << frame_discontinuity_inserted_packets
                << ",\"emitted_packets\":" << frame_emitted_packets
                << ",\"omitted_packets\":" << frame_omitted_packets
                << ",\"packets_per_codeword\":"
                << codeword_transport_bytes / kTsPacketSize
                << ",\"bch_clean_packet_salvage_emitted_packets\":"
                << frame_bch_clean_packet_salvage_emitted_packets;
            write_json_array(*frame_diagnostics, "codeword_converged", codeword_converged);
            write_json_array(
                *frame_diagnostics,
                "codeword_ldpc_iterations",
                codeword_ldpc_iterations);
            write_json_array(
                *frame_diagnostics,
                "codeword_initial_syndrome_weight",
                codeword_initial_syndrome_weight);
            write_json_array(
                *frame_diagnostics,
                "codeword_final_syndrome_weight",
                codeword_final_syndrome_weight);
            write_json_array(
                *frame_diagnostics,
                "codeword_early_rejected",
                codeword_early_rejected);
            write_json_array(*frame_diagnostics, "codeword_bch_clean", codeword_bch_clean);
            write_json_array(
                *frame_diagnostics,
                "codeword_bch_corrected_errors",
                codeword_bch_corrected_errors);
            write_json_array(
                *frame_diagnostics,
                "codeword_bch_unclean_blocks",
                codeword_bch_unclean_blocks);
            write_json_array(*frame_diagnostics, "codeword_clean", codeword_clean);
            write_json_array(*frame_diagnostics, "codeword_emitted", codeword_emitted);
            write_json_array(*frame_diagnostics, "codeword_omitted", codeword_omitted);
            write_json_array(
                *frame_diagnostics,
                "codeword_llr_abs_sum",
                codeword_llr_abs_sum);
            write_json_array(
                *frame_diagnostics,
                "codeword_llr_sample_count",
                codeword_llr_sample_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_weak_llr_count",
                codeword_weak_llr_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_hard_one_count",
                codeword_hard_one_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_attempts",
                codeword_retry_attempts);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_selected",
                codeword_retry_selected);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_layered",
                codeword_retry_layered);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_llr_scale",
                codeword_retry_llr_scale);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_llr_clip",
                codeword_retry_llr_clip);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_weak_erase_fraction",
                codeword_retry_weak_erase_fraction);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_attenuation",
                codeword_retry_attenuation);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_unsatisfied_erase_count",
                codeword_retry_unsatisfied_erase_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_final_failed_erase_count",
                codeword_retry_final_failed_erase_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_final_failed_flip_count",
                codeword_retry_final_failed_flip_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_final_failed_greedy_flip_count",
                codeword_retry_final_failed_greedy_flip_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_final_failed_syndrome_solve_count",
                codeword_retry_final_failed_syndrome_solve_count);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_qam_plane",
                codeword_retry_qam_plane);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_variable_range_first",
                codeword_retry_variable_range_first);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_variable_range_last",
                codeword_retry_variable_range_last);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_branch_window_first",
                codeword_retry_branch_window_first);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_branch_window_last",
                codeword_retry_branch_window_last);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_branch_window_group",
                std::span<const std::string>(codeword_retry_branch_window_group));
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_bch_corrected_errors",
                codeword_retry_bch_corrected_errors);
            write_json_array(
                *frame_diagnostics,
                "codeword_retry_ts_continuity_rejected",
                decoded_frame != nullptr
                    ? std::span<const std::uint8_t>(
                          decoded_frame->retry_ts_continuity_rejected)
                    : std::span<const std::uint8_t>());
            if (failed_variable_diagnostics_top != 0U) {
                write_final_failed_variable_summaries(
                    *frame_diagnostics,
                    decoded_frame,
                    graph,
                    profile,
                    codewords_per_frame,
                    failed_variable_diagnostics_top);
            }
            *frame_diagnostics << "}\n";
            if (!*frame_diagnostics) {
                throw std::runtime_error("failed to write frame diagnostics");
            }
        };

        const auto flush_pending_frames = [&] {
            if (pending_frames.empty()) {
                return;
            }
            const auto batch_started = std::chrono::steady_clock::now();

            std::vector<DecodedFecFrame> decoded(pending_frames.size());
            std::vector<std::pair<std::size_t, std::size_t>> jobs;
            for (std::size_t frame = 0; frame < pending_frames.size(); ++frame) {
                if (!pending_frames[frame].selected) {
                    continue;
                }
                auto& decoded_frame = decoded[frame];
                decoded_frame.full_llr.resize(
                    codewords_per_frame * profile.full_codeword_bits());
                decoded_frame.baseline_decoded_bits.resize(
                    codewords_per_frame * profile.full_codeword_bits());
                decoded_frame.decoded_bits.resize(
                    codewords_per_frame * profile.full_codeword_bits());
                decoded_frame.message_bits.resize(codewords_per_frame * profile.message_bits);
                decoded_frame.transport_bytes.resize(frame_transport_bytes);
                decoded_frame.results.resize(codewords_per_frame);
                decoded_frame.initial_syndrome_weights.assign(codewords_per_frame, 0U);
                decoded_frame.early_rejected.assign(codewords_per_frame, 0U);
                decoded_frame.clean_codewords.assign(codewords_per_frame, 0U);
                decoded_frame.bch_clean_codewords.assign(codewords_per_frame, 0U);
                decoded_frame.retry_attempts.assign(codewords_per_frame, 0U);
                decoded_frame.retry_selected.assign(codewords_per_frame, 0U);
                decoded_frame.retry_layered.assign(codewords_per_frame, 0U);
                decoded_frame.retry_llr_scale.assign(codewords_per_frame, 0.0);
                decoded_frame.retry_llr_clip.assign(codewords_per_frame, 0.0);
                decoded_frame.retry_weak_erase_fraction.assign(
                    codewords_per_frame,
                    0.0);
                decoded_frame.retry_attenuation.assign(codewords_per_frame, 0.0);
                decoded_frame.retry_unsatisfied_erase_count.assign(
                    codewords_per_frame,
                    0U);
                decoded_frame.retry_final_failed_erase_count.assign(
                    codewords_per_frame,
                    0U);
                decoded_frame.retry_final_failed_flip_count.assign(
                    codewords_per_frame,
                    0U);
                decoded_frame.retry_final_failed_greedy_flip_count.assign(
                    codewords_per_frame,
                    0U);
                decoded_frame.retry_final_failed_syndrome_solve_count.assign(
                    codewords_per_frame,
                    0U);
                decoded_frame.retry_qam_plane.assign(
                    codewords_per_frame,
                    kNoRetryQamPlane);
                decoded_frame.retry_variable_range_first.assign(
                    codewords_per_frame,
                    kNoRetryVariable);
                decoded_frame.retry_variable_range_last.assign(
                    codewords_per_frame,
                    kNoRetryVariable);
                decoded_frame.retry_branch_window_first.assign(
                    codewords_per_frame,
                    kNoRetryBranchWindow);
                decoded_frame.retry_branch_window_last.assign(
                    codewords_per_frame,
                    kNoRetryBranchWindow);
                decoded_frame.retry_branch_window_group.assign(
                    codewords_per_frame,
                    std::string{});
                decoded_frame.retry_bch_corrected_errors.assign(
                    codewords_per_frame,
                    0U);
                decoded_frame.retry_ts_continuity_rejected.assign(
                    codewords_per_frame,
                    0U);

                const auto& frame_llr = pending_frames[frame].transmitted_llr;
                for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                    const auto tx_offset = codeword * profile.transmitted_bits;
                    const auto full_offset = codeword * profile.full_codeword_bits();
                    std::fill_n(
                        decoded_frame.full_llr.data() + full_offset,
                        profile.erased_parity_bits,
                        0.0F);
                    std::copy_n(
                        frame_llr.data() + tx_offset,
                        profile.transmitted_bits - profile.message_bits,
                        decoded_frame.full_llr.data()
                            + full_offset
                            + profile.erased_parity_bits);
                    std::copy_n(
                        frame_llr.data()
                            + tx_offset
                            + profile.transmitted_bits
                            - profile.message_bits,
                        profile.message_bits,
                        decoded_frame.full_llr.data()
                            + full_offset
                            + profile.full_parity_bits());
                    jobs.emplace_back(frame, codeword);
                }
            }
            const auto batch_prepared = std::chrono::steady_clock::now();

            std::vector<std::uint8_t> baseline_ready(jobs.size(), 0U);
            if (ldpc_accel == LdpcAccel::cuda && !jobs.empty()) {
                std::vector<std::size_t> cuda_job_indexes;
                std::vector<float> cuda_llr;
                cuda_job_indexes.reserve(jobs.size());
                cuda_llr.reserve(jobs.size() * profile.full_codeword_bits());
                for (std::size_t job_index = 0; job_index < jobs.size(); ++job_index) {
                    const auto [frame, codeword] = jobs[job_index];
                    auto& decoded_frame = decoded[frame];
                    const auto full_offset = codeword * profile.full_codeword_bits();
                    const auto llr_span = std::span<const float>(
                        decoded_frame.full_llr.data() + full_offset,
                        profile.full_codeword_bits());
                    const auto decoded_span = std::span<std::uint8_t>(
                        decoded_frame.decoded_bits.data() + full_offset,
                        profile.full_codeword_bits());
                    cuda_job_indexes.push_back(job_index);
                    if (primary_llr_clip.has_value()) {
                        const auto output_offset = cuda_llr.size();
                        cuda_llr.resize(output_offset + llr_span.size());
                        std::transform(
                            llr_span.begin(),
                            llr_span.end(),
                            cuda_llr.begin() + static_cast<std::ptrdiff_t>(output_offset),
                            [&](float llr) {
                                return std::clamp(
                                    llr,
                                    -*primary_llr_clip,
                                    *primary_llr_clip);
                            });
                    } else {
                        cuda_llr.insert(cuda_llr.end(), llr_span.begin(), llr_span.end());
                    }
                }

                if (!cuda_job_indexes.empty()) {
                    std::vector<std::uint8_t> cuda_bits(
                        cuda_job_indexes.size() * profile.full_codeword_bits());
                    const auto cuda_result = dtmb::tools::ldpc_cuda::decode_min_sum_batch(
                        cuda_llr,
                        cuda_job_indexes.size(),
                        graph,
                        cuda_bits,
                        decode_options,
                        early_syndrome_reject_ratio);
                    if (cuda_result.results.size() != cuda_job_indexes.size()) {
                        throw std::runtime_error("CUDA LDPC backend returned wrong result count");
                    }
                    if (cuda_result.initial_syndrome_weights.size()
                            != cuda_job_indexes.size()
                        || cuda_result.early_rejected.size()
                            != cuda_job_indexes.size()) {
                        throw std::runtime_error(
                            "CUDA LDPC backend returned wrong gate result count");
                    }
                    for (std::size_t batch_index = 0;
                         batch_index < cuda_job_indexes.size();
                         ++batch_index) {
                        const auto job_index = cuda_job_indexes[batch_index];
                        const auto [frame, codeword] = jobs[job_index];
                        auto& decoded_frame = decoded[frame];
                        const auto full_offset = codeword * profile.full_codeword_bits();
                        const auto decoded_span = std::span<std::uint8_t>(
                            decoded_frame.decoded_bits.data() + full_offset,
                            profile.full_codeword_bits());
                        std::copy_n(
                            cuda_bits.data() + batch_index * profile.full_codeword_bits(),
                            profile.full_codeword_bits(),
                            decoded_span.begin());
                        decoded_frame.results[codeword] = cuda_result.results[batch_index];
                        decoded_frame.initial_syndrome_weights[codeword] =
                            cuda_result.initial_syndrome_weights[batch_index];
                        decoded_frame.early_rejected[codeword] =
                            cuda_result.early_rejected[batch_index];
                        baseline_ready[job_index] = 1U;
                    }
                    cuda_ldpc_codewords += cuda_result.stats.codewords;
                    cuda_ldpc_h2d_ms += cuda_result.stats.h2d_ms;
                    cuda_ldpc_kernel_ms += cuda_result.stats.kernel_ms;
                    cuda_ldpc_d2h_ms += cuda_result.stats.d2h_ms;
                    cuda_ldpc_total_ms += cuda_result.stats.total_ms;
                }
            }
            const auto batch_baseline_finished = std::chrono::steady_clock::now();

            std::vector<std::uint8_t> baseline_snapshot_ready(jobs.size(), 0U);
            for (std::size_t job_index = 0; job_index < jobs.size(); ++job_index) {
                if (baseline_ready[job_index] == 0U) {
                    continue;
                }
                const auto [frame, codeword] = jobs[job_index];
                auto& decoded_frame = decoded[frame];
                const auto full_offset = codeword * profile.full_codeword_bits();
                std::copy_n(
                    decoded_frame.decoded_bits.data() + full_offset,
                    profile.full_codeword_bits(),
                    decoded_frame.baseline_decoded_bits.data() + full_offset);
                baseline_snapshot_ready[job_index] = 1U;
            }
            const auto batch_snapshot_finished = std::chrono::steady_clock::now();

            std::vector<std::uint8_t> cuda_clip_retry_attempted(jobs.size(), 0U);
            if (cuda_retry_clips && !jobs.empty()) {
                struct CudaClipRetryCandidate {
                    std::size_t job_index = 0;
                    float clip = 0.0F;
                };
                std::vector<CudaClipRetryCandidate> candidates;
                std::vector<float> candidate_llr;
                candidates.reserve(jobs.size() * retry_llr_clips.size());
                candidate_llr.reserve(
                    jobs.size() * retry_llr_clips.size()
                    * profile.full_codeword_bits());
                for (std::size_t job_index = 0; job_index < jobs.size(); ++job_index) {
                    const auto [frame, codeword] = jobs[job_index];
                    auto& decoded_frame = decoded[frame];
                    if (decoded_frame.results[codeword].converged) {
                        continue;
                    }
                    cuda_clip_retry_attempted[job_index] = 1U;
                    const auto full_offset = codeword * profile.full_codeword_bits();
                    const auto llr_span = std::span<const float>(
                        decoded_frame.full_llr.data() + full_offset,
                        profile.full_codeword_bits());
                    for (const auto clip : retry_llr_clips) {
                        const auto candidate_offset = candidate_llr.size();
                        candidate_llr.resize(
                            candidate_offset + profile.full_codeword_bits());
                        auto retry_span = std::span<float>(
                            candidate_llr.data() + candidate_offset,
                            profile.full_codeword_bits());
                        condition_retry_llr(
                            llr_span,
                            retry_span,
                            profile.erased_parity_bits,
                            1.0F,
                            clip,
                            0.0F,
                            0U,
                            std::nullopt,
                            std::span<const BranchWindow>{},
                            kNoRetryQamPlane,
                            graph);
                        candidates.push_back(CudaClipRetryCandidate{job_index, clip});
                        ++decoded_frame.retry_attempts[codeword];
                    }
                }

                if (!candidates.empty()) {
                    std::vector<std::uint8_t> candidate_bits(
                        candidates.size() * profile.full_codeword_bits());
                    const auto retry_result =
                        dtmb::tools::ldpc_cuda::decode_min_sum_batch(
                            candidate_llr,
                            candidates.size(),
                            graph,
                            candidate_bits,
                            retry_decode_options,
                            -1.0F);
                    if (retry_result.results.size() != candidates.size()) {
                        throw std::runtime_error(
                            "CUDA LDPC retry backend returned wrong result count");
                    }
                    cuda_retry_codewords += retry_result.stats.codewords;
                    cuda_retry_h2d_ms += retry_result.stats.h2d_ms;
                    cuda_retry_kernel_ms += retry_result.stats.kernel_ms;
                    cuda_retry_d2h_ms += retry_result.stats.d2h_ms;
                    cuda_retry_total_ms += retry_result.stats.total_ms;

                    const auto no_candidate = std::numeric_limits<std::size_t>::max();
                    std::vector<std::size_t> best_candidate(jobs.size(), no_candidate);
                    std::vector<std::size_t> best_bch_corrected(
                        jobs.size(),
                        std::numeric_limits<std::size_t>::max());
                    std::vector<std::uint8_t> candidate_message(profile.message_bits);
                    std::vector<std::uint8_t> candidate_transport(codeword_transport_bytes);
                    for (std::size_t candidate = 0;
                         candidate < candidates.size();
                         ++candidate) {
                        const auto& result = retry_result.results[candidate];
                        if (!result.converged) {
                            continue;
                        }
                        const auto job_index = candidates[candidate].job_index;
                        const auto [frame, codeword] = jobs[job_index];
                        (void)frame;
                        const auto* bits = candidate_bits.data()
                            + candidate * profile.full_codeword_bits();
                        std::copy_n(
                            bits + profile.full_parity_bits(),
                            profile.message_bits,
                            candidate_message.data());
                        const auto bch =
                            dtmb::core::dtmb_bch_descramble_message_bits(
                                candidate_message,
                                candidate_transport,
                                true,
                                codeword * codeword_transport_bytes * 8U);
                        if (bch.unclean_blocks != 0U) {
                            continue;
                        }
                        if (retry_require_ts_packet_syntax
                            && !ts_packets_syntax_valid(candidate_transport)) {
                            continue;
                        }
                        const auto previous = best_candidate[job_index];
                        if (bch.corrected_errors > best_bch_corrected[job_index]
                            || (bch.corrected_errors
                                    == best_bch_corrected[job_index]
                                && previous != no_candidate
                                && result.iterations
                                    >= retry_result.results[previous].iterations)) {
                            continue;
                        }
                        best_candidate[job_index] = candidate;
                        best_bch_corrected[job_index] = bch.corrected_errors;
                    }

                    for (std::size_t job_index = 0;
                         job_index < jobs.size();
                         ++job_index) {
                        const auto candidate = best_candidate[job_index];
                        if (candidate == no_candidate) {
                            continue;
                        }
                        const auto [frame, codeword] = jobs[job_index];
                        auto& decoded_frame = decoded[frame];
                        const auto full_offset = codeword * profile.full_codeword_bits();
                        std::copy_n(
                            candidate_bits.data()
                                + candidate * profile.full_codeword_bits(),
                            profile.full_codeword_bits(),
                            decoded_frame.decoded_bits.data() + full_offset);
                        decoded_frame.results[codeword] =
                            retry_result.results[candidate];
                        decoded_frame.retry_selected[codeword] = 1U;
                        decoded_frame.retry_llr_scale[codeword] = 1.0F;
                        decoded_frame.retry_llr_clip[codeword] =
                            candidates[candidate].clip;
                        decoded_frame.retry_attenuation[codeword] =
                            retry_decode_options.attenuation;
                        decoded_frame.retry_bch_corrected_errors[codeword] =
                            best_bch_corrected[job_index];
                    }
                }
            }
            const auto batch_retry_finished = std::chrono::steady_clock::now();

            const auto effective_workers = std::min<std::size_t>(worker_count, jobs.size());
            std::vector<std::thread> workers;
            workers.reserve(effective_workers);
            for (std::size_t worker = 0; worker < effective_workers; ++worker) {
                workers.emplace_back([&, worker] {
                    std::vector<float> primary_llr_scratch;
                    if (primary_llr_clip.has_value()) {
                        primary_llr_scratch.resize(profile.full_codeword_bits());
                    }
                    for (std::size_t job_index = worker;
                         job_index < jobs.size();
                         job_index += effective_workers) {
                        const auto [frame, codeword] = jobs[job_index];
                        auto& decoded_frame = decoded[frame];
                        const auto full_offset = codeword * profile.full_codeword_bits();
                        const auto llr_span = std::span<const float>(
                            decoded_frame.full_llr.data() + full_offset,
                            profile.full_codeword_bits());
                        const auto decoded_span = std::span<std::uint8_t>(
                            decoded_frame.decoded_bits.data() + full_offset,
                            profile.full_codeword_bits());
                        if (baseline_ready[job_index] == 0U) {
                            auto baseline_llr_span = llr_span;
                            if (primary_llr_clip.has_value()) {
                                std::transform(
                                    llr_span.begin(),
                                    llr_span.end(),
                                    primary_llr_scratch.begin(),
                                    [&](float llr) {
                                        return std::clamp(
                                            llr,
                                            -*primary_llr_clip,
                                            *primary_llr_clip);
                                    });
                                baseline_llr_span = primary_llr_scratch;
                            }
                            if (ldpc_accel == LdpcAccel::cuda) {
                                ++cuda_ldpc_cpu_fallback_codewords;
                            }
                            if (early_syndrome_reject_ratio >= 0.0F) {
                                for (std::size_t bit = 0; bit < profile.full_codeword_bits(); ++bit) {
                                    decoded_span[bit] =
                                        baseline_llr_span[bit] < 0.0F ? 1U : 0U;
                                }
                                const auto initial = dtmb::core::ldpc_syndrome_weight(
                                    decoded_span,
                                    graph);
                                decoded_frame.initial_syndrome_weights[codeword] = initial;
                                const auto initial_ratio =
                                    static_cast<float>(initial)
                                    / static_cast<float>(graph.check_count());
                                if (initial == 0) {
                                    decoded_frame.results[codeword] =
                                        dtmb::core::LdpcDecodeResult{0, true, 0};
                                } else if (initial_ratio >= early_syndrome_reject_ratio) {
                                    decoded_frame.results[codeword] =
                                        dtmb::core::LdpcDecodeResult{0, false, initial};
                                    decoded_frame.early_rejected[codeword] = 1U;
                                } else {
                                    decoded_frame.results[codeword] =
                                        dtmb::core::ldpc_decode_min_sum_sparse(
                                            baseline_llr_span,
                                            graph,
                                            decoded_span,
                                            decode_options);
                                }
                            } else {
                                decoded_frame.results[codeword] =
                                    dtmb::core::ldpc_decode_min_sum_sparse(
                                        baseline_llr_span,
                                        graph,
                                        decoded_span,
                                        decode_options);
                            }
                        }
                        if (baseline_snapshot_ready[job_index] == 0U) {
                            std::copy_n(
                                decoded_span.data(),
                                profile.full_codeword_bits(),
                                decoded_frame.baseline_decoded_bits.data() + full_offset);
                            baseline_snapshot_ready[job_index] = 1U;
                        }
                        std::vector<std::uint8_t> candidate_message(
                            profile.message_bits);
                        std::vector<std::uint8_t> candidate_transport(
                            codeword_transport_bytes);
                        const auto bch_score = [&](
                            std::span<const std::uint8_t> candidate_bits) {
                            std::copy_n(
                                candidate_bits.data() + profile.full_parity_bits(),
                                profile.message_bits,
                                candidate_message.data());
                            return dtmb::core::dtmb_bch_descramble_message_bits(
                                candidate_message,
                                candidate_transport,
                                true,
                                codeword * codeword_transport_bytes * 8U);
                        };
                        auto baseline_bch_clean = false;
                        if (decoded_frame.results[codeword].converged) {
                            baseline_bch_clean =
                                bch_score(decoded_span).unclean_blocks == 0;
                        }
                        if (retry_enabled && !baseline_bch_clean
                            && cuda_clip_retry_attempted[job_index] == 0U) {
                            const auto static_branch_windows =
                                active_branch_windows(
                                    pending_frames[frame].frame_index,
                                    std::span<const BranchWindow>(
                                        retry_branch_windows),
                                    std::span<const FrameRange>(
                                        retry_branch_window_frame_ranges),
                                    std::span<const BranchWindowRule>(
                                        retry_branch_window_rules));
                            const auto static_branch_window_groups =
                                active_branch_window_groups(
                                    pending_frames[frame].frame_index,
                                    std::span<const std::vector<BranchWindow>>(
                                        retry_branch_window_groups),
                                    std::span<const FrameRange>(
                                        retry_branch_window_frame_ranges));
                            const auto mode2_source_frame_windows =
                                active_mode2_source_frame_windows(
                                    pending_frames[frame].frame_index,
                                    std::span<const FrameRange>(
                                        retry_mode2_source_frame_windows),
                                    std::span<const SourceFrameWindowRule>(
                                        retry_mode2_source_frame_window_rules));
                            const auto mode2_source_branch_windows =
                                add_merged_branch_window_candidates(
                                    mode2_source_frame_branch_windows(
                                        pending_frames[frame].frame_index,
                                        std::span<const FrameRange>(
                                            mode2_source_frame_windows)));
                            std::vector<std::vector<BranchWindow>>
                                branch_window_candidates;
                            branch_window_candidates.push_back({});
                            for (const auto& window : static_branch_windows) {
                                branch_window_candidates.push_back({window});
                            }
                            for (const auto& window : mode2_source_branch_windows) {
                                branch_window_candidates.push_back({window});
                            }
                            branch_window_candidates.insert(
                                branch_window_candidates.end(),
                                static_branch_window_groups.begin(),
                                static_branch_window_groups.end());
                            std::vector<float> retry_llr(profile.full_codeword_bits());
                            std::vector<std::uint8_t> retry_bits(
                                profile.full_codeword_bits());
                            std::vector<std::uint8_t> best_bits;
                            dtmb::core::LdpcDecodeResult best_result;
                            std::size_t best_bch_corrected =
                                std::numeric_limits<std::size_t>::max();
                            float best_llr_scale = 0.0F;
                            float best_clip = 0.0F;
                            float best_erase = 0.0F;
                            float best_attenuation = 0.0F;
                            std::size_t best_unsatisfied_erase_count = 0;
                            std::size_t best_final_failed_erase_count = 0;
                            std::size_t best_final_failed_flip_count = 0;
                            std::size_t best_final_failed_greedy_flip_count = 0;
                            std::size_t best_final_failed_syndrome_solve_count = 0;
                            std::size_t best_qam_plane = kNoRetryQamPlane;
                            std::size_t best_variable_range_first = kNoRetryVariable;
                            std::size_t best_variable_range_last = kNoRetryVariable;
                            std::size_t best_branch_window_first =
                                kNoRetryBranchWindow;
                            std::size_t best_branch_window_last =
                                kNoRetryBranchWindow;
                            std::string best_branch_window_group;
                            bool best_layered = false;
                            for (const auto llr_scale : retry_llr_scale_axis) {
                                for (const auto clip : retry_clip_axis) {
                                    for (const auto erase_fraction : retry_erase_axis) {
                                        for (const auto unsatisfied_erase_count :
                                             retry_unsatisfied_erase_axis) {
                                        for (const auto final_failed_erase_count :
                                             retry_final_failed_erase_axis) {
                                        for (const auto final_failed_flip_count :
                                             retry_final_failed_flip_axis) {
                                        for (const auto final_failed_greedy_flip_count :
                                             retry_final_failed_greedy_flip_axis) {
                                        for (const auto final_failed_syndrome_solve_count :
                                             retry_final_failed_syndrome_solve_axis) {
                                            for (const auto attenuation :
                                                 retry_attenuation_axis) {
                                            for (std::size_t variable_range_index = 0;
                                                 variable_range_index
                                                 <= retry_variable_ranges.size();
                                                 ++variable_range_index) {
                                                const auto has_variable_range =
                                                    variable_range_index != 0U;
                                                std::optional<VariableRange> variable_range;
                                                if (has_variable_range) {
                                                    variable_range = retry_variable_ranges[
                                                        variable_range_index - 1U];
                                                }
                                                for (const auto& branch_windows :
                                                     branch_window_candidates) {
                                                const auto has_branch_selection =
                                                    !branch_windows.empty();
                                                for (const auto qam_plane :
                                                     retry_qam_plane_axis) {
                                                    const auto has_qam_plane =
                                                        qam_plane < kQam64LlrsPerSymbol;
                                                    const auto algorithm_count =
                                                        retry_layered ? 2U : 1U;
                                                    for (std::size_t algorithm = 0;
                                                         algorithm < algorithm_count;
                                                         ++algorithm) {
                                                        const auto layered = algorithm == 1U;
                                                        if (!layered
                                                            && llr_scale == 1.0F
                                                            && clip == 0.0F
                                                            && erase_fraction == 0.0F
                                                            && unsatisfied_erase_count == 0U
                                                            && final_failed_erase_count == 0U
                                                            && final_failed_flip_count == 0U
                                                            && final_failed_greedy_flip_count
                                                                == 0U
                                                            && final_failed_syndrome_solve_count
                                                                == 0U
                                                            && attenuation
                                                                == decode_options.attenuation
                                                            && !has_variable_range
                                                            && !has_branch_selection
                                                            && !has_qam_plane) {
                                                            continue;
                                                        }
                                                        ++decoded_frame.retry_attempts[codeword];
                                                        condition_retry_llr(
                                                            llr_span,
                                                            retry_llr,
                                                            profile.erased_parity_bits,
                                                            llr_scale,
                                                            clip,
                                                            erase_fraction,
                                                            unsatisfied_erase_count,
                                                            variable_range,
                                                            std::span<const BranchWindow>(
                                                                branch_windows),
                                                            qam_plane,
                                                            graph);
                                                        erase_final_failed_variables(
                                                            retry_llr,
                                                            decoded_span,
                                                            final_failed_erase_count,
                                                            graph,
                                                            profile);
                                                        flip_final_failed_variables(
                                                            retry_llr,
                                                            decoded_span,
                                                            final_failed_flip_count,
                                                            graph,
                                                            profile);
                                                        if (!has_transmitted_llr_evidence(
                                                                retry_llr,
                                                                profile.erased_parity_bits)) {
                                                            continue;
                                                        }
                                                        auto retry_options = retry_decode_options;
                                                        retry_options.attenuation = attenuation;
                                                        auto retry_result = layered
                                                            ? dtmb::core::
                                                                  ldpc_decode_layered_min_sum_sparse(
                                                                      retry_llr,
                                                                      graph,
                                                                      retry_bits,
                                                                      retry_options)
                                                            : dtmb::core::
                                                                  ldpc_decode_min_sum_sparse(
                                                                      retry_llr,
                                                                      graph,
                                                                      retry_bits,
                                                                      retry_options);
                                                        if (!retry_result.converged
                                                            && final_failed_greedy_flip_count
                                                                != 0U) {
                                                            const auto greedy =
                                                                greedy_flip_final_failed_variables(
                                                                    retry_bits,
                                                                    final_failed_greedy_flip_count,
                                                                    graph,
                                                                    profile);
                                                            retry_result.syndrome_weight =
                                                                greedy.final_syndrome_weight;
                                                            retry_result.converged =
                                                                greedy.final_syndrome_weight
                                                                == 0U;
                                                        }
                                                        if (!retry_result.converged
                                                            && final_failed_syndrome_solve_count
                                                                != 0U) {
                                                            const auto solved =
                                                                solve_final_failed_syndrome(
                                                                    retry_bits,
                                                                    retry_llr,
                                                                    final_failed_syndrome_solve_count,
                                                                    retry_final_failed_syndrome_solve_minimize_cost_passes,
                                                                    retry_final_failed_syndrome_solve_nullspace_search_count,
                                                                    retry_final_failed_syndrome_solve_nullspace_depth,
                                                                    std::span<const BitValueConstraint>{},
                                                                    [&](std::span<const std::uint8_t> bits) {
                                                                        if (!retry_require_ts_packet_syntax) {
                                                                            return true;
                                                                        }
                                                                        const auto candidate_bch = bch_score(bits);
                                                                        return candidate_bch.unclean_blocks == 0
                                                                            && ts_packets_syntax_valid(candidate_transport);
                                                                    },
                                                                    graph,
                                                                    profile);
                                                            retry_result.syndrome_weight =
                                                                solved.final_syndrome_weight;
                                                            retry_result.converged =
                                                                solved.solved;
                                                        }
                                                        if (!retry_result.converged) {
                                                            continue;
                                                        }
                                                        const auto retry_bch =
                                                            bch_score(retry_bits);
                                                        if (retry_bch.unclean_blocks != 0) {
                                                            continue;
                                                        }
                                                        if (retry_require_ts_packet_syntax
                                                            && !ts_packets_syntax_valid(
                                                                candidate_transport)) {
                                                            continue;
                                                        }
                                                        if (retry_bch.corrected_errors
                                                                > best_bch_corrected
                                                            || (retry_bch.corrected_errors
                                                                    == best_bch_corrected
                                                                && !best_bits.empty()
                                                                && retry_result.iterations
                                                                    >= best_result.iterations)) {
                                                            continue;
                                                        }
                                                        best_bits.assign(
                                                            retry_bits.begin(),
                                                            retry_bits.end());
                                                        best_result = retry_result;
                                                        best_bch_corrected =
                                                            retry_bch.corrected_errors;
                                                        best_llr_scale = llr_scale;
                                                        best_clip = clip;
                                                        best_erase = erase_fraction;
                                                        best_attenuation = attenuation;
                                                        best_unsatisfied_erase_count =
                                                            unsatisfied_erase_count;
                                                        best_final_failed_erase_count =
                                                            final_failed_erase_count;
                                                        best_final_failed_flip_count =
                                                            final_failed_flip_count;
                                                        best_final_failed_greedy_flip_count =
                                                            final_failed_greedy_flip_count;
                                                        best_final_failed_syndrome_solve_count =
                                                            final_failed_syndrome_solve_count;
                                                        best_qam_plane = qam_plane;
                                                        best_variable_range_first =
                                                            variable_range.has_value()
                                                                ? variable_range->first
                                                                : kNoRetryVariable;
                                                        best_variable_range_last =
                                                            variable_range.has_value()
                                                                ? variable_range->last
                                                                : kNoRetryVariable;
                                                        if (branch_windows.size() == 1U) {
                                                            best_branch_window_first =
                                                                branch_windows.front().first;
                                                            best_branch_window_last =
                                                                branch_windows.front().last;
                                                            best_branch_window_group.clear();
                                                        } else if (branch_windows.size() > 1U) {
                                                            best_branch_window_first =
                                                                kNoRetryBranchWindow;
                                                            best_branch_window_last =
                                                                kNoRetryBranchWindow;
                                                            best_branch_window_group =
                                                                branch_window_group_label(
                                                                    std::span<const BranchWindow>(
                                                                        branch_windows));
                                                        } else {
                                                            best_branch_window_first =
                                                                kNoRetryBranchWindow;
                                                            best_branch_window_last =
                                                                kNoRetryBranchWindow;
                                                            best_branch_window_group.clear();
                                                        }
                                                        best_layered = layered;
                                                    }
                                                }
                                            }
                                                }
                                            }
                                        }
                                        }
                                        }
                                        }
                                        }
                                    }
                                }
                            }
                            if (!best_bits.empty()) {
                                std::copy(
                                    best_bits.begin(),
                                    best_bits.end(),
                                    decoded_span.begin());
                                decoded_frame.results[codeword] = best_result;
                                decoded_frame.retry_selected[codeword] = 1U;
                                decoded_frame.retry_llr_scale[codeword] =
                                    best_llr_scale;
                                decoded_frame.retry_llr_clip[codeword] = best_clip;
                                decoded_frame.retry_weak_erase_fraction[codeword] =
                                    best_erase;
                                decoded_frame.retry_attenuation[codeword] =
                                    best_attenuation;
                                decoded_frame.retry_unsatisfied_erase_count[codeword] =
                                    best_unsatisfied_erase_count;
                                decoded_frame.retry_final_failed_erase_count[codeword] =
                                    best_final_failed_erase_count;
                                decoded_frame.retry_final_failed_flip_count[codeword] =
                                    best_final_failed_flip_count;
                                decoded_frame
                                    .retry_final_failed_greedy_flip_count[codeword] =
                                    best_final_failed_greedy_flip_count;
                                decoded_frame
                                    .retry_final_failed_syndrome_solve_count[codeword] =
                                    best_final_failed_syndrome_solve_count;
                                decoded_frame.retry_qam_plane[codeword] =
                                    best_qam_plane;
                                decoded_frame.retry_variable_range_first[codeword] =
                                    best_variable_range_first;
                                decoded_frame.retry_variable_range_last[codeword] =
                                    best_variable_range_last;
                                decoded_frame.retry_branch_window_first[codeword] =
                                    best_branch_window_first;
                                decoded_frame.retry_branch_window_last[codeword] =
                                    best_branch_window_last;
                                decoded_frame.retry_branch_window_group[codeword] =
                                    best_branch_window_group;
                                decoded_frame.retry_bch_corrected_errors[codeword] =
                                    best_bch_corrected;
                                decoded_frame.retry_layered[codeword] =
                                    best_layered ? 1U : 0U;
                            }
                        }
                        std::copy_n(
                            decoded_frame.decoded_bits.data()
                                + full_offset
                                + profile.full_parity_bits(),
                            profile.message_bits,
                            decoded_frame.message_bits.data()
                                + codeword * profile.message_bits);
                    }
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
            const auto batch_workers_finished = std::chrono::steady_clock::now();

            for (std::size_t frame = 0; frame < pending_frames.size(); ++frame) {
                if (!pending_frames[frame].selected) {
                    write_frame_diagnostics(
                        pending_frames[frame],
                        nullptr,
                        false,
                        0,
                        0,
                        0,
                        0,
                        0);
                    omitted_codeword_count += codewords_per_frame;
                    omit_frame();
                    continue;
                }
                auto& decoded_frame = decoded[frame];
                decoded_frame.bch = dtmb::core::dtmb_bch_descramble_message_bits(
                    decoded_frame.message_bits,
                    decoded_frame.transport_bytes);
                const auto bch_blocks_per_codeword =
                    decoded_frame.bch.block_count / codewords_per_frame;
                if (bch_blocks_per_codeword * codewords_per_frame
                    != decoded_frame.bch.block_count) {
                    throw std::runtime_error(
                        "BCH block count does not align to FEC codewords");
                }
                const auto bch_block_payload_bytes =
                    bch_blocks_per_codeword == 0
                        ? std::size_t{0}
                        : codeword_transport_bytes / bch_blocks_per_codeword;
                if (bch_block_payload_bytes == 0
                    || codeword_transport_bytes
                            != bch_block_payload_bytes * bch_blocks_per_codeword
                    || (kTsPacketSize % bch_block_payload_bytes) != 0U) {
                    throw std::runtime_error(
                        "BCH payload blocks do not align to MPEG-TS packets");
                }
                const auto bch_blocks_per_ts_packet =
                    kTsPacketSize / bch_block_payload_bytes;
                const auto packets_per_codeword =
                    codeword_transport_bytes / kTsPacketSize;
                for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                    bool bch_clean = true;
                    const auto first_block = codeword * bch_blocks_per_codeword;
                    const auto last_block = first_block + bch_blocks_per_codeword;
                    for (std::size_t block = first_block; block < last_block; ++block) {
                        bch_clean = bch_clean
                            && block < decoded_frame.bch.block_clean.size()
                            && decoded_frame.bch.block_clean[block] != 0U;
                    }
                    decoded_frame.bch_clean_codewords[codeword] =
                        bch_clean ? 1U : 0U;
                    decoded_frame.clean_codewords[codeword] =
                        (decoded_frame.results[codeword].converged && bch_clean) ? 1U : 0U;
                    bch_clean_payload_codeword_count += bch_clean ? 1U : 0U;
                    const auto transport_offset = codeword * codeword_transport_bytes;
                    const auto codeword_bytes = std::span<const std::uint8_t>(
                        decoded_frame.transport_bytes.data() + transport_offset,
                        codeword_transport_bytes);
                    for (std::size_t packet = 0; packet < packets_per_codeword; ++packet) {
                        bool packet_bch_clean = true;
                        const auto packet_first_block =
                            first_block + packet * bch_blocks_per_ts_packet;
                        const auto packet_last_block =
                            packet_first_block + bch_blocks_per_ts_packet;
                        for (std::size_t block = packet_first_block;
                             block < packet_last_block;
                             ++block) {
                            packet_bch_clean = packet_bch_clean
                                && block < decoded_frame.bch.block_clean.size()
                                && decoded_frame.bch.block_clean[block] != 0U;
                        }
                        const auto packet_bytes = codeword_bytes.subspan(
                            packet * kTsPacketSize,
                            kTsPacketSize);
                        if (packet_bch_clean && ts_packet_syntax_valid(packet_bytes)) {
                            ++bch_clean_payload_packet_count;
                        }
                    }
                }
                const auto try_continuity_context_syndrome_retry = [&](
                    std::size_t codeword,
                    const TsContinuityContext& continuity_context) {
                    if (codeword >= decoded_frame.retry_selected.size()
                        || decoded_frame.retry_selected[codeword] == 0U
                        || codeword >= decoded_frame.retry_final_failed_syndrome_solve_count.size()
                        || decoded_frame.retry_final_failed_syndrome_solve_count[codeword] == 0U
                        || codeword >= decoded_frame.retry_branch_window_group.size()
                        || !decoded_frame.retry_branch_window_group[codeword].empty()) {
                        return false;
                    }

                    const auto full_offset = codeword * profile.full_codeword_bits();
                    const auto transport_offset = codeword * codeword_transport_bytes;
                    const auto llr_span = std::span<const float>(
                        decoded_frame.full_llr.data() + full_offset,
                        profile.full_codeword_bits());
                    const auto baseline_span = std::span<const std::uint8_t>(
                        decoded_frame.baseline_decoded_bits.data() + full_offset,
                        profile.full_codeword_bits());

                    std::optional<VariableRange> variable_range;
                    if (decoded_frame.retry_variable_range_first[codeword]
                            != kNoRetryVariable
                        && decoded_frame.retry_variable_range_last[codeword]
                            != kNoRetryVariable) {
                        variable_range = VariableRange{
                            decoded_frame.retry_variable_range_first[codeword],
                            decoded_frame.retry_variable_range_last[codeword]};
                    }

                    std::vector<BranchWindow> branch_windows;
                    if (decoded_frame.retry_branch_window_first[codeword]
                            != kNoRetryBranchWindow
                        && decoded_frame.retry_branch_window_last[codeword]
                            != kNoRetryBranchWindow) {
                        branch_windows.push_back(
                            BranchWindow{
                                decoded_frame.retry_branch_window_first[codeword],
                                decoded_frame.retry_branch_window_last[codeword]});
                    }

                    std::vector<float> retry_llr(profile.full_codeword_bits());
                    std::vector<std::uint8_t> retry_bits(profile.full_codeword_bits());
                    condition_retry_llr(
                        llr_span,
                        retry_llr,
                        profile.erased_parity_bits,
                        static_cast<float>(decoded_frame.retry_llr_scale[codeword]),
                        static_cast<float>(decoded_frame.retry_llr_clip[codeword]),
                        static_cast<float>(decoded_frame.retry_weak_erase_fraction[codeword]),
                        decoded_frame.retry_unsatisfied_erase_count[codeword],
                        variable_range,
                        std::span<const BranchWindow>(branch_windows),
                        decoded_frame.retry_qam_plane[codeword],
                        graph);
                    erase_final_failed_variables(
                        retry_llr,
                        baseline_span,
                        decoded_frame.retry_final_failed_erase_count[codeword],
                        graph,
                        profile);
                    flip_final_failed_variables(
                        retry_llr,
                        baseline_span,
                        decoded_frame.retry_final_failed_flip_count[codeword],
                        graph,
                        profile);
                    if (!has_transmitted_llr_evidence(
                            retry_llr,
                            profile.erased_parity_bits)) {
                        return false;
                    }

                    auto retry_options = retry_decode_options;
                    retry_options.attenuation =
                        static_cast<float>(decoded_frame.retry_attenuation[codeword]);
                    auto retry_result = decoded_frame.retry_layered[codeword] != 0U
                        ? dtmb::core::ldpc_decode_layered_min_sum_sparse(
                              retry_llr,
                              graph,
                              retry_bits,
                              retry_options)
                        : dtmb::core::ldpc_decode_min_sum_sparse(
                              retry_llr,
                              graph,
                              retry_bits,
                              retry_options);
                    if (!retry_result.converged
                        && decoded_frame.retry_final_failed_greedy_flip_count[codeword] != 0U) {
                        const auto greedy = greedy_flip_final_failed_variables(
                            retry_bits,
                            decoded_frame.retry_final_failed_greedy_flip_count[codeword],
                            graph,
                            profile);
                        retry_result.syndrome_weight = greedy.final_syndrome_weight;
                        retry_result.converged = greedy.final_syndrome_weight == 0U;
                    }

                    std::vector<std::uint8_t> candidate_message(profile.message_bits);
                    std::vector<std::uint8_t> candidate_transport(
                        codeword_transport_bytes);
                    const auto bch_score = [&](
                        std::span<const std::uint8_t> candidate_bits) {
                        std::copy_n(
                            candidate_bits.data() + profile.full_parity_bits(),
                            profile.message_bits,
                            candidate_message.data());
                        return dtmb::core::dtmb_bch_descramble_message_bits(
                            candidate_message,
                            candidate_transport,
                            true,
                            codeword * codeword_transport_bytes * 8U);
                    };
                    const auto accept_solution = [&](
                        std::span<const std::uint8_t> bits) {
                        const auto candidate_bch = bch_score(bits);
                        return candidate_bch.unclean_blocks == 0
                            && ts_packets_syntax_valid(candidate_transport)
                            && ts_packets_fit_continuity_context(
                                candidate_transport,
                                continuity_context);
                    };
                    std::vector<BitValueConstraint> bit_constraints;
                    if (!retry_final_failed_syndrome_solve_ts_header_template_rules.empty()) {
                        bit_constraints = ts_explicit_header_template_constraints(
                            pending_frames[frame].frame_index,
                            codeword,
                            std::span<const TsHeaderTemplateRule>(
                                retry_final_failed_syndrome_solve_ts_header_template_rules),
                            codeword,
                            codeword_transport_bytes,
                            profile);
                    }
                    if (bit_constraints.empty()
                        && (retry_final_failed_syndrome_solve_ts_continuity_headers
                            || retry_final_failed_syndrome_solve_ts_continuity_header_templates)) {
                        const auto initial_bch = bch_score(retry_bits);
                        if (initial_bch.unclean_blocks == 0
                            && ts_packets_syntax_valid(candidate_transport)
                            && !ts_packets_fit_continuity_context(
                                candidate_transport,
                                continuity_context)) {
                            bit_constraints =
                                retry_final_failed_syndrome_solve_ts_continuity_header_templates
                                ? ts_continuity_header_template_constraints(
                                      candidate_transport,
                                      continuity_context,
                                      codeword,
                                      codeword_transport_bytes,
                                      profile)
                                : ts_continuity_header_constraints(
                                      candidate_transport,
                                      continuity_context,
                                      codeword,
                                      codeword_transport_bytes,
                                      profile);
                        }
                    }
                    if (!retry_result.converged
                        || (!bit_constraints.empty() && !accept_solution(retry_bits))) {
                        const auto solved = solve_final_failed_syndrome(
                            retry_bits,
                            retry_llr,
                            decoded_frame
                                .retry_final_failed_syndrome_solve_count[codeword],
                            retry_final_failed_syndrome_solve_minimize_cost_passes,
                            retry_final_failed_syndrome_solve_nullspace_search_count,
                            retry_final_failed_syndrome_solve_nullspace_depth,
                            std::span<const BitValueConstraint>(bit_constraints),
                            accept_solution,
                            graph,
                            profile);
                        retry_result.syndrome_weight = solved.final_syndrome_weight;
                        retry_result.converged = solved.solved;
                    } else if (!accept_solution(retry_bits)) {
                        return false;
                    }
                    if (!retry_result.converged) {
                        return false;
                    }
                    const auto retry_bch = bch_score(retry_bits);
                    if (retry_bch.unclean_blocks != 0
                        || !ts_packets_syntax_valid(candidate_transport)
                        || !ts_packets_fit_continuity_context(
                            candidate_transport,
                            continuity_context)) {
                        return false;
                    }

                    std::copy_n(
                        retry_bits.data(),
                        profile.full_codeword_bits(),
                        decoded_frame.decoded_bits.data() + full_offset);
                    std::copy_n(
                        candidate_message.data(),
                        profile.message_bits,
                        decoded_frame.message_bits.data()
                            + codeword * profile.message_bits);
                    std::copy_n(
                        candidate_transport.data(),
                        codeword_transport_bytes,
                        decoded_frame.transport_bytes.data() + transport_offset);
                    decoded_frame.results[codeword] = retry_result;
                    decoded_frame.bch_clean_codewords[codeword] = 1U;
                    decoded_frame.clean_codewords[codeword] = 1U;
                    decoded_frame.retry_bch_corrected_errors[codeword] =
                        retry_bch.corrected_errors;
                    decoded_frame.retry_ts_continuity_rejected[codeword] = 0U;
                    return true;
                };
                if (retry_require_ts_packet_continuity) {
                    for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                        const auto transport_offset = codeword * codeword_transport_bytes;
                        auto codeword_bytes = std::span<const std::uint8_t>(
                            decoded_frame.transport_bytes.data() + transport_offset,
                            codeword_transport_bytes);
                        const auto codeword_clean =
                            decoded_frame.results[codeword].converged
                            && codeword < decoded_frame.bch_clean_codewords.size()
                            && decoded_frame.bch_clean_codewords[codeword] != 0U
                            && ts_packets_syntax_valid(codeword_bytes);
                        if (!codeword_clean) {
                            continue;
                        }
                        if (decoded_frame.retry_selected[codeword] != 0U
                            && !ts_packets_fit_continuity_context(
                                codeword_bytes,
                                retry_ts_continuity_context)) {
                            if (try_continuity_context_syndrome_retry(
                                    codeword,
                                    retry_ts_continuity_context)) {
                                ++retry_ts_continuity_context_recovered_codeword_count;
                                codeword_bytes = std::span<const std::uint8_t>(
                                    decoded_frame.transport_bytes.data() + transport_offset,
                                    codeword_transport_bytes);
                                note_ts_packets_for_continuity_context(
                                    codeword_bytes,
                                    retry_ts_continuity_context);
                                continue;
                            }
                            decoded_frame.results[codeword].converged = false;
                            decoded_frame.clean_codewords[codeword] = 0U;
                            decoded_frame.retry_selected[codeword] = 0U;
                            decoded_frame.retry_ts_continuity_rejected[codeword] = 1U;
                            ++retry_ts_continuity_rejected_codeword_count;
                            continue;
                        }
                        note_ts_packets_for_continuity_context(
                            codeword_bytes,
                            retry_ts_continuity_context);
                    }
                }
                for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                    decoded_frame.clean_codewords[codeword] =
                        (decoded_frame.results[codeword].converged
                         && decoded_frame.bch_clean_codewords[codeword] != 0U)
                            ? 1U
                            : 0U;
                }
                const auto ldpc_clean = std::all_of(
                    decoded_frame.results.begin(),
                    decoded_frame.results.end(),
                    [](const auto& result) { return result.converged; });
                decoded_frame.frame_clean =
                    ldpc_clean && decoded_frame.bch.unclean_blocks == 0;
                clean_frame_count += decoded_frame.frame_clean ? 1U : 0U;
                for (const auto value : decoded_frame.clean_codewords) {
                    clean_codeword_count += value != 0U ? 1U : 0U;
                }
                bool emitted = false;
                std::size_t frame_discontinuity_marked_packets = 0;
                std::size_t frame_discontinuity_inserted_packets = 0;
                std::size_t frame_emitted_packets = 0;
                std::size_t frame_omitted_packets = 0;
                std::size_t frame_bch_clean_packet_salvage_emitted_packets = 0;
                if (emit_clean_codewords) {
                    for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                        const auto transport_offset = codeword * codeword_transport_bytes;
                        auto codeword_bytes = std::span<std::uint8_t>(
                            decoded_frame.transport_bytes.data() + transport_offset,
                            codeword_transport_bytes);
                        const auto force_emit_codeword =
                            pending_frames[frame].force_selected && emit_forced_codewords;
                        const auto bch_emit_codeword =
                            emit_bch_clean_codewords
                            && codeword < decoded_frame.bch_clean_codewords.size()
                            && decoded_frame.bch_clean_codewords[codeword] != 0U
                            && ts_packets_syntax_valid(codeword_bytes);
                        const auto bch_known_codeword =
                            codeword < decoded_frame.bch_clean_codewords.size()
                            && decoded_frame.bch_clean_codewords[codeword] != 0U
                            && ts_packets_syntax_valid(codeword_bytes);
                        const auto emit_whole_codeword =
                            decoded_frame.clean_codewords[codeword] != 0U
                            || bch_emit_codeword
                            || force_emit_codeword;
                        if (!emit_whole_codeword && !emit_bch_clean_packets) {
                            ++omitted_codeword_count;
                            frame_omitted_packets += packets_per_codeword;
                            if (bch_known_codeword) {
                                note_transport_gap_for_packets_or_unknown(codeword_bytes);
                            } else {
                                note_transport_gap();
                            }
                            continue;
                        }
                        if (emit_whole_codeword) {
                            note_continuity_guard_gaps(codeword_bytes);
                            const auto [marked, inserted] =
                                signal_pending_pid_discontinuities(
                                    codeword_bytes,
                                    pending_frames[frame].frame_index,
                                    codeword);
                            frame_discontinuity_marked_packets += marked;
                            discontinuity_marked_packets += marked;
                            frame_discontinuity_inserted_packets += inserted;
                            discontinuity_inserted_packets += inserted;
                            note_output_transport(
                                codeword_bytes,
                                "emitted",
                                pending_frames[frame].frame_index,
                                codeword,
                                0,
                                packets_per_codeword);
                            write_all(output, codeword_bytes);
                            output_byte_count += codeword_transport_bytes;
                            frame_emitted_packets += packets_per_codeword;
                            ++emitted_codeword_count;
                            note_emitted_transport(codeword_bytes);
                            emitted = true;
                            continue;
                        }

                        ++omitted_codeword_count;
                        for (std::size_t packet = 0;
                             packet < packets_per_codeword;
                             ++packet) {
                            bool packet_bch_clean = true;
                            const auto packet_first_block =
                                codeword * bch_blocks_per_codeword
                                + packet * bch_blocks_per_ts_packet;
                            const auto packet_last_block =
                                packet_first_block + bch_blocks_per_ts_packet;
                            for (std::size_t block = packet_first_block;
                                 block < packet_last_block;
                                 ++block) {
                                packet_bch_clean = packet_bch_clean
                                    && block < decoded_frame.bch.block_clean.size()
                                    && decoded_frame.bch.block_clean[block] != 0U;
                            }
                            auto packet_bytes = codeword_bytes.subspan(
                                packet * kTsPacketSize,
                                kTsPacketSize);
                            if (!packet_bch_clean || !ts_packet_syntax_valid(packet_bytes)) {
                                ++frame_omitted_packets;
                                if (packet_bch_clean) {
                                    note_transport_gap_for_packet_or_unknown(packet_bytes);
                                } else {
                                    note_transport_gap();
                                }
                                continue;
                            }
                            note_continuity_guard_gaps(packet_bytes);
                            const auto [marked, inserted] =
                                signal_pending_pid_discontinuities(
                                    packet_bytes,
                                    pending_frames[frame].frame_index,
                                    codeword);
                            frame_discontinuity_marked_packets += marked;
                            discontinuity_marked_packets += marked;
                            frame_discontinuity_inserted_packets += inserted;
                            discontinuity_inserted_packets += inserted;
                            note_output_transport(
                                packet_bytes,
                                "emitted_bch_clean_packet",
                                pending_frames[frame].frame_index,
                                codeword,
                                packet,
                                packets_per_codeword);
                            write_all(output, packet_bytes);
                            output_byte_count += kTsPacketSize;
                            ++frame_emitted_packets;
                            ++frame_bch_clean_packet_salvage_emitted_packets;
                            ++bch_clean_packet_salvage_emitted_packet_count;
                            note_emitted_transport(packet_bytes);
                            emitted = true;
                        }
                    }
                    emitted_frame_count += emitted ? 1U : 0U;
                    partial_emitted_frame_count +=
                        (emitted && !decoded_frame.frame_clean) ? 1U : 0U;
                    if (!emitted) {
                        ++omitted_frame_count;
                    }
                } else if (!clean_frames_only || decoded_frame.frame_clean) {
                    note_continuity_guard_gaps(decoded_frame.transport_bytes);
                    const auto [marked, inserted] =
                        signal_pending_pid_discontinuities(
                            decoded_frame.transport_bytes,
                            pending_frames[frame].frame_index,
                            std::optional<std::size_t>{});
                    frame_discontinuity_marked_packets += marked;
                    discontinuity_marked_packets += marked;
                    frame_discontinuity_inserted_packets += inserted;
                    discontinuity_inserted_packets += inserted;
                    for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                        const auto transport_offset = codeword * codeword_transport_bytes;
                        const auto codeword_bytes = std::span<const std::uint8_t>(
                            decoded_frame.transport_bytes.data() + transport_offset,
                            codeword_transport_bytes);
                        note_output_transport(
                            codeword_bytes,
                            "emitted",
                            pending_frames[frame].frame_index,
                            codeword,
                            0,
                            packets_per_codeword);
                    }
                    write_all(output, decoded_frame.transport_bytes);
                    output_byte_count += decoded_frame.transport_bytes.size();
                    frame_emitted_packets +=
                        decoded_frame.transport_bytes.size() / kTsPacketSize;
                    ++emitted_frame_count;
                    emitted_codeword_count += codewords_per_frame;
                    note_emitted_transport(decoded_frame.transport_bytes);
                    emitted = true;
                } else {
                    omitted_codeword_count += codewords_per_frame;
                    frame_omitted_packets +=
                        codewords_per_frame * codeword_transport_bytes / kTsPacketSize;
                    ++omitted_frame_count;
                    bool bch_known_frame = ts_packets_syntax_valid(decoded_frame.transport_bytes);
                    for (std::size_t codeword = 0;
                         codeword < decoded_frame.bch_clean_codewords.size();
                         ++codeword) {
                        bch_known_frame = bch_known_frame
                            && decoded_frame.bch_clean_codewords[codeword] != 0U;
                    }
                    if (bch_known_frame) {
                        note_transport_gap_for_packets_or_unknown(decoded_frame.transport_bytes);
                    } else {
                        note_transport_gap();
                    }
                }
                write_frame_diagnostics(
                    pending_frames[frame],
                    &decoded_frame,
                    emitted,
                    frame_discontinuity_marked_packets,
                    frame_discontinuity_inserted_packets,
                    frame_emitted_packets,
                    frame_omitted_packets,
                    frame_bch_clean_packet_salvage_emitted_packets);

                if (fail_on_unclean_frame && !decoded_frame.frame_clean) {
                    std::cerr << "strict_failed_frame="
                              << pending_frames[frame].frame_index << '\n';
                    throw std::runtime_error(
                        "strict decode stopped at the first unclean FEC frame");
                }

                bch_corrected_errors += decoded_frame.bch.corrected_errors;
                bch_unclean_blocks += decoded_frame.bch.unclean_blocks;
                for (const auto weight : decoded_frame.initial_syndrome_weights) {
                    initial_syndrome_weight += weight;
                }
                for (const auto rejected : decoded_frame.early_rejected) {
                    early_rejected_count += rejected != 0U ? 1U : 0U;
                }
                for (const auto attempts : decoded_frame.retry_attempts) {
                    retry_attempt_count += attempts;
                }
                for (std::size_t codeword = 0;
                     codeword < decoded_frame.retry_selected.size();
                     ++codeword) {
                    if (decoded_frame.retry_selected[codeword] == 0U) {
                        continue;
                    }
                    ++retry_recovered_codeword_count;
                    layered_retry_recovered_codeword_count +=
                        decoded_frame.retry_layered[codeword] != 0U ? 1U : 0U;
                    qam_plane_retry_recovered_codeword_count +=
                        decoded_frame.retry_qam_plane[codeword] != kNoRetryQamPlane ? 1U : 0U;
                    final_failed_flip_retry_recovered_codeword_count +=
                        decoded_frame.retry_final_failed_flip_count[codeword] != 0U ? 1U : 0U;
                    final_failed_greedy_flip_retry_recovered_codeword_count +=
                        decoded_frame.retry_final_failed_greedy_flip_count[codeword] != 0U
                            ? 1U
                            : 0U;
                    final_failed_syndrome_solve_retry_recovered_codeword_count +=
                        decoded_frame.retry_final_failed_syndrome_solve_count[codeword] != 0U
                            ? 1U
                            : 0U;
                    const auto recovered_by_branch_window =
                        decoded_frame.retry_branch_window_first[codeword]
                            != kNoRetryBranchWindow;
                    const auto recovered_by_group =
                        !decoded_frame.retry_branch_window_group[codeword].empty();
                    branch_window_retry_recovered_codeword_count +=
                        (recovered_by_branch_window || recovered_by_group) ? 1U : 0U;
                    branch_window_group_retry_recovered_codeword_count +=
                        recovered_by_group ? 1U : 0U;
                }
                for (const auto& result : decoded_frame.results) {
                    converged_count += result.converged ? 1U : 0U;
                    iteration_count += result.iterations;
                    final_syndrome_weight += result.syndrome_weight;
                }
                ++soft_decoded_frames;
            }
            pending_frames.clear();
            const auto batch_finished = std::chrono::steady_clock::now();
            const auto elapsed_ms = [](auto first, auto last) {
                return std::chrono::duration<double, std::milli>(last - first).count();
            };
            batch_prepare_ms += elapsed_ms(batch_started, batch_prepared);
            batch_baseline_ms += elapsed_ms(batch_prepared, batch_baseline_finished);
            batch_snapshot_ms +=
                elapsed_ms(batch_baseline_finished, batch_snapshot_finished);
            batch_retry_ms += elapsed_ms(batch_snapshot_finished, batch_retry_finished);
            batch_worker_ms += elapsed_ms(batch_retry_finished, batch_workers_finished);
            batch_finalize_ms += elapsed_ms(batch_workers_finished, batch_finished);
            batch_total_ms += elapsed_ms(batch_started, batch_finished);
        };

        const auto push_selected_frame = [&](
            std::size_t frame_index,
            std::span<const float> frame_llr,
            std::size_t clean_syndrome_weight,
            bool force_selected,
            bool force_omitted,
            bool mode2_source_frame_selected,
            bool hard_h_gap_filled) {
            PendingFecFrame frame;
            frame.selected = true;
            frame.force_selected = force_selected;
            frame.force_omitted = force_omitted;
            frame.mode2_source_frame_selected = mode2_source_frame_selected;
            frame.hard_h_gap_filled = hard_h_gap_filled;
            frame.frame_index = frame_index;
            frame.clean_syndrome_weight = clean_syndrome_weight;
            frame.transmitted_llr.assign(frame_llr.begin(), frame_llr.end());
            for (const auto& rule : codeword_llr_scale_rules) {
                if (frame_index < rule.first_frame || frame_index > rule.last_frame) {
                    continue;
                }
                const auto offset = rule.codeword * profile.transmitted_bits;
                auto codeword_llr = std::span<float>(
                    frame.transmitted_llr.data() + offset,
                    profile.transmitted_bits);
                for (auto& llr : codeword_llr) {
                    llr *= rule.scale;
                }
            }
            for (const auto& rule : variable_llr_scale_rules) {
                if (frame_index < rule.first_frame || frame_index > rule.last_frame) {
                    continue;
                }
                for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                    const auto offset = codeword * profile.transmitted_bits + rule.variable;
                    frame.transmitted_llr[offset] *= rule.scale;
                }
            }
            for (const auto& rule : variable_mod_llr_scale_rules) {
                if (frame_index < rule.first_frame || frame_index > rule.last_frame) {
                    continue;
                }
                for (std::size_t codeword = 0; codeword < codewords_per_frame; ++codeword) {
                    const auto codeword_offset = codeword * profile.transmitted_bits;
                    for (std::size_t variable = rule.remainder;
                         variable < profile.transmitted_bits;
                         variable += rule.modulus) {
                        frame.transmitted_llr[codeword_offset + variable] *= rule.scale;
                    }
                }
            }
            pending_frames.push_back(std::move(frame));
            if (pending_frames.size() >= decode_batch_frames) {
                flush_pending_frames();
            }
        };

        const auto push_omitted_frame = [&](
            std::size_t frame_index,
            std::size_t clean_syndrome_weight,
            bool force_omitted,
            bool mode2_source_frame_selected,
            bool hard_h_gap_filled,
            std::span<const float> frame_llr) {
            PendingFecFrame frame;
            frame.frame_index = frame_index;
            frame.clean_syndrome_weight = clean_syndrome_weight;
            frame.force_omitted = force_omitted;
            frame.mode2_source_frame_selected = mode2_source_frame_selected;
            frame.hard_h_gap_filled = hard_h_gap_filled;
            if (frame_diagnostics != nullptr) {
                frame.transmitted_llr.assign(frame_llr.begin(), frame_llr.end());
            }
            pending_frames.push_back(std::move(frame));
            if (pending_frames.size() >= decode_batch_frames) {
                flush_pending_frames();
            }
        };

        const auto consume_buffered_frame = [&](BufferedFecFrame&& frame) {
            if (frame.selected) {
                ++hard_h_gate_selected_frames;
                forced_selected_frames += frame.force_selected ? 1U : 0U;
                mode2_source_frame_selected_frames +=
                    frame.mode2_source_frame_selected ? 1U : 0U;
                push_selected_frame(
                    frame.frame_index,
                    frame.transmitted_llr,
                    frame.clean_syndrome_weight,
                    frame.force_selected,
                    frame.force_omitted,
                    frame.mode2_source_frame_selected,
                    frame.hard_h_gap_filled);
            } else {
                ++hard_h_gate_skipped_frames;
                push_omitted_frame(
                    frame.frame_index,
                    frame.clean_syndrome_weight,
                    frame.force_omitted,
                    frame.mode2_source_frame_selected,
                    frame.hard_h_gap_filled,
                    frame.transmitted_llr);
            }
        };

        std::deque<BufferedFecFrame> hard_h_gate_frames;
        std::deque<BufferedFecFrame> hard_h_gate_output_frames;
        bool previous_gate_anchor_selected = false;
        const auto hard_h_gate_window_frames = hard_h_gate_enabled
            ? *hard_h_gate_window_codewords / codewords_per_frame
            : std::size_t{0};
        const auto consume_gate_output_frame = [&](BufferedFecFrame&& frame) {
            const auto gate_anchor_selected = frame.selected && !frame.hard_h_gap_filled;
            consume_buffered_frame(std::move(frame));
            previous_gate_anchor_selected = gate_anchor_selected;
        };
        const auto mode2_source_selector_allows = [&](const BufferedFecFrame& frame) {
            return !has_select_mode2_source_frame_selection
                || frame.force_selected
                || frame.mode2_source_frame_selected;
        };
        const auto flush_decidable_gate_output = [&] {
            while (!hard_h_gate_output_frames.empty()) {
                if (hard_h_gate_output_frames.front().selected) {
                    auto oldest = std::move(hard_h_gate_output_frames.front());
                    hard_h_gate_output_frames.pop_front();
                    consume_gate_output_frame(std::move(oldest));
                    continue;
                }
                if (!previous_gate_anchor_selected) {
                    auto oldest = std::move(hard_h_gate_output_frames.front());
                    hard_h_gate_output_frames.pop_front();
                    consume_gate_output_frame(std::move(oldest));
                    continue;
                }
                if (!mode2_source_selector_allows(hard_h_gate_output_frames.front())) {
                    auto oldest = std::move(hard_h_gate_output_frames.front());
                    hard_h_gate_output_frames.pop_front();
                    consume_gate_output_frame(std::move(oldest));
                    continue;
                }

                std::size_t gap_frames = 0;
                bool bounded_by_selected_frame = false;
                for (const auto& frame : hard_h_gate_output_frames) {
                    if (frame.force_omitted) {
                        break;
                    }
                    if (!mode2_source_selector_allows(frame)) {
                        break;
                    }
                    if (frame.selected) {
                        bounded_by_selected_frame = true;
                        break;
                    }
                    ++gap_frames;
                    if (gap_frames > hard_h_gate_fill_max_gap_frames) {
                        break;
                    }
                }
                if (bounded_by_selected_frame
                    && gap_frames <= hard_h_gate_fill_max_gap_frames) {
                    for (std::size_t index = 0; index < gap_frames; ++index) {
                        auto& frame = hard_h_gate_output_frames[index];
                        if (!frame.selected && !frame.force_omitted
                            && mode2_source_selector_allows(frame)) {
                            frame.selected = true;
                            frame.hard_h_gap_filled = true;
                            ++hard_h_gate_gap_filled_frames;
                        }
                    }
                    continue;
                }
                if (gap_frames > hard_h_gate_fill_max_gap_frames) {
                    auto oldest = std::move(hard_h_gate_output_frames.front());
                    hard_h_gate_output_frames.pop_front();
                    consume_gate_output_frame(std::move(oldest));
                    continue;
                }
                break;
            }
        };
        const auto queue_gate_output_frame = [&](BufferedFecFrame&& frame) {
            if (hard_h_gate_fill_max_gap_frames == 0) {
                consume_buffered_frame(std::move(frame));
                return;
            }
            hard_h_gate_output_frames.push_back(std::move(frame));
            flush_decidable_gate_output();
        };

        while (true) {
            input.read(reinterpret_cast<char*>(transmitted_llr.data()), frame_bytes);
            const auto bytes_read = input.gcount();
            if (bytes_read == 0) {
                if (input.bad()) {
                    throw std::runtime_error("failed to read LLR input stream");
                }
                break;
            }
            if (bytes_read != frame_bytes) {
                throw std::runtime_error("LLR input ended before a complete DTMB FEC frame");
            }
            if (!std::all_of(
                    transmitted_llr.begin(),
                    transmitted_llr.end(),
                    [](float llr) { return std::isfinite(llr); })) {
                throw std::runtime_error("LLR input contains a nonfinite value");
            }

            ++frame_count;
            codeword_count += codewords_per_frame;
            if (frame_index_offset > std::numeric_limits<std::size_t>::max() - frame_count) {
                throw std::runtime_error("frame index offset overflow");
            }
            const auto frame_index = frame_index_offset + frame_count;
            if (!hard_h_gate_enabled) {
                const auto force_selected =
                    frame_in_ranges(frame_index, std::span<const FrameRange>(force_frame_ranges));
                const auto force_omitted =
                    frame_in_ranges(frame_index, std::span<const FrameRange>(omit_frame_ranges));
                const auto select_source_frame_windows =
                    active_mode2_source_frame_windows(
                        frame_index,
                        std::span<const FrameRange>(select_mode2_source_frame_windows),
                        std::span<const SourceFrameWindowRule>(
                            select_mode2_source_frame_window_rules));
                const auto mode2_source_frame_selected =
                    !mode2_source_frame_branch_windows(
                         frame_index,
                         std::span<const FrameRange>(select_source_frame_windows))
                         .empty();
                const auto default_selected =
                    force_frame_ranges.empty()
                    && !has_select_mode2_source_frame_selection;
                const auto selected =
                    !force_omitted
                    && (default_selected || force_selected
                        || mode2_source_frame_selected);
                if (selected) {
                    forced_selected_frames += force_selected ? 1U : 0U;
                    mode2_source_frame_selected_frames +=
                        mode2_source_frame_selected ? 1U : 0U;
                    push_selected_frame(
                        frame_index,
                        transmitted_llr,
                        0U,
                        force_selected,
                        force_omitted,
                        mode2_source_frame_selected,
                        false);
                } else {
                    push_omitted_frame(
                        frame_index,
                        0U,
                        force_omitted,
                        false,
                        false,
                        transmitted_llr);
                }
                continue;
            }

            BufferedFecFrame frame;
            frame.frame_index = frame_index;
            frame.force_selected =
                frame_in_ranges(frame_index, std::span<const FrameRange>(force_frame_ranges));
            const auto force_omitted =
                frame_in_ranges(frame_index, std::span<const FrameRange>(omit_frame_ranges));
            const auto select_source_frame_windows =
                active_mode2_source_frame_windows(
                    frame_index,
                    std::span<const FrameRange>(select_mode2_source_frame_windows),
                    std::span<const SourceFrameWindowRule>(
                        select_mode2_source_frame_window_rules));
            frame.mode2_source_frame_selected =
                !mode2_source_frame_branch_windows(
                     frame_index,
                     std::span<const FrameRange>(select_source_frame_windows))
                     .empty();
            if (force_omitted) {
                frame.force_omitted = true;
                frame.force_selected = false;
                frame.mode2_source_frame_selected = false;
            }
            frame.clean_syndrome_weight = clean_syndrome_weight(
                transmitted_llr,
                profile.transmitted_bits,
                *hard_h_gate_graph);
            frame.transmitted_llr = transmitted_llr;
            hard_h_gate_frames.push_back(std::move(frame));

            if (hard_h_gate_frames.size() == hard_h_gate_window_frames) {
                const auto total_weight = std::accumulate(
                    hard_h_gate_frames.begin(),
                    hard_h_gate_frames.end(),
                    std::size_t{0},
                    [](std::size_t total, const BufferedFecFrame& value) {
                        return total + value.clean_syndrome_weight;
                    });
                const auto denominator =
                    static_cast<double>(*hard_h_gate_window_codewords)
                    * static_cast<double>(hard_h_gate_graph->check_count());
                const auto ratio = static_cast<double>(total_weight)
                    / denominator;
                ++hard_h_gate_scored_windows;
                if (ratio <= *hard_h_gate_threshold) {
                    ++hard_h_gate_pass_windows;
                    for (auto& buffered : hard_h_gate_frames) {
                        buffered.selected = true;
                    }
                }
                for (auto& buffered : hard_h_gate_frames) {
                    buffered.selected = buffered.selected
                        || buffered.force_selected
                        || buffered.mode2_source_frame_selected;
                    if (!mode2_source_selector_allows(buffered)) {
                        buffered.selected = false;
                        buffered.hard_h_gap_filled = false;
                    }
                    if (frame_in_ranges(
                            buffered.frame_index,
                            std::span<const FrameRange>(omit_frame_ranges))) {
                        buffered.selected = false;
                        buffered.force_selected = false;
                        buffered.force_omitted = true;
                        buffered.mode2_source_frame_selected = false;
                        buffered.hard_h_gap_filled = false;
                    }
                }
                auto oldest = std::move(hard_h_gate_frames.front());
                hard_h_gate_frames.pop_front();
                queue_gate_output_frame(std::move(oldest));
            }
        }
        for (auto& buffered : hard_h_gate_frames) {
            buffered.selected = buffered.selected
                || buffered.force_selected
                || buffered.mode2_source_frame_selected;
            if (!mode2_source_selector_allows(buffered)) {
                buffered.selected = false;
                buffered.hard_h_gap_filled = false;
            }
            if (frame_in_ranges(
                    buffered.frame_index,
                    std::span<const FrameRange>(omit_frame_ranges))) {
                buffered.selected = false;
                buffered.force_selected = false;
                buffered.force_omitted = true;
                buffered.mode2_source_frame_selected = false;
                buffered.hard_h_gap_filled = false;
            }
        }
        while (!hard_h_gate_frames.empty()) {
            auto oldest = std::move(hard_h_gate_frames.front());
            hard_h_gate_frames.pop_front();
            queue_gate_output_frame(std::move(oldest));
        }
        while (!hard_h_gate_output_frames.empty()) {
            auto oldest = std::move(hard_h_gate_output_frames.front());
            hard_h_gate_output_frames.pop_front();
            consume_gate_output_frame(std::move(oldest));
        }
        flush_pending_frames();

        std::cerr << "frames=" << frame_count << '\n'
                  << "frame_index_offset=" << frame_index_offset << '\n'
                  << "codewords=" << codeword_count << '\n'
                  << "worker_count=" << worker_count << '\n'
                  << "decode_batch_frames=" << decode_batch_frames << '\n'
                  << "max_iterations=" << decode_options.max_iterations << '\n'
                  << "retry_max_iterations="
                  << retry_decode_options.max_iterations << '\n'
                  << "llr_clip=";
        if (primary_llr_clip.has_value()) {
            std::cerr << *primary_llr_clip;
        } else {
            std::cerr << "off";
        }
        std::cerr << '\n'
                  << "ldpc_accel=" << ldpc_accel_name(ldpc_accel) << '\n'
                  << "cuda_ldpc_codewords=" << cuda_ldpc_codewords << '\n'
                  << "cuda_ldpc_kernel_ms=" << cuda_ldpc_kernel_ms << '\n'
                  << "cuda_ldpc_h2d_ms=" << cuda_ldpc_h2d_ms << '\n'
                  << "cuda_ldpc_d2h_ms=" << cuda_ldpc_d2h_ms << '\n'
                  << "cuda_ldpc_total_ms=" << cuda_ldpc_total_ms << '\n'
                  << "cuda_ldpc_cpu_fallback_codewords="
                  << cuda_ldpc_cpu_fallback_codewords << '\n'
                  << "cuda_retry_clips=" << (cuda_retry_clips ? "true" : "false") << '\n'
                  << "cuda_retry_codewords=" << cuda_retry_codewords << '\n'
                  << "cuda_retry_kernel_ms=" << cuda_retry_kernel_ms << '\n'
                  << "cuda_retry_h2d_ms=" << cuda_retry_h2d_ms << '\n'
                  << "cuda_retry_d2h_ms=" << cuda_retry_d2h_ms << '\n'
                  << "cuda_retry_total_ms=" << cuda_retry_total_ms << '\n'
                  << "batch_prepare_ms=" << batch_prepare_ms << '\n'
                  << "batch_baseline_ms=" << batch_baseline_ms << '\n'
                  << "batch_snapshot_ms=" << batch_snapshot_ms << '\n'
                  << "batch_retry_ms=" << batch_retry_ms << '\n'
                  << "batch_worker_ms=" << batch_worker_ms << '\n'
                  << "batch_finalize_ms=" << batch_finalize_ms << '\n'
                  << "batch_total_ms=" << batch_total_ms << '\n'
                  << "converged_codewords=" << converged_count << '\n'
                  << "ldpc_iterations=" << iteration_count << '\n'
                  << "early_syndrome_reject_ratio="
                  << (early_syndrome_reject_ratio >= 0.0F
                          ? std::to_string(early_syndrome_reject_ratio)
                          : "off")
                  << '\n'
                  << "initial_syndrome_weight=" << initial_syndrome_weight << '\n'
                  << "early_rejected_codewords=" << early_rejected_count << '\n'
                  << "final_syndrome_weight=" << final_syndrome_weight << '\n'
                  << "bch_corrected_errors=" << bch_corrected_errors << '\n'
                  << "bch_unclean_blocks=" << bch_unclean_blocks << '\n'
                  << "clean_frames=" << clean_frame_count << '\n'
                  << "emitted_frames=" << emitted_frame_count << '\n'
                  << "partial_emitted_frames=" << partial_emitted_frame_count << '\n'
                  << "omitted_frames=" << omitted_frame_count << '\n'
                  << "clean_frames_only=" << (clean_frames_only ? "true" : "false") << '\n'
                  << "fail_on_unclean_frame="
                  << (fail_on_unclean_frame ? "true" : "false") << '\n'
                  << "emit_clean_codewords=" << (emit_clean_codewords ? "true" : "false") << '\n'
                  << "emit_bch_clean_codewords="
                  << (emit_bch_clean_codewords ? "true" : "false") << '\n'
                  << "emit_bch_clean_packets="
                  << (emit_bch_clean_packets ? "true" : "false") << '\n'
                  << "clean_codewords=" << clean_codeword_count << '\n'
                  << "bch_clean_payload_codewords="
                  << bch_clean_payload_codeword_count << '\n'
                  << "bch_clean_payload_packets="
                  << bch_clean_payload_packet_count << '\n'
                  << "bch_clean_packet_salvage_emitted_packets="
                  << bch_clean_packet_salvage_emitted_packet_count << '\n'
                  << "emitted_codewords=" << emitted_codeword_count << '\n'
                  << "omitted_codewords=" << omitted_codeword_count << '\n'
                  << "mark_discontinuities=" << (mark_discontinuities ? "true" : "false") << '\n'
                  << "discontinuity_marked_packets=" << discontinuity_marked_packets << '\n'
                  << "insert_discontinuity_packets="
                  << (insert_discontinuity_packets ? "true" : "false") << '\n'
                  << "discontinuity_inserted_packets=" << discontinuity_inserted_packets << '\n'
                  << "continuity_guard_pending_pids="
                  << continuity_guard_pending_pids << '\n'
                  << "hard_h_gate_enabled=" << (hard_h_gate_enabled ? "true" : "false") << '\n'
                  << "hard_h_gate_window_codewords="
                  << (hard_h_gate_enabled
                          ? std::to_string(*hard_h_gate_window_codewords)
                          : "off")
                  << '\n'
                  << "hard_h_gate_step_codewords="
                  << (hard_h_gate_enabled
                          ? std::to_string(*hard_h_gate_step_codewords)
                          : "off")
                  << '\n'
                  << "hard_h_gate_threshold="
                  << (hard_h_gate_enabled
                          ? std::to_string(*hard_h_gate_threshold)
                          : "off")
                  << '\n'
                  << "hard_h_gate_scored_windows=" << hard_h_gate_scored_windows << '\n'
                  << "hard_h_gate_pass_windows=" << hard_h_gate_pass_windows << '\n'
                  << "hard_h_gate_selected_frames=" << hard_h_gate_selected_frames << '\n'
                  << "forced_selected_frames=" << forced_selected_frames << '\n'
                  << "mode2_source_frame_selected_frames="
                  << mode2_source_frame_selected_frames << '\n'
                  << "hard_h_gate_gap_filled_frames="
                  << hard_h_gate_gap_filled_frames << '\n'
                  << "codeword_llr_scale_rule_count="
                  << codeword_llr_scale_rules.size() << '\n'
                  << "variable_llr_scale_rule_count="
                  << variable_llr_scale_rules.size() << '\n'
                  << "variable_mod_llr_scale_rule_count="
                  << variable_mod_llr_scale_rules.size() << '\n'
                  << "failed_variable_diagnostics_top="
                  << failed_variable_diagnostics_top << '\n'
                  << "retry_enabled=" << (retry_enabled ? "true" : "false") << '\n'
                  << "retry_llr_clip_count=" << retry_llr_clips.size() << '\n'
                  << "retry_llr_scale_count=" << retry_llr_scales.size() << '\n'
                  << "retry_weak_erase_fraction_count="
                  << retry_weak_erase_fractions.size() << '\n'
                  << "retry_attenuation_count=" << retry_attenuations.size() << '\n'
                  << "retry_unsatisfied_erase_count_count="
                  << retry_unsatisfied_erase_counts.size() << '\n'
                  << "retry_final_failed_erase_count_count="
                  << retry_final_failed_erase_counts.size() << '\n'
                  << "retry_final_failed_flip_count_count="
                  << retry_final_failed_flip_counts.size() << '\n'
                  << "retry_final_failed_greedy_flip_count_count="
                  << retry_final_failed_greedy_flip_counts.size() << '\n'
                  << "retry_final_failed_syndrome_solve_count_count="
                  << retry_final_failed_syndrome_solve_counts.size() << '\n'
                  << "retry_final_failed_syndrome_solve_minimize_cost_passes="
                  << retry_final_failed_syndrome_solve_minimize_cost_passes << '\n'
                  << "retry_final_failed_syndrome_solve_nullspace_search_count="
                  << retry_final_failed_syndrome_solve_nullspace_search_count << '\n'
                  << "retry_final_failed_syndrome_solve_nullspace_depth="
                  << retry_final_failed_syndrome_solve_nullspace_depth << '\n'
                  << "retry_final_failed_syndrome_solve_ts_continuity_headers="
                  << (retry_final_failed_syndrome_solve_ts_continuity_headers ? "true" : "false")
                  << '\n'
                  << "retry_final_failed_syndrome_solve_ts_continuity_header_templates="
                  << (retry_final_failed_syndrome_solve_ts_continuity_header_templates
                          ? "true"
                          : "false")
                  << '\n'
                  << "retry_final_failed_syndrome_solve_ts_header_template_rule_count="
                  << retry_final_failed_syndrome_solve_ts_header_template_rules.size()
                  << '\n'
                  << "retry_qam_plane_count="
                  << retry_qam_planes.size() << '\n'
                  << "retry_variable_range_count="
                  << retry_variable_ranges.size() << '\n'
                  << "retry_branch_window_count="
                  << retry_branch_windows.size() << '\n'
                  << "retry_branch_window_group_count="
                  << retry_branch_window_groups.size() << '\n'
                  << "retry_branch_window_frame_range_count="
                  << retry_branch_window_frame_ranges.size() << '\n'
                  << "retry_branch_window_rule_count="
                  << retry_branch_window_rules.size() << '\n'
                  << "retry_mode2_source_frame_window_count="
                  << retry_mode2_source_frame_windows.size() << '\n'
                  << "retry_mode2_source_frame_window_rule_count="
                  << retry_mode2_source_frame_window_rules.size() << '\n'
                  << "select_mode2_source_frame_window_count="
                  << select_mode2_source_frame_windows.size() << '\n'
                  << "select_mode2_source_frame_window_rule_count="
                  << select_mode2_source_frame_window_rules.size() << '\n'
                  << "retry_layered=" << (retry_layered ? "true" : "false") << '\n'
                  << "retry_require_ts_packet_syntax="
                  << (retry_require_ts_packet_syntax ? "true" : "false") << '\n'
                  << "retry_require_ts_packet_continuity="
                  << (retry_require_ts_packet_continuity ? "true" : "false") << '\n'
                  << "retry_attempts=" << retry_attempt_count << '\n'
                  << "retry_recovered_codewords="
                  << retry_recovered_codeword_count << '\n'
                  << "layered_retry_recovered_codewords="
                  << layered_retry_recovered_codeword_count << '\n'
                  << "qam_plane_retry_recovered_codewords="
                  << qam_plane_retry_recovered_codeword_count << '\n'
                  << "final_failed_flip_retry_recovered_codewords="
                  << final_failed_flip_retry_recovered_codeword_count << '\n'
                  << "final_failed_greedy_flip_retry_recovered_codewords="
                  << final_failed_greedy_flip_retry_recovered_codeword_count << '\n'
                  << "final_failed_syndrome_solve_retry_recovered_codewords="
                  << final_failed_syndrome_solve_retry_recovered_codeword_count << '\n'
                  << "branch_window_retry_recovered_codewords="
                  << branch_window_retry_recovered_codeword_count << '\n'
                  << "branch_window_group_retry_recovered_codewords="
                  << branch_window_group_retry_recovered_codeword_count << '\n'
                  << "retry_ts_continuity_context_recovered_codewords="
                  << retry_ts_continuity_context_recovered_codeword_count << '\n'
                  << "retry_ts_continuity_rejected_codewords="
                  << retry_ts_continuity_rejected_codeword_count << '\n'
                  << "hard_h_gate_skipped_frames=" << hard_h_gate_skipped_frames << '\n'
                  << "soft_decoded_frames=" << soft_decoded_frames << '\n'
                  << "output_packets=" << output_packet_count << '\n'
                  << "output_bytes=" << output_byte_count << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_ldpc_bch_decode: " << exc.what() << '\n';
        return 1;
    }
}
