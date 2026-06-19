#include "dtmb/core.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFloatsPerSymbol = 2;
constexpr std::size_t kLlrsPerSymbol = 6;
constexpr std::array<float, 8> kQam64Levels{-7.0F, -5.0F, -3.0F, -1.0F, 1.0F, 3.0F, 5.0F, 7.0F};

struct BranchGainOptions {
    std::vector<std::size_t> branches;
    float reliability_threshold = 0.55F;
    std::size_t min_symbols = 32;
    std::string diagnostics_path;
    std::vector<std::pair<std::size_t, std::size_t>> source_frame_ranges;
};

struct BranchGainStats {
    std::size_t branch = 0;
    std::size_t chunks_seen = 0;
    std::size_t applied_chunks = 0;
    std::size_t skipped_chunks = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    double gain_real_sum = 0.0;
    double gain_imag_sum = 0.0;
    double gain_abs_sum = 0.0;
    double reliable_relative_error_sum = 0.0;
    double corrected_relative_error_sum = 0.0;
};

struct BranchGainFrameStats {
    std::size_t branch = 0;
    std::size_t source_frame = 0;
    std::size_t symbols = 0;
    std::size_t reliable_symbols = 0;
    std::complex<double> numerator{0.0, 0.0};
    double denominator = 0.0;
    double relative_error_sum = 0.0;
    double reliable_relative_error_sum = 0.0;
};

struct SourceFrameLlrScaleRule {
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float scale = 1.0F;
};

struct SourceFrameLlrScaleStats {
    std::size_t rule_index = 0;
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float scale = 1.0F;
    std::size_t scaled_symbols = 0;
};

struct SourceCarrierLlrScaleRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float scale = 1.0F;
};

struct SourceCarrierLlrScaleStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float scale = 1.0F;
    std::size_t scaled_symbols = 0;
};

struct SourceCarrierSymbolGainRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceCarrierSymbolGainStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t chunks_seen = 0;
    std::size_t applied_chunks = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    double gain_real_sum = 0.0;
    double gain_imag_sum = 0.0;
    double gain_abs_sum = 0.0;
};

struct SourceFrameSymbolGainRule {
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceFrameSymbolGainStats {
    std::size_t rule_index = 0;
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t chunks_seen = 0;
    std::size_t applied_chunks = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    double gain_real_sum = 0.0;
    double gain_imag_sum = 0.0;
    double gain_abs_sum = 0.0;
};

struct SourceFrameAxisAffineRule {
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceFrameAxisAffineStats {
    std::size_t rule_index = 0;
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t chunks_seen = 0;
    std::size_t applied_chunks = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    double real_scale_sum = 0.0;
    double real_bias_sum = 0.0;
    double imag_scale_sum = 0.0;
    double imag_bias_sum = 0.0;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --mode mode1|mode2 [--phase N] [--keep-latency]"
        << " [--workers N] [--min-parallel-symbols N] [--chunk-symbols N]"
        << " [--branch-gain-branches CSV]"
        << " [--branch-gain-reliability-threshold X]"
        << " [--branch-gain-min-symbols N]"
        << " [--branch-gain-source-frame-range FIRST:LAST]"
        << " [--branch-gain-diagnostics-out PATH]"
        << " [--source-frame-symbol-gain BRANCH:FIRST:LAST]"
        << " [--source-carrier-symbol-gain CARRIER:FIRST:LAST]"
        << " [--source-frame-axis-affine BRANCH:FIRST:LAST]"
        << " [--source-frame-llr-scale BRANCH:FIRST:LAST:SCALE]"
        << " [--source-carrier-llr-scale CARRIER:FIRST:LAST:SCALE]"
        << " [--noise-variance X] [input.cf32|-] [output.llr.f32|-]\n";
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

std::vector<std::size_t> parse_branch_list(const std::string& text) {
    std::vector<std::size_t> branches;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        branches.push_back(parse_size(item, "branch"));
    }
    std::sort(branches.begin(), branches.end());
    branches.erase(std::unique(branches.begin(), branches.end()), branches.end());
    return branches;
}

std::pair<std::size_t, std::size_t> parse_range(const std::string& text, const char* field) {
    const auto separator = text.find(':');
    if (separator == std::string::npos) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    auto first = parse_size(text.substr(0, separator), field);
    auto last = parse_size(text.substr(separator + 1U), field);
    if (last < first) {
        throw std::invalid_argument(std::string(field) + " end is before start: " + text);
    }
    return {first, last};
}

SourceFrameLlrScaleRule parse_llr_scale_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source frame LLR scale rule: " + text);
    }
    SourceFrameLlrScaleRule rule;
    rule.branch = parse_size(parts[0], "source frame LLR scale branch");
    rule.first_frame = parse_size(parts[1], "source frame LLR scale first frame");
    rule.last_frame = parse_size(parts[2], "source frame LLR scale last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument("source frame LLR scale end is before start: " + text);
    }
    rule.scale = parse_float(parts[3], "source frame LLR scale");
    if (!std::isfinite(rule.scale) || rule.scale < 0.0F) {
        throw std::invalid_argument("source frame LLR scale must be finite and non-negative");
    }
    return rule;
}

SourceCarrierLlrScaleRule parse_carrier_llr_scale_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier LLR scale rule: " + text);
    }
    SourceCarrierLlrScaleRule rule;
    rule.carrier = parse_size(parts[0], "source carrier LLR scale carrier");
    rule.first_frame = parse_size(parts[1], "source carrier LLR scale first frame");
    rule.last_frame = parse_size(parts[2], "source carrier LLR scale last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument("source carrier LLR scale end is before start: " + text);
    }
    rule.scale = parse_float(parts[3], "source carrier LLR scale");
    if (!std::isfinite(rule.scale) || rule.scale < 0.0F) {
        throw std::invalid_argument("source carrier LLR scale must be finite and non-negative");
    }
    return rule;
}

SourceFrameSymbolGainRule parse_symbol_gain_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source frame symbol gain rule: " + text);
    }
    SourceFrameSymbolGainRule rule;
    rule.branch = parse_size(parts[0], "source frame symbol gain branch");
    rule.first_frame = parse_size(parts[1], "source frame symbol gain first frame");
    rule.last_frame = parse_size(parts[2], "source frame symbol gain last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument("source frame symbol gain end is before start: " + text);
    }
    return rule;
}

SourceCarrierSymbolGainRule parse_carrier_symbol_gain_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source carrier symbol gain rule: " + text);
    }
    SourceCarrierSymbolGainRule rule;
    rule.carrier = parse_size(parts[0], "source carrier symbol gain carrier");
    rule.first_frame = parse_size(parts[1], "source carrier symbol gain first frame");
    rule.last_frame = parse_size(parts[2], "source carrier symbol gain last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument("source carrier symbol gain end is before start: " + text);
    }
    return rule;
}

SourceFrameAxisAffineRule parse_axis_affine_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source frame axis affine rule: " + text);
    }
    SourceFrameAxisAffineRule rule;
    rule.branch = parse_size(parts[0], "source frame axis affine branch");
    rule.first_frame = parse_size(parts[1], "source frame axis affine first frame");
    rule.last_frame = parse_size(parts[2], "source frame axis affine last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument("source frame axis affine end is before start: " + text);
    }
    return rule;
}

dtmb::core::SymbolInterleaverMode parse_mode(const std::string& text) {
    if (text == "mode1") {
        return dtmb::core::SymbolInterleaverMode::mode1;
    }
    if (text == "mode2") {
        return dtmb::core::SymbolInterleaverMode::mode2;
    }
    throw std::invalid_argument("mode must be mode1 or mode2");
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

void write_all(std::ostream& output, std::span<const float> values) {
    const auto bytes = static_cast<std::streamsize>(values.size() * sizeof(float));
    output.write(reinterpret_cast<const char*>(values.data()), bytes);
    if (!output) {
        throw std::runtime_error("failed to write output LLR stream");
    }
}

float nearest_qam64_level(float value) noexcept {
    auto nearest = kQam64Levels.front();
    auto best_distance = std::abs(value - nearest);
    for (const auto level : kQam64Levels) {
        const auto distance = std::abs(value - level);
        if (distance < best_distance) {
            best_distance = distance;
            nearest = level;
        }
    }
    return nearest;
}

std::size_t source_symbol_for_output_symbol(
    std::size_t output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase) {
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const auto latency = spec.full_stream_latency_symbols();
    if (output_symbol > std::numeric_limits<std::size_t>::max() - latency) {
        throw std::overflow_error("branch-gain output symbol index overflow");
    }
    const auto deinterleaver_output_index = output_symbol + latency;
    const auto start = deinterleaver_output_index % spec.branch_count;
    const auto physical_branch = (start + phase) % spec.branch_count;
    const auto branch_delay = (spec.branch_count - 1U - physical_branch)
        * spec.delay_step;
    const auto branch_offset =
        (deinterleaver_output_index - start) / spec.branch_count;
    if (branch_offset < branch_delay) {
        throw std::runtime_error("branch-gain source index is negative");
    }
    const auto source_offset = branch_offset - branch_delay;
    if (source_offset > (
            std::numeric_limits<std::size_t>::max() - start) / spec.branch_count) {
        throw std::overflow_error("branch-gain source symbol index overflow");
    }
    return start + spec.branch_count * source_offset;
}

std::size_t source_branch_for_output_symbol(
    std::size_t output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase) {
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const auto source_symbol = source_symbol_for_output_symbol(output_symbol, mode, phase);
    return (source_symbol % spec.branch_count + phase) % spec.branch_count;
}

BranchGainFrameStats* find_frame_stats(
    std::vector<BranchGainFrameStats>& rows,
    std::size_t branch,
    std::size_t source_frame) {
    for (auto& row : rows) {
        if (row.branch == branch && row.source_frame == source_frame) {
            return &row;
        }
    }
    return nullptr;
}

void update_frame_diagnostics(
    std::vector<BranchGainFrameStats>& rows,
    std::size_t branch,
    std::size_t source_frame,
    std::complex<float> value,
    std::complex<float> nearest,
    float relative_error,
    const BranchGainOptions& options) {
    if (rows.empty()) {
        return;
    }
    auto* row = find_frame_stats(rows, branch, source_frame);
    if (row == nullptr) {
        return;
    }
    ++row->symbols;
    row->relative_error_sum += static_cast<double>(relative_error);
    if (relative_error > options.reliability_threshold) {
        return;
    }
    ++row->reliable_symbols;
    row->numerator += static_cast<std::complex<double>>(std::conj(nearest) * value);
    row->denominator += static_cast<double>(std::norm(nearest));
    row->reliable_relative_error_sum += static_cast<double>(relative_error);
}

void apply_branch_gain_correction(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    std::vector<BranchGainStats>& stats,
    std::vector<BranchGainFrameStats>& frame_stats) {
    if (options.branches.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    for (std::size_t branch_index = 0; branch_index < options.branches.size(); ++branch_index) {
        const auto branch = options.branches[branch_index];
        auto& branch_stats = stats[branch_index];
        ++branch_stats.chunks_seen;
        std::complex<float> numerator{0.0F, 0.0F};
        float denominator = 0.0F;
        double reliable_relative_error_sum = 0.0;
        std::size_t reliable_count = 0;
        std::size_t branch_symbols = 0;
        for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
            const auto source_symbol = source_symbol_for_output_symbol(
                first_output_symbol + symbol,
                mode,
                phase);
            const auto spec = dtmb::core::symbol_interleaver_spec(mode);
            const auto source_branch = (source_symbol % spec.branch_count + phase)
                % spec.branch_count;
            if (source_branch != branch) {
                continue;
            }
            ++branch_symbols;
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> nearest{
                nearest_qam64_level(value.real()),
                nearest_qam64_level(value.imag())};
            const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
            const auto relative_error = std::abs(value - nearest) / nearest_power;
            update_frame_diagnostics(
                frame_stats,
                branch,
                source_symbol / dtmb::core::kC3780DataSymbols,
                value,
                nearest,
                relative_error,
                options);
            if (relative_error > options.reliability_threshold) {
                continue;
            }
            numerator += std::conj(nearest) * value;
            denominator += std::norm(nearest);
            reliable_relative_error_sum += static_cast<double>(relative_error);
            ++reliable_count;
        }
        branch_stats.reliable_symbols += reliable_count;
        branch_stats.reliable_relative_error_sum += reliable_relative_error_sum;
        if (reliable_count < options.min_symbols || denominator <= 1.0e-9F) {
            ++branch_stats.skipped_chunks;
            continue;
        }
        const auto gain = numerator / denominator;
        if (std::abs(gain) <= 1.0e-9F || !std::isfinite(gain.real())
            || !std::isfinite(gain.imag())) {
            ++branch_stats.skipped_chunks;
            continue;
        }
        for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
            if (source_branch_for_output_symbol(first_output_symbol + symbol, mode, phase)
                != branch) {
                continue;
            }
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> nearest{
                nearest_qam64_level((value / gain).real()),
                nearest_qam64_level((value / gain).imag())};
            const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
            const auto corrected = value / gain;
            symbols[symbol * kFloatsPerSymbol] = corrected.real();
            symbols[symbol * kFloatsPerSymbol + 1U] = corrected.imag();
            branch_stats.corrected_relative_error_sum += static_cast<double>(
                std::abs(corrected - nearest) / nearest_power);
        }
        branch_stats.gain_real_sum += static_cast<double>(gain.real());
        branch_stats.gain_imag_sum += static_cast<double>(gain.imag());
        branch_stats.gain_abs_sum += static_cast<double>(std::abs(gain));
        branch_stats.corrected_symbols += branch_symbols;
        ++branch_stats.applied_chunks;
    }
}

std::vector<SourceFrameSymbolGainStats> make_symbol_gain_stats(
    const std::vector<SourceFrameSymbolGainRule>& rules) {
    std::vector<SourceFrameSymbolGainStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceFrameSymbolGainStats row;
        row.rule_index = index;
        row.branch = rules[index].branch;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceFrameAxisAffineStats> make_axis_affine_stats(
    const std::vector<SourceFrameAxisAffineRule>& rules) {
    std::vector<SourceFrameAxisAffineStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceFrameAxisAffineStats row;
        row.rule_index = index;
        row.branch = rules[index].branch;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

struct AxisAffineFit {
    float real_scale = 1.0F;
    float real_bias = 0.0F;
    float imag_scale = 1.0F;
    float imag_bias = 0.0F;
    bool valid = false;
};

AxisAffineFit fit_axis_affine(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const SourceFrameAxisAffineRule& rule,
    std::size_t& reliable_count) {
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    double sum_nearest_real = 0.0;
    double sum_nearest_imag = 0.0;
    double sum_value_real = 0.0;
    double sum_value_imag = 0.0;
    double sum_nearest_real2 = 0.0;
    double sum_nearest_imag2 = 0.0;
    double sum_nearest_value_real = 0.0;
    double sum_nearest_value_imag = 0.0;
    reliable_count = 0;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_branch = (source_symbol % spec.branch_count + phase)
            % spec.branch_count;
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        if (source_branch != rule.branch || source_frame < rule.first_frame
            || source_frame > rule.last_frame) {
            continue;
        }
        const std::complex<float> value{
            symbols[symbol * kFloatsPerSymbol],
            symbols[symbol * kFloatsPerSymbol + 1U]};
        const std::complex<float> nearest{
            nearest_qam64_level(value.real()),
            nearest_qam64_level(value.imag())};
        const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
        const auto relative_error = std::abs(value - nearest) / nearest_power;
        if (relative_error > options.reliability_threshold) {
            continue;
        }
        ++reliable_count;
        sum_nearest_real += nearest.real();
        sum_nearest_imag += nearest.imag();
        sum_value_real += value.real();
        sum_value_imag += value.imag();
        sum_nearest_real2 += static_cast<double>(nearest.real()) * nearest.real();
        sum_nearest_imag2 += static_cast<double>(nearest.imag()) * nearest.imag();
        sum_nearest_value_real += static_cast<double>(nearest.real()) * value.real();
        sum_nearest_value_imag += static_cast<double>(nearest.imag()) * value.imag();
    }
    if (reliable_count < options.min_symbols) {
        return {};
    }
    const auto count = static_cast<double>(reliable_count);
    const auto real_denominator = sum_nearest_real2 - sum_nearest_real * sum_nearest_real / count;
    const auto imag_denominator = sum_nearest_imag2 - sum_nearest_imag * sum_nearest_imag / count;
    if (std::abs(real_denominator) <= 1.0e-9 || std::abs(imag_denominator) <= 1.0e-9) {
        return {};
    }
    AxisAffineFit fit;
    const auto real_scale = (
        sum_nearest_value_real - sum_nearest_real * sum_value_real / count)
        / real_denominator;
    const auto imag_scale = (
        sum_nearest_value_imag - sum_nearest_imag * sum_value_imag / count)
        / imag_denominator;
    const auto real_bias = (sum_value_real - real_scale * sum_nearest_real) / count;
    const auto imag_bias = (sum_value_imag - imag_scale * sum_nearest_imag) / count;
    if (!std::isfinite(real_scale) || !std::isfinite(imag_scale)
        || !std::isfinite(real_bias) || !std::isfinite(imag_bias)
        || std::abs(real_scale) <= 1.0e-9 || std::abs(imag_scale) <= 1.0e-9) {
        return {};
    }
    fit.real_scale = static_cast<float>(real_scale);
    fit.imag_scale = static_cast<float>(imag_scale);
    fit.real_bias = static_cast<float>(real_bias);
    fit.imag_bias = static_cast<float>(imag_bias);
    fit.valid = true;
    return fit;
}

void apply_source_frame_axis_affine(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceFrameAxisAffineRule>& rules,
    std::vector<SourceFrameAxisAffineStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        const auto& rule = rules[rule_index];
        auto& row = stats[rule_index];
        ++row.chunks_seen;
        std::size_t reliable_count = 0;
        const auto fit = fit_axis_affine(
            symbols,
            first_output_symbol,
            mode,
            phase,
            options,
            rule,
            reliable_count);
        row.reliable_symbols += reliable_count;
        if (!fit.valid) {
            continue;
        }
        std::size_t corrected_count = 0;
        for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
            const auto source_symbol = source_symbol_for_output_symbol(
                first_output_symbol + symbol,
                mode,
                phase);
            const auto source_branch = (source_symbol % spec.branch_count + phase)
                % spec.branch_count;
            const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
            if (source_branch != rule.branch || source_frame < rule.first_frame
                || source_frame > rule.last_frame) {
                continue;
            }
            const auto real = symbols[symbol * kFloatsPerSymbol];
            const auto imag = symbols[symbol * kFloatsPerSymbol + 1U];
            symbols[symbol * kFloatsPerSymbol] = (real - fit.real_bias) / fit.real_scale;
            symbols[symbol * kFloatsPerSymbol + 1U] = (imag - fit.imag_bias) / fit.imag_scale;
            ++corrected_count;
        }
        row.corrected_symbols += corrected_count;
        row.real_scale_sum += fit.real_scale;
        row.real_bias_sum += fit.real_bias;
        row.imag_scale_sum += fit.imag_scale;
        row.imag_bias_sum += fit.imag_bias;
        ++row.applied_chunks;
    }
}

void apply_source_frame_symbol_gain(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceFrameSymbolGainRule>& rules,
    std::vector<SourceFrameSymbolGainStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        const auto& rule = rules[rule_index];
        auto& row = stats[rule_index];
        ++row.chunks_seen;
        std::complex<float> numerator{0.0F, 0.0F};
        float denominator = 0.0F;
        std::size_t reliable_count = 0;
        std::size_t matched_count = 0;
        for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
            const auto source_symbol = source_symbol_for_output_symbol(
                first_output_symbol + symbol,
                mode,
                phase);
            const auto source_branch = (source_symbol % spec.branch_count + phase)
                % spec.branch_count;
            const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
            if (source_branch != rule.branch || source_frame < rule.first_frame
                || source_frame > rule.last_frame) {
                continue;
            }
            ++matched_count;
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> nearest{
                nearest_qam64_level(value.real()),
                nearest_qam64_level(value.imag())};
            const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
            const auto relative_error = std::abs(value - nearest) / nearest_power;
            if (relative_error > options.reliability_threshold) {
                continue;
            }
            numerator += std::conj(nearest) * value;
            denominator += std::norm(nearest);
            ++reliable_count;
        }
        row.reliable_symbols += reliable_count;
        if (matched_count == 0 || reliable_count < options.min_symbols
            || denominator <= 1.0e-9F) {
            continue;
        }
        const auto gain = numerator / denominator;
        if (std::abs(gain) <= 1.0e-9F || !std::isfinite(gain.real())
            || !std::isfinite(gain.imag())) {
            continue;
        }
        for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
            const auto source_symbol = source_symbol_for_output_symbol(
                first_output_symbol + symbol,
                mode,
                phase);
            const auto source_branch = (source_symbol % spec.branch_count + phase)
                % spec.branch_count;
            const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
            if (source_branch != rule.branch || source_frame < rule.first_frame
                || source_frame > rule.last_frame) {
                continue;
            }
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const auto corrected = value / gain;
            symbols[symbol * kFloatsPerSymbol] = corrected.real();
            symbols[symbol * kFloatsPerSymbol + 1U] = corrected.imag();
        }
        row.corrected_symbols += matched_count;
        row.gain_real_sum += static_cast<double>(gain.real());
        row.gain_imag_sum += static_cast<double>(gain.imag());
        row.gain_abs_sum += static_cast<double>(std::abs(gain));
        ++row.applied_chunks;
    }
}

void apply_source_carrier_symbol_gain(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierSymbolGainRule>& rules,
    std::vector<SourceCarrierSymbolGainStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        const auto& rule = rules[rule_index];
        auto& row = stats[rule_index];
        ++row.chunks_seen;
        std::complex<float> numerator{0.0F, 0.0F};
        float denominator = 0.0F;
        std::size_t reliable_count = 0;
        std::size_t matched_count = 0;
        for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
            const auto source_symbol = source_symbol_for_output_symbol(
                first_output_symbol + symbol,
                mode,
                phase);
            const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
            const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
            if (source_carrier != rule.carrier || source_frame < rule.first_frame
                || source_frame > rule.last_frame) {
                continue;
            }
            ++matched_count;
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> nearest{
                nearest_qam64_level(value.real()),
                nearest_qam64_level(value.imag())};
            const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
            const auto relative_error = std::abs(value - nearest) / nearest_power;
            if (relative_error > options.reliability_threshold) {
                continue;
            }
            numerator += std::conj(nearest) * value;
            denominator += std::norm(nearest);
            ++reliable_count;
        }
        row.reliable_symbols += reliable_count;
        if (matched_count == 0 || reliable_count < options.min_symbols
            || denominator <= 1.0e-9F) {
            continue;
        }
        const auto gain = numerator / denominator;
        if (std::abs(gain) <= 1.0e-9F || !std::isfinite(gain.real())
            || !std::isfinite(gain.imag())) {
            continue;
        }
        for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
            const auto source_symbol = source_symbol_for_output_symbol(
                first_output_symbol + symbol,
                mode,
                phase);
            const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
            const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
            if (source_carrier != rule.carrier || source_frame < rule.first_frame
                || source_frame > rule.last_frame) {
                continue;
            }
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const auto corrected = value / gain;
            symbols[symbol * kFloatsPerSymbol] = corrected.real();
            symbols[symbol * kFloatsPerSymbol + 1U] = corrected.imag();
        }
        row.corrected_symbols += matched_count;
        row.gain_real_sum += static_cast<double>(gain.real());
        row.gain_imag_sum += static_cast<double>(gain.imag());
        row.gain_abs_sum += static_cast<double>(std::abs(gain));
        ++row.applied_chunks;
    }
}

std::vector<BranchGainFrameStats> make_frame_stats_rows(const BranchGainOptions& options) {
    std::vector<BranchGainFrameStats> rows;
    if (options.diagnostics_path.empty() || options.branches.empty()) {
        return rows;
    }
    if (options.source_frame_ranges.empty()) {
        throw std::invalid_argument(
            "branch gain diagnostics require at least one source frame range");
    }
    for (const auto branch : options.branches) {
        for (const auto [first, last] : options.source_frame_ranges) {
            for (auto frame = first; frame <= last; ++frame) {
                BranchGainFrameStats row;
                row.branch = branch;
                row.source_frame = frame;
                rows.push_back(row);
                if (frame == std::numeric_limits<std::size_t>::max()) {
                    break;
                }
            }
        }
    }
    return rows;
}

void write_frame_diagnostics(
    const std::string& path,
    const BranchGainOptions& options,
    const std::vector<BranchGainFrameStats>& rows) {
    if (path.empty()) {
        return;
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open branch gain diagnostics: " + path);
    }
    output << "{\"schema\":\"dtmb.branch_gain_diagnostics.v1\"";
    output << ",\"branches\":[";
    for (std::size_t index = 0; index < options.branches.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << options.branches[index];
    }
    output << "],\"source_frame_ranges\":[";
    for (std::size_t index = 0; index < options.source_frame_ranges.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << "{\"first\":" << options.source_frame_ranges[index].first
               << ",\"last\":" << options.source_frame_ranges[index].second << '}';
    }
    output << "],\"rows\":[";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        if (index != 0) {
            output << ',';
        }
        output << "{\"source_frame\":" << row.source_frame
               << ",\"branch\":" << row.branch
               << ",\"symbols\":" << row.symbols
               << ",\"reliable_symbols\":" << row.reliable_symbols;
        if (row.symbols != 0) {
            output << ",\"mean_relative_error\":"
                   << (row.relative_error_sum / static_cast<double>(row.symbols));
        } else {
            output << ",\"mean_relative_error\":null";
        }
        if (row.reliable_symbols != 0) {
            output << ",\"mean_reliable_relative_error\":"
                   << (row.reliable_relative_error_sum
                       / static_cast<double>(row.reliable_symbols));
        } else {
            output << ",\"mean_reliable_relative_error\":null";
        }
        if (row.denominator > 1.0e-12) {
            const auto gain = row.numerator / row.denominator;
            output << ",\"estimated_gain_real\":" << gain.real()
                   << ",\"estimated_gain_imag\":" << gain.imag()
                   << ",\"estimated_gain_abs\":" << std::abs(gain);
        } else {
            output << ",\"estimated_gain_real\":null"
                   << ",\"estimated_gain_imag\":null"
                   << ",\"estimated_gain_abs\":null";
        }
        output << '}';
    }
    output << "]}\n";
    if (!output) {
        throw std::runtime_error("failed to write branch gain diagnostics: " + path);
    }
}

std::vector<SourceFrameLlrScaleStats> make_llr_scale_stats(
    const std::vector<SourceFrameLlrScaleRule>& rules) {
    std::vector<SourceFrameLlrScaleStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceFrameLlrScaleStats row;
        row.rule_index = index;
        row.branch = rules[index].branch;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.scale = rules[index].scale;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierSymbolGainStats> make_carrier_symbol_gain_stats(
    const std::vector<SourceCarrierSymbolGainRule>& rules) {
    std::vector<SourceCarrierSymbolGainStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierSymbolGainStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierLlrScaleStats> make_carrier_llr_scale_stats(
    const std::vector<SourceCarrierLlrScaleRule>& rules) {
    std::vector<SourceCarrierLlrScaleStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierLlrScaleStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.scale = rules[index].scale;
        stats.push_back(row);
    }
    return stats;
}

void apply_source_frame_llr_scale(
    std::span<float> llrs,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const std::vector<SourceFrameLlrScaleRule>& rules,
    std::vector<SourceFrameLlrScaleStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const auto symbol_count = llrs.size() / kLlrsPerSymbol;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_branch = (source_symbol % spec.branch_count + phase)
            % spec.branch_count;
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
            const auto& rule = rules[rule_index];
            if (source_branch != rule.branch || source_frame < rule.first_frame
                || source_frame > rule.last_frame) {
                continue;
            }
            for (std::size_t bit = 0; bit < kLlrsPerSymbol; ++bit) {
                llrs[symbol * kLlrsPerSymbol + bit] *= rule.scale;
            }
            ++stats[rule_index].scaled_symbols;
            break;
        }
    }
}

void apply_source_carrier_llr_scale(
    std::span<float> llrs,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const std::vector<SourceCarrierLlrScaleRule>& rules,
    std::vector<SourceCarrierLlrScaleStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = llrs.size() / kLlrsPerSymbol;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
            const auto& rule = rules[rule_index];
            if (source_carrier != rule.carrier || source_frame < rule.first_frame
                || source_frame > rule.last_frame) {
                continue;
            }
            for (std::size_t bit = 0; bit < kLlrsPerSymbol; ++bit) {
                llrs[symbol * kLlrsPerSymbol + bit] *= rule.scale;
            }
            ++stats[rule_index].scaled_symbols;
            break;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    dtmb::core::QamSoftDemapOptions demap_options{};
    BranchGainOptions branch_gain_options{};
    std::vector<SourceFrameSymbolGainRule> symbol_gain_rules;
    std::vector<SourceCarrierSymbolGainRule> carrier_symbol_gain_rules;
    std::vector<SourceFrameAxisAffineRule> axis_affine_rules;
    std::vector<SourceFrameLlrScaleRule> llr_scale_rules;
    std::vector<SourceCarrierLlrScaleRule> carrier_llr_scale_rules;
    auto mode = dtmb::core::SymbolInterleaverMode::mode1;
    bool mode_set = false;
    bool keep_latency = false;
    std::size_t phase = 0;
    std::size_t chunk_symbols = 1U << 16U;
    std::string input_path = "-";
    std::string output_path = "-";
    std::vector<std::string> positional;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--mode") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                mode = parse_mode(argv[index]);
                mode_set = true;
            } else if (arg == "--phase") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                phase = parse_size(argv[index], "phase");
            } else if (arg == "--keep-latency") {
                keep_latency = true;
            } else if (arg == "--workers") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                demap_options.requested_workers = parse_size(argv[index], "worker count");
            } else if (arg == "--min-parallel-symbols") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                demap_options.min_parallel_symbols = parse_size(argv[index], "parallel threshold");
            } else if (arg == "--chunk-symbols") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                chunk_symbols = parse_size(argv[index], "chunk size");
            } else if (arg == "--noise-variance") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                demap_options.noise_variance = parse_float(argv[index], "noise variance");
            } else if (arg == "--branch-gain-branches") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options.branches = parse_branch_list(argv[index]);
            } else if (arg == "--branch-gain-reliability-threshold") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options.reliability_threshold =
                    parse_float(argv[index], "branch gain reliability threshold");
            } else if (arg == "--branch-gain-min-symbols") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options.min_symbols =
                    parse_size(argv[index], "branch gain min symbols");
            } else if (arg == "--branch-gain-source-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options.source_frame_ranges.push_back(
                    parse_range(argv[index], "branch gain source frame range"));
            } else if (arg == "--branch-gain-diagnostics-out") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options.diagnostics_path = argv[index];
            } else if (arg == "--source-frame-symbol-gain") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                symbol_gain_rules.push_back(parse_symbol_gain_rule(argv[index]));
            } else if (arg == "--source-carrier-symbol-gain") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_symbol_gain_rules.push_back(
                    parse_carrier_symbol_gain_rule(argv[index]));
            } else if (arg == "--source-frame-axis-affine") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                axis_affine_rules.push_back(parse_axis_affine_rule(argv[index]));
            } else if (arg == "--source-frame-llr-scale") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                llr_scale_rules.push_back(parse_llr_scale_rule(argv[index]));
            } else if (arg == "--source-carrier-llr-scale") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_llr_scale_rules.push_back(
                    parse_carrier_llr_scale_rule(argv[index]));
            } else if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                positional.push_back(arg);
            }
        }

        if (!mode_set || positional.size() > 2 || chunk_symbols == 0) {
            usage(argv[0]);
            return 2;
        }
        const auto mode_spec = dtmb::core::symbol_interleaver_spec(mode);
        if (phase >= mode_spec.branch_count) {
            throw std::invalid_argument("phase is outside the selected mode");
        }
        if (!std::isfinite(branch_gain_options.reliability_threshold)
            || branch_gain_options.reliability_threshold < 0.0F) {
            throw std::invalid_argument(
                "branch gain reliability threshold must be finite and non-negative");
        }
        if (branch_gain_options.min_symbols == 0) {
            throw std::invalid_argument("branch gain min symbols must be positive");
        }
        for (const auto branch : branch_gain_options.branches) {
            if (branch >= mode_spec.branch_count) {
                throw std::invalid_argument("branch gain branch is outside the selected mode");
            }
        }
        for (const auto& rule : llr_scale_rules) {
            if (rule.branch >= mode_spec.branch_count) {
                throw std::invalid_argument(
                    "source frame LLR scale branch is outside the selected mode");
            }
        }
        for (const auto& rule : symbol_gain_rules) {
            if (rule.branch >= mode_spec.branch_count) {
                throw std::invalid_argument(
                    "source frame symbol gain branch is outside the selected mode");
            }
        }
        for (const auto& rule : axis_affine_rules) {
            if (rule.branch >= mode_spec.branch_count) {
                throw std::invalid_argument(
                    "source frame axis affine branch is outside the selected mode");
            }
        }
        for (const auto& rule : carrier_llr_scale_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier LLR scale carrier is outside the C3780 data span");
            }
        }
        for (const auto& rule : carrier_symbol_gain_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier symbol gain carrier is outside the C3780 data span");
            }
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }
        if (positional.size() == 2) {
            output_path = positional[1];
        }

        dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);

        dtmb::core::SymbolDeinterleaverCf32 deinterleaver(mode, phase);
        auto discard_remaining = keep_latency ? std::size_t{0} : deinterleaver.latency_symbols();
        std::size_t input_symbols = 0;
        std::size_t output_symbols = 0;
        std::vector<BranchGainStats> branch_gain_stats;
        branch_gain_stats.reserve(branch_gain_options.branches.size());
        for (const auto branch : branch_gain_options.branches) {
            BranchGainStats stats;
            stats.branch = branch;
            branch_gain_stats.push_back(stats);
        }
        auto branch_gain_frame_stats = make_frame_stats_rows(branch_gain_options);
        auto symbol_gain_stats = make_symbol_gain_stats(symbol_gain_rules);
        auto carrier_symbol_gain_stats =
            make_carrier_symbol_gain_stats(carrier_symbol_gain_rules);
        auto axis_affine_stats = make_axis_affine_stats(axis_affine_rules);
        auto llr_scale_stats = make_llr_scale_stats(llr_scale_rules);
        auto carrier_llr_scale_stats = make_carrier_llr_scale_stats(carrier_llr_scale_rules);

        std::vector<float> input_chunk(chunk_symbols * kFloatsPerSymbol);
        std::vector<float> deinterleaved_chunk(chunk_symbols * kFloatsPerSymbol);
        std::vector<float> output_chunk(chunk_symbols * kLlrsPerSymbol);
        const auto input_chunk_bytes = static_cast<std::streamsize>(
            input_chunk.size() * sizeof(float));

        while (true) {
            input.read(reinterpret_cast<char*>(input_chunk.data()), input_chunk_bytes);
            const auto bytes_read = input.gcount();
            if (bytes_read == 0) {
                break;
            }
            if ((bytes_read % static_cast<std::streamsize>(sizeof(float) * kFloatsPerSymbol)) != 0) {
                throw std::runtime_error("CF32 input byte count is not a whole number of symbols");
            }

            const auto symbols_read = static_cast<std::size_t>(bytes_read)
                / (sizeof(float) * kFloatsPerSymbol);
            const auto input_values = std::span<const float>(
                input_chunk.data(),
                symbols_read * kFloatsPerSymbol);
            auto deinterleaved_values = std::span<float>(
                deinterleaved_chunk.data(),
                symbols_read * kFloatsPerSymbol);
            deinterleaver.process(input_values, deinterleaved_values);
            input_symbols += symbols_read;

            const auto discard_now = std::min(discard_remaining, symbols_read);
            discard_remaining -= discard_now;
            const auto useful_symbols = symbols_read - discard_now;
            if (useful_symbols != 0) {
                auto useful_input = std::span<float>(
                    deinterleaved_chunk.data() + discard_now * kFloatsPerSymbol,
                    useful_symbols * kFloatsPerSymbol);
                apply_branch_gain_correction(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    branch_gain_stats,
                    branch_gain_frame_stats);
                apply_source_frame_symbol_gain(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    symbol_gain_rules,
                    symbol_gain_stats);
                apply_source_carrier_symbol_gain(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_symbol_gain_rules,
                    carrier_symbol_gain_stats);
                apply_source_frame_axis_affine(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    axis_affine_rules,
                    axis_affine_stats);
                auto useful_output = std::span<float>(
                    output_chunk.data(),
                    useful_symbols * kLlrsPerSymbol);
                dtmb::core::qam64_soft_demodulate_cf32(
                    useful_input,
                    useful_output,
                    demap_options);
                apply_source_frame_llr_scale(
                    useful_output,
                    output_symbols,
                    mode,
                    phase,
                    llr_scale_rules,
                    llr_scale_stats);
                apply_source_carrier_llr_scale(
                    useful_output,
                    output_symbols,
                    mode,
                    phase,
                    carrier_llr_scale_rules,
                    carrier_llr_scale_stats);
                write_all(output, useful_output);
                output_symbols += useful_symbols;
            }

            if (bytes_read < input_chunk_bytes) {
                break;
            }
        }

        std::cerr << "input_symbols=" << input_symbols << '\n'
                  << "discarded_latency_symbols="
                  << (keep_latency ? 0 : deinterleaver.latency_symbols() - discard_remaining)
                  << '\n'
                  << "output_symbols=" << output_symbols << '\n'
                  << "branch_gain_enabled="
                  << (!branch_gain_options.branches.empty() ? "true" : "false") << '\n';
        for (const auto& stats : branch_gain_stats) {
            std::cerr << "branch_gain_branch_" << stats.branch
                      << "_chunks_seen=" << stats.chunks_seen << '\n'
                      << "branch_gain_branch_" << stats.branch
                      << "_applied_chunks=" << stats.applied_chunks << '\n'
                      << "branch_gain_branch_" << stats.branch
                      << "_skipped_chunks=" << stats.skipped_chunks << '\n'
                      << "branch_gain_branch_" << stats.branch
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "branch_gain_branch_" << stats.branch
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n';
            if (stats.applied_chunks != 0) {
                const auto applied_chunks = static_cast<double>(stats.applied_chunks);
                std::cerr << "branch_gain_branch_" << stats.branch
                          << "_mean_gain_real=" << (stats.gain_real_sum / applied_chunks)
                          << '\n'
                          << "branch_gain_branch_" << stats.branch
                          << "_mean_gain_imag=" << (stats.gain_imag_sum / applied_chunks)
                          << '\n'
                          << "branch_gain_branch_" << stats.branch
                          << "_mean_gain_abs=" << (stats.gain_abs_sum / applied_chunks)
                          << '\n';
            }
            if (stats.reliable_symbols != 0) {
                std::cerr << "branch_gain_branch_" << stats.branch
                          << "_mean_reliable_relative_error="
                          << (stats.reliable_relative_error_sum
                              / static_cast<double>(stats.reliable_symbols))
                          << '\n';
            }
            if (stats.corrected_symbols != 0) {
                std::cerr << "branch_gain_branch_" << stats.branch
                          << "_mean_corrected_relative_error="
                          << (stats.corrected_relative_error_sum
                              / static_cast<double>(stats.corrected_symbols))
                          << '\n';
            }
        }
        std::cerr << "source_frame_symbol_gain_rule_count=" << symbol_gain_stats.size() << '\n';
        for (const auto& stats : symbol_gain_stats) {
            std::cerr << "source_frame_symbol_gain_rule_" << stats.rule_index
                      << "_branch=" << stats.branch << '\n'
                      << "source_frame_symbol_gain_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_frame_symbol_gain_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_frame_symbol_gain_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_frame_symbol_gain_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n';
            if (stats.applied_chunks != 0) {
                const auto applied_chunks = static_cast<double>(stats.applied_chunks);
                std::cerr << "source_frame_symbol_gain_rule_" << stats.rule_index
                          << "_mean_gain_real="
                          << (stats.gain_real_sum / applied_chunks) << '\n'
                          << "source_frame_symbol_gain_rule_" << stats.rule_index
                          << "_mean_gain_imag="
                          << (stats.gain_imag_sum / applied_chunks) << '\n'
                          << "source_frame_symbol_gain_rule_" << stats.rule_index
                          << "_mean_gain_abs="
                          << (stats.gain_abs_sum / applied_chunks) << '\n';
            }
        }
        std::cerr << "source_carrier_symbol_gain_rule_count="
                  << carrier_symbol_gain_stats.size() << '\n';
        for (const auto& stats : carrier_symbol_gain_stats) {
            std::cerr << "source_carrier_symbol_gain_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_symbol_gain_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_symbol_gain_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_symbol_gain_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_symbol_gain_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n';
            if (stats.applied_chunks != 0) {
                const auto applied_chunks = static_cast<double>(stats.applied_chunks);
                std::cerr << "source_carrier_symbol_gain_rule_" << stats.rule_index
                          << "_mean_gain_real="
                          << (stats.gain_real_sum / applied_chunks) << '\n'
                          << "source_carrier_symbol_gain_rule_" << stats.rule_index
                          << "_mean_gain_imag="
                          << (stats.gain_imag_sum / applied_chunks) << '\n'
                          << "source_carrier_symbol_gain_rule_" << stats.rule_index
                          << "_mean_gain_abs="
                          << (stats.gain_abs_sum / applied_chunks) << '\n';
            }
        }
        std::cerr << "source_frame_axis_affine_rule_count=" << axis_affine_stats.size() << '\n';
        for (const auto& stats : axis_affine_stats) {
            std::cerr << "source_frame_axis_affine_rule_" << stats.rule_index
                      << "_branch=" << stats.branch << '\n'
                      << "source_frame_axis_affine_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_frame_axis_affine_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_frame_axis_affine_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_frame_axis_affine_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n';
            if (stats.applied_chunks != 0) {
                const auto applied_chunks = static_cast<double>(stats.applied_chunks);
                std::cerr << "source_frame_axis_affine_rule_" << stats.rule_index
                          << "_mean_real_scale="
                          << (stats.real_scale_sum / applied_chunks) << '\n'
                          << "source_frame_axis_affine_rule_" << stats.rule_index
                          << "_mean_real_bias="
                          << (stats.real_bias_sum / applied_chunks) << '\n'
                          << "source_frame_axis_affine_rule_" << stats.rule_index
                          << "_mean_imag_scale="
                          << (stats.imag_scale_sum / applied_chunks) << '\n'
                          << "source_frame_axis_affine_rule_" << stats.rule_index
                          << "_mean_imag_bias="
                          << (stats.imag_bias_sum / applied_chunks) << '\n';
            }
        }
        std::cerr << "source_frame_llr_scale_rule_count=" << llr_scale_stats.size() << '\n';
        for (const auto& stats : llr_scale_stats) {
            std::cerr << "source_frame_llr_scale_rule_" << stats.rule_index
                      << "_branch=" << stats.branch << '\n'
                      << "source_frame_llr_scale_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_frame_llr_scale_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_frame_llr_scale_rule_" << stats.rule_index
                      << "_scale=" << stats.scale << '\n'
                      << "source_frame_llr_scale_rule_" << stats.rule_index
                      << "_scaled_symbols=" << stats.scaled_symbols << '\n';
        }
        std::cerr << "source_carrier_llr_scale_rule_count="
                  << carrier_llr_scale_stats.size() << '\n';
        for (const auto& stats : carrier_llr_scale_stats) {
            std::cerr << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_scale=" << stats.scale << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_scaled_symbols=" << stats.scaled_symbols << '\n';
        }
        write_frame_diagnostics(
            branch_gain_options.diagnostics_path,
            branch_gain_options,
            branch_gain_frame_stats);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_deinterleave_qam64: " << exc.what() << '\n';
        return 1;
    }
}
