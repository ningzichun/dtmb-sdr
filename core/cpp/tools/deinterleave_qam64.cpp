#include "dtmb/core.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFloatsPerSymbol = 2;
constexpr std::size_t kLlrsPerSymbol = 6;
constexpr std::array<float, 8> kQam64Levels{-7.0F, -5.0F, -3.0F, -1.0F, 1.0F, 3.0F, 5.0F, 7.0F};
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::array<std::size_t, dtmb::core::kC3780SystemInfoSymbols>
    kC3780SystemInfoPositions{
        0, 140, 279, 419, 420, 560, 699, 839, 840, 980, 1119, 1259,
        1260, 1400, 1539, 1679, 1680, 1820, 1959, 2099, 2100, 2240,
        2379, 2519, 2520, 2660, 2799, 2939, 2940, 3080, 3219, 3359,
        3360, 3500, 3639, 3779,
    };

enum class SourceFrameRampCoordinate {
    data,
    physical,
};

struct SourceFrameRampGeometry {
    std::array<std::size_t, dtmb::core::kC3780DataSymbols> order{};
    std::array<double, 32> x{};
    double span = 32.0;
};

struct SourceCarrierFrameRange {
    std::size_t first_carrier = 0;
    std::size_t last_carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct BranchGainOptions {
    std::vector<std::size_t> branches;
    float reliability_threshold = 0.55F;
    std::size_t min_symbols = 32;
    std::string diagnostics_path;
    std::vector<std::pair<std::size_t, std::size_t>> source_frame_ranges;
    std::vector<std::pair<std::size_t, std::size_t>> carrier_fit_exclude_source_frame_ranges;
    std::vector<std::pair<std::size_t, std::size_t>>
        carrier_symbol_gain_fit_exclude_source_frame_ranges;
    std::vector<std::pair<std::size_t, std::size_t>>
        carrier_linear_affine_fit_exclude_source_frame_ranges;
    std::vector<std::pair<std::size_t, std::size_t>>
        carrier_axis_centroid_fit_exclude_source_frame_ranges;
    std::vector<SourceCarrierFrameRange>
        carrier_symbol_gain_fit_exclude_source_carrier_frame_ranges;
    std::vector<SourceCarrierFrameRange>
        carrier_linear_affine_fit_exclude_source_carrier_frame_ranges;
    std::vector<SourceCarrierFrameRange>
        carrier_axis_centroid_fit_exclude_source_carrier_frame_ranges;
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
    std::size_t last_carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float scale = 1.0F;
};

struct SourceCarrierLlrScaleStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t last_carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float scale = 1.0F;
    std::size_t scaled_symbols = 0;
};

struct SourceCarrierResidualLlrConfidenceRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float residual_knee = 0.45F;
    float min_scale = 0.15F;
};

struct SourceCarrierResidualLlrConfidenceStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    float residual_knee = 0.45F;
    float min_scale = 0.15F;
    std::size_t matched_symbols = 0;
    std::size_t scaled_symbols = 0;
    double scale_sum = 0.0;
    double scaled_residual_ratio_sum = 0.0;
    float min_observed_scale = 1.0F;
};

struct SourceFrameRampLlrErasureOptions {
    bool enabled = false;
    float threshold = 0.2F;
    float scale = 0.0F;
    SourceFrameRampCoordinate coordinate = SourceFrameRampCoordinate::data;
};

struct SourceFrameRampLlrErasureStats {
    std::size_t scored_frames = 0;
    std::size_t selected_frames = 0;
    std::size_t touched_symbols = 0;
    std::size_t scaled_symbols = 0;
    std::size_t out_of_metric_symbols = 0;
    double abs_ramp_sum = 0.0;
    double selected_abs_ramp_sum = 0.0;
    float max_abs_ramp = 0.0F;
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

struct SourceCarrierAxisAffineRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceCarrierLinearAffineRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceCarrierAxisQuantileRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceCarrierAxisCentroidRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceCarrierQamCentroidRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
};

struct SourceCarrierNeighborLeakageRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    int offset = 0;
};

struct SourceCarrierNeighborLeakageSetRule {
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::vector<int> offsets;
};

struct SourceCarrierAxisAffineStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
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

struct SourceCarrierLinearAffineStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t chunks_seen = 0;
    std::size_t applied_chunks = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    double real_from_real_sum = 0.0;
    double real_from_imag_sum = 0.0;
    double real_bias_sum = 0.0;
    double imag_from_real_sum = 0.0;
    double imag_from_imag_sum = 0.0;
    double imag_bias_sum = 0.0;
};

struct SourceCarrierAxisQuantileStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t chunks_seen = 0;
    std::size_t matched_symbols = 0;
    std::size_t corrected_symbols = 0;
    std::size_t applied_chunks = 0;
};

struct SourceCarrierAxisCentroidStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t chunks_seen = 0;
    std::size_t matched_symbols = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    std::size_t applied_chunks = 0;
};

struct SourceCarrierQamCentroidStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t chunks_seen = 0;
    std::size_t matched_symbols = 0;
    std::size_t reliable_symbols = 0;
    std::size_t centroid_points = 0;
    std::size_t corrected_symbols = 0;
    std::size_t applied_chunks = 0;
};

struct SourceCarrierNeighborLeakageStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    int offset = 0;
    std::size_t chunks_seen = 0;
    std::size_t matched_symbols = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    std::size_t applied_chunks = 0;
    double leakage_real_sum = 0.0;
    double leakage_imag_sum = 0.0;
    double leakage_abs_sum = 0.0;
};

struct SourceCarrierNeighborLeakageSetStats {
    std::size_t rule_index = 0;
    std::size_t carrier = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::size_t offset_count = 0;
    std::size_t chunks_seen = 0;
    std::size_t matched_symbols = 0;
    std::size_t reliable_symbols = 0;
    std::size_t corrected_symbols = 0;
    std::size_t applied_chunks = 0;
    double leakage_abs_sum = 0.0;
    double leakage_abs_max = 0.0;
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

struct SourceFrameFixedComplexGainRule {
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::complex<float> gain{1.0F, 0.0F};
};

struct SourceFrameFixedComplexGainStats {
    std::size_t rule_index = 0;
    std::size_t branch = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::complex<float> gain{1.0F, 0.0F};
    std::size_t corrected_symbols = 0;
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

struct SourceFrameCarrierRollRule {
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::int64_t shift = 0;
};

struct SourceFrameCarrierRollStats {
    std::size_t rule_index = 0;
    std::size_t first_frame = 0;
    std::size_t last_frame = 0;
    std::int64_t shift = 0;
    std::size_t rolled_frames = 0;
    std::size_t rolled_symbols = 0;
};

struct SourceFrameDisplacementRule {
    std::size_t fec_frame = 0;
    std::size_t codeword_slot = 0;
    std::int64_t source_frame_shift = 0;
};

struct SourceFrameDisplacementStats {
    std::size_t rule_index = 0;
    std::size_t fec_frame = 0;
    std::size_t codeword_slot = 0;
    std::int64_t source_frame_shift = 0;
    std::size_t replaced_symbols = 0;
};

struct SourceFrameDisplacementWindow {
    bool enabled = false;
    std::size_t first_symbol = 0;
    std::size_t last_symbol = 0;
};

struct DdLlrConfidenceOptions {
    bool enabled = false;
    float residual_knee = 0.45F;
    float min_scale = 0.15F;
};

struct DdLlrConfidenceStats {
    std::size_t symbols = 0;
    std::size_t scaled_symbols = 0;
    double scale_sum = 0.0;
    double scaled_residual_ratio_sum = 0.0;
    float min_observed_scale = 1.0F;
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
        << " [--source-frame-fixed-complex-gain BRANCH:FIRST:LAST:REAL:IMAG]"
        << " [--source-carrier-fixed-complex-gains PATH]"
        << " [--source-carrier-symbol-gain CARRIER:FIRST:LAST]"
        << " [--source-carrier-symbol-gain-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-affine CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-affine-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-linear-affine CARRIER:FIRST:LAST]"
        << " [--source-carrier-linear-affine-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-quantile CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-quantile-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-centroid CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-centroid-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-centroid-median CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-centroid-median-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-qam-centroid CARRIER:FIRST:LAST]"
        << " [--source-carrier-qam-centroid-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-neighbor-leakage CARRIER:FIRST:LAST:OFFSET]"
        << " [--source-carrier-neighbor-leakage-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST:OFFSET]"
        << " [--source-carrier-neighbor-leakage-set-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST:OFFSETS_CSV]"
        << " [--source-carrier-fit-exclude-source-frame-range FIRST:LAST]"
        << " [--source-carrier-symbol-gain-fit-exclude-source-frame-range FIRST:LAST]"
        << " [--source-carrier-symbol-gain-fit-exclude-source-carrier-frame-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-linear-affine-fit-exclude-source-frame-range FIRST:LAST]"
        << " [--source-carrier-linear-affine-fit-exclude-source-carrier-frame-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-carrier-axis-centroid-fit-exclude-source-frame-range FIRST:LAST]"
        << " [--source-carrier-axis-centroid-fit-exclude-source-carrier-frame-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST]"
        << " [--source-frame-carrier-roll FIRST:LAST:SHIFT]"
        << " [--source-frame-displacement FEC_FRAME:CODEWORD_SLOT:SHIFT]"
        << " [--source-frame-axis-affine BRANCH:FIRST:LAST]"
        << " [--source-frame-axis-affine-range FIRST_BRANCH:LAST_BRANCH:FIRST:LAST]"
        << " [--source-frame-llr-scale BRANCH:FIRST:LAST:SCALE]"
        << " [--source-carrier-llr-scale CARRIER:FIRST:LAST:SCALE]"
        << " [--source-carrier-llr-scale-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST:SCALE]"
        << " [--source-carrier-residual-llr-confidence CARRIER:FIRST:LAST:KNEE:MIN_SCALE]"
        << " [--source-carrier-residual-llr-confidence-range FIRST_CARRIER:LAST_CARRIER:FIRST:LAST:KNEE:MIN_SCALE]"
        << " [--source-frame-ramp-llr-erasure-threshold X]"
        << " [--source-frame-ramp-llr-erasure-scale X]"
        << " [--source-frame-ramp-llr-erasure-coordinate data|physical]"
        << " [--output-frame-range FIRST:LAST]"
        << " [--symbols-output PATH]"
        << " [--dd-llr-confidence]"
        << " [--dd-llr-confidence-knee X]"
        << " [--dd-llr-confidence-min-scale X]"
        << " [--csi-weights PATH]"
        << " [--noise-variance X] [--soft-demod-method max-log|log-sum-exp]"
        << " [input.cf32|-] [output.llr.f32|-]\n";
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

dtmb::core::QamSoftDemapMethod parse_soft_demap_method(const std::string& text) {
    if (text == "max-log" || text == "maxlog") {
        return dtmb::core::QamSoftDemapMethod::max_log;
    }
    if (text == "log-sum-exp" || text == "logsumexp" || text == "exact") {
        return dtmb::core::QamSoftDemapMethod::log_sum_exp;
    }
    throw std::invalid_argument("invalid soft demod method: " + text);
}

SourceFrameRampCoordinate parse_ramp_coordinate(const std::string& text) {
    if (text == "data") {
        return SourceFrameRampCoordinate::data;
    }
    if (text == "physical") {
        return SourceFrameRampCoordinate::physical;
    }
    throw std::invalid_argument(
        "source frame ramp LLR erasure coordinate must be data or physical");
}

const char* ramp_coordinate_name(SourceFrameRampCoordinate coordinate) noexcept {
    switch (coordinate) {
    case SourceFrameRampCoordinate::data:
        return "data";
    case SourceFrameRampCoordinate::physical:
        return "physical";
    }
    return "unknown";
}

std::int64_t parse_i64(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoll(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::int64_t>(value);
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

SourceCarrierFrameRange parse_carrier_frame_range(
    const std::string& text,
    const char* field) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4U) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    SourceCarrierFrameRange range;
    range.first_carrier = parse_size(parts[0], field);
    range.last_carrier = parse_size(parts[1], field);
    if (range.last_carrier < range.first_carrier) {
        throw std::invalid_argument(
            std::string(field) + " carrier range is reversed: " + text);
    }
    range.first_frame = parse_size(parts[2], field);
    range.last_frame = parse_size(parts[3], field);
    if (range.last_frame < range.first_frame) {
        throw std::invalid_argument(
            std::string(field) + " frame range is reversed: " + text);
    }
    return range;
}

bool source_frame_in_ranges(
    std::size_t source_frame,
    const std::vector<std::pair<std::size_t, std::size_t>>& ranges) {
    for (const auto [first, last] : ranges) {
        if (source_frame >= first && source_frame <= last) {
            return true;
        }
    }
    return false;
}

bool source_carrier_frame_in_ranges(
    std::size_t source_carrier,
    std::size_t source_frame,
    const std::vector<SourceCarrierFrameRange>& ranges) {
    for (const auto& range : ranges) {
        if (source_carrier >= range.first_carrier
            && source_carrier <= range.last_carrier
            && source_frame >= range.first_frame
            && source_frame <= range.last_frame) {
            return true;
        }
    }
    return false;
}

bool source_carrier_fit_uses_frame(
    const BranchGainOptions& options,
    std::size_t source_carrier,
    std::size_t source_frame,
    const std::vector<std::pair<std::size_t, std::size_t>>& stage_ranges = {},
    const std::vector<SourceCarrierFrameRange>& stage_carrier_frame_ranges = {}) {
    return !source_frame_in_ranges(
        source_frame,
        options.carrier_fit_exclude_source_frame_ranges)
        && !source_frame_in_ranges(source_frame, stage_ranges)
        && !source_carrier_frame_in_ranges(
            source_carrier,
            source_frame,
            stage_carrier_frame_ranges);
}

void print_source_frame_ranges(
    const char* prefix,
    const std::vector<std::pair<std::size_t, std::size_t>>& ranges) {
    std::cerr << prefix << "_count=" << ranges.size() << '\n';
    for (std::size_t range_index = 0; range_index < ranges.size(); ++range_index) {
        const auto [first, last] = ranges[range_index];
        std::cerr << prefix << "_" << range_index << "_first=" << first << '\n'
                  << prefix << "_" << range_index << "_last=" << last << '\n';
    }
}

void print_source_carrier_frame_ranges(
    const char* prefix,
    const std::vector<SourceCarrierFrameRange>& ranges) {
    std::cerr << prefix << "_count=" << ranges.size() << '\n';
    for (std::size_t range_index = 0; range_index < ranges.size(); ++range_index) {
        const auto& range = ranges[range_index];
        std::cerr << prefix << "_" << range_index << "_first_carrier="
                  << range.first_carrier << '\n'
                  << prefix << "_" << range_index << "_last_carrier="
                  << range.last_carrier << '\n'
                  << prefix << "_" << range_index << "_first_frame="
                  << range.first_frame << '\n'
                  << prefix << "_" << range_index << "_last_frame="
                  << range.last_frame << '\n';
    }
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

SourceFrameCarrierRollRule parse_carrier_roll_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source frame carrier roll rule: " + text);
    }
    SourceFrameCarrierRollRule rule;
    rule.first_frame = parse_size(parts[0], "source frame carrier roll first frame");
    rule.last_frame = parse_size(parts[1], "source frame carrier roll last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument("source frame carrier roll end is before start: " + text);
    }
    rule.shift = parse_i64(parts[2], "source frame carrier roll shift");
    return rule;
}

SourceFrameDisplacementRule parse_source_frame_displacement_rule(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument(
            "invalid source frame displacement rule: " + text);
    }
    SourceFrameDisplacementRule rule;
    rule.fec_frame = parse_size(parts[0], "source frame displacement FEC frame");
    rule.codeword_slot =
        parse_size(parts[1], "source frame displacement codeword slot");
    rule.source_frame_shift =
        parse_i64(parts[2], "source frame displacement shift");
    if (rule.fec_frame == 0U) {
        throw std::invalid_argument(
            "source frame displacement FEC frame must be 1-based");
    }
    if (rule.codeword_slot >= 3U) {
        throw std::invalid_argument(
            "source frame displacement codeword slot must be 0, 1, or 2");
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
    rule.last_carrier = rule.carrier;
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

SourceCarrierLlrScaleRule parse_carrier_llr_scale_range_rule(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 5) {
        throw std::invalid_argument(
            "invalid source carrier LLR scale range rule: " + text);
    }
    SourceCarrierLlrScaleRule rule;
    rule.carrier = parse_size(parts[0], "source carrier LLR scale first carrier");
    rule.last_carrier =
        parse_size(parts[1], "source carrier LLR scale last carrier");
    if (rule.last_carrier < rule.carrier) {
        throw std::invalid_argument(
            "source carrier LLR scale carrier range is reversed: " + text);
    }
    rule.first_frame = parse_size(parts[2], "source carrier LLR scale first frame");
    rule.last_frame = parse_size(parts[3], "source carrier LLR scale last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument("source carrier LLR scale end is before start: " + text);
    }
    rule.scale = parse_float(parts[4], "source carrier LLR scale");
    if (!std::isfinite(rule.scale) || rule.scale < 0.0F) {
        throw std::invalid_argument("source carrier LLR scale must be finite and non-negative");
    }
    return rule;
}

SourceCarrierResidualLlrConfidenceRule parse_carrier_residual_llr_confidence_rule(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 5) {
        throw std::invalid_argument(
            "invalid source carrier residual LLR confidence rule: " + text);
    }
    SourceCarrierResidualLlrConfidenceRule rule;
    rule.carrier = parse_size(parts[0], "source carrier residual LLR confidence carrier");
    rule.first_frame =
        parse_size(parts[1], "source carrier residual LLR confidence first frame");
    rule.last_frame =
        parse_size(parts[2], "source carrier residual LLR confidence last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier residual LLR confidence end is before start: " + text);
    }
    rule.residual_knee =
        parse_float(parts[3], "source carrier residual LLR confidence knee");
    rule.min_scale =
        parse_float(parts[4], "source carrier residual LLR confidence min scale");
    if (!std::isfinite(rule.residual_knee) || rule.residual_knee <= 0.0F) {
        throw std::invalid_argument(
            "source carrier residual LLR confidence knee must be positive and finite");
    }
    if (!std::isfinite(rule.min_scale) || rule.min_scale < 0.0F
        || rule.min_scale > 1.0F) {
        throw std::invalid_argument(
            "source carrier residual LLR confidence min scale must be within 0..1");
    }
    return rule;
}

std::vector<SourceCarrierResidualLlrConfidenceRule>
parse_carrier_residual_llr_confidence_range(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 6) {
        throw std::invalid_argument(
            "invalid source carrier residual LLR confidence range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier residual LLR confidence range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier residual LLR confidence range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier residual LLR confidence range carrier end is before start: "
            + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier residual LLR confidence range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier residual LLR confidence range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier residual LLR confidence range frame end is before start: "
            + text);
    }
    const auto residual_knee =
        parse_float(parts[4], "source carrier residual LLR confidence range knee");
    const auto min_scale =
        parse_float(parts[5], "source carrier residual LLR confidence range min scale");
    if (!std::isfinite(residual_knee) || residual_knee <= 0.0F) {
        throw std::invalid_argument(
            "source carrier residual LLR confidence knee must be positive and finite");
    }
    if (!std::isfinite(min_scale) || min_scale < 0.0F || min_scale > 1.0F) {
        throw std::invalid_argument(
            "source carrier residual LLR confidence min scale must be within 0..1");
    }
    std::vector<SourceCarrierResidualLlrConfidenceRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierResidualLlrConfidenceRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rule.residual_knee = residual_knee;
        rule.min_scale = min_scale;
        rules.push_back(rule);
    }
    return rules;
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

SourceFrameFixedComplexGainRule parse_fixed_complex_gain_rule(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 5) {
        throw std::invalid_argument(
            "invalid source frame fixed complex gain rule: " + text);
    }
    SourceFrameFixedComplexGainRule rule;
    rule.branch = parse_size(parts[0], "source frame fixed complex gain branch");
    rule.first_frame = parse_size(
        parts[1], "source frame fixed complex gain first frame");
    rule.last_frame = parse_size(
        parts[2], "source frame fixed complex gain last frame");
    rule.gain = {
        parse_float(parts[3], "source frame fixed complex gain real"),
        parse_float(parts[4], "source frame fixed complex gain imag")};
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source frame fixed complex gain end is before start: " + text);
    }
    if (!std::isfinite(rule.gain.real()) || !std::isfinite(rule.gain.imag())
        || std::abs(rule.gain) <= 1.0e-9F) {
        throw std::invalid_argument(
            "source frame fixed complex gain must be finite and non-zero: " + text);
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

std::vector<SourceCarrierSymbolGainRule> parse_carrier_symbol_gain_range(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier symbol gain range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier symbol gain range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier symbol gain range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier symbol gain range carrier end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier symbol gain range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier symbol gain range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier symbol gain range frame end is before start: " + text);
    }
    std::vector<SourceCarrierSymbolGainRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierSymbolGainRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rules.push_back(rule);
    }
    return rules;
}

SourceCarrierAxisAffineRule parse_carrier_axis_affine_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source carrier axis affine rule: " + text);
    }
    SourceCarrierAxisAffineRule rule;
    rule.carrier = parse_size(parts[0], "source carrier axis affine carrier");
    rule.first_frame = parse_size(parts[1], "source carrier axis affine first frame");
    rule.last_frame = parse_size(parts[2], "source carrier axis affine last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier axis affine end is before start: " + text);
    }
    return rule;
}

std::vector<SourceCarrierAxisAffineRule> parse_carrier_axis_affine_range(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier axis affine range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier axis affine range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier axis affine range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier axis affine range carrier end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier axis affine range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier axis affine range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier axis affine range frame end is before start: " + text);
    }
    std::vector<SourceCarrierAxisAffineRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierAxisAffineRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rules.push_back(rule);
    }
    return rules;
}

SourceCarrierLinearAffineRule parse_carrier_linear_affine_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source carrier linear affine rule: " + text);
    }
    SourceCarrierLinearAffineRule rule;
    rule.carrier = parse_size(parts[0], "source carrier linear affine carrier");
    rule.first_frame = parse_size(parts[1], "source carrier linear affine first frame");
    rule.last_frame = parse_size(parts[2], "source carrier linear affine last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier linear affine end is before start: " + text);
    }
    return rule;
}

std::vector<SourceCarrierLinearAffineRule> parse_carrier_linear_affine_range(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier linear affine range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier linear affine range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier linear affine range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier linear affine range carrier end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier linear affine range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier linear affine range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier linear affine range frame end is before start: " + text);
    }
    std::vector<SourceCarrierLinearAffineRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierLinearAffineRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rules.push_back(rule);
    }
    return rules;
}

SourceCarrierAxisQuantileRule parse_carrier_axis_quantile_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source carrier axis quantile rule: " + text);
    }
    SourceCarrierAxisQuantileRule rule;
    rule.carrier = parse_size(parts[0], "source carrier axis quantile carrier");
    rule.first_frame = parse_size(parts[1], "source carrier axis quantile first frame");
    rule.last_frame = parse_size(parts[2], "source carrier axis quantile last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier axis quantile end is before start: " + text);
    }
    return rule;
}

std::vector<SourceCarrierAxisQuantileRule> parse_carrier_axis_quantile_range(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier axis quantile range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier axis quantile range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier axis quantile range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier axis quantile range carrier end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier axis quantile range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier axis quantile range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier axis quantile range frame end is before start: " + text);
    }
    std::vector<SourceCarrierAxisQuantileRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierAxisQuantileRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rules.push_back(rule);
    }
    return rules;
}

SourceCarrierAxisCentroidRule parse_carrier_axis_centroid_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source carrier axis centroid rule: " + text);
    }
    SourceCarrierAxisCentroidRule rule;
    rule.carrier = parse_size(parts[0], "source carrier axis centroid carrier");
    rule.first_frame = parse_size(parts[1], "source carrier axis centroid first frame");
    rule.last_frame = parse_size(parts[2], "source carrier axis centroid last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier axis centroid end is before start: " + text);
    }
    return rule;
}

std::vector<SourceCarrierAxisCentroidRule> parse_carrier_axis_centroid_range(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier axis centroid range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier axis centroid range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier axis centroid range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier axis centroid range carrier end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier axis centroid range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier axis centroid range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier axis centroid range frame end is before start: " + text);
    }
    std::vector<SourceCarrierAxisCentroidRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierAxisCentroidRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rules.push_back(rule);
    }
    return rules;
}

SourceCarrierQamCentroidRule parse_carrier_qam_centroid_rule(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 3) {
        throw std::invalid_argument("invalid source carrier QAM centroid rule: " + text);
    }
    SourceCarrierQamCentroidRule rule;
    rule.carrier = parse_size(parts[0], "source carrier QAM centroid carrier");
    rule.first_frame = parse_size(parts[1], "source carrier QAM centroid first frame");
    rule.last_frame = parse_size(parts[2], "source carrier QAM centroid last frame");
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier QAM centroid end is before start: " + text);
    }
    return rule;
}

std::vector<SourceCarrierQamCentroidRule> parse_carrier_qam_centroid_range(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier QAM centroid range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier QAM centroid range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier QAM centroid range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier QAM centroid range carrier end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier QAM centroid range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier QAM centroid range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier QAM centroid range frame end is before start: " + text);
    }
    std::vector<SourceCarrierQamCentroidRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierQamCentroidRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rules.push_back(rule);
    }
    return rules;
}

SourceCarrierNeighborLeakageRule parse_carrier_neighbor_leakage_rule(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source carrier neighbor leakage rule: " + text);
    }
    SourceCarrierNeighborLeakageRule rule;
    rule.carrier = parse_size(parts[0], "source carrier neighbor leakage carrier");
    rule.first_frame = parse_size(parts[1], "source carrier neighbor leakage first frame");
    rule.last_frame = parse_size(parts[2], "source carrier neighbor leakage last frame");
    rule.offset = static_cast<int>(
        parse_i64(parts[3], "source carrier neighbor leakage offset"));
    if (rule.last_frame < rule.first_frame) {
        throw std::invalid_argument(
            "source carrier neighbor leakage end is before start: " + text);
    }
    if (rule.offset == 0) {
        throw std::invalid_argument("source carrier neighbor leakage offset must be nonzero");
    }
    return rule;
}

std::vector<SourceCarrierNeighborLeakageRule> parse_carrier_neighbor_leakage_range(
    const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 5) {
        throw std::invalid_argument("invalid source carrier neighbor leakage range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier neighbor leakage range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier neighbor leakage range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier neighbor leakage range carrier end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier neighbor leakage range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier neighbor leakage range last frame");
    const auto offset = static_cast<int>(
        parse_i64(parts[4], "source carrier neighbor leakage range offset"));
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier neighbor leakage range frame end is before start: " + text);
    }
    if (offset == 0) {
        throw std::invalid_argument("source carrier neighbor leakage offset must be nonzero");
    }
    std::vector<SourceCarrierNeighborLeakageRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierNeighborLeakageRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rule.offset = offset;
        rules.push_back(rule);
    }
    return rules;
}

std::vector<int> parse_neighbor_offset_list(const std::string& text) {
    std::vector<int> offsets;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument("empty source carrier neighbor leakage offset");
        }
        const auto offset =
            static_cast<int>(parse_i64(item, "source carrier neighbor leakage set offset"));
        if (offset == 0) {
            throw std::invalid_argument(
                "source carrier neighbor leakage set offsets must be nonzero");
        }
        if (std::find(offsets.begin(), offsets.end(), offset) == offsets.end()) {
            offsets.push_back(offset);
        }
    }
    if (offsets.empty()) {
        throw std::invalid_argument("source carrier neighbor leakage set needs offsets");
    }
    return offsets;
}

std::vector<SourceCarrierNeighborLeakageSetRule>
parse_carrier_neighbor_leakage_set_range(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 5) {
        throw std::invalid_argument(
            "invalid source carrier neighbor leakage set range: " + text);
    }
    const auto first_carrier =
        parse_size(parts[0], "source carrier neighbor leakage set range first carrier");
    const auto last_carrier =
        parse_size(parts[1], "source carrier neighbor leakage set range last carrier");
    if (last_carrier < first_carrier) {
        throw std::invalid_argument(
            "source carrier neighbor leakage set range carrier end is before start: "
            + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source carrier neighbor leakage set range first frame");
    const auto last_frame =
        parse_size(parts[3], "source carrier neighbor leakage set range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source carrier neighbor leakage set range frame end is before start: "
            + text);
    }
    auto offsets = parse_neighbor_offset_list(parts[4]);
    std::vector<SourceCarrierNeighborLeakageSetRule> rules;
    rules.reserve(last_carrier - first_carrier + 1U);
    for (auto carrier = first_carrier; carrier <= last_carrier; ++carrier) {
        SourceCarrierNeighborLeakageSetRule rule;
        rule.carrier = carrier;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rule.offsets = offsets;
        rules.push_back(std::move(rule));
    }
    return rules;
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

std::vector<SourceFrameAxisAffineRule> parse_axis_affine_range(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        parts.push_back(item);
    }
    if (parts.size() != 4) {
        throw std::invalid_argument("invalid source frame axis affine range: " + text);
    }
    const auto first_branch =
        parse_size(parts[0], "source frame axis affine range first branch");
    const auto last_branch =
        parse_size(parts[1], "source frame axis affine range last branch");
    if (last_branch < first_branch) {
        throw std::invalid_argument(
            "source frame axis affine range branch end is before start: " + text);
    }
    const auto first_frame =
        parse_size(parts[2], "source frame axis affine range first frame");
    const auto last_frame =
        parse_size(parts[3], "source frame axis affine range last frame");
    if (last_frame < first_frame) {
        throw std::invalid_argument(
            "source frame axis affine range frame end is before start: " + text);
    }
    std::vector<SourceFrameAxisAffineRule> rules;
    rules.reserve(last_branch - first_branch + 1U);
    for (auto branch = first_branch; branch <= last_branch; ++branch) {
        SourceFrameAxisAffineRule rule;
        rule.branch = branch;
        rule.first_frame = first_frame;
        rule.last_frame = last_frame;
        rules.push_back(rule);
    }
    return rules;
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

bool output_frame_selected(
    std::size_t frame_index,
    const std::vector<std::pair<std::size_t, std::size_t>>& ranges) {
    if (ranges.empty()) {
        return true;
    }
    return std::any_of(
        ranges.begin(),
        ranges.end(),
        [frame_index](const auto& range) {
            return frame_index >= range.first && frame_index <= range.second;
        });
}

std::size_t write_output_frame_ranges(
    std::ostream& output,
    std::span<const float> llr_values,
    std::size_t first_output_symbol,
    const std::vector<std::pair<std::size_t, std::size_t>>& ranges) {
    constexpr std::size_t kOutputSymbolsPerFecFrame =
        dtmb::core::kC3780DataSymbols;
    const auto symbol_count = llr_values.size() / kLlrsPerSymbol;
    if (ranges.empty()) {
        write_all(output, llr_values);
        return symbol_count;
    }

    std::size_t written_symbols = 0;
    std::size_t local_symbol = 0;
    while (local_symbol < symbol_count) {
        const auto global_symbol = first_output_symbol + local_symbol;
        const auto frame_index = global_symbol / kOutputSymbolsPerFecFrame + 1U;
        const auto frame_end_symbol = frame_index * kOutputSymbolsPerFecFrame;
        const auto segment_symbols = std::min(
            symbol_count - local_symbol,
            frame_end_symbol - global_symbol);
        if (output_frame_selected(frame_index, ranges)) {
            const auto first_llr = local_symbol * kLlrsPerSymbol;
            const auto llr_count = segment_symbols * kLlrsPerSymbol;
            write_all(output, llr_values.subspan(first_llr, llr_count));
            written_symbols += segment_symbols;
        }
        local_symbol += segment_symbols;
    }
    return written_symbols;
}

std::size_t write_symbol_output_frame_ranges(
    std::ostream& output,
    std::span<const float> symbol_values,
    std::size_t first_output_symbol,
    const std::vector<std::pair<std::size_t, std::size_t>>& ranges) {
    constexpr std::size_t kOutputSymbolsPerFecFrame =
        dtmb::core::kC3780DataSymbols;
    const auto symbol_count = symbol_values.size() / kFloatsPerSymbol;
    if (ranges.empty()) {
        write_all(output, symbol_values);
        return symbol_count;
    }

    std::size_t written_symbols = 0;
    std::size_t local_symbol = 0;
    while (local_symbol < symbol_count) {
        const auto global_symbol = first_output_symbol + local_symbol;
        const auto frame_index = global_symbol / kOutputSymbolsPerFecFrame + 1U;
        const auto frame_end_symbol = frame_index * kOutputSymbolsPerFecFrame;
        const auto segment_symbols = std::min(
            symbol_count - local_symbol,
            frame_end_symbol - global_symbol);
        if (output_frame_selected(frame_index, ranges)) {
            const auto first_value = local_symbol * kFloatsPerSymbol;
            const auto value_count = segment_symbols * kFloatsPerSymbol;
            write_all(output, symbol_values.subspan(first_value, value_count));
            written_symbols += segment_symbols;
        }
        local_symbol += segment_symbols;
    }
    return written_symbols;
}

std::size_t nearest_qam64_level_index(float value) noexcept {
    std::size_t nearest = 0;
    auto best_distance = std::abs(value - kQam64Levels[nearest]);
    for (std::size_t index = 1; index < kQam64Levels.size(); ++index) {
        const auto level = kQam64Levels[index];
        const auto distance = std::abs(value - level);
        if (distance < best_distance) {
            best_distance = distance;
            nearest = index;
        }
    }
    return nearest;
}

float nearest_qam64_level(float value) noexcept {
    return kQam64Levels[nearest_qam64_level_index(value)];
}

std::size_t qam64_point_index(std::size_t real_level, std::size_t imag_level) noexcept {
    return real_level * kQam64Levels.size() + imag_level;
}

std::complex<float> qam64_point_from_index(std::size_t index) noexcept {
    const auto real_level = index / kQam64Levels.size();
    const auto imag_level = index % kQam64Levels.size();
    return {kQam64Levels[real_level], kQam64Levels[imag_level]};
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

std::vector<SourceFrameFixedComplexGainStats> make_fixed_complex_gain_stats(
    const std::vector<SourceFrameFixedComplexGainRule>& rules) {
    std::vector<SourceFrameFixedComplexGainStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceFrameFixedComplexGainStats row;
        row.rule_index = index;
        row.branch = rules[index].branch;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.gain = rules[index].gain;
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

std::vector<SourceFrameCarrierRollStats> make_carrier_roll_stats(
    const std::vector<SourceFrameCarrierRollRule>& rules) {
    std::vector<SourceFrameCarrierRollStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceFrameCarrierRollStats row;
        row.rule_index = index;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.shift = rules[index].shift;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceFrameDisplacementStats> make_source_frame_displacement_stats(
    const std::vector<SourceFrameDisplacementRule>& rules) {
    std::vector<SourceFrameDisplacementStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceFrameDisplacementStats row;
        row.rule_index = index;
        row.fec_frame = rules[index].fec_frame;
        row.codeword_slot = rules[index].codeword_slot;
        row.source_frame_shift = rules[index].source_frame_shift;
        stats.push_back(row);
    }
    return stats;
}

SourceFrameDisplacementWindow make_source_frame_displacement_window(
    const std::vector<SourceFrameDisplacementRule>& rules,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase) {
    SourceFrameDisplacementWindow window;
    if (rules.empty()) {
        return window;
    }

    constexpr auto kSymbolsPerFrame = dtmb::core::kC3780DataSymbols;
    constexpr auto kSymbolsPerCodeword = kSymbolsPerFrame / 3U;
    auto first_symbol = std::numeric_limits<std::size_t>::max();
    std::size_t last_symbol = 0;
    for (const auto& rule : rules) {
        const auto zero_based_frame = rule.fec_frame - 1U;
        if (zero_based_frame
            > (std::numeric_limits<std::size_t>::max()
               - rule.codeword_slot * kSymbolsPerCodeword)
                / kSymbolsPerFrame) {
            throw std::overflow_error(
                "source frame displacement output symbol index overflow");
        }
        const auto first_output_symbol =
            zero_based_frame * kSymbolsPerFrame
            + rule.codeword_slot * kSymbolsPerCodeword;
        for (std::size_t symbol = 0; symbol < kSymbolsPerCodeword; ++symbol) {
            const auto base_source = source_symbol_for_output_symbol(
                first_output_symbol + symbol,
                mode,
                phase);
            const auto shifted_source =
                static_cast<std::int64_t>(base_source)
                + rule.source_frame_shift
                    * static_cast<std::int64_t>(kSymbolsPerFrame);
            if (shifted_source < 0) {
                throw std::runtime_error(
                    "source frame displacement selected a negative source symbol");
            }
            const auto source_index = static_cast<std::size_t>(shifted_source);
            first_symbol = std::min(first_symbol, source_index);
            last_symbol = std::max(last_symbol, source_index);
        }
    }
    window.enabled = true;
    window.first_symbol = first_symbol;
    window.last_symbol = last_symbol;
    return window;
}

std::size_t source_frame_displacement_window_symbols(
    const SourceFrameDisplacementWindow& window) {
    if (!window.enabled) {
        return 0;
    }
    return window.last_symbol - window.first_symbol + 1U;
}

void store_source_frame_displacement_history(
    std::span<const float> input_symbols,
    std::size_t first_input_symbol,
    const SourceFrameDisplacementWindow& window,
    std::vector<float>& source_history,
    std::size_t& available_end_symbol) {
    if (!window.enabled) {
        return;
    }
    const auto symbol_count = input_symbols.size() / kFloatsPerSymbol;
    if (symbol_count == 0) {
        return;
    }
    const auto input_end_symbol = first_input_symbol + symbol_count;
    const auto window_end_symbol = window.last_symbol + 1U;
    const auto overlap_start = std::max(first_input_symbol, window.first_symbol);
    const auto overlap_end = std::min(input_end_symbol, window_end_symbol);
    if (overlap_start >= overlap_end) {
        return;
    }
    const auto source_offset = (overlap_start - first_input_symbol) * kFloatsPerSymbol;
    const auto history_offset = (overlap_start - window.first_symbol) * kFloatsPerSymbol;
    const auto float_count = (overlap_end - overlap_start) * kFloatsPerSymbol;
    std::copy_n(
        input_symbols.begin() + static_cast<std::ptrdiff_t>(source_offset),
        float_count,
        source_history.begin() + static_cast<std::ptrdiff_t>(history_offset));
    available_end_symbol = std::max(available_end_symbol, overlap_end);
}

void apply_source_frame_carrier_roll(
    std::span<float> input_symbols,
    std::size_t first_input_symbol,
    const std::vector<SourceFrameCarrierRollRule>& rules,
    std::vector<SourceFrameCarrierRollStats>& stats,
    std::vector<float>& scratch_frame) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = input_symbols.size() / kFloatsPerSymbol;
    if (symbol_count == 0) {
        return;
    }
    if ((first_input_symbol % dtmb::core::kC3780DataSymbols) != 0
        || (symbol_count % dtmb::core::kC3780DataSymbols) != 0) {
        throw std::runtime_error(
            "source frame carrier roll requires C3780-frame-aligned chunks");
    }
    const auto frame_count = symbol_count / dtmb::core::kC3780DataSymbols;
    if (scratch_frame.size() < dtmb::core::kC3780DataSymbols * kFloatsPerSymbol) {
        scratch_frame.resize(dtmb::core::kC3780DataSymbols * kFloatsPerSymbol);
    }
    for (std::size_t local_frame = 0; local_frame < frame_count; ++local_frame) {
        const auto source_frame =
            (first_input_symbol / dtmb::core::kC3780DataSymbols) + local_frame;
        std::size_t matched_rule = rules.size();
        for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
            const auto& rule = rules[rule_index];
            if (source_frame >= rule.first_frame && source_frame <= rule.last_frame) {
                matched_rule = rule_index;
                break;
            }
        }
        if (matched_rule == rules.size()) {
            continue;
        }
        const auto& rule = rules[matched_rule];
        const auto frame_offset =
            local_frame * dtmb::core::kC3780DataSymbols * kFloatsPerSymbol;
        const auto frame_values = input_symbols.subspan(
            frame_offset,
            dtmb::core::kC3780DataSymbols * kFloatsPerSymbol);
        const auto carrier_count =
            static_cast<std::int64_t>(dtmb::core::kC3780DataSymbols);
        auto shift = rule.shift % carrier_count;
        if (shift < 0) {
            shift += carrier_count;
        }
        if (shift != 0) {
            std::copy(frame_values.begin(), frame_values.end(), scratch_frame.begin());
            for (std::size_t carrier = 0; carrier < dtmb::core::kC3780DataSymbols; ++carrier) {
                const auto source_carrier = (
                    carrier + dtmb::core::kC3780DataSymbols
                    - static_cast<std::size_t>(shift))
                    % dtmb::core::kC3780DataSymbols;
                frame_values[carrier * kFloatsPerSymbol] =
                    scratch_frame[source_carrier * kFloatsPerSymbol];
                frame_values[carrier * kFloatsPerSymbol + 1U] =
                    scratch_frame[source_carrier * kFloatsPerSymbol + 1U];
            }
        }
        ++stats[matched_rule].rolled_frames;
        stats[matched_rule].rolled_symbols += dtmb::core::kC3780DataSymbols;
    }
}

void apply_source_frame_displacement(
    std::span<float> deinterleaved_symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const SourceFrameDisplacementWindow& source_window,
    const std::vector<float>& source_history,
    std::size_t source_history_available_end_symbol,
    const std::vector<SourceFrameDisplacementRule>& rules,
    std::vector<SourceFrameDisplacementStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = deinterleaved_symbols.size() / kFloatsPerSymbol;
    const auto source_history_symbols = source_history.size() / kFloatsPerSymbol;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto output_symbol = first_output_symbol + symbol;
        const auto fec_frame =
            output_symbol / dtmb::core::kC3780DataSymbols + 1U;
        const auto carrier = output_symbol % dtmb::core::kC3780DataSymbols;
        const auto codeword_slot =
            carrier / (dtmb::core::kC3780DataSymbols / 3U);
        std::size_t matched_rule = rules.size();
        for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
            const auto& rule = rules[rule_index];
            if (rule.fec_frame == fec_frame
                && rule.codeword_slot == codeword_slot) {
                matched_rule = rule_index;
                break;
            }
        }
        if (matched_rule == rules.size()) {
            continue;
        }
        const auto base_source = source_symbol_for_output_symbol(
            output_symbol,
            mode,
            phase);
        const auto shifted_source =
            static_cast<std::int64_t>(base_source)
            + rules[matched_rule].source_frame_shift
                * static_cast<std::int64_t>(dtmb::core::kC3780DataSymbols);
        if (shifted_source < 0) {
            throw std::runtime_error(
                "source frame displacement selected a negative source symbol");
        }
        const auto source_index = static_cast<std::size_t>(shifted_source);
        if (!source_window.enabled
            || source_index < source_window.first_symbol
            || source_index > source_window.last_symbol) {
            throw std::runtime_error(
                "source frame displacement requires a source symbol outside "
                "the retained source window");
        }
        if (source_index >= source_history_available_end_symbol) {
            throw std::runtime_error(
                "source frame displacement requires an unavailable future source symbol");
        }
        const auto local_source_index = source_index - source_window.first_symbol;
        if (local_source_index >= source_history_symbols) {
            throw std::runtime_error(
                "source frame displacement source window bookkeeping overflow");
        }
        deinterleaved_symbols[symbol * kFloatsPerSymbol] =
            source_history[local_source_index * kFloatsPerSymbol];
        deinterleaved_symbols[symbol * kFloatsPerSymbol + 1U] =
            source_history[local_source_index * kFloatsPerSymbol + 1U];
        ++stats[matched_rule].replaced_symbols;
    }
}

struct AxisAffineFit {
    float real_scale = 1.0F;
    float real_bias = 0.0F;
    float imag_scale = 1.0F;
    float imag_bias = 0.0F;
    bool valid = false;
};

struct LinearAffineFit {
    float real_from_real = 1.0F;
    float real_from_imag = 0.0F;
    float real_bias = 0.0F;
    float imag_from_real = 0.0F;
    float imag_from_imag = 1.0F;
    float imag_bias = 0.0F;
    bool valid = false;
};

struct SourceFrameAxisAffineAccumulator {
    double sum_nearest_real = 0.0;
    double sum_nearest_imag = 0.0;
    double sum_value_real = 0.0;
    double sum_value_imag = 0.0;
    double sum_nearest_real2 = 0.0;
    double sum_nearest_imag2 = 0.0;
    double sum_nearest_value_real = 0.0;
    double sum_nearest_value_imag = 0.0;
    std::size_t reliable_count = 0;
};

struct AxisQuantileFit {
    std::array<float, 8> real_centers{};
    std::array<float, 8> imag_centers{};
    bool valid = false;
};

struct AxisCentroidFit {
    std::array<float, 8> real_centers{};
    std::array<float, 8> imag_centers{};
    bool valid = false;
};

struct QamCentroidFit {
    std::array<std::complex<float>, 64> centroids{};
    std::array<std::uint8_t, 64> valid_centroids{};
    bool valid = false;
};

bool solve_3x3(
    std::array<std::array<double, 4>, 3> matrix,
    std::array<double, 3>& solution) {
    for (std::size_t column = 0; column < 3; ++column) {
        auto pivot = column;
        auto pivot_abs = std::abs(matrix[column][column]);
        for (std::size_t row = column + 1U; row < 3; ++row) {
            const auto candidate_abs = std::abs(matrix[row][column]);
            if (candidate_abs > pivot_abs) {
                pivot = row;
                pivot_abs = candidate_abs;
            }
        }
        if (pivot_abs <= 1.0e-9) {
            return false;
        }
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
        }
        const auto divisor = matrix[column][column];
        for (std::size_t item = column; item < 4; ++item) {
            matrix[column][item] /= divisor;
        }
        for (std::size_t row = 0; row < 3; ++row) {
            if (row == column) {
                continue;
            }
            const auto factor = matrix[row][column];
            for (std::size_t item = column; item < 4; ++item) {
                matrix[row][item] -= factor * matrix[column][item];
            }
        }
    }
    for (std::size_t row = 0; row < 3; ++row) {
        if (!std::isfinite(matrix[row][3])) {
            return false;
        }
        solution[row] = matrix[row][3];
    }
    return true;
}

bool solve_complex_system(
    std::vector<std::vector<std::complex<double>>> matrix,
    std::vector<std::complex<float>>& solution) {
    const auto size = matrix.size();
    if (size == 0) {
        return false;
    }
    for (const auto& row : matrix) {
        if (row.size() != size + 1U) {
            return false;
        }
    }
    for (std::size_t column = 0; column < size; ++column) {
        auto pivot = column;
        auto pivot_abs = std::abs(matrix[column][column]);
        for (std::size_t row = column + 1U; row < size; ++row) {
            const auto candidate_abs = std::abs(matrix[row][column]);
            if (candidate_abs > pivot_abs) {
                pivot = row;
                pivot_abs = candidate_abs;
            }
        }
        if (pivot_abs <= 1.0e-12) {
            return false;
        }
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
        }
        const auto divisor = matrix[column][column];
        for (std::size_t item = column; item <= size; ++item) {
            matrix[column][item] /= divisor;
        }
        for (std::size_t row = 0; row < size; ++row) {
            if (row == column) {
                continue;
            }
            const auto factor = matrix[row][column];
            for (std::size_t item = column; item <= size; ++item) {
                matrix[row][item] -= factor * matrix[column][item];
            }
        }
    }
    solution.assign(size, {0.0F, 0.0F});
    for (std::size_t row = 0; row < size; ++row) {
        const auto value = matrix[row][size];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            return false;
        }
        solution[row] = {
            static_cast<float>(value.real()),
            static_cast<float>(value.imag())};
    }
    return true;
}

AxisAffineFit fit_source_frame_axis_affine_from_accumulator(
    const SourceFrameAxisAffineAccumulator& accumulator,
    std::size_t min_symbols) {
    if (accumulator.reliable_count < min_symbols) {
        return {};
    }
    const auto count = static_cast<double>(accumulator.reliable_count);
    const auto real_denominator = accumulator.sum_nearest_real2
        - accumulator.sum_nearest_real * accumulator.sum_nearest_real / count;
    const auto imag_denominator = accumulator.sum_nearest_imag2
        - accumulator.sum_nearest_imag * accumulator.sum_nearest_imag / count;
    if (std::abs(real_denominator) <= 1.0e-9
        || std::abs(imag_denominator) <= 1.0e-9) {
        return {};
    }
    AxisAffineFit fit;
    const auto real_scale = (
        accumulator.sum_nearest_value_real
        - accumulator.sum_nearest_real * accumulator.sum_value_real / count)
        / real_denominator;
    const auto imag_scale = (
        accumulator.sum_nearest_value_imag
        - accumulator.sum_nearest_imag * accumulator.sum_value_imag / count)
        / imag_denominator;
    const auto real_bias = (
        accumulator.sum_value_real - real_scale * accumulator.sum_nearest_real) / count;
    const auto imag_bias = (
        accumulator.sum_value_imag - imag_scale * accumulator.sum_nearest_imag) / count;
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

void add_source_frame_axis_affine_sample(
    SourceFrameAxisAffineAccumulator& accumulator,
    const std::complex<float>& value) {
    const std::complex<float> nearest{
        nearest_qam64_level(value.real()),
        nearest_qam64_level(value.imag())};
    ++accumulator.reliable_count;
    accumulator.sum_nearest_real += nearest.real();
    accumulator.sum_nearest_imag += nearest.imag();
    accumulator.sum_value_real += value.real();
    accumulator.sum_value_imag += value.imag();
    accumulator.sum_nearest_real2 += static_cast<double>(nearest.real()) * nearest.real();
    accumulator.sum_nearest_imag2 += static_cast<double>(nearest.imag()) * nearest.imag();
    accumulator.sum_nearest_value_real +=
        static_cast<double>(nearest.real()) * value.real();
    accumulator.sum_nearest_value_imag +=
        static_cast<double>(nearest.imag()) * value.imag();
}

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
    std::vector<std::vector<std::size_t>> rule_indices_by_branch(spec.branch_count);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rule_indices_by_branch[rules[rule_index].branch].push_back(rule_index);
        auto& row = stats[rule_index];
        ++row.chunks_seen;
    }

    std::vector<SourceFrameAxisAffineAccumulator> accumulators(rules.size());
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_branch = (source_symbol % spec.branch_count + phase)
            % spec.branch_count;
        const auto& branch_rules = rule_indices_by_branch[source_branch];
        if (branch_rules.empty()) {
            continue;
        }
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        std::size_t matched_rule = rules.size();
        for (const auto rule_index : branch_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame >= rule.first_frame && source_frame <= rule.last_frame) {
                matched_rule = rule_index;
                break;
            }
        }
        if (matched_rule == rules.size()) {
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
        if (relative_error <= options.reliability_threshold) {
            add_source_frame_axis_affine_sample(accumulators[matched_rule], value);
        }
    }

    std::vector<AxisAffineFit> fits(rules.size());
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        auto& row = stats[rule_index];
        row.reliable_symbols += accumulators[rule_index].reliable_count;
        fits[rule_index] = fit_source_frame_axis_affine_from_accumulator(
            accumulators[rule_index],
            options.min_symbols);
        const auto& fit = fits[rule_index];
        if (!fit.valid) {
            continue;
        }
        row.real_scale_sum += fit.real_scale;
        row.real_bias_sum += fit.real_bias;
        row.imag_scale_sum += fit.imag_scale;
        row.imag_bias_sum += fit.imag_bias;
        ++row.applied_chunks;
    }

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_branch = (source_symbol % spec.branch_count + phase)
            % spec.branch_count;
        const auto& branch_rules = rule_indices_by_branch[source_branch];
        if (branch_rules.empty()) {
            continue;
        }
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        std::size_t matched_rule = rules.size();
        for (const auto rule_index : branch_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame >= rule.first_frame && source_frame <= rule.last_frame) {
                matched_rule = rule_index;
                break;
            }
        }
        if (matched_rule == rules.size() || !fits[matched_rule].valid) {
            continue;
        }
        const auto& fit = fits[matched_rule];
        const auto real = symbols[symbol * kFloatsPerSymbol];
        const auto imag = symbols[symbol * kFloatsPerSymbol + 1U];
        symbols[symbol * kFloatsPerSymbol] = (real - fit.real_bias) / fit.real_scale;
        symbols[symbol * kFloatsPerSymbol + 1U] = (imag - fit.imag_bias) / fit.imag_scale;
        ++stats[matched_rule].corrected_symbols;
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

void apply_source_frame_fixed_complex_gain(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const std::vector<SourceFrameFixedComplexGainRule>& rules,
    std::vector<SourceFrameFixedComplexGainStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto spec = dtmb::core::symbol_interleaver_spec(mode);
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rule_indices_by_branch(spec.branch_count);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rule_indices_by_branch[rules[rule_index].branch].push_back(rule_index);
    }
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_branch = (source_symbol % spec.branch_count + phase)
            % spec.branch_count;
        const auto& branch_rules = rule_indices_by_branch[source_branch];
        if (branch_rules.empty()) {
            continue;
        }
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        for (const auto rule_index : branch_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const auto corrected = value / rule.gain;
            symbols[symbol * kFloatsPerSymbol] = corrected.real();
            symbols[symbol * kFloatsPerSymbol + 1U] = corrected.imag();
            ++stats[rule_index].corrected_symbols;
            break;
        }
    }
}

std::vector<std::complex<float>> read_source_carrier_fixed_complex_gains(
    const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "failed to open source carrier fixed complex gains: " + path);
    }
    std::vector<float> packed(dtmb::core::kC3780DataSymbols * 2U);
    input.read(
        reinterpret_cast<char*>(packed.data()),
        static_cast<std::streamsize>(packed.size() * sizeof(float)));
    if (input.gcount()
        != static_cast<std::streamsize>(packed.size() * sizeof(float))) {
        throw std::runtime_error(
            "source carrier fixed complex gains must contain exactly 3744 CF32 values");
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error(
            "source carrier fixed complex gains contain trailing bytes");
    }
    std::vector<std::complex<float>> gains(dtmb::core::kC3780DataSymbols);
    for (std::size_t carrier = 0; carrier < gains.size(); ++carrier) {
        gains[carrier] = {
            packed[carrier * 2U],
            packed[carrier * 2U + 1U]};
        if (!std::isfinite(gains[carrier].real())
            || !std::isfinite(gains[carrier].imag())
            || std::abs(gains[carrier]) <= 1.0e-9F) {
            throw std::runtime_error(
                "source carrier fixed complex gains must be finite and non-zero");
        }
    }
    return gains;
}

std::size_t apply_source_carrier_fixed_complex_gains(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    std::span<const std::complex<float>> gains) {
    if (gains.empty()) {
        return 0;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::size_t corrected_symbols = 0;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const std::complex<float> value{
            symbols[symbol * kFloatsPerSymbol],
            symbols[symbol * kFloatsPerSymbol + 1U]};
        const auto corrected = value / gains[source_carrier];
        symbols[symbol * kFloatsPerSymbol] = corrected.real();
        symbols[symbol * kFloatsPerSymbol + 1U] = corrected.imag();
        ++corrected_symbols;
    }
    return corrected_symbols;
}

bool source_carrier_symbol_gain_ranges_overlap(
    const SourceCarrierSymbolGainRule& left,
    const SourceCarrierSymbolGainRule& right) noexcept {
    return left.carrier == right.carrier && left.first_frame <= right.last_frame
        && right.first_frame <= left.last_frame;
}

bool source_carrier_symbol_gain_has_overlaps(
    const std::vector<SourceCarrierSymbolGainRule>& rules) {
    for (std::size_t left = 0; left < rules.size(); ++left) {
        for (std::size_t right = left + 1U; right < rules.size(); ++right) {
            if (source_carrier_symbol_gain_ranges_overlap(rules[left], rules[right])) {
                return true;
            }
        }
    }
    return false;
}

void apply_source_carrier_symbol_gain_sequential(
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
            if (!source_carrier_fit_uses_frame(
                    options,
                    source_carrier,
                    source_frame,
                    options.carrier_symbol_gain_fit_exclude_source_frame_ranges,
                    options
                        .carrier_symbol_gain_fit_exclude_source_carrier_frame_ranges)) {
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

bool source_carrier_axis_affine_ranges_overlap(
    const SourceCarrierAxisAffineRule& left,
    const SourceCarrierAxisAffineRule& right) noexcept {
    return left.carrier == right.carrier && left.first_frame <= right.last_frame
        && right.first_frame <= left.last_frame;
}

bool source_carrier_linear_affine_ranges_overlap(
    const SourceCarrierLinearAffineRule& left,
    const SourceCarrierLinearAffineRule& right) noexcept {
    return left.carrier == right.carrier && left.first_frame <= right.last_frame
        && right.first_frame <= left.last_frame;
}

AxisAffineFit fit_axis_affine_from_sums(
    std::size_t reliable_count,
    double sum_nearest_real,
    double sum_nearest_imag,
    double sum_value_real,
    double sum_value_imag,
    double sum_nearest_real2,
    double sum_nearest_imag2,
    double sum_nearest_value_real,
    double sum_nearest_value_imag,
    const BranchGainOptions& options) {
    if (reliable_count < options.min_symbols) {
        return {};
    }
    const auto count = static_cast<double>(reliable_count);
    const auto real_denominator =
        sum_nearest_real2 - sum_nearest_real * sum_nearest_real / count;
    const auto imag_denominator =
        sum_nearest_imag2 - sum_nearest_imag * sum_nearest_imag / count;
    if (std::abs(real_denominator) <= 1.0e-9
        || std::abs(imag_denominator) <= 1.0e-9) {
        return {};
    }
    const auto real_scale =
        (sum_nearest_value_real - sum_nearest_real * sum_value_real / count)
        / real_denominator;
    const auto imag_scale =
        (sum_nearest_value_imag - sum_nearest_imag * sum_value_imag / count)
        / imag_denominator;
    const auto real_bias = (sum_value_real - real_scale * sum_nearest_real) / count;
    const auto imag_bias = (sum_value_imag - imag_scale * sum_nearest_imag) / count;
    if (!std::isfinite(real_scale) || !std::isfinite(imag_scale)
        || !std::isfinite(real_bias) || !std::isfinite(imag_bias)
        || std::abs(real_scale) <= 1.0e-9 || std::abs(imag_scale) <= 1.0e-9) {
        return {};
    }
    AxisAffineFit fit;
    fit.real_scale = static_cast<float>(real_scale);
    fit.imag_scale = static_cast<float>(imag_scale);
    fit.real_bias = static_cast<float>(real_bias);
    fit.imag_bias = static_cast<float>(imag_bias);
    fit.valid = true;
    return fit;
}

LinearAffineFit fit_linear_affine_from_sums(
    std::size_t reliable_count,
    double sum_nearest_real,
    double sum_nearest_imag,
    double sum_value_real,
    double sum_value_imag,
    double sum_nearest_real2,
    double sum_nearest_imag2,
    double sum_nearest_real_imag,
    double sum_nearest_real_value_real,
    double sum_nearest_imag_value_real,
    double sum_nearest_real_value_imag,
    double sum_nearest_imag_value_imag,
    const BranchGainOptions& options) {
    if (reliable_count < options.min_symbols) {
        return {};
    }
    const auto count = static_cast<double>(reliable_count);
    const std::array<std::array<double, 3>, 3> normal{{
        {sum_nearest_real2, sum_nearest_real_imag, sum_nearest_real},
        {sum_nearest_real_imag, sum_nearest_imag2, sum_nearest_imag},
        {sum_nearest_real, sum_nearest_imag, count},
    }};

    std::array<double, 3> real_solution{};
    std::array<double, 3> imag_solution{};
    std::array<std::array<double, 4>, 3> real_matrix{{
        {normal[0][0], normal[0][1], normal[0][2], sum_nearest_real_value_real},
        {normal[1][0], normal[1][1], normal[1][2], sum_nearest_imag_value_real},
        {normal[2][0], normal[2][1], normal[2][2], sum_value_real},
    }};
    std::array<std::array<double, 4>, 3> imag_matrix{{
        {normal[0][0], normal[0][1], normal[0][2], sum_nearest_real_value_imag},
        {normal[1][0], normal[1][1], normal[1][2], sum_nearest_imag_value_imag},
        {normal[2][0], normal[2][1], normal[2][2], sum_value_imag},
    }};
    if (!solve_3x3(real_matrix, real_solution)
        || !solve_3x3(imag_matrix, imag_solution)) {
        return {};
    }
    const auto determinant =
        real_solution[0] * imag_solution[1] - real_solution[1] * imag_solution[0];
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-9) {
        return {};
    }
    for (const auto value :
         {real_solution[0], real_solution[1], real_solution[2],
          imag_solution[0], imag_solution[1], imag_solution[2]}) {
        if (!std::isfinite(value)) {
            return {};
        }
    }
    LinearAffineFit fit;
    fit.real_from_real = static_cast<float>(real_solution[0]);
    fit.real_from_imag = static_cast<float>(real_solution[1]);
    fit.real_bias = static_cast<float>(real_solution[2]);
    fit.imag_from_real = static_cast<float>(imag_solution[0]);
    fit.imag_from_imag = static_cast<float>(imag_solution[1]);
    fit.imag_bias = static_cast<float>(imag_solution[2]);
    fit.valid = true;
    return fit;
}

void apply_source_carrier_axis_affine(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierAxisAffineRule>& rules,
    std::vector<SourceCarrierAxisAffineStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<double> sum_nearest_real(rules.size(), 0.0);
    std::vector<double> sum_nearest_imag(rules.size(), 0.0);
    std::vector<double> sum_value_real(rules.size(), 0.0);
    std::vector<double> sum_value_imag(rules.size(), 0.0);
    std::vector<double> sum_nearest_real2(rules.size(), 0.0);
    std::vector<double> sum_nearest_imag2(rules.size(), 0.0);
    std::vector<double> sum_nearest_value_real(rules.size(), 0.0);
    std::vector<double> sum_nearest_value_imag(rules.size(), 0.0);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        if (!source_carrier_fit_uses_frame(
                options,
                source_carrier,
                source_frame,
                options.carrier_linear_affine_fit_exclude_source_frame_ranges,
                options
                    .carrier_linear_affine_fit_exclude_source_carrier_frame_ranges)) {
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
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            ++reliable_counts[rule_index];
            sum_nearest_real[rule_index] += nearest.real();
            sum_nearest_imag[rule_index] += nearest.imag();
            sum_value_real[rule_index] += value.real();
            sum_value_imag[rule_index] += value.imag();
            sum_nearest_real2[rule_index] +=
                static_cast<double>(nearest.real()) * nearest.real();
            sum_nearest_imag2[rule_index] +=
                static_cast<double>(nearest.imag()) * nearest.imag();
            sum_nearest_value_real[rule_index] +=
                static_cast<double>(nearest.real()) * value.real();
            sum_nearest_value_imag[rule_index] +=
                static_cast<double>(nearest.imag()) * value.imag();
        }
    }

    std::vector<AxisAffineFit> fits(rules.size());
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        stats[rule_index].reliable_symbols += reliable_counts[rule_index];
        fits[rule_index] = fit_axis_affine_from_sums(
            reliable_counts[rule_index],
            sum_nearest_real[rule_index],
            sum_nearest_imag[rule_index],
            sum_value_real[rule_index],
            sum_value_imag[rule_index],
            sum_nearest_real2[rule_index],
            sum_nearest_imag2[rule_index],
            sum_nearest_value_real[rule_index],
            sum_nearest_value_imag[rule_index],
            options);
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& fit = fits[rule_index];
            if (!fit.valid) {
                continue;
            }
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto real = symbols[symbol * kFloatsPerSymbol];
            const auto imag = symbols[symbol * kFloatsPerSymbol + 1U];
            symbols[symbol * kFloatsPerSymbol] = (real - fit.real_bias) / fit.real_scale;
            symbols[symbol * kFloatsPerSymbol + 1U] = (imag - fit.imag_bias) / fit.imag_scale;
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        const auto& fit = fits[rule_index];
        if (!fit.valid) {
            continue;
        }
        auto& row = stats[rule_index];
        row.corrected_symbols += corrected_counts[rule_index];
        row.real_scale_sum += fit.real_scale;
        row.real_bias_sum += fit.real_bias;
        row.imag_scale_sum += fit.imag_scale;
        row.imag_bias_sum += fit.imag_bias;
        ++row.applied_chunks;
    }
}

void apply_source_carrier_linear_affine(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierLinearAffineRule>& rules,
    std::vector<SourceCarrierLinearAffineStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<double> sum_nearest_real(rules.size(), 0.0);
    std::vector<double> sum_nearest_imag(rules.size(), 0.0);
    std::vector<double> sum_value_real(rules.size(), 0.0);
    std::vector<double> sum_value_imag(rules.size(), 0.0);
    std::vector<double> sum_nearest_real2(rules.size(), 0.0);
    std::vector<double> sum_nearest_imag2(rules.size(), 0.0);
    std::vector<double> sum_nearest_real_imag(rules.size(), 0.0);
    std::vector<double> sum_nearest_real_value_real(rules.size(), 0.0);
    std::vector<double> sum_nearest_imag_value_real(rules.size(), 0.0);
    std::vector<double> sum_nearest_real_value_imag(rules.size(), 0.0);
    std::vector<double> sum_nearest_imag_value_imag(rules.size(), 0.0);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        if (!source_carrier_fit_uses_frame(
                options,
                source_carrier,
                source_frame,
                options.carrier_linear_affine_fit_exclude_source_frame_ranges,
                options
                    .carrier_linear_affine_fit_exclude_source_carrier_frame_ranges)) {
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
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto nearest_real = static_cast<double>(nearest.real());
            const auto nearest_imag = static_cast<double>(nearest.imag());
            const auto value_real = static_cast<double>(value.real());
            const auto value_imag = static_cast<double>(value.imag());
            ++reliable_counts[rule_index];
            sum_nearest_real[rule_index] += nearest_real;
            sum_nearest_imag[rule_index] += nearest_imag;
            sum_value_real[rule_index] += value_real;
            sum_value_imag[rule_index] += value_imag;
            sum_nearest_real2[rule_index] += nearest_real * nearest_real;
            sum_nearest_imag2[rule_index] += nearest_imag * nearest_imag;
            sum_nearest_real_imag[rule_index] += nearest_real * nearest_imag;
            sum_nearest_real_value_real[rule_index] += nearest_real * value_real;
            sum_nearest_imag_value_real[rule_index] += nearest_imag * value_real;
            sum_nearest_real_value_imag[rule_index] += nearest_real * value_imag;
            sum_nearest_imag_value_imag[rule_index] += nearest_imag * value_imag;
        }
    }

    std::vector<LinearAffineFit> fits(rules.size());
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        stats[rule_index].reliable_symbols += reliable_counts[rule_index];
        fits[rule_index] = fit_linear_affine_from_sums(
            reliable_counts[rule_index],
            sum_nearest_real[rule_index],
            sum_nearest_imag[rule_index],
            sum_value_real[rule_index],
            sum_value_imag[rule_index],
            sum_nearest_real2[rule_index],
            sum_nearest_imag2[rule_index],
            sum_nearest_real_imag[rule_index],
            sum_nearest_real_value_real[rule_index],
            sum_nearest_imag_value_real[rule_index],
            sum_nearest_real_value_imag[rule_index],
            sum_nearest_imag_value_imag[rule_index],
            options);
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& fit = fits[rule_index];
            if (!fit.valid) {
                continue;
            }
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto real =
                static_cast<double>(symbols[symbol * kFloatsPerSymbol]) - fit.real_bias;
            const auto imag =
                static_cast<double>(symbols[symbol * kFloatsPerSymbol + 1U])
                - fit.imag_bias;
            const auto determinant =
                static_cast<double>(fit.real_from_real) * fit.imag_from_imag
                - static_cast<double>(fit.real_from_imag) * fit.imag_from_real;
            const auto corrected_real =
                (static_cast<double>(fit.imag_from_imag) * real
                 - static_cast<double>(fit.real_from_imag) * imag)
                / determinant;
            const auto corrected_imag =
                (-static_cast<double>(fit.imag_from_real) * real
                 + static_cast<double>(fit.real_from_real) * imag)
                / determinant;
            symbols[symbol * kFloatsPerSymbol] = static_cast<float>(corrected_real);
            symbols[symbol * kFloatsPerSymbol + 1U] = static_cast<float>(corrected_imag);
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        const auto& fit = fits[rule_index];
        if (!fit.valid) {
            continue;
        }
        auto& row = stats[rule_index];
        row.corrected_symbols += corrected_counts[rule_index];
        row.real_from_real_sum += fit.real_from_real;
        row.real_from_imag_sum += fit.real_from_imag;
        row.real_bias_sum += fit.real_bias;
        row.imag_from_real_sum += fit.imag_from_real;
        row.imag_from_imag_sum += fit.imag_from_imag;
        row.imag_bias_sum += fit.imag_bias;
        ++row.applied_chunks;
    }
}

bool source_carrier_axis_quantile_ranges_overlap(
    const SourceCarrierAxisQuantileRule& left,
    const SourceCarrierAxisQuantileRule& right) noexcept {
    return left.carrier == right.carrier && left.first_frame <= right.last_frame
        && right.first_frame <= left.last_frame;
}

bool source_carrier_axis_centroid_ranges_overlap(
    const SourceCarrierAxisCentroidRule& left,
    const SourceCarrierAxisCentroidRule& right) noexcept {
    return left.carrier == right.carrier && left.first_frame <= right.last_frame
        && right.first_frame <= left.last_frame;
}

bool source_carrier_qam_centroid_ranges_overlap(
    const SourceCarrierQamCentroidRule& left,
    const SourceCarrierQamCentroidRule& right) noexcept {
    return left.carrier == right.carrier && left.first_frame <= right.last_frame
        && right.first_frame <= left.last_frame;
}

bool axis_quantile_centers(
    std::vector<float> values,
    std::array<float, 8>& centers) {
    if (values.size() < centers.size()) {
        return false;
    }
    std::sort(values.begin(), values.end());
    for (std::size_t level = 0; level < centers.size(); ++level) {
        const auto first = level * values.size() / centers.size();
        const auto last = (level + 1U) * values.size() / centers.size();
        if (last <= first) {
            return false;
        }
        double sum = 0.0;
        for (std::size_t index = first; index < last; ++index) {
            sum += static_cast<double>(values[index]);
        }
        centers[level] = static_cast<float>(sum / static_cast<double>(last - first));
        if (!std::isfinite(centers[level])) {
            return false;
        }
        if (level != 0 && centers[level] <= centers[level - 1U] + 1.0e-6F) {
            return false;
        }
    }
    return true;
}

float remap_axis_quantile(float value, const std::array<float, 8>& centers) {
    if (value <= centers.front()) {
        const auto slope = (kQam64Levels[1] - kQam64Levels[0])
            / (centers[1] - centers[0]);
        return kQam64Levels[0] + (value - centers[0]) * slope;
    }
    if (value >= centers.back()) {
        const auto slope = (kQam64Levels[7] - kQam64Levels[6])
            / (centers[7] - centers[6]);
        return kQam64Levels[7] + (value - centers[7]) * slope;
    }
    const auto upper = std::upper_bound(centers.begin(), centers.end(), value);
    const auto high = static_cast<std::size_t>(upper - centers.begin());
    const auto low = high - 1U;
    const auto span = centers[high] - centers[low];
    const auto fraction = (value - centers[low]) / span;
    return kQam64Levels[low] + fraction * (kQam64Levels[high] - kQam64Levels[low]);
}

bool axis_centroid_centers(
    const std::array<double, 8>& sums,
    const std::array<std::size_t, 8>& counts,
    std::array<float, 8>& centers) {
    for (std::size_t level = 0; level < centers.size(); ++level) {
        if (counts[level] == 0U) {
            return false;
        }
        centers[level] = static_cast<float>(
            sums[level] / static_cast<double>(counts[level]));
        if (!std::isfinite(centers[level])) {
            return false;
        }
        if (level != 0 && centers[level] <= centers[level - 1U] + 1.0e-6F) {
            return false;
        }
    }
    return true;
}

float median_value(std::vector<float>& values) {
    const auto middle = values.size() / 2U;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const auto high = values[middle];
    if ((values.size() & 1U) != 0U) {
        return high;
    }
    std::nth_element(values.begin(), values.begin() + middle - 1U, values.begin() + middle);
    return 0.5F * (values[middle - 1U] + high);
}

bool axis_centroid_median_centers(
    std::array<std::vector<float>, 8>& values,
    std::array<float, 8>& centers) {
    for (std::size_t level = 0; level < centers.size(); ++level) {
        if (values[level].empty()) {
            return false;
        }
        centers[level] = median_value(values[level]);
        if (!std::isfinite(centers[level])) {
            return false;
        }
        if (level != 0 && centers[level] <= centers[level - 1U] + 1.0e-6F) {
            return false;
        }
    }
    return true;
}

void apply_source_carrier_axis_centroid(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierAxisCentroidRule>& rules,
    std::vector<SourceCarrierAxisCentroidStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<std::array<double, 8>> real_sums(rules.size());
    std::vector<std::array<double, 8>> imag_sums(rules.size());
    std::vector<std::array<std::size_t, 8>> real_counts(rules.size());
    std::vector<std::array<std::size_t, 8>> imag_counts(rules.size());
    std::vector<std::size_t> matched_counts(rules.size(), 0U);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        const auto real = symbols[symbol * kFloatsPerSymbol];
        const auto imag = symbols[symbol * kFloatsPerSymbol + 1U];
        const auto real_level = nearest_qam64_level_index(real);
        const auto imag_level = nearest_qam64_level_index(imag);
        const std::complex<float> nearest{
            kQam64Levels[real_level],
            kQam64Levels[imag_level]};
        const std::complex<float> value{real, imag};
        const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
        const auto relative_error = std::abs(value - nearest) / nearest_power;
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            ++matched_counts[rule_index];
            if (!source_carrier_fit_uses_frame(
                    options,
                    source_carrier,
                    source_frame,
                    options.carrier_axis_centroid_fit_exclude_source_frame_ranges,
                    options
                        .carrier_axis_centroid_fit_exclude_source_carrier_frame_ranges)) {
                continue;
            }
            if (relative_error > options.reliability_threshold) {
                continue;
            }
            ++reliable_counts[rule_index];
            real_sums[rule_index][real_level] += static_cast<double>(real);
            imag_sums[rule_index][imag_level] += static_cast<double>(imag);
            ++real_counts[rule_index][real_level];
            ++imag_counts[rule_index][imag_level];
            break;
        }
    }

    std::vector<AxisCentroidFit> fits(rules.size());
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        auto& row = stats[rule_index];
        row.matched_symbols += matched_counts[rule_index];
        row.reliable_symbols += reliable_counts[rule_index];
        if (reliable_counts[rule_index] < options.min_symbols) {
            continue;
        }
        auto& fit = fits[rule_index];
        fit.valid =
            axis_centroid_centers(
                real_sums[rule_index],
                real_counts[rule_index],
                fit.real_centers)
            && axis_centroid_centers(
                imag_sums[rule_index],
                imag_counts[rule_index],
                fit.imag_centers);
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto& fit = fits[rule_index];
            if (!fit.valid) {
                continue;
            }
            symbols[symbol * kFloatsPerSymbol] =
                remap_axis_quantile(symbols[symbol * kFloatsPerSymbol], fit.real_centers);
            symbols[symbol * kFloatsPerSymbol + 1U] = remap_axis_quantile(
                symbols[symbol * kFloatsPerSymbol + 1U],
                fit.imag_centers);
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        if (!fits[rule_index].valid) {
            continue;
        }
        stats[rule_index].corrected_symbols += corrected_counts[rule_index];
        ++stats[rule_index].applied_chunks;
    }
}

void apply_source_carrier_axis_centroid_median(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierAxisCentroidRule>& rules,
    std::vector<SourceCarrierAxisCentroidStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<std::array<std::vector<float>, 8>> real_values(rules.size());
    std::vector<std::array<std::vector<float>, 8>> imag_values(rules.size());
    std::vector<std::size_t> matched_counts(rules.size(), 0U);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        const auto real = symbols[symbol * kFloatsPerSymbol];
        const auto imag = symbols[symbol * kFloatsPerSymbol + 1U];
        const auto real_level = nearest_qam64_level_index(real);
        const auto imag_level = nearest_qam64_level_index(imag);
        const std::complex<float> nearest{
            kQam64Levels[real_level],
            kQam64Levels[imag_level]};
        const std::complex<float> value{real, imag};
        const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
        const auto relative_error = std::abs(value - nearest) / nearest_power;
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            ++matched_counts[rule_index];
            if (!source_carrier_fit_uses_frame(
                    options,
                    source_carrier,
                    source_frame,
                    options.carrier_axis_centroid_fit_exclude_source_frame_ranges,
                    options
                        .carrier_axis_centroid_fit_exclude_source_carrier_frame_ranges)) {
                continue;
            }
            if (relative_error > options.reliability_threshold) {
                continue;
            }
            ++reliable_counts[rule_index];
            real_values[rule_index][real_level].push_back(real);
            imag_values[rule_index][imag_level].push_back(imag);
            break;
        }
    }

    std::vector<AxisCentroidFit> fits(rules.size());
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        auto& row = stats[rule_index];
        row.matched_symbols += matched_counts[rule_index];
        row.reliable_symbols += reliable_counts[rule_index];
        if (reliable_counts[rule_index] < options.min_symbols) {
            continue;
        }
        auto& fit = fits[rule_index];
        fit.valid =
            axis_centroid_median_centers(
                real_values[rule_index],
                fit.real_centers)
            && axis_centroid_median_centers(
                imag_values[rule_index],
                fit.imag_centers);
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto& fit = fits[rule_index];
            if (!fit.valid) {
                continue;
            }
            symbols[symbol * kFloatsPerSymbol] =
                remap_axis_quantile(symbols[symbol * kFloatsPerSymbol], fit.real_centers);
            symbols[symbol * kFloatsPerSymbol + 1U] = remap_axis_quantile(
                symbols[symbol * kFloatsPerSymbol + 1U],
                fit.imag_centers);
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        if (!fits[rule_index].valid) {
            continue;
        }
        stats[rule_index].corrected_symbols += corrected_counts[rule_index];
        ++stats[rule_index].applied_chunks;
    }
}

void apply_source_carrier_qam_centroid(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierQamCentroidRule>& rules,
    std::vector<SourceCarrierQamCentroidStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<std::array<double, 64>> real_sums(rules.size());
    std::vector<std::array<double, 64>> imag_sums(rules.size());
    std::vector<std::array<std::size_t, 64>> point_counts(rules.size());
    std::vector<std::size_t> matched_counts(rules.size(), 0U);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        const auto real = symbols[symbol * kFloatsPerSymbol];
        const auto imag = symbols[symbol * kFloatsPerSymbol + 1U];
        const auto real_level = nearest_qam64_level_index(real);
        const auto imag_level = nearest_qam64_level_index(imag);
        const auto point_index = qam64_point_index(real_level, imag_level);
        const auto nearest = qam64_point_from_index(point_index);
        const std::complex<float> value{real, imag};
        const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
        const auto relative_error = std::abs(value - nearest) / nearest_power;
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            ++matched_counts[rule_index];
            if (!source_carrier_fit_uses_frame(
                    options,
                    source_carrier,
                    source_frame,
                    options.carrier_symbol_gain_fit_exclude_source_frame_ranges,
                    options
                        .carrier_symbol_gain_fit_exclude_source_carrier_frame_ranges)) {
                continue;
            }
            if (relative_error > options.reliability_threshold) {
                continue;
            }
            ++reliable_counts[rule_index];
            real_sums[rule_index][point_index] += static_cast<double>(real);
            imag_sums[rule_index][point_index] += static_cast<double>(imag);
            ++point_counts[rule_index][point_index];
            break;
        }
    }

    std::vector<QamCentroidFit> fits(rules.size());
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        auto& row = stats[rule_index];
        row.matched_symbols += matched_counts[rule_index];
        row.reliable_symbols += reliable_counts[rule_index];
        if (reliable_counts[rule_index] < options.min_symbols) {
            continue;
        }
        auto& fit = fits[rule_index];
        std::size_t valid_points = 0;
        for (std::size_t point_index = 0; point_index < point_counts[rule_index].size();
             ++point_index) {
            const auto count = point_counts[rule_index][point_index];
            if (count == 0U) {
                continue;
            }
            const auto real =
                static_cast<float>(real_sums[rule_index][point_index] / count);
            const auto imag =
                static_cast<float>(imag_sums[rule_index][point_index] / count);
            if (!std::isfinite(real) || !std::isfinite(imag)) {
                continue;
            }
            fit.centroids[point_index] = {real, imag};
            fit.valid_centroids[point_index] = 1U;
            ++valid_points;
        }
        if (valid_points == 0U) {
            continue;
        }
        fit.valid = true;
        row.centroid_points += valid_points;
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto& fit = fits[rule_index];
            if (!fit.valid) {
                continue;
            }
            const auto real = symbols[symbol * kFloatsPerSymbol];
            const auto imag = symbols[symbol * kFloatsPerSymbol + 1U];
            const auto real_level = nearest_qam64_level_index(real);
            const auto imag_level = nearest_qam64_level_index(imag);
            const auto point_index = qam64_point_index(real_level, imag_level);
            if (fit.valid_centroids[point_index] == 0U) {
                continue;
            }
            const auto nearest = qam64_point_from_index(point_index);
            const auto centroid = fit.centroids[point_index];
            symbols[symbol * kFloatsPerSymbol] = real + nearest.real() - centroid.real();
            symbols[symbol * kFloatsPerSymbol + 1U] =
                imag + nearest.imag() - centroid.imag();
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        if (!fits[rule_index].valid) {
            continue;
        }
        stats[rule_index].corrected_symbols += corrected_counts[rule_index];
        ++stats[rule_index].applied_chunks;
    }
}

void apply_source_carrier_neighbor_leakage(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierNeighborLeakageRule>& rules,
    std::vector<SourceCarrierNeighborLeakageStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::pair<std::size_t, std::size_t>> source_to_symbol;
    source_to_symbol.reserve(symbol_count);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        source_to_symbol.emplace_back(
            source_symbol_for_output_symbol(first_output_symbol + symbol, mode, phase),
            symbol);
    }
    std::sort(source_to_symbol.begin(), source_to_symbol.end());
    const auto find_symbol = [&](std::size_t source_symbol) -> std::size_t {
        const auto iter = std::lower_bound(
            source_to_symbol.begin(),
            source_to_symbol.end(),
            std::pair<std::size_t, std::size_t>{source_symbol, 0U});
        if (iter == source_to_symbol.end() || iter->first != source_symbol) {
            return std::numeric_limits<std::size_t>::max();
        }
        return iter->second;
    };

    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<std::complex<float>> numerators(rules.size(), {0.0F, 0.0F});
    std::vector<float> denominators(rules.size(), 0.0F);
    std::vector<std::size_t> matched_counts(rules.size(), 0U);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    for (const auto& [source_symbol, local_symbol] : source_to_symbol) {
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto neighbor_carrier =
                static_cast<std::int64_t>(source_carrier) + rule.offset;
            if (neighbor_carrier < 0
                || neighbor_carrier >= static_cast<std::int64_t>(
                    dtmb::core::kC3780DataSymbols)) {
                continue;
            }
            const auto neighbor_source =
                source_frame * dtmb::core::kC3780DataSymbols
                + static_cast<std::size_t>(neighbor_carrier);
            const auto neighbor_symbol = find_symbol(neighbor_source);
            if (neighbor_symbol == std::numeric_limits<std::size_t>::max()) {
                continue;
            }
            ++matched_counts[rule_index];
            if (!source_carrier_fit_uses_frame(
                    options,
                    source_carrier,
                    source_frame,
                    options.carrier_symbol_gain_fit_exclude_source_frame_ranges,
                    options
                        .carrier_symbol_gain_fit_exclude_source_carrier_frame_ranges)) {
                continue;
            }

            const std::complex<float> value{
                symbols[local_symbol * kFloatsPerSymbol],
                symbols[local_symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> nearest{
                nearest_qam64_level(value.real()),
                nearest_qam64_level(value.imag())};
            const std::complex<float> neighbor_value{
                symbols[neighbor_symbol * kFloatsPerSymbol],
                symbols[neighbor_symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> neighbor_nearest{
                nearest_qam64_level(neighbor_value.real()),
                nearest_qam64_level(neighbor_value.imag())};
            const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
            const auto neighbor_power = std::max(std::abs(neighbor_nearest), 1.0e-6F);
            const auto relative_error = std::abs(value - nearest) / nearest_power;
            const auto neighbor_relative_error =
                std::abs(neighbor_value - neighbor_nearest) / neighbor_power;
            if (relative_error > options.reliability_threshold
                || neighbor_relative_error > options.reliability_threshold) {
                continue;
            }
            numerators[rule_index] += std::conj(neighbor_nearest) * (value - nearest);
            denominators[rule_index] += std::norm(neighbor_nearest);
            ++reliable_counts[rule_index];
            break;
        }
    }

    std::vector<std::complex<float>> leakage(rules.size(), {0.0F, 0.0F});
    std::vector<bool> has_leakage(rules.size(), false);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        auto& row = stats[rule_index];
        row.matched_symbols += matched_counts[rule_index];
        row.reliable_symbols += reliable_counts[rule_index];
        if (matched_counts[rule_index] == 0
            || reliable_counts[rule_index] < options.min_symbols
            || denominators[rule_index] <= 1.0e-9F) {
            continue;
        }
        const auto fit = numerators[rule_index] / denominators[rule_index];
        if (!std::isfinite(fit.real()) || !std::isfinite(fit.imag())) {
            continue;
        }
        leakage[rule_index] = fit;
        has_leakage[rule_index] = true;
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (const auto& [source_symbol, local_symbol] : source_to_symbol) {
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            if (!has_leakage[rule_index]) {
                continue;
            }
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto neighbor_carrier =
                static_cast<std::int64_t>(source_carrier) + rule.offset;
            if (neighbor_carrier < 0
                || neighbor_carrier >= static_cast<std::int64_t>(
                    dtmb::core::kC3780DataSymbols)) {
                continue;
            }
            const auto neighbor_source =
                source_frame * dtmb::core::kC3780DataSymbols
                + static_cast<std::size_t>(neighbor_carrier);
            const auto neighbor_symbol = find_symbol(neighbor_source);
            if (neighbor_symbol == std::numeric_limits<std::size_t>::max()) {
                continue;
            }
            const std::complex<float> value{
                symbols[local_symbol * kFloatsPerSymbol],
                symbols[local_symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> neighbor_value{
                symbols[neighbor_symbol * kFloatsPerSymbol],
                symbols[neighbor_symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> neighbor_nearest{
                nearest_qam64_level(neighbor_value.real()),
                nearest_qam64_level(neighbor_value.imag())};
            const auto corrected = value - leakage[rule_index] * neighbor_nearest;
            symbols[local_symbol * kFloatsPerSymbol] = corrected.real();
            symbols[local_symbol * kFloatsPerSymbol + 1U] = corrected.imag();
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        if (!has_leakage[rule_index]) {
            continue;
        }
        auto& row = stats[rule_index];
        row.corrected_symbols += corrected_counts[rule_index];
        row.leakage_real_sum += leakage[rule_index].real();
        row.leakage_imag_sum += leakage[rule_index].imag();
        row.leakage_abs_sum += static_cast<double>(std::abs(leakage[rule_index]));
        ++row.applied_chunks;
    }
}

void apply_source_carrier_neighbor_leakage_set(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierNeighborLeakageSetRule>& rules,
    std::vector<SourceCarrierNeighborLeakageSetStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::pair<std::size_t, std::size_t>> source_to_symbol;
    source_to_symbol.reserve(symbol_count);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        source_to_symbol.emplace_back(
            source_symbol_for_output_symbol(first_output_symbol + symbol, mode, phase),
            symbol);
    }
    std::sort(source_to_symbol.begin(), source_to_symbol.end());
    const auto find_symbol = [&](std::size_t source_symbol) -> std::size_t {
        const auto iter = std::lower_bound(
            source_to_symbol.begin(),
            source_to_symbol.end(),
            std::pair<std::size_t, std::size_t>{source_symbol, 0U});
        if (iter == source_to_symbol.end() || iter->first != source_symbol) {
            return std::numeric_limits<std::size_t>::max();
        }
        return iter->second;
    };

    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<std::vector<std::vector<std::complex<double>>>> normal_matrices;
    normal_matrices.reserve(rules.size());
    for (const auto& rule : rules) {
        const auto count = rule.offsets.size();
        normal_matrices.emplace_back(
            count,
            std::vector<std::complex<double>>(count + 1U, {0.0, 0.0}));
    }
    std::vector<std::size_t> matched_counts(rules.size(), 0U);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    const auto neighbor_nearests = [&](std::size_t source_frame,
                                       std::size_t source_carrier,
                                       const SourceCarrierNeighborLeakageSetRule& rule,
                                       std::vector<std::complex<float>>& values,
                                       bool require_reliable) -> bool {
        values.clear();
        values.reserve(rule.offsets.size());
        for (const auto offset : rule.offsets) {
            const auto neighbor_carrier =
                static_cast<std::int64_t>(source_carrier) + offset;
            if (neighbor_carrier < 0
                || neighbor_carrier >= static_cast<std::int64_t>(
                    dtmb::core::kC3780DataSymbols)) {
                return false;
            }
            const auto neighbor_source =
                source_frame * dtmb::core::kC3780DataSymbols
                + static_cast<std::size_t>(neighbor_carrier);
            const auto neighbor_symbol = find_symbol(neighbor_source);
            if (neighbor_symbol == std::numeric_limits<std::size_t>::max()) {
                return false;
            }
            const std::complex<float> neighbor_value{
                symbols[neighbor_symbol * kFloatsPerSymbol],
                symbols[neighbor_symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> neighbor_nearest{
                nearest_qam64_level(neighbor_value.real()),
                nearest_qam64_level(neighbor_value.imag())};
            if (require_reliable) {
                const auto neighbor_power = std::max(std::abs(neighbor_nearest), 1.0e-6F);
                const auto neighbor_relative_error =
                    std::abs(neighbor_value - neighbor_nearest) / neighbor_power;
                if (neighbor_relative_error > options.reliability_threshold) {
                    return false;
                }
            }
            values.push_back(neighbor_nearest);
        }
        return true;
    };

    std::vector<std::complex<float>> features;
    for (const auto& [source_symbol, local_symbol] : source_to_symbol) {
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            if (!neighbor_nearests(source_frame, source_carrier, rule, features, false)) {
                continue;
            }
            ++matched_counts[rule_index];
            if (!source_carrier_fit_uses_frame(options, source_carrier, source_frame)) {
                continue;
            }
            const std::complex<float> value{
                symbols[local_symbol * kFloatsPerSymbol],
                symbols[local_symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> nearest{
                nearest_qam64_level(value.real()),
                nearest_qam64_level(value.imag())};
            const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
            const auto relative_error = std::abs(value - nearest) / nearest_power;
            if (relative_error > options.reliability_threshold
                || !neighbor_nearests(source_frame, source_carrier, rule, features, true)) {
                continue;
            }
            const auto residual = value - nearest;
            auto& matrix = normal_matrices[rule_index];
            for (std::size_t row = 0; row < features.size(); ++row) {
                const auto feature_conj = std::conj(
                    static_cast<std::complex<double>>(features[row]));
                for (std::size_t column = 0; column < features.size(); ++column) {
                    matrix[row][column] +=
                        feature_conj * static_cast<std::complex<double>>(features[column]);
                }
                matrix[row][features.size()] +=
                    feature_conj * static_cast<std::complex<double>>(residual);
            }
            ++reliable_counts[rule_index];
            break;
        }
    }

    std::vector<std::vector<std::complex<float>>> leakage(rules.size());
    std::vector<bool> has_leakage(rules.size(), false);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        auto& row = stats[rule_index];
        row.matched_symbols += matched_counts[rule_index];
        row.reliable_symbols += reliable_counts[rule_index];
        if (matched_counts[rule_index] == 0
            || reliable_counts[rule_index] < std::max(options.min_symbols, rules[rule_index].offsets.size())) {
            continue;
        }
        if (!solve_complex_system(normal_matrices[rule_index], leakage[rule_index])) {
            continue;
        }
        has_leakage[rule_index] = true;
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (const auto& [source_symbol, local_symbol] : source_to_symbol) {
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (!has_leakage[rule_index]
                || source_frame < rule.first_frame
                || source_frame > rule.last_frame
                || !neighbor_nearests(source_frame, source_carrier, rule, features, false)) {
                continue;
            }
            std::complex<float> correction{0.0F, 0.0F};
            for (std::size_t index = 0; index < features.size(); ++index) {
                correction += leakage[rule_index][index] * features[index];
            }
            const std::complex<float> value{
                symbols[local_symbol * kFloatsPerSymbol],
                symbols[local_symbol * kFloatsPerSymbol + 1U]};
            const auto corrected = value - correction;
            symbols[local_symbol * kFloatsPerSymbol] = corrected.real();
            symbols[local_symbol * kFloatsPerSymbol + 1U] = corrected.imag();
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        if (!has_leakage[rule_index]) {
            continue;
        }
        auto& row = stats[rule_index];
        row.corrected_symbols += corrected_counts[rule_index];
        for (const auto coefficient : leakage[rule_index]) {
            const auto abs_value = static_cast<double>(std::abs(coefficient));
            row.leakage_abs_sum += abs_value;
            row.leakage_abs_max = std::max(row.leakage_abs_max, abs_value);
        }
        ++row.applied_chunks;
    }
}

void apply_source_carrier_axis_quantile(
    std::span<float> symbols,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const BranchGainOptions& options,
    const std::vector<SourceCarrierAxisQuantileRule>& rules,
    std::vector<SourceCarrierAxisQuantileStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<std::vector<float>> real_values(rules.size());
    std::vector<std::vector<float>> imag_values(rules.size());
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            if (!source_carrier_fit_uses_frame(options, source_carrier, source_frame)) {
                continue;
            }
            real_values[rule_index].push_back(symbols[symbol * kFloatsPerSymbol]);
            imag_values[rule_index].push_back(symbols[symbol * kFloatsPerSymbol + 1U]);
            break;
        }
    }

    std::vector<AxisQuantileFit> fits(rules.size());
    const auto min_symbols = std::max<std::size_t>(options.min_symbols, kQam64Levels.size());
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        const auto matched = real_values[rule_index].size();
        stats[rule_index].matched_symbols += matched;
        if (matched < min_symbols || imag_values[rule_index].size() != matched) {
            continue;
        }
        auto& fit = fits[rule_index];
        fit.valid =
            axis_quantile_centers(std::move(real_values[rule_index]), fit.real_centers)
            && axis_quantile_centers(std::move(imag_values[rule_index]), fit.imag_centers);
    }

    std::vector<std::size_t> corrected_counts(rules.size(), 0U);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const auto& fit = fits[rule_index];
            if (!fit.valid) {
                continue;
            }
            symbols[symbol * kFloatsPerSymbol] =
                remap_axis_quantile(symbols[symbol * kFloatsPerSymbol], fit.real_centers);
            symbols[symbol * kFloatsPerSymbol + 1U] = remap_axis_quantile(
                symbols[symbol * kFloatsPerSymbol + 1U],
                fit.imag_centers);
            ++corrected_counts[rule_index];
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        if (!fits[rule_index].valid) {
            continue;
        }
        stats[rule_index].corrected_symbols += corrected_counts[rule_index];
        ++stats[rule_index].applied_chunks;
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
    if (source_carrier_symbol_gain_has_overlaps(rules)) {
        apply_source_carrier_symbol_gain_sequential(
            symbols,
            first_output_symbol,
            mode,
            phase,
            options,
            rules,
            stats);
        return;
    }

    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
        ++stats[rule_index].chunks_seen;
    }

    std::vector<std::complex<float>> numerators(rules.size(), {0.0F, 0.0F});
    std::vector<float> denominators(rules.size(), 0.0F);
    std::vector<std::size_t> matched_counts(rules.size(), 0U);
    std::vector<std::size_t> reliable_counts(rules.size(), 0U);

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            ++matched_counts[rule_index];
            if (!source_carrier_fit_uses_frame(options, source_carrier, source_frame)) {
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
            numerators[rule_index] += std::conj(nearest) * value;
            denominators[rule_index] += std::norm(nearest);
            ++reliable_counts[rule_index];
        }
    }

    std::vector<std::complex<float>> gains(rules.size(), {1.0F, 0.0F});
    std::vector<bool> has_gain(rules.size(), false);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        auto& row = stats[rule_index];
        row.reliable_symbols += reliable_counts[rule_index];
        if (matched_counts[rule_index] == 0
            || reliable_counts[rule_index] < options.min_symbols
            || denominators[rule_index] <= 1.0e-9F) {
            continue;
        }
        const auto gain = numerators[rule_index] / denominators[rule_index];
        if (std::abs(gain) <= 1.0e-9F || !std::isfinite(gain.real())
            || !std::isfinite(gain.imag())) {
            continue;
        }
        gains[rule_index] = gain;
        has_gain[rule_index] = true;
    }

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            if (!has_gain[rule_index]) {
                continue;
            }
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const auto corrected = value / gains[rule_index];
            symbols[symbol * kFloatsPerSymbol] = corrected.real();
            symbols[symbol * kFloatsPerSymbol + 1U] = corrected.imag();
            break;
        }
    }

    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        if (!has_gain[rule_index]) {
            continue;
        }
        auto& row = stats[rule_index];
        row.corrected_symbols += matched_counts[rule_index];
        row.gain_real_sum += static_cast<double>(gains[rule_index].real());
        row.gain_imag_sum += static_cast<double>(gains[rule_index].imag());
        row.gain_abs_sum += static_cast<double>(std::abs(gains[rule_index]));
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

std::vector<SourceCarrierAxisAffineStats> make_carrier_axis_affine_stats(
    const std::vector<SourceCarrierAxisAffineRule>& rules) {
    std::vector<SourceCarrierAxisAffineStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierAxisAffineStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierLinearAffineStats> make_carrier_linear_affine_stats(
    const std::vector<SourceCarrierLinearAffineRule>& rules) {
    std::vector<SourceCarrierLinearAffineStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierLinearAffineStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierAxisQuantileStats> make_carrier_axis_quantile_stats(
    const std::vector<SourceCarrierAxisQuantileRule>& rules) {
    std::vector<SourceCarrierAxisQuantileStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierAxisQuantileStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierAxisCentroidStats> make_carrier_axis_centroid_stats(
    const std::vector<SourceCarrierAxisCentroidRule>& rules) {
    std::vector<SourceCarrierAxisCentroidStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierAxisCentroidStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierQamCentroidStats> make_carrier_qam_centroid_stats(
    const std::vector<SourceCarrierQamCentroidRule>& rules) {
    std::vector<SourceCarrierQamCentroidStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierQamCentroidStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierNeighborLeakageStats> make_carrier_neighbor_leakage_stats(
    const std::vector<SourceCarrierNeighborLeakageRule>& rules) {
    std::vector<SourceCarrierNeighborLeakageStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierNeighborLeakageStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.offset = rules[index].offset;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierNeighborLeakageSetStats>
make_carrier_neighbor_leakage_set_stats(
    const std::vector<SourceCarrierNeighborLeakageSetRule>& rules) {
    std::vector<SourceCarrierNeighborLeakageSetStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierNeighborLeakageSetStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.offset_count = rules[index].offsets.size();
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
        row.last_carrier = rules[index].last_carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.scale = rules[index].scale;
        stats.push_back(row);
    }
    return stats;
}

std::vector<SourceCarrierResidualLlrConfidenceStats>
make_carrier_residual_llr_confidence_stats(
    const std::vector<SourceCarrierResidualLlrConfidenceRule>& rules) {
    std::vector<SourceCarrierResidualLlrConfidenceStats> stats;
    stats.reserve(rules.size());
    for (std::size_t index = 0; index < rules.size(); ++index) {
        SourceCarrierResidualLlrConfidenceStats row;
        row.rule_index = index;
        row.carrier = rules[index].carrier;
        row.first_frame = rules[index].first_frame;
        row.last_frame = rules[index].last_frame;
        row.residual_knee = rules[index].residual_knee;
        row.min_scale = rules[index].min_scale;
        stats.push_back(row);
    }
    return stats;
}

bool source_carrier_residual_llr_confidence_ranges_overlap(
    const SourceCarrierResidualLlrConfidenceRule& left,
    const SourceCarrierResidualLlrConfidenceRule& right) noexcept {
    return left.carrier == right.carrier && left.first_frame <= right.last_frame
        && right.first_frame <= left.last_frame;
}

double unwrap_phase_delta(double delta) noexcept {
    while (delta > kPi) {
        delta -= 2.0 * kPi;
    }
    while (delta <= -kPi) {
        delta += 2.0 * kPi;
    }
    return delta;
}

bool is_system_info_position(std::size_t inserted_logical) {
    return std::find(
        kC3780SystemInfoPositions.begin(),
        kC3780SystemInfoPositions.end(),
        inserted_logical)
        != kC3780SystemInfoPositions.end();
}

std::size_t c3780_logical_to_physical(std::size_t inserted_logical) {
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                for (std::size_t l = 0; l < 2; ++l) {
                    for (std::size_t m = 0; m < 2; ++m) {
                        for (std::size_t n = 0; n < 5; ++n) {
                            for (std::size_t o = 0; o < 7; ++o) {
                                const auto logical =
                                    i * 1260 + j * 420 + k * 140 + l * 70
                                    + m * 35 + n * 7 + o;
                                if (logical == inserted_logical) {
                                    return o * 540 + n * 108 + m * 54 + l * 27
                                        + k * 9 + j * 3 + i;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    throw std::logic_error("invalid C3780 logical carrier index");
}

std::array<std::size_t, dtmb::core::kC3780DataSymbols> build_data_physical_bins() {
    std::array<std::size_t, dtmb::core::kC3780DataSymbols> physical_bins{};
    std::size_t data_index = 0;
    for (std::size_t inserted_logical = 0;
         inserted_logical < dtmb::core::kC3780FrameBodySymbols;
         ++inserted_logical) {
        if (is_system_info_position(inserted_logical)) {
            continue;
        }
        physical_bins[data_index] = c3780_logical_to_physical(inserted_logical);
        ++data_index;
    }
    if (data_index != dtmb::core::kC3780DataSymbols) {
        throw std::logic_error("invalid C3780 data carrier count");
    }
    return physical_bins;
}

SourceFrameRampGeometry build_data_ramp_geometry() {
    constexpr std::size_t kWindowCount = 32;
    constexpr std::size_t kFrameSymbols = dtmb::core::kC3780DataSymbols;
    constexpr std::size_t kWindowSymbols = kFrameSymbols / kWindowCount;
    SourceFrameRampGeometry geometry;
    for (std::size_t index = 0; index < kFrameSymbols; ++index) {
        geometry.order[index] = index;
    }
    for (std::size_t window = 0; window < kWindowCount; ++window) {
        geometry.x[window] = static_cast<double>(window)
            - (static_cast<double>(kWindowCount) - 1.0) / 2.0;
    }
    geometry.span = static_cast<double>(kWindowCount);
    static_assert(kFrameSymbols == kWindowCount * kWindowSymbols);
    return geometry;
}

SourceFrameRampGeometry build_physical_ramp_geometry() {
    constexpr std::size_t kWindowCount = 32;
    constexpr std::size_t kFrameSymbols = dtmb::core::kC3780DataSymbols;
    constexpr std::size_t kWindowSymbols = kFrameSymbols / kWindowCount;
    static_assert(kFrameSymbols == kWindowCount * kWindowSymbols);
    const auto physical_bins = build_data_physical_bins();
    SourceFrameRampGeometry geometry;
    for (std::size_t index = 0; index < kFrameSymbols; ++index) {
        geometry.order[index] = index;
    }
    std::sort(
        geometry.order.begin(),
        geometry.order.end(),
        [&physical_bins](std::size_t left, std::size_t right) {
            if (physical_bins[left] != physical_bins[right]) {
                return physical_bins[left] < physical_bins[right];
            }
            return left < right;
        });

    const auto [min_it, max_it] =
        std::minmax_element(physical_bins.begin(), physical_bins.end());
    double coordinate_sum = 0.0;
    for (const auto physical_bin : physical_bins) {
        coordinate_sum += static_cast<double>(physical_bin);
    }
    const auto coordinate_mean =
        coordinate_sum / static_cast<double>(physical_bins.size());
    for (std::size_t window = 0; window < kWindowCount; ++window) {
        const auto first_symbol = window * kWindowSymbols;
        double window_sum = 0.0;
        for (std::size_t offset = 0; offset < kWindowSymbols; ++offset) {
            window_sum += static_cast<double>(
                physical_bins[geometry.order[first_symbol + offset]]);
        }
        geometry.x[window] =
            window_sum / static_cast<double>(kWindowSymbols) - coordinate_mean;
    }
    geometry.span = static_cast<double>(*max_it) - static_cast<double>(*min_it);
    return geometry;
}

const SourceFrameRampGeometry& ramp_geometry_for(
    SourceFrameRampCoordinate coordinate) {
    static const auto data_geometry = build_data_ramp_geometry();
    static const auto physical_geometry = build_physical_ramp_geometry();
    switch (coordinate) {
    case SourceFrameRampCoordinate::data:
        return data_geometry;
    case SourceFrameRampCoordinate::physical:
        return physical_geometry;
    }
    return data_geometry;
}

float estimate_source_frame_phase_ramp(
    std::span<const float> frame_symbols,
    const SourceFrameRampGeometry& geometry) {
    constexpr std::size_t kFrameSymbols = dtmb::core::kC3780DataSymbols;
    constexpr std::size_t kWindowCount = 32;
    constexpr std::size_t kWindowSymbols = kFrameSymbols / kWindowCount;
    static_assert(kFrameSymbols == kWindowCount * kWindowSymbols);

    double power_sum = 0.0;
    for (std::size_t symbol = 0; symbol < kFrameSymbols; ++symbol) {
        const auto real = static_cast<double>(frame_symbols[symbol * kFloatsPerSymbol]);
        const auto imag = static_cast<double>(
            frame_symbols[symbol * kFloatsPerSymbol + 1U]);
        power_sum += real * real + imag * imag;
    }
    const auto mean_power = power_sum / static_cast<double>(kFrameSymbols);
    const auto inv_rms = mean_power > 0.0 ? 1.0 / std::sqrt(mean_power) : 1.0;

    std::array<double, kWindowCount> phases{};
    std::array<double, kWindowCount> weights{};
    for (std::size_t window = 0; window < kWindowCount; ++window) {
        std::complex<double> sum{0.0, 0.0};
        const auto first_symbol = window * kWindowSymbols;
        for (std::size_t offset = 0; offset < kWindowSymbols; ++offset) {
            const auto symbol = geometry.order[first_symbol + offset];
            const std::complex<double> value{
                static_cast<double>(frame_symbols[symbol * kFloatsPerSymbol])
                    * inv_rms,
                static_cast<double>(
                    frame_symbols[symbol * kFloatsPerSymbol + 1U])
                    * inv_rms};
            const auto squared = value * value;
            sum += squared * squared;
        }
        phases[window] = std::atan2(sum.imag(), sum.real());
        weights[window] = std::abs(sum);
    }

    for (std::size_t window = 1; window < kWindowCount; ++window) {
        phases[window] = phases[window - 1U]
            + unwrap_phase_delta(phases[window] - phases[window - 1U]);
    }

    double weight_sum = 0.0;
    double weighted_x_sum = 0.0;
    double weighted_phase_sum = 0.0;
    for (std::size_t window = 0; window < kWindowCount; ++window) {
        const auto x = geometry.x[window];
        weight_sum += weights[window];
        weighted_x_sum += weights[window] * x;
        weighted_phase_sum += weights[window] * phases[window];
    }
    if (weight_sum <= 0.0) {
        return 0.0F;
    }

    const auto x_mean = weighted_x_sum / weight_sum;
    const auto phase_mean = weighted_phase_sum / weight_sum;
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t window = 0; window < kWindowCount; ++window) {
        const auto x = geometry.x[window];
        const auto dx = x - x_mean;
        const auto dp = phases[window] - phase_mean;
        numerator += weights[window] * dx * dp;
        denominator += weights[window] * dx * dx;
    }
    if (denominator <= 0.0) {
        return 0.0F;
    }

    const auto slope4_per_window = numerator / denominator;
    const auto ramp = slope4_per_window * geometry.span / 4.0;
    return std::isfinite(ramp) ? static_cast<float>(ramp) : 0.0F;
}

void score_source_frame_phase_ramps(
    std::span<const float> input_symbols,
    std::size_t first_input_symbol,
    const SourceFrameRampLlrErasureOptions& options,
    const SourceFrameRampGeometry& geometry,
    std::vector<float>& ramp_metrics,
    SourceFrameRampLlrErasureStats& stats) {
    if (!options.enabled) {
        return;
    }
    constexpr auto kFrameSymbols = dtmb::core::kC3780DataSymbols;
    const auto symbol_count = input_symbols.size() / kFloatsPerSymbol;
    if ((first_input_symbol % kFrameSymbols) != 0
        || (symbol_count % kFrameSymbols) != 0) {
        throw std::runtime_error(
            "source frame ramp LLR erasure requires C3780-frame-aligned chunks");
    }
    const auto frame_count = symbol_count / kFrameSymbols;
    for (std::size_t local_frame = 0; local_frame < frame_count; ++local_frame) {
        const auto source_frame = first_input_symbol / kFrameSymbols + local_frame;
        const auto frame_offset = local_frame * kFrameSymbols * kFloatsPerSymbol;
        const auto ramp = estimate_source_frame_phase_ramp(
            input_symbols.subspan(frame_offset, kFrameSymbols * kFloatsPerSymbol),
            geometry);
        if (ramp_metrics.size() <= source_frame) {
            ramp_metrics.resize(
                source_frame + 1U,
                std::numeric_limits<float>::quiet_NaN());
        }
        ramp_metrics[source_frame] = ramp;
        const auto abs_ramp = std::abs(ramp);
        stats.abs_ramp_sum += static_cast<double>(abs_ramp);
        stats.max_abs_ramp = std::max(stats.max_abs_ramp, abs_ramp);
        ++stats.scored_frames;
        if (abs_ramp >= options.threshold) {
            ++stats.selected_frames;
            stats.selected_abs_ramp_sum += static_cast<double>(abs_ramp);
        }
    }
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

void apply_source_frame_ramp_llr_erasure(
    std::span<float> llrs,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const SourceFrameRampLlrErasureOptions& options,
    const std::vector<float>& ramp_metrics,
    SourceFrameRampLlrErasureStats& stats) {
    if (!options.enabled) {
        return;
    }
    const auto symbol_count = llrs.size() / kLlrsPerSymbol;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        ++stats.touched_symbols;
        if (source_frame >= ramp_metrics.size()
            || !std::isfinite(ramp_metrics[source_frame])) {
            ++stats.out_of_metric_symbols;
            continue;
        }
        if (std::abs(ramp_metrics[source_frame]) < options.threshold) {
            continue;
        }
        for (std::size_t bit = 0; bit < kLlrsPerSymbol; ++bit) {
            llrs[symbol * kLlrsPerSymbol + bit] *= options.scale;
        }
        ++stats.scaled_symbols;
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
            if (source_carrier < rule.carrier || source_carrier > rule.last_carrier
                || source_frame < rule.first_frame
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

void apply_source_carrier_residual_llr_confidence(
    std::span<const float> symbols,
    std::span<float> llrs,
    std::size_t first_output_symbol,
    dtmb::core::SymbolInterleaverMode mode,
    std::size_t phase,
    const std::vector<SourceCarrierResidualLlrConfidenceRule>& rules,
    std::vector<SourceCarrierResidualLlrConfidenceStats>& stats) {
    if (rules.empty()) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    std::vector<std::vector<std::size_t>> rules_by_carrier(
        dtmb::core::kC3780DataSymbols);
    for (std::size_t rule_index = 0; rule_index < rules.size(); ++rule_index) {
        rules_by_carrier[rules[rule_index].carrier].push_back(rule_index);
    }
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto source_symbol = source_symbol_for_output_symbol(
            first_output_symbol + symbol,
            mode,
            phase);
        const auto source_frame = source_symbol / dtmb::core::kC3780DataSymbols;
        const auto source_carrier = source_symbol % dtmb::core::kC3780DataSymbols;
        const auto& carrier_rules = rules_by_carrier[source_carrier];
        if (carrier_rules.empty()) {
            continue;
        }
        for (const auto rule_index : carrier_rules) {
            const auto& rule = rules[rule_index];
            if (source_frame < rule.first_frame || source_frame > rule.last_frame) {
                continue;
            }
            auto& row = stats[rule_index];
            ++row.matched_symbols;
            const std::complex<float> value{
                symbols[symbol * kFloatsPerSymbol],
                symbols[symbol * kFloatsPerSymbol + 1U]};
            const std::complex<float> nearest{
                nearest_qam64_level(value.real()),
                nearest_qam64_level(value.imag())};
            const auto nearest_power = std::max(std::abs(nearest), 1.0e-6F);
            const auto residual_ratio = std::abs(value - nearest) / nearest_power;
            if (!std::isfinite(residual_ratio)
                || residual_ratio <= rule.residual_knee) {
                break;
            }
            const auto scale = std::clamp(
                rule.residual_knee / residual_ratio,
                rule.min_scale,
                1.0F);
            auto symbol_llrs = llrs.subspan(symbol * kLlrsPerSymbol, kLlrsPerSymbol);
            for (auto& llr : symbol_llrs) {
                llr *= scale;
            }
            ++row.scaled_symbols;
            row.scale_sum += static_cast<double>(scale);
            row.scaled_residual_ratio_sum += static_cast<double>(residual_ratio);
            row.min_observed_scale = std::min(row.min_observed_scale, scale);
            break;
        }
    }
}

void apply_dd_llr_confidence(
    std::span<const float> symbols,
    std::span<float> llrs,
    const DdLlrConfidenceOptions& options,
    DdLlrConfidenceStats& stats) {
    if (!options.enabled) {
        return;
    }
    const auto symbol_count = symbols.size() / kFloatsPerSymbol;
    stats.symbols += symbol_count;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const std::complex<float> value{
            symbols[symbol * kFloatsPerSymbol],
            symbols[symbol * kFloatsPerSymbol + 1U]};
        const std::complex<float> nearest{
            nearest_qam64_level(value.real()),
            nearest_qam64_level(value.imag())};
        const auto decision_abs = std::max(std::abs(nearest), 1.0e-6F);
        const auto residual_ratio = std::abs(value - nearest) / decision_abs;
        if (!std::isfinite(residual_ratio)
            || residual_ratio <= options.residual_knee) {
            continue;
        }
        const auto scale = std::clamp(
            options.residual_knee / residual_ratio,
            options.min_scale,
            1.0F);
        auto symbol_llrs = llrs.subspan(symbol * kLlrsPerSymbol, kLlrsPerSymbol);
        for (auto& llr : symbol_llrs) {
            llr *= scale;
        }
        ++stats.scaled_symbols;
        stats.scale_sum += static_cast<double>(scale);
        stats.scaled_residual_ratio_sum += static_cast<double>(residual_ratio);
        stats.min_observed_scale = std::min(stats.min_observed_scale, scale);
    }
}

}  // namespace

int main(int argc, char** argv) {
    dtmb::core::QamSoftDemapOptions demap_options{};
    BranchGainOptions branch_gain_options{};
    DdLlrConfidenceOptions dd_llr_confidence_options{};
    DdLlrConfidenceStats dd_llr_confidence_stats{};
    SourceFrameRampLlrErasureOptions ramp_llr_erasure_options{};
    SourceFrameRampLlrErasureStats ramp_llr_erasure_stats{};
    std::vector<SourceFrameSymbolGainRule> symbol_gain_rules;
    std::vector<SourceFrameFixedComplexGainRule> fixed_complex_gain_rules;
    std::vector<SourceCarrierSymbolGainRule> carrier_symbol_gain_rules;
    std::vector<SourceCarrierAxisAffineRule> carrier_axis_affine_rules;
    std::vector<SourceCarrierLinearAffineRule> carrier_linear_affine_rules;
    std::vector<SourceCarrierAxisQuantileRule> carrier_axis_quantile_rules;
    std::vector<SourceCarrierAxisCentroidRule> carrier_axis_centroid_rules;
    std::vector<SourceCarrierAxisCentroidRule> carrier_axis_centroid_median_rules;
    std::vector<SourceCarrierQamCentroidRule> carrier_qam_centroid_rules;
    std::vector<SourceCarrierNeighborLeakageRule> carrier_neighbor_leakage_rules;
    std::vector<SourceCarrierNeighborLeakageSetRule> carrier_neighbor_leakage_set_rules;
    std::vector<SourceFrameCarrierRollRule> carrier_roll_rules;
    std::vector<SourceFrameDisplacementRule> source_displacement_rules;
    std::vector<SourceFrameAxisAffineRule> axis_affine_rules;
    std::vector<SourceFrameLlrScaleRule> llr_scale_rules;
    std::vector<SourceCarrierLlrScaleRule> carrier_llr_scale_rules;
    std::vector<SourceCarrierResidualLlrConfidenceRule>
        carrier_residual_llr_confidence_rules;
    auto mode = dtmb::core::SymbolInterleaverMode::mode1;
    bool mode_set = false;
    bool keep_latency = false;
    std::size_t phase = 0;
    std::size_t chunk_symbols = 1U << 16U;
    std::string input_path = "-";
    std::string output_path = "-";
    std::string csi_weights_path;
    std::string symbols_output_path;
    std::string source_carrier_fixed_complex_gains_path;
    std::vector<std::pair<std::size_t, std::size_t>> output_frame_ranges;
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
            } else if (arg == "--csi-weights") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                csi_weights_path = argv[index];
            } else if (arg == "--soft-demod-method") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                demap_options.method = parse_soft_demap_method(argv[index]);
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
            } else if (arg == "--source-frame-fixed-complex-gain") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                fixed_complex_gain_rules.push_back(
                    parse_fixed_complex_gain_rule(argv[index]));
            } else if (arg == "--source-carrier-fixed-complex-gains") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                source_carrier_fixed_complex_gains_path = argv[index];
            } else if (arg == "--source-carrier-symbol-gain") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_symbol_gain_rules.push_back(
                    parse_carrier_symbol_gain_rule(argv[index]));
            } else if (arg == "--source-carrier-symbol-gain-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_symbol_gain_range(argv[index]);
                carrier_symbol_gain_rules.insert(
                    carrier_symbol_gain_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-axis-affine") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_axis_affine_rules.push_back(
                    parse_carrier_axis_affine_rule(argv[index]));
            } else if (arg == "--source-carrier-axis-affine-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_axis_affine_range(argv[index]);
                carrier_axis_affine_rules.insert(
                    carrier_axis_affine_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-linear-affine") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_linear_affine_rules.push_back(
                    parse_carrier_linear_affine_rule(argv[index]));
            } else if (arg == "--source-carrier-linear-affine-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_linear_affine_range(argv[index]);
                carrier_linear_affine_rules.insert(
                    carrier_linear_affine_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-axis-quantile") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_axis_quantile_rules.push_back(
                    parse_carrier_axis_quantile_rule(argv[index]));
            } else if (arg == "--source-carrier-axis-quantile-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_axis_quantile_range(argv[index]);
                carrier_axis_quantile_rules.insert(
                    carrier_axis_quantile_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-axis-centroid") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_axis_centroid_rules.push_back(
                    parse_carrier_axis_centroid_rule(argv[index]));
            } else if (arg == "--source-carrier-axis-centroid-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_axis_centroid_range(argv[index]);
                carrier_axis_centroid_rules.insert(
                    carrier_axis_centroid_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-axis-centroid-median") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_axis_centroid_median_rules.push_back(
                    parse_carrier_axis_centroid_rule(argv[index]));
            } else if (arg == "--source-carrier-axis-centroid-median-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_axis_centroid_range(argv[index]);
                carrier_axis_centroid_median_rules.insert(
                    carrier_axis_centroid_median_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-qam-centroid") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_qam_centroid_rules.push_back(
                    parse_carrier_qam_centroid_rule(argv[index]));
            } else if (arg == "--source-carrier-qam-centroid-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_qam_centroid_range(argv[index]);
                carrier_qam_centroid_rules.insert(
                    carrier_qam_centroid_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-neighbor-leakage") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_neighbor_leakage_rules.push_back(
                    parse_carrier_neighbor_leakage_rule(argv[index]));
            } else if (arg == "--source-carrier-neighbor-leakage-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_neighbor_leakage_range(argv[index]);
                carrier_neighbor_leakage_rules.insert(
                    carrier_neighbor_leakage_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-neighbor-leakage-set-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_carrier_neighbor_leakage_set_range(argv[index]);
                carrier_neighbor_leakage_set_rules.insert(
                    carrier_neighbor_leakage_set_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-carrier-fit-exclude-source-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options.carrier_fit_exclude_source_frame_ranges.push_back(
                    parse_range(
                        argv[index],
                        "source carrier fit exclude source frame range"));
            } else if (arg == "--source-carrier-symbol-gain-fit-exclude-source-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options
                    .carrier_symbol_gain_fit_exclude_source_frame_ranges
                    .push_back(
                        parse_range(
                            argv[index],
                            "source carrier symbol gain fit exclude source frame range"));
            } else if (
                arg
                == "--source-carrier-symbol-gain-fit-exclude-source-carrier-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options
                    .carrier_symbol_gain_fit_exclude_source_carrier_frame_ranges
                    .push_back(
                        parse_carrier_frame_range(
                            argv[index],
                            "source carrier symbol gain fit exclude source carrier frame range"));
            } else if (arg == "--source-carrier-linear-affine-fit-exclude-source-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options
                    .carrier_linear_affine_fit_exclude_source_frame_ranges
                    .push_back(
                        parse_range(
                            argv[index],
                            "source carrier linear affine fit exclude source frame range"));
            } else if (
                arg
                == "--source-carrier-linear-affine-fit-exclude-source-carrier-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options
                    .carrier_linear_affine_fit_exclude_source_carrier_frame_ranges
                    .push_back(
                        parse_carrier_frame_range(
                            argv[index],
                            "source carrier linear affine fit exclude source carrier frame range"));
            } else if (arg == "--source-carrier-axis-centroid-fit-exclude-source-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options
                    .carrier_axis_centroid_fit_exclude_source_frame_ranges
                    .push_back(
                        parse_range(
                            argv[index],
                            "source carrier axis centroid fit exclude source frame range"));
            } else if (
                arg
                == "--source-carrier-axis-centroid-fit-exclude-source-carrier-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                branch_gain_options
                    .carrier_axis_centroid_fit_exclude_source_carrier_frame_ranges
                    .push_back(
                        parse_carrier_frame_range(
                            argv[index],
                            "source carrier axis centroid fit exclude source carrier frame range"));
            } else if (arg == "--source-frame-carrier-roll") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_roll_rules.push_back(parse_carrier_roll_rule(argv[index]));
            } else if (arg == "--source-frame-displacement") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                source_displacement_rules.push_back(
                    parse_source_frame_displacement_rule(argv[index]));
            } else if (arg == "--source-frame-axis-affine") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                axis_affine_rules.push_back(parse_axis_affine_rule(argv[index]));
            } else if (arg == "--source-frame-axis-affine-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules = parse_axis_affine_range(argv[index]);
                axis_affine_rules.insert(
                    axis_affine_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
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
            } else if (arg == "--source-carrier-llr-scale-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_llr_scale_rules.push_back(
                    parse_carrier_llr_scale_range_rule(argv[index]));
            } else if (arg == "--source-carrier-residual-llr-confidence") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                carrier_residual_llr_confidence_rules.push_back(
                    parse_carrier_residual_llr_confidence_rule(argv[index]));
            } else if (arg == "--source-carrier-residual-llr-confidence-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                auto range_rules =
                    parse_carrier_residual_llr_confidence_range(argv[index]);
                carrier_residual_llr_confidence_rules.insert(
                    carrier_residual_llr_confidence_rules.end(),
                    range_rules.begin(),
                    range_rules.end());
            } else if (arg == "--source-frame-ramp-llr-erasure-threshold") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                ramp_llr_erasure_options.enabled = true;
                ramp_llr_erasure_options.threshold =
                    parse_float(argv[index], "source frame ramp LLR erasure threshold");
            } else if (arg == "--source-frame-ramp-llr-erasure-scale") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                ramp_llr_erasure_options.enabled = true;
                ramp_llr_erasure_options.scale =
                    parse_float(argv[index], "source frame ramp LLR erasure scale");
            } else if (arg == "--source-frame-ramp-llr-erasure-coordinate") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                ramp_llr_erasure_options.coordinate = parse_ramp_coordinate(argv[index]);
            } else if (arg == "--output-frame-range") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                const auto range = parse_range(argv[index], "output frame range");
                if (range.first == 0U) {
                    throw std::invalid_argument(
                        "output frame range must use 1-based frame indexes");
                }
                output_frame_ranges.push_back(range);
            } else if (arg == "--symbols-output") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                symbols_output_path = argv[index];
            } else if (arg == "--dd-llr-confidence") {
                dd_llr_confidence_options.enabled = true;
            } else if (arg == "--dd-llr-confidence-knee") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                dd_llr_confidence_options.enabled = true;
                dd_llr_confidence_options.residual_knee =
                    parse_float(argv[index], "DD LLR confidence knee");
            } else if (arg == "--dd-llr-confidence-min-scale") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                dd_llr_confidence_options.enabled = true;
                dd_llr_confidence_options.min_scale =
                    parse_float(argv[index], "DD LLR confidence min scale");
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
        if (!carrier_roll_rules.empty()
            && (chunk_symbols % dtmb::core::kC3780DataSymbols) != 0) {
            throw std::invalid_argument(
                "source frame carrier roll requires --chunk-symbols to be a multiple "
                "of one C3780 data frame");
        }
        if (ramp_llr_erasure_options.enabled
            && (chunk_symbols % dtmb::core::kC3780DataSymbols) != 0) {
            throw std::invalid_argument(
                "source frame ramp LLR erasure requires --chunk-symbols to be a "
                "multiple of one C3780 data frame");
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
        if (!std::isfinite(dd_llr_confidence_options.residual_knee)
            || dd_llr_confidence_options.residual_knee <= 0.0F) {
            throw std::invalid_argument(
                "DD LLR confidence knee must be positive and finite");
        }
        if (!std::isfinite(dd_llr_confidence_options.min_scale)
            || dd_llr_confidence_options.min_scale < 0.0F
            || dd_llr_confidence_options.min_scale > 1.0F) {
            throw std::invalid_argument(
                "DD LLR confidence min scale must be finite and within 0..1");
        }
        if (!std::isfinite(ramp_llr_erasure_options.threshold)
            || ramp_llr_erasure_options.threshold < 0.0F) {
            throw std::invalid_argument(
                "source frame ramp LLR erasure threshold must be finite and non-negative");
        }
        if (!std::isfinite(ramp_llr_erasure_options.scale)
            || ramp_llr_erasure_options.scale < 0.0F) {
            throw std::invalid_argument(
                "source frame ramp LLR erasure scale must be finite and non-negative");
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
        for (const auto& rule : fixed_complex_gain_rules) {
            if (rule.branch >= mode_spec.branch_count) {
                throw std::invalid_argument(
                    "source frame fixed complex gain branch is outside the selected mode");
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
        for (const auto& rule : carrier_residual_llr_confidence_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier residual LLR confidence carrier is outside "
                    "the C3780 data span");
            }
        }
        for (std::size_t left = 0;
             left < carrier_residual_llr_confidence_rules.size();
             ++left) {
            for (std::size_t right = left + 1U;
                 right < carrier_residual_llr_confidence_rules.size();
                 ++right) {
                if (source_carrier_residual_llr_confidence_ranges_overlap(
                        carrier_residual_llr_confidence_rules[left],
                        carrier_residual_llr_confidence_rules[right])) {
                    throw std::invalid_argument(
                        "overlapping source carrier residual LLR confidence rules");
                }
            }
        }
        for (const auto& rule : carrier_symbol_gain_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier symbol gain carrier is outside the C3780 data span");
            }
        }
        for (const auto& rule : carrier_axis_affine_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier axis affine carrier is outside the C3780 data span");
            }
        }
        for (std::size_t left = 0; left < carrier_axis_affine_rules.size(); ++left) {
            for (std::size_t right = left + 1U; right < carrier_axis_affine_rules.size(); ++right) {
                if (source_carrier_axis_affine_ranges_overlap(
                        carrier_axis_affine_rules[left],
                        carrier_axis_affine_rules[right])) {
                    throw std::invalid_argument(
                        "overlapping source carrier axis affine rules");
                }
            }
        }
        for (const auto& rule : carrier_linear_affine_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier linear affine carrier is outside the C3780 data span");
            }
        }
        for (std::size_t left = 0; left < carrier_linear_affine_rules.size(); ++left) {
            for (std::size_t right = left + 1U; right < carrier_linear_affine_rules.size(); ++right) {
                if (source_carrier_linear_affine_ranges_overlap(
                        carrier_linear_affine_rules[left],
                        carrier_linear_affine_rules[right])) {
                    throw std::invalid_argument(
                        "overlapping source carrier linear affine rules");
                }
            }
        }
        for (const auto& rule : carrier_axis_quantile_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier axis quantile carrier is outside the C3780 data span");
            }
        }
        for (std::size_t left = 0; left < carrier_axis_quantile_rules.size(); ++left) {
            for (std::size_t right = left + 1U; right < carrier_axis_quantile_rules.size(); ++right) {
                if (source_carrier_axis_quantile_ranges_overlap(
                        carrier_axis_quantile_rules[left],
                        carrier_axis_quantile_rules[right])) {
                    throw std::invalid_argument(
                        "overlapping source carrier axis quantile rules");
                }
            }
        }
        for (const auto& rule : carrier_axis_centroid_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier axis centroid carrier is outside the C3780 data span");
            }
        }
        for (std::size_t left = 0; left < carrier_axis_centroid_rules.size(); ++left) {
            for (std::size_t right = left + 1U; right < carrier_axis_centroid_rules.size(); ++right) {
                if (source_carrier_axis_centroid_ranges_overlap(
                        carrier_axis_centroid_rules[left],
                        carrier_axis_centroid_rules[right])) {
                    throw std::invalid_argument(
                        "overlapping source carrier axis centroid rules");
                }
            }
        }
        for (const auto& rule : carrier_axis_centroid_median_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier axis centroid median carrier is outside the C3780 data span");
            }
        }
        for (std::size_t left = 0; left < carrier_axis_centroid_median_rules.size(); ++left) {
            for (std::size_t right = left + 1U;
                 right < carrier_axis_centroid_median_rules.size();
                 ++right) {
                if (source_carrier_axis_centroid_ranges_overlap(
                        carrier_axis_centroid_median_rules[left],
                        carrier_axis_centroid_median_rules[right])) {
                    throw std::invalid_argument(
                        "overlapping source carrier axis centroid median rules");
                }
            }
        }
        for (const auto& rule : carrier_qam_centroid_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier QAM centroid carrier is outside the C3780 data span");
            }
        }
        for (std::size_t left = 0; left < carrier_qam_centroid_rules.size(); ++left) {
            for (std::size_t right = left + 1U; right < carrier_qam_centroid_rules.size(); ++right) {
                if (source_carrier_qam_centroid_ranges_overlap(
                        carrier_qam_centroid_rules[left],
                        carrier_qam_centroid_rules[right])) {
                    throw std::invalid_argument(
                        "overlapping source carrier QAM centroid rules");
                }
            }
        }
        for (const auto& rule : carrier_neighbor_leakage_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier neighbor leakage carrier is outside the C3780 data span");
            }
            const auto neighbor_carrier =
                static_cast<std::int64_t>(rule.carrier) + rule.offset;
            if (neighbor_carrier < 0
                || neighbor_carrier >= static_cast<std::int64_t>(
                    dtmb::core::kC3780DataSymbols)) {
                throw std::invalid_argument(
                    "source carrier neighbor leakage offset points outside the C3780 data span");
            }
        }
        for (const auto& rule : carrier_neighbor_leakage_set_rules) {
            if (rule.carrier >= dtmb::core::kC3780DataSymbols) {
                throw std::invalid_argument(
                    "source carrier neighbor leakage set carrier is outside the C3780 data span");
            }
            for (const auto offset : rule.offsets) {
                const auto neighbor_carrier =
                    static_cast<std::int64_t>(rule.carrier) + offset;
                if (neighbor_carrier < 0
                    || neighbor_carrier >= static_cast<std::int64_t>(
                        dtmb::core::kC3780DataSymbols)) {
                    throw std::invalid_argument(
                        "source carrier neighbor leakage set offset points outside the C3780 data span");
                }
            }
        }
        for (std::size_t left = 0; left < source_displacement_rules.size(); ++left) {
            for (std::size_t right = left + 1U; right < source_displacement_rules.size(); ++right) {
                if (source_displacement_rules[left].fec_frame
                        == source_displacement_rules[right].fec_frame
                    && source_displacement_rules[left].codeword_slot
                        == source_displacement_rules[right].codeword_slot) {
                    throw std::invalid_argument(
                        "duplicate source frame displacement rule target");
                }
            }
        }
        if (!positional.empty()) {
            input_path = positional[0];
        }
        if (positional.size() == 2) {
            output_path = positional[1];
        }
        if (symbols_output_path == "-") {
            throw std::invalid_argument("--symbols-output does not support stdout");
        }
        if (!csi_weights_path.empty()
            && (!source_displacement_rules.empty() || !carrier_roll_rules.empty())) {
            throw std::invalid_argument(
                "--csi-weights is incompatible with source displacement or carrier roll");
        }

        dtmb::tools::configure_binary_stdio(input_path == "-", output_path == "-");
        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);
        std::ifstream csi_weights_input;
        if (!csi_weights_path.empty()) {
            csi_weights_input.open(csi_weights_path, std::ios::binary);
            if (!csi_weights_input) {
                throw std::runtime_error(
                    "failed to open CSI weights input: " + csi_weights_path);
            }
        }
        std::unique_ptr<std::ofstream> symbols_output_file;
        std::ostream* symbols_output = nullptr;
        if (!symbols_output_path.empty()) {
            symbols_output_file = std::make_unique<std::ofstream>(
                symbols_output_path,
                std::ios::binary);
            if (!*symbols_output_file) {
                throw std::runtime_error(
                    "failed to open symbols output: " + symbols_output_path);
            }
            symbols_output = symbols_output_file.get();
        }

        dtmb::core::SymbolDeinterleaverCf32 deinterleaver(mode, phase);
        dtmb::core::SymbolDeinterleaverCf32 csi_deinterleaver(mode, phase);
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
        const auto source_carrier_fixed_complex_gains =
            source_carrier_fixed_complex_gains_path.empty()
            ? std::vector<std::complex<float>>{}
            : read_source_carrier_fixed_complex_gains(
                source_carrier_fixed_complex_gains_path);
        std::size_t source_carrier_fixed_complex_gain_corrected_symbols = 0;
        auto symbol_gain_stats = make_symbol_gain_stats(symbol_gain_rules);
        auto fixed_complex_gain_stats =
            make_fixed_complex_gain_stats(fixed_complex_gain_rules);
        auto carrier_symbol_gain_stats =
            make_carrier_symbol_gain_stats(carrier_symbol_gain_rules);
        auto carrier_axis_affine_stats =
            make_carrier_axis_affine_stats(carrier_axis_affine_rules);
        auto carrier_linear_affine_stats =
            make_carrier_linear_affine_stats(carrier_linear_affine_rules);
        auto carrier_axis_quantile_stats =
            make_carrier_axis_quantile_stats(carrier_axis_quantile_rules);
        auto carrier_axis_centroid_stats =
            make_carrier_axis_centroid_stats(carrier_axis_centroid_rules);
        auto carrier_axis_centroid_median_stats =
            make_carrier_axis_centroid_stats(carrier_axis_centroid_median_rules);
        auto carrier_qam_centroid_stats =
            make_carrier_qam_centroid_stats(carrier_qam_centroid_rules);
        auto carrier_neighbor_leakage_stats =
            make_carrier_neighbor_leakage_stats(carrier_neighbor_leakage_rules);
        auto carrier_neighbor_leakage_set_stats =
            make_carrier_neighbor_leakage_set_stats(carrier_neighbor_leakage_set_rules);
        auto carrier_roll_stats = make_carrier_roll_stats(carrier_roll_rules);
        auto source_displacement_stats =
            make_source_frame_displacement_stats(source_displacement_rules);
        const auto source_displacement_window =
            make_source_frame_displacement_window(
                source_displacement_rules,
                mode,
                phase);
        const auto source_displacement_window_symbol_count =
            source_frame_displacement_window_symbols(source_displacement_window);
        auto axis_affine_stats = make_axis_affine_stats(axis_affine_rules);
        auto llr_scale_stats = make_llr_scale_stats(llr_scale_rules);
        auto carrier_llr_scale_stats = make_carrier_llr_scale_stats(carrier_llr_scale_rules);
        auto carrier_residual_llr_confidence_stats =
            make_carrier_residual_llr_confidence_stats(
                carrier_residual_llr_confidence_rules);
        const auto& ramp_geometry = ramp_geometry_for(ramp_llr_erasure_options.coordinate);
        std::vector<float> source_frame_ramp_metrics;

        std::vector<float> input_chunk(chunk_symbols * kFloatsPerSymbol);
        std::vector<float> csi_scalar_chunk(chunk_symbols);
        std::vector<float> csi_input_chunk(chunk_symbols * kFloatsPerSymbol);
        std::vector<float> csi_deinterleaved_chunk(chunk_symbols * kFloatsPerSymbol);
        std::vector<float> carrier_roll_scratch;
        std::vector<float> source_history(
            source_displacement_window_symbol_count * kFloatsPerSymbol);
        std::size_t source_history_available_end_symbol =
            source_displacement_window.enabled
                ? source_displacement_window.first_symbol
                : std::size_t{0};
        std::vector<float> deinterleaved_chunk(chunk_symbols * kFloatsPerSymbol);
        std::vector<float> output_chunk(chunk_symbols * kLlrsPerSymbol);
        std::size_t written_output_symbols = 0;
        const auto input_chunk_bytes = static_cast<std::streamsize>(
            input_chunk.size() * sizeof(float));

        std::size_t written_symbol_output_symbols = 0;
        std::size_t csi_weighted_symbols = 0;
        double csi_weight_sum = 0.0;
        float csi_weight_min = std::numeric_limits<float>::infinity();
        float csi_weight_max = 0.0F;
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
            if (csi_weights_input.is_open()) {
                const auto csi_bytes = static_cast<std::streamsize>(
                    symbols_read * sizeof(float));
                csi_weights_input.read(
                    reinterpret_cast<char*>(csi_scalar_chunk.data()),
                    csi_bytes);
                if (csi_weights_input.gcount() != csi_bytes) {
                    throw std::runtime_error(
                        "CSI weight count does not match input symbol count");
                }
                for (std::size_t symbol = 0; symbol < symbols_read; ++symbol) {
                    csi_input_chunk[symbol * 2] = csi_scalar_chunk[symbol];
                    csi_input_chunk[symbol * 2 + 1] = 0.0F;
                }
                csi_deinterleaver.process(
                    std::span<const float>(csi_input_chunk.data(), symbols_read * 2),
                    std::span<float>(
                        csi_deinterleaved_chunk.data(),
                        symbols_read * 2));
            }
            const auto input_values = std::span<const float>(
                input_chunk.data(),
                symbols_read * kFloatsPerSymbol);
            auto deinterleaved_values = std::span<float>(
                deinterleaved_chunk.data(),
                symbols_read * kFloatsPerSymbol);
            auto mutable_input_values = std::span<float>(
                input_chunk.data(),
                symbols_read * kFloatsPerSymbol);
            apply_source_frame_carrier_roll(
                mutable_input_values,
                input_symbols,
                carrier_roll_rules,
                carrier_roll_stats,
                carrier_roll_scratch);
            score_source_frame_phase_ramps(
                mutable_input_values,
                input_symbols,
                ramp_llr_erasure_options,
                ramp_geometry,
                source_frame_ramp_metrics,
                ramp_llr_erasure_stats);
            store_source_frame_displacement_history(
                mutable_input_values,
                input_symbols,
                source_displacement_window,
                source_history,
                source_history_available_end_symbol);
            deinterleaver.process(input_values, deinterleaved_values);
            input_symbols += symbols_read;

            const auto discard_now = std::min(discard_remaining, symbols_read);
            discard_remaining -= discard_now;
            const auto useful_symbols = symbols_read - discard_now;
            if (useful_symbols != 0) {
                auto useful_input = std::span<float>(
                    deinterleaved_chunk.data() + discard_now * kFloatsPerSymbol,
                    useful_symbols * kFloatsPerSymbol);
                apply_source_frame_displacement(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    source_displacement_window,
                    source_history,
                    source_history_available_end_symbol,
                    source_displacement_rules,
                    source_displacement_stats);
                apply_branch_gain_correction(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    branch_gain_stats,
                    branch_gain_frame_stats);
                apply_source_frame_fixed_complex_gain(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    fixed_complex_gain_rules,
                    fixed_complex_gain_stats);
                source_carrier_fixed_complex_gain_corrected_symbols +=
                    apply_source_carrier_fixed_complex_gains(
                        useful_input,
                        output_symbols,
                        mode,
                        phase,
                        source_carrier_fixed_complex_gains);
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
                apply_source_carrier_linear_affine(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_linear_affine_rules,
                    carrier_linear_affine_stats);
                apply_source_carrier_axis_affine(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_axis_affine_rules,
                    carrier_axis_affine_stats);
                apply_source_carrier_axis_centroid(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_axis_centroid_rules,
                    carrier_axis_centroid_stats);
                apply_source_carrier_axis_centroid_median(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_axis_centroid_median_rules,
                    carrier_axis_centroid_median_stats);
                apply_source_carrier_qam_centroid(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_qam_centroid_rules,
                    carrier_qam_centroid_stats);
                apply_source_carrier_neighbor_leakage(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_neighbor_leakage_rules,
                    carrier_neighbor_leakage_stats);
                apply_source_carrier_neighbor_leakage_set(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_neighbor_leakage_set_rules,
                    carrier_neighbor_leakage_set_stats);
                apply_source_carrier_axis_quantile(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    carrier_axis_quantile_rules,
                    carrier_axis_quantile_stats);
                apply_source_frame_axis_affine(
                    useful_input,
                    output_symbols,
                    mode,
                    phase,
                    branch_gain_options,
                    axis_affine_rules,
                    axis_affine_stats);
                if (symbols_output != nullptr) {
                    written_symbol_output_symbols += write_symbol_output_frame_ranges(
                        *symbols_output,
                        useful_input,
                        output_symbols,
                        output_frame_ranges);
                }
                auto useful_output = std::span<float>(
                    output_chunk.data(),
                    useful_symbols * kLlrsPerSymbol);
                dtmb::core::qam64_soft_demodulate_cf32(
                    useful_input,
                    useful_output,
                    demap_options);
                if (csi_weights_input.is_open()) {
                    const auto* weights = csi_deinterleaved_chunk.data()
                        + discard_now * kFloatsPerSymbol;
                    for (std::size_t symbol = 0; symbol < useful_symbols; ++symbol) {
                        const auto weight = weights[symbol * 2];
                        if (!std::isfinite(weight) || weight < 0.0F) {
                            throw std::runtime_error(
                                "CSI weights must be finite and non-negative");
                        }
                        for (std::size_t bit = 0; bit < kLlrsPerSymbol; ++bit) {
                            useful_output[symbol * kLlrsPerSymbol + bit] *= weight;
                        }
                        ++csi_weighted_symbols;
                        csi_weight_sum += weight;
                        csi_weight_min = std::min(csi_weight_min, weight);
                        csi_weight_max = std::max(csi_weight_max, weight);
                    }
                }
                apply_dd_llr_confidence(
                    useful_input,
                    useful_output,
                    dd_llr_confidence_options,
                    dd_llr_confidence_stats);
                apply_source_frame_llr_scale(
                    useful_output,
                    output_symbols,
                    mode,
                    phase,
                    llr_scale_rules,
                    llr_scale_stats);
                apply_source_frame_ramp_llr_erasure(
                    useful_output,
                    output_symbols,
                    mode,
                    phase,
                    ramp_llr_erasure_options,
                    source_frame_ramp_metrics,
                    ramp_llr_erasure_stats);
                apply_source_carrier_residual_llr_confidence(
                    useful_input,
                    useful_output,
                    output_symbols,
                    mode,
                    phase,
                    carrier_residual_llr_confidence_rules,
                    carrier_residual_llr_confidence_stats);
                apply_source_carrier_llr_scale(
                    useful_output,
                    output_symbols,
                    mode,
                    phase,
                    carrier_llr_scale_rules,
                    carrier_llr_scale_stats);
                written_output_symbols += write_output_frame_ranges(
                    output,
                    useful_output,
                    output_symbols,
                    output_frame_ranges);
                output_symbols += useful_symbols;
            }

            if (bytes_read < input_chunk_bytes) {
                break;
            }
        }

        if (csi_weights_input.is_open()) {
            char trailing = 0;
            if (csi_weights_input.read(&trailing, 1)) {
                throw std::runtime_error("CSI weight input has trailing values");
            }
        }

        std::cerr << "input_symbols=" << input_symbols << '\n'
                  << "discarded_latency_symbols="
                  << (keep_latency ? 0 : deinterleaver.latency_symbols() - discard_remaining)
                  << '\n'
                  << "output_symbols=" << output_symbols << '\n'
                  << "written_output_symbols=" << written_output_symbols << '\n'
                  << "written_symbol_output_symbols="
                  << written_symbol_output_symbols << '\n'
                  << "csi_weights="
                  << (csi_weights_path.empty() ? "none" : csi_weights_path) << '\n'
                  << "csi_weighted_symbols=" << csi_weighted_symbols << '\n'
                  << "csi_weight_mean="
                  << (csi_weighted_symbols == 0
                          ? 0.0
                          : csi_weight_sum / static_cast<double>(csi_weighted_symbols))
                  << '\n'
                  << "csi_weight_min="
                  << (csi_weighted_symbols == 0 ? 0.0F : csi_weight_min) << '\n'
                  << "csi_weight_max=" << csi_weight_max << '\n'
                  << "output_frame_range_count=" << output_frame_ranges.size() << '\n'
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
        std::cerr << "source_frame_fixed_complex_gain_rule_count="
                  << fixed_complex_gain_stats.size() << '\n';
        for (const auto& stats : fixed_complex_gain_stats) {
            std::cerr << "source_frame_fixed_complex_gain_rule_" << stats.rule_index
                      << "_branch=" << stats.branch << '\n'
                      << "source_frame_fixed_complex_gain_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_frame_fixed_complex_gain_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_frame_fixed_complex_gain_rule_" << stats.rule_index
                      << "_gain_real=" << stats.gain.real() << '\n'
                      << "source_frame_fixed_complex_gain_rule_" << stats.rule_index
                      << "_gain_imag=" << stats.gain.imag() << '\n'
                      << "source_frame_fixed_complex_gain_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n';
        }
        std::cerr << "source_carrier_fixed_complex_gains="
                  << (source_carrier_fixed_complex_gains.empty()
                      ? "none"
                      : source_carrier_fixed_complex_gains_path)
                  << '\n'
                  << "source_carrier_fixed_complex_gain_corrected_symbols="
                  << source_carrier_fixed_complex_gain_corrected_symbols << '\n';
        std::cerr << "source_carrier_symbol_gain_rule_count="
                  << carrier_symbol_gain_stats.size() << '\n';
        print_source_frame_ranges(
            "source_carrier_fit_exclude_source_frame_range",
            branch_gain_options.carrier_fit_exclude_source_frame_ranges);
        print_source_frame_ranges(
            "source_carrier_symbol_gain_fit_exclude_source_frame_range",
            branch_gain_options.carrier_symbol_gain_fit_exclude_source_frame_ranges);
        print_source_carrier_frame_ranges(
            "source_carrier_symbol_gain_fit_exclude_source_carrier_frame_range",
            branch_gain_options
                .carrier_symbol_gain_fit_exclude_source_carrier_frame_ranges);
        print_source_frame_ranges(
            "source_carrier_linear_affine_fit_exclude_source_frame_range",
            branch_gain_options.carrier_linear_affine_fit_exclude_source_frame_ranges);
        print_source_carrier_frame_ranges(
            "source_carrier_linear_affine_fit_exclude_source_carrier_frame_range",
            branch_gain_options
                .carrier_linear_affine_fit_exclude_source_carrier_frame_ranges);
        print_source_frame_ranges(
            "source_carrier_axis_centroid_fit_exclude_source_frame_range",
            branch_gain_options.carrier_axis_centroid_fit_exclude_source_frame_ranges);
        print_source_carrier_frame_ranges(
            "source_carrier_axis_centroid_fit_exclude_source_carrier_frame_range",
            branch_gain_options
                .carrier_axis_centroid_fit_exclude_source_carrier_frame_ranges);
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
        std::cerr << "source_carrier_axis_affine_rule_count="
                  << carrier_axis_affine_stats.size() << '\n';
        for (const auto& stats : carrier_axis_affine_stats) {
            std::cerr << "source_carrier_axis_affine_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_axis_affine_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_axis_affine_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_axis_affine_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_axis_affine_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n';
            if (stats.applied_chunks != 0) {
                const auto applied_chunks = static_cast<double>(stats.applied_chunks);
                std::cerr << "source_carrier_axis_affine_rule_" << stats.rule_index
                          << "_mean_real_scale="
                          << (stats.real_scale_sum / applied_chunks) << '\n'
                          << "source_carrier_axis_affine_rule_" << stats.rule_index
                          << "_mean_real_bias="
                          << (stats.real_bias_sum / applied_chunks) << '\n'
                          << "source_carrier_axis_affine_rule_" << stats.rule_index
                          << "_mean_imag_scale="
                          << (stats.imag_scale_sum / applied_chunks) << '\n'
                          << "source_carrier_axis_affine_rule_" << stats.rule_index
                          << "_mean_imag_bias="
                          << (stats.imag_bias_sum / applied_chunks) << '\n';
            }
        }
        std::cerr << "source_carrier_linear_affine_rule_count="
                  << carrier_linear_affine_stats.size() << '\n';
        for (const auto& stats : carrier_linear_affine_stats) {
            std::cerr << "source_carrier_linear_affine_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_linear_affine_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_linear_affine_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_linear_affine_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_linear_affine_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n';
            if (stats.applied_chunks != 0) {
                const auto applied_chunks = static_cast<double>(stats.applied_chunks);
                std::cerr << "source_carrier_linear_affine_rule_" << stats.rule_index
                          << "_mean_real_from_real="
                          << (stats.real_from_real_sum / applied_chunks) << '\n'
                          << "source_carrier_linear_affine_rule_" << stats.rule_index
                          << "_mean_real_from_imag="
                          << (stats.real_from_imag_sum / applied_chunks) << '\n'
                          << "source_carrier_linear_affine_rule_" << stats.rule_index
                          << "_mean_real_bias="
                          << (stats.real_bias_sum / applied_chunks) << '\n'
                          << "source_carrier_linear_affine_rule_" << stats.rule_index
                          << "_mean_imag_from_real="
                          << (stats.imag_from_real_sum / applied_chunks) << '\n'
                          << "source_carrier_linear_affine_rule_" << stats.rule_index
                          << "_mean_imag_from_imag="
                          << (stats.imag_from_imag_sum / applied_chunks) << '\n'
                          << "source_carrier_linear_affine_rule_" << stats.rule_index
                          << "_mean_imag_bias="
                          << (stats.imag_bias_sum / applied_chunks) << '\n';
            }
        }
        std::cerr << "source_carrier_axis_centroid_rule_count="
                  << carrier_axis_centroid_stats.size() << '\n';
        for (const auto& stats : carrier_axis_centroid_stats) {
            std::cerr << "source_carrier_axis_centroid_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_axis_centroid_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_axis_centroid_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_axis_centroid_rule_" << stats.rule_index
                      << "_matched_symbols=" << stats.matched_symbols << '\n'
                      << "source_carrier_axis_centroid_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_axis_centroid_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n'
                      << "source_carrier_axis_centroid_rule_" << stats.rule_index
                      << "_applied_chunks=" << stats.applied_chunks << '\n';
        }
        std::cerr << "source_carrier_axis_centroid_median_rule_count="
                  << carrier_axis_centroid_median_stats.size() << '\n';
        for (const auto& stats : carrier_axis_centroid_median_stats) {
            std::cerr << "source_carrier_axis_centroid_median_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_axis_centroid_median_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_axis_centroid_median_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_axis_centroid_median_rule_" << stats.rule_index
                      << "_matched_symbols=" << stats.matched_symbols << '\n'
                      << "source_carrier_axis_centroid_median_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_axis_centroid_median_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n'
                      << "source_carrier_axis_centroid_median_rule_" << stats.rule_index
                      << "_applied_chunks=" << stats.applied_chunks << '\n';
        }
        std::cerr << "source_carrier_qam_centroid_rule_count="
                  << carrier_qam_centroid_stats.size() << '\n';
        for (const auto& stats : carrier_qam_centroid_stats) {
            std::cerr << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_matched_symbols=" << stats.matched_symbols << '\n'
                      << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_centroid_points=" << stats.centroid_points << '\n'
                      << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n'
                      << "source_carrier_qam_centroid_rule_" << stats.rule_index
                      << "_applied_chunks=" << stats.applied_chunks << '\n';
        }
        std::cerr << "source_carrier_neighbor_leakage_rule_count="
                  << carrier_neighbor_leakage_stats.size() << '\n';
        for (const auto& stats : carrier_neighbor_leakage_stats) {
            std::cerr << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_offset=" << stats.offset << '\n'
                      << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_matched_symbols=" << stats.matched_symbols << '\n'
                      << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n'
                      << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                      << "_applied_chunks=" << stats.applied_chunks << '\n';
            if (stats.applied_chunks != 0) {
                const auto applied_chunks = static_cast<double>(stats.applied_chunks);
                std::cerr << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                          << "_mean_leakage_real="
                          << (stats.leakage_real_sum / applied_chunks) << '\n'
                          << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                          << "_mean_leakage_imag="
                          << (stats.leakage_imag_sum / applied_chunks) << '\n'
                          << "source_carrier_neighbor_leakage_rule_" << stats.rule_index
                          << "_mean_leakage_abs="
                          << (stats.leakage_abs_sum / applied_chunks) << '\n';
            }
        }
        std::cerr << "source_carrier_neighbor_leakage_set_rule_count="
                  << carrier_neighbor_leakage_set_stats.size() << '\n';
        for (const auto& stats : carrier_neighbor_leakage_set_stats) {
            std::cerr << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_offset_count=" << stats.offset_count << '\n'
                      << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_matched_symbols=" << stats.matched_symbols << '\n'
                      << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_reliable_symbols=" << stats.reliable_symbols << '\n'
                      << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n'
                      << "source_carrier_neighbor_leakage_set_rule_" << stats.rule_index
                      << "_applied_chunks=" << stats.applied_chunks << '\n';
            if (stats.applied_chunks != 0 && stats.offset_count != 0) {
                const auto scale = static_cast<double>(
                    stats.applied_chunks * stats.offset_count);
                std::cerr << "source_carrier_neighbor_leakage_set_rule_"
                          << stats.rule_index << "_mean_leakage_abs="
                          << (stats.leakage_abs_sum / scale) << '\n'
                          << "source_carrier_neighbor_leakage_set_rule_"
                          << stats.rule_index << "_max_leakage_abs="
                          << stats.leakage_abs_max << '\n';
            }
        }
        std::cerr << "source_carrier_axis_quantile_rule_count="
                  << carrier_axis_quantile_stats.size() << '\n';
        for (const auto& stats : carrier_axis_quantile_stats) {
            std::cerr << "source_carrier_axis_quantile_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_axis_quantile_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_axis_quantile_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_axis_quantile_rule_" << stats.rule_index
                      << "_matched_symbols=" << stats.matched_symbols << '\n'
                      << "source_carrier_axis_quantile_rule_" << stats.rule_index
                      << "_corrected_symbols=" << stats.corrected_symbols << '\n'
                      << "source_carrier_axis_quantile_rule_" << stats.rule_index
                      << "_applied_chunks=" << stats.applied_chunks << '\n';
        }
        std::cerr << "source_frame_carrier_roll_rule_count="
                  << carrier_roll_stats.size() << '\n';
        for (const auto& stats : carrier_roll_stats) {
            std::cerr << "source_frame_carrier_roll_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_frame_carrier_roll_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_frame_carrier_roll_rule_" << stats.rule_index
                      << "_shift=" << stats.shift << '\n'
                      << "source_frame_carrier_roll_rule_" << stats.rule_index
                      << "_rolled_frames=" << stats.rolled_frames << '\n'
                      << "source_frame_carrier_roll_rule_" << stats.rule_index
                      << "_rolled_symbols=" << stats.rolled_symbols << '\n';
        }
        std::cerr << "source_frame_displacement_rule_count="
                  << source_displacement_stats.size() << '\n';
        std::cerr << "source_frame_displacement_source_window_enabled="
                  << (source_displacement_window.enabled ? "true" : "false") << '\n'
                  << "source_frame_displacement_source_window_first_symbol="
                  << source_displacement_window.first_symbol << '\n'
                  << "source_frame_displacement_source_window_last_symbol="
                  << source_displacement_window.last_symbol << '\n'
                  << "source_frame_displacement_source_window_symbols="
                  << source_displacement_window_symbol_count << '\n'
                  << "source_frame_displacement_source_history_available_end_symbol="
                  << source_history_available_end_symbol << '\n';
        for (const auto& stats : source_displacement_stats) {
            std::cerr << "source_frame_displacement_rule_" << stats.rule_index
                      << "_fec_frame=" << stats.fec_frame << '\n'
                      << "source_frame_displacement_rule_" << stats.rule_index
                      << "_codeword_slot=" << stats.codeword_slot << '\n'
                      << "source_frame_displacement_rule_" << stats.rule_index
                      << "_source_frame_shift=" << stats.source_frame_shift << '\n'
                      << "source_frame_displacement_rule_" << stats.rule_index
                      << "_replaced_symbols=" << stats.replaced_symbols << '\n';
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
        std::cerr << "source_frame_ramp_llr_erasure_enabled="
                  << (ramp_llr_erasure_options.enabled ? "true" : "false") << '\n'
                  << "source_frame_ramp_llr_erasure_threshold="
                  << ramp_llr_erasure_options.threshold << '\n'
                  << "source_frame_ramp_llr_erasure_scale="
                  << ramp_llr_erasure_options.scale << '\n'
                  << "source_frame_ramp_llr_erasure_coordinate="
                  << ramp_coordinate_name(ramp_llr_erasure_options.coordinate) << '\n'
                  << "source_frame_ramp_llr_erasure_scored_frames="
                  << ramp_llr_erasure_stats.scored_frames << '\n'
                  << "source_frame_ramp_llr_erasure_selected_frames="
                  << ramp_llr_erasure_stats.selected_frames << '\n'
                  << "source_frame_ramp_llr_erasure_touched_symbols="
                  << ramp_llr_erasure_stats.touched_symbols << '\n'
                  << "source_frame_ramp_llr_erasure_scaled_symbols="
                  << ramp_llr_erasure_stats.scaled_symbols << '\n'
                  << "source_frame_ramp_llr_erasure_out_of_metric_symbols="
                  << ramp_llr_erasure_stats.out_of_metric_symbols << '\n';
        if (ramp_llr_erasure_stats.scored_frames != 0) {
            std::cerr << "source_frame_ramp_llr_erasure_mean_abs_ramp="
                      << (ramp_llr_erasure_stats.abs_ramp_sum
                          / static_cast<double>(ramp_llr_erasure_stats.scored_frames))
                      << '\n'
                      << "source_frame_ramp_llr_erasure_max_abs_ramp="
                      << ramp_llr_erasure_stats.max_abs_ramp << '\n';
        }
        if (ramp_llr_erasure_stats.selected_frames != 0) {
            std::cerr << "source_frame_ramp_llr_erasure_mean_selected_abs_ramp="
                      << (ramp_llr_erasure_stats.selected_abs_ramp_sum
                          / static_cast<double>(ramp_llr_erasure_stats.selected_frames))
                      << '\n';
        }
        std::cerr << "source_carrier_residual_llr_confidence_rule_count="
                  << carrier_residual_llr_confidence_stats.size() << '\n';
        for (const auto& stats : carrier_residual_llr_confidence_stats) {
            std::cerr << "source_carrier_residual_llr_confidence_rule_"
                      << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_residual_llr_confidence_rule_"
                      << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_residual_llr_confidence_rule_"
                      << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_residual_llr_confidence_rule_"
                      << stats.rule_index
                      << "_residual_knee=" << stats.residual_knee << '\n'
                      << "source_carrier_residual_llr_confidence_rule_"
                      << stats.rule_index
                      << "_min_scale=" << stats.min_scale << '\n'
                      << "source_carrier_residual_llr_confidence_rule_"
                      << stats.rule_index
                      << "_matched_symbols=" << stats.matched_symbols << '\n'
                      << "source_carrier_residual_llr_confidence_rule_"
                      << stats.rule_index
                      << "_scaled_symbols=" << stats.scaled_symbols << '\n';
            if (stats.scaled_symbols != 0) {
                const auto scaled = static_cast<double>(stats.scaled_symbols);
                std::cerr << "source_carrier_residual_llr_confidence_rule_"
                          << stats.rule_index
                          << "_mean_scale=" << (stats.scale_sum / scaled) << '\n'
                          << "source_carrier_residual_llr_confidence_rule_"
                          << stats.rule_index
                          << "_min_observed_scale=" << stats.min_observed_scale << '\n'
                          << "source_carrier_residual_llr_confidence_rule_"
                          << stats.rule_index
                          << "_mean_scaled_residual_ratio="
                          << (stats.scaled_residual_ratio_sum / scaled) << '\n';
            }
        }
        std::cerr << "source_carrier_llr_scale_rule_count="
                  << carrier_llr_scale_stats.size() << '\n';
        for (const auto& stats : carrier_llr_scale_stats) {
            std::cerr << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_carrier=" << stats.carrier << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_last_carrier=" << stats.last_carrier << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_first_frame=" << stats.first_frame << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_last_frame=" << stats.last_frame << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_scale=" << stats.scale << '\n'
                      << "source_carrier_llr_scale_rule_" << stats.rule_index
                      << "_scaled_symbols=" << stats.scaled_symbols << '\n';
        }
        std::cerr << "dd_llr_confidence_enabled="
                  << (dd_llr_confidence_options.enabled ? "true" : "false") << '\n'
                  << "dd_llr_confidence_knee="
                  << dd_llr_confidence_options.residual_knee << '\n'
                  << "dd_llr_confidence_min_scale="
                  << dd_llr_confidence_options.min_scale << '\n'
                  << "dd_llr_confidence_symbols="
                  << dd_llr_confidence_stats.symbols << '\n'
                  << "dd_llr_confidence_scaled_symbols="
                  << dd_llr_confidence_stats.scaled_symbols << '\n';
        if (dd_llr_confidence_stats.scaled_symbols != 0) {
            const auto scaled = static_cast<double>(
                dd_llr_confidence_stats.scaled_symbols);
            std::cerr << "dd_llr_confidence_mean_scale="
                      << (dd_llr_confidence_stats.scale_sum / scaled) << '\n'
                      << "dd_llr_confidence_min_observed_scale="
                      << dd_llr_confidence_stats.min_observed_scale << '\n'
                      << "dd_llr_confidence_mean_scaled_residual_ratio="
                      << (dd_llr_confidence_stats.scaled_residual_ratio_sum / scaled)
                      << '\n';
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
