#include "dtmb/core.hpp"

#include "binary_stdio.hpp"
#include "ldpc_h_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <thread>
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
        << " [--workers N] [--max-iterations N] [--attenuation X]"
        << " [--retry-llr-clip X] [--retry-weak-erase-fraction X]"
        << " [--retry-attenuation X] [--retry-layered]"
        << " [--decode-batch-frames N]"
        << " [--early-syndrome-reject-ratio off|X]"
        << " [--clean-frames-only] [--mark-discontinuities]"
        << " [--emit-clean-codewords]"
        << " [--emit-bch-clean-codewords]"
        << " [--insert-discontinuity-packets]"
        << " [--hard-h-gate-window-codewords N"
        << " --hard-h-gate-step-codewords N --hard-h-gate-threshold R]"
        << " [--hard-h-gate-fill-max-gap-frames N]"
        << " [--force-frame-range FIRST:LAST]"
        << " [--llr-scale-codeword FIRST:LAST:SLOT:SCALE]"
        << " [--llr-scale-variable FIRST:LAST:VARIABLE:SCALE]"
        << " [--llr-scale-variable-mod FIRST:LAST:MOD:REMAINDER:SCALE]"
        << " [--emit-forced-codewords]"
        << " [--frame-diagnostics-out PATH]"
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

bool frame_in_ranges(std::size_t frame_index, std::span<const FrameRange> ranges) {
    return std::any_of(
        ranges.begin(),
        ranges.end(),
        [frame_index](const FrameRange& range) {
            return frame_index >= range.first && frame_index <= range.last;
        });
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

[[nodiscard]] bool ts_packet_aligned(std::span<const std::uint8_t> bytes) {
    if (bytes.empty() || (bytes.size() % kTsPacketSize) != 0U) {
        return false;
    }
    for (std::size_t offset = 0; offset < bytes.size(); offset += kTsPacketSize) {
        if (bytes[offset] != kTsSyncByte) {
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

struct BufferedFecFrame {
    std::vector<float> transmitted_llr;
    std::size_t frame_index = 0;
    std::size_t clean_syndrome_weight = 0;
    bool selected = false;
    bool force_selected = false;
    bool hard_h_gap_filled = false;
};

struct PendingFecFrame {
    std::vector<float> transmitted_llr;
    std::size_t frame_index = 0;
    std::size_t clean_syndrome_weight = 0;
    bool selected = false;
    bool force_selected = false;
    bool hard_h_gap_filled = false;
};

struct DecodedFecFrame {
    std::vector<float> full_llr;
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
    std::vector<double> retry_llr_clip;
    std::vector<double> retry_weak_erase_fraction;
    std::vector<double> retry_attenuation;
    std::vector<std::size_t> retry_bch_corrected_errors;
    dtmb::core::DtmbBchDecodeStats bch;
    bool frame_clean = false;
};

void condition_retry_llr(
    std::span<const float> input,
    std::span<float> output,
    std::size_t first_transmitted_variable,
    float clip,
    float erase_fraction) {
    std::copy(input.begin(), input.end(), output.begin());
    auto transmitted = output.subspan(first_transmitted_variable);
    if (clip > 0.0F) {
        for (auto& llr : transmitted) {
            llr = std::clamp(llr, -clip, clip);
        }
    }
    const auto erase_count = static_cast<std::size_t>(
        std::floor(static_cast<double>(transmitted.size()) * erase_fraction));
    if (erase_count == 0) {
        return;
    }
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
    std::vector<std::string> positional;
    dtmb::core::LdpcDecodeOptions decode_options{};
    std::size_t decode_batch_frames = 16;
    bool clean_frames_only = false;
    bool emit_clean_codewords = false;
    bool emit_bch_clean_codewords = false;
    bool emit_forced_codewords = false;
    bool mark_discontinuities = false;
    bool insert_discontinuity_packets = false;
    float early_syndrome_reject_ratio = -1.0F;
    std::optional<std::size_t> hard_h_gate_window_codewords;
    std::optional<std::size_t> hard_h_gate_step_codewords;
    std::optional<double> hard_h_gate_threshold;
    std::size_t hard_h_gate_fill_max_gap_frames = 0;
    std::vector<FrameRange> force_frame_ranges;
    std::vector<CodewordLlrScaleRule> codeword_llr_scale_rules;
    std::vector<VariableLlrScaleRule> variable_llr_scale_rules;
    std::vector<VariableModLlrScaleRule> variable_mod_llr_scale_rules;
    std::vector<float> retry_llr_clips;
    std::vector<float> retry_weak_erase_fractions;
    std::vector<float> retry_attenuations;
    bool retry_layered = false;

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
            } else if (arg == "--attenuation") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                decode_options.attenuation = parse_float(argv[index], "attenuation");
            } else if (arg == "--retry-llr-clip") {
                if (++index >= argc) {
                    usage(argv[0]);
                    return 2;
                }
                retry_llr_clips.push_back(parse_float(argv[index], "retry LLR clip"));
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
            } else if (arg == "--retry-layered") {
                retry_layered = true;
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
            } else if (arg == "--emit-clean-codewords") {
                emit_clean_codewords = true;
            } else if (arg == "--emit-bch-clean-codewords") {
                emit_bch_clean_codewords = true;
            } else if (arg == "--emit-forced-codewords") {
                emit_forced_codewords = true;
            } else if (arg == "--mark-discontinuities") {
                mark_discontinuities = true;
            } else if (arg == "--insert-discontinuity-packets") {
                insert_discontinuity_packets = true;
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
        if (emit_clean_codewords && !clean_frames_only) {
            throw std::invalid_argument(
                "emit-clean-codewords requires --clean-frames-only");
        }
        if (emit_bch_clean_codewords && !emit_clean_codewords) {
            throw std::invalid_argument(
                "emit-bch-clean-codewords requires --emit-clean-codewords");
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
            || !retry_weak_erase_fractions.empty()
            || !retry_attenuations.empty()
            || retry_layered;
        const auto retry_clip_axis =
            retry_llr_clips.empty() ? std::vector<float>{0.0F} : retry_llr_clips;
        const auto retry_erase_axis =
            retry_weak_erase_fractions.empty()
                ? std::vector<float>{0.0F}
                : retry_weak_erase_fractions;
        const auto retry_attenuation_axis =
            retry_attenuations.empty()
                ? std::vector<float>{decode_options.attenuation}
                : retry_attenuations;
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

        auto worker_count = requested_workers;
        if (worker_count == 0) {
            worker_count = std::thread::hardware_concurrency();
        }
        worker_count = std::max<std::size_t>(worker_count, 1);

        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> output_file;
        std::unique_ptr<std::ofstream> frame_diagnostics_file;
        auto& input = input_stream(input_path, input_file);
        auto& output = output_stream(output_path, output_file);
        std::ostream* frame_diagnostics = nullptr;
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

        const auto frame_llr_count = codewords_per_frame * profile.transmitted_bits;
        std::vector<float> transmitted_llr(frame_llr_count);
        const auto frame_transport_bytes =
            codewords_per_frame * profile.output_bytes_per_codeword();
        const auto codeword_transport_bytes = profile.output_bytes_per_codeword();
        if ((codeword_transport_bytes % kTsPacketSize) != 0) {
            throw std::runtime_error(
                "DTMB codeword transport bytes must align to MPEG-TS packets");
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
        std::size_t emitted_codeword_count = 0;
        std::size_t omitted_codeword_count = 0;
        std::size_t output_byte_count = 0;
        std::size_t discontinuity_marked_packets = 0;
        std::size_t discontinuity_inserted_packets = 0;
        std::size_t hard_h_gate_scored_windows = 0;
        std::size_t hard_h_gate_pass_windows = 0;
        std::size_t hard_h_gate_selected_frames = 0;
        std::size_t forced_selected_frames = 0;
        std::size_t hard_h_gate_gap_filled_frames = 0;
        std::size_t hard_h_gate_skipped_frames = 0;
        std::size_t soft_decoded_frames = 0;
        std::size_t retry_attempt_count = 0;
        std::size_t retry_recovered_codeword_count = 0;
        std::size_t layered_retry_recovered_codeword_count = 0;
        std::unordered_set<std::uint16_t> seen_payload_pids;
        std::unordered_set<std::uint16_t> pending_discontinuity_pids;

        const auto note_transport_gap = [&] {
            if (!mark_discontinuities && !insert_discontinuity_packets) {
                return;
            }
            pending_discontinuity_pids.insert(
                seen_payload_pids.begin(),
                seen_payload_pids.end());
        };

        const auto signal_pending_pid_discontinuities = [&](
            std::span<std::uint8_t> bytes) {
            std::unordered_set<std::uint16_t> signaled_pids;
            std::size_t inserted = 0;
            std::size_t marked = 0;
            if (!pending_discontinuity_pids.empty() && insert_discontinuity_packets) {
                const auto packets = make_discontinuity_packets_for_pending_pids(
                    bytes,
                    pending_discontinuity_pids,
                    signaled_pids);
                for (const auto& packet : packets) {
                    write_all(output, packet);
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
            std::size_t frame_discontinuity_inserted_packets) {
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
            std::vector<double> codeword_retry_llr_clip(codewords_per_frame, 0.0);
            std::vector<double> codeword_retry_weak_erase_fraction(
                codewords_per_frame,
                0.0);
            std::vector<double> codeword_retry_attenuation(codewords_per_frame, 0.0);
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
                codeword_retry_llr_clip = decoded_frame->retry_llr_clip;
                codeword_retry_weak_erase_fraction =
                    decoded_frame->retry_weak_erase_fraction;
                codeword_retry_attenuation = decoded_frame->retry_attenuation;
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
                        && ts_packet_aligned(codeword_bytes);
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
                << frame_discontinuity_inserted_packets;
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
                "codeword_retry_bch_corrected_errors",
                codeword_retry_bch_corrected_errors);
            *frame_diagnostics << "}\n";
            if (!*frame_diagnostics) {
                throw std::runtime_error("failed to write frame diagnostics");
            }
        };

        const auto flush_pending_frames = [&] {
            if (pending_frames.empty()) {
                return;
            }

            std::vector<DecodedFecFrame> decoded(pending_frames.size());
            std::vector<std::pair<std::size_t, std::size_t>> jobs;
            for (std::size_t frame = 0; frame < pending_frames.size(); ++frame) {
                if (!pending_frames[frame].selected) {
                    continue;
                }
                auto& decoded_frame = decoded[frame];
                decoded_frame.full_llr.resize(
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
                decoded_frame.retry_llr_clip.assign(codewords_per_frame, 0.0);
                decoded_frame.retry_weak_erase_fraction.assign(
                    codewords_per_frame,
                    0.0);
                decoded_frame.retry_attenuation.assign(codewords_per_frame, 0.0);
                decoded_frame.retry_bch_corrected_errors.assign(
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

            const auto effective_workers = std::min<std::size_t>(worker_count, jobs.size());
            std::vector<std::thread> workers;
            workers.reserve(effective_workers);
            for (std::size_t worker = 0; worker < effective_workers; ++worker) {
                workers.emplace_back([&, worker] {
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
                        if (early_syndrome_reject_ratio >= 0.0F) {
                            for (std::size_t bit = 0; bit < profile.full_codeword_bits(); ++bit) {
                                decoded_span[bit] = llr_span[bit] < 0.0F ? 1U : 0U;
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
                                        llr_span,
                                        graph,
                                        decoded_span,
                                        decode_options);
                            }
                        } else {
                            decoded_frame.results[codeword] =
                                dtmb::core::ldpc_decode_min_sum_sparse(
                                    llr_span,
                                    graph,
                                    decoded_span,
                                    decode_options);
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
                                candidate_transport);
                        };
                        auto baseline_bch_clean = false;
                        if (decoded_frame.results[codeword].converged) {
                            baseline_bch_clean =
                                bch_score(decoded_span).unclean_blocks == 0;
                        }
                        if (retry_enabled && !baseline_bch_clean) {
                            std::vector<float> retry_llr(profile.full_codeword_bits());
                            std::vector<std::uint8_t> retry_bits(
                                profile.full_codeword_bits());
                            std::vector<std::uint8_t> best_bits;
                            dtmb::core::LdpcDecodeResult best_result;
                            std::size_t best_bch_corrected =
                                std::numeric_limits<std::size_t>::max();
                            float best_clip = 0.0F;
                            float best_erase = 0.0F;
                            float best_attenuation = 0.0F;
                            bool best_layered = false;
                            for (const auto clip : retry_clip_axis) {
                                for (const auto erase_fraction : retry_erase_axis) {
                                    for (const auto attenuation :
                                         retry_attenuation_axis) {
                                        const auto algorithm_count =
                                            retry_layered ? 2U : 1U;
                                        for (std::size_t algorithm = 0;
                                             algorithm < algorithm_count;
                                             ++algorithm) {
                                                const auto layered = algorithm == 1U;
                                                if (!layered
                                                    && clip == 0.0F
                                                    && erase_fraction == 0.0F
                                                    && attenuation
                                                        == decode_options.attenuation) {
                                                    continue;
                                                }
                                                ++decoded_frame.retry_attempts[codeword];
                                                condition_retry_llr(
                                                    llr_span,
                                                    retry_llr,
                                                    profile.erased_parity_bits,
                                                    clip,
                                                    erase_fraction);
                                                auto retry_options = decode_options;
                                                retry_options.attenuation = attenuation;
                                                const auto retry_result = layered
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
                                                if (!retry_result.converged) {
                                                    continue;
                                                }
                                                const auto retry_bch =
                                                    bch_score(retry_bits);
                                                if (retry_bch.unclean_blocks != 0) {
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
                                                best_clip = clip;
                                                best_erase = erase_fraction;
                                                best_attenuation = attenuation;
                                                best_layered = layered;
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
                                decoded_frame.retry_llr_clip[codeword] = best_clip;
                                decoded_frame.retry_weak_erase_fraction[codeword] =
                                    best_erase;
                                decoded_frame.retry_attenuation[codeword] =
                                    best_attenuation;
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

            for (std::size_t frame = 0; frame < pending_frames.size(); ++frame) {
                if (!pending_frames[frame].selected) {
                    write_frame_diagnostics(pending_frames[frame], nullptr, false, 0, 0);
                    omitted_codeword_count += codewords_per_frame;
                    omit_frame();
                    continue;
                }
                auto& decoded_frame = decoded[frame];
                decoded_frame.bch = dtmb::core::dtmb_bch_descramble_message_bits(
                    decoded_frame.message_bits,
                    decoded_frame.transport_bytes);
                const auto ldpc_clean = std::all_of(
                    decoded_frame.results.begin(),
                    decoded_frame.results.end(),
                    [](const auto& result) { return result.converged; });
                decoded_frame.frame_clean =
                    ldpc_clean && decoded_frame.bch.unclean_blocks == 0;
                const auto bch_blocks_per_codeword =
                    decoded_frame.bch.block_count / codewords_per_frame;
                if (bch_blocks_per_codeword * codewords_per_frame
                    != decoded_frame.bch.block_count) {
                    throw std::runtime_error(
                        "BCH block count does not align to FEC codewords");
                }
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
                }
                clean_frame_count += decoded_frame.frame_clean ? 1U : 0U;
                for (const auto value : decoded_frame.clean_codewords) {
                    clean_codeword_count += value != 0U ? 1U : 0U;
                }
                bool emitted = false;
                std::size_t frame_discontinuity_marked_packets = 0;
                std::size_t frame_discontinuity_inserted_packets = 0;
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
                            && ts_packet_aligned(codeword_bytes);
                        if (decoded_frame.clean_codewords[codeword] == 0U
                            && !bch_emit_codeword
                            && !force_emit_codeword) {
                            ++omitted_codeword_count;
                            note_transport_gap();
                            continue;
                        }
                        const auto [marked, inserted] =
                            signal_pending_pid_discontinuities(codeword_bytes);
                        frame_discontinuity_marked_packets += marked;
                        discontinuity_marked_packets += marked;
                        frame_discontinuity_inserted_packets += inserted;
                        discontinuity_inserted_packets += inserted;
                        write_all(output, codeword_bytes);
                        output_byte_count += codeword_transport_bytes;
                        ++emitted_codeword_count;
                        note_emitted_transport(codeword_bytes);
                        emitted = true;
                    }
                    emitted_frame_count += emitted ? 1U : 0U;
                    partial_emitted_frame_count +=
                        (emitted && !decoded_frame.frame_clean) ? 1U : 0U;
                    if (!emitted) {
                        omit_frame();
                    }
                } else if (!clean_frames_only || decoded_frame.frame_clean) {
                    const auto [marked, inserted] =
                        signal_pending_pid_discontinuities(decoded_frame.transport_bytes);
                    frame_discontinuity_marked_packets += marked;
                    discontinuity_marked_packets += marked;
                    frame_discontinuity_inserted_packets += inserted;
                    discontinuity_inserted_packets += inserted;
                    write_all(output, decoded_frame.transport_bytes);
                    output_byte_count += decoded_frame.transport_bytes.size();
                    ++emitted_frame_count;
                    emitted_codeword_count += codewords_per_frame;
                    note_emitted_transport(decoded_frame.transport_bytes);
                    emitted = true;
                } else {
                    omitted_codeword_count += codewords_per_frame;
                    omit_frame();
                }
                write_frame_diagnostics(
                    pending_frames[frame],
                    &decoded_frame,
                    emitted,
                    frame_discontinuity_marked_packets,
                    frame_discontinuity_inserted_packets);

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
                for (const auto selected : decoded_frame.retry_selected) {
                    retry_recovered_codeword_count += selected != 0U ? 1U : 0U;
                }
                for (const auto layered : decoded_frame.retry_layered) {
                    layered_retry_recovered_codeword_count +=
                        layered != 0U ? 1U : 0U;
                }
                for (const auto& result : decoded_frame.results) {
                    converged_count += result.converged ? 1U : 0U;
                    iteration_count += result.iterations;
                    final_syndrome_weight += result.syndrome_weight;
                }
                ++soft_decoded_frames;
            }
            pending_frames.clear();
        };

        const auto push_selected_frame = [&](
            std::size_t frame_index,
            std::span<const float> frame_llr,
            std::size_t clean_syndrome_weight,
            bool force_selected,
            bool hard_h_gap_filled) {
            PendingFecFrame frame;
            frame.selected = true;
            frame.force_selected = force_selected;
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
            bool hard_h_gap_filled) {
            PendingFecFrame frame;
            frame.frame_index = frame_index;
            frame.clean_syndrome_weight = clean_syndrome_weight;
            frame.hard_h_gap_filled = hard_h_gap_filled;
            pending_frames.push_back(std::move(frame));
            if (pending_frames.size() >= decode_batch_frames) {
                flush_pending_frames();
            }
        };

        const auto consume_buffered_frame = [&](BufferedFecFrame&& frame) {
            if (frame.selected) {
                ++hard_h_gate_selected_frames;
                forced_selected_frames += frame.force_selected ? 1U : 0U;
                push_selected_frame(
                    frame.frame_index,
                    frame.transmitted_llr,
                    frame.clean_syndrome_weight,
                    frame.force_selected,
                    frame.hard_h_gap_filled);
            } else {
                ++hard_h_gate_skipped_frames;
                push_omitted_frame(
                    frame.frame_index,
                    frame.clean_syndrome_weight,
                    frame.hard_h_gap_filled);
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

                std::size_t gap_frames = 0;
                bool bounded_by_selected_frame = false;
                for (const auto& frame : hard_h_gate_output_frames) {
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
                        if (!frame.selected) {
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
            if (!hard_h_gate_enabled) {
                const auto force_selected =
                    frame_in_ranges(frame_count, std::span<const FrameRange>(force_frame_ranges));
                forced_selected_frames += force_selected ? 1U : 0U;
                push_selected_frame(frame_count, transmitted_llr, 0U, force_selected, false);
                continue;
            }

            BufferedFecFrame frame;
            frame.frame_index = frame_count;
            frame.force_selected =
                frame_in_ranges(frame_count, std::span<const FrameRange>(force_frame_ranges));
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
                    buffered.selected = buffered.selected || buffered.force_selected;
                }
                auto oldest = std::move(hard_h_gate_frames.front());
                hard_h_gate_frames.pop_front();
                queue_gate_output_frame(std::move(oldest));
            }
        }
        for (auto& buffered : hard_h_gate_frames) {
            buffered.selected = buffered.selected || buffered.force_selected;
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
                  << "codewords=" << codeword_count << '\n'
                  << "worker_count=" << worker_count << '\n'
                  << "decode_batch_frames=" << decode_batch_frames << '\n'
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
                  << "emit_clean_codewords=" << (emit_clean_codewords ? "true" : "false") << '\n'
                  << "emit_bch_clean_codewords="
                  << (emit_bch_clean_codewords ? "true" : "false") << '\n'
                  << "clean_codewords=" << clean_codeword_count << '\n'
                  << "bch_clean_payload_codewords="
                  << bch_clean_payload_codeword_count << '\n'
                  << "emitted_codewords=" << emitted_codeword_count << '\n'
                  << "omitted_codewords=" << omitted_codeword_count << '\n'
                  << "mark_discontinuities=" << (mark_discontinuities ? "true" : "false") << '\n'
                  << "discontinuity_marked_packets=" << discontinuity_marked_packets << '\n'
                  << "insert_discontinuity_packets="
                  << (insert_discontinuity_packets ? "true" : "false") << '\n'
                  << "discontinuity_inserted_packets=" << discontinuity_inserted_packets << '\n'
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
                  << "hard_h_gate_gap_filled_frames="
                  << hard_h_gate_gap_filled_frames << '\n'
                  << "codeword_llr_scale_rule_count="
                  << codeword_llr_scale_rules.size() << '\n'
                  << "variable_llr_scale_rule_count="
                  << variable_llr_scale_rules.size() << '\n'
                  << "variable_mod_llr_scale_rule_count="
                  << variable_mod_llr_scale_rules.size() << '\n'
                  << "retry_enabled=" << (retry_enabled ? "true" : "false") << '\n'
                  << "retry_llr_clip_count=" << retry_llr_clips.size() << '\n'
                  << "retry_weak_erase_fraction_count="
                  << retry_weak_erase_fractions.size() << '\n'
                  << "retry_attenuation_count=" << retry_attenuations.size() << '\n'
                  << "retry_layered=" << (retry_layered ? "true" : "false") << '\n'
                  << "retry_attempts=" << retry_attempt_count << '\n'
                  << "retry_recovered_codewords="
                  << retry_recovered_codeword_count << '\n'
                  << "layered_retry_recovered_codewords="
                  << layered_retry_recovered_codeword_count << '\n'
                  << "hard_h_gate_skipped_frames=" << hard_h_gate_skipped_frames << '\n'
                  << "soft_decoded_frames=" << soft_decoded_frames << '\n'
                  << "output_bytes=" << output_byte_count << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_ldpc_bch_decode: " << exc.what() << '\n';
        return 1;
    }
}
