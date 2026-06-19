#include "dtmb/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dtmb::core {
namespace {

constexpr std::size_t kBchMessageBits = 752;
constexpr std::size_t kBchCodeBits = 762;
constexpr std::size_t kBchParityBits = kBchCodeBits - kBchMessageBits;
constexpr std::array<std::uint8_t, 11> kBchGeneratorBits{
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
};
constexpr std::array<std::uint8_t, 15> kScramblerInitialState{
    1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
};

void validate_graph(const LdpcSparseGraph& graph) {
    if (graph.check_offsets.empty() || graph.check_offsets.front() != 0) {
        throw std::invalid_argument("LDPC graph check offsets are invalid");
    }
    if (graph.check_offsets.back() != graph.edge_variables.size()) {
        throw std::invalid_argument("LDPC graph edge count is invalid");
    }
    if (!std::is_sorted(graph.check_offsets.begin(), graph.check_offsets.end())) {
        throw std::invalid_argument("LDPC graph check offsets must be sorted");
    }
    for (const auto variable : graph.edge_variables) {
        if (variable >= graph.variable_count) {
            throw std::invalid_argument("LDPC graph variable index is out of range");
        }
    }
}

std::size_t candidate_worker_count(std::size_t requested, std::size_t codewords) {
    if (codewords == 0) {
        return 0;
    }
    auto workers = requested;
    if (workers == 0) {
        workers = std::thread::hardware_concurrency();
    }
    if (workers == 0) {
        workers = 1;
    }
    return std::clamp<std::size_t>(workers, 1, codewords);
}

std::size_t ordered_variable(
    std::size_t variable,
    std::size_t codeword_bits,
    LdpcStreamBitOrder order) {
    switch (order) {
    case LdpcStreamBitOrder::identity:
        return variable;
    case LdpcStreamBitOrder::reverse_each_byte:
        return (variable / 8U) * 8U + (7U - (variable % 8U));
    case LdpcStreamBitOrder::reverse_each_codeword:
        return codeword_bits - 1U - variable;
    }
    throw std::invalid_argument("unknown LDPC stream bit order");
}

std::uint16_t bch_syndrome_key(std::span<const std::uint8_t> bits) {
    if (bits.size() != kBchCodeBits) {
        throw std::invalid_argument("BCH codeword requires 762 bits");
    }
    std::array<std::uint8_t, kBchCodeBits> work{};
    std::copy(bits.begin(), bits.end(), work.begin());
    for (std::size_t index = 0; index < kBchCodeBits - kBchParityBits; ++index) {
        if (work[index] == 0) {
            continue;
        }
        for (std::size_t bit = 0; bit < kBchGeneratorBits.size(); ++bit) {
            work[index + bit] ^= kBchGeneratorBits[bit];
        }
    }

    std::uint16_t key = 0;
    for (std::size_t index = kBchCodeBits - kBchParityBits; index < kBchCodeBits; ++index) {
        key = static_cast<std::uint16_t>((key << 1U) | work[index]);
    }
    return key;
}

const std::array<int, 1U << kBchParityBits>& bch_single_error_table() {
    static const auto table = [] {
        std::array<int, 1U << kBchParityBits> result{};
        result.fill(-1);
        std::array<std::uint8_t, kBchCodeBits> error{};
        for (std::size_t position = 0; position < kBchCodeBits; ++position) {
            error.fill(0);
            error[position] = 1;
            result[bch_syndrome_key(error)] = static_cast<int>(position);
        }
        return result;
    }();
    return table;
}

}  // namespace

std::size_t LdpcSparseGraph::check_count() const noexcept {
    return check_offsets.empty() ? 0 : check_offsets.size() - 1;
}

std::size_t LdpcSparseGraph::edge_count() const noexcept {
    return edge_variables.size();
}

LdpcSparseGraph make_ldpc_sparse_graph(
    const std::vector<std::vector<std::size_t>>& check_variables,
    std::size_t variable_count) {
    if (variable_count == 0) {
        throw std::invalid_argument("LDPC graph must have variables");
    }

    LdpcSparseGraph graph;
    graph.variable_count = variable_count;
    graph.check_offsets.reserve(check_variables.size() + 1);
    graph.check_offsets.push_back(0);
    for (const auto& variables : check_variables) {
        for (const auto variable : variables) {
            if (variable >= variable_count) {
                throw std::invalid_argument("LDPC graph variable index is out of range");
            }
            graph.edge_variables.push_back(variable);
        }
        graph.check_offsets.push_back(graph.edge_variables.size());
    }
    validate_graph(graph);
    return graph;
}

std::size_t ldpc_syndrome_weight(
    std::span<const std::uint8_t> bits,
    const LdpcSparseGraph& graph) {
    validate_graph(graph);
    if (bits.size() != graph.variable_count) {
        throw std::invalid_argument("LDPC bit count must match graph variable count");
    }

    std::size_t weight = 0;
    for (std::size_t check = 0; check < graph.check_count(); ++check) {
        std::uint8_t parity = 0;
        for (std::size_t edge = graph.check_offsets[check];
             edge < graph.check_offsets[check + 1];
             ++edge) {
            parity ^= bits[graph.edge_variables[edge]];
        }
        weight += parity;
    }
    return weight;
}

LdpcHardCandidateScore ldpc_score_hard_bit_candidate(
    std::span<const std::uint8_t> input_bits,
    const LdpcSparseGraph& clean_check_graph,
    LdpcHardCandidateScoreOptions options) {
    validate_graph(clean_check_graph);
    if (clean_check_graph.check_count() == 0) {
        throw std::invalid_argument("LDPC candidate scoring requires clean check rows");
    }
    const auto codeword_bits = clean_check_graph.variable_count;
    if (options.stream_bit_order == LdpcStreamBitOrder::reverse_each_byte
        && (codeword_bits % 8U) != 0) {
        throw std::invalid_argument(
            "reverse_each_byte requires an LDPC codeword divisible into whole bytes");
    }

    LdpcHardCandidateScore score;
    score.bit_offset = options.bit_offset;
    score.clean_rows = clean_check_graph.check_count();
    if (options.bit_offset >= input_bits.size()) {
        return score;
    }

    const auto available_bits = input_bits.size() - options.bit_offset;
    const auto available_codewords = available_bits / codeword_bits;
    score.codewords = options.max_codewords == 0
        ? available_codewords
        : std::min(available_codewords, options.max_codewords);
    score.scored_bits = score.codewords * codeword_bits;
    score.unused_bits = available_bits - score.scored_bits;
    score.worker_count = candidate_worker_count(options.requested_workers, score.codewords);
    if (score.codewords == 0) {
        return score;
    }

    const auto selected = input_bits.subspan(options.bit_offset, score.scored_bits);
    if (std::any_of(selected.begin(), selected.end(), [](std::uint8_t bit) {
            return bit > 1U;
        })) {
        throw std::invalid_argument("LDPC candidate input bits must be binary");
    }

    std::vector<std::size_t> ordered_edge_variables(clean_check_graph.edge_count());
    std::transform(
        clean_check_graph.edge_variables.begin(),
        clean_check_graph.edge_variables.end(),
        ordered_edge_variables.begin(),
        [&](std::size_t variable) {
            return ordered_variable(variable, codeword_bits, options.stream_bit_order);
        });

    score.syndrome_weights.assign(score.codewords, 0);
    const auto score_range = [&](std::size_t first, std::size_t last) {
        for (std::size_t codeword = first; codeword < last; ++codeword) {
            const auto base = options.bit_offset + codeword * codeword_bits;
            std::size_t weight = 0;
            for (std::size_t check = 0; check < clean_check_graph.check_count(); ++check) {
                std::uint8_t parity = 0;
                for (std::size_t edge = clean_check_graph.check_offsets[check];
                     edge < clean_check_graph.check_offsets[check + 1];
                     ++edge) {
                    parity ^= input_bits[base + ordered_edge_variables[edge]];
                }
                weight += parity;
            }
            score.syndrome_weights[codeword] = weight;
        }
    };

    if (score.worker_count == 1) {
        score_range(0, score.codewords);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(score.worker_count);
        for (std::size_t worker = 0; worker < score.worker_count; ++worker) {
            const auto first = (score.codewords * worker) / score.worker_count;
            const auto last = (score.codewords * (worker + 1)) / score.worker_count;
            threads.emplace_back(score_range, first, last);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    const auto minmax = std::minmax_element(
        score.syndrome_weights.begin(),
        score.syndrome_weights.end());
    const auto total = std::accumulate(
        score.syndrome_weights.begin(),
        score.syndrome_weights.end(),
        std::size_t{0});
    score.mean_syndrome_ratio = static_cast<double>(total)
        / static_cast<double>(score.codewords * score.clean_rows);
    score.min_syndrome_ratio = static_cast<double>(*minmax.first)
        / static_cast<double>(score.clean_rows);
    score.max_syndrome_ratio = static_cast<double>(*minmax.second)
        / static_cast<double>(score.clean_rows);
    score.zero_syndrome_codewords = static_cast<std::size_t>(std::count(
        score.syndrome_weights.begin(),
        score.syndrome_weights.end(),
        std::size_t{0}));
    return score;
}

LdpcDecodeResult ldpc_decode_min_sum_sparse(
    std::span<const float> llr,
    const LdpcSparseGraph& graph,
    std::span<std::uint8_t> output_bits,
    LdpcDecodeOptions options) {
    validate_graph(graph);
    if (llr.size() != graph.variable_count || output_bits.size() < graph.variable_count) {
        throw std::invalid_argument("LDPC input/output size must match graph variable count");
    }
    if (options.max_iterations == 0) {
        throw std::invalid_argument("LDPC max iterations must be positive");
    }
    if (options.attenuation <= 0.0F || !std::isfinite(options.attenuation)) {
        throw std::invalid_argument("LDPC attenuation must be positive and finite");
    }

    std::vector<float> check_to_var(graph.edge_count(), 0.0F);
    std::vector<float> var_to_check(graph.edge_count(), 0.0F);
    std::vector<float> posterior(graph.variable_count, 0.0F);
    for (std::size_t edge = 0; edge < graph.edge_count(); ++edge) {
        var_to_check[edge] = llr[graph.edge_variables[edge]];
    }
    for (std::size_t variable = 0; variable < graph.variable_count; ++variable) {
        output_bits[variable] = llr[variable] < 0.0F ? 1U : 0U;
    }

    auto syndrome_weight = ldpc_syndrome_weight(
        output_bits.first(graph.variable_count),
        graph);
    if (syndrome_weight == 0) {
        return LdpcDecodeResult{0, true, 0};
    }

    for (std::size_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
        for (std::size_t check = 0; check < graph.check_count(); ++check) {
            const auto first = graph.check_offsets[check];
            const auto last = graph.check_offsets[check + 1];
            if (first == last) {
                continue;
            }
            if (last - first == 1) {
                check_to_var[first] = 0.0F;
                continue;
            }

            float sign_product = 1.0F;
            float min1 = std::numeric_limits<float>::infinity();
            float min2 = std::numeric_limits<float>::infinity();
            std::size_t min_edge = first;
            for (std::size_t edge = first; edge < last; ++edge) {
                const auto message = var_to_check[edge];
                if (message < 0.0F) {
                    sign_product = -sign_product;
                }
                const auto magnitude = std::abs(message);
                if (magnitude < min1) {
                    min2 = min1;
                    min1 = magnitude;
                    min_edge = edge;
                } else if (magnitude < min2) {
                    min2 = magnitude;
                }
            }
            for (std::size_t edge = first; edge < last; ++edge) {
                const auto sign = var_to_check[edge] < 0.0F ? -1.0F : 1.0F;
                const auto magnitude = edge == min_edge ? min2 : min1;
                check_to_var[edge] = options.attenuation * sign_product * sign * magnitude;
            }
        }

        std::copy(llr.begin(), llr.end(), posterior.begin());
        for (std::size_t edge = 0; edge < graph.edge_count(); ++edge) {
            posterior[graph.edge_variables[edge]] += check_to_var[edge];
        }
        for (std::size_t edge = 0; edge < graph.edge_count(); ++edge) {
            var_to_check[edge] = posterior[graph.edge_variables[edge]] - check_to_var[edge];
        }
        for (std::size_t variable = 0; variable < graph.variable_count; ++variable) {
            output_bits[variable] = posterior[variable] < 0.0F ? 1U : 0U;
        }

        syndrome_weight = ldpc_syndrome_weight(
            output_bits.first(graph.variable_count),
            graph);
        if (syndrome_weight == 0) {
            return LdpcDecodeResult{iteration, true, 0};
        }
    }

    return LdpcDecodeResult{options.max_iterations, false, syndrome_weight};
}

LdpcDecodeResult ldpc_decode_layered_min_sum_sparse(
    std::span<const float> llr,
    const LdpcSparseGraph& graph,
    std::span<std::uint8_t> output_bits,
    LdpcDecodeOptions options) {
    validate_graph(graph);
    if (llr.size() != graph.variable_count || output_bits.size() < graph.variable_count) {
        throw std::invalid_argument("LDPC input/output size must match graph variable count");
    }
    if (options.max_iterations == 0) {
        throw std::invalid_argument("LDPC max iterations must be positive");
    }
    if (options.attenuation <= 0.0F || !std::isfinite(options.attenuation)) {
        throw std::invalid_argument("LDPC attenuation must be positive and finite");
    }

    std::vector<float> check_to_var(graph.edge_count(), 0.0F);
    std::vector<float> posterior(llr.begin(), llr.end());
    for (std::size_t variable = 0; variable < graph.variable_count; ++variable) {
        output_bits[variable] = posterior[variable] < 0.0F ? 1U : 0U;
    }

    auto syndrome_weight = ldpc_syndrome_weight(
        output_bits.first(graph.variable_count),
        graph);
    if (syndrome_weight == 0) {
        return LdpcDecodeResult{0, true, 0};
    }

    for (std::size_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
        for (std::size_t check = 0; check < graph.check_count(); ++check) {
            const auto first = graph.check_offsets[check];
            const auto last = graph.check_offsets[check + 1];
            if (first == last) {
                continue;
            }
            if (last - first == 1) {
                const auto variable = graph.edge_variables[first];
                posterior[variable] -= check_to_var[first];
                check_to_var[first] = 0.0F;
                continue;
            }

            float sign_product = 1.0F;
            float min1 = std::numeric_limits<float>::infinity();
            float min2 = std::numeric_limits<float>::infinity();
            std::size_t min_edge = first;
            for (std::size_t edge = first; edge < last; ++edge) {
                const auto variable = graph.edge_variables[edge];
                const auto message = posterior[variable] - check_to_var[edge];
                if (message < 0.0F) {
                    sign_product = -sign_product;
                }
                const auto magnitude = std::abs(message);
                if (magnitude < min1) {
                    min2 = min1;
                    min1 = magnitude;
                    min_edge = edge;
                } else if (magnitude < min2) {
                    min2 = magnitude;
                }
            }
            for (std::size_t edge = first; edge < last; ++edge) {
                const auto variable = graph.edge_variables[edge];
                const auto message = posterior[variable] - check_to_var[edge];
                const auto sign = message < 0.0F ? -1.0F : 1.0F;
                const auto magnitude = edge == min_edge ? min2 : min1;
                const auto updated =
                    options.attenuation * sign_product * sign * magnitude;
                posterior[variable] = message + updated;
                check_to_var[edge] = updated;
            }
        }

        for (std::size_t variable = 0; variable < graph.variable_count; ++variable) {
            output_bits[variable] = posterior[variable] < 0.0F ? 1U : 0U;
        }
        syndrome_weight = ldpc_syndrome_weight(
            output_bits.first(graph.variable_count),
            graph);
        if (syndrome_weight == 0) {
            return LdpcDecodeResult{iteration, true, 0};
        }
    }

    return LdpcDecodeResult{options.max_iterations, false, syndrome_weight};
}

DtmbBchDecodeStats dtmb_bch_descramble_message_bits(
    std::span<const std::uint8_t> ldpc_message_bits,
    std::span<std::uint8_t> output_bytes,
    bool correct) {
    if (ldpc_message_bits.empty() || (ldpc_message_bits.size() % kBchCodeBits) != 0) {
        throw std::invalid_argument("LDPC message bits must contain whole BCH blocks");
    }
    for (const auto bit : ldpc_message_bits) {
        if (bit > 1) {
            throw std::invalid_argument("BCH input bits must be binary");
        }
    }

    const auto block_count = ldpc_message_bits.size() / kBchCodeBits;
    const auto payload_bits = block_count * kBchMessageBits;
    const auto payload_bytes = payload_bits / 8;
    if (output_bytes.size() < payload_bytes) {
        throw std::invalid_argument("BCH output byte span is too small");
    }
    std::fill(output_bytes.begin(), output_bytes.begin() + payload_bytes, 0);

    DtmbBchDecodeStats stats;
    stats.block_count = block_count;
    stats.block_clean.assign(block_count, 1U);
    stats.block_corrected_errors.assign(block_count, 0U);
    auto scrambler = kScramblerInitialState;
    std::size_t output_bit = 0;
    std::array<std::uint8_t, kBchCodeBits> codeword{};
    for (std::size_t block = 0; block < block_count; ++block) {
        const auto input = ldpc_message_bits.subspan(block * kBchCodeBits, kBchCodeBits);
        std::copy(input.begin(), input.end(), codeword.begin());
        const auto syndrome = bch_syndrome_key(codeword);
        if (syndrome != 0) {
            const auto position = correct ? bch_single_error_table()[syndrome] : -1;
            if (position >= 0) {
                codeword[static_cast<std::size_t>(position)] ^= 1U;
                ++stats.corrected_errors;
                stats.block_corrected_errors[block] = 1U;
            } else {
                ++stats.unclean_blocks;
                stats.block_clean[block] = 0U;
            }
        }

        for (std::size_t bit = 0; bit < kBchMessageBits; ++bit) {
            const auto next_scrambler_bit = static_cast<std::uint8_t>(scrambler[13] ^ scrambler[14]);
            const auto decoded = static_cast<std::uint8_t>(codeword[bit] ^ next_scrambler_bit);
            if (decoded != 0) {
                output_bytes[output_bit / 8] |= static_cast<std::uint8_t>(1U << (7U - (output_bit % 8)));
            }
            for (std::size_t index = scrambler.size() - 1; index > 0; --index) {
                scrambler[index] = scrambler[index - 1];
            }
            scrambler[0] = next_scrambler_bit;
            ++output_bit;
        }
    }
    return stats;
}

}  // namespace dtmb::core
