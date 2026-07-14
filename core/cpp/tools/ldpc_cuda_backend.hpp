#pragma once

#include "dtmb/core.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace dtmb::tools::ldpc_cuda {

struct BatchDecodeStats {
    std::size_t codewords = 0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

struct BatchDecodeResult {
    std::vector<dtmb::core::LdpcDecodeResult> results;
    std::vector<std::size_t> initial_syndrome_weights;
    std::vector<std::uint8_t> early_rejected;
    BatchDecodeStats stats;
};

[[nodiscard]] bool backend_compiled() noexcept;

[[nodiscard]] BatchDecodeResult decode_min_sum_batch(
    std::span<const float> llr,
    std::size_t codeword_count,
    const dtmb::core::LdpcSparseGraph& graph,
    std::span<std::uint8_t> output_bits,
    dtmb::core::LdpcDecodeOptions options,
    float early_syndrome_reject_ratio = -1.0F);

}  // namespace dtmb::tools::ldpc_cuda
