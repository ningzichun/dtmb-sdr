#include "ldpc_cuda_backend.hpp"
#include "ldpc_cuda_plugin.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
void write_error(char* error, std::uint64_t capacity, const char* message) noexcept {
    if (error != nullptr && capacity != 0) {
        std::snprintf(error, static_cast<std::size_t>(capacity), "%s", message);
    }
}
}

extern "C" DTMB_CUDA_EXPORT std::uint32_t dtmb_cuda_abi_version() {
    return DTMB_CUDA_ABI_VERSION;
}

extern "C" DTMB_CUDA_EXPORT int dtmb_cuda_probe(char* error, std::uint64_t capacity) {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess) {
        write_error(error, capacity, cudaGetErrorString(status));
        return 1;
    }
    if (devices == 0) {
        write_error(error, capacity, "no compatible NVIDIA GPU found");
        return 1;
    }
    return 0;
}

extern "C" DTMB_CUDA_EXPORT int dtmb_cuda_decode(
    const DtmbCudaBatchRequest* request, DtmbCudaBatchStats* stats,
    char* error, std::uint64_t capacity) {
    try {
        if (request == nullptr || stats == nullptr
            || request->abi_version != DTMB_CUDA_ABI_VERSION
            || request->struct_size != sizeof(DtmbCudaBatchRequest)) {
            throw std::invalid_argument("incompatible CUDA backend request ABI");
        }
        constexpr auto index_limit = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
        if (request->variable_count == 0 || request->variable_count > index_limit
            || request->check_count == 0 || request->check_count > index_limit
            || request->edge_count > index_limit || request->codeword_count > index_limit
            || request->max_iterations > index_limit
            || request->codeword_count > std::numeric_limits<std::size_t>::max() / request->variable_count
            || request->llr == nullptr || request->check_offsets == nullptr
            || request->edge_variables == nullptr || request->output_bits == nullptr
            || request->results == nullptr) {
            throw std::invalid_argument("invalid CUDA batch dimensions or buffers");
        }
        auto graph = dtmb::core::LdpcSparseGraph{};
        graph.variable_count = static_cast<std::size_t>(request->variable_count);
        graph.check_offsets.assign(request->check_offsets, request->check_offsets + request->check_count + 1);
        graph.edge_variables.assign(request->edge_variables, request->edge_variables + request->edge_count);
        if (graph.check_offsets.front() != 0 || graph.check_offsets.back() != request->edge_count) {
            throw std::invalid_argument("invalid CUDA graph offsets");
        }
        for (std::size_t check = 0; check < graph.check_count(); ++check) {
            if (graph.check_offsets[check] > graph.check_offsets[check + 1]) {
                throw std::invalid_argument("nonmonotonic CUDA graph offsets");
            }
        }
        for (const auto variable : graph.edge_variables) {
            if (variable >= graph.variable_count) throw std::invalid_argument("invalid CUDA graph variable");
        }
        const auto count = static_cast<std::size_t>(request->codeword_count);
        const auto samples = count * graph.variable_count;
        const auto result = dtmb::tools::ldpc_cuda::decode_min_sum_batch(
            std::span<const float>(request->llr, samples), count, graph,
            std::span<std::uint8_t>(request->output_bits, samples),
            {static_cast<std::size_t>(request->max_iterations), request->attenuation},
            request->early_syndrome_reject_ratio);
        for (std::size_t codeword = 0; codeword < count; ++codeword) {
            request->results[codeword] = {
                result.results[codeword].iterations, result.results[codeword].syndrome_weight,
                result.initial_syndrome_weights[codeword],
                result.results[codeword].converged ? 1U : 0U, result.early_rejected[codeword],
            };
        }
        *stats = {result.stats.codewords, result.stats.h2d_ms, result.stats.kernel_ms,
                  result.stats.d2h_ms, result.stats.total_ms};
        return 0;
    } catch (const std::exception& failure) {
        write_error(error, capacity, failure.what());
    } catch (...) {
        write_error(error, capacity, "unknown CUDA backend failure");
    }
    return 1;
}
