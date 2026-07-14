#include "ldpc_cuda_backend.hpp"

#include <stdexcept>

namespace dtmb::tools::ldpc_cuda {

bool backend_compiled() noexcept {
    return false;
}

BatchDecodeResult decode_min_sum_batch(
    std::span<const float>,
    std::size_t,
    const dtmb::core::LdpcSparseGraph&,
    std::span<std::uint8_t>,
    dtmb::core::LdpcDecodeOptions,
    float) {
    throw std::runtime_error(
        "CUDA LDPC backend is not built; rebuild with "
        "-DDTMB_CORE_ENABLE_CUDA_LDPC=ON and a CUDA Toolkit visible to CMake");
}

}  // namespace dtmb::tools::ldpc_cuda
