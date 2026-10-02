#include "ldpc_cuda_backend.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
#ifdef _WIN32
    _putenv_s("DTMB_CUDA_BACKEND", argv[1]);
    _putenv_s("DTMB_CUDA_RUNTIME_DIR", "");
#else
    setenv("DTMB_CUDA_BACKEND", argv[1], 1);
    unsetenv("DTMB_CUDA_RUNTIME_DIR");
#endif
    try {
        dtmb::tools::ldpc_cuda::require_backend();
        if (std::string(argv[2]) != "batch") return 1;
        const auto graph = dtmb::core::LdpcSparseGraph{3, {0, 3}, {0, 1, 2}};
        const auto llr = std::array<float, 6>{2, -3, -5, 1, 2, -3};
        auto bits = std::array<std::uint8_t, 6>{};
        const auto result = dtmb::tools::ldpc_cuda::decode_min_sum_batch(llr, 2, graph, bits, {});
        if (bits != std::array<std::uint8_t, 6>{0, 1, 1, 0, 0, 1}
            || result.results.size() != 2 || result.results[1].iterations != 7
            || !result.results[0].converged || result.initial_syndrome_weights[0] != 2
            || result.early_rejected[1] != 0 || result.stats.codewords != 2
            || result.stats.kernel_ms != 2 || result.stats.total_ms != 6) return 1;
        return 0;
    } catch (const std::exception& error) {
        const auto expected = std::string(argv[2]) == "abi" ? "ABI version mismatch" : "library is missing";
        if (std::string(argv[2]) != "batch" && std::string(error.what()).find(expected) != std::string::npos) return 0;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
