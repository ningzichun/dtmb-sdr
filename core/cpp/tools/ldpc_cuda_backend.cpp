#include "ldpc_cuda_backend.hpp"
#include "ldpc_cuda_plugin.h"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace dtmb::tools::ldpc_cuda {
namespace {
#ifndef __EMSCRIPTEN__
std::filesystem::path executable_directory() {
#ifdef _WIN32
    auto buffer = std::vector<wchar_t>(32768);
    const auto length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length == buffer.size()) throw std::runtime_error("cannot locate native executable");
    return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
#elif defined(__APPLE__)
    std::uint32_t length = 0;
    _NSGetExecutablePath(nullptr, &length);
    auto buffer = std::vector<char>(length);
    if (_NSGetExecutablePath(buffer.data(), &length) != 0) throw std::runtime_error("cannot locate native executable");
    return std::filesystem::canonical(buffer.data()).parent_path();
#else
    return std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
}

std::filesystem::path backend_path() {
    if (const auto configured = std::getenv("DTMB_CUDA_BACKEND")) {
        const auto path = std::filesystem::path(configured);
        if (!path.is_absolute()) throw std::runtime_error("DTMB_CUDA_BACKEND must be an absolute library path");
        return path;
    }
#ifdef _WIN32
    return executable_directory() / "dtmb_cuda12.dll";
#elif defined(__APPLE__)
    return executable_directory() / "libdtmb_cuda12.dylib";
#else
    return executable_directory() / "libdtmb_cuda12.so";
#endif
}

class Backend {
public:
    Backend() {
        const auto path = backend_path();
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("CUDA 12 backend library is missing: " + path.string()
                + "; install a CUDA-enabled dtmb-sdr wheel or build with DTMB_CORE_ENABLE_CUDA_LDPC=ON");
        }
        if (const auto directory = std::getenv("DTMB_CUDA_RUNTIME_DIR"); directory != nullptr && *directory != '\0') {
            const auto runtime_directory = std::filesystem::path(directory);
            if (!runtime_directory.is_absolute()) throw std::runtime_error("DTMB_CUDA_RUNTIME_DIR must be absolute");
#ifdef _WIN32
            if (AddDllDirectory(runtime_directory.c_str()) == nullptr) {
                throw std::runtime_error("invalid DTMB_CUDA_RUNTIME_DIR");
            }
            const auto runtime = runtime_directory / "cudart64_12.dll";
            if (LoadLibraryExW(runtime.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) == nullptr) {
                throw std::runtime_error("cannot load CUDA 12 runtime; install dtmb-sdr[cuda]");
            }
#else
            const auto runtime = runtime_directory / "libcudart.so.12";
            if (dlopen(runtime.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) {
                throw std::runtime_error("cannot load CUDA 12 runtime: " + std::string(dlerror())
                    + "; install dtmb-sdr[cuda]");
            }
#endif
        }
#ifdef _WIN32
        handle_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (handle_ == nullptr) {
            throw std::runtime_error("cannot load CUDA 12 backend (Windows error " + std::to_string(GetLastError())
                + "); install dtmb-sdr[cuda] or set DTMB_CUDA_RUNTIME_DIR");
        }
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            throw std::runtime_error("cannot load CUDA 12 backend: " + std::string(dlerror())
                + "; install dtmb-sdr[cuda] or set DTMB_CUDA_RUNTIME_DIR");
        }
#endif
        const auto version = symbol<DtmbCudaAbiFunction>("dtmb_cuda_abi_version");
        if (version() != DTMB_CUDA_ABI_VERSION) throw std::runtime_error("CUDA backend ABI version mismatch");
        const auto probe = symbol<DtmbCudaProbeFunction>("dtmb_cuda_probe");
        decode = symbol<DtmbCudaDecodeFunction>("dtmb_cuda_decode");
        auto error = std::array<char, 1024>{};
        if (probe(error.data(), error.size()) != 0) {
            throw std::runtime_error("CUDA 12 device initialization failed: " + std::string(error.data())
                + "; a compatible NVIDIA GPU and driver are required");
        }
    }
    DtmbCudaDecodeFunction decode = nullptr;

private:
    template<typename Function>
    Function symbol(const char* name) {
#ifdef _WIN32
        const auto address = GetProcAddress(handle_, name);
#else
        const auto address = dlsym(handle_, name);
#endif
        if (address == nullptr) throw std::runtime_error(std::string("CUDA backend is missing ABI symbol ") + name);
        return reinterpret_cast<Function>(address);
    }
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
};

Backend& backend() {
    static Backend instance;
    return instance;
}
#endif
}

void require_backend() {
#ifdef __EMSCRIPTEN__
    throw std::runtime_error("CUDA acceleration is unavailable in WebAssembly");
#else
    (void)backend();
#endif
}

bool backend_compiled() noexcept {
    try { require_backend(); return true; } catch (...) { return false; }
}

BatchDecodeResult decode_min_sum_batch(std::span<const float> llr, std::size_t count,
    const dtmb::core::LdpcSparseGraph& graph, std::span<std::uint8_t> output_bits,
    dtmb::core::LdpcDecodeOptions options, float early_syndrome_reject_ratio) {
    if (count == 0) return {};
    if (graph.variable_count == 0 || count > std::numeric_limits<std::size_t>::max() / graph.variable_count
        || llr.size() != count * graph.variable_count || output_bits.size() < llr.size()) {
        throw std::invalid_argument("CUDA LDPC batch input/output size mismatch");
    }
    require_backend();
#ifndef __EMSCRIPTEN__
    const auto offsets = std::vector<std::uint64_t>(graph.check_offsets.begin(), graph.check_offsets.end());
    const auto variables = std::vector<std::uint64_t>(graph.edge_variables.begin(), graph.edge_variables.end());
    auto results = std::vector<DtmbCudaCodewordResult>(count);
    const auto request = DtmbCudaBatchRequest{DTMB_CUDA_ABI_VERSION, sizeof(DtmbCudaBatchRequest), count,
        graph.variable_count, graph.check_count(), graph.edge_count(), options.max_iterations,
        options.attenuation, early_syndrome_reject_ratio, llr.data(), offsets.data(), variables.data(),
        output_bits.data(), results.data()};
    auto stats = DtmbCudaBatchStats{};
    auto error = std::array<char, 1024>{};
    if (backend().decode(&request, &stats, error.data(), error.size()) != 0) {
        throw std::runtime_error("CUDA LDPC decode failed: " + std::string(error.data()));
    }
    if (stats.codewords != count) throw std::runtime_error("CUDA backend returned wrong codeword count");
    auto result = BatchDecodeResult{};
    for (const auto& codeword : results) {
        result.results.push_back({static_cast<std::size_t>(codeword.iterations), codeword.converged != 0,
            static_cast<std::size_t>(codeword.syndrome_weight)});
        result.initial_syndrome_weights.push_back(static_cast<std::size_t>(codeword.initial_syndrome_weight));
        result.early_rejected.push_back(codeword.early_rejected != 0 ? 1U : 0U);
    }
    result.stats = {count, stats.h2d_ms, stats.kernel_ms, stats.d2h_ms, stats.total_ms};
    return result;
#else
    return {};
#endif
}

}  // namespace dtmb::tools::ldpc_cuda
