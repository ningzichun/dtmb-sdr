#include "ldpc_cuda_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtmb::tools::ldpc_cuda {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) {
        reset(count);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() {
        release();
    }

    void reset(std::size_t count) {
        release();
        count_ = count;
        if (count_ != 0) {
            check_cuda(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
        }
    }

    void reserve(std::size_t count) {
        if (count > count_) {
            reset(count);
        }
    }

    [[nodiscard]] T* get() noexcept {
        return ptr_;
    }
    [[nodiscard]] const T* get() const noexcept {
        return ptr_;
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }

private:
    void release() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class CudaEvent {
public:
    CudaEvent() {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }

    [[nodiscard]] cudaEvent_t get() const noexcept {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

struct HostGraph {
    std::vector<int> check_offsets;
    std::vector<int> edge_variables;
    std::vector<int> variable_offsets;
    std::vector<int> variable_edges;

    bool operator==(const HostGraph&) const = default;
};

HostGraph build_host_graph(const dtmb::core::LdpcSparseGraph& graph) {
    if (graph.variable_count > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || graph.edge_count() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || graph.check_count() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("CUDA LDPC graph is too large for int indexing");
    }

    HostGraph host;
    host.check_offsets.reserve(graph.check_offsets.size());
    for (const auto value : graph.check_offsets) {
        host.check_offsets.push_back(static_cast<int>(value));
    }
    host.edge_variables.reserve(graph.edge_variables.size());
    for (const auto value : graph.edge_variables) {
        host.edge_variables.push_back(static_cast<int>(value));
    }

    std::vector<int> degrees(graph.variable_count, 0);
    for (const auto variable : graph.edge_variables) {
        ++degrees[variable];
    }
    host.variable_offsets.assign(graph.variable_count + 1, 0);
    for (std::size_t variable = 0; variable < graph.variable_count; ++variable) {
        host.variable_offsets[variable + 1] = host.variable_offsets[variable] + degrees[variable];
    }
    host.variable_edges.assign(graph.edge_count(), 0);
    std::vector<int> cursor = host.variable_offsets;
    for (std::size_t edge = 0; edge < graph.edge_variables.size(); ++edge) {
        const auto variable = graph.edge_variables[edge];
        host.variable_edges[cursor[variable]++] = static_cast<int>(edge);
    }
    return host;
}

__global__ void init_messages_kernel(
    const float* llr,
    const int* edge_variables,
    float* var_to_check,
    float* check_to_var,
    std::uint8_t* bits,
    int* active,
    int variable_count,
    int edge_count,
    int codeword_count) {
    const auto total_edges = edge_count * codeword_count;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_edges;
         index += blockDim.x * gridDim.x) {
        const int codeword = index / edge_count;
        const int edge = index - codeword * edge_count;
        const int variable = edge_variables[edge];
        var_to_check[index] = llr[codeword * variable_count + variable];
        check_to_var[index] = 0.0F;
    }

    const auto total_variables = variable_count * codeword_count;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_variables;
         index += blockDim.x * gridDim.x) {
        bits[index] = llr[index] < 0.0F ? 1U : 0U;
        if ((index % variable_count) == 0) {
            active[index / variable_count] = 1;
        }
    }
}

__global__ void reset_syndrome_kernel(int* syndrome, int codeword_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < codeword_count;
         index += blockDim.x * gridDim.x) {
        syndrome[index] = 0;
    }
}

__global__ void syndrome_kernel(
    const std::uint8_t* bits,
    const int* check_offsets,
    const int* edge_variables,
    const int* active,
    int* syndrome,
    int variable_count,
    int check_count,
    int edge_count,
    int codeword_count) {
    const auto total_checks = check_count * codeword_count;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_checks;
         index += blockDim.x * gridDim.x) {
        const int codeword = index / check_count;
        if (active[codeword] == 0) {
            continue;
        }
        const int check = index - codeword * check_count;
        const int first = check_offsets[check];
        const int last = check_offsets[check + 1];
        std::uint8_t parity = 0;
        for (int edge = first; edge < last; ++edge) {
            parity ^= bits[codeword * variable_count + edge_variables[edge]];
        }
        if (parity != 0) {
            atomicAdd(&syndrome[codeword], 1);
        }
    }
}

__global__ void finish_initial_kernel(
    const int* syndrome,
    int* active,
    int* iterations,
    int* converged,
    int* final_syndrome,
    int* initial_syndrome,
    int* early_rejected,
    float early_reject_ratio,
    int check_count,
    int codeword_count) {
    for (int codeword = blockIdx.x * blockDim.x + threadIdx.x;
         codeword < codeword_count;
         codeword += blockDim.x * gridDim.x) {
        const int initial = syndrome[codeword];
        initial_syndrome[codeword] = initial;
        early_rejected[codeword] = 0;
        if (initial == 0) {
            active[codeword] = 0;
            iterations[codeword] = 0;
            converged[codeword] = 1;
            final_syndrome[codeword] = 0;
        } else if (
            early_reject_ratio >= 0.0F
            && static_cast<float>(initial) / static_cast<float>(check_count)
                >= early_reject_ratio) {
            active[codeword] = 0;
            iterations[codeword] = 0;
            converged[codeword] = 0;
            final_syndrome[codeword] = initial;
            early_rejected[codeword] = 1;
        } else {
            final_syndrome[codeword] = initial;
        }
    }
}

__global__ void check_update_kernel(
    const float* var_to_check,
    float* check_to_var,
    const int* check_offsets,
    const int* active,
    float attenuation,
    int check_count,
    int edge_count,
    int codeword_count) {
    const auto total_checks = check_count * codeword_count;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_checks;
         index += blockDim.x * gridDim.x) {
        const int codeword = index / check_count;
        if (active[codeword] == 0) {
            continue;
        }
        const int check = index - codeword * check_count;
        const int first = check_offsets[check];
        const int last = check_offsets[check + 1];
        const int edge_base = codeword * edge_count;
        if (last - first <= 1) {
            if (last - first == 1) {
                check_to_var[edge_base + first] = 0.0F;
            }
            continue;
        }

        float sign_product = 1.0F;
        float min1 = FLT_MAX;
        float min2 = FLT_MAX;
        int min_edge = first;
        for (int edge = first; edge < last; ++edge) {
            const auto message = var_to_check[edge_base + edge];
            if (message < 0.0F) {
                sign_product = -sign_product;
            }
            const auto magnitude = fabsf(message);
            if (magnitude < min1) {
                min2 = min1;
                min1 = magnitude;
                min_edge = edge;
            } else if (magnitude < min2) {
                min2 = magnitude;
            }
        }
        for (int edge = first; edge < last; ++edge) {
            const auto sign = var_to_check[edge_base + edge] < 0.0F ? -1.0F : 1.0F;
            const auto magnitude = edge == min_edge ? min2 : min1;
            check_to_var[edge_base + edge] = attenuation * sign_product * sign * magnitude;
        }
    }
}

__global__ void variable_update_kernel(
    const float* llr,
    const float* check_to_var,
    float* var_to_check,
    std::uint8_t* bits,
    const int* variable_offsets,
    const int* variable_edges,
    const int* active,
    int variable_count,
    int edge_count,
    int codeword_count) {
    const auto total_variables = variable_count * codeword_count;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_variables;
         index += blockDim.x * gridDim.x) {
        const int codeword = index / variable_count;
        if (active[codeword] == 0) {
            continue;
        }
        const int variable = index - codeword * variable_count;
        const int edge_base = codeword * edge_count;
        float posterior = llr[index];
        for (int cursor = variable_offsets[variable];
             cursor < variable_offsets[variable + 1];
             ++cursor) {
            posterior += check_to_var[edge_base + variable_edges[cursor]];
        }
        bits[index] = posterior < 0.0F ? 1U : 0U;
        for (int cursor = variable_offsets[variable];
             cursor < variable_offsets[variable + 1];
             ++cursor) {
            const int edge = variable_edges[cursor];
            var_to_check[edge_base + edge] = posterior - check_to_var[edge_base + edge];
        }
    }
}

__global__ void finish_iteration_kernel(
    const int* syndrome,
    int* active,
    int* iterations,
    int* converged,
    int* final_syndrome,
    int iteration,
    int max_iterations,
    int codeword_count) {
    for (int codeword = blockIdx.x * blockDim.x + threadIdx.x;
         codeword < codeword_count;
         codeword += blockDim.x * gridDim.x) {
        if (active[codeword] == 0) {
            continue;
        }
        final_syndrome[codeword] = syndrome[codeword];
        if (syndrome[codeword] == 0) {
            active[codeword] = 0;
            iterations[codeword] = iteration;
            converged[codeword] = 1;
        } else if (iteration == max_iterations) {
            iterations[codeword] = max_iterations;
            converged[codeword] = 0;
        }
    }
}

void copy_to_device(DeviceBuffer<int>& device, const std::vector<int>& host, const char* what) {
    device.reserve(host.size());
    check_cuda(
        cudaMemcpy(device.get(), host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice),
        what);
}

struct DeviceWorkspace {
    HostGraph host_graph;
    bool graph_loaded = false;
    DeviceBuffer<int> check_offsets;
    DeviceBuffer<int> edge_variables;
    DeviceBuffer<int> variable_offsets;
    DeviceBuffer<int> variable_edges;
    DeviceBuffer<float> llr;
    DeviceBuffer<float> var_to_check;
    DeviceBuffer<float> check_to_var;
    DeviceBuffer<std::uint8_t> bits;
    DeviceBuffer<int> active;
    DeviceBuffer<int> iterations;
    DeviceBuffer<int> converged;
    DeviceBuffer<int> syndrome;
    DeviceBuffer<int> final_syndrome;
    DeviceBuffer<int> initial_syndrome;
    DeviceBuffer<int> early_rejected;
    CudaEvent h2d_start;
    CudaEvent h2d_stop;
    CudaEvent kernel_start;
    CudaEvent kernel_stop;
    CudaEvent d2h_start;
    CudaEvent d2h_stop;

    void prepare(
        const HostGraph& next_graph,
        std::size_t codeword_count,
        std::size_t variable_count,
        std::size_t edge_count) {
        if (!graph_loaded || host_graph != next_graph) {
            copy_to_device(check_offsets, next_graph.check_offsets, "copy check offsets");
            copy_to_device(edge_variables, next_graph.edge_variables, "copy edge variables");
            copy_to_device(variable_offsets, next_graph.variable_offsets, "copy variable offsets");
            copy_to_device(variable_edges, next_graph.variable_edges, "copy variable edges");
            host_graph = next_graph;
            graph_loaded = true;
        }

        llr.reserve(codeword_count * variable_count);
        var_to_check.reserve(codeword_count * edge_count);
        check_to_var.reserve(codeword_count * edge_count);
        bits.reserve(codeword_count * variable_count);
        active.reserve(codeword_count);
        iterations.reserve(codeword_count);
        converged.reserve(codeword_count);
        syndrome.reserve(codeword_count);
        final_syndrome.reserve(codeword_count);
        initial_syndrome.reserve(codeword_count);
        early_rejected.reserve(codeword_count);
    }
};

}  // namespace

bool backend_compiled() noexcept {
    return true;
}

BatchDecodeResult decode_min_sum_batch(
    std::span<const float> llr,
    std::size_t codeword_count,
    const dtmb::core::LdpcSparseGraph& graph,
    std::span<std::uint8_t> output_bits,
    dtmb::core::LdpcDecodeOptions options,
    float early_syndrome_reject_ratio) {
    if (codeword_count == 0) {
        return {};
    }
    if (llr.size() != codeword_count * graph.variable_count
        || output_bits.size() < codeword_count * graph.variable_count) {
        throw std::invalid_argument("CUDA LDPC batch input/output size mismatch");
    }
    if (options.max_iterations == 0) {
        throw std::invalid_argument("CUDA LDPC max iterations must be positive");
    }
    if (options.attenuation <= 0.0F || !std::isfinite(options.attenuation)) {
        throw std::invalid_argument("CUDA LDPC attenuation must be positive and finite");
    }
    if ((early_syndrome_reject_ratio < 0.0F
         && early_syndrome_reject_ratio != -1.0F)
        || early_syndrome_reject_ratio > 1.0F) {
        throw std::invalid_argument("CUDA LDPC early reject ratio must be off or within 0..1");
    }
    if (codeword_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("CUDA LDPC batch is too large");
    }

    static std::mutex workspace_mutex;
    static DeviceWorkspace workspace;
    const std::lock_guard workspace_lock(workspace_mutex);

    const auto started = std::chrono::steady_clock::now();
    const auto host_graph = build_host_graph(graph);
    const int codewords = static_cast<int>(codeword_count);
    const int variables = static_cast<int>(graph.variable_count);
    const int checks = static_cast<int>(graph.check_count());
    const int edges = static_cast<int>(graph.edge_count());
    const int max_iterations = static_cast<int>(options.max_iterations);
    constexpr int threads = 256;
    const auto blocks_for = [](std::size_t work) {
        return static_cast<int>(std::min<std::size_t>((work + threads - 1U) / threads, 65535U));
    };

    workspace.prepare(host_graph, codeword_count, graph.variable_count, graph.edge_count());
    auto& d_check_offsets = workspace.check_offsets;
    auto& d_edge_variables = workspace.edge_variables;
    auto& d_variable_offsets = workspace.variable_offsets;
    auto& d_variable_edges = workspace.variable_edges;
    auto& d_llr = workspace.llr;
    auto& d_var_to_check = workspace.var_to_check;
    auto& d_check_to_var = workspace.check_to_var;
    auto& d_bits = workspace.bits;
    auto& d_active = workspace.active;
    auto& d_iterations = workspace.iterations;
    auto& d_converged = workspace.converged;
    auto& d_syndrome = workspace.syndrome;
    auto& d_final_syndrome = workspace.final_syndrome;
    auto& d_initial_syndrome = workspace.initial_syndrome;
    auto& d_early_rejected = workspace.early_rejected;

    const auto h2d_start = workspace.h2d_start.get();
    const auto h2d_stop = workspace.h2d_stop.get();
    const auto kernel_start = workspace.kernel_start.get();
    const auto kernel_stop = workspace.kernel_stop.get();
    const auto d2h_start = workspace.d2h_start.get();
    const auto d2h_stop = workspace.d2h_stop.get();

    check_cuda(cudaEventRecord(h2d_start), "record h2d start");
    check_cuda(
        cudaMemcpy(d_llr.get(), llr.data(), llr.size() * sizeof(float), cudaMemcpyHostToDevice),
        "copy LLR batch to device");
    check_cuda(cudaMemset(d_iterations.get(), 0, codeword_count * sizeof(int)), "clear iterations");
    check_cuda(cudaMemset(d_converged.get(), 0, codeword_count * sizeof(int)), "clear converged");
    check_cuda(
        cudaMemset(d_final_syndrome.get(), 0, codeword_count * sizeof(int)),
        "clear final syndrome");
    check_cuda(cudaEventRecord(h2d_stop), "record h2d stop");

    check_cuda(cudaEventRecord(kernel_start), "record kernel start");
    init_messages_kernel<<<
        blocks_for(std::max<std::size_t>(llr.size(), codeword_count * graph.edge_count())),
        threads>>>(
        d_llr.get(),
        d_edge_variables.get(),
        d_var_to_check.get(),
        d_check_to_var.get(),
        d_bits.get(),
        d_active.get(),
        variables,
        edges,
        codewords);
    check_cuda(cudaGetLastError(), "launch init messages kernel");

    reset_syndrome_kernel<<<blocks_for(codeword_count), threads>>>(
        d_syndrome.get(),
        codewords);
    check_cuda(cudaGetLastError(), "launch reset syndrome kernel");
    syndrome_kernel<<<blocks_for(codeword_count * graph.check_count()), threads>>>(
        d_bits.get(),
        d_check_offsets.get(),
        d_edge_variables.get(),
        d_active.get(),
        d_syndrome.get(),
        variables,
        checks,
        edges,
        codewords);
    check_cuda(cudaGetLastError(), "launch initial syndrome kernel");
    finish_initial_kernel<<<blocks_for(codeword_count), threads>>>(
        d_syndrome.get(),
        d_active.get(),
        d_iterations.get(),
        d_converged.get(),
        d_final_syndrome.get(),
        d_initial_syndrome.get(),
        d_early_rejected.get(),
        early_syndrome_reject_ratio,
        checks,
        codewords);
    check_cuda(cudaGetLastError(), "launch initial finish kernel");

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        check_update_kernel<<<blocks_for(codeword_count * graph.check_count()), threads>>>(
            d_var_to_check.get(),
            d_check_to_var.get(),
            d_check_offsets.get(),
            d_active.get(),
            options.attenuation,
            checks,
            edges,
            codewords);
        check_cuda(cudaGetLastError(), "launch check update kernel");
        variable_update_kernel<<<blocks_for(codeword_count * graph.variable_count), threads>>>(
            d_llr.get(),
            d_check_to_var.get(),
            d_var_to_check.get(),
            d_bits.get(),
            d_variable_offsets.get(),
            d_variable_edges.get(),
            d_active.get(),
            variables,
            edges,
            codewords);
        check_cuda(cudaGetLastError(), "launch variable update kernel");
        reset_syndrome_kernel<<<blocks_for(codeword_count), threads>>>(
            d_syndrome.get(),
            codewords);
        check_cuda(cudaGetLastError(), "launch reset syndrome kernel");
        syndrome_kernel<<<blocks_for(codeword_count * graph.check_count()), threads>>>(
            d_bits.get(),
            d_check_offsets.get(),
            d_edge_variables.get(),
            d_active.get(),
            d_syndrome.get(),
            variables,
            checks,
            edges,
            codewords);
        check_cuda(cudaGetLastError(), "launch syndrome kernel");
        finish_iteration_kernel<<<blocks_for(codeword_count), threads>>>(
            d_syndrome.get(),
            d_active.get(),
            d_iterations.get(),
            d_converged.get(),
            d_final_syndrome.get(),
            iteration,
            max_iterations,
            codewords);
        check_cuda(cudaGetLastError(), "launch finish iteration kernel");
    }
    check_cuda(cudaEventRecord(kernel_stop), "record kernel stop");

    std::vector<int> iterations(codeword_count);
    std::vector<int> converged(codeword_count);
    std::vector<int> final_syndrome(codeword_count);
    std::vector<int> initial_syndrome(codeword_count);
    std::vector<int> early_rejected(codeword_count);
    check_cuda(cudaEventRecord(d2h_start), "record d2h start");
    check_cuda(
        cudaMemcpy(
            output_bits.data(),
            d_bits.get(),
            codeword_count * graph.variable_count * sizeof(std::uint8_t),
            cudaMemcpyDeviceToHost),
        "copy decoded bits from device");
    check_cuda(
        cudaMemcpy(iterations.data(), d_iterations.get(), iterations.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy iterations from device");
    check_cuda(
        cudaMemcpy(converged.data(), d_converged.get(), converged.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy convergence from device");
    check_cuda(
        cudaMemcpy(final_syndrome.data(), d_final_syndrome.get(), final_syndrome.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy final syndrome from device");
    check_cuda(
        cudaMemcpy(initial_syndrome.data(), d_initial_syndrome.get(), initial_syndrome.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy initial syndrome from device");
    check_cuda(
        cudaMemcpy(early_rejected.data(), d_early_rejected.get(), early_rejected.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy early rejected flags from device");
    check_cuda(cudaEventRecord(d2h_stop), "record d2h stop");
    check_cuda(cudaEventSynchronize(d2h_stop), "synchronize d2h");

    float h2d_ms = 0.0F;
    float kernel_ms = 0.0F;
    float d2h_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop), "measure h2d");
    check_cuda(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop), "measure kernel");
    check_cuda(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop), "measure d2h");
    BatchDecodeResult result;
    result.results.resize(codeword_count);
    result.initial_syndrome_weights.resize(codeword_count);
    result.early_rejected.resize(codeword_count);
    for (std::size_t codeword = 0; codeword < codeword_count; ++codeword) {
        result.results[codeword] = dtmb::core::LdpcDecodeResult{
            static_cast<std::size_t>(std::max(iterations[codeword], 0)),
            converged[codeword] != 0,
            static_cast<std::size_t>(std::max(final_syndrome[codeword], 0)),
        };
        result.initial_syndrome_weights[codeword] =
            static_cast<std::size_t>(std::max(initial_syndrome[codeword], 0));
        result.early_rejected[codeword] = early_rejected[codeword] != 0 ? 1U : 0U;
    }
    result.stats.codewords = codeword_count;
    result.stats.h2d_ms = h2d_ms;
    result.stats.kernel_ms = kernel_ms;
    result.stats.d2h_ms = d2h_ms;
    const auto stopped = std::chrono::steady_clock::now();
    result.stats.total_ms =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    return result;
}

}  // namespace dtmb::tools::ldpc_cuda
