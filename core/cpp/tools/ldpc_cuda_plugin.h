#pragma once
#include <stdint.h>

#define DTMB_CUDA_ABI_VERSION 1U
#ifdef _WIN32
#define DTMB_CUDA_EXPORT __declspec(dllexport)
#else
#define DTMB_CUDA_EXPORT __attribute__((visibility("default")))
#endif

struct DtmbCudaCodewordResult {
    uint64_t iterations;
    uint64_t syndrome_weight;
    uint64_t initial_syndrome_weight;
    uint32_t converged;
    uint32_t early_rejected;
};

struct DtmbCudaBatchRequest {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t codeword_count;
    uint64_t variable_count;
    uint64_t check_count;
    uint64_t edge_count;
    uint64_t max_iterations;
    float attenuation;
    float early_syndrome_reject_ratio;
    const float* llr;
    const uint64_t* check_offsets;
    const uint64_t* edge_variables;
    uint8_t* output_bits;
    struct DtmbCudaCodewordResult* results;
};

struct DtmbCudaBatchStats {
    uint64_t codewords;
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    double total_ms;
};

typedef uint32_t (*DtmbCudaAbiFunction)(void);
typedef int (*DtmbCudaProbeFunction)(char* error, uint64_t error_capacity);
typedef int (*DtmbCudaDecodeFunction)(const struct DtmbCudaBatchRequest* request,
    struct DtmbCudaBatchStats* stats, char* error, uint64_t error_capacity);
