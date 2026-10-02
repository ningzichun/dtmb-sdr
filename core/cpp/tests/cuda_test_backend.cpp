#include "ldpc_cuda_plugin.h"

extern "C" DTMB_CUDA_EXPORT uint32_t dtmb_cuda_abi_version() {
#ifdef DTMB_TEST_BAD_ABI
    return DTMB_CUDA_ABI_VERSION + 1;
#else
    return DTMB_CUDA_ABI_VERSION;
#endif
}

extern "C" DTMB_CUDA_EXPORT int dtmb_cuda_probe(char*, uint64_t) {
    return 0;
}

extern "C" DTMB_CUDA_EXPORT int dtmb_cuda_decode(const DtmbCudaBatchRequest* request,
    DtmbCudaBatchStats* stats, char*, uint64_t) {
    if (request->abi_version != DTMB_CUDA_ABI_VERSION || request->struct_size != sizeof(*request)
        || request->variable_count != 3 || request->check_count != 1 || request->edge_count != 3
        || request->check_offsets[1] != 3 || request->edge_variables[2] != 2
        || request->max_iterations != 50 || request->attenuation != 0.75F) return 1;
    for (uint64_t index = 0; index < request->codeword_count * request->variable_count; ++index) {
        request->output_bits[index] = request->llr[index] < 0 ? 1 : 0;
    }
    for (uint64_t codeword = 0; codeword < request->codeword_count; ++codeword) {
        request->results[codeword] = {7, 0, 2, 1, 0};
    }
    *stats = {request->codeword_count, 1, 2, 3, 6};
    return 0;
}
