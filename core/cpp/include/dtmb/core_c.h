#ifndef DTMB_CORE_C_H
#define DTMB_CORE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DTMB_CORE_STATUS_OK 0
#define DTMB_CORE_STATUS_INVALID_ARGUMENT -1
#define DTMB_CORE_STATUS_INVALID_LENGTH -2
#define DTMB_CORE_STATUS_OUT_OF_MEMORY -3
#define DTMB_CORE_STATUS_INTERNAL_ERROR -4

typedef struct DtmbCoreVersion {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t abi_major;
  uint32_t abi_minor;
} DtmbCoreVersion;

typedef struct DtmbCoreCi8PowerStats {
  size_t sample_count;
  float mean_i2q2;
  float rms_iq;
  size_t clip_count_i;
  size_t clip_count_q;
  size_t worker_count;
} DtmbCoreCi8PowerStats;

uint32_t dtmb_core_version_major(void);
uint32_t dtmb_core_version_minor(void);
uint32_t dtmb_core_version_patch(void);
uint32_t dtmb_core_abi_version_major(void);
uint32_t dtmb_core_abi_version_minor(void);
DtmbCoreVersion dtmb_core_version(void);
const char *dtmb_core_build_info(void);

int32_t dtmb_core_ci8_power_stats(
    const int8_t *interleaved_iq,
    size_t len_bytes,
    DtmbCoreCi8PowerStats *out_stats);

int32_t dtmb_core_ci8_power_stats_parallel(
    const int8_t *interleaved_iq,
    size_t len_bytes,
    size_t requested_workers,
    size_t min_parallel_samples,
    DtmbCoreCi8PowerStats *out_stats);

int32_t dtmb_core_qam64_soft_demodulate_cf32(
    const float *interleaved_symbols,
    size_t symbol_count,
    float noise_variance,
    float *output_llr);

int32_t dtmb_core_qam64_soft_demodulate_cf32_parallel(
    const float *interleaved_symbols,
    size_t symbol_count,
    float noise_variance,
    size_t requested_workers,
    size_t min_parallel_symbols,
    float *output_llr);

#ifdef __cplusplus
}
#endif

#endif
