#pragma once

#include "dtmb/core.hpp"

namespace dtmb::core {

inline constexpr std::size_t kC1Pn595ContextSymbols = 4970;

struct C1AcquisitionResult {
    bool locked = false;
    std::size_t observations = 0;
    std::size_t system_info_index = 0;
    float metric = 0.0F;
    float margin = 0.0F;
    float data_fourfold_metric = 0.0F;
    float frequency_shift_hz = 0.0F;
    float pn_fit_error = 0.0F;
    int cfo_alias = 0;
    int body_offset = 0;
};

std::vector<float> pn595_linear_channel_cf32(std::span<const float> header);
std::vector<float> c1_system_info_reference_cf32(std::size_t index);
float c1_equalize_pn595_cf32(std::span<const float> frame_with_next_header,
                            std::span<float> equalized);
C1AcquisitionResult acquire_c1_pn595_cf32(
    std::span<const float> samples, std::size_t phase_offset,
    float frequency_shift_hz, std::size_t observations = 32,
    float min_metric = 0.75F, float min_margin = 0.03F);

}
