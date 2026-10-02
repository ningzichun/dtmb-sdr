#ifdef NDEBUG
#undef NDEBUG
#endif
#include "dtmb/c1.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <random>

using namespace dtmb::core;

namespace {

std::vector<float> frame_samples(bool valid_info = true) {
    std::vector<float> result(kC1Pn595ContextSymbols * 2);
    pn_header_symbols_cf32(PnMode::pn595, std::span<float>(result).first(1190));
    pn_header_symbols_cf32(PnMode::pn595, std::span<float>(result).subspan(8750, 1190));
    std::mt19937 generator(595);
    for (std::size_t index = 595; index < 4375; ++index) {
        result[index * 2] = generator() % 2 ? 1.0F : -1.0F;
        result[index * 2 + 1] = generator() % 2 ? 1.0F : -1.0F;
    }
    if (valid_info) {
        const auto reference = c1_system_info_reference_cf32(10);
        std::copy(reference.begin(), reference.end(), result.begin() + 1190);
    }
    return result;
}

void check_equalization() {
    auto transmitted = frame_samples();
    std::vector<float> observed(transmitted.size()), equalized(transmitted.size());
    for (std::size_t index = 0; index < observed.size() / 2; ++index) {
        auto value = std::complex<float>{transmitted[index * 2], transmitted[index * 2 + 1]}
            * std::complex<float>{0.8F, 0.3F};
        if (index >= 13) {
            value += std::complex<float>{transmitted[(index - 13) * 2], transmitted[(index - 13) * 2 + 1]}
                * std::complex<float>{0.05F, -0.04F};
        }
        observed[index * 2] = value.real();
        observed[index * 2 + 1] = value.imag();
    }
    const auto error = c1_equalize_pn595_cf32(observed, equalized);
    assert(error < 1e-8F);
    for (std::size_t index = 595 * 2; index < 4375 * 2; ++index) {
        assert(std::abs(equalized[index] - transmitted[index]) < 0.04F);
    }
    const auto reference = c1_system_info_reference_cf32(10);
    assert(reference[0] == -1.0F && reference[1] == -1.0F);
    assert(reference[8] == 1.0F && reference[10] == -1.0F);
}

void check_acquisition() {
    const auto frame = frame_samples();
    constexpr std::size_t leading = 37;
    std::vector<float> input((leading + 33 * 4375 + 595) * 2);
    for (std::size_t index = leading; index < input.size() / 2; ++index) {
        const auto position = (index - leading) % 4375;
        const auto phase = 2.0 * std::numbers::pi * 228.0 * static_cast<double>(index) / 7560000.0;
        const auto value = std::complex<double>{frame[position * 2], frame[position * 2 + 1]}
            * std::polar(0.7, phase + 0.8);
        input[index * 2] = static_cast<float>(value.real());
        input[index * 2 + 1] = static_cast<float>(value.imag());
    }
    const auto acquired = acquire_c1_pn595_cf32(input, leading, -228.0F + 1728.0F);
    assert(acquired.locked && acquired.system_info_index == 10);
    assert(acquired.body_offset == 0 && acquired.cfo_alias == -1);
    assert(acquired.metric > 0.99F && acquired.margin > 0.1F);
    assert(acquired.data_fourfold_metric > 0.99F);
    assert(std::abs(acquired.frequency_shift_hz + 228.0F) < 0.01F);
    const auto noise_frame = frame_samples(false);
    for (std::size_t index = leading; index < input.size() / 2; ++index) {
        const auto position = (index - leading) % 4375;
        input[index * 2] = noise_frame[position * 2];
        input[index * 2 + 1] = noise_frame[position * 2 + 1];
    }
    assert(!acquire_c1_pn595_cf32(input, leading, 0).locked);
}

}

int main() {
    check_equalization();
    check_acquisition();
    std::cout << "C=1 PN-only equalization, standard SI and guarded CFO passed\n";
}
