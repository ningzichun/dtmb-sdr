#include "dtmb/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef DTMB_CORE_HAVE_FFTW3F
#include <fftw3.h>
#endif

namespace dtmb::core {
namespace {

using Complex = std::complex<float>;

#ifdef DTMB_CORE_HAVE_FFTW3F
class FftwPlanCache {
public:
    FftwPlanCache() = default;
    FftwPlanCache(const FftwPlanCache&) = delete;
    FftwPlanCache& operator=(const FftwPlanCache&) = delete;

    ~FftwPlanCache() {
        for (const auto& [size, plan] : plans_) {
            (void)size;
            fftwf_destroy_plan(plan);
        }
    }

    [[nodiscard]] fftwf_plan get(std::size_t size) {
        if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("FFT size exceeds FFTW int limit");
        }
        const std::lock_guard lock(mutex_);
        if (const auto found = plans_.find(size); found != plans_.end()) {
            return found->second;
        }
        auto* input = fftwf_alloc_complex(size);
        auto* output = fftwf_alloc_complex(size);
        if (input == nullptr || output == nullptr) {
            fftwf_free(input);
            fftwf_free(output);
            throw std::bad_alloc();
        }
        const auto plan = fftwf_plan_dft_1d(
            static_cast<int>(size),
            input,
            output,
            FFTW_FORWARD,
            FFTW_ESTIMATE | FFTW_UNALIGNED);
        fftwf_free(input);
        fftwf_free(output);
        if (plan == nullptr) {
            throw std::runtime_error("failed to create FFTW3f plan");
        }
        plans_.emplace(size, plan);
        return plan;
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::size_t, fftwf_plan> plans_;
};

[[nodiscard]] fftwf_plan fftw_plan(std::size_t size) {
    static FftwPlanCache cache;
    return cache.get(size);
}
#endif

constexpr std::array<std::size_t, kC3780SystemInfoSymbols> kSystemInfoPositions{
    0, 140, 279, 419, 420, 560, 699, 839, 840, 980, 1119, 1259,
    1260, 1400, 1539, 1679, 1680, 1820, 1959, 2099, 2100, 2240,
    2379, 2519, 2520, 2660, 2799, 2939, 2940, 3080, 3219, 3359,
    3360, 3500, 3639, 3779,
};

[[nodiscard]] std::array<std::size_t, kC3780FrameBodySymbols>
build_logical_to_physical() {
    // GB 20600-2006 Appendix F C=3780 frequency-interleaver permutation.
    std::array<std::size_t, kC3780FrameBodySymbols> logical_to_physical{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                for (std::size_t l = 0; l < 2; ++l) {
                    for (std::size_t m = 0; m < 2; ++m) {
                        for (std::size_t n = 0; n < 5; ++n) {
                            for (std::size_t o = 0; o < 7; ++o) {
                                const auto physical =
                                    o * 540 + n * 108 + m * 54 + l * 27
                                    + k * 9 + j * 3 + i;
                                const auto logical =
                                    i * 1260 + j * 420 + k * 140 + l * 70
                                    + m * 35 + n * 7 + o;
                                logical_to_physical[logical] = physical;
                            }
                        }
                    }
                }
            }
        }
    }
    return logical_to_physical;
}

[[nodiscard]] std::array<bool, kC3780FrameBodySymbols> build_system_info_mask() {
    std::array<bool, kC3780FrameBodySymbols> mask{};
    for (const auto position : kSystemInfoPositions) {
        mask[position] = true;
    }
    return mask;
}

[[nodiscard]] std::size_t fft_radix(std::size_t size) noexcept {
    for (const auto radix : std::array<std::size_t, 4>{2, 3, 5, 7}) {
        if ((size % radix) == 0) {
            return radix;
        }
    }
    return 0;
}

void direct_dft(
    const Complex* input,
    std::size_t stride,
    Complex* output,
    std::size_t size) {
    constexpr auto two_pi = 2.0F * std::numbers::pi_v<float>;
    for (std::size_t bin = 0; bin < size; ++bin) {
        const auto root = std::polar(
            1.0F,
            -two_pi * static_cast<float>(bin) / static_cast<float>(size));
        auto power = Complex{1.0F, 0.0F};
        auto sum = Complex{0.0F, 0.0F};
        for (std::size_t sample = 0; sample < size; ++sample) {
            sum += input[sample * stride] * power;
            power *= root;
        }
        output[bin] = sum;
    }
}

struct DirectDftTwiddles {
    std::size_t size = 0;
    std::vector<Complex> factors;
};

[[nodiscard]] DirectDftTwiddles build_direct_dft_twiddles(std::size_t size) {
    constexpr auto two_pi = 2.0F * std::numbers::pi_v<float>;
    DirectDftTwiddles result;
    result.size = size;
    result.factors.resize(size * size);
    for (std::size_t bin = 0; bin < size; ++bin) {
        const auto root = std::polar(
            1.0F,
            -two_pi * static_cast<float>(bin) / static_cast<float>(size));
        auto power = Complex{1.0F, 0.0F};
        for (std::size_t sample = 0; sample < size; ++sample) {
            result.factors[bin * size + sample] = power;
            power *= root;
        }
    }
    return result;
}

[[nodiscard]] const DirectDftTwiddles* cached_direct_dft_twiddles(
    std::size_t size) {
    if (size == 511) {
        static const auto twiddles = build_direct_dft_twiddles(511);
        return &twiddles;
    }
    return nullptr;
}

void direct_dft_cached(
    const Complex* input,
    std::size_t stride,
    Complex* output,
    const DirectDftTwiddles& twiddles) {
    for (std::size_t bin = 0; bin < twiddles.size; ++bin) {
        auto sum = Complex{0.0F, 0.0F};
        const auto* factors = twiddles.factors.data() + bin * twiddles.size;
        for (std::size_t sample = 0; sample < twiddles.size; ++sample) {
            sum += input[sample * stride] * factors[sample];
        }
        output[bin] = sum;
    }
}

struct FftCombineTwiddles {
    std::size_t size = 0;
    std::size_t radix = 0;
    std::size_t sub_size = 0;
    std::vector<Complex> factors;
};

[[nodiscard]] FftCombineTwiddles build_fft_combine_twiddles(std::size_t size) {
    const auto radix = fft_radix(size);
    const auto sub_size = size / radix;
    constexpr auto two_pi = 2.0F * std::numbers::pi_v<float>;
    FftCombineTwiddles result;
    result.size = size;
    result.radix = radix;
    result.sub_size = sub_size;
    result.factors.resize(sub_size * radix * radix);
    for (std::size_t sub_bin = 0; sub_bin < sub_size; ++sub_bin) {
        for (std::size_t outer_bin = 0; outer_bin < radix; ++outer_bin) {
            const auto bin = sub_bin + sub_size * outer_bin;
            const auto root = std::polar(
                1.0F,
                -two_pi * static_cast<float>(bin) / static_cast<float>(size));
            auto power = Complex{1.0F, 0.0F};
            const auto offset = (sub_bin * radix + outer_bin) * radix;
            for (std::size_t branch = 0; branch < radix; ++branch) {
                result.factors[offset + branch] = power;
                power *= root;
            }
        }
    }
    return result;
}

[[nodiscard]] const FftCombineTwiddles* cached_fft_combine_twiddles(
    std::size_t size) {
    switch (size) {
    case 3: {
        static const auto twiddles = build_fft_combine_twiddles(3);
        return &twiddles;
    }
    case 7: {
        static const auto twiddles = build_fft_combine_twiddles(7);
        return &twiddles;
    }
    case 35: {
        static const auto twiddles = build_fft_combine_twiddles(35);
        return &twiddles;
    }
    case 105: {
        static const auto twiddles = build_fft_combine_twiddles(105);
        return &twiddles;
    }
    case 315: {
        static const auto twiddles = build_fft_combine_twiddles(315);
        return &twiddles;
    }
    case 945: {
        static const auto twiddles = build_fft_combine_twiddles(945);
        return &twiddles;
    }
    case 1890: {
        static const auto twiddles = build_fft_combine_twiddles(1890);
        return &twiddles;
    }
    case 3780: {
        static const auto twiddles = build_fft_combine_twiddles(3780);
        return &twiddles;
    }
    default:
        return nullptr;
    }
}

void fft_recursive(
    const Complex* input,
    std::size_t stride,
    Complex* output,
    std::size_t size) {
    if (size == 1) {
        output[0] = input[0];
        return;
    }

    const auto radix = fft_radix(size);
    if (radix == 0) {
        if (const auto* twiddles = cached_direct_dft_twiddles(size)) {
            direct_dft_cached(input, stride, output, *twiddles);
            return;
        }
        direct_dft(input, stride, output, size);
        return;
    }
    const auto sub_size = size / radix;
    for (std::size_t branch = 0; branch < radix; ++branch) {
        fft_recursive(
            input + branch * stride,
            stride * radix,
            output + branch * sub_size,
            sub_size);
    }

    std::array<Complex, 7> branch_values{};
    std::array<Complex, 7> combined{};
    if (const auto* twiddles = cached_fft_combine_twiddles(size)) {
        for (std::size_t sub_bin = 0; sub_bin < sub_size; ++sub_bin) {
            for (std::size_t branch = 0; branch < radix; ++branch) {
                branch_values[branch] = output[branch * sub_size + sub_bin];
            }
            for (std::size_t outer_bin = 0; outer_bin < radix; ++outer_bin) {
                auto sum = Complex{0.0F, 0.0F};
                const auto* factors = twiddles->factors.data()
                    + (sub_bin * radix + outer_bin) * radix;
                for (std::size_t branch = 0; branch < radix; ++branch) {
                    sum += branch_values[branch] * factors[branch];
                }
                combined[outer_bin] = sum;
            }
            for (std::size_t outer_bin = 0; outer_bin < radix; ++outer_bin) {
                output[sub_bin + sub_size * outer_bin] = combined[outer_bin];
            }
        }
    } else {
        constexpr auto two_pi = 2.0F * std::numbers::pi_v<float>;
        for (std::size_t sub_bin = 0; sub_bin < sub_size; ++sub_bin) {
            for (std::size_t branch = 0; branch < radix; ++branch) {
                branch_values[branch] = output[branch * sub_size + sub_bin];
            }
            for (std::size_t outer_bin = 0; outer_bin < radix; ++outer_bin) {
                const auto bin = sub_bin + sub_size * outer_bin;
                const auto root = std::polar(
                    1.0F,
                    -two_pi * static_cast<float>(bin) / static_cast<float>(size));
                auto power = Complex{1.0F, 0.0F};
                auto sum = Complex{0.0F, 0.0F};
                for (std::size_t branch = 0; branch < radix; ++branch) {
                    sum += branch_values[branch] * power;
                    power *= root;
                }
                combined[outer_bin] = sum;
            }
            for (std::size_t outer_bin = 0; outer_bin < radix; ++outer_bin) {
                output[sub_bin + sub_size * outer_bin] = combined[outer_bin];
            }
        }
    }
}

[[nodiscard]] float nearest_qam64_level(float value) noexcept {
    // Midpoint comparisons preserve the previous tie behaviour (the lower
    // constellation level wins) without scanning all eight levels for every
    // I/Q component.  This function is on the per-frame normalization hot
    // path, so the fixed decision tree is materially cheaper and vectorizes
    // better than the generic nearest-neighbour loop.
    if (value <= -6.0F) {
        return -7.0F;
    }
    if (value <= -4.0F) {
        return -5.0F;
    }
    if (value <= -2.0F) {
        return -3.0F;
    }
    if (value <= 0.0F) {
        return -1.0F;
    }
    if (value <= 2.0F) {
        return 1.0F;
    }
    if (value <= 4.0F) {
        return 3.0F;
    }
    if (value <= 6.0F) {
        return 5.0F;
    }
    return 7.0F;
}

}  // namespace

void qam64_normalize_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols) {
    if ((interleaved_symbols.size() % 2) != 0) {
        throw std::invalid_argument("CF32 input must contain interleaved real/imag pairs");
    }
    if (output_symbols.size() < interleaved_symbols.size()) {
        throw std::invalid_argument("output CF32 span is too small");
    }
    const auto symbol_count = interleaved_symbols.size() / 2;
    if (symbol_count == 0) {
        return;
    }

    double power_sum = 0.0;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto value = Complex{
            interleaved_symbols[symbol * 2],
            interleaved_symbols[symbol * 2 + 1],
        };
        power_sum += std::norm(value);
    }
    const auto observed_power = power_sum / static_cast<double>(symbol_count);
    if (observed_power <= 1.0e-24) {
        std::copy(interleaved_symbols.begin(), interleaved_symbols.end(), output_symbols.begin());
        return;
    }

    constexpr double average_power = 42.0;
    const auto scale = static_cast<float>(std::sqrt(observed_power / average_power));
    auto numerator = Complex{0.0F, 0.0F};
    double denominator = 0.0;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto value = Complex{
            interleaved_symbols[symbol * 2],
            interleaved_symbols[symbol * 2 + 1],
        };
        const auto corrected = value / std::max(scale, 1.0e-12F);
        const auto nearest = Complex{
            nearest_qam64_level(corrected.real()),
            nearest_qam64_level(corrected.imag()),
        };
        numerator += std::conj(nearest) * value;
        denominator += std::norm(nearest);
    }

    auto gain = Complex{scale, 0.0F};
    if (denominator > 1.0e-12) {
        const auto candidate = numerator / static_cast<float>(denominator);
        if (std::abs(candidate) > 1.0e-12F) {
            gain = candidate;
        }
    }
    const auto gain_power = std::norm(gain);
    const auto inverse_gain = gain_power > 1.0e-24F
        ? std::conj(gain) / gain_power
        : Complex{1.0F, 0.0F};
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto normalized = Complex{
            interleaved_symbols[symbol * 2],
            interleaved_symbols[symbol * 2 + 1],
        } * inverse_gain;
        output_symbols[symbol * 2] = normalized.real();
        output_symbols[symbol * 2 + 1] = normalized.imag();
    }
}

void qam64_normalize_amplitude_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols) {
    if ((interleaved_symbols.size() % 2) != 0) {
        throw std::invalid_argument("CF32 input must contain interleaved real/imag pairs");
    }
    if (output_symbols.size() < interleaved_symbols.size()) {
        throw std::invalid_argument("output CF32 span is too small");
    }
    const auto symbol_count = interleaved_symbols.size() / 2;
    if (symbol_count == 0) {
        return;
    }

    double power_sum = 0.0;
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto value = Complex{
            interleaved_symbols[symbol * 2],
            interleaved_symbols[symbol * 2 + 1],
        };
        power_sum += std::norm(value);
    }
    const auto observed_power = power_sum / static_cast<double>(symbol_count);
    if (observed_power <= 1.0e-24) {
        std::copy(interleaved_symbols.begin(), interleaved_symbols.end(), output_symbols.begin());
        return;
    }

    constexpr double average_power = 42.0;
    const auto scale = static_cast<float>(std::sqrt(observed_power / average_power));
    const auto safe_scale = std::max(scale, 1.0e-12F);
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        output_symbols[symbol * 2] = interleaved_symbols[symbol * 2] / safe_scale;
        output_symbols[symbol * 2 + 1] = interleaved_symbols[symbol * 2 + 1] / safe_scale;
    }
}

void mixed_radix_fft_forward_cf32(
    std::span<const float> interleaved_time_samples,
    std::span<float> interleaved_frequency_bins) {
    if ((interleaved_time_samples.size() % 2) != 0) {
        throw std::invalid_argument("CF32 input must contain interleaved real/imag pairs");
    }
    if (interleaved_frequency_bins.size() < interleaved_time_samples.size()) {
        throw std::invalid_argument("FFT output CF32 span is too small");
    }
    const auto sample_count = interleaved_time_samples.size() / 2;
    if (sample_count == 0) {
        return;
    }

#ifdef DTMB_CORE_HAVE_FFTW3F
    // New-array execution is safe concurrently and FFTW_UNALIGNED makes the
    // plan independent of the allocator alignment used by caller-owned frame
    // buffers.  FFTW uses the same unnormalised forward-transform convention
    // as the built-in implementation.
    const auto plan = fftw_plan(sample_count);
    fftwf_execute_dft(
        plan,
        reinterpret_cast<fftwf_complex*>(
            const_cast<float*>(interleaved_time_samples.data())),
        reinterpret_cast<fftwf_complex*>(interleaved_frequency_bins.data()));
    return;
#else

    thread_local std::vector<Complex> input;
    thread_local std::vector<Complex> output;
    input.resize(sample_count);
    output.resize(sample_count);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        input[sample] = Complex{
            interleaved_time_samples[sample * 2],
            interleaved_time_samples[sample * 2 + 1],
        };
    }
    fft_recursive(input.data(), 1, output.data(), sample_count);
    for (std::size_t bin = 0; bin < sample_count; ++bin) {
        interleaved_frequency_bins[bin * 2] = output[bin].real();
        interleaved_frequency_bins[bin * 2 + 1] = output[bin].imag();
    }
#endif
}

void c3780_extract_frame_symbols_cf32(
    std::span<const float> interleaved_time_body,
    std::span<float> interleaved_logical_symbols) {
    if (interleaved_time_body.size() != kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("C=3780 body must contain exactly 3780 CF32 samples");
    }
    if (interleaved_logical_symbols.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("C=3780 logical output span is too small");
    }

    thread_local std::vector<float> spectrum;
    spectrum.resize(kC3780FrameBodySymbols * 2);
    mixed_radix_fft_forward_cf32(interleaved_time_body, spectrum);
    c3780_deinterleave_spectrum_cf32(spectrum, interleaved_logical_symbols);
}

void c3780_deinterleave_spectrum_cf32(
    std::span<const float> interleaved_physical_spectrum,
    std::span<float> interleaved_logical_symbols) {
    if (interleaved_physical_spectrum.size() != kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("C=3780 spectrum must contain exactly 3780 CF32 bins");
    }
    if (interleaved_logical_symbols.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("C=3780 logical output span is too small");
    }

    static const auto logical_to_physical = build_logical_to_physical();
    static const auto system_info_mask = build_system_info_mask();
    std::size_t output_symbol = 0;
    for (const auto inserted_position : kSystemInfoPositions) {
        const auto physical_bin = logical_to_physical[inserted_position];
        interleaved_logical_symbols[output_symbol * 2] =
            interleaved_physical_spectrum[physical_bin * 2];
        interleaved_logical_symbols[output_symbol * 2 + 1] =
            interleaved_physical_spectrum[physical_bin * 2 + 1];
        ++output_symbol;
    }
    for (std::size_t inserted_position = 0;
         inserted_position < kC3780FrameBodySymbols;
         ++inserted_position) {
        if (system_info_mask[inserted_position]) {
            continue;
        }
        const auto physical_bin = logical_to_physical[inserted_position];
        interleaved_logical_symbols[output_symbol * 2] =
            interleaved_physical_spectrum[physical_bin * 2];
        interleaved_logical_symbols[output_symbol * 2 + 1] =
            interleaved_physical_spectrum[physical_bin * 2 + 1];
        ++output_symbol;
    }
}

void c3780_extract_data_symbols_cf32(
    std::span<const float> interleaved_time_body,
    std::span<float> interleaved_data_symbols,
    bool normalize_qam64) {
    if (interleaved_data_symbols.size() < kC3780DataSymbols * 2) {
        throw std::invalid_argument("C=3780 data output span is too small");
    }
    std::vector<float> logical(kC3780FrameBodySymbols * 2);
    c3780_extract_frame_symbols_cf32(interleaved_time_body, logical);
    std::copy(
        logical.begin() + kC3780SystemInfoSymbols * 2,
        logical.end(),
        interleaved_data_symbols.begin());
    if (normalize_qam64) {
        qam64_normalize_cf32(
            interleaved_data_symbols.first(kC3780DataSymbols * 2),
            interleaved_data_symbols.first(kC3780DataSymbols * 2));
    }
}

}  // namespace dtmb::core
