#include "dtmb/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace dtmb::core {

PnMode parse_pn_mode(std::string_view name) {
    if (name == "pn420") return PnMode::pn420;
    if (name == "pn595") return PnMode::pn595;
    if (name == "pn945") return PnMode::pn945;
    throw std::invalid_argument("PN mode must be pn420, pn595 or pn945");
}

const char* pn_mode_name(PnMode mode) {
    switch (mode) {
    case PnMode::pn420: return "pn420";
    case PnMode::pn595: return "pn595";
    case PnMode::pn945: return "pn945";
    }
    throw std::invalid_argument("invalid PN mode");
}

std::span<const std::int8_t> pn_reference_chips(PnMode mode) {
    const auto generate = [](const PnDefinition& definition) {
        std::vector<std::int8_t> bits(definition.header_symbols);
        for (std::size_t n = 0; n < definition.seed.size(); ++n) {
            bits[n] = static_cast<std::int8_t>(definition.seed[n] - '0');
        }
        for (std::size_t n = definition.seed.size(); n < bits.size(); ++n) {
            for (std::size_t t = 0; t < definition.recurrence_tap_count; ++t) {
                bits[n] ^= bits[n - definition.seed.size() + definition.recurrence_taps[t]];
            }
        }
        for (auto& bit : bits) bit = bit == 0 ? 1 : -1;
        return bits;
    };
    static const auto pn420 = generate(kPn420Definition);
    static const auto pn595 = generate(kPn595Definition);
    static const auto pn945 = generate(kPn945Definition);
    switch (mode) {
    case PnMode::pn420: return pn420;
    case PnMode::pn595: return pn595;
    case PnMode::pn945: return pn945;
    }
    throw std::invalid_argument("invalid PN mode");
}

void pn_header_symbols_cf32(PnMode mode, std::span<float> output, std::size_t phase) {
    const auto& definition = pn_definition(mode);
    if (output.size() < definition.header_symbols * 2 || phase >= definition.phase_count()) {
        throw std::invalid_argument("PN reference needs a complete output header and a valid phase");
    }
    const auto chips = pn_reference_chips(mode);
    for (std::size_t n = 0; n < definition.header_symbols; ++n) {
        const auto index = definition.cyclic_extension()
            ? definition.prefix_symbols + (n + definition.core_symbols - definition.prefix_symbols + phase)
                % definition.core_symbols
            : n;
        output[n * 2] = output[n * 2 + 1] = static_cast<float>(chips[index]);
    }
}

namespace {

// Linear least squares against the fixed, truncated PN595 sequence. A row is
// admitted only if every delayed transmitted sample lies inside this header.
// There is no circular convolution, cyclic extension, or phase-family detector.
class DirectPnSolver {
public:
    DirectPnSolver(std::span<const std::int8_t> chips, int first_delay,
                   std::size_t count, float regularization)
        : columns_(count), header_size_(chips.size()) {
        const auto last_delay = first_delay + static_cast<int>(count) - 1;
        first_sample_ = static_cast<std::size_t>(std::max(last_delay, 0));
        const auto end = static_cast<int>(chips.size()) + std::min(first_delay, 0);
        if (end <= static_cast<int>(first_sample_) || count == 0
            || static_cast<std::size_t>(end) - first_sample_ < count
            || !std::isfinite(regularization) || regularization <= 0.0F) {
            throw std::invalid_argument("invalid PN595 direct least-squares support");
        }
        rows_ = static_cast<std::size_t>(end) - first_sample_;
        design_.resize(rows_ * columns_);
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t column = 0; column < columns_; ++column) {
                const auto index = static_cast<int>(first_sample_ + row) - first_delay - static_cast<int>(column);
                design_[row * columns_ + column] = chips[static_cast<std::size_t>(index)];
            }
        }
        lower_.assign(columns_ * columns_, 0.0);
        for (std::size_t i = 0; i < columns_; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double value = i == j ? static_cast<double>(regularization) / 2.0 : 0.0;
                for (std::size_t row = 0; row < rows_; ++row) {
                    value += design_[row * columns_ + i] * design_[row * columns_ + j];
                }
                for (std::size_t k = 0; k < j; ++k) value -= lower_[i * columns_ + k] * lower_[j * columns_ + k];
                if (i == j) {
                    if (!(value > 0.0)) throw std::runtime_error("singular PN595 channel fit");
                    lower_[i * columns_ + j] = std::sqrt(value);
                } else {
                    lower_[i * columns_ + j] = value / lower_[j * columns_ + j];
                }
            }
        }
    }

    [[nodiscard]] std::vector<std::complex<float>> fit(std::span<const float> header) const {
        if (header.size() != header_size_ * 2) throw std::invalid_argument("incomplete PN595 header");
        std::vector<std::complex<double>> solution(columns_);
        for (std::size_t row = 0; row < rows_; ++row) {
            const auto i = static_cast<double>(header[(first_sample_ + row) * 2]);
            const auto q = static_cast<double>(header[(first_sample_ + row) * 2 + 1]);
            const auto observation = std::complex<double>{(i + q) * 0.5, (q - i) * 0.5};
            for (std::size_t column = 0; column < columns_; ++column) {
                solution[column] += static_cast<double>(design_[row * columns_ + column]) * observation;
            }
        }
        for (std::size_t i = 0; i < columns_; ++i) {
            for (std::size_t j = 0; j < i; ++j) solution[i] -= lower_[i * columns_ + j] * solution[j];
            solution[i] /= lower_[i * columns_ + i];
        }
        for (std::size_t i = columns_; i-- > 0;) {
            for (std::size_t j = i + 1; j < columns_; ++j) solution[i] -= lower_[j * columns_ + i] * solution[j];
            solution[i] /= lower_[i * columns_ + i];
        }
        std::vector<std::complex<float>> result;
        result.reserve(columns_);
        for (auto tap : solution) result.emplace_back(static_cast<float>(tap.real()), static_cast<float>(tap.imag()));
        return result;
    }

private:
    std::size_t columns_, header_size_, first_sample_, rows_;
    std::vector<std::int8_t> design_;
    std::vector<double> lower_;
};

template<PnMode Mode>
struct PnChannel {
using Complex = std::complex<float>;

static constexpr auto definition = pn_definition(Mode);
static constexpr std::size_t kPnCoreSymbols = definition.core_symbols;
static constexpr std::size_t kPnPrefixSymbols = definition.prefix_symbols;
static constexpr std::size_t kPnSuffixSymbols = definition.suffix_symbols;
static constexpr std::size_t kPnHeaderSymbols = definition.header_symbols;
static constexpr std::size_t kPnFrameSymbols = definition.frame_symbols();
static constexpr std::size_t kPnPhaseCount = definition.phase_count();

[[nodiscard]] static const std::array<std::int8_t, kPnCoreSymbols>& pn_core_chips() {
    static const auto chips = [] {
        std::array<std::int8_t, kPnCoreSymbols> result{};
        const auto reference = pn_reference_chips(Mode);
        std::copy_n(reference.begin() + kPnPrefixSymbols, kPnCoreSymbols, result.begin());
        return result;
    }();
    return chips;
}

[[nodiscard]] static std::size_t rotate_phase(std::size_t phase, int rotation) {
    if constexpr (!definition.cyclic_extension()) return 0;
    const auto core = static_cast<int>(kPnCoreSymbols);
    return static_cast<std::size_t>((static_cast<int>(phase) - rotation + 2 * core) % core);
}

static constexpr std::size_t kDirectMaxFitCount = (kPnHeaderSymbols + 1) / 2;

[[nodiscard]] static bool tap_is_observed(std::size_t tap, std::size_t fit_count) {
    if constexpr (definition.cyclic_extension()) return true;
    const int first = -static_cast<int>((fit_count - 1) / 2);
    const int last = first + static_cast<int>(fit_count) - 1;
    return tap <= static_cast<std::size_t>(last)
        || tap >= static_cast<std::size_t>(static_cast<int>(kPnCoreSymbols) + first);
}

[[nodiscard]] static std::vector<Complex> estimate_direct_channel_taps(
    std::span<const float> header, std::size_t channel_taps, float regularization,
    std::size_t direct_fit_count) {
    const bool wideband = channel_taps == kPnCoreSymbols;
    const auto count = wideband ? direct_fit_count : channel_taps;
    if (count == 0 || count > kDirectMaxFitCount || !std::isfinite(regularization) || regularization <= 0.0F) {
        throw std::invalid_argument("PN595 direct channel fit requires at least as many observations as taps");
    }
    const int first = wideband ? -static_cast<int>((count - 1) / 2) : 0;
    // Reuse factorizations; each worker owns its cache. All observations used by
    // this linear fit lie wholly inside the known header, never in adjacent data.
    thread_local std::map<std::tuple<int, std::size_t, float>, DirectPnSolver> solvers;
    const auto key = std::make_tuple(first, count, regularization);
    auto entry = solvers.find(key);
    if (entry == solvers.end()) {
        entry = solvers.emplace(key, DirectPnSolver(pn_reference_chips(Mode), first, count, regularization)).first;
    }
    const auto fitted = entry->second.fit(header);
    std::vector<Complex> taps(channel_taps);
    for (std::size_t n = 0; n < fitted.size(); ++n) {
        const auto delay = first + static_cast<int>(n);
        const auto index = static_cast<std::size_t>((delay + static_cast<int>(channel_taps)) % static_cast<int>(channel_taps));
        taps[index] = fitted[n];
    }
    return taps;
}

[[nodiscard]] static Complex pn_core_symbol(std::size_t phase, std::size_t core_index) {
    const auto chip = static_cast<float>(
        pn_core_chips()[(core_index + phase) % kPnCoreSymbols]);
    return Complex{chip, chip};
}

[[nodiscard]] static Complex pn_header_symbol(std::size_t phase, std::size_t header_index) {
    if (header_index < kPnPrefixSymbols) {
        return pn_core_symbol(
            phase,
            kPnCoreSymbols - kPnPrefixSymbols + header_index);
    }
    if (header_index < kPnPrefixSymbols + kPnCoreSymbols) {
        return pn_core_symbol(phase, header_index - kPnPrefixSymbols);
    }
    return pn_core_symbol(
        phase,
        header_index - kPnPrefixSymbols - kPnCoreSymbols);
}

[[nodiscard]] static const std::vector<float>& pn_core_reference_fft(std::size_t phase) {
    static const auto cache = [] {
        std::array<std::vector<float>, kPnCoreSymbols> result{};
        for (std::size_t cached_phase = 0; cached_phase < kPnCoreSymbols; ++cached_phase) {
            std::vector<float> reference(kPnCoreSymbols * 2);
            for (std::size_t index = 0; index < kPnCoreSymbols; ++index) {
                const auto value = pn_core_symbol(cached_phase, index);
                reference[index * 2] = value.real();
                reference[index * 2 + 1] = value.imag();
            }
            result[cached_phase].resize(kPnCoreSymbols * 2);
            mixed_radix_fft_forward_cf32(reference, result[cached_phase]);
        }
        return result;
    }();
    return cache[phase % kPnCoreSymbols];
}

struct PnPhaseProjection {
    std::size_t phase = 0;
    Complex projection{};
};

struct PnHeaderCoreAccumulator {
    std::array<float, kPnCoreSymbols> real{};
    std::array<float, kPnCoreSymbols> imag{};
};

[[nodiscard]] static PnHeaderCoreAccumulator accumulate_pn_header_by_core_index(
    std::span<const float> interleaved_header) {
    PnHeaderCoreAccumulator accum{};
    for (std::size_t core_index = 0; core_index < kPnCoreSymbols; ++core_index) {
        const auto main_sample = kPnPrefixSymbols + core_index;
        accum.real[core_index] = interleaved_header[main_sample * 2];
        accum.imag[core_index] = interleaved_header[main_sample * 2 + 1];
    }
    for (std::size_t core_index = 0; core_index < kPnSuffixSymbols; ++core_index) {
        const auto post_sample = kPnPrefixSymbols + kPnCoreSymbols + core_index;
        accum.real[core_index] += interleaved_header[post_sample * 2];
        accum.imag[core_index] += interleaved_header[post_sample * 2 + 1];
    }
    for (std::size_t core_index = kPnCoreSymbols - kPnPrefixSymbols;
         core_index < kPnCoreSymbols;
         ++core_index) {
        const auto prefix_sample = core_index - (kPnCoreSymbols - kPnPrefixSymbols);
        accum.real[core_index] += interleaved_header[prefix_sample * 2];
        accum.imag[core_index] += interleaved_header[prefix_sample * 2 + 1];
    }
    return accum;
}

[[nodiscard]] static PnPhaseProjection pn_bipolar_phase_projection(
    std::span<const float> interleaved_header) {
    if constexpr (!definition.cyclic_extension()) {
        // A fixed aperiodic reference, with no cyclic phase search.
        Complex projection{};
        const auto chips = pn_reference_chips(Mode);
        for (std::size_t n = 0; n < kPnHeaderSymbols; ++n) {
            projection += static_cast<float>(chips[n]) * Complex{
                interleaved_header[n * 2], interleaved_header[n * 2 + 1]};
        }
        return PnPhaseProjection{0, projection};
    }
    const auto accum = accumulate_pn_header_by_core_index(interleaved_header);
    const auto& chips = pn_core_chips();
    std::size_t best_phase = 0;
    auto best_projection = Complex{};
    float best_power = -1.0F;
    for (std::size_t phase = 0; phase < kPnCoreSymbols; ++phase) {
        float projection_real = 0.0F;
        float projection_imag = 0.0F;
        auto chip_index = phase;
        for (std::size_t core_index = 0; core_index < kPnCoreSymbols; ++core_index) {
            if (chips[chip_index] > 0) {
                projection_real += accum.real[core_index];
                projection_imag += accum.imag[core_index];
            } else {
                projection_real -= accum.real[core_index];
                projection_imag -= accum.imag[core_index];
            }
            ++chip_index;
            if (chip_index == kPnCoreSymbols) {
                chip_index = 0;
            }
        }
        const auto projection = Complex{projection_real, projection_imag};
        const auto power = std::norm(projection);
        if (power > best_power) {
            best_power = power;
            best_phase = phase;
            best_projection = projection;
        }
    }
    return PnPhaseProjection{best_phase, best_projection};
}

[[nodiscard]] static std::size_t choose_residual_cfo_worker_count(
    std::size_t frame_count,
    std::size_t requested_workers) noexcept {
    if (frame_count == 0) {
        return 0;
    }
    auto worker_count = requested_workers;
    if (worker_count == 0) {
        worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    return std::clamp<std::size_t>(worker_count, 1, frame_count);
}

static void inverse_fft_cf32(
    std::span<const float> interleaved_frequency_bins,
    std::span<float> interleaved_time_samples) {
    std::vector<float> conjugated(interleaved_frequency_bins.size());
    for (std::size_t index = 0; index < interleaved_frequency_bins.size() / 2; ++index) {
        conjugated[index * 2] = interleaved_frequency_bins[index * 2];
        conjugated[index * 2 + 1] = -interleaved_frequency_bins[index * 2 + 1];
    }
    mixed_radix_fft_forward_cf32(conjugated, interleaved_time_samples);
    const auto size = static_cast<float>(interleaved_frequency_bins.size() / 2);
    for (std::size_t index = 0; index < interleaved_time_samples.size() / 2; ++index) {
        interleaved_time_samples[index * 2] /= size;
        interleaved_time_samples[index * 2 + 1] =
            -interleaved_time_samples[index * 2 + 1] / size;
    }
}

[[nodiscard]] static float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0F;
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if ((values.size() % 2) != 0) {
        return *middle;
    }
    const auto lower = std::max_element(values.begin(), middle);
    return (*lower + *middle) * 0.5F;
}

static void repair_low_energy_pn_dc_response(
    std::span<float> response_fft,
    std::span<const float> reference_fft) {
    std::vector<float> reference_power;
    reference_power.reserve(kPnCoreSymbols - 1);
    for (std::size_t bin = 1; bin < kPnCoreSymbols; ++bin) {
        reference_power.push_back(std::norm(Complex{
            reference_fft[bin * 2],
            reference_fft[bin * 2 + 1],
        }));
    }
    const auto typical_reference_power = median(std::move(reference_power));
    const auto dc_reference_power = std::norm(Complex{
        reference_fft[0],
        reference_fft[1],
    });
    if (typical_reference_power <= 0.0F
        || dc_reference_power > typical_reference_power * 0.01F) {
        return;
    }

    std::vector<float> provisional_taps(response_fft.size());
    inverse_fft_cf32(response_fft, provisional_taps);
    float peak = 0.0F;
    for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
        peak = std::max(peak, std::abs(Complex{
            provisional_taps[tap * 2],
            provisional_taps[tap * 2 + 1],
        }));
    }
    if (peak <= 0.0F) {
        return;
    }
    std::size_t last_significant = 0;
    for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
        if (std::abs(Complex{
                provisional_taps[tap * 2],
                provisional_taps[tap * 2 + 1],
            }) >= peak * 0.05F) {
            last_significant = tap;
        }
    }
    if (last_significant > 64) {
        return;
    }
    const auto interpolated = (
        Complex{response_fft[2], response_fft[3]}
        + Complex{
            response_fft[(kPnCoreSymbols - 1) * 2],
            response_fft[(kPnCoreSymbols - 1) * 2 + 1],
        }) * 0.5F;
    const auto current = Complex{response_fft[0], response_fft[1]};
    const auto scale = std::max(std::abs(interpolated), 1.0e-12F);
    if (std::abs(current - interpolated) <= scale * 0.25F) {
        return;
    }
    response_fft[0] = interpolated.real();
    response_fft[1] = interpolated.imag();
}

[[nodiscard]] static std::vector<Complex> estimate_channel_taps(
    std::span<const float> header,
    std::size_t phase,
    std::size_t channel_taps,
    float regularization,
    PnHeaderObservation observation = PnHeaderObservation::core_only,
    std::size_t safe_prefix_skip = kPnPrefixSymbols,
    std::size_t safe_postfix_skip = 0,
    std::size_t direct_fit_count = definition.default_channel_span()) {
    if constexpr (!definition.cyclic_extension()) {
        if (observation != PnHeaderObservation::core_only) {
            throw std::invalid_argument("PN595 requires direct header observations, without cyclic averaging");
        }
        return estimate_direct_channel_taps(header, channel_taps, regularization, direct_fit_count);
    }
    std::vector<float> observed(kPnCoreSymbols * 2);
    for (std::size_t index = 0; index < kPnCoreSymbols; ++index) {
        const auto main_sample = kPnPrefixSymbols + index;
        auto value = Complex{
            header[main_sample * 2],
            header[main_sample * 2 + 1],
        };
        std::size_t observation_count = 1;
        if (observation != PnHeaderObservation::core_only
            && index < kPnSuffixSymbols - safe_postfix_skip) {
            const auto postfix_sample =
                kPnPrefixSymbols + kPnCoreSymbols + index;
            value += Complex{
                header[postfix_sample * 2],
                header[postfix_sample * 2 + 1],
            };
            ++observation_count;
        }
        constexpr auto prefix_core_start = kPnCoreSymbols - kPnPrefixSymbols;
        if (observation == PnHeaderObservation::core_cyclic_safe_average
            && index >= prefix_core_start + safe_prefix_skip) {
            const auto prefix_sample = index - prefix_core_start;
            value += Complex{
                header[prefix_sample * 2],
                header[prefix_sample * 2 + 1],
            };
            ++observation_count;
        }
        value /= static_cast<float>(observation_count);
        observed[index * 2] = value.real();
        observed[index * 2 + 1] = value.imag();
    }

    std::vector<float> observed_fft(observed.size());
    const auto& reference_fft = pn_core_reference_fft(phase);
    std::vector<float> response_fft(reference_fft.size());
    std::vector<float> taps_cf32(reference_fft.size());
    mixed_radix_fft_forward_cf32(observed, observed_fft);
    for (std::size_t bin = 0; bin < kPnCoreSymbols; ++bin) {
        const auto obs = Complex{observed_fft[bin * 2], observed_fft[bin * 2 + 1]};
        const auto ref = Complex{reference_fft[bin * 2], reference_fft[bin * 2 + 1]};
        const auto response = obs * std::conj(ref) / (std::norm(ref) + regularization);
        response_fft[bin * 2] = response.real();
        response_fft[bin * 2 + 1] = response.imag();
    }
    repair_low_energy_pn_dc_response(response_fft, reference_fft);
    inverse_fft_cf32(response_fft, taps_cf32);

    std::vector<Complex> taps(
        std::min(channel_taps, kPnCoreSymbols),
        Complex{0.0F, 0.0F});
    for (std::size_t tap = 0; tap < taps.size(); ++tap) {
        taps[tap] = Complex{taps_cf32[tap * 2], taps_cf32[tap * 2 + 1]};
    }
    return taps;
}

[[nodiscard]] static std::vector<Complex> rotate_left(
    std::span<const Complex> values,
    std::size_t shift) {
    std::vector<Complex> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[(index + shift) % values.size()];
    }
    return result;
}

[[nodiscard]] static std::vector<bool> rotate_left(
    const std::vector<bool>& values,
    std::size_t shift) {
    std::vector<bool> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = values[(index + shift) % values.size()];
    }
    return result;
}

[[nodiscard]] static std::pair<std::size_t, std::size_t> longest_circular_false_run(
    const std::vector<bool>& mask) {
    if (mask.empty()) {
        return {0, 0};
    }
    if (std::all_of(mask.begin(), mask.end(), [](bool value) { return value; })) {
        return {0, 0};
    }
    if (std::none_of(mask.begin(), mask.end(), [](bool value) { return value; })) {
        return {0, mask.size()};
    }
    std::size_t best_start = 0;
    std::size_t best_length = 0;
    std::size_t run_start = 0;
    bool in_run = false;
    for (std::size_t index = 0; index < mask.size() * 2; ++index) {
        if (!mask[index % mask.size()]) {
            if (!in_run) {
                run_start = index;
                in_run = true;
            }
            const auto length = index - run_start + 1;
            if (run_start < mask.size() && length > best_length) {
                best_start = run_start;
                best_length = length;
            }
        } else {
            in_run = false;
        }
    }
    return {best_start % mask.size(), std::min(best_length, mask.size())};
}

static void dilate_circular_mask(std::vector<bool>& mask, std::size_t guard_taps) {
    if (guard_taps == 0 || mask.empty()
        || std::none_of(mask.begin(), mask.end(), [](bool value) { return value; })) {
        return;
    }
    if (guard_taps >= mask.size() / 2) {
        std::fill(mask.begin(), mask.end(), true);
        return;
    }
    const auto source = mask;
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (!source[index]) {
            continue;
        }
        for (std::size_t delta = 1; delta <= guard_taps; ++delta) {
            mask[(index + delta) % mask.size()] = true;
            mask[(index + mask.size() - delta) % mask.size()] = true;
        }
    }
}

[[nodiscard]] static Complex bounded_wideband_response_scale(
    Complex anchor,
    Complex reference,
    float per_frame_noise_tap_power) {
    auto scale = Complex{1.0F, 0.0F};
    const auto anchor_power = std::norm(anchor);
    if (std::abs(reference) > 0.0F
        && anchor_power > 4.0F * per_frame_noise_tap_power) {
        scale = anchor / reference;
        const auto magnitude = std::abs(scale);
        if (magnitude > 4.0F || magnitude < 0.25F) {
            scale /= magnitude;
        }
    }
    return scale;
}

[[nodiscard]] static Complex bounded_wideband_response_scale_from_sums(
    Complex cross_sum,
    float reference_power_sum,
    float observed_power_sum,
    std::size_t tap_count,
    float per_frame_noise_tap_power) {
    auto scale = Complex{1.0F, 0.0F};
    if (reference_power_sum > 0.0F
        && tap_count > 0
        && observed_power_sum
            > 4.0F * per_frame_noise_tap_power * static_cast<float>(tap_count)) {
        scale = cross_sum / reference_power_sum;
        const auto magnitude = std::abs(scale);
        if (magnitude > 4.0F || magnitude < 0.25F) {
            scale /= magnitude;
        }
    }
    return scale;
}

[[nodiscard]] static Complex wideband_response_scale_for_rotated_taps(
    std::span<const Complex> rotated_taps,
    const PnWidebandChannelModel& model,
    PnWidebandScaleEstimator estimator) {
    const auto template_size = model.template_taps.size() / 2;
    if (template_size == 0 || model.dominant_tap_index >= template_size) {
        throw std::invalid_argument("wideband PN model has invalid template taps");
    }
    if (estimator == PnWidebandScaleEstimator::dominant_tap) {
        const auto anchor = rotated_taps[model.dominant_tap_index];
        const auto reference = Complex{
            model.template_taps[model.dominant_tap_index * 2],
            model.template_taps[model.dominant_tap_index * 2 + 1],
        };
        return bounded_wideband_response_scale(
            anchor,
            reference,
            model.per_frame_noise_tap_power);
    }

    Complex cross_sum{0.0F, 0.0F};
    float reference_power_sum = 0.0F;
    float observed_power_sum = 0.0F;
    std::size_t tap_count = 0;
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        const auto reference = Complex{
            model.template_taps[tap * 2],
            model.template_taps[tap * 2 + 1],
        };
        const auto reference_power = std::norm(reference);
        if (reference_power <= 0.0F) {
            continue;
        }
        const auto observed = rotated_taps[tap];
        cross_sum += observed * std::conj(reference);
        reference_power_sum += reference_power;
        observed_power_sum += std::norm(observed);
        ++tap_count;
    }
    return bounded_wideband_response_scale_from_sums(
        cross_sum,
        reference_power_sum,
        observed_power_sum,
        tap_count,
        model.per_frame_noise_tap_power);
}

[[nodiscard]] static bool uses_masked_frame_taps(PnWidebandScaleEstimator estimator) noexcept {
    return estimator == PnWidebandScaleEstimator::masked_frame_taps;
}

[[nodiscard]] static std::vector<Complex> masked_frame_taps_from_rotated_taps(
    std::span<const Complex> rotated_taps,
    const PnWidebandChannelModel& model) {
    const auto template_size = model.template_taps.size() / 2;
    if (template_size == 0
        || model.template_tap_mask.size() != template_size
        || rotated_taps.size() < template_size) {
        throw std::invalid_argument("wideband PN frame-tap model has invalid mask");
    }
    std::vector<Complex> taps(template_size);
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        if (model.template_tap_mask[tap] != 0U) {
            taps[tap] = rotated_taps[tap];
        }
    }
    return taps;
}

[[nodiscard]] static std::vector<Complex> masked_frame_taps_for_model_frame(
    const PnWidebandChannelModel& model,
    std::size_t model_frame_index) {
    const auto template_size = model.template_taps.size() / 2;
    if (template_size == 0
        || (model_frame_index + 1) * template_size * 2 > model.frame_template_taps.size()) {
        throw std::invalid_argument("wideband PN cached frame taps are missing");
    }
    std::vector<Complex> taps(template_size);
    const auto base = model_frame_index * template_size * 2;
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        taps[tap] = Complex{
            model.frame_template_taps[base + tap * 2],
            model.frame_template_taps[base + tap * 2 + 1],
        };
    }
    return taps;
}

[[nodiscard]] static std::vector<Complex> scaled_template_taps(
    const PnWidebandChannelModel& model,
    Complex scale) {
    const auto template_size = model.template_taps.size() / 2;
    std::vector<Complex> result(template_size);
    for (std::size_t tap = 0; tap < template_size; ++tap) {
        result[tap] = Complex{
            model.template_taps[tap * 2],
            model.template_taps[tap * 2 + 1],
        } * scale;
    }
    return result;
}

[[nodiscard]] static std::size_t direct_fit_count_for_model(const PnWidebandChannelModel& model) {
    if constexpr (definition.cyclic_extension()) return definition.default_channel_span();
    auto first = static_cast<int>(model.rotation_symbols);
    if (first > static_cast<int>(kPnCoreSymbols / 2)) first -= static_cast<int>(kPnCoreSymbols);
    const auto last = first + static_cast<int>(model.template_taps.size() / 2) - 1;
    return std::min(kDirectMaxFitCount, std::max(definition.default_channel_span(),
        static_cast<std::size_t>(2 * std::max(std::abs(first), std::abs(last)) + 1)));
}

[[nodiscard]] static std::vector<Complex> instantiate_wideband_taps(
    std::span<const float> header,
    const PnWidebandChannelModel& model,
    float regularization,
    std::size_t& pn_phase,
    Complex& scale) {
    const auto header_phase = pn_detect_phase_cf32(header);
    const auto raw_taps = estimate_channel_taps(
        header,
        header_phase,
        kPnCoreSymbols,
        regularization,
        model.header_observation, kPnPrefixSymbols, 0, direct_fit_count_for_model(model));
    const auto rotated = rotate_left(raw_taps, model.rotation_symbols);
    if (uses_masked_frame_taps(model.scale_estimator)) {
        scale = Complex{1.0F, 0.0F};
        auto result = masked_frame_taps_from_rotated_taps(rotated, model);
        pn_phase = rotate_phase(header_phase, model.rotation_symbols);
        return result;
    }
    scale = wideband_response_scale_for_rotated_taps(
        rotated,
        model,
        model.scale_estimator);
    auto result = scaled_template_taps(model, scale);
    pn_phase = rotate_phase(header_phase, model.rotation_symbols);
    return result;
}

[[nodiscard]] static std::size_t wideband_pn_phase_only(
    std::span<const float> header,
    const PnWidebandChannelModel& model) {
    const auto header_phase = pn_detect_phase_cf32(header);
    return rotate_phase(header_phase, model.rotation_symbols);
}

[[nodiscard]] static Complex convolved_pn_header_sample(
    std::size_t phase,
    std::span<const Complex> taps,
    std::size_t output_index) {
    if (taps.empty()) {
        return {};
    }
    auto sum = Complex{};
    const auto max_tap = std::min(output_index, taps.size() - 1);
    for (std::size_t tap_count = max_tap + 1; tap_count > 0; --tap_count) {
        const auto tap = tap_count - 1;
        const auto header_index = output_index - tap;
        if (header_index >= kPnHeaderSymbols) {
            continue;
        }
        sum += pn_header_symbol(phase, header_index) * taps[tap];
    }
    return sum;
}

[[nodiscard]] static std::vector<Complex> midpoint_taps(
    std::span<const Complex> current_taps,
    std::span<const Complex> next_taps) {
    if (current_taps.size() != next_taps.size()) {
        throw std::invalid_argument("PN midpoint tap spans must match");
    }
    std::vector<Complex> result(current_taps.size());
    for (std::size_t tap = 0; tap < result.size(); ++tap) {
        result[tap] = 0.5F * (current_taps[tap] + next_taps[tap]);
    }
    return result;
}

static void restore_pn_boundaries(
    std::span<float> restored,
    std::span<const float> current_header,
    std::span<const float> next_header,
    std::size_t phase,
    std::size_t next_phase,
    std::span<const Complex> current_taps,
    std::span<const Complex> next_taps,
    int tap_origin) {
    if (tap_origin == 0) {
        // Preserve the established causal path, including its arithmetic order.
        const auto tail_length = std::min<std::size_t>(
            std::max(current_taps.size(), next_taps.size()) - 1, kC3780FrameBodySymbols);
        for (std::size_t sample = 0; sample < tail_length; ++sample) {
            auto body = Complex{restored[sample * 2], restored[sample * 2 + 1]};
            body -= convolved_pn_header_sample(phase, current_taps, kPnHeaderSymbols + sample);
            const auto observed_next = Complex{next_header[sample * 2], next_header[sample * 2 + 1]};
            body += observed_next - convolved_pn_header_sample(next_phase, next_taps, sample);
            restored[sample * 2] = body.real();
            restored[sample * 2 + 1] = body.imag();
        }
        return;
    }
    const auto first_delay = tap_origin;
    const auto last_delay = tap_origin + static_cast<int>(
        std::max(current_taps.size(), next_taps.size())) - 1;
    if (first_delay <= -static_cast<int>(kPnHeaderSymbols)
        || last_delay >= static_cast<int>(kPnHeaderSymbols)) {
        throw std::invalid_argument("PN channel exceeds available adjacent headers");
    }
    if (first_delay < 0 && current_header.size() != kPnHeaderSymbols * 2) {
        throw std::invalid_argument("PN precursor restoration requires the current received header");
    }
    // Tap rotation is a storage convention. PN boundary cancellation must use
    // the original transmitted phase and signed physical tap positions.
    phase = rotate_phase(phase, -tap_origin);
    next_phase = rotate_phase(next_phase, -tap_origin);
    for (int sample = 0; sample < last_delay; ++sample) {
        auto body = Complex{restored[sample * 2], restored[sample * 2 + 1]};
        for (std::size_t tap = 0; tap < current_taps.size(); ++tap) {
            const auto delay = tap_origin + static_cast<int>(tap);
            if (delay > sample) {
                body -= current_taps[tap] * pn_header_symbol(
                    phase, static_cast<std::size_t>(static_cast<int>(kPnHeaderSymbols) + sample - delay));
            }
        }
        auto residual = Complex{next_header[sample * 2], next_header[sample * 2 + 1]};
        for (std::size_t tap = 0; tap < next_taps.size(); ++tap) {
            const auto delay = tap_origin + static_cast<int>(tap);
            if (delay <= sample) {
                residual -= next_taps[tap] * pn_header_symbol(
                    next_phase, static_cast<std::size_t>(sample - delay));
            }
        }
        body += residual;
        restored[sample * 2] = body.real();
        restored[sample * 2 + 1] = body.imag();
    }
    for (int sample = first_delay; sample < 0; ++sample) {
        const auto output_sample = static_cast<int>(kC3780FrameBodySymbols) + sample;
        auto body = Complex{restored[output_sample * 2], restored[output_sample * 2 + 1]};
        for (std::size_t tap = 0; tap < next_taps.size(); ++tap) {
            const auto delay = tap_origin + static_cast<int>(tap);
            if (delay <= sample) {
                body -= next_taps[tap] * pn_header_symbol(
                    next_phase, static_cast<std::size_t>(sample - delay));
            }
        }
        const auto header_sample = static_cast<int>(kPnHeaderSymbols) + sample;
        auto residual = Complex{current_header[header_sample * 2], current_header[header_sample * 2 + 1]};
        for (std::size_t tap = 0; tap < current_taps.size(); ++tap) {
            const auto delay = tap_origin + static_cast<int>(tap);
            if (delay > sample) {
                residual -= current_taps[tap] * pn_header_symbol(
                    phase, static_cast<std::size_t>(header_sample - delay));
            }
        }
        body += residual;
        restored[output_sample * 2] = body.real();
        restored[output_sample * 2 + 1] = body.imag();
    }
}

static void restore_and_equalize_transition(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    std::size_t phase,
    std::size_t next_phase,
    std::span<const Complex> current_taps,
    std::span<const Complex> next_taps,
    std::span<const Complex> body_taps,
    float response_floor,
    float noise_variance,
    int response_window_offset,
    bool mmse_unbias,
    float mmse_unbias_gain_floor,
    std::span<float> interleaved_channel_power,
    std::span<const float> current_header = {},
    int tap_origin = 0) {
    if (current_taps.empty() || next_taps.empty() || body_taps.empty()) {
        throw std::invalid_argument("PN transition equalizer taps must not be empty");
    }
    if (!interleaved_channel_power.empty()
        && interleaved_channel_power.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("PN channel-power output span is too small");
    }
    if (mmse_unbias
        && (!std::isfinite(mmse_unbias_gain_floor)
            || mmse_unbias_gain_floor <= 0.0F
            || mmse_unbias_gain_floor > 1.0F)) {
        throw std::invalid_argument("PN MMSE unbias gain floor must be in (0,1]");
    }
    thread_local std::vector<float> restored;
    restored.assign(interleaved_time_body.begin(), interleaved_time_body.end());
    restore_pn_boundaries(restored, current_header, interleaved_next_header,
                          phase, next_phase, current_taps, next_taps, tap_origin);

    thread_local std::vector<float> padded_taps;
    thread_local std::vector<float> response;
    padded_taps.assign(kC3780FrameBodySymbols * 2, 0.0F);
    response.resize(kC3780FrameBodySymbols * 2);
    mixed_radix_fft_forward_cf32(restored, interleaved_equalized_spectrum);
    for (std::size_t tap = 0; tap < body_taps.size(); ++tap) {
        padded_taps[tap * 2] = body_taps[tap].real();
        padded_taps[tap * 2 + 1] = body_taps[tap].imag();
    }
    mixed_radix_fft_forward_cf32(padded_taps, response);
    for (std::size_t bin = 0; bin < kC3780FrameBodySymbols; ++bin) {
        const auto value = Complex{
            interleaved_equalized_spectrum[bin * 2],
            interleaved_equalized_spectrum[bin * 2 + 1],
        };
        auto channel = Complex{response[bin * 2], response[bin * 2 + 1]};
        if (response_window_offset != 0) {
            const auto angle = 2.0F * std::numbers::pi_v<float>
                * static_cast<float>(response_window_offset)
                * static_cast<float>(bin)
                / static_cast<float>(kC3780FrameBodySymbols);
            channel *= Complex{std::cos(angle), std::sin(angle)};
        }
        const auto channel_power = std::norm(channel);
        if (!interleaved_channel_power.empty()) {
            interleaved_channel_power[bin * 2] = channel_power;
            interleaved_channel_power[bin * 2 + 1] = 0.0F;
        }
        Complex equalized;
        if (noise_variance >= 0.0F) {
            equalized = value * std::conj(channel)
                / (channel_power + noise_variance);
            if (mmse_unbias) {
                const auto gain = channel_power / (channel_power + noise_variance);
                equalized /= std::max(gain, mmse_unbias_gain_floor);
            }
        } else {
            equalized = std::abs(channel) >= response_floor
                ? value / channel
                : value;
        }
        interleaved_equalized_spectrum[bin * 2] = equalized.real();
        interleaved_equalized_spectrum[bin * 2 + 1] = equalized.imag();
    }
}

static void restore_and_equalize(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    std::size_t phase,
    std::size_t next_phase,
    std::span<const Complex> taps,
    float response_floor,
    float noise_variance,
    int response_window_offset,
    bool mmse_unbias,
    float mmse_unbias_gain_floor,
    std::span<float> interleaved_channel_power,
    std::span<const float> current_header = {},
    int tap_origin = 0) {
    restore_and_equalize_transition(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        taps,
        taps,
        response_floor,
        noise_variance,
        response_window_offset,
        mmse_unbias,
        mmse_unbias_gain_floor,
        interleaved_channel_power,
        current_header,
        tap_origin);
}

static void restore_and_equalize_with_template_response(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    std::size_t phase,
    std::size_t next_phase,
    std::span<const Complex> taps,
    std::span<const float> template_response_fft,
    Complex response_scale,
    float response_floor,
    float noise_variance,
    int response_window_offset,
    bool mmse_unbias,
    float mmse_unbias_gain_floor,
    std::span<float> interleaved_channel_power,
    std::span<const float> current_header = {},
    int tap_origin = 0) {
    if (template_response_fft.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("wideband PN template response FFT is too small");
    }
    if (!interleaved_channel_power.empty()
        && interleaved_channel_power.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("PN channel-power output span is too small");
    }
    if (mmse_unbias
        && (!std::isfinite(mmse_unbias_gain_floor)
            || mmse_unbias_gain_floor <= 0.0F
            || mmse_unbias_gain_floor > 1.0F)) {
        throw std::invalid_argument("PN MMSE unbias gain floor must be in (0,1]");
    }
    thread_local std::vector<float> restored;
    restored.assign(interleaved_time_body.begin(), interleaved_time_body.end());
    restore_pn_boundaries(restored, current_header, interleaved_next_header,
                          phase, next_phase, taps, taps, tap_origin);

    mixed_radix_fft_forward_cf32(restored, interleaved_equalized_spectrum);
    for (std::size_t bin = 0; bin < kC3780FrameBodySymbols; ++bin) {
        const auto value = Complex{
            interleaved_equalized_spectrum[bin * 2],
            interleaved_equalized_spectrum[bin * 2 + 1],
        };
        auto channel = Complex{
            template_response_fft[bin * 2],
            template_response_fft[bin * 2 + 1],
        } * response_scale;
        if (response_window_offset != 0) {
            const auto angle = 2.0F * std::numbers::pi_v<float>
                * static_cast<float>(response_window_offset)
                * static_cast<float>(bin)
                / static_cast<float>(kC3780FrameBodySymbols);
            channel *= Complex{std::cos(angle), std::sin(angle)};
        }
        const auto channel_power = std::norm(channel);
        if (!interleaved_channel_power.empty()) {
            interleaved_channel_power[bin * 2] = channel_power;
            interleaved_channel_power[bin * 2 + 1] = 0.0F;
        }
        Complex equalized;
        if (noise_variance >= 0.0F) {
            equalized = value * std::conj(channel)
                / (channel_power + noise_variance);
            if (mmse_unbias) {
                const auto gain = channel_power / (channel_power + noise_variance);
                equalized /= std::max(gain, mmse_unbias_gain_floor);
            }
        } else {
            equalized = std::abs(channel) >= response_floor
                ? value / channel
                : value;
        }
        interleaved_equalized_spectrum[bin * 2] = equalized.real();
        interleaved_equalized_spectrum[bin * 2 + 1] = equalized.imag();
    }
}

static std::size_t pn_phase_for_frame(std::size_t index) noexcept {
    if constexpr (!definition.cyclic_extension()) return 0;
    index %= definition.frames_per_superframe;
    constexpr auto turnaround = definition.frames_per_superframe / 2;
    constexpr auto reflected_end = 2 * turnaround;
    if (index > turnaround) index = reflected_end - index;
    return index == 0 ? 0 : ((index % 2) != 0 ? (index + 1) / 2 : kPnCoreSymbols - index / 2);
}

static float pn_known_phase_metric_ci8(
    std::span<const std::int8_t> header, std::size_t phase) {
    if (header.size() != kPnHeaderSymbols * 2 || phase >= kPnPhaseCount) {
        throw std::invalid_argument("known-phase PN metric needs a complete header and a valid selected-mode phase");
    }
    double sum_i = 0.0;
    double sum_q = 0.0;
    double power = 0.0;
    for (std::size_t n = 0; n < kPnCoreSymbols; ++n) {
        const auto i = static_cast<double>(header[(n + kPnPrefixSymbols) * 2]);
        const auto q = static_cast<double>(header[(n + kPnPrefixSymbols) * 2 + 1]);
        const auto chip = pn_core_chips()[(n + phase) % kPnCoreSymbols];
        sum_i += chip * i;
        sum_q += chip * q;
        power += i * i + q * q;
    }
    if (power == 0.0) return 0.0F;
    return static_cast<float>(std::sqrt((sum_i * sum_i + sum_q * sum_q)
        / (kPnCoreSymbols * power)));
}

static PnScheduleAlignment fit_pn_phase_schedule(
    std::span<const std::size_t> observed_phases) {
    constexpr std::size_t period = definition.frames_per_superframe;
    if constexpr (!definition.cyclic_extension()) {
        throw std::invalid_argument("PN595 has a fixed header, not a cyclic phase schedule");
    }
    PnScheduleAlignment result;
    result.observations = std::min(observed_phases.size(), period);
    for (const auto phase : observed_phases.first(result.observations)) {
        if (phase >= kPnPhaseCount) {
            throw std::invalid_argument("PN phase must be within the selected PN phase family");
        }
    }
    if (result.observations < period) {
        return result;
    }
    const auto phase_at = [](std::size_t index) {
        return static_cast<int>(pn_phase_for_frame(index));
    };
    const auto distance = [](int observed, int expected) {
        constexpr auto core = static_cast<int>(kPnCoreSymbols);
        return (observed - expected + 2 * core + core / 2) % core - core / 2;
    };
    std::array<double, period> costs;
    costs.fill(std::numeric_limits<double>::infinity());
    std::array<int, period> biases{};
    for (std::size_t origin = 0; origin < period; ++origin) {
        for (int bias = -static_cast<int>(kPnPrefixSymbols); bias <= static_cast<int>(kPnSuffixSymbols); ++bias) {
            double sum = 0.0;
            for (std::size_t frame = 0; frame < period; ++frame) {
                const auto error = distance(static_cast<int>(observed_phases[frame]),
                    phase_at(origin + frame) + bias);
                sum += std::min(error * error, 25);
            }
            const auto cost = sum / period;
            if (cost < costs[origin]
                || (cost == costs[origin] && std::abs(bias) < std::abs(biases[origin]))) {
                costs[origin] = cost;
                biases[origin] = bias;
            }
        }
    }
    result.superframe_index = static_cast<std::size_t>(
        std::distance(costs.begin(), std::min_element(costs.begin(), costs.end())));
    result.phase_bias = biases[result.superframe_index];
    result.mean_squared_error = costs[result.superframe_index];
    result.runner_up_error = std::numeric_limits<double>::infinity();
    for (std::size_t origin = 0; origin < period; ++origin) {
        if (origin != result.superframe_index) {
            result.runner_up_error = std::min(result.runner_up_error, costs[origin]);
        }
    }
    for (std::size_t frame = 0; frame < period; ++frame) {
        result.inliers += std::abs(distance(static_cast<int>(observed_phases[frame]),
            phase_at(result.superframe_index + frame) + result.phase_bias)) <= 1;
    }
    result.valid = result.mean_squared_error <= 1.0
        && result.runner_up_error - result.mean_squared_error >= 0.25
        && result.inliers * 20 >= period * 19;
    return result;
}

static std::size_t pn_detect_phase_cf32(std::span<const float> interleaved_header) {
    if (interleaved_header.size() != kPnHeaderSymbols * 2) {
        throw std::invalid_argument("PN header must contain one complete selected-mode CF32 header");
    }
    return pn_bipolar_phase_projection(interleaved_header).phase;
}

static PnResidualCfoResult estimate_pn_residual_cfo_cf32(
    std::span<const float> interleaved_symbols,
    std::size_t phase_offset,
    PnResidualCfoOptions options) {
    if ((interleaved_symbols.size() % 2) != 0) {
        throw std::invalid_argument("PN residual-CFO input must contain CF32 pairs");
    }
    if (options.max_frames < 2) {
        throw std::invalid_argument("PN residual-CFO max_frames must be at least 2");
    }
    if (options.min_fit_r_squared < 0.0F || options.min_fit_r_squared > 1.0F) {
        throw std::invalid_argument("PN residual-CFO min_fit_r_squared must be in [0, 1]");
    }

    const auto sample_count = interleaved_symbols.size() / 2;
    if (phase_offset + kPnHeaderSymbols > sample_count) {
        return {};
    }
    const auto available_frames =
        1 + (sample_count - phase_offset - kPnHeaderSymbols) / kPnFrameSymbols;
    const auto frame_count = std::min(options.max_frames, available_frames);
    if (frame_count < 2) {
        return {};
    }

    std::vector<PnPhaseProjection> projections(frame_count);
    const auto worker_count = choose_residual_cfo_worker_count(
        frame_count,
        options.requested_workers);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::size_t frame = worker; frame < frame_count; frame += worker_count) {
                const auto start = phase_offset + frame * kPnFrameSymbols;
                const auto header = std::span<const float>(
                    interleaved_symbols.data() + start * 2,
                    kPnHeaderSymbols * 2);
                projections[frame] = pn_bipolar_phase_projection(header);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    std::vector<double> phases;
    std::vector<double> weights;
    std::vector<double> frame_indices;
    phases.reserve(frame_count);
    weights.reserve(frame_count);
    frame_indices.reserve(frame_count);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const auto weight = std::abs(projections[frame].projection);
        if (weight <= 0.0F) {
            continue;
        }
        auto phase = static_cast<double>(std::arg(projections[frame].projection));
        if (!phases.empty()) {
            while (phase - phases.back() > std::numbers::pi_v<double>) {
                phase -= 2.0 * std::numbers::pi_v<double>;
            }
            while (phase - phases.back() < -std::numbers::pi_v<double>) {
                phase += 2.0 * std::numbers::pi_v<double>;
            }
        }
        phases.push_back(phase);
        weights.push_back(weight);
        frame_indices.push_back(static_cast<double>(frame));
    }
    if (phases.size() < 2) {
        return PnResidualCfoResult{
            0.0F,
            0.0F,
            phases.size(),
            worker_count,
            false,
        };
    }

    const auto max_weight = *std::max_element(weights.begin(), weights.end());
    if (max_weight > 0.0) {
        for (auto& weight : weights) {
            weight /= max_weight;
        }
    } else {
        std::fill(weights.begin(), weights.end(), 1.0);
    }
    double weight_sum = 0.0;
    double weighted_index_sum = 0.0;
    double weighted_phase_sum = 0.0;
    for (std::size_t index = 0; index < phases.size(); ++index) {
        weight_sum += weights[index];
        weighted_index_sum += weights[index] * frame_indices[index];
        weighted_phase_sum += weights[index] * phases[index];
    }
    const auto mean_index = weighted_index_sum / weight_sum;
    const auto mean_phase = weighted_phase_sum / weight_sum;
    double covariance = 0.0;
    double index_variance = 0.0;
    double phase_variance = 0.0;
    for (std::size_t index = 0; index < phases.size(); ++index) {
        const auto index_delta = frame_indices[index] - mean_index;
        const auto phase_delta = phases[index] - mean_phase;
        covariance += weights[index] * index_delta * phase_delta;
        index_variance += weights[index] * index_delta * index_delta;
        phase_variance += weights[index] * phase_delta * phase_delta;
    }
    if (index_variance <= 0.0) {
        return PnResidualCfoResult{
            0.0F,
            0.0F,
            phases.size(),
            worker_count,
            false,
        };
    }
    const auto slope = covariance / index_variance;
    auto r_squared = 1.0;
    if (phase_variance > 0.0) {
        const auto residual_variance = phase_variance - slope * covariance;
        r_squared = 1.0 - residual_variance / phase_variance;
    }
    const auto cfo_hz = slope * static_cast<double>(kDtmbSymbolRateSps)
        / (2.0 * std::numbers::pi_v<double> * static_cast<double>(kPnFrameSymbols));
    return PnResidualCfoResult{
        static_cast<float>(cfo_hz),
        static_cast<float>(r_squared),
        phases.size(),
        worker_count,
        r_squared >= options.min_fit_r_squared,
    };
}

static PnEqualizeResult pn_equalize_c3780_frame_cf32(
    std::span<const float> interleaved_header,
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    PnEqualizeOptions options) {
    if (interleaved_header.size() != kPnHeaderSymbols * 2
        || interleaved_next_header.size() != kPnHeaderSymbols * 2) {
        throw std::invalid_argument("PN headers must contain one complete selected-mode CF32 header");
    }
    if (interleaved_time_body.size() != kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("C=3780 body must contain exactly 3780 CF32 samples");
    }
    if (interleaved_equalized_spectrum.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("equalized C=3780 spectrum output span is too small");
    }
    if (options.channel_taps == 0 || options.channel_taps > kPnCoreSymbols) {
        throw std::invalid_argument("PN channel taps must be within the selected header channel support");
    }
    if (options.regularization <= 0.0F || options.response_floor <= 0.0F) {
        throw std::invalid_argument("PN regularization and response floor must be positive");
    }

    const auto phase = pn_detect_phase_cf32(interleaved_header);
    const auto next_phase = pn_detect_phase_cf32(interleaved_next_header);
    const auto taps = estimate_channel_taps(
        interleaved_header,
        phase,
        options.channel_taps,
        options.regularization);

    restore_and_equalize(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        options.response_floor,
        options.noise_variance,
        0,
        options.mmse_unbias,
        options.mmse_unbias_gain_floor,
        options.interleaved_channel_power);
    return PnEqualizeResult{phase, next_phase};
}

static PnWidebandChannelModel build_pn_wideband_channel_model_cf32(
    std::span<const float> interleaved_headers,
    PnWidebandModelOptions options) {
    const auto header_stride = kPnHeaderSymbols * 2;
    if (interleaved_headers.empty()
        || (interleaved_headers.size() % header_stride) != 0) {
        throw std::invalid_argument(
            "wideband PN model needs one or more complete interleaved headers");
    }
    if (options.threshold_factor <= 0.0F
        || options.max_span_symbols == 0
        || options.max_span_symbols > kPnCoreSymbols
        || options.min_relative_power <= 0.0F
        || options.regularization <= 0.0F) {
        throw std::invalid_argument("invalid wideband PN model options");
    }
    const auto frame_count = interleaved_headers.size() / header_stride;
    if (!options.expected_phases.empty() && options.expected_phases.size() != frame_count) {
        throw std::invalid_argument("wideband PN phase count must match the header count");
    }
    for (const auto phase : options.expected_phases) {
        if (phase >= kPnPhaseCount) {
            throw std::invalid_argument("wideband PN expected phase must be within the selected PN phase family");
        }
    }
    std::vector<std::size_t> phases(frame_count);
    std::vector<std::vector<Complex>> rows;
    std::vector<std::vector<Complex>> structure_rows;
    rows.reserve(frame_count);
    structure_rows.reserve(frame_count);
    std::array<std::size_t, kPnCoreSymbols> phase_counts{};
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const auto header = interleaved_headers.subspan(frame * header_stride, header_stride);
        phases[frame] = options.expected_phases.empty()
            ? pn_detect_phase_cf32(header) : options.expected_phases[frame];
        ++phase_counts[phases[frame]];
        rows.push_back(estimate_channel_taps(
            header,
            phases[frame],
            kPnCoreSymbols,
            options.regularization,
            options.header_observation, kPnPrefixSymbols, 0, options.max_span_symbols));
        if (options.header_observation == PnHeaderObservation::core_only) {
            structure_rows.push_back(rows.back());
        } else {
            structure_rows.push_back(estimate_channel_taps(
                header,
                phases[frame],
                kPnCoreSymbols,
                options.regularization,
                PnHeaderObservation::core_only, kPnPrefixSymbols, 0, options.max_span_symbols));
        }
    }
    const auto base_phase = static_cast<std::size_t>(
        std::distance(
            phase_counts.begin(),
            std::max_element(phase_counts.begin(), phase_counts.end())));

    std::vector<float> raw_power(kPnCoreSymbols, 0.0F);
    for (const auto& row : structure_rows) {
        for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
            raw_power[tap] += std::norm(row[tap]) / static_cast<float>(frame_count);
        }
    }
    const auto dominant_raw = static_cast<std::size_t>(
        std::distance(raw_power.begin(), std::max_element(raw_power.begin(), raw_power.end())));
    std::vector<Complex> mean_taps(kPnCoreSymbols, Complex{0.0F, 0.0F});
    for (const auto& row : structure_rows) {
        const auto anchor = row[dominant_raw];
        const auto rotation = std::abs(anchor) > 0.0F
            ? std::conj(anchor) / std::abs(anchor)
            : Complex{1.0F, 0.0F};
        for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
            mean_taps[tap] += row[tap] * rotation / static_cast<float>(frame_count);
        }
    }

    std::vector<float> averaged_power(kPnCoreSymbols);
    for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
        averaged_power[tap] = std::norm(mean_taps[tap]);
    }
    std::vector<float> observable_power;
    for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
        if (tap_is_observed(tap, options.max_span_symbols)) observable_power.push_back(averaged_power[tap]);
    }
    auto noise_power = median(observable_power);
    const auto peak_power = *std::max_element(averaged_power.begin(), averaged_power.end());
    auto make_mask = [&](float noise) {
        const auto threshold = std::max(
            options.threshold_factor * options.threshold_factor * std::max(noise, 0.0F),
            peak_power * options.min_relative_power);
        std::vector<bool> result(kPnCoreSymbols);
        for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
            result[tap] = averaged_power[tap] >= threshold;
        }
        return result;
    };
    auto mask = make_mask(noise_power);
    std::vector<float> provisional_noise;
    for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
        if (!mask[tap] && tap_is_observed(tap, options.max_span_symbols)) {
            provisional_noise.push_back(averaged_power[tap]);
        }
    }
    const auto refined_noise = median(std::move(provisional_noise));
    if (refined_noise > 0.0F) {
        noise_power = refined_noise;
        mask = make_mask(noise_power);
    }
    if (std::none_of(mask.begin(), mask.end(), [](bool value) { return value; })) {
        mask[dominant_raw] = true;
    }

    std::vector<float> per_frame_noise_samples;
    for (const auto& row : structure_rows) {
        for (std::size_t tap = 0; tap < kPnCoreSymbols; ++tap) {
            if (!mask[tap] && tap_is_observed(tap, options.max_span_symbols)) {
                per_frame_noise_samples.push_back(std::norm(row[tap]));
            }
        }
    }
    const auto per_frame_noise = median(std::move(per_frame_noise_samples));
    // Guard while the impulse response is still circular. Cutting at the first
    // significant tap before dilation silently discarded every leading guard
    // tap, including weak physical precursors immediately before that tap.
    auto guarded_mask = mask;
    dilate_circular_mask(guarded_mask, options.guard_taps);
    const auto [gap_start, gap_length] = longest_circular_false_run(guarded_mask);
    const auto rotation = (gap_start + gap_length) % kPnCoreSymbols;
    auto rotated_taps = rotate_left(mean_taps, rotation);
    auto rotated_mask = rotate_left(guarded_mask, rotation);
    std::size_t span = 1;
    for (std::size_t tap = 0; tap < rotated_mask.size(); ++tap) {
        if (rotated_mask[tap]) {
            span = tap + 1;
        }
    }
    span = std::min(span, options.max_span_symbols);
    auto first_delay = static_cast<int>(rotation);
    if (first_delay > static_cast<int>(kPnCoreSymbols / 2)) {
        first_delay -= static_cast<int>(kPnCoreSymbols);
    }
    const auto last_delay = first_delay + static_cast<int>(span) - 1;
    if (options.header_observation == PnHeaderObservation::core_cyclic_safe_average
        || (options.header_observation == PnHeaderObservation::core_postfix_average
            && first_delay < 0)) {
        // Only repeated PN samples unaffected by adjacent data are usable.
        // These bounds are physical delays, not offsets in the rotated array.
        const auto safe_prefix_skip = static_cast<std::size_t>(
            std::clamp(last_delay, 0, static_cast<int>(kPnPrefixSymbols)));
        const auto safe_postfix_skip = static_cast<std::size_t>(
            std::clamp(-first_delay, 0, static_cast<int>(kPnSuffixSymbols)));
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            const auto header = interleaved_headers.subspan(frame * header_stride, header_stride);
            rows[frame] = estimate_channel_taps(
                header,
                phases[frame],
                kPnCoreSymbols,
                options.regularization,
                options.header_observation,
                safe_prefix_skip,
                safe_postfix_skip, options.max_span_symbols);
        }
    }
    std::vector<float> template_taps(span * 2, 0.0F);
    std::vector<std::uint8_t> template_tap_mask(span, 0U);
    std::size_t significant_taps = 0;
    float kept_energy = 0.0F;
    float dropped_energy = 0.0F;
    for (std::size_t tap = 0; tap < rotated_mask.size(); ++tap) {
        if (!rotated_mask[tap]) {
            continue;
        }
        if (tap < span) {
            template_taps[tap * 2] = rotated_taps[tap].real();
            template_taps[tap * 2 + 1] = rotated_taps[tap].imag();
            template_tap_mask[tap] = 1U;
            kept_energy += std::norm(rotated_taps[tap]);
            ++significant_taps;
        } else {
            dropped_energy += std::norm(rotated_taps[tap]);
        }
    }
    std::size_t dominant_tap = 0;
    float dominant_power = -1.0F;
    for (std::size_t tap = 0; tap < span; ++tap) {
        const auto power = std::norm(Complex{
            template_taps[tap * 2],
            template_taps[tap * 2 + 1],
        });
        if (power > dominant_power) {
            dominant_power = power;
            dominant_tap = tap;
        }
    }
    std::vector<float> padded_taps(kC3780FrameBodySymbols * 2, 0.0F);
    std::copy(template_taps.begin(), template_taps.end(), padded_taps.begin());
    std::vector<float> template_response_fft(kC3780FrameBodySymbols * 2);
    mixed_radix_fft_forward_cf32(padded_taps, template_response_fft);

    std::vector<std::size_t> frame_pn_phases(frame_count);
    std::vector<float> frame_response_scales(frame_count * 2);
    std::vector<float> frame_template_taps;
    if (uses_masked_frame_taps(options.scale_estimator)) {
        frame_template_taps.assign(frame_count * span * 2, 0.0F);
    }
    const auto dominant_reference = Complex{
        template_taps[dominant_tap * 2],
        template_taps[dominant_tap * 2 + 1],
    };
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        frame_pn_phases[frame] =
            rotate_phase(phases[frame], rotation);
        Complex scale{1.0F, 0.0F};
        if (uses_masked_frame_taps(options.scale_estimator)) {
            Complex cross_sum{0.0F, 0.0F};
            float reference_power_sum = 0.0F;
            float observed_power_sum = 0.0F;
            std::size_t tap_count = 0;
            for (std::size_t tap = 0; tap < span; ++tap) {
                if (template_tap_mask[tap] == 0U) {
                    continue;
                }
                const auto observed =
                    rows[frame][(tap + rotation) % kPnCoreSymbols];
                const auto base = (frame * span + tap) * 2;
                frame_template_taps[base] = observed.real();
                frame_template_taps[base + 1] = observed.imag();
                const auto reference = Complex{
                    template_taps[tap * 2],
                    template_taps[tap * 2 + 1],
                };
                const auto reference_power = std::norm(reference);
                if (reference_power <= 0.0F) {
                    continue;
                }
                cross_sum += observed * std::conj(reference);
                reference_power_sum += reference_power;
                observed_power_sum += std::norm(observed);
                ++tap_count;
            }
            scale = bounded_wideband_response_scale_from_sums(
                cross_sum,
                reference_power_sum,
                observed_power_sum,
                tap_count,
                per_frame_noise);
        } else if (options.scale_estimator
            == PnWidebandScaleEstimator::least_squares_template) {
            Complex cross_sum{0.0F, 0.0F};
            float reference_power_sum = 0.0F;
            float observed_power_sum = 0.0F;
            std::size_t tap_count = 0;
            for (std::size_t tap = 0; tap < span; ++tap) {
                const auto template_tap = Complex{
                    template_taps[tap * 2],
                    template_taps[tap * 2 + 1],
                };
                const auto template_power = std::norm(template_tap);
                if (template_power <= 0.0F) {
                    continue;
                }
                const auto observed =
                    rows[frame][(tap + rotation) % kPnCoreSymbols];
                cross_sum += observed * std::conj(template_tap);
                reference_power_sum += template_power;
                observed_power_sum += std::norm(observed);
                ++tap_count;
            }
            scale = bounded_wideband_response_scale_from_sums(
                cross_sum,
                reference_power_sum,
                observed_power_sum,
                tap_count,
                per_frame_noise);
        } else {
            const auto anchor =
                rows[frame][(dominant_tap + rotation) % kPnCoreSymbols];
            scale = bounded_wideband_response_scale(
                anchor,
                dominant_reference,
                per_frame_noise);
        }
        frame_response_scales[frame * 2] = scale.real();
        frame_response_scales[frame * 2 + 1] = scale.imag();
    }
    return PnWidebandChannelModel{
        base_phase,
        rotate_phase(base_phase, rotation),
        rotation,
        std::move(template_taps),
        std::move(template_response_fft),
        std::move(template_tap_mask),
        std::move(frame_pn_phases),
        std::move(frame_response_scales),
        std::move(frame_template_taps),
        significant_taps,
        dominant_tap,
        frame_count,
        static_cast<float>(phase_counts[base_phase]) / static_cast<float>(frame_count),
        per_frame_noise,
        per_frame_noise * static_cast<float>(kC3780FrameBodySymbols),
        (kept_energy + dropped_energy) > 0.0F
            ? dropped_energy / (kept_energy + dropped_energy)
            : 0.0F,
        options.scale_estimator,
        options.header_observation,
        std::vector<float>(interleaved_headers.begin(), interleaved_headers.end()),
    };
}

static PnEqualizeResult pn_equalize_c3780_frame_wideband_cf32(
    std::span<const float> interleaved_header,
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    const PnWidebandChannelModel& model,
    PnEqualizeOptions options) {
    if (interleaved_header.size() != kPnHeaderSymbols * 2
        || interleaved_next_header.size() != kPnHeaderSymbols * 2) {
        throw std::invalid_argument("PN headers must contain one complete selected-mode CF32 header");
    }
    if (interleaved_time_body.size() != kC3780FrameBodySymbols * 2
        || interleaved_equalized_spectrum.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("wideband PN equalizer needs one complete C=3780 body");
    }
    if (options.regularization <= 0.0F || options.response_floor <= 0.0F) {
        throw std::invalid_argument("PN regularization and response floor must be positive");
    }
    std::size_t phase = 0;
    Complex response_scale{1.0F, 0.0F};
    const auto taps = instantiate_wideband_taps(
        interleaved_header,
        model,
        options.regularization,
        phase,
        response_scale);
    std::size_t next_phase = 0;
    Complex next_response_scale{1.0F, 0.0F};
    std::vector<Complex> next_taps;
    if (options.interpolate_body_channel) {
        next_taps = instantiate_wideband_taps(
            interleaved_next_header,
            model,
            options.regularization,
            next_phase,
            next_response_scale);
    } else {
        next_phase = wideband_pn_phase_only(interleaved_next_header, model);
    }
    auto signed_rotation = static_cast<int>(model.rotation_symbols);
    if (signed_rotation > static_cast<int>(kPnCoreSymbols / 2)) {
        signed_rotation -= static_cast<int>(kPnCoreSymbols);
    }
    if (options.interpolate_body_channel) {
        const auto body_taps = midpoint_taps(taps, next_taps);
        restore_and_equalize_transition(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            next_taps,
            body_taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power,
            interleaved_header,
            signed_rotation);
        return PnEqualizeResult{phase, next_phase};
    }
    if (uses_masked_frame_taps(model.scale_estimator)) {
        restore_and_equalize(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power,
            interleaved_header,
            signed_rotation);
        return PnEqualizeResult{phase, next_phase};
    }
    restore_and_equalize_with_template_response(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        model.template_response_fft,
        response_scale,
        options.response_floor,
        options.noise_variance,
        -signed_rotation + options.response_window_offset_adjust,
        options.mmse_unbias,
        options.mmse_unbias_gain_floor,
        options.interleaved_channel_power,
        interleaved_header,
        signed_rotation);
    return PnEqualizeResult{phase, next_phase};
}

static PnEqualizeResult pn_equalize_c3780_frame_wideband_cached_cf32(
    std::span<const float> interleaved_time_body,
    std::span<const float> interleaved_next_header,
    std::span<float> interleaved_equalized_spectrum,
    const PnWidebandChannelModel& model,
    std::size_t model_frame_index,
    PnEqualizeOptions options) {
    if (interleaved_next_header.size() != kPnHeaderSymbols * 2) {
        throw std::invalid_argument("PN next header must contain one complete selected-mode CF32 header");
    }
    if (interleaved_time_body.size() != kC3780FrameBodySymbols * 2
        || interleaved_equalized_spectrum.size() < kC3780FrameBodySymbols * 2) {
        throw std::invalid_argument("wideband PN cached equalizer needs one C=3780 body");
    }
    if (options.response_floor <= 0.0F) {
        throw std::invalid_argument("PN response floor must be positive");
    }
    if (model.template_taps.empty() || (model.template_taps.size() % 2) != 0) {
        throw std::invalid_argument("wideband PN cached model has invalid template taps");
    }
    if (model_frame_index + 1 >= model.frame_pn_phases.size()
        || (model_frame_index + 1) * 2 + 1 >= model.frame_response_scales.size()) {
        throw std::invalid_argument("wideband PN cached frame index is outside model state");
    }
    const auto header_values = kPnHeaderSymbols * 2;
    const auto current_header = model_frame_index < model.frame_headers_cf32.size() / header_values
        ? std::span<const float>(model.frame_headers_cf32).subspan(model_frame_index * header_values, header_values)
        : std::span<const float>{};
    const auto phase = model.frame_pn_phases[model_frame_index];
    const auto next_phase = model.frame_pn_phases[model_frame_index + 1];
    const auto response_scale = Complex{
        model.frame_response_scales[model_frame_index * 2],
        model.frame_response_scales[model_frame_index * 2 + 1],
    };
    const auto taps = uses_masked_frame_taps(model.scale_estimator)
        ? masked_frame_taps_for_model_frame(model, model_frame_index)
        : scaled_template_taps(model, response_scale);
    auto signed_rotation = static_cast<int>(model.rotation_symbols);
    if (signed_rotation > static_cast<int>(kPnCoreSymbols / 2)) {
        signed_rotation -= static_cast<int>(kPnCoreSymbols);
    }
    if (options.interpolate_body_channel) {
        const auto next_response_scale = Complex{
            model.frame_response_scales[(model_frame_index + 1) * 2],
            model.frame_response_scales[(model_frame_index + 1) * 2 + 1],
        };
        const auto next_taps = uses_masked_frame_taps(model.scale_estimator)
            ? masked_frame_taps_for_model_frame(model, model_frame_index + 1)
            : scaled_template_taps(model, next_response_scale);
        const auto body_taps = midpoint_taps(taps, next_taps);
        restore_and_equalize_transition(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            next_taps,
            body_taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power,
            current_header,
            signed_rotation);
        return PnEqualizeResult{phase, next_phase};
    }
    if (uses_masked_frame_taps(model.scale_estimator)) {
        restore_and_equalize(
            interleaved_time_body,
            interleaved_next_header,
            interleaved_equalized_spectrum,
            phase,
            next_phase,
            taps,
            options.response_floor,
            options.noise_variance,
            -signed_rotation + options.response_window_offset_adjust,
            options.mmse_unbias,
            options.mmse_unbias_gain_floor,
            options.interleaved_channel_power,
            current_header,
            signed_rotation);
        return PnEqualizeResult{phase, next_phase};
    }
    restore_and_equalize_with_template_response(
        interleaved_time_body,
        interleaved_next_header,
        interleaved_equalized_spectrum,
        phase,
        next_phase,
        taps,
        model.template_response_fft,
        response_scale,
        options.response_floor,
        options.noise_variance,
        -signed_rotation + options.response_window_offset_adjust,
        options.mmse_unbias,
        options.mmse_unbias_gain_floor,
        options.interleaved_channel_power,
        current_header,
        signed_rotation);
    return PnEqualizeResult{phase, next_phase};
}

};

template<typename Function>
decltype(auto) dispatch_pn(PnMode mode, Function&& function) {
    switch (mode) {
    case PnMode::pn420: return function(PnChannel<PnMode::pn420>{});
    case PnMode::pn595: return function(PnChannel<PnMode::pn595>{});
    case PnMode::pn945: return function(PnChannel<PnMode::pn945>{});
    }
    throw std::invalid_argument("invalid PN mode");
}

}  // namespace

std::size_t pn_detect_phase_cf32(PnMode mode, std::span<const float> header) {
    return dispatch_pn(mode, [&](auto impl) { return impl.pn_detect_phase_cf32(header); });
}

float pn_known_phase_metric_ci8(PnMode mode, std::span<const std::int8_t> header, std::size_t phase) {
    return dispatch_pn(mode, [&](auto impl) { return impl.pn_known_phase_metric_ci8(header, phase); });
}

std::size_t pn_phase_for_frame(PnMode mode, std::size_t index) {
    return dispatch_pn(mode, [&](auto impl) { return impl.pn_phase_for_frame(index); });
}

PnScheduleAlignment fit_pn_phase_schedule(PnMode mode, std::span<const std::size_t> phases) {
    return dispatch_pn(mode, [&](auto impl) { return impl.fit_pn_phase_schedule(phases); });
}

PnResidualCfoResult estimate_pn_residual_cfo_cf32(
    PnMode mode, std::span<const float> symbols, std::size_t offset, PnResidualCfoOptions options) {
    return dispatch_pn(mode, [&](auto impl) { return impl.estimate_pn_residual_cfo_cf32(symbols, offset, options); });
}

PnEqualizeResult pn_equalize_c3780_frame_cf32(
    PnMode mode, std::span<const float> header, std::span<const float> body,
    std::span<const float> next_header, std::span<float> spectrum, PnEqualizeOptions options) {
    return dispatch_pn(mode, [&](auto impl) {
        return impl.pn_equalize_c3780_frame_cf32(header, body, next_header, spectrum, options);
    });
}

PnWidebandChannelModel build_pn_wideband_channel_model_cf32(
    PnMode mode, std::span<const float> headers, PnWidebandModelOptions options) {
    return dispatch_pn(mode, [&](auto impl) { return impl.build_pn_wideband_channel_model_cf32(headers, options); });
}

PnEqualizeResult pn_equalize_c3780_frame_wideband_cf32(
    PnMode mode, std::span<const float> header, std::span<const float> body,
    std::span<const float> next_header, std::span<float> spectrum,
    const PnWidebandChannelModel& model, PnEqualizeOptions options) {
    return dispatch_pn(mode, [&](auto impl) {
        return impl.pn_equalize_c3780_frame_wideband_cf32(header, body, next_header, spectrum, model, options);
    });
}

PnEqualizeResult pn_equalize_c3780_frame_wideband_cached_cf32(
    PnMode mode, std::span<const float> body, std::span<const float> next_header,
    std::span<float> spectrum, const PnWidebandChannelModel& model,
    std::size_t index, PnEqualizeOptions options) {
    return dispatch_pn(mode, [&](auto impl) {
        return impl.pn_equalize_c3780_frame_wideband_cached_cf32(body, next_header, spectrum, model, index, options);
    });
}

// Legacy entry points use the identical compile-time PN945 arithmetic path.
std::size_t pn945_detect_phase_cf32(std::span<const float> header) {
    return pn_detect_phase_cf32(PnMode::pn945, header);
}
float pn945_known_phase_metric_ci8(std::span<const std::int8_t> header, std::size_t phase) {
    return pn_known_phase_metric_ci8(PnMode::pn945, header, phase);
}
std::size_t pn945_phase_for_frame(std::size_t index) noexcept {
    return PnChannel<PnMode::pn945>::pn_phase_for_frame(index);
}
Pn945ScheduleAlignment fit_pn945_phase_schedule(std::span<const std::size_t> phases) {
    return fit_pn_phase_schedule(PnMode::pn945, phases);
}
Pn945ResidualCfoResult estimate_pn945_residual_cfo_cf32(
    std::span<const float> symbols, std::size_t offset, Pn945ResidualCfoOptions options) {
    return estimate_pn_residual_cfo_cf32(PnMode::pn945, symbols, offset, options);
}
Pn945EqualizeResult pn945_equalize_c3780_frame_cf32(
    std::span<const float> header, std::span<const float> body, std::span<const float> next_header,
    std::span<float> spectrum, Pn945EqualizeOptions options) {
    return pn_equalize_c3780_frame_cf32(PnMode::pn945, header, body, next_header, spectrum, options);
}
Pn945WidebandChannelModel build_pn945_wideband_channel_model_cf32(
    std::span<const float> headers, Pn945WidebandModelOptions options) {
    return build_pn_wideband_channel_model_cf32(PnMode::pn945, headers, options);
}
Pn945EqualizeResult pn945_equalize_c3780_frame_wideband_cf32(
    std::span<const float> header, std::span<const float> body, std::span<const float> next_header,
    std::span<float> spectrum, const Pn945WidebandChannelModel& model, Pn945EqualizeOptions options) {
    return pn_equalize_c3780_frame_wideband_cf32(PnMode::pn945, header, body, next_header, spectrum, model, options);
}
Pn945EqualizeResult pn945_equalize_c3780_frame_wideband_cached_cf32(
    std::span<const float> body, std::span<const float> next_header, std::span<float> spectrum,
    const Pn945WidebandChannelModel& model, std::size_t index, Pn945EqualizeOptions options) {
    return pn_equalize_c3780_frame_wideband_cached_cf32(PnMode::pn945, body, next_header, spectrum, model, index, options);
}

}  // namespace dtmb::core
