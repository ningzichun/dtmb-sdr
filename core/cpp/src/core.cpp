#include "dtmb/core.hpp"

#include "dtmb/core_c.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dtmb::core {
namespace {

constexpr std::uint32_t kVersionMajor = 0;
constexpr std::uint32_t kVersionMinor = 2;
constexpr std::uint32_t kVersionPatch = 0;
constexpr std::uint32_t kAbiVersionMajor = 0;
constexpr std::uint32_t kAbiVersionMinor = 1;

struct PartialStats {
    std::size_t sample_count = 0;
    std::uint64_t sum_i2q2 = 0;
    std::size_t clip_count_i = 0;
    std::size_t clip_count_q = 0;
};

[[nodiscard]] std::size_t choose_worker_count(
    std::size_t sample_count,
    const Ci8PowerStatsOptions& options) {
    if (sample_count == 0) {
        return 0;
    }

    const auto min_parallel_samples = std::max<std::size_t>(options.min_parallel_samples, 1);
    if (sample_count < min_parallel_samples) {
        return 1;
    }

    auto available_workers = options.requested_workers;
    if (available_workers == 0) {
        available_workers = std::thread::hardware_concurrency();
    }
    if (available_workers == 0) {
        available_workers = 1;
    }

    return std::clamp<std::size_t>(available_workers, 1, sample_count);
}

[[nodiscard]] std::size_t choose_qam_worker_count(
    std::size_t symbol_count,
    const QamSoftDemapOptions& options) {
    if (symbol_count == 0) {
        return 0;
    }

    const auto min_parallel_symbols = std::max<std::size_t>(options.min_parallel_symbols, 1);
    if (symbol_count < min_parallel_symbols) {
        return 1;
    }

    auto available_workers = options.requested_workers;
    if (available_workers == 0) {
        available_workers = std::thread::hardware_concurrency();
    }
    if (available_workers == 0) {
        available_workers = 1;
    }

    return std::clamp<std::size_t>(available_workers, 1, symbol_count);
}

[[nodiscard]] PartialStats reduce_ci8_range(
    std::span<const std::int8_t> interleaved_iq,
    std::size_t first_sample,
    std::size_t last_sample) {
    PartialStats stats;
    stats.sample_count = last_sample - first_sample;

    for (std::size_t sample = first_sample; sample < last_sample; ++sample) {
        const auto i = interleaved_iq[sample * 2];
        const auto q = interleaved_iq[sample * 2 + 1];
        const auto i_wide = static_cast<int>(i);
        const auto q_wide = static_cast<int>(q);

        stats.sum_i2q2 += static_cast<std::uint64_t>(i_wide * i_wide + q_wide * q_wide);
        stats.clip_count_i += (i == INT8_MIN || i == INT8_MAX) ? 1U : 0U;
        stats.clip_count_q += (q == INT8_MIN || q == INT8_MAX) ? 1U : 0U;
    }

    return stats;
}

[[nodiscard]] Ci8PowerStats finish_stats(PartialStats partial, std::size_t worker_count) {
    Ci8PowerStats stats;
    stats.sample_count = partial.sample_count;
    stats.clip_count_i = partial.clip_count_i;
    stats.clip_count_q = partial.clip_count_q;
    stats.worker_count = worker_count;

    if (partial.sample_count == 0) {
        return stats;
    }

    stats.mean_i2q2 = static_cast<float>(
        static_cast<double>(partial.sum_i2q2) / static_cast<double>(partial.sample_count));
    stats.rms_iq = std::sqrt(stats.mean_i2q2);
    return stats;
}

[[nodiscard]] DtmbCoreCi8PowerStats to_c_stats(const Ci8PowerStats& stats) noexcept {
    return DtmbCoreCi8PowerStats{
        stats.sample_count,
        stats.mean_i2q2,
        stats.rms_iq,
        stats.clip_count_i,
        stats.clip_count_q,
        stats.worker_count,
    };
}

constexpr std::array<float, 8> kQam64Levels{-7.0F, -5.0F, -3.0F, -1.0F, 1.0F, 3.0F, 5.0F, 7.0F};
constexpr std::array<std::array<std::uint8_t, 3>, 8> kQam64AxisBits{{
    {{0, 0, 0}},
    {{1, 0, 0}},
    {{1, 1, 0}},
    {{0, 1, 0}},
    {{0, 1, 1}},
    {{1, 1, 1}},
    {{1, 0, 1}},
    {{0, 0, 1}},
}};

[[nodiscard]] float axis_llr(float value, std::size_t bit_index, float noise_variance) noexcept {
    auto min_zero = std::numeric_limits<float>::infinity();
    auto min_one = std::numeric_limits<float>::infinity();

    for (std::size_t index = 0; index < kQam64Levels.size(); ++index) {
        const auto delta = value - kQam64Levels[index];
        const auto distance = delta * delta;
        if (kQam64AxisBits[index][bit_index] == 0) {
            min_zero = std::min(min_zero, distance);
        } else {
            min_one = std::min(min_one, distance);
        }
    }

    return (min_one - min_zero) / noise_variance;
}

[[nodiscard]] double log_sum_exp4(const std::array<double, 4>& values) noexcept {
    const auto max_value = *std::max_element(values.begin(), values.end());
    auto scaled_sum = 0.0;
    for (const auto value : values) {
        scaled_sum += std::exp(value - max_value);
    }
    return max_value + std::log(scaled_sum);
}

[[nodiscard]] float axis_llr_exact(
    float value,
    std::size_t bit_index,
    float noise_variance) noexcept {
    std::array<double, 4> zero_metrics{};
    std::array<double, 4> one_metrics{};
    std::size_t zero_count = 0;
    std::size_t one_count = 0;

    for (std::size_t index = 0; index < kQam64Levels.size(); ++index) {
        const auto delta = static_cast<double>(value) - static_cast<double>(kQam64Levels[index]);
        const auto metric = -(delta * delta) / static_cast<double>(noise_variance);
        if (kQam64AxisBits[index][bit_index] == 0) {
            zero_metrics[zero_count++] = metric;
        } else {
            one_metrics[one_count++] = metric;
        }
    }

    return static_cast<float>(log_sum_exp4(zero_metrics) - log_sum_exp4(one_metrics));
}

void qam64_soft_demodulate_range(
    std::span<const float> interleaved_symbols,
    std::span<float> output_llr,
    float noise_variance,
    QamSoftDemapMethod method,
    std::size_t first_symbol,
    std::size_t last_symbol) {
    for (std::size_t symbol = first_symbol; symbol < last_symbol; ++symbol) {
        const auto real = interleaved_symbols[symbol * 2];
        const auto imag = interleaved_symbols[symbol * 2 + 1];
        auto* out = output_llr.data() + symbol * 6;
        for (std::size_t bit = 0; bit < 3; ++bit) {
            if (method == QamSoftDemapMethod::log_sum_exp) {
                out[bit] = axis_llr_exact(real, bit, noise_variance);
                out[3 + bit] = axis_llr_exact(imag, bit, noise_variance);
            } else {
                out[bit] = axis_llr(real, bit, noise_variance);
                out[3 + bit] = axis_llr(imag, bit, noise_variance);
            }
        }
    }
}

}  // namespace

std::size_t SymbolInterleaverSpec::max_branch_delay() const noexcept {
    return branch_count == 0 ? 0 : (branch_count - 1) * delay_step;
}

std::size_t SymbolInterleaverSpec::full_stream_latency_symbols() const noexcept {
    return max_branch_delay() * branch_count;
}

SymbolInterleaverSpec symbol_interleaver_spec(SymbolInterleaverMode mode) noexcept {
    switch (mode) {
    case SymbolInterleaverMode::mode1:
        return SymbolInterleaverSpec{52, 240};
    case SymbolInterleaverMode::mode2:
        return SymbolInterleaverSpec{52, 720};
    }
    return SymbolInterleaverSpec{52, 240};
}

SymbolDeinterleaverCf32::SymbolDeinterleaverCf32(
    SymbolInterleaverMode mode,
    std::size_t phase,
    float fill_real,
    float fill_imag)
    : spec_(symbol_interleaver_spec(mode)),
      phase_(phase),
      fill_real_(fill_real),
      fill_imag_(fill_imag) {
    if (phase_ >= spec_.branch_count) {
        throw std::invalid_argument("symbol deinterleaver phase must be 0..51");
    }

    branch_offsets_.resize(spec_.branch_count);
    branch_lengths_.resize(spec_.branch_count);
    branch_positions_.resize(spec_.branch_count);

    std::size_t offset = 0;
    for (std::size_t branch = 0; branch < spec_.branch_count; ++branch) {
        const auto delay = (spec_.branch_count - 1 - branch) * spec_.delay_step;
        branch_offsets_[branch] = offset;
        branch_lengths_[branch] = delay;
        offset += delay;
    }
    delay_line_.resize(offset * 2);
    reset();
}

void SymbolDeinterleaverCf32::reset() {
    processed_symbols_ = 0;
    std::fill(branch_positions_.begin(), branch_positions_.end(), 0);
    for (std::size_t symbol = 0; symbol < delay_line_.size() / 2; ++symbol) {
        delay_line_[symbol * 2] = fill_real_;
        delay_line_[symbol * 2 + 1] = fill_imag_;
    }
}

void SymbolDeinterleaverCf32::process(
    std::span<const float> interleaved_symbols,
    std::span<float> output_symbols) {
    if ((interleaved_symbols.size() % 2) != 0) {
        throw std::invalid_argument("CF32 input must contain interleaved real/imag pairs");
    }
    if (output_symbols.size() < interleaved_symbols.size()) {
        throw std::invalid_argument("output CF32 span is too small");
    }

    const auto symbol_count = interleaved_symbols.size() / 2;
    for (std::size_t local = 0; local < symbol_count; ++local) {
        const auto branch = ((processed_symbols_ + local) % spec_.branch_count + phase_)
            % spec_.branch_count;
        const auto branch_length = branch_lengths_[branch];
        const auto input_real = interleaved_symbols[local * 2];
        const auto input_imag = interleaved_symbols[local * 2 + 1];

        if (branch_length == 0) {
            output_symbols[local * 2] = input_real;
            output_symbols[local * 2 + 1] = input_imag;
            continue;
        }

        const auto delay_symbol = branch_offsets_[branch] + branch_positions_[branch];
        output_symbols[local * 2] = delay_line_[delay_symbol * 2];
        output_symbols[local * 2 + 1] = delay_line_[delay_symbol * 2 + 1];
        delay_line_[delay_symbol * 2] = input_real;
        delay_line_[delay_symbol * 2 + 1] = input_imag;
        branch_positions_[branch] = (branch_positions_[branch] + 1) % branch_length;
    }
    processed_symbols_ += symbol_count;
}

SymbolInterleaverSpec SymbolDeinterleaverCf32::spec() const noexcept {
    return spec_;
}

std::size_t SymbolDeinterleaverCf32::phase() const noexcept {
    return phase_;
}

std::size_t SymbolDeinterleaverCf32::processed_symbols() const noexcept {
    return processed_symbols_;
}

std::size_t SymbolDeinterleaverCf32::latency_symbols() const noexcept {
    return spec_.full_stream_latency_symbols();
}

Version version() noexcept {
    return Version{kVersionMajor, kVersionMinor, kVersionPatch, kAbiVersionMajor, kAbiVersionMinor};
}

const char* build_info() noexcept {
    return "dtmb-core-cpp 0.2.0";
}

Ci8PowerStats ci8_power_stats(
    std::span<const std::int8_t> interleaved_iq,
    Ci8PowerStatsOptions options) {
    if ((interleaved_iq.size() % 2) != 0) {
        throw std::invalid_argument("CI8 byte length must be even");
    }

    const auto sample_count = interleaved_iq.size() / 2;
    const auto worker_count = choose_worker_count(sample_count, options);
    if (worker_count == 0) {
        return {};
    }
    if (worker_count == 1) {
        return finish_stats(reduce_ci8_range(interleaved_iq, 0, sample_count), worker_count);
    }

    std::vector<PartialStats> partials(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const auto first = (sample_count * worker) / worker_count;
        const auto last = (sample_count * (worker + 1)) / worker_count;
        workers.emplace_back([&, worker, first, last] {
            partials[worker] = reduce_ci8_range(interleaved_iq, first, last);
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    PartialStats merged;
    for (const auto& partial : partials) {
        merged.sample_count += partial.sample_count;
        merged.sum_i2q2 += partial.sum_i2q2;
        merged.clip_count_i += partial.clip_count_i;
        merged.clip_count_q += partial.clip_count_q;
    }

    return finish_stats(merged, worker_count);
}

void qam64_soft_demodulate_cf32(
    std::span<const float> interleaved_symbols,
    std::span<float> output_llr,
    QamSoftDemapOptions options) {
    if ((interleaved_symbols.size() % 2) != 0) {
        throw std::invalid_argument("CF32 input must contain interleaved real/imag pairs");
    }
    if (options.noise_variance <= 0.0F || !std::isfinite(options.noise_variance)) {
        throw std::invalid_argument("noise variance must be positive and finite");
    }

    const auto symbol_count = interleaved_symbols.size() / 2;
    if (output_llr.size() < symbol_count * 6) {
        throw std::invalid_argument("output LLR span is too small");
    }

    const auto worker_count = choose_qam_worker_count(symbol_count, options);
    if (worker_count == 0) {
        return;
    }
    if (worker_count == 1) {
        qam64_soft_demodulate_range(
            interleaved_symbols,
            output_llr,
            options.noise_variance,
            options.method,
            0,
            symbol_count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const auto first = (symbol_count * worker) / worker_count;
        const auto last = (symbol_count * (worker + 1)) / worker_count;
        workers.emplace_back([&, first, last] {
            qam64_soft_demodulate_range(
                interleaved_symbols,
                output_llr,
                options.noise_variance,
                options.method,
                first,
                last);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

}  // namespace dtmb::core

extern "C" {

uint32_t dtmb_core_version_major(void) {
    return dtmb::core::version().major;
}

uint32_t dtmb_core_version_minor(void) {
    return dtmb::core::version().minor;
}

uint32_t dtmb_core_version_patch(void) {
    return dtmb::core::version().patch;
}

uint32_t dtmb_core_abi_version_major(void) {
    return dtmb::core::version().abi_major;
}

uint32_t dtmb_core_abi_version_minor(void) {
    return dtmb::core::version().abi_minor;
}

DtmbCoreVersion dtmb_core_version(void) {
    const auto cpp_version = dtmb::core::version();
    return DtmbCoreVersion{
        cpp_version.major,
        cpp_version.minor,
        cpp_version.patch,
        cpp_version.abi_major,
        cpp_version.abi_minor,
    };
}

const char* dtmb_core_build_info(void) {
    return dtmb::core::build_info();
}

int32_t dtmb_core_ci8_power_stats(
    const int8_t* interleaved_iq,
    size_t len_bytes,
    DtmbCoreCi8PowerStats* out_stats) {
    return dtmb_core_ci8_power_stats_parallel(interleaved_iq, len_bytes, 0, 1U << 20U, out_stats);
}

int32_t dtmb_core_ci8_power_stats_parallel(
    const int8_t* interleaved_iq,
    size_t len_bytes,
    size_t requested_workers,
    size_t min_parallel_samples,
    DtmbCoreCi8PowerStats* out_stats) {
    if (out_stats == nullptr) {
        return DTMB_CORE_STATUS_INVALID_ARGUMENT;
    }
    if ((len_bytes % 2) != 0) {
        return DTMB_CORE_STATUS_INVALID_LENGTH;
    }
    if (interleaved_iq == nullptr && len_bytes != 0) {
        return DTMB_CORE_STATUS_INVALID_ARGUMENT;
    }

    try {
        const auto input = interleaved_iq == nullptr
            ? std::span<const std::int8_t>{}
            : std::span<const std::int8_t>{interleaved_iq, len_bytes};
        const auto stats = dtmb::core::ci8_power_stats(
            input,
            dtmb::core::Ci8PowerStatsOptions{requested_workers, min_parallel_samples});
        *out_stats = dtmb::core::to_c_stats(stats);
        return DTMB_CORE_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return DTMB_CORE_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
        return DTMB_CORE_STATUS_INVALID_LENGTH;
    } catch (const std::exception&) {
        return DTMB_CORE_STATUS_INTERNAL_ERROR;
    } catch (...) {
        return DTMB_CORE_STATUS_INTERNAL_ERROR;
    }
}

int32_t dtmb_core_qam64_soft_demodulate_cf32(
    const float* interleaved_symbols,
    size_t symbol_count,
    float noise_variance,
    float* output_llr) {
    return dtmb_core_qam64_soft_demodulate_cf32_parallel(
        interleaved_symbols,
        symbol_count,
        noise_variance,
        0,
        1U << 14U,
        output_llr);
}

int32_t dtmb_core_qam64_soft_demodulate_cf32_parallel(
    const float* interleaved_symbols,
    size_t symbol_count,
    float noise_variance,
    size_t requested_workers,
    size_t min_parallel_symbols,
    float* output_llr) {
    if (output_llr == nullptr) {
        return DTMB_CORE_STATUS_INVALID_ARGUMENT;
    }
    if (interleaved_symbols == nullptr && symbol_count != 0) {
        return DTMB_CORE_STATUS_INVALID_ARGUMENT;
    }

    try {
        const auto input = interleaved_symbols == nullptr
            ? std::span<const float>{}
            : std::span<const float>{interleaved_symbols, symbol_count * 2};
        const auto output = std::span<float>{output_llr, symbol_count * 6};
        dtmb::core::qam64_soft_demodulate_cf32(
            input,
            output,
            dtmb::core::QamSoftDemapOptions{
                noise_variance,
                requested_workers,
                min_parallel_symbols,
            });
        return DTMB_CORE_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return DTMB_CORE_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
        return DTMB_CORE_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception&) {
        return DTMB_CORE_STATUS_INTERNAL_ERROR;
    } catch (...) {
        return DTMB_CORE_STATUS_INTERNAL_ERROR;
    }
}

}  // extern "C"
