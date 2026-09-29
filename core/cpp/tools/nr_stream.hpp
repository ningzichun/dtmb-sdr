#pragma once

#include "dtmb/nr.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>

namespace dtmb::tools {

// Stream orchestration only. The NR likelihoods and bit interleaver live in
// the portable core; both demapper command names use this same path.
inline int run_nr_demapper(
    std::istream& input, std::ostream& output,
    core::QamSoftDemapOptions options, std::size_t chunk_symbols,
    std::optional<core::SymbolInterleaverMode> mode = std::nullopt,
    std::size_t phase = 0, bool keep_latency = false,
    std::istream* csi = nullptr, bool binary_mse = false, bool inverse_mse = false) {
    if (chunk_symbols == 0 || !std::isfinite(options.noise_variance) || options.noise_variance <= 0)
        throw std::invalid_argument("NR chunk size and finite noise variance must be positive");
    // Confidence is defined on complete source signal frames. NR blocks always
    // begin at the first data symbol; eight physical symbols carry eight bits.
    const std::size_t quantum = binary_mse || inverse_mse ? 3744 : 8;
    chunk_symbols = std::max(quantum, chunk_symbols / quantum * quantum);
    std::optional<core::BitDeinterleaverF32> deinterleaver;
    if (mode) deinterleaver.emplace(*mode, phase);
    const auto latency = deinterleaver && !keep_latency ? deinterleaver->latency_bits() : 0;
    auto discard = latency;
    std::vector<float> symbols(chunk_symbols * 2), physical_llr(chunk_symbols * 2);
    std::vector<float> nr_llr(chunk_symbols), decoded(chunk_symbols), weights(chunk_symbols);
    std::size_t received = 0, written = 0, weighted = 0, scored = 0;
    const auto capacity = static_cast<std::streamsize>(symbols.size() * sizeof(float));
    while (true) {
        input.read(reinterpret_cast<char*>(symbols.data()), capacity);
        const auto bytes = input.gcount();
        if (input.bad()) throw std::runtime_error("failed to read NR symbol stream");
        if (bytes == 0) break;
        if (bytes % (2 * sizeof(float)) != 0)
            throw std::runtime_error("CF32 input byte count is not a whole number of symbols");
        const auto count = static_cast<std::size_t>(bytes) / (2 * sizeof(float));
        if (count % 8 != 0) throw std::runtime_error("NR input ends inside an eight-symbol block");
        if ((binary_mse || inverse_mse) && count % 3744 != 0)
            throw std::runtime_error("frame confidence input ends inside a C3780 data frame");
        const auto source = std::span<const float>(symbols.data(), count * 2);
        if (!std::all_of(source.begin(), source.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("NR input symbols must be finite");
        std::fill_n(weights.begin(), count, 1.0F);
        if (csi) {
            const auto weight_bytes = static_cast<std::streamsize>(count * sizeof(float));
            csi->read(reinterpret_cast<char*>(weights.data()), weight_bytes);
            if (csi->gcount() != weight_bytes)
                throw std::runtime_error("CSI weight count does not match input symbol count");
        }
        if (binary_mse || inverse_mse) {
            for (std::size_t first = 0; first < count; first += 3744) {
                const auto mse = core::c3780_qam_frame_mse_cf32(source.subspan(first * 2, 3744 * 2), core::QamMode::qam4);
                const auto weight = inverse_mse
                    ? static_cast<float>(std::clamp(1.0 / std::max(mse, 1.0e-12), 0.05, 4.0))
                    : mse > 1.0 ? 0.05F : 1.0F;
                for (std::size_t n = first; n < first + 3744; ++n) weights[n] *= weight;
                ++scored;
            }
        }
        auto physical = std::span<float>(physical_llr.data(), count * 2);
        core::qam_soft_demodulate_cf32(source, physical, core::QamMode::qam4, options);
        for (std::size_t n = 0; n < count; ++n) {
            if (!std::isfinite(weights[n]) || weights[n] < 0)
                throw std::runtime_error("CSI weights must be finite and non-negative");
            physical[2 * n] *= weights[n];
            physical[2 * n + 1] *= weights[n];
        }
        weighted += csi || binary_mse || inverse_mse ? count : 0;
        // NR's 8->16 inner code means eight 4QAM symbols carry one byte;
        // decode_block consumes the 16 axis LLRs of a block and emits 8 bits.
        auto nr = std::span<float>(nr_llr.data(), count);
        core::nr_soft_decode(physical, nr, options);
        auto result = nr;
        if (deinterleaver) {
            result = std::span<float>(decoded.data(), count);
            deinterleaver->process(nr, result);
        }
        const auto skip = std::min(discard, count);
        discard -= skip;
        result = result.subspan(skip);
        output.write(reinterpret_cast<const char*>(result.data()),
                     static_cast<std::streamsize>(result.size_bytes()));
        if (!output) throw std::runtime_error("failed to write NR LLR stream");
        received += count;
        written += result.size();
        if (bytes < capacity) break;
    }
    if (csi) {
        char trailing;
        if (csi->read(&trailing, 1)) throw std::runtime_error("CSI weight input has trailing values");
    }
    std::cerr << "qam=4qam-nr\nllrs_per_symbol=1\n"
              << "input_symbols=" << received << '\n'
              << "nr_blocks=" << received / 8 << '\n'
              << "discarded_latency_bits=" << latency - discard << '\n'
              << "discarded_latency_symbols=" << latency - discard << '\n'
              << "output_symbols=" << written << '\n'
              << "written_output_symbols=" << written << '\n'
              << "output_llrs=" << written << '\n'
              << "csi_weighted_symbols=" << weighted << '\n'
              << "source_frame_confidence=" << (inverse_mse ? "inverse-mse" : binary_mse ? "binary-mse" : "off") << '\n'
              << "source_frame_mse_scored_frames=" << scored << '\n';
    return 0;
}
}  // namespace dtmb::tools
