#pragma once

#include "dtmb/core.hpp"

namespace dtmb::core {

// GB 20600-2006 4.4.3.6 equations (4-1)..(4-8). Input is x0..x7
// (x0 is the MSB); output is x0..x7,y0..y7, transmitted in that order.
[[nodiscard]] std::uint16_t nr_encode(std::uint8_t input) noexcept;

// Joint likelihood over all 256 NR codewords. Sixteen physical-bit LLRs
// produce eight x-bit LLRs; positive means zero. Complete blocks only.
void nr_soft_decode(std::span<const float> input_llr,
                    std::span<float> output_llr,
                    QamSoftDemapOptions options = {});

// 4.4.3.6 applies the 52-branch interleaver to individual FEC bits before
// NR encoding. At the receiver it follows NR soft decoding, not QAM symbols.
class BitDeinterleaverF32 {
public:
    explicit BitDeinterleaverF32(SymbolInterleaverMode mode, std::size_t phase = 0);
    void process(std::span<const float> input, std::span<float> output);
    [[nodiscard]] std::size_t latency_bits() const noexcept;
private:
    SymbolInterleaverSpec spec_;
    std::size_t branch_;
    std::array<std::size_t, 52> offsets_{}, lengths_{}, positions_{};
    std::vector<float> delays_;
};

}  // namespace dtmb::core
