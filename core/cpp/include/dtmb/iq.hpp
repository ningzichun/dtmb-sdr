#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <span>
#include <string_view>
#include <vector>

namespace dtmb::core {

enum class IqFormat { cu8, ci8, ci16, cf32 };
IqFormat parse_iq_format(std::string_view name);
std::size_t iq_sample_bytes(IqFormat format);
// Interleaved little-endian I,Q. Float output uses CI8-compatible amplitude
// units (full scale 128), retaining every CI16 bit through the frontend.
void decode_iq(std::span<const std::uint8_t> bytes, IqFormat format,
               std::span<float> output);
std::size_t read_iq(std::istream& input, IqFormat format, std::span<float> output);

// Bounded polyphase SRRC rate converter. The phase bank has fixed size even
// for relatively prime rates; the time accumulator keeps the exact ratio.
class IqResampler {
public:
    IqResampler(std::uint32_t input_rate, std::uint32_t output_rate,
                std::size_t span = 8, float roll_off = 0.05F);
    void process(std::span<const float> input, std::vector<float>& output);
    void finish(std::vector<float>& output);
    std::uint64_t input_samples() const { return received_; }
    std::uint64_t output_samples() const { return produced_; }
private:
    void emit(bool final, std::vector<float>& output);
    std::uint32_t input_rate_, output_rate_;
    std::size_t radius_;
    std::vector<float> filters_, history_;
    std::uint64_t base_ = 0, received_ = 0, produced_ = 0;
    bool finished_ = false;
};
}
