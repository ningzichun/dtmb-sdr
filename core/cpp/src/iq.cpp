#include "dtmb/iq.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace dtmb::core {
IqFormat parse_iq_format(std::string_view name) {
    if (name == "cu8") return IqFormat::cu8;
    if (name == "ci8" || name == "cs8") return IqFormat::ci8;
    if (name == "ci16" || name == "sc16" || name == "cs16") return IqFormat::ci16;
    if (name == "cf32" || name == "fc32") return IqFormat::cf32;
    throw std::invalid_argument("IQ format must be cu8, ci8/cs8, ci16/sc16, or cf32");
}
std::size_t iq_sample_bytes(IqFormat f) {
    return f == IqFormat::cf32 ? 8 : f == IqFormat::ci16 ? 4 : 2;
}
void decode_iq(std::span<const std::uint8_t> bytes, IqFormat f, std::span<float> out) {
    const auto width = iq_sample_bytes(f) / 2;
    if (bytes.size() % (width * 2) || out.size() != bytes.size() / width)
        throw std::invalid_argument("incomplete interleaved IQ sample");
    for (std::size_t n = 0; n < out.size(); ++n) {
        const auto* p = bytes.data() + n * width;
        if (f == IqFormat::cu8) out[n] = static_cast<float>(p[0]) - 128.0F;
        else if (f == IqFormat::ci8) out[n] = p[0] < 128 ? p[0] : static_cast<int>(p[0]) - 256;
        else if (f == IqFormat::ci16) {
            const int value = p[0] | (static_cast<int>(p[1]) << 8);
            out[n] = static_cast<float>(value < 32768 ? value : value - 65536) / 256.0F;
        } else {
            const std::uint32_t bits = p[0] | (std::uint32_t(p[1]) << 8)
                | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
            out[n] = std::bit_cast<float>(bits) * 128.0F;
            if (!std::isfinite(out[n])) throw std::invalid_argument("IQ contains NaN or infinity");
        }
    }
}
std::size_t read_iq(std::istream& input, IqFormat f, std::span<float> out) {
    if (out.size() % 2) throw std::invalid_argument("IQ destination must contain pairs");
    std::vector<std::uint8_t> bytes(out.size() / 2 * iq_sample_bytes(f));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    const auto count = static_cast<std::size_t>(input.gcount());
    if (input.bad()) throw std::runtime_error("failed to read IQ stream");
    if (count % iq_sample_bytes(f)) throw std::runtime_error("truncated IQ: incomplete I/Q pair");
    const auto values = count / iq_sample_bytes(f) * 2;
    decode_iq(std::span(bytes).first(count), f, out.first(values));
    return values;
}
namespace {
constexpr std::size_t phases = 1024;
double srrc(double t, double a) {
    const auto pi = std::numbers::pi_v<double>;
    if (std::abs(t) < 1e-12) return 1 - a + 4 * a / pi;
    if (std::abs(std::abs(4*a*t)-1) < 1e-9)
        return a/std::sqrt(2.0)*((1+2/pi)*std::sin(pi/(4*a))+(1-2/pi)*std::cos(pi/(4*a)));
    return (std::sin(pi*t*(1-a))+4*a*t*std::cos(pi*t*(1+a))) / (pi*t*(1-16*a*a*t*t));
}
}
IqResampler::IqResampler(std::uint32_t in, std::uint32_t out, std::size_t span, float roll)
    : input_rate_(in), output_rate_(out) {
    if (!in || !out || !span || span > 64 || !(roll > 0 && roll <= 1))
        throw std::invalid_argument("invalid resampler rate, span, or roll-off");
    const double ratio = std::min(1.0, double(out) / in);
    radius_ = static_cast<std::size_t>(std::ceil(span / ratio));
    if (radius_ > 8192) throw std::invalid_argument("sample-rate ratio exceeds bounded filter support");
    const auto width = 2 * radius_ + 1;
    filters_.resize((phases + 1) * width);
    for (std::size_t p = 0; p <= phases; ++p) {
        double sum = 0;
        for (std::size_t k = 0; k < width; ++k) {
            const auto t = (double(k) - radius_ - double(p)/phases) * ratio;
            const auto value = srrc(t, roll);
            filters_[p*width+k] = static_cast<float>(value);
            sum += value;
        }
        for (std::size_t k = 0; k < width; ++k) filters_[p*width+k] /= static_cast<float>(sum);
    }
}
void IqResampler::process(std::span<const float> input, std::vector<float>& out) {
    if (finished_ || input.size()%2) throw std::invalid_argument("invalid resampler input/state");
    history_.insert(history_.end(), input.begin(), input.end());
    received_ += input.size()/2;
    emit(false, out);
}
void IqResampler::finish(std::vector<float>& out) {
    if (!finished_) { emit(true, out); history_.clear(); finished_ = true; }
}
void IqResampler::emit(bool final, std::vector<float>& out) {
    const auto width = 2*radius_+1;
    while (true) {
        if (produced_ > std::numeric_limits<std::uint64_t>::max()/input_rate_)
            throw std::overflow_error("resampler time overflow");
        const auto time = produced_*input_rate_;
        const auto center = time/output_rate_;
        if (final ? center >= received_ : center+radius_ >= received_) break;
        const double phase = double(time%output_rate_)*phases/output_rate_;
        const auto p = static_cast<std::size_t>(phase);
        const auto blend = static_cast<float>(phase-p);
        float i = 0, q = 0;
        for (std::size_t k = 0; k < width; ++k) {
            const auto source = static_cast<std::int64_t>(center)+static_cast<std::int64_t>(k)-static_cast<std::int64_t>(radius_);
            if (source < 0 || static_cast<std::uint64_t>(source) >= received_) continue;
            const auto offset = static_cast<std::size_t>(source-base_)*2;
            const float h0 = filters_[p*width+k], h1 = filters_[(p+1)*width+k];
            const float h = h0+(h1-h0)*blend;
            i += history_[offset]*h; q += history_[offset+1]*h;
        }
        out.push_back(i); out.push_back(q); ++produced_;
    }
    const auto next = produced_*input_rate_/output_rate_;
    const auto keep = std::min(received_, next > radius_ ? next-radius_ : 0);
    history_.erase(history_.begin(), history_.begin()+static_cast<std::ptrdiff_t>((keep-base_)*2));
    base_ = keep;
}
}
