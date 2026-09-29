#include "dtmb/nr.hpp"
#include "dtmb/worker.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <thread>

namespace dtmb::core {

std::uint16_t nr_encode(std::uint8_t input) noexcept {
    std::array<unsigned, 8> x{};
    for (unsigned i = 0; i < 8; ++i) x[i] = (input >> (7 - i)) & 1U;
    unsigned parity = 0;
    unsigned last = std::popcount(input) & 1U;
    for (unsigned i = 0; i < 7; ++i) {
        const auto at = [&](unsigned j) { return x[(i + j) % 7]; };
        const auto y = x[7] ^ at(6) ^ at(0) ^ at(1) ^ at(3)
            ^ ((at(0) ^ at(4)) & (at(1) ^ at(2) ^ at(3) ^ at(5)))
            ^ ((at(1) ^ at(2)) & (at(3) ^ at(5)));
        parity |= y << (7 - i);
        last ^= y;
    }
    return static_cast<std::uint16_t>((input << 8) | parity | last);
}

namespace {
const auto kParity = [] {
    std::array<std::uint8_t, 256> table{};
    for (unsigned i = 0; i < 256; ++i) table[i] = nr_encode(static_cast<std::uint8_t>(i)) & 255;
    return table;
}();

void decode_block(const float* input, float* output, QamSoftDemapMethod method) {
    // A candidate's negative log likelihood, up to a common constant, is
    // the sum of input LLRs at its one bits. Two byte tables avoid rescoring
    // sixteen bits for every candidate; doubles keep finite inputs stable.
    std::array<double, 256> first{}, second{}, cost{};
    for (unsigned value = 1; value < 256; ++value) {
        const auto bit = std::countr_zero(value);
        const auto previous = value & (value - 1);
        first[value] = first[previous] + input[7 - bit];
        second[value] = second[previous] + input[15 - bit];
    }
    for (unsigned value = 0; value < 256; ++value)
        cost[value] = first[value] + second[kParity[value]];
    for (unsigned bit = 0; bit < 8; ++bit) {
        std::array<double, 2> minimum{std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()};
        for (unsigned value = 0; value < 256; ++value) {
            const auto label = (value >> (7 - bit)) & 1;
            minimum[label] = std::min(minimum[label], cost[value]);
        }
        auto llr = minimum[1] - minimum[0];
        if (method == QamSoftDemapMethod::log_sum_exp) {
            std::array<double, 2> sums{};
            for (unsigned value = 0; value < 256; ++value) {
                const auto label = (value >> (7 - bit)) & 1;
                sums[label] += std::exp(minimum[label] - cost[value]);
            }
            llr += std::log(sums[0]) - std::log(sums[1]);
        }
        // Only representational saturation, not an experimental LLR clip.
        const auto limit = static_cast<double>(std::numeric_limits<float>::max());
        output[bit] = static_cast<float>(std::clamp(llr, -limit, limit));
    }
}
}  // namespace

void nr_soft_decode(std::span<const float> input, std::span<float> output,
                    QamSoftDemapOptions options) {
    if (input.size() % 16 != 0 || output.size() != input.size() / 2)
        throw std::invalid_argument("NR decoding requires complete 16-to-8 LLR blocks");
    if (!std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("NR input LLRs must be finite");
    const auto blocks = input.size() / 16;
    if (blocks == 0) return;
    auto count = options.requested_workers ? options.requested_workers
                                          : std::max(1U, std::thread::hardware_concurrency());
    count = std::min(count, blocks);
    if (blocks * 8 < options.min_parallel_symbols) count = 1;
    const auto run = [&](std::size_t first, std::size_t last) {
        for (auto block = first; block < last; ++block)
            decode_block(input.data() + block * 16, output.data() + block * 8, options.method);
    };
    if (count == 1) { run(0, blocks); return; }
    std::vector<WorkerThread> workers;
    for (std::size_t i = 0; i < count; ++i)
        workers.emplace_back([&, i] { run(blocks * i / count, blocks * (i + 1) / count); });
    for (auto& worker : workers) worker.join();
}

BitDeinterleaverF32::BitDeinterleaverF32(SymbolInterleaverMode mode, std::size_t phase)
    : spec_(symbol_interleaver_spec(mode)), branch_(phase) {
    if (phase >= spec_.branch_count) throw std::invalid_argument("NR bit interleaver phase must be 0..51");
    std::size_t size = 0;
    for (std::size_t branch = 0; branch < spec_.branch_count; ++branch) {
        offsets_[branch] = size;
        lengths_[branch] = (spec_.branch_count - 1 - branch) * spec_.delay_step;
        size += lengths_[branch];
    }
    delays_.resize(size, 0.0F);
}

void BitDeinterleaverF32::process(std::span<const float> input, std::span<float> output) {
    if (output.size() != input.size()) throw std::invalid_argument("NR bit interleaver span mismatch");
    for (std::size_t n = 0; n < input.size(); ++n) {
        if (lengths_[branch_] == 0) output[n] = input[n];
        else {
            auto& delayed = delays_[offsets_[branch_] + positions_[branch_]];
            const auto value = input[n];
            output[n] = delayed;
            delayed = value;
            positions_[branch_] = (positions_[branch_] + 1) % lengths_[branch_];
        }
        branch_ = (branch_ + 1) % spec_.branch_count;
    }
}

std::size_t BitDeinterleaverF32::latency_bits() const noexcept {
    return spec_.full_stream_latency_symbols();
}

}  // namespace dtmb::core
