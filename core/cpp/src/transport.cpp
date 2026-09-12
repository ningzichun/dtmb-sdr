#include "dtmb/transport.hpp"

#include <algorithm>

namespace dtmb::core {

bool is_dtmb_sip_packet(std::span<const std::uint8_t> packet) noexcept {
    if (packet.size() != 188 || packet[0] != 0x47 || packet[1] != 0x40
        || packet[2] != 0x15 || packet[3] != 0x10) {
        return false;
    }
    const auto header_mode = packet[4] >> 6;
    const auto mapping = (packet[4] >> 2) & 7;
    const auto code_rate = packet[4] & 3;
    if (header_mode == 3 || mapping > 4 || code_rate == 3
        || ((mapping == 0 || mapping == 3) && code_rate != 2)) {
        return false;
    }
    const auto maximum_delay = (static_cast<std::uint32_t>(packet[6]) << 16)
        | (static_cast<std::uint32_t>(packet[7]) << 8) | packet[8];
    return maximum_delay <= 9'999'999
        && std::all_of(packet.begin() + 19, packet.end(),
                       [](std::uint8_t value) { return value == 0xFF; });
}

}  // namespace dtmb::core
