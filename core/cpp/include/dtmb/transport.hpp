#pragma once

#include <cstdint>
#include <span>

namespace dtmb::core {

// GB/T 28434-2012, 5.3.1.2, Figure 5 and Table 2. A SIP has a fixed
// continuity counter of zero. Recognition validates the complete packet,
// including defined mode values, maximum delay and 169 stuffing bytes.
[[nodiscard]] bool is_dtmb_sip_packet(std::span<const std::uint8_t> packet) noexcept;

}  // namespace dtmb::core
