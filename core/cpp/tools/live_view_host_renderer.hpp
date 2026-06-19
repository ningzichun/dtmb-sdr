#pragma once

#include <cstdint>
#include <istream>
#include <ostream>

namespace dtmb::tools::live_view_host {

struct Options {
    bool fail_fast = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

int run(std::istream &input, std::ostream &frame_output,
        std::ostream &diagnostics, const Options &options = {});

} // namespace dtmb::tools::live_view_host
