#ifdef NDEBUG
#undef NDEBUG
#endif
#include "dtmb/nr.hpp"

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    assert(argc == 2);
    std::ifstream input(argv[1]);
    assert(input);
    std::string line;
    unsigned count = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        unsigned byte, word;
        std::istringstream row(line);
        assert(row >> std::hex >> byte >> word);
        assert(byte == count++);
        assert(dtmb::core::nr_encode(static_cast<std::uint8_t>(byte)) == word);
    }
    assert(count == 256);
    const auto& definition = dtmb::core::qam_definition(dtmb::core::parse_qam_mode("4qam-nr"));
    assert(definition.bits_per_symbol == 2);
    assert(definition.coded_bits_per_symbol() == 1);
    assert(definition.signal_frames_per_fec_group() == 2);
    assert(definition.codewords_per_fec_group() == 1);
}
