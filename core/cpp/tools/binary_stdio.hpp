#pragma once

#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace dtmb::tools {

inline void configure_binary_stdio(bool binary_stdin, bool binary_stdout) {
#ifdef _WIN32
    if (binary_stdin && _setmode(_fileno(stdin), _O_BINARY) == -1) {
        throw std::runtime_error("failed to set stdin to binary mode");
    }
    if (binary_stdout && _setmode(_fileno(stdout), _O_BINARY) == -1) {
        throw std::runtime_error("failed to set stdout to binary mode");
    }
#else
    (void)binary_stdin;
    (void)binary_stdout;
#endif
}

}  // namespace dtmb::tools
