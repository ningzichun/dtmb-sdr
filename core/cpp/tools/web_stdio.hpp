#pragma once
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <algorithm>
#include <iostream>
#include <streambuf>

EM_ASYNC_JS(int, dtmb_web_read, (char* target, int capacity), {
    const bytes = await Module.dtmbRead(capacity);
    HEAPU8.set(bytes, target);
    return bytes.byteLength;
});
EM_ASYNC_JS(int, dtmb_web_write, (const char* data, int size), {
    await Module.dtmbWrite(HEAPU8.slice(data, data + size));
    return size;
});

namespace dtmb::tools {
class WebInput final : public std::streambuf {
    char pending_ = 0;
    int_type underflow() override {
        if (dtmb_web_read(&pending_, 1) == 0) return traits_type::eof();
        setg(&pending_, &pending_, &pending_ + 1);
        return traits_type::to_int_type(pending_);
    }
    std::streamsize xsgetn(char* target, std::streamsize count) override {
        std::streamsize total = 0;
        if (gptr() && gptr() < egptr() && count) { target[total++] = *gptr(); gbump(1); }
        while (total < count) {
            const auto n = dtmb_web_read(target + total,
                static_cast<int>(std::min<std::streamsize>(count-total, 1024*1024)));
            if (n == 0) break;
            total += n;
        }
        return total;
    }
};
class WebOutput final : public std::streambuf {
    int_type overflow(int_type value) override {
        if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
        const char byte = traits_type::to_char_type(value);
        return dtmb_web_write(&byte, 1) == 1 ? value : traits_type::eof();
    }
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        std::streamsize total = 0;
        while (total < count) {
            const auto n = static_cast<int>(std::min<std::streamsize>(count-total, 1024*1024));
            total += dtmb_web_write(data+total, n);
        }
        return total;
    }
};
inline void configure_web_stdio(bool input, bool output) {
    // These buffers live for the module lifetime, including iostream shutdown.
    if (input) { std::cin.rdbuf(new WebInput); std::cin.tie(nullptr); }
    if (output) std::cout.rdbuf(new WebOutput);
}
}
#endif
