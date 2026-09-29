#pragma once
#include <thread>
#include <utility>
namespace dtmb::core {
#ifdef __EMSCRIPTEN__
class WorkerThread {
public:
    template<class F, class... Args> explicit WorkerThread(F&& f, Args&&... args) {
        std::forward<F>(f)(std::forward<Args>(args)...);
    }
    void join() noexcept {}
    static unsigned hardware_concurrency() noexcept { return 1; }
};
#else
using WorkerThread = std::thread;
#endif
}
