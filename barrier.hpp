#pragma once

#include <atomic>
#include <cstddef>

namespace fastrnn {

class SpinBarrier {
private:
    std::atomic<size_t> cnt, gen;
    size_t max;
public:
    SpinBarrier(size_t x) : cnt(x), gen(0), max(x) {}
    void arrive() {
        size_t g = gen.load();
        cnt.fetch_sub(1);
        while (cnt.load() && g == gen.load()) {}
        if (g == gen.exchange(g + 1)) {
            cnt = max;
        }
        while (cnt.load() == 0) {}
    }
};

};