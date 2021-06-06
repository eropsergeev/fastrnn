#pragma once

#include "barrier.hpp"
#include <thread>
#include <array>
#include <functional>

namespace fastrnn {

template<size_t n>
class StaticExecuter {
private:
    SpinBarrier barrier;
    std::array<std::thread, n - 1> threads;
    std::atomic<bool> stoped;
    std::function<void(size_t)> cur_func;
public:
    static constexpr size_t static_thread_count = n;

    StaticExecuter(): barrier(n), stoped(0) {
        for (size_t i = 0; i < n - 1; ++i) {
            threads[i] = std::thread([i, this]() {
                barrier.arrive();
                if (stoped.load())
                    return;
                while (1) {
                    barrier.arrive();
                    cur_func(i);
                    barrier.arrive();
                    if (stoped.load())
                        break;
                }
            });
        }
    }

    size_t thread_count() const {
        return n;
    }

    template<class F>
    void run(const F &f) {
        barrier.arrive();
        cur_func = f;
        barrier.arrive();
        f(n - 1);
        // barrier.arrive();
    }

    ~StaticExecuter() {
        stoped.store(1);
        barrier.arrive();
        for (auto &t : threads) {
            t.join();
        }
    }
};

template<>
class StaticExecuter<1> {
public:
    static constexpr size_t static_thread_count = 1;

    size_t thread_count() const {
        return 1;
    }

    template<class F>
    void run(const F &f) {
        f(0);
    }

    static StaticExecuter<1> object;
};

};