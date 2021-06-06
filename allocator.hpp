#pragma once

#include "tensor.hpp"
#include <cstring>

namespace fastrnn {

template<class T, size_t n, bool zero = false>
class TensorAllocator {
private:
    T buf[n];
    size_t pos;
public:
    TensorAllocator(): pos(0) {}
    template<size_t... dims>
    Tensor<T, dims...> *allocate() {
        constexpr size_t size = sizeof(Tensor<T, dims...>) / sizeof(T);
        if (pos + size > n) {
            return nullptr;
        }
        if constexpr (zero) {
            memset(buf + pos, 0, size * sizeof(T));
        }
        pos += size;
        return reinterpret_cast<Tensor<T, dims...> *>(buf + pos - size);
    }
    void reset() {
        pos = 0;
    }
};

};


