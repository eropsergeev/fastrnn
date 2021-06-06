#pragma once

#include "executer.hpp"

#include <cassert>
#include <cstring>
#include <tuple>

namespace fastrnn {

template<class T>
class GDOptimizer {
private:
    std::vector<std::tuple<T*, T*, size_t>> grads;
public:
    T lr;
    GDOptimizer(T lr): lr(lr) {}
    template<class Executer = StaticExecuter<1>>
    void zero_grad(Executer &exe = Executer::object) {
        for (auto [grad, data, s] : grads) {
            exe.run([grad, data, s](size_t t) {
                size_t b = t * (s / Executer::static_thread_count);
                size_t e = t + s / Executer::static_thread_count;
                if (t + 1 == Executer::static_thread_count) {
                    e = s;
                }
                memset(grad + b, 0, (e - b) * sizeof(T));
            });
        }
    }
    template<class Executer = StaticExecuter<1>>
    void step(Executer &exe = Executer::object) {
        for (auto [grad, data, s] : grads) {
            exe.run([grad, data, s, this](size_t t) {
                size_t b = t * s / Executer::static_thread_count;
                size_t e = t + s / Executer::static_thread_count;
                if (t + 1 == Executer::static_thread_count) {
                    e = s;
                }
                T lr = this->lr;
                #pragma GCC ivdep
                for (size_t i = b; i < e; ++i) {
                    data[i] -= grad[i] * lr;
                }
            });
        }
    }
    void add(T *data, T *grad, size_t s) {
        grads.emplace_back(grad, data, s);
    }
    template<size_t... dims>
    void add(Variable<Tensor<T, dims...>> v) {
        add(v.data->data(), v.grad->data(), Tensor<T, dims...>::total_size);
    }
};

template<class T>
class RMSPropOptimizer {
private:
    std::vector<std::tuple<T*, T*, T*, size_t>> grads;
public:
    T lr, alpha, eps;
    RMSPropOptimizer(T lr, T alpha = 0.99, T eps = 1e-8): lr(lr), alpha(alpha), eps(eps) {}
    template<class Executer = StaticExecuter<1>>
    void zero_grad(Executer &exe = Executer::object) {
        for (auto [grad, data, avg, s] : grads) {
            // std::cout << *std::max_element(grad, grad + s) << std::endl;
            exe.run([grad, data, avg, s](size_t t) {
                size_t b = t * (s / Executer::static_thread_count);
                size_t e = t + s / Executer::static_thread_count;
                if (t + 1 == Executer::static_thread_count) {
                    e = s;
                }
                memset(grad + b, 0, (e - b) * sizeof(T));
            });
        }
    }
    template<class Executer = StaticExecuter<1>>
    void step(Executer &exe = Executer::object) {
        for (auto [grad, data, avg, s] : grads) {
            exe.run([grad, data, avg, s, this](size_t t) {
                size_t b = t * s / Executer::static_thread_count;
                size_t e = t + s / Executer::static_thread_count;
                if (t + 1 == Executer::static_thread_count) {
                    e = s;
                }
                T eps = this->eps;
                T alpha = this->alpha;
                T lr = this->lr;
                #pragma GCC ivdep
                for (size_t i = b; i < e; ++i) {
                    avg[i] = alpha * avg[i] + (grad[i] * grad[i]) * (1 - alpha);
                    data[i] -= grad[i] * lr / std::sqrt(avg[i] + eps);
                }
            });
        }
    }
    void add(T *data, T *grad, size_t s) {
        T *avg = new T[s];
        #pragma GCC ivdep
        for (size_t i = 0; i < s; ++i) {
            avg[i] = 0;
        }
        grads.emplace_back(grad, data, avg, s);
    }
    template<size_t... dims>
    void add(Variable<Tensor<T, dims...>> v) {
        add(v.data->data(), v.grad->data(), Tensor<T, dims...>::total_size);
    }
    ~RMSPropOptimizer() {
        for (auto &x : grads) {
            delete [] std::get<2>(x);
        }
    }
};

};