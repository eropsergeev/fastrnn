#pragma once

#include "tensor.hpp"
#include "variable.hpp"
#include "executer.hpp"

#include <algorithm>

namespace fastrnn {

template<class T, size_t input_size, size_t output_size, bool use_grad = true>
class Linear {};

template<class T, size_t input_size, size_t output_size>
class Linear<T, input_size, output_size, false> {
public:
    Tensor<T, output_size, input_size> W;
    Tensor<T, output_size> b;
public:
    Linear() = default;
    template<class F>
    explicit Linear(F f) {
        std::generate(W.template view<output_size * input_size>().begin(), W.template view<output_size * input_size>().end(), f);
        std::fill(b.template view<output_size>().begin(), b.template view<output_size>().end(), 0);
    }
    template<class Executer = StaticExecuter<1>>
    void operator()(
        const Tensor<T, input_size> &x,
        Tensor<T, output_size> &out,
        Executer &exe = Executer::object)
    {
        matmul_transposed(W, x.template view<1, input_size>(), out.template view<output_size, 1>(), exe);
        add_(b, out, exe);
    }
};

template<class T, size_t input_size, size_t output_size>
class Linear<T, input_size, output_size, true>: public Linear<T, input_size, output_size, false> {
public:
    Tensor<T, output_size, input_size> W_grad;
    Tensor<T, output_size> b_grad;
    using Base = Linear<T, input_size, output_size, false>;
public:
    Linear() = default;
    template<class F>
    explicit Linear(F f): Base(f) {
        W_grad = 0;
        b_grad = 0;
    }
    Linear<T, input_size, output_size, false> &no_grad() {
        return static_cast<Linear<T, input_size, output_size, false>&>(*this);
    }
    template<class Executer = StaticExecuter<1>>
    void operator()(
        Variable<Tensor<T, input_size>> x,
        Variable<Tensor<T, output_size>> out,
        GradientCalculator &calc,
        Executer &exe = Executer::object)
    {
        calc.matmul_transposed(Variable(Base::W, W_grad), x.template view<1, input_size>(), out.template view<output_size, 1>());
        calc.add_(Variable(Base::b, b_grad), out);
    }
    template<class Executer = StaticExecuter<1>>
    void operator()(
        const Tensor<T, input_size> &x,
        Variable<Tensor<T, output_size>> out,
        GradientCalculator &calc,
        Executer &exe = Executer::object)
    {
        calc.matmul_transposed(Variable(Base::W, W_grad), x.template view<1, input_size>(), out.template view<output_size, 1>());
        calc.add_(Variable(Base::b, b_grad), out);
    }
    template<class Optimizer>
    void register_in_optimizer(Optimizer &opt) {
        opt.add(Variable(Base::W, W_grad));
        opt.add(Variable(Base::b, b_grad));
    }
};

};