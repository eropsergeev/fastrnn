#pragma once

#include "tensor.hpp"
#include "variable.hpp"
#include "executer.hpp"

#include <algorithm>

namespace fastrnn {

template<class T, size_t input_size, size_t hidden_size, bool use_grad = true>
class GRUCell {};

template<class T, size_t input_size, size_t hidden_size>
class GRUCell<T, input_size, hidden_size, false> {
public:
    Tensor<T, hidden_size, input_size> Wz, Wr, Wh;
    Tensor<T, hidden_size, hidden_size> Uz, Ur, Uh;
    Tensor<T, hidden_size> bz, br, bh;
public:
    GRUCell() = default;
    template<class F>
    explicit GRUCell(F f) {
        std::generate(Wz.template view<hidden_size * input_size>().begin(), Wz.template view<hidden_size * input_size>().end(), f);
        std::generate(Wr.template view<hidden_size * input_size>().begin(), Wr.template view<hidden_size * input_size>().end(), f);
        std::generate(Wh.template view<hidden_size * input_size>().begin(), Wh.template view<hidden_size * input_size>().end(), f);
        std::generate(Uz.template view<hidden_size * hidden_size>().begin(), Uz.template view<hidden_size * hidden_size>().end(), f);
        std::generate(Ur.template view<hidden_size * hidden_size>().begin(), Ur.template view<hidden_size * hidden_size>().end(), f);
        std::generate(Uh.template view<hidden_size * hidden_size>().begin(), Uh.template view<hidden_size * hidden_size>().end(), f);
        std::fill(bz.template view<hidden_size>().begin(), bz.template view<hidden_size>().end(), 0);
        std::fill(br.template view<hidden_size>().begin(), br.template view<hidden_size>().end(), 0);
        std::fill(bh.template view<hidden_size>().begin(), bh.template view<hidden_size>().end(), 0);
    }
    template<class Executer = StaticExecuter<1>>
    void operator()(
        const Tensor<T, input_size> &x,
        const Tensor<T, hidden_size> &h,
        Tensor<T, hidden_size> &next_h,
        Executer &exe = Executer::object)
    {
        Tensor<T, hidden_size> z, r;
        matmul_transposed<ASSIGN>(Wz, x.template view<1, input_size>(), z.template view<hidden_size, 1>(), exe);
        matmul_transposed<ADD>(Uz, h.template view<1, hidden_size>(), z.template view<hidden_size, 1>(), exe);
        add_(bz, z, exe);
        fast_sigmoid(z, z, exe);
        matmul_transposed<ASSIGN>(Wr, x.template view<1, input_size>(), r.template view<hidden_size, 1>(), exe);
        matmul_transposed<ADD>(Ur, h.template view<1, hidden_size>(), r.template view<hidden_size, 1>(), exe);
        add_(br, r, exe);
        fast_sigmoid(r, r, exe);
        mul_(h, r);
        matmul_transposed<ASSIGN>(Wh, x.template view<1, input_size>(), next_h.template view<hidden_size, 1>(), exe);
        matmul_transposed<ADD>(Uh, r.template view<1, hidden_size>(), next_h.template view<hidden_size, 1>(), exe);
        add_(bh, next_h, exe);
        fast_tanh(next_h, next_h, exe);
        apply_func(h, z, next_h, [](auto h, auto z, auto next_h) {
            return z * h + (1 - z) * next_h;
        }, exe);
    }
};

template<class T, size_t input_size, size_t hidden_size>
class GRUCell<T, input_size, hidden_size, true>: public GRUCell<T, input_size, hidden_size, false> {
protected:
    Tensor<T, hidden_size, input_size> Wz_grad, Wr_grad, Wh_grad;
    Tensor<T, hidden_size, hidden_size> Uz_grad, Ur_grad, Uh_grad;
    Tensor<T, hidden_size> bz_grad, br_grad, bh_grad;
    using Base = GRUCell<T, input_size, hidden_size, false>;
public:
    GRUCell() = default;
    template<class F>
    explicit GRUCell(F f): Base(f) {
        Wz_grad = 0;
        Wr_grad = 0;
        Wh_grad = 0;
        Uz_grad = 0;
        Ur_grad = 0;
        Uh_grad = 0;
        bz_grad = 0;
        br_grad = 0;
        bh_grad = 0;
    }
    GRUCell<T, input_size, hidden_size, false> &no_grad() {
        return static_cast<GRUCell<T, input_size, hidden_size, false>&>(*this);
    }
    template<class Input, class Allocator, class Executer = StaticExecuter<1>>
    void operator()(
        Input &x,
        Variable<Tensor<T, hidden_size>> h,
        Variable<Tensor<T, hidden_size>> next_h,
        GradientCalculator &calc,
        Allocator &allocator,
        Executer &exe = Executer::object)
    {
        Variable<Tensor<T, hidden_size>> z(*allocator.template allocate<hidden_size>(), *allocator.template allocate<hidden_size>());
        Variable<Tensor<T, hidden_size>> r(*allocator.template allocate<hidden_size>(), *allocator.template allocate<hidden_size>());
        Variable<Tensor<T, hidden_size>> pre_z(*allocator.template allocate<hidden_size>(), *allocator.template allocate<hidden_size>());
        Variable<Tensor<T, hidden_size>> pre_r(*allocator.template allocate<hidden_size>(), *allocator.template allocate<hidden_size>());
        Variable<Tensor<T, hidden_size>> pre_h(*allocator.template allocate<hidden_size>(), *allocator.template allocate<hidden_size>());
        Variable<Tensor<T, hidden_size>> pre_h2(*allocator.template allocate<hidden_size>(), *allocator.template allocate<hidden_size>());
        calc.matmul_transposed<ASSIGN>(Variable(Base::Wz, Wz_grad), x.template view<1, input_size>(), pre_z.template view<hidden_size, 1>(), exe);
        calc.matmul_transposed<ADD>(Variable(Base::Uz, Uz_grad), h.template view<1, hidden_size>(), pre_z.template view<hidden_size, 1>(), exe);
        calc.add_(Variable(Base::bz, bz_grad), pre_z, exe);
        calc.fast_sigmoid(pre_z, z, exe);
        calc.matmul_transposed<ASSIGN>(Variable(Base::Wr, Wr_grad), x.template view<1, input_size>(), pre_r.template view<hidden_size, 1>(), exe);
        calc.matmul_transposed<ADD>(Variable(Base::Ur, Ur_grad), h.template view<1, hidden_size>(), pre_r.template view<hidden_size, 1>(), exe);
        calc.add_(Variable(Base::br, br_grad), pre_r, exe);
        calc.fast_sigmoid(pre_r, r, exe);
        Variable<Tensor<T, hidden_size>> rh(*allocator.template allocate<hidden_size>(), *allocator.template allocate<hidden_size>());
        calc.mul(h, r, rh);
        calc.matmul_transposed<ASSIGN>(Variable(Base::Wh, Wh_grad),
            x.template view<1, input_size>(),
            pre_h.template view<hidden_size, 1>(), exe);
        calc.matmul_transposed<ADD>(Variable(Base::Uh, Uh_grad),
            rh.template view<1, hidden_size>(),
            pre_h.template view<hidden_size, 1>(), exe);
        calc.add_(Variable(Base::bh, bh_grad), pre_h, exe);
        calc.fast_tanh(pre_h, pre_h2, exe);
        calc.apply_func(h, z, pre_h2, next_h, [](auto h, auto z, auto pre_h2, auto next_h) {
            return z * h + ((T) 1 - z) * pre_h2;
        }, [](auto h, auto z, auto pre_h2, auto next_h, auto &h_, auto &z_, auto &pre_h2_, auto next_h_) {
            h_ += z * next_h_;
            z_ += (h - pre_h2) * next_h_;
            pre_h2_ += ((T) 1 - z) * next_h_;
        }, exe);
    }
    template<class Optimizer>
    void register_in_optimizer(Optimizer &opt) {
        opt.add(Base::Wz.data(), Wz_grad.data(), input_size * hidden_size);
        opt.add(Base::Wr.data(), Wr_grad.data(), input_size * hidden_size);
        opt.add(Base::Wh.data(), Wh_grad.data(), input_size * hidden_size);
        opt.add(Base::Uz.data(), Uz_grad.data(), hidden_size * hidden_size);
        opt.add(Base::Ur.data(), Ur_grad.data(), hidden_size * hidden_size);
        opt.add(Base::Uh.data(), Uh_grad.data(), hidden_size * hidden_size);
        opt.add(Base::bz.data(), bz_grad.data(), hidden_size);
        opt.add(Base::br.data(), br_grad.data(), hidden_size);
        opt.add(Base::bh.data(), bh_grad.data(), hidden_size);
    }
};

};