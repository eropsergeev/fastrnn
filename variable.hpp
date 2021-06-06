#pragma once

#include <vector>
#include <functional>
#include <type_traits>
#include <cstring>
#include "executer.hpp"
#include "tensor.hpp"
#include "sysinfo.hpp"

namespace fastrnn {

template<class T>
struct Variable {
    T *data, *grad;
    Variable(T &data, T &grad): data(&data), grad(&grad) {}
    auto operator[](size_t i) {
        return Variable<typename std::remove_reference<decltype((*data)[i])>::type>((*data)[i], (*grad)[i]);
    }
    template<size_t... new_dims>
    Variable<Tensor<typename T::element_type, new_dims...>> view() {
        return Variable<Tensor<typename T::element_type, new_dims...>>
        (data->template view<new_dims...>(), grad->template view<new_dims...>());
    }
};

template<class T>
struct IsVariable {
    static constexpr bool value = false;
};

template<class T>
struct IsVariable<Variable<T>> {
    static constexpr bool value = true;
};

class GradientCalculator {
private:
    std::vector<std::function<void()>> backward_log;
public:
    template<class T>
    void backward(Variable<Tensor<T>> l) {
        *l.grad = 1;
        while (backward_log.size()) {
            backward_log.back()();
            backward_log.pop_back();
        }
    }

    // ===============

    template<CalculationMode mode = ASSIGN, class T, size_t n, size_t m, size_t k, class Executer = StaticExecuter<1>>
    void matmul_transposed(
        Variable<Tensor<T, n, m>> a,
        Variable<Tensor<T, k, m>> b,
        Variable<Tensor<T, n, k>> ans,
        Executer &exe = Executer::object)
    {
        fastrnn::matmul_transposed<mode>(*a.data, *b.data, *ans.data, exe);
        backward_log.emplace_back([a, b, ans, &exe]() {
            typename std::conditional<n == 1 || k == 1, Tensor<T, k, n>*, Tensor<T, k, n>>::type ans_grad_t;
            Tensor<T, k, n> *ans_grad_t_ptr;
            if constexpr (n == 1 || k == 1) {
                ans_grad_t_ptr = ans_grad_t = &ans.grad->template view<k, n>();
            } else {
                transpose(*ans.grad, ans_grad_t, exe);
                ans_grad_t_ptr = &ans_grad_t;
            }
            typename std::conditional<n == 1 || m == 1, Tensor<T, m, n>*, Tensor<T, m, n>>::type a_data_t;
            Tensor<T, m, n> *a_data_t_ptr;
            if constexpr (n == 1 || m == 1) {
                a_data_t_ptr = a_data_t = &a.data->template view<m, n>();
            } else {
                transpose(*a.data, a_data_t, exe);
                a_data_t_ptr = &a_data_t;
            }
            typename std::conditional<k == 1 || m == 1, Tensor<T, m, k>*, Tensor<T, m, k>>::type b_data_t;
            Tensor<T, m, k> *b_data_t_ptr;
            if constexpr (k == 1 || m == 1) {
                b_data_t_ptr = b_data_t = &b.data->template view<m, k>();
            } else {
                transpose(*b.data, b_data_t, exe);
                b_data_t_ptr = &b_data_t;
            }
            fastrnn::matmul_transposed<ADD>(*ans.grad, *b_data_t_ptr, *a.grad, exe);
            fastrnn::matmul_transposed<ADD>(*ans_grad_t_ptr, *a_data_t_ptr, *b.grad, exe);
        });
    }
    template<CalculationMode mode = ASSIGN, class T, size_t n, size_t m, size_t k, class Executer = StaticExecuter<1>>
    void matmul_transposed(
        const Tensor<T, n, m> &a,
        Variable<Tensor<T, k, m>> b,
        Variable<Tensor<T, n, k>> ans,
        Executer &exe = Executer::object)
    {
        fastrnn::matmul_transposed<mode>(a, *b.data, *ans.data, exe);
        backward_log.emplace_back([&a, b, ans, &exe]() {
            typename std::conditional<n == 1 || k == 1, Tensor<T, k, n>*, Tensor<T, k, n>>::type ans_grad_t;
            Tensor<T, k, n> *ans_grad_t_ptr;
            if constexpr (n == 1 || k == 1) {
                ans_grad_t_ptr = ans_grad_t = &ans.grad->template view<k, n>();
            } else {
                transpose(*ans.grad, ans_grad_t, exe);
                ans_grad_t_ptr = &ans_grad_t;
            }
            typename std::conditional<n == 1 || m == 1, const Tensor<T, m, n>*, Tensor<T, m, n>>::type a_data_t;
            const Tensor<T, m, n> *a_data_t_ptr;
            if constexpr (n == 1 || m == 1) {
                a_data_t_ptr = a_data_t = &a.template view<m, n>();
            } else {
                transpose(a, a_data_t, exe);
                a_data_t_ptr = &a_data_t;
            }
            fastrnn::matmul_transposed<ADD>(*ans_grad_t_ptr, *a_data_t_ptr, *b.grad, exe);
        });
    }
    template<CalculationMode mode = ASSIGN, class T, size_t n, size_t m, size_t k, class Executer = StaticExecuter<1>>
    void matmul_transposed(
        Variable<Tensor<T, n, m>> a,
        const Tensor<T, k, m> &b,
        Variable<Tensor<T, n, k>> ans,
        Executer &exe = Executer::object)
    {
        fastrnn::matmul_transposed<mode>(*a.data, b, *ans.data, exe);
        backward_log.emplace_back([a, &b, ans, &exe]() {
            typename std::conditional<k == 1 || m == 1, const Tensor<T, m, k>*, Tensor<T, m, k>>::type b_data_t;
            const Tensor<T, m, k> *b_data_t_ptr;
            if constexpr (k == 1 || m == 1) {
                b_data_t_ptr = b_data_t = &b.template view<m, k>();
            } else {
                transpose(b, b_data_t, exe);
                b_data_t_ptr = &b_data_t;
            }
            fastrnn::matmul_transposed<ADD>(*ans.grad, *b_data_t_ptr, *a.grad, exe);
        });
    }

    // ===============

    template<class T, class Executer = StaticExecuter<1>>
    void add_(Variable<T> a, Variable<T> ans, Executer &exe = Executer::object) {
        fastrnn::add_(*a.data, *ans.data, exe);
        backward_log.emplace_back([a, ans, &exe]() {
            fastrnn::add_(*ans.grad, *a.grad, exe);
        });
    }

    template<class T, class Executer = StaticExecuter<1>>
    void sub_(Variable<T> a, Variable<T> ans, Executer &exe = Executer::object) {
        fastrnn::sub_(*a.data, *ans.data, exe);
        backward_log.emplace_back([a, ans, &exe]() {
            fastrnn::sub_(*ans.grad, *a.grad, exe);
        });
    }

    // ===============

    template<CTensor Res, class F, class G, class Executer, CTensor... Args>
    void __apply_func(Variable<Res> res, F f, G g, Executer &exe, Variable<Args>... args) {
        static_assert((IsBroadcastableTo<Args, Res>::value && ...));
        fastrnn::__apply_func(*res.data, f, exe, (*args.data)...);
        backward_log.emplace_back([&exe, res, g, args...]() {
            fastrnn::__apply_func(*res.grad, g, exe, *args.data..., *res.data, *args.grad...);
        });
    }

    template<class... Args>
    void apply_func(Args&&... args) {
        static_assert(sizeof...(Args) > 2);
        if constexpr (sizeof...(Args) == 3) {
            __apply_func(args..., StaticExecuter<1>::object);
        } else {
            auto internal = [this]<class Arg1, class Arg2, class Arg3, class Arg4, class... OtherArgs>(
                auto internal,
                Arg1 &&arg1,
                Arg2 &&arg2,
                Arg3 &&arg3,
                Arg4 &&arg4,
                OtherArgs&&... other_args)
            {
                if constexpr (!IsVariable<typename std::remove_reference<Arg2>::type>::value) {
                    static_assert(!IsVariable<typename std::remove_reference<Arg3>::type>::value,
                        "at least 2 non-variable arguments is required");
                    if constexpr (!IsVariable<typename std::remove_reference<Arg4>::type>::value) {
                        __apply_func(arg1, arg2, arg3, arg4, other_args...);
                    } else {
                        __apply_func(arg1, arg2, arg3, StaticExecuter<1>::object, arg4, other_args...);
                    }
                } else {
                    internal(internal, arg2, arg3, arg4, other_args..., arg1);
                }
            };
            internal(internal, args...);
        }
    }

    template<CalculationMode mode = ASSIGN, class T, class Executer = StaticExecuter<1>>
    void add(Variable<T> a, Variable<T> b, Variable<T> ans, Executer &exe = Executer::object) {
        if constexpr (mode == ASSIGN)
            fastrnn::assign_(*a.data, *ans.data, exe);
        else
            fastrnn::add_(*a.data, *ans.data, exe);
        fastrnn::add_(*b.data, *ans.data);
        backward_log.emplace_back([a, b, ans, &exe]() {
            fastrnn::add_(*ans.grad, *a.grad, exe);
            fastrnn::add_(*ans.grad, *b.grad, exe);
        });
    }

    template<CalculationMode mode = ASSIGN, class T, class Executer = StaticExecuter<1>>
    void sub(Variable<T> a, Variable<T> b, Variable<T> ans, Executer &exe = Executer::object) {
        if constexpr (mode == ASSIGN)
            fastrnn::assign_(*a.data, *ans.data, exe);
        else
            fastrnn::sub_(*a.data, *ans.data, exe);
        fastrnn::sub_(*b.data, *ans.data);
        backward_log.emplace_back([a, b, ans, &exe]() {
            fastrnn::sub_(*ans.grad, *a.grad, exe);
            fastrnn::sub_(*ans.grad, *b.grad, exe);
        });
    }

    template<CalculationMode mode = ASSIGN, class T, size_t... dims, class Executer = StaticExecuter<1>>
    void mul(Variable<Tensor<T, dims...>> a, Variable<Tensor<T, dims...>> b, Variable<Tensor<T, dims...>> ans, Executer &exe = Executer::object) {
        static_assert(mode == ASSIGN, "Not implemented");
        fastrnn::assign_(*a.data, *ans.data, exe);
        fastrnn::mul_(*b.data, *ans.data, exe);
        backward_log.emplace_back([a, b, ans, &exe]() {
            constexpr size_t size = sizeof(typename std::remove_reference<decltype(*a.data)>::type) / sizeof(T);
            constexpr size_t block = size / Executer::static_thread_count;
            auto a_data = a.data->data();
            auto b_data = b.data->data();
            auto a_grad = a.grad->data();
            auto b_grad = b.grad->data();
            auto ans_grad = ans.grad->data();
            if constexpr (block) {
                exe.run([a_data, a_grad, b_data, b_grad, ans_grad](size_t t) {
                    t *= block;
                    #pragma GCC ivdep
                    for (size_t i = t; i < t + block; ++i) {
                        a_grad[i] += ans_grad[i] * b_data[i];
                        b_grad[i] += ans_grad[i] * a_data[i];
                    }
                });
            }
            #pragma GCC ivdep
            for (size_t i = size - (block ? size % block : size); i < size; ++i) {
                a_grad[i] += ans_grad[i] * b_data[i];
                b_grad[i] += ans_grad[i] * a_data[i];
            }
        });
    }

    // ===============

    template<class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
    void sum(Variable<Tensor<T, dims1...>> a, Variable<Tensor<T, dims2...>> ans, Executer &exe = Executer::object) {
        fastrnn::sum(*a.data, *ans.data, exe);
        backward_log.emplace_back([a, ans, &exe]() {
            constexpr size_t broadcast = sizeof(typename std::remove_reference<decltype(*a.data)>::type)
                / sizeof(typename std::remove_reference<decltype(*ans.data)>::type);
            constexpr size_t size = sizeof(typename std::remove_reference<decltype(*ans.data)>::type) / sizeof(T);
            constexpr size_t block = size / Executer::static_thread_count;
            if constexpr (block) {
                exe.run([a, ans](size_t t) {
                    t *= block;
                    for (size_t i = t; i < t + block; ++i) {
                        T val = ans.grad->data()[i];
                        #pragma GCC ivdep
                        for (size_t j = 0; j < broadcast; ++j) {
                            a.grad->data()[i * broadcast + j] += val;
                        }
                    }
                });
            }
            for (size_t i = size - (block ? size % block : size); i < size; ++i) {
                T val = ans.grad->data()[i];
                #pragma GCC ivdep
                for (size_t j = 0; j < broadcast; ++j) {
                    a.grad->data()[i * broadcast + j] += val;
                }
            }
        });
    }

    // ===============

    template<class T, size_t... dims, class Executer = StaticExecuter<1>>
    void relu(Variable<Tensor<T, dims...>> a, Variable<Tensor<T, dims...>> ans, Executer &exe = Executer::object) {
        __apply_func(ans,
            [](T x, T y) {
                return x <= 0 ? 0 : x;
            },
            [](T x, T y, T &x_, T y_) {
                x_ += (x > 0 ? y_ : 0);
            }, exe, a);
    }

    template<class T, size_t... dims, class Executer = StaticExecuter<1>>
    void fast_tanh(Variable<Tensor<T, dims...>> a, Variable<Tensor<T, dims...>> ans, Executer &exe = Executer::object) {
        __apply_func(ans,
            [](T x, T y) {
                T abs;
                if constexpr (std::is_same<T, float>()) {
                    union {
                        float f;
                        uint32_t i;
                    } u;
                    u.f = x;
                    u.i &= 0x7FFFFFFF;
                    abs = u.f;
                } else {
                    abs = std::abs(x);
                }
                return x / (abs + (T) 1);
            },
            [](T x, T y, T &x_, T y_) {
                T abs;
                if constexpr (std::is_same<T, float>()) {
                    union {
                        float f;
                        uint32_t i;
                    } u;
                    u.f = x;
                    u.i &= 0x7FFFFFFF;
                    abs = u.f;
                } else {
                    abs = std::abs(x);
                }
                x_ += y_ / (((T) 1 + abs) * ((T) 1 + abs));
            }, exe, a);
    }

    template<class T, size_t... dims, class Executer = StaticExecuter<1>>
    void fast_sigmoid(Variable<Tensor<T, dims...>> a, Variable<Tensor<T, dims...>> ans, Executer &exe = Executer::object) {
        __apply_func(ans,
            [](T x, T y) {
                T abs;
                if constexpr (std::is_same<T, float>()) {
                    union {
                        float f;
                        uint32_t i;
                    } u;
                    u.f = x;
                    u.i &= 0x7FFFFFFF;
                    abs = u.f;
                } else {
                    abs = std::abs(x);
                }
                return (T) 0.5 + x / (abs * (T) 2 + (T) 2);
            },
            [](T x, T y, T &x_, T y_) {
                T abs;
                if constexpr (std::is_same<T, float>()) {
                    union {
                        float f;
                        uint32_t i;
                    } u;
                    u.f = x;
                    u.i &= 0x7FFFFFFF;
                    abs = u.f;
                } else {
                    abs = std::abs(x);
                }
                x_ += y_ / (((T) 1 + abs) * ((T) 1 + abs) * (T) 2);
            }, exe, a);
    }

    template<CalculationMode mode = ASSIGN, class T, size_t... dims, class Executer = StaticExecuter<1>>
    void exp(Variable<Tensor<T, dims...>> a, Variable<Tensor<T, dims...>> ans, Executer &exe = Executer::object) {
        __apply_func(ans,
            [](T x, T y) {
                if constexpr (mode == ASSIGN)
                    return std::exp(x);
                else
                    return std::exp(x) + y;
            },
            [](T x, T y, T &x_, T y_) {
                x_ += y * y_;
            }, exe, a);
    }

    template<CalculationMode mode = ASSIGN, class T, size_t... dims, class Executer = StaticExecuter<1>>
    void log(Variable<Tensor<T, dims...>> a, Variable<Tensor<T, dims...>> ans, Executer &exe = Executer::object) {
        __apply_func(ans,
            [](T x, T y) {
                if constexpr (mode == ASSIGN)
                    return std::log(x);
                else
                    return std::log(x) + y;
            },
            [](T x, T y, T &x_, T y_) {
                x_ += y_ / x;
            }, exe, a);
    }
};

};
