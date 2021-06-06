#pragma once

#include <type_traits>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>
#include "executer.hpp"
#include "sysinfo.hpp"

namespace fastrnn {

template<class T, size_t... dims>
class Tensor {};

template<class T>
class Tensor<T> {
private:
    T data_;
public:
    static constexpr size_t total_size = 1;
    using element_type = T;
    Tensor() = default;
    Tensor(T x): data_(x) {}
    Tensor& operator=(T x) {
        data_ = x;
        return *this;
    }
    operator T&() {
        return data_;
    }
    operator const T&() const {
        return data_;
    }
    T *data() {
        return reinterpret_cast<T*>(&data_);
    }
    const T *data() const {
        return reinterpret_cast<const T*>(&data_);
    }
    template<size_t... new_dims>
    Tensor<T, new_dims...> &view() {
        static_assert(sizeof(*this) == sizeof(Tensor<T, new_dims...>));
        return *reinterpret_cast<Tensor<T, new_dims...> *>(this);
    }
    template<size_t... new_dims>
    const Tensor<T, new_dims...> &view() const {
        static_assert(sizeof(*this) == sizeof(Tensor<T, new_dims...>));
        return *reinterpret_cast<const Tensor<T, new_dims...> *>(this);
    }
};

template<class T, size_t n, size_t... dims>
class Tensor<T, n, dims...> {
    template<class U, size_t... other_dims>
    friend class Tensor;
private:
    std::array<Tensor<T, dims...>, n> data_;
public:
    using element_type = T;
    static constexpr size_t static_size = n;
    static constexpr size_t total_size = n * Tensor<T, dims...>::total_size;
    Tensor() = default;
    template<size_t... new_dims>
    Tensor<T, new_dims...> &view() {
        static_assert(sizeof(*this) == sizeof(Tensor<T, new_dims...>));
        return *reinterpret_cast<Tensor<T, new_dims...> *>(this);
    }
    template<size_t... new_dims>
    const Tensor<T, new_dims...> &view() const {
        static_assert(sizeof(*this) == sizeof(Tensor<T, new_dims...>));
        return *reinterpret_cast<const Tensor<T, new_dims...> *>(this);
    }
    T *data() {
        return reinterpret_cast<T*>(&data_);
    }
    const T *data() const {
        return reinterpret_cast<const T*>(&data_);
    }
    template<size_t... other_dims>
    explicit Tensor(const Tensor<T, other_dims...> &other) {
        (*this) = other;
    }
    explicit Tensor(T x) {
        (*this) = x;
    }

    template<size_t... other_dims>
    Tensor &operator=(const Tensor<T, other_dims...> &other) {
        if constexpr (sizeof...(dims) + 1 == sizeof...(other_dims)) {
            data_ = other.data_;
        } else {
            static_assert(sizeof...(dims) + 1 > sizeof...(other_dims));
            for (size_t i = 0; i < n; ++i) {
                data_[i] = other;
            }
        }
        return *this;
    }
    Tensor &operator=(T x) {
        for (size_t i = 0; i < n; ++i) {
            data_[i] = x;
        }
        return *this;
    }

    Tensor<T, dims...> &operator[](size_t i) {
        return data_[i];
    }
    const Tensor<T, dims...> &operator[](size_t i) const {
        return data_[i];
    }

    template<size_t from, size_t to>
    Tensor<T, to - from, dims...> &subtensor() {
        static_assert(from < to && to <= n);
        return *reinterpret_cast<Tensor<T, to - from, dims...> *>(&data_[from]);
    }

    template<size_t from, size_t to>
    const Tensor<T, to - from, dims...> &subtensor() const {
        static_assert(from < to && to <= n);
        return *reinterpret_cast<const Tensor<T, to - from, dims...> *>(&data_[from]);
    }

    auto begin() {
        return data_.begin();
    }
    auto begin() const {
        return data_.begin();
    }
    auto end() {
        return data_.end();
    }
    auto end() const {
        return data_.end();
    }
    size_t size() const {
        return n;
    }
};

template<class T>
struct IsTensor {
    static constexpr bool value = false;
};

template<class T, size_t... dims>
struct IsTensor<Tensor<T, dims...>> {
    static constexpr bool value = true;
};

template<class T, size_t... dims>
struct IsTensor<const Tensor<T, dims...>> {
    static constexpr bool value = true;
};

template<class T>
concept CTensor = IsTensor<T>::value;

template<CTensor A, CTensor B>
struct IsBroadcastableTo {
    static constexpr bool value = IsBroadcastableTo<A, typename std::remove_reference<decltype(std::declval<B>()[0])>::type>::value;
};

template<CTensor A, CTensor B>
struct IsBroadcastableTo<const A, B>
{
    static constexpr bool value = IsBroadcastableTo<A, B>::value;
};

template<CTensor A, CTensor B>
struct IsBroadcastableTo<A, const B>
{
    static constexpr bool value = IsBroadcastableTo<A, B>::value;
};

template<CTensor A, CTensor B>
struct IsBroadcastableTo<const A, const B>
{
    static constexpr bool value = IsBroadcastableTo<A, B>::value;
};

template<class T1, class T2, size_t... dims>
struct IsBroadcastableTo<Tensor<T1, dims...>, Tensor<T2>> {
    static constexpr bool value = false;
};

template<class T1, class T2, size_t first_dim, size_t... dims>
struct IsBroadcastableTo<Tensor<T1, first_dim, dims...>, Tensor<T2, first_dim, dims...>> {
    static constexpr bool value = true;
};

template<class T1, class T2>
struct IsBroadcastableTo<Tensor<T1>, Tensor<T2>> {
    static constexpr bool value = true;
};

// <op>= operations with Tensor

template<CTensor Res, class F, class Executer, CTensor... Args>
inline void __apply_func(Res &res, F f, Executer &exe, Args&... args) {
    static_assert((IsBroadcastableTo<Args, Res>::value && ...));
    if constexpr (((Res::total_size == Args::total_size) && ...)) {
        constexpr size_t size = Res::total_size;
        constexpr size_t block = size / Executer::static_thread_count;
        if constexpr (block) {
            exe.run([&](size_t t) {
                t *= block;
                #pragma GCC ivdep
                for (size_t i = t; i < t + block; ++i) {
                    if constexpr (std::is_void<decltype(f(args.data()[i]..., res.data()[i]))>::value)
                        f(args.data()[i]..., res.data()[i]);
                    else
                        res.data()[i] = f(args.data()[i]..., res.data()[i]);
                }
            });
        }
        #pragma GCC ivdep
        for (size_t i = size - (block ? size % block : size); i < size; ++i) {
            if constexpr (std::is_void<decltype(f(args.data()[i]..., res.data()[i]))>::value)
                f(args.data()[i]..., res.data()[i]);
            else
                res.data()[i] = f(args.data()[i]..., res.data()[i]);
        }
    } else {
        constexpr size_t size = Res::static_size;
        auto selector = [](auto &a, size_t i)->auto& {
            if constexpr (std::remove_reference<decltype(a)>::type::total_size == Res::total_size) {
                return a[i];
            } else {
                return a;
            }
        };
        for (size_t i = 0; i < size; ++i) {
            __apply_func(res[i], f, exe, selector(args, i)...);
        }
    }
}

template<class... Args>
inline void apply_func(Args&&... args) {
    static_assert(sizeof...(Args) > 1);
    if constexpr (sizeof...(Args) == 2) {
        __apply_func(args..., StaticExecuter<1>::object);
    } else {
        auto internal = []<class Arg1, class Arg2, class Arg3, class... OtherArgs>(
            auto internal,
            Arg1 &&arg1,
            Arg2 &&arg2,
            Arg3 &&arg3,
            OtherArgs&&... other_args)
        {
            if constexpr (!IsTensor<typename std::remove_reference<Arg2>::type>::value) {
                if constexpr (!IsTensor<typename std::remove_reference<Arg3>::type>::value) {
                    __apply_func(arg1, arg2, arg3, other_args...);
                } else {
                    __apply_func(arg1, arg2, StaticExecuter<1>::object, arg3, other_args...);
                }
            } else {
                internal(internal, arg2, arg3, other_args..., arg1);
            }
        };
        internal(internal, args...);
    }
}

template<class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
inline void assign_(const Tensor<T, dims2...> &b, Tensor<T, dims1...> &a, Executer &exe = Executer::object) {
    apply_func(b, a, [](T x, T y) {
        return x;
    }, exe);
}

template<class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
inline void add_(const Tensor<T, dims2...> &b, Tensor<T, dims1...> &a, Executer &exe = Executer::object) {
    apply_func(b, a, [](T x, T y) {
        return x + y;
    }, exe);
}

template<class T, size_t... dims1, size_t... dims2>
inline Tensor<T, dims1...> &operator+=(Tensor<T, dims1...> &a, const Tensor<T, dims2...> &b) {
    add_(b, a);
    return a;
}

template<class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
inline void sub_(const Tensor<T, dims2...> &b, Tensor<T, dims1...> &a, Executer &exe = Executer::object) {
    apply_func(b, a, [](T x, T y) {
        return y - x;
    }, exe);
}

template<class T, size_t... dims1, size_t... dims2>
inline Tensor<T, dims1...> &operator-=(Tensor<T, dims1...> &a, const Tensor<T, dims2...> &b) {
    sub_(b, a);
    return a;
}

template<class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
inline void mul_(const Tensor<T, dims2...> &b, Tensor<T, dims1...> &a, Executer &exe = Executer::object) {
    apply_func(b, a, [](T x, T y) {
        return x * y;
    }, exe);
}

template<class T, size_t... dims1, size_t... dims2>
inline Tensor<T, dims1...> &operator*=(Tensor<T, dims1...> &a, const Tensor<T, dims2...> &b) {
    mul_(b, a);
    return a;
}

template<class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
inline void div_(const Tensor<T, dims2...> &b, Tensor<T, dims1...> &a, Executer &exe = Executer::object) {
    apply_func(b, a, [](T x, T y) {
        return y / x;
    }, exe);
}

template<class T, size_t... dims1, size_t... dims2>
inline Tensor<T, dims1...> &operator/=(Tensor<T, dims1...> &a, const Tensor<T, dims2...> &b) {
    div_(b, a);
    return a;
}

template<class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
inline void mod_(const Tensor<T, dims2...> &b, Tensor<T, dims1...> &a, Executer &exe = Executer::object) {
    apply_func(b, a, [](T x, T y) {
        return y % x;
    }, exe);
}

template<class T, size_t... dims1, size_t... dims2>
inline Tensor<T, dims1...> &operator%=(Tensor<T, dims1...> &a, const Tensor<T, dims2...> &b) {
    mod_(b, a);
    return a;
}

// op= operations for scalar

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void add_(T b, Tensor<T, dims...> &a, Executer &exe = Executer::object) {
    add_(Tensor<T>(b), a, exe);
}

template<class T, size_t... dims>
inline Tensor<T, dims...> &operator+=(Tensor<T, dims...> &a, T b) {
    add_(b, a);
    return a;
}

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void sub_(T b, Tensor<T, dims...> &a, Executer &exe = Executer::object) {
    sub_(Tensor<T>(b), a, exe);
}

template<class T, size_t... dims>
inline Tensor<T, dims...> &operator-=(Tensor<T, dims...> &a, T b) {
    sub_(b, a);
    return a;
}

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void mul_(T b, Tensor<T, dims...> &a, Executer &exe = Executer::object) {
    mul(Tensor<T>(b), a, exe);
}

template<class T, size_t... dims>
inline Tensor<T, dims...> &operator*=(Tensor<T, dims...> &a, T b) {
    mul_(b, a);
    return a;
}

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void div_(T b, Tensor<T, dims...> &a, Executer &exe = Executer::object) {
    div_(Tensor<T>(b), a, exe);
}

template<class T, size_t... dims>
inline Tensor<T, dims...> &operator/=(Tensor<T, dims...> &a, T b) {
    div_(b, a);
    return a;
}

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void mod_(T b, Tensor<T, dims...> &a, Executer &exe = Executer::object) {
    mod_(Tensor<T>(b), a, exe);
}

template<class T, size_t... dims>
inline Tensor<T, dims...> &operator%=(Tensor<T, dims...> &a, T b) {
    mod_(b, a);
    return a;
}

// -----

enum CalculationMode {
    ASSIGN,
    ADD
};

// reduction

template<CalculationMode mode = ASSIGN, class T, size_t... dims1, size_t... dims2, class Executer = StaticExecuter<1>>
inline void sum(const Tensor<T, dims1...> &a, Tensor<T, dims2...> &ans, Executer &exe = Executer::object) {
    if constexpr (sizeof...(dims2) == 0) {
        constexpr size_t size = sizeof(typename std::remove_reference<decltype(a)>::type) / sizeof(T);
        constexpr size_t block = size / Executer::static_thread_count;
        std::atomic<T> result = 0;
        if constexpr (block) {
            exe.run([&a, &ans, &result](size_t t) {
                t *= block;
                T res = 0;
                #pragma GCC ivdep
                for (size_t i = t; i < t + block; ++i) {
                    res += a.data()[i];
                }
                result.fetch_add(res);
            });
        }
        T res = result;
        #pragma GCC ivdep
        for (size_t i = size - (block ? size % block : size); i < size; ++i) {
            res += a.data()[i];
        }
        if constexpr (mode == ASSIGN)
            ans = res;
        else
            ans += res;
    } else {
        static_assert(sizeof...(dims1) > sizeof...(dims2));
        static_assert(Tensor<T, dims1...>::static_size == Tensor<T, dims2...>::static_size);
        constexpr size_t size = Tensor<T, dims1...>::static_size;
        constexpr size_t block = size / Executer::static_thread_count;
        if constexpr (block) {
            exe.run([&a, &ans](size_t t) {
                t *= block;
                for (size_t i = t; i < t + block; ++i) {
                    sum(a[i], ans[i]);
                }
            });
        }
        for (size_t i = size - (block ? size % block : size); i < size; ++i) {
            sum(a[i], ans[i]);
        }
    }
}

// -----

template<CalculationMode mode = ASSIGN, class T = void, size_t n = 0, size_t m = 0, class Executer = StaticExecuter<1>>
inline void transpose(
    const Tensor<T, m, n> &a,
    Tensor<T, n, m> &ans,
    Executer &exe = Executer::object)
{
    constexpr size_t block = n / Executer::static_thread_count;
    if constexpr (block) {
        exe.run([&a, &ans](size_t t) {
            t *= block;
            for (size_t i = t; i < t + block; ++i) {
                #pragma GCC ivdep
                for (size_t j = 0; j < m; ++j) {
                    if constexpr (mode == ASSIGN)
                        ans[i][j] = a[j][i];
                    else
                        ans[i][j] += a[j][i];
                }
            }
        });
    }
    for (size_t i = n - (block ? n % block : n); i < n; ++i) {
        #pragma GCC ivdep
        for (size_t j = 0; j < m; ++j) {
            if constexpr (mode == ASSIGN)
                ans[i][j] = a[j][i];
            else
                ans[i][j] += a[j][i];
        }
    }
} 

template<CalculationMode mode = ASSIGN, class T = void, size_t n = 0, size_t m = 0, size_t k = 0, class Executer = StaticExecuter<1>>
inline void matmul_transposed(
    const Tensor<T, n, m> &a,
    const Tensor<T, k, m> &b,
    Tensor<T, n, k> &ans,
    Executer &exe = Executer::object)
{
    if constexpr (n == 1 && k != 1) {
        matmul_transposed(b, a, ans.template view<k, n>(), exe);
    } else {
        constexpr size_t block = n / Executer::static_thread_count;
        if constexpr (block) {
            exe.run([&a, &b, &ans](size_t t) {
                t *= block;
                #pragma GCC ivdep
                for (size_t i = t; i < t + block; ++i) {
                    #pragma GCC ivdep
                    for (size_t j = 0; j < k; ++j) {
                        T sum = 0;
                        #pragma GCC ivdep
                        for (size_t l = 0; l < m; ++l) {
                            sum += a[i][l] * b[j][l];
                        }
                        if constexpr (mode == ASSIGN)
                            ans[i][j] = sum;
                        else
                            ans[i][j] += sum;
                    }
                }
            });
        }
        #pragma GCC ivdep
        for (size_t i = n - (block ? n % block : n); i < n; ++i) {
            #pragma GCC ivdep
            for (size_t j = 0; j < k; ++j) {
                T sum = 0;
                #pragma GCC ivdep
                for (size_t l = 0; l < m; ++l) {
                    sum += a[i][l] * b[j][l];
                }
                if constexpr (mode == ASSIGN)
                    ans[i][j] = sum;
                else
                    ans[i][j] += sum;
            }
        }
    }
}

template<CalculationMode mode = ASSIGN, class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void exp(const Tensor<T, dims...> &a, Tensor<T, dims...> &ans, Executer &exe = Executer::object) {
    apply_func(a, ans, [] (auto x, auto y) {
        if constexpr (mode == ASSIGN)
            return std::exp(x);
        else
            return y + std::exp(x);
    }, exe);
}

template<CalculationMode mode = ASSIGN, class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void log(const Tensor<T, dims...> &a, Tensor<T, dims...> &ans, Executer &exe = Executer::object) {
    apply_func(a, ans, [] (auto x, auto y) {
        if constexpr (mode == ASSIGN)
            return std::log(x);
        else
            return y + std::log(x);
    }, exe);
}

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void fast_tanh(const Tensor<T, dims...> &a, Tensor<T, dims...> &ans, Executer &exe = Executer::object) {
    apply_func(a, ans, [](T x, T y) {
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
    }, exe);
}

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void fast_sigmoid(const Tensor<T, dims...> &a, Tensor<T, dims...> &ans, Executer &exe = Executer::object) {
    apply_func(a, ans, [](T x, T y) {
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
    }, exe);
}

template<class T, size_t... dims, class Executer = StaticExecuter<1>>
inline void relu(const Tensor<T, dims...> &a, Tensor<T, dims...> &ans, Executer &exe = Executer::object) {
    apply_func(a, ans, [](T x, T y) {
        return x <= 0 ? 0 : x;
    }, exe);
}

};