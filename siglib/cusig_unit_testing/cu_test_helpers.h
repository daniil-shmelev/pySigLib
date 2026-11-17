/* Copyright 2026 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once

#include <gtest/gtest.h>
#include "cusig.h"
#include "cuda_runtime.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#define EPSILON 1e-10
#define SINGLE_EPSILON 1e-4
#define DOUBLE_EPSILON 1e-10
#define TYPED_EPSILON(T) (std::is_same_v<T, float> ? SINGLE_EPSILON : DOUBLE_EPSILON)

inline bool is_prime_(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

inline int64_t mobius_(uint64_t N) {
    if (N == 1) return 1;
    uint64_t p = 0;
    for (uint64_t i = 2; i <= N; ++i) {
        if (N % i == 0 && is_prime_(i)) {
            if (N % (i * i) == 0) return 0;
            ++p;
        }
    }
    return (p % 2 == 0) ? 1 : -1;
}

inline uint64_t log_sig_length_(uint64_t dimension, uint64_t degree) {
    if (!dimension || !degree) return 0;
    uint64_t result = 0;
    for (uint64_t k = 1; k <= degree; ++k) {
        int64_t k_sum = 0;
        for (uint64_t d = 1; d <= k; ++d) {
            if (k % d != 0) continue;
            uint64_t p = 1;
            for (uint64_t j = 0; j < d; ++j) p *= dimension;
            int64_t m = mobius_(k / d);
            k_sum += m * static_cast<int64_t>(p);
        }
        result += static_cast<uint64_t>(k_sum / static_cast<int64_t>(k));
    }
    return result;
}

inline double dot_product(double* a, double* b, uint64_t N) {
    double out = 0;
    for (int i = 0; i < N; ++i)
        out += a[i] * b[i];
    return out;
}

inline void gram_(
    double* path1,
    double* path2,
    double* out,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2
) {
    double* out_ptr = out;

    uint64_t flat_path1_length = length1 * dimension;
    uint64_t flat_path2_length = length2 * dimension;

    double* path1_start = path1;
    double* path1_end = path1 + flat_path1_length;

    double* path2_start = path2;
    double* path2_end = path2 + flat_path2_length;

    for (uint64_t b = 0; b < batch_size; ++b) {

        for (double* path1_ptr = path1_start; path1_ptr < path1_end - dimension; path1_ptr += dimension) {
            for (double* path2_ptr = path2_start; path2_ptr < path2_end - dimension; path2_ptr += dimension) {
                *(out_ptr++) = dot_product(path1_ptr + dimension, path2_ptr + dimension, dimension)
                    - dot_product(path1_ptr + dimension, path2_ptr, dimension)
                    - dot_product(path1_ptr, path2_ptr + dimension, dimension)
                    + dot_product(path1_ptr, path2_ptr, dimension);
            }
        }

        path1_start += flat_path1_length;
        path1_end += flat_path1_length;
        path2_start += flat_path2_length;
        path2_end += flat_path2_length;
    }
}

inline std::vector<int> int_test_data(uint64_t dimension, uint64_t length) {
    std::vector<int> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(i);
    }
    return data;
}

template<typename FN, typename T, typename... Args>
void check_result(FN f, std::vector<T>& path, std::vector<double>& true_, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    T* d_a;
    double* d_out;
    cudaMalloc(&d_a, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);

    f(d_a, d_out, args...);

    cudaMemcpy(out.data(), d_out, sizeof(double) * true_.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_out);

    for (uint64_t i = 0; i < true_.size(); ++i)
        EXPECT_TRUE(abs(true_[i] - out[i]) < EPSILON);

    EXPECT_TRUE(abs(-1. - out[true_.size()]) < EPSILON);
}

template<typename FN, typename T, typename... Args>
void check_result_2(FN f, std::vector<T>& path1, std::vector<T>& path2, std::vector<double>& true_, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    T* d_a, * d_b;
    double * d_out;
    cudaMalloc(&d_a, sizeof(T) * path1.size());
    cudaMalloc(&d_b, sizeof(T) * path2.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, path1.data(), sizeof(T) * path1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, path2.data(), sizeof(T) * path2.size(), cudaMemcpyHostToDevice);

    f(d_a, d_b, d_out, args...);

    cudaMemcpy(out.data(), d_out, sizeof(double) * true_.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    for (uint64_t i = 0; i < true_.size(); ++i)
        EXPECT_TRUE(abs(true_[i] - out[i]) < EPSILON);

    EXPECT_TRUE(abs(-1. - out[true_.size()]) < EPSILON);
}

template<typename FN, typename T, typename... Args>
void check_result_4(FN f, std::vector<T>& path, std::vector<double>& true_, std::vector<double>& deriv, std::vector<double>& k_grid, Args... args) {
    std::vector<double> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    T* d_a;
    double* d_out;
    double* d_deriv;
    double* d_k_grid;
    cudaMalloc(&d_a, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());
    cudaMalloc(&d_deriv, sizeof(double) * deriv.size());
    cudaMalloc(&d_k_grid, sizeof(double) * k_grid.size());

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deriv, deriv.data(), sizeof(double) * deriv.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_grid, k_grid.data(), sizeof(double) * k_grid.size(), cudaMemcpyHostToDevice);

    f(d_a, d_out, d_deriv, d_k_grid, args...);

    cudaMemcpy(out.data(), d_out, sizeof(double) * true_.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_deriv);
    cudaFree(d_k_grid);

    for (uint64_t i = 0; i < true_.size(); ++i)
        EXPECT_TRUE(abs(true_[i] - out[i]) < EPSILON);

    EXPECT_TRUE(abs(-1. - out[true_.size()]) < EPSILON);
}

template<typename FN, typename T, typename... Args>
std::vector<double> run_backprop_cuda(FN f, std::vector<T>& gram, uint64_t out_size, std::vector<double>& deriv, std::vector<double>& k_grid, Args... args) {
    std::vector<double> out(out_size, 0.);

    T* d_gram;
    double* d_out;
    double* d_deriv;
    double* d_k_grid;
    cudaMalloc(&d_gram, sizeof(T) * gram.size());
    cudaMalloc(&d_out, sizeof(double) * out_size);
    cudaMalloc(&d_deriv, sizeof(double) * deriv.size());
    cudaMalloc(&d_k_grid, sizeof(double) * k_grid.size());

    cudaMemcpy(d_gram, gram.data(), sizeof(T) * gram.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deriv, deriv.data(), sizeof(double) * deriv.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_grid, k_grid.data(), sizeof(double) * k_grid.size(), cudaMemcpyHostToDevice);

    f(d_gram, d_out, d_deriv, d_k_grid, args...);

    cudaMemcpy(out.data(), d_out, sizeof(double) * out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_gram);
    cudaFree(d_out);
    cudaFree(d_deriv);
    cudaFree(d_k_grid);

    return out;
}

inline uint64_t sig_length_(uint64_t dimension, uint64_t degree) {
    if (dimension == 0) return 1;
    uint64_t result = 1;
    uint64_t power = 1;
    for (uint64_t i = 0; i < degree; ++i) {
        power *= dimension;
        result += power;
    }
    return result;
}

template<typename FN, typename T, typename... Args>
void check_result_typed(FN f, std::vector<T>& path, std::vector<T>& true_, Args... args) {
    std::vector<T> out;
    out.resize(true_.size() + 1);
    out[true_.size()] = static_cast<T>(-1.);

    T* d_path = nullptr;
    T* d_out = nullptr;
    if (path.size() > 0)
        cudaMalloc(&d_path, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(T) * out.size());

    // Copy sentinel to device so we can check it later
    cudaMemcpy(d_out, out.data(), sizeof(T) * out.size(), cudaMemcpyHostToDevice);

    if (path.size() > 0)
        cudaMemcpy(d_path, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);

    int err;
    if constexpr (std::is_invocable_v<FN, T*, T*, Args...>)
        err = f(d_path, d_out, args...);
    else if constexpr (std::is_invocable_v<FN, T*, T*, Args..., const T*, uint64_t>)
        err = f(d_path, d_out, args..., nullptr, (uint64_t)0);
    else
        err = f(d_path, d_out, args..., nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

    if (d_path) cudaFree(d_path);
    cudaFree(d_out);

    EXPECT_EQ(0, err) << "Signature function returned non-zero error code";

    const double eps = TYPED_EPSILON(T);
    for (uint64_t i = 0; i < true_.size(); ++i) {
        std::string msg = "Mismatch at index " + std::to_string(i) +
            ": expected " + std::to_string(static_cast<double>(true_[i])) +
            " got " + std::to_string(static_cast<double>(out[i]));
        EXPECT_TRUE(std::abs(static_cast<double>(true_[i]) - static_cast<double>(out[i])) < eps) << msg;
    }

    EXPECT_TRUE(std::abs(-1. - static_cast<double>(out[true_.size()])) < eps) << "Sentinel value was overwritten";
}

template<typename FN, typename T, typename... Args>
void check_result_2_typed(FN f, std::vector<T>& input1, std::vector<T>& input2, std::vector<T>& true_, Args... args) {
    std::vector<T> out;
    out.resize(true_.size() + 1);
    out[true_.size()] = static_cast<T>(-1.);

    T* d_input1 = nullptr;
    T* d_input2 = nullptr;
    T* d_out = nullptr;
    if (input1.size() > 0)
        cudaMalloc(&d_input1, sizeof(T) * input1.size());
    if (input2.size() > 0)
        cudaMalloc(&d_input2, sizeof(T) * input2.size());
    cudaMalloc(&d_out, sizeof(T) * out.size());

    // Copy sentinel to device so we can check it later
    cudaMemcpy(d_out, out.data(), sizeof(T) * out.size(), cudaMemcpyHostToDevice);

    if (input1.size() > 0)
        cudaMemcpy(d_input1, input1.data(), sizeof(T) * input1.size(), cudaMemcpyHostToDevice);
    if (input2.size() > 0)
        cudaMemcpy(d_input2, input2.data(), sizeof(T) * input2.size(), cudaMemcpyHostToDevice);

    int err = f(d_input1, d_input2, d_out, args...);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

    if (d_input1) cudaFree(d_input1);
    if (d_input2) cudaFree(d_input2);
    cudaFree(d_out);

    EXPECT_EQ(0, err) << "sig_combine_cuda returned non-zero error code";

    const double eps = TYPED_EPSILON(T);
    for (uint64_t i = 0; i < true_.size(); ++i) {
        std::string msg = "Mismatch at index " + std::to_string(i) +
            ": expected " + std::to_string(static_cast<double>(true_[i])) +
            " got " + std::to_string(static_cast<double>(out[i]));
        EXPECT_TRUE(std::abs(static_cast<double>(true_[i]) - static_cast<double>(out[i])) < eps) << msg;
    }

    EXPECT_TRUE(std::abs(-1. - static_cast<double>(out[true_.size()])) < eps) << "Sentinel value was overwritten";
}

template<typename T>
std::vector<T> compute_sig_on_gpu(const std::vector<T>& path, uint64_t dimension, uint64_t length, uint64_t degree) {
    uint64_t sig_len = sig_length_(dimension, degree);
    std::vector<T> sig(sig_len);

    T* d_path = nullptr;
    T* d_out = nullptr;
    cudaMalloc(&d_path, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(T) * sig_len);
    cudaMemcpy(d_path, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);

    int err;
    if constexpr (std::is_same_v<T, float>)
        err = signature_cuda_f(d_path, d_out, (uint64_t)1, dimension, length, degree, false, false, 1.f, true);
    else
        err = signature_cuda_d(d_path, d_out, (uint64_t)1, dimension, length, degree, false, false, 1., true);
    cudaDeviceSynchronize();

    cudaMemcpy(sig.data(), d_out, sizeof(T) * sig_len, cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);

    EXPECT_EQ(0, err) << "signature_cuda returned non-zero error code in helper";
    return sig;
}

template<typename FN, typename T>
void check_backprop_result(
    FN f,
    std::vector<T>& path,
    std::vector<T>& sig,
    std::vector<T>& sig_derivs,
    std::vector<T>& expected_out,
    uint64_t dimension, uint64_t length, uint64_t degree,
    bool time_aug, bool lead_lag, T end_time
) {
    std::vector<T> out(expected_out.size() + 1);
    out[expected_out.size()] = static_cast<T>(-1.);

    T* d_path = nullptr;
    T* d_out = nullptr;
    T* d_sig = nullptr;
    T* d_sig_derivs = nullptr;

    if (path.size() > 0)
        cudaMalloc(&d_path, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(T) * out.size());
    cudaMalloc(&d_sig, sizeof(T) * sig.size());
    cudaMalloc(&d_sig_derivs, sizeof(T) * sig_derivs.size());

    cudaMemcpy(d_out, out.data(), sizeof(T) * out.size(), cudaMemcpyHostToDevice);

    if (path.size() > 0)
        cudaMemcpy(d_path, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig.data(), sizeof(T) * sig.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_derivs, sig_derivs.data(), sizeof(T) * sig_derivs.size(), cudaMemcpyHostToDevice);

    int err;
    if constexpr (std::is_invocable_v<FN, T*, T*, T*, T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool>)
        err = f(d_path, d_out, d_sig_derivs, d_sig, (uint64_t)1, dimension, length, degree, time_aug, lead_lag, end_time, true);
    else if constexpr (std::is_invocable_v<FN, T*, T*, T*, T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, const T*, uint64_t>)
        err = f(d_path, d_out, d_sig_derivs, d_sig, (uint64_t)1, dimension, length, degree, time_aug, lead_lag, end_time, true, nullptr, (uint64_t)0);
    else
        err = f(d_path, d_out, d_sig_derivs, d_sig, (uint64_t)1, dimension, length, degree, time_aug, lead_lag, end_time, true, nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

    if (d_path) cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_sig);
    cudaFree(d_sig_derivs);

    EXPECT_EQ(0, err) << "sig_backprop returned non-zero error code";

    const double eps = TYPED_EPSILON(T);
    for (uint64_t i = 0; i < expected_out.size(); ++i) {
        std::string msg = "Backprop mismatch at index " + std::to_string(i) +
            ": expected " + std::to_string(static_cast<double>(expected_out[i])) +
            " got " + std::to_string(static_cast<double>(out[i]));
        EXPECT_TRUE(std::abs(static_cast<double>(expected_out[i]) - static_cast<double>(out[i])) < eps) << msg;
    }

    EXPECT_TRUE(std::abs(-1. - static_cast<double>(out[expected_out.size()])) < eps) << "Sentinel value was overwritten";
}

template<typename FN, typename T>
void check_batch_backprop_result(
    FN f,
    std::vector<T>& path,
    std::vector<T>& sig,
    std::vector<T>& sig_derivs,
    std::vector<T>& expected_out,
    uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree,
    bool time_aug, bool lead_lag, T end_time
) {
    std::vector<T> out(expected_out.size() + 1);
    out[expected_out.size()] = static_cast<T>(-1.);

    T* d_path = nullptr;
    T* d_out = nullptr;
    T* d_sig = nullptr;
    T* d_sig_derivs = nullptr;

    if (path.size() > 0)
        cudaMalloc(&d_path, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(T) * out.size());
    cudaMalloc(&d_sig, sizeof(T) * sig.size());
    cudaMalloc(&d_sig_derivs, sizeof(T) * sig_derivs.size());

    cudaMemcpy(d_out, out.data(), sizeof(T) * out.size(), cudaMemcpyHostToDevice);

    if (path.size() > 0)
        cudaMemcpy(d_path, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig.data(), sizeof(T) * sig.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_derivs, sig_derivs.data(), sizeof(T) * sig_derivs.size(), cudaMemcpyHostToDevice);

    int err;
    if constexpr (std::is_invocable_v<FN, T*, T*, T*, T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool>)
        err = f(d_path, d_out, d_sig_derivs, d_sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, true);
    else if constexpr (std::is_invocable_v<FN, T*, T*, T*, T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, const T*, uint64_t>)
        err = f(d_path, d_out, d_sig_derivs, d_sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, true, nullptr, (uint64_t)0);
    else
        err = f(d_path, d_out, d_sig_derivs, d_sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time, true, nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

    if (d_path) cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_sig);
    cudaFree(d_sig_derivs);

    EXPECT_EQ(0, err) << "sig_backprop returned non-zero error code";

    const double eps = TYPED_EPSILON(T);
    for (uint64_t i = 0; i < expected_out.size(); ++i) {
        std::string msg = "Batch backprop mismatch at index " + std::to_string(i) +
            ": expected " + std::to_string(static_cast<double>(expected_out[i])) +
            " got " + std::to_string(static_cast<double>(out[i]));
        EXPECT_TRUE(std::abs(static_cast<double>(expected_out[i]) - static_cast<double>(out[i])) < eps) << msg;
    }

    EXPECT_TRUE(std::abs(-1. - static_cast<double>(out[expected_out.size()])) < eps) << "Sentinel value was overwritten";
}

template<typename FN, typename T, typename... Args>
void check_result_backprop_typed(FN f, std::vector<T>& sig, std::vector<T>& true_, std::vector<T>& derivs, Args... args) {
    std::vector<T> out;
    out.resize(true_.size() + 1);
    out[true_.size()] = static_cast<T>(-1.);

    T* d_sig = nullptr;
    T* d_out = nullptr;
    T* d_derivs = nullptr;
    cudaMalloc(&d_sig, sizeof(T) * sig.size());
    cudaMalloc(&d_out, sizeof(T) * out.size());
    cudaMalloc(&d_derivs, sizeof(T) * derivs.size());

    cudaMemcpy(d_out, out.data(), sizeof(T) * out.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig.data(), sizeof(T) * sig.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), sizeof(T) * derivs.size(), cudaMemcpyHostToDevice);

    int err = f(d_sig, d_out, d_derivs, args...);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_sig);
    cudaFree(d_out);
    cudaFree(d_derivs);

    EXPECT_EQ(0, err) << "Backprop function returned non-zero error code";

    const double eps = TYPED_EPSILON(T);
    for (uint64_t i = 0; i < true_.size(); ++i) {
        std::string msg = "Mismatch at index " + std::to_string(i) +
            ": expected " + std::to_string(static_cast<double>(true_[i])) +
            " got " + std::to_string(static_cast<double>(out[i]));
        EXPECT_TRUE(std::abs(static_cast<double>(true_[i]) - static_cast<double>(out[i])) < eps) << msg;
    }

    EXPECT_TRUE(std::abs(-1. - static_cast<double>(out[true_.size()])) < eps) << "Sentinel value was overwritten";
}

template<typename FN, typename T, typename... Args>
void check_result_batch_backprop_typed(FN f, std::vector<T>& sig, std::vector<T>& true_, std::vector<T>& derivs, Args... args) {
    check_result_backprop_typed(f, sig, true_, derivs, args...);
}

template<typename T>
std::vector<T> compute_batch_sig_on_gpu(const std::vector<T>& path, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree) {
    uint64_t sig_len = sig_length_(dimension, degree) * batch_size;
    std::vector<T> sig(sig_len);

    T* d_path = nullptr;
    T* d_out = nullptr;
    cudaMalloc(&d_path, sizeof(T) * path.size());
    cudaMalloc(&d_out, sizeof(T) * sig_len);
    cudaMemcpy(d_path, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);

    int err;
    if constexpr (std::is_same_v<T, float>)
        err = signature_cuda_f(d_path, d_out, batch_size, dimension, length, degree, false, false, 1.f, true);
    else
        err = signature_cuda_d(d_path, d_out, batch_size, dimension, length, degree, false, false, 1., true);
    cudaDeviceSynchronize();

    cudaMemcpy(sig.data(), d_out, sizeof(T) * sig_len, cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);

    EXPECT_EQ(0, err) << "signature_cuda returned non-zero error code in helper";
    return sig;
}
