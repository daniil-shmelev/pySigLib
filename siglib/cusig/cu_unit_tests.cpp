/* Copyright 2025 Daniil Shmelev
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

#include "CppUnitTest.h"
#include "cusig.h"
#include "cuda_runtime.h"
#include <vector>
#include <cmath>


#define EPSILON 1e-10
#define SINGLE_EPSILON 1e-4
#define DOUBLE_EPSILON 1e-10
#define TYPED_EPSILON(T) (std::is_same_v<T, float> ? SINGLE_EPSILON : DOUBLE_EPSILON)

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

double dot_product(double* a, double* b, uint64_t N) {
    double out = 0;
    for (int i = 0; i < N; ++i)
        out += a[i] * b[i];
    return out;
}

void gram_(
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


std::vector<int> int_test_data(uint64_t dimension, uint64_t length) {
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
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
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
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
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
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
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

// Helper: compute sig_length on the host (not exported from cusig)
static uint64_t sig_length_(uint64_t dimension, uint64_t degree) {
    if (dimension == 0) return 1;
    uint64_t result = 1;
    uint64_t power = 1;
    for (uint64_t i = 0; i < degree; ++i) {
        power *= dimension;
        result += power;
    }
    return result;
}

// Typed check_result: input type T, output type T (for signature_cuda_f / signature_cuda_d)
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

    int err = f(d_path, d_out, args...);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

    if (d_path) cudaFree(d_path);
    cudaFree(d_out);

    Assert::AreEqual(0, err, L"Signature function returned non-zero error code");

    const double eps = TYPED_EPSILON(T);
    for (uint64_t i = 0; i < true_.size(); ++i) {
        std::wstring msg = L"Mismatch at index " + std::to_wstring(i) +
            L": expected " + std::to_wstring(static_cast<double>(true_[i])) +
            L" got " + std::to_wstring(static_cast<double>(out[i]));
        Assert::IsTrue(std::abs(static_cast<double>(true_[i]) - static_cast<double>(out[i])) < eps, msg.c_str());
    }

    Assert::IsTrue(std::abs(-1. - static_cast<double>(out[true_.size()])) < eps, L"Sentinel value was overwritten");
}

// Typed check_result_2: two device inputs, output type T (for sig_combine_cuda_f / sig_combine_cuda_d)
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

    Assert::AreEqual(0, err, L"sig_combine_cuda returned non-zero error code");

    const double eps = TYPED_EPSILON(T);
    for (uint64_t i = 0; i < true_.size(); ++i) {
        std::wstring msg = L"Mismatch at index " + std::to_wstring(i) +
            L": expected " + std::to_wstring(static_cast<double>(true_[i])) +
            L" got " + std::to_wstring(static_cast<double>(out[i]));
        Assert::IsTrue(std::abs(static_cast<double>(true_[i]) - static_cast<double>(out[i])) < eps, msg.c_str());
    }

    Assert::IsTrue(std::abs(-1. - static_cast<double>(out[true_.size()])) < eps, L"Sentinel value was overwritten");
}

// Helper: compute signature on GPU and return host vector
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
        err = signature_cuda_f(d_path, d_out, dimension, length, degree, false, false, 1.f, true);
    else
        err = signature_cuda_d(d_path, d_out, dimension, length, degree, false, false, 1., true);
    cudaDeviceSynchronize();

    cudaMemcpy(sig.data(), d_out, sizeof(T) * sig_len, cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);

    Assert::AreEqual(0, err, L"signature_cuda returned non-zero error code in helper");
    return sig;
}

// Helper: compute batch signature on GPU and return host vector
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
        err = batch_signature_cuda_f(d_path, d_out, batch_size, dimension, length, degree, false, false, 1.f, true);
    else
        err = batch_signature_cuda_d(d_path, d_out, batch_size, dimension, length, degree, false, false, 1., true);
    cudaDeviceSynchronize();

    cudaMemcpy(sig.data(), d_out, sizeof(T) * sig_len, cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);

    Assert::AreEqual(0, err, L"batch_signature_cuda returned non-zero error code in helper");
    return sig;
}

namespace MyTest
{
    TEST_CLASS(sigKernelTest) {
public:

    TEST_METHOD(Trivial) {
        auto f = sig_kernel_cuda_d;
        uint64_t dimension = 1, length = 1, batch_size = 1;
        std::vector<double> path = { 0. };
        std::vector<double> true_sig = { 1. };
        std::vector<double> gram = {};
        check_result(f, gram, true_sig, dimension, length, length, 0, 0, false);
    }

    TEST_METHOD(TrivialBatch) {
        auto f = batch_sig_kernel_cuda_d;
        uint64_t dimension = 1, length = 1, batch_size = 5;
        std::vector<double> path = { 0. };
        std::vector<double> true_sig = { 1., 1., 1., 1., 1. };
        std::vector<double> gram = {};
        check_result(f, gram, true_sig, batch_size, dimension, length, length, 0, 0, false);
    }
    TEST_METHOD(LinearPathTest) {
        auto f = sig_kernel_cuda_d;
        uint64_t dimension = 2, length = 3;
        std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
        std::vector<double> true_sig = { 4.256702149748847 };
        std::vector<double> gram(length * length);
        gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
        check_result(f, gram, true_sig, dimension, length, length, 2, 2, false);
    }

    TEST_METHOD(ManualTest) {
        auto f = sig_kernel_cuda_d;
        uint64_t dimension = 3, length = 4;
        std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
        std::vector<double> true_sig = { 2.1529809076880486 };
        std::vector<double> gram(length * length);
        gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
        check_result(f, gram, true_sig, dimension, length, length, 2, 2, false);
    }

    TEST_METHOD(NonSquare1) {
        auto f = sig_kernel_cuda_d;
        uint64_t dimension = 1, length1 = 3, length2 = 2;
        std::vector<double> path1 = { 0., 1., 2. };
        std::vector<double> path2 = { 0., 2. };
        std::vector<double> true_sig = { 11. };
        std::vector<double> gram(length1 * length2);
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(NonSquare2) {
        auto f = sig_kernel_cuda_d;
        uint64_t dimension = 1, length1 = 2, length2 = 3;
        std::vector<double> path2 = { 0., 1., 2. };
        std::vector<double> path1 = { 0., 2. };
        std::vector<double> true_sig = { 11. };
        std::vector<double> gram(length1 * length2);
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(FullGrid) {
        auto f = sig_kernel_cuda_d;
        uint64_t dimension = 1, length1 = 3, length2 = 2;
        std::vector<double> path1 = { 0., 1., 2. };
        std::vector<double> path2 = { 0., 2. };
        std::vector<double> true_sig = { 1., 1.,
            1., 4.,
            1., 11. };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, true);
    }

    TEST_METHOD(FullGrid2) {
        auto f = batch_sig_kernel_cuda_d;
        uint64_t dimension = 1, length1 = 3, length2 = 2, batch_size = 2;
        std::vector<double> path1 = { 0., 1., 2., 0., 1., 2.};
        std::vector<double> path2 = { 0., 2., 0., 2. };
        std::vector<double> true_sig = { 1., 1.,
            1., 4.,
            1., 11.,
            1., 1.,
            1., 4.,
            1., 11.};
        std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
        gram_(path1.data(), path2.data(), gram.data(), batch_size, dimension, length1, length2);
        check_result(f, gram, true_sig, batch_size, dimension, length1, length2, 0, 0, true);
    }

    TEST_METHOD(FullGridLarge) {
        auto f = batch_sig_kernel_cuda_d;
        uint64_t dimension = 1, length1 = 410, length2 = 410, batch_size = 32;
        double* d_gram, * d_out;
        cudaMalloc(&d_gram, sizeof(double) * (length1 - 1) * (length2 - 2) * batch_size);
        cudaMalloc(&d_out, sizeof(double) * length1 * length2 * batch_size);
        f(d_gram, d_out, batch_size, dimension, length1, length2, 0, 0, true);
        cudaFree(d_gram);
        cudaFree(d_out);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            const int error_code = static_cast<int>(err);
            throw std::runtime_error("CUDA Error (" + std::to_string(error_code) + "): " + cudaGetErrorString(err));
        }
    }
    };

    TEST_CLASS(sigKernelBackpropTest) {
public:
    TEST_METHOD(ManualTest1) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length1 = 2, length2 = 3;
        std::vector<double> path1 = { 0., 2. };
        std::vector<double> path2 = { 0., 1., 2. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 4.5 + 1. / 6, 4.5 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11. };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(ManualTest1Extended) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length1 = 34, length2 = 35;
        std::vector<double> path1(length1, 0.);
        path1[length1 - 1] = 2.;
        std::vector<double> path2(length2, 0.);
        path2[length2 - 2] = 1.;
        path2[length2 - 1] = 2.;
        std::vector<double> deriv = { 1. };
        std::vector<double> true_((length1 - 1) * (length2 - 1), 11.); //{ 4.5 + 1. / 6, 4.5 };

        for (uint64_t i = 1; i < length1 - 1; ++i) {
            true_[(length2 - 1) * i - 2] = 7. + 1. / 9;
            true_[(length2 - 1) * i - 1] = 2. + 1. / 3;
        }
        for (uint64_t i = (length1 - 2) * (length2 - 1); i < (length1 - 1) * (length2 - 1) - 2; ++i) {
            true_[i] = 5. + 4. / 9;
        }

        true_[(length1 - 1) * (length2 - 1) - 2] = 4.5 + 1. / 6;
        true_[(length1 - 1) * (length2 - 1) - 1] = 4.5;
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid(length1 * length2, 1.);// = { 1., 1., 1., 1., 4., 11. };
        k_grid[length1 * length2 - 2] = 4.;
        k_grid[length1 * length2 - 1] = 11.;
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(ManualTest1Rev) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length2 = 2, length1 = 3;
        std::vector<double> path2 = { 0., 2. };
        std::vector<double> path1 = { 0., 1., 2. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 4.5 + 1. / 6, 4.5 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = { 1., 1., 1., 4., 1., 11. };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(ManualTest2) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length1 = 3, length2 = 3;
        std::vector<double> path1 = { 0., 2., 3. };
        std::vector<double> path2 = { 0., 1., 2. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6 };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(ManualTest2Rev) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length2 = 3, length1 = 3;
        std::vector<double> path2 = { 0., 2., 3. };
        std::vector<double> path1 = { 0., 1., 2. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 761. / 72, 133. / 24, 7.125, 12.5 + 1. / 6 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = { 1., 1., 1., 1., 4., 7., 1., 11., 25. - 1. / 6 };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(ManualTest3) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length1 = 2, length2 = 3;
        std::vector<double> path1 = { 0., 2. };
        std::vector<double> path2 = { 0., 1., 2. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 5.1602194279800226, 5.1185673607720270 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = {
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.5625,
            2.27734375,
            3.1857910156249996,
            4.3402760823567705,
            1.0,
            2.27734375,
            4.25830078125,
            7.2303009033203125,
            11.584854549831814
        };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 1, 1, false);
    }

    TEST_METHOD(ManualTest3Rev) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length2 = 2, length1 = 3;
        std::vector<double> path2 = { 0., 2. };
        std::vector<double> path1 = { 0., 1., 2. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 5.1602194279800226, 5.1185673607720270 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = {
            1.0,
            1.0,
            1.0,
            1.0,
            1.5625,
            2.27734375,
            1.0,
            2.27734375,
            4.25830078125,
            1.0,
            3.1857910156249996,
            7.2303009033203125,
            1.0,
            4.3402760823567705,
            11.584854549831814
        };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 1, 1, false);
    }

    TEST_METHOD(ManualTest4) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 2, length1 = 3, length2 = 3;
        std::vector<double> path1 = { 0., 1., 2., 4., 5., 5. };
        std::vector<double> path2 = { 0., 2., 1., 3., 2., 1. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 1631. / 72, -437. / 96, 817. / 32, 1049. / 24 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = {
            1.0,
            1.0,
            1.0,
            1.0,
            12.25,
            4.75,
            1.0,
            57.75,
            87.729 + 1. / 6000
        };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(ManualTest4Rev) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 2, length2 = 3, length1 = 3;
        std::vector<double> path2 = { 0., 1., 2., 4., 5., 5. };
        std::vector<double> path1 = { 0., 2., 1., 3., 2., 1. };
        std::vector<double> deriv = { 1. };
        std::vector<double> true_ = { 1631. / 72, 817. / 32 , -437. / 96, 1049. / 24 };
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid = {
            1.0,
            1.0,
            1.0,
            1.0,
            12.25,
            57.75,
            1.0,
            4.75,
            87.729 + 1. / 6000
        };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    /*TEST_METHOD(ManualTest5) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 1, dimension = 1, length1 = 10, length2 = 40;
        std::vector<double> path1(length1);
        for (int i = 0; i < length1; ++i)
            path1[i] = i / 10.;
        std::vector<double> path2(40);
        for (int i = 0; i < length2; ++i)
            path2[i] = i / 10.;
        std::vector<double> deriv = { 1. };
        std::vector<double> true_((length1 - 1) * (length2 - 1));
        std::vector<double> gram((length1 - 1) * (length2 - 1));
        std::vector<double> k_grid(length1 * length2);
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        sig_kernel_cuda_d(gram.data(), k_grid.data(), dimension, length1, length2, 0, 0, true);
        check_result_4(f, gram, true_, deriv, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }*/

    TEST_METHOD(BatchManualTest1) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 2, dimension = 1, length1 = 2, length2 = 3;
        std::vector<double> path1 = { 0., 2., 0., 2. };
        std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
        std::vector<double> derivs = { 1., 1. };
        std::vector<double> true_ = { 4.5 + 1. / 6, 4.5, 4.5 + 1. / 6, 4.5 };
        std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
        std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 1., 1., 1., 4., 11. };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        gram_(path1.data(), path2.data(), gram.data() + 2, 1, dimension, length1, length2);
        check_result_4(f, gram, true_, derivs, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }

    TEST_METHOD(BatchManualTest2) {
        auto f = batch_sig_kernel_backprop_cuda_d;
        uint64_t batch_size = 2, dimension = 1, length1 = 3, length2 = 3;
        std::vector<double> path1 = { 0., 2., 3., 0., 2., 3. };
        std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
        std::vector<double> derivs = { 1., 1. };
        std::vector<double> true_ = { 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6, 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6 };
        std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
        std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6, 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6 };
        gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
        gram_(path1.data(), path2.data(), gram.data() + 4, 1, dimension, length1, length2);
        check_result_4(f, gram, true_, derivs, k_grid, batch_size, dimension, length1, length2, 0, 0, false);
    }
    };

    TEST_CLASS(sigKernelBackpropGridTest) {
    public:
        // When derivs_grid has 1.0 only at [-1,-1] and 0 elsewhere,
        // grid backprop should produce the same result as scalar backprop with deriv=1.0.
        TEST_METHOD(ConsistencyWithScalar) {
            auto f = batch_sig_kernel_backprop_cuda_d;
            uint64_t batch_size = 1, dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            uint64_t out_size = (length1 - 1) * (length2 - 1);

            // Scalar backprop
            std::vector<double> deriv_scalar = { 1. };
            auto out_scalar = run_backprop_cuda(f, gram, out_size, deriv_scalar, k_grid, batch_size, dimension, length1, length2, 0, 0, false);

            // Grid backprop with 1.0 only at [-1,-1]
            uint64_t grid_length = length1 * length2;
            std::vector<double> derivs_grid(grid_length, 0.);
            derivs_grid[grid_length - 1] = 1.0;
            auto out_grid = run_backprop_cuda(f, gram, out_size, derivs_grid, k_grid, batch_size, dimension, length1, length2, 0, 0, true);

            for (uint64_t i = 0; i < out_size; ++i)
                Assert::IsTrue(abs(out_scalar[i] - out_grid[i]) < EPSILON);
        }

        TEST_METHOD(BatchConsistencyWithScalar) {
            auto f = batch_sig_kernel_backprop_cuda_d;
            uint64_t batch_size = 2, dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2., 0., 2. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 2, 1, dimension, length1, length2);
            uint64_t out_size = (length1 - 1) * (length2 - 1) * batch_size;

            // Scalar batch backprop
            std::vector<double> derivs_scalar = { 1., 1. };
            auto out_scalar = run_backprop_cuda(f, gram, out_size, derivs_scalar, k_grid, batch_size, dimension, length1, length2, 0, 0, false);

            // Grid batch backprop
            uint64_t grid_length = length1 * length2;
            std::vector<double> derivs_grid(grid_length * batch_size, 0.);
            derivs_grid[grid_length - 1] = 1.0;
            derivs_grid[2 * grid_length - 1] = 1.0;
            auto out_grid = run_backprop_cuda(f, gram, out_size, derivs_grid, k_grid, batch_size, dimension, length1, length2, 0, 0, true);

            for (uint64_t i = 0; i < out_size; ++i)
                Assert::IsTrue(abs(out_scalar[i] - out_grid[i]) < EPSILON);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_kernel_backprop_cuda_d;
            uint64_t dimension = 2, length = 4;
            std::vector<double> path = { 0., 0., 1., .5, 4., 0., 0., 1. };
            std::vector<double> gram((length - 1) * (length - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 1., 2.640625, 10.571045, 3.154658, 1., 10.571045, 285.859342, 2372.95239, 1., 3.154658, 2372.95239, 165981.889 };
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            uint64_t out_size = (length - 1) * (length - 1);

            std::vector<double> true_ = { 8.0338748831219071, 3.0207107002152322, -0.041744818181351222, 3.0207107002152322, 2.6526166180712516, -1.6587152651909718, -0.041744818181351222, -1.6587152651909718, 1.6629617402333334 };
            uint64_t grid_length = length * length;
            std::vector<double> derivs_grid(grid_length, 0.0001);
            auto out_grid = run_backprop_cuda(f, gram, out_size, derivs_grid, k_grid, dimension, length, length, 0, 0, true);

            for (uint64_t i = 0; i < out_size; ++i)
                Assert::IsTrue(abs(true_[i] - out_grid[i]) < EPSILON);
        }

    };

    TEST_CLASS(transformPathBackprop) {
    public:

        TEST_METHOD(TimeAugTest) {
            auto f = transform_path_backprop_cuda_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((dimension + 1) * length, 1.);
            std::vector<double> true_ = { 1., 1., 1., 1., 1., 1. };
            check_result(f, derivs, true_, dimension, length, true, false, 1.);
        }
        TEST_METHOD(LeadLagTest) {
            auto f = transform_path_backprop_cuda_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs(2 * dimension * (2 * length - 1));
            for (int i = 0; i < derivs.size(); ++i)
                derivs[i] = i;
            std::vector<double> true_ = { 6., 9., 36., 40., 48., 51. };
            check_result(f, derivs, true_, dimension, length, false, true, 1.);
        }

        TEST_METHOD(LeadLagTest2) {
            auto f = transform_path_backprop_cuda_d;
            uint64_t dimension = 5, length = 100;
            std::vector<double> derivs(2 * dimension * (2 * length - 1));
            for (int i = 0; i < derivs.size(); ++i)
                derivs[i] = 1.;
            std::vector<double> true_(dimension * length);
            for (uint64_t i = 0; i < dimension; ++i)
                true_[i] = 3.;
            for (uint64_t i = dimension; i < true_.size() - dimension; ++i)
                true_[i] = 4.;
            for (uint64_t i = true_.size() - dimension; i < true_.size(); ++i)
                true_[i] = 3.;
            check_result(f, derivs, true_, dimension, length, false, true, 1.);
        }

        TEST_METHOD(TimeAugLeadLagTest) {
            auto f = transform_path_backprop_cuda_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((2 * dimension + 1) * (2 * length - 1), 1.);
            std::vector<double> true_ = { 3., 3., 4., 4., 3., 3. };
            check_result(f, derivs, true_, dimension, length, true, true, 1.);
        }
    };

    TEST_CLASS(signatureDoubleTest)
    {
    public:
        TEST_METHOD(TrivialCases) {
            auto f = signature_cuda_d;
            std::vector<double> path;
            std::vector<double> true_sig;
            Assert::AreEqual(2, f(nullptr, nullptr, 0, 0, 0, false, false, 1., true));

            true_sig.push_back(1.);
            check_result_typed(f, path, true_sig, 1, 0, 0, false, false, 1., true);

            path.push_back(0.);
            check_result_typed(f, path, true_sig, 1, 1, 0, false, false, 1., true);

            true_sig.push_back(0.);
            check_result_typed(f, path, true_sig, 1, 0, 1, false, false, 1., true);
            check_result_typed(f, path, true_sig, 1, 1, 1, false, false, 1., true);

            path.push_back(1.);
            true_sig[1] = 1.;
            check_result_typed(f, path, true_sig, 1, 2, 1, false, false, 1., true);
        }

        TEST_METHOD(LinearPathTest) {
            auto f = signature_cuda_d;
            uint64_t dimension = 2, length = 3, degree = 3;
            uint64_t level_3_start = sig_length_(dimension, 2);
            uint64_t level_4_start = sig_length_(dimension, 3);
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1., 1. };
            std::vector<double> true_sig;
            true_sig.resize(level_4_start);
            true_sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
            check_result_typed(f, path, true_sig, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = signature_cuda_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            uint64_t level_3_start = sig_length_(dimension, 2);
            uint64_t level_4_start = sig_length_(dimension, 3);
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.75, 0.75, 1., 1. };
            std::vector<double> true_sig;
            true_sig.resize(level_4_start);
            true_sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
            check_result_typed(f, path, true_sig, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(ManualSigTest) {
            auto f = signature_cuda_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result_typed(f, path, true_sig, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(ManualSigTestDirect) {
            auto f = signature_cuda_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result_typed(f, path, true_sig, dimension, length, degree, false, false, 1., false);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = signature_cuda_d;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<double> path = { 0., 5., 2., 4., 9. };
            std::vector<double> true_sig = { 1., 9., 4., 40.5, 15.5, 20.5, 8., 121.5, 37.5,
                                64.5, 24.5, 60., 13., 34.5, 10. + 2. / 3. };
            double end_time = length - 1.;
            check_result_typed(f, path, true_sig, dimension, length, degree, true, false, end_time, true);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = signature_cuda_d;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<double> path = { 0., 5., 2., 4., 9. };
            std::vector<double> true_sig = { 1., 9., 9., 40.5, 9., 72., 40.5, 121.5, 6.5, 68., -8.5, 290., 98., 275., 121.5 };
            check_result_typed(f, path, true_sig, dimension, length, degree, false, true, 1., true);
        }
    };

    TEST_CLASS(signatureFloatTest)
    {
    public:
        TEST_METHOD(ManualSigTest2) {
            auto f = signature_cuda_f;
            uint64_t dimension = 3, length = 4, degree = 3;
            std::vector<float> path = {
                 9.f, 5.f, 8.f, 5.f, 3.f, 0.f, 0.f, 2.f, 6.f, 4.f, 0.f, 2.f
            };

            std::vector<float> true_sig = {
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f
            };
            check_result_typed(f, path, true_sig, dimension, length, degree, false, false, 1.f, true);
        }

        TEST_METHOD(ManualSigTest2Direct) {
            auto f = signature_cuda_f;
            uint64_t dimension = 3, length = 4, degree = 3;
            std::vector<float> path = {
                 9.f, 5.f, 8.f, 5.f, 3.f, 0.f, 0.f, 2.f, 6.f, 4.f, 0.f, 2.f
            };

            std::vector<float> true_sig = {
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f
            };
            check_result_typed(f, path, true_sig, dimension, length, degree, false, false, 1.f, false);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = signature_cuda_f;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
            std::vector<float> true_sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                                64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
            float end_time = static_cast<float>(length - 1);
            check_result_typed(f, path, true_sig, dimension, length, degree, true, false, end_time, true);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = signature_cuda_f;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
            std::vector<float> true_sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
            check_result_typed(f, path, true_sig, dimension, length, degree, false, true, 1.f, true);
        }
    };

    TEST_CLASS(batchSignatureTest)
    {
    public:
        TEST_METHOD(BatchSigTest) {
            auto f = batch_signature_cuda_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> true_sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            check_result_typed(f, path, true_sig, (uint64_t)3, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(BatchSigTestDirect) {
            auto f = batch_signature_cuda_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> true_sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            check_result_typed(f, path, true_sig, (uint64_t)3, dimension, length, degree, false, false, 1., false);
        }

        TEST_METHOD(BatchSigTestDegree1) {
            auto f = batch_signature_cuda_d;
            uint64_t dimension = 2, length = 4, degree = 1;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> true_sig = { 1., 1., 1.,
                1., 1., 1.,
                1., 0., 1. };

            check_result_typed(f, path, true_sig, (uint64_t)3, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = batch_signature_cuda_d;
            uint64_t dimension = 2, length = 10, degree = 2, batch = 1;
            std::vector<double> path;
            path.resize(batch * length * dimension, 0.);
            std::vector<double> out;
            out.resize(batch * sig_length_(dimension * 2, degree) + 1);
            out.back() = -1.;

            double* d_path;
            double* d_out;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * out.size());
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);

            int err = f(d_path, d_out, batch, dimension, length, degree, false, true, 1., true);

            cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost);
            cudaFree(d_path);
            cudaFree(d_out);

            Assert::AreEqual(0, err);
        }

        TEST_METHOD(BatchSigTestFloat) {
            auto f = batch_signature_cuda_f;
            uint64_t dimension = 3, length = 4, degree = 3;
            std::vector<float> path = {
                 9.f, 5.f, 8.f, 5.f, 3.f, 0.f, 0.f, 2.f, 6.f, 4.f, 0.f, 2.f,
                 9.f, 5.f, 8.f, 5.f, 3.f, 0.f, 0.f, 2.f, 6.f, 4.f, 0.f, 2.f
            };

            std::vector<float> true_sig = {
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f,
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f
            };
            check_result_typed(f, path, true_sig, (uint64_t)2, dimension, length, degree, false, false, 1.f, true);
        }
    };

    // =====================================================================
    // Helper for sig_backprop tests
    // Allocates path, sig, sig_derivs, out on GPU, runs backprop, checks result
    // =====================================================================
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

        // Copy sentinel
        cudaMemcpy(d_out, out.data(), sizeof(T) * out.size(), cudaMemcpyHostToDevice);

        if (path.size() > 0)
            cudaMemcpy(d_path, path.data(), sizeof(T) * path.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sig, sig.data(), sizeof(T) * sig.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sig_derivs, sig_derivs.data(), sizeof(T) * sig_derivs.size(), cudaMemcpyHostToDevice);

        int err = f(d_path, d_out, d_sig_derivs, d_sig, dimension, length, degree, time_aug, lead_lag, end_time);
        cudaDeviceSynchronize();

        cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

        if (d_path) cudaFree(d_path);
        cudaFree(d_out);
        cudaFree(d_sig);
        cudaFree(d_sig_derivs);

        Assert::AreEqual(0, err, L"sig_backprop returned non-zero error code");

        const double eps = TYPED_EPSILON(T);
        for (uint64_t i = 0; i < expected_out.size(); ++i) {
            std::wstring msg = L"Backprop mismatch at index " + std::to_wstring(i) +
                L": expected " + std::to_wstring(static_cast<double>(expected_out[i])) +
                L" got " + std::to_wstring(static_cast<double>(out[i]));
            Assert::IsTrue(std::abs(static_cast<double>(expected_out[i]) - static_cast<double>(out[i])) < eps, msg.c_str());
        }

        Assert::IsTrue(std::abs(-1. - static_cast<double>(out[expected_out.size()])) < eps, L"Sentinel value was overwritten");
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

        int err = f(d_path, d_out, d_sig_derivs, d_sig, batch_size, dimension, length, degree, time_aug, lead_lag, end_time);
        cudaDeviceSynchronize();

        cudaMemcpy(out.data(), d_out, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);

        if (d_path) cudaFree(d_path);
        cudaFree(d_out);
        cudaFree(d_sig);
        cudaFree(d_sig_derivs);

        Assert::AreEqual(0, err, L"batch_sig_backprop returned non-zero error code");

        const double eps = TYPED_EPSILON(T);
        for (uint64_t i = 0; i < expected_out.size(); ++i) {
            std::wstring msg = L"Batch backprop mismatch at index " + std::to_wstring(i) +
                L": expected " + std::to_wstring(static_cast<double>(expected_out[i])) +
                L" got " + std::to_wstring(static_cast<double>(out[i]));
            Assert::IsTrue(std::abs(static_cast<double>(expected_out[i]) - static_cast<double>(out[i])) < eps, msg.c_str());
        }

        Assert::IsTrue(std::abs(-1. - static_cast<double>(out[expected_out.size()])) < eps, L"Sentinel value was overwritten");
    }

    TEST_CLASS(sigBackpropDoubleTest)
    {
    public:
        // Degree 0 or length <= 1: output should be all zeros
        TEST_METHOD(TrivialDegree0) {
            auto f = sig_backprop_cuda_d;
            uint64_t dimension = 2, length = 3, degree = 0;
            std::vector<double> path = { 0., 0., 1., 1., 2., 2. };
            std::vector<double> sig = { 1. };
            std::vector<double> sig_derivs = { 1. };
            std::vector<double> expected_out(dimension * length, 0.);
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(TrivialLength1) {
            auto f = sig_backprop_cuda_d;
            uint64_t dimension = 2, length = 1, degree = 3;
            std::vector<double> path = { 1., 2. };
            std::vector<double> sig = { 1., 0., 0. };
            std::vector<double> sig_derivs = { 1., 1., 1. };
            std::vector<double> expected_out(dimension * length, 0.);
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
        }
        TEST_METHOD(Degree1Dim1) {
            auto f = sig_backprop_cuda_d;
            uint64_t dimension = 1, length = 3, degree = 1;
            std::vector<double> path = { 0., 1., 3. };
            std::vector<double> sig = { 1., 3. };
            std::vector<double> sig_derivs = { 0., 2. };
            std::vector<double> expected_out = { -2., 0., 2. };
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(Degree1Dim2) {
            auto f = sig_backprop_cuda_d;
            uint64_t dimension = 2, length = 3, degree = 1;
            std::vector<double> path = { 0., 0., 1., 2., 3., 5. };
            std::vector<double> sig = { 1., 3., 5. };
            std::vector<double> sig_derivs = { 0., 1., 1. };
            std::vector<double> expected_out = { -1., -1., 0., 0., 1., 1. };
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
        }
        TEST_METHOD(Degree2Dim1) {
            auto f = sig_backprop_cuda_d;
            uint64_t dimension = 1, length = 3, degree = 2;
            std::vector<double> path = { 0., 1., 2. };
            std::vector<double> sig = { 1., 2., 2. };
            std::vector<double> sig_derivs = { 0., 0., 1. };
            std::vector<double> expected_out = { -2., 0., 2. };
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
        }
        TEST_METHOD(Degree2Dim2) {
            auto f = sig_backprop_cuda_d;
            uint64_t dimension = 2, length = 2, degree = 2;
            std::vector<double> path = { 0., 0., 1., 2. };
            std::vector<double> sig = { 1., 1., 2., 0.5, 1., 1., 2. };
            std::vector<double> sig_derivs = { 0., 1., 0., 0., 0., 0., 0. };
            std::vector<double> expected_out = { -1., 0., 1., 0. };
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
        }
        TEST_METHOD(ErrorDimension0) {
            int err = sig_backprop_cuda_d(nullptr, nullptr, nullptr, nullptr, 0, 3, 2, false, false, 1.);
            Assert::AreNotEqual(0, err);
        }
    };

    TEST_CLASS(sigBackpropFloatTest)
    {
    public:
        TEST_METHOD(Degree1Dim2Float) {
            auto f = sig_backprop_cuda_f;
            uint64_t dimension = 2, length = 3, degree = 1;
            std::vector<float> path = { 0.f, 0.f, 1.f, 2.f, 3.f, 5.f };
            std::vector<float> sig = { 1.f, 3.f, 5.f };
            std::vector<float> sig_derivs = { 0.f, 1.f, 1.f };
            std::vector<float> expected_out = { -1.f, -1.f, 0.f, 0.f, 1.f, 1.f };
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.f);
        }

        TEST_METHOD(Degree2Dim1Float) {
            auto f = sig_backprop_cuda_f;
            uint64_t dimension = 1, length = 3, degree = 2;
            std::vector<float> path = { 0.f, 1.f, 2.f };
            std::vector<float> sig = { 1.f, 2.f, 2.f };
            std::vector<float> sig_derivs = { 0.f, 0.f, 1.f };
            std::vector<float> expected_out = { -2.f, 0.f, 2.f };
            check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.f);
        }
    };

    TEST_CLASS(batchSigBackpropTest)
    {
    public:
        TEST_METHOD(BatchDegree1) {
            auto f = batch_sig_backprop_cuda_d;
            uint64_t dimension = 2, length = 3, degree = 1;
            // Two identical batch elements
            std::vector<double> path = {
                0., 0., 1., 2., 3., 5.,
                0., 0., 1., 2., 3., 5.
            };
            std::vector<double> sig = {
                1., 3., 5.,
                1., 3., 5.
            };
            std::vector<double> sig_derivs = {
                0., 1., 1.,
                0., 1., 1.
            };
            std::vector<double> expected_out = {
                -1., -1., 0., 0., 1., 1.,
                -1., -1., 0., 0., 1., 1.
            };
            check_batch_backprop_result(f, path, sig, sig_derivs, expected_out,
                (uint64_t)2, dimension, length, degree, false, false, 1.);
        }
    };

    TEST_CLASS(sigCombineDoubleTest)
    {
    public:
        TEST_METHOD(PolyMultTestLinear)
        {
            // Test signatures of linear 2d paths (same as CPU PolyMultTestLinear)
            auto f = sig_combine_cuda_d;
            std::vector<double> poly = { 1., 1., 1., 1. / 2, 1. / 2, 1. / 2, 1. / 2 };
            std::vector<double> true_res = { 1., 2., 2., 2., 2., 2., 2. };

            check_result_2_typed(f, poly, poly, true_res, (uint64_t)2, (uint64_t)2);
        }

        TEST_METHOD(PolyMultSigTest)
        {
            // Same as CPU PolyMultSigTest: compute sigs of two sub-paths, combine,
            // and compare against sig of the concatenated path
            uint64_t dimension = 2, degree = 5;
            auto f = sig_combine_cuda_d;
            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

            // Compute sigs on GPU
            std::vector<double> poly1 = compute_sig_on_gpu(path1, dimension, 3, degree);
            std::vector<double> poly2 = compute_sig_on_gpu(path2, dimension, 3, degree);
            std::vector<double> true_sig = compute_sig_on_gpu(path, dimension, 5, degree);

            check_result_2_typed(f, poly1, poly2, true_sig, dimension, (uint64_t)degree);
        }
    };

    TEST_CLASS(sigCombineFloatTest)
    {
    public:
        TEST_METHOD(PolyMultTestLinear)
        {
            auto f = sig_combine_cuda_f;
            std::vector<float> poly = { 1.f, 1.f, 1.f, 1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2 };
            std::vector<float> true_res = { 1.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };

            check_result_2_typed(f, poly, poly, true_res, (uint64_t)2, (uint64_t)2);
        }

        TEST_METHOD(PolyMultSigTest)
        {
            uint64_t dimension = 2, degree = 5;
            auto f = sig_combine_cuda_f;
            std::vector<float> path1 = { 0.f, 0.f, 1.f, 0.5f, 0.4f, 2.f };
            std::vector<float> path2 = { 0.4f, 2.f, 6.f, 0.1f, 2.3f, 4.1f };
            std::vector<float> path = { 0.f, 0.f, 1.f, 0.5f, 0.4f, 2.f, 6.f, 0.1f, 2.3f, 4.1f };

            std::vector<float> poly1 = compute_sig_on_gpu(path1, dimension, (uint64_t)3, degree);
            std::vector<float> poly2 = compute_sig_on_gpu(path2, dimension, (uint64_t)3, degree);
            std::vector<float> true_sig = compute_sig_on_gpu(path, dimension, (uint64_t)5, degree);

            check_result_2_typed(f, poly1, poly2, true_sig, dimension, (uint64_t)degree);
        }
    };

    TEST_CLASS(batchSigCombineTest)
    {
    public:
        TEST_METHOD(BatchPolyMultSigTest)
        {
            // Same as CPU BatchPolyMultSigTest
            uint64_t batch_size = 3, dimension = 2, degree = 2;
            auto f = batch_sig_combine_cuda_d;
            std::vector<double> path1 = { 0., 0., 0.25, 0.25, 0.5, 0.5,
                0., 0., 0.4, 0.4, 0.6, 0.6,
                0., 0., 1., 0.5, 4., 0. };
            std::vector<double> path2 = { 0.5, 0.5, 1., 1.,
                0.6, 0.6, 1., 1.,
                4., 0., 0., 1. };
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> poly1 = compute_batch_sig_on_gpu(path1, batch_size, dimension, 3, degree);
            std::vector<double> poly2 = compute_batch_sig_on_gpu(path2, batch_size, dimension, 2, degree);
            std::vector<double> true_sig = compute_batch_sig_on_gpu(path, batch_size, dimension, 4, degree);

            check_result_2_typed(f, poly1, poly2, true_sig, batch_size, dimension, (uint64_t)degree);
        }

        TEST_METHOD(BatchPolyMultSigTestFloat)
        {
            uint64_t batch_size = 3, dimension = 2, degree = 2;
            auto f = batch_sig_combine_cuda_f;
            std::vector<float> path1 = { 0.f, 0.f, 0.25f, 0.25f, 0.5f, 0.5f,
                0.f, 0.f, 0.4f, 0.4f, 0.6f, 0.6f,
                0.f, 0.f, 1.f, 0.5f, 4.f, 0.f };
            std::vector<float> path2 = { 0.5f, 0.5f, 1.f, 1.f,
                0.6f, 0.6f, 1.f, 1.f,
                4.f, 0.f, 0.f, 1.f };
            std::vector<float> path = { 0.f, 0.f, 0.25f, 0.25f, 0.5f, 0.5f, 1.f, 1.f,
                0.f, 0.f, 0.4f, 0.4f, 0.6f, 0.6f, 1.f, 1.f,
                0.f, 0.f, 1.f, 0.5f, 4.f, 0.f, 0.f, 1.f };

            std::vector<float> poly1 = compute_batch_sig_on_gpu<float>(path1, batch_size, dimension, 3, degree);
            std::vector<float> poly2 = compute_batch_sig_on_gpu<float>(path2, batch_size, dimension, 2, degree);
            std::vector<float> true_sig = compute_batch_sig_on_gpu<float>(path, batch_size, dimension, 4, degree);

            check_result_2_typed(f, poly1, poly2, true_sig, batch_size, dimension, (uint64_t)degree);
        }

        TEST_METHOD(BatchPolyMultStressTest)
        {
            // Same as CPU BatchPolyMultStressTest: just check it doesn't crash/error
            uint64_t batch_size = 1000, dimension = 5, degree = 5;
            uint64_t total_len = batch_size * sig_length_(dimension, degree);

            std::vector<double> poly(total_len, 1.);

            double* d_poly = nullptr;
            double* d_out = nullptr;
            cudaMalloc(&d_poly, sizeof(double) * total_len);
            cudaMalloc(&d_out, sizeof(double) * total_len);
            cudaMemcpy(d_poly, poly.data(), sizeof(double) * total_len, cudaMemcpyHostToDevice);

            int err = batch_sig_combine_cuda_d(d_poly, d_poly, d_out, batch_size, dimension, degree);
            cudaDeviceSynchronize();

            cudaFree(d_poly);
            cudaFree(d_out);

            Assert::AreEqual(0, err, L"BatchPolyMultStressTest returned non-zero error code");
        }
    };

    // =========================================================================
    // sig_combine_backprop CUDA tests
    // Ported from CPU sigCombineBackpropTest
    // =========================================================================

    TEST_CLASS(sigCombineBackpropCudaTest) {
    public:
        TEST_METHOD(ManualTest) {
            uint64_t dimension = 2, degree = 2;
            uint64_t sig_len = 7;

            std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5 };
            std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5 };
            std::vector<double> derivs = { 1., 1., 2., 3., 4., 5., 6. };
            // true_ = [sig1_deriv..., sig2_deriv...]
            std::vector<double> true_ = { 1., 5., 8., 3., 4., 5., 6., 1., 9., 12., 3., 4., 5., 6. };

            double* d_derivs = nullptr;
            double* d_sig1 = nullptr;
            double* d_sig2 = nullptr;
            double* d_sig1_deriv = nullptr;
            double* d_sig2_deriv = nullptr;
            cudaMalloc(&d_derivs, sizeof(double) * sig_len);
            cudaMalloc(&d_sig1, sizeof(double) * sig_len);
            cudaMalloc(&d_sig2, sizeof(double) * sig_len);
            cudaMalloc(&d_sig1_deriv, sizeof(double) * sig_len);
            cudaMalloc(&d_sig2_deriv, sizeof(double) * sig_len);

            cudaMemcpy(d_derivs, derivs.data(), sizeof(double) * sig_len, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sig1, sig1.data(), sizeof(double) * sig_len, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sig2, sig2.data(), sizeof(double) * sig_len, cudaMemcpyHostToDevice);

            int err = sig_combine_backprop_cuda_d(d_derivs, d_sig1_deriv, d_sig2_deriv, d_sig1, d_sig2, dimension, degree);
            cudaDeviceSynchronize();

            std::vector<double> sig1_deriv(sig_len), sig2_deriv(sig_len);
            cudaMemcpy(sig1_deriv.data(), d_sig1_deriv, sizeof(double) * sig_len, cudaMemcpyDeviceToHost);
            cudaMemcpy(sig2_deriv.data(), d_sig2_deriv, sizeof(double) * sig_len, cudaMemcpyDeviceToHost);

            cudaFree(d_derivs);
            cudaFree(d_sig1);
            cudaFree(d_sig2);
            cudaFree(d_sig1_deriv);
            cudaFree(d_sig2_deriv);

            Assert::AreEqual(0, err, L"sig_combine_backprop_cuda_d returned non-zero error code");

            for (uint64_t i = 0; i < sig_len; ++i) {
                std::wstring msg = L"sig1_deriv mismatch at " + std::to_wstring(i);
                Assert::IsTrue(std::abs(sig1_deriv[i] - true_[i]) < DOUBLE_EPSILON, msg.c_str());
            }
            for (uint64_t i = 0; i < sig_len; ++i) {
                std::wstring msg = L"sig2_deriv mismatch at " + std::to_wstring(i);
                Assert::IsTrue(std::abs(sig2_deriv[i] - true_[sig_len + i]) < DOUBLE_EPSILON, msg.c_str());
            }
        }

        TEST_METHOD(ManualBatchTest) {
            uint64_t dimension = 2, degree = 2, batch_size = 2;
            uint64_t sig_len = 7;
            uint64_t total = sig_len * batch_size;

            std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5,
                1., 0., 1., 0., 1., -1., .5 };
            std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5,
                1., 1., 1., .5, .5, .5, .5 };
            std::vector<double> derivs = { 1., 1., 2., 3., 4., 5., 6.,
                1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 1., 5., 8., 3., 4., 5., 6.,
                1., 8., 13., 3., 4., 5., 6.,
                1., 9., 12., 3., 4., 5., 6.,
                1., 6., 8., 3., 4., 5., 6. };

            double* d_derivs = nullptr;
            double* d_sig1 = nullptr;
            double* d_sig2 = nullptr;
            double* d_sig1_deriv = nullptr;
            double* d_sig2_deriv = nullptr;
            cudaMalloc(&d_derivs, sizeof(double) * total);
            cudaMalloc(&d_sig1, sizeof(double) * total);
            cudaMalloc(&d_sig2, sizeof(double) * total);
            cudaMalloc(&d_sig1_deriv, sizeof(double) * total);
            cudaMalloc(&d_sig2_deriv, sizeof(double) * total);

            cudaMemcpy(d_derivs, derivs.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sig1, sig1.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sig2, sig2.data(), sizeof(double) * total, cudaMemcpyHostToDevice);

            int err = batch_sig_combine_backprop_cuda_d(d_derivs, d_sig1_deriv, d_sig2_deriv, d_sig1, d_sig2, batch_size, dimension, degree);
            cudaDeviceSynchronize();

            std::vector<double> sig1_deriv(total), sig2_deriv(total);
            cudaMemcpy(sig1_deriv.data(), d_sig1_deriv, sizeof(double) * total, cudaMemcpyDeviceToHost);
            cudaMemcpy(sig2_deriv.data(), d_sig2_deriv, sizeof(double) * total, cudaMemcpyDeviceToHost);

            cudaFree(d_derivs);
            cudaFree(d_sig1);
            cudaFree(d_sig2);
            cudaFree(d_sig1_deriv);
            cudaFree(d_sig2_deriv);

            Assert::AreEqual(0, err, L"batch_sig_combine_backprop_cuda_d returned non-zero error code");

            for (uint64_t i = 0; i < total; ++i) {
                std::wstring msg = L"sig1_deriv mismatch at " + std::to_wstring(i);
                Assert::IsTrue(std::abs(sig1_deriv[i] - true_[i]) < DOUBLE_EPSILON, msg.c_str());
            }
            for (uint64_t i = 0; i < total; ++i) {
                std::wstring msg = L"sig2_deriv mismatch at " + std::to_wstring(i);
                Assert::IsTrue(std::abs(sig2_deriv[i] - true_[total + i]) < DOUBLE_EPSILON, msg.c_str());
            }
        }
    };

    // =========================================================================
    // sig_to_log_sig CUDA tests (expanded / method=0)
    // Ported from CPU logSignatureExpandedTest
    // =========================================================================
    TEST_CLASS(logSignatureExpandedCudaTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length_(dimension, 2);
            uint64_t level_4_start = sig_length_(dimension, 3);
            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
            std::vector<double> sig(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            check_result_typed(sig_to_log_sig_cuda_d, sig, true_, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualLogSigTest) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> true_ = { 0., 0., 1., 0., 1., -1., 0. };
            std::vector<double> sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result_typed(sig_to_log_sig_cuda_d, sig, true_, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualLogSigTest2Float) {
            uint64_t dimension = 3, degree = 3;
            std::vector<float> true_ = {
                 0.f, -5.f, -5.f, -6.f, 0.f, 12.f, -10.f, -12.f,
                 0.f, -6.f, 10.f, 6.f, 0.f, 0.f, -27.f,
                 11.f, 54.f, 5.f, 3.f + 2.f / 3.f, -22.f, 20.f + 2.f / 3.f, -18.f,
                -27.f, -10.f, -24.f - 1.f / 3.f, 5.f, 0.f, -9.f, 20.f + 2.f / 3.f,
                 18.f, -4.f - 2.f / 3.f, 11.f, -24.f - 1.f / 3.f, 36.f, 3.f + 2.f / 3.f, -9.f,
                 9.f + 1.f / 3.f, -18.f, -4.f - 2.f / 3.f, 0.f
            };

            std::vector<float> sig = {
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f
            };
            check_result_typed(sig_to_log_sig_cuda_f, sig, true_, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(BatchLogSigTest) {
            uint64_t dimension = 2, degree = 2;

            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0.,
                0., 1., 1., 0., 0., 0., 0.,
                0., 0., 1., 0., 1., -1., 0. };

            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            check_result_typed(batch_sig_to_log_sig_cuda_d, sig, true_, (uint64_t)3, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualTimeAugTest) {
            // CPU test passes time_aug=true with dimension=1 → aug_dimension=2
            // CUDA: we pass dimension=2 directly (the signature is already in the augmented space)
            uint64_t dimension = 2, degree = 3;
            std::vector<float> true_ = { 0.f, 9.f, 4.f, 0.f, -2.5f, 2.5f, 0.f, 0.f, -5.25f,
                                10.5f, 5.5f, -5.25f, -11.f, 5.5f, 0.f };
            std::vector<float> sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                                64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
            check_result_typed(sig_to_log_sig_cuda_f, sig, true_, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualLeadLagTest) {
            // CPU test passes lead_lag=true with dimension=1 → aug_dimension=2
            // CUDA: we pass dimension=2 directly
            uint64_t dimension = 2, degree = 3;
            std::vector<float> true_ = { 0.f, 9.f, 9.f, 0.f, -31.5f, 31.5f, 0.f, 0.f, 26.75f, -53.5f, 11.75f, 26.75f, -23.5f, 11.75f, 0.f };
            std::vector<float> sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
            check_result_typed(sig_to_log_sig_cuda_f, sig, true_, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(BigLeadLagTest) {
            // CPU test: lead_lag=true, dimension=2 → aug_dimension=4
            // Just check it doesn't crash/error
            uint64_t dimension = 4, degree = 2, batch = 1;
            uint64_t slen = sig_length_(dimension, degree);
            std::vector<double> sig(batch * slen, 0.);
            std::vector<double> out(batch * slen, 0.);

            double* d_sig = nullptr;
            double* d_out = nullptr;
            cudaMalloc(&d_sig, sizeof(double) * sig.size());
            cudaMalloc(&d_out, sizeof(double) * out.size());
            cudaMemcpy(d_sig, sig.data(), sizeof(double) * sig.size(), cudaMemcpyHostToDevice);

            int err = batch_sig_to_log_sig_cuda_d(d_sig, d_out, batch, dimension, degree, 0);
            cudaDeviceSynchronize();

            cudaFree(d_sig);
            cudaFree(d_out);

            Assert::AreEqual(0, err, L"BigLeadLagTest returned non-zero error code");
        }

        TEST_METHOD(Degree1Test) {
            // degree 1: log sig should just be [0, sig[1], sig[2], ...]
            uint64_t dimension = 3, degree = 1;
            std::vector<double> sig = { 1., 2., 3., 4. };
            std::vector<double> true_ = { 0., 2., 3., 4. };
            check_result_typed(sig_to_log_sig_cuda_d, sig, true_, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(StressTest) {
            // Just check it doesn't crash for large input
            uint64_t batch_size = 100, dimension = 5, degree = 5;
            uint64_t slen = sig_length_(dimension, degree);
            uint64_t total_len = batch_size * slen;

            std::vector<double> sig(total_len, 1.);

            double* d_sig = nullptr;
            double* d_out = nullptr;
            cudaMalloc(&d_sig, sizeof(double) * total_len);
            cudaMalloc(&d_out, sizeof(double) * total_len);
            cudaMemcpy(d_sig, sig.data(), sizeof(double) * total_len, cudaMemcpyHostToDevice);

            int err = batch_sig_to_log_sig_cuda_d(d_sig, d_out, batch_size, dimension, degree, 0);
            cudaDeviceSynchronize();

            cudaFree(d_sig);
            cudaFree(d_out);

            Assert::AreEqual(0, err, L"StressTest returned non-zero error code");
        }
    };

    // =========================================================================
    // Helper for backprop tests: 3 device inputs (sig, derivs → out)
    // fn signature: int f(const T* sig, T* out, const T* derivs, ...)
    // =========================================================================

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

        Assert::AreEqual(0, err, L"Backprop function returned non-zero error code");

        const double eps = TYPED_EPSILON(T);
        for (uint64_t i = 0; i < true_.size(); ++i) {
            std::wstring msg = L"Mismatch at index " + std::to_wstring(i) +
                L": expected " + std::to_wstring(static_cast<double>(true_[i])) +
                L" got " + std::to_wstring(static_cast<double>(out[i]));
            Assert::IsTrue(std::abs(static_cast<double>(true_[i]) - static_cast<double>(out[i])) < eps, msg.c_str());
        }

        Assert::IsTrue(std::abs(-1. - static_cast<double>(out[true_.size()])) < eps, L"Sentinel value was overwritten");
    }

    // For batch backprop: fn signature: int f(const T* sig, T* out, const T* derivs, batch_size, ...)
    template<typename FN, typename T, typename... Args>
    void check_result_batch_backprop_typed(FN f, std::vector<T>& sig, std::vector<T>& true_, std::vector<T>& derivs, Args... args) {
        check_result_backprop_typed(f, sig, true_, derivs, args...);
    }

    // =========================================================================
    // sig_to_log_sig_backprop CUDA tests (expanded / method=0)
    // Ported from CPU logSignatureExpandedBackpropTest
    // =========================================================================
    TEST_CLASS(logSignatureExpandedBackpropCudaTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 1., -1., -1.,  1.,  1.,  1.,  1. };
            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualTest) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 1., -5., -6.25, 3., 4., 5., 6. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualTest2) {
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { 1., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualTestAsBatch) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 1., -5., -6.25, 3., 4., 5., 6. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { 1., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualBatchTest) {
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 1., 6.5, 7.6875, -10., -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14., 1., 66., 30.25, -35., 15.5, -46., 14.5, 7., -8., 9., -10., 11., -12., 13., -14., 1., 1.625, 1.625, 1.5, 1.5, 1.5, 1.5, 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, batch_size, dimension, (uint64_t)degree, 0);
        }

        TEST_METHOD(ManualDim1Test) {
            uint64_t dimension = 1, degree = 8;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
            std::vector<double> true_ = { 1., -1., 8., 9., 1., -8., -9., -1., 8. };
            std::vector<double> sig = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 0);
        }
    };

    // =========================================================================
    // sig_to_log_sig_backprop CUDA tests (Lyndon words / method=1)
    // Ported from CPU logSignatureLyndonWordsBackpropTest
    // =========================================================================
    TEST_CLASS(logSignatureLyndonWordsBackpropCudaTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1. };
            std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
            prepare_log_sig_cuda(dimension, degree, 1);
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 1);
        }

        TEST_METHOD(ManualTest) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig_cuda(dimension, degree, 1);
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 1);
        }

        TEST_METHOD(ManualTest2) {
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig_cuda(dimension, degree, 1);
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 1);
        }

        TEST_METHOD(ManualTestAsBatch) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig_cuda(dimension, degree, 1);
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig_cuda(dimension, degree, 1);
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 1);
        }

        TEST_METHOD(ManualBatchTest) {
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            prepare_log_sig_cuda(dimension, degree, 1);
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, batch_size, dimension, (uint64_t)degree, 1);
        }
    };

    // =========================================================================
    // sig_to_log_sig_backprop CUDA tests (Lyndon basis / method=2)
    // Ported from CPU logSignatureLyndonBasisBackpropTest
    // =========================================================================
    TEST_CLASS(logSignatureLyndonBasisBackpropCudaTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1. };
            std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
            prepare_log_sig_cuda(dimension, degree, 2);
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 2);
        }

        TEST_METHOD(ManualTest) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig_cuda(dimension, degree, 2);
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 2);
        }

        TEST_METHOD(ManualTest2) {
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig_cuda(dimension, degree, 2);
            check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, dimension, (uint64_t)degree, 2);
        }

        TEST_METHOD(ManualTestAsBatch) {
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig_cuda(dimension, degree, 2);
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 2);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig_cuda(dimension, degree, 2);
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 2);
        }

        TEST_METHOD(ManualBatchTest) {
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            prepare_log_sig_cuda(dimension, degree, 2);
            check_result_backprop_typed(batch_sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, batch_size, dimension, (uint64_t)degree, 2);
        }
    };
}