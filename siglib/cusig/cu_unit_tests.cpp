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
            check_result_typed(f, path, true_sig, (uint64_t)1, (uint64_t)0, (uint64_t)0, false, false, 1., true);

            path.push_back(0.);
            check_result_typed(f, path, true_sig, (uint64_t)1, (uint64_t)1, (uint64_t)0, false, false, 1., true);

            true_sig.push_back(0.);
            check_result_typed(f, path, true_sig, (uint64_t)1, (uint64_t)0, (uint64_t)1, false, false, 1., true);
            check_result_typed(f, path, true_sig, (uint64_t)1, (uint64_t)1, (uint64_t)1, false, false, 1., true);

            path.push_back(1.);
            true_sig[1] = 1.;
            check_result_typed(f, path, true_sig, (uint64_t)1, (uint64_t)2, (uint64_t)1, false, false, 1., true);
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
}