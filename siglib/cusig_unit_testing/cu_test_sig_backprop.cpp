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

#include "cu_test_helpers.h"

// Degree 0 or length <= 1: output should be all zeros
TEST(sigBackpropDoubleTest, TrivialDegree0) {
    auto f = sig_backprop_cuda_d;
    uint64_t dimension = 2, length = 3, degree = 0;
    std::vector<double> path = { 0., 0., 1., 1., 2., 2. };
    std::vector<double> sig = { 1. };
    std::vector<double> sig_derivs = { 1. };
    std::vector<double> expected_out(dimension * length, 0.);
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
}

TEST(sigBackpropDoubleTest, TrivialLength1) {
    auto f = sig_backprop_cuda_d;
    uint64_t dimension = 2, length = 1, degree = 3;
    std::vector<double> path = { 1., 2. };
    std::vector<double> sig = { 1., 0., 0. };
    std::vector<double> sig_derivs = { 1., 1., 1. };
    std::vector<double> expected_out(dimension * length, 0.);
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
}

TEST(sigBackpropDoubleTest, Degree1Dim1) {
    auto f = sig_backprop_cuda_d;
    uint64_t dimension = 1, length = 3, degree = 1;
    std::vector<double> path = { 0., 1., 3. };
    std::vector<double> sig = { 1., 3. };
    std::vector<double> sig_derivs = { 0., 2. };
    std::vector<double> expected_out = { -2., 0., 2. };
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
}

TEST(sigBackpropDoubleTest, Degree1Dim2) {
    auto f = sig_backprop_cuda_d;
    uint64_t dimension = 2, length = 3, degree = 1;
    std::vector<double> path = { 0., 0., 1., 2., 3., 5. };
    std::vector<double> sig = { 1., 3., 5. };
    std::vector<double> sig_derivs = { 0., 1., 1. };
    std::vector<double> expected_out = { -1., -1., 0., 0., 1., 1. };
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
}

TEST(sigBackpropDoubleTest, Degree2Dim1) {
    auto f = sig_backprop_cuda_d;
    uint64_t dimension = 1, length = 3, degree = 2;
    std::vector<double> path = { 0., 1., 2. };
    std::vector<double> sig = { 1., 2., 2. };
    std::vector<double> sig_derivs = { 0., 0., 1. };
    std::vector<double> expected_out = { -2., 0., 2. };
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
}

TEST(sigBackpropDoubleTest, Degree2Dim2) {
    auto f = sig_backprop_cuda_d;
    uint64_t dimension = 2, length = 2, degree = 2;
    std::vector<double> path = { 0., 0., 1., 2. };
    std::vector<double> sig = { 1., 1., 2., 0.5, 1., 1., 2. };
    std::vector<double> sig_derivs = { 0., 1., 0., 0., 0., 0., 0. };
    std::vector<double> expected_out = { -1., 0., 1., 0. };
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.);
}

TEST(sigBackpropDoubleTest, PrimitiveSingleSegment) {
    auto f = sig_backprop_cuda_d;
    uint64_t dimension = 1, length = 2, degree = 4;
    std::vector<double> path = { 0., 3. };
    std::vector<double> primitives = { 2. };
    std::vector<double> sig = { 1., 3., 6.5, 10.5, 14.375 };
    std::vector<double> sig_derivs(sig.size(), 1.);
    std::vector<double> expected_out = { -21., 21. };
    std::vector<double> out(expected_out.size() + 1, 0.);
    out[expected_out.size()] = -1.;

    double* d_path;
    double* d_out;
    double* d_sig;
    double* d_sig_derivs;
    double* d_primitives;
    cudaMalloc(&d_path, sizeof(double) * path.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());
    cudaMalloc(&d_sig, sizeof(double) * sig.size());
    cudaMalloc(&d_sig_derivs, sizeof(double) * sig_derivs.size());
    cudaMalloc(&d_primitives, sizeof(double) * primitives.size());
    cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data(), sizeof(double) * out.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig.data(), sizeof(double) * sig.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_derivs, sig_derivs.data(), sizeof(double) * sig_derivs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primitives, primitives.data(), sizeof(double) * primitives.size(), cudaMemcpyHostToDevice);

    int err = f(d_path, d_out, d_sig_derivs, d_sig, (uint64_t)1, dimension, length, degree,
        false, false, 1., true, d_primitives, primitives.size());
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_sig);
    cudaFree(d_sig_derivs);
    cudaFree(d_primitives);

    EXPECT_EQ(0, err);
    for (uint64_t i = 0; i < expected_out.size(); ++i)
        EXPECT_TRUE(std::abs(expected_out[i] - out[i]) < DOUBLE_EPSILON);
    EXPECT_TRUE(std::abs(-1. - out[expected_out.size()]) < DOUBLE_EPSILON);
}

TEST(sigBackpropDoubleTest, ErrorDimension0) {
    int err = sig_backprop_cuda_d(nullptr, nullptr, nullptr, nullptr, (uint64_t)1, 0, 3, 2, false, false, 1., true, nullptr, 0);
    EXPECT_NE(0, err);
}

TEST(sigBackpropFloatTest, Degree1Dim2Float) {
    auto f = sig_backprop_cuda_f;
    uint64_t dimension = 2, length = 3, degree = 1;
    std::vector<float> path = { 0.f, 0.f, 1.f, 2.f, 3.f, 5.f };
    std::vector<float> sig = { 1.f, 3.f, 5.f };
    std::vector<float> sig_derivs = { 0.f, 1.f, 1.f };
    std::vector<float> expected_out = { -1.f, -1.f, 0.f, 0.f, 1.f, 1.f };
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.f);
}

TEST(sigBackpropFloatTest, Degree2Dim1Float) {
    auto f = sig_backprop_cuda_f;
    uint64_t dimension = 1, length = 3, degree = 2;
    std::vector<float> path = { 0.f, 1.f, 2.f };
    std::vector<float> sig = { 1.f, 2.f, 2.f };
    std::vector<float> sig_derivs = { 0.f, 0.f, 1.f };
    std::vector<float> expected_out = { -2.f, 0.f, 2.f };
    check_backprop_result(f, path, sig, sig_derivs, expected_out, dimension, length, degree, false, false, 1.f);
}

TEST(batchSigBackpropTest, BatchDegree1) {
    auto f = sig_backprop_cuda_d;
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
