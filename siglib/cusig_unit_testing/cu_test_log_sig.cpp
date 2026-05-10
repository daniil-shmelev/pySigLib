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

// =========================================================================
// sig_to_log_sig CUDA tests (expanded / method=0)
// Ported from CPU logSignatureExpandedTest
// =========================================================================

TEST(logSignatureExpandedCudaTest, LinearPathTest) {
    uint64_t dimension = 2, degree = 3;
    uint64_t level_3_start = sig_length_(dimension, 2);
    uint64_t level_4_start = sig_length_(dimension, 3);
    std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
    std::vector<double> sig(level_4_start);
    sig[0] = 1.;
    for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
    for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
    for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
    check_result_typed(sig_to_log_sig_cuda_d, sig, true_, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedCudaTest, ManualLogSigTest) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> true_ = { 0., 0., 1., 0., 1., -1., 0. };
    std::vector<double> sig = { 1., 0., 1., 0., 1., -1., 0.5 };
    check_result_typed(sig_to_log_sig_cuda_d, sig, true_, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedCudaTest, ManualLogSigTest2Float) {
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
    check_result_typed(sig_to_log_sig_cuda_f, sig, true_, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedCudaTest, BatchLogSigTest) {
    uint64_t dimension = 2, degree = 2;

    std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0.,
        0., 1., 1., 0., 0., 0., 0.,
        0., 0., 1., 0., 1., -1., 0. };

    std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 0., 1., 0., 1., -1., 0.5 };

    check_result_typed(sig_to_log_sig_cuda_d, sig, true_, (uint64_t)3, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedCudaTest, ManualTimeAugTest) {
    // CPU test passes time_aug=true with dimension=1 -> aug_dimension=2
    // CUDA: we pass dimension=2 directly (the signature is already in the augmented space)
    uint64_t dimension = 2, degree = 3;
    std::vector<float> true_ = { 0.f, 9.f, 4.f, 0.f, -2.5f, 2.5f, 0.f, 0.f, -5.25f,
                            10.5f, 5.5f, -5.25f, -11.f, 5.5f, 0.f };
    std::vector<float> sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                            64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
    check_result_typed(sig_to_log_sig_cuda_f, sig, true_, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedCudaTest, ManualLeadLagTest) {
    // CPU test passes lead_lag=true with dimension=1 -> aug_dimension=2
    // CUDA: we pass dimension=2 directly
    uint64_t dimension = 2, degree = 3;
    std::vector<float> true_ = { 0.f, 9.f, 9.f, 0.f, -31.5f, 31.5f, 0.f, 0.f, 26.75f, -53.5f, 11.75f, 26.75f, -23.5f, 11.75f, 0.f };
    std::vector<float> sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
    check_result_typed(sig_to_log_sig_cuda_f, sig, true_, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedCudaTest, BigLeadLagTest) {
    // CPU test: lead_lag=true, dimension=2 -> aug_dimension=4
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

    int err = sig_to_log_sig_cuda_d(d_sig, d_out, batch, dimension, degree, 0);
    cudaDeviceSynchronize();

    cudaFree(d_sig);
    cudaFree(d_out);

    EXPECT_EQ(0, err) << "BigLeadLagTest returned non-zero error code";
}

TEST(logSignatureExpandedCudaTest, Degree1Test) {
    // degree 1: log sig should just be [0, sig[1], sig[2], ...]
    uint64_t dimension = 3, degree = 1;
    std::vector<double> sig = { 1., 2., 3., 4. };
    std::vector<double> true_ = { 0., 2., 3., 4. };
    check_result_typed(sig_to_log_sig_cuda_d, sig, true_, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

// =========================================================================
// sig_to_log_sig_backprop CUDA tests (expanded / method=0)
// Ported from CPU logSignatureExpandedBackpropTest
// =========================================================================

TEST(logSignatureExpandedBackpropCudaTest, LinearPathTest) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { 0., -1., -1.,  1.,  1.,  1.,  1. };
    std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedBackpropCudaTest, ManualTest) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
    std::vector<double> true_ = { 0., -5., -6.25, 3., 4., 5., 6. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedBackpropCudaTest, ManualTest2) {
    uint64_t dimension = 2, degree = 3;
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> true_ = { 0., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedBackpropCudaTest, ManualTestAsBatch) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
    std::vector<double> true_ = { 0., -5., -6.25, 3., 4., 5., 6. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedBackpropCudaTest, ManualTest2AsBatch) {
    uint64_t dimension = 2, degree = 3;
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> true_ = { 0., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedBackpropCudaTest, ManualBatchTest) {
    uint64_t dimension = 2, degree = 3, batch_size = 3;
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { 0., 6.5, 7.6875, -10., -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14., 0., 66., 30.25, -35., 15.5, -46., 14.5, 7., -8., 9., -10., 11., -12., 13., -14., 0., 1.625, 1.625, 1.5, 1.5, 1.5, 1.5, 1., 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, batch_size, dimension, (uint64_t)degree, 0, true);
}

TEST(logSignatureExpandedBackpropCudaTest, ManualDim1Test) {
    uint64_t dimension = 1, degree = 8;
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
    std::vector<double> true_ = { 0., -1., 8., 9., 1., -8., -9., -1., 8. };
    std::vector<double> sig = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 0, true);
}

// =========================================================================
// sig_to_log_sig_backprop CUDA tests (Lyndon words / method=1)
// Ported from CPU logSignatureLyndonWordsBackpropTest
// =========================================================================

TEST(logSignatureLyndonWordsBackpropCudaTest, LinearPathTest) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 1., 1. };
    std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
    std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
    (void)prepare_log_sig_cuda(dimension, degree, 1);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 1, true);
}

TEST(logSignatureLyndonWordsBackpropCudaTest, ManualTest) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 2., 3. };
    std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    (void)prepare_log_sig_cuda(dimension, degree, 1);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 1, true);
}

TEST(logSignatureLyndonWordsBackpropCudaTest, ManualTest2) {
    uint64_t dimension = 2, degree = 3;
    std::vector<double> deriv = { 1., 2., 3., 4., 5. };
    std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    (void)prepare_log_sig_cuda(dimension, degree, 1);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 1, true);
}

TEST(logSignatureLyndonWordsBackpropCudaTest, ManualTestAsBatch) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 2., 3. };
    std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    (void)prepare_log_sig_cuda(dimension, degree, 1);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 1, true);
}

TEST(logSignatureLyndonWordsBackpropCudaTest, ManualTest2AsBatch) {
    uint64_t dimension = 2, degree = 3;
    std::vector<double> deriv = { 1., 2., 3., 4., 5. };
    std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    (void)prepare_log_sig_cuda(dimension, degree, 1);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 1, true);
}

TEST(logSignatureLyndonWordsBackpropCudaTest, ManualBatchTest) {
    uint64_t dimension = 2, degree = 3, batch_size = 3;
    std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
    (void)prepare_log_sig_cuda(dimension, degree, 1);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, batch_size, dimension, (uint64_t)degree, 1, true);
}

// =========================================================================
// sig_to_log_sig_backprop CUDA tests (Lyndon basis / method=2)
// Ported from CPU logSignatureLyndonBasisBackpropTest
// =========================================================================

TEST(logSignatureLyndonBasisBackpropCudaTest, LinearPathTest) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 1., 1. };
    std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
    std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 2, true);
}

TEST(logSignatureLyndonBasisBackpropCudaTest, ManualTest) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 2., 3. };
    std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 2, true);
}

TEST(logSignatureLyndonBasisBackpropCudaTest, ManualTest2) {
    uint64_t dimension = 2, degree = 3;
    std::vector<double> deriv = { 1., 2., 3., 4., 5. };
    std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 2, true);
}

TEST(logSignatureLyndonBasisBackpropCudaTest, ManualTestAsBatch) {
    uint64_t dimension = 2, degree = 2;
    std::vector<double> deriv = { 1., 2., 3. };
    std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 2, true);
}

TEST(logSignatureLyndonBasisBackpropCudaTest, ManualTest2AsBatch) {
    uint64_t dimension = 2, degree = 3;
    std::vector<double> deriv = { 1., 2., 3., 4., 5. };
    std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, (uint64_t)1, dimension, (uint64_t)degree, 2, true);
}

TEST(logSignatureLyndonBasisBackpropCudaTest, ManualBatchTest) {
    uint64_t dimension = 2, degree = 3, batch_size = 3;
    std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    check_result_backprop_typed(sig_to_log_sig_backprop_cuda_d, sig, true_, deriv, batch_size, dimension, (uint64_t)degree, 2, true);
}

TEST(logSigCombineCudaTest, ChenIdentity) {
    uint64_t dimension = 2, degree = 3;
    (void)prepare_log_sig_cuda(dimension, degree, 2);

    std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
    std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
    std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

    uint64_t slen = sig_length_(dimension, degree);
    uint64_t ls_len = log_sig_length_(dimension, degree);

    std::vector<double> sig1 = compute_sig_on_gpu(path1, dimension, (uint64_t)3, degree);
    std::vector<double> sig2 = compute_sig_on_gpu(path2, dimension, (uint64_t)3, degree);
    std::vector<double> sig_full = compute_sig_on_gpu(path, dimension, (uint64_t)5, degree);

    double* d_sig = nullptr;
    double* d_ls = nullptr;
    cudaMalloc(&d_sig, sizeof(double) * slen);
    cudaMalloc(&d_ls, sizeof(double) * ls_len);

    cudaMemcpy(d_sig, sig1.data(), sizeof(double) * slen, cudaMemcpyHostToDevice);
    int err = sig_to_log_sig_cuda_d(d_sig, d_ls, 1, dimension, degree, 2, true);
    cudaDeviceSynchronize();
    std::vector<double> ls1(ls_len);
    cudaMemcpy(ls1.data(), d_ls, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "sig_to_log_sig_cuda_d failed for sig1";

    cudaMemcpy(d_sig, sig2.data(), sizeof(double) * slen, cudaMemcpyHostToDevice);
    err = sig_to_log_sig_cuda_d(d_sig, d_ls, 1, dimension, degree, 2, true);
    cudaDeviceSynchronize();
    std::vector<double> ls2(ls_len);
    cudaMemcpy(ls2.data(), d_ls, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "sig_to_log_sig_cuda_d failed for sig2";

    cudaMemcpy(d_sig, sig_full.data(), sizeof(double) * slen, cudaMemcpyHostToDevice);
    err = sig_to_log_sig_cuda_d(d_sig, d_ls, 1, dimension, degree, 2, true);
    cudaDeviceSynchronize();
    std::vector<double> true_ls(ls_len);
    cudaMemcpy(true_ls.data(), d_ls, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "sig_to_log_sig_cuda_d failed for full sig";

    cudaFree(d_sig);
    cudaFree(d_ls);

    check_result_2_typed(log_sig_combine_cuda_d, ls1, ls2, true_ls, (uint64_t)1, dimension, (uint64_t)degree);
}

TEST(logSigCombineCudaTest, BatchChenIdentity) {
    uint64_t dimension = 2, degree = 3, batch_size = 2;
    (void)prepare_log_sig_cuda(dimension, degree, 2);

    std::vector<double> path1 = {
        0., 0., 1., 0.5, 0.4, 2.,
        0., 0., 0.25, 0.25, 0.5, 0.5 };
    std::vector<double> path2 = {
        0.4, 2., 6., 0.1, 2.3, 4.1,
        0.5, 0.5, 1., 1., 0.75, 0.75 };
    std::vector<double> path = {
        0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1,
        0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1., 0.75, 0.75 };

    uint64_t slen = sig_length_(dimension, degree);
    uint64_t ls_len = log_sig_length_(dimension, degree);
    uint64_t total_slen = slen * batch_size;
    uint64_t total_ls_len = ls_len * batch_size;

    std::vector<double> sig1 = compute_batch_sig_on_gpu(path1, batch_size, dimension, (uint64_t)3, degree);
    std::vector<double> sig2 = compute_batch_sig_on_gpu(path2, batch_size, dimension, (uint64_t)3, degree);
    std::vector<double> sig_full = compute_batch_sig_on_gpu(path, batch_size, dimension, (uint64_t)5, degree);

    double* d_sig = nullptr;
    double* d_ls = nullptr;
    cudaMalloc(&d_sig, sizeof(double) * total_slen);
    cudaMalloc(&d_ls, sizeof(double) * total_ls_len);

    cudaMemcpy(d_sig, sig1.data(), sizeof(double) * total_slen, cudaMemcpyHostToDevice);
    int err = sig_to_log_sig_cuda_d(d_sig, d_ls, batch_size, dimension, degree, 2, true);
    cudaDeviceSynchronize();
    std::vector<double> ls1(total_ls_len);
    cudaMemcpy(ls1.data(), d_ls, sizeof(double) * total_ls_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "sig_to_log_sig_cuda_d failed for batch sig1";

    cudaMemcpy(d_sig, sig2.data(), sizeof(double) * total_slen, cudaMemcpyHostToDevice);
    err = sig_to_log_sig_cuda_d(d_sig, d_ls, batch_size, dimension, degree, 2, true);
    cudaDeviceSynchronize();
    std::vector<double> ls2(total_ls_len);
    cudaMemcpy(ls2.data(), d_ls, sizeof(double) * total_ls_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "sig_to_log_sig_cuda_d failed for batch sig2";

    cudaMemcpy(d_sig, sig_full.data(), sizeof(double) * total_slen, cudaMemcpyHostToDevice);
    err = sig_to_log_sig_cuda_d(d_sig, d_ls, batch_size, dimension, degree, 2, true);
    cudaDeviceSynchronize();
    std::vector<double> true_ls(total_ls_len);
    cudaMemcpy(true_ls.data(), d_ls, sizeof(double) * total_ls_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "sig_to_log_sig_cuda_d failed for batch full sig";

    cudaFree(d_sig);
    cudaFree(d_ls);

    check_result_2_typed(log_sig_combine_cuda_d, ls1, ls2, true_ls, batch_size, dimension, (uint64_t)degree);
}

TEST(logSigCombineBackpropCudaTest, FiniteDifference) {
    uint64_t dimension = 2, degree = 3;
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    uint64_t ls_len = log_sig_length_(dimension, degree);

    std::vector<double> ls1(ls_len);
    std::vector<double> ls2(ls_len);
    std::vector<double> upstream(ls_len);
    for (uint64_t i = 0; i < ls_len; ++i) {
        ls1[i] = 0.1 * (i + 1);
        ls2[i] = 0.2 * (i + 1) - 0.5;
        upstream[i] = 0.3 * (i + 1) - 1.;
    }

    double* d_upstream = nullptr;
    double* d_ls1_gpu = nullptr;
    double* d_ls2_gpu = nullptr;
    double* d_grad1 = nullptr;
    double* d_grad2 = nullptr;
    cudaMalloc(&d_upstream, sizeof(double) * ls_len);
    cudaMalloc(&d_ls1_gpu, sizeof(double) * ls_len);
    cudaMalloc(&d_ls2_gpu, sizeof(double) * ls_len);
    cudaMalloc(&d_grad1, sizeof(double) * ls_len);
    cudaMalloc(&d_grad2, sizeof(double) * ls_len);

    cudaMemcpy(d_upstream, upstream.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls1_gpu, ls1.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls2_gpu, ls2.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);

    int err = log_sig_combine_backprop_cuda_d(d_upstream, d_grad1, d_grad2,
        d_ls1_gpu, d_ls2_gpu, (uint64_t)1, dimension, degree);
    cudaDeviceSynchronize();

    std::vector<double> grad1(ls_len), grad2(ls_len);
    cudaMemcpy(grad1.data(), d_grad1, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad2.data(), d_grad2, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);

    cudaFree(d_upstream);
    cudaFree(d_ls1_gpu);
    cudaFree(d_ls2_gpu);
    cudaFree(d_grad1);
    cudaFree(d_grad2);

    EXPECT_EQ(0, err) << "log_sig_combine_backprop_cuda_d failed";

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_fwd = nullptr;
    cudaMalloc(&d_a, sizeof(double) * ls_len);
    cudaMalloc(&d_b, sizeof(double) * ls_len);
    cudaMalloc(&d_fwd, sizeof(double) * ls_len);

    double eps = 1e-7;

    cudaMemcpy(d_b, ls2.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
    for (uint64_t i = 0; i < 5 && i < ls_len; ++i) {
        double orig = ls1[i];

        ls1[i] = orig + eps;
        cudaMemcpy(d_a, ls1.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
        (void)log_sig_combine_cuda_d(d_a, d_b, d_fwd, (uint64_t)1, dimension, degree);
        cudaDeviceSynchronize();
        std::vector<double> out_plus(ls_len);
        cudaMemcpy(out_plus.data(), d_fwd, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);

        ls1[i] = orig - eps;
        cudaMemcpy(d_a, ls1.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
        (void)log_sig_combine_cuda_d(d_a, d_b, d_fwd, (uint64_t)1, dimension, degree);
        cudaDeviceSynchronize();
        std::vector<double> out_minus(ls_len);
        cudaMemcpy(out_minus.data(), d_fwd, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);

        ls1[i] = orig;

        double numerical = 0.;
        for (uint64_t j = 0; j < ls_len; ++j)
            numerical += upstream[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
        EXPECT_TRUE(std::abs(numerical - grad1[i]) < 1e-4);
    }

    cudaMemcpy(d_a, ls1.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
    for (uint64_t i = 0; i < 5 && i < ls_len; ++i) {
        double orig = ls2[i];

        ls2[i] = orig + eps;
        cudaMemcpy(d_b, ls2.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
        (void)log_sig_combine_cuda_d(d_a, d_b, d_fwd, (uint64_t)1, dimension, degree);
        cudaDeviceSynchronize();
        std::vector<double> out_plus(ls_len);
        cudaMemcpy(out_plus.data(), d_fwd, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);

        ls2[i] = orig - eps;
        cudaMemcpy(d_b, ls2.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
        (void)log_sig_combine_cuda_d(d_a, d_b, d_fwd, (uint64_t)1, dimension, degree);
        cudaDeviceSynchronize();
        std::vector<double> out_minus(ls_len);
        cudaMemcpy(out_minus.data(), d_fwd, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);

        ls2[i] = orig;

        double numerical = 0.;
        for (uint64_t j = 0; j < ls_len; ++j)
            numerical += upstream[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
        EXPECT_TRUE(std::abs(numerical - grad2[i]) < 1e-4);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_fwd);
}

TEST(logSigCombineBackpropCudaTest, ZeroDerivative) {
    uint64_t dimension = 2, degree = 3;
    (void)prepare_log_sig_cuda(dimension, degree, 2);
    uint64_t ls_len = log_sig_length_(dimension, degree);

    std::vector<double> ls1(ls_len);
    std::vector<double> ls2(ls_len);
    for (uint64_t i = 0; i < ls_len; ++i) {
        ls1[i] = 0.1 * (i + 1);
        ls2[i] = 0.2 * (i + 1) - 0.5;
    }

    std::vector<double> upstream(ls_len, 0.);

    double* d_upstream = nullptr;
    double* d_ls1_gpu = nullptr;
    double* d_ls2_gpu = nullptr;
    double* d_grad1 = nullptr;
    double* d_grad2 = nullptr;
    cudaMalloc(&d_upstream, sizeof(double) * ls_len);
    cudaMalloc(&d_ls1_gpu, sizeof(double) * ls_len);
    cudaMalloc(&d_ls2_gpu, sizeof(double) * ls_len);
    cudaMalloc(&d_grad1, sizeof(double) * ls_len);
    cudaMalloc(&d_grad2, sizeof(double) * ls_len);

    cudaMemcpy(d_upstream, upstream.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls1_gpu, ls1.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls2_gpu, ls2.data(), sizeof(double) * ls_len, cudaMemcpyHostToDevice);

    int err = log_sig_combine_backprop_cuda_d(d_upstream, d_grad1, d_grad2,
        d_ls1_gpu, d_ls2_gpu, (uint64_t)1, dimension, degree);
    cudaDeviceSynchronize();

    std::vector<double> grad1(ls_len), grad2(ls_len);
    cudaMemcpy(grad1.data(), d_grad1, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad2.data(), d_grad2, sizeof(double) * ls_len, cudaMemcpyDeviceToHost);

    cudaFree(d_upstream);
    cudaFree(d_ls1_gpu);
    cudaFree(d_ls2_gpu);
    cudaFree(d_grad1);
    cudaFree(d_grad2);

    EXPECT_EQ(0, err) << "log_sig_combine_backprop_cuda_d failed";

    for (uint64_t i = 0; i < ls_len; ++i) {
        EXPECT_TRUE(std::abs(grad1[i]) < DOUBLE_EPSILON);
        EXPECT_TRUE(std::abs(grad2[i]) < DOUBLE_EPSILON);
    }
}