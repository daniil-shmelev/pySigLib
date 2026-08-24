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

#include <string>
#include <thread>

TEST(cudaErrorDetailTest, PreservesDetailAndClearsAfterSuccess) {
	EXPECT_EQ(2, signature_cuda_d(
		nullptr, nullptr, 1, 0, 0, 0, false, false, 1., true, true,
		nullptr, 0, 0, 0));
	EXPECT_STREQ(
		cusig_last_error_message(),
		"signature_cuda received path of dimension 0");

	EXPECT_EQ(0, signature_cuda_d(
		nullptr, nullptr, 0, 1, 0, 0, false, false, 1., true, true,
		nullptr, 0, 0, 0));
	EXPECT_STREQ(cusig_last_error_message(), "");
}

TEST(cudaErrorDetailTest, IsThreadLocal) {
	EXPECT_EQ(2, signature_cuda_d(
		nullptr, nullptr, 1, 0, 0, 0, false, false, 1., true, true,
		nullptr, 0, 0, 0));
	std::string child_message;
	std::thread child([&child_message]() {
		EXPECT_EQ(2, signature_cuda_d(
			nullptr, nullptr, 1, 1, 2, 2, false, false, 1., true, true,
			nullptr, 1, 0, 0));
		child_message = cusig_last_error_message();
	});
	child.join();
	EXPECT_NE(child_message.find("correction pointer is null"), std::string::npos);
	EXPECT_STREQ(
		cusig_last_error_message(),
		"signature_cuda received path of dimension 0");
}

TEST(signatureDoubleTest, TrivialCases) {
    auto f = signature_cuda_d;
    std::vector<double> path;
    std::vector<double> true_sig;
    EXPECT_EQ(2, f(nullptr, nullptr, (uint64_t)1, 0, 0, 0, false, false, 1., true, true, nullptr, 0, 0, 0));

    true_sig.push_back(1.);
    check_result_typed(f, path, true_sig, (uint64_t)1, 1, 0, 0, false, false, 1., true, true);

    path.push_back(0.);
    check_result_typed(f, path, true_sig, (uint64_t)1, 1, 1, 0, false, false, 1., true, true);

    true_sig.push_back(0.);
    check_result_typed(f, path, true_sig, (uint64_t)1, 1, 0, 1, false, false, 1., true, true);
    check_result_typed(f, path, true_sig, (uint64_t)1, 1, 1, 1, false, false, 1., true, true);

    path.push_back(1.);
    true_sig[1] = 1.;
    check_result_typed(f, path, true_sig, (uint64_t)1, 1, 2, 1, false, false, 1., true, true);
}

TEST(signatureDoubleTest, LinearPathTest) {
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
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1., true, true);
}

TEST(signatureDoubleTest, LinearPathTest2) {
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
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1., true, true);
}

TEST(signatureDoubleTest, ManualSigTest) {
    auto f = signature_cuda_d;
    uint64_t dimension = 2, length = 4, degree = 2;
    std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
    std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1., true, true);
}

TEST(signatureDoubleTest, ManualSigTestDirect) {
    auto f = signature_cuda_d;
    uint64_t dimension = 2, length = 4, degree = 2;
    std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
    std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1., false, true);
}

TEST(signatureDoubleTest, CorrectionSingleSegment) {
    auto f = signature_cuda_d;
    uint64_t dimension = 1, length = 2, degree = 4;
    std::vector<double> path = { 0., 3. };
    std::vector<double> correction = { 2. };
    std::vector<double> true_sig = { 1., 3., 6.5, 10.5, 14.375 };
    std::vector<double> out(true_sig.size() + 1, 0.);
    out[true_sig.size()] = -1.;

    double* d_path;
    double* d_out;
    double* d_correction;
    cudaMalloc(&d_path, sizeof(double) * path.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());
    cudaMalloc(&d_correction, sizeof(double) * correction.size());
    cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data(), sizeof(double) * out.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_correction, correction.data(), sizeof(double) * correction.size(), cudaMemcpyHostToDevice);

    int err = f(d_path, d_out, (uint64_t)1, dimension, length, degree, false, false, 1., true, true, d_correction, correction.size(), 0, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_correction);

    EXPECT_EQ(0, err);
    for (uint64_t i = 0; i < true_sig.size(); ++i)
        EXPECT_TRUE(std::abs(true_sig[i] - out[i]) < DOUBLE_EPSILON);
    EXPECT_TRUE(std::abs(-1. - out[true_sig.size()]) < DOUBLE_EPSILON);
}

TEST(signatureDoubleTest, CorrectionTimeAug) {
    auto f = signature_cuda_d;
    uint64_t dimension = 1, length = 2, degree = 2;
    std::vector<double> path = { 0., 3. };
    std::vector<double> correction = { 2. };
    std::vector<double> true_sig = { 1., 3., 1., 6.5, 1.5, 1.5, 0.5 };
    std::vector<double> out(true_sig.size() + 1, 0.);
    out[true_sig.size()] = -1.;

    double* d_path;
    double* d_out;
    double* d_correction;
    cudaMalloc(&d_path, sizeof(double) * path.size());
    cudaMalloc(&d_out, sizeof(double) * out.size());
    cudaMalloc(&d_correction, sizeof(double) * correction.size());
    cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data(), sizeof(double) * out.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_correction, correction.data(), sizeof(double) * correction.size(), cudaMemcpyHostToDevice);

    int err = f(d_path, d_out, (uint64_t)1, dimension, length, degree, true, false, 1., true, true, d_correction, correction.size(), 0, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_correction);

    EXPECT_EQ(0, err);
    for (uint64_t i = 0; i < true_sig.size(); ++i)
        EXPECT_TRUE(std::abs(true_sig[i] - out[i]) < DOUBLE_EPSILON);
    EXPECT_TRUE(std::abs(-1. - out[true_sig.size()]) < DOUBLE_EPSILON);
}

TEST(signatureDoubleTest, ManualTimeAugTest) {
    auto f = signature_cuda_d;
    uint64_t dimension = 1, length = 5, degree = 3;
    std::vector<double> path = { 0., 5., 2., 4., 9. };
    std::vector<double> true_sig = { 1., 9., 4., 40.5, 15.5, 20.5, 8., 121.5, 37.5,
                        64.5, 24.5, 60., 13., 34.5, 10. + 2. / 3. };
    double end_time = length - 1.;
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, true, false, end_time, true, true);
}

TEST(signatureDoubleTest, ManualLeadLagTest) {
    auto f = signature_cuda_d;
    uint64_t dimension = 1, length = 5, degree = 3;
    std::vector<double> path = { 0., 5., 2., 4., 9. };
    std::vector<double> true_sig = { 1., 9., 9., 40.5, 9., 72., 40.5, 121.5, 6.5, 68., -8.5, 290., 98., 275., 121.5 };
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, true, 1., true, true);
}

TEST(signatureFloatTest, ManualSigTest2) {
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
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1.f, true, true);
}

TEST(signatureFloatTest, ManualSigTest2Direct) {
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
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1.f, false, true);
}

TEST(signatureFloatTest, ManualTimeAugTest) {
    auto f = signature_cuda_f;
    uint64_t dimension = 1, length = 5, degree = 3;
    std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
    std::vector<float> true_sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                        64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
    float end_time = static_cast<float>(length - 1);
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, true, false, end_time, true, true);
}

TEST(signatureFloatTest, ManualLeadLagTest) {
    auto f = signature_cuda_f;
    uint64_t dimension = 1, length = 5, degree = 3;
    std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
    std::vector<float> true_sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
    check_result_typed(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, true, 1.f, true, true);
}

TEST(batchSignatureTest, BatchSigTest) {
    auto f = signature_cuda_d;
    uint64_t dimension = 2, length = 4, degree = 2;
    std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
        0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
        0., 0., 1., 0.5, 4., 0., 0., 1. };

    std::vector<double> true_sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 0., 1., 0., 1., -1., 0.5 };

    check_result_typed(f, path, true_sig, (uint64_t)3, dimension, length, degree, false, false, 1., true, true);
}

TEST(batchSignatureTest, BatchSigTestDirect) {
    auto f = signature_cuda_d;
    uint64_t dimension = 2, length = 4, degree = 2;
    std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
        0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
        0., 0., 1., 0.5, 4., 0., 0., 1. };

    std::vector<double> true_sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 0., 1., 0., 1., -1., 0.5 };

    check_result_typed(f, path, true_sig, (uint64_t)3, dimension, length, degree, false, false, 1., false, true);
}

TEST(batchSignatureTest, BatchSigTestDegree1) {
    auto f = signature_cuda_d;
    uint64_t dimension = 2, length = 4, degree = 1;
    std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
        0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
        0., 0., 1., 0.5, 4., 0., 0., 1. };

    std::vector<double> true_sig = { 1., 1., 1.,
        1., 1., 1.,
        1., 0., 1. };

    check_result_typed(f, path, true_sig, (uint64_t)3, dimension, length, degree, false, false, 1., true, true);
}

TEST(batchSignatureTest, BigLeadLagTest) {
    auto f = signature_cuda_d;
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

    int err = f(d_path, d_out, batch, dimension, length, degree, false, true, 1., true, true, nullptr, 0, 0, 0);

    cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_path);
    cudaFree(d_out);

    EXPECT_EQ(0, err);
}

TEST(batchSignatureTest, BatchSigTestFloat) {
    auto f = signature_cuda_f;
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
    check_result_typed(f, path, true_sig, (uint64_t)2, dimension, length, degree, false, false, 1.f, true, true);
}

TEST(sigCombineDoubleTest, SigCombineTestLinear) {
    auto f = sig_combine_cuda_d;
    std::vector<double> poly = { 1., 1., 1., 1. / 2, 1. / 2, 1. / 2, 1. / 2 };
    std::vector<double> true_res = { 1., 2., 2., 2., 2., 2., 2. };

    check_result_2_typed(f, poly, poly, true_res, (uint64_t)1, (uint64_t)2, (uint64_t)2, true);
}

TEST(sigCombineDoubleTest, SigCombineSigTest) {
    uint64_t dimension = 2, degree = 5;
    auto f = sig_combine_cuda_d;
    std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
    std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
    std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

    std::vector<double> poly1 = compute_sig_on_gpu(path1, dimension, 3, degree);
    std::vector<double> poly2 = compute_sig_on_gpu(path2, dimension, 3, degree);
    std::vector<double> true_sig = compute_sig_on_gpu(path, dimension, 5, degree);

    check_result_2_typed(f, poly1, poly2, true_sig, (uint64_t)1, dimension, (uint64_t)degree, true);
}

TEST(sigCombineFloatTest, SigCombineTestLinear) {
    auto f = sig_combine_cuda_f;
    std::vector<float> poly = { 1.f, 1.f, 1.f, 1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2 };
    std::vector<float> true_res = { 1.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };

    check_result_2_typed(f, poly, poly, true_res, (uint64_t)1, (uint64_t)2, (uint64_t)2, true);
}

TEST(sigCombineFloatTest, SigCombineSigTest) {
    uint64_t dimension = 2, degree = 5;
    auto f = sig_combine_cuda_f;
    std::vector<float> path1 = { 0.f, 0.f, 1.f, 0.5f, 0.4f, 2.f };
    std::vector<float> path2 = { 0.4f, 2.f, 6.f, 0.1f, 2.3f, 4.1f };
    std::vector<float> path = { 0.f, 0.f, 1.f, 0.5f, 0.4f, 2.f, 6.f, 0.1f, 2.3f, 4.1f };

    std::vector<float> poly1 = compute_sig_on_gpu(path1, dimension, (uint64_t)3, degree);
    std::vector<float> poly2 = compute_sig_on_gpu(path2, dimension, (uint64_t)3, degree);
    std::vector<float> true_sig = compute_sig_on_gpu(path, dimension, (uint64_t)5, degree);

    check_result_2_typed(f, poly1, poly2, true_sig, (uint64_t)1, dimension, (uint64_t)degree, true);
}

TEST(batchSigCombineTest, BatchSigCombineSigTest) {
    uint64_t batch_size = 3, dimension = 2, degree = 2;
    auto f = sig_combine_cuda_d;
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

    check_result_2_typed(f, poly1, poly2, true_sig, batch_size, dimension, (uint64_t)degree, true);
}

TEST(batchSigCombineTest, BatchSigCombineSigTestFloat) {
    uint64_t batch_size = 3, dimension = 2, degree = 2;
    auto f = sig_combine_cuda_f;
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

    check_result_2_typed(f, poly1, poly2, true_sig, batch_size, dimension, (uint64_t)degree, true);
}

// =========================================================================
// sig_combine_backprop CUDA tests
// Ported from CPU sigCombineBackpropTest
// =========================================================================

TEST(sigCombineBackpropCudaTest, ManualTest) {
    uint64_t dimension = 2, degree = 2;
    uint64_t sig_len = 7;

    std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5 };
    std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5 };
    std::vector<double> derivs = { 1., 1., 2., 3., 4., 5., 6. };
    std::vector<double> true_ = { 0., 5., 8., 3., 4., 5., 6., 0., 9., 12., 3., 4., 5., 6. };

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

    int err = sig_combine_backprop_cuda_d(d_derivs, d_sig1_deriv, d_sig2_deriv, d_sig1, d_sig2, (uint64_t)1, dimension, degree, true);
    cudaDeviceSynchronize();

    std::vector<double> sig1_deriv(sig_len), sig2_deriv(sig_len);
    cudaMemcpy(sig1_deriv.data(), d_sig1_deriv, sizeof(double) * sig_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(sig2_deriv.data(), d_sig2_deriv, sizeof(double) * sig_len, cudaMemcpyDeviceToHost);

    cudaFree(d_derivs);
    cudaFree(d_sig1);
    cudaFree(d_sig2);
    cudaFree(d_sig1_deriv);
    cudaFree(d_sig2_deriv);

    EXPECT_EQ(0, err) << "sig_combine_backprop_cuda_d returned non-zero error code";

    for (uint64_t i = 0; i < sig_len; ++i) {
        std::string msg = "sig1_deriv mismatch at " + std::to_string(i);
        EXPECT_TRUE(std::abs(sig1_deriv[i] - true_[i]) < DOUBLE_EPSILON) << msg;
    }
    for (uint64_t i = 0; i < sig_len; ++i) {
        std::string msg = "sig2_deriv mismatch at " + std::to_string(i);
        EXPECT_TRUE(std::abs(sig2_deriv[i] - true_[sig_len + i]) < DOUBLE_EPSILON) << msg;
    }
}

TEST(sigCombineBackpropCudaTest, ManualBatchTest) {
    uint64_t dimension = 2, degree = 2, batch_size = 2;
    uint64_t sig_len = 7;
    uint64_t total = sig_len * batch_size;

    std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5,
        1., 0., 1., 0., 1., -1., .5 };
    std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5,
        1., 1., 1., .5, .5, .5, .5 };
    std::vector<double> derivs = { 1., 1., 2., 3., 4., 5., 6.,
        1., 1., 2., 3., 4., 5., 6. };
    std::vector<double> true_ = { 0., 5., 8., 3., 4., 5., 6.,
        0., 8., 13., 3., 4., 5., 6.,
        0., 9., 12., 3., 4., 5., 6.,
        0., 6., 8., 3., 4., 5., 6. };

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

    int err = sig_combine_backprop_cuda_d(d_derivs, d_sig1_deriv, d_sig2_deriv, d_sig1, d_sig2, batch_size, dimension, degree, true);
    cudaDeviceSynchronize();

    std::vector<double> sig1_deriv(total), sig2_deriv(total);
    cudaMemcpy(sig1_deriv.data(), d_sig1_deriv, sizeof(double) * total, cudaMemcpyDeviceToHost);
    cudaMemcpy(sig2_deriv.data(), d_sig2_deriv, sizeof(double) * total, cudaMemcpyDeviceToHost);

    cudaFree(d_derivs);
    cudaFree(d_sig1);
    cudaFree(d_sig2);
    cudaFree(d_sig1_deriv);
    cudaFree(d_sig2_deriv);

    EXPECT_EQ(0, err) << "sig_combine_backprop_cuda_d returned non-zero error code";

    for (uint64_t i = 0; i < total; ++i) {
        std::string msg = "sig1_deriv mismatch at " + std::to_string(i);
        EXPECT_TRUE(std::abs(sig1_deriv[i] - true_[i]) < DOUBLE_EPSILON) << msg;
    }
    for (uint64_t i = 0; i < total; ++i) {
        std::string msg = "sig2_deriv mismatch at " + std::to_string(i);
        EXPECT_TRUE(std::abs(sig2_deriv[i] - true_[total + i]) < DOUBLE_EPSILON) << msg;
    }
}
