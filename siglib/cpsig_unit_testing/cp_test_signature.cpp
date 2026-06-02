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

#include "cp_test_helpers.h"

TEST(PolyTest, SigLengthTest) {
    EXPECT_EQ((uint64_t)1, sig_length(0, 0));
    EXPECT_EQ((uint64_t)1, sig_length(0, 1));
    EXPECT_EQ((uint64_t)1, sig_length(1, 0));

    EXPECT_EQ((uint64_t)435848050, sig_length(9, 9));
    EXPECT_EQ((uint64_t)11111111111, sig_length(10, 10));
    EXPECT_EQ((uint64_t)313842837672, sig_length(11, 11));

    EXPECT_EQ((uint64_t)10265664160401, sig_length(400, 5));

    EXPECT_EQ((uint64_t)0, sig_length(100, 100)); // overflow
}

TEST(PolyTest, LogSigLengthTest) {
    EXPECT_EQ((uint64_t)0, log_sig_length(0, 0));
    EXPECT_EQ((uint64_t)0, log_sig_length(0, 1));
    EXPECT_EQ((uint64_t)0, log_sig_length(1, 0));

    EXPECT_EQ((uint64_t)5, log_sig_length(2, 3));

    EXPECT_EQ((uint64_t)49212093, log_sig_length(9, 9));
    EXPECT_EQ((uint64_t)1125217654, log_sig_length(10, 10));
    EXPECT_EQ((uint64_t)26039187, log_sig_length(5, 12));

    EXPECT_EQ((uint64_t)0, log_sig_length(100, 100)); // overflow
}

TEST(PolyTest, BranchedSigLengthTest) {
    EXPECT_EQ((uint64_t)1, branched_sig_length(0, 0));
    EXPECT_EQ((uint64_t)1, branched_sig_length(0, 1));
    EXPECT_EQ((uint64_t)1, branched_sig_length(1, 0));

    EXPECT_EQ((uint64_t)2, branched_sig_length(1, 1));
    EXPECT_EQ((uint64_t)3, branched_sig_length(2, 1));
    EXPECT_EQ((uint64_t)7, branched_sig_length(2, 2));
    EXPECT_EQ((uint64_t)21, branched_sig_length(2, 3));
    EXPECT_EQ((uint64_t)73, branched_sig_length(2, 4));
    EXPECT_EQ((uint64_t)58, branched_sig_length(3, 3));
    EXPECT_EQ((uint64_t)19881, branched_sig_length(5, 5));

    EXPECT_EQ((uint64_t)51, branched_sig_length(2, 3, true));
    EXPECT_EQ((uint64_t)275, branched_sig_length(2, 4, true));
    EXPECT_EQ((uint64_t)157, branched_sig_length(3, 3, true));
}

TEST(PolyTest, SigCombineTestLinear) {
    // Test signatures of linear 2d paths
    auto f = sig_combine_d;
    std::vector<double> poly = { 1., 1., 1., 1./2, 1./2, 1./2, 1./2 };
    std::vector<double> true_res = { 1., 2., 2., 2., 2., 2., 2. };

    check_result_2(f, poly, poly, true_res, (uint64_t)1, 2, 2, true, 1);
}

TEST(PolyTest, SigCombineSigTest) {
    uint64_t dimension = 2, length = 4, degree = 5;
    auto f = sig_combine_d;
    std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
    std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
    std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

    uint64_t poly_len_ = sig_length(dimension, degree);

    std::vector<double> poly1;
    poly1.resize(poly_len_);
    (void)signature_d(path1.data(), poly1.data(), (uint64_t)1, dimension, 3, degree);

    std::vector<double> poly2;
    poly2.resize(poly_len_);
    (void)signature_d(path2.data(), poly2.data(), (uint64_t)1, dimension, 3, degree);

    std::vector<double> true_sig;
    true_sig.resize(poly_len_);
    (void)signature_d(path.data(), true_sig.data(), (uint64_t)1, dimension, 5, degree);
    check_result_2(f, poly1, poly2, true_sig, (uint64_t)1, dimension, degree, true, 1);
}

TEST(PolyTest, BatchSigCombineSigTest) {
    uint64_t batch_size = 3, dimension = 2, length = 4, degree = 2;
    auto f = sig_combine_d;
    std::vector<double> path1 = { 0., 0., 0.25, 0.25, 0.5, 0.5,
        0., 0., 0.4, 0.4, 0.6, 0.6,
        0., 0., 1., 0.5, 4., 0. };
    std::vector<double> path2 = { 0.5, 0.5, 1., 1.,
        0.6, 0.6, 1., 1.,
        4., 0., 0., 1. };
    std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
        0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
        0., 0., 1., 0.5, 4., 0., 0., 1. };

    uint64_t res_len_ = sig_length(dimension, degree) * batch_size;

    std::vector<double> poly1;
    poly1.resize(res_len_);
    (void)signature_d(path1.data(), poly1.data(), batch_size, dimension, 3, degree);

    std::vector<double> poly2;
    poly2.resize(res_len_);
    (void)signature_d(path2.data(), poly2.data(), batch_size, dimension, 2, degree);

    std::vector<double> true_sig;
    true_sig.resize(res_len_);
    (void)signature_d(path.data(), true_sig.data(), batch_size, dimension, 4, degree);
    check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, true, 1);
    check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, true, -1);
}

TEST(PolyTest, BatchSigCombineStressTest) {
    uint64_t batch_size = 1000, dimension = 5, degree = 5;

    std::vector<double> poly;
    poly.resize(batch_size * sig_length(dimension, degree));
    std::fill(poly.data(), poly.data() + poly.size(), 1.);

    std::vector<double> out;
    out.resize(batch_size * sig_length(dimension, degree));

    int err = sig_combine_d(poly.data(), poly.data(), out.data(), batch_size, dimension, degree, true, -1);
    EXPECT_FALSE(err);
}

TEST(signatureDoubleTest, TrivialCases) {
    auto f = signature_d;
    std::vector<double> path;
    std::vector<double> true_sig;
    EXPECT_EQ(2, f(path.data(), true_sig.data(), (uint64_t)1, 0, 0, 0, false, false, 1., true, true, 1, nullptr, 0, 0, 0));

    true_sig.push_back(1.);
    check_result(f, path, true_sig, (uint64_t)1, 1, 0, 0, false, false, 1., true, true, 1);

    path.push_back(0.);
    check_result(f, path, true_sig, (uint64_t)1, 1, 1, 0, false, false, 1., true, true, 1);

    true_sig.push_back(0.);
    check_result(f, path, true_sig, (uint64_t)1, 1, 0, 1, false, false, 1., true, true, 1);
    check_result(f, path, true_sig, (uint64_t)1, 1, 1, 1, false, false, 1., true, true, 1);

    path.push_back(1.);
    true_sig[1] = 1.;
    check_result(f, path, true_sig, (uint64_t)1, 1, 2, 1, false, false, 1., true, true, 1);
}

TEST(signatureDoubleTest, LinearPathTest) {
    auto f = signature_d;
    uint64_t dimension = 2, length = 3, degree = 3;
    uint64_t level_3_start = sig_length(dimension, 2);
    uint64_t level_4_start = sig_length(dimension, 3);
    std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
    std::vector<double> true_sig;
    true_sig.resize(level_4_start);
    true_sig[0] = 1.;
    for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
    for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
    for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
    check_result(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1., true, true, 1);
}

TEST(signatureDoubleTest, LinearPathTest2) {
    auto f = signature_d;
    uint64_t dimension = 2, length = 4, degree = 3;
    uint64_t level_3_start = sig_length(dimension, 2);
    uint64_t level_4_start = sig_length(dimension, 3);
    std::vector<double> path = { 0.,0., 0.25, 0.25, 0.75, 0.75, 1.,1. };
    std::vector<double> true_sig;
    true_sig.resize(level_4_start);
    true_sig[0] = 1.;
    for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
    for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
    for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
    check_result(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1., true, true, 1);
}

TEST(signatureDoubleTest, ManualSigTest) {
    auto f = signature_d;
    uint64_t dimension = 2, length = 4, degree = 2;
    std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
    std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
    check_result(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1., true, true, 1);
}

TEST(signatureDoubleTest, ManualSigTest2) {
    auto f = signature_f;
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
    check_result(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, false, 1.f, true, true, 1);
}

TEST(signatureDoubleTest, BatchSigTest) {
    auto f = signature_d;
    uint64_t dimension = 2, length = 4, degree = 2;
    std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
        0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
        0., 0., 1., 0.5, 4., 0., 0., 1. };

    std::vector<double> true_sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 1., 1., 0.5, 0.5, 0.5, 0.5,
        1., 0., 1., 0., 1., -1., 0.5 };

    check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, true, 1);
    check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, true, -1);
}

TEST(signatureDoubleTest, BatchSigTestDegree1) {
    auto f = signature_d;
    uint64_t dimension = 2, length = 4, degree = 1;
    std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
        0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
        0., 0., 1., 0.5, 4., 0., 0., 1. };

    std::vector<double> true_sig = { 1., 1., 1.,
        1., 1., 1.,
        1., 0., 1. };

    check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, true, 1);
    check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, true, -1);
}

TEST(signatureDoubleTest, ManualTimeAugTest) {
    auto f = signature_f;
    uint64_t dimension = 1, length = 5, degree = 3;
    std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
    std::vector<float> true_sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                            64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f/3.f };
    float end_time = length - 1.f;
    check_result(f, path, true_sig, (uint64_t)1, dimension, length, degree, true, false, end_time, true, true, 1);
}

TEST(signatureDoubleTest, ManualLeadLagTest) {
    auto f = signature_f;
    uint64_t dimension = 1, length = 5, degree = 3;
    std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
    std::vector<float> true_sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
    check_result(f, path, true_sig, (uint64_t)1, dimension, length, degree, false, true, 1.f, true, true, 1);
}

TEST(signatureDoubleTest, BigLeadLagTest) {
    auto f = signature_d;
    uint64_t dimension = 2, length = 10, degree = 2, batch = 1;
    std::vector<double> path;
    path.resize(batch * length * dimension);
    std::vector<double> out;
    out.resize(batch * sig_length(dimension * 2, degree));
    f(path.data(), out.data(), batch, dimension, length, degree, false, true, 1., true, true, 1, nullptr, 0, 0, 0);
}

TEST(sigBackpropTest, LinearPathTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 2, length = 2, degree = 2;
    std::vector<double> path = { 0., 0., 1.,1. };
    std::vector<double> deriv = { 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { -3., -3., 3., 3. };
    std::vector<double> sig = {1., 1., 1., 1./2, 1./2, 1./2, 1./2};
    check_result(f, path, true_, deriv.data(), sig.data(), (uint64_t)1, dimension, length, degree, false, false, 1., true, 1);
}

TEST(sigBackpropTest, ManualTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 2, length = 3, degree = 2;
    std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
    std::vector<double> true_ = { -7.5, -10., -0.5, 0.25, 8., 9.75 };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    check_result(f, path, true_, deriv.data(), sig.data(), (uint64_t)1, dimension, length, degree, false, false, 1., true, 1);
}

TEST(sigBackpropTest, ManualTest2) {
    auto f = sig_backprop_d;
    uint64_t dimension = 2, length = 3, degree = 3;
    std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    check_result(f, path, true_, deriv.data(), sig.data(), (uint64_t)1, dimension, length, degree, false, false, 1., true, 1);
}

TEST(sigBackpropTest, ManualTestAsBatch) {
    auto f = sig_backprop_d;
    uint64_t dimension = 2, length = 3, degree = 2;
    std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
    std::vector<double> true_ = { -7.5, -10., -0.5, 0.25, 8., 9.75 };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
    check_result(f, path, true_, deriv.data(), sig.data(), 1, dimension, length, degree, false, false, 1., true, 1);
}

TEST(sigBackpropTest, ManualTest2AsBatch) {
    auto f = sig_backprop_d;
    uint64_t dimension = 2, length = 3, degree = 3;
    std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23. };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
    check_result(f, path, true_, deriv.data(), sig.data(), 1, dimension, length, degree, false, false, 1., true, 1);
}

TEST(sigBackpropTest, ManualBatchTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 2, length = 3, degree = 3, batch_size = 3;
    std::vector<double> path = { 0., 0., 1., 2., 0.5, 1., 0., 0., 3., 2., 5., 2., 0., 0., -1., 2., 0.5, -1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23., -162.5, -103.5, -81.0, 245.5, 243.5, -142.0, -0.625, -0.625, 0., 0., 0.625, 0.625 };
    std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1./48, 1./24, 1./24, 1./12, 1./24, 1./12, 1./12, 1./6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5./6, 3., 9., 2., 13., 2., 6., 1. + 1./3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1./48, -1./24, -1./24,  1./12, -1./24, 1./48, 1./48, -1./6 };
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, false, 1., true, 1);
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, false, 1., true, -1);
}

TEST(sigBackpropTest, TimeAugTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 1, length = 3, degree = 3;
    std::vector<double> path = { 0., 2., 1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> true_ = { -54., -4.5, 58.5 };
    std::vector<double> sig = { 1., 1., 2., 0.5, 2.5, -0.5, 2., 1./6, 1.5 + 1./3, -1-1./6, 2 + 1./6, 1./3, 2./3, -0.5-1./3, 1 + 1./3 };
    double end_time = length - 1.;
    check_result(f, path, true_, deriv.data(), sig.data(), (uint64_t)1, dimension, length, degree, true, false, end_time, true, 1);
}

TEST(sigBackpropTest, BatchTimeAugTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 1, length = 3, degree = 3, batch_size = 2;
    std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { -54., -4.5, 58.5, -41., 0., 41. };
    std::vector<double> sig = { 1., 1., 2., 0.5, 2.5, -0.5, 2., 1. / 6, 1.5 + 1. / 3, -1 - 1. / 6, 2 + 1. / 6, 1. / 3, 2. / 3, -0.5 - 1. / 3, 1 + 1. / 3,
    1., 6., 2., 18., 6., 6., 2., 36., 12., 12., 4., 12., 4., 4., 4./3};
    double end_time = length - 1.;
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, false, end_time, true, 1);
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, false, end_time, true, -1);
}

TEST(sigBackpropTest, LeadLagTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 1, length = 3, degree = 3;
    std::vector<double> path = { 0., 2., 1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
    std::vector<double> true_ = { -76., 5.5, 70.5 };
    std::vector<double> sig = { 1., 1., 1., .5, -2., 3., .5, 1./6, -2., 2., 1., .5, -4., 3.5, 1./6 };
    check_result(f, path, true_, deriv.data(), sig.data(), (uint64_t)1, dimension, length, degree, false, true, 1., true, 1);
}

TEST(sigBackpropTest, BatchLeadLagTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 1, length = 3, degree = 3, batch_size = 2;
    std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { -76., 5.5, 70.5, -170., 0., 170. };
    std::vector<double> sig = { 1., 1., 1., .5, -2., 3., .5, 1. / 6, -2., 2., 1., .5, -4., 3.5, 1. / 6,
    1., 6., 6., 18., 9., 27., 18., 36., 13.5, 27., 13.5, 67.5, 27., 67.5, 36. };
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, true, 1., true, 1);
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, true, 1., true, -1);
}

TEST(sigBackpropTest, TimeAugLeadLagTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 1, length = 3, degree = 2;
    std::vector<double> path = { 0., 2., 1. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. };
    std::vector<double> true_ = { -98., -6., 104. };
    std::vector<double> sig = { 1., 1., 1., 4., .5, -2., 4.5, 3., .5, 5.5, -.5, -1.5, 8. };
    double end_time = length * 2. - 2.;
    check_result(f, path, true_, deriv.data(), sig.data(), (uint64_t)1, dimension, length, degree, true, true, end_time, true, 1);
}

TEST(sigBackpropTest, BatchTimeAugLeadLagTest) {
    auto f = sig_backprop_d;
    uint64_t dimension = 1, length = 3, degree = 2, batch_size = 2;
    std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
    std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
    std::vector<double> true_ = { -98., -6., 104., -34., 0., 34. };
    std::vector<double> sig = { 1., 1., 1., 4., .5, -2., 4.5, 3., .5, 5.5, -.5, -1.5, 8.,
    1., 6., 6., 4., 18., 9., 9., 27., 18., 15., 15., 9., 8. };
    double end_time = length * 2. - 2.;
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, true, end_time, true, 1);
    check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, true, end_time, true, -1);
}

TEST(sigCombineBackpropTest, ManualTest) {
    auto f = sig_combine_backprop_d;
    uint64_t dimension = 2, degree = 2;
    uint64_t result_length = 7;
    std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5 };
    std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5 };
    std::vector<double> derivs = {1., 1., 2., 3., 4., 5., 6.};
    std::vector<double> true_ = { 0., 5., 8., 3., 4., 5., 6., 0., 9., 12., 3., 4., 5., 6. };

    auto func = [&](double* sig_combined_derivs, double* out, double* sig1, double* sig2, uint64_t dimension, uint64_t degree) {
        f(sig_combined_derivs, out, out + result_length, sig1, sig2, (uint64_t)1, dimension, degree, true, 1);
        };

    check_result(func, derivs, true_, sig1.data(), sig2.data(), dimension, degree);
}

TEST(sigCombineBackpropTest, ManualBatchTest) {
    auto f = sig_combine_backprop_d;
    uint64_t dimension = 2, degree = 2, batch_size = 2;
    uint64_t result_length = 7 * batch_size;
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

    auto func = [&](double* sig_combined_derivs, double* out, double* sig1, double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) {
        f(sig_combined_derivs, out, out + result_length, sig1, sig2, batch_size, dimension, degree, true, n_jobs);
        };

    check_result(func, derivs, true_, sig1.data(), sig2.data(), batch_size, dimension, degree, 1);
    check_result(func, derivs, true_, sig1.data(), sig2.data(), batch_size, dimension, degree, -1);
}
