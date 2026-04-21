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

namespace cpSigTests {

    TEST_CLASS(lyndonWordsTest) {
    public:
        TEST_METHOD(AllLyndonWordsTest1) {
            uint64_t dimension = 2, degree = 3;
            std::vector<word> result = all_lyndon_words(dimension, degree);
            std::vector<word> true_ = {
                {0}, {1},
                {0,1},
                {0,0,1}, {0,1,1}
            };
            check_result_words(result, true_);
        }

        TEST_METHOD(AllLyndonWordsTest2) {
            uint64_t dimension = 5, degree = 2;
            std::vector<word> result = all_lyndon_words(dimension, degree);
            std::vector<word> true_ = {
                {0}, {1}, {2}, {3}, {4},
                {0,1}, {0,2}, {0,3}, {0,4},
                {1,2}, {1,3}, {1,4},
                {2,3}, {2,4}, {3,4}
            };
            check_result_words(result, true_);
        }

        TEST_METHOD(AllLyndonWordsTest3) {
            uint64_t dimension = 3, degree = 4;
            std::vector<word> result = all_lyndon_words(dimension, degree);
            std::vector<word> true_ = { 
                { 0 }, {1}, {2},
                {0, 1}, {0, 2}, {1, 2}, 
                {0, 0, 1}, {0, 0, 2}, {0, 1, 1}, {0, 1, 2}, {0, 2, 1}, {0, 2, 2}, {1, 1, 2}, {1, 2, 2},
                {0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 1, 1}, {0, 0, 1, 2}, {0, 0, 2, 1}, {0, 0, 2, 2},
                {0, 1, 0, 2}, {0, 1, 1, 1}, {0, 1, 1, 2}, {0, 1, 2, 1}, {0, 1, 2, 2}, {0, 2, 1, 1},
                {0, 2, 1, 2}, {0, 2, 2, 1}, {0, 2, 2, 2}, {1, 1, 1, 2}, {1, 1, 2, 2}, {1, 2, 2, 2}
            };
            check_result_words(result, true_);
        }
    };

    TEST_CLASS(lyndonMatrixTest) {
    public:
        TEST_METHOD(lyndonMatrixTest1) {
            uint64_t dimension = 2, degree = 2;
            std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
            std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
            SparseIntMatrix out;
            lyndon_proj_matrix(out, lyndon_words, lyndon_idx, dimension, degree);

            SparseIntMatrix true_(out.n);

            Assert::IsTrue(true_ == out);
        }
        TEST_METHOD(lyndonMatrixTest2) {
            uint64_t dimension = 3, degree = 4;
            std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
            std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
            SparseIntMatrix out;
            lyndon_proj_matrix(out, lyndon_words, lyndon_idx, dimension, degree);

            SparseIntMatrix true_(out.n);
            true_.insert_entry(10, 9, -1);
            true_.insert_entry(18, 17, -1);
            true_.insert_entry(20, 18, -1);
            true_.insert_entry(23, 22, -2);
            true_.insert_entry(25, 22, 1);
            true_.insert_entry(25, 23, -1);
            true_.insert_entry(26, 24, -2);
            true_.insert_entry(27, 24, 1);
            true_.insert_entry(27, 26, -1);

            Assert::IsTrue(true_ == out);
        }

        TEST_METHOD(lyndonMatrixTest3) {
            uint64_t dimension = 2, degree = 5;
            std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
            std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
            SparseIntMatrix out;
            lyndon_proj_matrix(out, lyndon_words, lyndon_idx, dimension, degree);

            SparseIntMatrix true_(out.n);
            true_.insert_entry(10, 9, -2);
            true_.insert_entry(12, 11, -3);

            Assert::IsTrue(true_ == out);
        }
    };

    TEST_CLASS(logSignatureExpandedTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(ManualLogSigTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> true_ = { 0., 0., 1., 0., 1., -1., 0. };
            std::vector<double> sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(ManualLogSigTest2) {
            auto f = sig_to_log_sig_f;
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
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(BatchLogSigTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;

            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0.,
                0., 1., 1., 0., 0., 0., 0.,
                0., 0., 1., 0., 1., -1., 0.};

            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            check_result(f, sig, true_, 3, dimension, degree, false, false, 0, true, 1);
            check_result(f, sig, true_, 3, dimension, degree, false, false, 0, true, -1);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 0.f, 9.f, 4.f, 0.f, -2.5f, 2.5f, 0.f, 0.f, -5.25f,
                                10.5f, 5.5f, -5.25f, -11.f, 5.5f, 0.f};
            std::vector<float> sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                                64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, true, false, 0, true, 1);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 0., 9., 9., 0., -31.5, 31.5, 0., 0., 26.75, -53.5, 11.75, 26.75, -23.5, 11.75, 0. };
            std::vector<float> sig = { 1., 9., 9., 40.5, 9., 72., 40.5, 121.5, 6.5, 68., -8.5, 290., 98., 275., 121.5 };
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, true, 0, true, 1);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2, batch = 1;
            std::vector<double> out;
            out.resize(batch * sig_length(dimension * 2, degree));
            std::vector<double> sig;
            sig.resize(batch * sig_length(dimension * 2, degree));
            f(sig.data(), out.data(), batch, dimension, degree, false, true, 0, true, 1);
        }
    };

    TEST_CLASS(logSignatureExpandedBackpropTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., -1., -1.,  1.,  1.,  1.,  1. };
            std::vector<double> sig = {1., 1., 1., 0.5, 0.5, 0.5, 0.5};
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 0., -5., -6.25, 3., 4., 5., 6. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { 0., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 0., -5., -6.25, 3., 4., 5., 6. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { 0., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., 6.5, 7.6875, -10., -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14., 0., 66., 30.25, -35., 15.5, -46., 14.5, 7., -8., 9., -10., 11., -12., 13., -14., 0., 1.625, 1.625, 1.5, 1.5, 1.5, 1.5, 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 0, true, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 0, true, -1);
        }

        TEST_METHOD(ManualDim1Test) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 1, degree = 8;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
            std::vector<double> true_ = { 0., -1., 8., 9., 1., -8., -9., -1., 8. };
            std::vector<double> sig = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }
    };

    TEST_CLASS(logSignatureLyndonWordsTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(LinearPathCacheTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            (void)clear_cache(true); // Clear disk
            (void)prepare_log_sig(dimension, degree, 1, true);
            (void)clear_cache(false); // Remove from memory but keep on disk
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(ManualLogSigTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> true_ = { 0., 1., 1. };
            std::vector<double> sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(ManualLogSigTest2) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 3, degree = 3;
            std::vector<float> true_ = {
                -5.f, -5.f, -6.f, 12.f, -10.f, -6.f, -27.f,
                 11.f, 5.f, 3.f + 2.f / 3.f, 20.f + 2.f / 3.f, -18.f, -9.f, -4.f - 2.f / 3.f
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
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(BatchLogSigTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> true_ = { 1., 1., 0.,
                1., 1., 0.,
                0., 1., 1. };

            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
               1., 1., 1., 0.5, 0.5, 0.5, 0.5,
               1., 0., 1., 0., 1., -1., 0.5 };

            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, 3, dimension, degree, false, false, 1, true, 1);
            check_result(f, sig, true_, 3, dimension, degree, false, false, 1, true, -1);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 9.f, 4.f, -2.5f, -5.25f, 5.5f };
            std::vector<float> sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                                64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, true, false, 1, true, 1);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 9.f, 9.f, -31.5f, 26.75f, 11.75f };
            std::vector<float> sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, true, 1, true, 1);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2, batch = 1;
            std::vector<double> out;
            out.resize(batch * sig_length(dimension * 2, degree));
            std::vector<double> sig;
            sig.resize(batch * sig_length(dimension * 2, degree));
            (void)prepare_log_sig(dimension, degree, 1);
            f(sig.data(), out.data(), batch, dimension, degree, false, true, 1, true, 1);
        }
    };

    TEST_CLASS(logSignatureLyndonWordsBackpropTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1. };
            std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 1, true, 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            (void)prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 1, true, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 1, true, -1);
        }
    };

    TEST_CLASS(logSignatureLyndonBasisTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 2, true, 1);
        }

        TEST_METHOD(LinearPathCacheTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            (void)clear_cache(true); // Clear disk
            (void)prepare_log_sig(dimension, degree, 2, true);
            (void)clear_cache(false); // Clear memory
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 2, true, 1);
        }

        TEST_METHOD(ManualLogSigTest2) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 3, degree = 3;
            std::vector<float> true_ = { -5.f, -5.f, -6.f, 12.f, -10.f, -6.f, -27.f,
            11.f, 5.f, 3.f + 2.f / 3.f, 24.f + 1.f / 3.f, -18.f, -9.f, -4.f - 2.f / 3.f };
            std::vector<float> sig = { 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                                                5.f, 0.5f, 12.5f, 9.f, 25.f,
                                               21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                                               33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                                              -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                                              -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                                              -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                                              -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f };
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, (uint64_t)1, dimension, degree, false, false, 2, true, 1);
        }
    };

    TEST_CLASS(logSignatureLyndonBasisBackpropTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1. };
            std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 2, true, 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 2, true, 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            (void)prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 2, true, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 2, true, -1);
        }
    };

    TEST_CLASS(logsigToSigTest) {
    public:

        // Round-trip: sig -> log_sig -> logsig_to_sig should recover original sig
        TEST_METHOD(RoundTripDeg1) {
            uint64_t dimension = 3, degree = 1;
            uint64_t s = sig_length(dimension, degree);
            // exp(x) = 1 + x for degree 1
            std::vector<double> log_sig = { 0., 2., 3., 5. };
            std::vector<double> true_ = { 1., 2., 3., 5. };
            check_result(logsig_to_sig_d, log_sig, true_, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(RoundTripDeg2) {
            uint64_t dimension = 2, degree = 2;
            // Use a known sig/log-sig pair
            std::vector<double> sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            std::vector<double> log_sig(sig.size());
            (void)sig_to_log_sig_d(sig.data(), log_sig.data(), (uint64_t)1, dimension, degree, false, false, 0, 1);
            // Now round-trip
            check_result(logsig_to_sig_d, log_sig, sig, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(RoundTripDeg3) {
            uint64_t dimension = 3, degree = 3;
            uint64_t s = sig_length(dimension, degree);
            // Create a random-ish signature via a known path
            std::vector<double> path = { 1., 2., 3., 4., 5., 6., 7., 8., 9. }; // 3x3 path
            std::vector<double> sig(s);
            (void)signature_d(path.data(), sig.data(), 1, dimension, 3, degree, false, false, 1., true, 1);
            std::vector<double> log_sig(s);
            (void)sig_to_log_sig_d(sig.data(), log_sig.data(), (uint64_t)1, dimension, degree, false, false, 0, 1);
            check_result(logsig_to_sig_d, log_sig, sig, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }

        TEST_METHOD(RoundTripFloat32) {
            uint64_t dimension = 2, degree = 3;
            uint64_t s = sig_length(dimension, degree);
            std::vector<float> path = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
            std::vector<float> sig(s);
            (void)signature_f(path.data(), sig.data(), 1, dimension, 3, degree, false, false, 1.f, true, 1);
            std::vector<float> log_sig(s);
            (void)sig_to_log_sig_f(sig.data(), log_sig.data(), (uint64_t)1, dimension, degree, false, false, 0, 1);
            std::vector<float> recovered(s);
            (void)logsig_to_sig_f(log_sig.data(), recovered.data(), (uint64_t)1, dimension, degree, false, false, 0, 1);
            for (uint64_t i = 0; i < s; ++i)
                Assert::IsTrue(abs(sig[i] - recovered[i]) < SINGLE_EPSILON);
        }

        TEST_METHOD(BatchRoundTrip) {
            uint64_t dimension = 2, degree = 3, batch_size = 4;
            uint64_t s = sig_length(dimension, degree);
            std::vector<double> path = {
                1., 2., 3., 4., 5., 6.,
                2., 3., 4., 5., 6., 7.,
                0., 1., 1., 0., 2., 1.,
                3., 1., 4., 1., 5., 9.
            };
            std::vector<double> sig(s * batch_size);
            (void)signature_d(path.data(), sig.data(), batch_size, dimension, 3, degree, false, false, 1., true, 1);
            std::vector<double> log_sig(s * batch_size);
            (void)sig_to_log_sig_d(sig.data(), log_sig.data(), batch_size, dimension, degree, false, false, 0, 1);
            std::vector<double> recovered(s * batch_size);
            (void)logsig_to_sig_d(log_sig.data(), recovered.data(), batch_size, dimension, degree, false, false, 0, 1);
            for (uint64_t i = 0; i < s * batch_size; ++i)
                Assert::IsTrue(abs(sig[i] - recovered[i]) < DOUBLE_EPSILON);
        }

        TEST_METHOD(ZeroLogSigGivesIdentity) {
            uint64_t dimension = 3, degree = 3;
            uint64_t s = sig_length(dimension, degree);
            std::vector<double> log_sig(s, 0.);
            std::vector<double> true_(s, 0.);
            true_[0] = 1.; // identity signature
            check_result(logsig_to_sig_d, log_sig, true_, (uint64_t)1, dimension, degree, false, false, 0, true, 1);
        }
    };

    TEST_CLASS(logSigCombineTest) {
    public:
        TEST_METHOD(ChenIdentity) {
            uint64_t dimension = 2, degree = 3;
            (void)prepare_log_sig(dimension, degree, 2, false);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

            uint64_t s_len = sig_length(dimension, degree);
            uint64_t ls_len = log_sig_length(dimension, degree);

            std::vector<double> sig1(s_len);
            (void)signature_d(path1.data(), sig1.data(), (uint64_t)1, dimension, 3, degree);
            std::vector<double> ls1(ls_len);
            (void)sig_to_log_sig_d(sig1.data(), ls1.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);

            std::vector<double> sig2(s_len);
            (void)signature_d(path2.data(), sig2.data(), (uint64_t)1, dimension, 3, degree);
            std::vector<double> ls2(ls_len);
            (void)sig_to_log_sig_d(sig2.data(), ls2.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);

            std::vector<double> sig_full(s_len);
            (void)signature_d(path.data(), sig_full.data(), (uint64_t)1, dimension, 5, degree);
            std::vector<double> true_ls(ls_len);
            (void)sig_to_log_sig_d(sig_full.data(), true_ls.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);

            check_result_2(log_sig_combine_d, ls1, ls2, true_ls, (uint64_t)1, dimension, degree, 1);
        }

        TEST_METHOD(ChenIdentityHighDeg) {
            uint64_t dimension = 3, degree = 5;
            (void)prepare_log_sig(dimension, degree, 2, false);

            std::vector<double> path1 = { 0., 0., 0., 1., 0.5, 0.3, 0.4, 2., 1.5 };
            std::vector<double> path2 = { 0.4, 2., 1.5, 6., 0.1, 0.8, 2.3, 4.1, 3.2 };
            std::vector<double> path = { 0., 0., 0., 1., 0.5, 0.3, 0.4, 2., 1.5, 6., 0.1, 0.8, 2.3, 4.1, 3.2 };

            uint64_t s_len = sig_length(dimension, degree);
            uint64_t ls_len = log_sig_length(dimension, degree);

            std::vector<double> sig1(s_len);
            (void)signature_d(path1.data(), sig1.data(), (uint64_t)1, dimension, 3, degree);
            std::vector<double> ls1(ls_len);
            (void)sig_to_log_sig_d(sig1.data(), ls1.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);

            std::vector<double> sig2(s_len);
            (void)signature_d(path2.data(), sig2.data(), (uint64_t)1, dimension, 3, degree);
            std::vector<double> ls2(ls_len);
            (void)sig_to_log_sig_d(sig2.data(), ls2.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);

            std::vector<double> sig_full(s_len);
            (void)signature_d(path.data(), sig_full.data(), (uint64_t)1, dimension, 5, degree);
            std::vector<double> true_ls(ls_len);
            (void)sig_to_log_sig_d(sig_full.data(), true_ls.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);

            check_result_2(log_sig_combine_d, ls1, ls2, true_ls, (uint64_t)1, dimension, degree, 1);
        }

        TEST_METHOD(BatchChenIdentity) {
            uint64_t dimension = 2, degree = 3, batch_size = 2;
            (void)prepare_log_sig(dimension, degree, 2, false);

            std::vector<double> path1 = {
                0., 0., 1., 0.5, 0.4, 2.,
                0., 0., 0.25, 0.25, 0.5, 0.5 };
            std::vector<double> path2 = {
                0.4, 2., 6., 0.1, 2.3, 4.1,
                0.5, 0.5, 1., 1., 0.75, 0.75 };
            std::vector<double> path = {
                0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1,
                0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1., 0.75, 0.75 };

            uint64_t s_len = sig_length(dimension, degree);
            uint64_t ls_len = log_sig_length(dimension, degree);

            std::vector<double> sig1(s_len * batch_size);
            (void)signature_d(path1.data(), sig1.data(), batch_size, dimension, 3, degree);
            std::vector<double> ls1(ls_len * batch_size);
            (void)sig_to_log_sig_d(sig1.data(), ls1.data(), batch_size, dimension, degree, false, false, 2, true, 1);

            std::vector<double> sig2(s_len * batch_size);
            (void)signature_d(path2.data(), sig2.data(), batch_size, dimension, 3, degree);
            std::vector<double> ls2(ls_len * batch_size);
            (void)sig_to_log_sig_d(sig2.data(), ls2.data(), batch_size, dimension, degree, false, false, 2, true, 1);

            std::vector<double> sig_full(s_len * batch_size);
            (void)signature_d(path.data(), sig_full.data(), batch_size, dimension, 5, degree);
            std::vector<double> true_ls(ls_len * batch_size);
            (void)sig_to_log_sig_d(sig_full.data(), true_ls.data(), batch_size, dimension, degree, false, false, 2, true, 1);

            check_result_2(log_sig_combine_d, ls1, ls2, true_ls, batch_size, dimension, degree, 1);
        }

        TEST_METHOD(IdentityElement) {
            uint64_t dimension = 2, degree = 3;
            (void)prepare_log_sig(dimension, degree, 2, false);

            uint64_t s_len = sig_length(dimension, degree);
            uint64_t ls_len = log_sig_length(dimension, degree);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> sig1(s_len);
            (void)signature_d(path1.data(), sig1.data(), (uint64_t)1, dimension, 3, degree);
            std::vector<double> ls1(ls_len);
            (void)sig_to_log_sig_d(sig1.data(), ls1.data(), (uint64_t)1, dimension, degree, false, false, 2, true, 1);

            std::vector<double> zero_ls(ls_len, 0.);

            check_result_2(log_sig_combine_d, zero_ls, ls1, ls1, (uint64_t)1, dimension, degree, 1);
            check_result_2(log_sig_combine_d, ls1, zero_ls, ls1, (uint64_t)1, dimension, degree, 1);
        }
    };

    TEST_CLASS(logSigCombineBackpropTest) {
    public:
        TEST_METHOD(FiniteDifference) {
            uint64_t dimension = 2, degree = 3;
            (void)prepare_log_sig(dimension, degree, 2, false);
            uint64_t ls_len = log_sig_length(dimension, degree);

            std::vector<double> ls1(ls_len);
            std::vector<double> ls2(ls_len);
            std::vector<double> d_out(ls_len);
            for (uint64_t i = 0; i < ls_len; ++i) {
                ls1[i] = 0.1 * (i + 1);
                ls2[i] = 0.2 * (i + 1) - 0.5;
                d_out[i] = 0.3 * (i + 1) - 1.;
            }

            std::vector<double> d_ls1(ls_len);
            std::vector<double> d_ls2(ls_len);
            (void)log_sig_combine_backprop_d(d_out.data(), d_ls1.data(), d_ls2.data(),
                ls1.data(), ls2.data(), (uint64_t)1, dimension, degree, 1);

            double eps = 1e-7;
            for (uint64_t i = 0; i < 5 && i < ls_len; ++i) {
                double orig = ls1[i];
                ls1[i] = orig + eps;
                std::vector<double> out_plus(ls_len);
                (void)log_sig_combine_d(ls1.data(), ls2.data(), out_plus.data(), (uint64_t)1, dimension, degree, 1);
                ls1[i] = orig - eps;
                std::vector<double> out_minus(ls_len);
                (void)log_sig_combine_d(ls1.data(), ls2.data(), out_minus.data(), (uint64_t)1, dimension, degree, 1);
                ls1[i] = orig;

                double numerical = 0.;
                for (uint64_t j = 0; j < ls_len; ++j)
                    numerical += d_out[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
                Assert::IsTrue(abs(numerical - d_ls1[i]) < 1e-4);
            }

            for (uint64_t i = 0; i < 5 && i < ls_len; ++i) {
                double orig = ls2[i];
                ls2[i] = orig + eps;
                std::vector<double> out_plus(ls_len);
                (void)log_sig_combine_d(ls1.data(), ls2.data(), out_plus.data(), (uint64_t)1, dimension, degree, 1);
                ls2[i] = orig - eps;
                std::vector<double> out_minus(ls_len);
                (void)log_sig_combine_d(ls1.data(), ls2.data(), out_minus.data(), (uint64_t)1, dimension, degree, 1);
                ls2[i] = orig;

                double numerical = 0.;
                for (uint64_t j = 0; j < ls_len; ++j)
                    numerical += d_out[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
                Assert::IsTrue(abs(numerical - d_ls2[i]) < 1e-4);
            }
        }

        TEST_METHOD(ZeroDerivative) {
            uint64_t dimension = 2, degree = 3;
            (void)prepare_log_sig(dimension, degree, 2, false);
            uint64_t ls_len = log_sig_length(dimension, degree);

            std::vector<double> ls1(ls_len);
            std::vector<double> ls2(ls_len);
            for (uint64_t i = 0; i < ls_len; ++i) {
                ls1[i] = 0.1 * (i + 1);
                ls2[i] = 0.2 * (i + 1) - 0.5;
            }

            std::vector<double> d_out(ls_len, 0.);
            std::vector<double> d_ls1(ls_len);
            std::vector<double> d_ls2(ls_len);
            (void)log_sig_combine_backprop_d(d_out.data(), d_ls1.data(), d_ls2.data(),
                ls1.data(), ls2.data(), (uint64_t)1, dimension, degree, 1);

            for (uint64_t i = 0; i < ls_len; ++i) {
                Assert::IsTrue(abs(d_ls1[i]) < DOUBLE_EPSILON);
                Assert::IsTrue(abs(d_ls2[i]) < DOUBLE_EPSILON);
            }
        }

        TEST_METHOD(BatchFiniteDifference) {
            uint64_t dimension = 2, degree = 3, batch_size = 2;
            (void)prepare_log_sig(dimension, degree, 2, false);
            uint64_t ls_len = log_sig_length(dimension, degree);

            std::vector<double> ls1(ls_len * batch_size);
            std::vector<double> ls2(ls_len * batch_size);
            std::vector<double> d_out(ls_len * batch_size);
            for (uint64_t i = 0; i < ls_len * batch_size; ++i) {
                ls1[i] = 0.1 * (i + 1);
                ls2[i] = 0.2 * (i + 1) - 0.5;
                d_out[i] = 0.3 * (i + 1) - 1.;
            }

            std::vector<double> d_ls1_serial(ls_len * batch_size);
            std::vector<double> d_ls2_serial(ls_len * batch_size);
            (void)log_sig_combine_backprop_d(d_out.data(), d_ls1_serial.data(), d_ls2_serial.data(),
                ls1.data(), ls2.data(), batch_size, dimension, degree, 1);

            std::vector<double> d_ls1_parallel(ls_len * batch_size);
            std::vector<double> d_ls2_parallel(ls_len * batch_size);
            (void)log_sig_combine_backprop_d(d_out.data(), d_ls1_parallel.data(), d_ls2_parallel.data(),
                ls1.data(), ls2.data(), batch_size, dimension, degree, -1);

            for (uint64_t i = 0; i < ls_len * batch_size; ++i) {
                Assert::IsTrue(abs(d_ls1_serial[i] - d_ls1_parallel[i]) < DOUBLE_EPSILON);
                Assert::IsTrue(abs(d_ls2_serial[i] - d_ls2_parallel[i]) < DOUBLE_EPSILON);
            }
        }
    };

}