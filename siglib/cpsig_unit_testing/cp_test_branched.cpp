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

    TEST_CLASS(branchedSigCombineTest) {
    public:
        TEST_METHOD(ChenIdentity) {
            uint64_t dimension = 2, max_nodes = 3;
            (void)prepare_branched_sig(dimension, max_nodes, false, false);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

            uint64_t bs_len = branched_sig_length(dimension, max_nodes);

            std::vector<double> bsig1(bs_len);
            (void)branched_sig_d(path1.data(), bsig1.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

            std::vector<double> bsig2(bs_len);
            (void)branched_sig_d(path2.data(), bsig2.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

            std::vector<double> true_bsig(bs_len);
            (void)branched_sig_d(path.data(), true_bsig.data(), (uint64_t)1, dimension, 5, max_nodes, 1);

            check_result_2(branched_sig_combine_d, bsig1, bsig2, true_bsig, (uint64_t)1, dimension, max_nodes, 1, false, true);
        }

        TEST_METHOD(BatchChenIdentity) {
            uint64_t dimension = 2, max_nodes = 3, batch_size = 2;
            (void)prepare_branched_sig(dimension, max_nodes, false, false);

            std::vector<double> path1 = {
                0., 0., 1., 0.5, 0.4, 2.,
                0., 0., 0.25, 0.25, 0.5, 0.5 };
            std::vector<double> path2 = {
                0.4, 2., 6., 0.1, 2.3, 4.1,
                0.5, 0.5, 1., 1., 0.75, 0.75 };
            std::vector<double> path = {
                0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1,
                0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1., 0.75, 0.75 };

            uint64_t bs_len = branched_sig_length(dimension, max_nodes);

            std::vector<double> bsig1(bs_len * batch_size);
            (void)branched_sig_d(path1.data(), bsig1.data(), batch_size, dimension, 3, max_nodes, 1);

            std::vector<double> bsig2(bs_len * batch_size);
            (void)branched_sig_d(path2.data(), bsig2.data(), batch_size, dimension, 3, max_nodes, 1);

            std::vector<double> true_bsig(bs_len * batch_size);
            (void)branched_sig_d(path.data(), true_bsig.data(), batch_size, dimension, 5, max_nodes, 1);

            check_result_2(branched_sig_combine_d, bsig1, bsig2, true_bsig, batch_size, dimension, max_nodes, 1, false, true);
        }
    };

    TEST_CLASS(branchedSigCombineBackpropTest) {
    public:
        TEST_METHOD(FiniteDifference) {
            uint64_t dimension = 2, max_nodes = 3;
            (void)prepare_branched_sig(dimension, max_nodes, false, false);
            uint64_t bs_len = branched_sig_length(dimension, max_nodes);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };

            std::vector<double> bsig1(bs_len);
            (void)branched_sig_d(path1.data(), bsig1.data(), (uint64_t)1, dimension, 3, max_nodes, 1);
            std::vector<double> bsig2(bs_len);
            (void)branched_sig_d(path2.data(), bsig2.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

            std::vector<double> derivs(bs_len);
            for (uint64_t i = 0; i < bs_len; ++i)
                derivs[i] = 0.3 * (i + 1) - 1.;

            std::vector<double> d_bsig1(bs_len);
            std::vector<double> d_bsig2(bs_len);
            (void)branched_sig_combine_backprop_d(bsig1.data(), bsig2.data(), derivs.data(),
                d_bsig1.data(), d_bsig2.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);

            double eps = 1e-7;
            for (uint64_t i = 0; i < 5 && i < bs_len; ++i) {
                double orig = bsig1[i];
                bsig1[i] = orig + eps;
                std::vector<double> out_plus(bs_len);
                (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_plus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
                bsig1[i] = orig - eps;
                std::vector<double> out_minus(bs_len);
                (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_minus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
                bsig1[i] = orig;

                double numerical = 0.;
                for (uint64_t j = 0; j < bs_len; ++j)
                    numerical += derivs[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
                Assert::IsTrue(abs(numerical - d_bsig1[i]) < 1e-4);
            }

            for (uint64_t i = 0; i < 5 && i < bs_len; ++i) {
                double orig = bsig2[i];
                bsig2[i] = orig + eps;
                std::vector<double> out_plus(bs_len);
                (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_plus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
                bsig2[i] = orig - eps;
                std::vector<double> out_minus(bs_len);
                (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_minus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
                bsig2[i] = orig;

                double numerical = 0.;
                for (uint64_t j = 0; j < bs_len; ++j)
                    numerical += derivs[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
                Assert::IsTrue(abs(numerical - d_bsig2[i]) < 1e-4);
            }
        }

        TEST_METHOD(ZeroDerivative) {
            uint64_t dimension = 2, max_nodes = 3;
            (void)prepare_branched_sig(dimension, max_nodes, false, false);
            uint64_t bs_len = branched_sig_length(dimension, max_nodes);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
            std::vector<double> bsig1(bs_len);
            (void)branched_sig_d(path1.data(), bsig1.data(), (uint64_t)1, dimension, 3, max_nodes, 1);
            std::vector<double> bsig2(bs_len);
            (void)branched_sig_d(path2.data(), bsig2.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

            std::vector<double> derivs(bs_len, 0.);
            std::vector<double> out1(bs_len);
            std::vector<double> out2(bs_len);
            (void)branched_sig_combine_backprop_d(bsig1.data(), bsig2.data(), derivs.data(),
                out1.data(), out2.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);

            for (uint64_t i = 0; i < bs_len; ++i) {
                Assert::IsTrue(abs(out1[i]) < DOUBLE_EPSILON);
                Assert::IsTrue(abs(out2[i]) < DOUBLE_EPSILON);
            }
        }
    };
}