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

    TEST_CLASS(sigKernelTest) {
    public:

        TEST_METHOD(Trivial) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length = 1;
            std::vector<double> path = { 0. };
            std::vector<double> true_sig = { 1. };
            std::vector<double> gram = {};
            check_result(f, gram, true_sig, (uint64_t)1, dimension, length, length, 0, 0, false, 1);
        }

        TEST_METHOD(TrivialBatch) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length = 1, batch_size = 5;
            std::vector<double> path = { 0. };
            std::vector<double> true_sig = { 1., 1., 1., 1., 1. };
            std::vector<double> gram = {};
            check_result(f, gram, true_sig, batch_size, dimension, length, length, 0, 0, 1, false);
        }
        TEST_METHOD(LinearPathTest) {
            auto f = sig_kernel_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> true_sig = { 4.256702149748847 };
            std::vector<double> gram((length - 1) * (length - 1));
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, (uint64_t)1, dimension, length, length, 2, 2, false, 1);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_kernel_d;
            uint64_t dimension = 3, length = 4;
            std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
            std::vector<double> true_sig = { 2.1529809076880486 };
            std::vector<double> gram((length - 1) * (length - 1));
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, (uint64_t)1, dimension, length, length, 2, 2, false, 1);
        }

        TEST_METHOD(NonSquare1) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length1 = 3, length2 = 2;
            std::vector<double> path1 = { 0., 1., 2. };
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> true_sig = { 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(NonSquare2) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path2 = { 0., 1., 2. };
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> true_sig = { 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(FullGrid) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length1 = 3, length2 = 2;
            std::vector<double> path1 = { 0., 1., 2. };
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> true_sig = { 1., 1.,
                1., 4.,
                1., 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, (uint64_t)1, dimension, length1, length2, 0, 0, true, 1);
        }
    };

    TEST_CLASS(sigKernelBackpropTest) {
    public:
        TEST_METHOD(ManualTest1) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 4.5 + 1./6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(ManualTest1Extended) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 34, length2 = 35;
            std::vector<double> path1(length1, 0.);
            path1[length1 - 1] = 2.;
            std::vector<double> path2(length2, 0.);
            path2[length2 - 2] = 1.;
            path2[length2 - 1] = 2.;
            double deriv = 1.;
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
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(ManualTest1Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length2 = 2, length1 = 3;
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> path1 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 4.5 + 1. / 6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 4., 1., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 2., 3. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 761./72, 7.125, 133./24, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6 };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(ManualTest2Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length2 = 3, length1 = 3;
            std::vector<double> path2 = { 0., 2., 3. };
            std::vector<double> path1 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 761. / 72, 133. / 24, 7.125, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 7., 1., 11., 25. - 1. / 6 };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(ManualTest3) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
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
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 1, 1, false, 1);
        }

        TEST_METHOD(ManualTest3Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length2 = 2, length1 = 3;
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> path1 = { 0., 1., 2. };
            double deriv = 1.;
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
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 1, 1, false, 1);
        }

        TEST_METHOD(ManualTest4) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 2, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 1., 2., 4., 5., 5. };
            std::vector<double> path2 = { 0., 2., 1., 3., 2., 1. };
            double deriv = 1.;
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
                87.729 + 1./6000
            };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(ManualTest4Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 2, length2 = 3, length1 = 3;
            std::vector<double> path2 = { 0., 1., 2., 4., 5., 5. };
            std::vector<double> path1 = { 0., 2., 1., 3., 2., 1. };
            double deriv = 1.;
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
            check_result(f, gram, true_, &deriv, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(BatchManualTest1) {
            auto f = sig_kernel_backprop_d;
            uint64_t batch_size = 2, dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2., 0., 2. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> derivs = { 1., 1. };
            std::vector<double> true_ = { 4.5 + 1. / 6, 4.5, 4.5 + 1. / 6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 2, 1, dimension, length1, length2);
            check_result(f, gram, true_, derivs.data(), k_grid.data(), batch_size, dimension, length1, length2, 0, 0, false, 1);
        }

        TEST_METHOD(BatchManualTest2) {
            auto f = sig_kernel_backprop_d;
            uint64_t batch_size = 2, dimension = 1, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 2., 3., 0., 2., 3. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> derivs = { 1., 1. };
            std::vector<double> true_ = { 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6, 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6, 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6 };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 4, 1, dimension, length1, length2);
            check_result(f, gram, true_, derivs.data(), k_grid.data(), batch_size, dimension, length1, length2, 0, 0, false, 1);
        }
    };

    TEST_CLASS(sigKernelBackpropGridTest) {
    public:
        // When derivs_grid has 1.0 only at position [-1,-1] and 0.0 elsewhere,
        // the grid backprop should produce the same result as scalar backprop with deriv=1.0.
        TEST_METHOD(ConsistencyWithScalar) {
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);

            // Scalar backprop with deriv=1.0
            double one = 1.0;
            std::vector<double> out_scalar((length1 - 1) * (length2 - 1), 0.);
            (void)sig_kernel_backprop_d(gram.data(), out_scalar.data(), &one, k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, false, 1);

            // Grid backprop with derivs_grid = 0 everywhere except [length1-1, length2-1] = 1.0
            uint64_t grid_length = length1 * length2;
            std::vector<double> derivs_grid(grid_length, 0.);
            derivs_grid[grid_length - 1] = 1.0; // last element
            std::vector<double> out_grid((length1 - 1) * (length2 - 1), 0.);
            (void)sig_kernel_backprop_d(gram.data(), out_grid.data(), derivs_grid.data(), k_grid.data(), (uint64_t)1, dimension, length1, length2, 0, 0, true, 1);

            for (uint64_t i = 0; i < out_scalar.size(); ++i)
                Assert::IsTrue(abs(out_scalar[i] - out_grid[i]) < DOUBLE_EPSILON);
        }
        TEST_METHOD(BatchConsistencyWithScalar) {
            uint64_t batch_size = 2, dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2., 0., 2. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 2, 1, dimension, length1, length2);

            // Scalar batch backprop with derivs = {1, 1}
            std::vector<double> derivs_scalar = { 1., 1. };
            uint64_t out_size = (length1 - 1) * (length2 - 1) * batch_size;
            std::vector<double> out_scalar(out_size, 0.);
            (void)sig_kernel_backprop_d(gram.data(), out_scalar.data(), derivs_scalar.data(), k_grid.data(), batch_size, dimension, length1, length2, 0, 0, false, 1);

            // Grid batch backprop with derivs_grid = 0 everywhere except [-1,-1] = 1.0
            uint64_t grid_length = length1 * length2;
            std::vector<double> derivs_grid(grid_length * batch_size, 0.);
            derivs_grid[grid_length - 1] = 1.0;
            derivs_grid[2 * grid_length - 1] = 1.0;
            std::vector<double> out_grid(out_size, 0.);
            (void)sig_kernel_backprop_d(gram.data(), out_grid.data(), derivs_grid.data(), k_grid.data(), batch_size, dimension, length1, length2, 0, 0, true, 1);

            for (uint64_t i = 0; i < out_size; ++i)
                Assert::IsTrue(abs(out_scalar[i] - out_grid[i]) < DOUBLE_EPSILON);
        }
        TEST_METHOD(ManualTest) {
            uint64_t dimension = 2, length = 4;
            std::vector<double> path = { 0., 0., 1., .5, 4., 0., 0., 1. };
            std::vector<double> gram((length - 1) * (length - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 1., 2.640625, 10.571045, 3.154658, 1., 10.571045, 285.859342, 2372.95239, 1., 3.154658, 2372.95239, 165981.889 };
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            uint64_t out_size = (length - 1) * (length - 1);

            std::vector<double> true_ = { 8.0338748831219071, 3.0207107002152322, -0.041744818181351222, 3.0207107002152322, 2.6526166180712516, -1.6587152651909718, -0.041744818181351222, -1.6587152651909718, 1.6629617402333334 };
            uint64_t grid_length = length * length;
            std::vector<double> derivs_grid(grid_length, 0.0001);
            std::vector<double> out_grid(out_size, 0.);
            (void)sig_kernel_backprop_d(gram.data(), out_grid.data(), derivs_grid.data(), k_grid.data(), (uint64_t)1, dimension, length, length, 0, 0, true, 1);

            for (uint64_t i = 0; i < out_size; ++i)
                Assert::IsTrue(abs(true_[i] - out_grid[i]) < DOUBLE_EPSILON);
        }
    };
}