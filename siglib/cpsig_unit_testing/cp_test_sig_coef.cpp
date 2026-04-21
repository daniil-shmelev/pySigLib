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
    TEST_CLASS(sigCoefDoubleTest)
    {

        TEST_METHOD(Trivial) {
            auto f = sig_coef_d;

            std::vector<double> path = { 0., 0. };
            std::vector<double> true_ = { 0., 0. };
            std::vector<uint64_t> multi_indices = { 0, 1 };
            std::vector<uint64_t> degrees = { 1, 1 };

            check_result(f, path, true_, multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, 2, 1, false, false, 1., false, 1);
        }
        TEST_METHOD(Linear) {
            auto f = sig_coef_d;

            std::vector<double> path = { 0., 1. };
            std::vector<double> true_ = { 1., 1. / 2, 1. / 6, 1. / 24, 1. / 120 };
            std::vector<uint64_t> multi_indices(15, 0);
            std::vector<uint64_t> degrees = { 1, 2, 3, 4, 5 };

            check_result(f, path, true_, multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, 1, 2, false, false, 1., false, 1);
        }

        TEST_METHOD(ManualSigTest) {
            auto f = sig_coef_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., 0., 1., -1., 0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1,
                0, 0,
                0, 1,
                1, 0,
                1, 1
            };

            std::vector<uint64_t> degrees = { 1, 1, 2, 2, 2, 2 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, false, 1., false, 1);
        }

        TEST_METHOD(BatchManualSigTest) {
            auto f = sig_coef_d;
            uint64_t batch_size = 2, dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1.,
            0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., 0., 1., -1., 0.5,
            0., 1., 0., 1., -1., 0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1,
                0, 0,
                0, 1,
                1, 0,
                1, 1
            };

            std::vector<uint64_t> degrees = { 1, 1, 2, 2, 2, 2 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, false, false, 1., false, 1);
        }

        TEST_METHOD(ManualSigTestPrefixes) {
            auto f = sig_coef_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., -1., 1., 0.5, -0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1, 0,
                1, 1, 0
            };

            std::vector<uint64_t> degrees = { 1, 2, 3 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, false, 1., true, 1);
        }

        TEST_METHOD(BatchManualSigTestPrefixes) {
            auto f = sig_coef_d;
            uint64_t batch_size = 2, dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1.,
            0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., -1., 1., 0.5, -0.5,
            0., 1., -1., 1., 0.5, -0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1, 0,
                1, 1, 0
            };

            std::vector<uint64_t> degrees = { 1, 2, 3 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, false, false, 1., true, 1);
        }
    };

    TEST_CLASS(sigCoefBackpropTest) {
    public:

        TEST_METHOD(ManualTest1) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 2, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5 };
            std::vector<double> true_deriv = { -1./6, -1./6, 1./6, 1./6 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0
            };

            std::vector<double> coefs = { 1., 0.25, 1./12 };
            std::vector<double> derivs = { 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, false, 1., 1);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 3, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0. };
            std::vector<double> true_deriv = { -1. / 6, -10./6, 2./3, -1. - 1./3, -0.5, 3.};

            std::vector<uint64_t> multi_indices = {
                0, 1, 0
            };

            std::vector<double> coefs = { 4., -1., -2./3 };
            std::vector<double> derivs = { 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, false, 1., 1);
        }
        TEST_METHOD(ManualTest3) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 0.5 + 1./3, 1./3, 2./3, 20./3, -1.-1./6, -1.-2./3, -1./3, -5-1./3 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0
            };

            std::vector<double> coefs = {0., 1., -2.};
            std::vector<double> derivs = { 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, false, 1., 1);
        }

        TEST_METHOD(ManualTest4) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 19./24, 17./12, 2./3, 7.+1./3, -31./24, -25./12, -1./6, -20./3 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0,
                1, 1, 0
            };

            std::vector<double> coefs = { 0., 1., -2., 1., 0.5, -0.5 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, false, 1., 1);
        }

        TEST_METHOD(ManualTestBatch) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1., 0., 0., 2., 0., 1., 2., 0., 1. };
            std::vector<double> true_deriv = { 19. / 24, 17. / 12, 2. / 3, 7. + 1. / 3, -31. / 24, -25. / 12, -1. / 6, -20. / 3, 2.5, 3.5 + 1./3, -4., 2./3, -2.5, -3.5, 4., -1. };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0,
                1, 1, 0
            };

            std::vector<double> coefs = { 0., 1., -2., 1., 0.5, -0.5, 0., 2.5, -4-1./3, 1., 0.5, -1.5-1./3 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, false, false, 1., 1);

            std::vector<double> derivs2 = { 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1. };
            check_result(f, path, true_deriv, coefs.data(), derivs2.data(), multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, false, false, 1., -1);
        }

        TEST_METHOD(ManualTest4Empty) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 19./24, 17./12, 2./3, 7.+1./3, -31./24, -25./12, -1./6, -20./3 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0,
                1, 1, 0
            };

            std::vector<double> coefs = { 1., 0., 1., -2., 1., 0.5, -0.5 };
            std::vector<double> derivs = { 1., 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 0, 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, false, 1., 1);
        }

        TEST_METHOD(ManualTestTimeAug) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 55./36, 29./18, -1., -4./9, -71./36, -17./18, 13./9, -2./9 };

            std::vector<uint64_t> multi_indices = {
                0, 2, 0,
                1, 2, 0
            };

            std::vector<double> coefs = { 0., 1. + 2./3, -4. - 2./9, 1., 1./3, -4./9 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, true, false, 1., 1);
        }

        TEST_METHOD(ManualTestTimeAugBatch) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1.,
            0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 55. / 36, 29. / 18, -1., -4. / 9, -71. / 36, -17. / 18, 13. / 9, -2. / 9,
            55. / 36, 29. / 18, -1., -4. / 9, -71. / 36, -17. / 18, 13. / 9, -2. / 9 };

            std::vector<uint64_t> multi_indices = {
                0, 2, 0,
                1, 2, 0
            };

            std::vector<double> coefs = { 0., 1. + 2. / 3, -4. - 2. / 9, 1., 1. / 3, -4. / 9,
            0., 1. + 2. / 3, -4. - 2. / 9, 1., 1. / 3, -4. / 9 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1.,
            0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, true, false, 1., 1);
            
            std::vector<double> derivs2 = { 0., 0., 1., 0., 0., 1.,
            0., 0., 1., 0., 0., 1. };
            check_result(f, path, true_deriv, coefs.data(), derivs2.data(), multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, true, false, 1., -1);
        }

        TEST_METHOD(ManualTestLeadLag) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { -13., -.25, -5.375, 1.5, 47.375, -2.75, -29., 1.5 };

            std::vector<uint64_t> multi_indices = {
                0, 2, 0,
                1, 2, 3
            };

            std::vector<double> coefs = { 0., -13., 61., 1., 1.5, 1.125 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), (uint64_t)1, dimension, length, false, true, 1., 1);
        }
    };
}