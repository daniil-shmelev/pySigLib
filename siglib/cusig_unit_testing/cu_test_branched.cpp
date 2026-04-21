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
#include "branched_trees.h"

namespace MyTest {

    TEST_CLASS(branchedSigCombineCudaTest) {
    public:
        TEST_METHOD(ChenIdentity) {
            uint64_t dimension = 2, max_nodes = 3;
            uint64_t bs_len = compute_branched_sig_length(dimension, max_nodes);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

            double* d_path = nullptr;
            double* d_bsig = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_bsig, sizeof(double) * bs_len);

            cudaMemcpy(d_path, path1.data(), sizeof(double) * path1.size(), cudaMemcpyHostToDevice);
            int err = branched_sig_cuda_d(d_path, d_bsig, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
            cudaDeviceSynchronize();
            std::vector<double> bsig1(bs_len);
            cudaMemcpy(bsig1.data(), d_bsig, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
            Assert::AreEqual(0, err, L"branched_sig_cuda_d failed for path1");

            cudaMemcpy(d_path, path2.data(), sizeof(double) * path2.size(), cudaMemcpyHostToDevice);
            err = branched_sig_cuda_d(d_path, d_bsig, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
            cudaDeviceSynchronize();
            std::vector<double> bsig2(bs_len);
            cudaMemcpy(bsig2.data(), d_bsig, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
            Assert::AreEqual(0, err, L"branched_sig_cuda_d failed for path2");

            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            err = branched_sig_cuda_d(d_path, d_bsig, (uint64_t)1, dimension, (uint64_t)5, max_nodes);
            cudaDeviceSynchronize();
            std::vector<double> true_bsig(bs_len);
            cudaMemcpy(true_bsig.data(), d_bsig, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
            Assert::AreEqual(0, err, L"branched_sig_cuda_d failed for full path");

            cudaFree(d_path);
            cudaFree(d_bsig);

            check_result_2_typed(branched_sig_combine_cuda_d, bsig1, bsig2, true_bsig,
                (uint64_t)1, dimension, (uint64_t)max_nodes, false, true);
        }
    };

    TEST_CLASS(branchedSigCombineBackpropCudaTest) {
    public:
        TEST_METHOD(FiniteDifference) {
            uint64_t dimension = 2, max_nodes = 3;
            uint64_t bs_len = compute_branched_sig_length(dimension, max_nodes);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };

            double* d_path = nullptr;
            double* d_bsig_tmp = nullptr;
            cudaMalloc(&d_path, sizeof(double) * 6);
            cudaMalloc(&d_bsig_tmp, sizeof(double) * bs_len);

            cudaMemcpy(d_path, path1.data(), sizeof(double) * 6, cudaMemcpyHostToDevice);
            int err = branched_sig_cuda_d(d_path, d_bsig_tmp, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
            cudaDeviceSynchronize();
            std::vector<double> bsig1(bs_len);
            cudaMemcpy(bsig1.data(), d_bsig_tmp, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
            Assert::AreEqual(0, err, L"branched_sig_cuda_d failed for path1");

            cudaMemcpy(d_path, path2.data(), sizeof(double) * 6, cudaMemcpyHostToDevice);
            err = branched_sig_cuda_d(d_path, d_bsig_tmp, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
            cudaDeviceSynchronize();
            std::vector<double> bsig2(bs_len);
            cudaMemcpy(bsig2.data(), d_bsig_tmp, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
            Assert::AreEqual(0, err, L"branched_sig_cuda_d failed for path2");

            cudaFree(d_path);
            cudaFree(d_bsig_tmp);

            std::vector<double> derivs(bs_len);
            for (uint64_t i = 0; i < bs_len; ++i)
                derivs[i] = 0.3 * (i + 1) - 1.;

            double* d_bsig1 = nullptr;
            double* d_bsig2 = nullptr;
            double* d_derivs = nullptr;
            double* d_out1 = nullptr;
            double* d_out2 = nullptr;
            cudaMalloc(&d_bsig1, sizeof(double) * bs_len);
            cudaMalloc(&d_bsig2, sizeof(double) * bs_len);
            cudaMalloc(&d_derivs, sizeof(double) * bs_len);
            cudaMalloc(&d_out1, sizeof(double) * bs_len);
            cudaMalloc(&d_out2, sizeof(double) * bs_len);

            cudaMemcpy(d_bsig1, bsig1.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
            cudaMemcpy(d_bsig2, bsig2.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
            cudaMemcpy(d_derivs, derivs.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);

            err = branched_sig_combine_backprop_cuda_d(d_bsig1, d_bsig2, d_derivs,
                d_out1, d_out2, (uint64_t)1, dimension, max_nodes, false, true);
            cudaDeviceSynchronize();

            std::vector<double> grad1(bs_len), grad2(bs_len);
            cudaMemcpy(grad1.data(), d_out1, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad2.data(), d_out2, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

            cudaFree(d_bsig1);
            cudaFree(d_bsig2);
            cudaFree(d_derivs);
            cudaFree(d_out1);
            cudaFree(d_out2);

            Assert::AreEqual(0, err, L"branched_sig_combine_backprop_cuda_d failed");

            double* d_a = nullptr;
            double* d_b = nullptr;
            double* d_fwd = nullptr;
            cudaMalloc(&d_a, sizeof(double) * bs_len);
            cudaMalloc(&d_b, sizeof(double) * bs_len);
            cudaMalloc(&d_fwd, sizeof(double) * bs_len);

            double eps = 1e-7;

            cudaMemcpy(d_b, bsig2.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
            for (uint64_t i = 0; i < 5 && i < bs_len; ++i) {
                double orig = bsig1[i];

                bsig1[i] = orig + eps;
                cudaMemcpy(d_a, bsig1.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
                (void)branched_sig_combine_cuda_d(d_a, d_b, d_fwd,
                    (uint64_t)1, dimension, max_nodes, false, true);
                cudaDeviceSynchronize();
                std::vector<double> out_plus(bs_len);
                cudaMemcpy(out_plus.data(), d_fwd, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

                bsig1[i] = orig - eps;
                cudaMemcpy(d_a, bsig1.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
                (void)branched_sig_combine_cuda_d(d_a, d_b, d_fwd,
                    (uint64_t)1, dimension, max_nodes, false, true);
                cudaDeviceSynchronize();
                std::vector<double> out_minus(bs_len);
                cudaMemcpy(out_minus.data(), d_fwd, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

                bsig1[i] = orig;

                double numerical = 0.;
                for (uint64_t j = 0; j < bs_len; ++j)
                    numerical += derivs[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
                Assert::IsTrue(std::abs(numerical - grad1[i]) < 1e-4);
            }

            cudaMemcpy(d_a, bsig1.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
            for (uint64_t i = 0; i < 5 && i < bs_len; ++i) {
                double orig = bsig2[i];

                bsig2[i] = orig + eps;
                cudaMemcpy(d_b, bsig2.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
                (void)branched_sig_combine_cuda_d(d_a, d_b, d_fwd,
                    (uint64_t)1, dimension, max_nodes, false, true);
                cudaDeviceSynchronize();
                std::vector<double> out_plus(bs_len);
                cudaMemcpy(out_plus.data(), d_fwd, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

                bsig2[i] = orig - eps;
                cudaMemcpy(d_b, bsig2.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
                (void)branched_sig_combine_cuda_d(d_a, d_b, d_fwd,
                    (uint64_t)1, dimension, max_nodes, false, true);
                cudaDeviceSynchronize();
                std::vector<double> out_minus(bs_len);
                cudaMemcpy(out_minus.data(), d_fwd, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

                bsig2[i] = orig;

                double numerical = 0.;
                for (uint64_t j = 0; j < bs_len; ++j)
                    numerical += derivs[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
                Assert::IsTrue(std::abs(numerical - grad2[i]) < 1e-4);
            }

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_fwd);
        }

        TEST_METHOD(ZeroDerivative) {
            uint64_t dimension = 2, max_nodes = 3;
            uint64_t bs_len = compute_branched_sig_length(dimension, max_nodes);

            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };

            double* d_path = nullptr;
            double* d_bsig_tmp = nullptr;
            cudaMalloc(&d_path, sizeof(double) * 6);
            cudaMalloc(&d_bsig_tmp, sizeof(double) * bs_len);

            cudaMemcpy(d_path, path1.data(), sizeof(double) * 6, cudaMemcpyHostToDevice);
            (void)branched_sig_cuda_d(d_path, d_bsig_tmp, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
            cudaDeviceSynchronize();
            std::vector<double> bsig1(bs_len);
            cudaMemcpy(bsig1.data(), d_bsig_tmp, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

            cudaMemcpy(d_path, path2.data(), sizeof(double) * 6, cudaMemcpyHostToDevice);
            (void)branched_sig_cuda_d(d_path, d_bsig_tmp, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
            cudaDeviceSynchronize();
            std::vector<double> bsig2(bs_len);
            cudaMemcpy(bsig2.data(), d_bsig_tmp, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

            cudaFree(d_path);
            cudaFree(d_bsig_tmp);

            std::vector<double> derivs(bs_len, 0.);

            double* d_bsig1 = nullptr;
            double* d_bsig2 = nullptr;
            double* d_derivs = nullptr;
            double* d_out1 = nullptr;
            double* d_out2 = nullptr;
            cudaMalloc(&d_bsig1, sizeof(double) * bs_len);
            cudaMalloc(&d_bsig2, sizeof(double) * bs_len);
            cudaMalloc(&d_derivs, sizeof(double) * bs_len);
            cudaMalloc(&d_out1, sizeof(double) * bs_len);
            cudaMalloc(&d_out2, sizeof(double) * bs_len);

            cudaMemcpy(d_bsig1, bsig1.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
            cudaMemcpy(d_bsig2, bsig2.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);
            cudaMemcpy(d_derivs, derivs.data(), sizeof(double) * bs_len, cudaMemcpyHostToDevice);

            int err = branched_sig_combine_backprop_cuda_d(d_bsig1, d_bsig2, d_derivs,
                d_out1, d_out2, (uint64_t)1, dimension, max_nodes, false, true);
            cudaDeviceSynchronize();

            std::vector<double> out1(bs_len), out2(bs_len);
            cudaMemcpy(out1.data(), d_out1, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
            cudaMemcpy(out2.data(), d_out2, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);

            cudaFree(d_bsig1);
            cudaFree(d_bsig2);
            cudaFree(d_derivs);
            cudaFree(d_out1);
            cudaFree(d_out2);

            Assert::AreEqual(0, err, L"branched_sig_combine_backprop_cuda_d failed");

            for (uint64_t i = 0; i < bs_len; ++i) {
                Assert::IsTrue(std::abs(out1[i]) < DOUBLE_EPSILON);
                Assert::IsTrue(std::abs(out2[i]) < DOUBLE_EPSILON);
            }
        }
    };
}