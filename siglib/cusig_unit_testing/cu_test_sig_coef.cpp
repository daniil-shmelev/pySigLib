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

namespace MyTest {

    // =========================================================================
    // sig_coef CUDA tests
    // Ported from CPU sig_coef tests
    // =========================================================================

    TEST_CLASS(sigCoefCudaDoubleTest) {
    public:

        // Trivial test: linear path [0,0] -> [1,1], words (0,) and (1,)
        TEST_METHOD(TrivialTest) {
            uint64_t dimension = 2, length = 2;
            std::vector<double> path = { 0., 0., 1., 1. };
            std::vector<uint64_t> multi_idx = { 0, 1 };
            std::vector<uint64_t> degrees = { 1, 1 };
            std::vector<double> true_ = { 1., 1. };

            double* d_path = nullptr; double* d_out = nullptr;
            uint64_t* d_idx = nullptr; uint64_t* d_deg = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * true_.size());
            cudaMalloc(&d_idx, sizeof(uint64_t) * multi_idx.size());
            cudaMalloc(&d_deg, sizeof(uint64_t) * degrees.size());
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_deg, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

            int err = sig_coef_cuda_d(d_path, d_out, d_idx, (uint64_t)2, d_deg, (uint64_t)1, dimension, length, false);
            cudaDeviceSynchronize();

            std::vector<double> out(true_.size());
            cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost);
            cudaFree(d_path); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_deg);

            Assert::AreEqual(0, err, L"sig_coef_cuda_d returned non-zero error code");
            for (uint64_t i = 0; i < true_.size(); ++i) {
                Assert::IsTrue(std::abs(true_[i] - out[i]) < DOUBLE_EPSILON,
                    (L"Mismatch at index " + std::to_wstring(i)).c_str());
            }
        }

        // Trivial test: single-point path should return zeros
        TEST_METHOD(SinglePointPath) {
            uint64_t dimension = 2, length = 1;
            std::vector<double> path = { 5., 3. };
            std::vector<uint64_t> multi_idx = { 0, 1 };
            std::vector<uint64_t> degrees = { 1, 1 };
            std::vector<double> true_ = { 0., 0. };

            double* d_path = nullptr; double* d_out = nullptr;
            uint64_t* d_idx = nullptr; uint64_t* d_deg = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * true_.size());
            cudaMalloc(&d_idx, sizeof(uint64_t) * multi_idx.size());
            cudaMalloc(&d_deg, sizeof(uint64_t) * degrees.size());
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_deg, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

            int err = sig_coef_cuda_d(d_path, d_out, d_idx, (uint64_t)2, d_deg, (uint64_t)1, dimension, length, false);
            cudaDeviceSynchronize();

            std::vector<double> out(true_.size());
            cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost);
            cudaFree(d_path); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_deg);

            Assert::AreEqual(0, err, L"sig_coef_cuda_d returned non-zero error code");
            for (uint64_t i = 0; i < true_.size(); ++i) {
                Assert::IsTrue(std::abs(true_[i] - out[i]) < DOUBLE_EPSILON,
                    (L"Mismatch at index " + std::to_wstring(i)).c_str());
            }
        }

        // Verify sig_coef_cuda matches full signature for all words
        TEST_METHOD(FullSigCompare) {
            uint64_t dimension = 2, length = 5, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };
            uint64_t sig_len = sig_length_(dimension, degree);

            // Compute full signature on GPU for ground truth
            std::vector<double> true_sig = compute_sig_on_gpu(path, dimension, length, degree);

            // Build all words of degree 1..3 for dim=2: (0),(1),(0,0),(0,1),(1,0),(1,1),(0,0,0),...
            std::vector<uint64_t> multi_idx;
            std::vector<uint64_t> degrees;
            // Degree 1: 2 words
            for (uint64_t a = 0; a < dimension; ++a) {
                multi_idx.push_back(a);
                degrees.push_back(1);
            }
            // Degree 2: 4 words
            for (uint64_t a = 0; a < dimension; ++a)
                for (uint64_t b = 0; b < dimension; ++b) {
                    multi_idx.push_back(a); multi_idx.push_back(b);
                    degrees.push_back(2);
                }
            // Degree 3: 8 words
            for (uint64_t a = 0; a < dimension; ++a)
                for (uint64_t b = 0; b < dimension; ++b)
                    for (uint64_t c = 0; c < dimension; ++c) {
                        multi_idx.push_back(a); multi_idx.push_back(b); multi_idx.push_back(c);
                        degrees.push_back(3);
                    }

            uint64_t num_words = degrees.size(); // 2 + 4 + 8 = 14 = sig_len - 1
            std::vector<double> out(num_words);

            double* d_path = nullptr; double* d_out = nullptr;
            uint64_t* d_idx = nullptr; uint64_t* d_deg = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * num_words);
            cudaMalloc(&d_idx, sizeof(uint64_t) * multi_idx.size());
            cudaMalloc(&d_deg, sizeof(uint64_t) * degrees.size());
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_deg, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

            int err = sig_coef_cuda_d(d_path, d_out, d_idx, num_words, d_deg, (uint64_t)1, dimension, length, false);
            cudaDeviceSynchronize();

            cudaMemcpy(out.data(), d_out, sizeof(double) * num_words, cudaMemcpyDeviceToHost);
            cudaFree(d_path); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_deg);

            Assert::AreEqual(0, err, L"sig_coef_cuda_d returned non-zero error code");

            // true_sig[0] = 1 (degree 0), true_sig[1..] = degree 1,2,3 coeffs
            for (uint64_t i = 0; i < num_words; ++i) {
                std::wstring msg = L"Mismatch at word " + std::to_wstring(i) +
                    L": expected " + std::to_wstring(true_sig[i + 1]) +
                    L" got " + std::to_wstring(out[i]);
                Assert::IsTrue(std::abs(true_sig[i + 1] - out[i]) < DOUBLE_EPSILON, msg.c_str());
            }
        }

        // Batch test: verify sig_coef_cuda matches batch signature
        TEST_METHOD(BatchFullSigCompare) {
            uint64_t batch_size = 3, dimension = 2, length = 4, degree = 2;
            std::vector<double> path = {
                0., 0., 1., 0.5, 0.4, 2., 6., 0.1,
                0., 0., 0.5, 1., 3., 2., 1., 0.5,
                1., 2., 3., 4., 5., 6., 7., 8.
            };

            std::vector<double> true_sig = compute_batch_sig_on_gpu(path, batch_size, dimension, length, degree);

            // Build all words of degree 1..2 for dim=2
            std::vector<uint64_t> multi_idx;
            std::vector<uint64_t> degrees;
            for (uint64_t a = 0; a < dimension; ++a) { multi_idx.push_back(a); degrees.push_back(1); }
            for (uint64_t a = 0; a < dimension; ++a)
                for (uint64_t b = 0; b < dimension; ++b) {
                    multi_idx.push_back(a); multi_idx.push_back(b);
                    degrees.push_back(2);
                }

            uint64_t num_words = degrees.size(); // 2 + 4 = 6
            uint64_t total_out = batch_size * num_words;
            std::vector<double> out(total_out);

            double* d_path = nullptr; double* d_out = nullptr;
            uint64_t* d_idx = nullptr; uint64_t* d_deg = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * total_out);
            cudaMalloc(&d_idx, sizeof(uint64_t) * multi_idx.size());
            cudaMalloc(&d_deg, sizeof(uint64_t) * degrees.size());
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_deg, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

            int err = sig_coef_cuda_d(d_path, d_out, d_idx, num_words, d_deg, batch_size, dimension, length, false);
            cudaDeviceSynchronize();

            cudaMemcpy(out.data(), d_out, sizeof(double) * total_out, cudaMemcpyDeviceToHost);
            cudaFree(d_path); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_deg);

            Assert::AreEqual(0, err, L"sig_coef_cuda_d returned non-zero error code");

            uint64_t sig_len = sig_length_(dimension, degree);
            for (uint64_t b = 0; b < batch_size; ++b) {
                for (uint64_t i = 0; i < num_words; ++i) {
                    double expected = true_sig[b * sig_len + i + 1];
                    double actual = out[b * num_words + i];
                    std::wstring msg = L"Batch " + std::to_wstring(b) + L" word " + std::to_wstring(i) +
                        L": expected " + std::to_wstring(expected) +
                        L" got " + std::to_wstring(actual);
                    Assert::IsTrue(std::abs(expected - actual) < DOUBLE_EPSILON, msg.c_str());
                }
            }
        }

        // Prefixes test
        TEST_METHOD(PrefixesTest) {
            uint64_t dimension = 2, length = 5;
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

            // Word (0,1) with prefixes returns coefs for (0) and (0,1)
            // Word (1) with prefixes returns coef for (1)
            std::vector<uint64_t> multi_idx = { 0, 1, 1 };
            std::vector<uint64_t> degrees = { 2, 1 };
            uint64_t num_words = 2;
            uint64_t out_size = 3; // prefix sizes: 2 + 1

            // Get ground truth from full signature
            uint64_t degree = 2;
            std::vector<double> sig = compute_sig_on_gpu(path, dimension, length, degree);
            // sig layout for dim=2, deg=2: [1, s(0), s(1), s(0,0), s(0,1), s(1,0), s(1,1)]
            // Word (0,1) prefixes: s(0)=sig[1], s(0,1)=sig[4]
            // Word (1) prefixes: s(1)=sig[2]
            std::vector<double> true_ = { sig[1], sig[4], sig[2] };

            double* d_path = nullptr; double* d_out = nullptr;
            uint64_t* d_idx = nullptr; uint64_t* d_deg = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * out_size);
            cudaMalloc(&d_idx, sizeof(uint64_t) * multi_idx.size());
            cudaMalloc(&d_deg, sizeof(uint64_t) * degrees.size());
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_deg, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

            int err = sig_coef_cuda_d(d_path, d_out, d_idx, num_words, d_deg, (uint64_t)1, dimension, length, true);
            cudaDeviceSynchronize();

            std::vector<double> out(out_size);
            cudaMemcpy(out.data(), d_out, sizeof(double) * out_size, cudaMemcpyDeviceToHost);
            cudaFree(d_path); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_deg);

            Assert::AreEqual(0, err, L"sig_coef_cuda_d returned non-zero error code");
            for (uint64_t i = 0; i < out_size; ++i) {
                std::wstring msg = L"Mismatch at index " + std::to_wstring(i) +
                    L": expected " + std::to_wstring(true_[i]) +
                    L" got " + std::to_wstring(out[i]);
                Assert::IsTrue(std::abs(true_[i] - out[i]) < DOUBLE_EPSILON, msg.c_str());
            }
        }

        // Degree 0 word test: should return 1
        TEST_METHOD(DegreeZeroWord) {
            uint64_t dimension = 2, length = 3;
            std::vector<double> path = { 0., 0., 1., 2., 3., 4. };
            std::vector<uint64_t> multi_idx = {}; // empty word
            std::vector<uint64_t> degrees = { 0 };
            std::vector<double> true_ = { 1. };

            double* d_path = nullptr; double* d_out = nullptr;
            uint64_t* d_idx = nullptr; uint64_t* d_deg = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * 1);
            cudaMalloc(&d_idx, sizeof(uint64_t) * 1); // at least 1 byte
            cudaMalloc(&d_deg, sizeof(uint64_t) * 1);
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_deg, degrees.data(), sizeof(uint64_t) * 1, cudaMemcpyHostToDevice);

            int err = sig_coef_cuda_d(d_path, d_out, d_idx, (uint64_t)1, d_deg, (uint64_t)1, dimension, length, false);
            cudaDeviceSynchronize();

            std::vector<double> out(1);
            cudaMemcpy(out.data(), d_out, sizeof(double) * 1, cudaMemcpyDeviceToHost);
            cudaFree(d_path); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_deg);

            Assert::AreEqual(0, err, L"sig_coef_cuda_d returned non-zero error code");
            Assert::IsTrue(std::abs(1. - out[0]) < DOUBLE_EPSILON, L"Degree 0 word should return 1");
        }

        // Mixed degrees test
        TEST_METHOD(MixedDegrees) {
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1 };

            std::vector<double> sig = compute_sig_on_gpu(path, dimension, length, degree);
            // sig layout dim=2, deg=3: [1, s(0), s(1), s(0,0), s(0,1), s(1,0), s(1,1),
            //                           s(0,0,0), s(0,0,1), s(0,1,0), s(0,1,1), s(1,0,0), s(1,0,1), s(1,1,0), s(1,1,1)]

            // Words with mixed degrees: (0) deg1, (1,0) deg2, (0,1,1) deg3
            std::vector<uint64_t> multi_idx = { 0, 1, 0, 0, 1, 1 };
            std::vector<uint64_t> degrees = { 1, 2, 3 };
            // s(0) = sig[1], s(1,0) = sig[5], s(0,1,1) = sig[10]
            std::vector<double> true_ = { sig[1], sig[5], sig[10] };

            double* d_path = nullptr; double* d_out = nullptr;
            uint64_t* d_idx = nullptr; uint64_t* d_deg = nullptr;
            cudaMalloc(&d_path, sizeof(double) * path.size());
            cudaMalloc(&d_out, sizeof(double) * 3);
            cudaMalloc(&d_idx, sizeof(uint64_t) * multi_idx.size());
            cudaMalloc(&d_deg, sizeof(uint64_t) * degrees.size());
            cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_deg, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

            int err = sig_coef_cuda_d(d_path, d_out, d_idx, (uint64_t)3, d_deg, (uint64_t)1, dimension, length, false);
            cudaDeviceSynchronize();

            std::vector<double> out(3);
            cudaMemcpy(out.data(), d_out, sizeof(double) * 3, cudaMemcpyDeviceToHost);
            cudaFree(d_path); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_deg);

            Assert::AreEqual(0, err, L"sig_coef_cuda_d returned non-zero error code");
            for (uint64_t i = 0; i < 3; ++i) {
                std::wstring msg = L"Mismatch at index " + std::to_wstring(i) +
                    L": expected " + std::to_wstring(true_[i]) +
                    L" got " + std::to_wstring(out[i]);
                Assert::IsTrue(std::abs(true_[i] - out[i]) < DOUBLE_EPSILON, msg.c_str());
            }
        }
    };

}