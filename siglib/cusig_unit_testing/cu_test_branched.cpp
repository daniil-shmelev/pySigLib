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
#include "trees/basis_counts.h"

TEST(branchedSigCombineCudaTest, ChenIdentity) {
    uint64_t dimension = 2, max_nodes = 3;
    uint64_t bs_len = compute_branched_sig_length(dimension, max_nodes);
    ASSERT_EQ(0, prepare_branched_sig_cuda(dimension, max_nodes));

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
    EXPECT_EQ(0, err) << "branched_sig_cuda_d failed for path1";

    cudaMemcpy(d_path, path2.data(), sizeof(double) * path2.size(), cudaMemcpyHostToDevice);
    err = branched_sig_cuda_d(d_path, d_bsig, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
    cudaDeviceSynchronize();
    std::vector<double> bsig2(bs_len);
    cudaMemcpy(bsig2.data(), d_bsig, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "branched_sig_cuda_d failed for path2";

    cudaMemcpy(d_path, path.data(), sizeof(double) * path.size(), cudaMemcpyHostToDevice);
    err = branched_sig_cuda_d(d_path, d_bsig, (uint64_t)1, dimension, (uint64_t)5, max_nodes);
    cudaDeviceSynchronize();
    std::vector<double> true_bsig(bs_len);
    cudaMemcpy(true_bsig.data(), d_bsig, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "branched_sig_cuda_d failed for full path";

    cudaFree(d_path);
    cudaFree(d_bsig);

    check_result_2_typed(branched_sig_combine_cuda_d, bsig1, bsig2, true_bsig,
        (uint64_t)1, dimension, (uint64_t)max_nodes, false, true);
}

TEST(branchedSigCombineBackpropCudaTest, FiniteDifference) {
    uint64_t dimension = 2, max_nodes = 3;
    uint64_t bs_len = compute_branched_sig_length(dimension, max_nodes);
    ASSERT_EQ(0, prepare_branched_sig_cuda(dimension, max_nodes));

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
    EXPECT_EQ(0, err) << "branched_sig_cuda_d failed for path1";

    cudaMemcpy(d_path, path2.data(), sizeof(double) * 6, cudaMemcpyHostToDevice);
    err = branched_sig_cuda_d(d_path, d_bsig_tmp, (uint64_t)1, dimension, (uint64_t)3, max_nodes);
    cudaDeviceSynchronize();
    std::vector<double> bsig2(bs_len);
    cudaMemcpy(bsig2.data(), d_bsig_tmp, sizeof(double) * bs_len, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, err) << "branched_sig_cuda_d failed for path2";

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

    EXPECT_EQ(0, err) << "branched_sig_combine_backprop_cuda_d failed";

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
        EXPECT_TRUE(std::abs(numerical - grad1[i]) < 1e-4);
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
        EXPECT_TRUE(std::abs(numerical - grad2[i]) < 1e-4);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_fwd);
}

TEST(branchedSigCombineBackpropCudaTest, ZeroDerivative) {
    uint64_t dimension = 2, max_nodes = 3;
    uint64_t bs_len = compute_branched_sig_length(dimension, max_nodes);
    ASSERT_EQ(0, prepare_branched_sig_cuda(dimension, max_nodes));

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

    EXPECT_EQ(0, err) << "branched_sig_combine_backprop_cuda_d failed";

    for (uint64_t i = 0; i < bs_len; ++i) {
        EXPECT_TRUE(std::abs(out1[i]) < DOUBLE_EPSILON);
        EXPECT_TRUE(std::abs(out2[i]) < DOUBLE_EPSILON);
    }
}

TEST(branchedSigCoefCudaTest, ForwardAndBackpropMatchFull) {
    const uint64_t batch_size = 2;
    const uint64_t dimension = 2;
    const uint64_t length = 4;
    const uint64_t max_nodes = 3;
    const uint64_t bs_len = compute_branched_sig_length(dimension, max_nodes);
    const std::vector<uint64_t> indices = {1, 3, bs_len - 1, 3};
    const uint64_t num_indices = indices.size();
    const std::vector<uint64_t> tree_data{
        4, 1, 0, 0, 1, 0, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0
    };

    std::vector<double> path(batch_size * length * dimension);
    for (uint64_t i = 0; i < path.size(); ++i)
        path[i] = 0.07 * static_cast<double>(i) - 0.2;
    std::vector<double> derivs(batch_size * num_indices);
    for (uint64_t i = 0; i < derivs.size(); ++i)
        derivs[i] = 0.13 * static_cast<double>(i + 1) - 0.4;

    double* d_path = nullptr;
    double* d_full = nullptr;
    double* d_coefs = nullptr;
    double* d_derivs = nullptr;
    double* d_full_derivs = nullptr;
    double* d_sparse_grad = nullptr;
    double* d_full_grad = nullptr;
    cudaMalloc(&d_path, path.size() * sizeof(double));
    cudaMalloc(&d_full, batch_size * bs_len * sizeof(double));
    cudaMalloc(&d_coefs, derivs.size() * sizeof(double));
    cudaMalloc(&d_derivs, derivs.size() * sizeof(double));
    cudaMalloc(&d_full_derivs, batch_size * bs_len * sizeof(double));
    cudaMalloc(&d_sparse_grad, path.size() * sizeof(double));
    cudaMalloc(&d_full_grad, path.size() * sizeof(double));
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), derivs.size() * sizeof(double), cudaMemcpyHostToDevice);

    cusig_shutdown();
    EXPECT_NE(0, branched_sig_coef_cuda_d(
        d_path, d_coefs, tree_data.data(), tree_data.size(), batch_size,
        dimension, length, max_nodes));
    ASSERT_EQ(0, prepare_branched_sig_cuda(dimension, max_nodes));
    ASSERT_EQ(0, prepare_branched_sig_coef_cuda(
        tree_data.data(), tree_data.size(), dimension, dimension, max_nodes));

    int err = branched_sig_cuda_d(
        d_path, d_full, batch_size, dimension, length, max_nodes);
    EXPECT_EQ(0, err);
    err = branched_sig_coef_cuda_d(
        d_path, d_coefs, tree_data.data(), tree_data.size(), batch_size,
        dimension, length, max_nodes);
    EXPECT_EQ(0, err);

    std::vector<double> full(batch_size * bs_len);
    std::vector<double> coefs(derivs.size());
    cudaMemcpy(full.data(), d_full, full.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(coefs.data(), d_coefs, coefs.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (uint64_t batch = 0; batch < batch_size; ++batch) {
        for (uint64_t i = 0; i < num_indices; ++i)
            EXPECT_NEAR(coefs[batch * num_indices + i],
                full[batch * bs_len + indices[i]], 1e-12);
    }

    std::vector<double> full_derivs(batch_size * bs_len, 0.);
    for (uint64_t batch = 0; batch < batch_size; ++batch) {
        for (uint64_t i = 0; i < num_indices; ++i)
            full_derivs[batch * bs_len + indices[i]] += derivs[batch * num_indices + i];
    }
    cudaMemcpy(d_full_derivs, full_derivs.data(),
        full_derivs.size() * sizeof(double), cudaMemcpyHostToDevice);

    err = branched_sig_coef_backprop_cuda_d(
        d_path, d_sparse_grad, d_coefs, d_derivs, tree_data.data(),
        tree_data.size(), batch_size, dimension, length, max_nodes);
    EXPECT_EQ(0, err);
    err = branched_sig_backprop_cuda_d(
        d_path, d_full_grad, d_full_derivs, d_full, batch_size,
        dimension, length, max_nodes);
    EXPECT_EQ(0, err);

    std::vector<double> sparse_grad(path.size());
    std::vector<double> full_grad(path.size());
    cudaMemcpy(sparse_grad.data(), d_sparse_grad,
        sparse_grad.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(full_grad.data(), d_full_grad,
        full_grad.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (uint64_t i = 0; i < path.size(); ++i)
        EXPECT_NEAR(sparse_grad[i], full_grad[i], 1e-11);

    cudaFree(d_path);
    cudaFree(d_full);
    cudaFree(d_coefs);
    cudaFree(d_derivs);
    cudaFree(d_full_derivs);
    cudaFree(d_sparse_grad);
    cudaFree(d_full_grad);
}

TEST(branchedLogSigFromPathCudaTest, ForwardAndBackwardMatchMethodTwo) {
    const uint64_t batch_size = 2;
    const uint64_t length = 5;
    const uint64_t dimension = 2;
    const uint64_t max_nodes = 4;
    const uint64_t full_length = compute_branched_sig_length(
        dimension, max_nodes, true);
    const uint64_t log_length = compute_branched_log_sig_length(
        dimension, max_nodes, true);
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 3, true, false));

    const std::vector<double> path{
        0.1, -0.2, 0.4, 0.3, 0.4, 0.3, -0.1, 0.8, 0.2, -0.4,
        -0.3, 0.2, 0.6, -0.5, 0.6, -0.5, 0.9, 0.1, -0.2, 0.7
    };
    std::vector<double> derivs(batch_size * log_length);
    for (uint64_t index = 0; index < derivs.size(); ++index)
        derivs[index] = 0.09 * static_cast<double>(index + 1) - 0.4;

    double* d_path = nullptr;
    double* d_out = nullptr;
    double* d_derivs = nullptr;
    double* d_path_derivs = nullptr;
    double* d_bsig = nullptr;
    double* d_method_two_out = nullptr;
    double* d_bsig_derivs = nullptr;
    double* d_method_two_path_derivs = nullptr;
    cudaMalloc(&d_path, path.size() * sizeof(double));
    cudaMalloc(&d_out, derivs.size() * sizeof(double));
    cudaMalloc(&d_derivs, derivs.size() * sizeof(double));
    cudaMalloc(&d_path_derivs, path.size() * sizeof(double));
    cudaMalloc(&d_bsig, batch_size * full_length * sizeof(double));
    cudaMalloc(&d_method_two_out, derivs.size() * sizeof(double));
    cudaMalloc(&d_bsig_derivs,
        batch_size * full_length * sizeof(double));
    cudaMalloc(&d_method_two_path_derivs, path.size() * sizeof(double));
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), derivs.size() * sizeof(double),
        cudaMemcpyHostToDevice);

    ASSERT_EQ(0, branched_sig_cuda_d(
        d_path, d_bsig, batch_size, dimension, length, max_nodes,
        false, false, 1., true, true));
    ASSERT_EQ(0, branched_sig_to_log_sig_cuda_d(
        d_bsig, d_method_two_out, batch_size, dimension, max_nodes,
        2, true, true));
    ASSERT_EQ(0, branched_sig_to_log_sig_backprop_cuda_d(
        d_bsig, d_derivs, d_bsig_derivs, batch_size, dimension,
        max_nodes, 2, true, true));
    ASSERT_EQ(0, branched_sig_backprop_cuda_d(
        d_path, d_method_two_path_derivs, d_bsig_derivs, d_bsig,
        batch_size, dimension, length, max_nodes,
        false, false, 1., true, true));
    ASSERT_EQ(0, branched_log_sig_from_path_cuda_d(
        d_path, d_out, batch_size, length, dimension, max_nodes));
    ASSERT_EQ(0, branched_log_sig_from_path_backprop_cuda_d(
        d_derivs, d_path_derivs, d_path, batch_size, length,
        dimension, max_nodes));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<double> expected(derivs.size());
    std::vector<double> expected_gradient(path.size());
    std::vector<double> actual(derivs.size());
    std::vector<double> actual_gradient(path.size());
    std::vector<double> actual_derivs(derivs.size());
    cudaMemcpy(expected.data(), d_method_two_out,
        expected.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(expected_gradient.data(), d_method_two_path_derivs,
        expected_gradient.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(double),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_gradient.data(), d_path_derivs,
        actual_gradient.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_derivs.data(), d_derivs,
        actual_derivs.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (uint64_t index = 0; index < actual.size(); ++index)
        EXPECT_NEAR(actual[index], expected[index], 2e-11);
    for (uint64_t index = 0; index < actual_gradient.size(); ++index)
        EXPECT_NEAR(actual_gradient[index], expected_gradient[index], 2e-10);
    EXPECT_EQ(actual_derivs, derivs);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_derivs);
    cudaFree(d_path_derivs);
    cudaFree(d_bsig);
    cudaFree(d_method_two_out);
    cudaFree(d_bsig_derivs);
    cudaFree(d_method_two_path_derivs);
}

TEST(branchedLogSigFromPathCudaTest, FloatForwardAndBackwardMatchMethodTwo) {
    const uint64_t batch_size = 1;
    const uint64_t length = 4;
    const uint64_t dimension = 2;
    const uint64_t max_nodes = 3;
    const uint64_t full_length = compute_branched_sig_length(
        dimension, max_nodes, true);
    const uint64_t log_length = compute_branched_log_sig_length(
        dimension, max_nodes, true);
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 3, true, false));

    const std::vector<float> path{
        0.2f, -0.1f, 0.5f, 0.3f, -0.4f, 0.8f, 0.7f, -0.2f
    };
    std::vector<float> derivs(log_length);
    for (uint64_t index = 0; index < derivs.size(); ++index)
        derivs[index] = 0.05f * static_cast<float>(index + 1) - 0.3f;

    float* d_path = nullptr;
    float* d_out = nullptr;
    float* d_derivs = nullptr;
    float* d_path_derivs = nullptr;
    float* d_bsig = nullptr;
    float* d_method_two_out = nullptr;
    float* d_bsig_derivs = nullptr;
    float* d_method_two_path_derivs = nullptr;
    cudaMalloc(&d_path, path.size() * sizeof(float));
    cudaMalloc(&d_out, derivs.size() * sizeof(float));
    cudaMalloc(&d_derivs, derivs.size() * sizeof(float));
    cudaMalloc(&d_path_derivs, path.size() * sizeof(float));
    cudaMalloc(&d_bsig, batch_size * full_length * sizeof(float));
    cudaMalloc(&d_method_two_out, derivs.size() * sizeof(float));
    cudaMalloc(&d_bsig_derivs,
        batch_size * full_length * sizeof(float));
    cudaMalloc(&d_method_two_path_derivs, path.size() * sizeof(float));
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), derivs.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    ASSERT_EQ(0, branched_sig_cuda_f(
        d_path, d_bsig, batch_size, dimension, length, max_nodes,
        false, false, 1.f, true, true));
    ASSERT_EQ(0, branched_sig_to_log_sig_cuda_f(
        d_bsig, d_method_two_out, batch_size, dimension, max_nodes,
        2, true, true));
    ASSERT_EQ(0, branched_sig_to_log_sig_backprop_cuda_f(
        d_bsig, d_derivs, d_bsig_derivs, batch_size, dimension,
        max_nodes, 2, true, true));
    ASSERT_EQ(0, branched_sig_backprop_cuda_f(
        d_path, d_method_two_path_derivs, d_bsig_derivs, d_bsig,
        batch_size, dimension, length, max_nodes,
        false, false, 1.f, true, true));
    ASSERT_EQ(0, branched_log_sig_from_path_cuda_f(
        d_path, d_out, batch_size, length, dimension, max_nodes));
    ASSERT_EQ(0, branched_log_sig_from_path_backprop_cuda_f(
        d_derivs, d_path_derivs, d_path, batch_size, length,
        dimension, max_nodes));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float> expected(derivs.size());
    std::vector<float> expected_gradient(path.size());
    std::vector<float> actual(derivs.size());
    std::vector<float> actual_gradient(path.size());
    cudaMemcpy(expected.data(), d_method_two_out,
        expected.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(expected_gradient.data(), d_method_two_path_derivs,
        expected_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_gradient.data(), d_path_derivs,
        actual_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (uint64_t index = 0; index < actual.size(); ++index)
        EXPECT_NEAR(actual[index], expected[index], 2e-5);
    for (uint64_t index = 0; index < actual_gradient.size(); ++index)
        EXPECT_NEAR(actual_gradient[index], expected_gradient[index], 2e-4);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_derivs);
    cudaFree(d_path_derivs);
    cudaFree(d_bsig);
    cudaFree(d_method_two_out);
    cudaFree(d_bsig_derivs);
    cudaFree(d_method_two_path_derivs);
}

TEST(branchedLogSigFromPathCudaTest, ZeroIncrementGradientFiniteDifference) {
    const uint64_t length = 5;
    const uint64_t dimension = 2;
    const uint64_t max_nodes = 3;
    const uint64_t log_length = compute_branched_log_sig_length(
        dimension, max_nodes, true);
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 3, true, false));

    std::vector<double> path{
        0.1, -0.2, 0.4, 0.3, 0.4, 0.3, -0.1, 0.8, 0.2, -0.4
    };
    std::vector<double> derivs(log_length);
    for (uint64_t index = 0; index < derivs.size(); ++index)
        derivs[index] = 0.07 * static_cast<double>(index + 1) - 0.25;

    double* d_path = nullptr;
    double* d_out = nullptr;
    double* d_derivs = nullptr;
    double* d_path_derivs = nullptr;
    cudaMalloc(&d_path, path.size() * sizeof(double));
    cudaMalloc(&d_out, log_length * sizeof(double));
    cudaMalloc(&d_derivs, derivs.size() * sizeof(double));
    cudaMalloc(&d_path_derivs, path.size() * sizeof(double));
    cudaMemcpy(d_derivs, derivs.data(), derivs.size() * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(double),
        cudaMemcpyHostToDevice);
    ASSERT_EQ(0, branched_log_sig_from_path_backprop_cuda_d(
        d_derivs, d_path_derivs, d_path, 1, length, dimension, max_nodes));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<double> gradient(path.size());
    cudaMemcpy(gradient.data(), d_path_derivs, gradient.size() * sizeof(double),
        cudaMemcpyDeviceToHost);

    const double epsilon = 1e-6;
    std::vector<double> out_plus(log_length);
    std::vector<double> out_minus(log_length);
    for (uint64_t path_index = 0; path_index < path.size(); ++path_index) {
        const double original = path[path_index];
        path[path_index] = original + epsilon;
        cudaMemcpy(d_path, path.data(), path.size() * sizeof(double),
            cudaMemcpyHostToDevice);
        ASSERT_EQ(0, branched_log_sig_from_path_cuda_d(
            d_path, d_out, 1, length, dimension, max_nodes));
        cudaMemcpy(out_plus.data(), d_out, out_plus.size() * sizeof(double),
            cudaMemcpyDeviceToHost);

        path[path_index] = original - epsilon;
        cudaMemcpy(d_path, path.data(), path.size() * sizeof(double),
            cudaMemcpyHostToDevice);
        ASSERT_EQ(0, branched_log_sig_from_path_cuda_d(
            d_path, d_out, 1, length, dimension, max_nodes));
        cudaMemcpy(out_minus.data(), d_out, out_minus.size() * sizeof(double),
            cudaMemcpyDeviceToHost);
        path[path_index] = original;

        double numerical = 0.;
        for (uint64_t output_index = 0; output_index < log_length;
            ++output_index) {
            numerical += derivs[output_index]
                * (out_plus[output_index] - out_minus[output_index])
                / (2. * epsilon);
        }
        EXPECT_NEAR(gradient[path_index], numerical, 2e-7);
    }

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_derivs);
    cudaFree(d_path_derivs);
}

TEST(branchedLogSigFromPathCudaTest, LengthOneReturnsZeros) {
    const uint64_t batch_size = 2;
    const uint64_t length = 1;
    const uint64_t dimension = 2;
    const uint64_t max_nodes = 3;
    const uint64_t log_length = compute_branched_log_sig_length(
        dimension, max_nodes, true);
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 3, true, false));

    const std::vector<double> path{ 0.2, -0.3, 0.7, 0.1 };
    const std::vector<double> derivs(batch_size * log_length, 1.);
    double* d_path = nullptr;
    double* d_out = nullptr;
    double* d_derivs = nullptr;
    double* d_path_derivs = nullptr;
    cudaMalloc(&d_path, path.size() * sizeof(double));
    cudaMalloc(&d_out, derivs.size() * sizeof(double));
    cudaMalloc(&d_derivs, derivs.size() * sizeof(double));
    cudaMalloc(&d_path_derivs, path.size() * sizeof(double));
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), derivs.size() * sizeof(double),
        cudaMemcpyHostToDevice);

    ASSERT_EQ(0, branched_log_sig_from_path_cuda_d(
        d_path, d_out, batch_size, length, dimension, max_nodes));
    ASSERT_EQ(0, branched_log_sig_from_path_backprop_cuda_d(
        d_derivs, d_path_derivs, d_path, batch_size, length,
        dimension, max_nodes));
    std::vector<double> output(derivs.size());
    std::vector<double> gradient(path.size());
    cudaMemcpy(output.data(), d_out, output.size() * sizeof(double),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(gradient.data(), d_path_derivs, gradient.size() * sizeof(double),
        cudaMemcpyDeviceToHost);
    for (double value : output)
        EXPECT_DOUBLE_EQ(value, 0.);
    for (double value : gradient)
        EXPECT_DOUBLE_EQ(value, 0.);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_derivs);
    cudaFree(d_path_derivs);
}

TEST(branchedLogSigFromPathCudaTest, DegreeLimit) {
    EXPECT_EQ(0, prepare_branched_log_sig_cuda(0, 20, 3, true, false));
    EXPECT_NE(0, prepare_branched_log_sig_cuda(0, 21, 3, true, false));
}

TEST(branchedLogSigFromPathCudaTest, EmptyPathRejected) {
    EXPECT_NE(0, branched_log_sig_from_path_cuda_d(
        nullptr, nullptr, 1, 0, 2, 3));
    EXPECT_NE(0, branched_log_sig_from_path_backprop_cuda_d(
        nullptr, nullptr, nullptr, 1, 0, 2, 3));
}

TEST(branchedSigToLogSigCudaTest, MethodsZeroToTwoFiniteDifference) {
    const uint64_t batch_size = 2;
    const uint64_t dimension = 3;
    const uint64_t max_nodes = 4;
    const uint64_t full_length = compute_branched_sig_length(
        dimension, max_nodes, true);
    const uint64_t compact_length = compute_branched_log_sig_length(
        dimension, max_nodes, true);
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 2, true, false));

    for (bool scalar_term : { true, false }) {
        const uint64_t input_length = scalar_term
            ? full_length : full_length - 1;
        std::vector<double> input(batch_size * input_length);
        for (uint64_t batch = 0; batch < batch_size; ++batch) {
            for (uint64_t index = 0; index < input_length; ++index) {
                if (scalar_term && index == 0) {
                    input[batch * input_length] = 1.;
                } else {
                    const int64_t centered = static_cast<int64_t>(
                        (batch * 31 + index * 7) % 19) - 9;
                    input[batch * input_length + index]
                        = 0.015 * static_cast<double>(centered);
                }
            }
        }

        double* d_input = nullptr;
        cudaMalloc(&d_input, input.size() * sizeof(double));
        cudaMemcpy(d_input, input.data(), input.size() * sizeof(double),
            cudaMemcpyHostToDevice);

        for (int method = 0; method <= 2; ++method) {
            const uint64_t output_length = method == 0
                ? input_length : compact_length;
            std::vector<double> derivs(batch_size * output_length);
            for (uint64_t index = 0; index < derivs.size(); ++index) {
                const int64_t centered = static_cast<int64_t>(
                    (index * 11) % 23) - 11;
                derivs[index] = 0.02 * static_cast<double>(centered);
            }

            double* d_output = nullptr;
            double* d_derivs = nullptr;
            double* d_gradient = nullptr;
            cudaMalloc(&d_output,
                batch_size * output_length * sizeof(double));
            cudaMalloc(&d_derivs, derivs.size() * sizeof(double));
            cudaMalloc(&d_gradient, input.size() * sizeof(double));
            cudaMemcpy(d_derivs, derivs.data(), derivs.size() * sizeof(double),
                cudaMemcpyHostToDevice);

            ASSERT_EQ(0, branched_sig_to_log_sig_cuda_d(
                d_input, d_output, batch_size, dimension, max_nodes,
                method, true, scalar_term));
            ASSERT_EQ(0, branched_sig_to_log_sig_backprop_cuda_d(
                d_input, d_derivs, d_gradient, batch_size, dimension,
                max_nodes, method, true, scalar_term));
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

            std::vector<double> actual(batch_size * output_length);
            std::vector<double> actual_gradient(input.size());
            std::vector<double> actual_derivs(derivs.size());
            cudaMemcpy(actual.data(), d_output,
                actual.size() * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(actual_gradient.data(), d_gradient,
                actual_gradient.size() * sizeof(double),
                cudaMemcpyDeviceToHost);
            cudaMemcpy(actual_derivs.data(), d_derivs,
                actual_derivs.size() * sizeof(double), cudaMemcpyDeviceToHost);
            for (double value : actual)
                EXPECT_TRUE(std::isfinite(value));
            EXPECT_EQ(actual_derivs, derivs);

            const double epsilon = 1e-6;
            std::vector<double> out_plus(actual.size());
            std::vector<double> out_minus(actual.size());
            uint64_t checked = 0;
            for (uint64_t input_index = 0;
                input_index < input.size() && checked < 8; ++input_index) {
                if (scalar_term && input_index % input_length == 0)
                    continue;
                const double original = input[input_index];
                input[input_index] = original + epsilon;
                cudaMemcpy(d_input, input.data(), input.size() * sizeof(double),
                    cudaMemcpyHostToDevice);
                ASSERT_EQ(0, branched_sig_to_log_sig_cuda_d(
                    d_input, d_output, batch_size, dimension, max_nodes,
                    method, true, scalar_term));
                cudaMemcpy(out_plus.data(), d_output,
                    out_plus.size() * sizeof(double), cudaMemcpyDeviceToHost);

                input[input_index] = original - epsilon;
                cudaMemcpy(d_input, input.data(), input.size() * sizeof(double),
                    cudaMemcpyHostToDevice);
                ASSERT_EQ(0, branched_sig_to_log_sig_cuda_d(
                    d_input, d_output, batch_size, dimension, max_nodes,
                    method, true, scalar_term));
                cudaMemcpy(out_minus.data(), d_output,
                    out_minus.size() * sizeof(double), cudaMemcpyDeviceToHost);
                input[input_index] = original;

                double numerical = 0.;
                for (uint64_t output_index = 0;
                    output_index < derivs.size(); ++output_index) {
                    numerical += derivs[output_index]
                        * (out_plus[output_index] - out_minus[output_index])
                        / (2. * epsilon);
                }
                EXPECT_NEAR(actual_gradient[input_index], numerical, 2e-7);
                ++checked;
            }
            cudaMemcpy(d_input, input.data(), input.size() * sizeof(double),
                cudaMemcpyHostToDevice);

            cudaFree(d_output);
            cudaFree(d_derivs);
            cudaFree(d_gradient);
        }
        cudaFree(d_input);
    }
}

TEST(branchedSigToLogSigCudaTest, CacheUpgradeClearAndDiskRoundTrip) {
    const uint64_t dimension = 1;
    const uint64_t max_nodes = 3;
    const uint64_t full_length = compute_branched_sig_length(
        dimension, max_nodes, true);
    const uint64_t compact_length = compute_branched_log_sig_length(
        dimension, max_nodes, true);
    const std::vector<double> input(full_length, 0.);

    double* d_input = nullptr;
    double* d_output = nullptr;
    cudaMalloc(&d_input, input.size() * sizeof(double));
    cudaMalloc(&d_output, compact_length * sizeof(double));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(double),
        cudaMemcpyHostToDevice);

    ASSERT_EQ(0, clear_cache_cuda(true));
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 1, true, false));
    EXPECT_EQ(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 1, true, true));
    EXPECT_NE(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 2, true, true));

    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 2, true, false));
    EXPECT_EQ(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 1, true, true));
    EXPECT_EQ(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 2, true, true));

    ASSERT_EQ(0, clear_cache_cuda(false));
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 3, true, false));
    EXPECT_EQ(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 1, true, true));
    EXPECT_EQ(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 2, true, true));

    ASSERT_EQ(0, clear_cache_cuda(false));
    EXPECT_NE(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 1, true, true));
    EXPECT_NE(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 2, true, true));

    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 2, true, true));
    ASSERT_EQ(0, clear_cache_cuda(false));
    ASSERT_EQ(0, prepare_branched_log_sig_cuda(
        dimension, max_nodes, 1, true, true));
    EXPECT_EQ(0, branched_sig_to_log_sig_cuda_d(
        d_input, d_output, 1, dimension, max_nodes, 2, true, true));

    cudaFree(d_input);
    cudaFree(d_output);
    EXPECT_EQ(0, clear_cache_cuda(true));
}
