/* Copyright 2025 Daniil Shmelev
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

#include "dll_funcs.h"
#include "utils.h"

void example_signature_f(
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    bool horner,
    int num_runs
) {
    print_header("Signature Float");

    std::vector<float> path = test_data<float>(dimension * length);

    uint64_t out_size = sig_length(dimension, degree);
    std::vector<float> out(out_size, 0.);

    time_function(num_runs, signature_f, path.data(), out.data(), dimension, length, degree, time_aug, lead_lag, 1.f, horner);

    std::cout << "done\n";
}

void example_signature_d(
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    bool horner,
    int num_runs
) {
    print_header("Signature Double");

    std::vector<double> path = test_data<double>(dimension * length);

    uint64_t out_size = sig_length(dimension, degree);
    std::vector<double> out(out_size, 0.);

    time_function(num_runs, signature_d, path.data(), out.data(), dimension, length, degree, time_aug, lead_lag, 1., horner);

    std::cout << "done\n";
}

void example_batch_signature_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    bool horner,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Signature Double");

    std::vector<double> path = test_data<double>(batch_size * dimension * length);

    uint64_t out_size = sig_length(dimension, degree) * batch_size;
    std::vector<double> out(out_size, 0.);

    time_function(num_runs, batch_signature_d, path.data(), out.data(), batch_size, dimension, length, degree, time_aug, lead_lag, 1., horner, n_jobs);

    std::cout << "done\n";
}

void example_batch_signature_kernel_f(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2,
    uint64_t dyadic_order_1,
    uint64_t dyadic_order_2,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Signature Kernel");

    std::vector<float> out(batch_size, 0.);
    std::vector<float> gram = test_data<float>(length1 * length2 * batch_size);

    time_function(num_runs, batch_sig_kernel_f, gram.data(), out.data(), batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, n_jobs, false);

    std::cout << "done\n";
}

void example_batch_signature_kernel_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2,
    uint64_t dyadic_order_1,
    uint64_t dyadic_order_2,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Signature Kernel");

    std::vector<double> out(batch_size, 0.);
    std::vector<double> gram = test_data<double>(length1 * length2 * batch_size);

    time_function(num_runs, batch_sig_kernel_d, gram.data(), out.data(), batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, n_jobs, false);

    std::cout << "done\n";
}

void example_batch_signature_kernel_cuda(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2,
    uint64_t dyadic_order_1,
    uint64_t dyadic_order_2,
    int num_runs
) {
    print_header("Batch Signature Kernel CUDA");

    uint64_t gram_size = length1 * length2 * batch_size;
    std::vector<double> gram = test_data<double>(gram_size);
    
    double* d_gram;
    double* d_out;
    cudaMalloc(&d_gram, sizeof(double) * gram_size);
    cudaMalloc(&d_out, sizeof(double) * batch_size);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_gram, gram.data(), sizeof(double) * gram_size, cudaMemcpyHostToDevice);

    time_function(num_runs, batch_sig_kernel_cuda_d, d_gram, d_out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, false);

    cudaFree(d_gram);
    cudaFree(d_out);
}

void example_batch_signature_kernel_cuda_full_grid(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2,
    uint64_t dyadic_order_1,
    uint64_t dyadic_order_2,
    int num_runs
) {
    print_header("Batch Signature Kernel CUDA (Full Grid)");

    uint64_t gram_size = length1 * length2 * batch_size;
    uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
    uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;
    uint64_t grid_size = dyadic_length_1 * dyadic_length_2 * batch_size;
    std::vector<double> gram = test_data<double>(gram_size);
    
    double* d_gram;
    double* d_out;
    cudaMalloc(&d_gram, sizeof(double) * gram_size);
    cudaMalloc(&d_out, sizeof(double) * grid_size);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_gram, gram.data(), sizeof(double) * gram_size, cudaMemcpyHostToDevice);

    time_function(num_runs, batch_sig_kernel_cuda_d, d_gram, d_out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, true);

    cudaFree(d_gram);
    cudaFree(d_out);
}

void example_sig_backprop_d(
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    int num_runs
) {
    print_header("Sig Backprop Double");

    std::vector<double> path = test_data<double>(dimension * length);
    uint64_t sig_len = sig_length(dimension, degree);
    std::vector<double> sig_derivs = test_data<double>(sig_len);
    std::vector<double> sig = test_data<double>(sig_len);

    uint64_t out_size = dimension * length;
    std::vector<double> out(out_size, 0.);

    time_function(num_runs, sig_backprop_d, path.data(), out.data(), sig_derivs.data(), sig.data(), dimension, length, degree, time_aug, lead_lag, 1.);

    std::cout << "done\n";
}

void example_batch_sig_kernel_backprop(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2,
    uint64_t dyadic_order_1,
    uint64_t dyadic_order_2,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Sig Kernel Backprop");

    uint64_t gram_size = (length1 - 1) * (length2 - 1) * batch_size;
    std::vector<double> gram = test_data<double>(gram_size);
    std::vector<double> deriv = test_data<double>(batch_size);
    std::vector<double> out(batch_size * (length1 - 1) * (length2 - 1));
    std::vector<double> k_grid = test_data<double>(batch_size * length1 * length2);

    time_function(num_runs, batch_sig_kernel_backprop_d, gram.data(), out.data(), deriv.data(), k_grid.data(), batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, n_jobs);

    std::cout << "done\n";
}

void example_batch_sig_kernel_backprop_cuda(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2,
    uint64_t dyadic_order_1,
    uint64_t dyadic_order_2,
    int num_runs
) {
    print_header("Batch Sig Kernel Backprop CUDA");

    uint64_t dyadic_length_1 = ((length1 - 1) << dyadic_order_1) + 1;
    uint64_t dyadic_length_2 = ((length2 - 1) << dyadic_order_2) + 1;
    uint64_t grid_size = dyadic_length_1 * dyadic_length_2;
    uint64_t gram_size = (length1 - 1) * (length2 - 1) * batch_size;
    std::vector<double> gram = test_data<double>(gram_size);
    std::vector<double> deriv = test_data<double>(batch_size);
    std::vector<double> k_grid = test_data<double>(batch_size * length1 * length2);

    double* d_gram;
    double* d_out;
    double* d_deriv;
    double* d_k_grid;
    cudaMalloc(&d_gram, sizeof(double) * batch_size * gram_size);
    cudaMalloc(&d_out, sizeof(double) * batch_size * gram_size);
    cudaMalloc(&d_deriv, sizeof(double) * batch_size);
    cudaMalloc(&d_k_grid, sizeof(double) * batch_size * grid_size);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_gram, gram.data(), sizeof(double) * gram_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deriv, deriv.data(), sizeof(double) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_grid, k_grid.data(), sizeof(double) * batch_size * grid_size, cudaMemcpyHostToDevice);

    time_function(num_runs, batch_sig_kernel_backprop_cuda_d, d_gram, d_out, d_deriv, d_k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2);

    cudaFree(d_gram);
    cudaFree(d_deriv);
    cudaFree(d_k_grid);
    cudaFree(d_out);
 
    std::cout << "done\n";
}

void example_prepare_log_sig(
    uint64_t dimension,
    uint64_t degree,
    int method,
    int num_runs
) {
    print_header("Prepare Log Sig");

    auto f = [&]()
    {
            prepare_log_sig(dimension, degree, method, false);
            clear_cache(false);
    };

    time_function(num_runs, f);

    std::cout << "done\n";
}

void example_sig_to_log_sig_d(
    uint64_t dimension,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    int method,
    int num_runs
) {
    print_header("Log Signature Double");

    std::vector<double> sig = test_data<double>(sig_length(dimension, degree));

    uint64_t out_size = method ? log_sig_length(dimension, degree) : sig_length(dimension, degree);
    std::vector<double> out(out_size, 0.);

    prepare_log_sig(dimension, degree, method, true);
    time_function(num_runs, sig_to_log_sig_d, sig.data(), out.data(), dimension, degree, time_aug, lead_lag, method);

    std::cout << "done\n";
}

void example_batch_sig_coef(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    bool time_aug,
    bool lead_lag,
    double end_time,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Sig Coef");

    uint64_t path_size = dimension * length * batch_size;
    std::vector<double> path = test_data<double>(path_size);

    std::vector<uint64_t> degrees(num_idx, degree);
    std::vector<uint64_t> multi_idx(num_idx * degree, 0);

    std::vector<double> out(batch_size * num_idx);

    time_function(num_runs, batch_sig_coef_d, path.data(), out.data(), multi_idx.data(), degrees.size(), degrees.data(), batch_size, dimension, length, time_aug, lead_lag, end_time, false, n_jobs);
}

void example_batch_sig_coef_backprop(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    bool time_aug,
    bool lead_lag,
    double end_time,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Sig Coef Backprop");

    uint64_t path_size = dimension * length * batch_size;
    std::vector<double> path = test_data<double>(path_size);
    path[0] = 0.;
    path[1] = 1.;

    std::vector<uint64_t> degrees(num_idx, degree);
    std::vector<uint64_t> multi_idx(num_idx * degree, 0);

    uint64_t prefix_coef_size = degree * num_idx;

    std::vector<double> out(path_size);
    std::vector<double> coefs(batch_size * prefix_coef_size);
    std::vector<double> derivs(batch_size * prefix_coef_size);

    time_function(num_runs, batch_sig_coef_backprop_d, path.data(), out.data(), coefs.data(), derivs.data(), multi_idx.data(), degrees.size(), degrees.data(), batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs);
}
