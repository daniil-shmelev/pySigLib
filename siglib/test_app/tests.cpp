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

void example_signature_cuda_f(
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    bool horner,
    int num_runs
) {
    print_header("Signature CUDA Float");

    uint64_t path_size = dimension * length;
    std::vector<float> path = test_data<float>(path_size);

    uint64_t out_size = sig_length(dimension, degree);

    float* d_path;
    float* d_out;
    cudaMalloc(&d_path, sizeof(float) * path_size);
    cudaMalloc(&d_out, sizeof(float) * out_size);

    cudaMemcpy(d_path, path.data(), sizeof(float) * path_size, cudaMemcpyHostToDevice);

    time_function(num_runs, signature_cuda_f, d_path, d_out, dimension, length, degree, time_aug, lead_lag, 1.f, horner);

    cudaFree(d_path);
    cudaFree(d_out);

    std::cout << "done\n";
}

void example_signature_cuda_d(
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    bool horner,
    int num_runs
) {
    print_header("Signature CUDA Double");

    uint64_t path_size = dimension * length;
    std::vector<double> path = test_data<double>(path_size);

    uint64_t out_size = sig_length(dimension, degree);

    double* d_path;
    double* d_out;
    cudaMalloc(&d_path, sizeof(double) * path_size);
    cudaMalloc(&d_out, sizeof(double) * out_size);

    cudaMemcpy(d_path, path.data(), sizeof(double) * path_size, cudaMemcpyHostToDevice);

    time_function(num_runs, signature_cuda_d, d_path, d_out, dimension, length, degree, time_aug, lead_lag, 1., horner);

    cudaFree(d_path);
    cudaFree(d_out);

    std::cout << "done\n";
}

void example_batch_signature_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    bool horner,
    int num_runs
) {
    print_header("Batch Signature CUDA Double");

    uint64_t path_size = batch_size * dimension * length;
    std::vector<double> path = test_data<double>(path_size);

    uint64_t out_size = sig_length(dimension, degree) * batch_size;

    double* d_path;
    double* d_out;
    cudaMalloc(&d_path, sizeof(double) * path_size);
    cudaMalloc(&d_out, sizeof(double) * out_size);

    cudaMemcpy(d_path, path.data(), sizeof(double) * path_size, cudaMemcpyHostToDevice);

    time_function(num_runs, batch_signature_cuda_d, d_path, d_out, batch_size, dimension, length, degree, time_aug, lead_lag, 1., horner);

    cudaFree(d_path);
    cudaFree(d_out);

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

void example_batch_sig_backprop_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Sig Backprop Double");

    uint64_t path_size = batch_size * dimension * length;
    std::vector<double> path = test_data<double>(path_size);
    uint64_t sig_len = sig_length(dimension, degree);
    std::vector<double> sig_derivs = test_data<double>(batch_size * sig_len);
    std::vector<double> sig = test_data<double>(batch_size * sig_len);

    uint64_t out_size = batch_size * dimension * length;
    std::vector<double> out(out_size, 0.);

    time_function(num_runs, batch_sig_backprop_d, path.data(), out.data(), sig_derivs.data(), sig.data(), batch_size, dimension, length, degree, time_aug, lead_lag, 1., n_jobs);

    std::cout << "done\n";
}

void example_sig_backprop_cuda_d(
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    int num_runs
) {
    print_header("Sig Backprop CUDA Double");

    std::vector<double> path = test_data<double>(dimension * length);
    uint64_t sig_len = sig_length(dimension, degree);
    std::vector<double> sig_derivs = test_data<double>(sig_len);
    std::vector<double> sig = test_data<double>(sig_len);

    uint64_t out_size = dimension * length;

    double* d_path;
    double* d_out;
    double* d_sig_derivs;
    double* d_sig;
    cudaMalloc(&d_path, sizeof(double) * dimension * length);
    cudaMalloc(&d_out, sizeof(double) * out_size);
    cudaMalloc(&d_sig_derivs, sizeof(double) * sig_len);
    cudaMalloc(&d_sig, sizeof(double) * sig_len);

    cudaMemcpy(d_path, path.data(), sizeof(double) * dimension * length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_derivs, sig_derivs.data(), sizeof(double) * sig_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig.data(), sizeof(double) * sig_len, cudaMemcpyHostToDevice);

    time_function(num_runs, sig_backprop_cuda_d, d_path, d_out, d_sig_derivs, d_sig, dimension, length, degree, time_aug, lead_lag, 1.);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_sig_derivs);
    cudaFree(d_sig);

    std::cout << "done\n";
}

void example_batch_sig_backprop_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    int num_runs
) {
    print_header("Batch Sig Backprop CUDA Double");

    uint64_t path_size = batch_size * dimension * length;
    std::vector<double> path = test_data<double>(path_size);
    uint64_t sig_len = sig_length(dimension, degree);
    std::vector<double> sig_derivs = test_data<double>(batch_size * sig_len);
    std::vector<double> sig = test_data<double>(batch_size * sig_len);

    uint64_t out_size = batch_size * dimension * length;

    double* d_path;
    double* d_out;
    double* d_sig_derivs;
    double* d_sig;
    cudaMalloc(&d_path, sizeof(double) * path_size);
    cudaMalloc(&d_out, sizeof(double) * out_size);
    cudaMalloc(&d_sig_derivs, sizeof(double) * batch_size * sig_len);
    cudaMalloc(&d_sig, sizeof(double) * batch_size * sig_len);

    cudaMemcpy(d_path, path.data(), sizeof(double) * path_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_derivs, sig_derivs.data(), sizeof(double) * batch_size * sig_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig.data(), sizeof(double) * batch_size * sig_len, cudaMemcpyHostToDevice);

    time_function(num_runs, batch_sig_backprop_cuda_d, d_path, d_out, d_sig_derivs, d_sig, batch_size, dimension, length, degree, time_aug, lead_lag, 1.);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_sig_derivs);
    cudaFree(d_sig);

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

void example_batch_sig_to_log_sig_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    int method,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Log Signature Double");

    uint64_t slen = sig_length(dimension, degree);
    std::vector<double> sig = test_data<double>(batch_size * slen);

    uint64_t out_len = method ? log_sig_length(dimension, degree) : slen;
    std::vector<double> out(batch_size * out_len, 0.);

    prepare_log_sig(dimension, degree, method, true);
    time_function(num_runs, batch_sig_to_log_sig_d, sig.data(), out.data(), batch_size, dimension, degree, time_aug, lead_lag, method, n_jobs);

    std::cout << "done\n";
}

void example_sig_to_log_sig_cuda_d(
    uint64_t dimension,
    uint64_t degree,
    int method,
    int num_runs
) {
    print_header("Log Signature CUDA Double");

    uint64_t slen = sig_length(dimension, degree);
    std::vector<double> sig = test_data<double>(slen);

    double* d_sig;
    double* d_out;
    cudaMalloc(&d_sig, sizeof(double) * slen);
    cudaMalloc(&d_out, sizeof(double) * slen);

    cudaMemcpy(d_sig, sig.data(), sizeof(double) * slen, cudaMemcpyHostToDevice);

    prepare_log_sig_cuda(dimension, degree, method);
    time_function(num_runs, sig_to_log_sig_cuda_d, d_sig, d_out, dimension, degree, method);

    cudaFree(d_sig);
    cudaFree(d_out);

    std::cout << "done\n";
}

void example_batch_sig_to_log_sig_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int method,
    int num_runs
) {
    print_header("Batch Log Signature CUDA Double");

    uint64_t slen = sig_length(dimension, degree);
    uint64_t total = batch_size * slen;
    std::vector<double> sig = test_data<double>(total);

    double* d_sig;
    double* d_out;
    cudaMalloc(&d_sig, sizeof(double) * total);
    cudaMalloc(&d_out, sizeof(double) * total);

    cudaMemcpy(d_sig, sig.data(), sizeof(double) * total, cudaMemcpyHostToDevice);

    prepare_log_sig_cuda(dimension, degree, method);
    time_function(num_runs, batch_sig_to_log_sig_cuda_d, d_sig, d_out, batch_size, dimension, degree, method);

    cudaFree(d_sig);
    cudaFree(d_out);

    std::cout << "done\n";
}

void example_batch_sig_to_log_sig_backprop_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    bool time_aug,
    bool lead_lag,
    int method,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Log Signature Backprop Double");

    uint64_t slen = sig_length(dimension, degree);
    std::vector<double> sig = test_data<double>(batch_size * slen);

    uint64_t derivs_len = method ? log_sig_length(dimension, degree) : slen;
    std::vector<double> derivs = test_data<double>(batch_size * derivs_len);

    std::vector<double> out(batch_size * slen, 0.);

    prepare_log_sig(dimension, degree, method, true);
    time_function(num_runs, batch_sig_to_log_sig_backprop_d, sig.data(), out.data(), derivs.data(), batch_size, dimension, degree, time_aug, lead_lag, method, n_jobs);

    std::cout << "done\n";
}

void example_batch_sig_to_log_sig_backprop_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int method,
    int num_runs
) {
    print_header("Batch Log Signature Backprop CUDA Double");

    uint64_t slen = sig_length(dimension, degree);
    uint64_t derivs_len = method ? log_sig_length(dimension, degree) : slen;
    std::vector<double> sig = test_data<double>(batch_size * slen);
    std::vector<double> derivs = test_data<double>(batch_size * derivs_len);

    double* d_sig;
    double* d_out;
    double* d_derivs;
    cudaMalloc(&d_sig, sizeof(double) * batch_size * slen);
    cudaMalloc(&d_out, sizeof(double) * batch_size * slen);
    cudaMalloc(&d_derivs, sizeof(double) * batch_size * derivs_len);

    cudaMemcpy(d_sig, sig.data(), sizeof(double) * batch_size * slen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), sizeof(double) * batch_size * derivs_len, cudaMemcpyHostToDevice);

    prepare_log_sig_cuda(dimension, degree, method);
    time_function(num_runs, batch_sig_to_log_sig_backprop_cuda_d, d_sig, d_out, d_derivs, batch_size, dimension, degree, method);

    cudaFree(d_sig);
    cudaFree(d_out);
    cudaFree(d_derivs);

    std::cout << "done\n";
}

void example_batch_sig_combine_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int num_runs
) {
    print_header("Batch Sig Combine CUDA Double");

    uint64_t sig_len = sig_length(dimension, degree);
    uint64_t total = batch_size * sig_len;
    std::vector<double> sig1 = test_data<double>(total);
    std::vector<double> sig2 = test_data<double>(total);

    double* d_sig1;
    double* d_sig2;
    double* d_out;
    cudaMalloc(&d_sig1, sizeof(double) * total);
    cudaMalloc(&d_sig2, sizeof(double) * total);
    cudaMalloc(&d_out, sizeof(double) * total);

    cudaMemcpy(d_sig1, sig1.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig2, sig2.data(), sizeof(double) * total, cudaMemcpyHostToDevice);

    time_function(num_runs, batch_sig_combine_cuda_d, d_sig1, d_sig2, d_out, batch_size, dimension, degree);

    cudaFree(d_sig1);
    cudaFree(d_sig2);
    cudaFree(d_out);

    std::cout << "done\n";
}

void example_batch_sig_combine_backprop_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Sig Combine Backprop CPU Double");

    uint64_t sig_len = sig_length(dimension, degree);
    uint64_t total = batch_size * sig_len;
    std::vector<double> sig1 = test_data<double>(total);
    std::vector<double> sig2 = test_data<double>(total);
    std::vector<double> sig_combined_deriv = test_data<double>(total);
    std::vector<double> sig1_deriv(total);
    std::vector<double> sig2_deriv(total);

    time_function(num_runs, batch_sig_combine_backprop_d, sig_combined_deriv.data(), sig1_deriv.data(), sig2_deriv.data(), sig1.data(), sig2.data(), batch_size, dimension, degree, n_jobs);

    std::cout << "done\n";
}

void example_batch_sig_combine_backprop_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int num_runs
) {
    print_header("Batch Sig Combine Backprop CUDA Double");

    uint64_t sig_len = sig_length(dimension, degree);
    uint64_t total = batch_size * sig_len;
    std::vector<double> sig1 = test_data<double>(total);
    std::vector<double> sig2 = test_data<double>(total);
    std::vector<double> sig_combined_deriv = test_data<double>(total);

    double* d_sig1;
    double* d_sig2;
    double* d_sig_combined_deriv;
    double* d_sig1_deriv;
    double* d_sig2_deriv;
    cudaMalloc(&d_sig1, sizeof(double) * total);
    cudaMalloc(&d_sig2, sizeof(double) * total);
    cudaMalloc(&d_sig_combined_deriv, sizeof(double) * total);
    cudaMalloc(&d_sig1_deriv, sizeof(double) * total);
    cudaMalloc(&d_sig2_deriv, sizeof(double) * total);

    cudaMemcpy(d_sig1, sig1.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig2, sig2.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_combined_deriv, sig_combined_deriv.data(), sizeof(double) * total, cudaMemcpyHostToDevice);

    time_function(num_runs, batch_sig_combine_backprop_cuda_d, d_sig_combined_deriv, d_sig1_deriv, d_sig2_deriv, d_sig1, d_sig2, batch_size, dimension, degree);

    cudaFree(d_sig1);
    cudaFree(d_sig2);
    cudaFree(d_sig_combined_deriv);
    cudaFree(d_sig1_deriv);
    cudaFree(d_sig2_deriv);

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
