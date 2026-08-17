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

#include <algorithm>
#include <cmath>
#include <stdexcept>


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

    time_function(num_runs, signature_d, path.data(), out.data(), batch_size, dimension, length, degree, time_aug, lead_lag, 1., horner, true, n_jobs, nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);

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

    time_function(num_runs, signature_cuda_d, d_path, d_out, batch_size, dimension, length, degree, time_aug, lead_lag, 1., horner, true, nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);

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

    time_function(num_runs, sig_kernel_f, gram.data(), out.data(), batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, false, n_jobs);

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

    time_function(num_runs, sig_kernel_d, gram.data(), out.data(), batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, false, n_jobs);

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

    time_function(num_runs, sig_kernel_cuda_d, d_gram, d_out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, false);

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

    time_function(num_runs, sig_kernel_cuda_d, d_gram, d_out, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, true);

    cudaFree(d_gram);
    cudaFree(d_out);
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

    time_function(num_runs, sig_backprop_d, path.data(), out.data(), sig_derivs.data(), sig.data(), batch_size, dimension, length, degree, time_aug, lead_lag, 1., true, n_jobs, nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);

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

    time_function(num_runs, sig_backprop_cuda_d, d_path, d_out, d_sig_derivs, d_sig, batch_size, dimension, length, degree, time_aug, lead_lag, 1., true, nullptr, (uint64_t)0, (uint64_t)0, (uint64_t)0);

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

    time_function(num_runs, sig_kernel_backprop_d, gram.data(), out.data(), deriv.data(), k_grid.data(), batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, false, n_jobs);

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

    time_function(num_runs, sig_kernel_backprop_cuda_d, d_gram, d_out, d_deriv, d_k_grid, batch_size, dimension, length1, length2, dyadic_order_1, dyadic_order_2, false);

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
    time_function(num_runs, sig_to_log_sig_d, sig.data(), out.data(), batch_size, dimension, degree, time_aug, lead_lag, method, n_jobs);

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

    prepare_log_sig_cuda(dimension, degree, method, false);
    time_function(num_runs, sig_to_log_sig_cuda_d, d_sig, d_out, batch_size, dimension, degree, method);

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
    time_function(num_runs, sig_to_log_sig_backprop_d, sig.data(), out.data(), derivs.data(), batch_size, dimension, degree, time_aug, lead_lag, method, n_jobs);

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

    prepare_log_sig_cuda(dimension, degree, method, false);
    time_function(num_runs, sig_to_log_sig_backprop_cuda_d, d_sig, d_out, d_derivs, batch_size, dimension, degree, method);

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

    time_function(num_runs, sig_combine_cuda_d, d_sig1, d_sig2, d_out, batch_size, dimension, degree);

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

    time_function(num_runs, sig_combine_backprop_d, sig_combined_deriv.data(), sig1_deriv.data(), sig2_deriv.data(), sig1.data(), sig2.data(), batch_size, dimension, degree, n_jobs);

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

    time_function(num_runs, sig_combine_backprop_cuda_d, d_sig_combined_deriv, d_sig1_deriv, d_sig2_deriv, d_sig1, d_sig2, batch_size, dimension, degree);

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

    time_function(num_runs, sig_coef_d, path.data(), out.data(), multi_idx.data(), degrees.size(), degrees.data(), batch_size, dimension, length, time_aug, lead_lag, end_time, false, n_jobs);
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

    time_function(num_runs, sig_coef_backprop_d, path.data(), out.data(), coefs.data(), derivs.data(), multi_idx.data(), degrees.size(), degrees.data(), batch_size, dimension, length, time_aug, lead_lag, end_time, n_jobs);
}

static std::vector<uint64_t> branched_sig_coef_tree_data_(
    uint64_t num_idx,
    uint64_t dimension,
    uint64_t degree
) {
    if (dimension == 0)
        throw std::invalid_argument("dimension must be positive");
    uint64_t max_unique = 1;
    for (uint64_t node = 0; node < degree; ++node) {
        if (max_unique > UINT64_MAX / dimension) {
            max_unique = UINT64_MAX;
            break;
        }
        max_unique *= dimension;
    }
    if (num_idx > max_unique)
        throw std::invalid_argument("num_idx exceeds the number of unique chain trees");

    std::vector<uint64_t> tree_data{ num_idx };
    for (uint64_t i = 0; i < num_idx; ++i) {
        uint64_t word = i;
        tree_data.push_back(degree == 0 ? 0 : 1);
        for (uint64_t node = 0; node < degree; ++node) {
            tree_data.push_back(word % dimension);
            word /= dimension;
            tree_data.push_back(node + 1 < degree ? 1 : 0);
        }
    }
    return tree_data;
}

void example_batch_branched_sig_coef(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    int n_jobs,
    int num_runs,
    bool planar
) {
    print_header("Batch Branched Sig Coef");
    if (num_idx == 0)
        throw std::invalid_argument("num_idx must be positive");
    const auto tree_data = branched_sig_coef_tree_data_(num_idx, dimension, degree);
    if (prepare_branched_sig_coef(
        tree_data.data(), tree_data.size(), dimension, dimension, degree, planar, false) != 0)
        throw std::runtime_error("prepare_branched_sig_coef failed");
    std::vector<double> path = test_data<double>(batch_size * dimension * length);
    std::vector<double> out(batch_size * num_idx);

    std::cout << "planar: " << planar << "\n";
    std::cout << "num_idx: " << num_idx << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "dimension: " << dimension << "\n";
    std::cout << "length: " << length << "\n";
    std::cout << "degree: " << degree << "\n";
    std::cout << "n_jobs: " << n_jobs << "\n";
    std::cout << "num_runs: " << num_runs << "\n";

    time_function(num_runs, branched_sig_coef_d, path.data(), out.data(),
        tree_data.data(), tree_data.size(), batch_size, dimension, length, degree, n_jobs,
        false, false, 1., planar, nullptr, 0, 0, 0);
}

void example_batch_branched_sig_coef_backprop(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    int n_jobs,
    int num_runs,
    bool planar
) {
    print_header("Batch Branched Sig Coef Backprop");
    if (num_idx == 0)
        throw std::invalid_argument("num_idx must be positive");
    const auto tree_data = branched_sig_coef_tree_data_(num_idx, dimension, degree);
    if (prepare_branched_sig_coef(
        tree_data.data(), tree_data.size(), dimension, dimension, degree, planar, false) != 0)
        throw std::runtime_error("prepare_branched_sig_coef failed");
    const uint64_t path_size = batch_size * dimension * length;
    std::vector<double> path = test_data<double>(path_size);
    std::vector<double> out(path_size);
    std::vector<double> coefs(batch_size * num_idx);
    std::vector<double> derivs(batch_size * num_idx, 1.);
    if (branched_sig_coef_d(path.data(), coefs.data(), tree_data.data(), tree_data.size(),
        batch_size, dimension, length, degree, n_jobs, false, false, 1., planar,
        nullptr, 0, 0, 0) != 0)
        throw std::runtime_error("branched_sig_coef_d failed");

    std::cout << "planar: " << planar << "\n";
    std::cout << "num_idx: " << num_idx << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "dimension: " << dimension << "\n";
    std::cout << "length: " << length << "\n";
    std::cout << "degree: " << degree << "\n";
    std::cout << "n_jobs: " << n_jobs << "\n";
    std::cout << "num_runs: " << num_runs << "\n";

    time_function(num_runs, branched_sig_coef_backprop_d, path.data(), out.data(),
        coefs.data(), derivs.data(), tree_data.data(), tree_data.size(), batch_size,
        dimension, length, degree, n_jobs, false, false, 1., planar, nullptr, 0, 0, 0);
}

void example_batch_branched_sig_coef_cuda(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    int num_runs,
    bool planar
) {
    print_header("Batch Branched Sig Coef CUDA");
    if (num_idx == 0)
        throw std::invalid_argument("num_idx must be positive");
    const auto tree_data = branched_sig_coef_tree_data_(num_idx, dimension, degree);
    if (prepare_branched_sig_coef_cuda(
        tree_data.data(), tree_data.size(), dimension, dimension, degree, planar, false) != 0)
        throw std::runtime_error("prepare_branched_sig_coef_cuda failed");
    std::vector<double> path = test_data<double>(batch_size * dimension * length);

    double* d_path = nullptr;
    double* d_out = nullptr;
    cudaMalloc(&d_path, path.size() * sizeof(double));
    cudaMalloc(&d_out, batch_size * num_idx * sizeof(double));
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "planar: " << planar << "\n";
    std::cout << "num_idx: " << num_idx << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "dimension: " << dimension << "\n";
    std::cout << "length: " << length << "\n";
    std::cout << "degree: " << degree << "\n";
    std::cout << "num_runs: " << num_runs << "\n";

    time_function(num_runs, branched_sig_coef_cuda_d, d_path, d_out,
        tree_data.data(), tree_data.size(), batch_size, dimension, length, degree,
        false, false, 1., planar, nullptr, 0, 0, 0);

    cudaFree(d_path);
    cudaFree(d_out);
}

void example_batch_branched_sig_coef_backprop_cuda(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    int num_runs,
    bool planar
) {
    print_header("Batch Branched Sig Coef Backprop CUDA");
    if (num_idx == 0)
        throw std::invalid_argument("num_idx must be positive");
    const auto tree_data = branched_sig_coef_tree_data_(num_idx, dimension, degree);
    if (prepare_branched_sig_coef_cuda(
        tree_data.data(), tree_data.size(), dimension, dimension, degree, planar, false) != 0)
        throw std::runtime_error("prepare_branched_sig_coef_cuda failed");
    std::vector<double> path = test_data<double>(batch_size * dimension * length);
    std::vector<double> derivs(batch_size * num_idx, 1.);

    double* d_path = nullptr;
    double* d_out = nullptr;
    double* d_coefs = nullptr;
    double* d_derivs = nullptr;
    cudaMalloc(&d_path, path.size() * sizeof(double));
    cudaMalloc(&d_out, path.size() * sizeof(double));
    cudaMalloc(&d_coefs, derivs.size() * sizeof(double));
    cudaMalloc(&d_derivs, derivs.size() * sizeof(double));
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), derivs.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (branched_sig_coef_cuda_d(
        d_path, d_coefs, tree_data.data(), tree_data.size(), batch_size,
        dimension, length, degree, false, false, 1., planar,
        nullptr, 0, 0, 0) != 0)
        throw std::runtime_error("branched_sig_coef_cuda_d failed");

    std::cout << "planar: " << planar << "\n";
    std::cout << "num_idx: " << num_idx << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "dimension: " << dimension << "\n";
    std::cout << "length: " << length << "\n";
    std::cout << "degree: " << degree << "\n";
    std::cout << "num_runs: " << num_runs << "\n";

    time_function(num_runs, branched_sig_coef_backprop_cuda_d,
        d_path, d_out, d_coefs, d_derivs, tree_data.data(), tree_data.size(),
        batch_size, dimension, length, degree, false, false, 1., planar,
        nullptr, 0, 0, 0);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_coefs);
    cudaFree(d_derivs);
}

void example_batch_sig_coef_cuda_d(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    int num_runs
) {
    print_header("Batch Sig Coef CUDA Double");

    uint64_t path_size = dimension * length * batch_size;
    std::vector<double> path = test_data<double>(path_size);

    std::vector<uint64_t> degrees(num_idx, degree);
    std::vector<uint64_t> multi_idx(num_idx * degree, 0);

    uint64_t out_size = batch_size * num_idx;

    // Allocate device memory
    double* d_path;
    double* d_out;
    uint64_t* d_multi_idx;
    uint64_t* d_degrees;
    cudaMalloc(&d_path, sizeof(double) * path_size);
    cudaMalloc(&d_out, sizeof(double) * out_size);
    cudaMalloc(&d_multi_idx, sizeof(uint64_t) * multi_idx.size());
    cudaMalloc(&d_degrees, sizeof(uint64_t) * degrees.size());

    // Copy to device
    cudaMemcpy(d_path, path.data(), sizeof(double) * path_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_multi_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_degrees, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

    time_function(num_runs, sig_coef_cuda_d, d_path, d_out, d_multi_idx, num_idx, d_degrees, batch_size, dimension, length, false);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_multi_idx);
    cudaFree(d_degrees);

    std::cout << "done\n";
}

void example_batch_sig_coef_backprop_cuda_d(
    uint64_t num_idx,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    uint64_t length,
    int num_runs
) {
    print_header("Batch Sig Coef Backprop CUDA Double");

    uint64_t path_size = dimension * length * batch_size;
    std::vector<double> path = test_data<double>(path_size);

    std::vector<uint64_t> degrees(num_idx, degree);
    std::vector<uint64_t> multi_idx(num_idx * degree, 0);

    uint64_t prefix_coef_size = degree * num_idx;
    std::vector<double> coefs(batch_size * prefix_coef_size, 1.);
    std::vector<double> derivs(batch_size * prefix_coef_size, 1.);

    uint64_t out_size = path_size;

    // Allocate device memory
    double* d_path;
    double* d_out;
    double* d_coefs;
    double* d_derivs;
    uint64_t* d_multi_idx;
    uint64_t* d_degrees;
    cudaMalloc(&d_path, sizeof(double) * path_size);
    cudaMalloc(&d_out, sizeof(double) * out_size);
    cudaMalloc(&d_coefs, sizeof(double) * coefs.size());
    cudaMalloc(&d_derivs, sizeof(double) * derivs.size());
    cudaMalloc(&d_multi_idx, sizeof(uint64_t) * multi_idx.size());
    cudaMalloc(&d_degrees, sizeof(uint64_t) * degrees.size());

    // Copy to device
    cudaMemcpy(d_path, path.data(), sizeof(double) * path_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coefs, coefs.data(), sizeof(double) * coefs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivs, derivs.data(), sizeof(double) * derivs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_multi_idx, multi_idx.data(), sizeof(uint64_t) * multi_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_degrees, degrees.data(), sizeof(uint64_t) * degrees.size(), cudaMemcpyHostToDevice);

    time_function(num_runs, sig_coef_backprop_cuda_d, d_path, d_out, d_coefs, d_derivs, d_multi_idx, num_idx, d_degrees, batch_size, dimension, length);

    cudaFree(d_path);
    cudaFree(d_out);
    cudaFree(d_coefs);
    cudaFree(d_derivs);
    cudaFree(d_multi_idx);
    cudaFree(d_degrees);

    std::cout << "done\n";
}

void example_batch_log_sig_combine_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Log Sig Combine CPU Double");

    uint64_t ls_len = log_sig_length(dimension, degree);
    uint64_t total = batch_size * ls_len;
    std::vector<double> ls1 = test_data<double>(total);
    std::vector<double> ls2 = test_data<double>(total);
    std::vector<double> out(total);

    time_function(num_runs, log_sig_combine_d, ls1.data(), ls2.data(), out.data(), batch_size, dimension, degree, n_jobs);

    std::cout << "done\n";
}

void example_batch_log_sig_combine_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int num_runs
) {
    print_header("Batch Log Sig Combine CUDA Double");

    uint64_t ls_len = log_sig_length(dimension, degree);
    uint64_t total = batch_size * ls_len;
    std::vector<double> ls1 = test_data<double>(total);
    std::vector<double> ls2 = test_data<double>(total);

    double* d_ls1;
    double* d_ls2;
    double* d_out;
    cudaMalloc(&d_ls1, sizeof(double) * total);
    cudaMalloc(&d_ls2, sizeof(double) * total);
    cudaMalloc(&d_out, sizeof(double) * total);

    cudaMemcpy(d_ls1, ls1.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls2, ls2.data(), sizeof(double) * total, cudaMemcpyHostToDevice);

    time_function(num_runs, log_sig_combine_cuda_d, d_ls1, d_ls2, d_out, batch_size, dimension, degree);

    cudaFree(d_ls1);
    cudaFree(d_ls2);
    cudaFree(d_out);

    std::cout << "done\n";
}

void example_batch_log_sig_combine_backprop_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int n_jobs,
    int num_runs
) {
    print_header("Batch Log Sig Combine Backprop CPU Double");

    uint64_t ls_len = log_sig_length(dimension, degree);
    uint64_t total = batch_size * ls_len;
    std::vector<double> ls1 = test_data<double>(total);
    std::vector<double> ls2 = test_data<double>(total);
    std::vector<double> d_out = test_data<double>(total);
    std::vector<double> d_ls1(total);
    std::vector<double> d_ls2(total);

    time_function(num_runs, log_sig_combine_backprop_d, d_out.data(), d_ls1.data(), d_ls2.data(), ls1.data(), ls2.data(), batch_size, dimension, degree, n_jobs);

    std::cout << "done\n";
}

void example_batch_log_sig_combine_backprop_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t degree,
    int num_runs
) {
    print_header("Batch Log Sig Combine Backprop CUDA Double");

    uint64_t ls_len = log_sig_length(dimension, degree);
    uint64_t total = batch_size * ls_len;
    std::vector<double> ls1 = test_data<double>(total);
    std::vector<double> ls2 = test_data<double>(total);
    std::vector<double> d_out = test_data<double>(total);

    double* d_ls1_dev;
    double* d_ls2_dev;
    double* d_ls1_out;
    double* d_ls2_out;
    double* d_out_dev;
    cudaMalloc(&d_ls1_dev, sizeof(double) * total);
    cudaMalloc(&d_ls2_dev, sizeof(double) * total);
    cudaMalloc(&d_ls1_out, sizeof(double) * total);
    cudaMalloc(&d_ls2_out, sizeof(double) * total);
    cudaMalloc(&d_out_dev, sizeof(double) * total);

    cudaMemcpy(d_ls1_dev, ls1.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ls2_dev, ls2.data(), sizeof(double) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_dev, d_out.data(), sizeof(double) * total, cudaMemcpyHostToDevice);

    time_function(num_runs, log_sig_combine_backprop_cuda_d, d_out_dev, d_ls1_out, d_ls2_out, d_ls1_dev, d_ls2_dev, batch_size, dimension, degree);

    cudaFree(d_ls1_dev);
    cudaFree(d_ls2_dev);
    cudaFree(d_ls1_out);
    cudaFree(d_ls2_out);
    cudaFree(d_out_dev);

    std::cout << "done\n";
}


void example_batch_branched_sig_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length,
    uint64_t max_nodes,
    int n_jobs,
    int num_runs,
    bool planar
) {
    print_header("Batch Branched Signature Double");

    prepare_branched_sig(dimension, max_nodes, false, planar);
    uint64_t out_len = branched_sig_length(dimension, max_nodes, planar);
    uint64_t out_size = out_len * batch_size;

    std::vector<double> path = test_data<double>(batch_size * dimension * length);
    std::vector<double> out(out_size, 0.);

    std::cout << "planar: " << planar << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "dimension: " << dimension << "\n";
    std::cout << "length: " << length << "\n";
    std::cout << "max_nodes: " << max_nodes << "\n";
    std::cout << "n_jobs: " << n_jobs << "\n";
    std::cout << "num_runs: " << num_runs << "\n";
    std::cout << "output length: " << out_len << "\n";

    time_function(num_runs, branched_sig_d, path.data(), out.data(), batch_size, dimension, length, max_nodes, n_jobs, false, false, 1., planar, true, nullptr, 0, 0, 0);

    std::cout << "done\n";
}

void example_batch_branched_sig_cuda_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length,
    uint64_t max_nodes,
    int num_runs,
    bool planar
) {
    print_header("Batch Branched Signature CUDA Double");

    prepare_branched_sig(dimension, max_nodes, false, planar);
    prepare_branched_sig_cuda(dimension, max_nodes, planar, false);
    uint64_t out_len = branched_sig_length(dimension, max_nodes, planar);
    uint64_t path_size = batch_size * dimension * length;
    uint64_t out_size = out_len * batch_size;

    std::vector<double> path = test_data<double>(path_size);

    double* d_path;
    double* d_out;
    cudaMalloc(&d_path, sizeof(double) * path_size);
    cudaMalloc(&d_out, sizeof(double) * out_size);

    cudaMemcpy(d_path, path.data(), sizeof(double) * path_size, cudaMemcpyHostToDevice);

    const uint64_t check_batch = 2;
    const uint64_t check_length = 5;
    const uint64_t check_path_size = check_batch * dimension * check_length;
    const uint64_t check_out_size = check_batch * out_len;
    std::vector<double> check_path = test_data<double>(check_path_size);
    std::vector<double> cpu_check(check_out_size, 0.);
    std::vector<double> cuda_check(check_out_size, 0.);
    int cpu_check_err = branched_sig_d(check_path.data(), cpu_check.data(), check_batch, dimension, check_length,
        max_nodes, 1, false, false, 1., planar, true, nullptr, 0, 0, 0);

    double* d_check_path;
    double* d_check_out;
    cudaMalloc(&d_check_path, sizeof(double) * check_path_size);
    cudaMalloc(&d_check_out, sizeof(double) * check_out_size);
    cudaMemcpy(d_check_path, check_path.data(), sizeof(double) * check_path_size, cudaMemcpyHostToDevice);
    int check_err = branched_sig_cuda_d(d_check_path, d_check_out, check_batch, dimension,
        check_length, max_nodes, false, false, 1., planar, true, nullptr, 0, 0, 0);
    cudaMemcpy(cuda_check.data(), d_check_out, sizeof(double) * check_out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_check_path);
    cudaFree(d_check_out);

    double max_diff = 0.;
    double max_ref = 0.;
    for (uint64_t i = 0; i < check_out_size; ++i) {
        max_diff = (std::max)(max_diff, std::abs(cpu_check[i] - cuda_check[i]));
        max_ref = (std::max)(max_ref, std::abs(cpu_check[i]));
    }
    if (cpu_check_err != 0 || check_err != 0 || max_diff > 1e-9 * (std::max)(1., max_ref))
        throw std::runtime_error("branched_sig_cuda_d correctness check failed");

    std::cout << "planar: " << planar << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "dimension: " << dimension << "\n";
    std::cout << "length: " << length << "\n";
    std::cout << "max_nodes: " << max_nodes << "\n";
    std::cout << "num_runs: " << num_runs << "\n";
    std::cout << "output length: " << out_len << "\n";

    time_function(num_runs, branched_sig_cuda_d, d_path, d_out, batch_size, dimension, length, max_nodes, false, false, 1., planar, true, nullptr, 0, 0, 0);

    cudaFree(d_path);
    cudaFree(d_out);

    std::cout << "done\n";
}

void example_batch_branched_log_sig_d(
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length,
    uint64_t max_nodes,
    int n_jobs
) {
    print_header("Batch Branched Log Signature Double");

    bool planar = false;
    bool scalar_term = true;
    prepare_branched_log_sig(dimension, max_nodes, 0, false, planar);
    prepare_branched_log_sig_cuda(dimension, max_nodes, planar, false);

    uint64_t bsig_len = branched_sig_length(dimension, max_nodes, planar);
    uint64_t total = batch_size * bsig_len;

    std::vector<double> path = test_data<double>(batch_size * dimension * length);
    std::vector<double> bsig(total, 0.);
    std::vector<double> cpu_out(total, 0.);
    std::vector<double> cuda_out(total, 0.);

    branched_sig_d(path.data(), bsig.data(), batch_size, dimension, length, max_nodes, n_jobs, false, false, 1., planar, scalar_term, nullptr, 0, 0, 0);
    branched_sig_to_log_sig_d(bsig.data(), cpu_out.data(), batch_size, dimension, max_nodes, 0, n_jobs, planar, scalar_term);

    double* d_bsig;
    double* d_out;
    cudaMalloc(&d_bsig, sizeof(double) * total);
    cudaMalloc(&d_out, sizeof(double) * total);
    cudaMemcpy(d_bsig, bsig.data(), sizeof(double) * total, cudaMemcpyHostToDevice);

    branched_sig_to_log_sig_cuda_d(d_bsig, d_out, batch_size, dimension, max_nodes, planar, scalar_term);
    cudaMemcpy(cuda_out.data(), d_out, sizeof(double) * total, cudaMemcpyDeviceToHost);

    cudaFree(d_bsig);
    cudaFree(d_out);

    std::cout << "bsig length: " << bsig_len << "\n";
    std::cout << "first CPU / CUDA entries:\n";
    for (uint64_t i = 0; i < total && i < 8; ++i) {
        std::cout << i << ": " << cpu_out[i] << " / " << cuda_out[i] << "\n";
    }

    std::cout << "done\n";
}
