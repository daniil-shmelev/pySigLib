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

#pragma once

void example_batch_signature_d(
	uint64_t batch_size = 100,
	uint64_t dimension = 6,
	uint64_t length = 1000,
	uint64_t degree = 5,
	bool time_aug = false,
	bool lead_lag = false,
	bool horner = true,
	int n_jobs = -1,
	int num_runs = 50
);

void example_batch_signature_cuda_d(
	uint64_t batch_size = 100,
	uint64_t dimension = 6,
	uint64_t length = 1000,
	uint64_t degree = 5,
	bool time_aug = false,
	bool lead_lag = false,
	bool horner = true,
	int num_runs = 50
);

void example_batch_signature_kernel_f(
	uint64_t batch_size = 100,
	uint64_t dimension = 5,
	uint64_t length1 = 1000,
	uint64_t length2 = 1000,
	uint64_t dyadic_order_1 = 0,
	uint64_t dyadic_order_2 = 0,
	int n_jobs = -1,
	int num_runs = 50
);

void example_batch_signature_kernel_d(
	uint64_t batch_size = 100,
	uint64_t dimension = 5,
	uint64_t length1 = 1000,
	uint64_t length2 = 1000,
	uint64_t dyadic_order_1 = 0,
	uint64_t dyadic_order_2 = 0,
	int n_jobs = -1,
	int num_runs = 50
);

void example_batch_signature_kernel_cuda(
	uint64_t batch_size = 100,
	uint64_t dimension = 5,
	uint64_t length1 = 1000,
	uint64_t length2 = 1000,
	uint64_t dyadic_order_1 = 0,
	uint64_t dyadic_order_2 = 0,
	int num_runs = 50
);

void example_batch_signature_kernel_cuda_full_grid(
	uint64_t batch_size = 10,
	uint64_t dimension = 5,
	uint64_t length1 = 100,
	uint64_t length2 = 100,
	uint64_t dyadic_order_1 = 0,
	uint64_t dyadic_order_2 = 0,
	int num_runs = 50
);

void example_batch_sig_backprop_d(
	uint64_t batch_size = 100,
	uint64_t dimension = 5,
	uint64_t length = 10,
	uint64_t degree = 5,
	bool time_aug = false,
	bool lead_lag = false,
	int n_jobs = -1,
	int num_runs = 50
);

void example_batch_sig_backprop_cuda_d(
	uint64_t batch_size = 100,
	uint64_t dimension = 5,
	uint64_t length = 10,
	uint64_t degree = 5,
	bool time_aug = false,
	bool lead_lag = false,
	int num_runs = 50
);

void example_batch_sig_kernel_backprop(
	uint64_t batch_size = 32,
	uint64_t dimension = 5,
	uint64_t length1 = 1000,
	uint64_t length2 = 1000,
	uint64_t dyadic_order_1 = 0,
	uint64_t dyadic_order_2 = 0,
	int n_jobs = -1,
	int num_runs = 50
);

void example_batch_sig_kernel_backprop_cuda(
	uint64_t batch_size = 1,
	uint64_t dimension = 1,
	uint64_t length1 = 2,
	uint64_t length2 = 3,
	uint64_t dyadic_order_1 = 0,
	uint64_t dyadic_order_2 = 0,
	int num_runs = 1
);

void example_prepare_log_sig(
	uint64_t dimension = 5,
	uint64_t degree = 8,
	int method = 2,
	int num_runs = 10
);

void example_batch_sig_to_log_sig_d(
	uint64_t batch_size = 1000,
	uint64_t dimension = 5,
	uint64_t degree = 7,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 2,
	int n_jobs = -1,
	int num_runs = 50
);

void example_batch_sig_to_log_sig_cuda_d(
	uint64_t batch_size = 1000,
	uint64_t dimension = 5,
	uint64_t degree = 7,
	int method = 2,
	int num_runs = 50
);

void example_batch_sig_to_log_sig_backprop_d(
	uint64_t batch_size = 1000,
	uint64_t dimension = 5,
	uint64_t degree = 7,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 2,
	int n_jobs = -1,
	int num_runs = 50
);

void example_batch_sig_to_log_sig_backprop_cuda_d(
	uint64_t batch_size = 1000,
	uint64_t dimension = 5,
	uint64_t degree = 7,
	int method = 2,
	int num_runs = 50
);

void example_batch_sig_combine_cuda_d(
	uint64_t batch_size = 1000,
	uint64_t dimension = 8,
	uint64_t degree = 6,
	int num_runs = 50
);

void example_batch_sig_combine_backprop_d(
	uint64_t batch_size = 5200,
	uint64_t dimension = 6,
	uint64_t degree = 6,
	int n_jobs = 1,
	int num_runs = 50
);

void example_batch_sig_combine_backprop_cuda_d(
	uint64_t batch_size = 5200,
	uint64_t dimension = 6,
	uint64_t degree = 6,
	int num_runs = 50
);

void example_batch_sig_coef(
	uint64_t num_idx = 10,
	uint64_t batch_size = 1000,
	uint64_t dimension = 5,
	uint64_t degree = 5,
	uint64_t length = 1000,
	bool time_aug = false,
	bool lead_lag = false,
	double end_time = 1.,
	int n_jobs = 1,
	int num_runs = 50
);

void example_batch_sig_coef_backprop(
	uint64_t num_idx = 100,
	uint64_t batch_size = 1,
	uint64_t dimension = 5,
	uint64_t degree = 5,
	uint64_t length = 1000,
	bool time_aug = false,
	bool lead_lag = false,
	double end_time = 1.,
	int n_jobs = 1,
    int num_runs = 50
);

void example_batch_branched_sig_coef(
	uint64_t num_idx = 32,
	uint64_t batch_size = 32,
	uint64_t dimension = 3,
	uint64_t degree = 4,
	uint64_t length = 128,
	int n_jobs = 1,
	int num_runs = 50,
	bool planar = false
);

void example_batch_branched_sig_coef_backprop(
	uint64_t num_idx = 32,
	uint64_t batch_size = 32,
	uint64_t dimension = 3,
	uint64_t degree = 4,
	uint64_t length = 128,
	int n_jobs = 1,
	int num_runs = 50,
	bool planar = false
);

void example_batch_sig_coef_cuda_d(
	uint64_t num_idx = 10,
	uint64_t batch_size = 1000,
	uint64_t dimension = 5,
	uint64_t degree = 5,
	uint64_t length = 1000,
	int num_runs = 50
);

void example_batch_sig_coef_backprop_cuda_d(
	uint64_t num_idx = 100,
	uint64_t batch_size = 64,
	uint64_t dimension = 8,
	uint64_t degree = 5,
	uint64_t length = 100,
	int num_runs = 10
);

void example_batch_log_sig_combine_d(
	uint64_t batch_size = 120,
	uint64_t dimension = 5,
	uint64_t degree = 5,
	int n_jobs = -1,
	int num_runs = 20
);

void example_batch_log_sig_combine_cuda_d(
	uint64_t batch_size = 120,
	uint64_t dimension = 5,
	uint64_t degree = 5,
	int num_runs = 20
);

void example_batch_log_sig_combine_backprop_d(
	uint64_t batch_size = 120,
	uint64_t dimension = 5,
	uint64_t degree = 5,
	int n_jobs = -1,
	int num_runs = 20
);

void example_batch_log_sig_combine_backprop_cuda_d(
	uint64_t batch_size = 120,
	uint64_t dimension = 5,
	uint64_t degree = 5,
	int num_runs = 20
);

void example_batch_branched_sig_d(
	uint64_t batch_size = 100,
	uint64_t dimension = 3,
	uint64_t length = 100,
	uint64_t max_nodes = 4,
	int n_jobs = -1,
	int num_runs = 50,
	bool planar = false
);

void example_batch_branched_sig_cuda_d(
	uint64_t batch_size = 100,
	uint64_t dimension = 3,
	uint64_t length = 100,
	uint64_t max_nodes = 4,
	int num_runs = 50,
	bool planar = false
);

void example_batch_branched_log_sig_d(
	uint64_t batch_size = 2,
	uint64_t dimension = 3,
	uint64_t length = 8,
	uint64_t max_nodes = 4,
	int n_jobs = 1
);
