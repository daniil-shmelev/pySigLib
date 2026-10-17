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

// Included inside each target namespace.

template<std::floating_point T>
void log_sig_combine_(const T* log_sig1, const T* log_sig2, T* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template<std::floating_point T>
void batch_log_sig_from_path_(const T* path, T* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs);

template<std::floating_point T>
void batch_log_sig_from_path_backprop_(const T* d_out, T* d_path, const T* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs);

template<std::floating_point T>
void log_sig_combine_backprop_(const T* d_out, T* d_ls1, T* d_ls2, const T* ls1, const T* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template<std::floating_point T>
void log_sig_join_(const T* log_sig, const T* displacement, T* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template<std::floating_point T>
void log_sig_join_backprop_(const T* d_out, T* d_logsig, T* d_displacement, const T* log_sig, const T* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template<std::floating_point T>
void branched_sig_to_log_sig_(const T* bsig, T* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, BranchedLogSigMethod method, int n_jobs, bool planar, bool scalar_term);

template<std::floating_point T>
void branched_sig_to_log_sig_backprop_(const T* bsig, const T* derivs, T* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, BranchedLogSigMethod method, int n_jobs, bool planar, bool scalar_term);

template<std::floating_point T>
void branched_log_sig_from_path_(const T* path, T* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs);

template<std::floating_point T>
void batch_branched_log_sig_join_(const T* logsig, const T* displacement, T* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool prepend, int n_jobs);

template<std::floating_point T>
void batch_branched_log_sig_join_backprop_(const T* derivs, T* d_logsig, T* d_displacement, const T* logsig, const T* displacement, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool prepend, int n_jobs);

template<std::floating_point T>
void branched_log_sig_from_path_backprop_(const T* derivs, T* path_derivs, const T* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes, int n_jobs);

template<std::floating_point T>
void branched_sig_kernel_poly_(const T* gram, T* out, T* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs);

template<std::floating_point T>
void branched_sig_kernel_poly_backprop_(const T* gram, T* grad, const T* derivs, const T* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs);

template<std::floating_point T>
void branched_sig_coef_(const T* path, T* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, T end_time, bool planar, const T* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template<std::floating_point T>
void branched_sig_coef_backprop_(const T* path, T* out, const T* coefs, const T* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, T end_time, bool planar, const T* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template<std::floating_point T>
void branched_sig_kernel_(const T* gram, T* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs);

template<std::floating_point T>
void batch_branched_sig_kernel_backprop_(const T* gram, T* out, const T* derivs, const T* k_stack, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs);

template<std::floating_point T>
void branched_signature_(const T* path, T* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, T end_time, bool planar, bool scalar_term, const T* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template<std::floating_point T>
void branched_sig_combine_(const T* bsig1, const T* bsig2, T* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term);

template<std::floating_point T>
void branched_sig_combine_backprop_(const T* bsig1, const T* bsig2, const T* derivs, T* out1, T* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term);

template<std::floating_point T>
void branched_sig_backprop_(const T* path, T* out, const T* bsig_derivs, const T* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, T end_time, bool planar, bool scalar_term, const T* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template<std::floating_point T>
void sig_to_log_sig_(const T* sig, T* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template<std::floating_point T>
void sig_to_log_sig_backprop_(const T* sig, T* out, const T* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template<std::floating_point T>
void logsig_to_sig_(const T* log_sig, T* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template<std::floating_point T>
void logsig_to_sig_backprop_(const T* log_sig, T* out, const T* sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template<std::floating_point T>
void batch_transform_path_(const T* data_in, T* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time, int n_jobs);

template<std::floating_point T>
void batch_transform_path_backprop_(const T* derivs, T* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time, int n_jobs);

template<std::floating_point T>
void sig_kernel_poly_(const T* gram, T* out, T* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid, int n_jobs);

template<std::floating_point T>
void sig_kernel_poly_backprop_(const T* gram, T* gram_derivs, const T* output_derivs, const T* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid, int n_jobs);

template<std::floating_point T>
void sig_coef_(const T* path, T* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time, bool prefixes, int n_jobs);

template<std::floating_point T>
void sig_coef_backprop_(const T* path, T* out, const T* coefs, T* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time, int n_jobs);

template<std::floating_point T>
void batch_sig_combine_(const T* sig1, const T* sig2, T* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term, int n_jobs);

template<std::floating_point T>
void batch_sig_combine_backprop_(const T* sig_combined_deriv, T* sig1_deriv, T* sig2_deriv, const T* sig1, const T* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term, int n_jobs);

template<std::floating_point T>
void batch_linear_sig_(const T* displacement, T* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term, int n_jobs);

template<std::floating_point T>
void sig_join_(const T* sig, const T* displacement, T* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs);

template<std::floating_point T>
void batch_sig_join_backprop_(const T* d_out, T* d_sig, T* d_displacement, const T* sig, const T* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs);

template<std::floating_point T>
void sig_kernel_(const T* gram, T* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs);

template<std::floating_point T>
void batch_sig_kernel_backprop_(const T* gram, T* out, const T* derivs, const T* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid, int n_jobs);

template<std::floating_point T>
void signature_(const T* path, T* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, T end_time, bool horner, bool scalar_term, int n_jobs, const T* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template<std::floating_point T>
void sig_backprop_(const T* path, T* out, const T* sig_derivs, const T* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, T end_time, bool scalar_term, int n_jobs, const T* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);
