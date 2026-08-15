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

#if defined(CUSIG_EXPORTS)
	#if defined(_MSC_VER)
		#define CUSIG_API __declspec(dllexport)
	#elif defined(__GNUC__)
		#define CUSIG_API __attribute__((visibility("default")))
	#else
		#define CUSIG_API
	#endif
#else
	#if defined(_MSC_VER)
		#define CUSIG_API __declspec(dllimport)
	#elif defined(__GNUC__)
		#define CUSIG_API
	#else
		#define CUSIG_API
	#endif
#endif

extern "C" {

	
	/** @defgroup transform_path_cuda_functions Transform path CUDA functions
	* @{
	*/

	/**
	* @brief Applies time-augmentation and/or the lead-lag transformation to a batch of paths.
	*
	*
	* @param data_in Pointer to input path data (row-major), size = `batch_size * length * dimension`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `batch_size * transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length_ * 2 - 1` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the paths.
	* @param length Length of the paths.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply the lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int transform_path_cuda_f(const float* data_in, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time = 1.) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int transform_path_cuda_d(const double* data_in, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time = 1.) noexcept;
	/** @} */
	

	/** @defgroup transform_path_backprop_cuda_functions Transform path backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagation through the transform_path_cuda function
	*
	*
	* @param derivs Pointer to derivatives with respect to transformed path (row-major), size = `batch_size * transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length_ * 2 - 1` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `batch_size * length * dimension`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the original (pre-transformation) paths.
	* @param length Length of the original (pre-transformation) paths.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int transform_path_backprop_cuda_f(const float* derivs, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time = 1.) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int transform_path_backprop_cuda_d(const double* derivs, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time = 1.) noexcept;
	/** @} */

	/** @defgroup sig_kernel_cuda_functions Sig kernel CUDA functions
	* @{
	*/

	/**
	* @brief Computes signature kernels of a batch of paths from their gram matrices.
	*
	* @param gram Pointer to batch gram matrix data (row-major), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `batch_size * (return_grid ? (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1) : 1)`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the path.
	* @param length1 Length of the first path.
	* @param length2 Length of the second path.
	* @param dyadic_order_1 Dyadic refinement for the first path.
	* @param dyadic_order_2 Dyadic refinement for the second path.
	* @param return_grid Whether to return the entire PDE grid (default = false).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_kernel_cuda_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_kernel_cuda_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup sig_kernel_poly_cuda_functions Polynomial sig kernel CUDA functions
	* @{
	*/

	/**
	* @brief Computes polynomial signature kernels from batch gram matrices on the GPU.
	*
	* @param gram Pointer to batch gram matrix data on the device, size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to device output, size = `batch_size * (return_grid ? length1 * length2 : 1)`.
	* @param state Optional device output for incoming cell coefficients, size = `2 * batch_size * (length1 - 1) * (length2 - 1) * (order + 1)`. May be null.
	* @param batch_size Number of path pairs.
	* @param dimension Path dimension, retained for API parity with sig_kernel_cuda_f.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param order Highest retained polynomial degree, between 2 and 64.
	* @param return_grid Whether to return the full original-vertex grid.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_kernel_poly_cuda_f(const float* gram, float* out, float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false) noexcept;
	/** @brief Double-precision variant of sig_kernel_poly_cuda_f. */
	[[nodiscard]] CUSIG_API int sig_kernel_poly_cuda_d(const double* gram, double* out, double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false) noexcept;

	/**
	* @brief Backpropagates through polynomial signature kernels on the GPU.
	*
	* @param gram Pointer to batch gram matrix data on the device, size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param gram_derivs Pointer to device gram derivatives, with the same size as `gram`.
	* @param output_derivs Pointer to device output derivatives, size = `batch_size * (return_grid ? length1 * length2 : 1)`.
	* @param state Optional incoming-coefficient state produced by sig_kernel_poly_cuda_f. If null, the state is regenerated.
	* @param batch_size Number of path pairs.
	* @param dimension Path dimension, retained for API parity.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param order Highest retained polynomial degree, between 2 and 64.
	* @param return_grid Whether `output_derivs` contains full-grid derivatives.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_kernel_poly_backprop_cuda_f(const float* gram, float* gram_derivs, const float* output_derivs, const float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false) noexcept;
	/** @brief Double-precision variant of sig_kernel_poly_backprop_cuda_f. */
	[[nodiscard]] CUSIG_API int sig_kernel_poly_backprop_cuda_d(const double* gram, double* gram_derivs, const double* output_derivs, const double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup branched_sig_kernel_cuda_functions Branched sig kernel CUDA functions
	* @{
	*/

	/**
	* @brief Computes the depth-recursive branched signature kernel from batch gram matrices on the GPU.
	*
	* Computes the non-planar BCK branched signature kernel using the Chevyrev-Oberhauser recursion.
	* The core consumes precomputed static-kernel increments rather than path coordinates.
	*
	* @param gram Pointer to batch gram matrix data (row-major, on device), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer (row-major, preallocated, on device), size = `batch_size * (return_grid ? (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1) : 1)`.
	* @param batch_size Batch size of the path pairs.
	* @param dimension Dimension of the original paths. The kernel core uses `gram`, so this is retained for API parity.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param depth Truncation depth of the branched kernel recursion. If zero, the output is filled with ones.
	* @param dyadic_order_1 Dyadic refinement for the first paths.
	* @param dyadic_order_2 Dyadic refinement for the second paths.
	* @param return_grid If true, returns the final-depth grid; otherwise returns the endpoint scalar per batch item.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int branched_sig_kernel_cuda_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief Double-precision variant of branched_sig_kernel_cuda_f. */
	[[nodiscard]] CUSIG_API int branched_sig_kernel_cuda_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup sig_kernel_backprop_cuda_functions Sig kernel backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagation through sig_kernel.
	*
	* @param gram Pointer to batch gram matrix data (row-major), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param derivs Pointer to input derivatives. If `return_grid` is false, size = `batch_size`. If `return_grid` is true, size = `batch_size * (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param k_grid Pointer to batch of signature kernel PDE grids (row-major, precomputed), size = `batch_size * (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the paths.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param dyadic_order_1 Dyadic refinement for the first paths.
	* @param dyadic_order_2 Dyadic refinement for the second paths.
	* @param return_grid If true, derivs is expected to be grid-sized per batch element; if false, derivs has one scalar per batch element.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_kernel_backprop_cuda_f(const float* gram, float* out, const float* derivs, const float* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_kernel_backprop_cuda_d(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup branched_sig_kernel_backprop_cuda_functions Branched sig kernel backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagates through branched_sig_kernel_cuda_f with respect to the gram matrix on the GPU.
	*
	* Computes derivatives with respect to the precomputed static-kernel increments used by the branched
	* signature kernel. Path-coordinate derivatives are handled by the Python static-kernel wrappers.
	*
	* @param gram Pointer to batch gram matrix data (row-major, on device), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer for dF/d(gram) (row-major, preallocated, on device), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param derivs Pointer to input derivatives (on device). If `return_grid` is false, size = `batch_size`. If `return_grid` is true, size = `batch_size * (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param k_stack Optional pointer to precomputed forward grids K_0, ..., K_depth (row-major, on device). May be null. If supplied, size = `batch_size * (depth + 1) * (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param batch_size Batch size of the path pairs.
	* @param dimension Dimension of the original paths. The kernel core uses `gram`, so this is retained for API parity.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param depth Truncation depth of the branched kernel recursion. If zero, derivatives with respect to `gram` are zero.
	* @param dyadic_order_1 Dyadic refinement for the first paths.
	* @param dyadic_order_2 Dyadic refinement for the second paths.
	* @param return_grid If true, `derivs` is expected to be grid-sized per batch item; otherwise it has one scalar per batch item.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int branched_sig_kernel_backprop_cuda_f(const float* gram, float* out, const float* derivs, const float* k_stack, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief Double-precision variant of branched_sig_kernel_backprop_cuda_f. */
	[[nodiscard]] CUSIG_API int branched_sig_kernel_backprop_cuda_d(const double* gram, double* out, const double* derivs, const double* k_stack, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup signature_cuda_functions Signature CUDA functions
	* @{
	*/

	/**
	* @brief Computes the truncated signatures of a batch of paths on the GPU.
	*
	* @param path Pointer to input batch path data (row-major, on device), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (row-major, preallocated, on device), size = `batch_size * sig_length(transformed_dimension, degree)`,
	*			  where `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param batch_size Batch size.
	* @param dimension Dimension of the paths.
	* @param length Length of the paths.
	* @param degree Truncation degree of the signature.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply the lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param horner Whether to use the Horner algorithm (default = true).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int signature_cuda_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool horner = true, bool scalar_term = true, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int signature_cuda_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true, bool scalar_term = true, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup sig_backprop_cuda_functions Signature backpropagation CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagates through the batch signature computation on the GPU (float).
	*
	* @param path Pointer to input batch path data (row-major, on device), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (row-major, preallocated, on device), same size as path.
	* @param sig_derivs Pointer to dF/d(sig) (on device), size = `batch_size * sig_length(transformed_dimension, degree)`.
	* @param sig Pointer to forward signatures (on device), same size as sig_derivs.
	* @param batch_size Batch size.
	* @param dimension Dimension of the paths.
	* @param length Length of the paths.
	* @param degree Truncation degree of the signature.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_backprop_cuda_f(const float* path, float* out, const float* sig_derivs, const float* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool scalar_term = true, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_backprop_cuda_d(const double* path, double* out, const double* sig_derivs, const double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool scalar_term = true, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup sig_combine_cuda_functions Signature combine CUDA functions
	* @{
	*/

	/**
	* @brief Combines batches of truncated signatures on the GPU using Chen's identity.
	*
	* @param sig1 Pointer to batch of first signatures (row-major, on device), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig2 Pointer to batch of second signatures (row-major, on device), same size as sig1.
	* @param out Pointer to output buffer (row-major, preallocated, on device), same size as sig1.
	* @param batch_size Batch size.
	* @param dimension Dimension of the underlying path space.
	* @param degree Truncation degree of the signatures.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_combine_cuda_f(const float* sig1, const float* sig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_combine_cuda_d(const double* sig1, const double* sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup sig_combine_backprop_cuda_functions Signature combine backprop CUDA functions
	* @{
	*/

	/**
	* @brief Batch backpropagation through sig_combine on the GPU.
	*
	* @param sig_combined_deriv Pointer to input derivatives (on device), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig1_deriv Pointer to output sig1 derivatives (on device, preallocated).
	* @param sig2_deriv Pointer to output sig2 derivatives (on device, preallocated).
	* @param sig1 Pointer to first signatures (on device).
	* @param sig2 Pointer to second signatures (on device).
	* @param batch_size Batch size.
	* @param dimension Dimension of the underlying path space.
	* @param degree Truncation degree of the signatures.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_combine_backprop_cuda_f(const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv, const float* sig1, const float* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_combine_backprop_cuda_d(const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup sig_to_log_sig_cuda_functions Sig to log sig CUDA functions
	* @{
	*/

	/**
	* @brief Converts a batch of signatures to log signatures on the GPU using the specified method.
	*
	* @param sig Pointer to batch of input signatures (on device), size = `batch_size * sig_length(dimension, degree)`.
	* @param out Pointer to output buffer (on device, preallocated).
	* @param batch_size Batch size.
	* @param dimension Dimension of the underlying path space.
	* @param degree Truncation degree of the signature.
	* @param method Method for log signature computation (0, 1, or 2).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_to_log_sig_cuda_f(const float* sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_to_log_sig_cuda_d(const double* sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup sig_to_log_sig_backprop_cuda_functions Sig to log sig backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagates through the sig_to_log_sig_cuda function.
	*
	* @param sig Pointer to batch of input signatures (on device), size = `batch_size * sig_length(dimension, degree)`.
	* @param out Pointer to output buffer (on device, preallocated), size = `batch_size * sig_length(dimension, degree)`.
	* @param log_sig_derivs Pointer to dF/d(log_sig) (on device).
	* @param batch_size Batch size.
	* @param dimension Dimension of the underlying path space.
	* @param degree Truncation degree of the signature.
	* @param method Method for log signature computation (0, 1, or 2).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_to_log_sig_backprop_cuda_f(const float* sig, float* out, const float* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_to_log_sig_backprop_cuda_d(const double* sig, double* out, const double* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup prepare_log_sig_cuda_functions Prepare log sig CUDA functions
	* @{
	*/

	/**
	* @brief Prepares GPU-side data structures needed for log signature methods 1, 2, and 3.
	*
	* @param dimension Dimension of the underlying path space.
	* @param degree Truncation degree.
	* @param method Method (1, 2, or 3).
	* @param use_disk If true, check the shared disk cache before computing, and save to disk if not found.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int prepare_log_sig_cuda(uint64_t dimension, uint64_t degree, int method, bool use_disk = false) noexcept;
	/** @} */

	/** @defgroup sig_coef_cuda_functions Sig coef CUDA functions
	* @{
	*/

	/**
	* @brief Computes signature coefficients for a batch of paths on the GPU.
	*
	* @param path Pointer to input batch path data (row-major, on device), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (on device, preallocated).
	* @param multi_idx Pointer to flattened multi-indices (on device).
	* @param num_multi_idx Number of multi-indices (words).
	* @param degrees Pointer to degree of each multi-index (on device), size = `num_multi_idx`.
	* @param batch_size Batch size.
	* @param dimension Dimension of the paths.
	* @param length Length of the paths.
	* @param prefixes If true, output all prefix coefficients; if false, output only the final coefficient.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_coef_cuda_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_coef_cuda_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes) noexcept;
	/** @} */

	/** @defgroup sig_coef_backprop_cuda_functions Sig coef backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagates through sig_coef_cuda on the GPU.
	*
	* @param path Pointer to input batch path data (row-major, on device), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (on device, preallocated), size = `batch_size * length * dimension`.
	* @param coefs Pointer to prefix coefficients from forward pass (on device).
	* @param derivs Pointer to dF/d(coefs) (on device), same size as coefs.
	* @param multi_idx Pointer to flattened multi-indices (on device).
	* @param num_multi_idx Number of multi-indices (words).
	* @param degrees Pointer to degree of each multi-index (on device), size = `num_multi_idx`.
	* @param batch_size Batch size.
	* @param dimension Dimension of the paths.
	* @param length Length of the paths.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int sig_coef_backprop_cuda_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int sig_coef_backprop_cuda_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length) noexcept;
	/** @} */

	/** @defgroup log_sig_combine_cuda_functions Log sig combine CUDA functions
	* @{
	*/

	/**
	* @brief Combines batches of truncated log-signatures on the GPU using the BCH formula.
	*
	* @param log_sig1 Pointer to batch of first log-signatures (on device), size = `batch_size * log_sig_length(dimension, degree)`.
	* @param log_sig2 Pointer to batch of second log-signatures (on device), same size as log_sig1.
	* @param out Pointer to output buffer (on device, preallocated), same size as log_sig1.
	* @param batch_size Batch size.
	* @param dimension Dimension of the underlying path space.
	* @param degree Truncation degree of the log-signatures.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int log_sig_combine_cuda_f(const float* log_sig1, const float* log_sig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	/** @brief */
	[[nodiscard]] CUSIG_API int log_sig_combine_cuda_d(const double* log_sig1, const double* log_sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	/** @} */

	/** @defgroup log_sig_combine_backprop_cuda_functions Log sig combine backprop CUDA functions
	* @{
	*/

	[[nodiscard]] CUSIG_API int log_sig_combine_backprop_cuda_f(const float* d_out, float* d_ls1, float* d_ls2,
		const float* ls1, const float* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	[[nodiscard]] CUSIG_API int log_sig_combine_backprop_cuda_d(const double* d_out, double* d_ls1, double* d_ls2,
		const double* ls1, const double* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	/** @} */

	/** @defgroup logsig_to_sig_cuda_functions Log-signature to signature (tensor exp) CUDA functions
	* @{
	*/
	[[nodiscard]] CUSIG_API int logsig_to_sig_cuda_f(const float* log_sig, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int logsig_to_sig_cuda_d(const double* log_sig, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;

	[[nodiscard]] CUSIG_API int logsig_to_sig_backprop_cuda_f(const float* log_sig, float* d_logsig, const float* d_sig,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int logsig_to_sig_backprop_cuda_d(const double* log_sig, double* d_logsig, const double* d_sig,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int method, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup log_sig_from_path_cuda_functions Log-signature from path CUDA functions
	* @{
	*/
	[[nodiscard]] CUSIG_API int log_sig_from_path_cuda_f(const float* path, float* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree) noexcept;
	[[nodiscard]] CUSIG_API int log_sig_from_path_cuda_d(const double* path, double* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree) noexcept;

	[[nodiscard]] CUSIG_API int log_sig_from_path_backprop_cuda_f(const float* d_out, float* d_path, const float* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree) noexcept;
	[[nodiscard]] CUSIG_API int log_sig_from_path_backprop_cuda_d(const double* d_out, double* d_path, const double* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree) noexcept;
	/** @} */

	/** @defgroup clear_cache_cuda_functions Clear cache CUDA functions
	* @{
	*/

	/**
	* @brief Clears all GPU-side cached data for log signature computation.
	*
	* @param use_disk If true, also delete the shared disk cache.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int clear_cache_cuda(bool use_disk = false) noexcept;
	/** @} */

	/** @defgroup set_cache_dir_cuda_functions Set cache dir CUDA functions
	* @{
	*/

	/**
	* @brief Sets the disk cache directory for CUDA log signature data.
	*
	* This should match the directory used by cpsig, as both libraries
	* share the same disk cache format.
	*
	* @param dir Path to the cache directory.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int set_cache_dir_cuda(const char* dir) noexcept;
	/** @} */

	/** @defgroup branched_sig_coef_cuda_functions Branched signature coefficient CUDA functions
	* @{
	*/

	/**
	* @brief Prepares a sparse coefficient cache on the active CUDA device.
	*
	* @param tree_data Serialized requested trees and ordered forests in host memory.
	* @param tree_data_len Number of uint64_t values in `tree_data`.
	* @param data_dimension Dimension of the unaugmented path.
	* @param dimension Dimension after time or lead-lag augmentation.
	* @param max_nodes Maximum number of nodes in the requested basis elements.
	* @param planar Whether the data uses the planar MKW basis rather than the non-planar BCK basis.
	* @param use_disk Whether to persist and load the shared host-side sparse plan.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int prepare_branched_sig_coef_cuda(const uint64_t* tree_data, uint64_t tree_data_len, uint64_t data_dimension, uint64_t dimension, uint64_t max_nodes, bool planar = false, bool use_disk = false) noexcept;

	/**
	* @brief Computes selected branched signature coefficients on the GPU.
	*
	* Call prepare_branched_sig_coef_cuda with matching arguments on the active
	* CUDA device before this function.
	*
	* @param path Input paths on device, size = `batch_size * length * dimension`.
	* @param out Output coefficients on device, size = `batch_size * tree_data[0]`.
	* @param tree_data Serialized requested trees passed to preparation.
	* @param tree_data_len Number of uint64_t values in `tree_data`.
	* @param batch_size Number of paths.
	* @param dimension Dimension of the input paths.
	* @param length Number of points in each input path.
	* @param max_nodes Maximum number of nodes in the requested basis elements.
	* @param time_aug Whether to prepend a time channel.
	* @param lead_lag Whether to apply the lead-lag transform.
	* @param end_time Final value of the time channel.
	* @param planar Whether the data uses the planar MKW basis rather than the non-planar BCK basis.
	* @param correction Optional log-signature correction data on device.
	* @param correction_len Number of correction values per segment.
	* @param correction_batch_stride Correction stride between paths.
	* @param correction_segment_stride Correction stride between segments.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int branched_sig_coef_cuda_f(const float* path, float* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @brief Double-precision variant of branched_sig_coef_cuda_f. */
	[[nodiscard]] CUSIG_API int branched_sig_coef_cuda_d(const double* path, double* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup branched_sig_coef_backprop_cuda_functions Branched signature coefficient backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagates selected branched signature coefficients to path coordinates on the GPU.
	*
	* @param path Input paths on device, size = `batch_size * length * dimension`.
	* @param out Output path derivatives on device, with the same size as `path`.
	* @param coefs Coefficients returned by branched_sig_coef_cuda_f.
	* @param derivs Derivatives with respect to `coefs`.
	* @param tree_data Serialized requested trees passed to the forward call.
	* @param tree_data_len Number of uint64_t values in `tree_data`.
	* @param batch_size Number of paths.
	* @param dimension Dimension of the input paths.
	* @param length Number of points in each input path.
	* @param max_nodes Maximum number of nodes in the requested basis elements.
	* @param time_aug Whether the forward pass prepended a time channel.
	* @param lead_lag Whether the forward pass applied the lead-lag transform.
	* @param end_time Final value of the time channel.
	* @param planar Whether the data uses the planar MKW basis rather than the non-planar BCK basis.
	* @param correction Optional log-signature correction data on device.
	* @param correction_len Number of correction values per segment.
	* @param correction_batch_stride Correction stride between paths.
	* @param correction_segment_stride Correction stride between segments.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int branched_sig_coef_backprop_cuda_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @brief Double-precision variant of branched_sig_coef_backprop_cuda_f. */
	[[nodiscard]] CUSIG_API int branched_sig_coef_backprop_cuda_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup branched_sig_cuda_functions Branched signature CUDA functions
	* @{
	*/

	/**
	* @brief Prepares the branched-signature cache on the active CUDA device.
	*
	* @param dimension Dimension after time or lead-lag augmentation.
	* @param max_nodes Maximum number of nodes in the branched-signature basis.
	* @param planar Whether to prepare the planar MKW basis rather than the non-planar BCK basis.
	* @param use_disk Whether to persist and load the shared host-side cache tables.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CUSIG_API int prepare_branched_sig_cuda(uint64_t dimension, uint64_t max_nodes, bool planar = false, bool use_disk = false) noexcept;

	[[nodiscard]] CUSIG_API int branched_sig_cuda_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, bool scalar_term = true, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	[[nodiscard]] CUSIG_API int branched_sig_cuda_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, bool scalar_term = true, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;

	[[nodiscard]] CUSIG_API int branched_sig_combine_cuda_f(const float* bsig1, const float* bsig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int branched_sig_combine_cuda_d(const double* bsig1, const double* bsig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar = false, bool scalar_term = true) noexcept;

	[[nodiscard]] CUSIG_API int branched_sig_combine_backprop_cuda_f(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int branched_sig_combine_backprop_cuda_d(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar = false, bool scalar_term = true) noexcept;

	[[nodiscard]] CUSIG_API int branched_sig_backprop_cuda_f(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, bool scalar_term = true, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	[[nodiscard]] CUSIG_API int branched_sig_backprop_cuda_d(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, bool scalar_term = true, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup branched_log_sig_cuda_functions Branched log signature CUDA functions
	* @{
	*/
	/**
	* @brief Prepares the branched-signature and derived branched-log caches on the active CUDA device.
	* @param method Branched log-signature method in the range 0 to 3. Methods 1 to 3 require planar=true. Method 3 supports max_nodes up to 12.
	*/
	[[nodiscard]] CUSIG_API int prepare_branched_log_sig_cuda(uint64_t dimension, uint64_t max_nodes, int method, bool planar = false, bool use_disk = false) noexcept;

	[[nodiscard]] CUSIG_API int branched_sig_to_log_sig_cuda_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int branched_sig_to_log_sig_cuda_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar = false, bool scalar_term = true) noexcept;

	[[nodiscard]] CUSIG_API int branched_sig_to_log_sig_backprop_cuda_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int branched_sig_to_log_sig_backprop_cuda_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int method, bool planar = false, bool scalar_term = true) noexcept;

	/** @brief Computes planar method 3 directly from a path. */
	[[nodiscard]] CUSIG_API int branched_log_sig_from_path_cuda_f(const float* path, float* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes) noexcept;
	[[nodiscard]] CUSIG_API int branched_log_sig_from_path_cuda_d(const double* path, double* out, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes) noexcept;

	[[nodiscard]] CUSIG_API int branched_log_sig_from_path_backprop_cuda_f(const float* derivs, float* path_derivs, const float* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes) noexcept;
	[[nodiscard]] CUSIG_API int branched_log_sig_from_path_backprop_cuda_d(const double* derivs, double* path_derivs, const double* path, uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t max_nodes) noexcept;
	/** @} */

	/** @defgroup linear_sig_cuda_functions Linear sig CUDA functions
	* @{
	*/
	[[nodiscard]] CUSIG_API int linear_sig_cuda_f(const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int linear_sig_cuda_d(const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup sig_join_cuda_functions Sig join CUDA functions
	* @{
	*/
	[[nodiscard]] CUSIG_API int sig_join_cuda_f(const float* sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int sig_join_cuda_d(const double* sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup sig_join_backprop_cuda_functions Sig join backprop CUDA functions
	* @{
	*/
	[[nodiscard]] CUSIG_API int sig_join_backprop_cuda_f(const float* d_out, float* d_sig, float* d_displacement, const float* sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CUSIG_API int sig_join_backprop_cuda_d(const double* d_out, double* d_sig, double* d_displacement, const double* sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true) noexcept;
	/** @} */

	/** @defgroup log_sig_join_cuda_functions Log sig join CUDA functions
	* @{
	*/
	[[nodiscard]] CUSIG_API int log_sig_join_cuda_f(const float* log_sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	[[nodiscard]] CUSIG_API int log_sig_join_cuda_d(const double* log_sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	/** @} */

	/** @defgroup log_sig_join_backprop_cuda_functions Log sig join backprop CUDA functions
	* @{
	*/
	[[nodiscard]] CUSIG_API int log_sig_join_backprop_cuda_f(const float* d_out, float* d_logsig, float* d_displacement, const float* log_sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	[[nodiscard]] CUSIG_API int log_sig_join_backprop_cuda_d(const double* d_out, double* d_logsig, double* d_displacement, const double* log_sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree) noexcept;
	/** @} */

	/** @brief Release all cached device allocations and the CUDA stream pool. Synchronizes the device first. Idempotent. */
	CUSIG_API void cusig_shutdown() noexcept;
}
