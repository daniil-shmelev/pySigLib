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

#ifdef CUSIG_EXPORTS
#ifdef _WIN32
#define CUSIG_API __declspec(dllexport)
#else
#define CUSIG_API
#endif
#else
#define CUSIG_API __declspec(dllimport)
#endif

extern "C" {

	/** @defgroup transform_path_cuda_functions Transform path CUDA functions
	* @{
	*/

	/**
	* @brief Applies time-augmentation and/or the lead-lag transformation to a path of type float.
	*
	*
	* @param data_in Pointer to input path data (row-major), size = `length * dimension`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length_ * 2 - 1` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply the lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	CUSIG_API int transform_path_cuda_f(const float* data_in, float* data_out, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time = 1.) noexcept;
	/** @brief */
	CUSIG_API int transform_path_cuda_d(const double* data_in, double* data_out, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time = 1.) noexcept;
	/** @} */
	
	/** @defgroup batch_transform_path_cuda_functions Batch transform path CUDA functions
	* @{
	*/

	/**
	* @brief Applies time-augmentation and/or the lead-lag transformation to a batch of paths of type float.
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
	CUSIG_API int batch_transform_path_cuda_f(const float* data_in, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time = 1.) noexcept;
	/** @brief */
	CUSIG_API int batch_transform_path_cuda_d(const double* data_in, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time = 1.) noexcept;
	/** @} */
	
	/** @defgroup transform_path_backprop_cuda_functions Transform path backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagation through the transform_path_cuda function
	*
	*
	* @param derivs Pointer to derivatives with respect to transformed path (row-major), size = `transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length_ * 2 - 1` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `length * dimension`.
	* @param dimension Dimension of the original (pre-transformation) path.
	* @param length Length of the original (pre-transformation) path.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	CUSIG_API int transform_path_backprop_cuda_f(const float* derivs, float* data_out, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time = 1.) noexcept;
	/** @brief */
	CUSIG_API int transform_path_backprop_cuda_d(const double* derivs, double* data_out, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time = 1.) noexcept;
	/** @} */

	/** @defgroup batch_transform_path_backprop_cuda_functions Batch transform path backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagation through the batch_transform_path_cuda function
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
	CUSIG_API int batch_transform_path_backprop_cuda_f(const float* derivs, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time = 1.) noexcept;
	/** @brief */
	CUSIG_API int batch_transform_path_backprop_cuda_d(const double* derivs, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time = 1.) noexcept;
	/** @} */

	/** @defgroup sig_kernel_cuda_functions Sig kernel CUDA functions
	* @{
	*/

	/**
	* @brief Computes the signature kernel of two paths from their gram matrix.
	*
	* @param gram Pointer to gram matrix data (row-major), size = `(length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `return_grid ? (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1) : 1`.
	* @param dimension Dimension of the path.
	* @param length1 Length of the first path.
	* @param length2 Length of the second path.
	* @param dyadic_order_1 Dyadic refinement for the first path.
	* @param dyadic_order_2 Dyadic refinement for the second path.
	* @param return_grid Whether to return the entire PDE grid (default = false).
	* @return Status code (0 = success).
	*/
	CUSIG_API int sig_kernel_cuda_f(const float* gram, float* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */
	CUSIG_API int sig_kernel_cuda_d(const double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup batch_sig_kernel_cuda_functions Batch sig kernel CUDA functions
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
	CUSIG_API int batch_sig_kernel_cuda_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */
	CUSIG_API int batch_sig_kernel_cuda_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup sig_kernel_backprop_cuda_functions Sig kernel backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagation through sig_kernel.
	*
	* @param gram Pointer to gram matrix data (row-major), size = `(length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `(length1 - 1) * (length2 - 1)`.
	* @param derivs Pointer to input derivatives. If `return_grid` is false, points to a single scalar. If `return_grid` is true, points to a grid-sized array of size `(((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param k_grid Pointer to signature kernel PDE grid (row-major, precomputed), size = `(((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param dimension Dimension of the path.
	* @param length1 Length of the first path.
	* @param length2 Length of the second path.
	* @param dyadic_order_1 Dyadic refinement for the first path.
	* @param dyadic_order_2 Dyadic refinement for the second path.
	* @param return_grid If true, derivs is expected to be grid-sized; if false, derivs points to a single scalar.
	* @return Status code (0 = success).
	*/
	CUSIG_API int sig_kernel_backprop_cuda_f(const float* gram, float* out, const float* derivs, const float* k_grid, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */
	CUSIG_API int sig_kernel_backprop_cuda_d(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup batch_sig_kernel_backprop_cuda_functions Batch sig kernel backprop CUDA functions
	* @{
	*/

	/**
	* @brief Backpropagation through batch_sig_kernel.
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
	CUSIG_API int batch_sig_kernel_backprop_cuda_f(const float* gram, float* out, const float* derivs, const float* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */
	CUSIG_API int batch_sig_kernel_backprop_cuda_d(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */
}
