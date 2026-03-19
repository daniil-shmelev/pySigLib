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
#include "cppch.h"

#if defined(CPSIG_EXPORTS)
	#if defined (_MSC_VER)
		#define CPSIG_API __declspec(dllexport)
	#elif defined (__GNUC__)
		#define CPSIG_API __attribute__((visibility("default")))
	#else
		#define CPSIG_API
	#endif
#else
	#if defined (_MSC_VER)
		#define CPSIG_API __declspec(dllimport)
	#elif defined (__GNUC__)
		#define CPSIG_API 
	#else
		#define CPSIG_API 
	#endif
#endif


extern "C" {

	/** @defgroup transform_path_functions Transform path functions
	* @{
	*/

	/**
	* @brief Applies time-augmentation and/or the lead-lag transformation to a path of type float.
	*
	*
	* @param data_in Pointer to input path data (row-major), size = `length * dimension`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length * 2 - 1 : length` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply the lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	CPSIG_API int transform_path_f(const float* data_in, float* data_out, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1.) noexcept;
	/** @brief */
	CPSIG_API int transform_path_d(const double* data_in, double* data_out, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	/** @} */

	/** @defgroup batch_transform_path_functions Batch transform path functions
	* @{
	*/

	/**
	* @brief Applies time-augmentation and/or the lead-lag transformation to a batch of paths of type float.
	*
	*
	* @param data_in Pointer to input path data (row-major), size = `batch_size * length * dimension`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `batch_size * transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length * 2 - 1 : length` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the paths.
	* @param length Length of the paths.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply the lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_transform_path_f(const float* data_in, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_transform_path_d(const double* data_in, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup transform_path_backprop_functions Transform path backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the transform_path function.
	*
	*
	* @param derivs Pointer to derivatives with respect to transformed path (row-major), size = `transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length * 2 - 1 : length` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `length * dimension`.
	* @param dimension Dimension of the original (pre-transformation) path.
	* @param length Length of the original (pre-transformation) path.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	CPSIG_API int transform_path_backprop_f(const float* derivs, float* data_out, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1.) noexcept;
	/** @brief */
	CPSIG_API int transform_path_backprop_d(const double* derivs, double* data_out, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	/** @} */

	/** @defgroup batch_transform_path_backprop_functions Batch transform path backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the batch_transform_path function.
	*
	*
	* @param derivs Pointer to derivatives with respect to transformed path (row-major), size = `batch_size * transformed_length * transformed_dimension`, where
	*					`transformed_length = lead_lag ? length * 2 - 1 : length` and `transformed_dimension = (lead_lag ? 2 : 1) * dimension + (time_aug ? 1 : 0)`.
	* @param data_out Pointer to output buffer (row-major, preallocated), size = `batch_size * length * dimension`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the original (pre-transformation) paths.
	* @param length Length of the original (pre-transformation) paths.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_transform_path_backprop_f(const float* derivs, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_transform_path_backprop_d(const double* derivs, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	/** @} */

	/**
	* @brief Returns the length of a truncated signature.
	*
	*
	* @param dimension Dimension of the underlying path.
	* @param degree Truncation degree of the signature.
	* @return Length of a truncated signature. A returned value of 0 indicates integer overflow.
	*/
	CPSIG_API uint64_t sig_length(uint64_t dimension, uint64_t degree) noexcept;

	/** @defgroup sig_combine_functions Sig combine functions
	* @{
	*/

	/**
	* @brief Combines two truncated signatures of the same degree and dimension into one signature.
	*
	*
	* @param sig1 Pointer to the first truncated signature, size = `sig_length(dimension, degree)`.
	* @param sig2 Pointer to the second truncated signature, size = `sig_length(dimension, degree)`. Must have the same degree and dimension as the first.
	* @param out Pointer to the output buffer (preallocated), size = `sig_length(dimension, degree)`.
	* @param dimension Dimension of the underlying paths.
	* @param degree Truncation degree of the signatures.
	* @return Status code (0 = success).
	*/
	CPSIG_API int sig_combine_f(const float* sig1, const float* sig2, float* out, uint64_t dimension, uint64_t degree) noexcept;
	/** @brief */
	CPSIG_API int sig_combine_d(const double* sig1, const double* sig2, double* out, uint64_t dimension, uint64_t degree) noexcept;
	/** @} */

	/** @defgroup batch_sig_combine_functions Batch sig combine functions
	* @{
	*/

	/**
	* @brief Combines a batch of pairs of truncated signatures of the same degree and dimension.
	*
	*
	* @param sig1 Pointer to the batch of first truncated signatures (row-major), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig2 Pointer to the batch of second truncated signatures (row-major), size = `batch_size * sig_length(dimension, degree)`.
	*					Must have the same batch size, degree and dimension as the first.
	* @param out Pointer to the output buffer (row-major, preallocated), size = `batch_size * sig_length(dimension, degree)`.
	* @param batch_size Batch size of sig1 and sig2.
	* @param dimension Dimension of the underlying paths.
	* @param degree Truncation degree of the signatures.
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_combine_f(const float* sig1, const float* sig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_combine_d(const double* sig1, const double* sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_combine_backprop_functions Sig combine backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the sig_combine function.
	*
	*
	* @param sig_combined_derivs Pointer to the derivatives with respect to the combined signature, size = `sig_length(dimension, degree)`.
	* @param sig1_deriv Pointer to the output buffer for the derivatives with respect to sig1 (preallocated), size = `sig_length(dimension, degree)`.
	* @param sig2_deriv Pointer to the output buffer for the derivatives with respect to sig2 (preallocated), size = `sig_length(dimension, degree)`.
	* @param sig1 Pointer to the first truncated signature (precomputed), size = `sig_length(dimension, degree)`.
	* @param sig2 Pointer to the second truncated signature (precomputed), size = `sig_length(dimension, degree)`. Must have the same degree and dimension as the first.
	* @param dimension Dimension of the underlying path.
	* @param degree Truncation degree of the signature.
	* @return Status code (0 = success).
	*/
	CPSIG_API int sig_combine_backprop_f(const float* sig_combined_derivs, float* sig1_deriv, float* sig2_deriv, const float* sig1, const float* sig2, uint64_t dimension, uint64_t degree) noexcept;
	/** @brief */
	CPSIG_API int sig_combine_backprop_d(const double* sig_combined_derivs, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t dimension, uint64_t degree) noexcept;
	/** @} */

	/** @defgroup batch_sig_combine_backprop_functions Batch sig combine backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the batch_sig_combine function.
	*
	*
	* @param sig_combined_derivs Pointer to the derivatives with respect to the combined signatures (row-major), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig1_deriv Pointer to the output buffer for the derivatives with respect to sig1 (row-major, preallocated), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig2_deriv Pointer to the output buffer for the derivatives with respect to sig2 (row-major, preallocated), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig1 Pointer to the batch of first truncated signatures (row-major, precomputed), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig2 Pointer to the batch of second truncated signatures (row-major, precomputed), size = `batch_size * sig_length(dimension, degree)`. Must have the same batch size, degree and dimension as sig1.
	* @param batch_size Batch size of the signatures.
	* @param dimension Dimension of the underlying paths.
	* @param degree Truncation degree of the signatures.
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_combine_backprop_f(const float* sig_combined_derivs, float* sig1_deriv, float* sig2_deriv, const float* sig1, const float* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_combine_backprop_d(const double* sig_combined_derivs, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_coef_functions Signature coefficient functions
	* @{
	*/

	/**
	* @brief For a path of type float, computes coefficients of its signature.
	*
	*
	* @param path Pointer to path data (row-major), size = `length * dimension`.
	* @param out Pointer to output buffer (preallocated), size = `prefixes ? sum(max(degrees[i], 1)) : num_multi_idx`.
	* @param multi_idx Pointer to flattened array of multi indices, size = `sum(degrees[i])`.
	* @param num_multi_idx Number of multi indices.
	* @param degrees Pointer to array of degrees of the multi indices, size = `num_multi_idx`.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param prefixes If `true`, will additionally return coefficients for all prefixes of words (default = false).
	* @return Status code (0 = success).
	*/
	CPSIG_API int sig_coef_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., bool prefixes = false) noexcept;
	/** @brief */
	CPSIG_API int sig_coef_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool prefixes = false) noexcept;
	/** @} */

	/** @defgroup batch_sig_coef_functions Batch signature coefficient functions
	* @{
	*/

	/**
	* @brief For a batch of paths of type float, computes coefficients of their signatures.
	*
	*
	* @param path Pointer to path data (row-major), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (preallocated), size = `batch_size * (prefixes ? sum(max(degrees[i], 1)) : num_multi_idx)`.
	* @param multi_idx Pointer to flattened array of multi indices, size = `sum(degrees[i])`.
	* @param num_multi_idx Number of multi indices.
	* @param degrees Pointer to array of degrees of the multi indices, size = `num_multi_idx`.
	* @param batch_size Batch size.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param prefixes If `true`, will additionally return coefficients for all prefixes of words (default = false).
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_coef_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., bool prefixes = false, int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_coef_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool prefixes = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_coef_backprop_functions Signature coefficient backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the sig_coef functions
	*
	*
	* @param path Pointer to path data (row-major), size = `length * dimension`.
	* @param out Pointer to output buffer (preallocated), size = `length * dimension`.
	* @param coefs Pointer to coefficients computed using `sig_coef` with `prefixes=true`, size = `sum(degrees[i])`.
	* @param derivs Pointer to derivatives with respect to coefficients,  size = `sum(degrees[i])`. **Modified in-place.**
	* @param multi_idx Pointer to flattened array of multi indices, size = `sum(degrees[i])`.
	* @param num_multi_idx Number of multi indices.
	* @param degrees Pointer to array of degrees of the multi indices, size = `num_multi_idx`.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	CPSIG_API int sig_coef_backprop_f(const float* path, float* out, const float* coefs, float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1.) noexcept;
	/** @brief */
	CPSIG_API int sig_coef_backprop_d(const double* path, double* out, const double* coefs, double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	/** @} */

	/** @defgroup batch_sig_coef_backprop_functions Batch signature coefficient backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the batch_sig_coef functions
	*
	*
	* @param path Pointer to path data (row-major), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (preallocated), size = `batch_size * length * dimension`.
	* @param coefs Pointer to coefficients computed using `sig_coef` with `prefixes=true`, size = `batch_size * sum(degrees[i])`.
	* @param derivs Pointer to derivatives with respect to coefficients,  size = `batch_size * sum(degrees[i])`. **Modified in-place.**
	* @param multi_idx Pointer to flattened array of multi indices, size = `sum(degrees[i])`.
	* @param num_multi_idx Number of multi indices.
	* @param degrees Pointer to array of degrees of the multi indices, size = `num_multi_idx`.
	* @param batch_size Batch size.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_coef_backprop_f(const float* path, float* out, const float* coefs, float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_coef_backprop_d(const double* path, double* out, const double* coefs, double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup signature_functions Signature functions
	* @{
	*/

	/**
	* @brief Computes the signature of a path of type float.
	* @param path Pointer to path data (row-major), size = `length * dimension`.
	* @param out Pointer to output buffer (preallocated), size = `sig_length(dimension, degree)`.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param degree Truncation degree of the signature.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param horner Whether to use Horner's scheme (default = true).
	* @return Status code (0 = success).
	*/
	CPSIG_API int signature_f(const float* path, float* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1., bool horner = true) noexcept; //bool time_aug = false, bool lead_lag = false, bool horner = true);
	/** @brief */
	CPSIG_API int signature_d(const double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true) noexcept;
	/** @} */


	/** @defgroup batch_signature_functions Batch signature functions
	* @{
	*/

	/**
	* @brief Computes the signatures of a batch of paths of type float.
	* @param path Pointer to path batch data (row-major), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `batch_size * sig_length(dimension, degree)`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param degree Truncation degree of the signature.
	* @param time_aug Whether to add time augmentation (default = false).
	* @param lead_lag Whether to apply lead-lag transform (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param horner Whether to use Horner's scheme (default = true).
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_signature_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1., bool horner = true, int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_signature_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true, int n_jobs = 1) noexcept;
	/** @} */


	/** @defgroup sig_backprop_functions Signature backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the signature_f function.
	* 
	* @param path Pointer to path data (row-major), size = `length * dimension`.
	* @param out Pointer to output buffer (preallocated), size = `length * dimension`.
	* @param sig_derivs Pointer to derivatives with respect to the signature, size = `sig_length(dimension, degree)`.
	* @param sig Pointer to signature of the path (precomputed), size = `sig_length(dimension, degree)`.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param degree Truncation degree of the signature.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @return Status code (0 = success).
	*/
	CPSIG_API int sig_backprop_f(const float* path, float* out, const float* sig_derivs, const float* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1.) noexcept;
	/** @brief */
	CPSIG_API int sig_backprop_d(const double* path, double* out, const double* sig_derivs, const double* sig, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1.) noexcept;
	/** @} */


	/** @defgroup batch_sig_backprop_functions Signature backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the batch_signature_f function.
	*
	* @param path Pointer to path batch data (row-major), size = `batch_size * length * dimension`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `batch_size * length * dimension`.
	* @param sig_derivs Pointer to derivatives with respect to the signatures (row-major), size = `batch_size * sig_length(dimension, degree)`.
	* @param sig Pointer to signatures of the paths (row-major, precomputed), size = `batch_size * sig_length(dimension, degree)`.
	* @param batch_size Batch size of the paths.
	* @param dimension Dimension of the path.
	* @param length Length of the path.
	* @param degree Truncation degree of the signature.
	* @param time_aug Whether time augmentation was applied (default = false).
	* @param lead_lag Whether the lead-lag transform was applied (default = false).
	* @param end_time End time for time augmentation (default = 1.0).
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_backprop_f(const float* path, float* out, const float* sig_derivs, const float* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1., int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_backprop_d(const double* path, double* out, const double* sig_derivs, const double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	/** @} */

	/**
	* @brief Returns the length of a truncated log signature.
	*
	*
	* @param dimension Dimension of the underlying path.
	* @param degree Truncation degree of the log signature.
	* @return Length of a truncated log signature. A returned value of 0 indicates integer overflow or invalid parameters (dimension == 0 or degree == 0).
	*/
	CPSIG_API uint64_t log_sig_length(uint64_t dimension, uint64_t degree) noexcept;

	/**
	* @brief Sets the cache directory to use in ``prepare_log_sig`` when ``use_disk=true``.
	If the cache directory is unset, a default directory will be used. This function is not thread safe.
	*
	*
	* @param dir Cache directory
	* @return Status code (0 = success).
	*/
	CPSIG_API int set_cache_dir(const char* dir) noexcept;

	/**
	* @brief Prepares for log signature calculations. This function is not thread safe.
	*
	*
	* @param dimension Dimension of the underlying path.
	* @param degree Truncation degree of the log signature.
	* @param method Method to use for log signature calculation. Please see Python documentation for details.
	* @param use_disk If ``false``, will cache prepared objects in memory only.
        If ``true``, will also save these objects in a cache directory to be
        re-used for future runs.
	* @return Status code (0 = success).
	*/
	CPSIG_API int prepare_log_sig(uint64_t dimension, uint64_t degree, int method, bool use_disk = false) noexcept;

	/**
	* @brief Clear the log signature cache generated by `prepare_log_sig`. This function is not thread safe.
	* 
	* @param use_disk If ``false``, will clear the cache from memory only.
        If ``true``, will also clear the cache directory.
	* @return Status code (0 = success).
	*/
	CPSIG_API int clear_cache(bool use_disk = false) noexcept;

	/** @defgroup sig_to_log_sig_functions Sig to log sig functions
	* @{
	*/

	/**
	* @brief Converts a signature to a log signature using the specified method.
	* @param sig Pointer to signature, size = `sig_length(dimension, degree)`.
	* @param out Pointer to output buffer (preallocated), size = `method ? log_sig_length(dimension, degree) : sig_length(dimension, degree)`.
	* @param dimension Dimension of the path.
	* @param degree Truncation degree of the (log) signature.
	* @param time_aug Whether time augmentation was used for the signature computation (default = false).
	* @param lead_lag Whether the lead-lag transform was used for the signature computation (default = false).
	* @param method The method to use for the log calculation (`0`, `1` or `2`). Please see the Python documentation for details.
	* @return Status code (0 = success).
	*/
	CPSIG_API int sig_to_log_sig_f(const float* sig, float* out, uint64_t dimension, uint64_t degree, bool time_aug = false, bool lead_lag = false, int method = 0) noexcept; //bool time_aug = false, bool lead_lag = false, bool horner = true);
	/** @brief */
	CPSIG_API int sig_to_log_sig_d(const double* sig, double* out, uint64_t dimension, uint64_t degree, bool time_aug = false, bool lead_lag = false, int method = 0) noexcept;
	/** @} */

	/** @defgroup batch_sig_to_log_sig_functions Batch sig to log sig functions
	* @{
	*/

	/**
	* @brief Converts a batch of signatures to log signatures using the specified method.
	* @param sig Pointer to batch of signatures (row-major), size = `batch_size * sig_length(dimension, degree)`.
	* @param out Pointer to output buffer (preallocated), size = `batch_size * (method ? log_sig_length(dimension, degree) : sig_length(dimension, degree))`.
	* @param batch_size Batch size.
	* @param dimension Dimension of the paths.
	* @param degree Truncation degree of the (log) signatures.
	* @param time_aug Whether time augmentation was used for the signature computations (default = false).
	* @param lead_lag Whether the lead-lag transform was used for the signature computations (default = false).
	* @param method The method to use for the log calculation (`0`, `1` or `2`). Please see the Python documentation for details.
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_to_log_sig_f(const float* sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug = false, bool lead_lag = false, int method = 0, int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_to_log_sig_d(const double* sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug = false, bool lead_lag = false, int method = 0, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_to_log_sig_backprop_functions Sig to log sig backprop functions
	* @{
	*/

	/**
	* @brief Backpropagates derivatives through the `sig_to_log_sig_f` function.
	* @param sig Pointer to signature, size = `sig_length(dimension, degree)`.
	* @param out Pointer to output buffer (preallocated), size = `sig_length(dimension, degree)`.
	* @param log_sig_derivs Pointer to derivatives with respect to the log signature, size = `method ? log_sig_length(dimension, degree) : sig_length(dimension, degree)`.
	* @param dimension Dimension of the path.
	* @param degree Truncation degree of the (log) signature.
	* @param time_aug Whether time augmentation was used for the signature computation (default = false).
	* @param lead_lag Whether the lead-lag transform was used for the signature computation (default = false).
	* @param method The method used for the log calculation (`0`, `1` or `2`). Please see the Python documentation for details.
	* @return Status code (0 = success).
	*/
	CPSIG_API int sig_to_log_sig_backprop_f(const float* sig, float* out, const float* log_sig_derivs, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method) noexcept;
	/** @brief */
	CPSIG_API int sig_to_log_sig_backprop_d(const double* sig, double* out, const double* log_sig_derivs, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method) noexcept;
	/** @} */
	
	/** @defgroup batch_sig_to_log_sig_backprop_functions Batch sig to log sig backprop functions
	* @{
	*/

	/**
	* @brief Backpropagates derivatives through the `batch_sig_to_log_sig_f` function.
	* @param sig Pointer to batch of signatures (row-major), size = `batch_size * sig_length(dimension, degree)`.
	* @param out Pointer to output buffer (preallocated), size = `batch_size * sig_length(dimension, degree)`.
	* @param log_sig_derivs Pointer to derivatives with respect to the log signature, size = `batch_size * (method ? log_sig_length(dimension, degree) : sig_length(dimension, degree))`.
	* @param batch_size Batch size.
	* @param dimension Dimension of the paths.
	* @param degree Truncation degree of the (log) signatures.
	* @param time_aug Whether time augmentation was used for the signature computations (default = false).
	* @param lead_lag Whether the lead-lag transform was used for the signature computations (default = false).
	* @param method The method used for the log calculation (`0`, `1` or `2`). Please see the Python documentation for details.
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_to_log_sig_backprop_f(const float* sig, float* out, const float* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, int n_jobs) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_to_log_sig_backprop_d(const double* sig, double* out, const double* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, int n_jobs) noexcept;
	/** @} */

	/** @defgroup log_sig_combine_functions Log sig combine functions
	* @{
	*/

	CPSIG_API int log_sig_combine_f(const float* log_sig1, const float* log_sig2, float* out,
		uint64_t dimension, uint64_t degree) noexcept;
	CPSIG_API int log_sig_combine_d(const double* log_sig1, const double* log_sig2, double* out,
		uint64_t dimension, uint64_t degree) noexcept;

	CPSIG_API int batch_log_sig_combine_f(const float* log_sig1, const float* log_sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	CPSIG_API int batch_log_sig_combine_d(const double* log_sig1, const double* log_sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_kernel_functions Signature kernel functions
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
	CPSIG_API int sig_kernel_f(const float* gram, float* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */
	CPSIG_API int sig_kernel_d(const double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup batch_sig_kernel_functions Batch signature kernel functions
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
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @param return_grid Whether to return the entire PDE grid (default = false).
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_kernel_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_kernel_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_kernel_backprop_functions Signature kernel backprop functions
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
	CPSIG_API int sig_kernel_backprop_f(const float* gram, float* out, const float* derivs, const float* k_grid, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @brief */	
	CPSIG_API int sig_kernel_backprop_d(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false) noexcept;
	/** @} */

	/** @defgroup batch_sig_kernel_backprop_functions Batch signature kernel backprop functions
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
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all 
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example 
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @param return_grid If true, derivs is expected to be grid-sized per batch element; if false, derivs has one scalar per batch element.
	* @return Status code (0 = success).
	*/
	CPSIG_API int batch_sig_kernel_backprop_f(const float* gram, float* out, const float* derivs, const float* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief */
	CPSIG_API int batch_sig_kernel_backprop_d(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @} */
}


