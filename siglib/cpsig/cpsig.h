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
	* @brief Applies time-augmentation and/or the lead-lag transformation to a batch of paths.
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
	[[nodiscard]] CPSIG_API int transform_path_f(const float* data_in, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int transform_path_d(const double* data_in, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup transform_path_backprop_functions Transform path backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the transform_path function.
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
	[[nodiscard]] CPSIG_API int transform_path_backprop_f(const float* derivs, float* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int transform_path_backprop_d(const double* derivs, double* data_out, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
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
	* @brief Combines pairs of truncated signatures of the same degree and dimension.
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
	[[nodiscard]] CPSIG_API int sig_combine_f(const float* sig1, const float* sig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_combine_d(const double* sig1, const double* sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_combine_backprop_functions Sig combine backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the sig_combine function.
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
	[[nodiscard]] CPSIG_API int sig_combine_backprop_f(const float* sig_combined_derivs, float* sig1_deriv, float* sig2_deriv, const float* sig1, const float* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_combine_backprop_d(const double* sig_combined_derivs, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup linear_sig_functions Linear sig functions
	* @{
	*/
	[[nodiscard]] CPSIG_API int linear_sig_f(const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int linear_sig_d(const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_join_functions Sig join functions
	* @{
	*/
	[[nodiscard]] CPSIG_API int sig_join_f(const float* sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int sig_join_d(const double* sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_join_backprop_functions Sig join backprop functions
	* @{
	*/
	[[nodiscard]] CPSIG_API int sig_join_backprop_f(const float* d_out, float* d_sig, float* d_displacement, const float* sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int sig_join_backprop_d(const double* d_out, double* d_sig, double* d_displacement, const double* sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend = false, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_coef_functions Signature coefficient functions
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
	[[nodiscard]] CPSIG_API int sig_coef_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., bool prefixes = false, int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_coef_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool prefixes = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_coef_backprop_functions Signature coefficient backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the sig_coef functions
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
	[[nodiscard]] CPSIG_API int sig_coef_backprop_f(const float* path, float* out, const float* coefs, float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, float end_time = 1., int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_coef_backprop_d(const double* path, double* out, const double* coefs, double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug = false, bool lead_lag = false, double end_time = 1., int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup signature_functions Signature functions
	* @{
	*/

	/**
	* @brief Computes the signatures of a batch of paths.
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
	[[nodiscard]] CPSIG_API int signature_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1., bool horner = true, bool scalar_term = true, int n_jobs = 1, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int signature_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool horner = true, bool scalar_term = true, int n_jobs = 1, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup sig_backprop_functions Signature backprop functions
	* @{
	*/

	/**
	* @brief Backpropagation through the signature_f function.
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
	[[nodiscard]] CPSIG_API int sig_backprop_f(const float* path, float* out, const float* sig_derivs, const float* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, float end_time = 1., bool scalar_term = true, int n_jobs = 1, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_backprop_d(const double* path, double* out, const double* sig_derivs, const double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool scalar_term = true, int n_jobs = 1, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
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
	[[nodiscard]] CPSIG_API int set_cache_dir(const char* dir) noexcept;

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
	[[nodiscard]] CPSIG_API int prepare_log_sig(uint64_t dimension, uint64_t degree, int method, bool use_disk = false) noexcept;

	/**
	* @brief Clear the log signature cache generated by `prepare_log_sig`. This function is not thread safe.
	* 
	* @param use_disk If ``false``, will clear the cache from memory only.
        If ``true``, will also clear the cache directory.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int clear_cache(bool use_disk = false) noexcept;

	/** @brief Release all in-memory cpsig caches. Idempotent; cache-backed computations require preparation again. */
	CPSIG_API void cpsig_shutdown() noexcept;

	/** @defgroup sig_to_log_sig_functions Sig to log sig functions
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
	[[nodiscard]] CPSIG_API int sig_to_log_sig_f(const float* sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug = false, bool lead_lag = false, int method = 0, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_to_log_sig_d(const double* sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug = false, bool lead_lag = false, int method = 0, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	
	/** @defgroup sig_to_log_sig_backprop_functions Sig to log sig backprop functions
	* @{
	*/

	/**
	* @brief Backpropagates derivatives through the `sig_to_log_sig_f` function.
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
	[[nodiscard]] CPSIG_API int sig_to_log_sig_backprop_f(const float* sig, float* out, const float* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_to_log_sig_backprop_d(const double* sig, double* out, const double* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, int method, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup log_sig_combine_functions Log sig combine functions
	* @{
	*/

	[[nodiscard]] CPSIG_API int log_sig_combine_f(const float* log_sig1, const float* log_sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int log_sig_combine_d(const double* log_sig1, const double* log_sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup log_sig_combine_backprop_functions Log sig combine backprop functions
	* @{
	*/

	[[nodiscard]] CPSIG_API int log_sig_combine_backprop_f(const float* d_out, float* d_ls1, float* d_ls2,
		const float* ls1, const float* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int log_sig_combine_backprop_d(const double* d_out, double* d_ls1, double* d_ls2,
		const double* ls1, const double* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup log_sig_join_functions Log sig join functions
	* @{
	*/
	[[nodiscard]] CPSIG_API int log_sig_join_f(const float* log_sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int log_sig_join_d(const double* log_sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup log_sig_join_backprop_functions Log sig join backprop functions
	* @{
	*/
	[[nodiscard]] CPSIG_API int log_sig_join_backprop_f(const float* d_out, float* d_logsig, float* d_displacement, const float* log_sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int log_sig_join_backprop_d(const double* d_out, double* d_logsig, double* d_displacement, const double* log_sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup log_sig_from_path_functions Log-signature from path functions
	* @{
	*/
	[[nodiscard]] CPSIG_API int log_sig_from_path_f(const float* path, float* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int log_sig_from_path_d(const double* path, double* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;

	[[nodiscard]] CPSIG_API int log_sig_from_path_backprop_f(const float* d_out, float* d_path, const float* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int log_sig_from_path_backprop_d(const double* d_out, double* d_path, const double* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup logsig_to_sig_functions Log-signature to signature (tensor exponential) functions
	* @{
	*/

	[[nodiscard]] CPSIG_API int logsig_to_sig_f(const float* log_sig, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree,
		bool time_aug = false, bool lead_lag = false, int method = 0, bool scalar_term = true, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int logsig_to_sig_d(const double* log_sig, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree,
		bool time_aug = false, bool lead_lag = false, int method = 0, bool scalar_term = true, int n_jobs = 1) noexcept;

	[[nodiscard]] CPSIG_API int logsig_to_sig_backprop_f(const float* log_sig, float* out, const float* sig_derivs,
		uint64_t batch_size, uint64_t dimension, uint64_t degree,
		bool time_aug = false, bool lead_lag = false, int method = 0, bool scalar_term = true, int n_jobs = 1) noexcept;
	[[nodiscard]] CPSIG_API int logsig_to_sig_backprop_d(const double* log_sig, double* out, const double* sig_derivs,
		uint64_t batch_size, uint64_t dimension, uint64_t degree,
		bool time_aug = false, bool lead_lag = false, int method = 0, bool scalar_term = true, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_kernel_functions Signature kernel functions
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
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int sig_kernel_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_kernel_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup polynomial_sig_kernel_functions Polynomial signature kernel functions
	* @{
	*/

	/**
	* @brief Computes monomial polynomial approximations from increment gram matrices.
	*
	* @param gram Pointer to batch gram matrix data (row-major), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `batch_size * (return_grid ? length1 * length2 : 1)`.
	* @param state Optional output buffer for incoming tile coefficients, size = `batch_size * 2 * (length1 - 1) * (length2 - 1) * (order + 1)`.
	* @param batch_size Batch size of the path pairs.
	* @param dimension Dimension of the original paths.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param order Highest retained polynomial degree, between 2 and 64.
	* @param return_grid Whether to return values at all original path vertices.
	* @param n_jobs Number of threads used across independent batch pairs.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int sig_kernel_poly_f(const float* gram, float* out, float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief Double-precision variant of sig_kernel_poly_f. */
	[[nodiscard]] CPSIG_API int sig_kernel_poly_d(const double* gram, double* out, double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief Backpropagates through sig_kernel_poly_f with respect to gram. */
	[[nodiscard]] CPSIG_API int sig_kernel_poly_backprop_f(const float* gram, float* gram_derivs, const float* output_derivs, const float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief Double-precision variant of sig_kernel_poly_backprop_f. */
	[[nodiscard]] CPSIG_API int sig_kernel_poly_backprop_d(const double* gram, double* gram_derivs, const double* output_derivs, const double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup branched_sig_kernel_functions Branched signature kernel functions
	* @{
	*/

	/**
	* @brief Computes the depth-recursive branched signature kernel from batch gram matrices.
	*
	* Computes the non-planar BCK branched signature kernel using the Chevyrev-Oberhauser recursion.
	* The core consumes precomputed static-kernel increments rather than path coordinates.
	*
	* @param gram Pointer to batch gram matrix data (row-major), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer (row-major, preallocated), size = `batch_size * (return_grid ? (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1) : 1)`.
	* @param batch_size Batch size of the path pairs.
	* @param dimension Dimension of the original paths. The kernel core uses `gram`, so this is retained for API parity.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param depth Truncation depth of the branched kernel recursion. If zero, the output is filled with ones.
	* @param dyadic_order_1 Dyadic refinement for the first paths.
	* @param dyadic_order_2 Dyadic refinement for the second paths.
	* @param return_grid If true, returns the final-depth grid; otherwise returns the endpoint scalar per batch item.
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int branched_sig_kernel_f(const float* gram, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief Double-precision variant of branched_sig_kernel_f. */
	[[nodiscard]] CPSIG_API int branched_sig_kernel_d(const double* gram, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup sig_kernel_backprop_functions Signature kernel backprop functions
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
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int sig_kernel_backprop_f(const float* gram, float* out, const float* derivs, const float* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int sig_kernel_backprop_d(const double* gram, double* out, const double* derivs, const double* k_grid, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup branched_sig_kernel_backprop_functions Branched signature kernel backprop functions
	* @{
	*/

	/**
	* @brief Backpropagates through branched_sig_kernel_f with respect to the gram matrix.
	*
	* Computes derivatives with respect to the precomputed static-kernel increments used by the branched
	* signature kernel. Path-coordinate derivatives are handled by the Python static-kernel wrappers.
	*
	* @param gram Pointer to batch gram matrix data (row-major), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param out Pointer to output buffer for dF/d(gram) (row-major, preallocated), size = `batch_size * (length1 - 1) * (length2 - 1)`.
	* @param derivs Pointer to input derivatives. If `return_grid` is false, size = `batch_size`. If `return_grid` is true, size = `batch_size * (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param k_stack Optional pointer to precomputed forward grids K_0, ..., K_depth (row-major). May be null. If supplied, size = `batch_size * (depth + 1) * (((length1 - 1) << dyadic_order_1) + 1) * (((length2 - 1) << dyadic_order_2) + 1)`.
	* @param batch_size Batch size of the path pairs.
	* @param dimension Dimension of the original paths. The kernel core uses `gram`, so this is retained for API parity.
	* @param length1 Length of the first paths.
	* @param length2 Length of the second paths.
	* @param depth Truncation depth of the branched kernel recursion. If zero, derivatives with respect to `gram` are zero.
	* @param dyadic_order_1 Dyadic refinement for the first paths.
	* @param dyadic_order_2 Dyadic refinement for the second paths.
	* @param return_grid If true, `derivs` is expected to be grid-sized per batch item; otherwise it has one scalar per batch item.
	* @param n_jobs Number of threads to run in parallel. If n_jobs = 1, the computation is run serially. If set to -1, all
	*				available threads are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
	*				if n_jobs = -2, all threads but one are used (default = 1).
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int branched_sig_kernel_backprop_f(const float* gram, float* out, const float* derivs, const float* k_stack, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @brief Double-precision variant of branched_sig_kernel_backprop_f. */
	[[nodiscard]] CPSIG_API int branched_sig_kernel_backprop_d(const double* gram, double* out, const double* derivs, const double* k_stack, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t depth, uint64_t dyadic_order_1, uint64_t dyadic_order_2, bool return_grid = false, int n_jobs = 1) noexcept;
	/** @} */

	/** @defgroup branched_sig_functions Branched signature functions
	* @{
	*/

	/** @brief Prepares the branched-signature cache. */
	[[nodiscard]] CPSIG_API int prepare_branched_sig(uint64_t dimension, uint64_t max_nodes, bool use_disk = false, bool planar = false) noexcept;
	CPSIG_API uint64_t branched_sig_length(uint64_t dimension, uint64_t max_nodes, bool planar = false) noexcept;

	[[nodiscard]] CPSIG_API int branched_sig_f(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, bool scalar_term = true, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	[[nodiscard]] CPSIG_API int branched_sig_d(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, bool scalar_term = true, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup branched_sig_coef_functions Branched signature coefficient functions
	* @{
	*/

	/**
	* @brief Prepares a sparse branched-signature coefficient cache.
	*
	* The exact dimensions, degree, planar convention, and serialized tree data
	* must be reused by the forward and backpropagation calls. `tree_data` starts
	* with the number of requested basis elements. Each basis element then starts
	* with its number of roots, followed by preorder `(label, num_children)` pairs.
	*
	* @param tree_data Serialized requested trees and ordered forests.
	* @param tree_data_len Number of uint64_t values in `tree_data`.
	* @param data_dimension Dimension of the unaugmented path.
	* @param dimension Dimension after time or lead-lag augmentation.
	* @param max_nodes Maximum number of nodes in the requested basis elements.
	* @param planar Whether the data uses the planar MKW ordered-forest basis.
	* @param use_disk Whether to persist and load the shared host-side sparse plan.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int prepare_branched_sig_coef(const uint64_t* tree_data, uint64_t tree_data_len, uint64_t data_dimension, uint64_t dimension, uint64_t max_nodes, bool planar = false, bool use_disk = false) noexcept;

	/**
	* @brief Computes selected branched-signature coefficients for float paths.
	*
	* The sparse computation keeps the union of the coproduct dependency closures
	* of the requested basis elements. Call prepare_branched_sig_coef with matching
	* dimensions, degree, planar convention, and tree data before this function.
	*
	* @param path Pointer to row-major path data, size = `batch_size * length * dimension`.
	* @param out Pointer to the preallocated coefficient output, size = `batch_size * tree_data[0]`.
	* @param tree_data Serialized requested trees passed to prepare_branched_sig_coef.
	* @param tree_data_len Number of uint64_t values in `tree_data`.
	* @param batch_size Number of paths.
	* @param dimension Dimension of the unaugmented path.
	* @param length Number of points in each path.
	* @param max_nodes Maximum number of nodes in the prepared branched basis.
	* @param n_jobs Number of CPU threads. Use 1 for serial execution or -1 for all available threads.
	* @param time_aug Whether to add time augmentation.
	* @param lead_lag Whether to apply the lead-lag transform.
	* @param end_time End time for time augmentation.
	* @param planar Whether the data uses the planar MKW ordered-forest basis.
	* @param correction Optional segment correction data, with the layout used by branched_sig_f.
	* @param correction_len Width of one correction row.
	* @param correction_batch_stride Correction stride between batch items, in elements.
	* @param correction_segment_stride Correction stride between path segments, in elements.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int branched_sig_coef_f(const float* path, float* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @copydoc branched_sig_coef_f */
	[[nodiscard]] CPSIG_API int branched_sig_coef_d(const double* path, double* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @defgroup branched_sig_coef_backprop_functions Branched signature coefficient backprop functions
	* @{
	*/

	/**
	* @brief Backpropagates through selected float branched-signature coefficients.
	*
	* Reverse mode reuses the prepared sparse coproduct closure, seeds cotangents
	* at the requested coefficients, and replays segment products backward.
	* Correction data is treated as constant.
	*
	* @param path Pointer to row-major path data, size = `batch_size * length * dimension`.
	* @param out Pointer to the preallocated path derivative output, size = `batch_size * length * dimension`.
	* @param coefs Forward coefficient values from branched_sig_coef_f, size = `batch_size * tree_data[0]`.
	* @param derivs Coefficient cotangents, size = `batch_size * tree_data[0]`.
	* @param tree_data Serialized requested trees passed to the forward call.
	* @param tree_data_len Number of uint64_t values in `tree_data`.
	* @param batch_size Number of paths.
	* @param dimension Dimension of the unaugmented path.
	* @param length Number of points in each path.
	* @param max_nodes Maximum number of nodes in the prepared branched basis.
	* @param n_jobs Number of CPU threads. Use 1 for serial execution or -1 for all available threads.
	* @param time_aug Whether time augmentation was used in the forward call.
	* @param lead_lag Whether lead-lag was used in the forward call.
	* @param end_time End time used for time augmentation.
	* @param planar Whether the data uses the planar MKW ordered-forest basis.
	* @param correction Optional correction data from the forward call.
	* @param correction_len Width of one correction row.
	* @param correction_batch_stride Correction stride between batch items, in elements.
	* @param correction_segment_stride Correction stride between path segments, in elements.
	* @return Status code (0 = success).
	*/
	[[nodiscard]] CPSIG_API int branched_sig_coef_backprop_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @copydoc branched_sig_coef_backprop_f */
	[[nodiscard]] CPSIG_API int branched_sig_coef_backprop_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */

	/** @addtogroup branched_sig_functions
	* @{
	*/
	/** @brief Prepares the branched-signature and derived branched-log caches. */
	[[nodiscard]] CPSIG_API int prepare_branched_log_sig(uint64_t dimension, uint64_t max_nodes, bool use_disk = false, bool planar = false) noexcept;

	[[nodiscard]] CPSIG_API int branched_sig_combine_f(const float* bsig1, const float* bsig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CPSIG_API int branched_sig_combine_d(const double* bsig1, const double* bsig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;

	/**
	* @brief Computes the truncated Hopf logarithm of a branched signature.
	*
	* The input and output use the same decorated-tree basis and the same scalar-term
	* convention. If scalar_term is true, the output scalar coefficient is zero.
	*/
	[[nodiscard]] CPSIG_API int branched_sig_to_log_sig_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int branched_sig_to_log_sig_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;

	/**
	* @brief Backpropagation through branched_sig_to_log_sig.
	*
	* Computes the vector-Jacobian product from derivatives with respect to the
	* branched log signature to derivatives with respect to the branched signature.
	*/
	[[nodiscard]] CPSIG_API int branched_sig_to_log_sig_backprop_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;
	/** @brief */
	[[nodiscard]] CPSIG_API int branched_sig_to_log_sig_backprop_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;

	[[nodiscard]] CPSIG_API int branched_sig_combine_backprop_f(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;
	[[nodiscard]] CPSIG_API int branched_sig_combine_backprop_d(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs = 1, bool planar = false, bool scalar_term = true) noexcept;

	[[nodiscard]] CPSIG_API int branched_sig_backprop_f(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, float end_time = 1.f, bool planar = false, bool scalar_term = true, const float* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	[[nodiscard]] CPSIG_API int branched_sig_backprop_d(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs = 1, bool time_aug = false, bool lead_lag = false, double end_time = 1., bool planar = false, bool scalar_term = true, const double* correction = nullptr, uint64_t correction_len = 0, uint64_t correction_batch_stride = 0, uint64_t correction_segment_stride = 0) noexcept;
	/** @} */
}
