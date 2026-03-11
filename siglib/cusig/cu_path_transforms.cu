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

#include "cupch.h"
#include "cusig.h"
#include "cu_path_transforms.h"
//#include "cuda_constants.h"

__constant__ uint64_t path_dimension;
__constant__ uint64_t length;
__constant__ bool time_aug;
__constant__ bool lead_lag;
__constant__ uint64_t path_size;

__constant__ uint64_t transformed_dimension;
__constant__ uint64_t transformed_length;
__constant__ uint64_t transformed_path_size;

template<typename T>
__global__ void transform_path_internal_(
	const T* data_in,
	T* data_out,
	T end_time
) {
	const int thread_id = threadIdx.x;

	const T* const data_in_ = data_in + blockIdx.x * path_size;
	T* const data_out_ = data_out + blockIdx.x * transformed_path_size;

	if (!(time_aug || lead_lag)) {
		for (uint64_t i = thread_id; i < path_size; i += 32)
			data_out_[i] = static_cast<T>(data_in_[i]);
	}

	if (lead_lag) {
		const uint64_t twice_dimension = 2 * path_dimension;
		const uint64_t twice_transformed_dimension = 2 * transformed_dimension;

		for (uint64_t i = thread_id; i < length; i += 32) {
			for (uint64_t j = 0; j < path_dimension; ++j) {
				data_out_[i * twice_transformed_dimension + j] = static_cast<T>(data_in_[i * path_dimension + j]);
				data_out_[i * twice_transformed_dimension + j + path_dimension] = static_cast<T>(data_in_[i * path_dimension + j]);
			}
		}

		for (uint64_t i = thread_id; i < length - 1; i += 32) {
			for (uint64_t j = 0; j < twice_dimension; ++j) {
				data_out_[i * twice_transformed_dimension + transformed_dimension + j] = static_cast<T>(data_in_[i * path_dimension + j]);
			}
		}
	}
	else {
		for (uint64_t i = thread_id; i < length; i += 32) {
			for (uint64_t j = 0; j < path_dimension; ++j) {
				data_out_[i * transformed_dimension + j] = static_cast<T>(data_in_[i * path_dimension + j]);
			}
		}
	}

	if (time_aug) {
		const T scale = end_time / (transformed_length - 1);

		for (uint64_t i = thread_id; i < transformed_length; i += 32) {
			data_out_[(i + 1) * transformed_dimension - 1] = i * scale;
		}
	}
}

//template __global__ void transform_path_internal_<float>(
//	const float* data_in,
//	float* data_out
//);
//
//template __global__ void transform_path_internal_<double>(
//	const double* data_in,
//	double* data_out
//);

template<typename T>
void transform_path_(
	const T* data_in,
	T* data_out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length_,
	bool time_aug_,
	bool lead_lag_,
	T end_time
) {
	const uint64_t path_size_ = dimension_ * length_;

	const uint64_t transformed_length_ = lead_lag_ ? 2 * length_ - 1 : length_;
	const uint64_t transformed_dimension_ = (lead_lag_ ? 2 * dimension_ : dimension_) + (time_aug_ ? 1 : 0);
	const uint64_t transformed_path_size_ = transformed_length_ * transformed_dimension_;

	cudaMemcpyToSymbol(path_dimension, &dimension_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length, &length_, sizeof(uint64_t));
	cudaMemcpyToSymbol(time_aug, &time_aug_, sizeof(bool));
	cudaMemcpyToSymbol(lead_lag, &lead_lag_, sizeof(bool));
	cudaMemcpyToSymbol(path_size, &path_size_, sizeof(uint64_t));

	cudaMemcpyToSymbol(transformed_dimension, &transformed_dimension_, sizeof(uint64_t));
	cudaMemcpyToSymbol(transformed_length, &transformed_length_, sizeof(uint64_t));
	cudaMemcpyToSymbol(transformed_path_size, &transformed_path_size_, sizeof(uint64_t));

	transform_path_internal_ << <static_cast<unsigned int>(batch_size_), 32U >> > (data_in, data_out, end_time);

	check_cuda_error();
}

template<typename T>
__global__ void transform_path_backprop_internal_(
	const T* derivs,
	T* data_out,
	T end_time
) {
	const int thread_id = threadIdx.x;

	const T* const derivs_ = derivs + blockIdx.x * transformed_path_size;
	T* const data_out_ = data_out + blockIdx.x * path_size;

	if (lead_lag) {
		const uint64_t twice_dimension = 2 * path_dimension;
		const uint64_t twice_transformed_dimension = 2 * transformed_dimension;

		for (uint64_t i = thread_id; i < length; i += 32) {
			for (uint64_t j = 0; j < path_dimension; ++j) {
				data_out_[i * path_dimension + j] = derivs_[i * twice_transformed_dimension + j];
				data_out_[i * path_dimension + j] += derivs_[i * twice_transformed_dimension + path_dimension + j];
			}
		}

		for (uint64_t i = thread_id; i < length - 1; i += 32) {
			for (uint64_t j = 0; j < twice_dimension; ++j) {
				data_out_[i * path_dimension + j] += derivs_[i * twice_transformed_dimension + transformed_dimension + j];
			}
		}
	}
	else {
		for (uint64_t i = thread_id; i < length; i += 32) {
			for (uint64_t j = 0; j < path_dimension; ++j) {
				data_out_[i * path_dimension + j] = derivs_[i * transformed_dimension + j];
			}
		}
	}
}

template<typename T>
void transform_path_backprop_(
	const T* derivs,
	T* data_out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length_,
	bool time_aug_,
	bool lead_lag_,
	T end_time
) {
	const uint64_t path_size_ = dimension_ * length_;

	const uint64_t transformed_length_ = lead_lag_ ? 2 * length_ - 1 : length_;
	const uint64_t transformed_dimension_ = (lead_lag_ ? 2 * dimension_ : dimension_) + (time_aug_ ? 1 : 0);
	const uint64_t transformed_path_size_ = transformed_length_ * transformed_dimension_;

	cudaMemcpyToSymbol(path_dimension, &dimension_, sizeof(uint64_t));
	cudaMemcpyToSymbol(length, &length_, sizeof(uint64_t));
	cudaMemcpyToSymbol(time_aug, &time_aug_, sizeof(bool));
	cudaMemcpyToSymbol(lead_lag, &lead_lag_, sizeof(bool));
	cudaMemcpyToSymbol(path_size, &path_size_, sizeof(uint64_t));

	cudaMemcpyToSymbol(transformed_dimension, &transformed_dimension_, sizeof(uint64_t));
	cudaMemcpyToSymbol(transformed_length, &transformed_length_, sizeof(uint64_t));
	cudaMemcpyToSymbol(transformed_path_size, &transformed_path_size_, sizeof(uint64_t));

	if (!(lead_lag_ || time_aug_)) {
		cudaMemcpy(data_out, derivs, batch_size_ * path_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	else {
		transform_path_backprop_internal_ << <static_cast<unsigned int>(batch_size_), 32U >> > (derivs, data_out, end_time);
	}

	check_cuda_error();
}

#include "cu_macros.h"

extern "C" {

	CUSIG_API int transform_path_cuda_f(const float* const data_in, float* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const float end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_<float>(data_in, data_out, 1, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int transform_path_cuda_d(const double* const data_in, double* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_<double>(data_in, data_out, 1, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_transform_path_cuda_f(const float* const data_in, float* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const float end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_<float>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_transform_path_cuda_d(const double* const data_in, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_<double>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int transform_path_backprop_cuda_f(const float* const derivs, float* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const float end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_backprop_<float>(derivs, data_out, 1, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int transform_path_backprop_cuda_d(const double* const derivs, double* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_backprop_<double>(derivs, data_out, 1, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_transform_path_backprop_cuda_f(const float* const derivs, float* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const float end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_backprop_<float>(derivs, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int batch_transform_path_backprop_cuda_d(const double* const derivs, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept {
		CUSIG_SAFE_CALL(transform_path_backprop_<double>(derivs, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}
}