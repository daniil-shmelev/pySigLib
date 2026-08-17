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

#include "cusig.h"
#include "cu_runtime_utils.h"
#include "cu_path_transforms.h"

#include <cstdint>

namespace {
struct TransformPathParams {
	uint64_t path_dimension;
	uint64_t length;
	uint64_t path_size;
	uint64_t transformed_dimension;
	uint64_t transformed_length;
	uint64_t transformed_path_size;
	bool     time_aug;
	bool     lead_lag;
};

static TransformPathParams make_params(uint64_t dimension_, uint64_t length_, bool time_aug_, bool lead_lag_) {
	TransformPathParams p;
	p.path_dimension         = dimension_;
	p.length                 = length_;
	p.path_size              = dimension_ * length_;
	p.transformed_length     = lead_lag_ ? 2 * length_ - 1 : length_;
	p.transformed_dimension  = (lead_lag_ ? 2 * dimension_ : dimension_) + (time_aug_ ? 1 : 0);
	p.transformed_path_size  = p.transformed_length * p.transformed_dimension;
	p.time_aug               = time_aug_;
	p.lead_lag               = lead_lag_;
	return p;
}
}  // anonymous namespace

template<typename T>
__global__ void transform_path_internal_(
	const T* data_in,
	T* data_out,
	TransformPathParams p,
	T end_time
) {
	const int thread_id = threadIdx.x;

	const T* const data_in_ = data_in + blockIdx.x * p.path_size;
	T* const data_out_ = data_out + blockIdx.x * p.transformed_path_size;

	if (!(p.time_aug || p.lead_lag)) {
		for (uint64_t i = thread_id; i < p.path_size; i += blockDim.x)
			data_out_[i] = static_cast<T>(data_in_[i]);
	}

	if (p.lead_lag) {
		const uint64_t twice_dimension = 2 * p.path_dimension;
		const uint64_t twice_transformed_dimension = 2 * p.transformed_dimension;

		for (uint64_t i = thread_id; i < p.length; i += blockDim.x) {
			for (uint64_t j = 0; j < p.path_dimension; ++j) {
				data_out_[i * twice_transformed_dimension + j] = static_cast<T>(data_in_[i * p.path_dimension + j]);
				data_out_[i * twice_transformed_dimension + j + p.path_dimension] = static_cast<T>(data_in_[i * p.path_dimension + j]);
			}
		}

		for (uint64_t i = thread_id; i < p.length - 1; i += blockDim.x) {
			for (uint64_t j = 0; j < twice_dimension; ++j) {
				data_out_[i * twice_transformed_dimension + p.transformed_dimension + j] = static_cast<T>(data_in_[i * p.path_dimension + j]);
			}
		}
	}
	else {
		for (uint64_t i = thread_id; i < p.length; i += blockDim.x) {
			for (uint64_t j = 0; j < p.path_dimension; ++j) {
				data_out_[i * p.transformed_dimension + j] = static_cast<T>(data_in_[i * p.path_dimension + j]);
			}
		}
	}

	if (p.time_aug) {
		const T scale = end_time / (p.transformed_length - 1);

		for (uint64_t i = thread_id; i < p.transformed_length; i += blockDim.x) {
			data_out_[(i + 1) * p.transformed_dimension - 1] = i * scale;
		}
	}
}

template<typename T>
void cu_transform_path_(
	const T* data_in,
	T* data_out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length_,
	bool time_aug_,
	bool lead_lag_,
	T end_time
) {
	const TransformPathParams p = make_params(dimension_, length_, time_aug_, lead_lag_);

	transform_path_internal_ << <static_cast<unsigned int>(batch_size_), 32U >> > (data_in, data_out, p, end_time);

	check_cuda_kernel_launch();
}

template<typename T>
__global__ void transform_path_backprop_internal_(
	const T* derivs,
	T* data_out,
	TransformPathParams p,
	T end_time
) {
	const int thread_id = threadIdx.x;

	const T* const derivs_ = derivs + blockIdx.x * p.transformed_path_size;
	T* const data_out_ = data_out + blockIdx.x * p.path_size;

	if (p.lead_lag) {
		const uint64_t td = p.transformed_dimension;
		const uint64_t D  = p.path_dimension;
		const uint64_t L  = p.length;

		for (uint64_t k = thread_id; k < L; k += blockDim.x) {
			for (uint64_t j = 0; j < D; ++j) {
				T sum  = derivs_[(2 * k) * td + j];
				sum   += derivs_[(2 * k) * td + D + j];
				if (k + 1 < L) {
					sum += derivs_[(2 * k + 1) * td + j];
				}
				if (k >= 1) {
					sum += derivs_[(2 * k - 1) * td + D + j];
				}
				data_out_[k * D + j] = sum;
			}
		}
	}
	else {
		for (uint64_t i = thread_id; i < p.length; i += blockDim.x) {
			for (uint64_t j = 0; j < p.path_dimension; ++j) {
				data_out_[i * p.path_dimension + j] = derivs_[i * p.transformed_dimension + j];
			}
		}
	}
}

template<typename T>
void cu_transform_path_backprop_(
	const T* derivs,
	T* data_out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length_,
	bool time_aug_,
	bool lead_lag_,
	T end_time
) {
	if (!(lead_lag_ || time_aug_)) {
		cudaMemcpy(data_out, derivs, batch_size_ * dimension_ * length_ * sizeof(T), cudaMemcpyDeviceToDevice);
	}
	else {
		const TransformPathParams p = make_params(dimension_, length_, time_aug_, lead_lag_);
		transform_path_backprop_internal_ << <static_cast<unsigned int>(batch_size_), 32U >> > (derivs, data_out, p, end_time);
	}

	check_cuda_kernel_launch();
}

template void cu_transform_path_<float>(const float*, float*, uint64_t, uint64_t, uint64_t, bool, bool, float);
template void cu_transform_path_<double>(const double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, double);
template void cu_transform_path_backprop_<float>(const float*, float*, uint64_t, uint64_t, uint64_t, bool, bool, float);
template void cu_transform_path_backprop_<double>(const double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, double);

#include "cu_macros.h"

extern "C" {


	CUSIG_API int transform_path_cuda_f(const float* const data_in, float* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const float end_time) noexcept {
		CUSIG_SAFE_CALL(cu_transform_path_<float>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int transform_path_cuda_d(const double* const data_in, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept {
		CUSIG_SAFE_CALL(cu_transform_path_<double>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}


	CUSIG_API int transform_path_backprop_cuda_f(const float* const derivs, float* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const float end_time) noexcept {
		CUSIG_SAFE_CALL(cu_transform_path_backprop_<float>(derivs, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}

	CUSIG_API int transform_path_backprop_cuda_d(const double* const derivs, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept {
		CUSIG_SAFE_CALL(cu_transform_path_backprop_<double>(derivs, data_out, batch_size, dimension, length, time_aug, lead_lag, end_time));
	}
}
