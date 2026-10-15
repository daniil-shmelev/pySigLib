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

#include "cp_simd_config.h"
#include "hwy/highway.h"
#if defined(CPSIG_CP_PATH_TRANSFORMS_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_CP_PATH_TRANSFORMS_H_
#undef CPSIG_CP_PATH_TRANSFORMS_H_
#else
#define CPSIG_CP_PATH_TRANSFORMS_H_
#endif
#include "cppch.h"
#include "cpsig.h"
#include "macros.h"
#include "cp_path.h"
#include "cp_simd-inl.h"
#include "hwy/contrib/algo/copy-inl.h"
#include "multithreading.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template<std::floating_point T>
void transform_path_(
	const T* data_in,
	T* data_out,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time
) {
	// Note that here 'dimension' and 'length' refer to the underlying path, not the transformed one.
	// Also note that if both time_aug and lead_lag are false, we still copy memory to data_out

	const Path<T> dummy_path(nullptr, dimension, length, time_aug, lead_lag, end_time);
	const uint64_t transformed_dimension = dummy_path.dimension();
	const uint64_t transformed_length = dummy_path.length();

	if (!(time_aug || lead_lag)) {
		const uint64_t path_size = dimension * length;
		if (path_size == 0 || data_in == data_out)
			return;
		const uintptr_t input_address = reinterpret_cast<uintptr_t>(data_in);
		const uintptr_t output_address = reinterpret_cast<uintptr_t>(data_out);
		const uintptr_t distance = input_address < output_address
			? output_address - input_address : input_address - output_address;
		if (distance >= path_size * sizeof(T)) {
#ifdef _MSC_VER
			std::memcpy(data_out, data_in, path_size * sizeof(T));
#else
			const Simd<T> d;
			constexpr uint64_t lanes = vec_batch_lanes<T>;
			const uint64_t prefix = std::min(path_size,
				((uintptr_t(0) - output_address) & (lanes * sizeof(T) - 1)) / sizeof(T));
			uint64_t i = 0;
			for (; i < prefix; ++i)
				data_out[i] = data_in[i];
			for (; i + 4 * lanes <= path_size; i += 4 * lanes) {
				const auto v0 = hn::LoadU(d, data_in + i);
				const auto v1 = hn::LoadU(d, data_in + i + lanes);
				const auto v2 = hn::LoadU(d, data_in + i + 2 * lanes);
				const auto v3 = hn::LoadU(d, data_in + i + 3 * lanes);
				hn::Store(v0, d, data_out + i);
				hn::Store(v1, d, data_out + i + lanes);
				hn::Store(v2, d, data_out + i + 2 * lanes);
				hn::Store(v3, d, data_out + i + 3 * lanes);
			}
			hn::Copy(d, data_in + i, path_size - i, data_out + i);
#endif
			return;
		}
		T* out_ptr = data_out;
		const T* in_ptr = data_in;
		for (uint64_t i = 0; i < path_size; ++i)
			*(out_ptr++) = *(in_ptr++);
		return;
	}

	if (lead_lag) {
		const uint64_t twice_dimension = 2 * dimension;
		uint64_t dim_update = 2 * transformed_dimension - dimension;

		T* out_ptr = data_out;
		T* out_ptr_2 = data_out + dimension;
		const T* in_ptr = data_in;

		for (uint64_t i = 0; i < length; ++i) {
			for (uint64_t j = 0; j < dimension; ++j) {
				*(out_ptr++) = *in_ptr;
				*(out_ptr_2++) = *(in_ptr++);
			}
			out_ptr += dim_update;
			out_ptr_2 += dim_update;
		}

		dim_update = 2 * transformed_dimension - twice_dimension;
		out_ptr = data_out + transformed_dimension;
		in_ptr = data_in;

		for (uint64_t i = 0; i < length - 1; ++i) {
			for (uint64_t j = 0; j < twice_dimension; ++j) {
				*(out_ptr++) = *(in_ptr++);
			}
			out_ptr += dim_update;
			in_ptr -= dimension;
		}
	}
	else {
		const uint64_t dim_update = transformed_dimension - dimension;
		T* out_ptr = data_out;
		const T* in_ptr = data_in;

		for (uint64_t i = 0; i < length; ++i) {
			for (uint64_t j = 0; j < dimension; ++j) {
				*(out_ptr++) = *(in_ptr++);
			}
			out_ptr += dim_update;
		}
	}

	if (time_aug) {
		T* const out_ptr = data_out + transformed_dimension - 1;
		const T scale = end_time / (transformed_length - 1);

		for (uint64_t i = 0; i < transformed_length; ++i) {
			out_ptr[i * transformed_dimension] = i * scale;
		}
	}

}

template<std::floating_point T>
void batch_transform_path_(
	const T* data_in,
	T* data_out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time,
	int n_jobs
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("transform_path received path of dimension 0"); }

	const Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t result_length = dummy_path_obj.length() * dummy_path_obj.dimension();

	const uint64_t flat_path_length = dimension * length;

	if (n_jobs == 1 && !(time_aug || lead_lag)) {
		transform_path_<T>(data_in, data_out, dimension, batch_size * length,
			false, false, end_time);
		return;
	}

	auto transform_func = [&](const T* const path_ptr, T* const out_ptr) {
		transform_path_<T>(path_ptr, out_ptr, dimension, length, time_aug, lead_lag, end_time);
	};

	multi_threaded_batch(transform_func, batch_size, n_jobs,
		make_batch(data_in, flat_path_length), make_batch(data_out, result_length));
	return;
}

template<std::floating_point T>
void transform_path_backprop_(
	const T* derivs,
	T* data_out,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time
) {
	// Note that here 'dimension' and 'length' refer to the underlying path, not the transformed one.
	// Also note that if both time_aug and lead_lag are false, we still copy memory to data_out

	const Path<T> dummy_path(nullptr, dimension, length, time_aug, lead_lag, end_time);
	const uint64_t transformed_dimension = dummy_path.dimension();
	const uint64_t dim_update = 2 * transformed_dimension - dimension;

	if (!(lead_lag || time_aug)) {
		std::memcpy(data_out, derivs, length * dimension * sizeof(T));
		return;
	}

	if (lead_lag) {
		for (uint64_t i = 0; i < length; ++i) {
			std::memcpy(data_out + i * dimension, derivs + 2 * i * transformed_dimension, dimension * sizeof(T));
		}

		T* out_ptr = data_out;
		const T* derivs_ptr = derivs + transformed_dimension;
		for (uint64_t i = 0; i < length - 1; ++i) {
			for (uint64_t j = 0; j < dimension; ++j) {
				*(out_ptr++) += *(derivs_ptr++);
			}
			derivs_ptr += dim_update;
		}

		out_ptr = data_out;
		derivs_ptr = derivs + dimension;
		for (uint64_t i = 0; i < length; ++i) {
			for (uint64_t j = 0; j < dimension; ++j) {
				*(out_ptr++) += *(derivs_ptr++);
			}
			derivs_ptr += dim_update;
		}

		out_ptr = data_out + dimension;
		derivs_ptr = derivs + transformed_dimension + dimension;
		for (uint64_t i = 1; i < length; ++i) {
			for (uint64_t j = 0; j < dimension; ++j) {
				*(out_ptr++) += *(derivs_ptr++);
			}
			derivs_ptr += dim_update;
		}
	}
	else {
		for (uint64_t i = 0; i < length; ++i) {
			std::memcpy(data_out + i * dimension, derivs + i * transformed_dimension, dimension * sizeof(T));
		}
	}
}

template<std::floating_point T>
void batch_transform_path_backprop_(
	const T* derivs,
	T* data_out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	T end_time,
	int n_jobs
) {
	// Note that here 'dimension' and 'length' refer to the underlying path, not the transformed one.

	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("transform_path_backprop received path of dimension 0"); }

	const Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t result_length = length * dimension;

	const uint64_t flat_path_length = dummy_path_obj.length() * dummy_path_obj.dimension();

	auto transform_func = [&](const T* const derivs_ptr, T* const out_ptr) {
		transform_path_backprop_(derivs_ptr, out_ptr, dimension, length, time_aug, lead_lag, end_time);
	};

	multi_threaded_batch(transform_func, batch_size, n_jobs,
		make_batch(derivs, flat_path_length), make_batch(data_out, result_length));
	return;
}

}  // namespace HWY_NAMESPACE
}  // namespace cpsig
HWY_AFTER_NAMESPACE();
#endif
