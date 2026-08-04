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

#pragma once

template<typename T>
__device__ void sparse_branched_hopf_convolution_block_(
	const T* X,
	const T* Y,
	T* out,
	uint32_t cache_size,
	const uint32_t* coprod_data,
	const uint32_t* coprod_offsets,
	uint32_t tid
) {
	if (tid == 0)
		out[0] = X[0] * Y[0];
	const uint32_t local = tid + 1;
	if (local < cache_size) {
		T value = X[local] * Y[0] + X[0] * Y[local];
		uint32_t pos = coprod_offsets[local];
		const uint32_t end = coprod_offsets[local + 1];
		while (pos < end) {
			const uint32_t num_forest = coprod_data[pos++];
			T term = Y[coprod_data[pos++]];
			#pragma unroll 4
			for (uint32_t j = 0; j < num_forest; ++j)
				term *= X[coprod_data[pos++]];
			value += term;
		}
		out[local] = value;
	}
	__syncthreads();
}

template<typename T>
__device__ void sparse_branched_hopf_convolution_deriv_block_(
	const T* X,
	const T* Y,
	const T* d_out,
	T* d_X,
	T* d_Y,
	uint32_t cache_size,
	const uint32_t* coprod_data,
	const uint32_t* coprod_offsets,
	uint32_t tid
) {
	if (tid == 0) {
		myAtomicAdd(&d_X[0], d_out[0] * Y[0]);
		myAtomicAdd(&d_Y[0], d_out[0] * X[0]);
	}
	const uint32_t local = tid + 1;
	if (local < cache_size) {
		const T d = d_out[local];
		if (d != T(0)) {
			myAtomicAdd(&d_X[local], d * Y[0]);
			myAtomicAdd(&d_Y[0], d * X[local]);
			myAtomicAdd(&d_X[0], d * Y[local]);
			myAtomicAdd(&d_Y[local], d * X[0]);

			uint32_t pos = coprod_offsets[local];
			const uint32_t end = coprod_offsets[local + 1];
			while (pos < end) {
				const uint32_t num_forest = coprod_data[pos++];
				const uint32_t trunk = coprod_data[pos++];
				const uint32_t forest_start = pos;
				T forest_product = T(1);
				for (uint32_t j = 0; j < num_forest; ++j)
					forest_product *= X[coprod_data[pos++]];
				myAtomicAdd(&d_Y[trunk], d * forest_product);

				for (uint32_t k = 0; k < num_forest; ++k) {
					T partial = d * Y[trunk];
					for (uint32_t j = 0; j < num_forest; ++j) {
						if (j != k)
							partial *= X[coprod_data[forest_start + j]];
					}
					myAtomicAdd(&d_X[coprod_data[forest_start + k]], partial);
				}
			}
		}
	}
	__syncthreads();
}

template<typename T>
__device__ void sparse_branched_add_correction_block_(
	T* out,
	const T* correction,
	const uint32_t* correction_offsets,
	const uint32_t* correction_locals,
	uint32_t num_corrections,
	uint32_t tid
) {
	if (correction != nullptr) {
		for (uint32_t i = tid; i < num_corrections; i += blockDim.x)
			myAtomicAdd(&out[correction_locals[i]], correction[correction_offsets[i]]);
	}
	__syncthreads();
}

template<typename T>
__device__ void sparse_linear_branched_sig_block_(
	const T* increment,
	T* out,
	uint32_t cache_size,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	uint32_t tid
) {
	if (tid == 0)
		out[0] = T(1);
	const uint32_t local = tid + 1;
	if (local < cache_size) {
		T product = T(1);
		const uint32_t start = labels_offsets[local];
		const uint32_t end = labels_offsets[local + 1];
		#pragma unroll 8
		for (uint32_t pos = start; pos < end; ++pos)
			product *= increment[labels_data[pos]];
		out[local] = product * inv_factorial[local];
	}
	__syncthreads();
}

template<typename T>
__device__ void sparse_local_branched_sig_block_(
	const T* increment,
	T* out,
	T* local_log,
	T* power,
	T* next_power,
	uint32_t dimension,
	const T* correction,
	uint32_t cache_size,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	const uint32_t* coprod_data,
	const uint32_t* coprod_offsets,
	const uint32_t* leaf_indices,
	const uint32_t* correction_offsets,
	const uint32_t* correction_locals,
	uint32_t num_corrections,
	uint32_t max_nodes,
	uint32_t tid
) {
	if (num_corrections == 0) {
		sparse_linear_branched_sig_block_(increment, out, cache_size,
			labels_data, labels_offsets, inv_factorial, tid);
		return;
	}

	if (max_nodes <= 2) {
		sparse_linear_branched_sig_block_(increment, out, cache_size,
			labels_data, labels_offsets, inv_factorial, tid);
		sparse_branched_add_correction_block_(out, correction,
			correction_offsets, correction_locals, num_corrections, tid);
		return;
	}

	for (uint32_t i = tid; i < cache_size; i += blockDim.x)
		local_log[i] = T(0);
	__syncthreads();
	for (uint32_t d = tid; d < dimension; d += blockDim.x) {
		const uint32_t local = leaf_indices[d];
		if (local != 0)
			local_log[local] = increment[d];
	}
	__syncthreads();
	sparse_branched_add_correction_block_(local_log, correction,
		correction_offsets, correction_locals, num_corrections, tid);

	for (uint32_t i = tid; i < cache_size; i += blockDim.x) {
		out[i] = T(0);
		power[i] = local_log[i];
	}
	if (tid == 0)
		out[0] = T(1);
	__syncthreads();

	T inv_k_factorial = T(1);
	T* current = power;
	T* next = next_power;
	for (uint32_t k = 1; k <= max_nodes; ++k) {
		inv_k_factorial /= static_cast<T>(k);
		for (uint32_t i = tid; i < cache_size; i += blockDim.x)
			out[i] += inv_k_factorial * current[i];
		__syncthreads();
		if (k < max_nodes) {
			sparse_branched_hopf_convolution_block_(current, local_log, next,
				cache_size, coprod_data, coprod_offsets, tid);
			T* swap = current;
			current = next;
			next = swap;
		}
	}
}

template<typename T>
__device__ void sparse_linear_branched_sig_deriv_block_(
	const T* local_derivs,
	const T* increment,
	T* increment_derivs,
	uint32_t dimension,
	uint32_t cache_size,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	uint32_t tid
) {
	for (uint32_t d = tid; d < dimension; d += blockDim.x)
		increment_derivs[d] = T(0);
	__syncthreads();

	const uint32_t local = tid + 1;
	if (local < cache_size) {
		const T derivative = local_derivs[local];
		if (derivative != T(0)) {
			const uint32_t start = labels_offsets[local];
			const uint32_t end = labels_offsets[local + 1];
			const T base = derivative * inv_factorial[local];
			for (uint32_t k = start; k < end; ++k) {
				T partial = base;
				for (uint32_t j = start; j < end; ++j) {
					if (j != k)
						partial *= increment[labels_data[j]];
				}
				myAtomicAdd(&increment_derivs[labels_data[k]], partial);
			}
		}
	}
	__syncthreads();
}

template<typename T>
__device__ void sparse_local_branched_sig_deriv_block_(
	const T* local_derivs,
	const T* increment,
	T* increment_derivs,
	T* local_log,
	T* powers,
	T* power_derivs,
	T* d_local_log,
	uint32_t dimension,
	const T* correction,
	uint32_t cache_size,
	const uint8_t* labels_data,
	const uint32_t* labels_offsets,
	const T* inv_factorial,
	const uint32_t* coprod_data,
	const uint32_t* coprod_offsets,
	const uint32_t* leaf_indices,
	const uint32_t* correction_offsets,
	const uint32_t* correction_locals,
	uint32_t num_corrections,
	uint32_t max_nodes,
	uint32_t tid
) {
	if (num_corrections == 0 || max_nodes <= 2) {
		sparse_linear_branched_sig_deriv_block_(local_derivs, increment,
			increment_derivs, dimension, cache_size, labels_data,
			labels_offsets, inv_factorial, tid);
		return;
	}

	for (uint32_t i = tid; i < cache_size; i += blockDim.x)
		local_log[i] = T(0);
	__syncthreads();
	for (uint32_t d = tid; d < dimension; d += blockDim.x) {
		const uint32_t local = leaf_indices[d];
		if (local != 0)
			local_log[local] = increment[d];
	}
	__syncthreads();
	sparse_branched_add_correction_block_(local_log, correction,
		correction_offsets, correction_locals, num_corrections, tid);

	for (uint32_t i = tid; i < cache_size; i += blockDim.x)
		powers[i] = local_log[i];
	__syncthreads();
	for (uint32_t k = 2; k <= max_nodes; ++k)
		sparse_branched_hopf_convolution_block_(
			powers + (k - 2) * cache_size, local_log,
			powers + (k - 1) * cache_size, cache_size,
			coprod_data, coprod_offsets, tid);

	for (uint32_t i = tid; i < max_nodes * cache_size; i += blockDim.x)
		power_derivs[i] = T(0);
	for (uint32_t i = tid; i < cache_size; i += blockDim.x)
		d_local_log[i] = T(0);
	__syncthreads();

	T inv_k_factorial = T(1);
	for (uint32_t k = 1; k <= max_nodes; ++k) {
		inv_k_factorial /= static_cast<T>(k);
		for (uint32_t local = tid + 1; local < cache_size; local += blockDim.x)
			power_derivs[(k - 1) * cache_size + local]
				+= inv_k_factorial * local_derivs[local];
	}
	__syncthreads();

	for (uint32_t k = max_nodes; k > 1; --k)
		sparse_branched_hopf_convolution_deriv_block_(
			powers + (k - 2) * cache_size, local_log,
			power_derivs + (k - 1) * cache_size,
			power_derivs + (k - 2) * cache_size, d_local_log,
			cache_size, coprod_data, coprod_offsets, tid);

	for (uint32_t i = tid; i < cache_size; i += blockDim.x)
		d_local_log[i] += power_derivs[i];
	__syncthreads();
	for (uint32_t d = tid; d < dimension; d += blockDim.x) {
		const uint32_t local = leaf_indices[d];
		increment_derivs[d] = local == 0 ? T(0) : d_local_log[local];
	}
	__syncthreads();
}

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_coef_forward_kernel_(
	const T* __restrict__ path,
	T* __restrict__ state,
	T* __restrict__ out,
	uint32_t dimension,
	uint32_t steps,
	uint64_t path_stride,
	uint32_t cache_size,
	const uint32_t* __restrict__ target_indices,
	uint32_t num_targets,
	const uint8_t* __restrict__ labels_data,
	const uint32_t* __restrict__ labels_offsets,
	const T* __restrict__ inv_factorial,
	const uint32_t* __restrict__ global_coprod_data,
	const uint32_t* __restrict__ global_coprod_offsets,
	const uint32_t* __restrict__ global_order_index,
	const uint32_t* __restrict__ leaf_indices,
	const uint32_t* __restrict__ correction_offsets,
	const uint32_t* __restrict__ correction_locals,
	uint32_t num_corrections,
	uint32_t max_nodes,
	uint32_t coprod_data_len,
	const T* __restrict__ correction,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	const uint32_t batch = blockIdx.x;
	const uint32_t tid = threadIdx.x;
	const bool has_correction = num_corrections != 0;

	extern __shared__ char shared[];
	T* shared_state = reinterpret_cast<T*>(shared);
	T* batch_state = state == nullptr
		? shared_state : state + static_cast<uint64_t>(batch) * cache_size;
	T* temp = state == nullptr ? shared_state + cache_size : shared_state;
	T* local_log = has_correction ? temp + cache_size : temp;
	T* power = has_correction ? local_log + cache_size : temp;
	T* next_power = has_correction ? power + cache_size : temp;
	T* increment = has_correction ? next_power + cache_size : temp + cache_size;
	uint32_t* coprod_data = reinterpret_cast<uint32_t*>(increment + dimension);
	uint32_t* coprod_offsets = coprod_data + coprod_data_len;
	uint32_t* order_index = coprod_offsets + cache_size + 1;

	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		coprod_data[i] = global_coprod_data[i];
	for (uint32_t i = tid; i < cache_size + 1; i += blockDim.x)
		coprod_offsets[i] = global_coprod_offsets[i];
	for (uint32_t i = tid; i < max_nodes + 2; i += blockDim.x)
		order_index[i] = global_order_index[i];

	for (uint32_t i = tid; i < cache_size; i += blockDim.x)
		batch_state[i] = T(0);
	if (tid == 0)
		batch_state[0] = T(1);
	__syncthreads();

	const T* batch_path = path + static_cast<uint64_t>(batch) * path_stride;
	const T* batch_correction = has_correction
		? correction + static_cast<uint64_t>(batch) * correction_batch_stride : nullptr;
	for (uint32_t segment = 0; segment < steps; ++segment) {
		for (uint32_t d = tid; d < dimension; d += blockDim.x)
			increment[d] = batch_path[(segment + 1) * dimension + d]
				- batch_path[segment * dimension + d];
		__syncthreads();
		const T* segment_correction = has_correction
			? batch_correction + static_cast<uint64_t>(segment) * correction_segment_stride
			: nullptr;
		T* local = segment == 0 ? batch_state : temp;
		sparse_local_branched_sig_block_(increment, local, local_log, power,
			next_power, dimension, segment_correction, cache_size, labels_data,
			labels_offsets, inv_factorial, coprod_data, coprod_offsets,
			leaf_indices, correction_offsets, correction_locals,
			num_corrections, max_nodes, tid);

		if (segment > 0) {
			const uint32_t local_index = tid + 1;
			for (uint32_t order = max_nodes; order > 0; --order) {
				if (local_index >= order_index[order]
					&& local_index < order_index[order + 1]) {
					T value = batch_state[local_index] + temp[local_index];
					uint32_t pos = coprod_offsets[local_index];
					const uint32_t end = coprod_offsets[local_index + 1];
					while (pos < end) {
						const uint32_t num_forest = coprod_data[pos++];
						T term = temp[coprod_data[pos++]];
						for (uint32_t j = 0; j < num_forest; ++j)
							term *= batch_state[coprod_data[pos++]];
						value += term;
					}
					batch_state[local_index] = value;
				}
				__syncthreads();
			}
		}
	}

	if (out != nullptr) {
		T* batch_out = out + static_cast<uint64_t>(batch) * num_targets;
		for (uint32_t i = tid; i < num_targets; i += blockDim.x)
			batch_out[i] = batch_state[target_indices[i]];
	}
}

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_coef_backprop_kernel_(
	const T* __restrict__ path,
	T* __restrict__ path_derivs,
	const T* __restrict__ state,
	const T* __restrict__ coefs,
	const T* __restrict__ derivs,
	uint32_t dimension,
	uint32_t steps,
	uint64_t path_stride,
	uint32_t cache_size,
	const uint32_t* __restrict__ target_indices,
	uint32_t num_targets,
	const uint8_t* __restrict__ labels_data,
	const uint32_t* __restrict__ labels_offsets,
	const T* __restrict__ inv_factorial,
	const uint32_t* __restrict__ global_coprod_data,
	const uint32_t* __restrict__ global_coprod_offsets,
	const uint32_t* __restrict__ global_order_index,
	const uint32_t* __restrict__ leaf_indices,
	const uint32_t* __restrict__ correction_offsets,
	const uint32_t* __restrict__ correction_locals,
	uint32_t num_corrections,
	uint32_t max_nodes,
	uint32_t coprod_data_len,
	const T* __restrict__ correction,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	const uint32_t batch = blockIdx.x;
	const uint32_t tid = threadIdx.x;
	const bool has_correction = num_corrections != 0;

	extern __shared__ char shared[];
	T* batch_state = reinterpret_cast<T*>(shared);
	T* state_derivs = batch_state + cache_size;
	T* local_sig = state_derivs + cache_size;
	T* local_derivs = local_sig + cache_size;
	T* increment = local_derivs + cache_size;
	T* increment_derivs = increment + dimension;
	T* local_log = increment_derivs + dimension;
	T* powers = has_correction ? local_log + cache_size : local_log;
	T* power_derivs = has_correction
		? powers + static_cast<uint64_t>(max_nodes) * cache_size : powers;
	T* d_local_log = has_correction
		? power_derivs + static_cast<uint64_t>(max_nodes) * cache_size : power_derivs;
	T* arrays_end = has_correction ? d_local_log + cache_size : increment_derivs + dimension;
	uint32_t* coprod_data = reinterpret_cast<uint32_t*>(arrays_end);
	uint32_t* coprod_offsets = coprod_data + coprod_data_len;
	uint32_t* order_index = coprod_offsets + cache_size + 1;

	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		coprod_data[i] = global_coprod_data[i];
	for (uint32_t i = tid; i < cache_size + 1; i += blockDim.x)
		coprod_offsets[i] = global_coprod_offsets[i];
	for (uint32_t i = tid; i < max_nodes + 2; i += blockDim.x)
		order_index[i] = global_order_index[i];

	const uint64_t state_offset = static_cast<uint64_t>(batch) * cache_size;
	for (uint32_t i = tid; i < cache_size; i += blockDim.x) {
		batch_state[i] = state[state_offset + i];
		state_derivs[i] = T(0);
	}
	__syncthreads();
	const T* batch_coefs = coefs + static_cast<uint64_t>(batch) * num_targets;
	const T* batch_derivs = derivs + static_cast<uint64_t>(batch) * num_targets;
	for (uint32_t i = tid; i < num_targets; i += blockDim.x) {
		const uint32_t local = target_indices[i];
		batch_state[local] = batch_coefs[i];
		if (local != 0)
			myAtomicAdd(&state_derivs[local], batch_derivs[i]);
	}

	const T* batch_path = path + static_cast<uint64_t>(batch) * path_stride;
	T* batch_path_derivs = path_derivs + static_cast<uint64_t>(batch) * path_stride;
	for (uint64_t i = tid; i < path_stride; i += blockDim.x)
		batch_path_derivs[i] = T(0);
	__syncthreads();

	const T* batch_correction = has_correction
		? correction + static_cast<uint64_t>(batch) * correction_batch_stride : nullptr;
	for (int32_t segment = static_cast<int32_t>(steps) - 1; segment >= 0; --segment) {
		for (uint32_t d = tid; d < dimension; d += blockDim.x)
			increment[d] = batch_path[(segment + 1) * dimension + d]
				- batch_path[segment * dimension + d];
		__syncthreads();
		const T* segment_correction = has_correction
			? batch_correction + static_cast<uint64_t>(segment) * correction_segment_stride
			: nullptr;
		sparse_local_branched_sig_block_(increment, local_sig, local_log,
			powers, power_derivs, dimension, segment_correction, cache_size,
			labels_data, labels_offsets, inv_factorial, coprod_data,
			coprod_offsets, leaf_indices, correction_offsets, correction_locals,
			num_corrections, max_nodes, tid);

		const uint32_t local_index = tid + 1;
		if (segment > 0) {
			for (uint32_t order = 1; order <= max_nodes; ++order) {
				if (local_index >= order_index[order]
					&& local_index < order_index[order + 1]) {
					T value = batch_state[local_index] - local_sig[local_index];
					uint32_t pos = coprod_offsets[local_index];
					const uint32_t end = coprod_offsets[local_index + 1];
					while (pos < end) {
						const uint32_t num_forest = coprod_data[pos++];
						T term = local_sig[coprod_data[pos++]];
						for (uint32_t j = 0; j < num_forest; ++j)
							term *= batch_state[coprod_data[pos++]];
						value -= term;
					}
					batch_state[local_index] = value;
				}
				__syncthreads();
			}

			if (tid == 0)
				local_derivs[0] = T(0);
			if (local_index < cache_size)
				local_derivs[local_index] = state_derivs[local_index];
			__syncthreads();

			for (uint32_t order = 1; order <= max_nodes; ++order) {
				if (local_index >= order_index[order]
					&& local_index < order_index[order + 1]) {
					const T d_out = state_derivs[local_index];
					if (d_out != T(0)) {
						uint32_t pos = coprod_offsets[local_index];
						const uint32_t end = coprod_offsets[local_index + 1];
						while (pos < end) {
							const uint32_t num_forest = coprod_data[pos++];
							const uint32_t trunk = coprod_data[pos++];
							const uint32_t forest_start = pos;
							T forest_product = T(1);
							for (uint32_t j = 0; j < num_forest; ++j)
								forest_product *= batch_state[coprod_data[pos++]];
							myAtomicAdd(&local_derivs[trunk], d_out * forest_product);
							for (uint32_t k = 0; k < num_forest; ++k) {
								T partial = d_out * local_sig[trunk];
								for (uint32_t j = 0; j < num_forest; ++j) {
									if (j != k)
										partial *= batch_state[coprod_data[forest_start + j]];
								}
								myAtomicAdd(&state_derivs[coprod_data[forest_start + k]], partial);
							}
						}
					}
				}
				__syncthreads();
			}
		}
		else {
			for (uint32_t i = tid; i < cache_size; i += blockDim.x)
				local_derivs[i] = state_derivs[i];
			__syncthreads();
		}

		sparse_local_branched_sig_deriv_block_(local_derivs, increment,
			increment_derivs, local_log, powers, power_derivs, d_local_log,
			dimension, segment_correction, cache_size, labels_data, labels_offsets,
			inv_factorial, coprod_data, coprod_offsets, leaf_indices,
			correction_offsets, correction_locals, num_corrections, max_nodes, tid);

		for (uint32_t d = tid; d < dimension; d += blockDim.x) {
			batch_path_derivs[(segment + 1) * dimension + d] += increment_derivs[d];
			batch_path_derivs[segment * dimension + d] -= increment_derivs[d];
		}
		__syncthreads();
	}
}
