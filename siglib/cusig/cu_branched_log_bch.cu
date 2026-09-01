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

#include "cusig.h"
#include "cache_lifecycle/cu_branched_log_sig_cache.h"
#include "cu_log_sig_combine.h"
#include "cu_macros.h"
#include "cu_utils.h"
#include "../shared/branched_log_horner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct CuMkwBchDeviceData {
	const double* bch_coefficients = nullptr;
	const uint64_t* bch_left_factor = nullptr;
	const uint64_t* bch_right_factor = nullptr;
	CUDABchPlan bch_plan;
	const uint32_t* comm_k_ptr = nullptr;
	const uint32_t* comm_k_i = nullptr;
	const uint32_t* comm_k_j = nullptr;
	const int* comm_k_val = nullptr;
	const uint32_t* comm_a_ptr = nullptr;
	const uint32_t* comm_a_k = nullptr;
	const uint32_t* comm_a_partner = nullptr;
	const int* comm_a_signed_c = nullptr;
	const uint64_t* linear_a_ptr = nullptr;
	const uint32_t* linear_a_idx = nullptr;
	const uint64_t* linear_output_ptr = nullptr;
	const uint32_t* linear_output_idx = nullptr;
	const uint64_t* linear_op_ptr = nullptr;
	const uint32_t* linear_op_idx = nullptr;
	const uint64_t* linear_reverse_ptr = nullptr;
	const uint32_t* linear_reverse_idx = nullptr;
	const uint32_t* segment_idx = nullptr;
	const double* segment_coefficients = nullptr;
	const uint32_t* segment_label_offsets = nullptr;
	const uint8_t* segment_labels = nullptr;
	uint64_t m = 0;
	uint64_t m2 = 0;
	uint64_t segment_count = 0;
};

struct CuMkwBchCache {
	CUDABchCache bch;
	uint64_t* d_linear_output_ptr = nullptr;
	uint32_t* d_linear_output_idx = nullptr;
	uint64_t* d_linear_op_ptr = nullptr;
	uint32_t* d_linear_op_idx = nullptr;
	uint64_t* d_linear_reverse_ptr = nullptr;
	uint32_t* d_linear_reverse_idx = nullptr;
	uint32_t* d_segment_idx = nullptr;
	double* d_segment_coefficients = nullptr;
	uint32_t* d_segment_label_offsets = nullptr;
	uint8_t* d_segment_labels = nullptr;
	uint64_t segment_count = 0;
	uint64_t exact_forward_work = 0;
	uint64_t exact_zero_work = 0;
	bool prune_reverse = false;

	CuMkwBchCache() = default;
	CuMkwBchCache(const CuMkwBchCache&) = delete;
	CuMkwBchCache& operator=(const CuMkwBchCache&) = delete;

	~CuMkwBchCache() {
		if (d_linear_output_ptr) cudaFree(d_linear_output_ptr);
		if (d_linear_output_idx) cudaFree(d_linear_output_idx);
		if (d_linear_op_ptr) cudaFree(d_linear_op_ptr);
		if (d_linear_op_idx) cudaFree(d_linear_op_idx);
		if (d_linear_reverse_ptr) cudaFree(d_linear_reverse_ptr);
		if (d_linear_reverse_idx) cudaFree(d_linear_reverse_idx);
		if (d_segment_idx) cudaFree(d_segment_idx);
		if (d_segment_coefficients) cudaFree(d_segment_coefficients);
		if (d_segment_label_offsets) cudaFree(d_segment_label_offsets);
		if (d_segment_labels) cudaFree(d_segment_labels);
	}

	CuMkwBchDeviceData device_data() const noexcept {
		return {
			bch.d_bch_coefficients,
			bch.d_bch_left_factor,
			bch.d_bch_right_factor,
			bch.bch_plan,
			bch.d_comm_k_ptr,
			bch.d_comm_k_i,
			bch.d_comm_k_j,
			bch.d_comm_k_val,
			bch.d_comm_a_ptr,
			bch.d_comm_a_k,
			bch.d_comm_a_partner,
			bch.d_comm_a_signed_c,
			bch.d_linear_a_ptr,
			bch.d_linear_a_idx,
			d_linear_output_ptr,
			d_linear_output_idx,
			d_linear_op_ptr,
			d_linear_op_idx,
			d_linear_reverse_ptr,
			d_linear_reverse_idx,
			d_segment_idx,
			d_segment_coefficients,
			d_segment_label_offsets,
			d_segment_labels,
			bch.m,
			bch.m2,
			segment_count
		};
	}
};

template<typename T>
void upload_mkw_vector_(T*& device, const std::vector<T>& host) {
	if (host.empty())
		return;
	CudaBuf<T> buffer(host.size() * sizeof(T));
	CUDA_CHECK(cudaMemcpy(
		buffer.get(), host.data(), host.size() * sizeof(T),
		cudaMemcpyHostToDevice));
	device = buffer.release();
}

uint32_t narrow_mkw_u32_(uint64_t value, const char* message) {
	if (value > UINT32_MAX)
		throw std::overflow_error(message);
	return static_cast<uint32_t>(value);
}

struct HostMkwBchTables_ {
	std::vector<uint32_t> k_ptr;
	std::vector<uint32_t> k_i;
	std::vector<uint32_t> k_j;
	std::vector<int> k_val;
	std::vector<uint32_t> a_ptr;
	std::vector<uint32_t> a_k;
	std::vector<uint32_t> a_partner;
	std::vector<int> a_signed_c;
};

struct HostMkwPruning_ {
	std::vector<uint64_t> range;
	std::vector<uint64_t> output_ptr;
	std::vector<uint32_t> output_idx;
	std::vector<uint64_t> op_ptr{ 0 };
	std::vector<uint32_t> op_idx;
	std::vector<uint64_t> linear_a_ptr;
	std::vector<uint32_t> linear_a_idx;
	std::vector<uint64_t> reverse_ptr;
	std::vector<uint32_t> reverse_idx;
	uint64_t dense_forward_work = 0;
	uint64_t exact_forward_work = 0;
	uint64_t exact_zero_work = 0;
	bool prune_reverse = false;
};

HostMkwPruning_ build_mkw_pruning_(
	const BchHardcodedData& formula,
	const HostMkwBchTables_& tables,
	const std::vector<uint64_t>& weights,
	const std::vector<uint32_t>& segment_idx,
	const std::vector<uint32_t>& bch_nodes
) {
	// Propagate exact coordinate support through the BCH expression tree.
	// This selects sparse plans only when they save enough work to justify them.
	const uint64_t m = weights.size();
	const uint64_t m2 = formula.size;
	std::vector<uint8_t> input_mask(m, 0);
	for (uint32_t index : segment_idx)
		input_mask[index] = 1;

	std::vector<uint64_t> min_weight(m2, 1);
	std::vector<uint64_t> max_weight(m2, weights.empty() ? 0 : weights.back());
	if (m2 > 1) {
		min_weight[1] = weights.empty() ? 1 : weights.back() + 1;
		max_weight[1] = 0;
		for (uint32_t index : segment_idx) {
			min_weight[1] = std::min(min_weight[1], weights[index]);
			max_weight[1] = std::max(max_weight[1], weights[index]);
		}
		if (segment_idx.empty())
			min_weight[1] = 1;
	}
	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t left = formula.left_factor[w];
		const uint64_t right = formula.right_factor[w];
		min_weight[w] = min_weight[left] + min_weight[right];
		max_weight[w] = std::min(
			weights.empty() ? uint64_t(0) : weights.back(),
			max_weight[left] + max_weight[right]);
	}

	HostMkwPruning_ pruning;
	pruning.range.resize(2 * m2);
	for (uint64_t w = 0; w < m2; ++w) {
		pruning.range[2 * w] = std::lower_bound(
			weights.begin(), weights.end(), min_weight[w]) - weights.begin();
		pruning.range[2 * w + 1] = std::upper_bound(
			weights.begin(), weights.end(), max_weight[w]) - weights.begin();
	}

	std::vector<std::vector<uint8_t>> support(m2, std::vector<uint8_t>(m, 0));
	if (m2 > 0)
		std::fill(support[0].begin(), support[0].end(), uint8_t(1));
	if (m2 > 1)
		support[1] = input_mask;
	pruning.output_ptr.assign(m2 + 1, 0);
	for (uint64_t w = 2; w < m2; ++w) {
		const auto& left_support = support[formula.left_factor[w]];
		const auto& right_support = support[formula.right_factor[w]];
		pruning.output_ptr[w] = pruning.output_idx.size();
		for (uint64_t k = 0; k < m; ++k) {
			const uint64_t before = pruning.op_idx.size();
			for (uint32_t index = tables.k_ptr[k]; index < tables.k_ptr[k + 1]; ++index) {
				const uint32_t i = tables.k_i[index];
				const uint32_t j = tables.k_j[index];
				if ((left_support[i] && right_support[j])
					|| (left_support[j] && right_support[i]))
					pruning.op_idx.push_back(index);
			}
			if (pruning.op_idx.size() != before) {
				support[w][k] = 1;
				pruning.output_idx.push_back(static_cast<uint32_t>(k));
				pruning.op_ptr.push_back(pruning.op_idx.size());
			}
		}
		pruning.output_ptr[w + 1] = pruning.output_idx.size();
	}

	pruning.linear_a_ptr.resize(m2 * m + 1, 0);
	pruning.reverse_ptr.assign(m2 + 1, 0);
	for (uint64_t w = 2; w < m2; ++w) {
		const auto& left_support = support[formula.left_factor[w]];
		const auto& right_support = support[formula.right_factor[w]];
		pruning.reverse_ptr[w] = pruning.reverse_idx.size();
		for (uint64_t a = 0; a < m; ++a) {
			const uint64_t row = w * m + a;
			pruning.linear_a_ptr[row] = pruning.linear_a_idx.size();
			for (uint32_t index = tables.a_ptr[a]; index < tables.a_ptr[a + 1]; ++index) {
				const uint32_t partner = tables.a_partner[index];
				if (left_support[a] && right_support[partner])
					pruning.linear_a_idx.push_back(index << 1);
				if (right_support[a] && left_support[partner])
					pruning.linear_a_idx.push_back((index << 1) | 1);
			}
			pruning.linear_a_ptr[row + 1] = pruning.linear_a_idx.size();
			if (pruning.linear_a_ptr[row] != pruning.linear_a_ptr[row + 1])
				pruning.reverse_idx.push_back(static_cast<uint32_t>(a));
		}
		pruning.reverse_ptr[w + 1] = pruning.reverse_idx.size();
	}

	const uint64_t node_count = bch_nodes.size();
	const uint64_t nnz = tables.k_i.size();
	pruning.dense_forward_work = node_count * (m + nnz);
	for (uint32_t node : bch_nodes) {
		const uint64_t output_begin = pruning.output_ptr[node];
		const uint64_t output_end = pruning.output_ptr[node + 1];
		pruning.exact_forward_work += output_end - output_begin;
		pruning.exact_forward_work +=
			pruning.op_ptr[output_end] - pruning.op_ptr[output_begin];
	}
	pruning.exact_zero_work = node_count * m;
	const uint64_t dense_reverse_work = node_count
		* (m + tables.a_k.size());
	uint64_t exact_reverse_work = 0;
	for (uint32_t node : bch_nodes) {
		const uint64_t reverse_begin = pruning.reverse_ptr[node];
		const uint64_t reverse_end = pruning.reverse_ptr[node + 1];
		exact_reverse_work += reverse_end - reverse_begin;
		for (uint64_t position = reverse_begin;
			position < reverse_end; ++position) {
			const uint64_t row = uint64_t(node) * m
				+ pruning.reverse_idx[position];
			exact_reverse_work += pruning.linear_a_ptr[row + 1]
				- pruning.linear_a_ptr[row];
		}
	}
	pruning.prune_reverse = dense_reverse_work > 0
		&& exact_reverse_work <= dense_reverse_work - dense_reverse_work / 3;
	return pruning;
}

std::unique_ptr<CuMkwBchCache> build_cuda_mkw_bch_cache_(
	const BranchedSigCache& cache,
	const BranchedBchCache& host_cache
) {
	if (!cache.planar)
		throw std::invalid_argument("MKW BCH requires planar=True");
	if (cache.max_nodes > BCH_MAX_HARDCODED_DEGREE)
		throw std::runtime_error(
			"CUDA MKW BCH method supports degree at most 12");

	const BchCache& host_bch = host_cache.bch;
	const uint64_t m = host_bch.m;
	if (m > UINT32_MAX)
		throw std::overflow_error("CUDA MKW BCH basis exceeds uint32 range");

	auto result = std::make_unique<CuMkwBchCache>();
	result->bch.m = m;
	if (cache.max_nodes == 0)
		return result;

	const BchHardcodedData* formula = get_hardcoded_bch_data(cache.max_nodes);
	if (formula == nullptr)
		throw std::runtime_error(
			"CUDA MKW BCH method supports degree at most 12");
	result->bch.m2 = formula->size;
	result->bch.bch_plan.size = host_bch.live_bch_nodes.size();

	std::vector<uint32_t> segment_idx;
	std::vector<double> segment_coefficients;
	std::vector<uint32_t> segment_label_offsets{ 0 };
	std::vector<uint8_t> segment_labels;
	for (uint64_t coordinate = 0; coordinate < m; ++coordinate) {
		if (host_cache.linear_coefficients[coordinate] == 0.0)
			continue;
		segment_idx.push_back(static_cast<uint32_t>(coordinate));
		segment_coefficients.push_back(
			host_cache.linear_coefficients[coordinate]);
		const uint64_t basis_idx = host_cache.linear_basis_idx[coordinate];
		const uint64_t start = cache.node_labels_offsets[basis_idx];
		const uint64_t end = cache.node_labels_offsets[basis_idx + 1];
		segment_labels.insert(
			segment_labels.end(),
			cache.node_labels_data.begin() + static_cast<std::ptrdiff_t>(start),
			cache.node_labels_data.begin() + static_cast<std::ptrdiff_t>(end));
		segment_label_offsets.push_back(narrow_mkw_u32_(
			segment_labels.size(), "MKW segment label plan exceeds uint32 range"));
	}
	result->segment_count = segment_idx.size();

	HostMkwBchTables_ tables;
	tables.k_ptr = host_bch.comm_k_ptr;
	tables.k_i = host_bch.comm_k_i;
	tables.k_j = host_bch.comm_k_j;
	tables.k_val = host_bch.comm_k_val;
	tables.a_ptr = host_bch.comm_a_ptr;
	tables.a_k = host_bch.comm_a_k;
	tables.a_partner = host_bch.comm_a_partner;
	tables.a_signed_c = host_bch.comm_a_signed_c;
	const HostMkwPruning_ pruning = build_mkw_pruning_(
		*formula, tables, host_bch.coordinate_weights, segment_idx,
		host_bch.live_bch_nodes);
	result->bch.linear_dense_forward_work = pruning.dense_forward_work;
	result->bch.linear_active_forward_work = pruning.exact_forward_work;
	result->bch.linear_zero_work = pruning.exact_zero_work;
	result->exact_forward_work = pruning.exact_forward_work;
	result->exact_zero_work = pruning.exact_zero_work;
	result->prune_reverse = pruning.prune_reverse;

	std::vector<double> bch_coefficients(
		formula->coefficients, formula->coefficients + formula->size);
	std::vector<uint64_t> bch_left(
		formula->left_factor, formula->left_factor + formula->size);
	std::vector<uint64_t> bch_right(
		formula->right_factor, formula->right_factor + formula->size);
	upload_mkw_vector_(result->bch.d_bch_coefficients, bch_coefficients);
	upload_mkw_vector_(result->bch.d_bch_left_factor, bch_left);
	upload_mkw_vector_(result->bch.d_bch_right_factor, bch_right);
	if (!host_bch.all_bch_nodes_live)
		upload_mkw_vector_(
			result->bch.bch_plan.nodes, host_bch.live_bch_nodes);
	upload_mkw_vector_(result->bch.d_linear_range, pruning.range);
	upload_mkw_vector_(result->bch.d_comm_k_ptr, tables.k_ptr);
	upload_mkw_vector_(result->bch.d_comm_k_i, tables.k_i);
	upload_mkw_vector_(result->bch.d_comm_k_j, tables.k_j);
	upload_mkw_vector_(result->bch.d_comm_k_val, tables.k_val);
	upload_mkw_vector_(result->bch.d_comm_a_ptr, tables.a_ptr);
	upload_mkw_vector_(result->bch.d_comm_a_k, tables.a_k);
	upload_mkw_vector_(result->bch.d_comm_a_partner, tables.a_partner);
	upload_mkw_vector_(result->bch.d_comm_a_signed_c, tables.a_signed_c);
	upload_mkw_vector_(result->bch.d_linear_a_ptr, pruning.linear_a_ptr);
	upload_mkw_vector_(result->bch.d_linear_a_idx, pruning.linear_a_idx);
	upload_mkw_vector_(result->d_linear_output_ptr, pruning.output_ptr);
	upload_mkw_vector_(result->d_linear_output_idx, pruning.output_idx);
	upload_mkw_vector_(result->d_linear_op_ptr, pruning.op_ptr);
	upload_mkw_vector_(result->d_linear_op_idx, pruning.op_idx);
	upload_mkw_vector_(result->d_linear_reverse_ptr, pruning.reverse_ptr);
	upload_mkw_vector_(result->d_linear_reverse_idx, pruning.reverse_idx);
	upload_mkw_vector_(result->d_segment_idx, segment_idx);
	upload_mkw_vector_(result->d_segment_coefficients, segment_coefficients);
	upload_mkw_vector_(result->d_segment_label_offsets, segment_label_offsets);
	upload_mkw_vector_(result->d_segment_labels, segment_labels);
	return result;
}

const CuMkwBchCache& get_cuda_mkw_bch_cache_(
	uint64_t dimension,
	uint64_t max_nodes
) {
	const auto key = make_cuda_branched_log_cache_key_(
		dimension, max_nodes, true);
	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	const auto found = get_cuda_branched_log_sig_cache_map_().find(key);
	if (found == get_cuda_branched_log_sig_cache_map_().end()
		|| !found->second.bch)
		throw cache_not_found_error(
			"CUDA MKW BCH cache not found - call prepare_branched_log_sig with method=3 first");
	return *static_cast<const CuMkwBchCache*>(found->second.bch.get());
}

template<typename T>
__device__ __forceinline__ void evaluate_mkw_segment_(
	const T* left,
	const T* right,
	T* target,
	T sign,
	const CuMkwBchDeviceData& data
) {
	for (uint64_t q = threadIdx.x; q < data.segment_count; q += blockDim.x) {
		T value = sign * static_cast<T>(data.segment_coefficients[q]);
		for (uint32_t position = data.segment_label_offsets[q];
			position < data.segment_label_offsets[q + 1]; ++position) {
			const uint8_t label = data.segment_labels[position];
			value *= right[label] - left[label];
		}
		target[data.segment_idx[q]] = value;
	}
}

template<typename T, bool Sparse, bool Shared, bool Accumulate>
__device__ __forceinline__ void evaluate_mkw_bch_nodes_(
	T* memo,
	T* out,
	T* shared_left,
	T* shared_right,
	const CuMkwBchDeviceData& data
) {
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	for (uint64_t plan_index = 0;
		plan_index < data.bch_plan.size; ++plan_index) {
		const uint32_t w = data.bch_plan[plan_index];
		const T* left_global = memo + data.bch_left_factor[w] * data.m;
		const T* right_global = memo + data.bch_right_factor[w] * data.m;
		const T* left = left_global;
		const T* right = right_global;
		if constexpr (Shared) {
			for (uint64_t k = tid; k < data.m; k += stride) {
				shared_left[k] = left_global[k];
				shared_right[k] = right_global[k];
			}
			__syncthreads();
			left = shared_left;
			right = shared_right;
		}

		T* result = memo + w * data.m;
		if constexpr (Sparse) {
			for (uint64_t q = data.linear_output_ptr[w] + tid;
				q < data.linear_output_ptr[w + 1]; q += stride) {
				const uint32_t k = data.linear_output_idx[q];
				T value = T(0);
				for (uint64_t position = data.linear_op_ptr[q];
					position < data.linear_op_ptr[q + 1]; ++position) {
					const uint32_t comm_index = data.linear_op_idx[position];
					const uint32_t i = data.comm_k_i[comm_index];
					const uint32_t j = data.comm_k_j[comm_index];
					value += static_cast<T>(data.comm_k_val[comm_index])
						* (left[i] * right[j] - left[j] * right[i]);
				}
				result[k] = value;
				if constexpr (Accumulate) {
					const T bch_coefficient = static_cast<T>(
						data.bch_coefficients[w]);
					if (bch_coefficient != T(0))
						out[k] += bch_coefficient * value;
				}
			}
		}
		else {
			for (uint64_t k = tid; k < data.m; k += stride) {
				T value = T(0);
				for (uint32_t position = data.comm_k_ptr[k];
					position < data.comm_k_ptr[k + 1]; ++position) {
					const uint32_t i = data.comm_k_i[position];
					const uint32_t j = data.comm_k_j[position];
					value += static_cast<T>(data.comm_k_val[position])
						* (left[i] * right[j] - left[j] * right[i]);
				}
				result[k] = value;
				if constexpr (Accumulate) {
					const T bch_coefficient = static_cast<T>(
						data.bch_coefficients[w]);
					if (bch_coefficient != T(0))
						out[k] += bch_coefficient * value;
				}
			}
		}
		__syncthreads();
	}
}

template<typename T, bool Sparse, bool Shared>
__global__ void branched_log_sig_from_path_kernel_(
	const T* __restrict__ path,
	T* __restrict__ out,
	T* __restrict__ workspace,
	uint64_t length,
	uint64_t dimension,
	CuMkwBchDeviceData data
) {
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	const T* path_row = path + batch_idx * length * dimension;
	T* out_row = out + batch_idx * data.m;
	T* memo = workspace + batch_idx * data.m2 * data.m;

	T* shared_left = nullptr;
	T* shared_right = nullptr;
	if constexpr (Shared) {
		extern __shared__ char shared_bytes[];
		shared_left = reinterpret_cast<T*>(shared_bytes);
		shared_right = shared_left + data.m;
	}

	for (uint64_t k = tid; k < data.m; k += stride) {
		out_row[k] = T(0);
		memo[data.m + k] = T(0);
	}
	if constexpr (Sparse) {
		for (uint64_t plan_index = 0;
			plan_index < data.bch_plan.size; ++plan_index) {
			T* result = memo + data.bch_plan[plan_index] * data.m;
			for (uint64_t k = tid; k < data.m; k += stride)
				result[k] = T(0);
		}
	}
	__syncthreads();
	if (length == 1)
		return;

	evaluate_mkw_segment_(
		path_row, path_row + dimension, out_row, T(1), data);
	__syncthreads();
	for (uint64_t segment = 1; segment + 1 < length; ++segment) {
		const T* left = path_row + segment * dimension;
		const T* right = left + dimension;
		for (uint64_t k = tid; k < data.m; k += stride)
			memo[k] = out_row[k];
		evaluate_mkw_segment_(left, right, memo + data.m, T(1), data);
		__syncthreads();
		for (uint64_t q = tid; q < data.segment_count; q += stride) {
			const uint32_t coordinate = data.segment_idx[q];
			out_row[coordinate] += memo[data.m + coordinate];
		}
		__syncthreads();
		evaluate_mkw_bch_nodes_<T, Sparse, Shared, true>(
			memo, out_row, shared_left, shared_right, data);
	}
}

uint64_t checked_mkw_product_(
	uint64_t left,
	uint64_t right,
	const char* message
) {
	if (right != 0 && left > UINT64_MAX / right)
		throw std::overflow_error(message);
	return left * right;
}

uint64_t checked_mkw_sum_(
	uint64_t left,
	uint64_t right,
	const char* message
) {
	if (left > UINT64_MAX - right)
		throw std::overflow_error(message);
	return left + right;
}

template<typename T, bool Shared>
void launch_mkw_forward_(
	bool sparse,
	const T* path,
	T* out,
	T* workspace,
	uint64_t batch,
	uint64_t length,
	uint64_t dimension,
	unsigned int threads,
	size_t shared_size,
	const CuMkwBchDeviceData& data
) {
	if (sparse) {
		branched_log_sig_from_path_kernel_<T, true, Shared><<<
			static_cast<unsigned int>(batch), threads, shared_size>>>(
				path, out, workspace, length, dimension, data);
	}
	else {
		branched_log_sig_from_path_kernel_<T, false, Shared><<<
			static_cast<unsigned int>(batch), threads, shared_size>>>(
				path, out, workspace, length, dimension, data);
	}
}

template<typename T>
void branched_log_sig_from_path_cuda_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t max_nodes
) {
	if (length == 0)
		throw std::invalid_argument(
			"branched_log_sig method 3 received an empty path");
	const CuMkwBchCache& cache = get_cuda_mkw_bch_cache_(
		dimension, max_nodes);
	const CuMkwBchDeviceData data = cache.device_data();
	if (batch_size == 0 || data.m == 0)
		return;
	if (length == 1) {
		const uint64_t output_elements = checked_mkw_product_(
			batch_size, data.m, "MKW BCH output size overflow");
		CUDA_CHECK(cudaMemset(
			out, 0, checked_mkw_product_(output_elements, sizeof(T),
				"MKW BCH output byte size overflow")));
		check_cuda_kernel_launch();
		return;
	}

	const uint64_t workspace_per_batch = checked_mkw_product_(
		data.m2, data.m, "MKW BCH workspace size overflow");
	size_t free_memory = 0;
	size_t total_memory = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
	const uint64_t workspace_bytes = checked_mkw_product_(
		workspace_per_batch, sizeof(T), "MKW BCH workspace byte size overflow");
	const uint64_t reservation_bytes = checked_mkw_product_(
		2, workspace_bytes, "MKW BCH memory reservation overflow");
	uint64_t chunk_size = workspace_bytes == 0
		? batch_size
		: free_memory / reservation_bytes;
	chunk_size = std::max<uint64_t>(1, chunk_size);
	chunk_size = std::min<uint64_t>(
		std::min<uint64_t>(batch_size, chunk_size), CUDA_GRID_X_LIMIT);
	CudaBuf<T> workspace(checked_mkw_product_(
		chunk_size, workspace_bytes, "MKW BCH workspace allocation overflow"));

	unsigned int threads = static_cast<unsigned int>(
		std::min<uint64_t>(64, data.m));
	threads = std::max(32U, ((threads + 31) / 32) * 32);
	const size_t shared_size = checked_mkw_product_(
		checked_mkw_product_(2, data.m, "MKW BCH shared size overflow"),
		sizeof(T), "MKW BCH shared byte size overflow");
	const bool shared = shared_size <= CUDA_BASE_DYNAMIC_SMEM;
	const uint64_t passes = length > 2 ? length - 2 : 0;
	const long double dense_work = static_cast<long double>(passes)
		* cache.bch.linear_dense_forward_work;
	const long double exact_work = static_cast<long double>(passes)
		* cache.exact_forward_work + cache.exact_zero_work;
	const bool sparse = data.bch_plan.size != 0 && exact_work < dense_work;
	const uint64_t path_stride = checked_mkw_product_(
		length, dimension, "MKW BCH path stride overflow");
	checked_mkw_product_(
		batch_size, path_stride, "MKW BCH path size overflow");
	checked_mkw_product_(
		batch_size, data.m, "MKW BCH output size overflow");

	for (uint64_t offset = 0; offset < batch_size; offset += chunk_size) {
		const uint64_t current_batch = std::min(
			chunk_size, batch_size - offset);
		if (shared) {
			launch_mkw_forward_<T, true>(
				sparse, path + offset * path_stride,
				out + offset * data.m, workspace.get(), current_batch,
				length, dimension, threads, shared_size, data);
		}
		else {
			launch_mkw_forward_<T, false>(
				sparse, path + offset * path_stride,
				out + offset * data.m, workspace.get(), current_batch,
				length, dimension, threads, 0, data);
		}
		check_cuda_kernel_launch();
	}
}

template<typename T>
__device__ __forceinline__ void add_mkw_segment_vjp_(
	const T* left,
	const T* right,
	const T* accumulated_derivs,
	const T* leaf_derivs,
	T* left_derivs,
	T* right_derivs,
	uint64_t dimension,
	const CuMkwBchDeviceData& data
) {
	for (uint64_t label = threadIdx.x; label < dimension; label += blockDim.x) {
		T derivative = T(0);
		for (uint64_t q = 0; q < data.segment_count; ++q) {
			const uint32_t coordinate = data.segment_idx[q];
			T base = accumulated_derivs[coordinate];
			if (leaf_derivs != nullptr)
				base += leaf_derivs[coordinate];
			base *= static_cast<T>(data.segment_coefficients[q]);
			const uint32_t start = data.segment_label_offsets[q];
			const uint32_t end = data.segment_label_offsets[q + 1];
			for (uint32_t position = start; position < end; ++position) {
				if (data.segment_labels[position] != label)
					continue;
				T product = base;
				for (uint32_t other = start; other < end; ++other) {
					if (other == position)
						continue;
					const uint8_t other_label = data.segment_labels[other];
					product *= right[other_label] - left[other_label];
				}
				derivative += product;
			}
		}
		right_derivs[label] += derivative;
		left_derivs[label] -= derivative;
	}
}

template<typename T, bool Sparse, bool Shared>
__device__ __forceinline__ void reverse_mkw_bch_nodes_(
	const T* memo,
	T* deriv_memo,
	T* shared_left,
	T* shared_right,
	const CuMkwBchDeviceData& data
) {
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	for (uint64_t plan_index = data.bch_plan.size;
		plan_index-- > 0;) {
		const uint32_t w = data.bch_plan[plan_index];
		const uint64_t left_factor = data.bch_left_factor[w];
		const uint64_t right_factor = data.bch_right_factor[w];
		const T* left_global = memo + left_factor * data.m;
		const T* right_global = memo + right_factor * data.m;
		const T* left = left_global;
		const T* right = right_global;
		if constexpr (Shared) {
			for (uint64_t k = tid; k < data.m; k += stride) {
				shared_left[k] = left_global[k];
				shared_right[k] = right_global[k];
			}
			__syncthreads();
			left = shared_left;
			right = shared_right;
		}

		const T* node_derivs = deriv_memo + w * data.m;
		T* left_derivs = deriv_memo + left_factor * data.m;
		T* right_derivs = deriv_memo + right_factor * data.m;
		if constexpr (Sparse) {
			for (uint64_t q = data.linear_reverse_ptr[w] + tid;
				q < data.linear_reverse_ptr[w + 1]; q += stride) {
				const uint32_t a = data.linear_reverse_idx[q];
				T left_value = T(0);
				T right_value = T(0);
				const uint64_t row = w * data.m + a;
				for (uint64_t position = data.linear_a_ptr[row];
					position < data.linear_a_ptr[row + 1]; ++position) {
					const uint32_t packed = data.linear_a_idx[position];
					const uint32_t index = packed >> 1;
					const uint32_t k = data.comm_a_k[index];
					const uint32_t partner = data.comm_a_partner[index];
					const T coefficient = static_cast<T>(
						data.comm_a_signed_c[index]);
					if (packed & 1)
						right_value -= coefficient * left[partner] * node_derivs[k];
					else
						left_value += coefficient * right[partner] * node_derivs[k];
				}
				left_derivs[a] += left_value;
				right_derivs[a] += right_value;
			}
		}
		else {
			for (uint64_t a = tid; a < data.m; a += stride) {
				T left_value = T(0);
				T right_value = T(0);
				for (uint32_t index = data.comm_a_ptr[a];
					index < data.comm_a_ptr[a + 1]; ++index) {
					const uint32_t k = data.comm_a_k[index];
					const uint32_t partner = data.comm_a_partner[index];
					const T coefficient = static_cast<T>(
						data.comm_a_signed_c[index]);
					left_value += coefficient * right[partner] * node_derivs[k];
					right_value -= coefficient * left[partner] * node_derivs[k];
				}
				left_derivs[a] += left_value;
				right_derivs[a] += right_value;
			}
		}
		__syncthreads();
	}
}

template<typename T, bool SparseForward, bool SaveStates,
	bool SparseReverse, bool Shared>
__global__ void branched_log_sig_from_path_backprop_kernel_(
	const T* __restrict__ out_derivs,
	T* __restrict__ path_derivs,
	const T* __restrict__ path,
	T* __restrict__ workspace,
	uint64_t length,
	uint64_t dimension,
	CuMkwBchDeviceData data
) {
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	const uint64_t segment_count = length - 1;
	const uint64_t state_count = SaveStates ? segment_count : 2;
	const uint64_t workspace_per_batch = (state_count + 1) * data.m
		+ 2 * data.m2 * data.m;
	T* states = workspace + batch_idx * workspace_per_batch;
	T* current = states;
	T* previous = current + data.m;
	T* memo = states + state_count * data.m;
	T* deriv_memo = memo + data.m2 * data.m;
	T* accumulated_derivs = deriv_memo + data.m2 * data.m;

	const T* path_row = path + batch_idx * length * dimension;
	T* path_derivs_row = path_derivs + batch_idx * length * dimension;
	const T* out_derivs_row = out_derivs + batch_idx * data.m;
	T* shared_left = nullptr;
	T* shared_right = nullptr;
	if constexpr (Shared) {
		extern __shared__ char shared_bytes[];
		shared_left = reinterpret_cast<T*>(shared_bytes);
		shared_right = shared_left + data.m;
	}

	for (uint64_t index = tid; index < length * dimension; index += stride)
		path_derivs_row[index] = T(0);
	for (uint64_t k = tid; k < data.m; k += stride) {
		current[k] = T(0);
		memo[data.m + k] = T(0);
	}
	if constexpr (SparseForward) {
		for (uint64_t plan_index = 0;
			plan_index < data.bch_plan.size; ++plan_index) {
			T* result = memo + data.bch_plan[plan_index] * data.m;
			for (uint64_t k = tid; k < data.m; k += stride)
				result[k] = T(0);
		}
	}
	__syncthreads();

	evaluate_mkw_segment_(
		path_row, path_row + dimension, current, T(1), data);
	__syncthreads();
	for (uint64_t segment = 1; segment < segment_count; ++segment) {
		const T* left = path_row + segment * dimension;
		const T* right = left + dimension;
		const T* accumulator = current;
		T* next = previous;
		if constexpr (SaveStates) {
			accumulator = states + (segment - 1) * data.m;
			next = states + segment * data.m;
		}
		for (uint64_t k = tid; k < data.m; k += stride) {
			memo[k] = accumulator[k];
			next[k] = accumulator[k];
		}
		evaluate_mkw_segment_(left, right, memo + data.m, T(1), data);
		__syncthreads();
		for (uint64_t q = tid; q < data.segment_count; q += stride) {
			const uint32_t coordinate = data.segment_idx[q];
			next[coordinate] += memo[data.m + coordinate];
		}
		__syncthreads();
		evaluate_mkw_bch_nodes_<T, SparseForward, Shared, true>(
			memo, next, shared_left, shared_right, data);
		if constexpr (!SaveStates) {
			T* temporary = current;
			current = previous;
			previous = temporary;
		}
	}

	for (uint64_t k = tid; k < data.m; k += stride)
		accumulated_derivs[k] = out_derivs_row[k];
	__syncthreads();
	for (uint64_t segment_offset = 0;
		segment_offset + 1 < segment_count; ++segment_offset) {
		const uint64_t segment = segment_count - segment_offset - 1;
		const T* left = path_row + segment * dimension;
		const T* right = left + dimension;
		if constexpr (SaveStates) {
			previous = states + (segment - 1) * data.m;
		}
		else {
			for (uint64_t k = tid; k < data.m; k += stride) {
				memo[k] = current[k];
				previous[k] = current[k];
			}
			evaluate_mkw_segment_(left, right, memo + data.m, T(-1), data);
			__syncthreads();
			for (uint64_t q = tid; q < data.segment_count; q += stride) {
				const uint32_t coordinate = data.segment_idx[q];
				previous[coordinate] += memo[data.m + coordinate];
			}
			__syncthreads();
			evaluate_mkw_bch_nodes_<T, SparseForward, Shared, true>(
				memo, previous, shared_left, shared_right, data);
		}

		for (uint64_t k = tid; k < data.m; k += stride)
			memo[k] = previous[k];
		evaluate_mkw_segment_(left, right, memo + data.m, T(1), data);
		__syncthreads();
		evaluate_mkw_bch_nodes_<T, SparseForward, Shared, false>(
			memo, nullptr, shared_left, shared_right, data);

		for (uint64_t k = tid; k < data.m; k += stride) {
			deriv_memo[k] = T(0);
			deriv_memo[data.m + k] = T(0);
		}
		if constexpr (SparseForward && !SparseReverse) {
			for (uint64_t plan_index = 0;
				plan_index < data.bch_plan.size; ++plan_index) {
				T* node_derivs = deriv_memo
					+ data.bch_plan[plan_index] * data.m;
				for (uint64_t k = tid; k < data.m; k += stride)
					node_derivs[k] = T(0);
			}
		}
		__syncthreads();
		if constexpr (SparseForward) {
			for (uint64_t plan_index = 0;
				plan_index < data.bch_plan.size; ++plan_index) {
				const uint32_t w = data.bch_plan[plan_index];
				const T coefficient = static_cast<T>(data.bch_coefficients[w]);
				for (uint64_t q = data.linear_output_ptr[w] + tid;
					q < data.linear_output_ptr[w + 1]; q += stride) {
					const uint32_t k = data.linear_output_idx[q];
					deriv_memo[w * data.m + k]
						= coefficient * accumulated_derivs[k];
				}
			}
		}
		else {
			for (uint64_t plan_index = 0;
				plan_index < data.bch_plan.size; ++plan_index) {
				const uint32_t w = data.bch_plan[plan_index];
				const T coefficient = static_cast<T>(data.bch_coefficients[w]);
				for (uint64_t k = tid; k < data.m; k += stride)
					deriv_memo[w * data.m + k]
						= coefficient * accumulated_derivs[k];
			}
		}
		__syncthreads();
		reverse_mkw_bch_nodes_<T, SparseReverse, Shared>(
			memo, deriv_memo, shared_left, shared_right, data);

		add_mkw_segment_vjp_(
			left, right, accumulated_derivs, deriv_memo + data.m,
			path_derivs_row + segment * dimension,
			path_derivs_row + (segment + 1) * dimension,
			dimension, data);
		__syncthreads();
		for (uint64_t k = tid; k < data.m; k += stride)
			accumulated_derivs[k] += deriv_memo[k];
		__syncthreads();
		if constexpr (!SaveStates) {
			T* temporary = current;
			current = previous;
			previous = temporary;
		}
	}

	add_mkw_segment_vjp_(
		path_row, path_row + dimension, accumulated_derivs,
		static_cast<const T*>(nullptr),
		path_derivs_row, path_derivs_row + dimension, dimension, data);
}

template<typename T, bool SaveStates, bool Shared>
void launch_mkw_backward_(
	bool sparse_forward,
	bool sparse_reverse,
	const T* out_derivs,
	T* path_derivs,
	const T* path,
	T* workspace,
	uint64_t batch,
	uint64_t length,
	uint64_t dimension,
	unsigned int threads,
	size_t shared_size,
	const CuMkwBchDeviceData& data
) {
	if (sparse_forward && sparse_reverse) {
		branched_log_sig_from_path_backprop_kernel_<
			T, true, SaveStates, true, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
	else if (sparse_forward) {
		branched_log_sig_from_path_backprop_kernel_<
			T, true, SaveStates, false, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
	else if (sparse_reverse) {
		branched_log_sig_from_path_backprop_kernel_<
			T, false, SaveStates, true, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
	else {
		branched_log_sig_from_path_backprop_kernel_<
			T, false, SaveStates, false, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
}

template<typename T>
void branched_log_sig_from_path_backprop_cuda_(
	const T* out_derivs,
	T* path_derivs,
	const T* path,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t max_nodes
) {
	if (length == 0)
		throw std::invalid_argument(
			"branched_log_sig method 3 received an empty path");
	const CuMkwBchCache& cache = get_cuda_mkw_bch_cache_(
		dimension, max_nodes);
	const CuMkwBchDeviceData data = cache.device_data();
	if (batch_size == 0)
		return;
	const uint64_t path_stride = checked_mkw_product_(
		length, dimension, "MKW BCH path stride overflow");
	const uint64_t path_elements = checked_mkw_product_(
		batch_size, path_stride, "MKW BCH path gradient size overflow");
	checked_mkw_product_(
		batch_size, data.m, "MKW BCH output derivative size overflow");
	if (length == 1 || data.m == 0) {
		if (path_elements != 0) {
			CUDA_CHECK(cudaMemset(path_derivs, 0, checked_mkw_product_(
				path_elements, sizeof(T),
				"MKW BCH path gradient byte size overflow")));
			check_cuda_kernel_launch();
		}
		return;
	}

	const uint64_t segment_count = length - 1;
	const bool save_states = length > 2 && segment_count <= 2 * data.m2;
	const uint64_t state_count = save_states ? segment_count : 2;
	const uint64_t memo_size = checked_mkw_product_(
		data.m2, data.m, "MKW BCH backward memo size overflow");
	const uint64_t workspace_per_batch = checked_mkw_sum_(
		checked_mkw_product_(state_count + 1, data.m,
			"MKW BCH backward state size overflow"),
		checked_mkw_product_(2, memo_size,
			"MKW BCH backward workspace size overflow"),
		"MKW BCH backward workspace size overflow");
	const uint64_t workspace_bytes = checked_mkw_product_(
		workspace_per_batch, sizeof(T),
		"MKW BCH backward workspace byte size overflow");
	size_t free_memory = 0;
	size_t total_memory = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
	const uint64_t reservation_bytes = checked_mkw_product_(
		2, workspace_bytes,
		"MKW BCH backward memory reservation overflow");
	uint64_t chunk_size = workspace_bytes == 0
		? batch_size
		: free_memory / reservation_bytes;
	chunk_size = std::max<uint64_t>(1, chunk_size);
	chunk_size = std::min<uint64_t>(
		std::min<uint64_t>(batch_size, chunk_size), CUDA_GRID_X_LIMIT);
	CudaBuf<T> workspace(checked_mkw_product_(
		chunk_size, workspace_bytes,
		"MKW BCH backward workspace allocation overflow"));

	unsigned int threads = static_cast<unsigned int>(
		std::min<uint64_t>(64, data.m));
	threads = std::max(32U, ((threads + 31) / 32) * 32);
	const size_t shared_size = checked_mkw_product_(
		checked_mkw_product_(2, data.m,
			"MKW BCH backward shared size overflow"),
		sizeof(T), "MKW BCH backward shared byte size overflow");
	const bool shared = shared_size <= CUDA_BASE_DYNAMIC_SMEM;
	const uint64_t combine_count = length > 2 ? length - 2 : 0;
	const long double passes = static_cast<long double>(combine_count)
		* (save_states ? 2 : 3);
	const long double dense_work = passes
		* cache.bch.linear_dense_forward_work;
	const long double exact_work = passes
		* cache.exact_forward_work + cache.exact_zero_work;
	const bool sparse_forward = data.bch_plan.size != 0
		&& exact_work < dense_work;
	const bool sparse_reverse = data.bch_plan.size != 0
		&& cache.prune_reverse;

	for (uint64_t offset = 0; offset < batch_size; offset += chunk_size) {
		const uint64_t current_batch = std::min(
			chunk_size, batch_size - offset);
		const T* chunk_out_derivs = out_derivs + offset * data.m;
		T* chunk_path_derivs = path_derivs + offset * path_stride;
		const T* chunk_path = path + offset * path_stride;
		if (save_states && shared) {
			launch_mkw_backward_<T, true, true>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, shared_size, data);
		}
		else if (save_states) {
			launch_mkw_backward_<T, true, false>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, 0, data);
		}
		else if (shared) {
			launch_mkw_backward_<T, false, true>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, shared_size, data);
		}
		else {
			launch_mkw_backward_<T, false, false>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, 0, data);
		}
		check_cuda_kernel_launch();
	}
}

}  // namespace

void prepare_cuda_branched_bch_cache_(
	const BranchedSigCache& cache,
	const BranchedBchCache& host_cache
) {
	const auto key = make_cuda_branched_log_cache_key_(
		cache.dimension, cache.max_nodes, true);
	{
		std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
		const auto found = get_cuda_branched_log_sig_cache_map_().find(key);
		if (found != get_cuda_branched_log_sig_cache_map_().end()
			&& found->second.bch)
			return;
	}
	std::shared_ptr<CuMkwBchCache> built =
		build_cuda_mkw_bch_cache_(cache, host_cache);
	std::lock_guard<std::mutex> lock(get_cuda_branched_log_sig_cache_mu_());
	auto [found, inserted] =
		get_cuda_branched_log_sig_cache_map_().try_emplace(key);
	if (!found->second.bch)
		found->second.bch = std::move(built);
}

extern "C" {

	CUSIG_API int branched_log_sig_from_path_cuda_f(
		const float* path, float* out, uint64_t batch_size, uint64_t length,
		uint64_t dimension, uint64_t max_nodes
	) noexcept {
		CUDA_SAFE_CALL(branched_log_sig_from_path_cuda_<float>(
			path, out, batch_size, length, dimension, max_nodes));
	}

	CUSIG_API int branched_log_sig_from_path_cuda_d(
		const double* path, double* out, uint64_t batch_size, uint64_t length,
		uint64_t dimension, uint64_t max_nodes
	) noexcept {
		CUDA_SAFE_CALL(branched_log_sig_from_path_cuda_<double>(
			path, out, batch_size, length, dimension, max_nodes));
	}

	CUSIG_API int branched_log_sig_from_path_backprop_cuda_f(
		const float* derivs, float* path_derivs, const float* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension,
		uint64_t max_nodes
	) noexcept {
		CUDA_SAFE_CALL(branched_log_sig_from_path_backprop_cuda_<float>(
			derivs, path_derivs, path, batch_size, length, dimension, max_nodes));
	}

	CUSIG_API int branched_log_sig_from_path_backprop_cuda_d(
		const double* derivs, double* path_derivs, const double* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension,
		uint64_t max_nodes
	) noexcept {
		CUDA_SAFE_CALL(branched_log_sig_from_path_backprop_cuda_<double>(
			derivs, path_derivs, path, batch_size, length, dimension, max_nodes));
	}

}
