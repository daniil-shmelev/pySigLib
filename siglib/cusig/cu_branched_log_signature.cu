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

#include "cupch.h"
#include "cusig.h"
#include "cu_atomic.h"
#include "cu_log_sig_cache.h"
#include "cu_macros.h"
#include "cu_utils.h"
#include "../shared/branched_cache.h"
#include "../shared/branched_log_cache.h"

#include <cstdint>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct BranchedLogSigCacheGPU {
	uint32_t* d_forest_offsets32 = nullptr;
	uint32_t* d_forest_trees32 = nullptr;
	uint32_t* d_forest_coprod_offsets32 = nullptr;
	uint32_t* d_forest_coprod_data32 = nullptr;
	uint32_t* d_single_tree_forest32 = nullptr;

	uint32_t total_length = 0;
	uint32_t num_trees = 0;
	uint32_t num_forests = 0;
	uint32_t forest_trees_len = 0;
	uint32_t forest_coprod_data_len = 0;
	int max_nodes = 0;

	BranchedLogSigCacheGPU() = default;
	BranchedLogSigCacheGPU(const BranchedLogSigCacheGPU&) = delete;
	BranchedLogSigCacheGPU& operator=(const BranchedLogSigCacheGPU&) = delete;

	~BranchedLogSigCacheGPU() {
		if (d_forest_offsets32) cudaFree(d_forest_offsets32);
		if (d_forest_trees32) cudaFree(d_forest_trees32);
		if (d_forest_coprod_offsets32) cudaFree(d_forest_coprod_offsets32);
		if (d_forest_coprod_data32) cudaFree(d_forest_coprod_data32);
		if (d_single_tree_forest32) cudaFree(d_single_tree_forest32);
	}
};

static inline std::pair<uint64_t, uint64_t> make_cu_branched_log_key(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	return { dimension, max_nodes | (static_cast<uint64_t>(planar) << 63) };
}

static std::unordered_map<
	std::pair<uint64_t, uint64_t>,
	std::unique_ptr<BranchedLogSigCacheGPU>,
	CuPairHash
> s_branched_log_gpu_cache_map;
static std::mutex s_branched_log_gpu_cache_mu;

void release_branched_log_sig_gpu_state() {
	std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
	s_branched_log_gpu_cache_map.clear();
}

void clear_cuda_branched_log_sig_gpu_cache_() {
	release_branched_log_sig_gpu_state();
}

template<typename T>
static void upload_branched_log(T*& d_ptr, const T* h_data, size_t count) {
	CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_ptr, h_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

static const BranchedLogSigCacheGPU& get_or_upload_branched_log_gpu_cache(
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	const auto key = make_cu_branched_log_key(dimension, max_nodes, planar);
	{
		std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
		auto it = s_branched_log_gpu_cache_map.find(key);
		if (it != s_branched_log_gpu_cache_map.end())
			return *(it->second);
	}

	BranchedSigCache c = build_branched_sig_cache(dimension, max_nodes, planar);
	BranchedLogForestCache fc = build_branched_log_forest_cache(c);
	uint64_t num_trees = c.total_length - 1;

	auto safe_narrow = [](const uint64_t* src, uint32_t* dst, size_t n) {
		for (size_t i = 0; i < n; ++i) {
			if (src[i] > UINT32_MAX)
				throw std::overflow_error("Branched log sig cache value exceeds uint32 range");
			dst[i] = static_cast<uint32_t>(src[i]);
		}
	};

	std::vector<uint32_t> forest_offsets32(fc.forest_offsets.size());
	std::vector<uint32_t> forest_trees32(fc.forest_trees.size());
	std::vector<uint32_t> forest_coprod_offsets32(fc.forest_coprod_offsets.size());
	std::vector<uint32_t> forest_coprod_data32(fc.forest_coprod_data.size());
	std::vector<uint32_t> single_tree_forest32(fc.single_tree_forest.size());
	safe_narrow(fc.forest_offsets.data(), forest_offsets32.data(), fc.forest_offsets.size());
	safe_narrow(fc.forest_trees.data(), forest_trees32.data(), fc.forest_trees.size());
	safe_narrow(fc.forest_coprod_offsets.data(), forest_coprod_offsets32.data(), fc.forest_coprod_offsets.size());
	safe_narrow(fc.forest_coprod_data.data(), forest_coprod_data32.data(), fc.forest_coprod_data.size());
	safe_narrow(fc.single_tree_forest.data(), single_tree_forest32.data(), fc.single_tree_forest.size());

	auto narrow32 = [](uint64_t v) -> uint32_t {
		if (v > UINT32_MAX) throw std::overflow_error("Branched log sig cache value exceeds uint32 range");
		return static_cast<uint32_t>(v);
	};

	auto gpu = std::make_unique<BranchedLogSigCacheGPU>();
	gpu->total_length = narrow32(c.total_length);
	gpu->num_trees = narrow32(num_trees);
	gpu->num_forests = narrow32(fc.forest_offsets.size() - 1);
	gpu->forest_trees_len = narrow32(fc.forest_trees.size());
	gpu->forest_coprod_data_len = narrow32(fc.forest_coprod_data.size());
	gpu->max_nodes = static_cast<int>(max_nodes);
	upload_branched_log(gpu->d_forest_offsets32, forest_offsets32.data(), forest_offsets32.size());
	upload_branched_log(gpu->d_forest_trees32, forest_trees32.data(), forest_trees32.size());
	upload_branched_log(gpu->d_forest_coprod_offsets32, forest_coprod_offsets32.data(), forest_coprod_offsets32.size());
	upload_branched_log(gpu->d_forest_coprod_data32, forest_coprod_data32.data(), forest_coprod_data32.size());
	upload_branched_log(gpu->d_single_tree_forest32, single_tree_forest32.data(), single_tree_forest32.size());

	std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
	auto [ins, _] = s_branched_log_gpu_cache_map.try_emplace(key, std::move(gpu));
	return *(ins->second);
}

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_to_log_sig_ker(
	const T* __restrict__ bsig,
	T* __restrict__ out,
	uint32_t total_len,
	uint32_t num_forests,
	const uint32_t* __restrict__ g_forest_offsets,
	const uint32_t* __restrict__ g_forest_trees,
	const uint32_t* __restrict__ g_forest_coprod_offsets,
	const uint32_t* __restrict__ g_forest_coprod_data,
	const uint32_t* __restrict__ g_single_tree_forest,
	uint32_t forest_trees_len,
	uint32_t forest_coprod_data_len,
	int max_nodes,
	bool scalar_term
) {
	const uint32_t batch_idx = blockIdx.y;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;
	const uint64_t stride = scalar_term ? total_len : total_len - 1;

	extern __shared__ char smem[];
	T* h = reinterpret_cast<T*>(smem);
	T* h_forest = h + total_len;
	T* power = h_forest + num_forests;
	T* next_power = power + num_forests;
	T* full_out = next_power + num_forests;
	uint32_t* s_forest_offsets = reinterpret_cast<uint32_t*>(full_out + total_len);
	uint32_t* s_forest_trees = s_forest_offsets + num_forests + 1;
	uint32_t* s_forest_coprod_offsets = s_forest_trees + forest_trees_len;
	uint32_t* s_forest_coprod_data = s_forest_coprod_offsets + num_forests + 1;
	uint32_t* s_single_tree_forest = s_forest_coprod_data + forest_coprod_data_len;

	for (uint32_t i = tid; i < num_forests + 1; i += blockDim.x) {
		s_forest_offsets[i] = g_forest_offsets[i];
		s_forest_coprod_offsets[i] = g_forest_coprod_offsets[i];
	}
	for (uint32_t i = tid; i < forest_trees_len; i += blockDim.x)
		s_forest_trees[i] = g_forest_trees[i];
	for (uint32_t i = tid; i < forest_coprod_data_len; i += blockDim.x)
		s_forest_coprod_data[i] = g_forest_coprod_data[i];
	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		s_single_tree_forest[i] = g_single_tree_forest[i];
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
		full_out[i] = T(0);
	}
	for (uint32_t i = tid; i < num_forests; i += blockDim.x) {
		h_forest[i] = T(0);
		power[i] = T(0);
		next_power[i] = T(0);
	}
	__syncthreads();

	const T* src = bsig + static_cast<uint64_t>(batch_idx) * stride;
	T* dst = out + static_cast<uint64_t>(batch_idx) * stride;
	for (uint32_t i = tid; i < num_trees; i += blockDim.x) {
		const T v = scalar_term ? src[i + 1] : src[i];
		h[i + 1] = v;
	}
	__syncthreads();

	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x) {
		T val = T(1);
		const uint32_t start = s_forest_offsets[forest_idx];
		const uint32_t end = s_forest_offsets[forest_idx + 1];
		for (uint32_t pos = start; pos < end; ++pos)
			val *= h[s_forest_trees[pos]];
		h_forest[forest_idx] = val;
		power[forest_idx] = val;
	}
	__syncthreads();

	for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
		full_out[i] = power[s_single_tree_forest[i]];
	__syncthreads();

	T* cur = power;
	T* next = next_power;
	for (int k = 2; k <= max_nodes; ++k) {
		for (uint32_t forest_idx = tid; forest_idx < num_forests; forest_idx += blockDim.x) {
			T val = T(0);
			const uint32_t start = s_forest_coprod_offsets[forest_idx];
			const uint32_t end = s_forest_coprod_offsets[forest_idx + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = s_forest_coprod_data[pos];
				const uint32_t right = s_forest_coprod_data[pos + 1];
				val += cur[left] * h_forest[right];
			}
			next[forest_idx] = val;
		}
		__syncthreads();

		const T coeff = (k % 2 == 0) ? T(-1) / T(k) : T(1) / T(k);
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			full_out[i] += coeff * next[s_single_tree_forest[i]];
		__syncthreads();

		T* tmp = cur;
		cur = next;
		next = tmp;
	}

	if (scalar_term) {
		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			dst[i] = (i == 0) ? T(0) : full_out[i];
	} else {
		for (uint32_t i = tid; i < num_trees; i += blockDim.x)
			dst[i] = full_out[i + 1];
	}
}

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_to_log_sig_backprop_ker(
	const T* __restrict__ bsig,
	const T* __restrict__ derivs,
	T* __restrict__ out,
	uint32_t total_len,
	uint32_t num_forests,
	const uint32_t* __restrict__ g_forest_offsets,
	const uint32_t* __restrict__ g_forest_trees,
	const uint32_t* __restrict__ g_forest_coprod_offsets,
	const uint32_t* __restrict__ g_forest_coprod_data,
	const uint32_t* __restrict__ g_single_tree_forest,
	uint32_t forest_trees_len,
	uint32_t forest_coprod_data_len,
	int max_nodes,
	bool scalar_term
) {
	const uint32_t batch_idx = blockIdx.y;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;
	const uint32_t levels = static_cast<uint32_t>(max_nodes + 1);
	const uint64_t stride = scalar_term ? total_len : total_len - 1;

	extern __shared__ char smem[];
	T* powers = reinterpret_cast<T*>(smem);
	T* d_powers = powers + static_cast<uint64_t>(levels) * num_forests;
	T* h = d_powers + static_cast<uint64_t>(levels) * num_forests;
	T* h_forest = h + total_len;
	T* d_h_forest = h_forest + num_forests;
	T* d_h_tree = d_h_forest + num_forests;
	T* full_derivs = d_h_tree + total_len;
	uint32_t* s_forest_offsets = reinterpret_cast<uint32_t*>(full_derivs + total_len);
	uint32_t* s_forest_trees = s_forest_offsets + num_forests + 1;
	uint32_t* s_forest_coprod_offsets = s_forest_trees + forest_trees_len;
	uint32_t* s_forest_coprod_data = s_forest_coprod_offsets + num_forests + 1;
	uint32_t* s_single_tree_forest = s_forest_coprod_data + forest_coprod_data_len;

	const uint64_t level_size = static_cast<uint64_t>(levels) * num_forests;
	for (uint64_t i = tid; i < level_size; i += blockDim.x) {
		powers[i] = T(0);
		d_powers[i] = T(0);
	}
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
		d_h_tree[i] = T(0);
		full_derivs[i] = T(0);
	}
	for (uint32_t i = tid; i < num_forests; i += blockDim.x) {
		h_forest[i] = T(0);
		d_h_forest[i] = T(0);
	}
	for (uint32_t i = tid; i < num_forests + 1; i += blockDim.x) {
		s_forest_offsets[i] = g_forest_offsets[i];
		s_forest_coprod_offsets[i] = g_forest_coprod_offsets[i];
	}
	for (uint32_t i = tid; i < forest_trees_len; i += blockDim.x)
		s_forest_trees[i] = g_forest_trees[i];
	for (uint32_t i = tid; i < forest_coprod_data_len; i += blockDim.x)
		s_forest_coprod_data[i] = g_forest_coprod_data[i];
	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		s_single_tree_forest[i] = g_single_tree_forest[i];
	__syncthreads();

	const T* src = bsig + static_cast<uint64_t>(batch_idx) * stride;
	const T* dsrc = derivs + static_cast<uint64_t>(batch_idx) * stride;
	T* dst = out + static_cast<uint64_t>(batch_idx) * stride;

	for (uint32_t i = tid; i < num_trees; i += blockDim.x) {
		const T v = scalar_term ? src[i + 1] : src[i];
		const T d = scalar_term ? dsrc[i + 1] : dsrc[i];
		h[i + 1] = v;
		full_derivs[i + 1] = d;
	}
	__syncthreads();

	T* p1 = powers + num_forests;
	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x) {
		T val = T(1);
		const uint32_t start = s_forest_offsets[forest_idx];
		const uint32_t end = s_forest_offsets[forest_idx + 1];
		for (uint32_t pos = start; pos < end; ++pos)
			val *= h[s_forest_trees[pos]];
		h_forest[forest_idx] = val;
		p1[forest_idx] = val;
	}
	__syncthreads();

	for (int k = 2; k <= max_nodes; ++k) {
		T* prev = powers + static_cast<uint64_t>(k - 1) * num_forests;
		T* next = powers + static_cast<uint64_t>(k) * num_forests;
		for (uint32_t forest_idx = tid; forest_idx < num_forests; forest_idx += blockDim.x) {
			T val = T(0);
			const uint32_t start = s_forest_coprod_offsets[forest_idx];
			const uint32_t end = s_forest_coprod_offsets[forest_idx + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = s_forest_coprod_data[pos];
				const uint32_t right = s_forest_coprod_data[pos + 1];
				val += prev[left] * h_forest[right];
			}
			next[forest_idx] = val;
		}
		__syncthreads();
	}

	for (int k = 1; k <= max_nodes; ++k) {
		const T coeff = (k == 1) ? T(1) : ((k % 2 == 0) ? T(-1) / T(k) : T(1) / T(k));
		T* dk = d_powers + static_cast<uint64_t>(k) * num_forests;
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			dk[s_single_tree_forest[i]] += coeff * full_derivs[i];
	}
	__syncthreads();

	for (int k = max_nodes; k >= 2; --k) {
		const T* prev = powers + static_cast<uint64_t>(k - 1) * num_forests;
		const T* d_cur = d_powers + static_cast<uint64_t>(k) * num_forests;
		T* d_prev = d_powers + static_cast<uint64_t>(k - 1) * num_forests;
		for (uint32_t forest_idx = tid; forest_idx < num_forests; forest_idx += blockDim.x) {
			const T d = d_cur[forest_idx];
			if (d == T(0))
				continue;
			const uint32_t start = s_forest_coprod_offsets[forest_idx];
			const uint32_t end = s_forest_coprod_offsets[forest_idx + 1];
			for (uint32_t pos = start; pos < end; pos += 2) {
				const uint32_t left = s_forest_coprod_data[pos];
				const uint32_t right = s_forest_coprod_data[pos + 1];
				myAtomicAdd(&d_prev[left], d * h_forest[right]);
				myAtomicAdd(&d_h_forest[right], d * prev[left]);
			}
		}
		__syncthreads();
	}

	T* d1 = d_powers + num_forests;
	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x)
		d_h_forest[forest_idx] += d1[forest_idx];
	__syncthreads();

	for (uint32_t forest_idx = tid + 1; forest_idx < num_forests; forest_idx += blockDim.x) {
		const T d = d_h_forest[forest_idx];
		if (d == T(0))
			continue;
		const uint32_t start = s_forest_offsets[forest_idx];
		const uint32_t end = s_forest_offsets[forest_idx + 1];
		for (uint32_t pos = start; pos < end; ++pos) {
			T partial = d;
			for (uint32_t other = start; other < end; ++other) {
				if (other != pos)
					partial *= h[s_forest_trees[other]];
			}
			myAtomicAdd(&d_h_tree[s_forest_trees[pos]], partial);
		}
	}
	__syncthreads();
	if (tid == 0) d_h_tree[0] = T(0);
	__syncthreads();

	if (scalar_term) {
		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			dst[i] = d_h_tree[i];
	} else {
		for (uint32_t i = tid; i < num_trees; i += blockDim.x)
			dst[i] = d_h_tree[i + 1];
	}
}

template<typename T>
void branched_sig_to_log_sig_cuda_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	bool scalar_term
) {
	const auto& gc = get_or_upload_branched_log_gpu_cache(dimension, max_nodes, planar);
	const uint32_t work_items = std::max(gc.num_trees, gc.num_forests);
	unsigned int block = static_cast<unsigned int>((work_items + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	size_t smem = (2 * static_cast<uint64_t>(gc.total_length) + 3 * static_cast<uint64_t>(gc.num_forests)) * sizeof(T)
		+ (2 * (static_cast<uint64_t>(gc.num_forests) + 1)
			+ static_cast<uint64_t>(gc.forest_trees_len)
			+ static_cast<uint64_t>(gc.forest_coprod_data_len)
			+ static_cast<uint64_t>(gc.total_length)) * sizeof(uint32_t);
	configure_dynamic_smem(
		branched_sig_to_log_sig_ker<T>, smem, "CUDA branched log sig");
	dim3 grid(1, static_cast<unsigned int>(batch_size));
	branched_sig_to_log_sig_ker<T><<<grid, block, smem>>>(
		bsig, out, gc.total_length, gc.num_forests,
		gc.d_forest_offsets32, gc.d_forest_trees32,
		gc.d_forest_coprod_offsets32, gc.d_forest_coprod_data32,
		gc.d_single_tree_forest32,
		gc.forest_trees_len, gc.forest_coprod_data_len,
		gc.max_nodes, scalar_term);
	cudaDeviceSynchronize();
	check_cuda_error();
}

template<typename T>
void branched_sig_to_log_sig_backprop_cuda_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	bool scalar_term
) {
	const auto& gc = get_or_upload_branched_log_gpu_cache(dimension, max_nodes, planar);
	const uint32_t work_items = std::max(gc.num_trees, gc.num_forests);
	unsigned int block = static_cast<unsigned int>((work_items + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) block = 1024;

	const uint64_t levels = static_cast<uint64_t>(gc.max_nodes + 1);
	size_t smem = (2 * levels * static_cast<uint64_t>(gc.num_forests)
			+ 2 * static_cast<uint64_t>(gc.num_forests)
			+ 3 * static_cast<uint64_t>(gc.total_length)) * sizeof(T)
		+ (2 * (static_cast<uint64_t>(gc.num_forests) + 1)
			+ static_cast<uint64_t>(gc.forest_trees_len)
			+ static_cast<uint64_t>(gc.forest_coprod_data_len)
			+ static_cast<uint64_t>(gc.total_length)) * sizeof(uint32_t);
	configure_dynamic_smem(
		branched_sig_to_log_sig_backprop_ker<T>, smem, "CUDA branched log sig backprop");
	dim3 grid(1, static_cast<unsigned int>(batch_size));
	branched_sig_to_log_sig_backprop_ker<T><<<grid, block, smem>>>(
		bsig, derivs, out, gc.total_length, gc.num_forests,
		gc.d_forest_offsets32, gc.d_forest_trees32,
		gc.d_forest_coprod_offsets32, gc.d_forest_coprod_data32,
		gc.d_single_tree_forest32,
		gc.forest_trees_len, gc.forest_coprod_data_len,
		gc.max_nodes, scalar_term);
	cudaDeviceSynchronize();
	check_cuda_error();
}

extern "C" {

	CUSIG_API int branched_sig_to_log_sig_cuda_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_cuda_<float>(bsig, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}

	CUSIG_API int branched_sig_to_log_sig_cuda_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_cuda_<double>(bsig, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}

	CUSIG_API int branched_sig_to_log_sig_backprop_cuda_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_backprop_cuda_<float>(bsig, derivs, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}

	CUSIG_API int branched_sig_to_log_sig_backprop_cuda_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, bool planar, bool scalar_term) noexcept {
		CUSIG_SAFE_CALL(branched_sig_to_log_sig_backprop_cuda_<double>(bsig, derivs, out, batch_size, dimension, max_nodes, planar, scalar_term));
	}

}
