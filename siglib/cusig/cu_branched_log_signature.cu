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
#include "../shared/branched_cache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct BranchedLogSigCacheGPU {
	uint32_t* d_coprod_data32 = nullptr;
	uint32_t* d_coprod_offsets32 = nullptr;
	uint32_t* d_order_index32 = nullptr;

	uint32_t total_length = 0;
	uint32_t num_trees = 0;
	uint32_t coprod_data_len = 0;
	int max_nodes = 0;

	BranchedLogSigCacheGPU() = default;
	BranchedLogSigCacheGPU(const BranchedLogSigCacheGPU&) = delete;
	BranchedLogSigCacheGPU& operator=(const BranchedLogSigCacheGPU&) = delete;

	~BranchedLogSigCacheGPU() {
		if (d_coprod_data32) cudaFree(d_coprod_data32);
		if (d_coprod_offsets32) cudaFree(d_coprod_offsets32);
		if (d_order_index32) cudaFree(d_order_index32);
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

template<typename Kernel>
static void configure_dynamic_smem(Kernel kernel, size_t smem, const char* op_name) {
	int device = 0;
	int max_default = 0;
	int max_optin = 0;
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, device));
	CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
	if (max_optin == 0) max_optin = max_default;

	if (smem > static_cast<size_t>(max_optin)) {
		throw std::invalid_argument(
			std::string(op_name) + " requires more dynamic shared memory than this CUDA device supports");
	}
	if (smem > static_cast<size_t>(max_default)) {
		CUDA_CHECK(cudaFuncSetAttribute(
			kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
	}
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
	uint64_t num_trees = c.total_length - 1;

	auto safe_narrow = [](const uint64_t* src, uint32_t* dst, size_t n) {
		for (size_t i = 0; i < n; ++i) {
			if (src[i] > UINT32_MAX)
				throw std::overflow_error("Branched log sig cache value exceeds uint32 range");
			dst[i] = static_cast<uint32_t>(src[i]);
		}
	};

	std::vector<uint32_t> coprod_data32(c.coproduct_data.size());
	std::vector<uint32_t> coprod_offsets32(c.coproduct_offsets.size());
	std::vector<uint32_t> order_index32(c.order_index.size());
	safe_narrow(c.coproduct_data.data(), coprod_data32.data(), c.coproduct_data.size());
	safe_narrow(c.coproduct_offsets.data(), coprod_offsets32.data(), c.coproduct_offsets.size());
	safe_narrow(c.order_index.data(), order_index32.data(), c.order_index.size());

	auto narrow32 = [](uint64_t v) -> uint32_t {
		if (v > UINT32_MAX) throw std::overflow_error("Branched log sig cache value exceeds uint32 range");
		return static_cast<uint32_t>(v);
	};

	auto gpu = std::make_unique<BranchedLogSigCacheGPU>();
	gpu->total_length = narrow32(c.total_length);
	gpu->num_trees = narrow32(num_trees);
	gpu->coprod_data_len = narrow32(c.coproduct_data.size());
	gpu->max_nodes = static_cast<int>(max_nodes);
	upload_branched_log(gpu->d_coprod_data32, coprod_data32.data(), coprod_data32.size());
	upload_branched_log(gpu->d_coprod_offsets32, coprod_offsets32.data(), coprod_offsets32.size());
	upload_branched_log(gpu->d_order_index32, order_index32.data(), order_index32.size());

	std::lock_guard<std::mutex> lock(s_branched_log_gpu_cache_mu);
	auto [ins, _] = s_branched_log_gpu_cache_map.try_emplace(key, std::move(gpu));
	return *(ins->second);
}

template<typename T>
__device__ __forceinline__ void butcher_product_tree_device(
	const T* X,
	const T* Y,
	T* out,
	uint32_t tid,
	const uint32_t* coprod_data,
	const uint32_t* coprod_offsets
) {
	const uint32_t flat_idx = tid + 1;
	T val = X[flat_idx] * Y[0] + X[0] * Y[flat_idx];

	uint32_t pos = coprod_offsets[tid];
	const uint32_t pos_end = coprod_offsets[tid + 1];
	while (pos < pos_end) {
		const uint32_t num_forest = coprod_data[pos++];
		const uint32_t trunk_flat = coprod_data[pos++];
		T term = Y[trunk_flat];
		for (uint32_t j = 0; j < num_forest; ++j)
			term *= X[coprod_data[pos++]];
		val += term;
	}
	out[flat_idx] = val;
}

template<typename T>
__device__ __forceinline__ void butcher_product_deriv_tree_device(
	const T* X,
	const T* Y,
	const T* d_out,
	T* d_X,
	T* d_Y,
	uint32_t tid,
	const uint32_t* coprod_data,
	const uint32_t* coprod_offsets
) {
	const uint32_t flat_idx = tid + 1;
	const T d = d_out[flat_idx];
	if (d == T(0)) return;

	myAtomicAdd(&d_X[flat_idx], d * Y[0]);
	myAtomicAdd(&d_Y[0], d * X[flat_idx]);
	myAtomicAdd(&d_X[0], d * Y[flat_idx]);
	myAtomicAdd(&d_Y[flat_idx], d * X[0]);

	uint32_t pos = coprod_offsets[tid];
	const uint32_t pos_end = coprod_offsets[tid + 1];
	while (pos < pos_end) {
		const uint32_t num_forest = coprod_data[pos++];
		const uint32_t trunk_flat = coprod_data[pos++];
		const uint32_t forest_start = pos;
		T forest_product = T(1);

		for (uint32_t j = 0; j < num_forest; ++j)
			forest_product *= X[coprod_data[pos++]];

		myAtomicAdd(&d_Y[trunk_flat], d * forest_product);
		for (uint32_t k = 0; k < num_forest; ++k) {
			const uint32_t fk_flat = coprod_data[forest_start + k];
			T partial = d * Y[trunk_flat];
			for (uint32_t j = 0; j < num_forest; ++j) {
				if (j != k)
					partial *= X[coprod_data[forest_start + j]];
			}
			myAtomicAdd(&d_X[fk_flat], partial);
		}
	}
}

template<typename T>
__global__ __launch_bounds__(1024)
void branched_sig_to_log_sig_ker(
	const T* __restrict__ bsig,
	T* __restrict__ out,
	uint32_t total_len,
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	uint32_t coprod_data_len,
	int max_nodes,
	bool scalar_term
) {
	const uint32_t batch_idx = blockIdx.y;
	const uint32_t tid = threadIdx.x;
	const uint32_t num_trees = total_len - 1;
	const uint64_t stride = scalar_term ? total_len : total_len - 1;

	extern __shared__ char smem[];
	T* h = reinterpret_cast<T*>(smem);
	T* power = h + total_len;
	T* next_power = power + total_len;
	T* full_out = next_power + total_len;
	uint32_t* s_coprod_data = reinterpret_cast<uint32_t*>(full_out + total_len);
	uint32_t* s_coprod_offsets = s_coprod_data + coprod_data_len;

	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		s_coprod_data[i] = g_coprod_data[i];
	for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
		s_coprod_offsets[i] = g_coprod_offsets[i];
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
		power[i] = T(0);
		next_power[i] = T(0);
		full_out[i] = T(0);
	}
	__syncthreads();

	const T* src = bsig + static_cast<uint64_t>(batch_idx) * stride;
	T* dst = out + static_cast<uint64_t>(batch_idx) * stride;
	for (uint32_t i = tid; i < num_trees; i += blockDim.x) {
		const T v = scalar_term ? src[i + 1] : src[i];
		h[i + 1] = v;
		power[i + 1] = v;
		full_out[i + 1] = v;
	}
	__syncthreads();

	T* cur = power;
	T* next = next_power;
	for (int k = 2; k <= max_nodes; ++k) {
		if (tid == 0) next[0] = T(0);
		if (tid < num_trees)
			butcher_product_tree_device(cur, h, next, tid, s_coprod_data, s_coprod_offsets);
		__syncthreads();

		const T coeff = (k % 2 == 0) ? T(-1) / T(k) : T(1) / T(k);
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			full_out[i] += coeff * next[i];
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
	const uint32_t* __restrict__ g_coprod_data,
	const uint32_t* __restrict__ g_coprod_offsets,
	uint32_t coprod_data_len,
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
	T* d_powers = powers + static_cast<uint64_t>(levels) * total_len;
	T* h = d_powers + static_cast<uint64_t>(levels) * total_len;
	T* d_h = h + total_len;
	T* full_derivs = d_h + total_len;
	uint32_t* s_coprod_data = reinterpret_cast<uint32_t*>(full_derivs + total_len);
	uint32_t* s_coprod_offsets = s_coprod_data + coprod_data_len;

	const uint64_t level_size = static_cast<uint64_t>(levels) * total_len;
	for (uint64_t i = tid; i < level_size; i += blockDim.x) {
		powers[i] = T(0);
		d_powers[i] = T(0);
	}
	for (uint32_t i = tid; i < total_len; i += blockDim.x) {
		h[i] = T(0);
		d_h[i] = T(0);
		full_derivs[i] = T(0);
	}
	for (uint32_t i = tid; i < coprod_data_len; i += blockDim.x)
		s_coprod_data[i] = g_coprod_data[i];
	for (uint32_t i = tid; i < num_trees + 1; i += blockDim.x)
		s_coprod_offsets[i] = g_coprod_offsets[i];
	__syncthreads();

	const T* src = bsig + static_cast<uint64_t>(batch_idx) * stride;
	const T* dsrc = derivs + static_cast<uint64_t>(batch_idx) * stride;
	T* dst = out + static_cast<uint64_t>(batch_idx) * stride;

	T* p1 = powers + total_len;
	for (uint32_t i = tid; i < num_trees; i += blockDim.x) {
		const T v = scalar_term ? src[i + 1] : src[i];
		const T d = scalar_term ? dsrc[i + 1] : dsrc[i];
		h[i + 1] = v;
		p1[i + 1] = v;
		full_derivs[i + 1] = d;
	}
	__syncthreads();

	for (int k = 2; k <= max_nodes; ++k) {
		T* prev = powers + static_cast<uint64_t>(k - 1) * total_len;
		T* next = powers + static_cast<uint64_t>(k) * total_len;
		if (tid == 0) next[0] = T(0);
		if (tid < num_trees)
			butcher_product_tree_device(prev, h, next, tid, s_coprod_data, s_coprod_offsets);
		__syncthreads();
	}

	for (int k = 1; k <= max_nodes; ++k) {
		const T coeff = (k == 1) ? T(1) : ((k % 2 == 0) ? T(-1) / T(k) : T(1) / T(k));
		T* dk = d_powers + static_cast<uint64_t>(k) * total_len;
		for (uint32_t i = tid + 1; i < total_len; i += blockDim.x)
			dk[i] = coeff * full_derivs[i];
	}
	__syncthreads();

	for (int k = max_nodes; k >= 2; --k) {
		const T* X = powers + static_cast<uint64_t>(k - 1) * total_len;
		const T* Y = h;
		const T* d_out = d_powers + static_cast<uint64_t>(k) * total_len;
		T* d_X = d_powers + static_cast<uint64_t>(k - 1) * total_len;
		T* d_Y = d_h;
		if (tid == 0) {
			myAtomicAdd(&d_X[0], d_out[0] * Y[0]);
			myAtomicAdd(&d_Y[0], d_out[0] * X[0]);
		}
		if (tid < num_trees)
			butcher_product_deriv_tree_device(X, Y, d_out, d_X, d_Y, tid, s_coprod_data, s_coprod_offsets);
		__syncthreads();
	}

	T* d1 = d_powers + total_len;
	for (uint32_t i = tid; i < total_len; i += blockDim.x)
		d_h[i] += d1[i];
	__syncthreads();
	if (tid == 0) d_h[0] = T(0);
	__syncthreads();

	if (scalar_term) {
		for (uint32_t i = tid; i < total_len; i += blockDim.x)
			dst[i] = d_h[i];
	} else {
		for (uint32_t i = tid; i < num_trees; i += blockDim.x)
			dst[i] = d_h[i + 1];
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
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) throw std::invalid_argument("CUDA branched log sig: num_trees > 1024 not supported");

	size_t smem = 4 * gc.total_length * sizeof(T)
		+ gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t);
	configure_dynamic_smem(
		branched_sig_to_log_sig_ker<T>, smem, "CUDA branched log sig");
	dim3 grid(1, static_cast<unsigned int>(batch_size));
	branched_sig_to_log_sig_ker<T><<<grid, block, smem>>>(
		bsig, out, gc.total_length,
		gc.d_coprod_data32, gc.d_coprod_offsets32,
		gc.coprod_data_len, gc.max_nodes, scalar_term);
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
	unsigned int block = static_cast<unsigned int>((gc.num_trees + 31u) & ~31u);
	if (block < 32) block = 32;
	if (block > 1024) throw std::invalid_argument("CUDA branched log sig backprop: num_trees > 1024 not supported");

	const uint64_t levels = static_cast<uint64_t>(gc.max_nodes + 1);
	size_t smem = (2 * levels + 3) * gc.total_length * sizeof(T)
		+ gc.coprod_data_len * sizeof(uint32_t)
		+ (gc.num_trees + 1) * sizeof(uint32_t);
	configure_dynamic_smem(
		branched_sig_to_log_sig_backprop_ker<T>, smem, "CUDA branched log sig backprop");
	dim3 grid(1, static_cast<unsigned int>(batch_size));
	branched_sig_to_log_sig_backprop_ker<T><<<grid, block, smem>>>(
		bsig, derivs, out, gc.total_length,
		gc.d_coprod_data32, gc.d_coprod_offsets32,
		gc.coprod_data_len, gc.max_nodes, scalar_term);
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
