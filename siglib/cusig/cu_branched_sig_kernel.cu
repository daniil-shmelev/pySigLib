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
#include "cu_runtime_utils.h"
#include "cu_sig_kernel.h"

#include <cstdint>
#include <stdexcept>

struct SigKernelParams {
	uint64_t length2;
	uint64_t dyadic_order_1, dyadic_order_2;
	uint64_t dyadic_length_1, dyadic_length_2;
	uint64_t main_dyadic_length, num_anti_diag;
	uint64_t gram_length, grid_length;
};

static void validate_sig_kernel_dims_(uint64_t length1, uint64_t length2,
	uint64_t dyadic_order_1, uint64_t dyadic_order_2) {
	if (length1 == 0 || length2 == 0)
		throw std::invalid_argument("sig_kernel_cuda: paths must have length >= 1");
	if (dyadic_order_1 >= 63 || dyadic_order_2 >= 63 || dyadic_order_1 + dyadic_order_2 >= 63)
		throw std::invalid_argument("sig_kernel_cuda: dyadic_order too large");
	if ((length1 - 1) > (UINT64_MAX >> dyadic_order_1) ||
	    (length2 - 1) > (UINT64_MAX >> dyadic_order_2))
		throw std::invalid_argument("sig_kernel_cuda: path length * dyadic refinement would overflow");
}

static SigKernelParams make_params(uint64_t length1_, uint64_t length2_,
	uint64_t dyadic_order_1_, uint64_t dyadic_order_2_) {
	SigKernelParams p;
	p.length2 = length2_;
	p.dyadic_order_1 = dyadic_order_1_;
	p.dyadic_order_2 = dyadic_order_2_;
	p.dyadic_length_1 = ((length1_ - 1) << dyadic_order_1_) + 1;
	p.dyadic_length_2 = ((length2_ - 1) << dyadic_order_2_) + 1;
	p.main_dyadic_length = p.dyadic_length_2 <= p.dyadic_length_1 ? p.dyadic_length_1 : p.dyadic_length_2;
	p.num_anti_diag = 33 + p.main_dyadic_length - 1;
	p.gram_length = (length1_ - 1) * (length2_ - 1);
	p.grid_length = p.dyadic_length_1 * p.dyadic_length_2;
	return p;
}

template<typename T>
__global__ void bsk_fill_kernel(T* ptr, T val, uint64_t n) {
	for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		idx < n; idx += static_cast<uint64_t>(blockDim.x) * gridDim.x)
		ptr[idx] = val;
}
template<typename T>
__device__ inline T bsk_exp(T x);

template<>
__device__ inline float bsk_exp<float>(float x) { return expf(x); }

template<>
__device__ inline double bsk_exp<double>(double x) { return exp(x); }

template<typename T>
__device__ T bsk_reduce_sum(T val, T* shared) {
	const unsigned tid = threadIdx.x;
	shared[tid] = val;
	__syncthreads();
	for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
		if (tid < stride)
			shared[tid] += shared[tid + stride];
		__syncthreads();
	}
	return shared[0];
}

template<typename T>
__device__ inline uint64_t bsk_gram_idx(uint64_t row, uint64_t col, const SigKernelParams& p) {
	const uint64_t ii = p.dyadic_order_1 == 0 ? row - 1 : ((row - 1) >> p.dyadic_order_1);
	const uint64_t jj = p.dyadic_order_2 == 0 ? col - 1 : ((col - 1) >> p.dyadic_order_2);
	return ii * (p.length2 - 1) + jj;
}

template<typename T, bool prev_ones>
__device__ void bsk_depth_step_block(
	const T* gram,
	const T* prev,
	T* curr,
	T* accum,
	const SigKernelParams p,
	T quarter_scale
) {
	const uint64_t dl1 = p.dyadic_length_1;
	const uint64_t dl2 = p.dyadic_length_2;
	const unsigned tid = threadIdx.x;

	for (uint64_t idx = tid; idx < p.grid_length; idx += blockDim.x) {
		const uint64_t row = idx / dl2;
		const uint64_t col = idx - row * dl2;
		accum[idx] = static_cast<T>(0.);
		if (row == 0 || col == 0)
			curr[idx] = static_cast<T>(1.);
	}
	__syncthreads();

	const uint64_t last_diag = dl1 + dl2 - 2;
	for (uint64_t diag = 2; diag <= last_diag; ++diag) {
		const uint64_t row_start = diag > dl2 - 1 ? diag - (dl2 - 1) : 1;
		const uint64_t row_end = diag - 1 < dl1 - 1 ? diag - 1 : dl1 - 1;
		if (row_start <= row_end) {
			const uint64_t span = row_end - row_start + 1;
			for (uint64_t k = tid; k < span; k += blockDim.x) {
				const uint64_t row = row_start + k;
				const uint64_t col = diag - row;
				const uint64_t idx = row * dl2 + col;
				const uint64_t up = idx - dl2;
				const uint64_t left = idx - 1;
				const uint64_t diag_idx = up - 1;
				const T p00 = prev_ones ? static_cast<T>(1.) : prev[diag_idx];
				const T p10 = prev_ones ? static_cast<T>(1.) : prev[left];
				const T p01 = prev_ones ? static_cast<T>(1.) : prev[up];
				const T p11 = prev_ones ? static_cast<T>(1.) : prev[idx];
				const T cell = gram[bsk_gram_idx<T>(row, col, p)] * quarter_scale * (p00 + p10 + p01 + p11);
				const T val = accum[up] + accum[left] - accum[diag_idx] + cell;
				accum[idx] = val;
				curr[idx] = bsk_exp<T>(val);
			}
		}
		__syncthreads();
	}
}

template<typename T>
__device__ T bsk_final_integral_block(
	const T* gram,
	const T* prev,
	const SigKernelParams p,
	T quarter_scale,
	T* shared
) {
	const uint64_t dl2 = p.dyadic_length_2;
	T total = static_cast<T>(0.);
	for (uint64_t lin = threadIdx.x; lin < (p.dyadic_length_1 - 1) * (p.dyadic_length_2 - 1); lin += blockDim.x) {
		const uint64_t row = lin / (p.dyadic_length_2 - 1) + 1;
		const uint64_t col = lin - (row - 1) * (p.dyadic_length_2 - 1) + 1;
		const uint64_t idx = row * dl2 + col;
		total += gram[bsk_gram_idx<T>(row, col, p)] * quarter_scale * (
			prev[idx - dl2 - 1] + prev[idx - 1] + prev[idx - dl2] + prev[idx]);
	}
	return bsk_reduce_sum(total, shared);
}

template<typename T>
__global__ void bsk_depth_one_scalar_kernel(
	const T* gram,
	T* out,
	const SigKernelParams p,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t block_id = batch_offset + local_batch_idx;
	const T* gram_ = gram + block_id * p.gram_length;
	__shared__ T shared[256];
	T total = static_cast<T>(0.);
	for (uint64_t i = threadIdx.x; i < p.gram_length; i += blockDim.x)
		total += gram_[i];
	total = bsk_reduce_sum(total, shared);
	if (threadIdx.x == 0)
		out[block_id] = bsk_exp<T>(total);
}

template<typename T>
__global__ void bsk_scalar_forward_kernel(
	const T* gram,
	T* out,
	uint64_t depth,
	T quarter_scale,
	const SigKernelParams p,
	T* work,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t block_id = batch_offset + local_batch_idx;
	const T* gram_ = gram + block_id * p.gram_length;
	T* work_ = work + block_id * 3 * p.grid_length;
	T* buf_a = work_;
	T* buf_b = buf_a + p.grid_length;
	T* accum = buf_b + p.grid_length;
	__shared__ T shared[256];

	bsk_depth_step_block<T, true>(gram_, nullptr, buf_a, accum, p, quarter_scale);
	T* prev = buf_a;
	T* curr = buf_b;

	for (uint64_t m = 2; m < depth; ++m) {
		bsk_depth_step_block<T, false>(gram_, prev, curr, accum, p, quarter_scale);
		T* tmp = prev;
		prev = curr;
		curr = tmp;
	}

	const T total = bsk_final_integral_block(gram_, prev, p, quarter_scale, shared);
	if (threadIdx.x == 0)
		out[block_id] = bsk_exp<T>(total);
}

template<typename T>
__global__ void bsk_grid_forward_kernel(
	const T* gram,
	T* out,
	uint64_t depth,
	T quarter_scale,
	const SigKernelParams p,
	T* work,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t block_id = batch_offset + local_batch_idx;
	const T* gram_ = gram + block_id * p.gram_length;
	T* out_ = out + block_id * p.grid_length;
	T* work_ = work + block_id * 3 * p.grid_length;
	T* buf_a = work_;
	T* buf_b = buf_a + p.grid_length;
	T* accum = buf_b + p.grid_length;

	if (depth == 1) {
		bsk_depth_step_block<T, true>(gram_, nullptr, out_, accum, p, quarter_scale);
		return;
	}

	bsk_depth_step_block<T, true>(gram_, nullptr, buf_a, accum, p, quarter_scale);
	T* prev = buf_a;
	T* curr = buf_b;
	for (uint64_t m = 2; m <= depth; ++m) {
		T* target = m == depth ? out_ : curr;
		bsk_depth_step_block<T, false>(gram_, prev, target, accum, p, quarter_scale);
		prev = target;
		curr = curr == buf_a ? buf_b : buf_a;
	}
}

template<typename T>
void branched_sig_kernel_cuda_(
	const T* const gram,
	T* const out,
	const uint64_t batch_size_,
	const uint64_t dimension_,
	const uint64_t length1_,
	const uint64_t length2_,
	const uint64_t depth_,
	const uint64_t dyadic_order_1_,
	const uint64_t dyadic_order_2_,
	const bool return_grid
) {
	if (dimension_ == 0) { throw std::invalid_argument("branched signature kernel received path of dimension 0"); }
	validate_sig_kernel_dims_(length1_, length2_, dyadic_order_1_, dyadic_order_2_);

	const SigKernelParams p = make_params(length1_, length2_, dyadic_order_1_, dyadic_order_2_);
	const uint64_t result_length = return_grid ? p.grid_length : 1;

	if (!gram || depth_ == 0 || p.gram_length == 0) {
		bsk_fill_kernel<<<
			make_cuda_1d_grid(batch_size_ * result_length, 256), 256U>>>(
			out, static_cast<T>(1.), batch_size_ * result_length);
		check_cuda_kernel_launch();
		return;
	}

	const T cell_scale = static_cast<T>(1.) / (1ULL << (dyadic_order_1_ + dyadic_order_2_));
	const T quarter_scale = static_cast<T>(0.25) * cell_scale;
	constexpr unsigned int threads = 256U;

	if (!return_grid && depth_ == 1) {
		for (uint64_t batch_offset = 0; batch_offset < batch_size_;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size_, batch_offset);
			bsk_depth_one_scalar_kernel<T><<<batch_chunk.grid, threads>>>(
				gram, out, p, batch_chunk.offset, batch_chunk.size);
			batch_offset += batch_chunk.size;
		}
		check_cuda_kernel_launch();
		return;
	}

	if (!return_grid) {
		CudaBuf<T> work(3 * p.grid_length * batch_size_ * sizeof(T));
		for (uint64_t batch_offset = 0; batch_offset < batch_size_;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size_, batch_offset);
			bsk_scalar_forward_kernel<T><<<batch_chunk.grid, threads>>>(
				gram, out, depth_, quarter_scale, p, work.get(),
				batch_chunk.offset, batch_chunk.size);
			batch_offset += batch_chunk.size;
		}
		check_cuda_kernel_launch();
		return;
	}

	CudaBuf<T> work(3 * p.grid_length * batch_size_ * sizeof(T));
	for (uint64_t batch_offset = 0; batch_offset < batch_size_;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size_, batch_offset);
		bsk_grid_forward_kernel<T><<<batch_chunk.grid, threads>>>(
			gram, out, depth_, quarter_scale, p, work.get(),
			batch_chunk.offset, batch_chunk.size);
		batch_offset += batch_chunk.size;
	}

	check_cuda_kernel_launch();
}

template<typename T>
__device__ void bsk_build_stack_block(
	const T* gram,
	T* stack,
	T* accum,
	uint64_t depth,
	const SigKernelParams p,
	T quarter_scale
) {
	for (uint64_t i = threadIdx.x; i < p.grid_length; i += blockDim.x)
		stack[i] = static_cast<T>(1.);
	__syncthreads();

	for (uint64_t m = 1; m <= depth; ++m)
		bsk_depth_step_block<T, false>(
			gram, stack + (m - 1) * p.grid_length, stack + m * p.grid_length,
			accum, p, quarter_scale);
}

template<typename T>
__global__ void bsk_depth_one_backprop_kernel(
	const T* gram,
	T* out,
	const T* derivs,
	const SigKernelParams p,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t block_id = batch_offset + local_batch_idx;
	const T* gram_ = gram + block_id * p.gram_length;
	T* out_ = out + block_id * p.gram_length;
	__shared__ T shared[256];
	T total = static_cast<T>(0.);
	for (uint64_t i = threadIdx.x; i < p.gram_length; i += blockDim.x)
		total += gram_[i];
	total = bsk_reduce_sum(total, shared);
	const T val = derivs[block_id] * bsk_exp<T>(total);
	for (uint64_t i = threadIdx.x; i < p.gram_length; i += blockDim.x)
		out_[i] = val;
}

template<typename T>
__device__ void bsk_zero_grid(T* ptr, uint64_t n) {
	for (uint64_t i = threadIdx.x; i < n; i += blockDim.x)
		ptr[i] = static_cast<T>(0.);
	__syncthreads();
}

template<typename T>
__device__ void bsk_seed_scalar_reverse_block(
	const T* gram,
	const T* prev,
	T* out,
	T* d_curr,
	const T scalar_deriv,
	const SigKernelParams p,
	T quarter_scale
) {
	const uint64_t dl2 = p.dyadic_length_2;
	for (uint64_t lin = threadIdx.x; lin < (p.dyadic_length_1 - 1) * (p.dyadic_length_2 - 1); lin += blockDim.x) {
		const uint64_t row = lin / (p.dyadic_length_2 - 1) + 1;
		const uint64_t col = lin - (row - 1) * (p.dyadic_length_2 - 1) + 1;
		const uint64_t idx = row * dl2 + col;
		const uint64_t gi = bsk_gram_idx<T>(row, col, p);
		const T weight = scalar_deriv * quarter_scale;
		const T gram_weight = weight * gram[gi];
		const uint64_t i00 = idx - dl2 - 1;
		const uint64_t i10 = idx - 1;
		const uint64_t i01 = idx - dl2;
		myAtomicAdd(out + gi, weight * (prev[i00] + prev[i10] + prev[i01] + prev[idx]));
		myAtomicAdd(d_curr + i00, gram_weight);
		myAtomicAdd(d_curr + i10, gram_weight);
		myAtomicAdd(d_curr + i01, gram_weight);
		myAtomicAdd(d_curr + idx, gram_weight);
	}
	__threadfence_block();
	__syncthreads();
}

template<typename T>
__device__ void bsk_reverse_depth_block(
	const T* gram,
	const T* prev,
	const T* curr,
	T* out,
	T* d_curr,
	T* d_prev,
	T* d_inc,
	const SigKernelParams p,
	T quarter_scale
) {
	const uint64_t dl1 = p.dyadic_length_1;
	const uint64_t dl2 = p.dyadic_length_2;

	bsk_zero_grid(d_prev, p.grid_length);
	bsk_zero_grid(d_inc, p.grid_length);

	for (uint64_t idx = threadIdx.x; idx < p.grid_length; idx += blockDim.x) {
		const uint64_t row = idx / dl2;
		const uint64_t col = idx - row * dl2;
		if (row != 0 && col != 0)
			d_inc[idx] = d_curr[idx] * curr[idx];
	}
	__syncthreads();

	for (uint64_t diag = dl1 + dl2 - 2; diag >= 2; --diag) {
		const uint64_t row_start = diag > dl2 - 1 ? diag - (dl2 - 1) : 1;
		const uint64_t row_end = diag - 1 < dl1 - 1 ? diag - 1 : dl1 - 1;
		if (row_start <= row_end) {
			const uint64_t span = row_end - row_start + 1;
			for (uint64_t k = threadIdx.x; k < span; k += blockDim.x) {
				const uint64_t row = row_start + k;
				const uint64_t col = diag - row;
				const uint64_t idx = row * dl2 + col;
				T val = d_inc[idx];
				if (row + 1 < dl1) val += d_inc[idx + dl2];
				if (col + 1 < dl2) val += d_inc[idx + 1];
				if (row + 1 < dl1 && col + 1 < dl2) val -= d_inc[idx + dl2 + 1];
				d_inc[idx] = val;
			}
		}
		__syncthreads();
		if (diag == 2) break;
	}

	for (uint64_t lin = threadIdx.x; lin < (dl1 - 1) * (dl2 - 1); lin += blockDim.x) {
		const uint64_t row = lin / (dl2 - 1) + 1;
		const uint64_t col = lin - (row - 1) * (dl2 - 1) + 1;
		const uint64_t idx = row * dl2 + col;
		const uint64_t gi = bsk_gram_idx<T>(row, col, p);
		const T weight = d_inc[idx] * quarter_scale;
		const T gram_weight = weight * gram[gi];
		const uint64_t i00 = idx - dl2 - 1;
		const uint64_t i10 = idx - 1;
		const uint64_t i01 = idx - dl2;
		myAtomicAdd(out + gi, weight * (prev[i00] + prev[i10] + prev[i01] + prev[idx]));
		myAtomicAdd(d_prev + i00, gram_weight);
		myAtomicAdd(d_prev + i10, gram_weight);
		myAtomicAdd(d_prev + i01, gram_weight);
		myAtomicAdd(d_prev + idx, gram_weight);
	}
	__threadfence_block();
	__syncthreads();
}

template<typename T>
__global__ void bsk_backprop_kernel(
	const T* gram,
	T* out,
	const T* derivs,
	const T* k_stack,
	uint64_t depth,
	bool return_grid,
	T quarter_scale,
	const SigKernelParams p,
	T* work,
	uint64_t workspace_levels,
	uint64_t batch_offset,
	uint64_t batch_chunk_size
) {
	const uint64_t local_batch_idx = cuda_batch_index();
	if (local_batch_idx >= batch_chunk_size) return;
	const uint64_t block_id = batch_offset + local_batch_idx;
	const uint64_t derivs_stride = return_grid ? p.grid_length : 1;
	const uint64_t external_stack_stride = (depth + 1) * p.grid_length;
	const T* gram_ = gram + block_id * p.gram_length;
	T* out_ = out + block_id * p.gram_length;
	const T* derivs_ = derivs + block_id * derivs_stride;
	T* work_ = work + block_id * workspace_levels * p.grid_length;
	__shared__ T shared[256];

	for (uint64_t i = threadIdx.x; i < p.gram_length; i += blockDim.x)
		out_[i] = static_cast<T>(0.);
	__syncthreads();

	const uint64_t build_depth = return_grid ? depth : depth - 1;
	T* stack_storage = work_;
	T* accum = stack_storage + (k_stack ? 0 : (build_depth + 1) * p.grid_length);
	const T* stack = k_stack ? k_stack + block_id * external_stack_stride : stack_storage;
	if (!k_stack)
		bsk_build_stack_block(gram_, stack_storage, accum, build_depth, p, quarter_scale);

	T* d_curr = work_ + (k_stack ? 0 : (build_depth + 2) * p.grid_length);
	T* d_prev = d_curr + p.grid_length;
	T* d_inc = d_prev + p.grid_length;
	uint64_t start_depth = depth;

	if (return_grid) {
		for (uint64_t i = threadIdx.x; i < p.grid_length; i += blockDim.x)
			d_curr[i] = derivs_[i];
		__syncthreads();
	}
	else {
		const T* prev = stack + (depth - 1) * p.grid_length;
		const T total = bsk_final_integral_block(gram_, prev, p, quarter_scale, shared);
		const T scalar_deriv = (*derivs_) * bsk_exp<T>(total);
		bsk_zero_grid(d_curr, p.grid_length);
		bsk_seed_scalar_reverse_block(gram_, prev, out_, d_curr, scalar_deriv, p, quarter_scale);
		start_depth = depth - 1;
	}

	for (uint64_t m = start_depth; m > 0; --m) {
		const T* prev = stack + (m - 1) * p.grid_length;
		const T* curr = stack + m * p.grid_length;
		bsk_reverse_depth_block(gram_, prev, curr, out_, d_curr, d_prev, d_inc, p, quarter_scale);
		T* tmp = d_curr;
		d_curr = d_prev;
		d_prev = tmp;
	}
}

template<typename T>
void branched_sig_kernel_backprop_cuda_(
	const T* gram,
	T* const out,
	const T* derivs,
	const T* k_stack,
	const uint64_t batch_size_,
	const uint64_t dimension_,
	const uint64_t length1_,
	const uint64_t length2_,
	const uint64_t depth_,
	const uint64_t dyadic_order_1_,
	const uint64_t dyadic_order_2_,
	const bool return_grid
) {
	if (dimension_ == 0) { throw std::invalid_argument("branched signature kernel received path of dimension 0"); }
	validate_sig_kernel_dims_(length1_, length2_, dyadic_order_1_, dyadic_order_2_);

	const SigKernelParams p = make_params(length1_, length2_, dyadic_order_1_, dyadic_order_2_);

	if (!gram || depth_ == 0 || p.gram_length == 0) {
		CUDA_CHECK(cudaMemset(out, 0, batch_size_ * p.gram_length * sizeof(T)));
		return;
	}

	const T cell_scale = static_cast<T>(1.) / (1ULL << (dyadic_order_1_ + dyadic_order_2_));
	const T quarter_scale = static_cast<T>(0.25) * cell_scale;
	constexpr unsigned int threads = 256U;

	if (!return_grid && depth_ == 1) {
		for (uint64_t batch_offset = 0; batch_offset < batch_size_;) {
			const auto batch_chunk = make_cuda_batch_grid_chunk(
				1, batch_size_, batch_offset);
			bsk_depth_one_backprop_kernel<T><<<batch_chunk.grid, threads>>>(
				gram, out, derivs, p, batch_chunk.offset, batch_chunk.size);
			batch_offset += batch_chunk.size;
		}
		check_cuda_kernel_launch();
		return;
	}

	const uint64_t build_depth = return_grid ? depth_ : depth_ - 1;
	const uint64_t workspace_levels = k_stack ? 3 : build_depth + 5;
	CudaBuf<T> work(workspace_levels * p.grid_length * batch_size_ * sizeof(T));

	for (uint64_t batch_offset = 0; batch_offset < batch_size_;) {
		const auto batch_chunk = make_cuda_batch_grid_chunk(
			1, batch_size_, batch_offset);
		bsk_backprop_kernel<T><<<batch_chunk.grid, threads>>>(
			gram, out, derivs, k_stack, depth_, return_grid, quarter_scale,
			p, work.get(), workspace_levels,
			batch_chunk.offset, batch_chunk.size);
		batch_offset += batch_chunk.size;
	}

	check_cuda_kernel_launch();
}

#include "cu_macros.h"

extern "C" {

	CUSIG_API int branched_sig_kernel_cuda_f(const float* const gram, float* const out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t depth, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(branched_sig_kernel_cuda_<float>(gram, out, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int branched_sig_kernel_cuda_d(const double* const gram, double* const out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t depth, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(branched_sig_kernel_cuda_<double>(gram, out, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int branched_sig_kernel_backprop_cuda_f(const float* const gram, float* const out, const float* const derivs, const float* const k_stack, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t depth, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(branched_sig_kernel_backprop_cuda_<float>(gram, out, derivs, k_stack, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid));
	}

	CUSIG_API int branched_sig_kernel_backprop_cuda_d(const double* const gram, double* const out, const double* const derivs, const double* const k_stack, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t depth, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept {
		CUSIG_SAFE_CALL(branched_sig_kernel_backprop_cuda_<double>(gram, out, derivs, k_stack, batch_size, dimension, length1, length2, depth, dyadic_order_1, dyadic_order_2, return_grid));
	}
}
