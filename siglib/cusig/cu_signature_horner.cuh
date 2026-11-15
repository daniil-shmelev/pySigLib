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

namespace cuda_horner {

__host__ __device__ constexpr uint64_t power(int dimension, int level) {
	uint64_t result = 1;
	for (int i = 0; i < level; ++i) result *= dimension;
	return result;
}

// Each thread stores a prefix chain and the complete subtree below its last word.
template<typename T, int Dimension, int Degree, int Tail>
struct Tree {
	static constexpr int prefix = Degree - Tail;
	static constexpr int tail_width = power(Dimension, Tail);
	static constexpr int tail_size = (power(Dimension, Tail + 1) - 1) / (Dimension - 1);
	static constexpr int work_size = (power(Dimension, Tail) - 1) / (Dimension - 1);
	static constexpr int size = prefix + tail_size;
	T state[size];
	int letters[prefix];

	__device__ __forceinline__ explicit Tree(uint64_t word) {
		#pragma unroll
		for (int i = prefix - 1; i >= 0; --i) {
			letters[i] = static_cast<int>(word % Dimension);
			word /= Dimension;
		}
	}

	__device__ __forceinline__ void clear() {
		#pragma unroll
		for (int i = 0; i < size; ++i) state[i] = T(0);
		state[0] = T(1);
	}

	__host__ __device__ static constexpr uint64_t offset(int level) {
		return (power(Dimension, level) - 1) / (Dimension - 1);
	}

	__device__ __forceinline__ static T select(const T (&values)[Dimension], int letter) {
		T result = values[0];
		#pragma unroll
		for (int i = 1; i < Dimension; ++i)
			if (letter == i) result = values[i];
		return result;
	}

	__device__ __forceinline__ static void add(T (&values)[Dimension], int letter, T value) {
		#pragma unroll
		for (int i = 0; i < Dimension; ++i)
			if (letter == i) values[i] += value;
	}

	__device__ __forceinline__ void update(const T (&increment)[Dimension]) {
		T work[work_size];
		#pragma unroll
		for (int tail = Tail; tail > 0; --tail) {
			T common = T(0);
			#pragma unroll
			for (int i = 0; i < prefix; ++i)
				common = select(increment, letters[i]) * (T(1) / T(prefix + tail - i)) * (state[i] + common);
			work[0] = state[prefix] + common;
			// Fixed bounds let nvcc unroll the nested loops and keep state in registers.
			#pragma unroll
			for (int level = 1; level < Tail; ++level) {
				#pragma unroll
				for (int word = 0; word < tail_width; ++word) {
					if (level < tail && word < power(Dimension, level))
						work[offset(level) + word] = state[prefix + offset(level) + word]
							+ work[offset(level - 1) + word / Dimension]
								* increment[word % Dimension] * (T(1) / T(tail - level + 1));
				}
			}
			#pragma unroll
			for (int word = 0; word < tail_width; ++word)
				if (word < power(Dimension, tail))
					state[prefix + offset(tail) + word] += work[offset(tail - 1) + word / Dimension]
						* increment[word % Dimension];
		}
		#pragma unroll
		for (int level = prefix; level > 0; --level) {
			T value = T(0);
			#pragma unroll
			for (int i = 0; i < level; ++i)
				value = select(increment, letters[i]) * (T(1) / T(level - i)) * (state[i] + value);
			state[level] += value;
		}
	}

	__device__ __forceinline__ void store(T* output, uint64_t prefix_word, bool scalar) const {
		#pragma unroll
		for (int level = 1; level <= Tail; ++level) {
			#pragma unroll
			for (int word = 0; word < tail_width; ++word)
				if (word < power(Dimension, level))
					output[offset(prefix + level) + scalar - 1 + prefix_word * power(Dimension, level) + word]
						= state[prefix + offset(level) + word];
		}
		#pragma unroll
		for (int level = 1; level <= prefix; ++level) {
			const uint64_t divisor = power(Dimension, prefix - level);
			if (prefix_word % divisor == 0)
				output[offset(level) + scalar - 1 + prefix_word / divisor] = state[level];
		}
		if (scalar && prefix_word == 0) output[0] = T(1);
	}

	__device__ __forceinline__ void load(const T* signature, const T* seeds, T (&adjoint)[size], uint64_t prefix_word, bool scalar) {
		state[0] = T(1);
		adjoint[0] = T(0);
		#pragma unroll
		for (int level = 1; level <= prefix; ++level) {
			const uint64_t divisor = power(Dimension, prefix - level);
			const uint64_t index = offset(level) + scalar - 1 + prefix_word / divisor;
			state[level] = signature[index];
			adjoint[level] = prefix_word % divisor == 0 ? seeds[index] : T(0);
		}
		#pragma unroll
		for (int level = 1; level <= Tail; ++level) {
			#pragma unroll
			for (int word = 0; word < tail_width; ++word) {
				if (word < power(Dimension, level)) {
					const uint64_t index = offset(prefix + level) + scalar - 1 + prefix_word * power(Dimension, level) + word;
					state[prefix + offset(level) + word] = signature[index];
					adjoint[prefix + offset(level) + word] = seeds[index];
				}
			}
		}
	}

	__device__ __forceinline__ void pullback(const T (&increment)[Dimension], T (&adjoint)[size], T (&gradient)[Dimension]) {
		T history[prefix];
		#pragma unroll
		for (int level = 1; level <= prefix; ++level) {
			T value = T(0);
			#pragma unroll
			for (int i = 0; i < level; ++i) {
				history[i] = state[i] + value;
				value = select(increment, letters[i]) * (T(1) / T(level - i)) * history[i];
			}
			T derivative = adjoint[level];
			#pragma unroll
			for (int i = level - 1; i >= 0; --i) {
				derivative *= T(1) / T(level - i);
				add(gradient, letters[i], derivative * history[i]);
				derivative *= select(increment, letters[i]);
				adjoint[i] += derivative;
			}
		}
		#pragma unroll
		for (int tail = 1; tail <= Tail; ++tail) {
			T work[work_size];
			T work_adjoint[work_size];
			#pragma unroll
			for (int i = 0; i < work_size; ++i) work_adjoint[i] = T(0);
			T common = T(0);
			#pragma unroll
			for (int i = 0; i < prefix; ++i) {
				history[i] = state[i] + common;
				common = select(increment, letters[i]) * (T(1) / T(prefix + tail - i)) * history[i];
			}
			work[0] = state[prefix] + common;
			#pragma unroll
			for (int level = 1; level < Tail; ++level) {
				#pragma unroll
				for (int word = 0; word < tail_width; ++word)
					if (level < tail && word < power(Dimension, level))
						work[offset(level) + word] = state[prefix + offset(level) + word]
							+ work[offset(level - 1) + word / Dimension] * increment[word % Dimension] * (T(1) / T(tail - level + 1));
			}
			#pragma unroll
			for (int word = 0; word < tail_width; ++word) {
				if (word < power(Dimension, tail)) {
					const T derivative = adjoint[prefix + offset(tail) + word];
					gradient[word % Dimension] += derivative * work[offset(tail - 1) + word / Dimension];
					work_adjoint[offset(tail - 1) + word / Dimension] += derivative * increment[word % Dimension];
				}
			}
			#pragma unroll
			for (int level = Tail - 1; level > 0; --level) {
				#pragma unroll
				for (int word = 0; word < tail_width; ++word) {
					if (level < tail && word < power(Dimension, level)) {
						const T derivative = work_adjoint[offset(level) + word] * (T(1) / T(tail - level + 1));
						adjoint[prefix + offset(level) + word] += work_adjoint[offset(level) + word];
						gradient[word % Dimension] += derivative * work[offset(level - 1) + word / Dimension];
						work_adjoint[offset(level - 1) + word / Dimension] += derivative * increment[word % Dimension];
					}
				}
			}
			T derivative = work_adjoint[0];
			adjoint[prefix] += derivative;
			#pragma unroll
			for (int i = prefix - 1; i >= 0; --i) {
				derivative *= T(1) / T(prefix + tail - i);
				add(gradient, letters[i], derivative * history[i]);
				derivative *= select(increment, letters[i]);
				adjoint[i] += derivative;
			}
		}
	}
};

template<typename T, int Dimension, int Degree, int Tail>
__global__ void forward_kernel(const T* path, T* output, int steps, int chunk_size,
	uint64_t sig_stride, SignaturePath<T> input, bool scalar, uint64_t batch_offset, uint64_t count) {
	using State = Tree<T, Dimension, Degree, Tail>;
	const uint64_t sample = cuda_batch_index();
	if (sample >= count) return;
	const uint64_t word = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	const bool active = word < power(Dimension, State::prefix);
	const T* sample_path = path + (batch_offset + sample) * input.stride;
	extern __shared__ char storage[];
	T* increments = reinterpret_cast<T*>(storage);
	State state(word);
	state.clear();
	for (int start = 0; start < steps; start += chunk_size) {
		const int count_steps = min(chunk_size, steps - start);
		__syncthreads();
		for (int i = threadIdx.x; i < count_steps * Dimension; i += blockDim.x)
			increments[i] = input.template increment<false>(nullptr, sample_path, start + i / Dimension, i % Dimension);
		__syncthreads();
		if (active) {
			for (int step = 0; step < count_steps; ++step) {
				T increment[Dimension];
				#pragma unroll
				for (int letter = 0; letter < Dimension; ++letter)
					increment[letter] = increments[step * Dimension + letter];
				state.update(increment);
			}
		}
	}
	if (active) state.store(output + (batch_offset + sample) * sig_stride, word, scalar);
}

template<typename T, int Dimension, int Degree, int Tail>
void forward(const T* path, T* output, uint64_t batch, int steps, uint64_t stride,
	SignaturePath<T> input, bool scalar) {
	constexpr unsigned int block = 128;
	constexpr uint64_t words = power(Dimension, Degree - Tail);
	constexpr uint64_t grid = (words + block - 1) / block;
	const int chunk_size = std::min(steps, 128);
	for (uint64_t offset = 0; offset < batch;) {
		const auto chunk = make_cuda_batch_grid_chunk(grid, batch, offset);
		forward_kernel<T, Dimension, Degree, Tail><<<chunk.grid, block, chunk_size * Dimension * sizeof(T)>>>(
			path, output, steps, chunk_size, stride, input, scalar, offset, chunk.size);
		offset += chunk.size;
	}
	check_cuda_kernel_launch();
}

template<typename T>
bool try_forward(const T* path, T* output, uint64_t batch, int dimension, int degree,
	int steps, uint64_t stride, SignaturePath<T> input, bool scalar) {
	if (dimension < 2 || dimension > 32 || degree < (dimension > 4 ? 4 : 6)
		|| degree > (dimension == 6 ? 9 : (dimension > 4 ? 8 : 16))) return false;
	const int tail = dimension == 2 ? 4 : (dimension == 3 ? 3 : (dimension == 4 ? 2 : 1));
	const uint64_t words = power(dimension, degree - tail);
	if (batch < (1024 + words - 1) / words) return false;
	#define LAUNCH_TREE(K, D, R) case K: forward<T, D, K, R>(path, output, batch, steps, stride, input, scalar); return true;
	#define TREE_DEGREES(D, R) \
		LAUNCH_TREE(6, D, R) LAUNCH_TREE(7, D, R) LAUNCH_TREE(8, D, R) \
		LAUNCH_TREE(9, D, R) LAUNCH_TREE(10, D, R) LAUNCH_TREE(11, D, R) \
		LAUNCH_TREE(12, D, R) LAUNCH_TREE(13, D, R) LAUNCH_TREE(14, D, R) \
		LAUNCH_TREE(15, D, R) LAUNCH_TREE(16, D, R)
	if (dimension == 2) { switch (degree) { TREE_DEGREES(2, 4) } }
	if (dimension == 3) { switch (degree) { TREE_DEGREES(3, 3) } }
	if (dimension == 4) { switch (degree) { TREE_DEGREES(4, 2) } }
	if (dimension == 6) {
		switch (degree) { LAUNCH_TREE(7, 6, 2) LAUNCH_TREE(8, 6, 2) LAUNCH_TREE(9, 6, 2) }
	}
	#define TREE_SHORT_DEGREES(D) LAUNCH_TREE(4, D, 1) LAUNCH_TREE(5, D, 1) \
		LAUNCH_TREE(6, D, 1) LAUNCH_TREE(7, D, 1) LAUNCH_TREE(8, D, 1)
	if constexpr (sizeof(T) == sizeof(double)) {
		if (dimension == 6) { switch (degree) { TREE_SHORT_DEGREES(6) } }
		if (dimension == 8) { switch (degree) { TREE_SHORT_DEGREES(8) } }
		if (dimension == 10) { switch (degree) { TREE_SHORT_DEGREES(10) } }
		if (dimension == 12) { switch (degree) { TREE_SHORT_DEGREES(12) } }
		if (dimension == 16) { switch (degree) { TREE_SHORT_DEGREES(16) } }
		if (dimension == 24) { switch (degree) { TREE_SHORT_DEGREES(24) } }
		if (dimension == 32) { switch (degree) { TREE_SHORT_DEGREES(32) } }
	}
	#undef TREE_SHORT_DEGREES
	#undef TREE_DEGREES
	#undef LAUNCH_TREE
	return false;
}

template<typename T, int Dimension, int Degree, int Tail>
__global__ __launch_bounds__(128) void backward_kernel(const T* path, const T* signature, const T* seeds,
	T* increment_gradient, int steps, int chunk_size, uint64_t stride, bool scalar, uint64_t batch_offset, uint64_t count) {
	using State = Tree<T, Dimension, Degree, Tail>;
	const uint64_t sample = cuda_batch_index();
	if (sample >= count) return;
	const uint64_t word = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	const bool active = word < power(Dimension, State::prefix);
	const T* sample_path = path + (batch_offset + sample) * uint64_t(steps + 1) * Dimension;
	T* sample_gradient = increment_gradient + sample * uint64_t(steps) * Dimension;
	extern __shared__ char storage[];
	T* increments = reinterpret_cast<T*>(storage);
	T* reductions = increments + chunk_size * Dimension;
	constexpr int warps = 4;
	const int lane = threadIdx.x % 32;
	const int warp = threadIdx.x / 32;
	State state(word);
	T adjoint[State::size];
	if (active) state.load(signature + (batch_offset + sample) * stride,
		seeds + (batch_offset + sample) * stride, adjoint, word, scalar);
	for (int end = steps; end > 0; end -= chunk_size) {
		const int start = max(0, end - chunk_size);
		const int count_steps = end - start;
		__syncthreads();
		for (int i = threadIdx.x; i < count_steps * Dimension; i += blockDim.x)
			increments[i] = sample_path[(start + 1) * Dimension + i] - sample_path[start * Dimension + i];
		__syncthreads();
		for (int step = count_steps - 1; step >= 0; --step) {
			T gradient[Dimension] = {};
			if (active) {
				T increment[Dimension];
				#pragma unroll
				for (int letter = 0; letter < Dimension; ++letter) increment[letter] = -increments[step * Dimension + letter];
				state.update(increment);
				#pragma unroll
				for (int letter = 0; letter < Dimension; ++letter) increment[letter] = -increment[letter];
				state.pullback(increment, adjoint, gradient);
			}
			#pragma unroll
			for (int letter = 0; letter < Dimension; ++letter) {
				T value = gradient[letter];
				#pragma unroll
				for (int shift = 16; shift > 0; shift /= 2) value += __shfl_down_sync(0xffffffff, value, shift);
				if (lane == 0) reductions[(step * Dimension + letter) * warps + warp] = value;
			}
		}
		__syncthreads();
		for (int i = threadIdx.x; i < count_steps * Dimension; i += blockDim.x) {
			T value = T(0);
			#pragma unroll
			for (int warp_index = 0; warp_index < warps; ++warp_index) value += reductions[i * warps + warp_index];
			myAtomicAdd(sample_gradient + uint64_t(start) * Dimension + i, value);
		}
	}
}

template<typename T, int Dimension, int Degree, int Tail>
void backward(const T* path, T* output, const T* signature, const T* seeds, uint64_t batch,
	int steps, uint64_t stride, bool scalar, cudaStream_t stream) {
	constexpr unsigned int block = 128;
	constexpr uint64_t words = power(Dimension, Degree - Tail);
	constexpr uint64_t grid = (words + block - 1) / block;
	const int chunk_size = std::min(steps, 32);
	const size_t shared = chunk_size * Dimension * 5 * sizeof(T);
	const size_t row = checked_cuda_size_mul(steps, Dimension, "CUDA Horner backprop");
	CudaBatchWorkspace<T> workspace(batch, row, "CUDA Horner backprop");
	for (uint64_t offset = 0; offset < batch;) {
		const auto chunk = make_cuda_batch_grid_chunk(grid, batch, offset, workspace.capacity());
		CUDA_CHECK(cudaMemsetAsync(workspace.get(), 0, chunk.size * row * sizeof(T), stream));
		backward_kernel<T, Dimension, Degree, Tail><<<chunk.grid, block, shared, stream>>>(
			path, signature, seeds, workspace.get(), steps, chunk_size, stride, scalar, offset, chunk.size);
		const auto convert = make_cuda_batch_grid_chunk((uint64_t(steps + 1) * Dimension + 255) / 256,
			batch, offset, chunk.size);
		increment_to_path_grad_batch_ker<T><<<convert.grid, 256, 0, stream>>>(
			workspace.get(), output, steps + 1, Dimension, offset, chunk.size);
		offset += chunk.size;
	}
	check_cuda_kernel_launch();
}

template<typename T>
bool try_backward(const T* path, T* output, const T* signature, const T* seeds, uint64_t batch,
	int dimension, int degree, int steps, uint64_t stride, bool scalar, cudaStream_t stream) {
	if (dimension < 2 || dimension > 32 || degree < (dimension > 4 ? 4 : 6)
		|| degree > (dimension == 6 ? 9 : (dimension > 4 ? 8 : 16))) return false;
	const int tail = dimension >= 4 ? 1 : 2;
	const uint64_t words = power(dimension, degree - tail);
	if (batch < (256 + words - 1) / words) return false;
	#define LAUNCH_TREE(K, D, R) case K: backward<T, D, K, R>(path, output, signature, seeds, batch, steps, stride, scalar, stream); return true;
	#define TREE_DEGREES(D, R) \
		LAUNCH_TREE(6, D, R) LAUNCH_TREE(7, D, R) LAUNCH_TREE(8, D, R) \
		LAUNCH_TREE(9, D, R) LAUNCH_TREE(10, D, R) LAUNCH_TREE(11, D, R) \
		LAUNCH_TREE(12, D, R) LAUNCH_TREE(13, D, R) LAUNCH_TREE(14, D, R) \
		LAUNCH_TREE(15, D, R) LAUNCH_TREE(16, D, R)
	if (dimension == 2) { switch (degree) { TREE_DEGREES(2, 2) } }
	if (dimension == 3) { switch (degree) { TREE_DEGREES(3, 2) } }
	if (dimension == 4) { switch (degree) { TREE_DEGREES(4, 1) } }
	if (dimension == 6) {
		switch (degree) { LAUNCH_TREE(7, 6, 2) LAUNCH_TREE(8, 6, 2) LAUNCH_TREE(9, 6, 2) }
	}
	#define TREE_SHORT_DEGREES(D) LAUNCH_TREE(4, D, 1) LAUNCH_TREE(5, D, 1) \
		LAUNCH_TREE(6, D, 1) LAUNCH_TREE(7, D, 1) LAUNCH_TREE(8, D, 1)
	if (dimension == 6) { switch (degree) { TREE_SHORT_DEGREES(6) } }
	if (dimension == 8) { switch (degree) { TREE_SHORT_DEGREES(8) } }
	if (dimension == 10) { switch (degree) { TREE_SHORT_DEGREES(10) } }
	if (dimension == 12) { switch (degree) { TREE_SHORT_DEGREES(12) } }
	if (dimension == 16) { switch (degree) { TREE_SHORT_DEGREES(16) } }
	if (dimension == 24) { switch (degree) { TREE_SHORT_DEGREES(24) } }
	if (dimension == 32) { switch (degree) { TREE_SHORT_DEGREES(32) } }
	#undef TREE_SHORT_DEGREES
	#undef TREE_DEGREES
	#undef LAUNCH_TREE
	return false;
}

template<typename T>
__global__ void endpoint_backward_kernel(T* output, const T* signature, const T* adjoint,
	uint64_t dimension, uint64_t length, uint64_t degree, uint64_t stride, uint64_t total, bool scalar) {
	for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
		index += uint64_t(gridDim.x) * blockDim.x) {
		const uint64_t letter = index % dimension;
		const uint64_t vertex = index / dimension % length;
		const uint64_t sample = index / (dimension * length);
		T value = T(0);
		if (vertex == 0 || vertex + 1 == length) {
			if (degree == 1) value = adjoint[sample * stride + scalar + letter];
			else {
				for (uint64_t k = 1; k <= degree; ++k) {
					const T lower = k == 1 ? T(1) : signature[sample * stride + scalar + k - 2];
					value += adjoint[sample * stride + scalar + k - 1] * lower;
				}
			}
			if (vertex == 0) value = -value;
		}
		output[index] = value;
	}
}

} // namespace cuda_horner
