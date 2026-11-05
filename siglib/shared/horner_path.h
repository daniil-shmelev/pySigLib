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

// Included inside the backend's target namespace with HORNER_INLINE defined.
namespace horner_path {

inline constexpr uint64_t checkpoint_steps = 32;

template<typename T>
struct Path {

	const T* data;
	uint64_t dimension;
	uint64_t length;
	bool time_aug;
	bool lead_lag;
	T time_scale;

	HORNER_INLINE uint64_t steps() const {
		return (length > 0 ? length - 1 : 0) * (lead_lag ? 2 : 1);
	}

	HORNER_INLINE T increment(uint64_t step, int letter) const {
		if (time_aug && letter == (lead_lag ? 2 * dimension : dimension)) {
#ifdef __CUDA_ARCH__
			if constexpr (sizeof(T) == sizeof(float))
				return __fmul_rn(T(step + 1), time_scale) - __fmul_rn(T(step), time_scale);
			else
				return __dmul_rn(T(step + 1), time_scale) - __dmul_rn(T(step), time_scale);
#else
			return T(step + 1) * time_scale - T(step) * time_scale;
#endif
		}
		if (lead_lag) {
			if (letter < dimension) {
				if (step % 2 == 0) return T(0);
			}
			else {
				if (step % 2 != 0) return T(0);
				letter -= static_cast<int>(dimension);
			}
			step /= 2;
		}
		const uint64_t position = step * dimension + letter;
		return data[position + dimension] - data[position];
	}

	HORNER_INLINE void add_gradient(T* out, uint64_t step, int letter, T value) const {
		if (time_aug && letter == (lead_lag ? 2 * dimension : dimension)) return;
		if (lead_lag) {
			if (letter < dimension) {
				if (step % 2 == 0) return;
			}
			else {
				if (step % 2 != 0) return;
				letter -= static_cast<int>(dimension);
			}
			step /= 2;
		}
		const uint64_t position = step * dimension + letter;
		out[position] -= value;
		out[position + dimension] += value;
	}
};

template<typename Kernel, typename T>
HORNER_INLINE void forward_range(const Path<T>& path, T* out, uint64_t begin, uint64_t end) {
	T state[Kernel::state_size] = {};
	T increment[Kernel::dimension];
	for (uint64_t step = begin; step < end; ++step) {
		for (int i = 0; i < Kernel::dimension; ++i)
			increment[i] = path.increment(step, i);
		Kernel::update(state, increment);
	}
	Kernel::expand(state, out);
}

template<typename Kernel, typename T>
HORNER_INLINE void forward(const Path<T>& path, T* out) {
	forward_range<Kernel>(path, out, 0, path.steps());
}

template<typename Kernel, typename T, typename Workspace, typename StoreGradient>
HORNER_INLINE void backward_range(const Path<T>& path, const T* deriv, Workspace workspace,
	uint64_t first_step, uint64_t last_step, StoreGradient store_gradient) {
	constexpr int size = Kernel::state_size;
	const uint64_t steps = last_step - first_step;
	T state[size] = {};
	T gradient[size];
	T increment[Kernel::dimension];
	T increment_gradient[Kernel::dimension];
	auto replay = workspace + (steps / checkpoint_steps + 1) * size;
	for (uint64_t step = 0; step < steps; ++step) {
		if (step % checkpoint_steps == 0) {
			for (int i = 0; i < size; ++i)
				workspace[(step / checkpoint_steps) * size + i] = state[i];
		}
		for (int i = 0; i < Kernel::dimension; ++i)
			increment[i] = path.increment(first_step + step, i);
		Kernel::update(state, increment);
	}
	Kernel::expand_pullback(state, deriv, gradient);
	for (uint64_t end = steps; end > 0;) {
		const uint64_t begin = ((end - 1) / checkpoint_steps) * checkpoint_steps;
		for (int i = 0; i < size; ++i)
			state[i] = workspace[(begin / checkpoint_steps) * size + i];
		for (uint64_t step = begin; step < end; ++step) {
			for (int i = 0; i < size; ++i) replay[(step - begin) * size + i] = state[i];
			if (step + 1 < end) {
				for (int i = 0; i < Kernel::dimension; ++i)
					increment[i] = path.increment(first_step + step, i);
				Kernel::update(state, increment);
			}
		}
		for (uint64_t step = end; step-- > begin;) {
			for (int i = 0; i < Kernel::dimension; ++i)
				increment[i] = path.increment(first_step + step, i);
			for (int i = 0; i < size; ++i) state[i] = replay[(step - begin) * size + i];
			Kernel::pullback(state, increment, gradient, gradient, increment_gradient);
			for (int i = 0; i < Kernel::dimension; ++i)
				store_gradient(first_step + step, i, increment_gradient[i]);
		}
		end = begin;
	}
}

template<typename T>
struct PathGradient {
	const Path<T>& path;
	T* out;
	HORNER_INLINE void operator()(uint64_t step, int letter, T value) const {
		path.add_gradient(out, step, letter, value);
	}
};

template<typename Kernel, typename T>
HORNER_INLINE void backward(const Path<T>& path, const T* deriv, T* out, T* workspace) {
	for (uint64_t i = 0; i < path.length * path.dimension; ++i) out[i] = T(0);
	backward_range<Kernel>(path, deriv, workspace, 0, path.steps(), PathGradient<T>{path, out});
}

} // namespace horner_path
