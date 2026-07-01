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

// General finite state-space Volterra kernels: Lambda is an arbitrary real
// state matrix (diagonal, dense, oscillatory, or a defective Jordan block).
// The FSSK recursion is identical to the diagonal fast path except that the
// per-step propagator E = exp(-Lambda dt) is applied as a full R x R matrix
// product on the state (Z_new = Z . E + B (x) y) rather than a per-rate
// scaling. Everything is real: keeping Lambda real makes the contour
// coefficients (psi, phi) real, so a complex state is never needed.
//
// The coefficients E, psi, phi and the readout weights are precomputed by the
// Python layer (pysiglib/_general_fssk.py) via the matrix resolvent and a
// matrix exponential, and passed in here; this file only builds the (integer)
// multi-index layout and runs the recursion. This is the non-tiled advanced
// path; the diagonal common case stays on the SIMD batch-tiled path in
// cp_volterra_signature.cpp.

#include "cppch.h"
#include "cpsig.h"
#include "cp_utils.h"
#include "cp_volterra_common.h"
#include "macros.h"
#include "multithreading.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

using namespace volterra_detail;

template<std::floating_point T>
struct VolterraKernelCacheGeneral {
	std::vector<T> A;                 // (num_components, target_dimension, dimension)
	std::vector<T> E;                 // (state_dimension, state_dimension) propagator
	std::vector<T> readout_weights;   // (num_components, state_dimension)
	std::vector<T> psi;               // (mi_len_f, state_dimension)
	std::vector<T> phi;               // (num_components, mi_len_g, state_dimension, state_dimension)
	std::vector<uint64_t> mi_level_index;
	std::vector<uint64_t> minus_index;
	std::vector<uint64_t> level_index;
	std::vector<uint64_t> mi_off;
	std::vector<uint64_t> mi_ts;
	uint64_t dimension = 0;
	uint64_t num_components = 0;
	uint64_t target_dimension = 0;
	uint64_t state_dimension = 0;
	uint64_t degree = 0;
	uint64_t sig_len = 0;
	uint64_t state_len = 0;
	uint64_t f_len = 0;
	uint64_t mi_total = 0;
	uint64_t mi_len_g = 0;
};

template<std::floating_point T>
struct VolterraWorkspaceGeneral {
	explicit VolterraWorkspaceGeneral(const VolterraKernelCacheGeneral<T>& p)
		: state(p.num_components * p.state_dimension * p.state_len),
		next_state(p.num_components * p.state_dimension * p.state_len),
		F(p.state_dimension * p.f_len),
		G(p.num_components * p.state_dimension * p.state_dimension * p.f_len),
		B(p.state_dimension * p.f_len),
		y(p.num_components * p.target_dimension),
		dx(p.dimension),
		shuffle(p.mi_total)
	{}
	std::vector<T> state;
	std::vector<T> next_state;
	std::vector<T> F;
	std::vector<T> G;
	std::vector<T> B;
	std::vector<T> y;
	std::vector<T> dx;
	std::vector<T> shuffle;
};

template<std::floating_point T>
VolterraKernelCacheGeneral<T> make_prepared_volterra_sig_general(
	const T* E,
	const T* psi,
	const T* phi,
	const T* readout_weights,
	const T* A,
	uint64_t dimension,
	uint64_t num_components,
	uint64_t target_dimension,
	uint64_t state_dimension,
	uint64_t degree
) {
	if (dimension == 0 || num_components == 0 || target_dimension == 0 || state_dimension == 0)
		throw std::invalid_argument("prepare_volterra_sig received a zero dimension");

	VolterraKernelCacheGeneral<T> prepared;
	prepared.dimension = dimension;
	prepared.num_components = num_components;
	prepared.target_dimension = target_dimension;
	prepared.state_dimension = state_dimension;
	prepared.degree = degree;

	prepared.A.assign(A, A + num_components * target_dimension * dimension);
	prepared.E.assign(E, E + state_dimension * state_dimension);
	prepared.readout_weights.assign(readout_weights, readout_weights + num_components * state_dimension);

	prepared.sig_len = ::sig_length(target_dimension, degree);
	if (prepared.sig_len == 0)
		throw std::overflow_error("prepare_volterra_sig length overflow");
	prepared.state_len = prepared.sig_len - 1;
	prepared.f_len = degree == 0 ? 1 : ::sig_length(target_dimension, degree - 1);
	if (prepared.f_len == 0)
		throw std::overflow_error("prepare_volterra_sig workspace length overflow");

	const uint64_t level_count = degree + 2;
	prepared.level_index.resize(level_count);
	populate_level_index(prepared.level_index.data(), target_dimension, level_count);

	if (degree == 0) {
		prepared.mi_total = 0;
		return prepared;
	}

	// Integer multi-index layout (shuffle structure), shared with the diagonal path.
	std::vector<uint64_t> multiindices;
	populate_multiindex_layout(
		num_components, degree - 1, prepared.mi_level_index, multiindices, prepared.minus_index);
	const uint64_t mi_len_f = prepared.mi_level_index[degree];
	prepared.mi_len_g = degree >= 2 ? prepared.mi_level_index[degree - 1] : 0;

	prepared.psi.assign(psi, psi + mi_len_f * state_dimension);
	prepared.phi.assign(
		phi, phi + num_components * prepared.mi_len_g * state_dimension * state_dimension);

	const uint64_t n_levels = degree;
	prepared.mi_off.assign(n_levels, 0);
	prepared.mi_ts.assign(n_levels, 0);
	prepared.mi_total = populate_multiindex_tensor_layout(
		prepared.mi_off.data(), prepared.mi_ts.data(),
		prepared.mi_level_index.data(), target_dimension, n_levels);
	return prepared;
}

template<std::floating_point T>
void compute_projected_increment(
	const T* path,
	const VolterraKernelCacheGeneral<T>& prepared,
	T* y,
	T* dx,
	uint64_t step
) {
	const uint64_t dimension = prepared.dimension;
	const uint64_t num_components = prepared.num_components;
	const uint64_t target_dimension = prepared.target_dimension;
	const T* x0 = path + step * dimension;
	const T* x1 = x0 + dimension;
	for (uint64_t j = 0; j < dimension; ++j)
		dx[j] = x1[j] - x0[j];
	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t i = 0; i < target_dimension; ++i) {
			T acc = static_cast<T>(0);
			const T* A_row = prepared.A.data() + (p * target_dimension + i) * dimension;
			for (uint64_t j = 0; j < dimension; ++j)
				acc += A_row[j] * dx[j];
			y[p * target_dimension + i] = acc;
		}
	}
}

template<std::floating_point T>
void eval_fg(
	const VolterraKernelCacheGeneral<T>& prepared,
	const T* y,
	T* F,
	T* G,
	T* shuffle_tensors
) {
	const uint64_t num_components = prepared.num_components;
	const uint64_t target_dimension = prepared.target_dimension;
	const uint64_t state_dimension = prepared.state_dimension;
	const uint64_t degree = prepared.degree;
	const uint64_t* level_index = prepared.level_index.data();
	const uint64_t* mi_off = prepared.mi_off.data();
	const uint64_t* mi_ts = prepared.mi_ts.data();
	const uint64_t f_len = prepared.f_len;

	std::fill(F, F + state_dimension * f_len, static_cast<T>(0));

	build_shuffle_tensors<T, 1>(
		y, shuffle_tensors, num_components, target_dimension, degree,
		prepared.mi_level_index.data(), prepared.minus_index.data(), mi_off, mi_ts);

	for (uint64_t word_len = 0; word_len <= degree - 1; ++word_len) {
		const uint64_t mi_start = prepared.mi_level_index[word_len];
		const uint64_t mi_end = prepared.mi_level_index[word_len + 1];
		const uint64_t out_start = level_index[word_len];
		const uint64_t ts = mi_ts[word_len];
		const T* level_tensors = shuffle_tensors + mi_off[word_len];
		for (uint64_t mi_idx = mi_start; mi_idx < mi_end; ++mi_idx) {
			const T* tensor = level_tensors + (mi_idx - mi_start) * ts;
			for (uint64_t r = 0; r < state_dimension; ++r) {
				T* RESTRICT dst = F + r * f_len + out_start;
				const T c = prepared.psi[mi_idx * state_dimension + r];
				for (uint64_t i = 0; i < ts; ++i)
					dst[i] += c * tensor[i];
			}
		}
	}

	if (degree <= 1)
		return;

	std::fill(G, G + num_components * state_dimension * state_dimension * f_len, static_cast<T>(0));

	const uint64_t mi_len_g = prepared.mi_len_g;
	for (uint64_t word_len = 0; word_len <= degree - 2; ++word_len) {
		const uint64_t mi_start = prepared.mi_level_index[word_len];
		const uint64_t mi_end = prepared.mi_level_index[word_len + 1];
		const uint64_t out_start = level_index[word_len];
		const uint64_t ts = mi_ts[word_len];
		const T* level_tensors = shuffle_tensors + mi_off[word_len];
		for (uint64_t mi_idx = mi_start; mi_idx < mi_end; ++mi_idx) {
			const T* tensor = level_tensors + (mi_idx - mi_start) * ts;
			for (uint64_t p = 0; p < num_components; ++p) {
				const T* phi_ptr = prepared.phi.data()
					+ ((p * mi_len_g + mi_idx) * state_dimension * state_dimension);
				for (uint64_t r0 = 0; r0 < state_dimension; ++r0) {
					for (uint64_t r1 = 0; r1 < state_dimension; ++r1) {
						T* RESTRICT dst = G + (((p * state_dimension + r0) * state_dimension + r1) * f_len) + out_start;
						const T c = phi_ptr[r0 * state_dimension + r1];
						for (uint64_t i = 0; i < ts; ++i)
							dst[i] += c * tensor[i];
					}
				}
			}
		}
	}
}

template<std::floating_point T>
void update_state(
	const VolterraKernelCacheGeneral<T>& prepared,
	const T* y,
	T* state,
	T* next_state,
	T* F,
	T* G,
	T* B,
	T* shuffle_tensors
) {
	const uint64_t num_components = prepared.num_components;
	const uint64_t target_dimension = prepared.target_dimension;
	const uint64_t state_dimension = prepared.state_dimension;
	const uint64_t degree = prepared.degree;
	const uint64_t* level_index = prepared.level_index.data();
	const uint64_t f_len = prepared.f_len;
	const uint64_t state_len = prepared.state_len;
	const T* Emat = prepared.E.data();

	eval_fg<T>(prepared, y, F, G, shuffle_tensors);

	// Scaling: next_state[(p,r)] = sum_{r'} state[(p,r')] * E[r', r].
	std::fill(next_state, next_state + num_components * state_dimension * state_len, static_cast<T>(0));
	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t r1 = 0; r1 < state_dimension; ++r1) {
			const T* RESTRICT src = state + (p * state_dimension + r1) * state_len;
			for (uint64_t r = 0; r < state_dimension; ++r) {
				const T e = Emat[r1 * state_dimension + r];
				if (e == static_cast<T>(0))
					continue;
				T* RESTRICT dst = next_state + (p * state_dimension + r) * state_len;
				for (uint64_t i = 0; i < state_len; ++i)
					dst[i] += e * src[i];
			}
		}
	}

	std::memcpy(B, F, state_dimension * f_len * sizeof(T));

	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t left_level = 1; left_level <= degree - 1; ++left_level) {
			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
			const uint64_t left_state_start = level_index[left_level] - 1;
			for (uint64_t right_level = 0; right_level <= degree - 1 - left_level; ++right_level) {
				const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t out_start = level_index[left_level + right_level];
				for (uint64_t r0 = 0; r0 < state_dimension; ++r0) {
					const T* z = state + (p * state_dimension + r0) * state_len + left_state_start;
					for (uint64_t r1 = 0; r1 < state_dimension; ++r1) {
						const T* g = G + (((p * state_dimension + r0) * state_dimension + r1) * f_len) + right_start;
						T* dst = B + r1 * f_len + out_start;
						for (uint64_t i = 0; i < left_size; ++i) {
							const T z_val = z[i];
							T* RESTRICT dst_row = dst + i * right_size;
							for (uint64_t j = 0; j < right_size; ++j)
								dst_row[j] += z_val * g[j];
						}
					}
				}
			}
		}
	}

	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t b_level = 0; b_level <= degree - 1; ++b_level) {
			const uint64_t b_size = level_index[b_level + 1] - level_index[b_level];
			const uint64_t b_start = level_index[b_level];
			const uint64_t out_start = level_index[b_level + 1] - 1;
			for (uint64_t r = 0; r < state_dimension; ++r) {
				const T* b_ptr = B + r * f_len + b_start;
				T* dst = next_state + (p * state_dimension + r) * state_len + out_start;
				const T* y_ptr = y + p * target_dimension;
				for (uint64_t i = 0; i < b_size; ++i) {
					const T b_val = b_ptr[i];
					for (uint64_t d = 0; d < target_dimension; ++d)
						dst[i * target_dimension + d] += b_val * y_ptr[d];
				}
			}
		}
	}
}

template<std::floating_point T>
void readout_state(
	const T* state,
	const VolterraKernelCacheGeneral<T>& prepared,
	T* out,
	bool scalar_term
) {
	const uint64_t num_components = prepared.num_components;
	const uint64_t state_dimension = prepared.state_dimension;
	const uint64_t degree = prepared.degree;
	const uint64_t* level_index = prepared.level_index.data();
	const uint64_t sig_len = prepared.sig_len;
	const uint64_t state_len = prepared.state_len;
	const uint64_t out_len = scalar_term ? sig_len : sig_len - 1;

	std::fill(out, out + out_len, static_cast<T>(0));
	if (scalar_term)
		out[0] = static_cast<T>(1);

	for (uint64_t level = 1; level <= degree; ++level) {
		const uint64_t level_size = level_index[level + 1] - level_index[level];
		const uint64_t state_start = level_index[level] - 1;
		const uint64_t out_start = scalar_term ? level_index[level] : level_index[level] - 1;
		T* dst = out + out_start;
		for (uint64_t p = 0; p < num_components; ++p) {
			for (uint64_t r = 0; r < state_dimension; ++r) {
				const T weight = prepared.readout_weights[p * state_dimension + r];
				const T* RESTRICT src = state + (p * state_dimension + r) * state_len + state_start;
				for (uint64_t i = 0; i < level_size; ++i)
					dst[i] += weight * src[i];
			}
		}
	}
}

template<std::floating_point T>
void volterra_sig_single_general(
	const T* path,
	const VolterraKernelCacheGeneral<T>& prepared,
	T* out,
	VolterraWorkspaceGeneral<T>& ws,
	uint64_t length,
	bool scalar_term
) {
	T* cur = ws.state.data();
	T* nxt = ws.next_state.data();
	std::fill(ws.state.begin(), ws.state.end(), static_cast<T>(0));
	for (uint64_t step = 0; step + 1 < length; ++step) {
		compute_projected_increment<T>(path, prepared, ws.y.data(), ws.dx.data(), step);
		update_state<T>(prepared, ws.y.data(), cur, nxt, ws.F.data(), ws.G.data(), ws.B.data(), ws.shuffle.data());
		std::swap(cur, nxt);
	}
	readout_state<T>(cur, prepared, out, scalar_term);
}

HandleStore<VolterraKernelCacheGeneral> prepared_general_store;

template<std::floating_point T>
void volterra_sig_general_impl(
	const T* path,
	const VolterraKernelCacheGeneral<T>& prepared,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool scalar_term,
	int n_jobs
) {
	if (dimension != prepared.dimension)
		throw std::invalid_argument("volterra_sig path dimension does not match prepared kernel dimension");

	const uint64_t sig_len = prepared.sig_len;
	const uint64_t out_stride = scalar_term ? sig_len : sig_len - 1;

	if (length <= 1) {
		T* const out_end = out + out_stride * batch_size;
		std::fill(out, out_end, static_cast<T>(0));
		if (scalar_term) {
			for (T* out_ptr = out; out_ptr < out_end; out_ptr += out_stride)
				out_ptr[0] = static_cast<T>(1);
		}
		return;
	}
	if (prepared.degree == 0) {
		if (scalar_term)
			std::fill(out, out + batch_size, static_cast<T>(1));
		return;
	}

	auto run_range = [&](uint64_t start, uint64_t end) {
		VolterraWorkspaceGeneral<T> ws(prepared);
		for (uint64_t i = start; i < end; ++i) {
			volterra_sig_single_general<T>(
				path + i * length * prepared.dimension,
				prepared, out + i * out_stride, ws, length, scalar_term);
		}
	};

	if (batch_size == 0)
		return;
	if (n_jobs == 1 || batch_size == 1)
		run_range(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, run_range);
}

} // namespace

void clear_prepared_volterra_sig_general_cache() {
	prepared_general_store.clear();
}

extern "C" {

	CPSIG_API int prepare_volterra_sig_general_f(
		const float* E, const float* psi, const float* phi, const float* readout_weights,
		const float* A, uint64_t dimension, uint64_t num_components, uint64_t target_dimension,
		uint64_t state_dimension, uint64_t degree, uint64_t* handle
	) noexcept {
		SAFE_CALL({
			*handle = prepared_general_store.store<float>(make_prepared_volterra_sig_general<float>(
				E, psi, phi, readout_weights, A, dimension, num_components,
				target_dimension, state_dimension, degree));
		});
	}

	CPSIG_API int prepare_volterra_sig_general_d(
		const double* E, const double* psi, const double* phi, const double* readout_weights,
		const double* A, uint64_t dimension, uint64_t num_components, uint64_t target_dimension,
		uint64_t state_dimension, uint64_t degree, uint64_t* handle
	) noexcept {
		SAFE_CALL({
			*handle = prepared_general_store.store<double>(make_prepared_volterra_sig_general<double>(
				E, psi, phi, readout_weights, A, dimension, num_components,
				target_dimension, state_dimension, degree));
		});
	}

	CPSIG_API int free_volterra_sig_general_f(uint64_t handle) noexcept {
		SAFE_CALL(prepared_general_store.free<float>(handle));
	}

	CPSIG_API int free_volterra_sig_general_d(uint64_t handle) noexcept {
		SAFE_CALL(prepared_general_store.free<double>(handle));
	}

	CPSIG_API int volterra_sig_general_f(
		const float* path, float* out, uint64_t handle, uint64_t batch_size,
		uint64_t dimension, uint64_t length, bool scalar_term, int n_jobs
	) noexcept {
		SAFE_CALL(volterra_sig_general_impl<float>(
			path, prepared_general_store.get<float>(handle), out, batch_size, dimension, length, scalar_term, n_jobs));
	}

	CPSIG_API int volterra_sig_general_d(
		const double* path, double* out, uint64_t handle, uint64_t batch_size,
		uint64_t dimension, uint64_t length, bool scalar_term, int n_jobs
	) noexcept {
		SAFE_CALL(volterra_sig_general_impl<double>(
			path, prepared_general_store.get<double>(handle), out, batch_size, dimension, length, scalar_term, n_jobs));
	}
}
