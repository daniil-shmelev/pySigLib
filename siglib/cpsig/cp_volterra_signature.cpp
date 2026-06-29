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

#include "cppch.h"
#include "cpsig.h"
#include "cp_utils.h"
#include "cp_volterra_common.h"
#include "cp_volterra_signature.h"
#include "macros.h"
#include "multithreading.h"

#include <atomic>
#include <complex>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace {

using namespace volterra_detail;

template<std::floating_point T>
struct DiagFsskCoefficients {
	std::vector<T> E;
	std::vector<T> psi;
	std::vector<T> phi;
	std::vector<uint64_t> mi_level_index;
	std::vector<uint64_t> minus_index;
};

template<std::floating_point T>
struct VolterraKernelCache {
	std::vector<T> A;
	std::vector<T> readout_weights;
	DiagFsskCoefficients<T> coef;
	// Derived layout, depends only on target_dimension/degree, built once at prepare time.
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
};

// Internal batch-tiling width. Paths are processed in tiles of this many lanes
// so the per-step setup is amortized and the arithmetic vectorizes over the
// (always full and uniform) lane dimension. This is a tiling factor, not a
// kernel-parameter specialization: the same value is used for every input.
constexpr uint64_t VOLTERRA_LANES = 8;

// Per-thread scratch, sized for one full tile of VOLTERRA_LANES paths. The same
// workspace also serves the < VOLTERRA_LANES batch remainder, which runs at lane
// width 1 and so touches only the leading slice of each buffer.
template<std::floating_point T>
struct VolterraWorkspace {
	explicit VolterraWorkspace(const VolterraKernelCache<T>& prepared)
		: state_size(prepared.num_components * prepared.state_dimension * prepared.state_len * VOLTERRA_LANES),
		y_size(prepared.num_components * prepared.target_dimension * VOLTERRA_LANES),
		dx_size(prepared.dimension * VOLTERRA_LANES),
		f_size((prepared.num_components == 1 ? 0 : prepared.state_dimension * prepared.f_len) * VOLTERRA_LANES),
		g_size((prepared.num_components == 1
			? 0
			: prepared.num_components * prepared.state_dimension * prepared.state_dimension * prepared.f_len) * VOLTERRA_LANES),
		b_size(prepared.state_dimension * prepared.f_len * VOLTERRA_LANES),
		shuffle_size(prepared.mi_total * VOLTERRA_LANES),
		out_size(prepared.sig_len * VOLTERRA_LANES),
		buffer(2 * state_size + y_size + dx_size + f_size + g_size + b_size + shuffle_size + out_size)
	{
		T* ptr = buffer.data();
		state = ptr;
		ptr += state_size;
		next_state = ptr;
		ptr += state_size;
		y = ptr;
		ptr += y_size;
		dx = ptr;
		ptr += dx_size;
		F = ptr;
		ptr += f_size;
		G = ptr;
		ptr += g_size;
		B = ptr;
		ptr += b_size;
		shuffle_tensors = ptr;
		ptr += shuffle_size;
		tile_out = ptr;
	}

	uint64_t state_size;
	uint64_t y_size;
	uint64_t dx_size;
	uint64_t f_size;
	uint64_t g_size;
	uint64_t b_size;
	uint64_t shuffle_size;
	uint64_t out_size;
	std::vector<T> buffer;
	T* state;
	T* next_state;
	T* y;
	T* dx;
	T* F;
	T* G;
	T* B;
	T* shuffle_tensors;
	T* tile_out;
};

template<std::floating_point T>
DiagFsskCoefficients<T> build_diag_fssk_coefficients(
	const T* lambda_diag,
	const T* b,
	uint64_t num_components,
	uint64_t state_dimension,
	uint64_t degree,
	T dt,
	uint64_t quad_order
) {
	DiagFsskCoefficients<T> coef;
	coef.E.resize(state_dimension);
	for (uint64_t r = 0; r < state_dimension; ++r)
		coef.E[r] = std::exp(-lambda_diag[r] * dt);

	if (degree == 0)
		return coef;

	std::vector<uint64_t> multiindices;
	populate_multiindex_layout(
		num_components, degree - 1, coef.mi_level_index, multiindices, coef.minus_index);
	const uint64_t mi_len_f = coef.mi_level_index[degree];
	const uint64_t mi_len_g = degree >= 2 ? coef.mi_level_index[degree - 1] : 0;
	coef.psi.assign(mi_len_f * state_dimension, static_cast<T>(0));
	coef.phi.assign(num_components * mi_len_g * state_dimension * state_dimension, static_cast<T>(0));

	for (uint64_t r = 0; r < state_dimension; ++r)
		coef.psi[r] = phi1_neg(lambda_diag[r] * dt);

	const T pi = std::acos(static_cast<T>(-1));
	using C = std::complex<T>;

	for (uint64_t j = 1; j <= quad_order; ++j) {
		const T theta = (static_cast<T>(2 * j - 1) * pi) / static_cast<T>(2 * quad_order);
		const C zeta = static_cast<T>(2 * quad_order)
			* C(static_cast<T>(0.1309) - static_cast<T>(0.1194) * theta * theta,
				static_cast<T>(0.25) * theta);
		const C slope(static_cast<T>(0.25), static_cast<T>(0.2388) * theta);
		const C ez = std::exp(zeta);
		const C omega = ez * slope;
		const C tilde_omega = ez * slope / zeta;

		std::vector<C> inv(state_dimension);
		for (uint64_t r = 0; r < state_dimension; ++r)
			inv[r] = static_cast<T>(1) / (zeta + dt * lambda_diag[r]);

		std::vector<C> beta(num_components, C(static_cast<T>(0), static_cast<T>(0)));
		std::vector<C> u(num_components * state_dimension);
		for (uint64_t p = 0; p < num_components; ++p) {
			for (uint64_t r = 0; r < state_dimension; ++r) {
				const C val = b[p * state_dimension + r] * inv[r];
				u[p * state_dimension + r] = val;
				beta[p] += val;
			}
		}

		// psi covers word lengths 1..degree-1, phi covers 0..degree-2. Walk the
		// union once so multiindex_gamma is evaluated a single time per index.
		for (uint64_t word_len = 0; word_len <= degree - 1; ++word_len) {
			const uint64_t start = coef.mi_level_index[word_len];
			const uint64_t end = coef.mi_level_index[word_len + 1];
			const bool do_psi = word_len >= 1;
			const bool do_phi = word_len + 2 <= degree;
			for (uint64_t idx = start; idx < end; ++idx) {
				const C gamma = multiindex_gamma<T>(
					multiindices.data() + idx * num_components,
					num_components,
					beta);
				if (do_psi) {
					for (uint64_t r = 0; r < state_dimension; ++r) {
						coef.psi[idx * state_dimension + r] +=
							static_cast<T>(2) * std::real(tilde_omega * gamma * inv[r]);
					}
				}
				if (do_phi) {
					for (uint64_t p = 0; p < num_components; ++p) {
						T* phi_ptr = coef.phi.data()
							+ ((p * mi_len_g + idx) * state_dimension * state_dimension);
						for (uint64_t r0 = 0; r0 < state_dimension; ++r0) {
							for (uint64_t r1 = 0; r1 < state_dimension; ++r1) {
								phi_ptr[r0 * state_dimension + r1] += static_cast<T>(2)
									* std::real(omega * gamma * u[p * state_dimension + r0] * inv[r1]);
							}
						}
					}
				}
			}
		}
	}

	return coef;
}

template<std::floating_point T>
VolterraKernelCache<T> make_prepared_volterra_sig(
	const T* lambda_diag,
	const T* A,
	const T* b,
	uint64_t dimension,
	uint64_t num_components,
	uint64_t target_dimension,
	uint64_t state_dimension,
	uint64_t degree,
	T dt,
	T readout_lag,
	uint64_t quad_order
) {
	if (dimension == 0)
		throw std::invalid_argument("prepare_volterra_sig received path dimension 0");
	if (num_components == 0)
		throw std::invalid_argument("prepare_volterra_sig received num_components 0");
	if (target_dimension == 0)
		throw std::invalid_argument("prepare_volterra_sig received target_dimension 0");
	if (state_dimension == 0)
		throw std::invalid_argument("prepare_volterra_sig received state_dimension 0");
	if (!(dt > static_cast<T>(0)))
		throw std::invalid_argument("prepare_volterra_sig requires dt > 0");
	if (readout_lag < static_cast<T>(0))
		throw std::invalid_argument("prepare_volterra_sig requires readout_lag >= 0");
	if (quad_order == 0)
		throw std::invalid_argument("prepare_volterra_sig requires quad_order > 0");

	const uint64_t A_len = num_components * target_dimension * dimension;
	const uint64_t b_len = num_components * state_dimension;

	VolterraKernelCache<T> prepared;
	prepared.A.assign(A, A + A_len);
	prepared.readout_weights.resize(b_len);
	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t r = 0; r < state_dimension; ++r) {
			prepared.readout_weights[p * state_dimension + r] =
				std::exp(-lambda_diag[r] * readout_lag) * b[p * state_dimension + r];
		}
	}
	prepared.dimension = dimension;
	prepared.num_components = num_components;
	prepared.target_dimension = target_dimension;
	prepared.state_dimension = state_dimension;
	prepared.degree = degree;
	prepared.coef = build_diag_fssk_coefficients<T>(
		lambda_diag,
		b,
		num_components,
		state_dimension,
		degree,
		dt,
		quad_order);

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

	const uint64_t n_levels = degree;
	prepared.mi_off.assign(n_levels, 0);
	prepared.mi_ts.assign(n_levels, 0);
	prepared.mi_total = populate_multiindex_tensor_layout(
		prepared.mi_off.data(),
		prepared.mi_ts.data(),
		prepared.coef.mi_level_index.data(),
		target_dimension,
		n_levels);
	return prepared;
}

// ---------------------------------------------------------------------------
// Compute path, templated on the lane width L. Every per-path buffer carries a
// trailing lane dimension of width L (lane index innermost and contiguous), so
// each arithmetic loop vectorizes over the lane dimension and the per-step
// scalar setup is amortized over L paths. Each lane is an independent path, so
// the math is the single-path recursion applied lanewise. Instantiated at
// L == VOLTERRA_LANES for full tiles and at L == 1 for the batch remainder.
// ---------------------------------------------------------------------------

template<std::floating_point T, uint64_t L>
void compute_projected_increment(
	const T* path,
	uint64_t length,
	const VolterraKernelCache<T>& prepared,
	T* y,
	T* dx,
	uint64_t step
) {
	const uint64_t dimension = prepared.dimension;
	const uint64_t num_components = prepared.num_components;
	const uint64_t target_dimension = prepared.target_dimension;
	const uint64_t path_stride = length * dimension;

	for (uint64_t j = 0; j < dimension; ++j) {
		for (uint64_t b = 0; b < L; ++b) {
			const T* xb = path + b * path_stride + step * dimension;
			dx[j * L + b] = xb[dimension + j] - xb[j];
		}
	}
	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t i = 0; i < target_dimension; ++i) {
			T* RESTRICT yo = y + (p * target_dimension + i) * L;
			for (uint64_t b = 0; b < L; ++b)
				yo[b] = static_cast<T>(0);
			const T* A_row = prepared.A.data() + (p * target_dimension + i) * dimension;
			for (uint64_t j = 0; j < dimension; ++j) {
				const T a = A_row[j];
				const T* RESTRICT dxj = dx + j * L;
				for (uint64_t b = 0; b < L; ++b)
					yo[b] += a * dxj[b];
			}
		}
	}
}

template<std::floating_point T, uint64_t L>
void eval_fg(
	const VolterraKernelCache<T>& prepared,
	const T* y,
	T* F,
	T* G,
	T* shuffle_tensors
) {
	const auto& coef = prepared.coef;
	const uint64_t num_components = prepared.num_components;
	const uint64_t target_dimension = prepared.target_dimension;
	const uint64_t state_dimension = prepared.state_dimension;
	const uint64_t degree = prepared.degree;
	const uint64_t* level_index = prepared.level_index.data();
	const uint64_t* mi_off = prepared.mi_off.data();
	const uint64_t* mi_ts = prepared.mi_ts.data();
	const uint64_t f_len = prepared.f_len;

	std::fill(F, F + state_dimension * f_len * L, static_cast<T>(0));

	build_shuffle_tensors<T, L>(
		y, shuffle_tensors, num_components, target_dimension, degree,
		coef.mi_level_index.data(), coef.minus_index.data(), mi_off, mi_ts);

	for (uint64_t word_len = 0; word_len <= degree - 1; ++word_len) {
		const uint64_t mi_start = coef.mi_level_index[word_len];
		const uint64_t mi_end = coef.mi_level_index[word_len + 1];
		const uint64_t out_start = level_index[word_len];
		const uint64_t ts = mi_ts[word_len];
		const T* level_tensors = shuffle_tensors + mi_off[word_len] * L;
		for (uint64_t mi_idx = mi_start; mi_idx < mi_end; ++mi_idx) {
			const T* tensor = level_tensors + (mi_idx - mi_start) * ts * L;
			for (uint64_t r = 0; r < state_dimension; ++r) {
				T* RESTRICT dst = F + (r * f_len + out_start) * L;
				const T c = coef.psi[mi_idx * state_dimension + r];
				for (uint64_t i = 0; i < ts * L; ++i)
					dst[i] += c * tensor[i];
			}
		}
	}

	if (degree <= 1)
		return;

	std::fill(G, G + num_components * state_dimension * state_dimension * f_len * L, static_cast<T>(0));

	const uint64_t mi_len_g = coef.mi_level_index[degree - 1];
	for (uint64_t word_len = 0; word_len <= degree - 2; ++word_len) {
		const uint64_t mi_start = coef.mi_level_index[word_len];
		const uint64_t mi_end = coef.mi_level_index[word_len + 1];
		const uint64_t out_start = level_index[word_len];
		const uint64_t ts = mi_ts[word_len];
		const T* level_tensors = shuffle_tensors + mi_off[word_len] * L;
		for (uint64_t mi_idx = mi_start; mi_idx < mi_end; ++mi_idx) {
			const T* tensor = level_tensors + (mi_idx - mi_start) * ts * L;
			for (uint64_t p = 0; p < num_components; ++p) {
				const T* phi_ptr = coef.phi.data()
					+ ((p * mi_len_g + mi_idx) * state_dimension * state_dimension);
				for (uint64_t r0 = 0; r0 < state_dimension; ++r0) {
					for (uint64_t r1 = 0; r1 < state_dimension; ++r1) {
						T* RESTRICT dst = G + (((p * state_dimension + r0) * state_dimension + r1) * f_len + out_start) * L;
						const T c = phi_ptr[r0 * state_dimension + r1];
						for (uint64_t i = 0; i < ts * L; ++i)
							dst[i] += c * tensor[i];
					}
				}
			}
		}
	}
}

template<std::floating_point T, uint64_t L>
void build_scalar_tensor_powers(
	const T* y,
	T* tensors,
	const uint64_t* level_index,
	uint64_t target_dimension,
	uint64_t degree
) {
	for (uint64_t b = 0; b < L; ++b)
		tensors[b] = static_cast<T>(1);
	for (uint64_t level = 1; level <= degree - 1; ++level) {
		const uint64_t prev_start = level_index[level - 1];
		const uint64_t prev_size = level_index[level] - prev_start;
		T* dst = tensors + level_index[level] * L;
		const T* src = tensors + prev_start * L;
		for (uint64_t i = 0; i < prev_size; ++i) {
			const T* RESTRICT src_i = src + i * L;
			T* dst_row = dst + i * target_dimension * L;
			for (uint64_t d = 0; d < target_dimension; ++d) {
				const T* RESTRICT y_d = y + d * L;
				T* RESTRICT dst_d = dst_row + d * L;
				for (uint64_t b = 0; b < L; ++b)
					dst_d[b] = src_i[b] * y_d[b];
			}
		}
	}
}

template<std::floating_point T, uint64_t L>
void update_state_scalar(
	const VolterraKernelCache<T>& prepared,
	const T* y,
	T* state,
	T* next_state,
	T* B,
	T* tensor_powers
) {
	const auto& coef = prepared.coef;
	const uint64_t target_dimension = prepared.target_dimension;
	const uint64_t state_dimension = prepared.state_dimension;
	const uint64_t degree = prepared.degree;
	const uint64_t* level_index = prepared.level_index.data();
	const uint64_t f_len = prepared.f_len;
	const uint64_t state_len = prepared.state_len;

	build_scalar_tensor_powers<T, L>(y, tensor_powers, level_index, target_dimension, degree);

	for (uint64_t r = 0; r < state_dimension; ++r) {
		const T scale = coef.E[r];
		const T* RESTRICT src = state + r * state_len * L;
		T* RESTRICT dst = next_state + r * state_len * L;
		for (uint64_t i = 0; i < state_len * L; ++i)
			dst[i] = src[i] * scale;
	}

	for (uint64_t r = 0; r < state_dimension; ++r) {
		T* b_ptr = B + r * f_len * L;
		for (uint64_t level = 0; level <= degree - 1; ++level) {
			const uint64_t level_start = level_index[level];
			const uint64_t level_size = level_index[level + 1] - level_start;
			const T* RESTRICT tensor = tensor_powers + level_start * L;
			const T c = coef.psi[level * state_dimension + r];
			T* RESTRICT bdst = b_ptr + level_start * L;
			for (uint64_t i = 0; i < level_size * L; ++i)
				bdst[i] = c * tensor[i];
		}
	}

	for (uint64_t left_level = 1; left_level <= degree - 1; ++left_level) {
		const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
		const uint64_t left_state_start = level_index[left_level] - 1;
		for (uint64_t right_level = 0; right_level <= degree - 1 - left_level; ++right_level) {
			const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
			const uint64_t right_start = level_index[right_level];
			const uint64_t out_start = level_index[left_level + right_level];
			const T* right_tensor = tensor_powers + right_start * L;
			for (uint64_t r0 = 0; r0 < state_dimension; ++r0) {
				const T* z = state + (r0 * state_len + left_state_start) * L;
				for (uint64_t r1 = 0; r1 < state_dimension; ++r1) {
					const T g = coef.phi[(right_level * state_dimension + r0) * state_dimension + r1];
					T* dst = B + (r1 * f_len + out_start) * L;
					for (uint64_t i = 0; i < left_size; ++i) {
						const T* RESTRICT z_i = z + i * L;
						T* dst_row = dst + i * right_size * L;
						for (uint64_t j = 0; j < right_size; ++j) {
							const T* RESTRICT rt_j = right_tensor + j * L;
							T* RESTRICT dst_j = dst_row + j * L;
							for (uint64_t b = 0; b < L; ++b)
								dst_j[b] += (z_i[b] * g) * rt_j[b];
						}
					}
				}
			}
		}
	}

	for (uint64_t b_level = 0; b_level <= degree - 1; ++b_level) {
		const uint64_t b_size = level_index[b_level + 1] - level_index[b_level];
		const uint64_t b_start = level_index[b_level];
		const uint64_t out_start = level_index[b_level + 1] - 1;
		for (uint64_t r = 0; r < state_dimension; ++r) {
			const T* b_ptr = B + (r * f_len + b_start) * L;
			T* dst = next_state + (r * state_len + out_start) * L;
			for (uint64_t i = 0; i < b_size; ++i) {
				const T* RESTRICT b_i = b_ptr + i * L;
				T* dst_row = dst + i * target_dimension * L;
				for (uint64_t d = 0; d < target_dimension; ++d) {
					const T* RESTRICT y_d = y + d * L;
					T* RESTRICT dst_d = dst_row + d * L;
					for (uint64_t b = 0; b < L; ++b)
						dst_d[b] += b_i[b] * y_d[b];
				}
			}
		}
	}
}

template<std::floating_point T, uint64_t L>
void update_state(
	const VolterraKernelCache<T>& prepared,
	const T* y,
	T* state,
	T* next_state,
	T* F,
	T* G,
	T* B,
	T* shuffle_tensors
) {
	const auto& coef = prepared.coef;
	const uint64_t num_components = prepared.num_components;
	const uint64_t target_dimension = prepared.target_dimension;
	const uint64_t state_dimension = prepared.state_dimension;
	const uint64_t degree = prepared.degree;
	const uint64_t* level_index = prepared.level_index.data();
	const uint64_t f_len = prepared.f_len;
	const uint64_t state_len = prepared.state_len;

	if (num_components == 1) {
		update_state_scalar<T, L>(prepared, y, state, next_state, B, shuffle_tensors);
		return;
	}

	eval_fg<T, L>(prepared, y, F, G, shuffle_tensors);

	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t r = 0; r < state_dimension; ++r) {
			const T scale = coef.E[r];
			const T* RESTRICT src = state + (p * state_dimension + r) * state_len * L;
			T* RESTRICT dst = next_state + (p * state_dimension + r) * state_len * L;
			for (uint64_t i = 0; i < state_len * L; ++i)
				dst[i] = src[i] * scale;
		}
	}

	std::memcpy(B, F, state_dimension * f_len * L * sizeof(T));

	for (uint64_t p = 0; p < num_components; ++p) {
		for (uint64_t left_level = 1; left_level <= degree - 1; ++left_level) {
			const uint64_t left_size = level_index[left_level + 1] - level_index[left_level];
			const uint64_t left_state_start = level_index[left_level] - 1;
			for (uint64_t right_level = 0; right_level <= degree - 1 - left_level; ++right_level) {
				const uint64_t right_size = level_index[right_level + 1] - level_index[right_level];
				const uint64_t right_start = level_index[right_level];
				const uint64_t out_level = left_level + right_level;
				const uint64_t out_start = level_index[out_level];

				for (uint64_t r0 = 0; r0 < state_dimension; ++r0) {
					const T* z = state + ((p * state_dimension + r0) * state_len + left_state_start) * L;
					for (uint64_t r1 = 0; r1 < state_dimension; ++r1) {
						const T* g = G + (((p * state_dimension + r0) * state_dimension + r1) * f_len + right_start) * L;
						T* dst = B + (r1 * f_len + out_start) * L;
						for (uint64_t i = 0; i < left_size; ++i) {
							const T* RESTRICT z_i = z + i * L;
							T* dst_row = dst + i * right_size * L;
							for (uint64_t j = 0; j < right_size; ++j) {
								const T* RESTRICT g_j = g + j * L;
								T* RESTRICT dst_j = dst_row + j * L;
								for (uint64_t b = 0; b < L; ++b)
									dst_j[b] += z_i[b] * g_j[b];
							}
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
			const uint64_t out_level = b_level + 1;
			const uint64_t out_start = level_index[out_level] - 1;
			for (uint64_t r = 0; r < state_dimension; ++r) {
				const T* b_ptr = B + (r * f_len + b_start) * L;
				T* dst = next_state + ((p * state_dimension + r) * state_len + out_start) * L;
				const T* y_ptr = y + (p * target_dimension) * L;
				for (uint64_t i = 0; i < b_size; ++i) {
					const T* RESTRICT b_i = b_ptr + i * L;
					T* dst_row = dst + i * target_dimension * L;
					for (uint64_t d = 0; d < target_dimension; ++d) {
						const T* RESTRICT y_d = y_ptr + d * L;
						T* RESTRICT dst_d = dst_row + d * L;
						for (uint64_t b = 0; b < L; ++b)
							dst_d[b] += b_i[b] * y_d[b];
					}
				}
			}
		}
	}
}

template<std::floating_point T, uint64_t L>
void readout_state(
	const T* state,
	const VolterraKernelCache<T>& prepared,
	T* tile_out,
	bool scalar_term
) {
	const uint64_t num_components = prepared.num_components;
	const uint64_t state_dimension = prepared.state_dimension;
	const uint64_t degree = prepared.degree;
	const uint64_t* level_index = prepared.level_index.data();
	const uint64_t sig_len = prepared.sig_len;
	const uint64_t state_len = prepared.state_len;
	const uint64_t out_len = scalar_term ? sig_len : sig_len - 1;

	std::fill(tile_out, tile_out + out_len * L, static_cast<T>(0));
	if (scalar_term)
		for (uint64_t b = 0; b < L; ++b)
			tile_out[b] = static_cast<T>(1);

	for (uint64_t level = 1; level <= degree; ++level) {
		const uint64_t level_size = level_index[level + 1] - level_index[level];
		const uint64_t state_start = level_index[level] - 1;
		const uint64_t out_start = scalar_term ? level_index[level] : level_index[level] - 1;
		T* dst = tile_out + out_start * L;
		for (uint64_t p = 0; p < num_components; ++p) {
			for (uint64_t r = 0; r < state_dimension; ++r) {
				const T weight = prepared.readout_weights[p * state_dimension + r];
				const T* RESTRICT src = state + ((p * state_dimension + r) * state_len + state_start) * L;
				for (uint64_t i = 0; i < level_size * L; ++i)
					dst[i] += weight * src[i];
			}
		}
	}
}

template<std::floating_point T, uint64_t L>
void volterra_sig_run(
	const T* path,
	const VolterraKernelCache<T>& prepared,
	T* out,
	uint64_t out_stride,
	VolterraWorkspace<T>& ws,
	uint64_t length,
	bool scalar_term
) {
	T* cur = ws.state;
	T* nxt = ws.next_state;
	std::fill(ws.state, ws.state + ws.state_size, static_cast<T>(0));
	for (uint64_t step = 0; step + 1 < length; ++step) {
		compute_projected_increment<T, L>(path, length, prepared, ws.y, ws.dx, step);
		update_state<T, L>(prepared, ws.y, cur, nxt, ws.F, ws.G, ws.B, ws.shuffle_tensors);
		std::swap(cur, nxt);
	}

	readout_state<T, L>(cur, prepared, ws.tile_out, scalar_term);

	const uint64_t out_len = scalar_term ? prepared.sig_len : prepared.sig_len - 1;
	for (uint64_t b = 0; b < L; ++b) {
		T* RESTRICT dst = out + b * out_stride;
		const T* RESTRICT src = ws.tile_out + b;
		for (uint64_t pos = 0; pos < out_len; ++pos)
			dst[pos] = src[pos * L];
	}
}

std::mutex prepared_volterra_sig_mu;
std::atomic<uint64_t> next_prepared_volterra_sig_handle{ 1 };
std::unordered_map<uint64_t, VolterraKernelCache<float>> prepared_volterra_sig_f;
std::unordered_map<uint64_t, VolterraKernelCache<double>> prepared_volterra_sig_d;

template<std::floating_point T>
std::unordered_map<uint64_t, VolterraKernelCache<T>>& prepared_volterra_sig_map() {
	if constexpr (std::is_same_v<T, float>)
		return prepared_volterra_sig_f;
	else
		return prepared_volterra_sig_d;
}

template<std::floating_point T>
uint64_t store_prepared_volterra_sig(VolterraKernelCache<T>&& prepared) {
	const uint64_t handle = next_prepared_volterra_sig_handle.fetch_add(1);
	if (handle == 0)
		throw std::overflow_error("prepare_volterra_sig handle overflow");
	std::lock_guard lock(prepared_volterra_sig_mu);
	prepared_volterra_sig_map<T>().emplace(handle, std::move(prepared));
	return handle;
}

template<std::floating_point T>
const VolterraKernelCache<T>& get_prepared_volterra_sig(uint64_t handle) {
	std::lock_guard lock(prepared_volterra_sig_mu);
	auto& map = prepared_volterra_sig_map<T>();
	const auto it = map.find(handle);
	if (it == map.end())
		throw std::invalid_argument("volterra_sig received an invalid prepared handle");
	return it->second;
}

template<std::floating_point T>
void free_prepared_volterra_sig(uint64_t handle) {
	if (handle == 0)
		return;
	std::lock_guard lock(prepared_volterra_sig_mu);
	prepared_volterra_sig_map<T>().erase(handle);
}

} // namespace

void clear_prepared_volterra_sig_cache() {
	std::lock_guard lock(prepared_volterra_sig_mu);
	prepared_volterra_sig_f.clear();
	prepared_volterra_sig_d.clear();
}

template<std::floating_point T>
void volterra_sig_(
	const T* path,
	const VolterraKernelCache<T>& prepared,
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

	// One tile-sized workspace per thread, reused across every path in the
	// chunk. Full tiles of VOLTERRA_LANES paths take the batch-vectorized path;
	// the leftover (< VOLTERRA_LANES) paths run the same code at lane width 1.
	const uint64_t path_stride = length * prepared.dimension;
	auto run_range = [&](uint64_t start, uint64_t end) {
		VolterraWorkspace<T> ws(prepared);
		uint64_t i = start;
		for (; i + VOLTERRA_LANES <= end; i += VOLTERRA_LANES) {
			volterra_sig_run<T, VOLTERRA_LANES>(
				path + i * path_stride, prepared, out + i * out_stride,
				out_stride, ws, length, scalar_term);
		}
		for (; i < end; ++i) {
			volterra_sig_run<T, 1>(
				path + i * path_stride, prepared, out + i * out_stride,
				out_stride, ws, length, scalar_term);
		}
	};

	if (batch_size == 0)
		return;
	if (n_jobs == 1 || batch_size == 1)
		run_range(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, run_range);
}

extern "C" {

	CPSIG_API int prepare_volterra_sig_f(
		const float* lambda_diag,
		const float* A,
		const float* b,
		uint64_t dimension,
		uint64_t num_components,
		uint64_t target_dimension,
		uint64_t state_dimension,
		uint64_t degree,
		float dt,
		float readout_lag,
		uint64_t quad_order,
		uint64_t* handle
	) noexcept {
		SAFE_CALL({
			*handle = store_prepared_volterra_sig<float>(make_prepared_volterra_sig<float>(
				lambda_diag, A, b, dimension, num_components, target_dimension,
				state_dimension, degree, dt, readout_lag, quad_order));
		});
	}

	CPSIG_API int prepare_volterra_sig_d(
		const double* lambda_diag,
		const double* A,
		const double* b,
		uint64_t dimension,
		uint64_t num_components,
		uint64_t target_dimension,
		uint64_t state_dimension,
		uint64_t degree,
		double dt,
		double readout_lag,
		uint64_t quad_order,
		uint64_t* handle
	) noexcept {
		SAFE_CALL({
			*handle = store_prepared_volterra_sig<double>(make_prepared_volterra_sig<double>(
				lambda_diag, A, b, dimension, num_components, target_dimension,
				state_dimension, degree, dt, readout_lag, quad_order));
		});
	}

	CPSIG_API int free_volterra_sig_f(uint64_t handle) noexcept {
		SAFE_CALL(free_prepared_volterra_sig<float>(handle));
	}

	CPSIG_API int free_volterra_sig_d(uint64_t handle) noexcept {
		SAFE_CALL(free_prepared_volterra_sig<double>(handle));
	}

	CPSIG_API int volterra_sig_f(
		const float* path,
		float* out,
		uint64_t handle,
		uint64_t batch_size,
		uint64_t dimension,
		uint64_t length,
		bool scalar_term,
		int n_jobs
	) noexcept {
		SAFE_CALL(volterra_sig_<float>(
			path, get_prepared_volterra_sig<float>(handle), out, batch_size,
			dimension, length, scalar_term, n_jobs));
	}

	CPSIG_API int volterra_sig_d(
		const double* path,
		double* out,
		uint64_t handle,
		uint64_t batch_size,
		uint64_t dimension,
		uint64_t length,
		bool scalar_term,
		int n_jobs
	) noexcept {
		SAFE_CALL(volterra_sig_<double>(
			path, get_prepared_volterra_sig<double>(handle), out, batch_size,
			dimension, length, scalar_term, n_jobs));
	}
}
