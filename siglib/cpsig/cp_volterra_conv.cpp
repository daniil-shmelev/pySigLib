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

// General convolution scheme for Volterra signatures (tensordev "quadratic"
// scheme at order 0). Unlike the finite state-space scheme, the kernel is a
// general convolution kernel K(t,s) = sum_p k_p(t-s) A_p whose scalar parts
// k_p are NOT sums of exponentials (e.g. fractional t^(beta-1)/Gamma(beta) or
// Gamma kernels), so there is no finite state to carry. Instead the truncated
// signature is built by the quadratic Volterra-Chen recursion
//
//     V(t_k) = unit + sum_{i < k} evalVtE( V(t_i), y_i, alpha(s_i, t_i, t_k) ),
//
// where y_i = A dX_i is the projected increment over interval i and
// alpha(s_i, t_i, tau) are the normalized interval coefficients of the kernel.
// On a uniform grid alpha depends only on the lag (k - i), so the Python layer
// (pysiglib/_volterra_conv.py) precomputes the O(S) lag coefficients via the
// closed-form (fractional) or Gauss-Legendre (Gamma) expressions and passes
// them in here. This file only runs the O(S^2) recursion.
//
// This increment implements the scalar (q == 1) case: the Gamma kernel (q == 1
// by definition) and the scalar fractional kernel, using the Horner evaluation
// of evalVtE. The multivariate (q > 1) fractional case needs the shuffle
// algebra and is a separate follow-up; the Python layer rejects q > 1.

#include "cppch.h"
#include "cpsig.h"
#include "cp_utils.h"
#include "macros.h"
#include "multithreading.h"

#include <stdexcept>
#include <vector>

namespace {

// Scalar (q == 1) Horner evaluation of v (x)_N E, accumulating into acc.
// av holds the M = degree lag coefficients beta_n = av[n-1]; v and acc are
// signature-layout vectors (level n at level_index[n], size m^n).
template<std::floating_point T>
void eval_vte_accumulate(
	const T* RESTRICT v,
	const T* RESTRICT y,
	const T* RESTRICT av,
	uint64_t m,
	uint64_t degree,
	const uint64_t* level_index,
	T* Wcur,
	T* Wnxt,
	T* RESTRICT acc
) {
	for (uint64_t n = 1; n <= degree; ++n) {
		// W = v_level0 * beta_n  (size 1)
		Wcur[0] = v[0] * av[n - 1];
		uint64_t wlen = 1;
		for (uint64_t k = 1; k < n; ++k) {
			// W <- (W (x) y) + v_level_k * beta_{n-k}, fused into one pass.
			const T* RESTRICT v_k = v + level_index[k];
			const T coef = av[n - k - 1];
			for (uint64_t a = 0; a < wlen; ++a) {
				const T wa = Wcur[a];
				const T* RESTRICT vrow = v_k + a * m;
				T* RESTRICT drow = Wnxt + a * m;
				for (uint64_t d = 0; d < m; ++d)
					drow[d] = wa * y[d] + coef * vrow[d];
			}
			wlen *= m;
			std::swap(Wcur, Wnxt);
		}
		// acc_level_n += W (x) y
		T* RESTRICT acc_n = acc + level_index[n];
		for (uint64_t a = 0; a < wlen; ++a) {
			const T wa = Wcur[a];
			T* RESTRICT row = acc_n + a * m;
			for (uint64_t d = 0; d < m; ++d)
				row[d] += wa * y[d];
		}
	}
}

template<std::floating_point T>
struct ConvWorkspace {
	std::vector<T> y;             // (S, m) projected increments
	std::vector<T> V;             // (S+1, state_total) running signatures
	std::vector<T> Wa, Wb;        // Horner scratch
	std::vector<T> dx;            // (dimension,) increment scratch

	ConvWorkspace(uint64_t max_steps, uint64_t m, uint64_t state_total,
		uint64_t dimension)
		: y(max_steps * m, T(0)),
		  V((max_steps + 1) * state_total, T(0)),
		  Wa(state_total, T(0)),
		  Wb(state_total, T(0)),
		  dx(dimension, T(0)) {}
};

template<std::floating_point T>
void volterra_conv_single(
	const T* RESTRICT path,
	const T* RESTRICT A,
	const T* RESTRICT alpha_lag,
	T* RESTRICT out,
	ConvWorkspace<T>& ws,
	uint64_t dimension,
	uint64_t length,
	uint64_t m,
	uint64_t degree,
	uint64_t M,
	const uint64_t* level_index,
	uint64_t state_total,
	bool scalar_term
) {
	const uint64_t S = length - 1;
	T* RESTRICT y = ws.y.data();
	T* RESTRICT dx = ws.dx.data();

	// Projected increments y_i = A_0 dX_i  (q == 1).
	for (uint64_t i = 0; i < S; ++i) {
		const T* RESTRICT xb = path + i * dimension;
		for (uint64_t d = 0; d < dimension; ++d)
			dx[d] = xb[dimension + d] - xb[d];
		T* RESTRICT yi = y + i * m;
		for (uint64_t o = 0; o < m; ++o) {
			const T* RESTRICT A_row = A + o * dimension;
			T acc = T(0);
			for (uint64_t d = 0; d < dimension; ++d)
				acc += A_row[d] * dx[d];
			yi[o] = acc;
		}
	}

	T* RESTRICT V = ws.V.data();
	std::fill(ws.V.begin(), ws.V.end(), T(0));
	V[0] = T(1);  // V[0] = unit

	for (uint64_t k = 1; k <= S; ++k) {
		T* RESTRICT Vk = V + k * state_total;
		Vk[0] = T(1);  // unit level 0
		for (uint64_t i = 0; i < k; ++i) {
			const uint64_t lag = k - i;
			eval_vte_accumulate<T>(
				V + i * state_total, y + i * m, alpha_lag + lag * M,
				m, degree, level_index, ws.Wa.data(), ws.Wb.data(), Vk);
		}
	}

	const T* RESTRICT VS = V + S * state_total;
	if (scalar_term) {
		for (uint64_t i = 0; i < state_total; ++i)
			out[i] = VS[i];
	} else {
		for (uint64_t i = 1; i < state_total; ++i)
			out[i - 1] = VS[i];
	}
}

template<std::floating_point T>
void volterra_conv_sig_impl(
	const T* path,
	T* out,
	const T* A,
	const T* alpha_lag,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t num_components,
	uint64_t target_dimension,
	uint64_t degree,
	uint64_t num_multiindex,
	bool scalar_term,
	int n_jobs
) {
	if (num_components != 1)
		throw std::invalid_argument("volterra_conv_sig supports q == 1 in this build");
	if (degree == 0 || num_multiindex != degree)
		throw std::invalid_argument("volterra_conv_sig requires num_multiindex == degree >= 1 for q == 1");

	const uint64_t m = target_dimension;
	std::vector<uint64_t> level_index(degree + 2, 0);
	populate_level_index(level_index.data(), m, degree + 2);
	const uint64_t state_total = level_index[degree + 1];
	const uint64_t out_stride = scalar_term ? state_total : state_total - 1;

	if (batch_size == 0)
		return;
	if (length <= 1) {
		T* const out_end = out + out_stride * batch_size;
		std::fill(out, out_end, T(0));
		if (scalar_term)
			for (T* p = out; p < out_end; p += out_stride)
				p[0] = T(1);
		return;
	}

	auto run_range = [&](uint64_t start, uint64_t end) {
		ConvWorkspace<T> ws(length - 1, m, state_total, dimension);
		for (uint64_t b = start; b < end; ++b) {
			volterra_conv_single<T>(
				path + b * length * dimension, A, alpha_lag,
				out + b * out_stride, ws, dimension, length, m, degree,
				num_multiindex, level_index.data(), state_total, scalar_term);
		}
	};

	if (n_jobs == 1 || batch_size == 1)
		run_range(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, run_range);
}

} // namespace

extern "C" {

	CPSIG_API int volterra_conv_sig_f(
		const float* path, float* out, const float* A, const float* alpha_lag,
		uint64_t batch_size, uint64_t dimension, uint64_t length,
		uint64_t num_components, uint64_t target_dimension, uint64_t degree,
		uint64_t num_multiindex, bool scalar_term, int n_jobs
	) noexcept {
		SAFE_CALL(volterra_conv_sig_impl<float>(
			path, out, A, alpha_lag, batch_size, dimension, length,
			num_components, target_dimension, degree, num_multiindex,
			scalar_term, n_jobs));
	}

	CPSIG_API int volterra_conv_sig_d(
		const double* path, double* out, const double* A, const double* alpha_lag,
		uint64_t batch_size, uint64_t dimension, uint64_t length,
		uint64_t num_components, uint64_t target_dimension, uint64_t degree,
		uint64_t num_multiindex, bool scalar_term, int n_jobs
	) noexcept {
		SAFE_CALL(volterra_conv_sig_impl<double>(
			path, out, A, alpha_lag, batch_size, dimension, length,
			num_components, target_dimension, degree, num_multiindex,
			scalar_term, n_jobs));
	}

}
