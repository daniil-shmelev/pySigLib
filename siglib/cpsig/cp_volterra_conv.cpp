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
// The local increment E = evalVtE-without-v is
//
//     E^(n) = sum_{|ell| = n-1} ( T_ell (x) sum_p alpha[p, ell] y_p ),
//
// where T_ell is the normalized shuffle tensor (shuffle power / ell!) built by
// build_shuffle_tensors. For q == 1 this reduces to a scalar Horner evaluation
// (no shuffle layout needed); for q > 1 it uses the packed multi-index shuffle
// tensors, exactly as the finite state-space scheme does. evalVtE then Chen-
// multiplies the running signature: contribution^(n) = sum_{b=1..n} V_i^(n-b)
// (x) E^(b).

#include "cppch.h"
#include "cpsig.h"
#include "cp_utils.h"
#include "cp_volterra_common.h"
#include "macros.h"
#include "multithreading.h"

#include <stdexcept>
#include <vector>

namespace {

using namespace volterra_detail;

// Fixed per-call layout shared by the recursion and the local-increment eval.
// level_index is the signature layout (level n at level_index[n], size m^n);
// the mi_* arrays are the packed multi-index / shuffle-tensor layout and are
// only populated (non-null) for q > 1.
struct ConvLayout {
	uint64_t q;
	uint64_t m;
	uint64_t degree;
	uint64_t M;            // number of multi-indices of degree <= degree-1
	uint64_t mi_total;     // packed shuffle-tensor size (q > 1)
	uint64_t state_total;  // signature length incl. scalar term = level_index[degree+1]
	const uint64_t* level_index;
	const uint64_t* mi_level_index;
	const uint64_t* minus_index;
	const uint64_t* mi_off;
	const uint64_t* mi_ts;
};

// Scalar (q == 1) Horner evaluation of v (x)_N E, accumulating into acc.
// av holds the M = degree lag coefficients beta_n = av[n-1].
template<std::floating_point T>
void eval_vte_horner(
	const T* RESTRICT v,
	const T* RESTRICT y,
	const T* RESTRICT av,
	const ConvLayout& L,
	T* Wcur,
	T* Wnxt,
	T* RESTRICT acc
) {
	const uint64_t m = L.m;
	const uint64_t* level_index = L.level_index;
	for (uint64_t n = 1; n <= L.degree; ++n) {
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

// General (q >= 1) evaluation of v (x)_N E, accumulating into acc. Builds the
// local increment E from the precomputed shuffle tensors and the lag
// coefficients, then Chen-multiplies the running signature v. ``w`` is a
// length-m scratch buffer.
template<std::floating_point T>
void eval_vte_general(
	const T* RESTRICT v,            // V_i, signature layout
	const T* RESTRICT shuffle,      // T_ell for source i (mi_total)
	const T* RESTRICT y,            // y_i, (q, m)
	const T* RESTRICT alpha_row,    // (q, M) for this lag
	const ConvLayout& L,
	T* RESTRICT w,                  // length-m scratch
	T* RESTRICT E,                  // scratch, signature layout (level 0 unused)
	T* RESTRICT acc                 // V_k, signature layout
) {
	const uint64_t m = L.m;
	const uint64_t q = L.q;
	const uint64_t M = L.M;
	const uint64_t* level_index = L.level_index;
	std::fill(E, E + L.state_total, T(0));

	// E^(n) = sum_{|ell|=n-1} T_ell (x) (sum_p alpha[p, ell] y_p)
	for (uint64_t n = 1; n <= L.degree; ++n) {
		T* RESTRICT En = E + level_index[n];
		const uint64_t prev_ts = L.mi_ts[n - 1];          // m^(n-1)
		const uint64_t mi_start = L.mi_level_index[n - 1];
		const uint64_t mi_end = L.mi_level_index[n];
		const T* RESTRICT level_T = shuffle + L.mi_off[n - 1];
		for (uint64_t idx = mi_start; idx < mi_end; ++idx) {
			// w = sum_p alpha[p, idx] * y_p  (the trailing letter, length m)
			for (uint64_t d = 0; d < m; ++d)
				w[d] = T(0);
			for (uint64_t p = 0; p < q; ++p) {
				const T c = alpha_row[p * M + idx];
				const T* RESTRICT yp = y + p * m;
				for (uint64_t d = 0; d < m; ++d)
					w[d] += c * yp[d];
			}
			// En += T_ell (x) w
			const T* RESTRICT Tl = level_T + (idx - mi_start) * prev_ts;
			for (uint64_t cc = 0; cc < prev_ts; ++cc) {
				const T tlc = Tl[cc];
				T* RESTRICT dst = En + cc * m;
				for (uint64_t d = 0; d < m; ++d)
					dst[d] += tlc * w[d];
			}
		}
	}

	// acc += v (x) E : contribution^(n) = sum_{b=1..n} v^(n-b) (x) E^(b)
	for (uint64_t n = 1; n <= L.degree; ++n) {
		T* RESTRICT acc_n = acc + level_index[n];
		for (uint64_t b = 1; b <= n; ++b) {
			const uint64_t a = n - b;
			const T* RESTRICT Va = v + level_index[a];
			const T* RESTRICT Eb = E + level_index[b];
			const uint64_t size_a = level_index[a + 1] - level_index[a];   // m^a
			const uint64_t size_b = level_index[b + 1] - level_index[b];   // m^b
			for (uint64_t u = 0; u < size_a; ++u) {
				const T vu = Va[u];
				T* RESTRICT dst = acc_n + u * size_b;
				for (uint64_t ww = 0; ww < size_b; ++ww)
					dst[ww] += vu * Eb[ww];
			}
		}
	}
}

template<std::floating_point T>
struct ConvWorkspace {
	std::vector<T> y;             // (S, q, m) projected increments
	std::vector<T> V;             // (S+1, state_total) running signatures
	std::vector<T> dx;            // (dimension,) increment scratch
	std::vector<T> Wa, Wb;        // Horner scratch (q == 1)
	std::vector<T> shuffle;       // (S, mi_total) shuffle tensors (q > 1)
	std::vector<T> E;             // (state_total) local-increment scratch (q > 1)
	std::vector<T> w;             // (m,) trailing-letter scratch (q > 1)

	ConvWorkspace(uint64_t max_steps, const ConvLayout& L, uint64_t dimension)
		: y(max_steps * L.q * L.m, T(0)),
		  V((max_steps + 1) * L.state_total, T(0)),
		  dx(dimension, T(0)),
		  Wa(L.q == 1 ? L.state_total : 0, T(0)),
		  Wb(L.q == 1 ? L.state_total : 0, T(0)),
		  shuffle(L.q == 1 ? 0 : max_steps * L.mi_total, T(0)),
		  E(L.q == 1 ? 0 : L.state_total, T(0)),
		  w(L.q == 1 ? 0 : L.m, T(0)) {}
};

template<std::floating_point T>
void volterra_conv_single(
	const T* RESTRICT path,
	const T* RESTRICT A,
	const T* RESTRICT alpha_lag,
	T* RESTRICT out,
	ConvWorkspace<T>& ws,
	const ConvLayout& L,
	uint64_t dimension,
	uint64_t length,
	bool scalar_term
) {
	const uint64_t q = L.q;
	const uint64_t m = L.m;
	const uint64_t M = L.M;
	const uint64_t state_total = L.state_total;
	const uint64_t S = length - 1;
	const uint64_t qm = q * m;
	T* RESTRICT y = ws.y.data();
	T* RESTRICT dx = ws.dx.data();

	// Projected increments y_i = A dX_i, stored as (S, q, m).
	for (uint64_t i = 0; i < S; ++i) {
		const T* RESTRICT xb = path + i * dimension;
		for (uint64_t d = 0; d < dimension; ++d)
			dx[d] = xb[dimension + d] - xb[d];
		T* RESTRICT yi = y + i * qm;
		for (uint64_t po = 0; po < qm; ++po) {
			const T* RESTRICT A_row = A + po * dimension;
			T acc = T(0);
			for (uint64_t d = 0; d < dimension; ++d)
				acc += A_row[d] * dx[d];
			yi[po] = acc;
		}
	}

	// For q > 1, precompute the shuffle tensors T_ell once per source.
	if (q > 1) {
		for (uint64_t i = 0; i < S; ++i)
			build_shuffle_tensors<T, 1>(
				y + i * qm, ws.shuffle.data() + i * L.mi_total, q, m, L.degree,
				L.mi_level_index, L.minus_index, L.mi_off, L.mi_ts);
	}

	T* RESTRICT V = ws.V.data();
	std::fill(ws.V.begin(), ws.V.end(), T(0));
	V[0] = T(1);  // V[0] = unit

	for (uint64_t k = 1; k <= S; ++k) {
		T* RESTRICT Vk = V + k * state_total;
		Vk[0] = T(1);  // unit level 0
		for (uint64_t i = 0; i < k; ++i) {
			const uint64_t lag = k - i;
			const T* RESTRICT alpha_row = alpha_lag + lag * q * M;
			if (q == 1) {
				eval_vte_horner<T>(
					V + i * state_total, y + i * qm, alpha_row,
					L, ws.Wa.data(), ws.Wb.data(), Vk);
			} else {
				eval_vte_general<T>(
					V + i * state_total, ws.shuffle.data() + i * L.mi_total,
					y + i * qm, alpha_row, L, ws.w.data(), ws.E.data(), Vk);
			}
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
	if (num_components == 0)
		throw std::invalid_argument("volterra_conv_sig requires num_components >= 1");
	if (degree == 0)
		throw std::invalid_argument("volterra_conv_sig requires degree >= 1");

	const uint64_t q = num_components;
	const uint64_t m = target_dimension;
	std::vector<uint64_t> level_index(degree + 2, 0);
	populate_level_index(level_index.data(), m, degree + 2);

	// Multi-index + shuffle-tensor layout (q > 1). For q == 1 the Horner path
	// needs no shuffle layout, and num_multiindex is just the truncation degree.
	std::vector<uint64_t> mi_level_index, multiindices, minus_index, mi_off, mi_ts;
	uint64_t mi_total = 0;
	uint64_t expected_M = degree;
	if (q > 1) {
		populate_multiindex_layout(q, degree - 1, mi_level_index, multiindices, minus_index);
		expected_M = mi_level_index[degree];
		mi_off.assign(degree, 0);
		mi_ts.assign(degree, 0);
		mi_total = populate_multiindex_tensor_layout(
			mi_off.data(), mi_ts.data(), mi_level_index.data(), m, degree);
	}
	if (num_multiindex != expected_M)
		throw std::invalid_argument("volterra_conv_sig num_multiindex does not match the layout");

	const ConvLayout L{
		q, m, degree, num_multiindex, mi_total, level_index[degree + 1],
		level_index.data(), mi_level_index.data(), minus_index.data(),
		mi_off.data(), mi_ts.data()};
	const uint64_t out_stride = scalar_term ? L.state_total : L.state_total - 1;

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
		ConvWorkspace<T> ws(length - 1, L, dimension);
		for (uint64_t b = start; b < end; ++b) {
			volterra_conv_single<T>(
				path + b * length * dimension, A, alpha_lag,
				out + b * out_stride, ws, L, dimension, length, scalar_term);
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
