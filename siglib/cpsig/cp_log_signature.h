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
#include "cppch.h"

#include "multithreading.h"
#include "cp_tensor_log.h"

#include "cp_path.h"
#include "cp_bch.h"

template<std::floating_point T>
void get_log_sig_(
	const T* sig,
	T* out,
	uint64_t dimension,
	uint64_t degree,
	int method = 0
)
{
	switch (method) {
	case 0:
		log_sig_expanded<T>(sig, out, dimension, degree);
		break;
	case 1:
		log_sig_lyndon_words<T>(sig, out, dimension, degree);
		break;
	case 2:
		log_sig_lyndon_basis<T>(sig, out, dimension, degree);
		break;
	default:
		throw std::runtime_error("method must be one of 0, 1 or 2");
	}
}

template<std::floating_point T>
void sig_to_log_sig_(
	const T* sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	bool scalar_term = true,
	int n_jobs = 1
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("signature received dimension 0"); }
	if (degree == 0) { throw std::invalid_argument("log signature received degree 0"); }

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);

	const uint64_t sig_len = ::sig_length(aug_dimension, degree);
	const uint64_t logsig_len = ::log_sig_length(aug_dimension, degree);
	const uint64_t full_out_len = method ? logsig_len : sig_len;
	const uint64_t in_stride = scalar_term ? sig_len : sig_len - 1;
	const uint64_t out_stride = (method == 0 && !scalar_term) ? (sig_len - 1) : full_out_len;

	if (scalar_term) {
		auto f = [&](const T* sig_ptr, T* out_ptr) {
			get_log_sig_<T>(sig_ptr, out_ptr, aug_dimension, degree, method);
		};
		multi_threaded_batch(f, batch_size, n_jobs,
			make_batch(sig, in_stride), make_batch(out, out_stride));
	} else {
		auto f = [&](const T* sig_ptr, T* out_ptr) {
			std::vector<T> sig_full(sig_len);
			sig_full[0] = static_cast<T>(1);
			std::memcpy(sig_full.data() + 1, sig_ptr, (sig_len - 1) * sizeof(T));
			if (method == 0) {
				std::vector<T> out_full(sig_len);
				get_log_sig_<T>(sig_full.data(), out_full.data(), aug_dimension, degree, method);
				std::memcpy(out_ptr, out_full.data() + 1, (sig_len - 1) * sizeof(T));
			} else {
				get_log_sig_<T>(sig_full.data(), out_ptr, aug_dimension, degree, method);
			}
		};
		multi_threaded_batch(f, batch_size, n_jobs,
			make_batch(sig, in_stride), make_batch(out, out_stride));
	}
	return;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//// backpropagation
////////////////////////////////////////////////////////////////////////////////////////////////

template<std::floating_point T>
void get_sig_to_log_sig_backprop_(
	const T* sig,
	T* out,
	T* log_sig_derivs,
	uint64_t dimension,
	uint64_t degree,
	int method = 0
) {
	switch (method) {
	case 0:
		tensor_log_backprop_<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	case 1:
		tensor_log_backprop_lyndon_words<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	case 2:
		tensor_log_backprop_lyndon_basis<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	default:
		throw std::runtime_error("method must be one of 0, 1 or 2");
	}
}

template<std::floating_point T>
void sig_to_log_sig_backprop_(
	const T* sig,
	T* out,
	const T* log_sig_derivs,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	bool scalar_term = true,
	int n_jobs = 1
) {
	if (dimension == 0) { throw std::invalid_argument("sig_backprop received path of dimension 0"); }

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);

	const uint64_t sig_len_ = ::sig_length(aug_dimension, degree);
	const uint64_t log_sig_len_ = method ? ::log_sig_length(aug_dimension, degree) : sig_len_;

	// Determine caller strides
	const uint64_t sig_in_stride = scalar_term ? sig_len_ : sig_len_ - 1;
	// log_sig_derivs: logsig-shaped (method>0, unaffected) or sig-shaped (method==0, may be stripped)
	const uint64_t lsd_stride = (method == 0 && !scalar_term) ? (sig_len_ - 1) : log_sig_len_;
	// Output is sig-shaped
	const uint64_t out_stride = scalar_term ? sig_len_ : sig_len_ - 1;

	if (scalar_term) {
		auto log_sig_derivs_copy_uptr = std::make_unique<T[]>(log_sig_len_ * batch_size);
		T* log_sig_derivs_copy = log_sig_derivs_copy_uptr.get();
		std::memcpy(log_sig_derivs_copy, log_sig_derivs, log_sig_len_ * batch_size * sizeof(T));

		auto log_sig_backprop_func = [&](const T* sig_ptr, T* log_sig_derivs_ptr, T* out_ptr) {
			get_sig_to_log_sig_backprop_<T>(sig_ptr, out_ptr, log_sig_derivs_ptr, aug_dimension, degree, method);
		};

		multi_threaded_batch(log_sig_backprop_func, batch_size, n_jobs,
			make_batch(sig, sig_len_),
			make_batch(log_sig_derivs_copy, log_sig_len_),
			make_batch(out, sig_len_));
	} else {
		// Per-element: prepend scalars to sig input and (if method==0) to log_sig_derivs,
		// compute on full buffers, strip output
		auto log_sig_derivs_copy_uptr = std::make_unique<T[]>(log_sig_len_ * batch_size);
		T* log_sig_derivs_copy = log_sig_derivs_copy_uptr.get();

		if (method == 0) {
			// log_sig_derivs is sig-shaped with scalar stripped
			for (uint64_t b = 0; b < batch_size; ++b) {
				log_sig_derivs_copy[b * log_sig_len_] = static_cast<T>(0);
				std::memcpy(log_sig_derivs_copy + b * log_sig_len_ + 1,
					log_sig_derivs + b * lsd_stride, lsd_stride * sizeof(T));
			}
		} else {
			std::memcpy(log_sig_derivs_copy, log_sig_derivs, log_sig_len_ * batch_size * sizeof(T));
		}

		// sig input needs prepending per-element
		auto sig_full_uptr = std::make_unique<T[]>(sig_len_ * batch_size);
		T* sig_full = sig_full_uptr.get();
		for (uint64_t b = 0; b < batch_size; ++b) {
			sig_full[b * sig_len_] = static_cast<T>(1);
			std::memcpy(sig_full + b * sig_len_ + 1,
				sig + b * sig_in_stride, sig_in_stride * sizeof(T));
		}

		// Output buffer (full size), then strip
		auto out_full_uptr = std::make_unique<T[]>(sig_len_ * batch_size);
		T* out_full = out_full_uptr.get();

		auto log_sig_backprop_func = [&](const T* sig_ptr, T* log_sig_derivs_ptr, T* out_ptr) {
			get_sig_to_log_sig_backprop_<T>(sig_ptr, out_ptr, log_sig_derivs_ptr, aug_dimension, degree, method);
		};

		multi_threaded_batch(log_sig_backprop_func, batch_size, n_jobs,
			make_batch(sig_full, sig_len_),
			make_batch(log_sig_derivs_copy, log_sig_len_),
			make_batch(out_full, sig_len_));

		// Strip output
		for (uint64_t b = 0; b < batch_size; ++b) {
			std::memcpy(out + b * out_stride, out_full + b * sig_len_ + 1, out_stride * sizeof(T));
		}
	}
	return;
}

template<std::floating_point T>
void log_sig_combine_(
	const T* log_sig1, const T* log_sig2, T* out,
	uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t i = 0; i < m; ++i) out[i] = log_sig1[i] + log_sig2[i];
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();
	std::vector<T> memo(m2 * m);
	bch_combine_impl_<T>(log_sig1, log_sig2, out, cache, memo.data());
}

template<std::floating_point T>
void log_sig_combine_(
	const T* log_sig1, const T* log_sig2, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine received degree 0");

	// Resolve cache once before any parallel region (avoids data race on cache init)
	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t i = 0; i < batch_size; ++i) {
			for (uint64_t j = 0; j < m; ++j)
				out[i * m + j] = log_sig1[i * m + j] + log_sig2[i * m + j];
		}
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();

	auto func = [&](const T* ls1, const T* ls2, T* o) {
		thread_local std::vector<T> tl_memo;
		tl_memo.resize(m2 * m);
		bch_combine_impl_<T>(ls1, ls2, o, cache, tl_memo.data());
	};
	multi_threaded_batch(func, batch_size, n_jobs,
		make_batch(log_sig1, m), make_batch(log_sig2, m), make_batch(out, m));
}

// ========================================================================
// log_sig_join_: extend a log-signature by a displacement via BCH
// ========================================================================

template<std::floating_point T>
void log_sig_join_impl_(
	const T* RESTRICT log_sig, const T* RESTRICT displacement, T* RESTRICT out,
	const BchCache& cache, T* memo
) {
	uint64_t m = cache.m;
	uint64_t dim = cache.dimension;

	// Build the linear log-sig in memo[1]: zero-fill + copy dim entries
	std::fill(memo + m, memo + 2 * m, static_cast<T>(0));
	std::memcpy(memo + m, displacement, dim * sizeof(T));

	// Delegate to the sparse-aware BCH
	bch_combine_linear_impl_<T>(log_sig, memo + m, out, cache, memo);
}

template<std::floating_point T>
void log_sig_join_(
	const T* log_sig, const T* displacement, T* out,
	uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_join received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_join received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		std::memcpy(out, log_sig, m * sizeof(T));
		for (uint64_t i = 0; i < dimension; ++i) out[i] += displacement[i];
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();
	std::vector<T> memo(m2 * m);
	log_sig_join_impl_<T>(log_sig, displacement, out, cache, memo.data());
}

template<std::floating_point T>
void log_sig_join_(
	const T* log_sig, const T* displacement, T* out,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_join received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_join received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t b = 0; b < batch_size; ++b) {
			std::memcpy(out + b * m, log_sig + b * m, m * sizeof(T));
			for (uint64_t i = 0; i < dimension; ++i)
				out[b * m + i] += displacement[b * dimension + i];
		}
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();

	auto func = [&](const T* ls, const T* disp, T* o) {
		thread_local std::vector<T> tl_memo;
		tl_memo.resize(m2 * m);
		log_sig_join_impl_<T>(ls, disp, o, cache, tl_memo.data());
	};
	multi_threaded_batch(func, batch_size, n_jobs,
		make_batch(log_sig, m), make_batch(displacement, dimension), make_batch(out, m));
}

// ========================================================================
// log_sig_join_backprop_: backward pass through log_sig_join
// ========================================================================

template<std::floating_point T>
void log_sig_join_backprop_(
	const T* d_out, T* d_logsig, T* d_displacement,
	const T* log_sig, const T* displacement,
	uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_join_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_join_backprop received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		std::memcpy(d_logsig, d_out, m * sizeof(T));
		std::memcpy(d_displacement, d_out, dimension * sizeof(T));
		return;
	}

	// Construct the linear log-sig for the backprop
	auto linear_ls = std::make_unique<T[]>(m);
	std::fill(linear_ls.get(), linear_ls.get() + m, static_cast<T>(0));
	std::memcpy(linear_ls.get(), displacement, dimension * sizeof(T));

	// Reuse bch_combine_backprop_impl_ - it handles the full BCH backward
	auto d_ls2 = std::make_unique<T[]>(m);
	uint64_t m2 = cache.bch_coefficients.size();
	std::vector<T> workspace(2 * m2 * m);
	bch_combine_backprop_impl_<T>(d_out, d_logsig, d_ls2.get(), log_sig, linear_ls.get(),
		cache, workspace.data());

	// d_displacement = first dim elements of d_ls2
	std::memcpy(d_displacement, d_ls2.get(), dimension * sizeof(T));
}

template<std::floating_point T>
void log_sig_join_backprop_(
	const T* d_out, T* d_logsig, T* d_displacement,
	const T* log_sig, const T* displacement,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_join_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_join_backprop received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t b = 0; b < batch_size; ++b) {
			std::memcpy(d_logsig + b * m, d_out + b * m, m * sizeof(T));
			std::memcpy(d_displacement + b * dimension, d_out + b * m, dimension * sizeof(T));
		}
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();

	auto func = [&](uint64_t start, uint64_t end) {
		std::vector<T> workspace(2 * m2 * m);
		auto linear_ls = std::make_unique<T[]>(m);
		auto d_ls2 = std::make_unique<T[]>(m);
		for (uint64_t b = start; b < end; ++b) {
			std::fill(linear_ls.get(), linear_ls.get() + m, static_cast<T>(0));
			std::memcpy(linear_ls.get(), displacement + b * dimension, dimension * sizeof(T));
			bch_combine_backprop_impl_<T>(
				d_out + b * m, d_logsig + b * m, d_ls2.get(),
				log_sig + b * m, linear_ls.get(),
				cache, workspace.data());
			std::memcpy(d_displacement + b * dimension, d_ls2.get(), dimension * sizeof(T));
		}
	};

	spawn_batch_threads(batch_size, n_jobs, func);
}

// ========================================================================
// log_sig_from_path_: compute log-signature directly via sequential BCH
// ========================================================================

template<std::floating_point T>
void log_sig_from_path_(
	const T* path, T* out,
	uint64_t length, uint64_t dimension,
	const BchCache& cache, T* memo, T* seg, T* temp
) {
	uint64_t m = cache.m;

	const T* p0 = path;
	const T* p1 = path + dimension;
	for (uint64_t k = 0; k < dimension; ++k)
		out[k] = p1[k] - p0[k];
	for (uint64_t k = dimension; k < m; ++k)
		out[k] = T(0);

	if (length <= 2) return;

	std::memset(seg, 0, m * sizeof(T));

	T* acc = out;
	T* src = temp;
	for (uint64_t s = 1; s < length - 1; ++s) {
		const T* pa = path + s * dimension;
		const T* pb = path + (s + 1) * dimension;
		for (uint64_t k = 0; k < dimension; ++k)
			seg[k] = pb[k] - pa[k];

		std::swap(acc, src);
		bch_combine_linear_impl_<T>(src, seg, acc, cache, memo);
	}

	if (acc != out) std::memcpy(out, acc, m * sizeof(T));
}


// Shared across AVX2 and NEON - no platform-specific intrinsics needed.
#ifdef VEC
inline void log_sig_from_path_x4_(
	const double* paths[4], double* outs[4],
	uint64_t length, uint64_t dimension,
	const BchCache& cache, double* memo, double* seg, double* temp
) {
	uint64_t m = cache.m;

	for (uint64_t k = 0; k < dimension; ++k) {
		for (int b = 0; b < 4; ++b)
			temp[k * 4 + b] = paths[b][dimension + k] - paths[b][k];
	}
	std::memset(&temp[dimension * 4], 0, (m - dimension) * 4 * sizeof(double));

	if (length <= 2) {
		for (uint64_t k = 0; k < m; ++k)
			for (int b = 0; b < 4; ++b)
				outs[b][k] = temp[k * 4 + b];
		return;
	}

	std::memset(seg, 0, m * 4 * sizeof(double));

	double* acc = temp;
	double* src = memo + cache.bch_coefficients.size() * m * 4;

	for (uint64_t s = 1; s < length - 1; ++s) {
		for (uint64_t k = 0; k < dimension; ++k) {
			for (int b = 0; b < 4; ++b)
				seg[k * 4 + b] = paths[b][(s + 1) * dimension + k] - paths[b][s * dimension + k];
		}

		std::swap(acc, src);
		bch_combine_linear_impl_x4_(src, seg, acc, cache, memo);
	}

	for (uint64_t k = 0; k < m; ++k)
		for (int b = 0; b < 4; ++b)
			outs[b][k] = acc[k * 4 + b];
}
inline void log_sig_from_path_backprop_x4_(
	const double* d_outs[4], double* d_paths[4], const double* paths[4],
	uint64_t length, uint64_t dimension,
	const BchCache& cache, double* workspace
) {
	uint64_t m = cache.m;
	uint64_t m2 = cache.bch_coefficients.size();
	uint64_t n_segs = length - 1;

	// Workspace: curr[m*4] + prev[m*4] + seg[m*4] + neg_seg[m*4]
	//          + bch_ws[m2*m*4] + bch_bp_ws[2*m2*m*4] + d_acc[m*4] + d_ls1[m*4] + d_ls2[m*4]
	double* curr = workspace;
	double* prev = curr + m * 4;
	double* seg = prev + m * 4;
	double* neg_seg = seg + m * 4;
	double* bch_ws = neg_seg + m * 4;
	double* bch_bp_ws = bch_ws + m2 * m * 4;
	double* d_acc = bch_bp_ws + 2 * m2 * m * 4;
	double* d_ls1 = d_acc + m * 4;
	double* d_ls2 = d_ls1 + m * 4;

	// Forward into curr
	for (uint64_t k = 0; k < dimension; ++k)
		for (int b = 0; b < 4; ++b)
			curr[k * 4 + b] = paths[b][dimension + k] - paths[b][k];
	std::memset(&curr[dimension * 4], 0, (m - dimension) * 4 * sizeof(double));

	std::memset(seg, 0, m * 4 * sizeof(double));
	for (uint64_t s = 1; s < n_segs; ++s) {
		for (uint64_t k = 0; k < dimension; ++k)
			for (int b = 0; b < 4; ++b)
				seg[k * 4 + b] = paths[b][(s + 1) * dimension + k] - paths[b][s * dimension + k];

		bch_combine_linear_impl_x4_(curr, seg, prev, cache, bch_ws);
		std::swap(curr, prev);
	}

	// Init d_acc from d_out (interleaved)
	for (uint64_t k = 0; k < m; ++k)
		for (int b = 0; b < 4; ++b)
			d_acc[k * 4 + b] = d_outs[b][k];

	// Zero d_paths
	for (int b = 0; b < 4; ++b)
		std::memset(d_paths[b], 0, length * dimension * sizeof(double));

	// Backward with uncombination
	std::memset(neg_seg, 0, m * 4 * sizeof(double));
	for (uint64_t s = n_segs - 1; s >= 1; --s) {
		for (uint64_t k = 0; k < dimension; ++k) {
			for (int b = 0; b < 4; ++b) {
				double dx = paths[b][(s + 1) * dimension + k] - paths[b][s * dimension + k];
				seg[k * 4 + b] = dx;
				neg_seg[k * 4 + b] = -dx;
			}
		}

		// Uncombine: prev = BCH(curr, -seg)
		bch_combine_linear_impl_x4_(curr, neg_seg, prev, cache, bch_ws);

		// Backprop through BCH(prev, seg) -> curr
		bch_combine_linear_backprop_impl_x4_(d_acc, d_ls1, d_ls2, prev, seg, cache, bch_bp_ws);

		// Scatter d_ls2 to path gradients
		for (uint64_t k = 0; k < dimension; ++k) {
			for (int b = 0; b < 4; ++b) {
				d_paths[b][(s + 1) * dimension + k] += d_ls2[k * 4 + b];
				d_paths[b][s * dimension + k] -= d_ls2[k * 4 + b];
			}
		}

		std::memcpy(d_acc, d_ls1, m * 4 * sizeof(double));
		std::swap(curr, prev);
	}

	// Final step
	for (uint64_t k = 0; k < dimension; ++k) {
		for (int b = 0; b < 4; ++b) {
			d_paths[b][dimension + k] += d_acc[k * 4 + b];
			d_paths[b][k] -= d_acc[k * 4 + b];
		}
	}
}
#endif // VEC

template<std::floating_point T>
void batch_log_sig_from_path_(
	const T* path, T* out,
	uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_from_path received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_from_path received degree 0");
	if (length < 2) throw std::invalid_argument("log_sig_from_path received length < 2");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t i = 0; i < batch_size; ++i) {
			const T* first = path + i * length * dimension;
			const T* last = first + (length - 1) * dimension;
			for (uint64_t k = 0; k < m; ++k)
				out[i * m + k] = last[k] - first[k];
		}
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();
	uint64_t path_stride = length * dimension;
	uint64_t scalar_start = 0;

#ifdef VEC
	if constexpr (std::same_as<T, double>) {
		const uint64_t x4_batch_size = batch_size / 4;
		auto x4_func = [&](const double* p, double* o) {
			thread_local std::vector<double> tl_memo;
			thread_local std::vector<double> tl_seg;
			thread_local std::vector<double> tl_temp;
			tl_memo.resize((m2 + 1) * m * 4);
			tl_seg.resize(m * 4);
			tl_temp.resize(m * 4);

			const double* paths[4];
			double* outs[4];
			for (uint64_t b = 0; b < 4; ++b) {
				paths[b] = p + b * path_stride;
				outs[b] = o + b * m;
			}
			log_sig_from_path_x4_(paths, outs, length, dimension, cache,
				tl_memo.data(), tl_seg.data(), tl_temp.data());
		};
		multi_threaded_batch(x4_func, x4_batch_size, n_jobs,
			make_batch(path, path_stride * 4), make_batch(out, m * 4));
		scalar_start = x4_batch_size * 4;
	}
#endif

	auto func = [&](const T* p, T* o) {
		thread_local std::vector<T> tl_memo;
		thread_local std::vector<T> tl_seg;
		thread_local std::vector<T> tl_temp;
		tl_memo.resize(m2 * m);
		tl_seg.resize(m);
		tl_temp.resize(m);
		log_sig_from_path_<T>(p, o, length, dimension, cache, tl_memo.data(), tl_seg.data(), tl_temp.data());
	};
	multi_threaded_batch(func, batch_size - scalar_start, n_jobs,
		make_batch(path + scalar_start * path_stride, path_stride),
		make_batch(out + scalar_start * m, m));
}

// ========================================================================
// log_sig_from_path_backprop_: backward pass through sequential BCH chain
// ========================================================================

// Backward pass using BCH uncombination: recovers each intermediate on the fly
// via BCH(curr, -seg) = prev, avoiding O(N*m) storage of all intermediates.
// Uses O(m) memory instead, same total compute.
template<std::floating_point T>
void log_sig_from_path_backprop_(
	const T* d_out, T* d_path,
	const T* path,
	uint64_t length, uint64_t dimension,
	const BchCache& cache, T* workspace
) {
	uint64_t m = cache.m;
	uint64_t m2 = cache.bch_coefficients.size();
	uint64_t n_segs = length - 1;

	// Workspace layout:
	// curr: m (current accumulator, recovered via uncombination)
	// prev: m (previous accumulator, recovered via BCH(curr, -seg))
	// seg: m (segment log-sig buffer)
	// neg_seg: m (negated segment for uncombination)
	// bch_ws: m2 * m (BCH forward memo, shared between combine and backprop)
	// bch_bp_ws: 2 * m2 * m (backprop workspace: memo + d_memo)
	// d_acc: m (gradient flowing backward)
	// d_ls1: m, d_ls2: m
	T* curr = workspace;
	T* prev = curr + m;
	T* seg = prev + m;
	T* neg_seg = seg + m;
	T* bch_ws = neg_seg + m;
	T* bch_bp_ws = bch_ws + m2 * m;
	T* d_acc = bch_bp_ws + 2 * m2 * m;
	T* d_ls1 = d_acc + m;
	T* d_ls2 = d_ls1 + m;

	// Recompute the forward output into curr
	const T* p0 = path;
	const T* p1 = path + dimension;
	for (uint64_t k = 0; k < dimension; ++k)
		curr[k] = p1[k] - p0[k];
	for (uint64_t k = dimension; k < m; ++k)
		curr[k] = T(0);

	std::memset(seg, 0, m * sizeof(T));
	for (uint64_t s = 1; s < n_segs; ++s) {
		const T* pa = path + s * dimension;
		const T* pb = path + (s + 1) * dimension;
		for (uint64_t k = 0; k < dimension; ++k)
			seg[k] = pb[k] - pa[k];

		// curr = BCH(curr, seg) - reuse prev as temp output, then swap
		bch_combine_impl_<T>(curr, seg, prev, cache, bch_ws);
		std::swap(curr, prev);
	}

	// curr now holds the final forward output (= intermediates[n_segs-1])
	std::memcpy(d_acc, d_out, m * sizeof(T));
	std::memset(d_path, 0, length * dimension * sizeof(T));

	// Backward: reverse through segments, recovering prev via uncombination
	std::memset(neg_seg, 0, m * sizeof(T));
	for (uint64_t s = n_segs - 1; s >= 1; --s) {
		const T* pa = path + s * dimension;
		const T* pb = path + (s + 1) * dimension;
		for (uint64_t k = 0; k < dimension; ++k) {
			seg[k] = pb[k] - pa[k];
			neg_seg[k] = -(pb[k] - pa[k]);
		}

		// Recover prev = BCH(curr, -seg) - the "uncombine" step
		bch_combine_impl_<T>(curr, neg_seg, prev, cache, bch_ws);

		// Backprop through BCH(prev, seg) -> curr
		bch_combine_backprop_impl_<T>(d_acc, d_ls1, d_ls2, prev, seg, cache, bch_bp_ws);

		for (uint64_t k = 0; k < dimension; ++k) {
			d_path[(s + 1) * dimension + k] += d_ls2[k];
			d_path[s * dimension + k] -= d_ls2[k];
		}

		std::memcpy(d_acc, d_ls1, m * sizeof(T));
		std::swap(curr, prev); // curr = prev for next iteration
	}

	for (uint64_t k = 0; k < dimension; ++k) {
		d_path[dimension + k] += d_acc[k];
		d_path[k] -= d_acc[k];
	}
}

template<std::floating_point T>
void batch_log_sig_from_path_backprop_(
	const T* d_out, T* d_path,
	const T* path,
	uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_from_path_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_from_path_backprop received degree 0");
	if (length < 2) throw std::invalid_argument("log_sig_from_path_backprop received length < 2");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		// Forward was: out = path[last] - path[first]. Gradient: d_path[last] += d_out, d_path[first] -= d_out
		std::memset(d_path, 0, batch_size * length * dimension * sizeof(T));
		for (uint64_t i = 0; i < batch_size; ++i) {
			for (uint64_t k = 0; k < m; ++k) {
				d_path[i * length * dimension + (length - 1) * dimension + k] += d_out[i * m + k];
				d_path[i * length * dimension + k] -= d_out[i * m + k];
			}
		}
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();
	uint64_t path_stride = length * dimension;
	// Workspace: 4*m (curr, prev, seg, neg_seg) + 3*m2*m (bch_ws + bch_bp_ws) + 3*m (d_acc, d_ls1, d_ls2)
	uint64_t ws_size = 7 * m + 3 * m2 * m;
	uint64_t scalar_start = 0;

#ifdef VEC
	if constexpr (std::same_as<T, double>) {
		const uint64_t x4_batch_size = batch_size / 4;
		auto x4_func = [&](const double* dout, double* dp, const double* p) {
			thread_local std::vector<double> tl_ws;
			tl_ws.resize(ws_size * 4);

			const double* douts[4];
			double* dpaths[4];
			const double* paths[4];
			for (uint64_t b = 0; b < 4; ++b) {
				douts[b] = dout + b * m;
				dpaths[b] = dp + b * path_stride;
				paths[b] = p + b * path_stride;
			}
			log_sig_from_path_backprop_x4_(douts, dpaths, paths,
				length, dimension, cache, tl_ws.data());
		};
		multi_threaded_batch(x4_func, x4_batch_size, n_jobs,
			make_batch(d_out, m * 4), make_batch(d_path, path_stride * 4),
			make_batch(path, path_stride * 4));
		scalar_start = x4_batch_size * 4;
	}
#endif

	auto func = [&](const T* dout, T* dp, const T* p) {
		thread_local std::vector<T> tl_ws;
		tl_ws.resize(ws_size);
		log_sig_from_path_backprop_<T>(dout, dp, p, length, dimension, cache, tl_ws.data());
	};
	multi_threaded_batch(func, batch_size - scalar_start, n_jobs,
		make_batch(d_out + scalar_start * m, m),
		make_batch(d_path + scalar_start * path_stride, path_stride),
		make_batch(path + scalar_start * path_stride, path_stride));
}


template<std::floating_point T>
void log_sig_combine_backprop_(
	const T* d_out, T* d_ls1, T* d_ls2,
	const T* ls1, const T* ls2,
	uint64_t dimension, uint64_t degree
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine_backprop received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		std::memcpy(d_ls1, d_out, m * sizeof(T));
		std::memcpy(d_ls2, d_out, m * sizeof(T));
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();
	std::vector<T> workspace(2 * m2 * m);
	bch_combine_backprop_impl_<T>(d_out, d_ls1, d_ls2, ls1, ls2, cache, workspace.data());
}

template<std::floating_point T>
void log_sig_combine_backprop_(
	const T* d_out, T* d_ls1, T* d_ls2,
	const T* ls1, const T* ls2,
	uint64_t batch_size, uint64_t dimension, uint64_t degree,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("log_sig_combine_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("log_sig_combine_backprop received degree 0");

	const BchCache& cache = get_bch_cache(dimension, degree);
	uint64_t m = cache.m;

	if (degree < 2) {
		for (uint64_t i = 0; i < batch_size * m; ++i) {
			d_ls1[i] = d_out[i];
			d_ls2[i] = d_out[i];
		}
		return;
	}

	uint64_t m2 = cache.bch_coefficients.size();

	auto func = [&](const T* dout, T* dls1, T* dls2, const T* l1, const T* l2) {
		thread_local std::vector<T> tl_workspace;
		tl_workspace.resize(2 * m2 * m);
		bch_combine_backprop_impl_<T>(dout, dls1, dls2, l1, l2, cache, tl_workspace.data());
	};
	multi_threaded_batch(func, batch_size, n_jobs,
		make_batch(d_out, m), make_batch(d_ls1, m), make_batch(d_ls2, m), make_batch(ls1, m), make_batch(ls2, m));
}
