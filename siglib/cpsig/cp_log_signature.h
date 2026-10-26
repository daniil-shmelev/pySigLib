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

#include "cp_simd_config.h"
#include "hwy/highway.h"
#if defined(CPSIG_CP_LOG_SIGNATURE_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_CP_LOG_SIGNATURE_H_
#undef CPSIG_CP_LOG_SIGNATURE_H_
#else
#define CPSIG_CP_LOG_SIGNATURE_H_
#endif
#include "cppch.h"

#include "multithreading.h"
#include "cp_tensor_log.h"

#include "cp_path.h"
#include "cp_bch.h"
#include "log_sig_method.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template<std::floating_point T>
void get_log_sig_(
	const T* sig,
	T* out,
	uint64_t dimension,
	uint64_t degree,
	LogSigMethod method = LogSigMethod::Expanded
)
{
	switch (method) {
	case LogSigMethod::Expanded:
		log_sig_expanded<T>(sig, out, dimension, degree);
		break;
	case LogSigMethod::LyndonWords:
		log_sig_lyndon_words<T>(sig, out, dimension, degree);
		break;
	case LogSigMethod::LyndonBasis:
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
	LogSigMethod method = LogSigMethod::Expanded,
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
	const uint64_t full_out_len = method == LogSigMethod::Expanded ? sig_len : logsig_len;
	const uint64_t in_stride = scalar_term ? sig_len : sig_len - 1;
	const uint64_t out_stride = (method == LogSigMethod::Expanded && !scalar_term) ? (sig_len - 1) : full_out_len;

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
			if (method == LogSigMethod::Expanded) {
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
	LogSigMethod method = LogSigMethod::Expanded
) {
	switch (method) {
	case LogSigMethod::Expanded:
		tensor_log_backprop_<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	case LogSigMethod::LyndonWords:
		tensor_log_backprop_lyndon_words<T>(out, log_sig_derivs, sig, dimension, degree);
		break;
	case LogSigMethod::LyndonBasis:
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
	LogSigMethod method = LogSigMethod::Expanded,
	bool scalar_term = true,
	int n_jobs = 1
) {
	if (dimension == 0) { throw std::invalid_argument("sig_backprop received path of dimension 0"); }

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);

	const uint64_t sig_len_ = ::sig_length(aug_dimension, degree);
	const uint64_t log_sig_len_ = method == LogSigMethod::Expanded ? sig_len_ : ::log_sig_length(aug_dimension, degree);

	// Determine caller strides
	const uint64_t sig_in_stride = scalar_term ? sig_len_ : sig_len_ - 1;
	// log_sig_derivs: logsig-shaped (method>0, unaffected) or sig-shaped (method==0, may be stripped)
	const uint64_t lsd_stride = (method == LogSigMethod::Expanded && !scalar_term) ? (sig_len_ - 1) : log_sig_len_;
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

		if (method == LogSigMethod::Expanded) {
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

	batch_bch_combine_(log_sig1, log_sig2, out, batch_size, cache.general_schedule, m, n_jobs);
}

// Reduce arbitrary log-signature sequences; the same schedule computes prefixes.
template<std::floating_point T>
void log_sig_reduce_(const T* inputs, T* out, uint64_t batch, uint64_t length,
	uint64_t dimension, uint64_t degree, int n_jobs) {
	if (!dimension || !degree) throw std::invalid_argument("invalid BCH shape");
	const auto& cache = get_bch_cache(dimension, degree);
	const uint64_t m = cache.m;
	auto work = [&](uint64_t begin, uint64_t end) {
		thread_local std::vector<T> scratch;
		scratch.assign(cache.general_schedule.bch_memo_size * bch_tile, T(0));
		for (uint64_t b = begin; b < end; ++b) {
			T* result = out + b * m;
			std::fill_n(result, m, T(0));
			for (uint64_t start = 0; start < length; start += bch_tile)
				bch_combine_batch_<T, true>(nullptr, inputs + b * length * m + start,
					result, cache.general_schedule, scratch.data(), std::min(bch_tile, length - start), length);
		}
	};
	if (n_jobs == 1 || batch < 2) work(0, batch);
	else spawn_batch_threads(batch, n_jobs, work);
}

// ========================================================================
// log_sig_join_: extend a log-signature by a displacement via BCH
// ========================================================================

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

	batch_bch_combine_(log_sig, displacement, out, batch_size, cache.linear_schedule, dimension, n_jobs);
}

// ========================================================================
// log_sig_join_backprop_: backward pass through log_sig_join
// ========================================================================

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

	batch_bch_backprop_(d_out, d_logsig, d_displacement, log_sig, displacement,
		batch_size, cache.linear_schedule, dimension, n_jobs);
}

// ========================================================================
// batch_log_sig_from_path_: fused BCH scan of path increments
// ========================================================================

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

	const auto& schedule = cache.linear_schedule;
	auto work = [&](uint64_t begin, uint64_t end) {
		thread_local std::vector<T> scratch;
		scratch.assign((schedule.bch_memo_size + m) * bch_tile, T(0));
		T* increments = scratch.data() + schedule.bch_memo_size * bch_tile;
		for (uint64_t b = begin; b < end; ++b) {
			T* result = out + b * m;
			std::fill_n(result, m, T(0));
			for (uint64_t start = 0; start < length - 1; start += bch_tile) {
				const uint64_t count = std::min(bch_tile, length - 1 - start);
				for (uint64_t k = 0; k < dimension; ++k)
					for (uint64_t lane = 0; lane < count; ++lane) {
						const uint64_t i = (b * length + start + lane) * dimension + k;
						increments[k * bch_tile + lane] = path[i + dimension] - path[i];
					}
				bch_combine_batch_<T, true>(nullptr, increments, result, schedule,
					scratch.data(), count, bch_tile);
			}
		}
	};
	if (n_jobs == 1 || batch_size < 2) work(0, batch_size);
	else spawn_batch_threads(batch_size, n_jobs, work);
}

// ========================================================================
// batch_log_sig_from_path_backprop_: pull back the fused BCH scan
// ========================================================================

// Reconstruct tile starts via BCH inversion; storage is independent of path length.
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

	auto lift = [&](const T* increment, T* out, uint64_t stride) {
		for (uint64_t k = 0; k < dimension; ++k) out[k * stride] = increment[k];
	};
	auto pullback = [&](const T* gradient, const T*, T* out, uint64_t stride) {
		for (uint64_t k = 0; k < dimension; ++k) out[k] = gradient[k * stride];
	};
	auto work = [&](uint64_t begin, uint64_t end) {
		for (uint64_t b = begin; b < end; ++b)
			bch_path_backprop_(d_out + b * m, d_path + b * length * dimension,
				path + b * length * dimension, length, dimension, cache.linear_schedule,
				dimension, lift, pullback);
	};
	if (n_jobs == 1 || batch_size < 2) work(0, batch_size);
	else spawn_batch_threads(batch_size, n_jobs, work);
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

	batch_bch_backprop_(d_out, d_ls1, d_ls2, ls1, ls2,
		batch_size, cache.general_schedule, m, n_jobs);
}

}  // namespace HWY_NAMESPACE
}  // namespace cpsig
HWY_AFTER_NAMESPACE();
#endif
