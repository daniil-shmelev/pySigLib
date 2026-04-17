/* Copyright 2025 Daniil Shmelev
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
#include "cp_sig_combine.h"

// ---------------------------------------------------------------------------
// linear_sig_: signature of a single linear segment from a displacement vector
// out[0] = 1, out[level k] = dx^{⊗k} / k!
// ---------------------------------------------------------------------------

template<std::floating_point T>
FORCE_INLINE void linear_sig_with_level_index_(
	const T* displacement,
	T* out,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index,
	bool scalar_term = true
) {
	if (scalar_term) out[0] = static_cast<T>(1);
	if (degree == 0) return;
	std::memcpy(out + 1, displacement, dimension * sizeof(T));

	for (uint64_t level = 2; level <= degree; ++level) {
		T one_over_level = static_cast<T>(1.) / level;
		T* result_ptr = out + level_index[level];
		const T* const left_end = out + level_index[level];
#ifdef VEC
		for (const T* left_ptr = out + level_index[level - 1]; left_ptr != left_end; ++left_ptr, result_ptr += dimension) {
			vec_mult_assign(result_ptr, out + 1, (*left_ptr) * one_over_level, dimension);
		}
#else
		for (const T* left_ptr = out + level_index[level - 1]; left_ptr != left_end; ++left_ptr) {
			T val = (*left_ptr) * one_over_level;
			for (uint64_t d = 0; d < dimension; ++d)
				*(result_ptr++) = val * out[1 + d];
		}
#endif
	}
}

template<std::floating_point T>
void linear_sig_(
	const T* displacement,
	T* out,
	uint64_t dimension,
	uint64_t degree,
	bool scalar_term = true
) {
	if (dimension == 0) { throw std::invalid_argument("linear_sig received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	const uint64_t full_len = level_index[degree + 1];
	if (scalar_term) {
		linear_sig_with_level_index_(displacement, out, dimension, degree, level_index);
	} else {
		std::vector<T> buf(full_len);
		linear_sig_with_level_index_(displacement, buf.data(), dimension, degree, level_index);
		std::memcpy(out, buf.data() + 1, (full_len - 1) * sizeof(T));
	}
}

template<std::floating_point T>
void batch_linear_sig_(
	const T* displacement,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool scalar_term = true,
	int n_jobs = 1
) {
	if (dimension == 0) { throw std::invalid_argument("linear_sig received dimension 0"); }

	const uint64_t full_len = ::sig_length(dimension, degree);
	const uint64_t stride = scalar_term ? full_len : full_len - 1;

	auto func = [&](const T* in_ptr, T* out_ptr) {
		linear_sig_<T>(in_ptr, out_ptr, dimension, degree, scalar_term);
	};

	multi_threaded_batch(func, displacement, out, batch_size, dimension, stride, n_jobs);
}

// ---------------------------------------------------------------------------
// sig_join_: extend a signature by a displacement
// Computes sig_combine(sig, linear_sig(displacement)) in a single call.
// ---------------------------------------------------------------------------

template<std::floating_point T>
void sig_join_(
	const T* sig,
	const T* displacement,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool prepend = false,
	bool scalar_term = true,
	int n_jobs = 1
) {
	if (dimension == 0) { throw std::invalid_argument("sig_join received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	const uint64_t full_len = level_index[degree + 1];
	const uint64_t sig_stride = scalar_term ? full_len : full_len - 1;

	if (scalar_term) {
		auto func = [&](const T* sig_ptr, const T* disp_ptr, T* out_ptr) {
			auto lsig_uptr = std::make_unique<T[]>(full_len);
			T* lsig = lsig_uptr.get();
			linear_sig_with_level_index_(disp_ptr, lsig, dimension, degree, level_index);
			if (prepend) {
				std::memcpy(out_ptr, lsig, full_len * sizeof(T));
				sig_combine_inplace_(out_ptr, sig_ptr, degree, level_index);
			} else {
				std::memcpy(out_ptr, sig_ptr, full_len * sizeof(T));
				sig_combine_inplace_(out_ptr, lsig, degree, level_index);
			}
		};
		multi_threaded_batch_2<const T, const T, T>(func, sig, displacement, out, batch_size, full_len, dimension, full_len, n_jobs);
	} else {
		// Per-element copy: reconstruct full buffers, compute, strip
		auto func = [&](const T* sig_ptr, const T* disp_ptr, T* out_ptr) {
			std::vector<T> sig_full(full_len), lsig(full_len), out_full(full_len);
			sig_full[0] = static_cast<T>(1);
			std::memcpy(sig_full.data() + 1, sig_ptr, (full_len - 1) * sizeof(T));
			linear_sig_with_level_index_(disp_ptr, lsig.data(), dimension, degree, level_index);
			if (prepend) {
				std::memcpy(out_full.data(), lsig.data(), full_len * sizeof(T));
				sig_combine_inplace_(out_full.data(), sig_full.data(), degree, level_index);
			} else {
				std::memcpy(out_full.data(), sig_full.data(), full_len * sizeof(T));
				sig_combine_inplace_(out_full.data(), lsig.data(), degree, level_index);
			}
			std::memcpy(out_ptr, out_full.data() + 1, (full_len - 1) * sizeof(T));
		};
		multi_threaded_batch_2<const T, const T, T>(func, sig, displacement, out, batch_size, sig_stride, dimension, sig_stride, n_jobs);
	}
}

// ---------------------------------------------------------------------------
// sig_join_backprop_: backward pass through sig_join
// dF/dsig and dF/ddisplacement given dF/dout
// ---------------------------------------------------------------------------

template<std::floating_point T>
void sig_join_backprop_(
	const T* d_out,
	T* d_sig,
	T* d_displacement,
	const T* sig,
	const T* displacement,
	uint64_t dimension,
	uint64_t degree,
	bool prepend = false,
	bool scalar_term = true
) {
	if (dimension == 0) { throw std::invalid_argument("sig_join_backprop received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	const uint64_t full_len = level_index[degree + 1];
	const uint64_t in_stride = scalar_term ? full_len : full_len - 1;

	// Recompute linear_sig (always full)
	auto lsig_uptr = std::make_unique<T[]>(full_len);
	T* lsig = lsig_uptr.get();
	linear_sig_<T>(displacement, lsig, dimension, degree, /*scalar_term=*/true);

	// Full-size buffers for the inner backprop
	auto d_out_full_uptr = std::make_unique<T[]>(full_len);
	auto d_sig_full_uptr = std::make_unique<T[]>(full_len);
	auto sig_full_uptr = std::make_unique<T[]>(full_len);
	T* d_out_full = d_out_full_uptr.get();
	T* d_sig_full = d_sig_full_uptr.get();
	T* sig_full = sig_full_uptr.get();

	if (scalar_term) {
		std::memcpy(d_out_full, d_out, full_len * sizeof(T));
		std::memcpy(sig_full, sig, full_len * sizeof(T));
	} else {
		d_out_full[0] = static_cast<T>(0);
		std::memcpy(d_out_full + 1, d_out, (full_len - 1) * sizeof(T));
		sig_full[0] = static_cast<T>(1);
		std::memcpy(sig_full + 1, sig, (full_len - 1) * sizeof(T));
	}

	auto d_lsig_uptr = std::make_unique<T[]>(full_len);
	T* d_lsig = d_lsig_uptr.get();

	if (prepend) {
		std::memcpy(d_lsig, d_out_full, full_len * sizeof(T));
		uncombine_sig_deriv(lsig, sig_full, d_lsig, d_sig_full, dimension, degree, level_index);
	} else {
		std::memcpy(d_sig_full, d_out_full, full_len * sizeof(T));
		uncombine_sig_deriv(sig_full, lsig, d_sig_full, d_lsig, dimension, degree, level_index);
	}
	d_sig_full[0] = static_cast<T>(0);
	d_lsig[0] = static_cast<T>(0);

	linear_sig_deriv_to_increment_deriv(lsig, d_lsig, dimension, degree, level_index);
	std::memcpy(d_displacement, d_lsig + 1, dimension * sizeof(T));

	// Copy d_sig to caller's buffer
	if (scalar_term) {
		std::memcpy(d_sig, d_sig_full, full_len * sizeof(T));
	} else {
		std::memcpy(d_sig, d_sig_full + 1, (full_len - 1) * sizeof(T));
	}
}

template<std::floating_point T>
void batch_sig_join_backprop_(
	const T* d_out,
	T* d_sig,
	T* d_displacement,
	const T* sig,
	const T* displacement,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool prepend = false,
	bool scalar_term = true,
	int n_jobs = 1
) {
	if (dimension == 0) { throw std::invalid_argument("sig_join_backprop received dimension 0"); }

	const uint64_t full_len = ::sig_length(dimension, degree);
	const uint64_t sig_stride = scalar_term ? full_len : full_len - 1;

	auto func = [&](uint64_t start, uint64_t end) {
		for (uint64_t i = start; i < end; ++i) {
			sig_join_backprop_<T>(
				d_out + i * sig_stride,
				d_sig + i * sig_stride,
				d_displacement + i * dimension,
				sig + i * sig_stride,
				displacement + i * dimension,
				dimension, degree, prepend, scalar_term
			);
		}
	};

	if (n_jobs == 0) throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : static_cast<int>(get_max_threads()) + 1 + n_jobs;
	const uint64_t num_threads = std::min(static_cast<uint64_t>(std::max(max_threads, 1)), batch_size);

	if (num_threads > 1) {
		std::vector<std::thread> workers;
		const uint64_t chunk = batch_size / num_threads;
		const uint64_t remainder = batch_size % num_threads;
		uint64_t start = 0;
		for (uint64_t t = 0; t < num_threads; ++t) {
			uint64_t end = start + chunk + (t < remainder ? 1 : 0);
			workers.emplace_back(func, start, end);
			start = end;
		}
		for (auto& w : workers) w.join();
	}
	else {
		func(0, batch_size);
	}
}

