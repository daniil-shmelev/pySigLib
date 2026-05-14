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
#include "words.h"
#include "log_sig_cache.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

// ---------------------------------------------------------------------------
// tensor_exp_: truncated tensor exponential via power series
//
//   exp(x) = 1 + P_1 + P_2 + ... + P_N
//   P_1 = x, P_n = x \otimes P_{n-1} / n
//   P_n has min level n -> level-skipping reduces work for large n.
// ---------------------------------------------------------------------------

template<std::floating_point T>
void tensor_exp_(
	const T* log_sig,
	T* out,
	uint64_t dimension,
	uint64_t degree
) {
	const uint64_t sig_len = ::sig_length(dimension, degree);

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	out[0] = static_cast<T>(1.);
	if (degree == 0) return;
	std::memcpy(out + level_index[1], log_sig + level_index[1],
		(level_index[degree + 1] - level_index[1]) * sizeof(T));

	if (degree <= 1) return;

	auto buff1_uptr = std::make_unique<T[]>(sig_len);
	auto buff2_uptr = std::make_unique<T[]>(sig_len);
	T* P_prev = buff1_uptr.get();
	T* P_curr = buff2_uptr.get();

	std::memcpy(P_prev, log_sig, sig_len * sizeof(T));

	for (uint64_t n = 2; n <= degree; ++n) {
		T inv_n = static_cast<T>(1.) / static_cast<T>(n);

		for (uint64_t target_level = n; target_level <= degree; ++target_level) {
			std::fill(P_curr + level_index[target_level],
				P_curr + level_index[target_level + 1], static_cast<T>(0.));

			// l1 ranges from 1 to target_level-(n-1), so l2 >= n-1 (P_prev's support)
			const uint64_t max_left = target_level - (n - 1);

			for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
				uint64_t right_level = target_level - left_level;

				T* res_ptr = P_curr + level_index[target_level];
				const T* const left_ptr_end = log_sig + level_index[left_level + 1];
#ifdef VEC
				const uint64_t right_level_size = level_index[right_level + 1] - level_index[right_level];
				const T* right_start = P_prev + level_index[right_level];
				for (const T* left_ptr = log_sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr) {
					vec_mult_add(res_ptr, right_start, *left_ptr * inv_n, right_level_size);
					res_ptr += right_level_size;
				}
#else
				const T* const right_ptr_end = P_prev + level_index[right_level + 1];
				for (const T* left_ptr = log_sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr) {
					T val = *left_ptr * inv_n;
					for (const T* right_ptr = P_prev + level_index[right_level]; right_ptr < right_ptr_end; ++right_ptr) {
						*(res_ptr++) += val * *right_ptr;
					}
				}
#endif
			}

			// Fuse accumulation into per-level loop (keeps data in cache)
			for (uint64_t i = level_index[target_level]; i < level_index[target_level + 1]; ++i)
				out[i] += P_curr[i];
		}

		std::swap(P_prev, P_curr);
	}
}

// ---------------------------------------------------------------------------
// tensor_exp_backprop_: backward pass through tensor_exp_
// Recomputes P_1..P_N, then backprops from n=degree to 2.
// ---------------------------------------------------------------------------

template<std::floating_point T>
void tensor_exp_backprop_(
	T* d_logsig,
	const T* d_sig,
	const T* log_sig,
	uint64_t dimension,
	uint64_t degree
) {
	const uint64_t sig_len = ::sig_length(dimension, degree);

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	std::fill(d_logsig, d_logsig + sig_len, static_cast<T>(0.));

	if (degree <= 1) {
		for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i)
			d_logsig[i] = d_sig[i];
		return;
	}

	// Recompute and store P_1..P_N (full sig_len per P for simple indexing)
	auto P_all_uptr = std::make_unique<T[]>(sig_len * degree);
	T* P_all = P_all_uptr.get();
	std::fill(P_all + sig_len, P_all + sig_len * degree, static_cast<T>(0.));

	std::memcpy(P_all, log_sig, sig_len * sizeof(T));

	for (uint64_t n = 2; n <= degree; ++n) {
		T inv_n = static_cast<T>(1.) / static_cast<T>(n);
		T* P_curr = P_all + (n - 1) * sig_len;
		const T* P_prev = P_all + (n - 2) * sig_len;

		for (uint64_t target_level = n; target_level <= degree; ++target_level) {
			// l1 ranges from 1 to target_level-(n-1), so l2 >= n-1 (P_prev's support)
			const uint64_t max_left = target_level - (n - 1);
			for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
				uint64_t right_level = target_level - left_level;
				T* res_ptr = P_curr + level_index[target_level];
				const T* const left_end = log_sig + level_index[left_level + 1];
#ifdef VEC
				const uint64_t right_level_size = level_index[right_level + 1] - level_index[right_level];
				const T* right_start = P_prev + level_index[right_level];
				for (const T* lp = log_sig + level_index[left_level]; lp < left_end; ++lp) {
					vec_mult_add(res_ptr, right_start, *lp * inv_n, right_level_size);
					res_ptr += right_level_size;
				}
#else
				const T* const right_end = P_prev + level_index[right_level + 1];
				for (const T* lp = log_sig + level_index[left_level]; lp < left_end; ++lp) {
					T val = *lp * inv_n;
					for (const T* rp = P_prev + level_index[right_level]; rp < right_end; ++rp)
						*(res_ptr++) += val * *rp;
				}
#endif
			}
		}
	}

	for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i)
		d_logsig[i] = d_sig[i];

	auto dP1_uptr = std::make_unique<T[]>(sig_len);
	auto dP2_uptr = std::make_unique<T[]>(sig_len);
	T* dP = dP1_uptr.get();
	T* dP_next = dP2_uptr.get();
	std::fill(dP, dP + sig_len, static_cast<T>(0.));

#ifdef VEC
	// Scratch buffer for precomputed upstream values
	const uint64_t max_right_size = level_index[degree] - level_index[degree - 1];
	auto upstream_uptr = std::make_unique<T[]>(max_right_size);
	T* upstream_buf = upstream_uptr.get();
#endif

	for (int64_t n = static_cast<int64_t>(degree); n >= 2; --n) {
		T inv_n = static_cast<T>(1.) / static_cast<T>(n);
		const T* P_prev = P_all + (n - 2) * sig_len;

		std::fill(dP_next, dP_next + sig_len, static_cast<T>(0.));

		for (uint64_t target_level = static_cast<uint64_t>(n); target_level <= degree; ++target_level) {
			const uint64_t max_left = target_level - (n - 1);
			for (uint64_t left_level = 1; left_level <= max_left; ++left_level) {
				uint64_t right_level = target_level - left_level;

				const T* grad_ptr = d_sig + level_index[target_level];
				const T* dP_ptr_base = dP + level_index[target_level];
				T* d_left = d_logsig + level_index[left_level];
				const T* lp_start = log_sig + level_index[left_level];
				const T* const lp_end = log_sig + level_index[left_level + 1];
				T* d_right = dP_next + level_index[right_level];
				const T* rp_start = P_prev + level_index[right_level];
				const T* const rp_end = P_prev + level_index[right_level + 1];

#ifdef VEC
				const uint64_t right_size = rp_end - rp_start;
				const T* gp = grad_ptr;
				const T* dp = dP_ptr_base;
				for (const T* lp = lp_start; lp < lp_end; ++lp) {
					vec_add_scaled(upstream_buf, gp, dp, inv_n, right_size);
					gp += right_size;
					dp += right_size;
					*(d_left++) += dot_product(upstream_buf, rp_start, right_size);
					vec_mult_add(d_right, upstream_buf, *lp, right_size);
				}
#else
				const T* gp = grad_ptr;
				const T* dp = dP_ptr_base;
				for (const T* lp = lp_start; lp < lp_end; ++lp) {
					T d_left_acc = static_cast<T>(0.);
					T* drp = d_right;
					for (const T* rp = rp_start; rp < rp_end; ++rp) {
						T upstream = (*(gp++) + *(dp++)) * inv_n;
						d_left_acc += upstream * *rp;
						*(drp++) += upstream * *lp;
					}
					*(d_left++) += d_left_acc;
				}
#endif
			}
		}

		std::swap(dP, dP_next);
	}

	// dP holds chain gradient from P_2..P_N flowing back to P_1 = x
	for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i)
		d_logsig[i] += dP[i];
}

// ---------------------------------------------------------------------------
// Method dispatch: expand from compressed basis, apply tensor_exp_
// ---------------------------------------------------------------------------

template<std::floating_point T>
void sig_from_logsig_expanded(
	const T* log_sig,
	T* out,
	uint64_t dimension,
	uint64_t degree
) {
	tensor_exp_<T>(log_sig, out, dimension, degree);
}

// ---------------------------------------------------------------------------
// Build the bracket expansion matrix E[m x sig_len] for all Lyndon words.
// E[i] is the tensor algebra expansion of the i-th Lyndon bracket.
// ---------------------------------------------------------------------------

template<std::floating_point T>
std::unique_ptr<T[]> build_bracket_expansions_(
	uint64_t dimension, uint64_t degree
) {
	const BasisCache& cache = get_basis_cache(dimension, degree, 2);
	const uint64_t sig_len = ::sig_length(dimension, degree);
	const uint64_t m = cache.lyndon_idx.size();

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	auto lyndon_words = all_lyndon_words(dimension, degree);
	std::unordered_set<word, WordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());
	std::unordered_map<word, uint64_t, WordHash> word_idx;
	for (uint64_t i = 0; i < m; ++i)
		word_idx[lyndon_words[i]] = i;

	auto expansions = std::make_unique<T[]>(m * sig_len);

	for (uint64_t i = 0; i < m; ++i) {
		T* exp_i = expansions.get() + i * sig_len;
		std::fill(exp_i, exp_i + sig_len, static_cast<T>(0.));

		if (lyndon_words[i].size() == 1) {
			exp_i[cache.lyndon_idx[i]] = static_cast<T>(1.);
		}
		else {
			auto [u, v] = standard_factorization(lyndon_words[i], lyndon_set);
			const T* exp_u = expansions.get() + word_idx.at(u) * sig_len;
			const T* exp_v = expansions.get() + word_idx.at(v) * sig_len;

			// exp_i = u \otimes v - v \otimes u (Lie bracket in tensor algebra)
			for (uint64_t tl = 2; tl <= degree; ++tl) {
				for (uint64_t l1 = 1; l1 < tl; ++l1) {
					uint64_t l2 = tl - l1;
					T* r = exp_i + level_index[tl];
					for (const T* lu = exp_u + level_index[l1]; lu < exp_u + level_index[l1 + 1]; ++lu)
						for (const T* rv = exp_v + level_index[l2]; rv < exp_v + level_index[l2 + 1]; ++rv)
							*(r++) += *lu * *rv;
					r = exp_i + level_index[tl];
					for (const T* lv = exp_v + level_index[l1]; lv < exp_v + level_index[l1 + 1]; ++lv)
						for (const T* ru = exp_u + level_index[l2]; ru < exp_u + level_index[l2 + 1]; ++ru)
							*(r++) -= *lv * *ru;
				}
			}
		}
	}

	return expansions;
}

// ---------------------------------------------------------------------------
// Lyndon bracket expansion: reconstruct full tensor element from Lyndon coords
// ---------------------------------------------------------------------------

template<std::floating_point T>
void expand_lyndon_to_tensor_(
	const T* lyndon_coefs,
	T* expanded,
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	const BasisCache& cache = get_basis_cache(dimension, degree, 2);
	const uint64_t sig_len = ::sig_length(dimension, degree);
	const uint64_t m = cache.lyndon_idx.size();

	// method=1: apply P^{-1} to convert Lyndon word positions -> bracket coefficients
	auto coefs_uptr = std::make_unique<T[]>(m);
	T* coefs = coefs_uptr.get();
	std::memcpy(coefs, lyndon_coefs, m * sizeof(T));
	if (method == 1)
		cache.inv_proj_mat.mul_vec_inplace_lower(coefs);

	auto expansions = build_bracket_expansions_<T>(dimension, degree);

	std::fill(expanded, expanded + sig_len, static_cast<T>(0.));
	for (uint64_t i = 0; i < m; ++i) {
		if (coefs[i] == static_cast<T>(0.)) continue;
		const T* exp_i = expansions.get() + i * sig_len;
		for (uint64_t j = 0; j < sig_len; ++j)
			expanded[j] += coefs[i] * exp_i[j];
	}
}

template<std::floating_point T>
void get_logsig_to_sig_(
	const T* log_sig,
	T* out,
	uint64_t dimension,
	uint64_t degree,
	int method = 0
) {
	switch (method) {
	case 0:
		sig_from_logsig_expanded<T>(log_sig, out, dimension, degree);
		break;
	case 1:
	case 2: {
		const uint64_t sig_len = ::sig_length(dimension, degree);
		auto expanded = std::make_unique<T[]>(sig_len);
		expand_lyndon_to_tensor_<T>(log_sig, expanded.get(), dimension, degree, method);
		tensor_exp_<T>(expanded.get(), out, dimension, degree);
		break;
	}
	default:
		throw std::runtime_error("method must be one of 0, 1 or 2");
	}
}

// ---------------------------------------------------------------------------
// Backward with method dispatch
// ---------------------------------------------------------------------------

template<std::floating_point T>
void get_logsig_to_sig_backprop_(
	T* d_logsig,
	const T* d_sig,
	const T* log_sig,
	uint64_t dimension,
	uint64_t degree,
	int method = 0
) {
	switch (method) {
	case 0:
		tensor_exp_backprop_<T>(d_logsig, d_sig, log_sig, dimension, degree);
		break;
	case 1:
	case 2: {
		const uint64_t sig_len = ::sig_length(dimension, degree);
		const BasisCache& cache = get_basis_cache(dimension, degree, 2);
		const uint64_t m = cache.lyndon_idx.size();

		auto expanded = std::make_unique<T[]>(sig_len);
		expand_lyndon_to_tensor_<T>(log_sig, expanded.get(), dimension, degree, method);

		auto d_expanded = std::make_unique<T[]>(sig_len);
		tensor_exp_backprop_<T>(d_expanded.get(), d_sig, expanded.get(), dimension, degree);

		// Backprop through expand (linear map): d_coefs[i] = dot(d_expanded, expansion[i])
		auto expansions = build_bracket_expansions_<T>(dimension, degree);

		auto d_coefs = std::make_unique<T[]>(m);
		for (uint64_t i = 0; i < m; ++i) {
			T acc = static_cast<T>(0.);
			const T* exp_i = expansions.get() + i * sig_len;
			for (uint64_t j = 0; j < sig_len; ++j)
				acc += d_expanded[j] * exp_i[j];
			d_coefs[i] = acc;
		}

		// method=1: forward used P^{-1}, so backward applies (P^{-1})^T
		if (method == 1)
			cache.inv_proj_mat_transpose.mul_vec_inplace_upper(d_coefs.get());

		std::memcpy(d_logsig, d_coefs.get(), m * sizeof(T));
		break;
	}
	default:
		throw std::runtime_error("method must be one of 0, 1 or 2");
	}
}

// ---------------------------------------------------------------------------
// Single / batch wrappers with time_aug / lead_lag
// ---------------------------------------------------------------------------

template<std::floating_point T>
void logsig_to_sig_(
	const T* log_sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	bool scalar_term = true,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("logsig_to_sig received dimension 0");
	if (degree == 0) throw std::invalid_argument("logsig_to_sig received degree 0");

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t sig_len = ::sig_length(aug_dimension, degree);
	// Input: logsig-shaped (method>0, unaffected by scalar_term) or sig-shaped (method==0)
	const uint64_t full_in_len = method ? ::log_sig_length(aug_dimension, degree) : sig_len;
	const uint64_t in_stride = (method == 0 && !scalar_term) ? (sig_len - 1) : full_in_len;
	// Output is always sig-shaped
	const uint64_t out_stride = scalar_term ? sig_len : sig_len - 1;

	if (scalar_term) {
		auto func = [&](const T* in_ptr, T* out_ptr) {
			get_logsig_to_sig_<T>(in_ptr, out_ptr, aug_dimension, degree, method);
		};
		multi_threaded_batch(func, batch_size, n_jobs,
			make_batch(log_sig, full_in_len), make_batch(out, sig_len));
	} else {
		auto func = [&](const T* in_ptr, T* out_ptr) {
			// Prepare full input if method==0 (sig-shaped, prepend scalar)
			const T* actual_in = in_ptr;
			std::vector<T> in_full;
			if (method == 0) {
				in_full.resize(sig_len);
				in_full[0] = static_cast<T>(1);
				std::memcpy(in_full.data() + 1, in_ptr, (sig_len - 1) * sizeof(T));
				actual_in = in_full.data();
			}
			// Compute into full output, then strip
			std::vector<T> out_full(sig_len);
			get_logsig_to_sig_<T>(actual_in, out_full.data(), aug_dimension, degree, method);
			std::memcpy(out_ptr, out_full.data() + 1, (sig_len - 1) * sizeof(T));
		};
		multi_threaded_batch(func, batch_size, n_jobs,
			make_batch(log_sig, in_stride), make_batch(out, out_stride));
	}
}

// ---------------------------------------------------------------------------
// Backprop wrappers
// ---------------------------------------------------------------------------

template<std::floating_point T>
void logsig_to_sig_backprop_(
	const T* log_sig,
	T* d_logsig,
	const T* d_sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	bool scalar_term = true,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("logsig_to_sig_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("logsig_to_sig_backprop received degree 0");

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t sig_len = ::sig_length(aug_dimension, degree);
	// Input log_sig: logsig-shaped (method>0) or sig-shaped (method==0)
	const uint64_t full_in_len = method ? ::log_sig_length(aug_dimension, degree) : sig_len;
	const uint64_t in_stride = (method == 0 && !scalar_term) ? (sig_len - 1) : full_in_len;
	// d_sig (incoming grad) is sig-shaped
	const uint64_t dsig_stride = scalar_term ? sig_len : sig_len - 1;
	// d_logsig (output grad) is logsig-shaped (method>0) or sig-shaped (method==0)
	const uint64_t dout_stride = (method == 0 && !scalar_term) ? (sig_len - 1) : full_in_len;

	if (n_jobs == 0) throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : static_cast<int>(get_max_threads()) + 1 + n_jobs;
	if (max_threads < 1) throw std::invalid_argument("n_jobs too low");
	const uint64_t num_threads = std::min(static_cast<uint64_t>(max_threads), batch_size);

	if (scalar_term) {
		auto batch_func = [&](uint64_t start, uint64_t end) {
			for (uint64_t i = start; i < end; ++i) {
				get_logsig_to_sig_backprop_<T>(
					d_logsig + i * full_in_len, d_sig + i * sig_len,
					log_sig + i * full_in_len, aug_dimension, degree, method);
			}
		};

		if (num_threads > 1) {
			std::vector<std::thread> workers;
			const uint64_t chunk = batch_size / num_threads;
			const uint64_t remainder = batch_size % num_threads;
			uint64_t start = 0;
			for (uint64_t t = 0; t < num_threads; ++t) {
				uint64_t end = start + chunk + (t < remainder ? 1 : 0);
				workers.emplace_back(batch_func, start, end);
				start = end;
			}
			for (auto& w : workers) w.join();
		} else {
			batch_func(0, batch_size);
		}
	} else {
		auto batch_func = [&](uint64_t start, uint64_t end) {
			for (uint64_t i = start; i < end; ++i) {
				// Prepare d_sig (sig-shaped, prepend 0)
				std::vector<T> dsig_full(sig_len);
				dsig_full[0] = static_cast<T>(0);
				std::memcpy(dsig_full.data() + 1, d_sig + i * dsig_stride, dsig_stride * sizeof(T));

				// Prepare log_sig input (if method==0, prepend scalar)
				const T* actual_in;
				std::vector<T> in_full;
				if (method == 0) {
					in_full.resize(sig_len);
					in_full[0] = static_cast<T>(1);
					std::memcpy(in_full.data() + 1, log_sig + i * in_stride, in_stride * sizeof(T));
					actual_in = in_full.data();
				} else {
					actual_in = log_sig + i * in_stride;
				}

				if (method == 0) {
					// Output is sig-shaped, strip
					std::vector<T> dout_full(sig_len);
					get_logsig_to_sig_backprop_<T>(
						dout_full.data(), dsig_full.data(),
						actual_in, aug_dimension, degree, method);
					std::memcpy(d_logsig + i * dout_stride, dout_full.data() + 1, dout_stride * sizeof(T));
				} else {
					// Output is logsig-shaped, no stripping
					get_logsig_to_sig_backprop_<T>(
						d_logsig + i * dout_stride, dsig_full.data(),
						actual_in, aug_dimension, degree, method);
				}
			}
		};

		if (num_threads > 1) {
			std::vector<std::thread> workers;
			const uint64_t chunk = batch_size / num_threads;
			const uint64_t remainder = batch_size % num_threads;
			uint64_t start = 0;
			for (uint64_t t = 0; t < num_threads; ++t) {
				uint64_t end = start + chunk + (t < remainder ? 1 : 0);
				workers.emplace_back(batch_func, start, end);
				start = end;
			}
			for (auto& w : workers) w.join();
		} else {
			batch_func(0, batch_size);
		}
	}
}
