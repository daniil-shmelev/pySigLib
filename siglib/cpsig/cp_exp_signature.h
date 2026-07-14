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

#include "cp_sig_combine.h"
#include "multithreading.h"
#include "words.h"
#include "log_sig_cache.h"

// ---------------------------------------------------------------------------
// Truncated tensor exponential via power series
//
//   exp(x) = 1 + P_1 + P_2 + ... + P_N
//   P_1 = x, P_n = x \otimes P_{n-1} / n
//   P_n has min level n -> level-skipping reduces work for large n.
// ---------------------------------------------------------------------------

template<std::floating_point T>
void tensor_exp_with_level_index_(
	const T* log_sig,
	T* out,
	uint64_t log_degree,
	uint64_t out_degree,
	const uint64_t* level_index,
	std::vector<T>& powers,
	std::vector<T>& next
) {
	if (out_degree == 0) return;
	const uint64_t length = level_index[out_degree + 1];
	const uint64_t first = level_index[1];
	const uint64_t copy_degree = std::min(log_degree, out_degree);
	const uint64_t copy_end = level_index[copy_degree + 1];
	powers.assign(length, static_cast<T>(0));
	next.assign(length, static_cast<T>(0));
	std::fill(out + first, out + length, static_cast<T>(0));
	std::copy(log_sig + first, log_sig + copy_end, powers.begin() + first);
	std::copy(powers.begin() + first, powers.end(), out + first);

	for (uint64_t power = 2; power <= out_degree; ++power) {
		std::fill(next.begin() + level_index[power], next.end(), static_cast<T>(0));
		const T inv_power = static_cast<T>(1) / static_cast<T>(power);
		tensor_product_add_(log_sig, log_degree, powers.data(), out_degree,
			next.data(), out_degree, level_index, power, power - 1, inv_power);
		for (uint64_t i = level_index[power]; i < length; ++i) out[i] += next[i];
		powers.swap(next);
	}
}

template<std::floating_point T>
void tensor_exp_backprop_with_level_index_(
	T* d_logsig,
	const T* d_sig,
	const T* log_sig,
	uint64_t log_degree,
	uint64_t out_degree,
	const uint64_t* level_index,
	std::vector<T>& powers,
	std::vector<T>& d_powers
) {
	if (out_degree == 0) return;
	const uint64_t length = level_index[out_degree + 1];
	const uint64_t first = level_index[1];
	const uint64_t copy_degree = std::min(log_degree, out_degree);
	const uint64_t copy_end = level_index[copy_degree + 1];
	powers.assign(out_degree * length, static_cast<T>(0));
	d_powers.assign(out_degree * length, static_cast<T>(0));
	std::copy(log_sig + first, log_sig + copy_end, powers.begin() + first);
	for (uint64_t power = 0; power < out_degree; ++power)
		std::copy(d_sig + first, d_sig + length,
			d_powers.begin() + power * length + first);

	for (uint64_t power = 2; power <= out_degree; ++power) {
		const T inv_power = static_cast<T>(1) / static_cast<T>(power);
		tensor_product_add_(log_sig, log_degree,
			powers.data() + (power - 2) * length, out_degree,
			powers.data() + (power - 1) * length, out_degree,
			level_index, power, power - 1, inv_power);
	}

	for (uint64_t power = out_degree; power >= 2; --power) {
		const T inv_power = static_cast<T>(1) / static_cast<T>(power);
		tensor_product_backprop_(log_sig, log_degree,
			powers.data() + (power - 2) * length, out_degree,
			d_powers.data() + (power - 1) * length, out_degree,
			d_logsig, d_powers.data() + (power - 2) * length,
			level_index, power, power - 1, inv_power);
	}
	for (uint64_t i = first; i < copy_end; ++i) d_logsig[i] += d_powers[i];
}

template<std::floating_point T>
void tensor_exp_(
	const T* log_sig,
	T* out,
	uint64_t dimension,
	uint64_t degree
) {
	auto level_index = std::make_unique<uint64_t[]>(degree + 2);
	populate_level_index(level_index.get(), dimension, degree + 2);
	out[0] = static_cast<T>(1);
	std::vector<T> powers;
	std::vector<T> next;
	tensor_exp_with_level_index_(log_sig, out, degree, degree,
		level_index.get(), powers, next);
}

template<std::floating_point T>
void tensor_exp_backprop_(
	T* d_logsig,
	const T* d_sig,
	const T* log_sig,
	uint64_t dimension,
	uint64_t degree
) {
	auto level_index = std::make_unique<uint64_t[]>(degree + 2);
	populate_level_index(level_index.get(), dimension, degree + 2);
	std::fill(d_logsig, d_logsig + level_index[degree + 1], static_cast<T>(0));
	std::vector<T> powers;
	std::vector<T> d_powers;
	tensor_exp_backprop_with_level_index_(d_logsig, d_sig, log_sig, degree, degree,
		level_index.get(), powers, d_powers);
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
