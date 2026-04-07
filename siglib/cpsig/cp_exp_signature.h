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
// tensor_exp_: Horner scheme for the truncated tensor exponential
//
//   exp(x) = 1 + x(1 + x/2(1 + x/3(...(1 + x/N)...)))
//
// Input:  log_sig — expanded log-signature (sig_length elements, level 0 = 0)
// Output: out     — signature (sig_length elements, level 0 = 1)
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

	// Initialize: out = 1 + x / N
	out[0] = static_cast<T>(1.);
	if (degree == 0) return;

	T inv_k = static_cast<T>(1.) / static_cast<T>(degree);
	for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i) {
		out[i] = log_sig[i] * inv_k;
	}

	if (degree <= 1) return;

	// Scratch buffer for tensor product accumulation
	auto buff_uptr = std::make_unique<T[]>(sig_len);
	T* buff = buff_uptr.get();

	// Horner iterations: k = N-1 down to 1
	for (int64_t k = static_cast<int64_t>(degree) - 1; k >= 1; --k) {
		inv_k = static_cast<T>(1.) / static_cast<T>(k);

		// Compute buff[l] = sum_{l1+l2=l, l1>=1, l2>=1} (x[l1] * inv_k) * out[l2]
		for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
			std::fill(buff + level_index[target_level], buff + level_index[target_level + 1], static_cast<T>(0.));

			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				uint64_t right_level = target_level - left_level;

				T* res_ptr = buff + level_index[target_level];
				const T* const left_ptr_end = log_sig + level_index[left_level + 1];
#ifdef VEC
				const uint64_t right_level_size = level_index[right_level + 1] - level_index[right_level];
				const T* right_start = out + level_index[right_level];
				for (const T* left_ptr = log_sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr) {
					vec_mult_add(res_ptr, right_start, *left_ptr * inv_k, right_level_size);
					res_ptr += right_level_size;
				}
#else
				const T* const right_ptr_end = out + level_index[right_level + 1];
				for (const T* left_ptr = log_sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr) {
					T val = *left_ptr * inv_k;
					for (const T* right_ptr = out + level_index[right_level]; right_ptr < right_ptr_end; ++right_ptr) {
						*(res_ptr++) += val * *right_ptr;
					}
				}
#endif
			}
		}

		// Update out: out[l] = x[l]/k + buff[l] for l >= 2, out[1] = x[1]/k
		for (uint64_t l = 2; l <= degree; ++l) {
			for (uint64_t i = level_index[l]; i < level_index[l + 1]; ++i) {
				out[i] = log_sig[i] * inv_k + buff[i];
			}
		}
		for (uint64_t i = level_index[1]; i < level_index[2]; ++i) {
			out[i] = log_sig[i] * inv_k;
		}
		// out[0] = 1 (unchanged)
	}
}

// ---------------------------------------------------------------------------
// tensor_exp_backprop_: backward pass through the Horner scheme
//
// Given dL/d(out) (the upstream gradient w.r.t. the signature),
// computes dL/d(log_sig) (gradient w.r.t. the log-signature).
//
// Recomputes Horner intermediates rather than saving from forward.
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
		// exp(x) = 1 + x, so d_logsig = d_sig (levels 1+)
		for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i) {
			d_logsig[i] = d_sig[i];
		}
		return;
	}

	// Recompute all Horner intermediates S_N, S_{N-1}, ..., S_2
	// S_N = 1 + x/N, S_k = 1 + (x/k) * S_{k+1}
	// We need S_2, S_3, ..., S_N for the backward. Store them in a flat array.
	const uint64_t num_intermediates = degree - 1; // S_2 through S_N
	auto intermediates_uptr = std::make_unique<T[]>(sig_len * num_intermediates);
	T* intermediates = intermediates_uptr.get();

	auto buff_uptr = std::make_unique<T[]>(sig_len);
	T* buff = buff_uptr.get();

	// Compute S_N first
	T* S_current = intermediates + (num_intermediates - 1) * sig_len; // S_N slot
	S_current[0] = static_cast<T>(1.);
	T inv_k = static_cast<T>(1.) / static_cast<T>(degree);
	for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i) {
		S_current[i] = log_sig[i] * inv_k;
	}

	// Compute S_{N-1}, ..., S_2 via Horner
	for (int64_t k = static_cast<int64_t>(degree) - 1; k >= 2; --k) {
		T* S_prev = S_current; // S_{k+1}
		S_current = intermediates + (k - 2) * sig_len; // S_k slot (k=2 goes to index 0)
		inv_k = static_cast<T>(1.) / static_cast<T>(k);

		// Compute S_k = 1 + (x/k) * S_{k+1}
		for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
			std::fill(buff + level_index[target_level], buff + level_index[target_level + 1], static_cast<T>(0.));
			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				uint64_t right_level = target_level - left_level;
				T* res_ptr = buff + level_index[target_level];
				const T* const left_ptr_end = log_sig + level_index[left_level + 1];
				const T* const right_ptr_end = S_prev + level_index[right_level + 1];
				for (const T* left_ptr = log_sig + level_index[left_level]; left_ptr < left_ptr_end; ++left_ptr) {
					T val = *left_ptr * inv_k;
					for (const T* right_ptr = S_prev + level_index[right_level]; right_ptr < right_ptr_end; ++right_ptr) {
						*(res_ptr++) += val * *right_ptr;
					}
				}
			}
		}
		S_current[0] = static_cast<T>(1.);
		for (uint64_t i = level_index[1]; i < level_index[2]; ++i) {
			S_current[i] = log_sig[i] * inv_k;
		}
		for (uint64_t l = 2; l <= degree; ++l) {
			for (uint64_t i = level_index[l]; i < level_index[l + 1]; ++i) {
				S_current[i] = log_sig[i] * inv_k + buff[i];
			}
		}
	}

	// Backward pass: propagate dL/dS_1 back through Horner steps
	// dS = current upstream gradient (starts as dL/dS_1 = d_sig)
	auto dS_uptr = std::make_unique<T[]>(sig_len);
	T* dS = dS_uptr.get();
	std::memcpy(dS, d_sig, sig_len * sizeof(T));

	auto dS_next_uptr = std::make_unique<T[]>(sig_len);
	T* dS_next = dS_next_uptr.get();

	// For k = 1, 2, ..., N-1:
	// S_k = 1 + (x/k) * S_{k+1}
	// dL/dx[l] += dS[l] / k  (additive term)
	// dL/dS_{k+1}[l2] = sum_{l1} dS[l1+l2] / k * x[l1]  (tensor product backprop, right)
	// dL/dx[l1] += sum_{l2} dS[l1+l2] / k * S_{k+1}[l2]  (tensor product backprop, left)
	for (int64_t k = 1; k < static_cast<int64_t>(degree); ++k) {
		inv_k = static_cast<T>(1.) / static_cast<T>(k);
		const T* S_kp1 = intermediates + (k - 1) * sig_len; // S_{k+1} (k=1 -> index 0 = S_2)

		// Accumulate dL/dx from additive term: dL/dx[l] += dS[l] / k
		for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i) {
			d_logsig[i] += dS[i] * inv_k;
		}

		// Backprop through tensor product: T = (x/k) * S_{k+1}
		// where dS gives dL/dT at levels 2..degree
		// Compute dL/dS_{k+1} and additional dL/dx
		std::fill(dS_next, dS_next + sig_len, static_cast<T>(0.));

		for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
			for (uint64_t left_level = 1; left_level < target_level; ++left_level) {
				uint64_t right_level = target_level - left_level;

				const T* res_ptr = dS + level_index[target_level];
				T* d_left_ptr = d_logsig + level_index[left_level];
				const T* left_ptr = log_sig + level_index[left_level];
				const T* const left_ptr_end = log_sig + level_index[left_level + 1];
				T* d_right_ptr = dS_next + level_index[right_level];
				const T* right_ptr = S_kp1 + level_index[right_level];
				const T* const right_ptr_end = S_kp1 + level_index[right_level + 1];

				// dL/dx[l1] += sum_{l2} dS[l] * S_{k+1}[l2] * inv_k
				// dL/dS_{k+1}[l2] += sum_{l1} dS[l] * x[l1] * inv_k
				const T* rp = res_ptr;
				for (const T* lp = left_ptr; lp < left_ptr_end; ++lp) {
					T d_left_acc = static_cast<T>(0.);
					T* drp = d_right_ptr;
					for (const T* rrp = right_ptr; rrp < right_ptr_end; ++rrp) {
						T dS_val = *(rp++) * inv_k;
						d_left_acc += dS_val * *rrp;
						*(drp++) += dS_val * *lp;
					}
					*(d_left_ptr++) += d_left_acc;
				}
			}
		}

		// Swap dS and dS_next for the next iteration
		std::swap(dS, dS_next);
	}

	// Final step: k = N, S_N = 1 + x/N
	// dL/dx[l] += dS[l] / N
	inv_k = static_cast<T>(1.) / static_cast<T>(degree);
	for (uint64_t i = level_index[1]; i < level_index[degree + 1]; ++i) {
		d_logsig[i] += dS[i] * inv_k;
	}
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
// Lyndon bracket expansion: reconstruct full tensor element from Lyndon coords
//
// For each Lyndon word w, compute its tensor algebra expansion via recursive
// standard factorization: [u, v] = u⊗v - v⊗u. Then sum weighted by coefficients.
// ---------------------------------------------------------------------------

template<std::floating_point T>
void expand_lyndon_to_tensor_(
	const T* lyndon_coefs,
	T* expanded,
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	// Always use method=2 cache for the projection matrix and Lyndon indices
	const BasisCache& cache = get_basis_cache(dimension, degree, 2);
	const uint64_t sig_len = ::sig_length(dimension, degree);
	const uint64_t m = cache.lyndon_idx.size();

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	populate_level_index(level_index, dimension, degree + 2);

	// Convert to bracket coefficients:
	// method=1: input is Lyndon word POSITION values -> apply P^{-1} to get bracket coefficients
	// method=2: input IS already bracket coefficients
	auto coefs_uptr = std::make_unique<T[]>(m);
	T* coefs = coefs_uptr.get();
	std::memcpy(coefs, lyndon_coefs, m * sizeof(T));
	if (method == 1) {
		cache.inv_proj_mat.mul_vec_inplace_lower(coefs);
	}

	// Enumerate Lyndon words and build index map
	auto lyndon_words = all_lyndon_words(dimension, degree);
	std::unordered_set<word, WordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());

	// Map from word to its index in the Lyndon word list
	std::unordered_map<word, uint64_t, WordHash> word_to_lyndon_idx;
	for (uint64_t i = 0; i < m; ++i) {
		word_to_lyndon_idx[lyndon_words[i]] = i;
	}

	// Compute bracket expansions for each Lyndon word (in order of increasing length)
	// Each expansion is a sig_length vector
	auto expansions = std::make_unique<T[]>(m * sig_len);

	for (uint64_t i = 0; i < m; ++i) {
		T* exp_i = expansions.get() + i * sig_len;
		std::fill(exp_i, exp_i + sig_len, static_cast<T>(0.));

		if (lyndon_words[i].size() == 1) {
			// Single letter: unit vector at position word_to_idx in level 1
			exp_i[cache.lyndon_idx[i]] = static_cast<T>(1.);
		}
		else {
			// Standard factorization: w = [u, v]
			auto [u, v] = standard_factorization(lyndon_words[i], lyndon_set);
			uint64_t u_idx = word_to_lyndon_idx.at(u);
			uint64_t v_idx = word_to_lyndon_idx.at(v);
			const T* exp_u = expansions.get() + u_idx * sig_len;
			const T* exp_v = expansions.get() + v_idx * sig_len;

			// exp_i = u⊗v - v⊗u (Lie bracket in tensor algebra)
			for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
				for (uint64_t l1 = 1; l1 < target_level; ++l1) {
					uint64_t l2 = target_level - l1;
					T* res = exp_i + level_index[target_level];
					const T* left_u_end = exp_u + level_index[l1 + 1];
					const T* right_v_start = exp_v + level_index[l2];
					const T* right_v_end = exp_v + level_index[l2 + 1];
					const T* left_v_end = exp_v + level_index[l1 + 1];
					const T* right_u_start = exp_u + level_index[l2];
					const T* right_u_end = exp_u + level_index[l2 + 1];

					// u⊗v contribution
					T* r = res;
					for (const T* lu = exp_u + level_index[l1]; lu < left_u_end; ++lu)
						for (const T* rv = right_v_start; rv < right_v_end; ++rv)
							*(r++) += *lu * *rv;

					// -v⊗u contribution
					r = res;
					for (const T* lv = exp_v + level_index[l1]; lv < left_v_end; ++lv)
						for (const T* ru = right_u_start; ru < right_u_end; ++ru)
							*(r++) -= *lv * *ru;
				}
			}
		}
	}

	// Sum: expanded = sum_i coefs[i] * expansions[i]
	std::fill(expanded, expanded + sig_len, static_cast<T>(0.));
	for (uint64_t i = 0; i < m; ++i) {
		if (coefs[i] == static_cast<T>(0.)) continue;
		const T* exp_i = expansions.get() + i * sig_len;
		for (uint64_t j = 0; j < sig_len; ++j) {
			expanded[j] += coefs[i] * exp_i[j];
		}
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

		// Forward: expand lyndon coords to tensor, then tensor_exp
		// Backward: tensor_exp_backprop -> d_expanded, then backprop through expand

		// Recompute expanded tensor element
		auto expanded = std::make_unique<T[]>(sig_len);
		expand_lyndon_to_tensor_<T>(log_sig, expanded.get(), dimension, degree, method);

		// Backprop through tensor_exp
		auto d_expanded = std::make_unique<T[]>(sig_len);
		tensor_exp_backprop_<T>(d_expanded.get(), d_sig, expanded.get(), dimension, degree);

		// Backprop through expand_lyndon_to_tensor (linear map)
		// d_logsig[i] = sum_j d_expanded[j] * expansion[i][j]  (gather via E^T)
		const BasisCache& cache = get_basis_cache(dimension, degree, 2);
		uint64_t m = cache.lyndon_idx.size();

		// Recompute bracket expansions
		auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
		populate_level_index(level_index_uptr.get(), dimension, degree + 2);

		auto lyndon_words = all_lyndon_words(dimension, degree);
		std::unordered_set<word, WordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());
		std::unordered_map<word, uint64_t, WordHash> word_to_lyndon_idx_map;
		for (uint64_t i = 0; i < m; ++i)
			word_to_lyndon_idx_map[lyndon_words[i]] = i;

		auto expansions = std::make_unique<T[]>(m * sig_len);
		for (uint64_t i = 0; i < m; ++i) {
			T* exp_i = expansions.get() + i * sig_len;
			std::fill(exp_i, exp_i + sig_len, static_cast<T>(0.));

			if (lyndon_words[i].size() == 1) {
				exp_i[cache.lyndon_idx[i]] = static_cast<T>(1.);
			}
			else {
				auto [u, v] = standard_factorization(lyndon_words[i], lyndon_set);
				uint64_t u_idx = word_to_lyndon_idx_map.at(u);
				uint64_t v_idx = word_to_lyndon_idx_map.at(v);
				const T* exp_u = expansions.get() + u_idx * sig_len;
				const T* exp_v = expansions.get() + v_idx * sig_len;
				uint64_t* level_index = level_index_uptr.get();

				for (uint64_t target_level = 2; target_level <= degree; ++target_level) {
					for (uint64_t l1 = 1; l1 < target_level; ++l1) {
						uint64_t l2 = target_level - l1;
						T* r = exp_i + level_index[target_level];
						for (const T* lu = exp_u + level_index[l1]; lu < exp_u + level_index[l1 + 1]; ++lu)
							for (const T* rv = exp_v + level_index[l2]; rv < exp_v + level_index[l2 + 1]; ++rv)
								*(r++) += *lu * *rv;
						r = exp_i + level_index[target_level];
						for (const T* lv = exp_v + level_index[l1]; lv < exp_v + level_index[l1 + 1]; ++lv)
							for (const T* ru = exp_u + level_index[l2]; ru < exp_u + level_index[l2 + 1]; ++ru)
								*(r++) -= *lv * *ru;
					}
				}
			}
		}

		// d_lyndon_coefs[i] = dot(d_expanded, expansion[i])
		auto d_coefs = std::make_unique<T[]>(m);
		for (uint64_t i = 0; i < m; ++i) {
			T acc = static_cast<T>(0.);
			const T* exp_i = expansions.get() + i * sig_len;
			for (uint64_t j = 0; j < sig_len; ++j) {
				acc += d_expanded[j] * exp_i[j];
			}
			d_coefs[i] = acc;
		}

		// Backprop through coefficient conversion:
		// method=1: forward used P^{-1}, so backward applies (P^{-1})^T
		// method=2: no conversion, no backprop needed
		if (method == 1) {
			cache.inv_proj_mat_transpose.mul_vec_inplace_upper(d_coefs.get());
		}

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
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0
) {
	if (dimension == 0) throw std::invalid_argument("logsig_to_sig received dimension 0");
	if (degree == 0) throw std::invalid_argument("logsig_to_sig received degree 0");

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	get_logsig_to_sig_<T>(log_sig, out, aug_dimension, degree, method);
}

template<std::floating_point T>
void batch_logsig_to_sig_(
	const T* log_sig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("logsig_to_sig received dimension 0");
	if (degree == 0) throw std::invalid_argument("logsig_to_sig received degree 0");

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t input_length = method ? ::log_sig_length(aug_dimension, degree) : ::sig_length(aug_dimension, degree);
	const uint64_t output_length = ::sig_length(aug_dimension, degree);

	auto func = [&](const T* in_ptr, T* out_ptr) {
		get_logsig_to_sig_<T>(in_ptr, out_ptr, aug_dimension, degree, method);
	};

	if (n_jobs != 1) {
		multi_threaded_batch(func, log_sig, out, batch_size, input_length, output_length, n_jobs);
	}
	else {
		const T* in_ptr = log_sig;
		T* out_ptr = out;
		for (uint64_t i = 0; i < batch_size; ++i, in_ptr += input_length, out_ptr += output_length) {
			func(in_ptr, out_ptr);
		}
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
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0
) {
	if (dimension == 0) throw std::invalid_argument("logsig_to_sig_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("logsig_to_sig_backprop received degree 0");

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	get_logsig_to_sig_backprop_<T>(d_logsig, d_sig, log_sig, aug_dimension, degree, method);
}

template<std::floating_point T>
void batch_logsig_to_sig_backprop_(
	const T* log_sig,
	T* d_logsig,
	const T* d_sig,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	bool time_aug = false,
	bool lead_lag = false,
	int method = 0,
	int n_jobs = 1
) {
	if (dimension == 0) throw std::invalid_argument("logsig_to_sig_backprop received dimension 0");
	if (degree == 0) throw std::invalid_argument("logsig_to_sig_backprop received degree 0");

	uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const uint64_t input_length = method ? ::log_sig_length(aug_dimension, degree) : ::sig_length(aug_dimension, degree);
	const uint64_t sig_len = ::sig_length(aug_dimension, degree);

	if (n_jobs == 0) throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : static_cast<int>(get_max_threads()) + 1 + n_jobs;
	if (max_threads < 1) throw std::invalid_argument("n_jobs too low");

	const uint64_t num_threads = std::min(static_cast<uint64_t>(max_threads), batch_size);

	auto batch_func = [&](uint64_t start, uint64_t end) {
		for (uint64_t i = start; i < end; ++i) {
			get_logsig_to_sig_backprop_<T>(
				d_logsig + i * input_length, d_sig + i * sig_len,
				log_sig + i * input_length, aug_dimension, degree, method);
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
	}
	else {
		batch_func(0, batch_size);
	}
}
