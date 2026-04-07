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
#include "cp_branched_cache.h"
#include "cp_path.h"
#include "multithreading.h"
#include "macros.h"


template<std::floating_point T>
void linear_branched_sig_(
	const T* increment,
	T* out,
	const BranchedSigCache& cache
) {
	out[0] = static_cast<T>(1.);

	uint64_t num_trees = cache.total_length - 1;
	const uint8_t* labels = cache.node_labels_data.data();
	const uint64_t* offsets = cache.node_labels_offsets.data();

	for (uint64_t i = 0; i < num_trees; ++i) {
		T product = static_cast<T>(1.);
		uint64_t start = offsets[i];
		uint64_t end = offsets[i + 1];
		for (uint64_t j = start; j < end; ++j) {
			product *= increment[labels[j]];
		}
		out[i + 1] = product * static_cast<T>(cache.inv_tree_factorial[i]);
	}
}


// Processes trees from highest order down to order 1 so that forest
// references (always lower-order) use un-updated X values.
template<std::floating_point T>
void butcher_product_inplace_(
	T* X,
	const T* Y,
	const BranchedSigCache& cache
) {
	for (int64_t order = static_cast<int64_t>(cache.max_nodes); order >= 1; --order) {
		uint64_t start = cache.order_index[order];
		uint64_t end = cache.order_index[order + 1];

		for (uint64_t tree_idx = start; tree_idx < end; ++tree_idx) {
			uint64_t flat_idx = tree_idx + 1;
			T new_val = X[flat_idx] + Y[flat_idx];

			uint64_t pos = cache.coproduct_offsets[tree_idx];
			uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];

			while (pos < pos_end) {
				uint64_t num_forest = cache.coproduct_data[pos++];
				uint64_t trunk_flat = cache.coproduct_data[pos++];
				T term = Y[trunk_flat];

				for (uint64_t j = 0; j < num_forest; ++j) {
					term *= X[cache.coproduct_data[pos++]];
				}

				new_val += term;
			}

			X[flat_idx] = new_val;
		}
	}
}


template<std::floating_point T>
void branched_signature_with_buffers_(
	const Path<T>& path,
	T* out,
	T* increment,
	T* temp,
	const BranchedSigCache& cache
) {
	uint64_t total_len = cache.total_length;
	uint64_t dim = path.dimension();
	uint64_t len = path.length();

	if (len <= 1) {
		out[0] = static_cast<T>(1.);
		std::memset(out + 1, 0, (total_len - 1) * sizeof(T));
		return;
	}

	auto p0 = path[0];
	auto p1 = path[1];
	for (uint64_t d = 0; d < dim; ++d) {
		increment[d] = p1[d] - p0[d];
	}
	linear_branched_sig_(increment, out, cache);

	for (uint64_t seg = 1; seg < len - 1; ++seg) {
		auto seg_start = path[seg];
		auto seg_end = path[seg + 1];

		for (uint64_t d = 0; d < dim; ++d) {
			increment[d] = seg_end[d] - seg_start[d];
		}

		linear_branched_sig_(increment, temp, cache);
		butcher_product_inplace_(out, temp, cache);
	}
}


template<std::floating_point T>
void branched_signature_(
	const T* path,
	T* out,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.)
) {
	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);
	const auto& cache = get_branched_sig_cache(path_obj.dimension(), max_nodes);
	auto increment = std::make_unique<T[]>(path_obj.dimension());
	auto temp = std::make_unique<T[]>(cache.total_length);
	branched_signature_with_buffers_(path_obj, out, increment.get(), temp.get(), cache);
}


template<std::floating_point T>
void batch_branched_signature_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.)
) {
	uint64_t aug_dim = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const auto& cache = get_branched_sig_cache(aug_dim, max_nodes);
	uint64_t total_len = cache.total_length;
	uint64_t flat_path_length = length * dimension;

	if (n_jobs == 1 || batch_size == 1) {
		auto increment = std::make_unique<T[]>(aug_dim);
		auto temp = std::make_unique<T[]>(total_len);
		const T* path_ptr = path;
		T* out_ptr = out;
		for (uint64_t b = 0; b < batch_size; ++b) {
			Path<T> path_obj(path_ptr, dimension, length, time_aug, lead_lag, end_time);
			branched_signature_with_buffers_(path_obj, out_ptr, increment.get(), temp.get(), cache);
			path_ptr += flat_path_length;
			out_ptr += total_len;
		}
	}
	else {
		if (n_jobs == 0)
			throw std::invalid_argument("n_jobs cannot be 0");
		const unsigned int max_threads = n_jobs > 0
			? static_cast<unsigned int>(n_jobs)
			: get_max_threads() + 1 + n_jobs;
		if (max_threads < 1)
			throw std::invalid_argument("n_jobs too low");

		std::vector<std::thread> workers;
		for (unsigned int t = 0; t < max_threads && t < batch_size; ++t) {
			workers.emplace_back([&, t, max_threads]() {
				auto increment = std::make_unique<T[]>(aug_dim);
				auto temp = std::make_unique<T[]>(total_len);
				for (uint64_t b = t; b < batch_size; b += max_threads) {
					Path<T> path_obj(path + b * flat_path_length, dimension, length, time_aug, lead_lag, end_time);
					branched_signature_with_buffers_(
						path_obj,
						out + b * total_len,
						increment.get(), temp.get(), cache);
				}
			});
		}
		for (auto& w : workers) w.join();
	}
}


template<std::floating_point T>
void branched_sig_combine_(
	const T* bsig1,
	const T* bsig2,
	T* out,
	uint64_t dimension,
	uint64_t max_nodes
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes);
	std::memcpy(out, bsig1, cache.total_length * sizeof(T));
	butcher_product_inplace_(out, bsig2, cache);
}


template<std::floating_point T>
void batch_branched_sig_combine_(
	const T* bsig1,
	const T* bsig2,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes);
	uint64_t total_len = cache.total_length;

	auto thread_func = [&](const T* sig1_ptr, const T* sig2_ptr, T* out_ptr) {
		std::memcpy(out_ptr, sig1_ptr, total_len * sizeof(T));
		butcher_product_inplace_(out_ptr, sig2_ptr, cache);
	};

	if (n_jobs == 1 || batch_size == 1) {
		for (uint64_t b = 0; b < batch_size; ++b) {
			thread_func(bsig1 + b * total_len, bsig2 + b * total_len, out + b * total_len);
		}
	}
	else {
		multi_threaded_batch_2(thread_func,
			const_cast<T*>(bsig1), const_cast<T*>(bsig2), out,
			batch_size, total_len, total_len, total_len, n_jobs);
	}
}


template<std::floating_point T>
void branched_sig_combine_backprop_(
	const T* bsig1,
	const T* bsig2,
	const T* derivs_in,
	T* out1,
	T* out2,
	uint64_t dimension,
	uint64_t max_nodes
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes);
	uint64_t total_len = cache.total_length;

	// butcher_product_deriv_ expects dF_dX (modified in-place to dF/dX_prev=bsig1)
	// and writes dF_dY (dF/dbsig2)
	std::memcpy(out1, derivs_in, total_len * sizeof(T));
	butcher_product_deriv_(bsig1, bsig2, out1, out2, cache);
}


template<std::floating_point T>
void batch_branched_sig_combine_backprop_(
	const T* bsig1,
	const T* bsig2,
	const T* derivs_in,
	T* out1,
	T* out2,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes);
	uint64_t total_len = cache.total_length;

	auto work = [&](uint64_t b) {
		uint64_t off = b * total_len;
		std::memcpy(out1 + off, derivs_in + off, total_len * sizeof(T));
		butcher_product_deriv_(bsig1 + off, bsig2 + off, out1 + off, out2 + off, cache);
	};

	if (n_jobs == 1 || batch_size == 1) {
		for (uint64_t b = 0; b < batch_size; ++b)
			work(b);
	}
	else {
		if (n_jobs == 0)
			throw std::invalid_argument("n_jobs cannot be 0");
		const unsigned int max_threads = n_jobs > 0
			? static_cast<unsigned int>(n_jobs)
			: get_max_threads() + 1 + n_jobs;
		if (max_threads < 1)
			throw std::invalid_argument("n_jobs too low");

		std::vector<std::thread> workers;
		for (unsigned int t = 0; t < max_threads && t < batch_size; ++t) {
			workers.emplace_back([&, t, max_threads]() {
				for (uint64_t b = t; b < batch_size; b += max_threads)
					work(b);
			});
		}
		for (auto& w : workers) w.join();
	}
}


// =========================================================================
// Backpropagation
// =========================================================================

// Reverse the Butcher product: recover X_prev from X_combined and Y.
// X_combined[τ] = X_prev[τ] + Y[τ] + Σ_cuts Y[trunk] * ∏ X_prev[forest_i]
// Solved for X_prev by processing order 1 to max_nodes (forward order).
template<std::floating_point T>
void butcher_uncombine_inplace_(
	T* X,
	const T* Y,
	const BranchedSigCache& cache
) {
	for (int64_t order = 1; order <= static_cast<int64_t>(cache.max_nodes); ++order) {
		uint64_t start = cache.order_index[order];
		uint64_t end = cache.order_index[order + 1];

		for (uint64_t tree_idx = start; tree_idx < end; ++tree_idx) {
			uint64_t flat_idx = tree_idx + 1;
			T val = X[flat_idx] - Y[flat_idx];

			uint64_t pos = cache.coproduct_offsets[tree_idx];
			uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];

			while (pos < pos_end) {
				uint64_t num_forest = cache.coproduct_data[pos++];
				uint64_t trunk_flat = cache.coproduct_data[pos++];
				T term = Y[trunk_flat];

				for (uint64_t j = 0; j < num_forest; ++j) {
					term *= X[cache.coproduct_data[pos++]];
				}

				val -= term;
			}

			X[flat_idx] = val;
		}
	}
}


// Differentiate the Butcher product.
// Given dF/dX_combined, compute dF/dX_prev and dF/dY.
// X_prev and Y must be the values from BEFORE the product (X_prev recovered via uncombine).
template<std::floating_point T>
void butcher_product_deriv_(
	const T* X_prev,
	const T* Y,
	T* dF_dX,        // in: dF/dX_combined, out: dF/dX_prev
	T* dF_dY,        // out: dF/dY
	const BranchedSigCache& cache
) {
	uint64_t num_trees = cache.total_length - 1;

	// Initialize dF_dY = dF_dX (the Y[τ] direct term)
	dF_dY[0] = static_cast<T>(0);
	std::memcpy(dF_dY + 1, dF_dX + 1, num_trees * sizeof(T));

	for (uint64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
		uint64_t flat_idx = tree_idx + 1;
		T dF_dXcomb_tau = dF_dX[flat_idx];
		if (dF_dXcomb_tau == static_cast<T>(0)) continue;

		uint64_t pos = cache.coproduct_offsets[tree_idx];
		uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];

		while (pos < pos_end) {
			uint64_t num_forest = cache.coproduct_data[pos++];
			uint64_t trunk_flat = cache.coproduct_data[pos++];

			uint64_t forest_start = pos;
			T forest_product = static_cast<T>(1);
			for (uint64_t j = 0; j < num_forest; ++j)
				forest_product *= X_prev[cache.coproduct_data[pos++]];

			dF_dY[trunk_flat] += dF_dXcomb_tau * forest_product;

			if (num_forest > 0) {
				T base = dF_dXcomb_tau * Y[trunk_flat];
				for (uint64_t k = 0; k < num_forest; ++k) {
					uint64_t fk_flat = cache.coproduct_data[forest_start + k];
					T partial = base;
					for (uint64_t j = 0; j < num_forest; ++j) {
						if (j != k)
							partial *= X_prev[cache.coproduct_data[forest_start + j]];
					}
					dF_dX[fk_flat] += partial;
				}
			}
		}
	}
	// dF_dX now contains dF/dX_prev (initial copy + accumulated forest terms)
}


// Convert tree-level derivatives to increment derivatives.
// Y[τ] = (∏ increment[labels[j]]) / γ(τ)
// dF/d(increment[d]) = Σ_τ dF/dY[τ] * (1/γ(τ)) * Σ_{j:label[j]=d} ∏_{k≠j} increment[label[k]]
template<std::floating_point T>
void linear_bsig_deriv_to_increment_deriv_(
	const T* dF_dY,
	const T* increment,
	T* inc_derivs,
	uint64_t dimension,
	const BranchedSigCache& cache
) {
	uint64_t num_trees = cache.total_length - 1;
	const uint8_t* labels = cache.node_labels_data.data();
	const uint64_t* offsets = cache.node_labels_offsets.data();

	std::memset(inc_derivs, 0, dimension * sizeof(T));

	for (uint64_t i = 0; i < num_trees; ++i) {
		T dF_dYi = dF_dY[i + 1];
		if (dF_dYi == static_cast<T>(0)) continue;

		T inv_gamma = static_cast<T>(cache.inv_tree_factorial[i]);
		uint64_t lstart = offsets[i];
		uint64_t lend = offsets[i + 1];
		uint64_t n_labels = lend - lstart;

		// Leave-one-out product via prefix accumulation
		T base = inv_gamma * dF_dYi;
		T prefix = static_cast<T>(1);
		for (uint64_t j = 0; j < n_labels; ++j) {
			T suffix = static_cast<T>(1);
			for (uint64_t k = j + 1; k < n_labels; ++k)
				suffix *= increment[labels[lstart + k]];
			inc_derivs[labels[lstart + j]] += base * prefix * suffix;
			prefix *= increment[labels[lstart + j]];
		}
	}
}


// Main backward loop.
template<std::floating_point T>
void branched_sig_backprop_inplace_(
	const Path<T>& path,
	T* out,
	T* bsig_derivs,
	T* bsig,
	T* increment,
	T* temp_Y,
	T* local_derivs,
	T* inc_derivs,
	const BranchedSigCache& cache
) {
	uint64_t total_len = cache.total_length;
	uint64_t dim = path.dimension();
	uint64_t data_dim = path.data_dimension();
	uint64_t len = path.length();

	if (len <= 1) return;

	int steps = static_cast<int>(len - 1);

	if (!path.lead_lag()) {
		for (int seg = steps - 1; seg >= 0; --seg) {
			auto p0 = path[seg];
			auto p1 = path[seg + 1];
			for (uint64_t d = 0; d < dim; ++d)
				increment[d] = p1[d] - p0[d];

			linear_branched_sig_(increment, temp_Y, cache);

			if (seg > 0)
				butcher_uncombine_inplace_(bsig, temp_Y, cache);

			if (seg > 0)
				butcher_product_deriv_(bsig, temp_Y, bsig_derivs, local_derivs, cache);
			else
				std::memcpy(local_derivs, bsig_derivs, total_len * sizeof(T));

			linear_bsig_deriv_to_increment_deriv_(local_derivs, increment, inc_derivs, dim, cache);

			T* pos = out + (seg + 1) * data_dim;
			T* neg = out + seg * data_dim;
			for (uint64_t d = 0; d < data_dim; ++d) {
				pos[d] += inc_derivs[d];
				neg[d] -= inc_derivs[d];
			}
		}
	}
	else {
		uint64_t data_length = path.data_length();
		T* pos_ptr = out + (data_length - 1) * data_dim;
		T* neg_ptr = pos_ptr - data_dim;
		bool parity = false;

		for (int seg = steps - 1; seg >= 0; --seg) {
			auto p0 = path[seg];
			auto p1 = path[seg + 1];
			for (uint64_t d = 0; d < dim; ++d)
				increment[d] = p1[d] - p0[d];

			linear_branched_sig_(increment, temp_Y, cache);

			if (seg > 0)
				butcher_uncombine_inplace_(bsig, temp_Y, cache);

			if (seg > 0)
				butcher_product_deriv_(bsig, temp_Y, bsig_derivs, local_derivs, cache);
			else
				std::memcpy(local_derivs, bsig_derivs, total_len * sizeof(T));

			linear_bsig_deriv_to_increment_deriv_(local_derivs, increment, inc_derivs, dim, cache);

			T* s = parity ? inc_derivs + data_dim : inc_derivs;
			for (uint64_t d = 0; d < data_dim; ++d) {
				pos_ptr[d] += s[d];
				neg_ptr[d] -= s[d];
			}
			if (parity) {
				pos_ptr -= data_dim;
				neg_ptr -= data_dim;
			}
			parity = !parity;
		}
	}
}


template<std::floating_point T>
void branched_sig_backprop_(
	const T* path,
	T* out,
	const T* bsig_derivs_in,
	const T* bsig_in,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.)
) {
	Path<T> path_obj(path, dimension, length, time_aug, lead_lag, end_time);
	uint64_t aug_dim = path_obj.dimension();
	const auto& cache = get_branched_sig_cache(aug_dim, max_nodes);
	uint64_t total_len = cache.total_length;

	// Working copies (backprop modifies these in-place)
	auto bsig_copy = std::make_unique<T[]>(total_len);
	auto derivs_copy = std::make_unique<T[]>(total_len);
	std::memcpy(bsig_copy.get(), bsig_in, total_len * sizeof(T));
	std::memcpy(derivs_copy.get(), bsig_derivs_in, total_len * sizeof(T));

	auto increment = std::make_unique<T[]>(aug_dim);
	auto temp_Y = std::make_unique<T[]>(total_len);
	auto local_derivs = std::make_unique<T[]>(total_len);
	auto inc_derivs = std::make_unique<T[]>(aug_dim);

	std::memset(out, 0, path_obj.data_length() * path_obj.data_dimension() * sizeof(T));

	branched_sig_backprop_inplace_(
		path_obj, out, derivs_copy.get(), bsig_copy.get(),
		increment.get(), temp_Y.get(), local_derivs.get(), inc_derivs.get(), cache);
}


template<std::floating_point T>
void batch_branched_sig_backprop_(
	const T* path,
	T* out,
	const T* bsig_derivs_in,
	const T* bsig_in,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.)
) {
	uint64_t aug_dim = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const auto& cache = get_branched_sig_cache(aug_dim, max_nodes);
	uint64_t total_len = cache.total_length;
	uint64_t flat_path_length = length * dimension;

	auto work = [&](uint64_t b, T* increment, T* temp_Y, T* local_derivs, T* inc_derivs) {
		Path<T> path_obj(path + b * flat_path_length, dimension, length, time_aug, lead_lag, end_time);

		auto bsig_copy = std::make_unique<T[]>(total_len);
		auto derivs_copy = std::make_unique<T[]>(total_len);
		std::memcpy(bsig_copy.get(), bsig_in + b * total_len, total_len * sizeof(T));
		std::memcpy(derivs_copy.get(), bsig_derivs_in + b * total_len, total_len * sizeof(T));

		T* out_ptr = out + b * flat_path_length;
		std::memset(out_ptr, 0, flat_path_length * sizeof(T));

		branched_sig_backprop_inplace_(
			path_obj, out_ptr, derivs_copy.get(), bsig_copy.get(),
			increment, temp_Y, local_derivs, inc_derivs, cache);
	};

	if (n_jobs == 1 || batch_size == 1) {
		auto increment = std::make_unique<T[]>(aug_dim);
		auto temp_Y = std::make_unique<T[]>(total_len);
		auto local_derivs = std::make_unique<T[]>(total_len);
		auto inc_derivs = std::make_unique<T[]>(aug_dim);
		for (uint64_t b = 0; b < batch_size; ++b)
			work(b, increment.get(), temp_Y.get(), local_derivs.get(), inc_derivs.get());
	}
	else {
		if (n_jobs == 0)
			throw std::invalid_argument("n_jobs cannot be 0");
		const unsigned int max_threads = n_jobs > 0
			? static_cast<unsigned int>(n_jobs)
			: get_max_threads() + 1 + n_jobs;
		if (max_threads < 1)
			throw std::invalid_argument("n_jobs too low");

		std::vector<std::thread> workers;
		for (unsigned int t = 0; t < max_threads && t < batch_size; ++t) {
			workers.emplace_back([&, t, max_threads]() {
				auto increment = std::make_unique<T[]>(aug_dim);
				auto temp_Y = std::make_unique<T[]>(total_len);
				auto local_derivs = std::make_unique<T[]>(total_len);
				auto inc_derivs = std::make_unique<T[]>(aug_dim);
				for (uint64_t b = t; b < batch_size; b += max_threads)
					work(b, increment.get(), temp_Y.get(), local_derivs.get(), inc_derivs.get());
			});
		}
		for (auto& w : workers) w.join();
	}
}
