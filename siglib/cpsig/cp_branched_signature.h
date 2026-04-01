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
	const T* path,
	T* out,
	T* increment,
	T* temp,
	uint64_t dimension,
	uint64_t length,
	const BranchedSigCache& cache
) {
	uint64_t total_len = cache.total_length;

	if (length <= 1) {
		out[0] = static_cast<T>(1.);
		std::memset(out + 1, 0, (total_len - 1) * sizeof(T));
		return;
	}

	for (uint64_t d = 0; d < dimension; ++d) {
		increment[d] = path[dimension + d] - path[d];
	}
	linear_branched_sig_(increment, out, cache);

	for (uint64_t seg = 1; seg < length - 1; ++seg) {
		const T* seg_start = path + seg * dimension;
		const T* seg_end = path + (seg + 1) * dimension;

		for (uint64_t d = 0; d < dimension; ++d) {
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
	uint64_t max_nodes
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes);
	auto increment = std::make_unique<T[]>(dimension);
	auto temp = std::make_unique<T[]>(cache.total_length);
	branched_signature_with_buffers_(path, out, increment.get(), temp.get(), dimension, length, cache);
}


template<std::floating_point T>
void batch_branched_signature_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes);
	uint64_t total_len = cache.total_length;
	uint64_t flat_path_length = length * dimension;

	if (n_jobs == 1 || batch_size == 1) {
		auto increment = std::make_unique<T[]>(dimension);
		auto temp = std::make_unique<T[]>(total_len);
		const T* path_ptr = path;
		T* out_ptr = out;
		for (uint64_t b = 0; b < batch_size; ++b) {
			branched_signature_with_buffers_(path_ptr, out_ptr, increment.get(), temp.get(),
				dimension, length, cache);
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
		// Each thread allocates buffers once and reuses them across all assigned paths
		for (unsigned int t = 0; t < max_threads && t < batch_size; ++t) {
			workers.emplace_back([&, t, max_threads]() {
				auto increment = std::make_unique<T[]>(dimension);
				auto temp = std::make_unique<T[]>(total_len);
				for (uint64_t b = t; b < batch_size; b += max_threads) {
					branched_signature_with_buffers_(
						path + b * flat_path_length,
						out + b * total_len,
						increment.get(), temp.get(),
						dimension, length, cache);
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
