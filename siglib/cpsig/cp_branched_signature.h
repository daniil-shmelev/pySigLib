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
#include "cache_lifecycle/cp_branched_cache.h"
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

template<std::floating_point T>
void branched_hopf_convolution_(
	const T* X,
	const T* Y,
	T* out,
	const BranchedSigCache& cache
) {
	out[0] = X[0] * Y[0];

	for (uint64_t order = 1; order <= cache.max_nodes; ++order) {
		uint64_t start = cache.order_index[order];
		uint64_t end = cache.order_index[order + 1];

		for (uint64_t tree_idx = start; tree_idx < end; ++tree_idx) {
			uint64_t flat_idx = tree_idx + 1;
			T new_val = X[flat_idx] * Y[0] + X[0] * Y[flat_idx];

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

			out[flat_idx] = new_val;
		}
	}
}

template<std::floating_point T>
void branched_hopf_convolution_deriv_(
	const T* X,
	const T* Y,
	const T* d_out,
	T* d_X,
	T* d_Y,
	const BranchedSigCache& cache
) {
	d_X[0] += d_out[0] * Y[0];
	d_Y[0] += d_out[0] * X[0];

	uint64_t num_trees = cache.total_length - 1;
	for (uint64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
		uint64_t flat_idx = tree_idx + 1;
		T d = d_out[flat_idx];
		if (d == static_cast<T>(0)) continue;

		d_X[flat_idx] += d * Y[0];
		d_Y[0] += d * X[flat_idx];
		d_X[0] += d * Y[flat_idx];
		d_Y[flat_idx] += d * X[0];

		uint64_t pos = cache.coproduct_offsets[tree_idx];
		uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];

		while (pos < pos_end) {
			uint64_t num_forest = cache.coproduct_data[pos++];
			uint64_t trunk_flat = cache.coproduct_data[pos++];
			uint64_t forest_start = pos;
			T forest_product = static_cast<T>(1);

			for (uint64_t j = 0; j < num_forest; ++j) {
				forest_product *= X[cache.coproduct_data[pos++]];
			}

			d_Y[trunk_flat] += d * forest_product;
			for (uint64_t k = 0; k < num_forest; ++k) {
				uint64_t fk_flat = cache.coproduct_data[forest_start + k];
				T partial = d * Y[trunk_flat];
				for (uint64_t j = 0; j < num_forest; ++j) {
					if (j != k)
						partial *= X[cache.coproduct_data[forest_start + j]];
				}
				d_X[fk_flat] += partial;
			}
		}
	}
}

template<std::floating_point T>
void build_correction_base_(
	T* out,
	const T* correction,
	uint64_t correction_len,
	uint64_t data_dimension,
	const BranchedSigCache& cache
) {
	std::memset(out, 0, cache.total_length * sizeof(T));
	if (correction == nullptr || correction_len == 0)
		return;

	uint64_t offset = 0;
	uint64_t level_size = data_dimension;
	for (uint64_t level = 2; level <= cache.max_nodes; ++level) {
		level_size *= data_dimension;
		if (offset + level_size > correction_len)
			break;

		for (uint64_t word_idx = 0; word_idx < level_size; ++word_idx) {
			const T value = correction[offset + word_idx];
			if (value == static_cast<T>(0))
				continue;

			uint64_t tmp = word_idx;
			uint64_t aug_word_idx = 0;
			uint64_t pow = level_size / data_dimension;
			for (uint64_t pos = level; pos > 0; --pos) {
				const uint64_t label = tmp / pow;
				tmp -= label * pow;
				if (pos > 1)
					pow /= data_dimension;

				aug_word_idx = aug_word_idx * cache.dimension + label;
			}

			uint64_t idx = cache.chain_indices[cache.chain_index_offsets[level] + aug_word_idx];
			if (idx != 0)
				out[idx] += value;
		}
		offset += level_size;
	}
}

template<std::floating_point T>
void branched_correction_(
	const T* increment,
	T* out,
	const T* correction,
	uint64_t correction_len,
	uint64_t data_dimension,
	const BranchedSigCache& cache
) {
	build_correction_base_(out, correction, correction_len, data_dimension, cache);

	if (cache.max_nodes >= 1) {
		for (uint64_t d = 0; d < cache.dimension; ++d) {
			out[cache.order_index[1] + d + 1] = increment[d];
		}
	}
}

template<std::floating_point T>
void branched_hopf_exp_(
	const T* local_log,
	T* out,
	T* power,
	T* next_power,
	const BranchedSigCache& cache
) {
	const uint64_t total_len = cache.total_length;
	std::memset(out, 0, total_len * sizeof(T));
	out[0] = static_cast<T>(1);

	std::memcpy(power, local_log, total_len * sizeof(T));
	T inv_factorial = static_cast<T>(1);

	for (uint64_t k = 1; k <= cache.max_nodes; ++k) {
		inv_factorial /= static_cast<T>(k);
		for (uint64_t i = 0; i < total_len; ++i) {
			out[i] += inv_factorial * power[i];
		}

		if (k < cache.max_nodes) {
			branched_hopf_convolution_(power, local_log, next_power, cache);
			std::swap(power, next_power);
		}
	}
}

template<std::floating_point T>
void local_correction_branched_sig_(
	const T* increment,
	T* out,
	T* local_log,
	T* power,
	T* next_power,
	const T* correction,
	uint64_t correction_len,
	uint64_t data_dimension,
	const BranchedSigCache& cache
) {
	if (cache.max_nodes <= 2) {
		linear_branched_sig_(increment, out, cache);
		build_correction_base_(local_log, correction, correction_len, data_dimension, cache);
		for (uint64_t i = 1; i < cache.total_length; ++i)
			out[i] += local_log[i];
		return;
	}

	branched_correction_(increment, local_log, correction, correction_len, data_dimension, cache);
	branched_hopf_exp_(local_log, out, power, next_power, cache);
}

// Processes trees from highest order down to order 1 so that forest
// references (always lower-order) use un-updated X values.
template<std::floating_point T>
void butcher_product_inplace_(
	T* X,
	const T* Y,
	const BranchedSigCache& cache
) {
	const uint64_t* order_index = cache.order_index.data();
	const uint64_t* coproduct_offsets = cache.coproduct_offsets.data();
	const uint64_t* coproduct_data = cache.coproduct_data.data();

	if (cache.planar) {
		for (int64_t order = static_cast<int64_t>(cache.max_nodes); order >= 1; --order) {
			uint64_t start = order_index[order];
			uint64_t end = order_index[order + 1];

			for (uint64_t tree_idx = start; tree_idx < end; ++tree_idx) {
				uint64_t flat_idx = tree_idx + 1;
				T new_val = X[flat_idx] + Y[flat_idx];

				uint64_t pos = coproduct_offsets[tree_idx];
				uint64_t pos_end = coproduct_offsets[tree_idx + 1];

				while (pos < pos_end) {
					uint64_t has_forest = coproduct_data[pos++];
					uint64_t trunk_flat = coproduct_data[pos++];
					T term = Y[trunk_flat];

					if (has_forest) {
						term *= X[coproduct_data[pos++]];
					}

					new_val += term;
				}

				X[flat_idx] = new_val;
			}
		}
		return;
	}

	for (int64_t order = static_cast<int64_t>(cache.max_nodes); order >= 1; --order) {
		uint64_t start = order_index[order];
		uint64_t end = order_index[order + 1];

		for (uint64_t tree_idx = start; tree_idx < end; ++tree_idx) {
			uint64_t flat_idx = tree_idx + 1;
			T new_val = X[flat_idx] + Y[flat_idx];

			uint64_t pos = coproduct_offsets[tree_idx];
			uint64_t pos_end = coproduct_offsets[tree_idx + 1];

			while (pos < pos_end) {
				uint64_t num_forest = coproduct_data[pos++];
				uint64_t trunk_flat = coproduct_data[pos++];
				T term = Y[trunk_flat];

				for (uint64_t j = 0; j < num_forest; ++j) {
					term *= X[coproduct_data[pos++]];
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
	T* local_log,
	T* power,
	T* next_power,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_segment_stride,
	bool has_correction,
	const BranchedSigCache& cache
) {
	uint64_t total_len = cache.total_length;
	uint64_t dim = path.dimension();
	uint64_t data_dim = path.data_dimension();
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
	if (!has_correction) {
		linear_branched_sig_(increment, out, cache);
	} else {
		local_correction_branched_sig_(increment, out, local_log, power, next_power,
			correction, correction_len, data_dim, cache);
	}

	for (uint64_t seg = 1; seg < len - 1; ++seg) {
		auto seg_start = path[seg];
		auto seg_end = path[seg + 1];

		for (uint64_t d = 0; d < dim; ++d) {
			increment[d] = seg_end[d] - seg_start[d];
		}

		if (!has_correction) {
			linear_branched_sig_(increment, temp, cache);
		} else {
			const T* seg_correction = correction + seg * correction_segment_stride;
			local_correction_branched_sig_(increment, temp, local_log, power, next_power,
				seg_correction, correction_len, data_dim, cache);
		}
		butcher_product_inplace_(out, temp, cache);
	}
}


template<std::floating_point T>
void branched_signature_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.),
	bool planar = false,
	bool scalar_term = true,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");
	const bool has_correction = correction_len != 0;
	uint64_t aug_dim = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const auto& cache = get_branched_sig_cache(aug_dim, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t flat_path_length = length * dimension;
	uint64_t out_stride = scalar_term ? total_len : total_len - 1;

	auto compute_one = [&](const T* path_ptr, T* out_ptr, T* increment, T* temp,
		T* local_log, T* power, T* next_power, const T* correction_ptr) {
		Path<T> path_obj(path_ptr, dimension, length, time_aug, lead_lag, end_time);
		if (scalar_term) {
			branched_signature_with_buffers_(path_obj, out_ptr, increment, temp,
				local_log, power, next_power, correction_ptr, correction_len,
				correction_segment_stride, has_correction, cache);
		} else {
			std::vector<T> buf(total_len);
			branched_signature_with_buffers_(path_obj, buf.data(), increment, temp,
				local_log, power, next_power, correction_ptr, correction_len,
				correction_segment_stride, has_correction, cache);
			std::memcpy(out_ptr, buf.data() + 1, (total_len - 1) * sizeof(T));
		}
	};

	if (n_jobs == 1 || batch_size == 1) {
		auto increment = std::make_unique<T[]>(aug_dim);
		auto temp = std::make_unique<T[]>(total_len);
		std::unique_ptr<T[]> local_log;
		std::unique_ptr<T[]> power;
		std::unique_ptr<T[]> next_power;
		if (has_correction) {
			local_log = std::make_unique<T[]>(total_len);
			power = std::make_unique<T[]>(total_len);
			next_power = std::make_unique<T[]>(total_len);
		}
		const T* path_ptr = path;
		T* out_ptr = out;
		for (uint64_t b = 0; b < batch_size; ++b) {
			const T* correction_ptr = has_correction ? correction + b * correction_batch_stride : nullptr;
			compute_one(path_ptr, out_ptr, increment.get(), temp.get(),
				local_log.get(), power.get(), next_power.get(), correction_ptr);
			path_ptr += flat_path_length;
			out_ptr += out_stride;
		}
	}
	else {
		auto work_range = [&](uint64_t start, uint64_t end) {
			auto increment = std::make_unique<T[]>(aug_dim);
			auto temp = std::make_unique<T[]>(total_len);
			std::unique_ptr<T[]> local_log;
			std::unique_ptr<T[]> power;
			std::unique_ptr<T[]> next_power;
			if (has_correction) {
				local_log = std::make_unique<T[]>(total_len);
				power = std::make_unique<T[]>(total_len);
				next_power = std::make_unique<T[]>(total_len);
			}
			for (uint64_t b = start; b < end; ++b) {
				const T* correction_ptr = has_correction ? correction + b * correction_batch_stride : nullptr;
				compute_one(path + b * flat_path_length, out + b * out_stride,
					increment.get(), temp.get(), local_log.get(), power.get(), next_power.get(),
					correction_ptr);
			}
		};
		spawn_batch_threads(batch_size, n_jobs, work_range);
	}
}


template<std::floating_point T>
void branched_sig_combine_(
	const T* bsig1,
	const T* bsig2,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs,
	bool planar = false,
	bool scalar_term = true
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t stride = scalar_term ? total_len : total_len - 1;

	if (scalar_term) {
		auto thread_func = [&](const T* sig1_ptr, const T* sig2_ptr, T* out_ptr) {
			std::memcpy(out_ptr, sig1_ptr, total_len * sizeof(T));
			butcher_product_inplace_(out_ptr, sig2_ptr, cache);
		};

		if (n_jobs == 1 || batch_size == 1) {
			for (uint64_t b = 0; b < batch_size; ++b)
				thread_func(bsig1 + b * total_len, bsig2 + b * total_len, out + b * total_len);
		} else {
			multi_threaded_batch(thread_func, batch_size, n_jobs,
				make_batch(bsig1, total_len), make_batch(bsig2, total_len), make_batch(out, total_len));
		}
	} else {
		// Per-element copy: reconstruct full buffers, compute, strip
		auto thread_func = [&](const T* sig1_ptr, const T* sig2_ptr, T* out_ptr) {
			std::vector<T> s1(total_len), s2(total_len), buf(total_len);
			s1[0] = static_cast<T>(1);
			std::memcpy(s1.data() + 1, sig1_ptr, (total_len - 1) * sizeof(T));
			s2[0] = static_cast<T>(1);
			std::memcpy(s2.data() + 1, sig2_ptr, (total_len - 1) * sizeof(T));
			std::memcpy(buf.data(), s1.data(), total_len * sizeof(T));
			butcher_product_inplace_(buf.data(), s2.data(), cache);
			std::memcpy(out_ptr, buf.data() + 1, (total_len - 1) * sizeof(T));
		};

		if (n_jobs == 1 || batch_size == 1) {
			for (uint64_t b = 0; b < batch_size; ++b)
				thread_func(bsig1 + b * stride, bsig2 + b * stride, out + b * stride);
		} else {
			multi_threaded_batch(thread_func, batch_size, n_jobs,
				make_batch(bsig1, stride), make_batch(bsig2, stride), make_batch(out, stride));
		}
	}
}


template<std::floating_point T>
void branched_sig_combine_backprop_(
	const T* bsig1,
	const T* bsig2,
	const T* derivs_in,
	T* out1,
	T* out2,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs,
	bool planar = false,
	bool scalar_term = true
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t stride = scalar_term ? total_len : total_len - 1;

	auto work = [&](uint64_t b) {
		if (scalar_term) {
			uint64_t off = b * total_len;
			std::memcpy(out1 + off, derivs_in + off, total_len * sizeof(T));
			butcher_product_deriv_(bsig1 + off, bsig2 + off, out1 + off, out2 + off, cache);
		} else {
			// Per-element copy: reconstruct full buffers
			std::vector<T> s1(total_len), s2(total_len), df(total_len);
			std::vector<T> d1(total_len), d2(total_len);
			uint64_t off = b * stride;
			s1[0] = static_cast<T>(1);
			std::memcpy(s1.data() + 1, bsig1 + off, (total_len - 1) * sizeof(T));
			s2[0] = static_cast<T>(1);
			std::memcpy(s2.data() + 1, bsig2 + off, (total_len - 1) * sizeof(T));
			df[0] = static_cast<T>(0);
			std::memcpy(df.data() + 1, derivs_in + off, (total_len - 1) * sizeof(T));
			std::memcpy(d1.data(), df.data(), total_len * sizeof(T));
			butcher_product_deriv_(s1.data(), s2.data(), d1.data(), d2.data(), cache);
			std::memcpy(out1 + off, d1.data() + 1, (total_len - 1) * sizeof(T));
			std::memcpy(out2 + off, d2.data() + 1, (total_len - 1) * sizeof(T));
		}
	};

	if (n_jobs == 1 || batch_size == 1) {
		for (uint64_t b = 0; b < batch_size; ++b)
			work(b);
	}
	else {
		auto work_range = [&](uint64_t start, uint64_t end) {
			for (uint64_t b = start; b < end; ++b)
				work(b);
		};
		spawn_batch_threads(batch_size, n_jobs, work_range);
	}
}


// =========================================================================
// Backpropagation
// =========================================================================

// Reverse the Butcher product: recover X_prev from X_combined and Y.
// X_combined[\tau] = X_prev[\tau] + Y[\tau] + \Sigma_cuts Y[trunk] * \prod X_prev[forest_i]
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

	// Initialize dF_dY = dF_dX (the Y[\tau] direct term)
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
// Y[\tau] = (\prod increment[labels[j]]) / \gamma(\tau)
// dF/d(increment[d]) = \Sigma_\tau dF/dY[\tau] * (1/\gamma(\tau)) * \Sigma_{j:label[j]=d} \prod_{k!=j} increment[label[k]]
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

template<std::floating_point T>
void local_log_bsig_deriv_to_increment_deriv_(
	const T* dF_dY,
	const T* increment,
	T* inc_derivs,
	T* local_log,
	T* powers,
	T* power_derivs,
	T* d_correction,
	const T* correction,
	uint64_t correction_len,
	uint64_t data_dimension,
	const BranchedSigCache& cache
) {
	if (cache.max_nodes <= 2) {
		linear_bsig_deriv_to_increment_deriv_(dF_dY, increment, inc_derivs, cache.dimension, cache);
		return;
	}

	uint64_t total_len = cache.total_length;
	std::memset(power_derivs, 0, cache.max_nodes * total_len * sizeof(T));
	std::memset(d_correction, 0, total_len * sizeof(T));

	branched_correction_(increment, local_log, correction, correction_len, data_dimension, cache);
	std::memcpy(powers, local_log, total_len * sizeof(T));
	for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
		branched_hopf_convolution_(powers + (k - 2) * total_len, local_log,
			powers + (k - 1) * total_len, cache);
	}

	T inv_factorial = static_cast<T>(1);
	for (uint64_t k = 1; k <= cache.max_nodes; ++k) {
		inv_factorial /= static_cast<T>(k);
		T* d_power = power_derivs + (k - 1) * total_len;
		for (uint64_t i = 1; i < total_len; ++i) {
			d_power[i] += inv_factorial * dF_dY[i];
		}
	}

	for (uint64_t k = cache.max_nodes; k > 1; --k) {
		branched_hopf_convolution_deriv_(powers + (k - 2) * total_len, local_log,
			power_derivs + (k - 1) * total_len,
			power_derivs + (k - 2) * total_len, d_correction, cache);
	}

	for (uint64_t i = 0; i < total_len; ++i) {
		d_correction[i] += power_derivs[i];
	}

	std::memset(inc_derivs, 0, cache.dimension * sizeof(T));
	for (uint64_t d = 0; d < cache.dimension; ++d) {
		inc_derivs[d] = d_correction[cache.order_index[1] + d + 1];
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
	T* local_log,
	T* power,
	T* next_power,
	T* powers,
	T* power_derivs,
	T* d_correction,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_segment_stride,
	bool has_correction,
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

			if (!has_correction) {
				linear_branched_sig_(increment, temp_Y, cache);
			} else {
				const T* seg_correction = correction + static_cast<uint64_t>(seg) * correction_segment_stride;
				local_correction_branched_sig_(increment, temp_Y, local_log, power, next_power,
					seg_correction, correction_len, data_dim, cache);
			}

			if (seg > 0)
				butcher_uncombine_inplace_(bsig, temp_Y, cache);

			if (seg > 0)
				butcher_product_deriv_(bsig, temp_Y, bsig_derivs, local_derivs, cache);
			else
				std::memcpy(local_derivs, bsig_derivs, total_len * sizeof(T));

			if (!has_correction) {
				linear_bsig_deriv_to_increment_deriv_(local_derivs, increment, inc_derivs, dim, cache);
			} else {
				const T* seg_correction = correction + static_cast<uint64_t>(seg) * correction_segment_stride;
				local_log_bsig_deriv_to_increment_deriv_(local_derivs, increment, inc_derivs,
					local_log, powers, power_derivs, d_correction, seg_correction,
					correction_len, data_dim, cache);
			}

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

			if (!has_correction) {
				linear_branched_sig_(increment, temp_Y, cache);
			} else {
				const T* seg_correction = correction + static_cast<uint64_t>(seg) * correction_segment_stride;
				local_correction_branched_sig_(increment, temp_Y, local_log, power, next_power,
					seg_correction, correction_len, data_dim, cache);
			}

			if (seg > 0)
				butcher_uncombine_inplace_(bsig, temp_Y, cache);

			if (seg > 0)
				butcher_product_deriv_(bsig, temp_Y, bsig_derivs, local_derivs, cache);
			else
				std::memcpy(local_derivs, bsig_derivs, total_len * sizeof(T));

			if (!has_correction) {
				linear_bsig_deriv_to_increment_deriv_(local_derivs, increment, inc_derivs, dim, cache);
			} else {
				const T* seg_correction = correction + static_cast<uint64_t>(seg) * correction_segment_stride;
				local_log_bsig_deriv_to_increment_deriv_(local_derivs, increment, inc_derivs,
					local_log, powers, power_derivs, d_correction, seg_correction,
					correction_len, data_dim, cache);
			}

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
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs,
	bool time_aug = false,
	bool lead_lag = false,
	T end_time = static_cast<T>(1.),
	bool planar = false,
	bool scalar_term = true,
	const T* correction = nullptr,
	uint64_t correction_len = 0,
	uint64_t correction_batch_stride = 0,
	uint64_t correction_segment_stride = 0
) {
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");
	const bool has_correction = correction_len != 0;
	uint64_t aug_dim = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const auto& cache = get_branched_sig_cache(aug_dim, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t flat_path_length = length * dimension;
	uint64_t in_stride = scalar_term ? total_len : total_len - 1;

	auto work = [&](uint64_t b, T* increment, T* temp_Y, T* local_derivs, T* inc_derivs,
		T* local_log, T* power, T* next_power, T* powers, T* power_derivs, T* d_correction) {
		Path<T> path_obj(path + b * flat_path_length, dimension, length, time_aug, lead_lag, end_time);

		auto bsig_copy = std::make_unique<T[]>(total_len);
		auto derivs_copy = std::make_unique<T[]>(total_len);

		if (scalar_term) {
			std::memcpy(bsig_copy.get(), bsig_in + b * total_len, total_len * sizeof(T));
			std::memcpy(derivs_copy.get(), bsig_derivs_in + b * total_len, total_len * sizeof(T));
		} else {
			bsig_copy[0] = static_cast<T>(1);
			std::memcpy(bsig_copy.get() + 1, bsig_in + b * in_stride, (total_len - 1) * sizeof(T));
			derivs_copy[0] = static_cast<T>(0);
			std::memcpy(derivs_copy.get() + 1, bsig_derivs_in + b * in_stride, (total_len - 1) * sizeof(T));
		}

		T* out_ptr = out + b * flat_path_length;
		std::memset(out_ptr, 0, flat_path_length * sizeof(T));

		branched_sig_backprop_inplace_(
			path_obj, out_ptr, derivs_copy.get(), bsig_copy.get(),
			increment, temp_Y, local_derivs, inc_derivs,
			local_log, power, next_power, powers, power_derivs, d_correction,
			has_correction ? correction + b * correction_batch_stride : nullptr,
			correction_len, correction_segment_stride, has_correction, cache);
	};

	if (n_jobs == 1 || batch_size == 1) {
		auto increment = std::make_unique<T[]>(aug_dim);
		auto temp_Y = std::make_unique<T[]>(total_len);
		auto local_derivs = std::make_unique<T[]>(total_len);
		auto inc_derivs = std::make_unique<T[]>(aug_dim);
		std::unique_ptr<T[]> local_log;
		std::unique_ptr<T[]> power;
		std::unique_ptr<T[]> next_power;
		std::unique_ptr<T[]> powers;
		std::unique_ptr<T[]> power_derivs;
		std::unique_ptr<T[]> d_correction;
		if (has_correction) {
			local_log = std::make_unique<T[]>(total_len);
			power = std::make_unique<T[]>(total_len);
			next_power = std::make_unique<T[]>(total_len);
			powers = std::make_unique<T[]>(max_nodes * total_len);
			power_derivs = std::make_unique<T[]>(max_nodes * total_len);
			d_correction = std::make_unique<T[]>(total_len);
		}
		for (uint64_t b = 0; b < batch_size; ++b)
			work(b, increment.get(), temp_Y.get(), local_derivs.get(), inc_derivs.get(),
				local_log.get(), power.get(), next_power.get(), powers.get(), power_derivs.get(), d_correction.get());
	}
	else {
		auto work_range = [&](uint64_t start, uint64_t end) {
			auto increment = std::make_unique<T[]>(aug_dim);
			auto temp_Y = std::make_unique<T[]>(total_len);
			auto local_derivs = std::make_unique<T[]>(total_len);
			auto inc_derivs = std::make_unique<T[]>(aug_dim);
			std::unique_ptr<T[]> local_log;
			std::unique_ptr<T[]> power;
			std::unique_ptr<T[]> next_power;
			std::unique_ptr<T[]> powers;
			std::unique_ptr<T[]> power_derivs;
			std::unique_ptr<T[]> d_correction;
			if (has_correction) {
				local_log = std::make_unique<T[]>(total_len);
				power = std::make_unique<T[]>(total_len);
				next_power = std::make_unique<T[]>(total_len);
				powers = std::make_unique<T[]>(max_nodes * total_len);
				power_derivs = std::make_unique<T[]>(max_nodes * total_len);
				d_correction = std::make_unique<T[]>(total_len);
			}
			for (uint64_t b = start; b < end; ++b)
				work(b, increment.get(), temp_Y.get(), local_derivs.get(), inc_derivs.get(),
					local_log.get(), power.get(), next_power.get(), powers.get(), power_derivs.get(), d_correction.get());
		};
		spawn_batch_threads(batch_size, n_jobs, work_range);
	}
}
