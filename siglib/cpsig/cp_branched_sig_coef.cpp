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

#include "cppch.h"
#include "cpsig.h"
#include "cp_branched_sig_coef.h"
#include "cp_branched_cache.h"
#include "cp_branched_sig_coef_cache.h"
#include "cp_path.h"
#include "multithreading.h"
#include "macros.h"

template<std::floating_point T>
FORCE_INLINE void sparse_linear_branched_sig_(
	const T* increment,
	T* out,
	const BranchedSigCoefCache& cache
) {
	out[0] = static_cast<T>(1);
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t* offsets = cache.node_labels_offsets.data();
	const uint8_t* labels = cache.node_labels_data.data();
	for (uint64_t local = 1; local < cache_size; ++local) {
		const uint64_t start = offsets[local];
		const uint64_t num_labels = offsets[local + 1] - start;
		T product;
		switch (num_labels) {
		case 1:
			product = increment[labels[start]];
			break;
		case 2:
			product = increment[labels[start]] * increment[labels[start + 1]];
			break;
		case 3:
			product = increment[labels[start]] * increment[labels[start + 1]]
				* increment[labels[start + 2]];
			break;
		case 4:
			product = increment[labels[start]] * increment[labels[start + 1]]
				* increment[labels[start + 2]] * increment[labels[start + 3]];
			break;
		default:
			product = static_cast<T>(1);
			for (uint64_t pos = start; pos < offsets[local + 1]; ++pos)
				product *= increment[labels[pos]];
		}
		out[local] = product * static_cast<T>(cache.inv_tree_factorial[local]);
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_hopf_convolution_(
	const T* X,
	const T* Y,
	T* out,
	const BranchedSigCoefCache& cache
) {
	out[0] = X[0] * Y[0];
	for (uint64_t local = 1; local < cache.inv_tree_factorial.size(); ++local) {
		T value = X[local] * Y[0] + X[0] * Y[local];
		uint64_t pos = cache.coproduct_offsets[local];
		const uint64_t pos_end = cache.coproduct_offsets[local + 1];
		while (pos < pos_end) {
			const uint64_t num_forest = cache.coproduct_data[pos++];
			const uint64_t trunk = cache.coproduct_data[pos++];
			T term = Y[trunk];
			for (uint64_t j = 0; j < num_forest; ++j)
				term *= X[cache.coproduct_data[pos++]];
			value += term;
		}
		out[local] = value;
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_butcher_product_inplace_(
	T* X,
	const T* Y,
	const BranchedSigCoefCache& cache
) {
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t* offsets = cache.coproduct_offsets.data();
	const uint64_t* data = cache.coproduct_data.data();
	for (uint64_t local = cache_size; local-- > 1;) {
		T value = X[local] + Y[local];
		uint64_t pos = offsets[local];
		const uint64_t pos_end = offsets[local + 1];
		while (pos < pos_end) {
			const uint64_t num_forest = data[pos++];
			const uint64_t trunk = data[pos++];
			switch (num_forest) {
			case 0:
				value += Y[trunk];
				break;
			case 1:
				value += Y[trunk] * X[data[pos++]];
				break;
			case 2:
				value += Y[trunk] * X[data[pos]] * X[data[pos + 1]];
				pos += 2;
				break;
			case 3:
				value += Y[trunk] * X[data[pos]] * X[data[pos + 1]] * X[data[pos + 2]];
				pos += 3;
				break;
			default: {
				T term = Y[trunk];
				for (uint64_t j = 0; j < num_forest; ++j)
					term *= X[data[pos++]];
				value += term;
			}
			}
		}
		X[local] = value;
	}
}

template<std::floating_point T>
FORCE_INLINE void build_sparse_local_log_(
	const T* increment,
	const T* correction,
	uint64_t correction_len,
	T* local_log,
	const BranchedSigCoefCache& cache
) {
	std::fill(local_log, local_log + cache.inv_tree_factorial.size(), static_cast<T>(0));
	for (uint64_t d = 0; d < cache.leaf_indices.size(); ++d) {
		const uint64_t local = cache.leaf_indices[d];
		if (local != 0)
			local_log[local] = increment[d];
	}
	for (const auto& [correction_idx, local] : cache.correction_indices) {
		if (correction_idx >= correction_len)
			break;
		local_log[local] += correction[correction_idx];
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_local_branched_sig_(
	const T* increment,
	T* out,
	T* local_log,
	T* power,
	T* next_power,
	const T* correction,
	uint64_t correction_len,
	bool has_correction,
	const BranchedSigCoefCache& cache
) {
	if (!has_correction) {
		sparse_linear_branched_sig_(increment, out, cache);
		return;
	}

	if (cache.max_nodes <= 2) {
		sparse_linear_branched_sig_(increment, out, cache);
		for (const auto& [correction_idx, local] : cache.correction_indices) {
			if (correction_idx >= correction_len)
				break;
			out[local] += correction[correction_idx];
		}
		return;
	}

	const uint64_t cache_size = cache.inv_tree_factorial.size();
	build_sparse_local_log_(increment, correction, correction_len, local_log, cache);
	std::fill(out, out + cache_size, static_cast<T>(0));
	out[0] = static_cast<T>(1);
	std::copy(local_log, local_log + cache_size, power);

	T inv_factorial = static_cast<T>(1);
	for (uint64_t k = 1; k <= cache.max_nodes; ++k) {
		inv_factorial /= static_cast<T>(k);
		for (uint64_t local = 1; local < cache_size; ++local)
			out[local] += inv_factorial * power[local];
		if (k < cache.max_nodes) {
			sparse_hopf_convolution_(power, local_log, next_power, cache);
			std::swap(power, next_power);
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_branched_signature_with_buffers_(
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
	const BranchedSigCoefCache& cache
) {
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t dimension = path.dimension();
	const uint64_t length = path.length();
	if (length <= 1) {
		std::fill(out, out + cache_size, static_cast<T>(0));
		out[0] = static_cast<T>(1);
		return;
	}

	auto p0 = path[0];
	auto p1 = path[1];
	for (uint64_t d = 0; d < dimension; ++d)
		increment[d] = p1[d] - p0[d];
	sparse_local_branched_sig_(increment, out, local_log, power, next_power,
		correction, correction_len, has_correction, cache);

	for (uint64_t segment = 1; segment < length - 1; ++segment) {
		auto segment_start = path[segment];
		auto segment_end = path[segment + 1];
		for (uint64_t d = 0; d < dimension; ++d)
			increment[d] = segment_end[d] - segment_start[d];
		const T* segment_correction = has_correction
			? correction + segment * correction_segment_stride : nullptr;
		sparse_local_branched_sig_(increment, temp, local_log, power, next_power,
			segment_correction, correction_len, has_correction, cache);
		sparse_butcher_product_inplace_(out, temp, cache);
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_butcher_uncombine_inplace_(
	T* X,
	const T* Y,
	const BranchedSigCoefCache& cache
) {
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t* offsets = cache.coproduct_offsets.data();
	const uint64_t* data = cache.coproduct_data.data();
	for (uint64_t local = 1; local < cache_size; ++local) {
		T value = X[local] - Y[local];
		uint64_t pos = offsets[local];
		const uint64_t pos_end = offsets[local + 1];
		while (pos < pos_end) {
			const uint64_t num_forest = data[pos++];
			const uint64_t trunk = data[pos++];
			switch (num_forest) {
			case 0:
				value -= Y[trunk];
				break;
			case 1:
				value -= Y[trunk] * X[data[pos++]];
				break;
			case 2:
				value -= Y[trunk] * X[data[pos]] * X[data[pos + 1]];
				pos += 2;
				break;
			case 3:
				value -= Y[trunk] * X[data[pos]] * X[data[pos + 1]] * X[data[pos + 2]];
				pos += 3;
				break;
			default: {
				T term = Y[trunk];
				for (uint64_t j = 0; j < num_forest; ++j)
					term *= X[data[pos++]];
				value -= term;
			}
			}
		}
		X[local] = value;
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_butcher_product_deriv_(
	const T* X,
	const T* Y,
	T* d_X,
	T* d_Y,
	const BranchedSigCoefCache& cache
) {
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t* offsets = cache.coproduct_offsets.data();
	const uint64_t* data = cache.coproduct_data.data();
	d_Y[0] = static_cast<T>(0);
	std::copy(d_X + 1, d_X + cache_size, d_Y + 1);

	for (uint64_t local = 1; local < cache_size; ++local) {
		const T d_out = d_X[local];
		if (d_out == static_cast<T>(0))
			continue;
		uint64_t pos = offsets[local];
		const uint64_t pos_end = offsets[local + 1];
		while (pos < pos_end) {
			const uint64_t num_forest = data[pos++];
			const uint64_t trunk = data[pos++];
			if (num_forest == 0) {
				d_Y[trunk] += d_out;
				continue;
			}
			if (num_forest == 1) {
				const uint64_t a = data[pos++];
				d_Y[trunk] += d_out * X[a];
				d_X[a] += d_out * Y[trunk];
				continue;
			}
			if (num_forest == 2) {
				const uint64_t a = data[pos++];
				const uint64_t b = data[pos++];
				d_Y[trunk] += d_out * X[a] * X[b];
				const T base = d_out * Y[trunk];
				d_X[a] += base * X[b];
				d_X[b] += base * X[a];
				continue;
			}
			if (num_forest == 3) {
				const uint64_t a = data[pos++];
				const uint64_t b = data[pos++];
				const uint64_t c = data[pos++];
				d_Y[trunk] += d_out * X[a] * X[b] * X[c];
				const T base = d_out * Y[trunk];
				d_X[a] += base * X[b] * X[c];
				d_X[b] += base * X[a] * X[c];
				d_X[c] += base * X[a] * X[b];
				continue;
			}
			const uint64_t forest_start = pos;
			T forest_product = static_cast<T>(1);
			for (uint64_t j = 0; j < num_forest; ++j)
				forest_product *= X[data[pos++]];
			d_Y[trunk] += d_out * forest_product;

			const T base = d_out * Y[trunk];
			for (uint64_t k = 0; k < num_forest; ++k) {
				T partial = base;
				for (uint64_t j = 0; j < num_forest; ++j) {
					if (j != k)
						partial *= X[data[forest_start + j]];
				}
				d_X[data[forest_start + k]] += partial;
			}
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_hopf_convolution_deriv_(
	const T* X,
	const T* Y,
	const T* d_out,
	T* d_X,
	T* d_Y,
	const BranchedSigCoefCache& cache
) {
	d_X[0] += d_out[0] * Y[0];
	d_Y[0] += d_out[0] * X[0];
	for (uint64_t local = 1; local < cache.inv_tree_factorial.size(); ++local) {
		const T d = d_out[local];
		if (d == static_cast<T>(0))
			continue;
		d_X[local] += d * Y[0];
		d_Y[0] += d * X[local];
		d_X[0] += d * Y[local];
		d_Y[local] += d * X[0];

		uint64_t pos = cache.coproduct_offsets[local];
		const uint64_t pos_end = cache.coproduct_offsets[local + 1];
		while (pos < pos_end) {
			const uint64_t num_forest = cache.coproduct_data[pos++];
			const uint64_t trunk = cache.coproduct_data[pos++];
			const uint64_t forest_start = pos;
			T forest_product = static_cast<T>(1);
			for (uint64_t j = 0; j < num_forest; ++j)
				forest_product *= X[cache.coproduct_data[pos++]];
			d_Y[trunk] += d * forest_product;
			for (uint64_t k = 0; k < num_forest; ++k) {
				T partial = d * Y[trunk];
				for (uint64_t j = 0; j < num_forest; ++j) {
					if (j != k)
						partial *= X[cache.coproduct_data[forest_start + j]];
				}
				d_X[cache.coproduct_data[forest_start + k]] += partial;
			}
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_linear_bsig_deriv_to_increment_(
	const T* d_Y,
	const T* increment,
	T* increment_derivs,
	T* prefix_products,
	const BranchedSigCoefCache& cache
) {
	std::fill(increment_derivs, increment_derivs + cache.leaf_indices.size(), static_cast<T>(0));
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t* offsets = cache.node_labels_offsets.data();
	const uint8_t* labels = cache.node_labels_data.data();
	for (uint64_t local = 1; local < cache_size; ++local) {
		const T d = d_Y[local];
		if (d == static_cast<T>(0))
			continue;
		const uint64_t start = offsets[local];
		const uint64_t end = offsets[local + 1];
		const uint64_t num_labels = end - start;
		const T base = d * static_cast<T>(cache.inv_tree_factorial[local]);
		if (num_labels == 1) {
			increment_derivs[labels[start]] += base;
			continue;
		}
		if (num_labels == 2) {
			const uint64_t a = labels[start];
			const uint64_t b = labels[start + 1];
			increment_derivs[a] += base * increment[b];
			increment_derivs[b] += base * increment[a];
			continue;
		}
		if (num_labels == 3) {
			const uint64_t a = labels[start];
			const uint64_t b = labels[start + 1];
			const uint64_t c = labels[start + 2];
			increment_derivs[a] += base * increment[b] * increment[c];
			increment_derivs[b] += base * increment[a] * increment[c];
			increment_derivs[c] += base * increment[a] * increment[b];
			continue;
		}
		if (num_labels == 4) {
			const uint64_t a = labels[start];
			const uint64_t b = labels[start + 1];
			const uint64_t c = labels[start + 2];
			const uint64_t e = labels[start + 3];
			increment_derivs[a] += base * increment[b] * increment[c] * increment[e];
			increment_derivs[b] += base * increment[a] * increment[c] * increment[e];
			increment_derivs[c] += base * increment[a] * increment[b] * increment[e];
			increment_derivs[e] += base * increment[a] * increment[b] * increment[c];
			continue;
		}
		prefix_products[0] = static_cast<T>(1);
		for (uint64_t j = 0; j < num_labels; ++j)
			prefix_products[j + 1] = prefix_products[j] * increment[labels[start + j]];

		T suffix = static_cast<T>(1);
		for (uint64_t j = num_labels; j-- > 0;) {
			const uint64_t label = labels[start + j];
			increment_derivs[label] += base * prefix_products[j] * suffix;
			suffix *= increment[label];
		}
	}
}

template<std::floating_point T>
FORCE_INLINE void sparse_local_bsig_deriv_to_increment_(
	const T* d_Y,
	const T* increment,
	T* increment_derivs,
	T* prefix_products,
	T* local_log,
	T* powers,
	T* power_derivs,
	T* d_local_log,
	const T* correction,
	uint64_t correction_len,
	bool has_correction,
	const BranchedSigCoefCache& cache
) {
	if (!has_correction || cache.max_nodes <= 2) {
		sparse_linear_bsig_deriv_to_increment_(d_Y, increment, increment_derivs,
			prefix_products, cache);
		return;
	}

	const uint64_t cache_size = cache.inv_tree_factorial.size();
	build_sparse_local_log_(increment, correction, correction_len, local_log, cache);
	std::copy(local_log, local_log + cache_size, powers);
	for (uint64_t k = 2; k <= cache.max_nodes; ++k)
		sparse_hopf_convolution_(powers + (k - 2) * cache_size, local_log,
			powers + (k - 1) * cache_size, cache);

	std::fill(power_derivs, power_derivs + cache.max_nodes * cache_size, static_cast<T>(0));
	std::fill(d_local_log, d_local_log + cache_size, static_cast<T>(0));
	T inv_factorial = static_cast<T>(1);
	for (uint64_t k = 1; k <= cache.max_nodes; ++k) {
		inv_factorial /= static_cast<T>(k);
		T* d_power = power_derivs + (k - 1) * cache_size;
		for (uint64_t local = 1; local < cache_size; ++local)
			d_power[local] += inv_factorial * d_Y[local];
	}

	for (uint64_t k = cache.max_nodes; k > 1; --k) {
		sparse_hopf_convolution_deriv_(powers + (k - 2) * cache_size, local_log,
			power_derivs + (k - 1) * cache_size,
			power_derivs + (k - 2) * cache_size, d_local_log, cache);
	}
	for (uint64_t local = 0; local < cache_size; ++local)
		d_local_log[local] += power_derivs[local];

	std::fill(increment_derivs, increment_derivs + cache.leaf_indices.size(), static_cast<T>(0));
	for (uint64_t d = 0; d < cache.leaf_indices.size(); ++d) {
		const uint64_t local = cache.leaf_indices[d];
		if (local != 0)
			increment_derivs[d] = d_local_log[local];
	}
}

template<std::floating_point T>
void branched_sig_coef_(
	const T* path,
	T* out,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs,
	bool time_aug,
	bool lead_lag,
	T end_time,
	bool planar,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	if (dimension == 0)
		throw std::invalid_argument("branched_sig_coef received path of dimension 0");
	if (lead_lag && length == 0)
		throw std::invalid_argument("lead_lag requires a path with at least one point");
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");

	const bool has_correction = correction_len != 0;
	const uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const auto& cache = get_branched_sig_coef_cache(tree_data, tree_data_len, dimension,
		aug_dimension, max_nodes, planar);
	const uint64_t num_indices = cache.target_indices.size();
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t flat_path_length = length * dimension;

	auto worker = [&](uint64_t start, uint64_t end) {
		std::vector<T> increment(aug_dimension);
		std::vector<T> state(cache_size);
		std::vector<T> temp(cache_size);
		std::vector<T> local_log(has_correction ? cache_size : 0);
		std::vector<T> power(has_correction ? cache_size : 0);
		std::vector<T> next_power(has_correction ? cache_size : 0);
		for (uint64_t batch = start; batch < end; ++batch) {
			Path<T> path_obj(path + batch * flat_path_length, dimension, length,
				time_aug, lead_lag, end_time);
			const T* batch_correction = has_correction
				? correction + batch * correction_batch_stride : nullptr;
			sparse_branched_signature_with_buffers_(path_obj, state.data(), increment.data(),
				temp.data(), local_log.data(), power.data(), next_power.data(), batch_correction,
				correction_len, correction_segment_stride, has_correction, cache);
			T* out_ptr = out + batch * num_indices;
			for (uint64_t i = 0; i < num_indices; ++i)
				out_ptr[i] = state[cache.target_indices[i]];
		}
	};

	if (batch_size == 0 || n_jobs == 1 || batch_size == 1)
		worker(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, worker);
}

template<std::floating_point T>
void branched_sig_coef_backprop_(
	const T* path,
	T* out,
	const T* coefs,
	const T* derivs,
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	uint64_t max_nodes,
	int n_jobs,
	bool time_aug,
	bool lead_lag,
	T end_time,
	bool planar,
	const T* correction,
	uint64_t correction_len,
	uint64_t correction_batch_stride,
	uint64_t correction_segment_stride
) {
	if (dimension == 0)
		throw std::invalid_argument("branched_sig_coef_backprop received path of dimension 0");
	if (lead_lag && length == 0)
		throw std::invalid_argument("lead_lag requires a path with at least one point");
	validate_correction_len_(dimension, max_nodes, correction_len);
	if (correction == nullptr && correction_len != 0)
		throw std::invalid_argument("correction pointer is null but correction_len is nonzero");
	if (lead_lag && correction_len != 0)
		throw std::invalid_argument("correction cannot be used with lead_lag=true");

	const bool has_correction = correction_len != 0;
	const uint64_t aug_dimension = (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
	const auto& cache = get_branched_sig_coef_cache(tree_data, tree_data_len, dimension,
		aug_dimension, max_nodes, planar);
	const uint64_t num_indices = cache.target_indices.size();
	const uint64_t cache_size = cache.inv_tree_factorial.size();
	const uint64_t flat_path_length = length * dimension;

	auto worker = [&](uint64_t start, uint64_t end) {
		std::vector<T> increment(aug_dimension);
		std::vector<T> state(cache_size);
		std::vector<T> temp(cache_size);
		std::vector<T> state_derivs(cache_size);
		std::vector<T> local_derivs(cache_size);
		std::vector<T> increment_derivs(aug_dimension);
		std::vector<T> prefix_products(cache.max_nodes + 1);
		std::vector<T> local_log(has_correction ? cache_size : 0);
		std::vector<T> power(has_correction ? cache_size : 0);
		std::vector<T> next_power(has_correction ? cache_size : 0);
		std::vector<T> powers(has_correction && cache.max_nodes > 2 ? cache.max_nodes * cache_size : 0);
		std::vector<T> power_derivs(has_correction && cache.max_nodes > 2 ? cache.max_nodes * cache_size : 0);
		std::vector<T> d_local_log(has_correction && cache.max_nodes > 2 ? cache_size : 0);

		for (uint64_t batch = start; batch < end; ++batch) {
			Path<T> path_obj(path + batch * flat_path_length, dimension, length,
				time_aug, lead_lag, end_time);
			const T* batch_correction = has_correction
				? correction + batch * correction_batch_stride : nullptr;
			sparse_branched_signature_with_buffers_(path_obj, state.data(), increment.data(),
				temp.data(), local_log.data(), power.data(), next_power.data(), batch_correction,
				correction_len, correction_segment_stride, has_correction, cache);

			std::fill(state_derivs.begin(), state_derivs.end(), static_cast<T>(0));
			const T* coefs_ptr = coefs + batch * num_indices;
			const T* derivs_ptr = derivs + batch * num_indices;
			for (uint64_t i = 0; i < num_indices; ++i) {
				const uint64_t target = cache.target_indices[i];
				state[target] = coefs_ptr[i];
				if (target != 0)
					state_derivs[target] += derivs_ptr[i];
			}

			T* out_ptr = out + batch * flat_path_length;
			std::fill(out_ptr, out_ptr + flat_path_length, static_cast<T>(0));
			const uint64_t transformed_length = path_obj.length();
			if (transformed_length <= 1)
				continue;

			if (!lead_lag) {
				for (int64_t segment = static_cast<int64_t>(transformed_length) - 2;
					segment >= 0; --segment) {
					auto p0 = path_obj[segment];
					auto p1 = path_obj[segment + 1];
					for (uint64_t d = 0; d < aug_dimension; ++d)
						increment[d] = p1[d] - p0[d];
					const T* segment_correction = has_correction
						? batch_correction + static_cast<uint64_t>(segment) * correction_segment_stride
						: nullptr;
					sparse_local_branched_sig_(increment.data(), temp.data(), local_log.data(),
						power.data(), next_power.data(), segment_correction, correction_len,
						has_correction, cache);
					if (segment > 0)
						sparse_butcher_uncombine_inplace_(state.data(), temp.data(), cache);
					if (segment > 0)
						sparse_butcher_product_deriv_(state.data(), temp.data(), state_derivs.data(),
							local_derivs.data(), cache);
					else
						std::copy(state_derivs.begin(), state_derivs.end(), local_derivs.begin());

					sparse_local_bsig_deriv_to_increment_(local_derivs.data(), increment.data(),
						increment_derivs.data(), prefix_products.data(), local_log.data(), powers.data(),
						power_derivs.data(), d_local_log.data(), segment_correction, correction_len,
						has_correction, cache);
					T* positive = out_ptr + (static_cast<uint64_t>(segment) + 1) * dimension;
					T* negative = out_ptr + static_cast<uint64_t>(segment) * dimension;
					for (uint64_t d = 0; d < dimension; ++d) {
						positive[d] += increment_derivs[d];
						negative[d] -= increment_derivs[d];
					}
				}
			}
			else {
				T* positive = out_ptr + (length - 1) * dimension;
				T* negative = positive - dimension;
				bool parity = false;
				for (int64_t segment = static_cast<int64_t>(transformed_length) - 2;
					segment >= 0; --segment) {
					auto p0 = path_obj[segment];
					auto p1 = path_obj[segment + 1];
					for (uint64_t d = 0; d < aug_dimension; ++d)
						increment[d] = p1[d] - p0[d];
					sparse_local_branched_sig_(increment.data(), temp.data(), local_log.data(),
						power.data(), next_power.data(), static_cast<const T*>(nullptr), 0, false, cache);
					if (segment > 0)
						sparse_butcher_uncombine_inplace_(state.data(), temp.data(), cache);
					if (segment > 0)
						sparse_butcher_product_deriv_(state.data(), temp.data(), state_derivs.data(),
							local_derivs.data(), cache);
					else
						std::copy(state_derivs.begin(), state_derivs.end(), local_derivs.begin());
					sparse_local_bsig_deriv_to_increment_(local_derivs.data(), increment.data(),
						increment_derivs.data(), prefix_products.data(), static_cast<T*>(nullptr),
						static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr),
						static_cast<const T*>(nullptr), 0, false, cache);

					T* selected_derivs = parity ? increment_derivs.data() + dimension : increment_derivs.data();
					for (uint64_t d = 0; d < dimension; ++d) {
						positive[d] += selected_derivs[d];
						negative[d] -= selected_derivs[d];
					}
					if (parity) {
						positive -= dimension;
						negative -= dimension;
					}
					parity = !parity;
				}
			}
		}
	};

	if (batch_size == 0 || n_jobs == 1 || batch_size == 1)
		worker(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, worker);
}

extern "C" {

	CPSIG_API int prepare_branched_sig_coef(const uint64_t* tree_data, uint64_t tree_data_len, uint64_t data_dimension, uint64_t dimension, uint64_t max_nodes, bool planar) noexcept {
		SAFE_CALL(prepare_branched_sig_coef_cache(tree_data, tree_data_len, data_dimension, dimension, max_nodes, planar));
	}

	CPSIG_API int branched_sig_coef_f(const float* path, float* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(branched_sig_coef_<float>(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CPSIG_API int branched_sig_coef_d(const double* path, double* out, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(branched_sig_coef_<double>(path, out, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CPSIG_API int branched_sig_coef_backprop_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time, bool planar, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(branched_sig_coef_backprop_<float>(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

	CPSIG_API int branched_sig_coef_backprop_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* tree_data, uint64_t tree_data_len, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time, bool planar, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride) noexcept {
		SAFE_CALL(branched_sig_coef_backprop_<double>(path, out, coefs, derivs, tree_data, tree_data_len, batch_size, dimension, length, max_nodes, n_jobs, time_aug, lead_lag, end_time, planar, correction, correction_len, correction_batch_stride, correction_segment_stride));
	}

}
