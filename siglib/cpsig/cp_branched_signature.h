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
#include "cp_vector_funcs.h"
#include "multithreading.h"
#include "macros.h"

#include <new>

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
void linear_planar_log_(
	const T* increment,
	T* out,
	T* monomials,
	const BranchedSigCache& cache
) {
	out[0] = static_cast<T>(0);
	const auto& plan = cache.horner;
	monomials[0] = static_cast<T>(1);
	for (uint64_t i = 1; i < plan.planar_log_monomial_parent.size(); ++i) {
		monomials[i] = monomials[plan.planar_log_monomial_parent[i]]
			* increment[plan.planar_log_monomial_label[i]];
	}
	for (uint64_t pos = 0; pos < plan.planar_log_flats.size(); ++pos) {
		const uint64_t flat = plan.planar_log_flats[pos];
		out[flat] = static_cast<T>(plan.planar_log_coefficients[flat])
			* monomials[plan.planar_log_flat_monomial[pos]];
	}
}

template<std::floating_point T>
void linear_planar_log_deriv_to_increment_deriv_(
	const T* d_local_log,
	const T* increment,
	T* inc_derivs,
	const T* monomials,
	T* d_monomials,
	const BranchedSigCache& cache
) {
	std::memset(inc_derivs, 0, cache.dimension * sizeof(T));
	const auto& plan = cache.horner;
	const uint64_t monomial_count = plan.planar_log_monomial_parent.size();
	std::memset(d_monomials, 0, monomial_count * sizeof(T));
	for (uint64_t pos = 0; pos < plan.planar_log_flats.size(); ++pos) {
		const uint64_t flat = plan.planar_log_flats[pos];
		d_monomials[plan.planar_log_flat_monomial[pos]]
			+= static_cast<T>(plan.planar_log_coefficients[flat])
			* d_local_log[flat];
	}
	for (uint64_t i = monomial_count - 1; i > 0; --i) {
		const uint64_t parent = plan.planar_log_monomial_parent[i];
		const uint64_t label = plan.planar_log_monomial_label[i];
		const T deriv = d_monomials[i];
		d_monomials[parent] += deriv * increment[label];
		inc_derivs[label] += deriv * monomials[parent];
	}
}

template<std::floating_point T>
void fill_branched_horner_products_(
	const T* sig,
	T* products,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	products[0] = sig[0];
	for (uint64_t pos = 0; pos < plan.product_build.size(); ++pos) {
		const uint64_t product = plan.product_build[pos];
		products[product] = products[plan.product_build_parent[pos]]
			* sig[plan.product_build_factor[pos]];
	}
}

template<std::floating_point T>
void branched_horner_derivative_level_(
	const T* base,
	const T* extra,
	const T* increment,
	T* out,
	uint64_t target_order,
	uint64_t stage_order,
	T scale,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	const uint64_t stage_width = cache.max_nodes + 1;
	const uint64_t stage_key = target_order * stage_width + stage_order;
	const uint64_t start = plan.stage_offsets[stage_key];
	const uint64_t end = plan.stage_offsets[stage_key + 1];
	for (uint64_t stage_pos = start; stage_pos < end; ++stage_pos) {
		const uint64_t target = plan.stage_products[stage_pos];
		T value = static_cast<T>(0);
		const uint64_t term_start = plan.derivative_offsets[target];
		const uint64_t term_end = plan.derivative_offsets[target + 1];
		for (uint64_t pos = term_start; pos < term_end; ++pos) {
			const uint64_t left = plan.derivative_left[pos];
			const uint64_t label = plan.derivative_label[pos];
			const T source = base[left]
				+ (extra == nullptr ? static_cast<T>(0) : extra[left]);
			value += source * increment[label];
		}
		out[target] = scale * value;
	}
}

template<std::floating_point T>
void branched_horner_step_(
	T* sig,
	const T* increment,
	T* base,
	T* current,
	T* next,
	const BranchedSigCache& cache
) {
	if (cache.max_nodes == 0)
		return;
	const auto& plan = cache.horner;
	fill_branched_horner_products_(sig, base, cache);

	branched_horner_derivative_level_(
		base, static_cast<const T*>(nullptr), increment, current, 1, 1,
		static_cast<T>(1), cache);
	for (uint64_t flat_idx = cache.order_index[1];
		flat_idx < cache.order_index[2]; ++flat_idx) {
		const uint64_t flat = flat_idx + 1;
		const uint64_t product = plan.flat_to_product[flat];
		sig[flat] = base[product] + current[product];
	}

	for (uint64_t target_order = 2;
		target_order <= cache.max_nodes; ++target_order) {
		branched_horner_derivative_level_(
			base, static_cast<const T*>(nullptr), increment, current,
			target_order, 1,
			static_cast<T>(1) / static_cast<T>(target_order), cache);
		for (uint64_t source_order = 1;
			source_order + 1 < target_order; ++source_order) {
			branched_horner_derivative_level_(
				base, current, increment, next, target_order, source_order + 1,
				static_cast<T>(1)
					/ static_cast<T>(target_order - source_order), cache);
			std::swap(current, next);
		}

		branched_horner_derivative_level_(
			base, current, increment, next, target_order, target_order,
			static_cast<T>(1), cache);
		for (uint64_t flat_idx = cache.order_index[target_order];
			flat_idx < cache.order_index[target_order + 1]; ++flat_idx) {
			const uint64_t flat = flat_idx + 1;
			const uint64_t product = plan.flat_to_product[flat];
			sig[flat] = base[product] + next[product];
		}
	}
}

template<std::floating_point T>
void branched_horner_derivative_level_backprop_(
	const T* base,
	const T* extra,
	const T* increment,
	const T* d_target,
	T* d_base,
	T* d_extra,
	T* d_increment,
	uint64_t target_order,
	uint64_t stage_order,
	T scale,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	const uint64_t stage_width = cache.max_nodes + 1;
	const uint64_t stage_key = target_order * stage_width + stage_order;
	const uint64_t start = plan.stage_offsets[stage_key];
	const uint64_t end = plan.stage_offsets[stage_key + 1];
	for (uint64_t stage_pos = start; stage_pos < end; ++stage_pos) {
		const uint64_t target = plan.stage_products[stage_pos];
		const T deriv = scale * d_target[target];
		if (deriv == static_cast<T>(0))
			continue;
		const uint64_t term_start = plan.derivative_offsets[target];
		const uint64_t term_end = plan.derivative_offsets[target + 1];
		for (uint64_t pos = term_start; pos < term_end; ++pos) {
			const uint64_t left = plan.derivative_left[pos];
			const uint64_t label = plan.derivative_label[pos];
			const T source = base[left]
				+ (extra == nullptr ? static_cast<T>(0) : extra[left]);
			const T source_deriv = deriv * increment[label];
			d_base[left] += source_deriv;
			if (d_extra != nullptr)
				d_extra[left] += source_deriv;
			d_increment[label] += deriv * source;
		}
	}
}

template<std::floating_point T>
void branched_horner_step_backprop_(
	const T* sig,
	const T* increment,
	const T* d_out,
	T* d_sig,
	T* d_increment,
	T* base,
	T* states,
	T* d_base,
	T* d_current_buffer,
	T* d_next_buffer,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	const uint64_t product_count = plan.product_count;
	fill_branched_horner_products_(sig, base, cache);
	std::memset(d_base, 0, product_count * sizeof(T));
	std::memset(d_increment, 0, cache.dimension * sizeof(T));
	d_base[0] = d_out[0];
	for (uint64_t flat = 1; flat < cache.total_length; ++flat)
		d_base[plan.flat_to_product[flat]] += d_out[flat];

	for (uint64_t target_order = 1;
		target_order <= cache.max_nodes; ++target_order) {
		T* d_current = d_current_buffer;
		T* d_next = d_next_buffer;
		const uint64_t stage_width = cache.max_nodes + 1;
		uint64_t stage_key = target_order * stage_width + target_order;
		for (uint64_t pos = plan.stage_offsets[stage_key];
			pos < plan.stage_offsets[stage_key + 1]; ++pos) {
			d_current[plan.stage_products[pos]] = static_cast<T>(0);
		}
		for (uint64_t flat_idx = cache.order_index[target_order];
			flat_idx < cache.order_index[target_order + 1]; ++flat_idx) {
			const uint64_t flat = flat_idx + 1;
			d_current[plan.flat_to_product[flat]] += d_out[flat];
		}

		if (target_order == 1) {
			branched_horner_derivative_level_backprop_(
				base, static_cast<const T*>(nullptr), increment, d_current, d_base,
				static_cast<T*>(nullptr),
				d_increment, 1, 1, static_cast<T>(1), cache);
			continue;
		}

		T* state = states + product_count;
		branched_horner_derivative_level_(
			base, static_cast<const T*>(nullptr), increment, state,
			target_order, 1,
			static_cast<T>(1) / static_cast<T>(target_order), cache);
		for (uint64_t source_order = 1;
			source_order + 1 < target_order; ++source_order) {
			T* next_state = states + (source_order + 1) * product_count;
			branched_horner_derivative_level_(
				base, state, increment, next_state,
				target_order, source_order + 1,
				static_cast<T>(1)
					/ static_cast<T>(target_order - source_order), cache);
			state = next_state;
		}

		stage_key = target_order * stage_width + target_order - 1;
		for (uint64_t pos = plan.stage_offsets[stage_key];
			pos < plan.stage_offsets[stage_key + 1]; ++pos) {
			d_next[plan.stage_products[pos]] = static_cast<T>(0);
		}
		branched_horner_derivative_level_backprop_(
			base, state, increment, d_current, d_base, d_next,
			d_increment, target_order, target_order, static_cast<T>(1), cache);
		std::swap(d_current, d_next);

		for (uint64_t source_order = target_order - 2;
			source_order >= 1; --source_order) {
			state = states + source_order * product_count;
			stage_key = target_order * stage_width + source_order;
			for (uint64_t pos = plan.stage_offsets[stage_key];
				pos < plan.stage_offsets[stage_key + 1]; ++pos) {
				d_next[plan.stage_products[pos]] = static_cast<T>(0);
			}
			branched_horner_derivative_level_backprop_(
				base, state, increment, d_current, d_base, d_next,
				d_increment, target_order, source_order + 1,
				static_cast<T>(1)
					/ static_cast<T>(target_order - source_order), cache);
			std::swap(d_current, d_next);
		}

		branched_horner_derivative_level_backprop_(
			base, static_cast<const T*>(nullptr), increment, d_current, d_base,
			static_cast<T*>(nullptr),
			d_increment, target_order, 1,
			static_cast<T>(1) / static_cast<T>(target_order), cache);
	}

	std::memset(d_sig, 0, cache.total_length * sizeof(T));
	for (uint64_t pos = plan.product_build.size(); pos > 0; --pos) {
		const uint64_t product = plan.product_build[pos - 1];
		const uint64_t parent = plan.product_build_parent[pos - 1];
		const uint64_t factor = plan.product_build_factor[pos - 1];
		const T deriv = d_base[product];
		d_base[parent] += deriv * sig[factor];
		d_sig[factor] += deriv * base[parent];
	}
	d_sig[0] = d_base[0];
}

template<std::floating_point T>
FORCE_INLINE T planar_horner_convolution_value_(
	const T* left,
	const T* right,
	uint64_t target,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	T value = plan.planar_log_coefficients[target] == 0.0
		? static_cast<T>(0) : left[0] * right[target];
	const uint64_t start = plan.planar_coproduct_offsets[target];
	const uint64_t end = plan.planar_coproduct_offsets[target + 1];
	const uint64_t* left_indices = plan.planar_coproduct_left.data();
	const uint64_t* right_indices = plan.planar_coproduct_right.data();
	for (uint64_t pos = start; pos < end; ++pos) {
		value += left[left_indices[pos]] * right[right_indices[pos]];
	}
	return value;
}

template<std::floating_point T>
FORCE_INLINE void planar_horner_convolution_value_backprop_(
	const T* left,
	const T* right,
	T deriv,
	uint64_t target,
	T* d_left,
	T* d_right,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	if (plan.planar_log_coefficients[target] != 0.0)
		d_right[target] += deriv * left[0];
	const uint64_t start = plan.planar_coproduct_offsets[target];
	const uint64_t end = plan.planar_coproduct_offsets[target + 1];
	const uint64_t* left_indices = plan.planar_coproduct_left.data();
	const uint64_t* right_indices = plan.planar_coproduct_right.data();
	for (uint64_t pos = start; pos < end; ++pos) {
		const uint64_t left_idx = left_indices[pos];
		const uint64_t right_idx = right_indices[pos];
		d_left[left_idx] += deriv * right[right_idx];
		d_right[right_idx] += deriv * left[left_idx];
	}
}

template<std::floating_point T>
void planar_branched_horner_step_(
	T* sig,
	const T* local_log,
	T* current,
	T* next,
	const BranchedSigCache& cache
) {
	if (cache.max_nodes == 0)
		return;
	const T* source = sig;
	T* target = current;
	for (uint64_t stage = 1; stage <= cache.max_nodes; ++stage) {
		const uint64_t end = cache.order_index[stage + 1] + 1;
		const T scale = static_cast<T>(1)
			/ static_cast<T>(cache.max_nodes - stage + 1);
		target[0] = sig[0];
		for (uint64_t flat = 1; flat < end; ++flat) {
			target[flat] = sig[flat] + scale
				* planar_horner_convolution_value_(
					source, local_log, flat, cache);
		}
		source = target;
		target = target == current ? next : current;
	}
	std::memcpy(sig, source, cache.total_length * sizeof(T));
}

template<std::floating_point T>
void planar_branched_horner_step_backprop_(
	const T* sig,
	const T* local_log,
	const T* d_out,
	T* d_sig,
	T* d_local_log,
	T* states,
	T* d_current_buffer,
	T* d_previous_buffer,
	const BranchedSigCache& cache
) {
	const uint64_t total_length = cache.total_length;
	const T* source = sig;
	for (uint64_t stage = 1; stage <= cache.max_nodes; ++stage) {
		T* target = states + stage * total_length;
		const uint64_t end = cache.order_index[stage + 1] + 1;
		const T scale = static_cast<T>(1)
			/ static_cast<T>(cache.max_nodes - stage + 1);
		target[0] = sig[0];
		for (uint64_t flat = 1; flat < end; ++flat) {
			target[flat] = sig[flat] + scale
				* planar_horner_convolution_value_(
					source, local_log, flat, cache);
		}
		source = target;
	}

	std::memset(d_sig, 0, total_length * sizeof(T));
	for (const uint64_t flat : cache.horner.planar_log_flats)
		d_local_log[flat] = static_cast<T>(0);
	T* d_current = d_current_buffer;
	T* d_previous = d_previous_buffer;
	std::memcpy(d_current, d_out, total_length * sizeof(T));
	for (uint64_t stage = cache.max_nodes; stage > 0; --stage) {
		const uint64_t source_stage = stage - 1;
		const uint64_t source_end = source_stage == 0
			? 1 : cache.order_index[source_stage + 1] + 1;
		std::memset(d_previous, 0, source_end * sizeof(T));
		const T* stage_source = source_stage == 0
			? sig : states + source_stage * total_length;
		const uint64_t target_end = cache.order_index[stage + 1] + 1;
		const T scale = static_cast<T>(1)
			/ static_cast<T>(cache.max_nodes - stage + 1);
		for (uint64_t flat = 1; flat < target_end; ++flat) {
			const T deriv = d_current[flat];
			d_sig[flat] += deriv;
			planar_horner_convolution_value_backprop_(
				stage_source, local_log, scale * deriv, flat,
				d_previous, d_local_log, cache);
		}
		if (source_stage != 0)
			std::swap(d_current, d_previous);
	}
}

FORCE_INLINE uint64_t branched_horner_product_for_flat_(
	uint64_t flat,
	const BranchedSigCache& cache
) {
	return cache.planar ? flat : cache.horner.flat_to_product[flat];
}

template<std::floating_point T>
void fill_branched_horner_forest_products_(
	const T* values,
	T* products,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	if (cache.planar) {
		std::memcpy(products, values, cache.total_length * sizeof(T));
		return;
	}
	products[0] = values[0];
	for (uint64_t product = 1; product < plan.product_count; ++product) {
		const uint64_t parent = plan.product_parent[product];
		const uint64_t factor = plan.product_factor[product];
		products[product] = parent == 0
			? values[factor] : products[parent] * values[factor];
	}
}

template<std::floating_point T>
void branched_horner_forest_products_backprop_(
	const T* products,
	T* d_products,
	T* d_values,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	if (cache.planar) {
		for (uint64_t flat = 0; flat < cache.total_length; ++flat)
			d_values[flat] += d_products[flat];
		return;
	}
	d_values[0] += d_products[0];
	for (uint64_t product = plan.product_count - 1; product > 0; --product) {
		const uint64_t parent = plan.product_parent[product];
		const uint64_t factor = plan.product_factor[product];
		const T deriv = d_products[product];
		if (parent == 0) {
			d_values[factor] += deriv;
		}
		else {
			const uint64_t factor_product = plan.flat_to_product[factor];
			d_products[parent] += deriv * products[factor_product];
			d_values[factor] += deriv * products[parent];
		}
	}
}

template<std::floating_point T>
void correction_horner_exp_(
	const T* local_log,
	T* out,
	T* node_values,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	for (uint64_t node = 0;
		node < plan.correction_horner_constants.size(); ++node) {
		T value = static_cast<T>(plan.correction_horner_constants[node]);
		const uint64_t start = plan.correction_horner_node_offsets[node];
		const uint64_t end = plan.correction_horner_node_offsets[node + 1];
		for (uint64_t pos = start; pos < end; ++pos) {
			value += local_log[plan.correction_horner_variables[pos]]
				* node_values[plan.correction_horner_children[pos]];
		}
		node_values[node] = value;
	}
	out[0] = static_cast<T>(1);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		const uint64_t root = plan.correction_horner_roots[flat];
		out[flat] = root == UINT64_MAX
			? static_cast<T>(0) : node_values[root];
	}
}

template<std::floating_point T>
void correction_horner_exp_backprop_(
	const T* local_log,
	const T* d_out,
	T* d_local_log,
	const T* node_values,
	T* d_node_values,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	std::memset(d_local_log, 0, cache.total_length * sizeof(T));
	std::memset(d_node_values, 0,
		plan.correction_horner_constants.size() * sizeof(T));
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		const uint64_t root = plan.correction_horner_roots[flat];
		if (root != UINT64_MAX)
			d_node_values[root] += d_out[flat];
	}
	for (uint64_t node = plan.correction_horner_constants.size(); node > 0; --node) {
		const uint64_t node_idx = node - 1;
		const T deriv = d_node_values[node_idx];
		const uint64_t start = plan.correction_horner_node_offsets[node_idx];
		const uint64_t end = plan.correction_horner_node_offsets[node_idx + 1];
		for (uint64_t pos = start; pos < end; ++pos) {
			const uint64_t variable = plan.correction_horner_variables[pos];
			const uint64_t child = plan.correction_horner_children[pos];
			d_local_log[variable] += deriv * node_values[child];
			d_node_values[child] += deriv * local_log[variable];
		}
	}
}

template<std::floating_point T>
void branched_horner_combine_step_(
	T* sig,
	const T* local_sig,
	T* products,
	T* local_products,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	fill_branched_horner_forest_products_(sig, products, cache);
	fill_branched_horner_forest_products_(local_sig, local_products, cache);
	const T sig_scalar = sig[0];
	const T local_scalar = local_sig[0];
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		const uint64_t product = branched_horner_product_for_flat_(flat, cache);
		T value = sig[flat] * local_scalar + sig_scalar * local_sig[flat];
		const uint64_t start = plan.coproduct_offsets[product];
		const uint64_t end = plan.coproduct_offsets[product + 1];
		for (uint64_t pos = start; pos < end; pos += 2) {
			value += products[plan.coproduct_pairs[pos]]
				* local_products[plan.coproduct_pairs[pos + 1]];
		}
		sig[flat] = value;
	}
	sig[0] = sig_scalar * local_scalar;
}

template<std::floating_point T>
void branched_horner_uncombine_step_(
	T* combined,
	const T* local_sig,
	T* products,
	T* local_products,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	fill_branched_horner_forest_products_(local_sig, local_products, cache);
	std::memset(products, 0, plan.product_count * sizeof(T));
	const T local_scalar = local_sig[0];
	const T previous_scalar = combined[0] / local_scalar;
	products[0] = previous_scalar;
	for (uint64_t order = 1; order <= cache.max_nodes; ++order) {
		const uint64_t flat_start = cache.order_index[order] + 1;
		const uint64_t flat_end = cache.order_index[order + 1] + 1;
		for (uint64_t flat = flat_start; flat < flat_end; ++flat) {
			const uint64_t product = branched_horner_product_for_flat_(flat, cache);
			T value = combined[flat] - previous_scalar * local_sig[flat];
			const uint64_t start = plan.coproduct_offsets[product];
			const uint64_t end = plan.coproduct_offsets[product + 1];
			for (uint64_t pos = start; pos < end; pos += 2) {
				value -= products[plan.coproduct_pairs[pos]]
					* local_products[plan.coproduct_pairs[pos + 1]];
			}
			combined[flat] = value / local_scalar;
		}
		if (cache.planar) {
			for (uint64_t flat = flat_start; flat < flat_end; ++flat)
				products[flat] = combined[flat];
		}
		else {
			for (uint64_t product = 1; product < plan.product_count; ++product) {
				if (plan.product_node_counts[product] != order)
					continue;
				const uint64_t parent = plan.product_parent[product];
				const uint64_t factor = plan.product_factor[product];
				products[product] = parent == 0
					? combined[factor] : products[parent] * combined[factor];
			}
		}
	}
	combined[0] = previous_scalar;
}

template<std::floating_point T>
void branched_horner_combine_step_backprop_(
	const T* sig,
	const T* local_sig,
	const T* d_out,
	T* d_sig,
	T* d_local_sig,
	T* products,
	T* local_products,
	T* d_products,
	T* d_local_products,
	const BranchedSigCache& cache
) {
	const auto& plan = cache.horner;
	const uint64_t product_count = plan.product_count;
	fill_branched_horner_forest_products_(sig, products, cache);
	fill_branched_horner_forest_products_(local_sig, local_products, cache);
	std::memset(d_sig, 0, cache.total_length * sizeof(T));
	std::memset(d_local_sig, 0, cache.total_length * sizeof(T));
	std::memset(d_products, 0, product_count * sizeof(T));
	std::memset(d_local_products, 0, product_count * sizeof(T));
	d_sig[0] += d_out[0] * local_sig[0];
	d_local_sig[0] += d_out[0] * sig[0];
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		const T deriv = d_out[flat];
		d_sig[flat] += deriv * local_sig[0];
		d_local_sig[0] += deriv * sig[flat];
		d_sig[0] += deriv * local_sig[flat];
		d_local_sig[flat] += deriv * sig[0];
		const uint64_t product = branched_horner_product_for_flat_(flat, cache);
		const uint64_t start = plan.coproduct_offsets[product];
		const uint64_t end = plan.coproduct_offsets[product + 1];
		for (uint64_t pos = start; pos < end; pos += 2) {
			const uint64_t left = plan.coproduct_pairs[pos];
			const uint64_t right = plan.coproduct_pairs[pos + 1];
			d_products[left] += deriv * local_products[right];
			d_local_products[right] += deriv * products[left];
		}
	}
	branched_horner_forest_products_backprop_(
		products, d_products, d_sig, cache);
	branched_horner_forest_products_backprop_(
		local_products, d_local_products, d_local_sig, cache);
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

					if (has_forest)
						term *= X[coproduct_data[pos++]];

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
	T* local_log,
	T* local_sig,
	T* horner_base,
	T* horner_current,
	T* horner_next,
	T* correction_nodes,
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
		branched_correction_(
			increment, local_log, correction, correction_len, data_dim, cache);
		correction_horner_exp_(
			local_log, out, correction_nodes, cache);
	}

	for (uint64_t seg = 1; seg < len - 1; ++seg) {
		auto seg_start = path[seg];
		auto seg_end = path[seg + 1];

		for (uint64_t d = 0; d < dim; ++d) {
			increment[d] = seg_end[d] - seg_start[d];
		}

		if (!has_correction) {
			if (cache.planar) {
				linear_planar_log_(increment, local_log, horner_base, cache);
				planar_branched_horner_step_(
					out, local_log, horner_current, horner_next, cache);
			}
			else {
				branched_horner_step_(
					out, increment, horner_base, horner_current, horner_next, cache);
			}
		} else {
			const T* seg_correction = correction + seg * correction_segment_stride;
			branched_correction_(
				increment, local_log, seg_correction,
				correction_len, data_dim, cache);
			correction_horner_exp_(
				local_log, local_sig, correction_nodes, cache);
			branched_horner_combine_step_(
				out, local_sig, horner_base, horner_current, cache);
		}
	}
}


template<std::floating_point T>
struct BranchedSigForwardWorkspace_ {
	std::unique_ptr<T[]> increment;
	std::unique_ptr<T[]> local_log;
	std::unique_ptr<T[]> local_sig;
	std::unique_ptr<T[]> horner_base;
	std::unique_ptr<T[]> horner_current;
	std::unique_ptr<T[]> horner_next;
	std::unique_ptr<T[]> correction_nodes;
	std::unique_ptr<T[]> full_output;

	BranchedSigForwardWorkspace_(
		uint64_t dimension,
		uint64_t total_length,
		uint64_t product_count,
		uint64_t planar_monomial_count,
		uint64_t correction_node_count,
		bool has_correction,
		bool planar,
		bool scalar_term
	) : increment(std::make_unique<T[]>(dimension)) {
		if (has_correction) {
			local_log = std::make_unique<T[]>(total_length);
			local_sig = std::make_unique<T[]>(total_length);
			horner_base = std::make_unique<T[]>(product_count);
			horner_current = std::make_unique<T[]>(product_count);
			correction_nodes = std::make_unique<T[]>(correction_node_count);
		}
		else if (planar) {
			local_log = std::make_unique<T[]>(total_length);
			horner_base = std::make_unique<T[]>(planar_monomial_count);
			horner_current = std::make_unique<T[]>(total_length);
			horner_next = std::make_unique<T[]>(total_length);
		}
		else {
			horner_base = std::make_unique<T[]>(product_count);
			horner_current = std::make_unique<T[]>(product_count);
			horner_next = std::make_unique<T[]>(product_count);
		}
		if (!scalar_term)
			full_output = std::make_unique<T[]>(total_length);
	}
};


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

	auto compute_one = [&](const T* path_ptr, T* out_ptr,
		BranchedSigForwardWorkspace_<T>& workspace,
		const T* correction_ptr) {
		Path<T> path_obj(path_ptr, dimension, length, time_aug, lead_lag, end_time);
		if (scalar_term) {
			branched_signature_with_buffers_(path_obj, out_ptr,
				workspace.increment.get(), workspace.local_log.get(),
				workspace.local_sig.get(), workspace.horner_base.get(),
				workspace.horner_current.get(),
				workspace.horner_next.get(),
				workspace.correction_nodes.get(),
				correction_ptr, correction_len,
				correction_segment_stride, has_correction, cache);
		} else {
			branched_signature_with_buffers_(path_obj, workspace.full_output.get(),
				workspace.increment.get(), workspace.local_log.get(),
				workspace.local_sig.get(), workspace.horner_base.get(),
				workspace.horner_current.get(),
				workspace.horner_next.get(),
				workspace.correction_nodes.get(),
				correction_ptr, correction_len,
				correction_segment_stride, has_correction, cache);
			std::memcpy(out_ptr, workspace.full_output.get() + 1,
				(total_len - 1) * sizeof(T));
		}
	};

	auto work_range = [&](uint64_t start, uint64_t end) {
		if (start == end)
			return;
		BranchedSigForwardWorkspace_<T> workspace(
			aug_dim, total_len, cache.horner.product_count,
			cache.horner.planar_log_monomial_parent.size(),
			cache.horner.correction_horner_constants.size(),
			has_correction, cache.planar, scalar_term);
		for (uint64_t b = start; b < end; ++b) {
			const T* correction_ptr = has_correction
				? correction + b * correction_batch_stride
				: nullptr;
			compute_one(
				path + b * flat_path_length,
				out + b * out_stride,
				workspace,
				correction_ptr);
		}
	};
	if (n_jobs == 1 || batch_size == 1)
		work_range(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, work_range);
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

	auto work_range = [&](uint64_t start, uint64_t end) {
		if (start == end)
			return;
		if (scalar_term) {
			for (uint64_t b = start; b < end; ++b) {
				T* out_ptr = out + b * total_len;
				std::memcpy(out_ptr, bsig1 + b * total_len, total_len * sizeof(T));
				butcher_product_inplace_(out_ptr, bsig2 + b * total_len, cache);
			}
			return;
		}

		std::vector<T> s1(total_len);
		std::vector<T> s2(total_len);
		std::vector<T> combined(total_len);
		for (uint64_t b = start; b < end; ++b) {
			const T* sig1_ptr = bsig1 + b * stride;
			const T* sig2_ptr = bsig2 + b * stride;
			s1[0] = static_cast<T>(1);
			s2[0] = static_cast<T>(1);
			std::memcpy(s1.data() + 1, sig1_ptr, (total_len - 1) * sizeof(T));
			std::memcpy(s2.data() + 1, sig2_ptr, (total_len - 1) * sizeof(T));
			std::memcpy(combined.data(), s1.data(), total_len * sizeof(T));
			butcher_product_inplace_(combined.data(), s2.data(), cache);
			std::memcpy(out + b * stride, combined.data() + 1,
				(total_len - 1) * sizeof(T));
		}
	};
	if (n_jobs == 1 || batch_size == 1)
		work_range(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, work_range);
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
		}
	};

	auto work_range = [&](uint64_t start, uint64_t end) {
		if (start == end)
			return;
		if (scalar_term) {
			for (uint64_t b = start; b < end; ++b)
				work(b);
			return;
		}

		std::vector<T> s1(total_len);
		std::vector<T> s2(total_len);
		std::vector<T> derivs(total_len);
		std::vector<T> d1(total_len);
		std::vector<T> d2(total_len);
		for (uint64_t b = start; b < end; ++b) {
			const uint64_t off = b * stride;
			s1[0] = static_cast<T>(1);
			s2[0] = static_cast<T>(1);
			derivs[0] = static_cast<T>(0);
			std::memcpy(s1.data() + 1, bsig1 + off, (total_len - 1) * sizeof(T));
			std::memcpy(s2.data() + 1, bsig2 + off, (total_len - 1) * sizeof(T));
			std::memcpy(derivs.data() + 1, derivs_in + off, (total_len - 1) * sizeof(T));
			std::memcpy(d1.data(), derivs.data(), total_len * sizeof(T));
			butcher_product_deriv_(s1.data(), s2.data(), d1.data(), d2.data(), cache);
			std::memcpy(out1 + off, d1.data() + 1, (total_len - 1) * sizeof(T));
			std::memcpy(out2 + off, d2.data() + 1, (total_len - 1) * sizeof(T));
		}
	};
	if (n_jobs == 1 || batch_size == 1)
		work_range(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, work_range);
}


// =========================================================================
// Backpropagation
// =========================================================================

// Differentiate the Butcher product.
// Given dF/dX_combined, compute dF/dX_prev and dF/dY.
// X_prev and Y must be the values from before the product.
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
struct BranchedSigBackpropWorkspace_ {
	std::unique_ptr<T[]> bsig;
	std::unique_ptr<T[]> derivs;
	std::unique_ptr<T[]> increment;
	std::unique_ptr<T[]> local_derivs;
	std::unique_ptr<T[]> inc_derivs;
	std::unique_ptr<T[]> local_log;
	std::unique_ptr<T[]> local_sig;
	std::unique_ptr<T[]> d_local_sig;
	std::unique_ptr<T[]> d_local_log;
	std::unique_ptr<T[]> horner_base;
	std::unique_ptr<T[]> horner_log;
	std::unique_ptr<T[]> horner_current;
	std::unique_ptr<T[]> horner_next;
	std::unique_ptr<T[]> horner_states;
	std::unique_ptr<T[]> horner_d_base;
	std::unique_ptr<T[]> correction_node_derivs;

	BranchedSigBackpropWorkspace_(
		uint64_t dimension,
		uint64_t total_length,
		uint64_t max_nodes,
		uint64_t product_count,
		uint64_t planar_monomial_count,
		uint64_t correction_node_count,
		bool has_correction,
		bool planar
	) : bsig(std::make_unique<T[]>(total_length)),
		derivs(std::make_unique<T[]>(total_length)),
		increment(std::make_unique<T[]>(dimension)),
		local_derivs(std::make_unique<T[]>(total_length)),
		inc_derivs(std::make_unique<T[]>(dimension)) {
		if (has_correction) {
			local_log = std::make_unique<T[]>(total_length);
			local_sig = std::make_unique<T[]>(total_length);
			d_local_sig = std::make_unique<T[]>(total_length);
			d_local_log = std::make_unique<T[]>(total_length);
			horner_base = std::make_unique<T[]>(product_count);
			horner_log = std::make_unique<T[]>(product_count);
			horner_current = std::make_unique<T[]>(product_count);
			horner_next = std::make_unique<T[]>(product_count);
			horner_states = std::make_unique<T[]>(correction_node_count);
			correction_node_derivs = std::make_unique<T[]>(correction_node_count);
		}
		else if (planar) {
			local_log = std::make_unique<T[]>(total_length);
			d_local_log = std::make_unique<T[]>(total_length);
			horner_base = std::make_unique<T[]>(planar_monomial_count);
			horner_current = std::make_unique<T[]>(total_length);
			horner_next = std::make_unique<T[]>(total_length);
			horner_states = std::make_unique<T[]>((max_nodes + 1) * total_length);
			horner_d_base = std::make_unique<T[]>(planar_monomial_count);
		}
		else {
			horner_base = std::make_unique<T[]>(product_count);
			horner_current = std::make_unique<T[]>(product_count);
			horner_next = std::make_unique<T[]>(product_count);
			horner_states = std::make_unique<T[]>((max_nodes + 1) * product_count);
			horner_d_base = std::make_unique<T[]>(product_count);
		}
	}
};


// Main backward loop.
template<std::floating_point T>
void branched_sig_backprop_inplace_(
	const Path<T>& path,
	T* out,
	T* bsig_derivs,
	T* bsig,
	T* increment,
	T* local_derivs,
	T* inc_derivs,
	T* local_log,
	T* local_sig,
	T* d_local_sig,
	T* d_local_log,
	T* horner_base,
	T* horner_log,
	T* horner_current,
	T* horner_next,
	T* horner_states,
	T* horner_d_base,
	T* correction_node_derivs,
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
	const bool lead_lag = path.lead_lag();
	const uint64_t data_length = path.data_length();
	for (int seg = steps - 1; seg >= 0; --seg) {
		auto p0 = path[seg];
		auto p1 = path[seg + 1];
		for (uint64_t d = 0; d < dim; ++d)
			increment[d] = p1[d] - p0[d];

		if (!has_correction) {
			if (cache.planar) {
				if (seg > 0) {
					linear_planar_log_(
						increment, local_log, horner_base, cache);
					for (const uint64_t flat : cache.horner.planar_log_flats)
						local_log[flat] = -local_log[flat];
					planar_branched_horner_step_(
						bsig, local_log, horner_current, horner_next, cache);
					for (const uint64_t flat : cache.horner.planar_log_flats)
						local_log[flat] = -local_log[flat];
					planar_branched_horner_step_backprop_(
						bsig, local_log, bsig_derivs, local_derivs,
						d_local_log, horner_states, horner_current,
						horner_next, cache);
					linear_planar_log_deriv_to_increment_deriv_(
						d_local_log, increment, inc_derivs,
						horner_base, horner_d_base, cache);
					std::memcpy(
						bsig_derivs, local_derivs, total_len * sizeof(T));
				}
				else {
					linear_bsig_deriv_to_increment_deriv_(
						bsig_derivs, increment, inc_derivs, dim, cache);
				}
			}
			else {
				if (seg > 0) {
					for (uint64_t d = 0; d < dim; ++d)
						increment[d] = -increment[d];
					branched_horner_step_(
						bsig, increment, horner_base, horner_current, horner_next, cache);
					for (uint64_t d = 0; d < dim; ++d)
						increment[d] = -increment[d];

					branched_horner_step_backprop_(
						bsig, increment, bsig_derivs, local_derivs, inc_derivs,
						horner_base, horner_states, horner_d_base,
						horner_current, horner_next, cache);
					std::memcpy(
						bsig_derivs, local_derivs, total_len * sizeof(T));
				}
				else {
					linear_bsig_deriv_to_increment_deriv_(
						bsig_derivs, increment, inc_derivs, dim, cache);
				}
			}
		}
		else {
			const T* seg_correction = correction
				+ static_cast<uint64_t>(seg) * correction_segment_stride;
			branched_correction_(
				increment, local_log, seg_correction,
				correction_len, data_dim, cache);
			correction_horner_exp_(
				local_log, local_sig, horner_states, cache);
			if (seg > 0) {
				branched_horner_uncombine_step_(
					bsig, local_sig, horner_base, horner_log, cache);
				branched_horner_combine_step_backprop_(
					bsig, local_sig, bsig_derivs, local_derivs,
					d_local_sig, horner_base, horner_log,
					horner_current, horner_next, cache);
			}
			else {
				std::memcpy(
					d_local_sig, bsig_derivs, total_len * sizeof(T));
			}
			correction_horner_exp_backprop_(
				local_log, d_local_sig, d_local_log, horner_states,
				correction_node_derivs, cache);
			std::memset(inc_derivs, 0, dim * sizeof(T));
			for (uint64_t d = 0; d < dim; ++d)
				inc_derivs[d] = d_local_log[cache.order_index[1] + d + 1];
			if (seg > 0)
				std::memcpy(
					bsig_derivs, local_derivs, total_len * sizeof(T));
		}

		const uint64_t reverse_segment = static_cast<uint64_t>(steps - 1 - seg);
		const uint64_t value_offset = lead_lag && (reverse_segment & 1)
			? data_dim
			: 0;
		const uint64_t positive_offset = lead_lag
			? (data_length - 1 - reverse_segment / 2) * data_dim
			: static_cast<uint64_t>(seg + 1) * data_dim;
		T* positive = out + positive_offset;
		T* negative = positive - data_dim;
		const T* values = inc_derivs + value_offset;
		for (uint64_t d = 0; d < data_dim; ++d) {
			positive[d] += values[d];
			negative[d] -= values[d];
		}
	}
}


#ifdef VEC
template<std::floating_point T, size_t Width>
void linear_branched_sig_batch_(
	const T* increment,
	T* out,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	alignas(32) T product[Width];
	vec_batch_fill(out, static_cast<T>(1));

	const uint64_t num_trees = cache.total_length - 1;
	const uint8_t* labels = cache.node_labels_data.data();
	const uint64_t* offsets = cache.node_labels_offsets.data();
	for (uint64_t i = 0; i < num_trees; ++i) {
		vec_batch_fill(product, static_cast<T>(1));
		for (uint64_t pos = offsets[i]; pos < offsets[i + 1]; ++pos) {
			vec_batch_multiply_inplace(
				product, increment + static_cast<uint64_t>(labels[pos]) * Width);
		}
		vec_batch_scale(
			out + (i + 1) * Width, product,
			static_cast<T>(cache.inv_tree_factorial[i]));
	}
}


template<std::floating_point T, size_t Width>
void fill_branched_horner_products_batch_(
	const T* sig,
	T* products,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	const auto& plan = cache.horner;
	vec_batch_copy(products, sig);
	for (uint64_t pos = 0; pos < plan.product_build.size(); ++pos) {
		const uint64_t product = plan.product_build[pos];
		vec_batch_multiply(
			products + product * Width,
			products + plan.product_build_parent[pos] * Width,
			sig + plan.product_build_factor[pos] * Width);
	}
}


template<std::floating_point T, size_t Width>
void branched_horner_derivative_level_batch_(
	const T* base,
	const T* extra,
	const T* increment,
	T* out,
	uint64_t target_order,
	uint64_t stage_order,
	T scale,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	alignas(32) T value[Width];
	alignas(32) T source_sum[Width];
	const auto& plan = cache.horner;
	const uint64_t stage_width = cache.max_nodes + 1;
	const uint64_t stage_key = target_order * stage_width + stage_order;
	const uint64_t start = plan.stage_offsets[stage_key];
	const uint64_t end = plan.stage_offsets[stage_key + 1];
	for (uint64_t stage_pos = start; stage_pos < end; ++stage_pos) {
		const uint64_t target = plan.stage_products[stage_pos];
		vec_batch_fill(value, static_cast<T>(0));
		const uint64_t term_start = plan.derivative_offsets[target];
		const uint64_t term_end = plan.derivative_offsets[target + 1];
		for (uint64_t pos = term_start; pos < term_end; ++pos) {
			const uint64_t left = plan.derivative_left[pos];
			const uint64_t label = plan.derivative_label[pos];
			const T* source = base + left * Width;
			if (extra != nullptr) {
				vec_batch_add(source_sum, source, extra + left * Width);
				source = source_sum;
			}
			vec_batch_multiply_add(
				value, source, increment + label * Width);
		}
		vec_batch_scale(out + target * Width, value, scale);
	}
}


template<std::floating_point T, size_t Width>
void branched_horner_step_batch_(
	T* sig,
	const T* increment,
	T* base,
	T* current,
	T* next,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	if (cache.max_nodes == 0)
		return;
	const auto& plan = cache.horner;
	fill_branched_horner_products_batch_<T, Width>(sig, base, cache);

	branched_horner_derivative_level_batch_<T, Width>(
		base, nullptr, increment, current, 1, 1, static_cast<T>(1), cache);
	for (uint64_t flat_idx = cache.order_index[1];
		flat_idx < cache.order_index[2]; ++flat_idx) {
		const uint64_t flat = flat_idx + 1;
		const uint64_t product = plan.flat_to_product[flat];
		vec_batch_add(
			sig + flat * Width, base + product * Width,
			current + product * Width);
	}

	for (uint64_t target_order = 2;
		target_order <= cache.max_nodes; ++target_order) {
		branched_horner_derivative_level_batch_<T, Width>(
			base, nullptr, increment, current, target_order, 1,
			static_cast<T>(1) / static_cast<T>(target_order), cache);
		for (uint64_t source_order = 1;
			source_order + 1 < target_order; ++source_order) {
			branched_horner_derivative_level_batch_<T, Width>(
				base, current, increment, next, target_order, source_order + 1,
				static_cast<T>(1)
					/ static_cast<T>(target_order - source_order), cache);
			std::swap(current, next);
		}

		branched_horner_derivative_level_batch_<T, Width>(
			base, current, increment, next, target_order, target_order,
			static_cast<T>(1), cache);
		for (uint64_t flat_idx = cache.order_index[target_order];
			flat_idx < cache.order_index[target_order + 1]; ++flat_idx) {
			const uint64_t flat = flat_idx + 1;
			const uint64_t product = plan.flat_to_product[flat];
			vec_batch_add(
				sig + flat * Width, base + product * Width,
				next + product * Width);
		}
	}
}


template<std::floating_point T, size_t Width>
void branched_horner_derivative_level_backprop_batch_(
	const T* base,
	const T* extra,
	const T* increment,
	const T* d_target,
	T* d_base,
	T* d_extra,
	T* d_increment,
	uint64_t target_order,
	uint64_t stage_order,
	T scale,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	alignas(32) T deriv[Width];
	alignas(32) T source_sum[Width];
	alignas(32) T source_deriv[Width];
	const auto& plan = cache.horner;
	const uint64_t stage_width = cache.max_nodes + 1;
	const uint64_t stage_key = target_order * stage_width + stage_order;
	const uint64_t start = plan.stage_offsets[stage_key];
	const uint64_t end = plan.stage_offsets[stage_key + 1];
	for (uint64_t stage_pos = start; stage_pos < end; ++stage_pos) {
		const uint64_t target = plan.stage_products[stage_pos];
		vec_batch_scale(deriv, d_target + target * Width, scale);
		if (vec_batch_is_zero(deriv))
			continue;
		const uint64_t term_start = plan.derivative_offsets[target];
		const uint64_t term_end = plan.derivative_offsets[target + 1];
		for (uint64_t pos = term_start; pos < term_end; ++pos) {
			const uint64_t left = plan.derivative_left[pos];
			const uint64_t label = plan.derivative_label[pos];
			const T* source = base + left * Width;
			if (extra != nullptr) {
				vec_batch_add(source_sum, source, extra + left * Width);
				source = source_sum;
			}
			vec_batch_multiply(
				source_deriv, deriv, increment + label * Width);
			vec_batch_add_inplace(d_base + left * Width, source_deriv);
			if (d_extra != nullptr)
				vec_batch_add_inplace(d_extra + left * Width, source_deriv);
			vec_batch_multiply_add(
				d_increment + label * Width, deriv, source);
		}
	}
}


template<std::floating_point T, size_t Width>
void branched_horner_step_backprop_batch_(
	const T* sig,
	const T* increment,
	const T* d_out,
	T* d_sig,
	T* d_increment,
	T* base,
	T* states,
	T* d_base,
	T* d_current_buffer,
	T* d_next_buffer,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	const auto& plan = cache.horner;
	const uint64_t product_count = plan.product_count;
	fill_branched_horner_products_batch_<T, Width>(sig, base, cache);
	std::memset(d_base, 0, product_count * Width * sizeof(T));
	std::memset(d_increment, 0, cache.dimension * Width * sizeof(T));
	vec_batch_copy(d_base, d_out);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		vec_batch_add_inplace(
			d_base + plan.flat_to_product[flat] * Width,
			d_out + flat * Width);
	}

	for (uint64_t target_order = 1;
		target_order <= cache.max_nodes; ++target_order) {
		T* d_current = d_current_buffer;
		T* d_next = d_next_buffer;
		const uint64_t stage_width = cache.max_nodes + 1;
		uint64_t stage_key = target_order * stage_width + target_order;
		for (uint64_t pos = plan.stage_offsets[stage_key];
			pos < plan.stage_offsets[stage_key + 1]; ++pos) {
			vec_batch_fill(
				d_current + plan.stage_products[pos] * Width,
				static_cast<T>(0));
		}
		for (uint64_t flat_idx = cache.order_index[target_order];
			flat_idx < cache.order_index[target_order + 1]; ++flat_idx) {
			const uint64_t flat = flat_idx + 1;
			vec_batch_add_inplace(
				d_current + plan.flat_to_product[flat] * Width,
				d_out + flat * Width);
		}

		if (target_order == 1) {
			branched_horner_derivative_level_backprop_batch_<T, Width>(
				base, nullptr, increment, d_current, d_base, nullptr,
				d_increment, 1, 1, static_cast<T>(1), cache);
			continue;
		}

		T* state = states + product_count * Width;
		branched_horner_derivative_level_batch_<T, Width>(
			base, nullptr, increment, state, target_order, 1,
			static_cast<T>(1) / static_cast<T>(target_order), cache);
		for (uint64_t source_order = 1;
			source_order + 1 < target_order; ++source_order) {
			T* next_state = states
				+ (source_order + 1) * product_count * Width;
			branched_horner_derivative_level_batch_<T, Width>(
				base, state, increment, next_state, target_order, source_order + 1,
				static_cast<T>(1)
					/ static_cast<T>(target_order - source_order), cache);
			state = next_state;
		}

		stage_key = target_order * stage_width + target_order - 1;
		for (uint64_t pos = plan.stage_offsets[stage_key];
			pos < plan.stage_offsets[stage_key + 1]; ++pos) {
			vec_batch_fill(
				d_next + plan.stage_products[pos] * Width,
				static_cast<T>(0));
		}
		branched_horner_derivative_level_backprop_batch_<T, Width>(
			base, state, increment, d_current, d_base, d_next,
			d_increment, target_order, target_order, static_cast<T>(1), cache);
		std::swap(d_current, d_next);

		for (uint64_t source_order = target_order - 2;
			source_order >= 1; --source_order) {
			state = states + source_order * product_count * Width;
			stage_key = target_order * stage_width + source_order;
			for (uint64_t pos = plan.stage_offsets[stage_key];
				pos < plan.stage_offsets[stage_key + 1]; ++pos) {
				vec_batch_fill(
					d_next + plan.stage_products[pos] * Width,
					static_cast<T>(0));
			}
			branched_horner_derivative_level_backprop_batch_<T, Width>(
				base, state, increment, d_current, d_base, d_next,
				d_increment, target_order, source_order + 1,
				static_cast<T>(1)
					/ static_cast<T>(target_order - source_order), cache);
			std::swap(d_current, d_next);
		}

		branched_horner_derivative_level_backprop_batch_<T, Width>(
			base, nullptr, increment, d_current, d_base, nullptr,
			d_increment, target_order, 1,
			static_cast<T>(1) / static_cast<T>(target_order), cache);
	}

	std::memset(d_sig, 0, cache.total_length * Width * sizeof(T));
	for (uint64_t pos = plan.product_build.size(); pos > 0; --pos) {
		const uint64_t product = plan.product_build[pos - 1];
		const uint64_t parent = plan.product_build_parent[pos - 1];
		const uint64_t factor = plan.product_build_factor[pos - 1];
		const T* deriv = d_base + product * Width;
		vec_batch_multiply_add(
			d_base + parent * Width, deriv, sig + factor * Width);
		vec_batch_multiply_add(
			d_sig + factor * Width, deriv, base + parent * Width);
	}
	vec_batch_copy(d_sig, d_base);
}


template<std::floating_point T, size_t Width>
void linear_bsig_deriv_to_increment_deriv_batch_(
	const T* dF_dY,
	const T* increment,
	T* inc_derivs,
	uint64_t dimension,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	alignas(32) T base[Width];
	alignas(32) T prefix[Width];
	alignas(32) T suffix[Width];
	std::memset(inc_derivs, 0, dimension * Width * sizeof(T));

	const uint64_t num_trees = cache.total_length - 1;
	const uint8_t* labels = cache.node_labels_data.data();
	const uint64_t* offsets = cache.node_labels_offsets.data();
	for (uint64_t i = 0; i < num_trees; ++i) {
		const T* deriv = dF_dY + (i + 1) * Width;
		if (vec_batch_is_zero(deriv))
			continue;
		vec_batch_scale(
			base, deriv, static_cast<T>(cache.inv_tree_factorial[i]));
		vec_batch_fill(prefix, static_cast<T>(1));
		const uint64_t label_start = offsets[i];
		const uint64_t label_count = offsets[i + 1] - label_start;
		for (uint64_t j = 0; j < label_count; ++j) {
			vec_batch_fill(suffix, static_cast<T>(1));
			for (uint64_t k = j + 1; k < label_count; ++k) {
				vec_batch_multiply_inplace(
					suffix,
					increment + static_cast<uint64_t>(labels[label_start + k]) * Width);
			}
			const uint64_t label = labels[label_start + j];
			vec_batch_multiply_add3(
				inc_derivs + label * Width, base, prefix, suffix);
			vec_batch_multiply_inplace(prefix, increment + label * Width);
		}
	}
}


template<std::floating_point T, size_t Width>
void planar_monomial_derivs_to_increment_batch_(
	const T* monomial_derivs,
	const T* increment,
	T* inc_derivs,
	uint64_t dimension,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	std::memset(inc_derivs, 0, dimension * Width * sizeof(T));
	if (cache.max_nodes == 0)
		return;

	uint64_t offset = 1;
	for (uint64_t label = 0; label < dimension; ++label) {
		vec_batch_add_inplace(
			inc_derivs + label * Width,
			monomial_derivs + (offset + label) * Width);
	}
	if (cache.max_nodes == 1)
		return;

	offset += dimension;
	for (uint64_t label0 = 0; label0 < dimension; ++label0) {
		for (uint64_t label1 = 0; label1 < dimension; ++label1) {
			const T* deriv = monomial_derivs
				+ (offset + label0 * dimension + label1) * Width;
			if (vec_batch_is_zero(deriv))
				continue;
			vec_batch_multiply_add(
				inc_derivs + label0 * Width, deriv,
				increment + label1 * Width);
			vec_batch_multiply_add(
				inc_derivs + label1 * Width, deriv,
				increment + label0 * Width);
		}
	}
	if (cache.max_nodes == 2)
		return;

	offset += dimension * dimension;
	for (uint64_t label0 = 0; label0 < dimension; ++label0) {
		for (uint64_t label1 = 0; label1 < dimension; ++label1) {
			for (uint64_t label2 = 0; label2 < dimension; ++label2) {
				const T* deriv = monomial_derivs + (offset
					+ (label0 * dimension + label1) * dimension + label2) * Width;
				if (vec_batch_is_zero(deriv))
					continue;
				vec_batch_multiply_add3(
					inc_derivs + label0 * Width, deriv,
					increment + label1 * Width, increment + label2 * Width);
				vec_batch_multiply_add3(
					inc_derivs + label1 * Width, deriv,
					increment + label0 * Width, increment + label2 * Width);
				vec_batch_multiply_add3(
					inc_derivs + label2 * Width, deriv,
					increment + label0 * Width, increment + label1 * Width);
			}
		}
	}
}


template<std::floating_point T, size_t Width>
void planar_linear_bsig_deriv_to_increment_deriv_batch_(
	const T* dF_dY,
	const T* increment,
	T* inc_derivs,
	T* monomial_derivs,
	const uint64_t* flat_monomials,
	uint64_t monomial_count,
	uint64_t dimension,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	std::memset(monomial_derivs, 0, monomial_count * Width * sizeof(T));
	for (uint64_t tree_idx = 0; tree_idx + 1 < cache.total_length; ++tree_idx) {
		const T* deriv = dF_dY + (tree_idx + 1) * Width;
		if (vec_batch_is_zero(deriv))
			continue;
		vec_batch_scaled_add(
			monomial_derivs + flat_monomials[tree_idx + 1] * Width, deriv,
			static_cast<T>(cache.inv_tree_factorial[tree_idx]));
	}
	planar_monomial_derivs_to_increment_batch_<T, Width>(
		monomial_derivs, increment, inc_derivs, dimension, cache);
}


template<std::floating_point T, size_t Width>
void planar_butcher_uncombine_backprop_batch_(
	T* combined,
	const T* local_sig,
	T* d_previous,
	T* monomial_derivs,
	const uint64_t* flat_monomials,
	uint64_t monomial_count,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	alignas(32) T value[Width];
	std::memset(monomial_derivs, 0, monomial_count * Width * sizeof(T));
	const uint64_t* order_index = cache.order_index.data();
	const uint64_t* coproduct_offsets = cache.coproduct_offsets.data();
	const uint64_t* coproduct_data = cache.coproduct_data.data();
	for (uint64_t order = 1; order <= cache.max_nodes; ++order) {
		const uint64_t start = order_index[order];
		const uint64_t end = order_index[order + 1];
		for (uint64_t tree_idx = start; tree_idx < end; ++tree_idx) {
			const uint64_t flat = tree_idx + 1;
			const T* deriv = d_previous + flat * Width;
			vec_batch_scaled_add(
				monomial_derivs + flat_monomials[flat] * Width, deriv,
				static_cast<T>(cache.inv_tree_factorial[tree_idx]));
			vec_batch_subtract(
				value, combined + flat * Width, local_sig + flat * Width);
			uint64_t pos = coproduct_offsets[tree_idx];
			const uint64_t pos_end = coproduct_offsets[tree_idx + 1];
			while (pos < pos_end) {
				const uint64_t has_forest = coproduct_data[pos++];
				const uint64_t trunk = coproduct_data[pos++];
				if (has_forest == 0) {
					vec_batch_subtract_inplace(
						value, local_sig + trunk * Width);
					if (trunk != 0) {
						vec_batch_scaled_add(
							monomial_derivs + flat_monomials[trunk] * Width, deriv,
							static_cast<T>(cache.inv_tree_factorial[trunk - 1]));
					}
					continue;
				}
				const uint64_t forest = coproduct_data[pos++];
				vec_batch_subtract_product(
					value, local_sig + trunk * Width,
					combined + forest * Width);
				if (trunk != 0) {
					vec_batch_multiply_add_scaled(
						monomial_derivs + flat_monomials[trunk] * Width, deriv,
						combined + forest * Width,
						static_cast<T>(cache.inv_tree_factorial[trunk - 1]));
				}
				vec_batch_multiply_add(
					d_previous + forest * Width, deriv,
					local_sig + trunk * Width);
			}
			vec_batch_copy(combined + flat * Width, value);
		}
	}
}


template<std::floating_point T, size_t Width>
struct BranchedSigBatchBackpropWorkspace_ {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	struct AlignedDelete_ {
		void operator()(T* ptr) const noexcept {
			::operator delete[](ptr, std::align_val_t(32));
		}
	};
	using AlignedArray = std::unique_ptr<T[], AlignedDelete_>;

	static AlignedArray allocate_(uint64_t size) {
		return AlignedArray(static_cast<T*>(::operator new[](
			size * sizeof(T), std::align_val_t(32))));
	}

	AlignedArray bsig;
	AlignedArray derivs;
	AlignedArray increment;
	AlignedArray local_derivs;
	AlignedArray inc_derivs;
	AlignedArray local_log;
	AlignedArray horner_base;
	AlignedArray horner_current;
	AlignedArray horner_next;
	AlignedArray horner_states;
	AlignedArray horner_d_base;
	AlignedArray linear_monomial_derivs;

	BranchedSigBatchBackpropWorkspace_(
		uint64_t dimension,
		uint64_t total_length,
		uint64_t max_nodes,
		uint64_t product_count,
		uint64_t linear_monomial_count,
		bool planar
	) : bsig(allocate_(total_length * Width)),
		derivs(allocate_(total_length * Width)),
		increment(allocate_(dimension * Width)),
		inc_derivs(allocate_(dimension * Width)) {
		if (planar) {
			local_log = allocate_(total_length * Width);
			linear_monomial_derivs
				= allocate_(linear_monomial_count * Width);
		}
		else {
			local_derivs = allocate_(total_length * Width);
			horner_base = allocate_(product_count * Width);
			horner_current = allocate_(product_count * Width);
			horner_next = allocate_(product_count * Width);
			horner_states
				= allocate_((max_nodes + 1) * product_count * Width);
			horner_d_base = allocate_(product_count * Width);
		}
	}
};


template<std::floating_point T, size_t Width>
void branched_sig_backprop_batch_inplace_(
	const T* path,
	T* out,
	const T* bsig_derivs_in,
	const T* bsig_in,
	uint64_t batch_start,
	uint64_t active_width,
	uint64_t dimension,
	uint64_t length,
	bool scalar_term,
	const uint64_t* flat_monomials,
	uint64_t monomial_count,
	BranchedSigBatchBackpropWorkspace_<T, Width>& workspace,
	const BranchedSigCache& cache
) {
	static_assert(Width == vec_batch_bytes / sizeof(T));
	const uint64_t total_len = cache.total_length;
	const uint64_t in_stride = scalar_term ? total_len : total_len - 1;
	const uint64_t flat_path_length = length * dimension;

	for (uint64_t flat = 0; flat < total_len; ++flat) {
		T* sig_value = workspace.bsig.get() + flat * Width;
		T* deriv_value = workspace.derivs.get() + flat * Width;
		vec_batch_fill(
			sig_value, flat == 0 ? static_cast<T>(1) : static_cast<T>(0));
		vec_batch_fill(deriv_value, static_cast<T>(0));
		for (uint64_t lane = 0; lane < active_width; ++lane) {
			const uint64_t batch = batch_start + lane;
			if (scalar_term) {
				sig_value[lane] = bsig_in[batch * total_len + flat];
				deriv_value[lane] = bsig_derivs_in[batch * total_len + flat];
			}
			else if (flat != 0) {
				sig_value[lane] = bsig_in[batch * in_stride + flat - 1];
				deriv_value[lane] = bsig_derivs_in[batch * in_stride + flat - 1];
			}
		}
	}

	for (uint64_t lane = 0; lane < active_width; ++lane) {
		std::memset(
			out + (batch_start + lane) * flat_path_length,
			0, flat_path_length * sizeof(T));
	}

	if (length <= 1)
		return;

	const int steps = static_cast<int>(length - 1);
	for (int seg = steps - 1; seg >= 0; --seg) {
		for (uint64_t d = 0; d < dimension; ++d) {
			T* value = workspace.increment.get() + d * Width;
			vec_batch_fill(value, static_cast<T>(0));
			for (uint64_t lane = 0; lane < active_width; ++lane) {
				const T* path_ptr = path
					+ (batch_start + lane) * flat_path_length;
				value[lane] = path_ptr[(seg + 1) * dimension + d]
					- path_ptr[seg * dimension + d];
			}
		}

		if (cache.planar) {
			if (seg > 0) {
				linear_branched_sig_batch_<T, Width>(
					workspace.increment.get(), workspace.local_log.get(),
					cache);
				planar_butcher_uncombine_backprop_batch_<T, Width>(
					workspace.bsig.get(), workspace.local_log.get(),
					workspace.derivs.get(), workspace.linear_monomial_derivs.get(),
					flat_monomials, monomial_count, cache);
				planar_monomial_derivs_to_increment_batch_<T, Width>(
					workspace.linear_monomial_derivs.get(), workspace.increment.get(),
					workspace.inc_derivs.get(),
					dimension, cache);
			}
			else {
				planar_linear_bsig_deriv_to_increment_deriv_batch_<T, Width>(
					workspace.derivs.get(), workspace.increment.get(),
					workspace.inc_derivs.get(),
					workspace.linear_monomial_derivs.get(), flat_monomials,
					monomial_count, dimension, cache);
			}
		}
		else {
			if (seg > 0) {
				for (uint64_t d = 0; d < dimension; ++d) {
					vec_batch_negate_inplace(
						workspace.increment.get() + d * Width);
				}
				branched_horner_step_batch_<T, Width>(
					workspace.bsig.get(), workspace.increment.get(),
					workspace.horner_base.get(), workspace.horner_current.get(),
					workspace.horner_next.get(), cache);
				for (uint64_t d = 0; d < dimension; ++d) {
					vec_batch_negate_inplace(
						workspace.increment.get() + d * Width);
				}
				branched_horner_step_backprop_batch_<T, Width>(
					workspace.bsig.get(), workspace.increment.get(),
					workspace.derivs.get(), workspace.local_derivs.get(),
					workspace.inc_derivs.get(), workspace.horner_base.get(),
					workspace.horner_states.get(), workspace.horner_d_base.get(),
					workspace.horner_current.get(), workspace.horner_next.get(), cache);
				std::memcpy(
					workspace.derivs.get(), workspace.local_derivs.get(),
					total_len * Width * sizeof(T));
			}
			else {
				linear_bsig_deriv_to_increment_deriv_batch_<T, Width>(
					workspace.derivs.get(), workspace.increment.get(),
					workspace.inc_derivs.get(), dimension, cache);
			}
		}

		for (uint64_t lane = 0; lane < active_width; ++lane) {
			T* positive = out + (batch_start + lane) * flat_path_length
				+ static_cast<uint64_t>(seg + 1) * dimension;
			T* negative = positive - dimension;
			for (uint64_t d = 0; d < dimension; ++d) {
				const T value = workspace.inc_derivs[d * Width + lane];
				positive[d] += value;
				negative[d] -= value;
			}
		}
	}
}

#endif

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

	auto work = [&](uint64_t b, BranchedSigBackpropWorkspace_<T>& workspace) {
		Path<T> path_obj(path + b * flat_path_length, dimension, length, time_aug, lead_lag, end_time);

		if (scalar_term) {
			std::memcpy(workspace.bsig.get(), bsig_in + b * total_len, total_len * sizeof(T));
			std::memcpy(workspace.derivs.get(), bsig_derivs_in + b * total_len, total_len * sizeof(T));
		} else {
			workspace.bsig[0] = static_cast<T>(1);
			std::memcpy(workspace.bsig.get() + 1, bsig_in + b * in_stride, (total_len - 1) * sizeof(T));
			workspace.derivs[0] = static_cast<T>(0);
			std::memcpy(workspace.derivs.get() + 1, bsig_derivs_in + b * in_stride, (total_len - 1) * sizeof(T));
		}

		T* out_ptr = out + b * flat_path_length;
		std::memset(out_ptr, 0, flat_path_length * sizeof(T));

		branched_sig_backprop_inplace_(
			path_obj, out_ptr, workspace.derivs.get(), workspace.bsig.get(),
			workspace.increment.get(),
			workspace.local_derivs.get(), workspace.inc_derivs.get(),
			workspace.local_log.get(), workspace.local_sig.get(),
			workspace.d_local_sig.get(), workspace.d_local_log.get(),
			workspace.horner_base.get(), workspace.horner_log.get(),
			workspace.horner_current.get(),
			workspace.horner_next.get(), workspace.horner_states.get(),
			workspace.horner_d_base.get(),
			workspace.correction_node_derivs.get(),
			has_correction ? correction + b * correction_batch_stride : nullptr,
			correction_len, correction_segment_stride, has_correction, cache);
	};

	auto work_range = [&](uint64_t start, uint64_t end) {
		if (start == end)
			return;
		BranchedSigBackpropWorkspace_<T> workspace(
			aug_dim, total_len, max_nodes, cache.horner.product_count,
			cache.horner.planar_log_monomial_parent.size(),
			cache.horner.correction_horner_constants.size(),
			has_correction, cache.planar);
		for (uint64_t b = start; b < end; ++b)
			work(b, workspace);
	};
#ifdef VEC
	if (!has_correction && !time_aug && !lead_lag && batch_size > 1
		&& (!planar || max_nodes <= 3)) {
		constexpr size_t batch_width = vec_batch_bytes / sizeof(T);
		const uint64_t pack_count = (batch_size + batch_width - 1) / batch_width;
		std::vector<uint64_t> flat_monomials;
		uint64_t monomial_count = 0;
		if (planar) {
			flat_monomials.resize(total_len, 0);
			uint64_t level_size = 1;
			uint64_t level_offset = 1;
			for (uint64_t order = 1; order <= max_nodes; ++order) {
				level_size *= dimension;
				for (uint64_t tree_idx = cache.order_index[order];
					tree_idx < cache.order_index[order + 1]; ++tree_idx) {
					uint64_t encoded = 0;
					for (uint64_t pos = cache.node_labels_offsets[tree_idx];
						pos < cache.node_labels_offsets[tree_idx + 1]; ++pos) {
						encoded = encoded * dimension + cache.node_labels_data[pos];
					}
					flat_monomials[tree_idx + 1] = level_offset + encoded;
				}
				level_offset += level_size;
			}
			monomial_count = level_offset;
		}
		auto work_pack_range = [&](uint64_t start, uint64_t end) {
			BranchedSigBatchBackpropWorkspace_<T, batch_width> workspace(
				aug_dim, total_len, max_nodes, cache.horner.product_count,
				monomial_count, cache.planar);
			for (uint64_t pack = start; pack < end; ++pack) {
				const uint64_t batch_start = pack * batch_width;
				const uint64_t active_width = std::min<uint64_t>(
					batch_width, batch_size - batch_start);
				branched_sig_backprop_batch_inplace_(
					path, out, bsig_derivs_in, bsig_in, batch_start, active_width,
					dimension, length, scalar_term, flat_monomials.data(),
					monomial_count, workspace, cache);
			}
		};
		if (n_jobs == 1 || pack_count == 1)
			work_pack_range(0, pack_count);
		else {
			int pack_jobs = n_jobs;
			if (n_jobs < 0) {
				const int max_threads = get_max_threads() + 1 + n_jobs;
				if (max_threads < 1) {
					throw std::invalid_argument(
						"received negative n_jobs which is less than max_threads + 1; n_jobs too low");
				}
				const uint64_t packs_per_thread = (pack_count
					+ static_cast<uint64_t>(max_threads) - 1)
					/ static_cast<uint64_t>(max_threads);
				pack_jobs = static_cast<int>((pack_count + packs_per_thread - 1)
					/ packs_per_thread);
			}
			spawn_batch_threads(pack_count, pack_jobs, work_pack_range);
		}
		return;
	}
#endif
	if (n_jobs == 1 || batch_size == 1)
		work_range(0, batch_size);
	else
		spawn_batch_threads(batch_size, n_jobs, work_range);
}
