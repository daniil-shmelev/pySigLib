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

#include "preparation/branched_sig/branched_log_plan.h"

#include <concepts>
#include <cstdint>
#include <utility>
#include <vector>

template<std::floating_point T>
struct BranchedLogHornerWorkspace {
	std::vector<T> h;
	std::vector<T> current;
	std::vector<T> next;

	explicit BranchedLogHornerWorkspace(uint64_t product_count)
		: h(product_count), current(product_count), next(product_count) {}
};

inline uint64_t branched_log_product_for_flat(
	const BranchedLogHornerPlan& plan,
	bool planar,
	uint64_t flat
) {
	return planar ? flat : plan.flat_to_product[flat];
}

template<std::floating_point T, typename FlatValue>
void fill_branched_log_horner_products(
	const BranchedLogHornerPlan& plan,
	bool planar,
	FlatValue flat_value,
	T* h
) {
	h[0] = T(0);
	if (planar) {
		for (uint64_t product = 1; product < plan.product_count; ++product)
			h[product] = flat_value(product);
		return;
	}
	for (uint64_t product = 1; product < plan.product_count; ++product) {
		const uint64_t parent = plan.cpu_products.parent[product];
		const T factor = flat_value(plan.cpu_products.last_factor[product]);
		h[product] = parent == 0 ? factor : h[parent] * factor;
	}
}

template<std::floating_point T, typename FlatValue, typename SetOutput>
void branched_log_horner_forward(
	uint64_t total_length,
	uint64_t max_nodes,
	bool planar,
	const BranchedLogHornerPlan& plan,
	FlatValue flat_value,
	SetOutput set_output,
	BranchedLogHornerWorkspace<T>& workspace
) {
	if (max_nodes == 0)
		return;

	T* const h = workspace.h.data();
	T* current = workspace.current.data();
	T* next = workspace.next.data();
	fill_branched_log_horner_products<T>(plan, planar, flat_value, h);
	current[0] = T(0);
	next[0] = T(0);

	if (max_nodes == 1) {
		for (uint64_t flat = 1; flat < total_length; ++flat)
			set_output(flat, flat_value(flat));
		return;
	}

	const T initial_scale = T(1) / static_cast<T>(max_nodes);
	for (uint64_t product = 1; product < plan.product_count; ++product) {
		if (plan.product_node_counts[product] == 1)
			current[product] = initial_scale * h[product];
	}

	for (uint64_t k = max_nodes - 1; k > 1; --k) {
		const T scale = T(1) / static_cast<T>(k);
		const uint64_t max_product_nodes = max_nodes - k + 1;
		for (uint64_t product = 1; product < plan.product_count; ++product) {
			if (plan.product_node_counts[product] > max_product_nodes)
				continue;
			T value = T(0);
			const uint64_t start = plan.coproduct_offsets[product];
			const uint64_t end = plan.coproduct_offsets[product + 1];
			for (uint64_t pos = start; pos < end; pos += 2) {
				value += current[plan.coproduct_pairs[pos]]
					* h[plan.coproduct_pairs[pos + 1]];
			}
			next[product] = scale * h[product] - value;
		}
		std::swap(current, next);
	}

	for (uint64_t flat = 1; flat < total_length; ++flat) {
		const uint64_t product = branched_log_product_for_flat(
			plan, planar, flat);
		T value = T(0);
		const uint64_t start = plan.coproduct_offsets[product];
		const uint64_t end = plan.coproduct_offsets[product + 1];
		for (uint64_t pos = start; pos < end; pos += 2) {
			value += current[plan.coproduct_pairs[pos]]
				* h[plan.coproduct_pairs[pos + 1]];
		}
		set_output(flat, flat_value(flat) - value);
	}
}
