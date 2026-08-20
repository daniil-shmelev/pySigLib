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

#include "branched_sig_cache.h"

#include "branched_log_plan.h"
#include "../../branched_log_horner.h"

#include "../../trees/basis_counts.h"
#include "../../trees/coproduct.h"
#include "../../trees/tree.h"

#include <algorithm>
#include <map>
#include <unordered_map>

namespace {

bool chain_word_index(
	TreeId tree_id,
	const TreeTable& trees,
	uint64_t dimension,
	uint64_t& word_index
) {
	word_index = trees.tree(tree_id).root_label();
	while (true) {
		const Tree& tree = trees.tree(tree_id);
		if (tree.children().empty())
			return true;
		if (tree.children().size() != 1)
			return false;
		tree_id = tree.children()[0];
		word_index = checked_mul_add_(
			word_index, dimension, trees.tree(tree_id).root_label(),
			"chain word index overflow");
	}
}


void build_chain_indices(
	BranchedSigCache& cache,
	const TreeTable& trees,
	std::span<const Forest> forests = {}
) {
	cache.chain_index_offsets.assign(cache.max_nodes + 2, 0);
	uint64_t chain_size = 0;
	uint64_t level_size = 1;
	for (uint64_t level = 1; level <= cache.max_nodes; ++level) {
		cache.chain_index_offsets[level] = chain_size;
		if (cache.dimension != 0 && level_size > UINT64_MAX / cache.dimension)
			throw std::overflow_error("chain index size overflow");
		level_size *= cache.dimension;
		if (chain_size > UINT64_MAX - level_size)
			throw std::overflow_error("chain index size overflow");
		chain_size += level_size;
	}
	cache.chain_index_offsets[cache.max_nodes + 1] = chain_size;
	cache.chain_indices.assign(chain_size, 0);

	if (!forests.empty()) {
		for (uint64_t index = 0; index < forests.size(); ++index) {
			const Forest& forest = forests[index];
			if (forest.size() != 1)
				continue;
			const TreeId tree_id = forest[0];
			const uint64_t order = trees.tree(tree_id).node_count();
			uint64_t word_index = 0;
			if (chain_word_index(tree_id, trees, cache.dimension, word_index)) {
				cache.chain_indices[cache.chain_index_offsets[order] + word_index]
					= index + 1;
			}
		}
		return;
	}

	for (uint64_t order = 1; order <= cache.max_nodes; ++order) {
		for (TreeId tree_id = cache.order_index[order];
			tree_id < cache.order_index[order + 1]; ++tree_id) {
			uint64_t word_index = 0;
			if (chain_word_index(tree_id, trees, cache.dimension, word_index)) {
				cache.chain_indices[cache.chain_index_offsets[order] + word_index]
					= tree_id + 1;
			}
		}
	}
}


void build_tree_cuts(
	TreeTable& trees,
	const std::vector<uint64_t>& order_offsets,
	uint64_t max_nodes,
	std::vector<std::vector<TreeCut>>& cuts
) {
	cuts.resize(static_cast<size_t>(trees.size()));
	std::vector<uint8_t> ready(static_cast<size_t>(trees.size()), 0);
	const uint64_t dimension = trees.dimension();
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		const TreeId start = order_offsets[order];
		const TreeId end = order_offsets[order + 1];
		for (TreeId tree_id = start; tree_id < end; tree_id += dimension) {
			ensure_tree_cuts(tree_id, trees, cuts, ready);
			for (uint64_t label = 1; label < dimension && tree_id + label < end; ++label) {
				auto& destination = cuts[tree_id + label];
				destination.resize(cuts[tree_id].size());
				for (size_t cut_index = 0;
					cut_index < cuts[tree_id].size(); ++cut_index) {
					destination[cut_index].pruned = cuts[tree_id][cut_index].pruned;
					destination[cut_index].trunk =
						cuts[tree_id][cut_index].trunk + label;
				}
				ready[tree_id + label] = 1;
			}
		}
	}
}


void enumerate_ordered_forest_basis(
	const TreeTable& trees,
	uint64_t max_nodes,
	std::vector<Forest>& forests,
	std::vector<uint64_t>& order_offsets
) {
	forests.clear();
	order_offsets.assign(max_nodes + 2, 0);
	Forest current;
	const auto enumerate = [&](auto&& self, uint64_t remaining) -> void {
		if (remaining == 0) {
			forests.push_back(current);
			return;
		}
		for (TreeId tree_id = 0; tree_id < trees.size(); ++tree_id) {
			const uint64_t nodes = trees.tree(tree_id).node_count();
			if (nodes > remaining)
				break;
			current.push_back(tree_id);
			self(self, remaining - nodes);
			current.pop_back();
		}
	};
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		order_offsets[order] = forests.size();
		enumerate(enumerate, order);
	}
	order_offsets[max_nodes + 1] = forests.size();
}


BranchedSigCache build_nonplanar_cache(
	uint64_t dimension,
	uint64_t max_nodes
) {
	BranchedSigCache cache;
	cache.dimension = dimension;
	cache.max_nodes = max_nodes;

	TreeTable trees(dimension, TreeKind::NonPlanar);
	const uint64_t tree_count = compute_branched_sig_length(
		dimension, max_nodes, false) - 1;
	trees.reserve(static_cast<size_t>(tree_count));
	enumerate_trees(trees, max_nodes, cache.order_index);
	cache.total_length = tree_count + 1;
	cache.inv_tree_factorial.resize(tree_count);
	cache.node_labels_offsets.resize(tree_count + 1);
	for (TreeId tree_id = 0; tree_id < tree_count; ++tree_id) {
		const Tree& tree = trees.tree(tree_id);
		cache.inv_tree_factorial[tree_id] = 1.0 / tree.tree_factorial();
		cache.node_labels_offsets[tree_id] = cache.node_labels_data.size();
		const auto labels = tree.node_labels();
		cache.node_labels_data.insert(
			cache.node_labels_data.end(), labels.begin(), labels.end());
	}
	cache.node_labels_offsets[tree_count] = cache.node_labels_data.size();
	build_chain_indices(cache, trees);

	std::vector<std::vector<TreeCut>> tree_cuts;
	build_tree_cuts(trees, cache.order_index, max_nodes, tree_cuts);
	uint64_t coproduct_size = 0;
	for (const auto& cuts : tree_cuts) {
		for (const TreeCut& cut : cuts)
			coproduct_size += 2 + cut.pruned.size();
	}
	cache.coproduct_data.reserve(coproduct_size);
	cache.coproduct_offsets.resize(tree_count + 1, 0);
	for (TreeId tree_id = 0; tree_id < tree_count; ++tree_id) {
		cache.coproduct_offsets[tree_id] = cache.coproduct_data.size();
		for (const TreeCut& cut : tree_cuts[tree_id]) {
			cache.coproduct_data.push_back(cut.pruned.size());
			cache.coproduct_data.push_back(cut.trunk + 1);
			for (TreeId pruned : cut.pruned)
				cache.coproduct_data.push_back(pruned + 1);
		}
	}
	cache.coproduct_offsets[tree_count] = cache.coproduct_data.size();
	return cache;
}


BranchedSigCache build_planar_cache(
	uint64_t dimension,
	uint64_t max_nodes
) {
	BranchedSigCache cache;
	cache.dimension = dimension;
	cache.max_nodes = max_nodes;
	cache.planar = true;

	TreeTable trees(dimension, TreeKind::Planar);
	std::vector<uint64_t> tree_order_offsets;
	enumerate_trees(trees, max_nodes, tree_order_offsets);
	std::vector<Forest> forests;
	enumerate_ordered_forest_basis(
		trees, max_nodes, forests, cache.order_index);
	const uint64_t basis_size = static_cast<uint64_t>(forests.size());
	cache.total_length = basis_size + 1;

	std::unordered_map<Forest, uint64_t, Forest::Hash> flat_indices;
	flat_indices.reserve(static_cast<size_t>(basis_size));
	for (uint64_t index = 0; index < basis_size; ++index)
		flat_indices.emplace(forests[index], index + 1);
	const auto flat_index = [&flat_indices](const Forest& forest) {
		if (forest.empty())
			return uint64_t(0);
		const auto found = flat_indices.find(forest);
		if (found == flat_indices.end())
			throw std::runtime_error("forest not found in MKW basis");
		return found->second;
	};

	cache.inv_tree_factorial.resize(basis_size);
	cache.node_labels_offsets.resize(basis_size + 1);
	cache.basis_forest_offsets.resize(basis_size + 1);
	for (uint64_t index = 0; index < basis_size; ++index) {
		const Forest& forest = forests[index];
		double inverse_factorial = 1.0;
		double forest_factorial = 1.0;
		for (uint64_t k = 2; k <= forest.size(); ++k)
			forest_factorial *= static_cast<double>(k);
		for (TreeId tree_id : forest)
			inverse_factorial /= trees.tree(tree_id).tree_factorial();
		cache.inv_tree_factorial[index] = inverse_factorial / forest_factorial;

		cache.node_labels_offsets[index] = cache.node_labels_data.size();
		cache.basis_forest_offsets[index] = cache.basis_forest_data.size();
		const auto labels = forest.node_labels(trees);
		cache.node_labels_data.insert(
			cache.node_labels_data.end(), labels.begin(), labels.end());
		for (TreeId tree_id : forest)
			cache.basis_forest_data.push_back(tree_id);
	}
	cache.node_labels_offsets[basis_size] = cache.node_labels_data.size();
	cache.basis_forest_offsets[basis_size] = cache.basis_forest_data.size();
	build_chain_indices(cache, trees, forests);

	std::vector<std::vector<TreeCut>> tree_cuts;
	build_tree_cuts(trees, tree_order_offsets, max_nodes, tree_cuts);
	cache.coproduct_offsets.resize(basis_size + 1, 0);
	for (uint64_t index = 0; index < basis_size; ++index) {
		cache.coproduct_offsets[index] = cache.coproduct_data.size();
		std::vector<CoproductTerm> terms;
		if (forests[index].size() == 1) {
			const TreeId tree_id = forests[index][0];
			for (const TreeCut& cut : tree_cuts[tree_id])
				terms.push_back({ cut.pruned, { cut.trunk } });
		} else {
			enumerate_mkw_forest_coproduct_terms(
				forests[index], tree_cuts, terms);
		}

		const uint64_t self_flat = index + 1;
		for (const CoproductTerm& term : terms) {
			const uint64_t left_flat = flat_index(term.left);
			const uint64_t right_flat = flat_index(term.right);
			if ((left_flat == 0 && right_flat == self_flat)
				|| (left_flat == self_flat && right_flat == 0))
				continue;
			cache.coproduct_data.push_back(left_flat == 0 ? 0 : 1);
			cache.coproduct_data.push_back(right_flat);
			if (left_flat != 0)
				cache.coproduct_data.push_back(left_flat);
		}
	}
	cache.coproduct_offsets[basis_size] = cache.coproduct_data.size();
	return cache;
}


using CorrectionMonomial = std::vector<uint64_t>;
using CorrectionPolynomial = std::map<CorrectionMonomial, double>;

void add_correction_polynomial_(
	CorrectionPolynomial& out,
	const CorrectionPolynomial& source,
	double scale = 1.0
) {
	for (const auto& [monomial, coefficient] : source)
		out[monomial] += scale * coefficient;
}

CorrectionPolynomial multiply_correction_polynomials_(
	const CorrectionPolynomial& left,
	const CorrectionPolynomial& right
) {
	CorrectionPolynomial out;
	for (const auto& [left_monomial, left_coefficient] : left) {
		for (const auto& [right_monomial, right_coefficient] : right) {
			CorrectionMonomial monomial;
			monomial.reserve(left_monomial.size() + right_monomial.size());
			std::merge(
				left_monomial.begin(), left_monomial.end(),
				right_monomial.begin(), right_monomial.end(),
				std::back_inserter(monomial));
			out[std::move(monomial)] += left_coefficient * right_coefficient;
		}
	}
	return out;
}

uint64_t build_correction_horner_node_(
	const CorrectionPolynomial& polynomial,
	BranchedSigHornerPlan& plan
) {
	double constant = 0.0;
	std::map<uint64_t, CorrectionPolynomial> groups;
	for (const auto& [monomial, coefficient] : polynomial) {
		if (monomial.empty()) {
			constant += coefficient;
			continue;
		}
		CorrectionMonomial tail(monomial.begin() + 1, monomial.end());
		groups[monomial.front()][std::move(tail)] += coefficient;
	}

	std::vector<std::pair<uint64_t, uint64_t>> children;
	children.reserve(groups.size());
	for (const auto& [variable, child] : groups) {
		children.push_back({
			variable, build_correction_horner_node_(child, plan) });
	}

	const uint64_t node = plan.correction_horner_constants.size();
	plan.correction_horner_constants.push_back(constant);
	for (const auto& [variable, child] : children) {
		plan.correction_horner_variables.push_back(variable);
		plan.correction_horner_children.push_back(child);
	}
	plan.correction_horner_node_offsets.push_back(
		plan.correction_horner_variables.size());
	return node;
}

void populate_correction_horner_plan_(
	const BranchedSigCache& cache,
	BranchedSigHornerPlan& plan
) {
	std::vector<uint8_t> active(cache.total_length, 0);
	for (const uint64_t flat : cache.chain_indices) {
		if (flat != 0)
			active[flat] = 1;
	}

	std::vector<CorrectionPolynomial> output(cache.total_length);
	std::vector<CorrectionPolynomial> power(cache.total_length);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		if (active[flat] != 0)
			power[flat][CorrectionMonomial{ flat }] = 1.0;
	}

	double inverse_factorial = 1.0;
	for (uint64_t k = 1; k <= cache.max_nodes; ++k) {
		inverse_factorial /= static_cast<double>(k);
		for (uint64_t flat = 1; flat < cache.total_length; ++flat)
			add_correction_polynomial_(output[flat], power[flat], inverse_factorial);
		if (k == cache.max_nodes)
			break;

		std::vector<CorrectionPolynomial> next(cache.total_length);
		for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
			uint64_t pos = cache.coproduct_offsets[flat - 1];
			const uint64_t end = cache.coproduct_offsets[flat];
			while (pos < end) {
				const uint64_t forest_size = cache.coproduct_data[pos++];
				const uint64_t trunk = cache.coproduct_data[pos++];
				if (active[trunk] == 0) {
					pos += forest_size;
					continue;
				}

				CorrectionPolynomial term;
				term[CorrectionMonomial{ trunk }] = 1.0;
				for (uint64_t factor = 0; factor < forest_size; ++factor) {
					const uint64_t forest_flat = cache.coproduct_data[pos++];
					if (power[forest_flat].empty()) {
						term.clear();
						pos += forest_size - factor - 1;
						break;
					}
					term = multiply_correction_polynomials_(
						term, power[forest_flat]);
				}
				add_correction_polynomial_(next[flat], term);
			}
		}
		power.swap(next);
	}

	plan.correction_horner_node_offsets.push_back(0);
	plan.correction_horner_roots.assign(cache.total_length, UINT64_MAX);
	for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
		if (!output[flat].empty()) {
			plan.correction_horner_roots[flat]
				= build_correction_horner_node_(output[flat], plan);
		}
	}
}

}  // namespace


static void populate_branched_sig_horner_plan(BranchedSigCache& cache) {
	BranchedLogHornerPlan full = build_branched_log_horner_plan(cache);
	BranchedSigHornerPlan plan;
	plan.product_count = full.product_count;
	plan.product_parent = full.cpu_products.parent;
	plan.product_factor = full.cpu_products.last_factor;
	plan.product_node_counts = full.product_node_counts;
	plan.coproduct_offsets = full.coproduct_offsets;
	plan.coproduct_pairs = full.coproduct_pairs;
	populate_correction_horner_plan_(cache, plan);
	if (cache.planar) {
		plan.planar_log_coefficients.assign(cache.total_length, 0.0);
		BranchedLogHornerWorkspace<double> workspace(full.product_count);
		branched_log_horner_forward<double>(
			cache.total_length, cache.max_nodes, true, full,
			[&cache](uint64_t flat) {
				return cache.inv_tree_factorial[flat - 1];
			},
			[&plan](uint64_t flat, double value) {
				plan.planar_log_coefficients[flat] = value;
			},
			workspace);
		for (uint64_t flat = 1; flat < cache.total_length; ++flat) {
			if (plan.planar_log_coefficients[flat] != 0.0)
				plan.planar_log_flats.push_back(flat);
		}
		using Monomial = std::vector<uint64_t>;
		std::unordered_map<Monomial, uint64_t, BranchedLogProductHash>
			monomial_index;
		monomial_index.emplace(Monomial{}, 0);
		plan.planar_log_monomial_parent.push_back(0);
		plan.planar_log_monomial_label.push_back(0);
		const auto ensure_monomial = [&monomial_index, &plan](
			auto&& self, const Monomial& monomial) -> uint64_t {
			const auto found = monomial_index.find(monomial);
			if (found != monomial_index.end())
				return found->second;
			Monomial parent = monomial;
			const uint64_t label = parent.back();
			parent.pop_back();
			const uint64_t parent_index = self(self, parent);
			const uint64_t index = plan.planar_log_monomial_parent.size();
			monomial_index.emplace(monomial, index);
			plan.planar_log_monomial_parent.push_back(parent_index);
			plan.planar_log_monomial_label.push_back(label);
			return index;
		};
		for (const uint64_t flat : plan.planar_log_flats) {
			Monomial monomial;
			const uint64_t start = cache.node_labels_offsets[flat - 1];
			const uint64_t end = cache.node_labels_offsets[flat];
			monomial.reserve(end - start);
			for (uint64_t pos = start; pos < end; ++pos)
				monomial.push_back(cache.node_labels_data[pos]);
			std::sort(monomial.begin(), monomial.end());
			plan.planar_log_flat_monomial.push_back(
				ensure_monomial(ensure_monomial, monomial));
		}

		plan.planar_coproduct_offsets.resize(full.coproduct_offsets.size());
		plan.planar_coproduct_left.reserve(full.coproduct_pairs.size() / 2);
		plan.planar_coproduct_right.reserve(full.coproduct_pairs.size() / 2);
		for (uint64_t product = 0; product < full.product_count; ++product) {
			plan.planar_coproduct_offsets[product]
				= plan.planar_coproduct_left.size();
			const uint64_t start = full.coproduct_offsets[product];
			const uint64_t end = full.coproduct_offsets[product + 1];
			for (uint64_t pos = start; pos < end; pos += 2) {
				const uint64_t right = full.coproduct_pairs[pos + 1];
				if (plan.planar_log_coefficients[right] == 0.0)
					continue;
				plan.planar_coproduct_left.push_back(
					full.coproduct_pairs[pos]);
				plan.planar_coproduct_right.push_back(right);
			}
		}
		plan.planar_coproduct_offsets[full.product_count]
			= plan.planar_coproduct_left.size();
		cache.horner = std::move(plan);
		return;
	}
	plan.flat_to_product = full.flat_to_product;

	std::vector<uint64_t> product_labels(plan.product_count, UINT64_MAX);
	if (cache.max_nodes >= 1) {
		for (uint64_t label = 0; label < cache.dimension; ++label) {
			const uint64_t flat = cache.order_index[1] + label + 1;
			product_labels[plan.flat_to_product[flat]] = label;
		}
	}

	std::vector<uint64_t> full_derivative_offsets(plan.product_count + 1, 0);
	std::vector<uint64_t> full_derivative_left;
	std::vector<uint64_t> full_derivative_label;
	for (uint64_t product = 0; product < plan.product_count; ++product) {
		full_derivative_offsets[product] = full_derivative_left.size();
		if (full.product_node_counts[product] == 1) {
			const uint64_t label = product_labels[product];
			if (label == UINT64_MAX)
				throw std::runtime_error("Invalid degree-one branched Horner product");
			full_derivative_left.push_back(0);
			full_derivative_label.push_back(label);
		}

		const uint64_t start = full.coproduct_offsets[product];
		const uint64_t end = full.coproduct_offsets[product + 1];
		for (uint64_t pos = start; pos < end; pos += 2) {
			const uint64_t right = full.coproduct_pairs[pos + 1];
			const uint64_t label = product_labels[right];
			if (label == UINT64_MAX)
				continue;
			full_derivative_left.push_back(full.coproduct_pairs[pos]);
			full_derivative_label.push_back(label);
		}
	}
	full_derivative_offsets[plan.product_count] = full_derivative_left.size();

	const uint64_t stage_width = cache.max_nodes + 1;
	plan.stage_offsets.resize(stage_width * stage_width + 1, 0);
	for (uint64_t target_order = 0;
		target_order <= cache.max_nodes; ++target_order) {
		std::vector<std::vector<uint64_t>> active(stage_width);
		if (target_order > 0) {
			for (uint64_t flat_idx = cache.order_index[target_order];
				flat_idx < cache.order_index[target_order + 1]; ++flat_idx) {
				active[target_order].push_back(
					plan.flat_to_product[flat_idx + 1]);
			}
			std::sort(active[target_order].begin(), active[target_order].end());
			active[target_order].erase(
				std::unique(active[target_order].begin(), active[target_order].end()),
				active[target_order].end());

			for (uint64_t stage_order = target_order;
				stage_order > 1; --stage_order) {
				auto& previous = active[stage_order - 1];
				for (const uint64_t product : active[stage_order]) {
					const uint64_t start = full_derivative_offsets[product];
					const uint64_t end = full_derivative_offsets[product + 1];
					previous.insert(previous.end(),
						full_derivative_left.begin() + start,
						full_derivative_left.begin() + end);
				}
				std::sort(previous.begin(), previous.end());
				previous.erase(
					std::unique(previous.begin(), previous.end()), previous.end());
			}
		}

		for (uint64_t stage_order = 0;
			stage_order <= cache.max_nodes; ++stage_order) {
			const uint64_t key = target_order * stage_width + stage_order;
			plan.stage_offsets[key] = plan.stage_products.size();
			plan.stage_products.insert(plan.stage_products.end(),
				active[stage_order].begin(), active[stage_order].end());
		}
	}
	plan.stage_offsets[stage_width * stage_width] = plan.stage_products.size();

	std::vector<bool> active_products(plan.product_count, false);
	for (const uint64_t product : plan.stage_products)
		active_products[product] = true;
	plan.derivative_offsets.resize(plan.product_count + 1, 0);
	for (uint64_t product = 0; product < plan.product_count; ++product) {
		plan.derivative_offsets[product] = plan.derivative_left.size();
		if (!active_products[product])
			continue;
		const uint64_t start = full_derivative_offsets[product];
		const uint64_t end = full_derivative_offsets[product + 1];
		plan.derivative_left.insert(plan.derivative_left.end(),
			full_derivative_left.begin() + start,
			full_derivative_left.begin() + end);
		plan.derivative_label.insert(plan.derivative_label.end(),
			full_derivative_label.begin() + start,
			full_derivative_label.begin() + end);
	}
	plan.derivative_offsets[plan.product_count] = plan.derivative_left.size();

	std::vector<bool> needed_products(plan.product_count, false);
	needed_products[0] = true;
	for (const uint64_t product : plan.stage_products) {
		uint64_t current = product;
		while (!needed_products[current]) {
			needed_products[current] = true;
			current = full.cpu_products.parent[current];
		}
	}
	for (uint64_t product = 1; product < plan.product_count; ++product) {
		if (needed_products[product]) {
			plan.product_build.push_back(product);
			plan.product_build_parent.push_back(
				full.cpu_products.parent[product]);
			plan.product_build_factor.push_back(
				full.cpu_products.last_factor[product]);
		}
	}
	cache.horner = std::move(plan);
}


BranchedSigCache::BranchedSigCache(
	uint64_t dimension_value,
	uint64_t max_nodes_value,
	bool planar_value
) {
	*this = planar_value
		? build_planar_cache(dimension_value, max_nodes_value)
		: build_nonplanar_cache(dimension_value, max_nodes_value);
	populate_branched_sig_horner_plan(*this);
}
