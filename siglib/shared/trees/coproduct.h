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

#include "tree.h"

#include <cstdint>
#include <vector>

// A cut stores the removed forest and the tree that remains at the root.
struct TreeCut {
	Forest pruned;
	TreeId trunk = 0;

	bool operator==(const TreeCut& other) const noexcept {
		return pruned == other.pruned && trunk == other.trunk;
	}
};

struct CoproductTerm {
	Forest left;
	Forest right;

	bool operator==(const CoproductTerm& other) const noexcept {
		return left == other.left && right == other.right;
	}
};

inline void append_forest_shuffles_(
	const Forest& a,
	const Forest& b,
	uint64_t a_index,
	uint64_t b_index,
	Forest& current,
	std::vector<Forest>& out
) {
	if (a_index == a.size() && b_index == b.size()) {
		out.push_back(current);
		return;
	}
	if (a_index < a.size()) {
		current.push_back(a[a_index]);
		append_forest_shuffles_(a, b, a_index + 1, b_index, current, out);
		current.pop_back();
	}
	if (b_index < b.size()) {
		current.push_back(b[b_index]);
		append_forest_shuffles_(a, b, a_index, b_index + 1, current, out);
		current.pop_back();
	}
}

inline void shuffle_forest_groups_(
	const std::vector<Forest>& groups,
	std::vector<Forest>& out
) {
	// MKW shuffles groups while preserving the order inside each group.
	out.assign(1, {});
	for (const Forest& group : groups) {
		if (group.empty())
			continue;
		std::vector<Forest> next;
		for (const Forest& base : out) {
			Forest current;
			current.reserve(base.size() + group.size());
			append_forest_shuffles_(base, group, 0, 0, current, next);
		}
		out.swap(next);
	}
}

inline void append_tree_cut_(
	TreeLabel root_label,
	TreeTable& trees,
	Forest pruned,
	Forest trunk_children,
	std::vector<TreeCut>& results
) {
	pruned.canonicalize(trees.kind());
	const TreeId trunk = trees.intern(root_label, std::move(trunk_children));
	results.push_back({ std::move(pruned), trunk });
}

inline void enumerate_planar_tree_cuts_(
	TreeId tree_id,
	TreeTable& trees,
	const std::vector<std::vector<TreeCut>>& memo,
	std::vector<TreeCut>& results
) {
	const TreeLabel root_label = trees.tree(tree_id).root_label();
	const Forest children = trees.tree(tree_id).children();
	if (children.empty())
		return;

	struct ChildOption {
		const Forest* pruned;
		TreeId trunk;
	};

	std::vector<std::vector<ChildOption>> child_options;
	child_options.reserve(children.size());
	for (TreeId child : children) {
		std::vector<ChildOption> options;
		options.reserve(1 + memo[child].size());
		options.push_back({ nullptr, child });
		for (const TreeCut& cut : memo[child])
			options.push_back({ &cut.pruned, cut.trunk });
		child_options.push_back(std::move(options));
	}

	// A planar cut can prune only a left prefix at each root.
	for (uint64_t prefix_size = 0; prefix_size <= children.size(); ++prefix_size) {
		const auto child_span = children.trees();
		Forest prefix(child_span.begin(), child_span.begin() + prefix_size);
		if (prefix_size == children.size()) {
			append_tree_cut_(root_label, trees, std::move(prefix), {}, results);
			continue;
		}

		std::vector<uint64_t> indices(children.size() - prefix_size, 0);
		while (true) {
			std::vector<Forest> forest_groups;
			Forest trunk_children;
			bool nontrivial = prefix_size != 0;
			forest_groups.reserve(children.size() - prefix_size + 1);
			trunk_children.reserve(children.size() - prefix_size);

			for (uint64_t child_index = prefix_size;
				child_index < children.size(); ++child_index) {
				const ChildOption& option =
					child_options[child_index][indices[child_index - prefix_size]];
				if (option.pruned != nullptr) {
					forest_groups.push_back(*option.pruned);
					nontrivial = true;
				}
				trunk_children.push_back(option.trunk);
			}
			forest_groups.push_back(prefix);

			if (nontrivial) {
				std::vector<Forest> shuffled;
				shuffle_forest_groups_(forest_groups, shuffled);
				for (Forest& pruned : shuffled) {
					append_tree_cut_(
						root_label, trees, std::move(pruned), trunk_children, results);
				}
			}

			int64_t position = static_cast<int64_t>(indices.size()) - 1;
			while (position >= 0) {
				const uint64_t child_index =
					prefix_size + static_cast<uint64_t>(position);
				indices[position]++;
				if (indices[position] < child_options[child_index].size())
					break;
				indices[position] = 0;
				--position;
			}
			if (position < 0)
				break;
		}
	}
}

inline void enumerate_nonplanar_tree_cuts_(
	TreeId tree_id,
	TreeTable& trees,
	const std::vector<std::vector<TreeCut>>& memo,
	std::vector<TreeCut>& results
) {
	const TreeLabel root_label = trees.tree(tree_id).root_label();
	const Forest children = trees.tree(tree_id).children();
	if (children.empty())
		return;

	struct ChildOption {
		const Forest* pruned;
		TreeId single_pruned = 0;
		TreeId trunk = 0;
		bool has_single_pruned = false;
		bool has_trunk = false;
	};

	std::vector<std::vector<ChildOption>> child_options;
	child_options.reserve(children.size());
	for (TreeId child : children) {
		std::vector<ChildOption> options;
		options.reserve(2 + memo[child].size());
		options.push_back({ nullptr, child, 0, true, false });
		options.push_back({ nullptr, 0, child, false, true });
		for (const TreeCut& cut : memo[child])
			options.push_back({ &cut.pruned, 0, cut.trunk, false, true });
		child_options.push_back(std::move(options));
	}

	// Each non-planar child is pruned, retained, or cut recursively.
	std::vector<uint64_t> indices(children.size(), 0);
	while (true) {
		Forest pruned;
		Forest trunk_children;
		pruned.reserve(children.size());
		trunk_children.reserve(children.size());
		bool all_kept = true;

		for (uint64_t child_index = 0;
			child_index < children.size(); ++child_index) {
			const ChildOption& option = child_options[child_index][indices[child_index]];
			if (option.has_single_pruned)
				pruned.push_back(option.single_pruned);
			else if (option.pruned != nullptr)
				pruned.append(*option.pruned);
			if (option.has_trunk)
				trunk_children.push_back(option.trunk);
			if (indices[child_index] != 1)
				all_kept = false;
		}

		if (!all_kept) {
			append_tree_cut_(
				root_label, trees, std::move(pruned), std::move(trunk_children), results);
		}

		int64_t position = static_cast<int64_t>(indices.size()) - 1;
		while (position >= 0) {
			indices[position]++;
			if (indices[position] < child_options[position].size())
				break;
			indices[position] = 0;
			--position;
		}
		if (position < 0)
			break;
	}
}

inline void enumerate_tree_cuts(
	TreeId tree_id,
	TreeTable& trees,
	const std::vector<std::vector<TreeCut>>& memo,
	std::vector<TreeCut>& results
) {
	if (trees.kind() == TreeKind::Planar)
		enumerate_planar_tree_cuts_(tree_id, trees, memo, results);
	else
		enumerate_nonplanar_tree_cuts_(tree_id, trees, memo, results);
}

inline void ensure_tree_cuts(
	TreeId tree_id,
	TreeTable& trees,
	std::vector<std::vector<TreeCut>>& memo,
	std::vector<uint8_t>& ready
) {
	if (tree_id >= trees.size())
		throw std::out_of_range("tree cut ID is out of range");
	if (memo.size() < trees.size())
		memo.resize(static_cast<size_t>(trees.size()));
	if (ready.size() < trees.size())
		ready.resize(static_cast<size_t>(trees.size()), 0);
	if (ready[tree_id])
		return;

	const Forest children = trees.tree(tree_id).children();
	for (TreeId child : children)
		ensure_tree_cuts(child, trees, memo, ready);

	// Sparse construction interns trunks on demand, so the table can grow here.
	std::vector<TreeCut> results;
	enumerate_tree_cuts(tree_id, trees, memo, results);
	if (memo.size() < trees.size())
		memo.resize(static_cast<size_t>(trees.size()));
	if (ready.size() < trees.size())
		ready.resize(static_cast<size_t>(trees.size()), 0);
	memo[tree_id] = std::move(results);
	ready[tree_id] = 1;
}

inline void enumerate_mkw_forest_coproduct_terms(
	const Forest& forest,
	const std::vector<std::vector<TreeCut>>& tree_cuts,
	std::vector<CoproductTerm>& results
) {
	if (forest.empty())
		return;

	// Combine one cut choice per tree to form the forest coproduct.

	struct ChildOption {
		const Forest* pruned;
		TreeId trunk;
	};

	std::vector<std::vector<ChildOption>> child_options;
	child_options.reserve(forest.size());
	for (TreeId tree : forest) {
		std::vector<ChildOption> options;
		options.reserve(1 + tree_cuts[tree].size());
		options.push_back({ nullptr, tree });
		for (const TreeCut& cut : tree_cuts[tree])
			options.push_back({ &cut.pruned, cut.trunk });
		child_options.push_back(std::move(options));
	}

	for (uint64_t prefix_size = 0; prefix_size <= forest.size(); ++prefix_size) {
		const auto forest_span = forest.trees();
		Forest prefix(forest_span.begin(), forest_span.begin() + prefix_size);
		if (prefix_size == forest.size()) {
			results.push_back({ std::move(prefix), {} });
			continue;
		}

		std::vector<uint64_t> indices(forest.size() - prefix_size, 0);
		while (true) {
			std::vector<Forest> forest_groups;
			Forest right;
			bool nontrivial = prefix_size != 0;
			forest_groups.reserve(forest.size() - prefix_size + 1);
			right.reserve(forest.size() - prefix_size);

			for (uint64_t tree_index = prefix_size;
				tree_index < forest.size(); ++tree_index) {
				const ChildOption& option =
					child_options[tree_index][indices[tree_index - prefix_size]];
				if (option.pruned != nullptr) {
					forest_groups.push_back(*option.pruned);
					nontrivial = true;
				}
				right.push_back(option.trunk);
			}
			forest_groups.push_back(prefix);

			if (nontrivial) {
				std::vector<Forest> left_forests;
				shuffle_forest_groups_(forest_groups, left_forests);
				for (Forest& left : left_forests)
					results.push_back({ std::move(left), right });
			}

			int64_t position = static_cast<int64_t>(indices.size()) - 1;
			while (position >= 0) {
				const uint64_t tree_index =
					prefix_size + static_cast<uint64_t>(position);
				indices[position]++;
				if (indices[position] < child_options[tree_index].size())
					break;
				indices[position] = 0;
				--position;
			}
			if (position < 0)
				break;
		}
	}
}
