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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using TreeId = uint64_t;
using TreeLabel = uint8_t;

enum class TreeKind : uint8_t {
	NonPlanar,
	Planar
};

class TreeTable;

// A forest is an ordered tree collection. For non-planar trees, child order
// does not matter, so the table stores each forest in one sorted order.
class Forest {
public:
	Forest() = default;
	Forest(std::initializer_list<TreeId> trees) : trees_(trees) {}
	explicit Forest(std::vector<TreeId> trees) : trees_(std::move(trees)) {}

	template<typename Iterator>
	Forest(Iterator first, Iterator last) : trees_(first, last) {}

	std::span<const TreeId> trees() const noexcept { return trees_; }
	std::vector<TreeId>::const_iterator begin() const noexcept { return trees_.begin(); }
	std::vector<TreeId>::const_iterator end() const noexcept { return trees_.end(); }
	TreeId operator[](size_t index) const { return trees_.at(index); }

	bool empty() const noexcept { return trees_.empty(); }
	size_t size() const noexcept { return trees_.size(); }
	void reserve(size_t size) { trees_.reserve(size); }
	void push_back(TreeId tree) { trees_.push_back(tree); }
	void pop_back() { trees_.pop_back(); }

	void append(const Forest& other) {
		trees_.insert(trees_.end(), other.begin(), other.end());
	}

	void canonicalize(TreeKind kind) {
		if (kind == TreeKind::NonPlanar)
			std::sort(trees_.begin(), trees_.end());
	}

	uint64_t node_count(const TreeTable& table) const;
	std::vector<TreeLabel> node_labels(const TreeTable& table) const;

	bool operator==(const Forest& other) const noexcept = default;

	bool operator<(const Forest& other) const noexcept {
		return trees_ < other.trees_;
	}

	struct Hash {
		size_t operator()(const Forest& forest) const noexcept {
			size_t hash = forest.size();
			for (TreeId tree : forest)
				hash ^= std::hash<TreeId>()(tree)
					+ 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
			return hash;
		}
	};

private:
	std::vector<TreeId> trees_;
};

// Trees become immutable after TreeTable assigns their stable IDs.
class Tree {
public:
	TreeLabel root_label() const noexcept { return root_label_; }
	const Forest& children() const noexcept { return children_; }
	uint64_t node_count() const noexcept { return node_count_; }
	double tree_factorial() const noexcept { return tree_factorial_; }
	std::span<const TreeLabel> node_labels() const noexcept { return node_labels_; }

private:
	friend class TreeTable;

	Tree(
		TreeLabel root_label,
		Forest children,
		uint64_t node_count,
		double tree_factorial,
		std::vector<TreeLabel> node_labels
	) :
		root_label_(root_label),
		children_(std::move(children)),
		node_count_(node_count),
		tree_factorial_(tree_factorial),
		node_labels_(std::move(node_labels))
	{}

	TreeLabel root_label_ = 0;
	Forest children_;
	uint64_t node_count_ = 0;
	double tree_factorial_ = 0.0;
	std::vector<TreeLabel> node_labels_;
};

// Owns deduplicated trees and is the only type that creates TreeIds.
class TreeTable {
public:
	TreeTable(uint64_t dimension, TreeKind kind) : dimension_(dimension), kind_(kind) {
		if (dimension > 255)
			throw std::invalid_argument("branched signature dimension must be <= 255");
	}

	TreeId intern(uint64_t root_label, Forest children) {
		validate_root_label_(root_label);
		validate_children_(children);
		// Sorting makes child permutations identical only for non-planar trees.
		children.canonicalize(kind_);

		TreeKey key{ static_cast<TreeLabel>(root_label), children };
		const auto existing = indices_.find(key);
		if (existing != indices_.end())
			return existing->second;
		if (sealed_)
			throw std::runtime_error("cannot add a tree to a sealed TreeTable");

		// Store derived metadata once so all later cache builders can share it.
		uint64_t node_count = 1;
		double tree_factorial = 1.0;
		for (TreeId child : children) {
			const Tree& child_tree = trees_[child];
			if (node_count > UINT64_MAX - child_tree.node_count())
				throw std::overflow_error("tree node count overflow");
			node_count += child_tree.node_count();
			tree_factorial *= child_tree.tree_factorial();
		}
		tree_factorial *= static_cast<double>(node_count);

		std::vector<TreeLabel> node_labels;
		node_labels.reserve(static_cast<size_t>(node_count));
		node_labels.push_back(static_cast<TreeLabel>(root_label));
		for (TreeId child : children) {
			const auto labels = trees_[child].node_labels();
			node_labels.insert(node_labels.end(), labels.begin(), labels.end());
		}

		const TreeId id = static_cast<TreeId>(trees_.size());
		Tree tree(
			static_cast<TreeLabel>(root_label), std::move(children), node_count,
			tree_factorial, std::move(node_labels));
		trees_.push_back(std::move(tree));
		indices_.emplace(std::move(key), id);
		return id;
	}

	std::optional<TreeId> find(uint64_t root_label, Forest children) const {
		validate_root_label_(root_label);
		validate_children_(children);
		children.canonicalize(kind_);
		const TreeKey key{ static_cast<TreeLabel>(root_label), std::move(children) };
		const auto existing = indices_.find(key);
		if (existing == indices_.end())
			return std::nullopt;
		return existing->second;
	}

	const Tree& tree(TreeId id) const {
		if (id >= trees_.size())
			throw std::out_of_range("tree ID is out of range");
		return trees_[id];
	}

	uint64_t size() const noexcept { return static_cast<uint64_t>(trees_.size()); }
	uint64_t dimension() const noexcept { return dimension_; }
	TreeKind kind() const noexcept { return kind_; }
	void seal() noexcept { sealed_ = true; }
	bool sealed() const noexcept { return sealed_; }

private:
	struct TreeKey {
		TreeLabel root_label = 0;
		Forest children;

		bool operator==(const TreeKey& other) const noexcept {
			return root_label == other.root_label && children == other.children;
		}
	};

	struct TreeKeyHash {
		size_t operator()(const TreeKey& key) const noexcept {
			size_t hash = std::hash<TreeLabel>()(key.root_label);
			hash ^= Forest::Hash()(key.children)
				+ 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
			return hash;
		}
	};

	void validate_root_label_(uint64_t root_label) const {
		if (root_label >= dimension_)
			throw std::invalid_argument("tree root label is out of range");
	}

	void validate_children_(const Forest& children) const {
		for (TreeId child : children) {
			if (child >= trees_.size())
				throw std::out_of_range("tree child ID is out of range");
		}
	}

	uint64_t dimension_ = 0;
	TreeKind kind_ = TreeKind::NonPlanar;
	bool sealed_ = false;
	std::vector<Tree> trees_;
	std::unordered_map<TreeKey, TreeId, TreeKeyHash> indices_;
};

inline uint64_t Forest::node_count(const TreeTable& table) const {
	uint64_t count = 0;
	for (TreeId tree_id : trees_) {
		const uint64_t tree_nodes = table.tree(tree_id).node_count();
		if (count > UINT64_MAX - tree_nodes)
			throw std::overflow_error("forest node count overflow");
		count += tree_nodes;
	}
	return count;
}

inline std::vector<TreeLabel> Forest::node_labels(const TreeTable& table) const {
	const uint64_t count = node_count(table);
	std::vector<TreeLabel> labels;
	labels.reserve(static_cast<size_t>(count));
	for (TreeId tree_id : trees_) {
		const auto tree_labels = table.tree(tree_id).node_labels();
		labels.insert(labels.end(), tree_labels.begin(), tree_labels.end());
	}
	return labels;
}

inline void enumerate_trees(
	TreeTable& trees,
	uint64_t max_nodes,
	std::vector<uint64_t>& order_offsets
) {
	if (trees.sealed())
		throw std::invalid_argument("cannot enumerate into a sealed TreeTable");
	if (trees.size() != 0)
		throw std::invalid_argument("tree enumeration requires an empty TreeTable");
	if (max_nodes == UINT64_MAX)
		throw std::overflow_error("tree degree overflow");

	order_offsets.assign(max_nodes + 2, 0);
	// Keep the historical ordering: degree, child forest, then root label.
	for (uint64_t order = 1; order <= max_nodes; ++order) {
		order_offsets[order] = trees.size();
		if (order == 1) {
			for (uint64_t label = 0; label < trees.dimension(); ++label)
				trees.intern(label, {});
			continue;
		}

		// Enumerate child forests first to preserve the existing basis order.
		const TreeId tree_count = trees.size();
		std::vector<Forest> child_forests;
		Forest current;
		const auto enumerate_children = [&](auto&& self, uint64_t remaining, TreeId min_id) -> void {
			if (remaining == 0) {
				child_forests.push_back(current);
				return;
			}
			const TreeId start = trees.kind() == TreeKind::Planar ? 0 : min_id;
			for (TreeId tree_id = start; tree_id < tree_count; ++tree_id) {
				const uint64_t nodes = trees.tree(tree_id).node_count();
				if (nodes > remaining)
					break;
				current.push_back(tree_id);
				self(self, remaining - nodes, tree_id);
				current.pop_back();
			}
		};
		enumerate_children(enumerate_children, order - 1, 0);

		for (const Forest& children : child_forests) {
			if (children.empty())
				continue;
			for (uint64_t label = 0; label < trees.dimension(); ++label)
				trees.intern(label, children);
		}
	}
	order_offsets[max_nodes + 1] = trees.size();
	trees.seal();
}
