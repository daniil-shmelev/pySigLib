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

#include "cupch.h"
#include "cusig.h"
#include "cu_branched_log_sig_cache.h"
#include "cu_log_sig_combine.h"
#include "cu_macros.h"
#include "cu_utils.h"
#include "../shared/branched_log_horner.h"

#include <cmath>
#include <limits>
#include <unordered_set>

namespace {

using CuMkwTensorElem = std::unordered_map<uint64_t, int>;
using CuMkwInfinitesimalProduct = std::unordered_map<
	std::pair<uint64_t, uint64_t>, CuMkwTensorElem, CuPairHash>;

struct CuMkwBchDeviceData {
	const double* bch_coefficients = nullptr;
	const uint64_t* bch_left_factor = nullptr;
	const uint64_t* bch_right_factor = nullptr;
	const uint32_t* comm_k_ptr = nullptr;
	const uint32_t* comm_k_i = nullptr;
	const uint32_t* comm_k_j = nullptr;
	const int* comm_k_val = nullptr;
	const uint32_t* comm_a_ptr = nullptr;
	const uint32_t* comm_a_k = nullptr;
	const uint32_t* comm_a_partner = nullptr;
	const int* comm_a_signed_c = nullptr;
	const uint64_t* linear_a_ptr = nullptr;
	const uint32_t* linear_a_idx = nullptr;
	const uint64_t* linear_output_ptr = nullptr;
	const uint32_t* linear_output_idx = nullptr;
	const uint64_t* linear_op_ptr = nullptr;
	const uint32_t* linear_op_idx = nullptr;
	const uint64_t* linear_reverse_ptr = nullptr;
	const uint32_t* linear_reverse_idx = nullptr;
	const uint32_t* segment_idx = nullptr;
	const double* segment_coefficients = nullptr;
	const uint32_t* segment_label_offsets = nullptr;
	const uint8_t* segment_labels = nullptr;
	uint64_t m = 0;
	uint64_t m2 = 0;
	uint64_t segment_count = 0;
};

struct CuMkwBchCache {
	CUDABchCache bch;
	uint64_t* d_linear_output_ptr = nullptr;
	uint32_t* d_linear_output_idx = nullptr;
	uint64_t* d_linear_op_ptr = nullptr;
	uint32_t* d_linear_op_idx = nullptr;
	uint64_t* d_linear_reverse_ptr = nullptr;
	uint32_t* d_linear_reverse_idx = nullptr;
	uint32_t* d_segment_idx = nullptr;
	double* d_segment_coefficients = nullptr;
	uint32_t* d_segment_label_offsets = nullptr;
	uint8_t* d_segment_labels = nullptr;
	uint64_t segment_count = 0;
	uint64_t exact_forward_work = 0;
	uint64_t exact_zero_work = 0;
	bool prune_reverse = false;

	CuMkwBchCache() = default;
	CuMkwBchCache(const CuMkwBchCache&) = delete;
	CuMkwBchCache& operator=(const CuMkwBchCache&) = delete;

	~CuMkwBchCache() {
		if (d_linear_output_ptr) cudaFree(d_linear_output_ptr);
		if (d_linear_output_idx) cudaFree(d_linear_output_idx);
		if (d_linear_op_ptr) cudaFree(d_linear_op_ptr);
		if (d_linear_op_idx) cudaFree(d_linear_op_idx);
		if (d_linear_reverse_ptr) cudaFree(d_linear_reverse_ptr);
		if (d_linear_reverse_idx) cudaFree(d_linear_reverse_idx);
		if (d_segment_idx) cudaFree(d_segment_idx);
		if (d_segment_coefficients) cudaFree(d_segment_coefficients);
		if (d_segment_label_offsets) cudaFree(d_segment_label_offsets);
		if (d_segment_labels) cudaFree(d_segment_labels);
	}

	CuMkwBchDeviceData device_data() const noexcept {
		return {
			bch.d_bch_coefficients,
			bch.d_bch_left_factor,
			bch.d_bch_right_factor,
			bch.d_comm_k_ptr,
			bch.d_comm_k_i,
			bch.d_comm_k_j,
			bch.d_comm_k_val,
			bch.d_comm_a_ptr,
			bch.d_comm_a_k,
			bch.d_comm_a_partner,
			bch.d_comm_a_signed_c,
			bch.d_linear_a_ptr,
			bch.d_linear_a_idx,
			d_linear_output_ptr,
			d_linear_output_idx,
			d_linear_op_ptr,
			d_linear_op_idx,
			d_linear_reverse_ptr,
			d_linear_reverse_idx,
			d_segment_idx,
			d_segment_coefficients,
			d_segment_label_offsets,
			d_segment_labels,
			bch.m,
			bch.m2,
			segment_count
		};
	}
};

std::unordered_map<
	CuLogSigCacheKey,
	std::unique_ptr<CuMkwBchCache>,
	CuLogSigCacheKeyHash
> s_mkw_bch_cache;
std::mutex s_mkw_bch_cache_mu;

template<typename T>
void upload_mkw_vector_(T*& device, const std::vector<T>& host) {
	if (host.empty())
		return;
	CudaBuf<T> buffer(host.size() * sizeof(T));
	CUDA_CHECK(cudaMemcpy(
		buffer.get(), host.data(), host.size() * sizeof(T),
		cudaMemcpyHostToDevice));
	device = buffer.release();
}

uint32_t narrow_mkw_u32_(uint64_t value, const char* message) {
	if (value > UINT32_MAX)
		throw std::overflow_error(message);
	return static_cast<uint32_t>(value);
}

void add_mkw_coefficient_(int& target, int64_t value) {
	const int64_t result = static_cast<int64_t>(target) + value;
	if (result < std::numeric_limits<int>::min()
		|| result > std::numeric_limits<int>::max())
		throw std::overflow_error("MKW BCH integer coefficient overflow");
	target = static_cast<int>(result);
}

void remove_mkw_zero_entries_(CuMkwTensorElem& element) {
	for (auto it = element.begin(); it != element.end();) {
		if (it->second == 0)
			it = element.erase(it);
		else
			++it;
	}
}

CuMkwTensorElem mkw_tensor_product_(
	const CuMkwTensorElem& left,
	const CuMkwTensorElem& right,
	const CuMkwHostBasisData& basis
) {
	CuMkwTensorElem result;
	for (const auto& [left_idx, left_coefficient] : left) {
		for (const auto& [right_idx, right_coefficient] : right) {
			cu_word word = basis.flat_words.at(left_idx);
			const cu_word& suffix = basis.flat_words.at(right_idx);
			word.insert(word.end(), suffix.begin(), suffix.end());
			const auto flat = basis.flat_idx.find(word);
			if (flat == basis.flat_idx.end())
				continue;
			add_mkw_coefficient_(
				result[flat->second],
				static_cast<int64_t>(left_coefficient) * right_coefficient);
		}
	}
	remove_mkw_zero_entries_(result);
	return result;
}

CuMkwInfinitesimalProduct build_mkw_infinitesimal_product_(
	const BranchedSigCache& cache
) {
	CuMkwInfinitesimalProduct product;
	for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
		uint64_t position = cache.coproduct_offsets[basis_idx];
		const uint64_t end = cache.coproduct_offsets[basis_idx + 1];
		while (position < end) {
			const uint64_t forest_size = cache.coproduct_data[position++];
			const uint64_t trunk = cache.coproduct_data[position++];
			if (forest_size == 1) {
				const uint64_t branch = cache.coproduct_data[position++];
				add_mkw_coefficient_(
					product[{ branch, trunk }][basis_idx + 1], 1);
			}
			else {
				position += forest_size;
			}
		}
	}
	return product;
}

CuMkwTensorElem mkw_infinitesimal_product_(
	const CuMkwTensorElem& left,
	const CuMkwTensorElem& right,
	const CuMkwInfinitesimalProduct& product
) {
	CuMkwTensorElem result;
	for (const auto& [left_idx, left_coefficient] : left) {
		for (const auto& [right_idx, right_coefficient] : right) {
			const auto found = product.find({ left_idx, right_idx });
			if (found == product.end())
				continue;
			for (const auto& [out_idx, coefficient] : found->second) {
				add_mkw_coefficient_(
					result[out_idx],
					static_cast<int64_t>(left_coefficient)
						* right_coefficient * coefficient);
			}
		}
	}
	remove_mkw_zero_entries_(result);
	return result;
}

void apply_mkw_inverse_projection_(
	const CuSparseIntMatrix& inverse,
	std::vector<double>& coordinates
) {
	for (uint64_t offset = 0; offset < inverse.n; ++offset) {
		const uint64_t row = inverse.n - offset - 1;
		for (const CuEntry& entry : inverse.rows[row])
			coordinates[row] += static_cast<double>(entry.val)
				* coordinates[entry.col];
	}
}

std::vector<double> build_unit_segment_coefficients_(
	const BranchedSigCache& cache,
	const CuMkwHostBasisData& basis
) {
	// Evaluate log(signature) for a unit increment, then project to method 2.
	// The resulting coefficients scale the corresponding increment monomials.
	const BranchedLogHornerPlan plan = build_branched_log_horner_plan(cache);
	BranchedLogHornerWorkspace<double> workspace(plan.product_count);
	std::vector<double> h(cache.total_length, 0.0);
	std::vector<double> expanded(cache.total_length, 0.0);

	for (uint64_t flat = 1; flat < cache.total_length; ++flat)
		h[flat] = cache.inv_tree_factorial[flat - 1];
	auto flat_value = [&h](uint64_t flat) { return h[flat]; };
	auto set_output = [&expanded](uint64_t flat, double value) {
		expanded[flat] = value;
	};
	branched_log_horner_forward<double>(
		cache.total_length, cache.max_nodes, cache.planar,
		plan, flat_value, set_output, workspace);

	std::vector<double> compact(basis.lyndon_idx.size(), 0.0);
	for (uint64_t index = 0; index < compact.size(); ++index)
		compact[index] = expanded[basis.lyndon_idx[index]];
	apply_mkw_inverse_projection_(basis.inv_proj_mat, compact);
	return compact;
}

struct HostMkwBchTables_ {
	std::vector<uint32_t> k_ptr;
	std::vector<uint32_t> k_i;
	std::vector<uint32_t> k_j;
	std::vector<int> k_val;
	std::vector<uint32_t> a_ptr;
	std::vector<uint32_t> a_k;
	std::vector<uint32_t> a_partner;
	std::vector<int> a_signed_c;
};

HostMkwBchTables_ build_mkw_commutator_tables_(
	const BranchedSigCache& cache,
	const CuMkwHostBasisData& basis
) {
	// Build sparse forward and reverse tables for the MKW Lie commutator.
	const uint64_t m = basis.lyndon_words.size();
	std::vector<CuMkwTensorElem> tensor_representations(m);
	for (uint64_t index = 0; index < m; ++index) {
		if (basis.lyndon_words[index].size() == 1) {
			tensor_representations[index] = {
				{ basis.flat_idx.at(basis.lyndon_words[index]), 1 }
			};
			continue;
		}
		CuMkwTensorElem left_right = mkw_tensor_product_(
			tensor_representations[basis.left_factor[index]],
			tensor_representations[basis.right_factor[index]], basis);
		const CuMkwTensorElem right_left = mkw_tensor_product_(
			tensor_representations[basis.right_factor[index]],
			tensor_representations[basis.left_factor[index]], basis);
		for (const auto& [flat, coefficient] : right_left)
			add_mkw_coefficient_(left_right[flat], -static_cast<int64_t>(coefficient));
		remove_mkw_zero_entries_(left_right);
		tensor_representations[index] = std::move(left_right);
	}

	struct KEntry_ {
		uint32_t i;
		uint32_t j;
		int coefficient;
	};
	std::vector<std::vector<KEntry_>> by_output(m);
	const CuMkwInfinitesimalProduct product
		= build_mkw_infinitesimal_product_(cache);
	std::vector<double> coordinates(m, 0.0);
	for (uint64_t i = 0; i < m; ++i) {
		for (uint64_t j = i + 1; j < m; ++j) {
			const uint64_t weight = basis.lyndon_weights[i]
				+ basis.lyndon_weights[j];
			if (weight > cache.max_nodes)
				continue;
			CuMkwTensorElem commutator = mkw_infinitesimal_product_(
				tensor_representations[i], tensor_representations[j], product);
			const CuMkwTensorElem reverse = mkw_infinitesimal_product_(
				tensor_representations[j], tensor_representations[i], product);
			for (const auto& [flat, coefficient] : reverse)
				add_mkw_coefficient_(commutator[flat], -static_cast<int64_t>(coefficient));
			remove_mkw_zero_entries_(commutator);
			std::fill(coordinates.begin(), coordinates.end(), 0.0);
			for (uint64_t k = 0; k < m; ++k) {
				if (basis.lyndon_weights[k] != weight)
					continue;
				const auto found = commutator.find(basis.lyndon_idx[k]);
				if (found != commutator.end())
					coordinates[k] = static_cast<double>(found->second);
			}
			apply_mkw_inverse_projection_(basis.inv_proj_mat, coordinates);
			for (uint64_t k = 0; k < m; ++k) {
				const double rounded = std::round(coordinates[k]);
				if (rounded == 0.0)
					continue;
				if (rounded < std::numeric_limits<int>::min()
					|| rounded > std::numeric_limits<int>::max())
					throw std::overflow_error("MKW BCH commutator coefficient overflow");
				by_output[k].push_back({
					static_cast<uint32_t>(i),
					static_cast<uint32_t>(j),
					static_cast<int>(rounded)
				});
			}
		}
	}

	HostMkwBchTables_ tables;
	tables.k_ptr.resize(m + 1);
	uint64_t offset = 0;
	for (uint64_t k = 0; k < m; ++k) {
		tables.k_ptr[k] = narrow_mkw_u32_(
			offset, "MKW BCH commutator table exceeds uint32 range");
		offset += by_output[k].size();
	}
	tables.k_ptr[m] = narrow_mkw_u32_(
		offset, "MKW BCH commutator table exceeds uint32 range");
	tables.k_i.reserve(offset);
	tables.k_j.reserve(offset);
	tables.k_val.reserve(offset);
	for (const auto& entries : by_output) {
		for (const KEntry_& entry : entries) {
			tables.k_i.push_back(entry.i);
			tables.k_j.push_back(entry.j);
			tables.k_val.push_back(entry.coefficient);
		}
	}

	struct AEntry_ {
		uint32_t k;
		uint32_t partner;
		int signed_coefficient;
	};
	std::vector<std::vector<AEntry_>> by_input(m);
	for (uint64_t k = 0; k < m; ++k) {
		for (uint32_t index = tables.k_ptr[k]; index < tables.k_ptr[k + 1]; ++index) {
			const uint32_t i = tables.k_i[index];
			const uint32_t j = tables.k_j[index];
			const int coefficient = tables.k_val[index];
			by_input[i].push_back({ static_cast<uint32_t>(k), j, coefficient });
			by_input[j].push_back({ static_cast<uint32_t>(k), i, -coefficient });
		}
	}
	tables.a_ptr.resize(m + 1);
	offset = 0;
	for (uint64_t a = 0; a < m; ++a) {
		tables.a_ptr[a] = narrow_mkw_u32_(
			offset, "MKW BCH reverse table exceeds uint32 range");
		offset += by_input[a].size();
	}
	tables.a_ptr[m] = narrow_mkw_u32_(
		offset, "MKW BCH reverse table exceeds uint32 range");
	if (offset > (UINT32_MAX >> 1))
		throw std::overflow_error("MKW BCH reverse plan exceeds uint32 packing");
	tables.a_k.reserve(offset);
	tables.a_partner.reserve(offset);
	tables.a_signed_c.reserve(offset);
	for (const auto& entries : by_input) {
		for (const AEntry_& entry : entries) {
			tables.a_k.push_back(entry.k);
			tables.a_partner.push_back(entry.partner);
			tables.a_signed_c.push_back(entry.signed_coefficient);
		}
	}
	return tables;
}

struct HostMkwPruning_ {
	std::vector<uint64_t> range;
	std::vector<uint64_t> output_ptr;
	std::vector<uint32_t> output_idx;
	std::vector<uint64_t> op_ptr{ 0 };
	std::vector<uint32_t> op_idx;
	std::vector<uint64_t> linear_a_ptr;
	std::vector<uint32_t> linear_a_idx;
	std::vector<uint64_t> reverse_ptr;
	std::vector<uint32_t> reverse_idx;
	uint64_t dense_forward_work = 0;
	uint64_t exact_forward_work = 0;
	uint64_t exact_zero_work = 0;
	bool prune_reverse = false;
};

HostMkwPruning_ build_mkw_pruning_(
	const BchHardcodedData& formula,
	const HostMkwBchTables_& tables,
	const std::vector<uint64_t>& weights,
	const std::vector<uint32_t>& segment_idx
) {
	// Propagate exact coordinate support through the BCH expression tree.
	// This selects sparse plans only when they save enough work to justify them.
	const uint64_t m = weights.size();
	const uint64_t m2 = formula.size;
	std::vector<uint8_t> input_mask(m, 0);
	for (uint32_t index : segment_idx)
		input_mask[index] = 1;

	std::vector<uint64_t> min_weight(m2, 1);
	std::vector<uint64_t> max_weight(m2, weights.empty() ? 0 : weights.back());
	if (m2 > 1) {
		min_weight[1] = weights.empty() ? 1 : weights.back() + 1;
		max_weight[1] = 0;
		for (uint32_t index : segment_idx) {
			min_weight[1] = std::min(min_weight[1], weights[index]);
			max_weight[1] = std::max(max_weight[1], weights[index]);
		}
		if (segment_idx.empty())
			min_weight[1] = 1;
	}
	for (uint64_t w = 2; w < m2; ++w) {
		const uint64_t left = formula.left_factor[w];
		const uint64_t right = formula.right_factor[w];
		min_weight[w] = min_weight[left] + min_weight[right];
		max_weight[w] = std::min(
			weights.empty() ? uint64_t(0) : weights.back(),
			max_weight[left] + max_weight[right]);
	}

	HostMkwPruning_ pruning;
	pruning.range.resize(2 * m2);
	for (uint64_t w = 0; w < m2; ++w) {
		pruning.range[2 * w] = std::lower_bound(
			weights.begin(), weights.end(), min_weight[w]) - weights.begin();
		pruning.range[2 * w + 1] = std::upper_bound(
			weights.begin(), weights.end(), max_weight[w]) - weights.begin();
	}

	std::vector<std::vector<uint8_t>> support(m2, std::vector<uint8_t>(m, 0));
	if (m2 > 0)
		std::fill(support[0].begin(), support[0].end(), uint8_t(1));
	if (m2 > 1)
		support[1] = input_mask;
	pruning.output_ptr.assign(m2 + 1, 0);
	for (uint64_t w = 2; w < m2; ++w) {
		const auto& left_support = support[formula.left_factor[w]];
		const auto& right_support = support[formula.right_factor[w]];
		pruning.output_ptr[w] = pruning.output_idx.size();
		for (uint64_t k = 0; k < m; ++k) {
			const uint64_t before = pruning.op_idx.size();
			for (uint32_t index = tables.k_ptr[k]; index < tables.k_ptr[k + 1]; ++index) {
				const uint32_t i = tables.k_i[index];
				const uint32_t j = tables.k_j[index];
				if ((left_support[i] && right_support[j])
					|| (left_support[j] && right_support[i]))
					pruning.op_idx.push_back(index);
			}
			if (pruning.op_idx.size() != before) {
				support[w][k] = 1;
				pruning.output_idx.push_back(static_cast<uint32_t>(k));
				pruning.op_ptr.push_back(pruning.op_idx.size());
			}
		}
		pruning.output_ptr[w + 1] = pruning.output_idx.size();
	}

	pruning.linear_a_ptr.resize(m2 * m + 1, 0);
	pruning.reverse_ptr.assign(m2 + 1, 0);
	for (uint64_t w = 2; w < m2; ++w) {
		const auto& left_support = support[formula.left_factor[w]];
		const auto& right_support = support[formula.right_factor[w]];
		pruning.reverse_ptr[w] = pruning.reverse_idx.size();
		for (uint64_t a = 0; a < m; ++a) {
			const uint64_t row = w * m + a;
			pruning.linear_a_ptr[row] = pruning.linear_a_idx.size();
			for (uint32_t index = tables.a_ptr[a]; index < tables.a_ptr[a + 1]; ++index) {
				const uint32_t partner = tables.a_partner[index];
				if (left_support[a] && right_support[partner])
					pruning.linear_a_idx.push_back(index << 1);
				if (right_support[a] && left_support[partner])
					pruning.linear_a_idx.push_back((index << 1) | 1);
			}
			pruning.linear_a_ptr[row + 1] = pruning.linear_a_idx.size();
			if (pruning.linear_a_ptr[row] != pruning.linear_a_ptr[row + 1])
				pruning.reverse_idx.push_back(static_cast<uint32_t>(a));
		}
		pruning.reverse_ptr[w + 1] = pruning.reverse_idx.size();
	}

	const uint64_t node_count = m2 > 2 ? m2 - 2 : 0;
	const uint64_t nnz = tables.k_i.size();
	pruning.dense_forward_work = node_count * (m + nnz);
	pruning.exact_forward_work = pruning.output_idx.size() + pruning.op_idx.size();
	pruning.exact_zero_work = node_count * m;
	const uint64_t dense_reverse_work = node_count
		* (m + tables.a_k.size());
	const uint64_t exact_reverse_work = pruning.reverse_idx.size()
		+ pruning.linear_a_idx.size();
	pruning.prune_reverse = dense_reverse_work > 0
		&& exact_reverse_work <= dense_reverse_work - dense_reverse_work / 3;
	return pruning;
}

std::unique_ptr<CuMkwBchCache> build_cuda_mkw_bch_cache_(
	const BranchedSigCache& cache
) {
	if (!cache.planar)
		throw std::invalid_argument("MKW BCH requires planar=True");
	if (cache.max_nodes > BCH_MAX_HARDCODED_DEGREE)
		throw std::runtime_error(
			"CUDA MKW BCH method supports degree at most 12");

	// Method 3 reuses the prepared method 2 basis and adds BCH-specific data.
	const CuMkwHostBasisData& basis = get_cuda_mkw_host_basis_data_(
		cache.dimension, cache.max_nodes, 2);
	const uint64_t m = basis.lyndon_words.size();
	if (m > UINT32_MAX)
		throw std::overflow_error("CUDA MKW BCH basis exceeds uint32 range");

	auto result = std::make_unique<CuMkwBchCache>();
	result->bch.m = m;
	if (cache.max_nodes == 0)
		return result;

	const BchHardcodedData* formula = get_hardcoded_bch_data(cache.max_nodes);
	if (formula == nullptr)
		throw std::runtime_error(
			"CUDA MKW BCH method supports degree at most 12");
	result->bch.m2 = formula->size;

	std::vector<double> segment_coefficients_full
		= build_unit_segment_coefficients_(cache, basis);
	std::vector<uint32_t> segment_idx;
	std::vector<double> segment_coefficients;
	std::vector<uint32_t> segment_label_offsets{ 0 };
	std::vector<uint8_t> segment_labels;
	for (uint64_t coordinate = 0; coordinate < m; ++coordinate) {
		if (segment_coefficients_full[coordinate] == 0.0)
			continue;
		segment_idx.push_back(static_cast<uint32_t>(coordinate));
		segment_coefficients.push_back(segment_coefficients_full[coordinate]);
		const uint64_t basis_idx = basis.lyndon_idx[coordinate] - 1;
		const uint64_t start = cache.node_labels_offsets[basis_idx];
		const uint64_t end = cache.node_labels_offsets[basis_idx + 1];
		segment_labels.insert(
			segment_labels.end(),
			cache.node_labels_data.begin() + static_cast<std::ptrdiff_t>(start),
			cache.node_labels_data.begin() + static_cast<std::ptrdiff_t>(end));
		segment_label_offsets.push_back(narrow_mkw_u32_(
			segment_labels.size(), "MKW segment label plan exceeds uint32 range"));
	}
	result->segment_count = segment_idx.size();

	const HostMkwBchTables_ tables = build_mkw_commutator_tables_(cache, basis);
	const HostMkwPruning_ pruning = build_mkw_pruning_(
		*formula, tables, basis.lyndon_weights, segment_idx);
	result->bch.linear_dense_forward_work = pruning.dense_forward_work;
	result->bch.linear_active_forward_work = pruning.exact_forward_work;
	result->bch.linear_zero_work = pruning.exact_zero_work;
	result->exact_forward_work = pruning.exact_forward_work;
	result->exact_zero_work = pruning.exact_zero_work;
	result->prune_reverse = pruning.prune_reverse;

	std::vector<double> bch_coefficients(
		formula->coefficients, formula->coefficients + formula->size);
	std::vector<uint64_t> bch_left(
		formula->left_factor, formula->left_factor + formula->size);
	std::vector<uint64_t> bch_right(
		formula->right_factor, formula->right_factor + formula->size);
	upload_mkw_vector_(result->bch.d_bch_coefficients, bch_coefficients);
	upload_mkw_vector_(result->bch.d_bch_left_factor, bch_left);
	upload_mkw_vector_(result->bch.d_bch_right_factor, bch_right);
	upload_mkw_vector_(result->bch.d_linear_range, pruning.range);
	upload_mkw_vector_(result->bch.d_comm_k_ptr, tables.k_ptr);
	upload_mkw_vector_(result->bch.d_comm_k_i, tables.k_i);
	upload_mkw_vector_(result->bch.d_comm_k_j, tables.k_j);
	upload_mkw_vector_(result->bch.d_comm_k_val, tables.k_val);
	upload_mkw_vector_(result->bch.d_comm_a_ptr, tables.a_ptr);
	upload_mkw_vector_(result->bch.d_comm_a_k, tables.a_k);
	upload_mkw_vector_(result->bch.d_comm_a_partner, tables.a_partner);
	upload_mkw_vector_(result->bch.d_comm_a_signed_c, tables.a_signed_c);
	upload_mkw_vector_(result->bch.d_linear_a_ptr, pruning.linear_a_ptr);
	upload_mkw_vector_(result->bch.d_linear_a_idx, pruning.linear_a_idx);
	upload_mkw_vector_(result->d_linear_output_ptr, pruning.output_ptr);
	upload_mkw_vector_(result->d_linear_output_idx, pruning.output_idx);
	upload_mkw_vector_(result->d_linear_op_ptr, pruning.op_ptr);
	upload_mkw_vector_(result->d_linear_op_idx, pruning.op_idx);
	upload_mkw_vector_(result->d_linear_reverse_ptr, pruning.reverse_ptr);
	upload_mkw_vector_(result->d_linear_reverse_idx, pruning.reverse_idx);
	upload_mkw_vector_(result->d_segment_idx, segment_idx);
	upload_mkw_vector_(result->d_segment_coefficients, segment_coefficients);
	upload_mkw_vector_(result->d_segment_label_offsets, segment_label_offsets);
	upload_mkw_vector_(result->d_segment_labels, segment_labels);
	return result;
}

const CuMkwBchCache& get_cuda_mkw_bch_cache_(
	uint64_t dimension,
	uint64_t max_nodes
) {
	const CuLogSigCacheKey key = make_cuda_log_sig_cache_key_(
		dimension, max_nodes);
	std::lock_guard<std::mutex> lock(s_mkw_bch_cache_mu);
	const auto found = s_mkw_bch_cache.find(key);
	if (found == s_mkw_bch_cache.end())
		throw cache_not_found_error(
			"CUDA MKW BCH cache not found - call prepare_branched_log_sig with method=3 first");
	return *found->second;
}

template<typename T>
__device__ __forceinline__ void evaluate_mkw_segment_(
	const T* left,
	const T* right,
	T* target,
	T sign,
	const CuMkwBchDeviceData& data
) {
	for (uint64_t q = threadIdx.x; q < data.segment_count; q += blockDim.x) {
		T value = sign * static_cast<T>(data.segment_coefficients[q]);
		for (uint32_t position = data.segment_label_offsets[q];
			position < data.segment_label_offsets[q + 1]; ++position) {
			const uint8_t label = data.segment_labels[position];
			value *= right[label] - left[label];
		}
		target[data.segment_idx[q]] = value;
	}
}

template<typename T, bool Sparse, bool Shared, bool Accumulate>
__device__ __forceinline__ void evaluate_mkw_bch_nodes_(
	T* memo,
	T* out,
	T* shared_left,
	T* shared_right,
	const CuMkwBchDeviceData& data
) {
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	for (uint64_t w = 2; w < data.m2; ++w) {
		const T* left_global = memo + data.bch_left_factor[w] * data.m;
		const T* right_global = memo + data.bch_right_factor[w] * data.m;
		const T* left = left_global;
		const T* right = right_global;
		if constexpr (Shared) {
			for (uint64_t k = tid; k < data.m; k += stride) {
				shared_left[k] = left_global[k];
				shared_right[k] = right_global[k];
			}
			__syncthreads();
			left = shared_left;
			right = shared_right;
		}

		T* result = memo + w * data.m;
		if constexpr (Sparse) {
			for (uint64_t q = data.linear_output_ptr[w] + tid;
				q < data.linear_output_ptr[w + 1]; q += stride) {
				const uint32_t k = data.linear_output_idx[q];
				T value = T(0);
				for (uint64_t position = data.linear_op_ptr[q];
					position < data.linear_op_ptr[q + 1]; ++position) {
					const uint32_t comm_index = data.linear_op_idx[position];
					const uint32_t i = data.comm_k_i[comm_index];
					const uint32_t j = data.comm_k_j[comm_index];
					value += static_cast<T>(data.comm_k_val[comm_index])
						* (left[i] * right[j] - left[j] * right[i]);
				}
				result[k] = value;
				if constexpr (Accumulate) {
					const T bch_coefficient = static_cast<T>(
						data.bch_coefficients[w]);
					if (bch_coefficient != T(0))
						out[k] += bch_coefficient * value;
				}
			}
		}
		else {
			for (uint64_t k = tid; k < data.m; k += stride) {
				T value = T(0);
				for (uint32_t position = data.comm_k_ptr[k];
					position < data.comm_k_ptr[k + 1]; ++position) {
					const uint32_t i = data.comm_k_i[position];
					const uint32_t j = data.comm_k_j[position];
					value += static_cast<T>(data.comm_k_val[position])
						* (left[i] * right[j] - left[j] * right[i]);
				}
				result[k] = value;
				if constexpr (Accumulate) {
					const T bch_coefficient = static_cast<T>(
						data.bch_coefficients[w]);
					if (bch_coefficient != T(0))
						out[k] += bch_coefficient * value;
				}
			}
		}
		__syncthreads();
	}
}

template<typename T, bool Sparse, bool Shared>
__global__ void branched_log_sig_from_path_kernel_(
	const T* __restrict__ path,
	T* __restrict__ out,
	T* __restrict__ workspace,
	uint64_t length,
	uint64_t dimension,
	CuMkwBchDeviceData data
) {
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	const T* path_row = path + batch_idx * length * dimension;
	T* out_row = out + batch_idx * data.m;
	T* memo = workspace + batch_idx * data.m2 * data.m;

	T* shared_left = nullptr;
	T* shared_right = nullptr;
	if constexpr (Shared) {
		extern __shared__ char shared_bytes[];
		shared_left = reinterpret_cast<T*>(shared_bytes);
		shared_right = shared_left + data.m;
	}

	for (uint64_t k = tid; k < data.m; k += stride) {
		out_row[k] = T(0);
		memo[data.m + k] = T(0);
	}
	if constexpr (Sparse) {
		for (uint64_t index = 2 * data.m + tid;
			index < data.m2 * data.m; index += stride)
			memo[index] = T(0);
	}
	__syncthreads();
	if (length == 1)
		return;

	evaluate_mkw_segment_(
		path_row, path_row + dimension, out_row, T(1), data);
	__syncthreads();
	for (uint64_t segment = 1; segment + 1 < length; ++segment) {
		const T* left = path_row + segment * dimension;
		const T* right = left + dimension;
		for (uint64_t k = tid; k < data.m; k += stride)
			memo[k] = out_row[k];
		evaluate_mkw_segment_(left, right, memo + data.m, T(1), data);
		__syncthreads();
		for (uint64_t q = tid; q < data.segment_count; q += stride) {
			const uint32_t coordinate = data.segment_idx[q];
			out_row[coordinate] += memo[data.m + coordinate];
		}
		__syncthreads();
		evaluate_mkw_bch_nodes_<T, Sparse, Shared, true>(
			memo, out_row, shared_left, shared_right, data);
	}
}

uint64_t checked_mkw_product_(
	uint64_t left,
	uint64_t right,
	const char* message
) {
	if (right != 0 && left > UINT64_MAX / right)
		throw std::overflow_error(message);
	return left * right;
}

uint64_t checked_mkw_sum_(
	uint64_t left,
	uint64_t right,
	const char* message
) {
	if (left > UINT64_MAX - right)
		throw std::overflow_error(message);
	return left + right;
}

template<typename T, bool Shared>
void launch_mkw_forward_(
	bool sparse,
	const T* path,
	T* out,
	T* workspace,
	uint64_t batch,
	uint64_t length,
	uint64_t dimension,
	unsigned int threads,
	size_t shared_size,
	const CuMkwBchDeviceData& data
) {
	if (sparse) {
		branched_log_sig_from_path_kernel_<T, true, Shared><<<
			static_cast<unsigned int>(batch), threads, shared_size>>>(
				path, out, workspace, length, dimension, data);
	}
	else {
		branched_log_sig_from_path_kernel_<T, false, Shared><<<
			static_cast<unsigned int>(batch), threads, shared_size>>>(
				path, out, workspace, length, dimension, data);
	}
}

template<typename T>
void branched_log_sig_from_path_cuda_(
	const T* path,
	T* out,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t max_nodes
) {
	if (length == 0)
		throw std::invalid_argument(
			"branched_log_sig method 3 received an empty path");
	const CuMkwBchCache& cache = get_cuda_mkw_bch_cache_(
		dimension, max_nodes);
	const CuMkwBchDeviceData data = cache.device_data();
	if (batch_size == 0 || data.m == 0)
		return;
	if (length == 1) {
		const uint64_t output_elements = checked_mkw_product_(
			batch_size, data.m, "MKW BCH output size overflow");
		CUDA_CHECK(cudaMemset(
			out, 0, checked_mkw_product_(output_elements, sizeof(T),
				"MKW BCH output byte size overflow")));
		check_cuda_kernel_launch();
		return;
	}

	const uint64_t workspace_per_batch = checked_mkw_product_(
		data.m2, data.m, "MKW BCH workspace size overflow");
	size_t free_memory = 0;
	size_t total_memory = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
	const uint64_t workspace_bytes = checked_mkw_product_(
		workspace_per_batch, sizeof(T), "MKW BCH workspace byte size overflow");
	const uint64_t reservation_bytes = checked_mkw_product_(
		2, workspace_bytes, "MKW BCH memory reservation overflow");
	uint64_t chunk_size = workspace_bytes == 0
		? batch_size
		: free_memory / reservation_bytes;
	chunk_size = std::max<uint64_t>(1, chunk_size);
	chunk_size = std::min<uint64_t>(
		std::min<uint64_t>(batch_size, chunk_size), CUDA_GRID_X_LIMIT);
	CudaBuf<T> workspace(checked_mkw_product_(
		chunk_size, workspace_bytes, "MKW BCH workspace allocation overflow"));

	unsigned int threads = static_cast<unsigned int>(
		std::min<uint64_t>(64, data.m));
	threads = std::max(32U, ((threads + 31) / 32) * 32);
	const size_t shared_size = checked_mkw_product_(
		checked_mkw_product_(2, data.m, "MKW BCH shared size overflow"),
		sizeof(T), "MKW BCH shared byte size overflow");
	const bool shared = shared_size <= CUDA_BASE_DYNAMIC_SMEM;
	const uint64_t passes = length > 2 ? length - 2 : 0;
	const long double dense_work = static_cast<long double>(passes)
		* cache.bch.linear_dense_forward_work;
	const long double exact_work = static_cast<long double>(passes)
		* cache.exact_forward_work + cache.exact_zero_work;
	const bool sparse = data.m2 > 2 && exact_work < dense_work;
	const uint64_t path_stride = checked_mkw_product_(
		length, dimension, "MKW BCH path stride overflow");
	checked_mkw_product_(
		batch_size, path_stride, "MKW BCH path size overflow");
	checked_mkw_product_(
		batch_size, data.m, "MKW BCH output size overflow");

	for (uint64_t offset = 0; offset < batch_size; offset += chunk_size) {
		const uint64_t current_batch = std::min(
			chunk_size, batch_size - offset);
		if (shared) {
			launch_mkw_forward_<T, true>(
				sparse, path + offset * path_stride,
				out + offset * data.m, workspace.get(), current_batch,
				length, dimension, threads, shared_size, data);
		}
		else {
			launch_mkw_forward_<T, false>(
				sparse, path + offset * path_stride,
				out + offset * data.m, workspace.get(), current_batch,
				length, dimension, threads, 0, data);
		}
		check_cuda_kernel_launch();
	}
}

template<typename T>
__device__ __forceinline__ void add_mkw_segment_vjp_(
	const T* left,
	const T* right,
	const T* accumulated_derivs,
	const T* leaf_derivs,
	T* left_derivs,
	T* right_derivs,
	uint64_t dimension,
	const CuMkwBchDeviceData& data
) {
	for (uint64_t label = threadIdx.x; label < dimension; label += blockDim.x) {
		T derivative = T(0);
		for (uint64_t q = 0; q < data.segment_count; ++q) {
			const uint32_t coordinate = data.segment_idx[q];
			T base = accumulated_derivs[coordinate];
			if (leaf_derivs != nullptr)
				base += leaf_derivs[coordinate];
			base *= static_cast<T>(data.segment_coefficients[q]);
			const uint32_t start = data.segment_label_offsets[q];
			const uint32_t end = data.segment_label_offsets[q + 1];
			for (uint32_t position = start; position < end; ++position) {
				if (data.segment_labels[position] != label)
					continue;
				T product = base;
				for (uint32_t other = start; other < end; ++other) {
					if (other == position)
						continue;
					const uint8_t other_label = data.segment_labels[other];
					product *= right[other_label] - left[other_label];
				}
				derivative += product;
			}
		}
		right_derivs[label] += derivative;
		left_derivs[label] -= derivative;
	}
}

template<typename T, bool Sparse, bool Shared>
__device__ __forceinline__ void reverse_mkw_bch_nodes_(
	const T* memo,
	T* deriv_memo,
	T* shared_left,
	T* shared_right,
	const CuMkwBchDeviceData& data
) {
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	for (uint64_t offset = 0; offset + 2 < data.m2; ++offset) {
		const uint64_t w = data.m2 - offset - 1;
		const uint64_t left_factor = data.bch_left_factor[w];
		const uint64_t right_factor = data.bch_right_factor[w];
		const T* left_global = memo + left_factor * data.m;
		const T* right_global = memo + right_factor * data.m;
		const T* left = left_global;
		const T* right = right_global;
		if constexpr (Shared) {
			for (uint64_t k = tid; k < data.m; k += stride) {
				shared_left[k] = left_global[k];
				shared_right[k] = right_global[k];
			}
			__syncthreads();
			left = shared_left;
			right = shared_right;
		}

		const T* node_derivs = deriv_memo + w * data.m;
		T* left_derivs = deriv_memo + left_factor * data.m;
		T* right_derivs = deriv_memo + right_factor * data.m;
		if constexpr (Sparse) {
			for (uint64_t q = data.linear_reverse_ptr[w] + tid;
				q < data.linear_reverse_ptr[w + 1]; q += stride) {
				const uint32_t a = data.linear_reverse_idx[q];
				T left_value = T(0);
				T right_value = T(0);
				const uint64_t row = w * data.m + a;
				for (uint64_t position = data.linear_a_ptr[row];
					position < data.linear_a_ptr[row + 1]; ++position) {
					const uint32_t packed = data.linear_a_idx[position];
					const uint32_t index = packed >> 1;
					const uint32_t k = data.comm_a_k[index];
					const uint32_t partner = data.comm_a_partner[index];
					const T coefficient = static_cast<T>(
						data.comm_a_signed_c[index]);
					if (packed & 1)
						right_value -= coefficient * left[partner] * node_derivs[k];
					else
						left_value += coefficient * right[partner] * node_derivs[k];
				}
				left_derivs[a] += left_value;
				right_derivs[a] += right_value;
			}
		}
		else {
			for (uint64_t a = tid; a < data.m; a += stride) {
				T left_value = T(0);
				T right_value = T(0);
				for (uint32_t index = data.comm_a_ptr[a];
					index < data.comm_a_ptr[a + 1]; ++index) {
					const uint32_t k = data.comm_a_k[index];
					const uint32_t partner = data.comm_a_partner[index];
					const T coefficient = static_cast<T>(
						data.comm_a_signed_c[index]);
					left_value += coefficient * right[partner] * node_derivs[k];
					right_value -= coefficient * left[partner] * node_derivs[k];
				}
				left_derivs[a] += left_value;
				right_derivs[a] += right_value;
			}
		}
		__syncthreads();
	}
}

template<typename T, bool SparseForward, bool SaveStates,
	bool SparseReverse, bool Shared>
__global__ void branched_log_sig_from_path_backprop_kernel_(
	const T* __restrict__ out_derivs,
	T* __restrict__ path_derivs,
	const T* __restrict__ path,
	T* __restrict__ workspace,
	uint64_t length,
	uint64_t dimension,
	CuMkwBchDeviceData data
) {
	const uint64_t batch_idx = blockIdx.x;
	const uint64_t tid = threadIdx.x;
	const uint64_t stride = blockDim.x;
	const uint64_t segment_count = length - 1;
	const uint64_t state_count = SaveStates ? segment_count : 2;
	const uint64_t workspace_per_batch = (state_count + 1) * data.m
		+ 2 * data.m2 * data.m;
	T* states = workspace + batch_idx * workspace_per_batch;
	T* current = states;
	T* previous = current + data.m;
	T* memo = states + state_count * data.m;
	T* deriv_memo = memo + data.m2 * data.m;
	T* accumulated_derivs = deriv_memo + data.m2 * data.m;

	const T* path_row = path + batch_idx * length * dimension;
	T* path_derivs_row = path_derivs + batch_idx * length * dimension;
	const T* out_derivs_row = out_derivs + batch_idx * data.m;
	T* shared_left = nullptr;
	T* shared_right = nullptr;
	if constexpr (Shared) {
		extern __shared__ char shared_bytes[];
		shared_left = reinterpret_cast<T*>(shared_bytes);
		shared_right = shared_left + data.m;
	}

	for (uint64_t index = tid; index < length * dimension; index += stride)
		path_derivs_row[index] = T(0);
	for (uint64_t k = tid; k < data.m; k += stride) {
		current[k] = T(0);
		memo[data.m + k] = T(0);
	}
	if constexpr (SparseForward) {
		for (uint64_t index = 2 * data.m + tid;
			index < data.m2 * data.m; index += stride)
			memo[index] = T(0);
	}
	__syncthreads();

	evaluate_mkw_segment_(
		path_row, path_row + dimension, current, T(1), data);
	__syncthreads();
	for (uint64_t segment = 1; segment < segment_count; ++segment) {
		const T* left = path_row + segment * dimension;
		const T* right = left + dimension;
		const T* accumulator = current;
		T* next = previous;
		if constexpr (SaveStates) {
			accumulator = states + (segment - 1) * data.m;
			next = states + segment * data.m;
		}
		for (uint64_t k = tid; k < data.m; k += stride) {
			memo[k] = accumulator[k];
			next[k] = accumulator[k];
		}
		evaluate_mkw_segment_(left, right, memo + data.m, T(1), data);
		__syncthreads();
		for (uint64_t q = tid; q < data.segment_count; q += stride) {
			const uint32_t coordinate = data.segment_idx[q];
			next[coordinate] += memo[data.m + coordinate];
		}
		__syncthreads();
		evaluate_mkw_bch_nodes_<T, SparseForward, Shared, true>(
			memo, next, shared_left, shared_right, data);
		if constexpr (!SaveStates) {
			T* temporary = current;
			current = previous;
			previous = temporary;
		}
	}

	for (uint64_t k = tid; k < data.m; k += stride)
		accumulated_derivs[k] = out_derivs_row[k];
	__syncthreads();
	for (uint64_t segment_offset = 0;
		segment_offset + 1 < segment_count; ++segment_offset) {
		const uint64_t segment = segment_count - segment_offset - 1;
		const T* left = path_row + segment * dimension;
		const T* right = left + dimension;
		if constexpr (SaveStates) {
			previous = states + (segment - 1) * data.m;
		}
		else {
			for (uint64_t k = tid; k < data.m; k += stride) {
				memo[k] = current[k];
				previous[k] = current[k];
			}
			evaluate_mkw_segment_(left, right, memo + data.m, T(-1), data);
			__syncthreads();
			for (uint64_t q = tid; q < data.segment_count; q += stride) {
				const uint32_t coordinate = data.segment_idx[q];
				previous[coordinate] += memo[data.m + coordinate];
			}
			__syncthreads();
			evaluate_mkw_bch_nodes_<T, SparseForward, Shared, true>(
				memo, previous, shared_left, shared_right, data);
		}

		for (uint64_t k = tid; k < data.m; k += stride)
			memo[k] = previous[k];
		evaluate_mkw_segment_(left, right, memo + data.m, T(1), data);
		__syncthreads();
		evaluate_mkw_bch_nodes_<T, SparseForward, Shared, false>(
			memo, nullptr, shared_left, shared_right, data);

		for (uint64_t k = tid; k < data.m; k += stride) {
			deriv_memo[k] = T(0);
			deriv_memo[data.m + k] = T(0);
		}
		if constexpr (SparseForward && !SparseReverse) {
			for (uint64_t index = 2 * data.m + tid;
				index < data.m2 * data.m; index += stride)
				deriv_memo[index] = T(0);
		}
		__syncthreads();
		if constexpr (SparseForward) {
			for (uint64_t w = 2; w < data.m2; ++w) {
				const T coefficient = static_cast<T>(data.bch_coefficients[w]);
				for (uint64_t q = data.linear_output_ptr[w] + tid;
					q < data.linear_output_ptr[w + 1]; q += stride) {
					const uint32_t k = data.linear_output_idx[q];
					deriv_memo[w * data.m + k]
						= coefficient * accumulated_derivs[k];
				}
			}
		}
		else {
			for (uint64_t w = 2; w < data.m2; ++w) {
				const T coefficient = static_cast<T>(data.bch_coefficients[w]);
				for (uint64_t k = tid; k < data.m; k += stride)
					deriv_memo[w * data.m + k]
						= coefficient * accumulated_derivs[k];
			}
		}
		__syncthreads();
		reverse_mkw_bch_nodes_<T, SparseReverse, Shared>(
			memo, deriv_memo, shared_left, shared_right, data);

		add_mkw_segment_vjp_(
			left, right, accumulated_derivs, deriv_memo + data.m,
			path_derivs_row + segment * dimension,
			path_derivs_row + (segment + 1) * dimension,
			dimension, data);
		__syncthreads();
		for (uint64_t k = tid; k < data.m; k += stride)
			accumulated_derivs[k] += deriv_memo[k];
		__syncthreads();
		if constexpr (!SaveStates) {
			T* temporary = current;
			current = previous;
			previous = temporary;
		}
	}

	add_mkw_segment_vjp_(
		path_row, path_row + dimension, accumulated_derivs,
		static_cast<const T*>(nullptr),
		path_derivs_row, path_derivs_row + dimension, dimension, data);
}

template<typename T, bool SaveStates, bool Shared>
void launch_mkw_backward_(
	bool sparse_forward,
	bool sparse_reverse,
	const T* out_derivs,
	T* path_derivs,
	const T* path,
	T* workspace,
	uint64_t batch,
	uint64_t length,
	uint64_t dimension,
	unsigned int threads,
	size_t shared_size,
	const CuMkwBchDeviceData& data
) {
	if (sparse_forward && sparse_reverse) {
		branched_log_sig_from_path_backprop_kernel_<
			T, true, SaveStates, true, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
	else if (sparse_forward) {
		branched_log_sig_from_path_backprop_kernel_<
			T, true, SaveStates, false, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
	else if (sparse_reverse) {
		branched_log_sig_from_path_backprop_kernel_<
			T, false, SaveStates, true, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
	else {
		branched_log_sig_from_path_backprop_kernel_<
			T, false, SaveStates, false, Shared><<<
				static_cast<unsigned int>(batch), threads, shared_size>>>(
					out_derivs, path_derivs, path, workspace,
					length, dimension, data);
	}
}

template<typename T>
void branched_log_sig_from_path_backprop_cuda_(
	const T* out_derivs,
	T* path_derivs,
	const T* path,
	uint64_t batch_size,
	uint64_t length,
	uint64_t dimension,
	uint64_t max_nodes
) {
	if (length == 0)
		throw std::invalid_argument(
			"branched_log_sig method 3 received an empty path");
	const CuMkwBchCache& cache = get_cuda_mkw_bch_cache_(
		dimension, max_nodes);
	const CuMkwBchDeviceData data = cache.device_data();
	if (batch_size == 0)
		return;
	const uint64_t path_stride = checked_mkw_product_(
		length, dimension, "MKW BCH path stride overflow");
	const uint64_t path_elements = checked_mkw_product_(
		batch_size, path_stride, "MKW BCH path gradient size overflow");
	checked_mkw_product_(
		batch_size, data.m, "MKW BCH output derivative size overflow");
	if (length == 1 || data.m == 0) {
		if (path_elements != 0) {
			CUDA_CHECK(cudaMemset(path_derivs, 0, checked_mkw_product_(
				path_elements, sizeof(T),
				"MKW BCH path gradient byte size overflow")));
			check_cuda_kernel_launch();
		}
		return;
	}

	const uint64_t segment_count = length - 1;
	const bool save_states = length > 2 && segment_count <= 2 * data.m2;
	const uint64_t state_count = save_states ? segment_count : 2;
	const uint64_t memo_size = checked_mkw_product_(
		data.m2, data.m, "MKW BCH backward memo size overflow");
	const uint64_t workspace_per_batch = checked_mkw_sum_(
		checked_mkw_product_(state_count + 1, data.m,
			"MKW BCH backward state size overflow"),
		checked_mkw_product_(2, memo_size,
			"MKW BCH backward workspace size overflow"),
		"MKW BCH backward workspace size overflow");
	const uint64_t workspace_bytes = checked_mkw_product_(
		workspace_per_batch, sizeof(T),
		"MKW BCH backward workspace byte size overflow");
	size_t free_memory = 0;
	size_t total_memory = 0;
	CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
	const uint64_t reservation_bytes = checked_mkw_product_(
		2, workspace_bytes,
		"MKW BCH backward memory reservation overflow");
	uint64_t chunk_size = workspace_bytes == 0
		? batch_size
		: free_memory / reservation_bytes;
	chunk_size = std::max<uint64_t>(1, chunk_size);
	chunk_size = std::min<uint64_t>(
		std::min<uint64_t>(batch_size, chunk_size), CUDA_GRID_X_LIMIT);
	CudaBuf<T> workspace(checked_mkw_product_(
		chunk_size, workspace_bytes,
		"MKW BCH backward workspace allocation overflow"));

	unsigned int threads = static_cast<unsigned int>(
		std::min<uint64_t>(64, data.m));
	threads = std::max(32U, ((threads + 31) / 32) * 32);
	const size_t shared_size = checked_mkw_product_(
		checked_mkw_product_(2, data.m,
			"MKW BCH backward shared size overflow"),
		sizeof(T), "MKW BCH backward shared byte size overflow");
	const bool shared = shared_size <= CUDA_BASE_DYNAMIC_SMEM;
	const uint64_t combine_count = length > 2 ? length - 2 : 0;
	const long double passes = static_cast<long double>(combine_count)
		* (save_states ? 2 : 3);
	const long double dense_work = passes
		* cache.bch.linear_dense_forward_work;
	const long double exact_work = passes
		* cache.exact_forward_work + cache.exact_zero_work;
	const bool sparse_forward = data.m2 > 2 && exact_work < dense_work;
	const bool sparse_reverse = data.m2 > 2 && cache.prune_reverse;

	for (uint64_t offset = 0; offset < batch_size; offset += chunk_size) {
		const uint64_t current_batch = std::min(
			chunk_size, batch_size - offset);
		const T* chunk_out_derivs = out_derivs + offset * data.m;
		T* chunk_path_derivs = path_derivs + offset * path_stride;
		const T* chunk_path = path + offset * path_stride;
		if (save_states && shared) {
			launch_mkw_backward_<T, true, true>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, shared_size, data);
		}
		else if (save_states) {
			launch_mkw_backward_<T, true, false>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, 0, data);
		}
		else if (shared) {
			launch_mkw_backward_<T, false, true>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, shared_size, data);
		}
		else {
			launch_mkw_backward_<T, false, false>(
				sparse_forward, sparse_reverse, chunk_out_derivs,
				chunk_path_derivs, chunk_path, workspace.get(), current_batch,
				length, dimension, threads, 0, data);
		}
		check_cuda_kernel_launch();
	}
}

}  // namespace

void prepare_cuda_branched_bch_cache_(
	const BranchedSigCache& cache,
	bool
) {
	const CuLogSigCacheKey key = make_cuda_log_sig_cache_key_(
		cache.dimension, cache.max_nodes);
	{
		std::lock_guard<std::mutex> lock(s_mkw_bch_cache_mu);
		if (s_mkw_bch_cache.find(key) != s_mkw_bch_cache.end())
			return;
	}
	auto built = build_cuda_mkw_bch_cache_(cache);
	std::lock_guard<std::mutex> lock(s_mkw_bch_cache_mu);
	s_mkw_bch_cache.try_emplace(key, std::move(built));
}

void clear_cuda_branched_bch_cache_() {
	std::lock_guard<std::mutex> lock(s_mkw_bch_cache_mu);
	s_mkw_bch_cache.clear();
}

extern "C" {

	CUSIG_API int branched_log_sig_from_path_cuda_f(
		const float* path, float* out, uint64_t batch_size, uint64_t length,
		uint64_t dimension, uint64_t max_nodes
	) noexcept {
		CUSIG_SAFE_CALL(branched_log_sig_from_path_cuda_<float>(
			path, out, batch_size, length, dimension, max_nodes));
	}

	CUSIG_API int branched_log_sig_from_path_cuda_d(
		const double* path, double* out, uint64_t batch_size, uint64_t length,
		uint64_t dimension, uint64_t max_nodes
	) noexcept {
		CUSIG_SAFE_CALL(branched_log_sig_from_path_cuda_<double>(
			path, out, batch_size, length, dimension, max_nodes));
	}

	CUSIG_API int branched_log_sig_from_path_backprop_cuda_f(
		const float* derivs, float* path_derivs, const float* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension,
		uint64_t max_nodes
	) noexcept {
		CUSIG_SAFE_CALL(branched_log_sig_from_path_backprop_cuda_<float>(
			derivs, path_derivs, path, batch_size, length, dimension, max_nodes));
	}

	CUSIG_API int branched_log_sig_from_path_backprop_cuda_d(
		const double* derivs, double* path_derivs, const double* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension,
		uint64_t max_nodes
	) noexcept {
		CUSIG_SAFE_CALL(branched_log_sig_from_path_backprop_cuda_<double>(
			derivs, path_derivs, path, batch_size, length, dimension, max_nodes));
	}

}
