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

#include <gtest/gtest.h>

#include "cp_bch.h"
#include "log_sig/bch_cache.h"
#include "log_sig/bch_data.h"
#include "branched_sig/branched_log_sig_cache.h"
#include "branched_sig/branched_sig_cache_io.h"
#include "cache_io.h"
#include "polynomial_sig_kernel/polynomial_sig_kernel_tables.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace {
std::filesystem::path test_cache_directory_() {
	const auto stamp = std::chrono::steady_clock::now()
		.time_since_epoch().count();
	return std::filesystem::temp_directory_path()
		/ ("pysiglib_preparation_" + std::to_string(stamp));
}

TEST(preparationCacheTest, HardcodedBchCoversDegreeTwenty) {
	const BchHardcodedData* formula = get_hardcoded_bch_data(20);
	ASSERT_NE(formula, nullptr);
	ASSERT_EQ(formula->size, 111013);
	EXPECT_DOUBLE_EQ(formula->coefficients[0], 1.0);
	EXPECT_DOUBLE_EQ(formula->coefficients[1], 1.0);
	EXPECT_DOUBLE_EQ(formula->coefficients[2], 0.5);
	EXPECT_DOUBLE_EQ(formula->coefficients[3], 1.0 / 12.0);
	EXPECT_DOUBLE_EQ(formula->coefficients[4], 1.0 / 12.0);
	EXPECT_DOUBLE_EQ(formula->coefficients[5], 0.0);
	EXPECT_DOUBLE_EQ(
		formula->coefficients[110262],
		43867.0 / 10218188434341888000.0);
	EXPECT_EQ(formula->left_factor[formula->size - 1], 58635);
	EXPECT_EQ(formula->right_factor[formula->size - 1], 1);
	for (uint64_t node = 2; node < formula->size; ++node) {
		EXPECT_LT(formula->left_factor[node], node);
		EXPECT_LT(formula->right_factor[node], node);
	}
}

TEST(preparationCacheTest, BchHandlesZeroAndRejectsAboveHardcodedTable) {
	BchCache empty;
	empty.degree = 0;
	EXPECT_NO_THROW(build_bch_formula_data(empty));
	EXPECT_TRUE(empty.bch_operations.empty());

	BchCache too_large;
	too_large.degree = 21;
	EXPECT_THROW(build_bch_formula_data(too_large), std::invalid_argument);
}

TEST(preparationCacheTest, BchOperationsAreCompactAndTopological) {
	struct ExpectedPlanSize {
		uint64_t degree;
		size_t live_nodes;
	};
	const ExpectedPlanSize cases[] = {
		{ 1, 0 }, { 2, 1 }, { 3, 3 }, { 4, 4 }, { 5, 12 },
		{ 6, 17 }, { 7, 39 }, { 8, 56 }, { 9, 124 }, { 10, 180 },
		{ 11, 410 }, { 12, 595 }, { 13, 1375 }, { 14, 2004 },
		{ 15, 4717 }, { 16, 6899 }, { 17, 16508 }, { 18, 24217 },
		{ 19, 58634 }, { 20, 86227 }
	};
	for (const auto& [degree, expected_size] : cases) {
		SCOPED_TRACE(degree);
		BchCache cache;
		cache.degree = degree;
		build_bch_formula_data(cache);
		EXPECT_EQ(cache.bch_operations.size(), expected_size);
		EXPECT_EQ(cache.bch_size(), expected_size + 2);
		for (uint64_t operation_index = 0;
			operation_index < cache.bch_operations.size(); ++operation_index) {
			const BchOperation& operation = cache.bch_operations[operation_index];
			const uint64_t node = operation_index + 2;
			EXPECT_LT(operation.left, node);
			EXPECT_LT(operation.right, node);
		}
	}
}

TEST(preparationCacheTest, CompactAndRangedBchMatchFullFormula) {
	for (uint64_t degree : { 1, 3, 4, 5, 8 }) {
		SCOPED_TRACE(degree);
		LogSigCache cache(2, degree, 3);
		const BchCache& compact = cache.bch();
		BchCache full = compact;
		full.bch_operations.clear();
		const auto& formula = *get_hardcoded_bch_data(degree);
		for (uint64_t w = 2; w < formula.size; ++w)
			full.bch_operations.push_back({ formula.coefficients[w],
				static_cast<uint32_t>(formula.left_factor[w]),
				static_cast<uint32_t>(formula.right_factor[w]) });
		const uint64_t m = compact.m;
		std::vector<double> left(m), right(m), derivs(m), expected(m), actual(m);
		for (uint64_t k = 0; k < m; ++k) {
			left[k] = 0.03125 * (static_cast<int>(k % 9) - 4);
			right[k] = 0.015625 * (static_cast<int>(k % 7) - 3);
			derivs[k] = 0.0625 * (static_cast<int>(k % 5) - 2);
		}
		std::vector<double> workspace(2 * full.bch_size() * m);
		bch_combine_impl_(left.data(), right.data(), expected.data(), full, workspace.data());
		bch_combine_impl_(left.data(), right.data(), actual.data(), compact, workspace.data());
		EXPECT_EQ(actual, expected);
		std::vector<double> expected_left(m), expected_right(m), actual_left(m), actual_right(m);
		bch_combine_backprop_impl_(derivs.data(), expected_left.data(), expected_right.data(),
			left.data(), right.data(), full, workspace.data());
		bch_combine_backprop_impl_(derivs.data(), actual_left.data(), actual_right.data(),
			left.data(), right.data(), compact, workspace.data());
		EXPECT_EQ(actual_left, expected_left);
		EXPECT_EQ(actual_right, expected_right);

		// Independently evaluate brackets, discarding coordinates outside the plan.
		std::copy(left.begin(), left.end(), workspace.begin());
		std::copy(right.begin(), right.end(), workspace.begin() + m);
		for (uint64_t k = 0; k < m; ++k)
			actual[k] = left[k] + right[k];
		ASSERT_EQ(compact.bch_ranges.size(), compact.bch_operations.size());
		for (uint64_t w = 2; w < compact.bch_size(); ++w) {
			const auto& op = compact.bch_operation(w);
			const auto [begin, end] = compact.bch_ranges[w - 2];
			ASSERT_LE(begin, end);
			ASSERT_LE(end, m);
			double* result = workspace.data() + w * m;
			lie_bracket(workspace.data() + op.left * m, workspace.data() + op.right * m,
				result, m, compact.commutator_table);
			std::fill(result, result + begin, 0.0);
			std::fill(result + end, result + m, 0.0);
			for (uint64_t k = begin; k < end; ++k)
				actual[k] += op.coefficient * result[k];
		}
		for (uint64_t k = 0; k < m; ++k)
			EXPECT_NEAR(actual[k], expected[k], 1e-12);
	}
}

}

TEST(preparationCacheTest, StandardLogMethodsUpgradeInPlace) {
	LogSigCache cache(2, 4, 1);
	ASSERT_TRUE(cache.supports(1));
	ASSERT_FALSE(cache.supports(2));
	const uint64_t* indices = cache.basis(1).lyndon_idx.data();

	cache.upgrade(2);
	EXPECT_TRUE(cache.supports(2));
	EXPECT_EQ(indices, cache.basis(2).lyndon_idx.data());
	EXPECT_FALSE(cache.basis(2).inv_proj_mat.rows.empty());

	cache.upgrade(3);
	EXPECT_TRUE(cache.supports(3));
	EXPECT_TRUE(cache.has_bch());
	EXPECT_EQ(cache.bch().m, cache.basis(2).lyndon_idx.size());
}

TEST(preparationCacheTest, BranchedLogMethodsOwnPreparedData) {
	const BranchedSigCache nonplanar(2, 3, false);
	BranchedLogSigCache expanded(nonplanar, 0);
	EXPECT_TRUE(expanded.supports(0));
	EXPECT_GT(expanded.horner_plan().product_count, nonplanar.total_length);

	const BranchedSigCache planar(2, 3, true);
	BranchedLogSigCache compact(planar, 1);
	const uint64_t* indices = compact.basis_cache(1).lyndon_idx.data();
	compact.upgrade(planar, 2);
	EXPECT_EQ(indices, compact.basis_cache(2).lyndon_idx.data());
	EXPECT_FALSE(compact.basis_cache(2).inv_proj_mat.rows.empty());
	compact.upgrade(planar, 3);
	EXPECT_TRUE(compact.supports(3));
	EXPECT_EQ(
		compact.bch_cache().bch.m,
		compact.basis_cache(2).lyndon_idx.size());
	EXPECT_EQ(
		compact.bch_cache().linear_coefficients.size(),
		compact.bch_cache().bch.m);
}

TEST(preparationCacheTest, PolynomialTablesMatchAcrossDtypes) {
	const PolynomialSigKernelTables<float> float_tables(7);
	const PolynomialSigKernelTables<double> double_tables(7);
	ASSERT_EQ(float_tables.mat1.size(), double_tables.mat1.size());
	ASSERT_EQ(float_tables.mat2.size(), double_tables.mat2.size());
	for (uint64_t i = 0; i < float_tables.mat1.size(); ++i) {
		EXPECT_NEAR(float_tables.mat1[i], double_tables.mat1[i], 1e-7);
		EXPECT_NEAR(float_tables.mat2[i], double_tables.mat2[i], 1e-7);
	}
}

TEST(preparationCacheTest, BranchedDiskRoundTripAndTruncation) {
	const std::filesystem::path directory = test_cache_directory_();
	const BranchedSigCache expected(2, 3, true);
	write_branched_sig_cache(directory, expected);
	BranchedSigCache loaded;
	ASSERT_TRUE(read_branched_sig_cache(directory, 2, 3, true, loaded));
	EXPECT_EQ(loaded.order_index, expected.order_index);
	EXPECT_EQ(loaded.basis_forest_data, expected.basis_forest_data);
	EXPECT_EQ(loaded.coproduct_data, expected.coproduct_data);
	EXPECT_EQ(loaded.horner.product_count, expected.horner.product_count);
	EXPECT_EQ(
		loaded.horner.planar_coproduct_left,
		expected.horner.planar_coproduct_left);
	EXPECT_EQ(
		loaded.horner.coproduct_pairs,
		expected.horner.coproduct_pairs);
	EXPECT_EQ(
		loaded.horner.correction_horner_roots,
		expected.horner.correction_horner_roots);
	EXPECT_EQ(
		loaded.horner.correction_horner_node_offsets,
		expected.horner.correction_horner_node_offsets);
	EXPECT_EQ(
		loaded.horner.correction_horner_variables,
		expected.horner.correction_horner_variables);
	EXPECT_EQ(
		loaded.horner.correction_horner_children,
		expected.horner.correction_horner_children);
	EXPECT_EQ(
		loaded.horner.correction_horner_constants,
		expected.horner.correction_horner_constants);
	EXPECT_EQ(
		loaded.horner.planar_log_coefficients,
		expected.horner.planar_log_coefficients);
	EXPECT_EQ(
		loaded.horner.planar_log_monomial_parent,
		expected.horner.planar_log_monomial_parent);

	const auto path = branched_sig_cache_file_path(directory, 2, 3, true);
	std::ofstream truncated(path, std::ios::binary | std::ios::trunc);
	truncated.write(
		reinterpret_cast<const char*>(&cache_magic_number),
		sizeof(cache_magic_number));
	truncated.close();
	EXPECT_FALSE(read_branched_sig_cache(directory, 2, 3, true, loaded));
	std::filesystem::remove_all(directory);
}
