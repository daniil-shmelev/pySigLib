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
#include "branched_sig/branched_log_sig_cache.h"
#include "branched_sig/branched_sig_cache_io.h"
#include "cache_io.h"
#include "polynomial_sig_kernel/polynomial_sig_kernel_tables.h"

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
	BchCache cache;
	cache.degree = 20;
	build_bch_formula_data(cache);
	ASSERT_EQ(cache.bch_coefficients.size(), 111013);
	EXPECT_DOUBLE_EQ(cache.bch_coefficients[0], 1.0);
	EXPECT_DOUBLE_EQ(cache.bch_coefficients[1], 1.0);
	EXPECT_DOUBLE_EQ(cache.bch_coefficients[2], 0.5);
	EXPECT_DOUBLE_EQ(cache.bch_coefficients[3], 1.0 / 12.0);
	EXPECT_DOUBLE_EQ(cache.bch_coefficients[4], 1.0 / 12.0);
	EXPECT_DOUBLE_EQ(cache.bch_coefficients[5], 0.0);
	EXPECT_DOUBLE_EQ(
		cache.bch_coefficients[110262],
		43867.0 / 10218188434341888000.0);
	EXPECT_EQ(cache.bch_left_factor.back(), 58635);
	EXPECT_EQ(cache.bch_right_factor.back(), 1);
	ASSERT_EQ(cache.bch_left_factor.size(), cache.bch_coefficients.size());
	ASSERT_EQ(cache.bch_right_factor.size(), cache.bch_coefficients.size());
	for (uint64_t node = 2; node < cache.bch_coefficients.size(); ++node) {
		EXPECT_LT(cache.bch_left_factor[node], node);
		EXPECT_LT(cache.bch_right_factor[node], node);
	}
}

TEST(preparationCacheTest, BchHandlesZeroAndRejectsAboveHardcodedTable) {
	BchCache empty;
	empty.degree = 0;
	EXPECT_NO_THROW(build_bch_formula_data(empty));
	EXPECT_TRUE(empty.bch_coefficients.empty());

	BchCache too_large;
	too_large.degree = 21;
	EXPECT_THROW(build_bch_formula_data(too_large), std::invalid_argument);
}

TEST(preparationCacheTest, LiveBchPlansAreMinimalAndTopological) {
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
		build_live_bch_nodes(cache);
		EXPECT_EQ(cache.live_bch_nodes.size(), expected_size);
		EXPECT_EQ(cache.all_bch_nodes_live,
			expected_size + 2 == cache.bch_coefficients.size());

		std::vector<uint8_t> available(cache.bch_coefficients.size(), 0);
		if (!available.empty())
			available[0] = 1;
		if (available.size() > 1)
			available[1] = 1;
		for (uint32_t node : cache.live_bch_nodes) {
			EXPECT_TRUE(available[cache.bch_left_factor[node]]);
			EXPECT_TRUE(available[cache.bch_right_factor[node]]);
			available[node] = 1;
		}
		for (uint64_t node = 2; node < cache.bch_coefficients.size(); ++node) {
			if (cache.bch_coefficients[node] != 0.0)
				EXPECT_TRUE(available[node]);
		}
	}
}

TEST(preparationCacheTest, PrunedAndFullBchPlansAgree) {
	const uint64_t degrees[] = { 3, 4, 5, 8 };
	for (uint64_t degree : degrees) {
		SCOPED_TRACE(degree);
		LogSigCache log_cache(2, degree, 3);
		const BchCache& pruned = log_cache.bch();
		BchCache full = pruned;
		full.live_bch_nodes.clear();
		for (uint64_t node = 2; node < full.bch_coefficients.size(); ++node)
			full.live_bch_nodes.push_back(static_cast<uint32_t>(node));
		full.all_bch_nodes_live = true;

		const uint64_t m = pruned.m;
		const uint64_t m2 = pruned.bch_coefficients.size();
		std::vector<double> ls1(m), ls2(m), d_out(m);
		for (uint64_t index = 0; index < m; ++index) {
			ls1[index] = 0.03125 * (static_cast<int>(index % 9) - 4);
			ls2[index] = 0.015625 * (static_cast<int>(index % 7) - 3);
			d_out[index] = 0.0625 * (static_cast<int>(index % 5) - 2);
		}
		std::vector<double> pruned_out(m), full_out(m);
		std::vector<double> pruned_memo(m2 * m), full_memo(m2 * m);
		bch_combine_impl_(
			ls1.data(), ls2.data(), pruned_out.data(), pruned,
			pruned_memo.data());
		bch_combine_impl_(
			ls1.data(), ls2.data(), full_out.data(), full,
			full_memo.data());
		EXPECT_EQ(pruned_out, full_out);

		std::vector<double> pruned_d_ls1(m), pruned_d_ls2(m);
		std::vector<double> full_d_ls1(m), full_d_ls2(m);
		std::vector<double> pruned_workspace(2 * m2 * m);
		std::vector<double> full_workspace(2 * m2 * m);
		bch_combine_backprop_impl_<double>(
			d_out.data(), pruned_d_ls1.data(), pruned_d_ls2.data(),
			ls1.data(), ls2.data(), pruned, pruned_workspace.data());
		bch_combine_backprop_impl_<double>(
			d_out.data(), full_d_ls1.data(), full_d_ls2.data(),
			ls1.data(), ls2.data(), full, full_workspace.data());
		EXPECT_EQ(pruned_d_ls1, full_d_ls1);
		EXPECT_EQ(pruned_d_ls2, full_d_ls2);
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
