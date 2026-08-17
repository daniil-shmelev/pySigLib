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

#include "bch_cache.h"
#include "branched_log_sig_cache.h"
#include "branched_sig_cache_io.h"
#include "cache_io.h"
#include "polynomial_sig_kernel_tables.h"

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

	const auto path = branched_sig_cache_file_path(directory, 2, 3, true);
	std::ofstream truncated(path, std::ios::binary | std::ios::trunc);
	truncated.write(
		reinterpret_cast<const char*>(&cache_magic_number),
		sizeof(cache_magic_number));
	truncated.close();
	EXPECT_FALSE(read_branched_sig_cache(directory, 2, 3, true, loaded));
	std::filesystem::remove_all(directory);
}
