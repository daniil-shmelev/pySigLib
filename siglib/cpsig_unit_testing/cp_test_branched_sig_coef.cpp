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
#include "cpsig.h"

TEST(BranchedSigCoef, ForwardAndBackpropMatchFullSignature) {
	constexpr uint64_t batch_size = 2;
	constexpr uint64_t dimension = 2;
	constexpr uint64_t length = 6;
	constexpr uint64_t degree = 3;
	ASSERT_EQ(prepare_branched_sig(dimension, degree, false, false), 0);
	const uint64_t full_length = branched_sig_length(dimension, degree, false);

	std::vector<double> path(batch_size * length * dimension);
	for (uint64_t i = 0; i < path.size(); ++i)
		path[i] = 0.03 * static_cast<double>(i) - 0.2;
	const std::vector<uint64_t> tree_indices{ 0, 1, 4, full_length - 1, 4 };
	const std::vector<uint64_t> tree_data{
		5, 0, 1, 0, 0, 1, 1, 1, 0, 0,
		1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0
	};
	ASSERT_EQ(prepare_branched_sig_coef(
		tree_data.data(), tree_data.size(), dimension, dimension, degree), 0);
	std::vector<double> full(batch_size * full_length);
	std::vector<double> coefs(batch_size * tree_indices.size());
	ASSERT_EQ(branched_sig_d(
		path.data(), full.data(), batch_size, dimension, length, degree), 0);
	ASSERT_EQ(branched_sig_coef_d(
		path.data(), coefs.data(), tree_data.data(), tree_data.size(),
		batch_size, dimension, length, degree), 0);

	for (uint64_t batch = 0; batch < batch_size; ++batch) {
		for (uint64_t i = 0; i < tree_indices.size(); ++i)
			EXPECT_NEAR(coefs[batch * tree_indices.size() + i],
				full[batch * full_length + tree_indices[i]], 1e-14);
	}

	std::vector<double> derivs(batch_size * tree_indices.size());
	for (uint64_t i = 0; i < derivs.size(); ++i)
		derivs[i] = 0.1 + 0.02 * static_cast<double>(i);
	std::vector<double> full_derivs(batch_size * full_length, 0.0);
	for (uint64_t batch = 0; batch < batch_size; ++batch) {
		for (uint64_t i = 0; i < tree_indices.size(); ++i)
			full_derivs[batch * full_length + tree_indices[i]] +=
				derivs[batch * tree_indices.size() + i];
	}

	std::vector<double> expected(batch_size * length * dimension);
	std::vector<double> actual(batch_size * length * dimension);
	ASSERT_EQ(branched_sig_backprop_d(
		path.data(), expected.data(), full_derivs.data(), full.data(),
		batch_size, dimension, length, degree), 0);
	ASSERT_EQ(branched_sig_coef_backprop_d(
		path.data(), actual.data(), coefs.data(), derivs.data(),
		tree_data.data(), tree_data.size(), batch_size, dimension, length, degree), 0);
	for (uint64_t i = 0; i < actual.size(); ++i)
		EXPECT_NEAR(actual[i], expected[i], 2e-13);
}

TEST(BranchedSigCoef, RequiresPreparation) {
	ASSERT_EQ(clear_cache(false), 0);
	constexpr uint64_t tree_data[] = { 1, 1, 0, 0 };
	constexpr double path[] = { 0.0, 0.0, 1.0, 1.0 };
	double out = 0.0;

	EXPECT_NE(branched_sig_coef_d(
		path, &out, tree_data, std::size(tree_data), 1, 2, 2, 1), 0);
	ASSERT_EQ(prepare_branched_sig_coef(
		tree_data, std::size(tree_data), 2, 2, 1), 0);
	EXPECT_EQ(branched_sig_coef_d(
		path, &out, tree_data, std::size(tree_data), 1, 2, 2, 1), 0);
}

TEST(BranchedSigCoef, RejectsEmptyLeadLagPath) {
	constexpr uint64_t tree_data[] = { 1, 0 };
	constexpr double coef = 1.0;
	constexpr double deriv = 1.0;
	double out = 0.0;

	EXPECT_NE(branched_sig_coef_d(
		nullptr, &out, tree_data, std::size(tree_data), 1, 2, 0, 0,
		1, false, true), 0);
	EXPECT_NE(branched_sig_coef_backprop_d(
		nullptr, &out, &coef, &deriv, tree_data, std::size(tree_data), 1, 2, 0, 0,
		1, false, true), 0);
}
