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

#include "cp_test_helpers.h"


namespace {

using polynomial_kernel_d = int (*)(const double*, double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool, int);

void check_golden(
	polynomial_kernel_d kernel,
	const std::vector<double>& gram,
	uint64_t length1,
	uint64_t length2,
	const std::vector<double>& expected
) {
	const uint64_t orders[] = { 2, 5, 7, 10 };
	for (uint64_t i = 0; i < 4; ++i) {
		double out = 0;
		ASSERT_EQ(kernel(gram.data(), &out, nullptr, 1, 2, length1, length2, orders[i], false, 1), 0);
		EXPECT_NEAR(out, expected[i], 2e-14) << "order = " << orders[i];
	}
}

}


TEST(polynomialSigKernelTest, PinnedGoldenOneTile) {
	check_golden(polysig_kernel_d, { 2.0 }, 2, 2,
		{ 4.0, 4.252222222222223, 4.252350718065004, 4.252350879501321 });

	check_golden(polysig_kernel_d, { -0.75 }, 2, 2,
		{ 0.390625, 0.3794390869140625, 0.3794394249818762, 0.37943942504289163 });

	check_golden(polysig_kernel_d, { 0.0 }, 2, 2, { 1.0, 1.0, 1.0, 1.0 });
}

TEST(polynomialSigKernelTest, PinnedGoldenRectangular) {
	const std::vector<double> gram = {
		0.04, -0.115,
		0.02, 0.055,
		0.065, -0.08
	};
	check_golden(polysig_kernel_d, gram, 4, 3,
		{ 0.9868383588452052, 0.9868046051227348, 0.9868046051301949, 0.9868046051301949 });
}

TEST(polynomialSigKernelTest, PinnedGoldenBatchAndThreads) {
	const std::vector<double> gram = {
		0.04, -0.115,
		0.02, 0.055,
		0.065, -0.08,
		-0.04, -0.01,
		-0.045, 0.1575,
		0.055, -0.0875
	};
	const uint64_t orders[] = { 2, 5, 7, 10 };
	const double polysig_expected[][2] = {
		{ 0.9868383588452052, 1.0265309020995697 },
		{ 0.9868046051227348, 1.026566588558175 },
		{ 0.9868046051301949, 1.026566588567865 },
		{ 0.9868046051301949, 1.026566588567865 }
	};

	for (uint64_t i = 0; i < 4; ++i) {
		std::vector<double> serial(2);
		std::vector<double> parallel(2);
		ASSERT_EQ(polysig_kernel_d(gram.data(), serial.data(), nullptr, 2, 2, 4, 3, orders[i], false, 1), 0);
		ASSERT_EQ(polysig_kernel_d(gram.data(), parallel.data(), nullptr, 2, 2, 4, 3, orders[i], false, 2), 0);
		for (uint64_t batch = 0; batch < 2; ++batch) {
			EXPECT_NEAR(serial[batch], polysig_expected[i][batch], 2e-14);
			EXPECT_DOUBLE_EQ(serial[batch], parallel[batch]);
		}
	}
}

TEST(polynomialSigKernelTest, Float32MatchesFloat64) {
	const std::vector<float> gram = {
		0.04f, -0.115f,
		0.02f, 0.055f,
		0.065f, -0.08f
	};
	float polysig_out = 0;
	ASSERT_EQ(polysig_kernel_f(gram.data(), &polysig_out, nullptr, 1, 2, 4, 3, 7, false, 1), 0);
	EXPECT_NEAR(polysig_out, 0.9868046051301949, 2e-6);
}

TEST(polynomialSigKernelTest, TrivialAndNullGram) {
	std::vector<double> out(3, 0);
	EXPECT_EQ(polysig_kernel_d(nullptr, out.data(), nullptr, 3, 2, 1, 5, 7, false, 2), 0);
	EXPECT_EQ(out, std::vector<double>({ 1.0, 1.0, 1.0 }));
	EXPECT_EQ(polysig_kernel_d(nullptr, out.data(), nullptr, 3, 2, 5, 5, 7, false, 2), 2);
}

TEST(polynomialSigKernelTest, RejectsInvalidOrder) {
	double gram = 0;
	double out = 0;
	EXPECT_EQ(polysig_kernel_d(&gram, &out, nullptr, 1, 1, 2, 2, 1, false, 1), 2);
	EXPECT_EQ(polysig_kernel_d(&gram, &out, nullptr, 1, 1, 2, 2, 65, false, 1), 2);
}

TEST(polynomialSigKernelTest, GridEndsAtScalarAndStoresState) {
	const std::vector<double> gram = {
		0.04, -0.115,
		0.02, 0.055,
		0.065, -0.08
	};
	std::vector<double> grid(12);
	std::vector<double> state(2 * 6 * 8);
	double scalar = 0;
	ASSERT_EQ(polysig_kernel_d(gram.data(), &scalar, nullptr, 1, 2, 4, 3, 7, false, 1), 0);
	ASSERT_EQ(polysig_kernel_d(gram.data(), grid.data(), state.data(), 1, 2, 4, 3, 7, true, 1), 0);
	EXPECT_DOUBLE_EQ(grid.back(), scalar);
	for (uint64_t j = 0; j < 3; ++j)
		EXPECT_DOUBLE_EQ(grid[j], 1.0);
	for (uint64_t i = 0; i < 4; ++i)
		EXPECT_DOUBLE_EQ(grid[i * 3], 1.0);
	EXPECT_DOUBLE_EQ(state[0], 1.0);
	EXPECT_DOUBLE_EQ(state[8], 1.0);
}

TEST(polynomialSigKernelTest, BackpropMatchesFiniteDifferences) {
	std::vector<double> gram = {
		0.04, -0.115,
		0.02, 0.055,
		0.065, -0.08
	};
	std::vector<double> grid(12);
	std::vector<double> state(2 * 6 * 8);
	std::vector<double> output_derivs = {
		0.1, -0.2, 0.3,
		-0.4, 0.5, -0.6,
		0.7, -0.8, 0.9,
		-1.0, 1.1, -1.2
	};
	std::vector<double> gram_derivs(6);
	ASSERT_EQ(polysig_kernel_d(gram.data(), grid.data(), state.data(), 1, 2, 4, 3, 7, true, 1), 0);
	ASSERT_EQ(polysig_kernel_backprop_d(
		gram.data(), gram_derivs.data(), output_derivs.data(), state.data(),
		1, 2, 4, 3, 7, true, 1), 0);

	const double epsilon = 1e-6;
	for (uint64_t k = 0; k < gram.size(); ++k) {
		const double original = gram[k];
		gram[k] = original + epsilon;
		std::vector<double> plus(12);
		ASSERT_EQ(polysig_kernel_d(gram.data(), plus.data(), nullptr, 1, 2, 4, 3, 7, true, 1), 0);
		gram[k] = original - epsilon;
		std::vector<double> minus(12);
		ASSERT_EQ(polysig_kernel_d(gram.data(), minus.data(), nullptr, 1, 2, 4, 3, 7, true, 1), 0);
		gram[k] = original;
		double expected = 0;
		for (uint64_t p = 0; p < grid.size(); ++p)
			expected += output_derivs[p] * (plus[p] - minus[p]) / (2 * epsilon);
		EXPECT_NEAR(gram_derivs[k], expected, 2e-9);
	}

	std::vector<double> regenerated_derivs(6);
	ASSERT_EQ(polysig_kernel_backprop_d(
		gram.data(), regenerated_derivs.data(), output_derivs.data(), nullptr,
		1, 2, 4, 3, 7, true, 1), 0);
	EXPECT_EQ(gram_derivs, regenerated_derivs);
}
