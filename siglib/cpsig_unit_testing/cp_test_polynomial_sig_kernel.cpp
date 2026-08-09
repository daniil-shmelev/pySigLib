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

using polynomial_kernel_d = int (*)(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int);

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
		ASSERT_EQ(kernel(gram.data(), &out, 1, 2, length1, length2, orders[i], 1), 0);
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
		ASSERT_EQ(polysig_kernel_d(gram.data(), serial.data(), 2, 2, 4, 3, orders[i], 1), 0);
		ASSERT_EQ(polysig_kernel_d(gram.data(), parallel.data(), 2, 2, 4, 3, orders[i], 2), 0);
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
	ASSERT_EQ(polysig_kernel_f(gram.data(), &polysig_out, 1, 2, 4, 3, 7, 1), 0);
	EXPECT_NEAR(polysig_out, 0.9868046051301949, 2e-6);
}

TEST(polynomialSigKernelTest, TrivialAndNullGram) {
	std::vector<double> out(3, 0);
	EXPECT_EQ(polysig_kernel_d(nullptr, out.data(), 3, 2, 1, 5, 7, 2), 0);
	EXPECT_EQ(out, std::vector<double>({ 1.0, 1.0, 1.0 }));
	EXPECT_EQ(polysig_kernel_d(nullptr, out.data(), 3, 2, 5, 5, 7, 2), 2);
}

TEST(polynomialSigKernelTest, RejectsInvalidOrder) {
	double gram = 0;
	double out = 0;
	EXPECT_EQ(polysig_kernel_d(&gram, &out, 1, 1, 2, 2, 1, 1), 2);
	EXPECT_EQ(polysig_kernel_d(&gram, &out, 1, 1, 2, 2, 65, 1), 2);
}
