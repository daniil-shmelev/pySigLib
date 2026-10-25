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
#include "../shared/chebyshev_sig_kernel.h"

#include <array>
#include <random>
#include <type_traits>

namespace {

template<typename T, int P, int K = P>
void check_cell_specialization() {
	using namespace chebyshev_sig_kernel;
	std::mt19937 generator(20261024);
	std::uniform_real_distribution<double> random(-1., 1.);
	const T tolerance = std::is_same_v<T, float> ? T(3e-5) : T(8e-14);
	for (int trial = 0; trial < 80; ++trial) {
		std::array<T, P + 1> f{}, g{}, top{}, right{}, expected_top{}, expected_right{};
		for (int k = 1; k <= P; ++k) { f[k] = T(random(generator)); g[k] = T(random(generator)); }
		const Corner<T> start{ T(random(generator)), T(random(generator) * 1e-9) };
		Corner<T> f_end = start, g_end = start;
		for (int k = 1; k <= P; k += 2) {
			f_end = corner_add(f_end, { T(2) * f[k], T(0) });
			g_end = corner_add(g_end, { T(2) * g[k], T(0) });
		}
		const T coefficients[] = { T(0), T(1e-12), T(-1e-12), T(-4), T(4), T(random(generator)) };
		for (T c : coefficients) {
			const auto actual = cell_forward<T, P, K, true>(c, f.data(), g.data(), start, f_end, g_end,
				top.data(), right.data(), P, K);
			const auto expected = cell_forward<T>(c, f.data(), g.data(), start, f_end, g_end,
				expected_top.data(), expected_right.data(), P, K);
			const T scale = T(1) + std::abs(corner_value(expected));
			EXPECT_NEAR(corner_value(actual), corner_value(expected), tolerance * scale);
			for (int k = 0; k <= P; ++k) {
				EXPECT_NEAR(top[k], expected_top[k], tolerance * (scale + std::abs(expected_top[k])));
				EXPECT_NEAR(right[k], expected_right[k], tolerance * (scale + std::abs(expected_right[k])));
			}
		}
	}
}

template<int P>
void check_cell_gradient() {
	using namespace chebyshev_sig_kernel;
	std::array<double, P + 1> f{}, g{}, top_weight{}, right_weight{}, f_grad{}, g_grad{};
	top_weight[0] = 0.31; right_weight[0] = -0.29;
	for (int k = 1; k <= P; ++k) {
		f[k] = 0.17 * k; g[k] = -0.11 * k;
		top_weight[k] = 0.13 * k; right_weight[k] = -0.23 * k;
	}
	Corner<double> start{ 0.7, 0. }, f_end{ 1.1, 0. }, g_end{ -0.2, 0. };
	double c = 0.;
	const auto evaluate = [&]() {
		std::array<double, P + 1> top{}, right{};
		const auto end = cell_forward<double, P, P>(c, f.data(), g.data(), start, f_end, g_end,
			top.data(), right.data(), P, P);
		double value = 0.37 * corner_value(end);
		for (int k = 0; k <= P; ++k) value += top_weight[k] * top[k] + right_weight[k] * right[k];
		return value;
	};
	const auto finite_difference = [&](double& value) {
		const double original = value, step = 1e-6;
		value = original + step;
		const double plus = evaluate();
		value = original - step;
		const double minus = evaluate();
		value = original;
		return (plus - minus) / (2 * step);
	};
	for (double coefficient : { 0., -0.7, 0.9 }) {
		c = coefficient;
		const auto derivative = cell_backward<double, P, P>(c, f.data(), g.data(), start,
			top_weight.data(), right_weight.data(), 0.37, f_grad.data(), g_grad.data(), P, P);
		EXPECT_NEAR(derivative.coefficient, finite_difference(c), 2e-9);
		EXPECT_NEAR(derivative.start, finite_difference(start.hi), 2e-9);
		EXPECT_NEAR(derivative.bottom_end, finite_difference(f_end.hi), 2e-9);
		EXPECT_NEAR(derivative.left_end, finite_difference(g_end.hi), 2e-9);
		for (int k = 0; k <= P; ++k) {
			EXPECT_NEAR(f_grad[k], finite_difference(f[k]), 2e-9);
			EXPECT_NEAR(g_grad[k], finite_difference(g[k]), 2e-9);
		}
	}
}

}

TEST(chebyshevSigKernelTest, SpecializedCellsMatchGenericFloat) {
	check_cell_specialization<float, 1>();
	check_cell_specialization<float, 2>();
	check_cell_specialization<float, 3>();
	check_cell_specialization<float, 1, 2>();
	check_cell_specialization<float, 1, 3>();
	check_cell_specialization<float, 2, 3>();
}

TEST(chebyshevSigKernelTest, SpecializedCellsMatchGenericDouble) {
	check_cell_specialization<double, 1>();
	check_cell_specialization<double, 2>();
	check_cell_specialization<double, 3>();
	check_cell_specialization<double, 1, 2>();
	check_cell_specialization<double, 1, 3>();
	check_cell_specialization<double, 2, 3>();
}

TEST(chebyshevSigKernelTest, SpecializedCellGradients) {
	check_cell_gradient<1>();
	check_cell_gradient<2>();
	check_cell_gradient<3>();
}
