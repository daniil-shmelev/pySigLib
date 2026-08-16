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

#include <concepts>
#include <cstdint>
#include <vector>

template<std::floating_point T>
struct PolynomialSigKernelTables {
	uint64_t size;
	std::vector<T> mat1;
	std::vector<T> mat1_deriv;
	std::vector<T> mat2;

	explicit PolynomialSigKernelTables(uint64_t order)
		: size(order + 1),
		mat1(size * size, static_cast<T>(0)),
		mat1_deriv(size * size, static_cast<T>(0)),
		mat2(size * size, static_cast<T>(0)) {
		long double inverse_factorial = 1.0L;
		for (uint64_t n = 1; n < size; ++n) {
			inverse_factorial /= static_cast<long double>(n);
			long double value = inverse_factorial * inverse_factorial;
			mat2[n * size] = static_cast<T>(value);
			for (uint64_t k = 1; k < size; ++k) {
				value *= static_cast<long double>(k)
					/ static_cast<long double>(n + k);
				mat2[n * size + k] = static_cast<T>(value);
			}
		}

		long double inverse_nm1_factorial = 1.0L;
		for (uint64_t n = 2; n < size; ++n) {
			const long double inverse_n_factorial = inverse_nm1_factorial
				/ static_cast<long double>(n);
			long double value = inverse_n_factorial * inverse_nm1_factorial;
			mat1[n * size + 1] = static_cast<T>(value);
			mat1_deriv[n * size + 1]
				= static_cast<T>(n - 1) * static_cast<T>(value);
			for (uint64_t k = 2; k < n; ++k) {
				value *= static_cast<long double>(k * (n - k + 1));
				mat1[n * size + k] = static_cast<T>(value);
				mat1_deriv[n * size + k]
					= static_cast<T>(n - k) * static_cast<T>(value);
			}
			inverse_nm1_factorial = inverse_n_factorial;
		}
	}
};
