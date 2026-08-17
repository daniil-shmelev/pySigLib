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

#include <cstdint>
#include <limits>

inline uint64_t tensor_power(uint64_t base, uint64_t exponent) noexcept {
	uint64_t result = 1;
	while (exponent != 0) {
		if ((exponent & 1) != 0) {
			if (result != 0 && base > UINT64_MAX / result)
				return 0;
			result *= base;
		}
		exponent >>= 1;
		if (exponent != 0) {
			if (base != 0 && base > UINT64_MAX / base)
				return 0;
			base *= base;
		}
	}
	return result;
}

inline uint64_t tensor_sig_length(
	uint64_t dimension,
	uint64_t degree
) noexcept {
	if (dimension == 0)
		return 1;
	if (dimension == 1)
		return degree + 1;
	const uint64_t value = tensor_power(dimension, degree + 1);
	return value == 0 ? 0 : (value - 1) / (dimension - 1);
}
