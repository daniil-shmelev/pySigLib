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

#include "bch_formula.h"

extern "C" {
#include "bch.h"
}

#include <memory>
#include <stdexcept>

namespace {
using LieSeriesPtr = std::unique_ptr<lie_series_t, decltype(&free_lie_series)>;
}  // namespace

LyndonBchFormula compute_lyndon_bch_formula(uint64_t degree) {
	if (degree == 0 || degree > 30)
		throw std::invalid_argument("BCH truncation degree must be between 1 and 30");
	LieSeriesPtr series(
		BCH(static_cast<int>(degree), LYNDON_BASIS), &free_lie_series);
	if (!series)
		throw std::runtime_error("BCH failed to construct the Lyndon series");
	const int size = dimension(series.get());
	if (size < 0)
		throw std::runtime_error("BCH returned an invalid series dimension");
	const int generator_count = number_of_generators(series.get());
	if (generator_count != 2
		|| maximum_degree(series.get()) != static_cast<int>(degree))
		throw std::runtime_error("BCH returned incompatible series metadata");
	const INTEGER common_denominator = denominator(series.get());
	if (common_denominator == 0)
		throw std::runtime_error("BCH returned a zero common denominator");

	LyndonBchFormula result;
	result.coefficients.resize(static_cast<size_t>(size));
	result.left_factor.assign(static_cast<size_t>(size), UINT64_MAX);
	result.right_factor.assign(static_cast<size_t>(size), UINT64_MAX);
	const long double denominator_value = static_cast<long double>(common_denominator);
	for (int index = 0; index < size; ++index) {
		const INTEGER numerator = numerator_of_coefficient(series.get(), index);
		result.coefficients[static_cast<size_t>(index)] = numerator == 0
			? 0.0
			: static_cast<double>(
				static_cast<long double>(numerator) / denominator_value);
		if (index >= generator_count) {
			const int left = left_factor(series.get(), index);
			const int right = right_factor(series.get(), index);
			if (left < 0 || right < 0 || left >= index || right >= index)
				throw std::runtime_error("BCH returned an invalid factorization");
			result.left_factor[static_cast<size_t>(index)]
				= static_cast<uint64_t>(left);
			result.right_factor[static_cast<size_t>(index)]
				= static_cast<uint64_t>(right);
		}
	}
	return result;
}
