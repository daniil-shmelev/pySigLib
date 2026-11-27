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
#include <cmath>
#include <cstddef>
#include <limits>

namespace siglasso {

inline void prepare_sample(const double* increments, double* variation, double* after,
    size_t i, size_t n, size_t segments, size_t dimension) {
    double tail_variation = 0;
    for (size_t j = segments; j-- > 0;) {
        double v = 0;
        for (size_t a = 0; a < dimension; ++a) v = fmax(v, fabs(increments[(j * dimension + a) * n + i]));
        variation[j * n + i] = v;
        after[j * n + i] = tail_variation;
        tail_variation += v;
    }
}

inline double terminal_roundoff(size_t k, size_t n, size_t segments) {
    const double operations = 32.0 * (k + 2) * (k + 2) * (segments + 1) + 8.0 * n;
    const double error = operations * 2.2204460492503131e-16;
    return error < .25 ? error / (1 - error) : std::numeric_limits<double>::infinity();
}

inline double energy_sample(const double* integrals, const double* increments,
    size_t low, size_t high, size_t i, size_t n, size_t segments, size_t dimension, const size_t* order = nullptr) {
    double squared = 0, magnitude = 0;
    for (size_t a = low; a < high; ++a) {
        const size_t c = order ? order[a] : a;
        double value = 0, absolute = 0;
        for (size_t j = 0; j < segments; ++j) {
            const double term = integrals[j * n + i] * increments[(j * dimension + c) * n + i];
            value += term; absolute += fabs(term);
        }
        squared += value * value; magnitude += absolute * absolute;
    }
    return sqrt(fmax(0.0, squared) + 1e-11 * magnitude + 1e-300);
}

} // namespace siglasso
