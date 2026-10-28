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
#include <vector>

#ifdef __CUDACC__
#define SIGLASSO_HD __host__ __device__
#else
#define SIGLASSO_HD
#endif

namespace siglasso {

SIGLASSO_HD inline void prepare_sample(const double* increments, double* variation, double* after,
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

// Expand the remaining-variation power and integrate against each Bernstein basis polynomial.
inline std::vector<double> integration_coefficients(size_t k, size_t levels, const std::vector<double>& weights) {
    std::vector<double> out(levels * (levels + 1) / 2 * (k + 1));
    double factorial = 1;
    for (size_t q = 1; q <= levels; ++q) {
        if (q > 1) factorial *= q - 1;
        for (size_t a = 0; a <= k; ++a) {
            double value = weights[k + q] / (k + 1) / factorial;
            for (size_t b = 0; b < q; ++b) {
                out[(q * (q - 1) / 2 + b) * (k + 1) + a] = value;
                value *= static_cast<double>(k - a + b + 1) / (k + b + 2) * static_cast<double>(q - 1 - b) / (b + 1);
            }
        }
    }
    return out;
}

// Weighted integral of B_a^k(s) * step * (tail + step * (1-s))^(q-1) / (q-1)!.
SIGLASSO_HD inline double factor_value(const double* variation, const double* after, const double* constants,
    size_t k, size_t q, size_t a, size_t j, size_t i, size_t n) {
    const double step = variation[j * n + i], tail = after[j * n + i];
    if (step == 0) return 0;
    const size_t offset = q * (q - 1) / 2 * (k + 1) + a;
    if (q == 1) return constants[offset] * step;
    double value;
    if (step >= tail) {
        const double ratio = tail / step;
        value = constants[offset];
        for (size_t b = 1; b < q; ++b) value = value * ratio + constants[offset + b * (k + 1)];
    } else {
        const double ratio = step / tail;
        value = constants[offset + (q - 1) * (k + 1)];
        for (size_t b = q - 1; b-- > 0;) value = value * ratio + constants[offset + b * (k + 1)];
    }
    // Scale after Horner evaluation to avoid underflow before degree weighting.
    const double scale = fmax(step, tail);
    for (size_t b = 1; b < q; ++b) value *= scale;
    return value * step;
}

SIGLASSO_HD inline void child_sample(const double* increments, const double* parent, const double* integrals, double* child, double* values,
    size_t letter, size_t k, size_t i, size_t n, size_t segments, size_t dimension) {
    double carry = 0;
    for (size_t j = 0; j < segments; ++j) {
        if (!child) { carry += increments[(j * dimension + letter) * n + i] * integrals[j * n + i]; continue; }
        const double step = increments[(j * dimension + letter) * n + i] / (k + 1);
        if (child) child[(j * (k + 2)) * n + i] = carry;
        for (size_t a = 0; a <= k; ++a) {
            carry += step * parent[(j * (k + 1) + a) * n + i];
            if (child) child[(j * (k + 2) + a + 1) * n + i] = carry;
        }
    }
    values[i] = carry;
}

SIGLASSO_HD inline double moment_value(const double* coefficients, const double* factors, size_t k, size_t q,
    size_t j, size_t i, size_t n, size_t segments) {
    double value = 0;
    for (size_t a = 0; a <= k; ++a)
        value += fabs(coefficients[(j * (k + 1) + a) * n + i]) * factors[((q * (k + 1) + a) * segments + j) * n + i];
    return value;
}

// Correlation-only terminal contraction; no descendant feature column is stored.
SIGLASSO_HD inline double terminal_sample(const double* coefficients, const double* integrals, const double* increments,
    size_t a, size_t b, size_t q, size_t k, size_t i, size_t n, size_t segments, size_t dimension, double& magnitude) {
    double value = 0, correction = 0, tail = 0;
    magnitude = 0;
    for (size_t j = segments; j-- > 0;) {
        const double step = increments[(j * dimension + a) * n + i];
        const double integral = integrals[j * n + i];
        double term = step * integral, absolute = fabs(term);
        if (q == 2) {
            const double other = increments[(j * dimension + b) * n + i];
            tail += other;
            double linear = 0;
            for (size_t h = 0; h <= k; ++h) linear += (h + 1) * coefficients[(j * (k + 1) + h) * n + i];
            linear /= (k + 1) * (k + 2);
            term = step * (integral * tail - linear * other);
            absolute = fabs(step) * (fabs(integral * tail) + fabs(linear * other));
        }
        const double adjusted = term - correction, next = value + adjusted;
        correction = (next - value) - adjusted;
        value = next; magnitude += absolute;
    }
    return value;
}

// Gamma_n padding includes prefix recurrence, suffix sums and correlation reduction.
// The variation majorant also covers cancellation inside computed prefix coefficients.
SIGLASSO_HD inline double terminal_roundoff(size_t k, size_t n, size_t segments) {
    const double operations = 32.0 * (k + 2) * (k + 2) * (segments + 1) + 8.0 * n;
    const double error = operations * 2.2204460492503131e-16;
    return error < .25 ? error / (1 - error) : std::numeric_limits<double>::infinity();
}

SIGLASSO_HD inline double energy_sample(const double* integrals, const double* gram, const double* increments,
    size_t low, size_t high, size_t i, size_t n, size_t segments, size_t dimension) {
    double squared = 0, magnitude = 0;
    if (gram) {
        for (size_t j = 0; j < segments; ++j) for (size_t h = 0; h <= j; ++h) {
            double term = integrals[j * n + i] * gram[(j * segments + h) * n + i] * integrals[h * n + i];
            if (j != h) term *= 2;
            squared += term; magnitude += fabs(term);
        }
    } else {
        for (size_t a = low; a < high; ++a) {
            double value = 0, absolute = 0;
            for (size_t j = 0; j < segments; ++j) {
                const double term = integrals[j * n + i] * increments[(j * dimension + a) * n + i];
                value += term; absolute += fabs(term);
            }
            squared += value * value; magnitude += absolute * absolute;
        }
    }
    return sqrt(fmax(0.0, squared) + 1e-11 * magnitude + 1e-300);
}

} // namespace siglasso
#undef SIGLASSO_HD
