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
#include <vector>

#ifdef __CUDACC__
#define SIGLASSO_HD __host__ __device__
#else
#define SIGLASSO_HD
#endif

namespace siglasso {

SIGLASSO_HD inline size_t envelope_offset(size_t q, size_t a, size_t j, size_t i, size_t n, size_t segments) {
    return (q * (q + 1) / 2 + a) * segments * n + j * n + i;
}

SIGLASSO_HD inline void prepare_sample(const double* increments, double* variation, double* after, double* envelopes,
    size_t i, size_t n, size_t segments, size_t dimension, size_t degree) {
    double tail_variation = 0;
    for (size_t j = segments; j-- > 0;) {
        double v = 0;
        for (size_t a = 0; a < dimension; ++a) v = fmax(v, fabs(increments[(j * dimension + a) * n + i]));
        variation[j * n + i] = v;
        after[j * n + i] = tail_variation;
        tail_variation += v;
        envelopes[j * n + i] = 1;
    }
    if (degree > 1) {
        for (size_t a = 0; a < dimension; ++a) {
            double tail = 0;
            for (size_t j = segments; j-- > 0;) {
                auto right = envelope_offset(1, 1, j, i, n, segments);
                envelopes[right] = fmax(envelopes[right], fabs(tail));
                tail += increments[(j * dimension + a) * n + i];
                auto left = envelope_offset(1, 0, j, i, n, segments);
                envelopes[left] = fmax(envelopes[left], fabs(tail));
            }
        }
    }
    for (size_t q = 2; q < degree; ++q) {
        double carry = 0;
        for (size_t j = segments; j-- > 0;) {
            envelopes[envelope_offset(q, q, j, i, n, segments)] = carry;
            for (size_t a = q; a-- > 0;) {
                carry += variation[j * n + i] * envelopes[envelope_offset(q - 1, a, j, i, n, segments)] / q;
                envelopes[envelope_offset(q, a, j, i, n, segments)] = carry;
            }
        }
    }
}

SIGLASSO_HD inline double binomial_log(size_t n, size_t k) {
    return lgamma(static_cast<double>(n + 1)) - lgamma(static_cast<double>(k + 1)) - lgamma(static_cast<double>(n - k + 1));
}

inline std::vector<double> integration_coefficients(size_t k, size_t levels) {
    std::vector<double> out(levels * (levels + 1) * (k + 1));
    for (size_t q = 1; q <= levels; ++q) for (size_t b = 0; b < q; ++b) for (size_t a = 0; a <= k; ++a) {
        const size_t offset = 2 * ((q * (q - 1) / 2 + b) * (k + 1) + a);
        out[offset] = exp(binomial_log(k, a) + binomial_log(q - 1, b) - binomial_log(k + q - 1, a + b)) / (k + q);
        out[offset + 1] = exp(lgamma(static_cast<double>(k + 1)) + lgamma(static_cast<double>(k - a + b + 1))
            - lgamma(static_cast<double>(k - a + 1)) - lgamma(static_cast<double>(k + b + 2))
            - lgamma(static_cast<double>(b + 1)) - lgamma(static_cast<double>(q - b)));
    }
    return out;
}

SIGLASSO_HD inline double factor_value(const double* variation, const double* after, const double* envelopes, const double* constants,
    size_t k, size_t q, size_t a, size_t j, size_t i, size_t n, size_t segments) {
    const double step = variation[j * n + i], tail = after[j * n + i];
    double integral = 0, total_variation = 0;
    for (size_t b = 0; b < q; ++b) {
        const size_t offset = 2 * ((q * (q - 1) / 2 + b) * (k + 1) + a);
        const double beta = constants[offset];
        integral += step * envelopes[envelope_offset(q - 1, b, j, i, n, segments)] * beta;
        const double coefficient = constants[offset + 1];
        total_variation += pow(step, static_cast<double>(b + 1)) * pow(tail, static_cast<double>(q - 1 - b)) * coefficient;
    }
    return fmin(integral, total_variation);
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
