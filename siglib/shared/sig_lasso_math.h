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
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef __CUDACC__
#define SIGLASSO_HD __host__ __device__
#else
#define SIGLASSO_HD
#endif

namespace siglasso {

struct TerminalCache {
    double* data = nullptr;
    size_t capacity = 0, degree = 0;
    double generation = 1;
    SIGLASSO_HD size_t slot(const uint64_t* prefix, size_t k, uint64_t a, uint64_t b, size_t q) const {
        uint64_t hash = 0xcbf29ce484222325ULL;
        for (size_t j = 0; j < k; ++j) hash = (hash ^ prefix[j]) * 0x100000001b3ULL;
        hash = (hash ^ a) * 0x100000001b3ULL;
        if (q == 2) hash = (hash ^ b) * 0x100000001b3ULL;
        hash ^= hash >> 32;
        return hash & (capacity - 1);
    }
    SIGLASSO_HD bool get(size_t index, const uint64_t* prefix, size_t k, uint64_t a, uint64_t b, size_t q, double& value) const {
        const double* entry = data + index * (degree + 3);
        if (entry[0] != k + q || entry[1] != generation) return false;
        for (size_t j = 0; j < k + q; ++j) {
            uint64_t letter;
            memcpy(&letter, entry + 2 + j, sizeof(letter));
            if (letter != (j < k ? prefix[j] : j == k ? a : b)) return false;
        }
        value = entry[degree + 2];
        return true;
    }
    SIGLASSO_HD void put(size_t index, const uint64_t* prefix, size_t k, uint64_t a, uint64_t b, size_t q, double value) const {
        double* entry = data + index * (degree + 3);
        entry[0] = k + q; entry[1] = generation;
        for (size_t j = 0; j < k + q; ++j) {
            const uint64_t letter = j < k ? prefix[j] : j == k ? a : b;
            memcpy(entry + 2 + j, &letter, sizeof(letter));
        }
        entry[degree + 2] = value;
    }
};

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

SIGLASSO_HD inline size_t integral_scratch_size(size_t degree) {
    return 57 * (degree + 1) + 162 + degree;
}

// A negative moment requests expansion when subdivision cannot resolve the sign.
// Terminal blocks can use the Bernstein absolute-coefficient envelope before signed checks.
SIGLASSO_HD inline void absolute_moments(const double* coefficients, const double* factors,
    double step, double tail, size_t k, size_t levels, size_t j, size_t i, size_t n, size_t segments,
    double* scratch, double* out, bool envelope = false) {
    const size_t width = k + 1, nl = n * segments;
    bool positive = false, negative = false;
    for (size_t a = 0; a <= k; ++a) {
        const double value = coefficients[(j * width + a) * n + i];
        const double c = envelope ? fabs(value) : value;
        scratch[a] = c; positive |= c > 0; negative |= c < 0;
    }
    if (!positive || !negative || !step) {
        for (size_t q = 0; q < levels; ++q) {
            double value = 0;
            for (size_t a = 0; a <= k; ++a)
                value += scratch[a] * factors[(q * width + a) * nl + j * n + i];
            out[q * nl] = fabs(value);
        }
        return;
    }
    if (k == 1) {
        const double fraction = fabs(scratch[0]) / (fabs(scratch[0]) + fabs(scratch[1]));
        const double middle = fabs((1 - fraction) * scratch[0] + fraction * scratch[1]);
        const double left_step = step * fraction, right_step = step * (1 - fraction), left_tail = tail + right_step;
        const double* constants = factors + levels * width * nl;
        for (size_t q = 0; q < levels; ++q) {
            out[q * nl] = (fabs(scratch[0]) * factor_value(&left_step, &left_tail, constants, k, q + 1, 0, 0, 0, 1)
                + middle * factor_value(&left_step, &left_tail, constants, k, q + 1, 1, 0, 0, 1)
                + middle * factor_value(&right_step, &tail, constants, k, q + 1, 0, 0, 0, 1)
                + fabs(scratch[1]) * factor_value(&right_step, &tail, constants, k, q + 1, 1, 0, 0, 1)) / n;
        }
        return;
    }
    double* left = scratch + 54 * width;
    double* right = left + 54;
    double* depth = right + 54;
    double* work = depth + 54;
    double* split_left = work + width;
    double* split_right = split_left + width;
    double* original = split_right + width;
    const double* constants = factors + levels * width * nl;
    for (size_t q = 0; q < levels; ++q) {
        original[q] = 0; out[q * nl] = 0;
        for (size_t a = 0; a <= k; ++a)
            original[q] += fabs(scratch[a]) * factors[(q * width + a) * nl + j * n + i];
    }
    left[0] = 0; right[0] = 1; depth[0] = 0;
    size_t count = 1, visits = 0;
    while (count) {
        const size_t slot = --count;
        double* c = scratch + slot * width;
        const double low = left[slot], high = right[slot], level = depth[slot];
        positive = negative = false;
        for (size_t a = 0; a <= k; ++a) { positive |= c[a] > 0; negative |= c[a] < 0; }
        const double local_step = step * (high - low), local_tail = tail + step * (1 - high);
        bool resolved = !positive || !negative;
        if (!resolved) {
            resolved = true;
            for (size_t q = 0; q < levels; ++q) {
                double upper = 0, signed_value = 0;
                for (size_t a = 0; a <= k; ++a) {
                    const double f = factor_value(&local_step, &local_tail, constants, k, q + 1, a, 0, 0, 1) / n;
                    upper += fabs(c[a]) * f; signed_value += c[a] * f;
                }
                if (upper - fabs(signed_value) > 128 * 2.2204460492503131e-16 * original[q] * (high - low)) resolved = false;
            }
        }
        if (resolved) {
            for (size_t q = 0; q < levels; ++q) {
                double value = 0;
                for (size_t a = 0; a <= k; ++a)
                    value += fabs(c[a]) * factor_value(&local_step, &local_tail, constants, k, q + 1, a, 0, 0, 1) / n;
                out[q * nl] += value;
            }
            continue;
        }
        const double middle = (low + high) * .5;
        if (level >= 52 || middle == low || middle == high || ++visits > 256 * width) {
            for (size_t q = 0; q < levels; ++q) out[q * nl] = -1;
            return;
        }
        for (size_t a = 0; a <= k; ++a) work[a] = c[a];
        split_left[0] = work[0]; split_right[k] = work[k];
        for (size_t h = 1; h <= k; ++h) {
            for (size_t a = 0; a <= k - h; ++a) work[a] = .5 * work[a] + .5 * work[a + 1];
            split_left[h] = work[0]; split_right[k - h] = work[k - h];
        }
        for (size_t a = 0; a <= k; ++a) { c[a] = split_right[a]; c[width + a] = split_left[a]; }
        left[slot] = middle; right[slot] = high; depth[slot] = level + 1;
        left[slot + 1] = low; right[slot + 1] = middle; depth[slot + 1] = level + 1;
        count += 2;
    }
}

SIGLASSO_HD inline void trapezoidal_moments(double start, double end, double step, double tail,
    double weight, size_t k, size_t levels, size_t stride, double* out, const double* reciprocals = nullptr) {
    double left = fabs(start) * step * weight * .5, right = fabs(end) * step * weight * .5;
    for (size_t q = 0; q < levels; ++q) {
        out[q * stride] = left + right;
        const double ratio = reciprocals ? (k + q + 2) * reciprocals[q + 1] : static_cast<double>(k + q + 2) / (q + 1);
        left *= (tail + step) * ratio; right *= tail * ratio;
    }
}

// Correlation-only terminal contraction; no descendant feature column is stored.
SIGLASSO_HD inline double terminal_sample(const double* integrals, const double* linear_integrals, const double* increments,
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
            const double linear = linear_integrals[j * n + i];
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
