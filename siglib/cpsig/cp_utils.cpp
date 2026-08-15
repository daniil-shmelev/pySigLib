/* Copyright 2025 Daniil Shmelev
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

#include "cppch.h"
#include "cpsig.h"
#include "cp_utils.h"
#include "../shared/branched_trees.h"
#include "macros.h"

uint64_t power(uint64_t base, uint64_t exp) noexcept {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            if (result != 0 && base > UINT64_MAX / result)
                return 0;
            result *= base;
        }
        exp >>= 1;
        if (exp > 0) {
            if (base != 0 && base > UINT64_MAX / base)
                return 0;
            base *= base;
        }
    }
    return result;
}

void populate_level_index(uint64_t* level_index, uint64_t dimension, uint64_t degree) {
    level_index[0] = 0;
    for (uint64_t i = 1; i < degree; i++) {
        if (dimension != 0 && level_index[i - 1] > UINT64_MAX / dimension)
            throw std::overflow_error("populate_level_index: level_index overflow");
        const uint64_t mul = level_index[i - 1] * dimension;
        if (mul > UINT64_MAX - 1)
            throw std::overflow_error("populate_level_index: level_index overflow");
        level_index[i] = mul + 1;
    }
}

static std::vector<std::vector<uint64_t>> compute_divisors(uint64_t N) {
    std::vector<std::vector<uint64_t>> divisors(N + 1);

    for (uint64_t d = 1; d <= N; ++d) {
        for (uint64_t multiple = d; multiple <= N; multiple += d) {
            divisors[multiple].push_back(d);
        }
    }
    return divisors;
}

static bool is_prime(uint64_t n)
{
    if (n < 2)
        return false;
    for (uint64_t i = 2; i * i <= n; i++)
        if (n % i == 0)
            return false;
    return true;
}

static int64_t mobius(uint64_t N)
{
    if (N == 1)
        return 1;

    uint64_t p = 0;
    for (uint64_t i = 1; i <= N; i++) {
        if (N % i == 0 && is_prime(i)) {
            if (N % (i * i) == 0)
                return 0;
            else
                p++;
        }
    }

    return (p % 2 != 0) ? -1 : 1;
}

extern "C" CPSIG_API uint64_t sig_length(uint64_t dimension, uint64_t degree) noexcept {
    if (dimension == 0) {
        return 1;
    }
    else if (dimension == 1) {
        return degree + 1;
    }
    else {
        const auto pwr = power(dimension, degree + 1);
        if (pwr)
            return (pwr - 1) / (dimension - 1);
        else
            return 0;
    }
}

extern "C" CPSIG_API uint64_t log_sig_length(uint64_t dimension, uint64_t degree) noexcept {
    if (!dimension || !degree) {
        return 0;
    }
    std::vector<std::vector<uint64_t>> divisors = compute_divisors(degree);
    uint64_t result = 0;
    for (uint64_t i = 1; i <= degree; ++i) {
        int64_t i_sum = 0;
        for (uint64_t d : divisors[i]) {
            uint64_t p = power(dimension, d);
            if (!p || p > static_cast<uint64_t>(INT64_MAX))
                return 0;

            int64_t m = mobius(i / d);

            int64_t term = 0;
            if (m == 1)
                term = static_cast<int64_t>(p);
            else if (m == -1)
                term = -static_cast<int64_t>(p);

            if ((term > 0 && i_sum > INT64_MAX - term) ||
                (term < 0 && i_sum < INT64_MIN - term))
                return 0;

            i_sum += term;
        }
        result += i_sum / i;
    }
    return result;
}

uint64_t branched_sig_length_(uint64_t dimension, uint64_t max_nodes, bool planar) {
    return compute_branched_sig_length(dimension, max_nodes, planar);
}

uint64_t branched_log_sig_length_(uint64_t dimension, uint64_t max_nodes, bool planar) {
    return compute_branched_log_sig_length(dimension, max_nodes, planar);
}

extern "C" {
    CPSIG_API uint64_t branched_sig_length(uint64_t dimension, uint64_t max_nodes, bool planar) noexcept {
        try {
            return branched_sig_length_(dimension, max_nodes, planar);
        }
        catch (...) {
            return 0;
        }
    }

    CPSIG_API uint64_t branched_log_sig_length(uint64_t dimension, uint64_t max_nodes, bool planar) noexcept {
        try {
            return branched_log_sig_length_(dimension, max_nodes, planar);
        }
        catch (...) {
            return 0;
        }
    }
}
