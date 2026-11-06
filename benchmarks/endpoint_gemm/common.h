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
#include "prefix_suffix.h"
#include <array>
#include <type_traits>

#define ENDPOINT_API extern "C" __declspec(dllexport)

namespace endpoint {
using prefix_suffix::mul;
using prefix_suffix::add;
struct Shape {
    uint64_t d, m, n, batch, block, chunks, width, lower, total;
    std::array<uint64_t, 33> power{}, offset{};
    Shape(uint64_t batch_, uint64_t d_, uint64_t points, uint64_t m_, uint64_t block_)
        : d(d_), m(m_), n(points ? points - 1 : 0), batch(batch_), block(block_) {
        if (!d || m < 2 || m > 30 || !block || block > INT_MAX)
            throw std::invalid_argument("invalid endpoint dimensions");
        power[0] = 1;
        for (uint64_t k = 0; k <= m; ++k) {
            offset[k + 1] = add(offset[k], power[k]);
            if (k < m) power[k + 1] = mul(power[k], d);
        }
        chunks = std::max<uint64_t>(1, (n + block - 1) / block);
        width = power[m - 1]; lower = offset[m]; total = offset[m + 1];
        if (d > INT_MAX || width > INT_MAX) throw std::overflow_error("GEMM dimension exceeds int32");
        mul(mul(batch, chunks), total);
    }
};
inline thread_local std::string error;
}
