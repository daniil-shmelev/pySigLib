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
#include "pascal_coefficient.h"
#include <cstring>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace pascal_selection {

// Threshold evidence: benchmarks/pascal_short_words.rst.
template<class T>
bool cpu_region(const pascal_coefficient::Words& words, uint64_t batch, uint64_t dimension,
    uint64_t segments, int jobs, bool time_aug, bool lead_lag, bool calibrated) {
    return calibrated && sizeof(T) == 4 && jobs == 1 && !time_aug && !lead_lag
        && batch >= 32 && batch <= 512 && dimension >= 2 && dimension <= 32
        && segments >= 128 && segments <= 4096 && words.count >= 8 && words.count <= 64
        && words.max_degree >= 8 && words.max_degree <= 16
        && words.index_count / words.count == words.max_degree;
}

template<class T>
bool cuda_region(const pascal_coefficient::Words&, uint64_t, uint64_t,
    uint64_t, uint64_t, bool, bool, bool) {
    // No Pascal region is validated against the fully unrolled Horner kernels.
    return false;
}

inline bool cpu_hardware() {
#if defined(_MSC_VER) && _MSC_VER == 1944 && defined(_M_X64) && !defined(__CUDACC__)
    static const bool calibrated = [] {
        int registers[4];
        __cpuid(registers, static_cast<int>(0x80000000u));
        if (static_cast<unsigned int>(registers[0]) < 0x80000004u) return false;
        char brand[49] = {};
        for (int leaf = 0; leaf < 3; ++leaf) {
            __cpuid(registers, static_cast<int>(0x80000002u) + leaf);
            std::memcpy(brand + 16 * leaf, registers, 16);
        }
        return std::strcmp(brand, "13th Gen Intel(R) Core(TM) i7-13700H") == 0;
    }();
    return calibrated;
#else
    return false;
#endif
}

} // namespace pascal_selection
