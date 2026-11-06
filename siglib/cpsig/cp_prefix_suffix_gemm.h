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
#include <cblas.h>
#include <type_traits>

namespace cpsig {
static_assert(sizeof(blasint) == sizeof(int64_t), "prefix-suffix requires ILP64 OpenBLAS");

template<typename T>
inline void factor_gemm(prefix_suffix::Matrix<T> a, prefix_suffix::Matrix<T> b,
    prefix_suffix::Matrix<T> c, bool first) {
    const auto ta = a.rs == 1 ? CblasTrans : CblasNoTrans;
    const auto tb = b.rs == 1 ? CblasTrans : CblasNoTrans;
    if constexpr (std::is_same_v<T, float>)
        cblas_sgemm(CblasRowMajor, ta, tb, a.rows, b.cols, a.cols, 1, a.data, a.leading_dimension(),
            b.data, b.leading_dimension(), first ? 0 : 1, c.data, c.rs);
    else
        cblas_dgemm(CblasRowMajor, ta, tb, a.rows, b.cols, a.cols, 1, a.data, a.leading_dimension(),
            b.data, b.leading_dimension(), first ? 0 : 1, c.data, c.rs);
}
} // namespace cpsig
