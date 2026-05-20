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

#include <gtest/gtest.h>
#include "cpsig.h"
#include "cp_sig_combine.h"
#include "cp_path.h"
#include "cp_signature.h"
#include "cp_sig_kernel.h"
#include "sparse.h"
#include "words.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <span>
#include <cmath>
#include <type_traits>

#define SINGLE_EPSILON 1e-4
#define DOUBLE_EPSILON 1e-10
#define EPSILON (std::is_same_v<T, float> ? SINGLE_EPSILON : DOUBLE_EPSILON)

inline double dot_product(double* a, double* b, uint64_t n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += *(a + i) * *(b + i);
    }
    return res;
}

inline void gram_(
    double* path1,
    double* path2,
    double* out,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2
) {
    double* out_ptr = out;

    uint64_t flat_path1_length = length1 * dimension;
    uint64_t flat_path2_length = length2 * dimension;

    double* path1_start = path1;
    double* path1_end = path1 + flat_path1_length;

    double* path2_start = path2;
    double* path2_end = path2 + flat_path2_length;

    for (uint64_t b = 0; b < batch_size; ++b) {

        for (double* path1_ptr = path1_start; path1_ptr < path1_end - dimension; path1_ptr += dimension) {
            for (double* path2_ptr = path2_start; path2_ptr < path2_end - dimension; path2_ptr += dimension) {
                *(out_ptr++) = dot_product(path1_ptr + dimension, path2_ptr + dimension, dimension)
                    - dot_product(path1_ptr + dimension, path2_ptr, dimension)
                    - dot_product(path1_ptr, path2_ptr + dimension, dimension)
                    + dot_product(path1_ptr, path2_ptr, dimension);
            }
        }

        path1_start += flat_path1_length;
        path1_end += flat_path1_length;
        path2_start += flat_path2_length;
        path2_end += flat_path2_length;
    }
}


inline std::vector<float> int_test_data(uint64_t dimension, uint64_t length) {
    std::vector<float> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(static_cast<float>(i));
    }
    return data;
}

template<typename FN, std::floating_point T, typename... Args>
void check_result(FN f, std::vector<T>& path, std::vector<T>& true_, Args... args) {
    std::vector<T> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    if constexpr (std::is_invocable_v<FN, T*, T*, Args...>) {
        f(path.data(), out.data(), args...);
    }
    else {
        f(path.data(), out.data(), args..., static_cast<const T*>(nullptr), static_cast<uint64_t>(0));
    }

    for (uint64_t i = 0; i < true_.size(); ++i)
        EXPECT_LT(abs(true_[i] - out[i]), EPSILON);

    EXPECT_LT(abs( - 1. - out[true_.size()]), EPSILON);
}

template<typename FN, std::floating_point T, typename... Args>
void check_result_2(FN f, std::vector<T>& path1, std::vector<T>& path2, std::vector<T>& true_, Args... args) {
    std::vector<T> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    f(path1.data(), path2.data(), out.data(), args...);

    for (uint64_t i = 0; i < true_.size(); ++i)
        EXPECT_LT(abs(true_[i] - out[i]), EPSILON);

    EXPECT_LT(abs(-1. - out[true_.size()]), EPSILON);
}

inline void check_result_words(std::vector<word> a, std::vector<word> b) {
    ASSERT_EQ(a.size(), b.size());
    for (uint64_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i].size(), b[i].size());
        for (uint64_t j = 0; j < a[i].size(); ++j) {
            ASSERT_EQ(a[i][j], b[i][j]);
        }
    }
}
