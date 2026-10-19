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

#include <gtest/gtest.h>
#include "cpsig.h"
#include "pascal_coefficient.h"
#include "pascal_selection.h"
#include <cmath>
#include <vector>

TEST(PascalCoefficient, AllPrefixAdjoints) {
    const uint64_t word[] = {0, 0, 0, 0}, degree = 4;
    for (const uint64_t length : {2, 65, 66, 131}) {
        std::vector<double> path(length), grad(length);
        for (uint64_t i = 0; i < length; ++i) path[i] = double(i) / double(length - 1);
        for (int method : {0, 1, 2}) {
            double seeds[] = {1, 2, 3, 4};
            double result[4];
            ASSERT_EQ(sig_coef_d(path.data(), result, word, 1, &degree, 1, 1, length,
                false, false, 1., true, 1, method), 0);
            double fact = 1;
            for (int j = 0; j < 4; ++j) {
                fact *= j + 1;
                EXPECT_NEAR(result[j], 1. / fact, 1e-12);
            }
            ASSERT_EQ(sig_coef_backprop_d(path.data(), grad.data(), result, seeds, word, 1, &degree,
                1, 1, length, false, false, 1., 1, method), 0);
            EXPECT_NEAR(grad.front(), -(1. + 2. + 1.5 + 4. / 6.), 1e-10);
            EXPECT_NEAR(grad.back(), -grad.front(), 1e-10);
            for (uint64_t i = 1; i + 1 < length; ++i) EXPECT_NEAR(grad[i], 0, 1e-10);
        }
    }
}

TEST(PascalCoefficient, IdentityAndCheckedSizes) {
    const uint64_t degrees[] = {0, 1}, word = 0;
    float path = 0, out[2];
    ASSERT_EQ(sig_coef_f(&path, out, &word, 2, degrees, 1, 1, 1,
        false, false, 1.f, false, 1, 1), 0);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 0);
    for (uint64_t length : {0, 1}) {
        ASSERT_EQ(sig_coef_f(&path, out, &word, 2, degrees, 1, 1, length,
            false, false, 1.f, true, 1), 0);
        EXPECT_EQ(out[0], 1);
        EXPECT_EQ(out[1], 0);
    }
    EXPECT_THROW(pascal_coefficient::mul(UINT64_MAX, 2), std::overflow_error);
    for (int method : {-1, 3, 4, 5, 6}) {
        EXPECT_THROW(pascal_coefficient::variant(method, 4), std::invalid_argument);
        EXPECT_NE(sig_coef_f(&path, out, &word, 2, degrees, 1, 1, 1,
            false, false, 1.f, true, 1, method), 0);
    }
    const pascal_coefficient::Checkpoints plan(4096, 16);
    EXPECT_EQ(plan.block, 64);
    EXPECT_LT(plan.count, 4096 * 17);
}

TEST(PascalCoefficient, GuardedTinyLink) {
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double path[] = {0, 0, 1e300, 0, 1e300, 3 * tiny};
    const uint64_t word[] = {0, 1}, degree = 2;
    double result[2];
    ASSERT_EQ(sig_coef_d(path, result, word, 1, &degree, 1, 2, 3,
        false, false, 1., true, 1, 1), 0);
    EXPECT_NEAR(result[1] / (1e300 * (3 * tiny)), 1., 1e-14);
}

TEST(PascalCoefficient, SimdScratchAlignment) {
    const uint64_t word[] = {0, 1, 1, 1, 0, 1}, degrees[] = {0, 2, 2, 2};
    for (uint64_t batch : {6, 8, 9}) {
        std::vector<double> path(batch * 16, 0), result(batch * 7), seeds(batch * 7, 1), gradient(path.size());
        for (int method : {1, 2}) {
            for (int repeat = 0; repeat < 20; ++repeat) {
                ASSERT_EQ(sig_coef_d(path.data(), result.data(), word, 4, degrees, batch, 2, 8,
                    false, false, 1., true, 1, method), 0);
                ASSERT_EQ(sig_coef_backprop_d(path.data(), gradient.data(), result.data(), seeds.data(),
                    word, 4, degrees, batch, 2, 8, false, false, 1., 1, method), 0);
                for (uint64_t b = 0; b < batch; ++b) {
                    EXPECT_EQ(result[b * 7], 1);
                    for (uint64_t j = 1; j < 7; ++j) EXPECT_EQ(result[b * 7 + j], 0);
                    EXPECT_EQ(gradient[b * 16], -2);
                    EXPECT_EQ(gradient[b * 16 + 1], -1);
                    EXPECT_EQ(gradient[b * 16 + 14], 2);
                    EXPECT_EQ(gradient[b * 16 + 15], 1);
                }
            }
        }
    }
}

TEST(PascalCoefficient, CalibratedSelection) {
    std::vector<uint64_t> indices(96, 0), degrees(8, 12);
    const pascal_coefficient::Words words(indices.data(), degrees.data(), 8, 8, true);
    EXPECT_TRUE(pascal_selection::cpu_region<float>(words, 32, 8, 256, 1, false, false, true));
    EXPECT_FALSE(pascal_selection::cpu_region<float>(words, 32, 8, 256, 1, false, false, false));
    EXPECT_FALSE(pascal_selection::cpu_region<double>(words, 32, 8, 256, 1, false, false, true));
    EXPECT_FALSE(pascal_selection::cpu_region<float>(words, 32, 8, 256, 20, false, false, true));
    EXPECT_FALSE(pascal_selection::cpu_region<float>(words, 32, 8, 256, 1, true, false, true));
    EXPECT_FALSE(pascal_selection::cuda_region<float>(words, 32, 8, 256, 256, false, false, true));
    EXPECT_FALSE(pascal_selection::cuda_region<double>(words, 32, 8, 256, 256, false, false, true));
    EXPECT_FALSE(pascal_selection::cuda_region<float>(words, 32, 8, 256, 256, false, false, false));
    EXPECT_FALSE(pascal_selection::cuda_region<float>(words, 512, 8, 256, 4096, false, false, true));
    degrees[0] = 11;
    const pascal_coefficient::Words mixed(indices.data(), degrees.data(), 8, 8, true);
    EXPECT_FALSE(pascal_selection::cpu_region<float>(mixed, 32, 8, 256, 1, false, false, true));
    EXPECT_FALSE(pascal_selection::cuda_region<float>(mixed, 32, 8, 256, 256, false, false, true));
}
