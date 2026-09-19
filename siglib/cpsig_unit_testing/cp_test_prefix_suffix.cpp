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
#include "cp_prefix_suffix_detail.h"
#include "cpsig.h"
#include "prefix_suffix.h"

TEST(PrefixSuffix, CheckedSizes) {
    EXPECT_THROW(prefix_suffix::Layout(2, 64), std::overflow_error);
    EXPECT_THROW(prefix_suffix::mul(UINT64_MAX, 2), std::overflow_error);
    EXPECT_THROW(prefix_suffix::add(UINT64_MAX, 1), std::overflow_error);
    EXPECT_THROW(prefix_suffix::Layout(0, 3), std::invalid_argument);
    EXPECT_THROW(prefix_suffix::Layout(1, UINT64_MAX), std::overflow_error);
    EXPECT_EQ(prefix_suffix::Layout(1, 65536).offset.back(), 65537);
}

TEST(PrefixSuffix, NativeForward) {
    const double path[] = {0, 0, 0.2, -0.1, 0.1, 0.3, -0.2, 0.1};
    std::vector<double> expected(126), result(126);
    ASSERT_EQ(signature_d(path, expected.data(), 1, 2, 4, 6, false, false, 1., true, false, 1), 0);
    ASSERT_EQ(signature_prefix_suffix_d(path, result.data(), 1, 2, 4, 6, false, false, 1., false, 2), 0);
    for (size_t i = 0; i < result.size(); ++i) EXPECT_NEAR(result[i], expected[i], 1e-12);
    EXPECT_NE(signature_prefix_suffix_d(path, result.data(), 1, 2, 4, 6, false, false, 1., false, 0), 0);
}

TEST(PrefixSuffix, NativeDirectLowDepth) {
    for (uint64_t d : {2, 3, 4, 8}) for (uint64_t degree : {2, 3}) {
        const uint64_t length = 9, output_size = prefix_suffix::Layout(d, degree).offset[degree + 1];
        std::vector<double> path(length * d), expected(output_size), actual(output_size);
        for (uint64_t i = 0; i < length; ++i)
            for (uint64_t x = 0; x < d; ++x)
                path[i * d + x] = double(int((i * 5 + x * 3) % 17) - 8) / 32;
        ASSERT_EQ(signature_d(path.data(), expected.data(), 1, d, length, degree, false, false, 1., true, true, 1), 0);
        ASSERT_EQ(signature_prefix_suffix_d(path.data(), actual.data(), 1, d, length, degree,
            false, false, 1., true, 3), 0);
        for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-12);
    }
}

TEST(PrefixSuffix, GemmLargeStrides) {
    using Matrix = prefix_suffix::Matrix<double>;
    struct Operations {
        uint64_t scratch_stride = 0, calls = 0;
        void clear(Matrix c) { c.data[0] = 0; }
        void pack(Matrix a, Matrix c) { c.data[0] = a.data[0]; }
        void accumulate(Matrix a, Matrix c, double sign) { c.data[0] += sign * a.data[0]; }
        void gemm(Matrix a, Matrix b, Matrix c, double beta) {
            EXPECT_EQ(a.rows, 1); EXPECT_EQ(a.cols, 1);
            EXPECT_EQ(b.rows, 1); EXPECT_EQ(b.cols, 1);
            EXPECT_LE(a.rs, INT_MAX); EXPECT_LE(b.rs, INT_MAX); EXPECT_LE(c.rs, INT_MAX);
            c.data[0] = a.data[0] * b.data[0] + beta * c.data[0];
            ++calls;
        }
    } ops;
    double a[] = {1, 2, 3}, b[] = {4, 5, 6}, c = -1, scratch[3]{};
    const uint64_t stride = uint64_t(INT_MAX) + 1;
    for (bool first : {true, false}) {
        prefix_suffix::product(ops, Matrix{a, 1, 3, stride, 1}, Matrix{b, 3, 1, 1, stride},
            Matrix{&c, 1, 1, stride, 1}, 4096, scratch, first);
        EXPECT_DOUBLE_EQ(c, first ? 32 : 64);
    }
    EXPECT_EQ(ops.calls, 6);
}

TEST(PrefixSuffix, RightFactorDefiningSum) {
    for (uint64_t d : {2, 3}) for (uint64_t b = 1; b <= 5; ++b) {
        prefix_suffix::Layout l(d, 2 * b + 1);
        std::vector<double> suffix(l.offset[b + 1]), v(d), result(l.power[b]);
        for (size_t x = 0; x < suffix.size(); ++x) suffix[x] = double(int(x % 11) - 5) / 16;
        suffix[0] = 1;
        for (size_t x = 0; x < d; ++x) v[x] = double(int(x) - 1) / 4;
        for (uint64_t j = 0; j <= b; ++j) {
            prefix_suffix::right_factor(result.data(), suffix.data(), v.data(), j, b, l);
            for (uint64_t word = 0; word < l.power[b]; ++word) {
                long double expected = 0;
                for (uint64_t r = 0; r <= b; ++r) {
                    uint64_t head = word / l.power[b - r];
                    long double value = suffix[l.offset[b - r] + word % l.power[b - r]];
                    for (uint64_t f = 2; f <= j + r + 1; ++f) value /= f;
                    for (uint64_t c = 0; c < r; ++c) { value *= v[head % d]; head /= d; }
                    expected += value;
                }
                EXPECT_NEAR(result[word], double(expected), 1e-14);
            }
        }
    }
}

TEST(PrefixSuffix, RightFactorExtension) {
    for (uint64_t d : {2, 3, 5}) for (uint64_t b = 0; b <= 4; ++b) {
        prefix_suffix::Layout l(d, 2 * b + 2);
        std::vector<double> suffix(l.offset[b + 2]), v(d), previous(l.power[b]);
        std::vector<double> expected(l.power[b + 1]), actual(l.power[b + 1]);
        for (size_t x = 0; x < suffix.size(); ++x) suffix[x] = double(int(x % 13) - 6) / 16;
        suffix[0] = 1;
        for (size_t x = 0; x < d; ++x) v[x] = double(int(x) - 2) / 8;
        for (uint64_t j = 0; j <= b + 2; ++j) {
            prefix_suffix::right_factor(previous.data(), suffix.data(), v.data(), j + 1, b, l);
            prefix_suffix::right_factor(expected.data(), suffix.data(), v.data(), j, b + 1, l);
            prefix_suffix::extend_right_factor(actual.data(), previous.data(), suffix.data(), v.data(), j, b, l);
            for (size_t x = 0; x < actual.size(); ++x) EXPECT_NEAR(actual[x], expected[x], 1e-14);
        }
    }
}

TEST(PrefixSuffix, BackwardUpdate) {
    for (uint64_t d : {2, 3, 4, 5, 8}) for (uint64_t degree = 1; degree <= 5; ++degree) {
        prefix_suffix::Layout l(d, 2 * degree);
        std::vector<double> expected(l.offset[degree + 1]), actual(l.offset[degree + 1]);
        std::vector<double> v(d), expected_work(l.power[degree]), actual_work(l.power[degree]);
        expected[0] = actual[0] = 1;
        for (size_t x = 1; x < expected.size(); ++x)
            expected[x] = actual[x] = double(int(x % 17) - 8) / 32;
        for (size_t x = 0; x < v.size(); ++x) v[x] = double(int(x) - 2) / 8;
        for (uint64_t k = degree; k > 1; --k) {
            for (uint64_t x = 0; x < d; ++x) expected_work[x] = v[x] / double(k);
            for (uint64_t level = 1; level < k; ++level) {
                const uint64_t previous = l.power[level], size = l.power[level + 1];
                for (uint64_t x = size; x-- > 0;) {
                    const double value = (expected[l.offset[level] + x % previous] + expected_work[x % previous])
                        * v[x / previous] / double(k - level);
                    if (level + 1 == k) expected[l.offset[k] + x] += value;
                    else expected_work[x] = value;
                }
            }
        }
        for (uint64_t x = 0; x < d; ++x) expected[1 + x] += v[x];
        prefix_suffix::cpu_detail::prepend(actual.data(), v.data(), l, actual_work.data());
        for (size_t x = 0; x < actual.size(); ++x) EXPECT_NEAR(actual[x], expected[x], 1e-14);
    }
}
