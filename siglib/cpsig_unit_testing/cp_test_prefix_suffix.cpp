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
#include "cp_prefix_suffix_backprop.h"
#include "cp_prefix_suffix_endpoint_plan.h"
#include "cpsig.h"
#include "prefix_suffix.h"
#include "signature_method.h"
#include <cmath>
#ifndef CPSIG_PREBUILT_TESTS
#include "cp_prefix_suffix_gemm.h"
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#endif

namespace corrected_test {
using namespace prefix_suffix;
#include "prefix_suffix_correction-inl.h"
}

TEST(PrefixSuffix, CheckedSizes) {
    EXPECT_THROW(prefix_suffix::Layout(2, 64), std::overflow_error);
    EXPECT_THROW(prefix_suffix::mul(UINT64_MAX, 2), std::overflow_error);
    EXPECT_THROW(prefix_suffix::add(UINT64_MAX, 1), std::overflow_error);
    EXPECT_THROW(prefix_suffix::Layout(0, 3), std::invalid_argument);
    EXPECT_THROW(prefix_suffix::Layout(1, UINT64_MAX), std::overflow_error);
    EXPECT_EQ(prefix_suffix::Layout(1, 65536).offset.back(), 65537);
}

TEST(PrefixSuffix, CorrectionSelectionAndQuadrature) {
    using corrected_test::Plan;
    for (uint64_t degree = 2; degree <= 20; ++degree) {
        const Plan plan(1, 1, degree, 8, 0, 1, 1, sizeof(double), true);
        for (uint64_t k = 0; k < degree; ++k) {
            double value = 0;
            for (uint64_t j = 0; j < plan.nodes_count; ++j) {
                EXPECT_GT(plan.nodes[j], 0.); EXPECT_LT(plan.nodes[j], 1.);
                EXPECT_GT(plan.weights[j], 0.);
                value += plan.weights[j] * std::pow(plan.nodes[j], int(k));
            }
            EXPECT_NEAR(value, 1. / double(k + 1), 2e-14);
        }
    }
    for (uint64_t a = 1; a <= 6; ++a) {
        const Plan plan(2, 2, 6, 9, a, 4, 2, sizeof(double), false);
        EXPECT_EQ(plan.layout.left_degree(6), a);
        EXPECT_TRUE(signature_method::use_prefix_suffix<double>(1, a, 2, 2, 9, 6, false, 4, false, false));
    }
    EXPECT_FALSE(signature_method::use_prefix_suffix<float>(2, 0, 1, 2, 257, 19, false, 4, false, false));
    EXPECT_TRUE(signature_method::use_prefix_suffix<float>(2, 0, 1, 4, 17, 9, false, 16, false, false));
    EXPECT_TRUE(signature_method::use_prefix_suffix<float>(2, 0, 1, 4, 17, 9, false, 16, false, true));
    EXPECT_FALSE(signature_method::use_prefix_suffix<float>(2, 0, 1, 4, 17, 9, false, 80, false, false));
    EXPECT_FALSE(signature_method::use_prefix_suffix<float>(2, 0, 1, 4, 257, 9, false, 16, true, true));
    EXPECT_FALSE(signature_method::use_prefix_suffix<double>(2, 0, 1, 4, 17, 9, false, 16, false, false));
    EXPECT_FALSE(signature_method::use_prefix_suffix<float>(2, 0, 1, 4, 17, 9, true, 16, false, false));
    EXPECT_FALSE(signature_method::use_prefix_suffix<float>(2, 0, 1, 4, 17, 8, false, 16, false, false));
}

TEST(PrefixSuffix, InvalidSignatureMethod) {
    float path[4]{}, sig[7]{};
    EXPECT_NE(signature_f(path, sig, 1, 2, 2, 2, false, false, 1.f, true, 0,
        nullptr, 0, 0, 0, 1), 0);
    EXPECT_EQ(signature_f(path, sig, 1, 2, 2, 2, false, false, 1.f, true, 1,
        nullptr, 0, 0, 0, 3), 2);
    EXPECT_EQ(signature_f(path, sig, 1, 2, 2, 2, false, false, 1.f, true, 1,
        nullptr, 0, 0, 0, 2, 1), 2);
    EXPECT_EQ(signature_f(path, sig, 1, 2, 2, 2, false, false, 1.f, true, 1,
        nullptr, 0, 0, 0, 1, 3), 2);
}

TEST(PrefixSuffix, EndpointConcurrency) {
    using prefix_suffix_endpoint::select_cpu_block;
    const prefix_suffix::Layout cubic(129, 3), quadratic(24, 2);
    EXPECT_EQ(select_cpu_block(1, 256, cubic, 1, 4, false), 256);
    EXPECT_EQ(select_cpu_block(1, 256, cubic, 20, 4, false), 0);
    EXPECT_EQ(select_cpu_block(1, 256, prefix_suffix::Layout(128, 3), 2, 4, false), 128);
    EXPECT_EQ(select_cpu_block(32, 256, cubic, 1, 4, false), 0);
    EXPECT_EQ(select_cpu_block(32, 4096, prefix_suffix::Layout(48, 3), 20, 8, false), 1024);
    EXPECT_EQ(select_cpu_block(8, 513, quadratic, 4, 8, false), 513);
    EXPECT_EQ(select_cpu_block(8, 513, quadratic, 16, 8, false), 0);
    EXPECT_EQ(select_cpu_block(8, 513, quadratic, 4, 8, true), 0);

    using prefix_suffix_endpoint::CpuWorkspace;
    // Two persistent buffers consume 80 elements, each active worker needs 10.
    EXPECT_EQ(CpuWorkspace(4, 10, 2, 2, 1, 2, 4, 89).workers, 0);
    EXPECT_EQ(CpuWorkspace(4, 10, 2, 2, 1, 2, 4, 100).workers, 2);
    EXPECT_EQ(CpuWorkspace(4, UINT64_MAX, 2, 2, 1, 2, 4, 100).workers, 0);
}

#ifndef CPSIG_PREBUILT_TESTS
TEST(PrefixSuffix, EndpointRespectsSchedulerLimits) {
    constexpr uint64_t dimension = 32, points = 257;
    std::vector<float> path(points * dimension);
    for (size_t i = 0; i < path.size(); ++i)
        path[i] = float(int((i * 37 + i / dimension) % 103) - 51) / 79;
    std::vector<float> expected(prefix_suffix::Layout(dimension, 3).offset.back() - 1), actual(expected.size());
    auto run = [&](int jobs, std::vector<float>& output) {
        return signature_f(path.data(), output.data(), 1, dimension, points, 3,
            false, false, 1.f, false, jobs, nullptr, 0, 0, 0, 1);
    };
    ASSERT_EQ(run(1, expected), 0);
    auto check_serial_split = [&] {
        ASSERT_EQ(run(20, actual), 0);
        EXPECT_TRUE(expected == actual);
    };
    oneapi::tbb::task_arena(1).execute(check_serial_split);
    using oneapi::tbb::global_control;
    global_control limit(global_control::max_allowed_parallelism, 1);
    check_serial_split();
}
#endif

#ifndef CPSIG_PREBUILT_TESTS
TEST(PrefixSuffix, GemmLargeStrides) {
    using Matrix = prefix_suffix::Matrix<double>;
    double a[] = {1, 2, 3}, b[] = {4, 5, 6}, c = -1;
    const uint64_t stride = uint64_t(INT_MAX) + 1;
    for (bool first : {true, false}) {
        cpsig::factor_gemm(Matrix{a, 1, 3, stride, 1}, Matrix{b, 3, 1, 1, stride},
            Matrix{&c, 1, 1, stride, 1}, first);
        EXPECT_DOUBLE_EQ(c, first ? 32 : 64);
    }
}
#endif

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
        std::vector<double> actual(l.offset[degree + 1]), expected(actual.size()), segment(actual.size());
        std::vector<double> path(2 * d), actual_work(l.power[degree]);
        actual[0] = 1;
        for (size_t x = 1; x < actual.size(); ++x) actual[x] = double(int(x % 17) - 8) / 32;
        double* v = path.data() + d;
        for (size_t x = 0; x < d; ++x) v[x] = double(int(x) - 2) / 8;
        ASSERT_EQ(signature_d(path.data(), segment.data(), 1, d, 2, degree), 0);
        ASSERT_EQ(sig_combine_d(segment.data(), actual.data(), expected.data(), 1, d, degree), 0);
        cpsig::HWY_NAMESPACE::cpu_detail::prepend(actual.data(), v, l, actual_work.data());
        for (size_t x = 0; x < actual.size(); ++x) EXPECT_NEAR(actual[x], expected[x], 1e-14);
    }
}

TEST(PrefixSuffix, RightFactorBackprop) {
    for (uint64_t d : {2, 3, 5}) for (uint64_t b : {1, 2, 3}) for (uint64_t j : {0, 1, 3}) {
        prefix_suffix::Layout l(d, 2 * b + 2);
        const uint64_t size = l.offset[b + 1];
        std::vector<double> suffix(size), v(d), deriv(l.power[b]), dsuffix(size), dv(d);
        std::vector<double> history(size), work(size), result(l.power[b]);
        suffix[0] = 1;
        for (uint64_t x = 1; x < size; ++x) suffix[x] = double(int(x % 11) - 5) / 16;
        for (uint64_t c = 0; c < d; ++c) v[c] = double(int(c) - 1) / 8;
        for (uint64_t x = 0; x < deriv.size(); ++x) deriv[x] = double(int(x % 7) - 3) / 8;
        cpsig::HWY_NAMESPACE::cpu_detail::right_factor_backprop(suffix.data(), v.data(), deriv.data(),
            dsuffix.data(), dv.data(), j, b, l, history.data(), work.data());
        auto loss = [&] {
            prefix_suffix::right_factor(result.data(), suffix.data(), v.data(), j, b, l);
            double value = 0;
            for (uint64_t x = 0; x < deriv.size(); ++x) value += deriv[x] * result[x];
            return value;
        };
        auto check = [&](double& parameter, double expected) {
            const double original = parameter, epsilon = 1e-6;
            parameter = original + epsilon; const double plus = loss();
            parameter = original - epsilon; const double minus = loss();
            parameter = original;
            EXPECT_NEAR(expected, (plus - minus) / (2 * epsilon), 1e-9);
        };
        for (uint64_t x = 1; x < size; ++x) check(suffix[x], dsuffix[x]);
        for (uint64_t c = 0; c < d; ++c) check(v[c], dv[c]);
    }
}

TEST(PrefixSuffix, NativeForwardAndBackprop) {
    for (uint64_t dimension : {1, 2, 3, 4, 8}) for (uint64_t degree : {0, 1, 2, 3, 4, 5, 6})
        for (bool time_aug : {false, true}) for (bool lead_lag : {false, true}) for (bool scalar : {false, true}) {
            if (dimension > 3 && (degree < 2 || degree > 3 || time_aug || lead_lag)) continue;
            const uint64_t batch = 3, length = 5;
            const uint64_t d = dimension * (lead_lag ? 2 : 1) + time_aug;
            const uint64_t size = prefix_suffix::Layout(d, degree).offset.back() - !scalar;
            std::vector<double> path(batch * length * dimension), sig(batch * size), deriv(batch * size);
            std::vector<double> actual(path.size()), expected(path.size());
            for (uint64_t x = 0; x < path.size(); ++x) path[x] = double(int((x * 7) % 19) - 9) / 32;
            for (uint64_t x = 0; x < deriv.size(); ++x) deriv[x] = double(int((x * 5) % 13) - 6) / 16;
            ASSERT_EQ(signature_d(path.data(), sig.data(), batch, dimension, length, degree,
                time_aug, lead_lag, 0.7, scalar, 2), 0);
            std::vector<double> ps(sig.size());
            ASSERT_EQ(signature_d(path.data(), ps.data(), batch, dimension, length, degree,
                time_aug, lead_lag, 0.7, scalar, 2, nullptr, 0, 0, 0, 1), 0);
            for (size_t i = 0; i < sig.size(); ++i) EXPECT_NEAR(ps[i], sig[i], 1e-12);
            ASSERT_EQ(sig_backprop_d(path.data(), expected.data(), deriv.data(), sig.data(), batch,
                dimension, length, degree, time_aug, lead_lag, 0.7, scalar, 2), 0);
            ASSERT_EQ(sig_backprop_d(path.data(), actual.data(), deriv.data(), nullptr, batch,
                dimension, length, degree, time_aug, lead_lag, 0.7, scalar, 2, nullptr, 0, 0, 0, 1), 0) << cpsig_last_error_message();
            for (uint64_t x = 0; x < path.size(); ++x)
                EXPECT_NEAR(actual[x], expected[x], 1e-10) << dimension << "," << degree << "," << time_aug << "," << lead_lag;
        }
}

TEST(PrefixSuffix, BackpropFiniteDifference) {
    const uint64_t d = 3, length = 6, degree = 6, size = prefix_suffix::Layout(d, degree).offset.back();
    std::vector<double> path(length * d), sig(size), deriv(size), actual(path.size());
    for (uint64_t x = 0; x < path.size(); ++x) path[x] = double(int((x * 7) % 19) - 9) / 16;
    for (uint64_t x = 0; x < size; ++x) deriv[x] = double(int(x % 11) - 5) / 8;
    ASSERT_EQ(sig_backprop_d(path.data(), actual.data(), deriv.data(), nullptr, 1, d, length,
        degree, false, false, 1., true, 3, nullptr, 0, 0, 0, 1), 0);
    for (uint64_t x = 0; x < path.size(); ++x) {
        const double original = path[x], epsilon = 1e-5;
        long double values[2]{};
        for (int side = 0; side < 2; ++side) {
            path[x] = original + (side ? -epsilon : epsilon);
            ASSERT_EQ(signature_d(path.data(), sig.data(), 1, d, length, degree, false, false, 1., true, 1), 0);
            for (uint64_t word = 0; word < size; ++word) values[side] += static_cast<long double>(deriv[word]) * sig[word];
        }
        path[x] = original;
        EXPECT_NEAR(actual[x], (values[0] - values[1]) / (2 * epsilon), 2e-9);
    }
}
