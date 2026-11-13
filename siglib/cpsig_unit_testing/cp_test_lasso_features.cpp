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

#include "cp_test_helpers.h"
#include "cp_lasso_features.h"
#include "sig_lasso_math.h"
#include <random>

namespace {
class BoundBackend final : public siglasso::Backend {
public:
    explicit BoundBackend(const siglasso::SearchOptions& o) : Backend(o) {
        const size_t n = o.samples, l = o.length - 1, d = o.dimension;
        increments = alloc(n * l * d); variation = alloc(n * l); after = alloc(n * l);
        std::mt19937 generator(307); std::normal_distribution<double> normal;
        for (size_t j = 0; j < increments.size; ++j) increments.data[j] = normal(generator);
        for (size_t i = 0; i < n; ++i) {
            siglasso::prepare_sample(increments.data, variation.data, after.data, i, n, l, d);
            const double v = variation.data[i] + after.data[i];
            for (size_t j = 0; j < l * d; ++j) increments.data[j * n + i] /= v;
            siglasso::prepare_sample(increments.data, variation.data, after.data, i, n, l, d);
        }
    }
    siglasso::Array alloc(size_t n) override {
        auto charge = account(n * sizeof(double)); auto data = std::shared_ptr<double[]>(new double[n]());
        return {data.get(), n, std::shared_ptr<void>(data.get(), [data, charge](void*) {})};
    }
    void fill(const siglasso::Array& a, double v) override { std::fill_n(a.data, a.size, v); }
    void copy(const double* a, double* b, size_t n) override { std::copy_n(a, n, b); }
    double dot(const double* a, const double* b, size_t n) override { return std::inner_product(a, a + n, b, 0.); }
    double absolute_sum(const siglasso::Array& a) override { double s = 0; for (size_t i = 0; i < a.size; ++i) s += std::abs(a.data[i]); return s; }
    void axpy(double c, const double* a, double* b, size_t n) override { for (size_t i = 0; i < n; ++i) b[i] += c * a[i]; }
    void scale(double c, double* a, size_t n) override { for (size_t i = 0; i < n; ++i) a[i] *= c; }
    double center(double* a, size_t n, bool enabled) override {
        const double mean = enabled ? std::accumulate(a, a + n, 0.) / n : 0.; for (size_t i = 0; i < n; ++i) a[i] -= mean; return mean;
    }
    void correlations(const std::vector<siglasso::Column>&, const siglasso::Array&, std::vector<double>&, const std::vector<size_t>*) override { throw std::logic_error("Unused test operation"); }
    void prepare() override {}
    siglasso::Array factors(size_t) override { return {}; }
    void moments(siglasso::Node&, const siglasso::Array&) override {}
    siglasso::Array energy_gram(size_t, size_t) override { return {}; }
    void children(const std::vector<siglasso::Probe>&, std::vector<siglasso::NodePtr>&) override {}
    siglasso::Array response(size_t) override { return {}; }
    void export_result(const siglasso::Result&, const SigLassoOutput&) override {}
};
}

TEST(LassoFeatures, ExhaustiveDescendantAndGroupBounds) {
    for (uint32_t mode : {1U, 2U}) for (size_t degree : {4U, 6U}) for (unsigned flags : {0U, 15U, 31U, 119U, 127U, 1015U, 2039U}) {
        siglasso::SearchOptions options(SigLassoOptions{}, mode); options.samples = 7; options.length = 5; options.dimension = 2;
        options.degree = degree; options.n_jobs = 1; options.cache_bytes = 32 * 1024 * 1024;
        options.feature_kernel = flags;
        BoundBackend backend(options);
        EXPECT_GT(siglasso::audit_cpu_feature_bounds(backend, mode), 5);
    }
}

TEST(LassoFeatures, FactorialProductsByDegree) {
    std::mt19937 generator(6371); std::uniform_real_distribution<double> uniform(-5, 5);
    for (size_t count : {1U, 3U, 8U, 32U}) for (size_t levels = 1; levels <= 12; ++levels) {
        std::vector<siglasso::HallPowerFactor> factors;
        std::vector<double> side(count), dp(levels + 1), next(levels + 1);
        std::vector<siglasso::HallPowerItem> heap;
        std::vector<long double> expected(levels + 1), update(levels + 1); expected[0] = 1;
        for (size_t v = 0; v < count; ++v) {
            side[v] = v % 7 ? std::exp(uniform(generator)) : 0;
            const size_t degree = 1 + v % 4, run = v % 3;
            factors.push_back({degree, v, run}); update = expected;
            long double power = 1;
            for (size_t k = 1; k * degree <= levels; ++k) {
                power *= static_cast<long double>(side[v]) / (run + k);
                for (size_t q = k * degree; q <= levels; ++q) update[q] = std::max(update[q], expected[q - k * degree] * power);
            }
            expected.swap(update);
        }
        std::sort(factors.begin(), factors.end(), [](const auto& a, const auto& b) { return a.degree < b.degree; });
        siglasso::hall_factorial_products(factors, side.data(), 1, 0, levels, dp, next, heap);
        for (size_t q = 0; q <= levels; ++q)
            EXPECT_NEAR(dp[q], static_cast<double>(expected[q]), 2e-13 * std::max(1., static_cast<double>(expected[q])));
    }
}
