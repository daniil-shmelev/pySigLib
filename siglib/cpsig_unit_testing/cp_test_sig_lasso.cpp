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
#include "sig_lasso_core.h"
#include "sig_lasso_math.h"
#include <random>

TEST(sigLassoTest, BernsteinColumnsAndDescendantBounds) {
    constexpr size_t n = 5, l = 6, d = 3, m = 4;
    std::mt19937 generator(712);
    std::normal_distribution<double> normal(0, 0.4);
    std::vector<double> paths(n * (l + 1) * d), increments(n * l * d), residual(n);
    for (auto& value : paths) value = normal(generator);
    for (auto& value : residual) value = normal(generator);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < l; ++j) for (size_t a = 0; a < d; ++a)
        increments[(j * d + a) * n + i] = paths[(i * (l + 1) + j + 1) * d + a] - paths[(i * (l + 1) + j) * d + a];
    std::vector<double> variation(n * l), after(n * l), envelopes(m * (m + 1) / 2 * n * l);
    for (size_t i = 0; i < n; ++i) siglasso::prepare_sample(increments.data(), variation.data(), after.data(), envelopes.data(), i, n, l, d, m);
    struct Prefix { siglasso::Word word; std::vector<double> coefficients; };
    std::vector<Prefix> pending{{{}, std::vector<double>(n * l, 1)}};
    std::vector<siglasso::Word> words;
    std::vector<uint64_t> letters, degrees;
    for (size_t k = 1; k <= m; ++k) {
        siglasso::Word word(k);
        do {
            words.push_back(word); degrees.push_back(k); letters.insert(letters.end(), word.begin(), word.end());
            size_t j = k; while (j && ++word[j - 1] == d) { word[j - 1] = 0; --j; }
            if (!j) break;
        } while (true);
    }
    std::vector<double> reference(n * words.size()), weights(m + 1, 1);
    for (size_t k = 1; k <= m; ++k) weights[k] = weights[k - 1] * std::sqrt(static_cast<double>(k));
    ASSERT_EQ(sig_coef_d(paths.data(), reference.data(), letters.data(), words.size(), degrees.data(), n, d, l + 1, false, false, 1., false, 1), 0);
    while (!pending.empty()) {
        auto prefix = std::move(pending.back()); pending.pop_back();
        const size_t k = prefix.word.size();
        auto constants = siglasso::integration_coefficients(k, m - k);
        std::vector<double> factors((m - k) * (k + 1) * n * l), integrals(n * l);
        for (size_t q = 1; q <= m - k; ++q) for (size_t a = 0; a <= k; ++a) for (size_t j = 0; j < l; ++j) for (size_t i = 0; i < n; ++i)
            factors[(((q - 1) * (k + 1) + a) * l + j) * n + i] = siglasso::factor_value(
                variation.data(), after.data(), envelopes.data(), constants.data(), k, q, a, j, i, n, l) * weights[k + q] / n;
        for (size_t j = 0; j < l; ++j) for (size_t i = 0; i < n; ++i) for (size_t a = 0; a <= k; ++a)
            integrals[j * n + i] += prefix.coefficients[(j * (k + 1) + a) * n + i] / (k + 1);
        for (size_t low = 0; low < d; ++low) for (size_t high = low + 1; high <= d; ++high) {
            double bound = 0;
            for (size_t q = 0; q < m - k; ++q) {
                double total = 0;
                for (size_t j = 0; j < l; ++j) for (size_t i = 0; i < n; ++i) {
                    double step = 0;
                    for (size_t a = low; a < high; ++a) step = std::max(step, std::abs(increments[(j * d + a) * n + i]));
                    total += std::abs(residual[i]) * siglasso::moment_value(prefix.coefficients.data(), factors.data(), k, q, j, i, n, l) * step / variation[j * n + i];
                }
                if (!q) {
                    double energy = 0;
                    for (size_t i = 0; i < n; ++i) energy += std::abs(residual[i]) * siglasso::energy_sample(integrals.data(), nullptr, increments.data(), low, high, i, n, l, d);
                    total = std::min(total, energy * weights[k + 1] / n);
                }
                bound = std::max(bound, total);
            }
            for (size_t w = 0; w < words.size(); ++w) if (words[w].size() > k && words[w][k] >= low && words[w][k] < high
                && std::equal(prefix.word.begin(), prefix.word.end(), words[w].begin())) {
                double correlation = 0;
                for (size_t i = 0; i < n; ++i) correlation += residual[i] * reference[i * words.size() + w];
                EXPECT_LE(std::abs(correlation) * weights[words[w].size()] / n, bound + 1e-12);
            }
        }
        for (size_t a = 0; a < d; ++a) {
            auto word = prefix.word; word.push_back(a);
            std::vector<double> child(k + 1 < m ? (k + 2) * n * l : 0), values(n);
            for (size_t i = 0; i < n; ++i) siglasso::child_sample(increments.data(), prefix.coefficients.data(), integrals.data(), child.empty() ? nullptr : child.data(), values.data(), a, k, i, n, l, d);
            const size_t position = std::find(words.begin(), words.end(), word) - words.begin();
            for (size_t i = 0; i < n; ++i) EXPECT_NEAR(values[i], reference[i * words.size() + position], 2e-13);
            if (!child.empty()) pending.push_back({std::move(word), std::move(child)});
        }
    }
}

TEST(sigLassoTest, VersionedABI) {
    EXPECT_EQ(sig_lasso_abi_version(), 2);
}

TEST(sigLassoTest, NativeFitAndReleasedHandle) {
    const std::vector<double> paths{0, -2, 0, -1, 0, 1, 0, 2}, response{-2, -1, 1, 2};
    const double weights[]{1, 1}, alpha = .1;
    SigLassoOptions options{};
    options.samples = 4; options.length = 2; options.dimension = options.degree = options.targets = 1;
    options.max_iter = 100; options.batch_size = 4; options.tolerance = 1e-10; options.n_jobs = 1; options.intercept = 1;
    for (uint32_t full : {0U, 1U}) {
        options.full = full; uint64_t handle = 0;
        ASSERT_EQ(sig_lasso_v2_d(paths.data(), response.data(), weights, &alpha, nullptr, nullptr, 0, &options, &handle), 0);
        SigLassoShape shape{}; ASSERT_EQ(sig_lasso_shape_v2(handle, &shape), 0); ASSERT_EQ(shape.words, 1);
        uint64_t letters[1], offsets[2], converged[1], iterations[1];
        double coefficient, intercept, objective, gap;
        ASSERT_EQ(sig_lasso_result_v2_d(handle, letters, offsets, &coefficient, &intercept, &objective, &gap, converged, iterations), 0);
        EXPECT_NEAR(coefficient, .96, 1e-10); EXPECT_NEAR(intercept, 0, 1e-12); EXPECT_LE(gap, 1e-10); EXPECT_TRUE(converged[0]);
        EXPECT_EQ(sig_lasso_release_v2(handle), 0); EXPECT_NE(sig_lasso_shape_v2(handle, &shape), 0);
    }
}
