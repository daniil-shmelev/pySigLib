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

TEST(sigLassoTest, TerminalCacheChecksWordsAndPrefixDegree) {
    std::vector<double> storage(7, 0);
    siglasso::TerminalCache cache{storage.data(), 1, 4};
    const uint64_t prefix[]{5, 11};
    double value = 0;
    EXPECT_FALSE(cache.get(0, prefix, 2, 7, 9, 2, value));
    cache.put(0, prefix, 2, 7, 9, 2, .25);
    EXPECT_TRUE(cache.get(0, prefix, 2, 7, 9, 2, value));
    EXPECT_EQ(value, .25);
    EXPECT_FALSE(cache.get(0, prefix, 2, 7, 10, 2, value));
    const uint64_t longer[]{5, 11, 7};
    EXPECT_FALSE(cache.get(0, longer, 3, 9, 0, 1, value));
    cache.put(0, longer, 3, 9, 0, 1, .125);
    EXPECT_TRUE(cache.get(0, longer, 3, 9, 0, 1, value));
    EXPECT_EQ(value, .125);
    EXPECT_FALSE(cache.get(0, prefix, 2, 7, 9, 2, value));
    storage[0] = 0;
    EXPECT_FALSE(cache.get(0, longer, 3, 9, 0, 1, value));
}

TEST(sigLassoTest, RemainingVariationIgnoresCancellation) {
    constexpr size_t n = 2, l = 4, d = 2;
    const std::vector<double> increments{1, 0, -2, 0, 0, 0, 0, 0, -1, 0, 2, 0, .25, 0, -.5, 0};
    std::vector<double> variation(n * l), after(n * l);
    for (size_t i = 0; i < n; ++i) siglasso::prepare_sample(increments.data(), variation.data(), after.data(), i, n, l, d);
    EXPECT_EQ(variation, (std::vector<double>{2, 0, 0, 0, 2, 0, .5, 0}));
    EXPECT_EQ(after, (std::vector<double>{2.5, 0, 2.5, 0, .5, 0, 0, 0}));
}

TEST(sigLassoTest, RemainingVariationFactorsMatchQuadrature) {
    const std::array<double, 4> nodes{.18343464249564980494, .52553240991632898582, .79666647741362673959, .96028985649753623168};
    const std::array<double, 4> quadrature{.36268378337836198297, .31370664587788728734, .22238103445337447054, .10122853629037625915};
    const std::array<double, 8> steps{0, 0, .2, .7, .2, .5, 1e-12, .5}, tails{0, .2, 0, .1, .6, .5, .5, 1e-12};
    constexpr size_t levels = 8;
    for (size_t k = 0; k <= 6; ++k) {
        std::vector<double> weights(k + levels + 1, 1);
        for (size_t j = 1; j < weights.size(); ++j) weights[j] = weights[j - 1] * j;
        const auto constants = siglasso::integration_coefficients(k, levels, weights);
        for (size_t q = 1; q <= levels; ++q) for (size_t a = 0; a <= k; ++a) for (size_t i = 0; i < steps.size(); ++i) {
            double expected = 0;
            const double binomial = std::tgamma(k + 1.) / (std::tgamma(a + 1.) * std::tgamma(k - a + 1.));
            for (size_t j = 0; j < nodes.size(); ++j) for (double sign : {-1., 1.}) {
                const double t = (1 + sign * nodes[j]) / 2;
                expected += quadrature[j] / 2 * binomial * std::pow(t, a) * std::pow(1 - t, k - a)
                    * steps[i] * std::pow(tails[i] + steps[i] * (1 - t), q - 1) / std::tgamma(static_cast<double>(q));
            }
            expected *= weights[k + q];
            const double actual = siglasso::factor_value(steps.data(), tails.data(), constants.data(), k, q, a, 0, i, steps.size());
            EXPECT_NEAR(actual, expected, 2e-13 * std::abs(expected) + 1e-300) << "k=" << k << " q=" << q << " a=" << a << " i=" << i;
        }
    }
}

TEST(sigLassoTest, RemainingVariationFactorsAtDegreeLimit) {
    constexpr size_t degree = 170;
    std::vector<double> weights(degree + 1, 1);
    for (size_t j = 1; j <= degree; ++j) weights[j] = weights[j - 1] * j;
    for (size_t k : {0U, 30U, 100U, 169U}) {
        const size_t q = degree - k;
        const auto constants = siglasso::integration_coefficients(k, q, weights);
        for (size_t a : {size_t(0), k / 2, k}) for (double step : {.02, .7, 1.}) {
            const double tail = 0;
            const double expected = std::exp(std::lgamma(k + 1.) + std::lgamma(k - a + q)
                - std::lgamma(k - a + 1.) - std::lgamma(static_cast<double>(q)) + q * std::log(step));
            const double actual = siglasso::factor_value(&step, &tail, constants.data(), k, q, a, 0, 0, 1);
            EXPECT_TRUE(std::isfinite(actual));
            EXPECT_NEAR(actual, expected, 2e-12 * expected + 1e-300);
        }
    }
}

TEST(sigLassoTest, BernsteinColumnsAndDescendantBounds) {
    constexpr size_t n = 5, l = 6, d = 3, m = 4;
    std::mt19937 generator(712);
    std::normal_distribution<double> normal(0, 0.4);
    std::vector<double> paths(n * (l + 1) * d), increments(n * l * d), residual(n);
    for (auto& value : paths) value = normal(generator);
    for (auto& value : residual) value = normal(generator);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < l; ++j) for (size_t a = 0; a < d; ++a)
        increments[(j * d + a) * n + i] = paths[(i * (l + 1) + j + 1) * d + a] - paths[(i * (l + 1) + j) * d + a];
    std::vector<double> variation(n * l), after(n * l);
    for (size_t i = 0; i < n; ++i) siglasso::prepare_sample(increments.data(), variation.data(), after.data(), i, n, l, d);
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
    for (size_t k = 1; k <= m; ++k) weights[k] = weights[k - 1] * k;
    ASSERT_EQ(sig_coef_d(paths.data(), reference.data(), letters.data(), words.size(), degrees.data(), n, d, l + 1, false, false, 1., false, 1), 0);
    while (!pending.empty()) {
        auto prefix = std::move(pending.back()); pending.pop_back();
        const size_t k = prefix.word.size();
        auto constants = siglasso::integration_coefficients(k, m - k, weights);
        const size_t factor_size = (m - k) * (k + 1) * n * l;
        std::vector<double> factors(factor_size + constants.size()), integrals(n * l), linear_integrals(n * l), moments((m - k) * n * l);
        std::copy(constants.begin(), constants.end(), factors.begin() + factor_size);
        for (size_t q = 1; q <= m - k; ++q) for (size_t a = 0; a <= k; ++a) for (size_t j = 0; j < l; ++j) for (size_t i = 0; i < n; ++i)
            factors[(((q - 1) * (k + 1) + a) * l + j) * n + i] = siglasso::factor_value(
                variation.data(), after.data(), constants.data(), k, q, a, j, i, n) / n;
        for (size_t j = 0; j < l; ++j) for (size_t i = 0; i < n; ++i) for (size_t a = 0; a <= k; ++a) {
            integrals[j * n + i] += prefix.coefficients[(j * (k + 1) + a) * n + i] / (k + 1);
            linear_integrals[j * n + i] += (a + 1) * prefix.coefficients[(j * (k + 1) + a) * n + i] / ((k + 1) * (k + 2));
        }
        std::vector<double> scratch(siglasso::integral_scratch_size(m));
        for (size_t j = 0; j < l; ++j) for (size_t i = 0; i < n; ++i)
            siglasso::absolute_moments(prefix.coefficients.data(), factors.data(), variation[j * n + i], after[j * n + i],
                k, m - k, j, i, n, l, scratch.data(), moments.data() + j * n + i);
        for (size_t q = 1; q <= std::min<size_t>(2, m - k); ++q) for (size_t a = 0; a < d; ++a)
            for (size_t b = 0; b < (q == 2 ? d : 1); ++b) {
                auto word = prefix.word; word.push_back(a); if (q == 2) word.push_back(b);
                const size_t position = std::find(words.begin(), words.end(), word) - words.begin();
                for (size_t i = 0; i < n; ++i) {
                    double magnitude;
                    const double value = siglasso::terminal_sample(integrals.data(), linear_integrals.data(), increments.data(),
                        a, b, q, k, i, n, l, d, magnitude);
                    EXPECT_NEAR(value, reference[i * words.size() + position], 2e-12);
                    EXPECT_GE(magnitude + 1e-13, std::abs(value));
                }
            }
        for (size_t low = 0; low < d; ++low) for (size_t high = low + 1; high <= d; ++high) {
            double bound = 0;
            for (size_t q = 0; q < m - k; ++q) {
                double total = 0;
                for (size_t j = 0; j < l; ++j) for (size_t i = 0; i < n; ++i) {
                    double step = 0;
                    for (size_t a = low; a < high; ++a) step = std::max(step, std::abs(increments[(j * d + a) * n + i]));
                    const double moment = moments[(q * l + j) * n + i];
                    total += moment < 0 ? 1e100 : std::abs(residual[i]) * moment * step / variation[j * n + i];
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

TEST(sigLassoTest, AbsolutePolynomialIntegrals) {
    for (size_t k : {1U, 2U, 3U, 7U, 12U, 21U}) for (double root : {.0, .37, .5, 1.}) {
        std::vector<double> weights(k + 3, 1), coefficients(k + 1);
        for (size_t j = 1; j < weights.size(); ++j) weights[j] = weights[j - 1] * j;
        for (size_t a = 0; a <= k; ++a) coefficients[a] = std::pow(-root, k - a) * std::pow(1 - root, a);
        auto constants = siglasso::integration_coefficients(k, 2, weights);
        std::vector<double> factors(2 * (k + 1) + constants.size()), scratch(siglasso::integral_scratch_size(k + 2));
        std::copy(constants.begin(), constants.end(), factors.begin() + 2 * (k + 1));
        const double step = 1, tail = 0;
        for (size_t q = 0; q < 2; ++q) for (size_t a = 0; a <= k; ++a)
            factors[q * (k + 1) + a] = siglasso::factor_value(&step, &tail, constants.data(), k, q + 1, a, 0, 0, 1);
        double result[2];
        siglasso::absolute_moments(coefficients.data(), factors.data(), step, tail, k, 2, 0, 0, 1, 1, scratch.data(), result);
        const double integral = (std::pow(1 - root, k + 1) + std::pow(root, k + 1)) / (k + 1);
        const double weighted = (1 - root) * integral - (std::pow(1 - root, k + 2) - std::pow(root, k + 2)) / (k + 2);
        ASSERT_GE(result[0], 0); ASSERT_GE(result[1], 0);
        EXPECT_NEAR(result[0] / weights[k + 1], integral, 2e-13);
        EXPECT_NEAR(result[1] / weights[k + 2], weighted, 2e-13);
    }
}

TEST(sigLassoTest, TrapezoidalIntegral) {
    double result[8];
    const double start = -.37, end = .51, step = .23, tail = .47;
    siglasso::trapezoidal_moments(start, end, step, tail, 24., 3, 8, 1, result);
    double weight = 24, factorial = 1;
    for (size_t q = 0; q < 8; ++q) {
        const double reference = weight * step * .5 / factorial *
            (std::abs(start) * std::pow(step + tail, q) + std::abs(end) * std::pow(tail, q));
        EXPECT_NEAR(result[q], reference, 1e-12 * reference);
        weight *= 5 + q; factorial *= q + 1;
    }
}

TEST(sigLassoTest, NativeFitAndOutputAllocation) {
    const std::vector<double> paths{0, -2, 0, -1, 0, 1, 0, 2}, response{-2, -1, 1, 2};
    const double alpha = .1;
    SigLassoOptions options{};
    options.samples = 4; options.length = 2; options.dimension = options.degree = options.targets = 1;
    options.max_iter = 100; options.batch_size = 4; options.tolerance = 1e-10; options.n_jobs = 1; options.intercept = 1;
    for (uint32_t full : {0U, 1U}) {
        options.full = full;
        uint64_t letters[1], offsets[2], converged[1], iterations[1];
        double coefficient, intercept, objective, gap, signatures[4];
        SigLassoStats stats{};
        struct Output { SigLassoShape shape; SigLassoOutput buffers; size_t calls = 0; } output{
            {}, {letters, offsets, converged, iterations, &coefficient, &intercept, &objective, &gap, signatures, &stats}};
        const auto allocate = [](void* context, const SigLassoShape* shape, SigLassoOutput* buffers) {
            auto& output = *static_cast<Output*>(context);
            output.shape = *shape; ++output.calls;
            if (shape->words != 1 || shape->letters != 1 || shape->targets != 1) return 1;
            *buffers = output.buffers;
            return 0;
        };
        EXPECT_NE(sig_lasso_d(paths.data(), response.data(), &alpha, nullptr, nullptr, 0, &options, nullptr, nullptr), 0);
        const auto fail = [](void*, const SigLassoShape*, SigLassoOutput*) { return 1; };
        EXPECT_NE(sig_lasso_d(paths.data(), response.data(), &alpha, nullptr, nullptr, 0, &options, fail, nullptr), 0);
        const auto empty = [](void*, const SigLassoShape*, SigLassoOutput*) { return 0; };
        EXPECT_NE(sig_lasso_d(paths.data(), response.data(), &alpha, nullptr, nullptr, 0, &options, empty, nullptr), 0);
        ASSERT_EQ(sig_lasso_d(paths.data(), response.data(), &alpha, nullptr, nullptr, 0, &options, allocate, &output), 0);
        EXPECT_EQ(output.calls, 1); EXPECT_EQ(output.shape.stats.activations, 1);
        EXPECT_NEAR(coefficient, 1.4, 1e-10); EXPECT_NEAR(intercept, 0, 1e-12); EXPECT_LE(gap, 1e-10); EXPECT_TRUE(converged[0]);
        EXPECT_EQ(letters[0], 0); EXPECT_EQ(offsets[0], 0); EXPECT_EQ(offsets[1], 1);
        EXPECT_EQ(signatures[0], -1); EXPECT_EQ(signatures[3], 1);
        EXPECT_GT(stats.output_bytes, 0); EXPECT_GE(stats.peak_bytes, stats.output_bytes);
    }
}
