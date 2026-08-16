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
#include "branched_cache.h"

    TEST(branchedSigCombineTest, ChenIdentity) {
        uint64_t dimension = 2, max_nodes = 3;
        (void)prepare_branched_sig(dimension, max_nodes, false, false);

        std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
        std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
        std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

        uint64_t bs_len = branched_sig_length(dimension, max_nodes);

        std::vector<double> bsig1(bs_len);
        (void)branched_sig_d(path1.data(), bsig1.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

        std::vector<double> bsig2(bs_len);
        (void)branched_sig_d(path2.data(), bsig2.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

        std::vector<double> true_bsig(bs_len);
        (void)branched_sig_d(path.data(), true_bsig.data(), (uint64_t)1, dimension, 5, max_nodes, 1);

        check_result_2(branched_sig_combine_d, bsig1, bsig2, true_bsig, (uint64_t)1, dimension, max_nodes, 1, false, true);
    }

    TEST(branchedSigCombineTest, BatchChenIdentity) {
        uint64_t dimension = 2, max_nodes = 3, batch_size = 2;
        (void)prepare_branched_sig(dimension, max_nodes, false, false);

        std::vector<double> path1 = {
            0., 0., 1., 0.5, 0.4, 2.,
            0., 0., 0.25, 0.25, 0.5, 0.5 };
        std::vector<double> path2 = {
            0.4, 2., 6., 0.1, 2.3, 4.1,
            0.5, 0.5, 1., 1., 0.75, 0.75 };
        std::vector<double> path = {
            0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1,
            0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1., 0.75, 0.75 };

        uint64_t bs_len = branched_sig_length(dimension, max_nodes);

        std::vector<double> bsig1(bs_len * batch_size);
        (void)branched_sig_d(path1.data(), bsig1.data(), batch_size, dimension, 3, max_nodes, 1);

        std::vector<double> bsig2(bs_len * batch_size);
        (void)branched_sig_d(path2.data(), bsig2.data(), batch_size, dimension, 3, max_nodes, 1);

        std::vector<double> true_bsig(bs_len * batch_size);
        (void)branched_sig_d(path.data(), true_bsig.data(), batch_size, dimension, 5, max_nodes, 1);

        check_result_2(branched_sig_combine_d, bsig1, bsig2, true_bsig, batch_size, dimension, max_nodes, 1, false, true);
    }

    TEST(branchedSigCombineBackpropTest, FiniteDifference) {
        uint64_t dimension = 2, max_nodes = 3;
        (void)prepare_branched_sig(dimension, max_nodes, false, false);
        uint64_t bs_len = branched_sig_length(dimension, max_nodes);

        std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
        std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };

        std::vector<double> bsig1(bs_len);
        (void)branched_sig_d(path1.data(), bsig1.data(), (uint64_t)1, dimension, 3, max_nodes, 1);
        std::vector<double> bsig2(bs_len);
        (void)branched_sig_d(path2.data(), bsig2.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

        std::vector<double> derivs(bs_len);
        for (uint64_t i = 0; i < bs_len; ++i)
            derivs[i] = 0.3 * (i + 1) - 1.;

        std::vector<double> d_bsig1(bs_len);
        std::vector<double> d_bsig2(bs_len);
        (void)branched_sig_combine_backprop_d(bsig1.data(), bsig2.data(), derivs.data(),
            d_bsig1.data(), d_bsig2.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);

        double eps = 1e-7;
        for (uint64_t i = 0; i < 5 && i < bs_len; ++i) {
            double orig = bsig1[i];
            bsig1[i] = orig + eps;
            std::vector<double> out_plus(bs_len);
            (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_plus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
            bsig1[i] = orig - eps;
            std::vector<double> out_minus(bs_len);
            (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_minus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
            bsig1[i] = orig;

            double numerical = 0.;
            for (uint64_t j = 0; j < bs_len; ++j)
                numerical += derivs[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
            EXPECT_TRUE(abs(numerical - d_bsig1[i]) < 1e-4);
        }

        for (uint64_t i = 0; i < 5 && i < bs_len; ++i) {
            double orig = bsig2[i];
            bsig2[i] = orig + eps;
            std::vector<double> out_plus(bs_len);
            (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_plus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
            bsig2[i] = orig - eps;
            std::vector<double> out_minus(bs_len);
            (void)branched_sig_combine_d(bsig1.data(), bsig2.data(), out_minus.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);
            bsig2[i] = orig;

            double numerical = 0.;
            for (uint64_t j = 0; j < bs_len; ++j)
                numerical += derivs[j] * (out_plus[j] - out_minus[j]) / (2. * eps);
            EXPECT_TRUE(abs(numerical - d_bsig2[i]) < 1e-4);
        }
    }

    TEST(branchedSigCombineBackpropTest, ZeroDerivative) {
        uint64_t dimension = 2, max_nodes = 3;
        (void)prepare_branched_sig(dimension, max_nodes, false, false);
        uint64_t bs_len = branched_sig_length(dimension, max_nodes);

        std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
        std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
        std::vector<double> bsig1(bs_len);
        (void)branched_sig_d(path1.data(), bsig1.data(), (uint64_t)1, dimension, 3, max_nodes, 1);
        std::vector<double> bsig2(bs_len);
        (void)branched_sig_d(path2.data(), bsig2.data(), (uint64_t)1, dimension, 3, max_nodes, 1);

        std::vector<double> derivs(bs_len, 0.);
        std::vector<double> out1(bs_len);
        std::vector<double> out2(bs_len);
        (void)branched_sig_combine_backprop_d(bsig1.data(), bsig2.data(), derivs.data(),
            out1.data(), out2.data(), (uint64_t)1, dimension, max_nodes, 1, false, true);

        for (uint64_t i = 0; i < bs_len; ++i) {
            EXPECT_TRUE(abs(out1[i]) < DOUBLE_EPSILON);
            EXPECT_TRUE(abs(out2[i]) < DOUBLE_EPSILON);
        }
    }

    TEST(branchedLogSigLengthTest, MatchesExplicitWeightedLyndonEnumeration) {
        EXPECT_EQ(branched_log_sig_length(0, 3, true), 0);
        EXPECT_EQ(branched_log_sig_length(2, 0, true), 0);
        EXPECT_EQ(branched_log_sig_length(2, 3, false),
            branched_sig_length(2, 3, false) - 1);
        EXPECT_EQ(branched_log_sig_length(1, UINT64_MAX, true), 0);

        const std::vector<std::pair<uint64_t, uint64_t>> cases = {
            { 1, 1 }, { 1, 3 }, { 2, 3 }, { 3, 2 }
        };
        for (const auto& [dimension, max_nodes] : cases) {
            const auto cache = build_branched_sig_cache(dimension, max_nodes, true);
            uint64_t expected = 0;
            for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
                const uint64_t start = cache.basis_forest_offsets[basis_idx];
                const uint64_t end = cache.basis_forest_offsets[basis_idx + 1];
                const word forest(
                    cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(start),
                    cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(end));
                expected += is_lyndon(forest);
            }
            EXPECT_EQ(branched_log_sig_length(dimension, max_nodes, true), expected);
        }
    }

    TEST(branchedLogSigMethodsTest, LyndonCoordinatesAndBasisProjection) {
        const uint64_t dimension = 2;
        const uint64_t max_nodes = 3;
        ASSERT_EQ(prepare_branched_log_sig(
            dimension, max_nodes, 2, false, true), 0);

        const uint64_t bsig_len = branched_sig_length(dimension, max_nodes, true);
        const uint64_t logsig_len = branched_log_sig_length(dimension, max_nodes, true);
        const std::vector<double> path = {
            0., 0., 0.2, -0.3, 0.7, 0.4, 0.5, 0.8
        };
        std::vector<double> bsig(bsig_len);
        ASSERT_EQ(branched_sig_d(
            path.data(), bsig.data(), 1, dimension, 4, max_nodes,
            1, false, false, 1., true, true), 0);

        std::vector<double> expanded(bsig_len);
        std::vector<double> method_one(logsig_len);
        std::vector<double> method_two(logsig_len);
        ASSERT_EQ(branched_sig_to_log_sig_d(
            bsig.data(), expanded.data(), 1, dimension, max_nodes,
            0, 1, true, true), 0);
        ASSERT_EQ(branched_sig_to_log_sig_d(
            bsig.data(), method_one.data(), 1, dimension, max_nodes,
            1, 1, true, true), 0);
        ASSERT_EQ(branched_sig_to_log_sig_d(
            bsig.data(), method_two.data(), 1, dimension, max_nodes,
            2, 1, true, true), 0);

        const auto cache = build_branched_sig_cache(dimension, max_nodes, true);
        std::vector<word> lyndon_words;
        std::vector<uint64_t> lyndon_idx;
        std::vector<word> flat_words(cache.total_length);
        for (uint64_t basis_idx = 0; basis_idx + 1 < cache.total_length; ++basis_idx) {
            const uint64_t start = cache.basis_forest_offsets[basis_idx];
            const uint64_t end = cache.basis_forest_offsets[basis_idx + 1];
            word forest(
                cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(start),
                cache.basis_forest_data.begin() + static_cast<std::ptrdiff_t>(end));
            flat_words[basis_idx + 1] = forest;
            if (is_lyndon(forest)) {
                lyndon_words.push_back(forest);
                lyndon_idx.push_back(basis_idx + 1);
            }
        }
        ASSERT_EQ(lyndon_idx.size(), logsig_len);
        for (uint64_t i = 0; i < logsig_len; ++i)
            EXPECT_NEAR(method_one[i], expanded[lyndon_idx[i]], 1e-12);

        SparseIntMatrix projection;
        std::unordered_map<word, uint64_t, WordHash> flat_idx;
        for (uint64_t i = 0; i < flat_words.size(); ++i)
            flat_idx[flat_words[i]] = i;
        lyndon_proj_matrix_from_words(
            projection,
            lyndon_words,
            cache.total_length,
            [&flat_idx](const word& w) {
                return flat_idx.at(w);
            },
            [&flat_words, &flat_idx](uint64_t i, uint64_t j, uint64_t) {
                return flat_idx.at(concatenate_words(flat_words.at(i), flat_words.at(j)));
            });
        for (uint64_t row = 0; row < logsig_len; ++row) {
            double reconstructed = method_two[row];
            for (const auto& entry : projection.rows[row])
                reconstructed += entry.val * method_two[entry.col];
            EXPECT_NEAR(reconstructed, method_one[row], 1e-12);
        }
    }
