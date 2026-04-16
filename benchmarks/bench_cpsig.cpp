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

#include <benchmark/benchmark.h>
#include "cpsig.h"

#include <cstdint>
#include <random>
#include <vector>

// Suppress [[nodiscard]] warnings — we ignore status codes in hot loops.
#ifdef _MSC_VER
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

static void check(int rc, const char* fn) {
    if (rc != 0) {
        fprintf(stderr, "FATAL: %s returned %d\n", fn, rc);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::vector<double> random_data(uint64_t n, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ---------------------------------------------------------------------------
// sig forward
// Args: batch_size, dimension, length, degree
// ---------------------------------------------------------------------------
static void BM_sig(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);
    const uint64_t deg   = state.range(3);

    auto path = random_data(batch * dim * len);
    const uint64_t sig_len = ::sig_length(dim, deg);
    std::vector<double> out(batch * sig_len);

    for (auto _ : state) {
        ::signature_d(path.data(), out.data(), batch, dim, len, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig)
    ->Args({16, 3, 64,  6})
    ->Args({8,  4, 128, 5})
    ->Args({4,  5, 32,  7})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// sig backprop
// ---------------------------------------------------------------------------
static void BM_sig_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);
    const uint64_t deg   = state.range(3);

    auto path = random_data(batch * dim * len, 1);
    const uint64_t sig_len = ::sig_length(dim, deg);
    auto sig       = random_data(batch * sig_len, 2);
    auto sig_deriv = random_data(batch * sig_len, 3);
    std::vector<double> out(batch * dim * len);

    for (auto _ : state) {
        ::sig_backprop_d(path.data(), out.data(), sig_deriv.data(),
                              sig.data(), batch, dim, len, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_backprop)
    ->Args({16, 3, 64,  6})
    ->Args({8,  4, 128, 5})
    ->Args({4,  5, 32,  7})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// sig_combine
// ---------------------------------------------------------------------------
static void BM_sig_combine(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t sig_len = ::sig_length(dim, deg);
    auto sig1 = random_data(batch * sig_len, 1);
    auto sig2 = random_data(batch * sig_len, 2);
    std::vector<double> out(batch * sig_len);

    for (auto _ : state) {
        ::sig_combine_d(sig1.data(), sig2.data(), out.data(),
                             batch, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_combine)
    ->Args({64, 3, 6})
    ->Args({32, 4, 5})
    ->Args({16, 5, 7})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// sig_combine_backprop
// ---------------------------------------------------------------------------
static void BM_sig_combine_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t sig_len = ::sig_length(dim, deg);
    auto sig1       = random_data(batch * sig_len, 1);
    auto sig2       = random_data(batch * sig_len, 2);
    auto combo_deriv = random_data(batch * sig_len, 3);
    std::vector<double> d_sig1(batch * sig_len);
    std::vector<double> d_sig2(batch * sig_len);

    for (auto _ : state) {
        ::sig_combine_backprop_d(
            combo_deriv.data(), d_sig1.data(), d_sig2.data(),
            sig1.data(), sig2.data(), batch, dim, deg);
        benchmark::DoNotOptimize(d_sig1.data());
    }
}
BENCHMARK(BM_sig_combine_backprop)
    ->Args({64, 3, 6})
    ->Args({32, 4, 5})
    ->Args({16, 5, 7})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// sig_to_log_sig (method 0 = expanded)
// ---------------------------------------------------------------------------
static void BM_sig_to_log_sig(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t sig_len = ::sig_length(dim, deg);
    auto sig = random_data(batch * sig_len, 1);
    std::vector<double> out(batch * sig_len);

    check(::prepare_log_sig(dim, deg, 0, false), "prepare_log_sig");

    for (auto _ : state) {
        ::sig_to_log_sig_d(sig.data(), out.data(), batch, dim, deg,
                                false, false, 0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_to_log_sig)
    ->Args({64, 3, 6})
    ->Args({32, 4, 5})
    ->Args({16, 5, 7})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// log_sig (method 3 = BCH from path)
// ---------------------------------------------------------------------------
static void BM_log_sig_from_path(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);
    const uint64_t deg   = state.range(3);

    auto path = random_data(batch * dim * len, 1);
    const uint64_t ls_len = ::log_sig_length(dim, deg);
    std::vector<double> out(batch * ls_len);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_from_path_d(path.data(), out.data(), batch,
                                   len, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_from_path)
    ->Args({16, 3, 64,  6})
    ->Args({8,  4, 128, 5})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// log_sig_combine
// ---------------------------------------------------------------------------
static void BM_log_sig_combine(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t ls_len = ::log_sig_length(dim, deg);
    auto ls1 = random_data(batch * ls_len, 1);
    auto ls2 = random_data(batch * ls_len, 2);
    std::vector<double> out(batch * ls_len);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_combine_d(ls1.data(), ls2.data(), out.data(),
                                 batch, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_combine)
    ->Args({64, 3, 6})
    ->Args({32, 4, 5})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// sig_kernel (PDE)
// ---------------------------------------------------------------------------
static void BM_sig_kernel(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len1  = state.range(2);
    const uint64_t len2  = state.range(3);

    auto gram = random_data(batch * len1 * len2, 1);
    std::vector<double> out(batch);

    for (auto _ : state) {
        ::sig_kernel_d(gram.data(), out.data(), batch, dim,
                            len1, len2, 0, 0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_kernel)
    ->Args({32, 3, 64, 64})
    ->Args({16, 5, 128, 128})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// branched_sig
// ---------------------------------------------------------------------------
static void BM_branched_sig(benchmark::State& state) {
    const uint64_t batch     = state.range(0);
    const uint64_t dim       = state.range(1);
    const uint64_t len       = state.range(2);
    const uint64_t max_nodes = state.range(3);

    check(::prepare_branched_sig(dim, max_nodes, false, false), "prepare_branched_sig");
    const uint64_t bsig_len = ::branched_sig_length(dim, max_nodes, false);

    auto path = random_data(batch * dim * len, 1);
    std::vector<double> out(batch * bsig_len);

    for (auto _ : state) {
        ::branched_sig_d(path.data(), out.data(), batch, dim, len,
                              max_nodes);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig)
    ->Args({16, 3, 64, 4})
    ->Args({8,  4, 32, 3})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// transform_path (lead_lag)
// ---------------------------------------------------------------------------
static void BM_transform_path_lead_lag(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);

    auto path = random_data(batch * dim * len, 1);
    const uint64_t out_dim = 2 * dim;
    const uint64_t out_len = 2 * len - 1;
    std::vector<double> out(batch * out_dim * out_len);

    for (auto _ : state) {
        ::transform_path_d(path.data(), out.data(), batch, dim, len,
                                false, true, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_transform_path_lead_lag)
    ->Args({256, 8,  1024})
    ->Args({128, 16, 2048})
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// transform_path_backprop (lead_lag)
// ---------------------------------------------------------------------------
static void BM_transform_path_backprop_lead_lag(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);

    const uint64_t t_dim = 2 * dim;
    const uint64_t t_len = 2 * len - 1;
    auto derivs = random_data(batch * t_dim * t_len, 1);
    std::vector<double> out(batch * dim * len);

    for (auto _ : state) {
        ::transform_path_backprop_d(derivs.data(), out.data(),
                                         batch, dim, len, false, true, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_transform_path_backprop_lead_lag)
    ->Args({256, 8,  1024})
    ->Args({128, 16, 2048})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
