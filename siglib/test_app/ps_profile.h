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
#include "../cusig/cu_runtime_utils.h"
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

template<typename T>
int ps_profile(int argc, char** argv) {
    if (argc != 12) throw std::invalid_argument("--ps-profile cpu|cuda forward|backward float32|float64 batch segments dimension degree jobs iterations");
    const bool gpu = std::string(argv[3]) == "cuda", backward = std::string(argv[4]) == "backward";
    if ((!gpu && std::string(argv[3]) != "cpu") || (!backward && std::string(argv[4]) != "forward")
        || (std::string(argv[5]) != "float32" && std::string(argv[5]) != "float64"))
        throw std::invalid_argument("invalid profile device, pass or dtype");
    const uint64_t batch = std::stoull(argv[6]), n = std::stoull(argv[7]), d = std::stoull(argv[8]), m = std::stoull(argv[9]);
    const int jobs = std::stoi(argv[10]), iterations = std::stoi(argv[11]);
    if (!batch || !n || !d || !m || batch > 256 || n > 8192 || d > 64 || m > 24 || !jobs || iterations < 1)
        throw std::invalid_argument("profile parameters are outside the supported range");
    uint64_t power = 1, size = 1;
    for (uint64_t k = 1; k <= m; ++k) {
        if (power > (uint64_t(1) << 27) / d) throw std::invalid_argument("profile output is too large");
        power *= d; size += power;
    }
    if (batch * size > (uint64_t(1) << 27)) throw std::invalid_argument("profile batch output is too large");
    std::vector<T> path(batch * (n + 1) * d), signature(batch * size), seed(batch * size), gradient(path.size());
    std::mt19937 engine(11011);
    std::normal_distribution<double> normal(0, 0.03);
    for (auto& value : path) value = T(normal(engine));
    for (auto& value : seed) value = T(normal(engine));
    using Forward = int(*)(const T*, T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, int, const T*, uint64_t, uint64_t, uint64_t, int, uint64_t);
    using Backward = int(*)(const T*, T*, const T*, const T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, int, const T*, uint64_t, uint64_t, uint64_t, int, uint64_t);
    using CuForward = int(*)(const T*, T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, const T*, uint64_t, uint64_t, uint64_t, int, uint64_t, bool);
    using CuBackward = int(*)(const T*, T*, const T*, const T*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, const T*, uint64_t, uint64_t, uint64_t, int, uint64_t, bool);
    const auto forward = reinterpret_cast<Forward>(GET_FN_PTR(cpsig, sizeof(T) == 4 ? "signature_f" : "signature_d"));
    const auto reverse = reinterpret_cast<Backward>(GET_FN_PTR(cpsig, sizeof(T) == 4 ? "sig_backprop_f" : "sig_backprop_d"));
    const auto cu_forward = reinterpret_cast<CuForward>(GET_FN_PTR(cusig, sizeof(T) == 4 ? "signature_cuda_f" : "signature_cuda_d"));
    const auto cu_reverse = reinterpret_cast<CuBackward>(GET_FN_PTR(cusig, sizeof(T) == 4 ? "sig_backprop_cuda_f" : "sig_backprop_cuda_d"));
    CudaBuf<T> dp, ds, dg, dy;
    T *p = path.data(), *s = signature.data(), *g = seed.data(), *y = gradient.data();
    if (gpu) {
        dp = CudaBuf<T>(path.size() * sizeof(T)); ds = CudaBuf<T>(signature.size() * sizeof(T));
        dg = CudaBuf<T>(seed.size() * sizeof(T)); dy = CudaBuf<T>(gradient.size() * sizeof(T));
        p = dp.get(); s = ds.get(); g = dg.get(); y = dy.get();
        CUDA_CHECK(cudaMemcpy(p, path.data(), path.size() * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g, seed.data(), seed.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    auto run = [&](bool reverse_pass) {
        int status;
        if (gpu) status = reverse_pass ? cu_reverse(p, y, g, s, batch, d, n + 1, m, false, false, T(1), true, nullptr, 0, 0, 0, 1, 0, false)
            : cu_forward(p, s, batch, d, n + 1, m, false, false, T(1), true, nullptr, 0, 0, 0, 1, 0, false);
        else status = reverse_pass ? reverse(p, y, g, s, batch, d, n + 1, m, false, false, T(1), true, jobs, nullptr, 0, 0, 0, 1, 0)
            : forward(p, s, batch, d, n + 1, m, false, false, T(1), true, jobs, nullptr, 0, 0, 0, 1, 0);
        if (status) throw std::runtime_error("prefix-suffix status " + std::to_string(status));
    };
    run(false);
    for (int i = 0; i < 3; ++i) run(backward);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) run(backward);
    const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iterations;
    std::cout << milliseconds << " ms\n";
    return 0;
}
