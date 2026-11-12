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

#include "../siglib/cusig/cu_pascal_coef.cuh"

template<class T>
int launch(const T* path, T* out, const T* coefs, const T* seeds, const uint64_t* metadata,
    int* retry, uint64_t batch, uint64_t words, uint64_t dimension, uint64_t length,
    uint64_t degree, bool backward, unsigned int threads) {
    const uint64_t work = batch * words;
    const auto blocks = static_cast<unsigned int>((work + threads - 1) / threads);
    if (backward) {
        auto error = cudaMemsetAsync(out, 0, batch * length * dimension * sizeof(T));
        if (error != cudaSuccess) return error;
    }
    auto run = [&]<uint64_t Degree>() {
        auto kernel = [&]<bool Backward>() {
            auto grouped = [&]<bool Merge>() {
                sig_coef_pascal_kernel_fixed_degree<T, Degree, Backward, false, false, Merge><<<blocks, threads>>>(
                    path, out, coefs, seeds, metadata + 3 * words, metadata, metadata + words,
                    words, dimension, length, words * degree, true, 0, work, metadata + 2 * words,
                    false, false, T(1), retry);
            };
            if constexpr (Backward && sizeof(T) == sizeof(float) && Degree >= 8) {
                if (work >= 4096 && dimension <= 2) grouped.template operator()<true>();
                else grouped.template operator()<false>();
            } else grouped.template operator()<false>();
        };
        if (backward) kernel.template operator()<true>();
        else kernel.template operator()<false>();
    };
    switch (degree) {
        case 2: run.template operator()<2>(); break;
        case 4: run.template operator()<4>(); break;
        case 8: run.template operator()<8>(); break;
        case 12: run.template operator()<12>(); break;
        case 16: run.template operator()<16>(); break;
        case 20: run.template operator()<20>(); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define ENTRY(T, S) \
extern "C" EXPORT int pascal_tune_##S(const T* path, T* out, const T* coefs, const T* seeds, const uint64_t* metadata, \
    int* retry, uint64_t batch, uint64_t words, uint64_t dimension, uint64_t length, uint64_t degree, bool backward, unsigned int threads) { \
    return launch(path, out, coefs, seeds, metadata, retry, batch, words, dimension, length, degree, backward, threads); \
}
ENTRY(float, f)
ENTRY(double, d)
