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

#include <cuda_runtime.h>
#include <cstdint>

#define SIG_COEF_CUDA_MAX_DEGREE 20

template<class T, bool Backward>
void sig_coef_fixed_metadata_cuda(const T* path, T* out, const T* coefs, const T* deriv,
    const uint64_t* metadata, uint64_t words, uint64_t width, uint64_t max_degree,
    uint64_t batch, uint64_t dimension, uint64_t length, bool prefixes, cudaStream_t stream);

template<class T, bool Backward>
void sig_coef_dispatch_cuda(const T* path, T* out, const T* coefs, const T* deriv,
    const uint64_t* indices, uint64_t words, const uint64_t* degrees, uint64_t batch,
    uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time,
    bool prefixes, int method, void* stream);
