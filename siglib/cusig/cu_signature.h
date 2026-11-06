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

#include <cstdint>
#include <cuda_runtime.h>

template<bool Backward, typename T>
void signature_prefix_suffix_correction_cuda(const T* path, T* out, const T* deriv, uint64_t batch, uint64_t dimension, uint64_t length, uint64_t degree, bool ta, T end_time, bool scalar, cudaStream_t stream, uint64_t split, const T* correction, uint64_t correction_len, uint64_t batch_stride, uint64_t segment_stride);

template<typename T>
void signature_prefix_suffix_cuda(const T* path, T* out, uint64_t batch, uint64_t dimension, uint64_t length,
    uint64_t degree, bool time_aug, bool lead_lag, T end_time, bool scalar, cudaStream_t stream, uint64_t split_a);

template<typename T>
void signature_prefix_suffix_backprop_cuda(const T* path, T* out, const T* deriv, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, T end_time, bool scalar, cudaStream_t stream, uint64_t split_a);
