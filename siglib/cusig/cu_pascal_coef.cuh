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

#include "cu_atomic.h"
#include "pascal_coefficient.h"

template<class T, uint64_t Degree, bool Augmented, bool Guarded>
__device__ bool pascal_forward_cuda(const T* path, const uint64_t* channels, T* u,
    uint64_t dimension, uint64_t segments, bool time_aug, bool lead_lag, T end_time) {
    using Ops = pascal_coefficient::ScalarOps<T>;
    u[0] = T(1);
    #pragma unroll
    for (uint64_t j = 1; j <= Degree; ++j) u[j] = T(0);
    for (uint64_t i = 0; i < segments; ++i) {
        T a[Degree];
        #pragma unroll
        for (uint64_t j = 0; j < Degree; ++j) {
            if constexpr (Augmented)
                a[j] = pascal_coefficient::point(path, i + 1, channels[j], dimension, segments, time_aug, lead_lag, end_time)
                    - pascal_coefficient::point(path, i, channels[j], dimension, segments, time_aug, lead_lag, end_time);
            else a[j] = path[(i + 1) * dimension + channels[j]] - path[i * dimension + channels[j]];
        }
        if (!i) {
            #pragma unroll
            for (uint64_t j = 1; j <= Degree; ++j) {
                if constexpr (Guarded) u[j] = Ops::mul_div(a[j - 1], u[j - 1], j);
                else u[j] = a[j - 1] * u[j - 1];
            }
        } else if constexpr (Guarded) {
            #pragma unroll 1
            for (uint64_t s = Degree; s > 0; --s) {
                #pragma unroll
                for (uint64_t j = s; j <= Degree; ++j) u[j] += Ops::mul_div(a[j - 1], u[j - 1], j);
            }
        } else {
            T q = u[Degree - 1];
            #pragma unroll
            for (uint64_t s = Degree - 1; s > 0; --s) {
                #pragma unroll
                for (uint64_t j = s; j < Degree; ++j) u[j] += a[j - 1] * u[j - 1];
                q += u[Degree - 1];
            }
            u[Degree] += a[Degree - 1] * q;
        }
    }
    if constexpr (!Guarded) {
        // A nonfinite accumulator cannot become finite in a later segment.
        if (!Ops::all_finite(u + 1, Degree)) return false;
        T reciprocal = T(1);
        #pragma unroll
        for (uint64_t j = 1; j <= Degree; ++j) {
            reciprocal /= T(j);
            u[j] *= reciprocal;
        }
    }
    return true;
}

template<class T, uint64_t Degree, bool Backward, bool Augmented, bool Guarded = false, bool Merge = false>
__global__ void sig_coef_pascal_kernel_fixed_degree(const T* __restrict__ path, T* __restrict__ out,
    const T* __restrict__ coefs, const T* __restrict__ deriv, const uint64_t* __restrict__ indices,
    const uint64_t* __restrict__ index_offsets, const uint64_t* __restrict__ output_offsets,
    uint64_t words, uint64_t dimension, uint64_t length, uint64_t width, bool prefixes,
    uint64_t offset, uint64_t work, const uint64_t* __restrict__ order,
    bool time_aug, bool lead_lag, T end_time, int* retry) {
    using Ops = pascal_coefficient::ScalarOps<T>;
    const uint64_t local = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= work) return;
    const uint64_t job = offset + local, sample = job / words, word = order[job % words];
    const uint64_t segments = (length - 1) * (Augmented && lead_lag ? 2 : 1);
    const T* x = path + sample * length * dimension;
    uint64_t channels[Degree];
    #pragma unroll
    for (uint64_t j = 0; j < Degree; ++j) channels[j] = indices[index_offsets[word] + j];
    T u[Degree + 1];
    if constexpr (!Backward) {
        if (!pascal_forward_cuda<T, Degree, Augmented, false>(x, channels, u, dimension, segments, time_aug, lead_lag, end_time))
            pascal_forward_cuda<T, Degree, Augmented, true>(x, channels, u, dimension, segments, time_aug, lead_lag, end_time);
        T* result = out + sample * width + output_offsets[word];
        if (prefixes) {
            #pragma unroll
            for (uint64_t j = 1; j <= Degree; ++j) result[j - 1] = u[j];
        } else *result = u[Degree];
    } else {
        const uint64_t start = sample * width + output_offsets[word];
        T lambda[Degree + 1], previous[Degree];
        u[0] = T(1);
        lambda[0] = T(0);
        T factor = T(1), reciprocal = T(1);
        bool valid = true;
        #pragma unroll
        for (uint64_t j = 1; j <= Degree; ++j) {
            u[j] = j < Degree ? coefs[start + j - 1] : T(0);
            lambda[j] = deriv[start + j - 1];
            previous[j - 1] = T(0);
            if constexpr (!Guarded) {
                factor *= T(j);
                reciprocal /= T(j);
                u[j] *= factor;
                const T raw = lambda[j];
                lambda[j] *= reciprocal;
                valid = !Ops::tiny(raw, lambda[j]) && valid;
            }
        }
        if constexpr (!Guarded) {
            if (!valid || !Ops::all_finite(u + 1, Degree - 1)) { atomicExch(retry, 1); return; }
        }
        T* gradient = out + sample * length * dimension;
        uint32_t unique = 0;
        if constexpr (Merge) {
            #pragma unroll
            for (uint64_t j = 0; j < Degree; ++j) {
                bool first = true;
                #pragma unroll
                for (uint64_t k = 0; k < j; ++k) first = first && channels[k] != channels[j];
                if (first) unique |= uint32_t(1) << j;
            }
        }
        for (uint64_t i = segments; i-- > 0;) {
            T a[Degree], current[Degree];
            #pragma unroll
            for (uint64_t j = 0; j < Degree; ++j) {
                if constexpr (Augmented)
                    a[j] = pascal_coefficient::point(x, i + 1, channels[j], dimension, segments, time_aug, lead_lag, end_time)
                        - pascal_coefficient::point(x, i, channels[j], dimension, segments, time_aug, lead_lag, end_time);
                else a[j] = x[(i + 1) * dimension + channels[j]] - x[i * dimension + channels[j]];
                current[j] = T(0);
            }
            if (!i) {
                #pragma unroll
                for (uint64_t j = 1; j < Degree; ++j) {
                    if constexpr (Guarded) u[j] = Ops::mul_div(a[j - 1], u[j - 1], j);
                    else u[j] = a[j - 1] * u[j - 1];
                }
                #pragma unroll
                for (uint64_t j = Degree; j > 0; --j) {
                    if constexpr (Guarded) {
                        current[j - 1] = Ops::mul_div(u[j - 1], lambda[j], j);
                        if (j > 1) lambda[j - 1] += Ops::mul_div(a[j - 1], lambda[j], j);
                    } else {
                        current[j - 1] = u[j - 1] * lambda[j];
                        if (j > 1) lambda[j - 1] += a[j - 1] * lambda[j];
                    }
                }
            } else if constexpr (Guarded) {
                #pragma unroll
                for (uint64_t s = 1; s <= Degree; ++s) {
                    #pragma unroll
                    for (uint64_t j = Degree; j >= s; --j) {
                        current[j - 1] += Ops::mul_div(u[j - 1], lambda[j], j);
                        if (j > 1) lambda[j - 1] += Ops::mul_div(a[j - 1], lambda[j], j);
                        if (j < Degree) u[j] -= Ops::mul_div(a[j - 1], u[j - 1], j);
                    }
                }
            } else {
                T q = u[Degree - 1];
                const T dq = a[Degree - 1] * lambda[Degree];
                #pragma unroll
                for (uint64_t s = 1; s < Degree; ++s) {
                    lambda[Degree - 1] += dq;
                    #pragma unroll
                    for (uint64_t j = Degree - 1; j >= s; --j) {
                        current[j - 1] += u[j - 1] * lambda[j];
                        if (j > 1) lambda[j - 1] += a[j - 1] * lambda[j];
                        u[j] -= a[j - 1] * u[j - 1];
                    }
                    q += u[Degree - 1];
                }
                if constexpr (Degree > 1) lambda[Degree - 1] += dq;
                current[Degree - 1] = lambda[Degree] * q;
            }
            if constexpr (!Guarded) {
                if (!Ops::all_finite(u + 1, Degree - 1) || !Ops::all_finite(lambda + 1, Degree)
                    || !Ops::all_finite(current, Degree)) { atomicExch(retry, 1); return; }
            }
            #pragma unroll
            for (uint64_t j = 0; j < Degree; ++j) {
                if constexpr (Augmented) if (time_aug && channels[j] == dimension * (lead_lag ? 2 : 1)) continue;
                if constexpr (Merge) if (!(unique & (uint32_t(1) << j))) continue;
                const uint64_t vertex = i + 1;
                const uint64_t row = Augmented && lead_lag ? vertex / 2 + ((vertex % 2) && channels[j] >= dimension) : vertex;
                const uint64_t column = Augmented && lead_lag && channels[j] >= dimension ? channels[j] - dimension : channels[j];
                T value = current[j] - previous[j];
                if constexpr (Merge) {
                    #pragma unroll
                    for (uint64_t k = j + 1; k < Degree; ++k)
                        if (channels[k] == channels[j]) value += current[k] - previous[k];
                    if (!Ops::finite(value)) { atomicExch(retry, 1); return; }
                }
                myAtomicAdd(gradient + row * dimension + column, value);
            }
            #pragma unroll
            for (uint64_t j = 0; j < Degree; ++j) previous[j] = current[j];
        }
        #pragma unroll
        for (uint64_t j = 0; j < Degree; ++j) {
            if constexpr (Augmented) if (time_aug && channels[j] == dimension * (lead_lag ? 2 : 1)) continue;
            if constexpr (Merge) if (!(unique & (uint32_t(1) << j))) continue;
            const uint64_t column = Augmented && lead_lag && channels[j] >= dimension ? channels[j] - dimension : channels[j];
            T value = -previous[j];
            if constexpr (Merge) {
                #pragma unroll
                for (uint64_t k = j + 1; k < Degree; ++k)
                    if (channels[k] == channels[j]) value -= previous[k];
                if (!Ops::finite(value)) { atomicExch(retry, 1); return; }
            }
            myAtomicAdd(gradient + column, value);
        }
    }
}
