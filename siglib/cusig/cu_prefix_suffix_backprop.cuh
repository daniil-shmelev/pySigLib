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

namespace prefix_suffix::cuda_detail {

template<typename T>
__device__ T warp_sum(T value) {
    for (int shift = 16; shift; shift /= 2) value += __shfl_down_sync(0xffffffff, value, shift);
    return value;
}

template<bool Prepend, typename T>
__device__ void horner_backprop(const T* state, const T* v, T* deriv, T* dv, uint64_t d, uint64_t h,
    const uint64_t* power, const uint64_t* offset, T* history, T* work) {
    for (uint64_t c = threadIdx.x; c < d; c += blockDim.x) dv[c] += deriv[1 + c];
    __syncthreads();
    for (uint64_t target = 2; target <= h; ++target) {
        for (uint64_t c = threadIdx.x; c < d; c += blockDim.x) history[1 + c] = v[c] / T(target);
        __syncthreads();
        for (uint64_t k = 1; k + 1 < target; ++k) {
            const uint64_t size = power[k];
            for (uint64_t word = threadIdx.x; word < power[k + 1]; word += blockDim.x) {
                const uint64_t parent = Prepend ? word % size : word / d;
                const uint64_t c = Prepend ? word / size : word % d;
                history[offset[k + 1] + word] = (state[offset[k] + parent] + history[offset[k] + parent])
                    * v[c] / T(target - k);
            }
            __syncthreads();
        }
        const T* upper = deriv + offset[target];
        for (uint64_t k = target - 1; k > 0; --k) {
            const uint64_t size = power[k];
            const T reciprocal = T(1) / T(target - k);
            T* lower = work + offset[k];
            for (uint64_t c = threadIdx.x / 32; c < d; c += blockDim.x / 32) {
                T value = 0;
                for (uint64_t x = threadIdx.x % 32; x < size; x += 32) {
                    const uint64_t word = Prepend ? c * size + x : x * d + c;
                    value += upper[word] * (state[offset[k] + x] + history[offset[k] + x]);
                }
                value = warp_sum(value);
                if (!(threadIdx.x % 32)) dv[c] += value * reciprocal;
            }
            for (uint64_t x = threadIdx.x; x < size; x += blockDim.x) {
                T value = 0;
                for (uint64_t c = 0; c < d; ++c)
                    value += upper[Prepend ? c * size + x : x * d + c] * v[c];
                lower[x] = value * reciprocal;
                deriv[offset[k] + x] += lower[x];
            }
            __syncthreads();
            upper = lower;
        }
        for (uint64_t c = threadIdx.x; c < d; c += blockDim.x) dv[c] += work[1 + c] / T(target);
        __syncthreads();
    }
}

template<typename T>
__global__ void factors_backprop_kernel(const T* prefixes, const T* suffixes, const T* increments,
    const T* dleft, const T* dright, T* dprefix, T* dsuffix, T* dincrements, T* scratch,
    uint64_t batch, uint64_t d, uint64_t qlen, uint64_t a, uint64_t b, uint64_t base, uint64_t rows,
    uint64_t slots, uint64_t temporary, uint64_t stride, const uint64_t* power, const uint64_t* offset) {
    const uint64_t first_segment = base / a, segments = (base + rows - 1) / a - first_segment + 1;
    const uint64_t p = power[a], q = power[b];
    T* powers = scratch + uint64_t(blockIdx.x) / slots * stride + (blockIdx.x % slots) * 4 * temporary;
    T* dpowers = powers + temporary;
    T* history = dpowers + temporary;
    T* work = history + temporary;
    for (uint64_t task = blockIdx.x; task < batch * segments; task += gridDim.x) {
        const uint64_t sample = task / segments, i = first_segment + task % segments;
        const T* v = increments + sample * stride + i * d;
        T* dv = dincrements + sample * stride + i * d;
        const T* prefix = prefixes + sample * stride + i * qlen;
        const T* suffix = suffixes + sample * stride + (i + 1) * qlen;
        T* dp = dprefix + sample * stride + i * qlen;
        T* dq = dsuffix + sample * stride + (i + 1) * qlen;
        if (!threadIdx.x) powers[0] = 1;
        for (uint64_t x = threadIdx.x; x < offset[a + 1]; x += blockDim.x) dpowers[x] = 0;
        __syncthreads();
        for (uint64_t k = 1; k <= a; ++k) {
            for (uint64_t x = threadIdx.x; x < power[k]; x += blockDim.x)
                powers[offset[k] + x] = powers[offset[k - 1] + x / d] * v[x % d];
            __syncthreads();
        }
        for (uint64_t t = max(base, i * a); t < min(base + rows, (i + 1) * a); ++t) {
            const uint64_t j = t % a, row = t - base, tail = power[j + 1], head = power[a - 1 - j];
            const uint64_t level = offset[a - 1 - j];
            const T* gl = dleft + sample * stride + row * p;
            const T* gr = dright + sample * stride + row * q;
            for (uint64_t x = threadIdx.x; x < head; x += blockDim.x) {
                T value = 0;
                for (uint64_t y = 0; y < tail; ++y) value += gl[x * tail + y] * powers[offset[j + 1] + y];
                dp[level + x] += value;
            }
            for (uint64_t y = threadIdx.x; y < tail; y += blockDim.x) {
                T value = 0;
                for (uint64_t x = 0; x < head; ++x) value += gl[x * tail + y] * prefix[level + x];
                dpowers[offset[j + 1] + y] += value;
            }
            if (!threadIdx.x) history[0] = 1;
            __syncthreads();
            for (uint64_t k = 1; k < b; ++k) {
                const uint64_t previous = power[k - 1];
                for (uint64_t x = threadIdx.x; x < power[k]; x += blockDim.x)
                    history[offset[k] + x] = suffix[offset[k] + x]
                        + v[x / previous] * history[offset[k - 1] + x % previous] / T(j + b - k + 2);
                __syncthreads();
            }
            T scale = 1;
            for (uint64_t f = 2; f <= j + 1; ++f) scale /= T(f);
            for (uint64_t x = threadIdx.x; x < q; x += blockDim.x) work[offset[b] + x] = gr[x] * scale;
            __syncthreads();
            for (uint64_t k = b; k > 0; --k) {
                const uint64_t previous = power[k - 1];
                const T reciprocal = T(1) / T(j + b - k + 2);
                const T* upper = work + offset[k];
                for (uint64_t x = threadIdx.x; x < power[k]; x += blockDim.x) dq[offset[k] + x] += upper[x];
                for (uint64_t x = threadIdx.x; x < previous; x += blockDim.x) {
                    T value = 0;
                    for (uint64_t c = 0; c < d; ++c) value += upper[c * previous + x] * v[c];
                    work[offset[k - 1] + x] = value * reciprocal;
                }
                for (uint64_t c = threadIdx.x / 32; c < d; c += blockDim.x / 32) {
                    T value = 0;
                    for (uint64_t x = threadIdx.x % 32; x < previous; x += 32)
                        value += upper[c * previous + x] * history[offset[k - 1] + x];
                    value = warp_sum(value);
                    if (!(threadIdx.x % 32)) dv[c] += value * reciprocal;
                }
                __syncthreads();
            }
        }
        for (uint64_t k = a; k > 0; --k) {
            const uint64_t previous = power[k - 1];
            const T* upper = dpowers + offset[k];
            for (uint64_t x = threadIdx.x; x < previous; x += blockDim.x) {
                T value = 0;
                for (uint64_t c = 0; c < d; ++c) value += upper[x * d + c] * v[c];
                dpowers[offset[k - 1] + x] += value;
            }
            for (uint64_t c = threadIdx.x / 32; c < d; c += blockDim.x / 32) {
                T value = 0;
                for (uint64_t x = threadIdx.x % 32; x < previous; x += 32)
                    value += upper[x * d + c] * powers[offset[k - 1] + x];
                value = warp_sum(value);
                if (!(threadIdx.x % 32)) dv[c] += value;
            }
            __syncthreads();
        }
    }
}

template<bool Prepend, typename T>
__device__ void adjoint_step(T* adjoint, const T* v, T* work, uint64_t d, uint64_t h,
    const uint64_t* power, const uint64_t* offset) {
    for (uint64_t target = 2; target <= h; ++target) {
        const T* upper = adjoint + offset[target];
        for (uint64_t k = target - 1; k > 0; --k) {
            const uint64_t size = power[k];
            T* lower = work + offset[k];
            for (uint64_t x = threadIdx.x; x < size; x += blockDim.x) {
                T value = 0;
                for (uint64_t c = 0; c < d; ++c)
                    value += upper[Prepend ? c * size + x : x * d + c] * v[c];
                lower[x] = value / T(target - k);
                adjoint[offset[k] + x] += lower[x];
            }
            __syncthreads();
            upper = lower;
        }
    }
}

template<bool Prepend, typename T>
__global__ void scan_adjoint_seeds_kernel(const T* increments, const T* sources, const T* deriv,
    T* summaries, T* scratch, uint64_t batch, uint64_t d, uint64_t n, uint64_t h, uint64_t qlen,
    uint64_t chunks, uint64_t chunk_length, uint64_t stride, uint64_t sig_stride, bool scalar,
    const uint64_t* power, const uint64_t* offset, bool shared) {
    extern __shared__ unsigned char raw[];
    for (uint64_t task = blockIdx.x; task < batch * chunks; task += gridDim.x) {
        const uint64_t sample = task / chunks, chunk = task % chunks;
        const uint64_t first = chunk * chunk_length, end = min(n, first + chunk_length);
        T* adjoint = shared ? reinterpret_cast<T*>(raw) : scratch + sample * stride + chunk * 3 * qlen;
        T* work = adjoint + qlen;
        for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x)
            adjoint[x] = !Prepend && end == n && x ? deriv[sample * sig_stride + x - !scalar] : T(0);
        __syncthreads();
        for (uint64_t t = first; t < end; ++t) {
            const uint64_t i = Prepend ? t : end - 1 - (t - first), vertex = Prepend ? i + 1 : i;
            adjoint_step<Prepend>(adjoint, increments + sample * stride + i * d, work, d, h, power, offset);
            for (uint64_t x = threadIdx.x + 1; x < qlen; x += blockDim.x)
                adjoint[x] += sources[sample * stride + vertex * qlen + x];
            __syncthreads();
        }
        for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x)
            summaries[sample * stride + chunk * qlen + x] = adjoint[x];
        __syncthreads();
    }
}

template<bool Prepend, typename T>
__global__ void summary_adjoint_kernel(const T* signatures, const T* in, T* out, uint64_t batch,
    uint64_t chunks, uint64_t qlen, uint64_t stride, uint64_t shift, uint64_t h,
    const uint64_t* power, const uint64_t* offset) {
    const uint64_t total = batch * chunks * qlen;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < total; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / (chunks * qlen), chunk = x / qlen % chunks, word = x % qlen;
        const uint64_t position = sample * stride + chunk * qlen;
        T value = in[position + word];
        if (word && (Prepend ? chunk >= shift : chunk + shift < chunks)) {
            const T* other = in + sample * stride + (Prepend ? chunk - shift : chunk + shift) * qlen;
            const T* sig = signatures + position;
            uint64_t level = 1;
            while (word >= offset[level + 1]) ++level;
            const uint64_t index = word - offset[level];
            for (uint64_t k = 0; k <= h - level; ++k)
                for (uint64_t tail = 0; tail < power[k]; ++tail) {
                    const uint64_t coefficient = Prepend ? tail * power[level] + index : index * power[k] + tail;
                    value += other[offset[level + k] + coefficient] * sig[offset[k] + tail];
                }
        }
        out[position + word] = value;
    }
}

template<bool Prepend, typename T>
__global__ void scans_backprop_kernel(const T* states, const T* increments, const T* sources,
    const T* boundary, const T* deriv, T* dincrements, T* scratch, uint64_t batch, uint64_t d, uint64_t n,
    uint64_t h, uint64_t qlen, uint64_t chunks, uint64_t chunk_length, uint64_t stride, uint64_t sig_stride,
    bool scalar, const uint64_t* power, const uint64_t* offset, bool shared) {
    extern __shared__ unsigned char raw[];
    for (uint64_t task = blockIdx.x; task < batch * chunks; task += gridDim.x) {
        const uint64_t sample = task / chunks, chunk = task % chunks;
        const uint64_t first = chunk * chunk_length, end = min(n, first + chunk_length);
        const bool has_boundary = Prepend ? chunk != 0 : chunk + 1 < chunks;
        T* adjoint = shared ? reinterpret_cast<T*>(raw) : scratch + sample * stride + chunk * 3 * qlen;
        T* history = adjoint + qlen; T* work = history + qlen;
        for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x)
            adjoint[x] = has_boundary ? boundary[sample * stride + (Prepend ? chunk - 1 : chunk + 1) * qlen + x]
                : (!Prepend && x ? deriv[sample * sig_stride + x - !scalar] : T(0));
        __syncthreads();
        for (uint64_t t = first; t < end; ++t) {
            const uint64_t i = Prepend ? t : end - 1 - (t - first), vertex = Prepend ? i + 1 : i;
            horner_backprop<Prepend>(states + sample * stride + vertex * qlen, increments + sample * stride + i * d,
                adjoint, dincrements + sample * stride + i * d, d, h, power, offset, history, work);
            for (uint64_t x = threadIdx.x + 1; x < qlen; x += blockDim.x)
                adjoint[x] += sources[sample * stride + vertex * qlen + x];
            __syncthreads();
        }
    }
}

template<typename T>
__global__ void path_backprop_kernel(const T* dincrements, T* out, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t d, uint64_t n, uint64_t stride, bool lead_lag) {
    const uint64_t path_stride = length * dimension;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * path_stride; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / path_stride, point = x / dimension % length, channel = x % dimension;
        const T* g = dincrements + sample * stride;
        T value = 0;
        if (!lead_lag) {
            if (point) value += g[(point - 1) * d + channel];
            if (point < n) value -= g[point * d + channel];
        } else {
            if (point) {
                value += g[(2 * point - 1) * d + channel];
                value += g[(2 * point - 2) * d + dimension + channel];
            }
            if (point + 1 < length) {
                value -= g[(2 * point + 1) * d + channel];
                value -= g[2 * point * d + dimension + channel];
            }
        }
        out[x] = value;
    }
}

template<typename T>
__global__ void trivial_backprop_kernel(const T* path, const T* deriv, T* out, uint64_t batch,
    uint64_t dimension, uint64_t length, uint64_t d, uint64_t degree, uint64_t sig_stride, bool scalar, bool lead_lag) {
    const uint64_t path_stride = length * dimension;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * path_stride; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / path_stride, point = x / dimension % length, c = x % dimension;
        T value = 0;
        if (degree && length > 1 && (point == 0 || point + 1 == length)) {
            const T* g = deriv + sample * sig_stride + (scalar ? 1 : 0);
            value = g[c];
            if (lead_lag) value += g[dimension + c];
            if (d == 1) {
                const T v = path[sample * path_stride + length - 1] - path[sample * path_stride];
                T power = 1;
                for (uint64_t k = 2; k <= degree; ++k) { power *= v / T(k - 1); value += g[k - 1] * power; }
            }
            if (!point) value = -value;
        }
        out[x] = value;
    }
}

} // namespace prefix_suffix::cuda_detail
