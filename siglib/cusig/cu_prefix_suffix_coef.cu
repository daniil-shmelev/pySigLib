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

#include "cu_sig_coef.h"
#include "cu_runtime_utils.h"
#include "cu_utils.h"
#include "cu_atomic.h"
#include "cu_macros.h"
#include "word_coefficient.h"
#include <memory>

namespace {
namespace pc = word_coefficient;
constexpr unsigned scan_threads = 128;

struct WordPacket {
    static constexpr unsigned capacity = 16;
    uint64_t outputs[capacity];
    unsigned channels[capacity * SIG_COEF_CUDA_MAX_DEGREE];
    const uint64_t* table;
    uint64_t count, first;
    __device__ __forceinline__ uint64_t output(uint64_t w) const {
        return table ? table[count + table[2 * count + first + w]] : outputs[w];
    }
    __device__ __forceinline__ uint64_t channel(uint64_t w, unsigned degree, unsigned j) const {
        return table ? table[3 * count + table[table[2 * count + first + w]] + j] : channels[w * degree + j];
    }
};

struct ScanShape {
    uint64_t words, dimension, length, width, segments;
    bool prefixes, time_aug, lead_lag, terminal_seeds;
};

struct FreeScratch {
    cudaStream_t stream;
    bool ordered = true;
    void operator()(void* ptr) const {
        if (ordered) (void)cudaFreeAsync(ptr, stream);
        else (void)cudaFree(ptr);
    }
};

struct MappedWorkspace {
    uint64_t* host = nullptr;
    uint64_t capacity = 0;
    void* scratch = nullptr;
    uint64_t scratch_bytes = 0;
    int device_id = -1;
    void release_scratch() noexcept {
        if (!scratch) return;
        int active = -1;
        if (cudaGetDevice(&active) == cudaSuccess && (device_id == active || cudaSetDevice(device_id) == cudaSuccess)) {
            (void)cudaFree(scratch);
            if (active != device_id) (void)cudaSetDevice(active);
        }
        scratch = nullptr;
        scratch_bytes = 0;
    }
    ~MappedWorkspace() {
        release_scratch();
        if (host) (void)cudaFreeHost(host);
    }
    void* small_scratch(uint64_t bytes) {
        int active;
        CUDA_CHECK(cudaGetDevice(&active));
        if (scratch && (active != device_id || bytes > scratch_bytes)) release_scratch();
        if (bytes > (4u << 20)) return nullptr;
        if (!scratch) {
            const auto status = cudaMalloc(&scratch, bytes);
            if (status == cudaErrorMemoryAllocation) { (void)cudaGetLastError(); return nullptr; }
            CUDA_CHECK(status);
            device_id = active;
            scratch_bytes = bytes;
        }
        return scratch;
    }
    uint64_t* acquire(uint64_t elements) {
        if (elements > capacity) {
            uint64_t* fresh = nullptr;
            const auto status = cudaHostAlloc(&fresh, pc::mul(elements, sizeof(uint64_t)), cudaHostAllocMapped | cudaHostAllocPortable);
            if (status == cudaErrorMemoryAllocation || status == cudaErrorNotSupported) {
                (void)cudaGetLastError();
                return nullptr;
            }
            CUDA_CHECK(status);
            if (host) (void)cudaFreeHost(host);
            host = fresh;
            capacity = elements;
        }
        uint64_t* device = nullptr;
        const auto status = cudaHostGetDevicePointer(&device, host, 0);
        if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
            (void)cudaGetLastError();
            return nullptr;
        }
        CUDA_CHECK(status);
        *reinterpret_cast<int*>(host) = 0;
        return device;
    }
};

MappedWorkspace& packet_workspace() {
    static thread_local MappedWorkspace workspace;
    return workspace;
}

template<bool Reverse, class T>
__device__ __forceinline__ T segment_midpoint(T z, T& carry, T* totals) {
    constexpr unsigned warps = scan_threads / 32;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    T sum = z;
    #pragma unroll
    for (unsigned s = 1; s < 32; s *= 2) {
        const T v = Reverse ? __shfl_down_sync(0xffffffff, sum, s) : __shfl_up_sync(0xffffffff, sum, s);
        if (Reverse ? lane + s < 32 : lane >= s) sum += v;
    }
    if (lane == (Reverse ? 0 : 31)) totals[warp] = sum;
    const T previous = Reverse ? __shfl_down_sync(0xffffffff, sum, 1) : __shfl_up_sync(0xffffffff, sum, 1);
    __syncthreads();
    T block = lane < warps ? totals[lane] : T(0);
    #pragma unroll
    for (unsigned s = 1; s < warps; s *= 2) {
        const T v = Reverse ? __shfl_down_sync(0xffffffff, block, s) : __shfl_up_sync(0xffffffff, block, s);
        if (Reverse ? lane + s < warps : lane >= s) block += v;
    }
    const T w = __shfl_sync(0xffffffff, block, (warp + (Reverse ? 1 : warps - 1)) % warps);
    const T before = carry + (warp == (Reverse ? warps - 1 : 0) ? T(0) : w)
        + (lane == (Reverse ? 31 : 0) ? T(0) : previous);
    carry += __shfl_sync(0xffffffff, block, Reverse ? 0 : warps - 1);
    __syncthreads();
    return before + T(.5) * z;
}

template<int Degree>
constexpr __host__ __device__ int modes(int level) {
    return 1 + (level < Degree - 1 - level ? level : Degree - 1 - level);
}

// The two sets of Legendre moments use reciprocal scalings. Their ordinary
// dot product integrates the prefix times its adjoint on one segment.
template<int Count, int Previous, bool Dual, class T>
__device__ __forceinline__ void integrate(T* moment, T v, T& carry, T* totals) {
    const T mid = segment_midpoint<Dual>(v * moment[0], carry, totals);
    T lo = moment[0];
    if constexpr (Previous > 1) moment[0] = mid - (T(Dual ? 1 : 1. / 12.) * moment[1]) * v;
    else moment[0] = mid;
    #pragma unroll
    for (int r = 1; r < Count; ++r) {
        const T old = moment[r], hi = r + 1 < Previous ? moment[r + 1] : T(0);
        if constexpr (Dual) moment[r] = (lo * (T(1) / T((4 * r - 2) * (4 * r + 2))) - hi) * v;
        else moment[r] = (lo - hi * (T(1) / T((4 * r + 2) * (4 * r + 6)))) * v;
        lo = old;
    }
}

template<int D, int J, int Offset, bool Save, bool Last, bool Short = false, class T>
__device__ __forceinline__ void prefix_rows(T* moment, const T* dx, T* carry, T* tape, T* totals, int length = D) {
    if constexpr (Short) if (J >= length) return;
    constexpr int count = modes<D>(J);
    if constexpr (Save) {
        #pragma unroll
        for (int r = 0; r < count; ++r) tape[Offset + r] = r % 2 ? -moment[r] : moment[r];
    }
    if constexpr (J + 1 < D) {
        integrate<modes<D>(J + 1), count, false>(moment, dx[J], carry[J + 1], totals);
        prefix_rows<D, J + 1, Offset + count, Save, Last, Short>(moment, dx, carry, tape, totals, length);
    } else if constexpr (Last) integrate<1, count, false>(moment, dx[J], carry[D], totals);
}

template<int D, int J, int End, class T>
__device__ __forceinline__ void suffix_rows(T* moment, const T* dx, T* carry, const T* tape, T* grad, T* totals) {
    constexpr int count = modes<D>(J - 1);
    T value = tape[End - count] * moment[0];
    #pragma unroll
    for (int r = 1; r < count; ++r) value += tape[End - count + r] * moment[r];
    grad[J - 1] = value;
    if constexpr (J > 1) {
        integrate<modes<D>(J - 2), count, true>(moment, dx[J - 1], carry[J - 1], totals);
        suffix_rows<D, J - 1, End - count>(moment, dx, carry, tape, grad, totals);
    }
}

template<class T, int D>
__device__ __forceinline__ void increments(const T* x, uint64_t i, const unsigned (&channels)[D], T (&dx)[D],
    const ScanShape& s, T end_time, bool& valid) {
    #pragma unroll
    for (int j = 0; j < D; ++j) {
        T v = 0;
        if (i < s.segments) {
            const unsigned c = channels[j];
            if (s.time_aug || s.lead_lag)
                v = pc::point(x, i + 1, c, s.dimension, s.segments, s.time_aug, s.lead_lag, end_time)
                  - pc::point(x, i, c, s.dimension, s.segments, s.time_aug, s.lead_lag, end_time);
            else v = x[(i + 1) * s.dimension + c] - x[i * s.dimension + c];
        }
        dx[j] = v;
        const T mag = abs(v);
        valid = valid && mag <= T(65536) && (v == 0 || mag >= T(1e-20));
    }
}

template<class T, int D, bool Backward>
__global__ void scan_word(const T* path, T* out, const T* coefs, const T* seeds, const WordPacket packet,
    ScanShape s, T time, uint64_t offset, T* external, int* retry, const T* boundaries = nullptr) {
    constexpr int size = ((D + 1) / 2) * ((D + 2) / 2);
    __shared__ T totals[scan_threads / 32], edges[Backward ? scan_threads / 32 : 1][Backward ? D : 1];
    extern __shared__ double tile_storage[];
    const uint64_t pieces = (s.segments - 1) / scan_threads + 1;
    const uint64_t job = offset + (boundaries ? blockIdx.x / pieces : blockIdx.x), sample = job / s.words, w = job % s.words;
    const uint64_t pos = sample * s.width + packet.output(w);
    const uint64_t tiles = boundaries ? 1 : pieces;
    T* checkpoints = external ? external + uint64_t(blockIdx.x) * (tiles - 1) * (D - 1) : reinterpret_cast<T*>(tile_storage);
    const T* x = path + sample * s.length * s.dimension;
    unsigned channel[D];
    bool valid = true;
    T primal[D + 1] = {}, dual[D + 1], next[D], invfact = 1;
    #pragma unroll
    for (int j = 0; j < D; ++j) {
        channel[j] = unsigned(packet.channel(w, D, j));
        if constexpr (Backward) {
            const T seed = s.terminal_seeds ? (j + 1 == D ? seeds[pos] : T(0)) : seeds[pos + j];
            dual[j + 1] = boundaries ? boundaries[uint64_t(blockIdx.x) * 2 * D + D + j] : seed;
            invfact /= T(j + 1);
            valid = valid && (!coefs || isfinite(coefs[pos + j])) && isfinite(seed) && isfinite(dual[j + 1]) &&
                !pc::ScalarOps<T>::tiny(seed, seed * invfact);
            next[j] = T(0);
        }
    }
    if constexpr (Backward) {
        for (uint64_t block = 0; block + 1 < tiles; ++block) {
            T dx[D], moment[(D + 1) / 2 + 1] = {};
            increments(x, block * scan_threads + threadIdx.x, channel, dx, s, time, valid);
            moment[0] = 1;
            prefix_rows<D, 0, 0, false, false>(moment, dx, primal, static_cast<T*>(nullptr), totals);
            if (!threadIdx.x) {
                #pragma unroll
                for (int j = 1; j < D; ++j) checkpoints[block * (D - 1) + j - 1] = primal[j];
            }
        }
        __syncthreads();
    }
    for (uint64_t t = 0; t < tiles; ++t) {
        const uint64_t block = boundaries ? blockIdx.x % pieces : Backward ? tiles - 1 - t : t, i = block * scan_threads + threadIdx.x;
        T dx[D], moment[(D + 1) / 2 + 1] = {}, tape[Backward ? size : 1];
        increments(x, i, channel, dx, s, time, valid);
        if constexpr (Backward) {
            #pragma unroll
            for (int j = 1; j < D; ++j) primal[j] = boundaries ? boundaries[uint64_t(blockIdx.x) * 2 * D + j - 1]
                : block ? checkpoints[(block - 1) * (D - 1) + j - 1] : T(0);
        }
        moment[0] = 1;
        prefix_rows<D, 0, 0, Backward, !Backward>(moment, dx, primal, tape, totals);
        if constexpr (Backward) {
            moment[0] = dual[D];
            T grad[D];
            suffix_rows<D, D, size>(moment, dx, dual, tape, grad, totals);
            const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
            if (!lane) {
                #pragma unroll
                for (int j = 0; j < D; ++j) edges[warp][j] = grad[j];
            }
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                const T neighbour = __shfl_down_sync(0xffffffff, grad[j], 1);
                const T after = lane < 31 ? neighbour : (warp + 1 < scan_threads / 32 ? edges[warp + 1][j] : next[j]);
                const T value = grad[j] - (i + 1 == s.segments ? T(0) : after);
                next[j] = edges[0][j];
                valid = valid && isfinite(value);
                const unsigned c = channel[j];
                if (i >= s.segments || (s.time_aug && c == s.dimension * (s.lead_lag ? 2 : 1))) continue;
                const bool lead = s.lead_lag && c >= s.dimension;
                const uint64_t row = s.lead_lag ? (i + 1) / 2 + (((i + 1) % 2) && lead) : i + 1;
                T* dst = out + sample * s.length * s.dimension + (lead ? c - s.dimension : c);
                myAtomicAdd(dst + row * s.dimension, value);
                if (!i || (boundaries && !threadIdx.x)) {
                    const uint64_t before = s.lead_lag ? i / 2 + ((i % 2) && lead) : i;
                    myAtomicAdd(dst + before * s.dimension, -grad[j]);
                }
            }
            __syncthreads();
        }
    }
    if constexpr (!Backward) if (!threadIdx.x) {
        #pragma unroll
        for (int j = 1; j <= D; ++j) {
            valid = valid && isfinite(primal[j]);
            if (s.prefixes || j == D) out[pos + (s.prefixes ? j - 1 : 0)] = primal[j];
        }
    }
    if (__syncthreads_or(!valid) && !threadIdx.x) atomicExch(retry, 1);
}

template<int D, int J, bool Dual, class T>
__device__ __forceinline__ void half_word(T* moment, const T* dx, T* carry, T* totals) {
    constexpr int centre = (D - 1) / 2;
    if constexpr (J != centre) {
        constexpr int next = Dual ? J - 1 : J + 1;
        integrate<modes<D>(next), modes<D>(J), Dual>(moment, dx[J], carry[Dual ? J : J + 1], totals);
        half_word<D, next, Dual>(moment, dx, carry, totals);
    }
}

template<class T>
__device__ __forceinline__ T tile_sum(T value, T* totals) {
    const unsigned lane = threadIdx.x % 32;
    #pragma unroll
    for (unsigned k = 16; k; k /= 2) value += __shfl_down_sync(0xffffffff, value, k);
    if (!lane) totals[threadIdx.x / 32] = value;
    __syncthreads();
    value = lane < scan_threads / 32 ? totals[lane] : T(0);
    #pragma unroll
    for (unsigned k = scan_threads / 64; k; k /= 2) value += __shfl_down_sync(0xffffffff, value, k);
    __syncthreads();
    return value;
}

template<class T, int D>
__global__ void paired_word(const T* path, T* out, WordPacket packet, ScanShape s, T time, uint64_t offset,
    T* external, int* retry, const T* boundary = nullptr) {
    constexpr int centre = (D - 1) / 2;
    __shared__ T totals[scan_threads / 32];
    extern __shared__ double tile_storage[];
    const uint64_t pieces = (s.segments - 1) / scan_threads + 1;
    const uint64_t job = offset + (boundary ? blockIdx.x / pieces : blockIdx.x), w = job % s.words;
    const T* x = path + (job / s.words) * s.length * s.dimension;
    const uint64_t tiles = boundary ? 1 : pieces;
    T* saved = external ? external + uint64_t(blockIdx.x) * (tiles - 1) * centre : reinterpret_cast<T*>(tile_storage);
    unsigned channel[D];
    #pragma unroll
    for (int j = 0; j < D; ++j) channel[j] = unsigned(packet.channel(w, D, j));
    bool valid = true;
    T primal[D + 1] = {}, dual[D + 1] = {}, sum = 0;
    if constexpr (centre) for (uint64_t k = 0; k + 1 < tiles; ++k) {
        T dx[D], p[(D + 1) / 2 + 1] = {T(1)};
        increments(x, k * scan_threads + threadIdx.x, channel, dx, s, time, valid);
        half_word<D, 0, false>(p, dx, primal, totals);
        if (!threadIdx.x) for (int j = 1; j <= centre; ++j) saved[k * centre + j - 1] = primal[j];
    }
    __syncthreads();
    dual[D] = 1;
    if (boundary) for (int j = centre + 1; j <= D; ++j) dual[j] = boundary[uint64_t(blockIdx.x) * 2 * D + D + j - 1];
    for (uint64_t t = tiles; t-- > 0;) {
        const uint64_t k = boundary ? blockIdx.x % pieces : t;
        T dx[D], p[(D + 1) / 2 + 1] = {T(1)}, q[(D + 1) / 2 + 1] = {dual[D]};
        increments(x, k * scan_threads + threadIdx.x, channel, dx, s, time, valid);
        #pragma unroll
        for (int j = 1; j <= centre; ++j) primal[j] = boundary ? boundary[uint64_t(blockIdx.x) * 2 * D + j - 1] :
            k ? saved[(k - 1) * centre + j - 1] : T(0);
        half_word<D, 0, false>(p, dx, primal, totals);
        half_word<D, D - 1, true>(q, dx, dual, totals);
        T dot = p[0] * q[0];
        #pragma unroll
        for (int r = 1; r <= centre; ++r) dot += (r % 2 ? -p[r] : p[r]) * q[r];
        sum += dx[centre] * dot;
    }
    sum = tile_sum(sum, totals);
    if (!threadIdx.x) {
        out[boundary ? blockIdx.x : (job / s.words) * s.width + packet.output(w)] = sum;
        valid = valid && isfinite(sum);
    }
    if (__syncthreads_or(!valid) && !threadIdx.x) atomicExch(retry, 1);
}

template<class T>
__global__ void sum_chunks(const T* partial, T* out, WordPacket packet, ScanShape s, uint64_t tiles, uint64_t offset, int* retry) {
    __shared__ T totals[scan_threads / 32];
    T value = 0;
    for (uint64_t k = threadIdx.x; k < tiles; k += scan_threads) value += partial[uint64_t(blockIdx.x) * tiles + k];
    value = tile_sum(value, totals);
    if (!threadIdx.x) {
        const uint64_t job = offset + blockIdx.x;
        out[(job / s.words) * s.width + packet.output(job % s.words)] = value;
        if (!isfinite(value)) atomicExch(retry, 1);
    }
}

// Each thread owns a segment. A block reuses its path and increment gradients across words.
template<int Count, int Previous, bool Dual, class T>
__device__ __forceinline__ void path_moments(T* moment, T v, T seed, T* totals) {
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const T z = v * moment[0];
    T sum = z;
    #pragma unroll
    for (int k = 1; k < 32; k *= 2) {
        const T other = Dual ? __shfl_down_sync(0xffffffff, sum, k) : __shfl_up_sync(0xffffffff, sum, k);
        if (Dual ? lane + k < 32 : lane >= k) sum += other;
    }
    if (lane == (Dual ? 0 : 31)) totals[warp] = sum;
    __syncthreads();
    for (unsigned w = Dual ? warp + 1 : 0; w < (Dual ? blockDim.x / 32 : warp); ++w)
        seed += totals[w];
    T lo = moment[0];
    moment[0] = seed + (sum - T(.5) * z);
    if constexpr (Previous > 1) moment[0] -= (T(Dual ? 1 : 1. / 12.) * moment[1]) * v;
    #pragma unroll
    for (int r = 1; r < Count; ++r) {
        const T old = moment[r], hi = r + 1 < Previous ? moment[r + 1] : T(0);
        if constexpr (Dual) moment[r] = (lo * (T(1) / T((4 * r - 2) * (4 * r + 2))) - hi) * v;
        else moment[r] = (lo - hi * (T(1) / T((4 * r + 2) * (4 * r + 6)))) * v;
        lo = old;
    }
}

template<int D, int J, int Offset, bool Dual, bool Backward, class T>
__device__ __forceinline__ void path_rows(T* moments, const T* dx, const unsigned* channels, const T* seeds,
    T* tape, T* grad, unsigned pitch, T* totals) {
    constexpr int count = modes<D>(J), centre = (D - 1) / 2;
    if constexpr (Backward) {
        if constexpr (Dual) {
            T value = tape[Offset] * moments[0];
            #pragma unroll
            for (int r = 1; r < count; ++r) value += tape[Offset + r] * moments[r];
            grad[channels[J] * pitch + threadIdx.x] += value;
        } else {
            #pragma unroll
            for (int r = 0; r < count; ++r) tape[Offset + r] = r % 2 ? -moments[r] : moments[r];
        }
    }
    if constexpr (Dual ? J > (Backward ? 0 : centre) : J < (Backward ? D - 1 : centre)) {
        constexpr int next = Dual ? J - 1 : J + 1;
        constexpr int offset = Dual ? Offset - modes<D>(next) : Offset + count;
        path_moments<modes<D>(next), count, Dual>(moments, dx[J], Dual && Backward && seeds ? seeds[next] : T(0),
            totals + 32 * (J % 2));
        path_rows<D, next, offset, Dual, Backward>(moments, dx, channels, seeds, tape, grad, pitch, totals);
    }
}

template<class T, int D, bool Backward>
__global__ __launch_bounds__(D <= 10 ? 512 : 256, (sizeof(T) == 4 && D > 10 && D <= 16 ? 2 : 1))
void path_words(const T* path, T* out, const T* coefs, const T* seeds, WordPacket packet,
    ScanShape s, uint64_t first, unsigned group_words, bool all_words, bool direct, int* retry) {
    constexpr int centre = (D - 1) / 2, modes_max = (D + 1) / 2, tape_size = modes_max * ((D + 2) / 2);
    const unsigned t = threadIdx.x, n = unsigned(s.segments), dim = unsigned(s.dimension), pitch = blockDim.x + 1;
    const unsigned groups = (unsigned(s.words) + group_words - 1) / group_words;
    const uint64_t job = first + blockIdx.x, sample = job / groups, stride = s.length * dim;
    const unsigned begin = unsigned(job % groups) * group_words;
    const T* x = path + sample * stride;
    extern __shared__ double storage[];
    T* increments = reinterpret_cast<T*>(storage);
    T* grads = Backward ? increments + (direct ? 0 : dim * pitch) : nullptr;
    __shared__ T totals[2 * 32];
    bool valid = true;
    for (unsigned k = t; k < n * dim; k += blockDim.x) {
        const unsigned i = k / dim, c = k % dim;
        const T v = x[k + dim] - x[k];
        if (!direct) increments[c * pitch + i] = v;
        valid = valid && abs(v) <= T(65536) && (v == 0 || abs(v) >= T(1e-20));
    }
    __syncthreads();
    if constexpr (Backward) {
        for (unsigned c = 0; c < dim; ++c) grads[c * pitch + t] = 0;
    }
    for (unsigned w = begin; w < s.words && w < begin + group_words; ++w) {
        const uint64_t pos = sample * s.width + packet.output(w);
        T dx[D], p[modes_max] = {T(1)}, q[modes_max] = {}, tape[Backward ? tape_size : 1];
        unsigned channels[D];
        const T* adjoint = Backward && !s.terminal_seeds ? seeds + pos : nullptr;
        const T terminal = Backward ? seeds[pos + (s.terminal_seeds ? 0 : D - 1)] : T(1);
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            channels[j] = unsigned(packet.channel(w, D, j));
            dx[j] = t >= n ? T(0) : direct ? x[(t + 1) * dim + channels[j]] - x[t * dim + channels[j]] :
                increments[channels[j] * pitch + t];
        }
        if constexpr (Backward) if (!t) {
            T invfact = T(1);
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                invfact /= T(j + 1);
                if (adjoint) valid = valid && isfinite(coefs[pos + j]) && isfinite(adjoint[j]) &&
                    !pc::ScalarOps<T>::tiny(adjoint[j], adjoint[j] * invfact);
            }
            valid = valid && isfinite(terminal) && !pc::ScalarOps<T>::tiny(terminal, terminal * invfact);
        }
        if constexpr (D > 4) {
            q[0] = terminal;
            path_rows<D, 0, 0, false, Backward>(p, dx, channels, adjoint, tape, grads, pitch, totals);
            __syncthreads();
            path_rows<D, D - 1, tape_size - 1, true, Backward>(q, dx, channels, adjoint, tape, grads, pitch, totals);
            __syncthreads();
        } else if constexpr (D >= 2) {
            if constexpr (Backward || centre) {
                p[0] = (x[(t < n ? t : n) * dim + channels[0]] - x[channels[0]]) + T(.5) * dx[0];
                if constexpr (modes<D>(1) > 1) p[1] = dx[0];
                if constexpr (Backward) tape[0] = T(1);
                path_rows<D, 1, 1, false, Backward>(p, dx, channels, adjoint, tape, grads, pitch, totals);
                __syncthreads();
            }
            const unsigned last = channels[D - 1];
            const T mean = (x[n * dim + last] - x[(t < n ? t + 1 : n) * dim + last]) + T(.5) * dx[D - 1];
            q[0] = (adjoint ? adjoint[D - 2] : T(0)) + terminal * mean;
            if constexpr (modes<D>(D - 2) > 1) q[1] = T(1. / 12.) * terminal * dx[D - 1];
            if constexpr (Backward) grads[last * pitch + t] += tape[tape_size - 1] * terminal;
            path_rows<D, D - 2, tape_size - 1 - modes<D>(D - 2), true, Backward>(
                q, dx, channels, adjoint, tape, grads, pitch, totals);
            __syncthreads();
        }
        if constexpr (!Backward) {
            T dot = p[0] * q[0];
            #pragma unroll
            for (int r = 1; r <= centre; ++r) dot += (r % 2 ? -p[r] : p[r]) * q[r];
            T value = dot * dx[centre];
            #pragma unroll
            for (int k = 16; k; k /= 2) value += __shfl_down_sync(0xffffffff, value, k);
            if (!(t % 32)) totals[t / 32] = value;
            __syncthreads();
            if (!t) {
                value = totals[0];
                for (unsigned warp = 1; warp < blockDim.x / 32; ++warp) value += totals[warp];
                valid = valid && isfinite(value);
                out[pos] = value;
            }
            __syncthreads();
        }
    }
    if constexpr (Backward) {
        __syncthreads();
        for (unsigned k = t; k < (n + 1) * dim; k += blockDim.x) {
            const unsigned i = k / dim, c = k % dim;
            const T* g = grads + c * pitch;
            const T value = (i ? g[i - 1] : T(0)) - (i < n ? g[i] : T(0));
            valid = valid && isfinite(value);
            if (all_words) out[sample * stride + k] = value;
            else myAtomicAdd(out + sample * stride + k, value);
        }
    }
    if (__syncthreads_or(!valid) && !t) atomicExch(retry, 1);
}

template<int D, int J, int K, bool Right, class T>
__device__ __forceinline__ void warp_factor(T (&moments)[K][(D + 1) / 2], const T* increments,
    const WordPacket& packet, unsigned w, T bias) {
    constexpr int centre = (D - 1) / 2;
    if constexpr (J == centre) {
        #pragma unroll
        for (int t = 0; t < K; ++t) moments[t][0] += bias;
    } else {
        constexpr int next = Right ? J - 1 : J + 1, count = modes<D>(next);
        const int lane = threadIdx.x % 32;
        const T* a = increments + packet.channel(w, D, J) * (32 * K + 1) + lane;
        T total = 0;
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            const int t = Right ? K - 1 - i : i;
            const T v = a[t * 32];
            T lo = moments[t][0] + bias;
            const T z = v * lo;
            T mid = total + T(.5) * z;
            total += z;
            if constexpr (modes<D>(J) > 1) mid -= (T(Right ? 1 : 1. / 12.) * moments[t][1]) * v;
            moments[t][0] = mid;
            #pragma unroll
            for (int r = 1; r < count; ++r) {
                const T old = moments[t][r], hi = r + 1 < modes<D>(J) ? moments[t][r + 1] : T(0);
                if constexpr (Right) moments[t][r] = (lo * T(1. / ((4*r-2)*(4*r+2))) - hi) * v;
                else moments[t][r] = (lo - hi * T(1. / ((4*r+2)*(4*r+6)))) * v;
                lo = old;
            }
        }
        #pragma unroll
        for (int k = 1; k < 32; k *= 2) {
            const T v = Right ? __shfl_down_sync(0xffffffff, total, k) : __shfl_up_sync(0xffffffff, total, k);
            if (Right ? lane + k < 32 : lane >= k) total += v;
        }
        T carry = Right ? __shfl_down_sync(0xffffffff, total, 1) : __shfl_up_sync(0xffffffff, total, 1);
        if (lane == (Right ? 31 : 0)) carry = 0;
        warp_factor<D, next, K, Right>(moments, increments, packet, w, carry);
    }
}

template<class T, int D, int K>
__global__ void warp_words(const T* path, T* out, WordPacket packet, ScanShape s, uint64_t first, int* retry) {
    constexpr int pitch = 32 * K + 1, centre = (D - 1) / 2;
    const unsigned dim = unsigned(s.dimension), n = unsigned(s.segments);
    const unsigned lane = threadIdx.x % 32, warps = blockDim.x / 32;
    const uint64_t job = first + blockIdx.x, groups = (s.words + warps - 1) / warps;
    const uint64_t sample = job / groups, w = (job % groups) * warps + threadIdx.x / 32;
    const T* x = path + sample * s.length * dim;
    extern __shared__ double storage[];
    T* dx = reinterpret_cast<T*>(storage);
    bool valid = true;
    for (unsigned k = threadIdx.x; k < 32 * K * dim; k += blockDim.x) {
        const unsigned i = k / dim, c = k % dim;
        const T v = i < n ? x[k + dim] - x[k] : T(0);
        dx[c * pitch + i % K * 32 + i / K] = v;
        valid = valid && abs(v) <= T(65536) && (v == 0 || abs(v) >= T(1e-20));
    }
    __syncthreads();
    if (w < s.words) {
        T p[K][(D + 1) / 2] = {}, q[K][(D + 1) / 2] = {};
        #pragma unroll
        for (int t = 0; t < K; ++t) p[t][0] = q[t][0] = 1;
        warp_factor<D, 0, K, false>(p, dx, packet, unsigned(w), T(0));
        warp_factor<D, D - 1, K, true>(q, dx, packet, unsigned(w), T(0));
        const T* v = dx + packet.channel(w, D, centre) * pitch + lane;
        T sum = 0;
        #pragma unroll
        for (int t = 0; t < K; ++t) {
            T dot = p[t][0] * q[t][0];
            #pragma unroll
            for (int r = 1; r <= centre; ++r) dot += (r % 2 ? -p[t][r] : p[t][r]) * q[t][r];
            sum += v[t * 32] * dot;
        }
        #pragma unroll
        for (int k = 16; k; k /= 2) sum += __shfl_down_sync(0xffffffff, sum, k);
        if (!lane) {
            valid = valid && isfinite(sum);
            out[sample * s.width + packet.output(w)] = sum;
        }
    }
    if (__syncthreads_or(!valid) && !threadIdx.x) atomicExch(retry, 1);
}

// Column k of a chunk transfer contains the coefficients of subwords starting at k.
template<class T, int D>
__global__ void chunk_matrices(const T* path, T* matrices, WordPacket packet, ScanShape s, T time, uint64_t offset, int* retry) {
    constexpr unsigned entries = D * (D + 1) / 2;
    __shared__ T totals[scan_threads / 32];
    const uint64_t chunk = blockIdx.x / D, col = blockIdx.x % D, chunks = (s.segments - 1) / scan_threads + 1;
    const uint64_t job = offset + chunk / chunks, i = (chunk % chunks) * scan_threads + threadIdx.x;
    unsigned channels[D];
    #pragma unroll
    for (int j = 0; j < D; ++j) channels[j] = unsigned(packet.channel(job % s.words, D, min(unsigned(j) + unsigned(col), unsigned(D - 1))));
    T dx[D], moment[(D + 1) / 2 + 1] = {}, carry[D + 1] = {};
    bool valid = true;
    increments(path + (job / s.words) * s.length * s.dimension, i, channels, dx, s, time, valid);
    moment[0] = 1;
    prefix_rows<D, 0, 0, false, true, true>(moment, dx, carry, static_cast<T*>(nullptr), totals, D - int(col));
    if (!threadIdx.x) for (unsigned j = unsigned(col) + 1; j <= D; ++j) {
        const T value = carry[j - col];
        matrices[chunk * entries + j * (j - 1) / 2 + col] = value;
        valid = valid && isfinite(value);
    }
    if (__syncthreads_or(!valid) && !threadIdx.x) atomicExch(retry, 1);
}

constexpr unsigned chunk_group = 16;

template<int D, bool Transpose, class T>
__device__ __forceinline__ T apply_chunk(const T* matrix, T state) {
    const unsigned lane = threadIdx.x % 32;
    T next = state;
    #pragma unroll
    for (unsigned k = Transpose ? 2 : 0; k < (Transpose ? D + 1 : D); ++k) {
        const T v = __shfl_sync(0xffffffff, state, k);
        if constexpr (Transpose) {
            if (lane && lane < k) next += matrix[k * (k - 1) / 2 + lane] * v;
        } else if (lane > k && lane <= D) next += matrix[lane * (lane - 1) / 2 + k] * v;
    }
    return next;
}

template<class T, int D>
__global__ void group_matrices(const T* matrices, T* products, uint64_t chunks, uint64_t groups) {
    constexpr unsigned entries = D * (D + 1) / 2;
    const unsigned col = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint64_t group = blockIdx.x % groups, job = blockIdx.x / groups;
    T state = T(lane == col);
    for (uint64_t t = group * chunk_group; t < chunks && t < (group + 1) * chunk_group; ++t)
        state = apply_chunk<D, false>(matrices + (job * chunks + t) * entries, state);
    if (lane > col && lane <= D) products[uint64_t(blockIdx.x) * entries + lane * (lane - 1) / 2 + col] = state;
}

template<class T, int D, bool Backward>
__global__ void chunk_boundaries(const T* matrices, T* boundary, T* out, const T* seeds, WordPacket packet,
    ScanShape s, uint64_t chunks, uint64_t offset, int* retry, const T* outer = nullptr) {
    constexpr unsigned entries = D * (D + 1) / 2;
    const unsigned lane = threadIdx.x;
    const uint64_t groups = outer ? (chunks - 1) / chunk_group + 1 : 1;
    const uint64_t local = blockIdx.x / groups, group = blockIdx.x % groups, job = offset + local;
    const uint64_t pos = (job / s.words) * s.width + packet.output(job % s.words);
    const uint64_t first = local * chunks + group * chunk_group;
    const uint64_t last = outer ? first + min(uint64_t(chunk_group), chunks - group * chunk_group) : first + chunks;
    const uint64_t base = uint64_t(blockIdx.x) * 2 * D;
    T state = outer && lane && lane <= D ? outer[base + lane - 1] : T(lane == 0);
    for (uint64_t i = first; i < last; ++i) {
        if constexpr (Backward) if (lane && lane <= D) boundary[i * 2 * D + lane - 1] = state;
        state = apply_chunk<D, false>(matrices + i * entries, state);
    }
    if (lane <= D && !isfinite(state)) atomicExch(retry, 1);
    if constexpr (!Backward) {
        if (lane && lane <= D && (s.prefixes || lane == D)) out[pos + (s.prefixes ? lane - 1 : 0)] = state;
    } else {
        state = lane && lane <= D ? (outer ? outer[base + D + lane - 1] : !seeds ? T(lane == D) :
            s.terminal_seeds ? (lane == D ? seeds[pos] : T(0)) : seeds[pos + lane - 1]) : T(0);
        for (uint64_t i = last; i-- > first;) {
            if (lane && lane <= D) boundary[i * 2 * D + D + lane - 1] = state;
            state = apply_chunk<D, true>(matrices + i * entries, state);
        }
    }
}

template<class T, bool Backward>
__global__ void axis_words(const T* path, T* out, const T* seeds, const WordPacket packet, ScanShape s,
    unsigned degree, T time, uint64_t offset, uint64_t work, int* retry) {
    const uint64_t local = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (local >= work) return;
    const uint64_t job = local + offset, sample = job / s.words, w = job % s.words;
    const uint64_t pos = sample * s.width + packet.output(w);
    const unsigned c = degree ? unsigned(packet.channel(w, degree, 0)) : 0, lane = threadIdx.x % 32;
    const bool clock = s.time_aug && c == s.dimension * (s.lead_lag ? 2 : 1);
    const uint64_t col = s.lead_lag && c >= s.dimension ? c - s.dimension : c;
    const T* x = path + sample * s.length * s.dimension;
    bool valid = true;
    if (!clock && degree)
        for (uint64_t i = lane; i < s.length; i += 32) valid = valid && isfinite(x[i * s.dimension + col]);
    if (__any_sync(0xffffffff, !valid)) { if (!lane) atomicExch(retry, 1); return; }
    if (lane) return;
    if (!degree) { if constexpr (!Backward) out[pos] = 1; return; }
    const T dx = pc::point(x, s.segments, c, s.dimension, s.segments, s.time_aug, s.lead_lag, time)
        - pc::point(x, 0, c, s.dimension, s.segments, s.time_aug, s.lead_lag, time);
    if (!isfinite(dx)) { atomicExch(retry, 1); return; }
    if constexpr (Backward) {
        if (!clock) {
            const T value = pc::axis_backward(dx, seeds + pos, degree, s.terminal_seeds);
            if (!isfinite(value)) { atomicExch(retry, 1); return; }
            T* dst = out + sample * s.length * s.dimension + col;
            myAtomicAdd(dst, -value);
            myAtomicAdd(dst + (s.length - 1) * s.dimension, value);
        }
    } else pc::axis_forward(dx, degree, out + pos, s.prefixes);
}
} // namespace

template<class T, bool Backward>
bool prefix_suffix_coef_cuda(const T* path, T* out, const T* coefs, const T* seeds, const uint64_t* indices,
    const uint64_t* degrees, const uint64_t* metadata, uint64_t count, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t width, bool prefixes, bool time_aug, bool lead_lag, T end_time, cudaStream_t stream) {
    if (pc::add(pc::mul(dimension, lead_lag ? 2 : 1), time_aug) > UINT32_MAX) return false;
    const uint64_t segments = pc::mul(length - 1, lead_lag ? 2 : 1);
    const uint64_t max_degree = degrees[metadata[3 * count - 1]];
    if (segments == 1 || (max_degree >= 8 && segments < (max_degree + 3) / 4)) return false;
    const bool mapped = count > WordPacket::capacity;
    const uint64_t metadata_size = pc::add(pc::mul(3, count), pc::add(metadata[count - 1], degrees[count - 1]));
    const uint64_t entries = mapped ? metadata_size : 0;
    MappedWorkspace temporary;
    auto& workspace = entries > (1u << 16) ? temporary : packet_workspace();
    uint64_t* device = workspace.acquire(pc::add(entries, 1));
    if (!device) return false;
    if (mapped) std::copy_n(metadata, entries, workspace.host + 1);
    int* retry = reinterpret_cast<int*>(device);
    std::unique_ptr<T, FreeScratch> checkpoint_storage(nullptr, FreeScratch{stream});
    T* scratch = nullptr;
    uint64_t capacity = 0;
    const uint64_t budget = pc::workspace_limit() - pc::mul(metadata_size, sizeof(uint64_t)) - (Backward ? sizeof(int) : 0);
    if (workspace.scratch_bytes > budget) workspace.release_scratch();
    auto reserve = [&](uint64_t bytes) {
        if (bytes <= capacity) return true;
        if (capacity) CUDA_CHECK(cudaStreamSynchronize(stream));
        checkpoint_storage.reset();
        capacity = 0;
        scratch = static_cast<T*>(workspace.small_scratch(bytes));
        if (scratch) { capacity = bytes; return true; }
        void* ptr = nullptr;
        auto& ordered = checkpoint_storage.get_deleter().ordered;
        auto status = ordered ? cudaMallocAsync(&ptr, bytes, stream) : cudaMalloc(&ptr, bytes);
        if (status == cudaErrorNotSupported) {
            (void)cudaGetLastError();
            ordered = false;
            status = cudaMalloc(&ptr, bytes);
        }
        if (status == cudaErrorMemoryAllocation) { (void)cudaGetLastError(); return false; }
        CUDA_CHECK(status);
        checkpoint_storage.reset(static_cast<T*>(ptr));
        scratch = checkpoint_storage.get();
        capacity = bytes;
        return true;
    };
    auto clear_gradient = [&] { CUDA_CHECK(cudaMemsetAsync(out, 0, pc::mul(pc::mul(batch, pc::mul(length, dimension)), sizeof(T)), stream)); };
    bool initialized = false;
    try {
        for (uint64_t begin = 0; begin < count;) {
            const unsigned degree = unsigned(degrees[metadata[2 * count + begin]]);
            WordPacket packet{};
            packet.table = mapped ? device + 1 : nullptr;
            packet.count = count;
            packet.first = begin;
            bool axis = true;
            uint64_t words = 0;
            while (begin < count && (mapped || words < packet.capacity) && degrees[metadata[2 * count + begin]] == degree) {
                const uint64_t w = metadata[2 * count + begin++];
                if (degree && (axis || !mapped)) {
                    for (unsigned j = 0; j < degree; ++j) {
                        const unsigned c = unsigned(indices[metadata[w] + j]);
                        if (!mapped) packet.channels[words * degree + j] = c;
                        axis = axis && c == indices[metadata[w]];
                    }
                }
                if (!mapped) packet.outputs[words] = metadata[count + w];
                ++words;
            }
            if constexpr (Backward) if (!degree) continue;
            const uint64_t jobs = pc::mul(batch, words);
            const bool paired = !Backward && !prefixes && degree >= 2 && !axis;
            const unsigned path_threads = segments <= (degree <= 10 ? 512 : 256) ? unsigned((segments + 31) / 32) * 32 : 0;
            const uint64_t increments_bytes = (path_threads + 1) * dimension * sizeof(T);
            const bool direct = Backward && dimension >= 32 && 2 * increments_bytes + 64 * sizeof(T) > (48u << 10);
            const uint64_t path_bytes = increments_bytes * (Backward && !direct ? 2 : 1);
            const bool path_group = !axis && (Backward || paired) && !time_aug && !lead_lag && path_threads &&
                words <= 16 && path_bytes <= budget &&
                (path_bytes + 64 * sizeof(T) <= (48u << 10) ||
                    cuda_shared_memory_fits(64 * sizeof(T), path_bytes, cuda_shared_memory_limits()));
            const unsigned warp_items = segments <= 128 ? 4 : segments <= 256 ? 8 : segments <= 512 ? 16 : 32;
            const uint64_t warp_bytes = dimension * (32 * warp_items + 1) * sizeof(T);
            const bool warp_forward = paired && !time_aug && !lead_lag && words <= 16 && segments > 64 &&
                (words >= 4 || warp_items == 8 || (warp_items >= 16 && dimension <= 4 && degree >= 8)) &&
                segments <= (degree <= 8 ? 1024 : degree <= 16 ? 512 : 256) && warp_bytes <= budget &&
                (warp_bytes <= (48u << 10) || cuda_shared_memory_fits(0, warp_bytes, cuda_shared_memory_limits()));
            const bool grouped = path_group || warp_forward;
            const unsigned group_words = warp_forward ? unsigned(std::min<uint64_t>(words, 8)) : batch >= 64 ? unsigned(words) : 1;
            if constexpr (Backward) if (!initialized) {
                if (!grouped || group_words != count) clear_gradient();
                initialized = true;
            }
            const uint64_t tiles = (segments - 1) / scan_threads + 1;
            const uint64_t per_job = Backward && !axis ? pc::mul(tiles - 1, degree - 1) : paired ? pc::mul(tiles - 1, (degree - 1) / 2) : 0;
            const uint64_t shared = pc::mul(per_job, sizeof(T));
            const bool external = shared > (32u << 10);
            uint64_t limit = axis ? CUDA_GRID_X_LIMIT * 4 : CUDA_GRID_X_LIMIT;
            bool parallel_chunks = !axis && (Backward || degree <= 10 || jobs <= 4) && jobs <= (std::is_same_v<T, float> ? 32 : 16) &&
                segments >= std::max<uint64_t>(4096, 1024 * jobs) && tiles <= CUDA_GRID_X_LIMIT / (jobs * degree);
            if (parallel_chunks) {
                const uint64_t levels = tiles > 64 ? tiles + (tiles - 1) / chunk_group + 1 : tiles;
                const uint64_t chunk_bytes = (levels * (degree * (degree + 1) / 2 + (Backward || paired ? 2 * degree : 0))
                    + (paired ? tiles : 0)) * jobs * sizeof(T);
                parallel_chunks = chunk_bytes <= budget && reserve(chunk_bytes);
            }
            if (!parallel_chunks && external) {
                if (shared > budget) { CUDA_CHECK(cudaStreamSynchronize(stream)); return false; }
                limit = std::min(limit, budget / shared);
                const uint64_t needed = shared * std::min(limit, jobs);
                if (!reserve(needed)) { CUDA_CHECK(cudaStreamSynchronize(stream)); return false; }
            }
            const ScanShape s{words, dimension, length, width, segments, prefixes, time_aug, lead_lag, Backward && !coefs};
            const uint64_t tasks = grouped ? pc::mul(batch, (words - 1) / group_words + 1) : jobs;
            for (uint64_t offset = 0; offset < tasks;) {
                const uint64_t work = std::min(tasks - offset, limit);
                if (axis) {
                    axis_words<T, Backward><<<unsigned((work + 3) / 4), 128, 0, stream>>>(path, out, seeds, packet, s, degree, end_time, offset, work, retry);
                } else {
                    bool supported = true;
                    auto launch = [&]<unsigned D>() {
                        if (warp_forward) {
                            if constexpr (!Backward) {
                                auto run_warp = [&]<int K>() {
                                    if (warp_bytes > (48u << 10) && !try_configure_dynamic_smem(warp_words<T, D, K>, warp_bytes, cuda_shared_memory_limits())) {
                                        supported = false;
                                        return;
                                    }
                                    warp_words<T, D, K><<<unsigned(work), 32 * group_words, warp_bytes, stream>>>(path, out, packet, s, offset, retry);
                                };
                                if (warp_items == 4) run_warp.template operator()<4>();
                                if (warp_items == 8) run_warp.template operator()<8>();
                                if constexpr (D <= 16) if (warp_items == 16) run_warp.template operator()<16>();
                                if constexpr (D <= 8) if (warp_items == 32) run_warp.template operator()<32>();
                            }
                        } else if (grouped) {
                            if (path_bytes + 64 * sizeof(T) > (48u << 10) &&
                                !try_configure_dynamic_smem(path_words<T, D, Backward>, path_bytes, cuda_shared_memory_limits())) {
                                supported = false;
                                return;
                            }
                            path_words<T, D, Backward><<<unsigned(work), path_threads, path_bytes, stream>>>(path,
                                out, coefs, seeds, packet, s, offset, group_words, group_words == count, direct, retry);
                        } else if (parallel_chunks) {
                            T* matrices = scratch;
                            constexpr unsigned entries = D * (D + 1) / 2;
                            const uint64_t groups = tiles > 64 ? (tiles - 1) / chunk_group + 1 : 0;
                            T* products = matrices + work * tiles * entries;
                            T* boundary = products + work * groups * entries;
                            T* outer = boundary + (Backward || paired ? work * tiles * 2 * D : 0);
                            chunk_matrices<T, D><<<unsigned(work * tiles * D), scan_threads, 0, stream>>>(path, matrices, packet, s, end_time, offset, retry);
                            CUDA_CHECK(cudaGetLastError());
                            if (groups) {
                                group_matrices<T, D><<<unsigned(work * groups), 32 * D, 0, stream>>>(matrices, products, tiles, groups);
                                CUDA_CHECK(cudaGetLastError());
                            }
                            auto ends = [&]<bool Both>() {
                                chunk_boundaries<T, D, Both><<<unsigned(work), 32, 0, stream>>>(groups ? products : matrices,
                                    groups ? outer : boundary, out, seeds, packet, s, groups ? groups : tiles, offset, retry);
                            };
                            if (paired) ends.template operator()<true>(); else ends.template operator()<Backward>();
                            CUDA_CHECK(cudaGetLastError());
                            if (Backward || paired) {
                                if (groups) {
                                    chunk_boundaries<T, D, true><<<unsigned(work * groups), 32, 0, stream>>>(matrices, boundary, out, seeds, packet, s, tiles, offset, retry, outer);
                                    CUDA_CHECK(cudaGetLastError());
                                }
                                if (Backward) {
                                    scan_word<T, D, true><<<unsigned(work * tiles), scan_threads, 0, stream>>>(path, out, coefs, seeds, packet, s, end_time, offset, nullptr, retry, boundary);
                                } else {
                                    T* partial = outer + work * groups * 2 * D;
                                    paired_word<T, (D < 2 ? 2 : D)><<<unsigned(work * tiles), scan_threads, 0, stream>>>(path, partial, packet, s, end_time, offset, nullptr, retry, boundary);
                                    CUDA_CHECK(cudaGetLastError());
                                    sum_chunks<<<unsigned(work), scan_threads, 0, stream>>>(partial, out, packet, s, tiles, offset, retry);
                                }
                            }
                        } else {
                            if (paired) {
                                paired_word<T, (D < 2 ? 2 : D)><<<unsigned(work), scan_threads, external ? 0 : shared, stream>>>(
                                    path, out, packet, s, end_time, offset, external ? scratch : nullptr, retry);
                            } else {
                                scan_word<T, D, Backward><<<unsigned(work), scan_threads, external ? 0 : shared, stream>>>(
                                    path, out, coefs, seeds, packet, s, end_time, offset, external ? scratch : nullptr, retry);
                            }
                        }
                    };
#define WORD_PACKET_CASE(M) case M: launch.template operator()<M>(); break
                    switch (degree) {
                        WORD_PACKET_CASE(1); WORD_PACKET_CASE(2); WORD_PACKET_CASE(3); WORD_PACKET_CASE(4);
                        WORD_PACKET_CASE(5); WORD_PACKET_CASE(6); WORD_PACKET_CASE(7); WORD_PACKET_CASE(8);
                        WORD_PACKET_CASE(9); WORD_PACKET_CASE(10); WORD_PACKET_CASE(11); WORD_PACKET_CASE(12);
                        WORD_PACKET_CASE(13); WORD_PACKET_CASE(14); WORD_PACKET_CASE(15); WORD_PACKET_CASE(16);
                        WORD_PACKET_CASE(17); WORD_PACKET_CASE(18); WORD_PACKET_CASE(19); WORD_PACKET_CASE(20);
                    }
#undef WORD_PACKET_CASE
                    if (!supported) { CUDA_CHECK(cudaStreamSynchronize(stream)); return false; }
                }
                CUDA_CHECK(cudaGetLastError());
                offset += work;
            }
        }
        if constexpr (Backward) if (!initialized) clear_gradient();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
    return *reinterpret_cast<int*>(workspace.host) == 0;
}

#define INSTANTIATE_PACKET(T, B) template bool prefix_suffix_coef_cuda<T, B>(const T*, T*, const T*, const T*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, bool, T, cudaStream_t)
INSTANTIATE_PACKET(float, false);
INSTANTIATE_PACKET(float, true);
INSTANTIATE_PACKET(double, false);
INSTANTIATE_PACKET(double, true);
#undef INSTANTIATE_PACKET
