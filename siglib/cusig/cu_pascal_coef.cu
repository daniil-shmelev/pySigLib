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

#include "cusig.h"
#include "cu_macros.h"
#include "cu_runtime_utils.h"
#include "cu_atomic.h"
#include "cu_sig_coef.h"
#include "pascal_coefficient.h"
#include "pascal_selection.h"

namespace pascal_coefficient {
#include "pascal_coefficient-inl.h"
}

namespace {
namespace pc = pascal_coefficient;

struct DegreeGroup { uint64_t begin, count, degree; };
thread_local uint64_t last_workspace_bytes = 0;

bool calibrated_cuda() {
#if defined(_MSC_VER) && _MSC_VER == 1944 && __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 9
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    static thread_local int cached_device = -1;
    static thread_local bool calibrated = false;
    if (device != cached_device) {
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        calibrated = properties.major == 8 && properties.minor == 9 && properties.multiProcessorCount == 24
            && std::strcmp(properties.name, "NVIDIA GeForce RTX 4060 Laptop GPU") == 0;
        cached_device = device;
    }
    return calibrated;
#else
    return false;
#endif
}

template<class T>
struct Access {
    const T* path;
    T* out;
    const T* deriv;
    const T* coefs;
    const uint64_t* word;
    uint64_t dimension, n;
    bool ta, ll, independent;
    T time;
    __device__ T increment(uint64_t i, uint64_t j) const {
        return pc::point(path, i + 1, word[j], dimension, n, ta, ll, time)
            - pc::point(path, i, word[j], dimension, n, ta, ll, time);
    }
    __device__ T seed(uint64_t j) const { return deriv[j]; }
    __device__ void scatter(uint64_t i, const T* grad, uint64_t m) const {
        for (uint64_t j = 0; j < m; ++j) {
            const uint64_t c = word[j];
            if (ta && c == dimension * (ll ? 2 : 1)) continue;
            bool seen = false;
            for (uint64_t k = 0; k < j; ++k) seen |= word[k] == c;
            if (seen) continue;
            T value = grad[j];
            for (uint64_t k = j + 1; k < m; ++k) if (word[k] == c) value += grad[k];
            const uint64_t before = ll ? i / 2 + ((i % 2) && c >= dimension) : i;
            const uint64_t after = ll ? (i + 1) / 2 + (((i + 1) % 2) && c >= dimension) : i + 1;
            if (before != after) {
                const uint64_t column = ll && c >= dimension ? c - dimension : c;
                if (independent) {
                    out[after * dimension + column] += value;
                    out[before * dimension + column] -= value;
                } else {
                    myAtomicAdd(out + after * dimension + column, value);
                    myAtomicAdd(out + before * dimension + column, -value);
                }
            }
        }
    }
};

// Same Horner and inverse-Chen recurrences as the legacy coefficient kernels.
template<class T, int Degree, bool Backward>
__device__ void horner(Access<T>& access, uint64_t degree, T* u) {
    const uint64_t m = Degree ? Degree : degree;
    if constexpr (Backward) if (access.n == 1) {
        pc::single_segment_backward<pc::ScalarOps<T>>(access, m, u);
        return;
    }
    T fact[21], a[21], lambda[21], buffer[21], grad[21];
    fact[0] = T(1);
    u[0] = T(1);
    for (uint64_t j = 1; j <= m; ++j) {
        fact[j] = fact[j - 1] / T(j);
        u[j] = Backward ? access.coefs[j - 1] : T(0);
        if constexpr (Backward) lambda[j] = access.seed(j - 1);
    }
    if constexpr (!Backward) {
        for (uint64_t i = 0; i < access.n; ++i) {
            for (uint64_t j = 1; j <= m; ++j) a[j] = access.increment(i, j - 1);
            if (!i) {
                T product = T(1);
                for (uint64_t j = 1; j <= m; ++j) { product *= a[j]; u[j] = product * fact[j]; }
            } else for (uint64_t j = m; j > 0; --j) {
                T value = fact[j];
                for (uint64_t k = 1; k < j; ++k) value = value * a[k] + u[k] * fact[j - k];
                u[j] += a[j] * value;
            }
        }
    } else {
        for (uint64_t i = access.n; i-- > 0;) {
            for (uint64_t j = 1; j <= m; ++j) a[j] = access.increment(i, j - 1);
            T last = T(1);
            for (uint64_t j = 1; j <= m; ++j) {
                for (uint64_t k = 1; k < j; ++k) buffer[k] *= a[j];
                buffer[j] = last * a[j];
                T value = 0;
                for (uint64_t k = 1; k <= j; ++k) value += buffer[k] * ((j - k + 1) % 2 ? -fact[j - k + 1] : fact[j - k + 1]);
                last = u[j];
                u[j] += value;
            }
            for (uint64_t j = 1; j <= m; ++j) {
                if (j > 1) {
                    for (uint64_t k = 1; k + 1 < j; ++k) buffer[k] *= a[j - 1];
                    buffer[j - 1] = u[j - 2] * a[j - 1];
                }
                T value = u[j - 1];
                for (uint64_t k = 1; k < j; ++k) value += buffer[k] * fact[j - k + 1];
                T g = lambda[j] * value, product = T(1);
                for (uint64_t k = j + 1; k <= m; ++k) {
                    product *= a[k];
                    value = u[j - 1] * fact[k - j + 1];
                    for (uint64_t h = 1; h < j; ++h) value += buffer[h] * fact[k - h + 1];
                    g += lambda[k] * product * value;
                }
                grad[j - 1] = g;
            }
            access.scatter(i, grad, m);
            for (uint64_t j = 1; j < m; ++j) {
                T product = T(1), value = 0;
                for (uint64_t k = j + 1; k <= m; ++k) { product *= a[k]; value += lambda[k] * product * fact[k - j]; }
                lambda[j] += value;
            }
        }
    }
}

template<class T, int Mode, bool Backward, int Degree = 0>
__global__ void coefficient_kernel(const T* path, T* out, const T* coefs, const T* deriv,
    const uint64_t* metadata, uint64_t words, uint64_t width, uint64_t dimension, uint64_t length,
    uint64_t n, bool ta, bool ll, T time, bool prefixes, uint64_t offset, uint64_t work,
    T* checkpoints, uint64_t checkpoint_stride, uint64_t block, uint64_t blocks, int* failed,
    uint64_t order_offset, uint64_t group_begin, uint64_t group_count) {
    constexpr int capacity = Degree ? Degree : 20;
    const uint64_t local = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= work) return;
    const uint64_t job = offset + local, sample = job / group_count, w = metadata[order_offset + group_begin + job % group_count];
    const uint64_t m = Degree ? Degree : metadata[w], output_offset = metadata[2 * words + w];
    Access<T> access{path + sample * length * dimension, out + sample * length * dimension,
        deriv ? deriv + sample * width + output_offset : nullptr,
        coefs ? coefs + sample * width + output_offset : nullptr,
        metadata + 3 * words + metadata[words + w], dimension, n, ta, ll, words == 1, time};
    if constexpr (Backward) { if (!m || !n) return; }
    T scratch[Backward ? 4 * (capacity + 1) + capacity * (capacity + 1) / 2 : 2 * (capacity + 1)];
    bool valid = true;
    if constexpr (Mode == pc::horner) horner<T, Degree, Backward>(access, m, scratch);
    else if constexpr (Backward)
        valid = pc::backward<pc::ScalarOps<T>, Mode, Degree>(access, n, m, block, blocks,
            checkpoints + local * checkpoint_stride, scratch);
    else valid = pc::forward<pc::ScalarOps<T>, Mode, Degree>(access, n, m, scratch, scratch + capacity + 1);
    if constexpr (!Backward) {
        const uint64_t count = prefixes && m ? m : 1;
        for (uint64_t j = 0; j < count; ++j)
            out[sample * width + output_offset + j] = scratch[m ? (prefixes ? j + 1 : m) : 0];
    }
    if (!valid && Mode != pc::guarded) atomicExch(failed, 1);
}

template<class T, int Mode, bool Backward>
void launch(const T* path, T* out, const T* coefs, const T* deriv, const uint64_t* metadata,
    const pc::Words& words, uint64_t dimension, uint64_t length, uint64_t n, bool ta, bool ll, T time,
    bool prefixes, uint64_t offset, uint64_t work, T* checkpoints, const pc::Checkpoints& plan,
    int* failed, cudaStream_t stream, unsigned int threads, uint64_t order_offset, const DegreeGroup& group) {
    const unsigned int grid = static_cast<unsigned int>((work + threads - 1) / threads);
#define PASCAL_LAUNCH(M) coefficient_kernel<T, Mode, Backward, M><<<grid, threads, 0, stream>>>(path, out, coefs, deriv, metadata, words.count, words.width, dimension, length, n, ta, ll, time, prefixes, offset, work, checkpoints, plan.count, plan.block, plan.blocks, failed, order_offset, group.begin, group.count)
#define PASCAL_CUDA_CASE(M) case M: PASCAL_LAUNCH(M); break
    if constexpr (Mode == pc::guarded || Mode == pc::horner) { PASCAL_LAUNCH(0); }
    else switch (group.degree) {
        PASCAL_CUDA_CASE(1); PASCAL_CUDA_CASE(2); PASCAL_CUDA_CASE(3); PASCAL_CUDA_CASE(4);
        PASCAL_CUDA_CASE(5); PASCAL_CUDA_CASE(6); PASCAL_CUDA_CASE(7); PASCAL_CUDA_CASE(8);
        PASCAL_CUDA_CASE(9); PASCAL_CUDA_CASE(10); PASCAL_CUDA_CASE(11); PASCAL_CUDA_CASE(12);
        PASCAL_CUDA_CASE(13); PASCAL_CUDA_CASE(14); PASCAL_CUDA_CASE(15); PASCAL_CUDA_CASE(16);
        PASCAL_CUDA_CASE(17); PASCAL_CUDA_CASE(18); PASCAL_CUDA_CASE(19); PASCAL_CUDA_CASE(20);
        default: PASCAL_LAUNCH(0); break;
    }
#undef PASCAL_CUDA_CASE
#undef PASCAL_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

template<class T, int Mode>
__global__ void coefficient_warp_backward(const T* path, T* out, const T* deriv, const uint64_t* metadata,
    uint64_t words, uint64_t width, uint64_t dimension, uint64_t length, uint64_t n, bool ta, bool ll, T time,
    uint64_t offset, uint64_t work, T* checkpoints, uint64_t checkpoint_stride, uint64_t block,
    uint64_t blocks, int* failed, uint64_t order_offset, uint64_t group_begin, uint64_t group_count) {
    const unsigned int lane = threadIdx.x % 32;
    const uint64_t local = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (local >= work) return;
    const uint64_t job = offset + local, sample = job / group_count, w = metadata[order_offset + group_begin + job % group_count];
    const unsigned int m = static_cast<unsigned int>(metadata[w]), stride = m + 1;
    if (!m || !n) return;
    const uint64_t c = lane && lane <= m ? metadata[3 * words + metadata[words + w] + lane - 1] : UINT64_MAX;
    const T* x = path + sample * length * dimension;
    T* dx = out + sample * length * dimension;
    T* saved = checkpoints + local * checkpoint_stride;
    T* replay = saved + blocks * stride;
    const T inverse_lane = lane ? T(1) / T(lane) : T(0);
    T u = lane == 0 ? T(1) : T(0), lambda = 0, a = 0, raw = 0, q = 0;
    T tape[20];
    if (lane && lane <= m) lambda = deriv[sample * width + metadata[2 * words + w] + lane - 1];
    bool valid = true;
    if constexpr (Mode == pc::scaled) {
        T factor = 1;
        for (unsigned int j = 2; j <= lane; ++j) factor /= T(j);
        const T seed = lambda;
        lambda *= factor;
        valid = !pc::ScalarOps<T>::tiny(seed, lambda);
    }
    auto increment = [&](uint64_t i) {
        if (lane && lane <= m) {
            raw = pc::point(x, i + 1, c, dimension, n, ta, ll, time) - pc::point(x, i, c, dimension, n, ta, ll, time);
            a = Mode == pc::unscaled ? raw * inverse_lane : raw;
            if constexpr (Mode == pc::unscaled) valid = !pc::ScalarOps<T>::tiny(raw, a) && valid;
        }
    };
    auto step = [&](bool first, bool record) {
        if (first) {
            for (unsigned int j = 1; j <= m; ++j) {
                const T previous = __shfl_sync(0xffffffff, u, j - 1);
                if (lane == j) u = previous * a;
            }
        } else {
            q = lane == m - 1 ? u : T(0);
            for (unsigned int s = 1; s < m; ++s) {
                const T previous = __shfl_up_sync(0xffffffff, u, 1);
                if (lane >= s && lane < m) {
                    if (record) tape[s - 1] = previous;
                    u = a * previous + u;
                }
                if (lane == m - 1) q += u;
            }
            const T sum = __shfl_sync(0xffffffff, q, m - 1);
            if (lane == m) u = a * sum + u;
            if (lane == m - 1) valid = pc::ScalarOps<T>::finite(q) && valid;
        }
        if (lane <= m) valid = pc::ScalarOps<T>::finite(u) && valid;
    };
    for (uint64_t i = 0; i < n; ++i) {
        if (i % block == 0 && lane <= m) saved[(i / block) * stride + lane] = u;
        increment(i);
        step(i == 0, false);
    }
    if (!__all_sync(0xffffffff, valid)) {
        if (!lane) atomicExch(failed, 1);
        return;
    }
    unsigned int peers = 0;
#if __CUDA_ARCH__ >= 700
    peers = __match_any_sync(0xffffffff, c);
#else
    for (unsigned int k = 0; k < 32; ++k) if (c == __shfl_sync(0xffffffff, c, k)) peers |= 1u << k;
#endif
    for (uint64_t b = blocks; b-- > 0;) {
        const uint64_t begin = b * block, end = n < begin + block ? n : begin + block;
        if (block > 1) {
            if (lane <= m) u = saved[b * stride + lane];
            for (uint64_t i = begin; i < end; ++i) {
                if (lane <= m) replay[(i - begin) * stride + lane] = u;
                increment(i);
                step(i == 0, false);
            }
        }
        for (uint64_t i = end; i-- > begin;) {
            if (lane <= m) u = block > 1 ? replay[(i - begin) * stride + lane] : saved[b * stride + lane];
            increment(i);
            step(i == 0, true);
            T grad = 0;
            if (!i) {
                for (unsigned int j = m; j > 0; --j) {
                    const T previous = __shfl_sync(0xffffffff, u, j - 1);
                    if (lane == j) grad = previous * lambda;
                    const T contribution = __shfl_sync(0xffffffff, a * lambda, j);
                    if (lane + 1 == j && lane) lambda += contribution;
                }
            } else {
                const T sum = __shfl_sync(0xffffffff, q, m - 1);
                if (lane == m) grad = lambda * sum;
                const T dq = __shfl_sync(0xffffffff, a * lambda, m);
                for (unsigned int s = m - 1; s > 0; --s) {
                    if (lane == m - 1) lambda += dq;
                    if (lane >= s && lane < m) grad = tape[s - 1] * lambda + grad;
                    const T contribution = __shfl_down_sync(0xffffffff, a * lambda, 1);
                    if (lane && lane + 1 >= s && lane + 1 < m) lambda += contribution;
                }
                if (m > 1 && lane == m - 1) lambda += dq;
            }
            if constexpr (Mode == pc::unscaled) grad *= inverse_lane;
            if (lane && lane <= m) valid = pc::ScalarOps<T>::finite(grad) && pc::ScalarOps<T>::finite(lambda) && valid;
            T total = 0;
            if (lane && lane <= m) {
                for (unsigned int mask = peers; mask; mask &= mask - 1)
                    total += __shfl_sync(peers, grad, __ffs(mask) - 1);
                valid = pc::ScalarOps<T>::finite(total) && valid;
            }
            if (lane && lane <= m && lane == unsigned(__ffs(peers) - 1) && !(ta && c == dimension * (ll ? 2 : 1))) {
                const uint64_t before = ll ? i / 2 + ((i % 2) && c >= dimension) : i;
                const uint64_t after = ll ? (i + 1) / 2 + (((i + 1) % 2) && c >= dimension) : i + 1;
                const uint64_t column = ll && c >= dimension ? c - dimension : c;
                if (before != after) {
                    if (words == 1) { dx[after * dimension + column] += total; dx[before * dimension + column] -= total; }
                    else { myAtomicAdd(dx + after * dimension + column, total); myAtomicAdd(dx + before * dimension + column, -total); }
                }
            }
        }
    }
    if (!__all_sync(0xffffffff, valid) && !lane) atomicExch(failed, 1);
}

template<class T, bool Backward>
void coefficient_cuda(const T* path, T* out, const T* coefs, const T* deriv, const uint64_t* indices,
    uint64_t count, const uint64_t* degrees, uint64_t batch, uint64_t dimension, uint64_t length,
    bool ta, bool ll, T time, bool prefixes, int method, void* stream_ptr) {
    last_workspace_bytes = 0;
    pc::Words words(indices, degrees, count, pc::add(pc::mul(dimension, ll ? 2 : 1), ta ? 1 : 0), Backward || prefixes);
    if (words.max_degree > 20) throw std::invalid_argument("CUDA coefficients support word lengths through 20");
    const uint64_t n = length > 1 ? pc::mul(length - 1, ll ? 2 : 1) : 0;
    const uint64_t jobs = pc::mul(batch, count);
    const uint64_t output_elements = Backward ? pc::mul(batch, pc::mul(length, dimension)) : pc::mul(batch, words.width);
    const uint64_t output_bytes = pc::mul(output_elements, sizeof(T));
    if (!batch) return;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    int mode = pc::variant(method, words.max_degree);
    if (method == pc::horner || (method == pc::automatic && !(pascal_selection::cuda_region<T>(
        words, batch, dimension, n, jobs, ta, ll, true) && calibrated_cuda()))) mode = pc::horner;
    const uint64_t order_offset = words.packed.size();
    words.packed.resize(pc::add(order_offset, count));
    for (uint64_t w = 0; w < count; ++w) words.packed[order_offset + w] = w;
    std::stable_sort(words.packed.begin() + order_offset, words.packed.end(), [&](uint64_t a, uint64_t b) { return degrees[a] < degrees[b]; });
    std::vector<DegreeGroup> groups;
    for (uint64_t begin = 0; begin < count;) {
        uint64_t end = begin + 1;
        const uint64_t degree = degrees[words.packed[order_offset + begin]];
        while (end < count && degrees[words.packed[order_offset + end]] == degree) ++end;
        groups.push_back({begin, end - begin, degree});
        begin = end;
    }
    const uint64_t cutoff = method >= pc::unscaled ? pc::environment_option("PYSIGLIB_PASCAL_BENCH_CHECKPOINT", 64) : 64;
    const pc::Checkpoints plan(n, words.max_degree, cutoff);
    const uint64_t row = Backward && mode != pc::horner ? plan.count : 0;
    const uint64_t budget = pc::workspace_limit();
    const uint64_t state = words.max_degree + 1;
    const uint64_t local_count = Backward ? pc::add(pc::mul(4, state), pc::mul(state, words.max_degree) / 2) : pc::mul(2, state);
    const uint64_t job_bytes = pc::mul(pc::add(row, local_count), sizeof(T));
    if (job_bytes > budget) throw std::runtime_error("Pascal workspace limit is too small for one CUDA job");
    const uint64_t capacity = std::min<uint64_t>({jobs, budget / job_bytes, 65535ULL * 64});
    CudaBatchWorkspace<T> workspace(capacity, row, "Pascal coefficients");
    CudaBuf<uint64_t> device_words(pc::mul(pc::add(words.packed.size(), 1), sizeof(uint64_t)));
    last_workspace_bytes = pc::add(pc::mul(pc::mul(workspace.capacity(), row), sizeof(T)), pc::mul(pc::add(words.packed.size(), 1), sizeof(uint64_t)));
    int* device_failed = reinterpret_cast<int*>(device_words.get() + words.packed.size());
    const unsigned int threads = static_cast<unsigned int>(method >= pc::unscaled ? pc::environment_option("PYSIGLIB_PASCAL_BENCH_THREADS", 64) : 64);
    if (threads != 64 && threads != 128 && threads != 256) throw std::invalid_argument("Pascal CUDA block size must be 64, 128 or 256");
    const bool warp_default = words.max_degree >= 8 && (jobs <= 512 || (sizeof(T) == 4 && words.max_degree >= 16));
    const bool warp = Backward && mode != pc::horner && (method >= pc::unscaled ? pc::environment_option("PYSIGLIB_PASCAL_BENCH_WARP", warp_default) : warp_default);
    try {
        CUDA_CHECK(cudaMemcpyAsync(device_words.get(), words.packed.data(), words.packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        if (method == pc::automatic && mode == pc::horner && !ta && !ll && n && !(Backward && n == 1) && groups.size() == 1 && words.max_degree && words.max_degree <= 10) {
            sig_coef_fixed_metadata_cuda<T, Backward>(path, out, coefs, deriv, device_words.get(), words.count,
                words.width, words.max_degree, batch, dimension, length, prefixes, stream);
            return;
        }
        for (;;) {
            if constexpr (Backward) if (output_bytes) CUDA_CHECK(cudaMemsetAsync(out, 0, output_bytes, stream));
            if (mode != pc::horner) CUDA_CHECK(cudaMemsetAsync(device_failed, 0, sizeof(int), stream));
            for (const auto& group : groups) for (uint64_t offset = 0; offset < batch * group.count;) {
                const uint64_t work = std::min<uint64_t>(workspace.capacity(), batch * group.count - offset);
#define PASCAL_RUN(MODE) launch<T, MODE, Backward>(path, out, coefs, deriv, device_words.get(), words, dimension, length, n, ta, ll, time, prefixes, offset, work, workspace.get(), plan, device_failed, stream, threads, order_offset, group)
                if (warp && mode != pc::guarded) {
                    const uint64_t warp_work = std::min<uint64_t>(work, 65535ULL * (threads / 32));
                    const unsigned int grid = static_cast<unsigned int>((warp_work * 32 + threads - 1) / threads);
#define PASCAL_WARP(MODE) coefficient_warp_backward<T, MODE><<<grid, threads, 0, stream>>>(path, out, deriv, device_words.get(), words.count, words.width, dimension, length, n, ta, ll, time, offset, warp_work, workspace.get(), plan.count, plan.block, plan.blocks, device_failed, order_offset, group.begin, group.count)
                    if (mode == pc::scaled) { PASCAL_WARP(pc::scaled); }
                    else { PASCAL_WARP(pc::unscaled); }
#undef PASCAL_WARP
                    CUDA_CHECK(cudaGetLastError());
                    offset += warp_work;
                    continue;
                }
                if (mode == pc::horner) { PASCAL_RUN(pc::horner); }
                else if (mode == pc::scaled) { PASCAL_RUN(pc::scaled); }
                else if (mode == pc::guarded) { PASCAL_RUN(pc::guarded); }
                else { PASCAL_RUN(pc::unscaled); }
#undef PASCAL_RUN
                offset += work;
            }
            int failed = 0;
            if (mode != pc::horner) CUDA_CHECK(cudaMemcpyAsync(&failed, device_failed, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (!failed || mode == pc::guarded) break;
            mode = pc::guarded;
        }
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}
} // namespace

template<class T, bool Backward>
void sig_coef_dispatch_cuda(const T* path, T* out, const T* coefs, const T* deriv,
    const uint64_t* indices, uint64_t words, const uint64_t* degrees, uint64_t batch,
    uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time,
    bool prefixes, int method, void* stream) {
    coefficient_cuda<T, Backward>(path, out, coefs, deriv, indices, words, degrees,
        batch, dimension, length, time_aug, lead_lag, end_time, prefixes, method, stream);
}

#define INSTANTIATE_PASCAL_CUDA(T, B) \
template void sig_coef_dispatch_cuda<T, B>(const T*, T*, const T*, const T*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, int, void*)
INSTANTIATE_PASCAL_CUDA(float, false);
INSTANTIATE_PASCAL_CUDA(float, true);
INSTANTIATE_PASCAL_CUDA(double, false);
INSTANTIATE_PASCAL_CUDA(double, true);
#undef INSTANTIATE_PASCAL_CUDA

extern "C" {
CUSIG_API uint64_t cusig_coefficient_workspace_bytes() noexcept { return last_workspace_bytes; }
}
