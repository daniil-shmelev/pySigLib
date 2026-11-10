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

#include "../../siglib/cusig/cu_prefix_suffix.cu"

namespace {
template<bool Reverse, typename T>
__global__ void reconstruct(const T* path, const T* final, const T* deriv, T* prefixes,
    T* suffixes, T* sources, T* inc, T* dinc, T* state, T* scratch, uint64_t d, uint64_t n,
    uint64_t h, uint64_t qlen, uint64_t first, uint64_t size, uint64_t position,
    uint64_t stride, uint64_t sig_stride, bool scalar, const uint64_t* power, const uint64_t* offset) {
    const uint64_t sample = blockIdx.x;
    T* prefix = state + sample * stride; T* suffix = prefix + qlen; T* adjoint = suffix + qlen;
    T* work = scratch + sample * stride; T* negative = work + 2 * qlen;
    if (!position) {
        for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x) {
            const T value = x ? final[sample * sig_stride + x - !scalar] : T(1);
            prefix[x] = Reverse ? value : T(x == 0); suffix[x] = Reverse ? T(x == 0) : value;
            adjoint[x] = Reverse && x ? deriv[sample * sig_stride + x - !scalar] : T(0);
        }
    }
    for (uint64_t x = threadIdx.x; x < size * qlen; x += blockDim.x) sources[sample * stride + x] = 0;
    for (uint64_t x = threadIdx.x; x < size * d; x += blockDim.x) dinc[sample * stride + x] = 0;
    __syncthreads();
    for (uint64_t step = 0; step < size; ++step) {
        const uint64_t slot = Reverse ? size - 1 - step : step, i = first + slot;
        T* v = inc + sample * stride + slot * d;
        for (uint64_t c = threadIdx.x; c < d; c += blockDim.x)
            negative[c] = -(v[c] = path[(sample * (n + 1) + i + 1) * d + c] - path[(sample * (n + 1) + i) * d + c]);
        __syncthreads();
        fused_step(Reverse ? prefix : suffix, negative, work, work + qlen, d, h, power, offset, !Reverse);
        for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x) {
            prefixes[sample * stride + slot * qlen + x] = prefix[x];
            suffixes[sample * stride + slot * qlen + x] = suffix[x];
        }
        __syncthreads();
        fused_step(Reverse ? suffix : prefix, v, work, work + qlen, d, h, power, offset, Reverse);
    }
}

template<bool Reverse, typename T>
__global__ void factors(const T* prefixes, const T* suffixes, const T* inc, T* result,
    uint64_t batch, unsigned d, uint64_t qlen, unsigned a, unsigned b, unsigned width,
    uint64_t rows, uint64_t stride, const uint64_t* offset, FactorCoefficients<T> coefficients) {
    for (uint64_t task = uint64_t(blockIdx.x) * 8 + threadIdx.x / 32; task < batch * rows; task += uint64_t(gridDim.x) * 8) {
        const uint64_t sample = task / rows, row = task % rows, i = row / a;
        const unsigned j = row % a;
        const T* v = inc + sample * stride + i * d;
        for (unsigned word = threadIdx.x % 32; word < width; word += 32) {
            T value;
            if constexpr (Reverse) value = coefficient_right<T, 0>(suffixes + sample * stride + i * qlen, v, word, d, j, b, coefficients);
            else {
                unsigned index = word; value = 1;
                for (unsigned level = 0; level <= j; ++level) { value *= v[index % d]; index /= d; }
                value *= prefixes[sample * stride + i * qlen + offset[a - 1 - j] + index];
            }
            result[sample * stride + row * width + word] = value;
        }
    }
}

template<bool Reverse, typename T>
__global__ void factor_derivatives(const T* prefixes, const T* suffixes, const T* inc,
    const T* gradient, T* sources, T* dinc, T* scratch, uint64_t batch, uint64_t d,
    uint64_t qlen, uint64_t a, uint64_t b, uint64_t size, uint64_t slots, uint64_t temporary,
    uint64_t stride, const uint64_t* power, const uint64_t* offset) {
    const uint64_t p = power[a], q = power[b];
    T* powers = scratch + uint64_t(blockIdx.x) / slots * stride + blockIdx.x % slots * 2 * temporary;
    T* work = powers + temporary;
    for (uint64_t task = blockIdx.x; task < batch * size; task += gridDim.x) {
        const uint64_t sample = task / size, i = task % size;
        const T* v = inc + sample * stride + i * d;
        const T* prefix = prefixes + sample * stride + i * qlen;
        const T* suffix = suffixes + sample * stride + i * qlen;
        T* ds = sources + sample * stride + i * qlen; T* dv = dinc + sample * stride + i * d;
        if constexpr (Reverse) {
            if (!threadIdx.x) powers[0] = 1;
            for (uint64_t x = threadIdx.x; x < offset[a + 1]; x += blockDim.x) work[x] = 0;
            __syncthreads();
            for (uint64_t k = 1; k <= a; ++k) {
                for (uint64_t x = threadIdx.x; x < power[k]; x += blockDim.x)
                    powers[offset[k] + x] = powers[offset[k - 1] + x / d] * v[x % d];
                __syncthreads();
            }
        }
        for (uint64_t j = 0; j < a; ++j) {
            const T* g = gradient + sample * stride + (i * a + j) * (Reverse ? p : q);
            if constexpr (Reverse) {
                const uint64_t tail = power[j + 1], head = power[a - 1 - j], level = offset[a - 1 - j];
                for (uint64_t x = threadIdx.x; x < head; x += blockDim.x) {
                    T value = 0;
                    for (uint64_t y = 0; y < tail; ++y) value += g[x * tail + y] * powers[offset[j + 1] + y];
                    ds[level + x] += value;
                }
                for (uint64_t y = threadIdx.x; y < tail; y += blockDim.x) {
                    T value = 0;
                    for (uint64_t x = 0; x < head; ++x) value += g[x * tail + y] * prefix[level + x];
                    work[offset[j + 1] + y] += value;
                }
                __syncthreads();
            } else {
                if (!threadIdx.x) powers[0] = 1;
                __syncthreads();
                for (uint64_t k = 1; k < b; ++k) {
                    const uint64_t previous = power[k - 1];
                    for (uint64_t x = threadIdx.x; x < power[k]; x += blockDim.x)
                        powers[offset[k] + x] = suffix[offset[k] + x] + v[x / previous] * powers[offset[k - 1] + x % previous] / T(j + b - k + 2);
                    __syncthreads();
                }
                T scale = 1;
                for (uint64_t f = 2; f <= j + 1; ++f) scale /= T(f);
                for (uint64_t x = threadIdx.x; x < q; x += blockDim.x) work[offset[b] + x] = g[x] * scale;
                __syncthreads();
                for (uint64_t k = b; k > 0; --k) {
                    const uint64_t previous = power[k - 1];
                    const T* upper = work + offset[k]; const T reciprocal = T(1) / T(j + b - k + 2);
                    for (uint64_t x = threadIdx.x; x < power[k]; x += blockDim.x) ds[offset[k] + x] += upper[x];
                    for (uint64_t x = threadIdx.x; x < previous; x += blockDim.x) {
                        T value = 0;
                        for (uint64_t c = 0; c < d; ++c) value += upper[c * previous + x] * v[c];
                        work[offset[k - 1] + x] = value * reciprocal;
                    }
                    for (uint64_t c = threadIdx.x / 32; c < d; c += blockDim.x / 32) {
                        T value = 0;
                        for (uint64_t x = threadIdx.x % 32; x < previous; x += 32) value += upper[c * previous + x] * powers[offset[k - 1] + x];
                        value = cuda_detail::warp_sum(value);
                        if (!(threadIdx.x % 32)) dv[c] += value * reciprocal;
                    }
                    __syncthreads();
                }
            }
        }
        if constexpr (Reverse) for (uint64_t k = a; k > 0; --k) {
            const uint64_t previous = power[k - 1]; const T* upper = work + offset[k];
            for (uint64_t x = threadIdx.x; x < previous; x += blockDim.x) {
                T value = 0;
                for (uint64_t c = 0; c < d; ++c) value += upper[x * d + c] * v[c];
                work[offset[k - 1] + x] += value;
            }
            for (uint64_t c = threadIdx.x / 32; c < d; c += blockDim.x / 32) {
                T value = 0;
                for (uint64_t x = threadIdx.x % 32; x < previous; x += 32) value += upper[x * d + c] * powers[offset[k - 1] + x];
                value = cuda_detail::warp_sum(value);
                if (!(threadIdx.x % 32)) dv[c] += value;
            }
            __syncthreads();
        }
    }
}

template<bool Reverse, typename T>
__global__ void propagate(const T* states, const T* sources, const T* inc, T* dinc, T* adjoint,
    T* out, T* scratch, uint64_t d, uint64_t n, uint64_t h, uint64_t qlen, uint64_t first,
    uint64_t size, uint64_t stride, const uint64_t* power, const uint64_t* offset) {
    const uint64_t sample = blockIdx.x;
    T* adj = adjoint + sample * stride; T* work = scratch + sample * stride;
    for (uint64_t step = 0; step < size; ++step) {
        const uint64_t slot = Reverse ? size - 1 - step : step, i = first + slot;
        T* dv = dinc + sample * stride + slot * d;
        cuda_detail::horner_backprop<!Reverse>(states + sample * stride + slot * qlen,
            inc + sample * stride + slot * d, adj, dv, d, h, power, offset, work, work + qlen);
        for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x) adj[x] += sources[sample * stride + slot * qlen + x];
        for (uint64_t c = threadIdx.x; c < d; c += blockDim.x) {
            out[(sample * (n + 1) + i) * d + c] -= dv[c]; out[(sample * (n + 1) + i + 1) * d + c] += dv[c];
        }
        __syncthreads();
    }
}
}

namespace {
struct ExperimentResources {
    std::shared_ptr<void>& selected;
    std::shared_ptr<void> saved;
    static inline std::unordered_map<int, std::shared_ptr<void>> prototypes;
    int device;
    explicit ExperimentResources(int id) : selected(cuda_path_workspaces().devices[id].prefix_suffix_state), saved(selected), device(id) {
        selected = prototypes[id];
    }
    ~ExperimentResources() { prototypes[device] = selected; selected = saved; }
};

template<typename T>
void measure(const T* path, const T* final, const T* deriv, T* out, uint64_t batch,
    uint64_t d, uint64_t n, uint64_t m, uint64_t split, uint64_t block, bool scalar,
    uint64_t* peak, cudaStream_t stream) {
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    if (!batch || !n || m < 2) throw std::invalid_argument("invalid experiment shape");
    if (!block) {
        signature_prefix_suffix_backprop_cuda(path, out, deriv, batch, d, n + 1, m, false, false, T(1), scalar, stream, split);
        *peak = Resources::get(device).reserved() + (4ULL << 20);
        return;
    }
    Layout l(d, m, split);
    if (!split) {
        const auto choice = prefix_suffix_selection::select_cuda(true, batch, n, l, 1, false);
        if (choice.a) l.set_split(choice.a);
    }
    block = std::min(n, block);
    const uint64_t qlen = l.offset[l.scan_degree + 1], temporary = l.offset[std::max(l.scan_degree, l.left_degree(m)) + 1];
    const uint64_t rows = mul(block, l.left_degree(m)), maxp = l.power[l.left_degree(m)], maxq = l.power[l.right_degree(m)];
    const uint64_t factor_bytes = option("PYSIGLIB_PS_FACTOR_BYTES", 128ULL << 20);
    if (mul(mul(rows, add(maxp, maxq)), sizeof(T)) > factor_bytes) throw std::runtime_error("requested block exceeds factor workspace limit");
    const uint64_t slots = std::max<uint64_t>(1, std::min({block, uint64_t(256), factor_bytes / mul(2 * temporary, sizeof(T))}));
    const uint64_t scratch_count = std::max(mul(2 * slots, temporary), add(2 * qlen, d));
    uint64_t stride = add(mul(3 * block + 3, qlen), mul(2 * block, d));
    stride = add(stride, add(mul(rows, add(maxp, maxq)), scratch_count));
    stride = add(stride, 63) / 64 * 64;
    const uint64_t metadata = (l.power.size() + l.offset.size()) * sizeof(uint64_t);
    const uint64_t sig_stride = l.offset[m + 1] - !scalar, path_stride = (n + 1) * d;
    const FactorCoefficients<T> coefficients;
    ExperimentResources resources(device);
    Resources::get(device).storage.reset();
    with_factor_workspace<T>(l, device, batch, 32, stride, (metadata + 255) / 256 * 256,
        4ULL << 20, true, "reversal experiment", stream,
        [&](cublasHandle_t handle, uint64_t capacity, T* memory, uint64_t* power, uint64_t* offset) {
        *peak = Resources::get(device).reserved() + (4ULL << 20);
        for (uint64_t base = 0; base < batch; base += capacity) {
            const uint64_t count = std::min(capacity, batch - base);
            T* prefixes = memory; T* suffixes = prefixes + block * qlen;
            T* sources = suffixes + block * qlen; T* inc = sources + block * qlen;
            T* dinc = inc + block * d; T* state = dinc + block * d;
            T* left = state + 3 * qlen; T* right = left + rows * maxp;
            T* work = right + rows * maxq;
            const T* input = path + base * path_stride; const T* saved = final + base * sig_stride;
            const T* gradient = deriv + base * sig_stride; T* output = out + base * path_stride;
            CUDA_CHECK(cudaMemsetAsync(output, 0, count * path_stride * sizeof(T), stream));
            CudaMatrices<T> matrices{handle, count};
            auto sweep = [&]<bool Reverse>() {
                for (uint64_t position = 0; position < n; position += block) {
                    const uint64_t size = std::min(block, n - position), first = Reverse ? n - position - size : position;
                    reconstruct<Reverse><<<unsigned(count), 128, 0, stream>>>(input, saved, gradient, prefixes, suffixes,
                        sources, inc, dinc, state, work, d, n, l.scan_degree, qlen, first, size, position, stride, sig_stride, scalar, power, offset);
                    CUDA_CHECK(cudaGetLastError());
                    for (uint64_t k = l.scan_degree + 1; k <= m; ++k) {
                        const uint64_t a = l.left_degree(k), b = l.right_degree(k), p = l.power[a], q = l.power[b], nr = size * a;
                        factors<Reverse><<<make_cuda_1d_grid(count * nr, 8), 256, 0, stream>>>(prefixes, suffixes, inc,
                            Reverse ? right : left, count, unsigned(d), qlen, unsigned(a), unsigned(b), unsigned(Reverse ? q : p), nr, stride, offset, coefficients);
                        CUDA_CHECK(cudaGetLastError());
                        T* g = const_cast<T*>(gradient + l.offset[k] - !scalar);
                        if constexpr (Reverse) matrices.gemm(Matrix<T>{right, nr, q, q, 1, stride}, Matrix<T>{g, q, p, 1, q, sig_stride}, Matrix<T>{left, nr, p, p, 1, stride}, true);
                        else matrices.gemm(Matrix<T>{left, nr, p, p, 1, stride}, Matrix<T>{g, p, q, q, 1, sig_stride}, Matrix<T>{right, nr, q, q, 1, stride}, true);
                        const uint64_t active_slots = std::min(slots, size);
                        factor_derivatives<Reverse><<<unsigned(count * active_slots), 128, 0, stream>>>(prefixes, suffixes, inc,
                            Reverse ? left : right, sources, dinc, work, count, d, qlen, a, b, size, active_slots, temporary, stride, power, offset);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    propagate<Reverse><<<unsigned(count), 128, 0, stream>>>(Reverse ? prefixes : suffixes, sources, inc, dinc,
                        state + 2 * qlen, output, work, d, n, l.scan_degree, qlen, first, size, stride, power, offset);
                    CUDA_CHECK(cudaGetLastError());
                }
            };
            sweep.template operator()<true>(); sweep.template operator()<false>();
        }
    });
    Resources::get(device).storage.reset();
}
}

static thread_local std::string experiment_last_error;
extern "C" __declspec(dllexport) const char* experiment_error() noexcept { return experiment_last_error.c_str(); }
extern "C" __declspec(dllexport) void experiment_clear() noexcept {
    release_cuda_path_workspaces(); ExperimentResources::prototypes.clear();
}
#define EXPORT(T, S) \
extern "C" __declspec(dllexport) int experiment_##S(const T* path, const T* final, const T* deriv, T* out, \
    uint64_t batch, uint64_t d, uint64_t n, uint64_t m, uint64_t split, uint64_t block, bool scalar, int, uint64_t* peak, void* stream) noexcept { \
    try { measure(path, final, deriv, out, batch, d, n, m, split, block, scalar, peak, static_cast<cudaStream_t>(stream)); return 0; } \
    catch (const std::exception& e) { experiment_last_error = e.what(); return 1; } \
}
EXPORT(float, f)
EXPORT(double, d)
#undef EXPORT
