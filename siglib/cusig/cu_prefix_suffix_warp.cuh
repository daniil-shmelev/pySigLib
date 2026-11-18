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

// One warp owns all factor adjoints of a segment. Intermediate tensors stay in shared memory.
template<typename T, unsigned D>
__global__ void warp_factor_adjoints(const T* prefixes, const T* suffixes, const T* increments,
    const T* dleft, const T* dright, T* dprefix, T* dsuffix, T* dincrements, uint64_t batch,
    unsigned d, uint64_t qlen, unsigned a, unsigned b, uint64_t base, uint64_t rows, uint64_t stride,
    const uint64_t* power, const uint64_t* offset, unsigned workspace, FactorCoefficients<T> coefficients) {
    extern __shared__ unsigned char raw[];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    if constexpr (D) d = D;
    const uint64_t first = base / a, segments = (base + rows - 1) / a - first + 1;
    const unsigned p = unsigned(power[a]), q = unsigned(power[b]), plen = unsigned(offset[a + 1]);
    T* powers = reinterpret_cast<T*>(raw) + warp * workspace;
    T* dpowers = powers + plen;
    T* history = dpowers + plen;
    T* work = history + offset[b];
    for (uint64_t task = uint64_t(blockIdx.x) * warps + warp; task < batch * segments;
        task += uint64_t(gridDim.x) * warps) {
        const uint64_t sample = task / segments, i = first + task % segments, path = sample * stride;
        const T* v = increments + path + i * d;
        const T* pre = prefixes + path + i * qlen;
        const T* suf = suffixes + path + (i + 1) * qlen;
        T* dp = dprefix + path + i * qlen;
        T* dq = dsuffix + path + (i + 1) * qlen;
        T* dv = dincrements + path + i * d;
        if (!lane) powers[0] = 1;
        for (unsigned x = lane; x < plen; x += 32) dpowers[x] = 0;
        __syncwarp();
        for (unsigned k = 1; k <= a; ++k) {
            const unsigned prev = unsigned(offset[k - 1]), here = unsigned(offset[k]);
            for (unsigned x = lane; x < unsigned(power[k]); x += 32)
                powers[here + x] = powers[prev + x / d] * v[x % d];
            __syncwarp();
        }
        for (uint64_t t = max(base, i * a); t < min(base + rows, (i + 1) * a); ++t) {
            const unsigned j = unsigned(t - i * a), tail = unsigned(power[j + 1]), head = unsigned(power[a - 1 - j]);
            const unsigned po = unsigned(offset[j + 1]), level = unsigned(offset[a - 1 - j]);
            const T* gl = dleft + path + (t - base) * p;
            const T* gr = dright + path + (t - base) * q;
            if (head > 1) {
                if (head < 16 && tail >= 32) {
                    for (unsigned x = 0; x < head; ++x) {
                        T sum = 0;
                        for (unsigned y = lane; y < tail; y += 32) sum += gl[x * tail + y] * powers[po + y];
                        sum = cuda_detail::warp_sum(sum);
                        if (!lane) dp[level + x] += sum;
                    }
                } else for (unsigned x = lane; x < head; x += 32) {
                    T sum = 0;
                    for (unsigned y = 0; y < tail; ++y) sum += gl[x * tail + y] * powers[po + y];
                    dp[level + x] += sum;
                }
            }
            if (tail < 16 && head >= 32) {
                for (unsigned y = 0; y < tail; ++y) {
                    T sum = 0;
                    for (unsigned x = lane; x < head; x += 32) sum += gl[x * tail + y] * pre[level + x];
                    sum = cuda_detail::warp_sum(sum);
                    if (!lane) dpowers[po + y] = sum;
                }
            } else for (unsigned y = lane; y < tail; y += 32) {
                T sum = 0;
                for (unsigned x = 0; x < head; ++x) sum += gl[x * tail + y] * pre[level + x];
                dpowers[po + y] = sum;
            }

            if (!lane) history[0] = 1;
            __syncwarp();
            for (unsigned k = 1; k < b; ++k) {
                const unsigned prev = unsigned(power[k - 1]), here = unsigned(offset[k]), lower = unsigned(offset[k - 1]);
                const T reciprocal = coefficients.inverse[j + b - k + 2];
                for (unsigned x = lane; x < prev * d; x += 32)
                    history[here + x] = suf[here + x] + (v[x / prev] * reciprocal) * history[lower + x % prev];
                __syncwarp();
            }
            const T* upper = gr;
            for (unsigned k = b; k; --k) {
                const unsigned prev = unsigned(power[k - 1]), here = unsigned(offset[k]), lower = unsigned(offset[k - 1]);
                const T scale = k == b ? coefficients.factorial[j + 1] : T(1);
                const T reciprocal = coefficients.inverse[j + b - k + 2];
                const T weight = scale * reciprocal;
                for (unsigned x = lane; x < prev * d; x += 32) dq[here + x] += upper[x] * scale;
                if (k > 1) {
                    for (unsigned x = lane; x < prev; x += 32) {
                        T sum = 0;
                        for (unsigned c = 0; c < d; ++c) sum += (upper[c * prev + x] * weight) * v[c];
                        work[lower + x] = sum;
                    }
                }
                if (prev >= 32) {
                    for (unsigned c = 0; c < d; ++c) {
                        T sum = 0;
                        for (unsigned x = lane; x < prev; x += 32) sum += (upper[c * prev + x] * weight) * history[lower + x];
                        sum = cuda_detail::warp_sum(sum);
                        if (!lane) dv[c] += sum;
                    }
                } else for (unsigned c = lane; c < d; c += 32) {
                    T sum = 0;
                    for (unsigned x = 0; x < prev; ++x) sum += (upper[c * prev + x] * weight) * history[lower + x];
                    dv[c] += sum;
                }
                __syncwarp();
                upper = work + lower;
            }
        }
        for (unsigned k = a; k > 1; --k) {
            const unsigned prev = unsigned(power[k - 1]), here = unsigned(offset[k]), lower = unsigned(offset[k - 1]);
            for (unsigned x = lane; x < prev; x += 32) {
                T sum = 0;
                for (unsigned c = 0; c < d; ++c) sum += dpowers[here + x * d + c] * v[c];
                dpowers[lower + x] += sum;
            }
            if (prev >= 32) {
                for (unsigned c = 0; c < d; ++c) {
                    T sum = 0;
                    for (unsigned x = lane; x < prev; x += 32) sum += dpowers[here + x * d + c] * powers[lower + x];
                    sum = cuda_detail::warp_sum(sum);
                    if (!lane) dv[c] += sum;
                }
            } else for (unsigned c = lane; c < d; c += 32) {
                T sum = 0;
                for (unsigned x = 0; x < prev; ++x) sum += dpowers[here + x * d + c] * powers[lower + x];
                dv[c] += sum;
            }
            __syncwarp();
        }
        for (unsigned c = lane; c < d; c += 32) dv[c] += dpowers[1 + c];
        __syncwarp();
    }
}
