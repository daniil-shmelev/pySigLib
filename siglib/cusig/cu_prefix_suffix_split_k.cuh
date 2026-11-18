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

namespace gpu_split_k {

inline uint64_t chunks(uint64_t rows, uint64_t p, uint64_t q) {
    if (rows < 512 || p * q > 16384 || std::min(p, q) < 4) return 0;
    return std::min<uint64_t>(128, (rows + 127) / 128);
}

template<bool WordRows, bool Factored, typename T>
__global__ void product(const T* left, const T* right, T* partial, uint64_t batch, uint64_t n,
    unsigned p, unsigned q, unsigned pitch, unsigned splits, uint64_t stride, const T* centres, unsigned groups) {
    constexpr unsigned inner = WordRows ? 32 : 8;
    __shared__ T ltile[inner][33];
    __shared__ T rtile[inner][WordRows ? 33 : 32];
    const unsigned tx = threadIdx.x % 8, ty = threadIdx.x / 8;
    const unsigned mt = (p + 31) / 32, nt = (q + 31) / 32;
    const uint64_t tiles = uint64_t(mt) * nt;
    const uint64_t chunk = (n + splits - 1) / splits;
    for (uint64_t task = blockIdx.x; task < batch * splits * tiles; task += gridDim.x) {
        const uint64_t sample = task / (splits * tiles);
        const unsigned s = unsigned(task / tiles % splits), m = unsigned(task % tiles / nt) * 32;
        const unsigned col = unsigned(task % nt) * 32;
        T sums[4][4]{};
        const uint64_t end = min(n, (s + 1) * chunk);
        for (uint64_t k = s * chunk; k < end; k += inner) {
            #pragma unroll
            for (unsigned j = threadIdx.x; j < inner * 32; j += 64) {
                const unsigned x = WordRows ? j / inner : j % 32, y = WordRows ? j % inner : j / 32;
                const uint64_t ri = WordRows ? uint64_t(col + x) * pitch + k + y : (k + y) * pitch + col + x;
                T a = 0;
                if (k + y < end && m + x < p) {
                    if constexpr (Factored) {
                        const uint64_t base = sample * stride + k + y;
                        a = left[base + uint64_t((m + x) / groups) * pitch]
                            * centres[base + uint64_t((m + x) % groups) * pitch];
                    } else {
                        const uint64_t li = WordRows ? uint64_t(m + x) * pitch + k + y : (k + y) * p + m + x;
                        a = left[sample * stride + li];
                    }
                }
                ltile[y][x] = a;
                rtile[y][x] = k + y < end && col + x < q ? right[sample * stride + ri] : T(0);
            }
            __syncthreads();
            #pragma unroll
            for (unsigned j = 0; j < inner; ++j) {
                #pragma unroll
                for (unsigned r = 0; r < 4; ++r) {
                    const T v = ltile[j][ty + r * 8];
                    #pragma unroll
                    for (unsigned c = 0; c < 4; ++c) sums[r][c] += v * rtile[j][tx + c * 8];
                }
            }
            __syncthreads();
        }
        T* dest = partial + sample * stride + uint64_t(s) * p * q;
        #pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
            const unsigned row = m + ty + r * 8;
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c) {
                const unsigned x = col + tx + c * 8;
                if (row < p && x < q) dest[uint64_t(row) * q + x] = sums[r][c];
            }
        }
    }
}

template<typename T>
__global__ void reduce(const T* partial, T* result, uint64_t batch, uint64_t width, unsigned splits,
    uint64_t stride, uint64_t sig_stride, bool first) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * width;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / width, word = x % width;
        const T* v = partial + sample * stride + word;
        T sum = first ? T(0) : result[sample * sig_stride + word];
        for (unsigned split = 0; split < splits; ++split) sum += v[uint64_t(split) * width];
        result[sample * sig_stride + word] = sum;
    }
}

// Each warp contracts a 4 by 8 tile. Lanes traverse consecutive segments.
template<bool Factored, typename T>
__global__ void warp_product(const T* left, const T* right, T* partial, uint64_t batch,
    unsigned n, unsigned p, unsigned q, unsigned pitch, unsigned splits, uint64_t stride,
    const T* centres, unsigned groups) {
    const unsigned lane = threadIdx.x % 32, nw = (q + 7) / 8, tiles = ((p + 3) / 4) * nw;
    const unsigned chunk = (n + splits - 1) / splits;
    for (uint64_t task = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32; task < batch * splits * tiles;
        task += uint64_t(gridDim.x) * 4) {
        const uint64_t sample = task / (uint64_t(tiles) * splits), base = sample * stride;
        const unsigned tile = unsigned(task % tiles), row = tile / nw * 4, col = tile % nw * 8;
        const unsigned split = unsigned(task / tiles % splits), end = min(n, (split + 1) * chunk);
        const T* heads[4];
        const T* mid[4];
        #pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
            const unsigned w = min(row + r, p - 1);
            if constexpr (Factored) {
                heads[r] = left + base + uint64_t(w / groups) * pitch;
                mid[r] = centres + base + uint64_t(w % groups) * pitch;
            } else heads[r] = left + base + uint64_t(w) * pitch;
        }
        T sums[32]{};
        for (unsigned i = split * chunk + lane; i < end; i += 32) {
            T rhs[8];
            #pragma unroll
            for (unsigned c = 0; c < 8; ++c)
                rhs[c] = right[base + uint64_t(min(col + c, q - 1)) * pitch + i];
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                T a = heads[r][i];
                if constexpr (Factored) a *= mid[r][i];
                #pragma unroll
                for (unsigned c = 0; c < 8; ++c) sums[r * 8 + c] += a * rhs[c];
            }
        }
        #pragma unroll
        for (unsigned mask = 16; mask; mask /= 2) {
            #pragma unroll
            for (unsigned k = 0; k < mask; ++k) {
                const T a = sums[k], b = sums[k + mask];
                sums[k] = (lane & mask ? b : a) + __shfl_xor_sync(0xffffffff, lane & mask ? a : b, mask);
            }
        }
        if (row + lane / 8 < p && col + lane % 8 < q)
            partial[base + (uint64_t(split) * p + row + lane / 8) * q + col + lane % 8] = sums[0];
    }
}
template<bool WordRows = false, bool Factored = false, typename T>
void gemm(const T* left, const T* right, T* partial, T* result, uint64_t batch, uint64_t rows,
    uint64_t p, uint64_t q, uint64_t pitch, uint64_t splits, uint64_t stride,
    uint64_t result_stride, bool first, cudaStream_t stream, const T* centres = nullptr, unsigned groups = 0) {
    const uint64_t tiles = mul((p + 31) / 32, (q + 31) / 32);
    const bool direct = splits == 1 && first && result_stride == stride;
    if (WordRows && sizeof(T) == sizeof(double) && batch >= 8 && rows >= 128 && rows <= 512 && q <= 64) {
        const uint64_t warps = batch * splits * ((p + 3) / 4) * ((q + 7) / 8);
        warp_product<Factored><<<make_cuda_1d_grid(warps, 4), 128, 0, stream>>>(left, right,
            direct ? result : partial, batch, unsigned(rows), unsigned(p), unsigned(q), unsigned(pitch), unsigned(splits),
            stride, centres, groups);
    } else product<WordRows, Factored><<<make_cuda_1d_grid(mul(mul(batch, splits), tiles), 1), 64, 0, stream>>>(left, right,
        direct ? result : partial, batch, rows, unsigned(p), unsigned(q), unsigned(pitch), unsigned(splits), stride, centres, groups);
    CUDA_CHECK(cudaGetLastError());
    if (direct) return;
    reduce<<<make_cuda_1d_grid(mul(batch, p * q), 256), 256, 0, stream>>>(partial, result,
        batch, p * q, unsigned(splits), stride, result_stride, first);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace gpu_split_k
