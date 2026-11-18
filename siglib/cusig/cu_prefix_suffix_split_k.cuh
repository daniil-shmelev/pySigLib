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

template<typename T>
__global__ void product(const T* left, const T* right, T* partial, uint64_t batch, uint64_t n,
    unsigned p, unsigned q, unsigned pitch, unsigned splits, uint64_t stride) {
    __shared__ T ltile[8][33];
    __shared__ T rtile[8][32];
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
        for (uint64_t k = s * chunk; k < end; k += 8) {
            #pragma unroll
            for (unsigned j = threadIdx.x; j < 8 * 32; j += 64) {
                const unsigned x = j % 32, y = j / 32;
                ltile[y][x] = k + y < end && m + x < p ? left[sample * stride + (k + y) * p + m + x] : T(0);
                rtile[y][x] = k + y < end && col + x < q ? right[sample * stride + (k + y) * pitch + col + x] : T(0);
            }
            __syncthreads();
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
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
template<typename T>
void gemm(const T* left, const T* right, T* partial, T* result, uint64_t batch, uint64_t rows,
    uint64_t p, uint64_t q, uint64_t pitch, uint64_t splits, uint64_t stride,
    uint64_t result_stride, bool first, cudaStream_t stream) {
    const uint64_t tiles = mul((p + 31) / 32, (q + 31) / 32);
    product<<<make_cuda_1d_grid(mul(mul(batch, splits), tiles), 1), 64, 0, stream>>>(left, right, partial,
        batch, rows, unsigned(p), unsigned(q), unsigned(pitch), unsigned(splits), stride);
    CUDA_CHECK(cudaGetLastError());
    reduce<<<make_cuda_1d_grid(mul(batch, p * q), 256), 256, 0, stream>>>(partial, result,
        batch, p * q, unsigned(splits), stride, result_stride, first);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace gpu_split_k
