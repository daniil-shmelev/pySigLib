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

namespace gpu_correction {
using namespace prefix_suffix;
#include "prefix_suffix_correction-inl.h"

struct Shape {
    Algebra l;
    uint64_t n, nodes_count, slots, scratch_size, stride, sig_stride;
    uint64_t prefixes, suffixes, dprefixes, dsuffixes, gradient, left, right, dleft, dright, work;
    const double* nodes;
    const double* weights;
};

template<bool Backward, typename T>
__global__ void scans_kernel(const T* path, const T* correction, T* memory, T* out, const T* deriv,
    Shape s, uint64_t batch, uint64_t dimension, uint64_t length, bool ta, T end_time,
    bool scalar, uint64_t batch_stride, uint64_t segment_stride) {
    for (uint64_t sample = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; sample < batch;
        sample += uint64_t(blockDim.x) * gridDim.x) {
        T* storage = memory + sample * s.stride;
        T* prefixes = storage + s.prefixes; T* suffixes = storage + s.suffixes;
        const T* input = path + sample * length * dimension;
        for (uint64_t i = 0; i < s.n; ++i)
            for (uint64_t c = 0; c < s.l.d; ++c)
                storage[i * s.l.d + c] = point_value(input, i + 1, c, dimension, length, ta, false, end_time)
                    - point_value(input, i, c, dimension, length, ta, false, end_time);
        scans(s.l, s.n, storage, correction + sample * batch_stride, segment_stride, prefixes, suffixes, storage + s.work);
        if constexpr (Backward) {
            zero(storage + s.dprefixes, (s.n + 1) * s.l.size);
            zero(storage + s.dsuffixes, (s.n + 1) * s.l.size);
            zero(storage + s.gradient, s.n * s.l.d);
            for (uint64_t j = scalar ? 0 : 1; j < s.l.size; ++j)
                storage[s.dprefixes + s.n * s.l.size + j] = deriv[sample * s.sig_stride + j - (scalar ? 0 : 1)];
        } else {
            T* output = out + sample * s.sig_stride;
            zero(output, s.sig_stride);
            copy(prefixes + s.n * s.l.size + (scalar ? 0 : 1), output, s.l.size - (scalar ? 0 : 1));
        }
    }
}

template<bool Backward, typename T>
__global__ void factors_kernel(const T* correction, T* memory, Shape s, uint64_t batch,
    uint64_t a, uint64_t b, uint64_t base, uint64_t count, uint64_t rows,
    uint64_t batch_stride, uint64_t segment_stride) {
    for (uint64_t job = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; job < batch * s.slots;
        job += uint64_t(blockDim.x) * gridDim.x) {
        const uint64_t sample = job / s.slots, slot = job % s.slots;
        T* storage = memory + sample * s.stride;
        const T* c = correction + sample * batch_stride;
        const uint64_t p = s.l.power[a], q = s.l.power[b];
        for (uint64_t i = slot; i < count; i += s.slots) {
            const uint64_t segment = base + i;
            factors<Backward>(s.l, a, b, storage + segment * s.l.d, c + segment * segment_stride,
                storage + s.prefixes + segment * s.l.size, storage + s.suffixes + (segment + 1) * s.l.size,
                s.nodes_count, s.nodes, s.weights, storage + s.left + i * rows * p, storage + s.right + i * rows * q,
                storage + (Backward ? s.dleft + i * rows * p : s.work), storage + (Backward ? s.dright + i * rows * q : s.work),
                storage + (Backward ? s.dprefixes + segment * s.l.size : s.work), storage + (Backward ? s.dsuffixes + (segment + 1) * s.l.size : s.work),
                storage + (Backward ? s.gradient + segment * s.l.d : s.work), storage + s.work + slot * s.scratch_size);
        }
    }
}

template<typename T>
__global__ void reverse_kernel(const T* correction, T* memory, T* out, Shape s, uint64_t batch,
    uint64_t dimension, uint64_t length, uint64_t batch_stride, uint64_t segment_stride) {
    for (uint64_t sample = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; sample < batch;
        sample += uint64_t(blockDim.x) * gridDim.x) {
        T* storage = memory + sample * s.stride;
        T* dv = storage + s.gradient;
        reverse_scans(s.l, s.n, storage, correction + sample * batch_stride, segment_stride,
            storage + s.prefixes, storage + s.suffixes, storage + s.dprefixes, storage + s.dsuffixes, dv, storage + s.work);
        T* output = out + sample * length * dimension;
        for (uint64_t i = 0; i < length; ++i)
            for (uint64_t c = 0; c < dimension; ++c)
                output[i * dimension + c] = (i ? dv[(i - 1) * s.l.d + c] : T(0))
                    - (i < s.n ? dv[i * s.l.d + c] : T(0));
    }
}

template<bool Backward, typename T>
void compute(const T* path, T* out, const T* deriv, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t degree, bool ta, T end_time, bool scalar, cudaStream_t stream,
    uint64_t split, const T* correction, uint64_t correction_len, uint64_t batch_stride, uint64_t segment_stride) {
    if (!batch) return;
    const Plan plan(add(dimension, ta ? 1 : 0), dimension, degree, length, split, correction_len, 128, sizeof(T), Backward);
    const auto& l = plan.layout;
    const uint64_t sig_stride = l.offset[degree + 1] - (scalar ? 0 : 1), path_stride = mul(length, dimension);
    mul(mul(batch, sig_stride), sizeof(T)); mul(mul(batch, path_stride), sizeof(T));
    const uint64_t stride = mul(add(plan.count, 63) / 64, 64);
    const uint64_t metadata_count = add(add(l.power.size(), l.offset.size()), plan.raw.offset.size());
    const uint64_t metadata_bytes = mul(add(metadata_count, mul(2, plan.nodes_count)), sizeof(uint64_t));
    const uint64_t row_bytes = add(mul(stride, sizeof(T)), metadata_bytes);
    const uint64_t blas_bytes = 4ULL << 20, budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", UINT64_MAX);
    if (add(row_bytes, blas_bytes) > budget)
        throw std::runtime_error("CUDA corrected prefix-suffix requires " + std::to_string(add(row_bytes, blas_bytes))
            + " workspace bytes for one path; configured limit is " + std::to_string(budget));
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    auto& registry = cuda_path_workspaces();
    std::lock_guard lock(registry.mutex);
    auto& entry = registry.devices[device];
    if (!entry.prefix_suffix_state) entry.prefix_suffix_state = std::make_shared<Resources>(device);
    auto resources = std::static_pointer_cast<Resources>(entry.prefix_suffix_state);
    const uint64_t limit = std::min({batch, uint64_t(32), (budget - blas_bytes) / row_bytes});
    const uint64_t requested_bytes = mul(limit, row_bytes);
    const uint64_t allocated = resources->storage ? mul(resources->storage->row_elements(), resources->storage->capacity()) : 0;
    if (!resources->storage || allocated < row_bytes || allocated > budget - blas_bytes
        || (requested_bytes > resources->requested_bytes && allocated < requested_bytes)) {
        resources->storage.reset();
        resources->storage = std::make_unique<CudaBatchWorkspace<std::byte>>(limit, row_bytes, "corrected prefix-suffix factors");
        resources->requested_bytes = requested_bytes;
    }
    const uint64_t capacity = std::min(limit, mul(resources->storage->row_elements(), resources->storage->capacity()) / row_bytes);
    T* memory = reinterpret_cast<T*>(resources->storage->get());
    auto* power = reinterpret_cast<uint64_t*>(memory + capacity * stride);
    auto* offset = power + l.power.size();
    auto* raw_offset = offset + l.offset.size();
    auto* nodes = reinterpret_cast<double*>(raw_offset + plan.raw.offset.size());
    auto* weights = nodes + plan.nodes_count;
    Shape s{{l.d, l.scan_degree, plan.size, plan.correction_degree, dimension, power, offset, raw_offset},
        plan.n, plan.nodes_count, plan.slots, plan.scratch_size, stride, sig_stride,
        plan.prefixes, plan.suffixes, plan.dprefixes, plan.dsuffixes, plan.gradient,
        plan.left, plan.right, plan.dleft, plan.dright, plan.work, nodes, weights};
    try {
        CUDA_CHECK(cudaMemcpyAsync(power, l.power.data(), l.power.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(offset, l.offset.data(), l.offset.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(raw_offset, plan.raw.offset.data(), plan.raw.offset.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(nodes, plan.nodes.data(), plan.nodes_count * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(weights, plan.weights.data(), plan.nodes_count * sizeof(double), cudaMemcpyHostToDevice, stream));
        blas_check(cublasSetStream(resources->handle, stream));
        blas_check(cublasSetWorkspace(resources->handle, resources->blas_workspace.get(), blas_bytes));
        blas_check(cublasSetMathMode(resources->handle, CUBLAS_PEDANTIC_MATH));
        for (uint64_t base_batch = 0; base_batch < batch; base_batch += capacity) {
            const uint64_t count = std::min(capacity, batch - base_batch);
            const T* corrections = correction + base_batch * batch_stride;
            T* output = out + base_batch * (Backward ? path_stride : sig_stride);
            const T* g = Backward ? deriv + base_batch * sig_stride : nullptr;
            scans_kernel<Backward><<<make_cuda_1d_grid(count, 64), 64, 0, stream>>>(path + base_batch * path_stride,
                corrections, memory, output, g, s, count, dimension, length, ta, end_time, scalar, batch_stride, segment_stride);
            CUDA_CHECK(cudaGetLastError());
            CudaMatrices<T> matrices{resources->handle, stream, count, stride};
            for (uint64_t k = l.scan_degree + 1; k <= degree; ++k) {
                const uint64_t a = l.left_degree(k), b = k - a, p = l.power[a], q = l.power[b];
                const uint64_t rows = mul(plan.nodes_count, ranks(plan.algebra(), a, b));
                for (uint64_t base = 0; base < plan.n; base += plan.segments) {
                    const uint64_t segments = std::min(plan.segments, plan.n - base), total = mul(segments, rows);
                    factors_kernel<false><<<make_cuda_1d_grid(mul(count, s.slots), 128), 128, 0, stream>>>(
                        corrections, memory, s, count, a, b, base, segments, rows, batch_stride, segment_stride);
                    CUDA_CHECK(cudaGetLastError());
                    if constexpr (Backward) {
                        T* grad = const_cast<T*>(g + l.offset[k] - (scalar ? 0 : 1));
                        product(matrices, Matrix<T>{memory + plan.right, total, q, q, 1, stride}, Matrix<T>{grad, q, p, 1, q, sig_stride},
                            Matrix<T>{memory + plan.dleft, total, p, p, 1, stride}, plan.side, memory + plan.matrix_work, true);
                        product(matrices, Matrix<T>{memory + plan.left, total, p, p, 1, stride}, Matrix<T>{grad, p, q, q, 1, sig_stride},
                            Matrix<T>{memory + plan.dright, total, q, q, 1, stride}, plan.side, memory + plan.matrix_work, true);
                        factors_kernel<true><<<make_cuda_1d_grid(mul(count, s.slots), 128), 128, 0, stream>>>(
                            corrections, memory, s, count, a, b, base, segments, rows, batch_stride, segment_stride);
                        CUDA_CHECK(cudaGetLastError());
                    } else {
                        product(matrices, Matrix<T>{memory + plan.left, p, total, 1, p, stride}, Matrix<T>{memory + plan.right, total, q, q, 1, stride},
                            Matrix<T>{output + l.offset[k] - (scalar ? 0 : 1), p, q, q, 1, sig_stride},
                            plan.side, memory + plan.matrix_work, base == 0);
                    }
                }
            }
            if constexpr (Backward) {
                reverse_kernel<<<make_cuda_1d_grid(count, 64), 64, 0, stream>>>(corrections, memory, output, s, count,
                    dimension, length, batch_stride, segment_stride);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) { (void)cudaStreamSynchronize(stream); throw; }
}
} // namespace gpu_correction
