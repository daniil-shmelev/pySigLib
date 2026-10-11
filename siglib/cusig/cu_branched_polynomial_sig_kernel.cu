// Copyright 2026. Licensed under the Apache License, Version 2.0.
#include "cusig.h"
#include "cu_utils.h"
#include "cu_macros.h"
#include "../shared/branched_polynomial.h"
#include "../shared/preparation/polynomial_sig_kernel/branched_polynomial_tables.h"
#include <algorithm>
#include <mutex>
#include <type_traits>
template<typename T>
struct BranchedPolyTableCacheEntry {
	int device;
	uint64_t order;
	T* data;
};

static std::mutex branched_poly_table_cache_mutex_;

template<typename T>
static std::vector<BranchedPolyTableCacheEntry<T>>& branched_poly_table_cache_() {
	static std::vector<BranchedPolyTableCacheEntry<T>> cache;
	return cache;
}

template<typename T>
static const T* get_branched_poly_tables_(uint64_t order) {
	int device = 0;
	CUDA_CHECK(cudaGetDevice(&device));
	std::lock_guard<std::mutex> lock(branched_poly_table_cache_mutex_);
	auto& cache = branched_poly_table_cache_<T>();
	for (const auto& entry : cache) {
		if (entry.device == device && entry.order == order)
			return entry.data;
	}

    const BranchedPolynomialTables<T> prepared(order);
    const auto& host_tables = prepared.data;

	CudaBuf<T> device_tables(host_tables.size() * sizeof(T));
	CUDA_CHECK(cudaMemcpy(device_tables.get(), host_tables.data(),
		host_tables.size() * sizeof(T), cudaMemcpyHostToDevice));
	T* const result = device_tables.release();
	cache.push_back({device, order, result});
	return result;
}

template<typename T>
static void release_branched_poly_tables_() {
	auto& cache = branched_poly_table_cache_<T>();
	for (const auto& entry : cache) {
		(void)cudaSetDevice(entry.device);
		(void)cudaFree(entry.data);
	}
	cache.clear();
}

void release_branched_sig_kernel_poly_state() {
	std::lock_guard<std::mutex> lock(branched_poly_table_cache_mutex_);
	int original_device = 0;
	const bool restore_device = cudaGetDevice(&original_device) == cudaSuccess;
	release_branched_poly_tables_<float>();
	release_branched_poly_tables_<double>();
	if (restore_device)
		(void)cudaSetDevice(original_device);
}


namespace {
template<typename T> struct CudaPolyOps {
    T* reduction;
    __device__ static uint64_t index() { return threadIdx.x; }
    __device__ static uint64_t stride() { return blockDim.x; }
    __device__ static void sync() { __syncthreads(); }
    __device__ T sum(T val) {
        reduction[threadIdx.x] = val; sync();
        for (unsigned s = blockDim.x/2; s; s >>= 1) {
            if (threadIdx.x < s) reduction[threadIdx.x] += reduction[threadIdx.x+s];
            sync();
        }
        return reduction[0];
    }
    __device__ void integrate_x(const T* src, T* dst, uint64_t nx, uint64_t width,
                                uint64_t q, const T* A, const T* w, bool transpose) {
        for (uint64_t j = index(); j < width; j += stride()) {
            T prefix = 0;
            for (uint64_t k = 0; k < nx; ++k) {
                const auto i = transpose ? nx-1-k : k;
                for (uint64_t a = 0; a < q; ++a) {
                    T val = (transpose ? w[a] : T(1))*prefix;
                    for (uint64_t b = 0; b < q; ++b)
                        val += (transpose ? A[b*q+a] : A[a*q+b])*src[(i*q+b)*width+j];
                    dst[(i*q+a)*width+j] = val;
                }
                for (uint64_t b = 0; b < q; ++b)
                    prefix += (transpose ? T(1) : w[b])*src[(i*q+b)*width+j];
            }
        }
        sync();
    }
    __device__ void integrate_y(const T* src, T* dst, uint64_t height, uint64_t ny,
                                uint64_t q, const T* A, const T* w, bool transpose) {
        const uint64_t width = ny*q;
        for (uint64_t i = index(); i < height; i += stride()) {
            T prefix = 0;
            for (uint64_t k = 0; k < ny; ++k) {
                const auto j = transpose ? ny-1-k : k;
                for (uint64_t a = 0; a < q; ++a) {
                    T val = (transpose ? w[a] : T(1))*prefix;
                    for (uint64_t b = 0; b < q; ++b)
                        val += (transpose ? A[b*q+a] : A[a*q+b])*src[i*width+j*q+b];
                    dst[i*width+j*q+a] = val;
                }
                for (uint64_t b = 0; b < q; ++b)
                    prefix += (transpose ? T(1) : w[b])*src[i*width+j*q+b];
            }
        }
        sync();
    }
    __device__ void prefix(T* values, uint64_t nx, uint64_t ny, bool reverse) {
        for (uint64_t j = index(); j < ny; j += stride()) {
            T val = 0;
            for (uint64_t k = 0; k < nx; ++k) {
                const auto i = reverse ? nx-1-k : k;
                val += values[i*ny+j]; values[i*ny+j] = val;
            }
        }
        sync();
        for (uint64_t i = index(); i < nx; i += stride()) {
            T val = 0;
            for (uint64_t k = 0; k < ny; ++k) {
                const auto j = reverse ? ny-1-k : k;
                val += values[i*ny+j]; values[i*ny+j] = val;
            }
        }
        sync();
    }
};
template<typename T>
__global__ void forward_kernel(const T* gram, T* out, T* state, T* storage,
                               BranchedPolyDims d, uint64_t depth, bool grid,
                               const T* tables, uint64_t offset, uint64_t count) {
    const auto local = cuda_batch_index();
    if (local >= count) return;
    const auto b = local + offset;
    __shared__ T reduction[256];
    BranchedPolyWorkspace<T> work(storage+local*d.scratch, d);
    branched_poly_forward_pair(gram+b*d.cells, out+b*d.output,
        state ? state+b*d.state : nullptr, work, d, depth, tables, grid, CudaPolyOps<T>{reduction});
}

template<typename T>
__global__ void backprop_kernel(const T* gram, T* grad, const T* derivs,
                                const T* state, T* storage, BranchedPolyDims d,
                                uint64_t depth, bool grid, const T* tables, uint64_t row_elements,
                                uint64_t offset, uint64_t count) {
    const auto local = cuda_batch_index();
    if (local >= count) return;
    const auto b = local + offset;
    __shared__ T reduction[256];
    BranchedPolyWorkspace<T> work(storage+local*row_elements, d);
    branched_poly_backprop_pair(gram+b*d.cells, grad+b*d.cells, derivs+b*d.output,
        state ? state+b*d.state : nullptr, work, d, depth, tables, grid, CudaPolyOps<T>{reduction});
}

template<typename T>
void forward(const T* gram, T* out, T* state, uint64_t batch, uint64_t dim,
             uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid) {
    const BranchedPolyDims d(l1, l2, depth, order, grid, batch, dim);
    if (!batch) return;
    if (!out) throw std::invalid_argument("branched polynomial kernel output pointer must not be null");
    if (d.cells && !gram) throw std::invalid_argument("branched polynomial kernel gram pointer must not be null");
    if (depth <= 1 || !d.cells) {
        int error;
        if constexpr (std::is_same_v<T, float>)
            error = branched_sig_kernel_cuda_f(gram, out, batch, dim, l1, l2, depth, 0, 0, grid);
        else
            error = branched_sig_kernel_cuda_d(gram, out, batch, dim, l1, l2, depth, 0, 0, grid);
        if (error) throw std::runtime_error(get_pysiglib_last_error());
        return;
    }
    CudaBatchWorkspace<T> work(batch, d.scratch, "branched polynomial kernel");
    const T* tables = get_branched_poly_tables_<T>(order);
    for (uint64_t offset = 0; offset < batch;) {
        const auto count = std::min<uint64_t>(work.capacity(), batch-offset);
        const auto chunk = make_cuda_batch_grid_chunk(1, count, 0);
        forward_kernel<T><<<chunk.grid, 256>>>(gram, out, state,
            work.get(), d, depth, grid, tables, offset, chunk.size);
        offset += chunk.size;
    }
    check_cuda_kernel_launch();
}

template<typename T>
void backprop(const T* gram, T* grad, const T* derivs, const T* state,
              uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2,
              uint64_t depth, uint64_t order, bool grid) {
    const BranchedPolyDims d(l1, l2, depth, order, grid, batch, dim);
    if (!batch || !d.cells) return;
    if (!gram) throw std::invalid_argument("branched polynomial kernel gram pointer must not be null");
    if (!grad) throw std::invalid_argument("branched polynomial kernel gram derivative pointer must not be null");
    if (!derivs) throw std::invalid_argument("branched polynomial kernel output derivative pointer must not be null");
    if (depth <= 1) {
        int error;
        if constexpr (std::is_same_v<T, float>)
            error = branched_sig_kernel_backprop_cuda_f(gram, grad, derivs, nullptr, batch, dim, l1, l2, depth, 0, 0, grid);
        else
            error = branched_sig_kernel_backprop_cuda_d(gram, grad, derivs, nullptr, batch, dim, l1, l2, depth, 0, 0, grid);
        if (error) throw std::runtime_error(get_pysiglib_last_error());
        return;
    }
    const auto elements = branched_poly_add(d.scratch, state ? 0 : d.state);
    CudaBatchWorkspace<T> work(batch, elements, "branched polynomial kernel backprop");
    const T* tables = get_branched_poly_tables_<T>(order);
    for (uint64_t offset = 0; offset < batch;) {
        const auto count = std::min<uint64_t>(work.capacity(), batch-offset);
        const auto chunk = make_cuda_batch_grid_chunk(1, count, 0);
        backprop_kernel<T><<<chunk.grid, 256>>>(gram, grad, derivs, state,
            work.get(), d, depth, grid, tables, elements, offset, chunk.size);
        offset += chunk.size;
    }
    check_cuda_kernel_launch();
}
}
extern "C" CUSIG_API int branched_sig_kernel_poly_cuda_f(const float* gram, float* out, float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid) noexcept {
    CUDA_SAFE_CALL(forward<float>(gram, out, state, batch, dim, l1, l2, depth, order, grid));
}
extern "C" CUSIG_API int branched_sig_kernel_poly_backprop_cuda_f(const float* gram, float* grad, const float* derivs, const float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid) noexcept {
    CUDA_SAFE_CALL(backprop<float>(gram, grad, derivs, state, batch, dim, l1, l2, depth, order, grid));
}
extern "C" CUSIG_API int branched_sig_kernel_poly_cuda_d(const double* gram, double* out, double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid) noexcept {
    CUDA_SAFE_CALL(forward<double>(gram, out, state, batch, dim, l1, l2, depth, order, grid));
}
extern "C" CUSIG_API int branched_sig_kernel_poly_backprop_cuda_d(const double* gram, double* grad, const double* derivs, const double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid) noexcept {
    CUDA_SAFE_CALL(backprop<double>(gram, grad, derivs, state, batch, dim, l1, l2, depth, order, grid));
}
