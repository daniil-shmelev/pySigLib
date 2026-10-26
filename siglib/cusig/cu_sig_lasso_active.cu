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

#include "sig_lasso_core.h"
#include "cu_sig_lasso_active.cuh"
#include "cu_runtime_utils.h"
#include "cu_macros.h"
#include <cusolverDn.h>
#include <cfloat>
#include <cstring>
#include <math_constants.h>

namespace siglasso {
namespace {
void blas_check(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS feature-sign operation failed: " + std::to_string(status));
}
void solver_check(cusolverStatus_t status) {
    if (status == CUSOLVER_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    if (status != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER feature-sign operation failed: " + std::to_string(status));
}
struct Control { uint64_t rank, null_step, dropped, failed; double step; };
struct PoolColumn { const double* values; uint64_t slot; };

__device__ double reduce_sum(double value, double* scratch) {
    scratch[threadIdx.x] = value; __syncthreads();
    for (unsigned int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step) scratch[threadIdx.x] += scratch[threadIdx.x + step];
        __syncthreads();
    }
    const double result = scratch[0]; __syncthreads(); return result;
}
__device__ void compensated_add(double value, double& sum, double& error) {
    const double next = sum + value;
    error += fabs(sum) >= fabs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
}

__global__ void initialise_active(const double* column, double* normalized, const double** pointers, uint64_t* slots,
    double* scales, double* signs, double* gamma, size_t n, size_t p, size_t slot, double scale, double sign) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) normalized[p * n + i] = column[i] / scale;
    if (!blockIdx.x && !threadIdx.x) { pointers[p] = column; slots[p] = slot; scales[p] = scale; signs[p] = sign; gamma[p] = 0; }
}
__global__ void finish_append(double* q, double* r, const double* norm, size_t n, size_t p, size_t stride) {
    const double value = *norm;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        q[p * n + i] = value > DBL_EPSILON * static_cast<double>(n > p + 1 ? n : p + 1) ? q[p * n + i] / value : 0;
    if (!blockIdx.x && !threadIdx.x) r[p * stride + p] = value > DBL_EPSILON * static_cast<double>(n > p + 1 ? n : p + 1) ? value : 0;
}
__global__ void rank_check(const double* r, size_t stride, size_t n, size_t p, Control* out) {
    if (threadIdx.x) return;
    double smallest = CUDART_INF, largest = 0;
    if (p <= n) for (size_t j = 0; j < p; ++j) { smallest = fmin(smallest, fabs(r[j * stride + j])); largest = fmax(largest, fabs(r[j * stride + j])); }
    out->failed = p > n || !(smallest > DBL_EPSILON * static_cast<double>(n > p ? n : p) * largest);
}
__global__ void signed_rhs(const double* scales, const double* signs, double* out, size_t p, double penalty) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < p; j += static_cast<size_t>(gridDim.x) * blockDim.x) out[j] = penalty * signs[j] / scales[j];
}
__global__ void subtract_vectors(const double* a, const double* b, double* out, size_t p) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < p; j += static_cast<size_t>(gridDim.x) * blockDim.x) out[j] = a[j] - b[j];
}
__global__ void residual_from_active(const double* y, const double* const* columns, const double* gamma,
    const double* scales, double* residual, size_t n, size_t p) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        double value = y[i], error = 0;
        for (size_t j = 0; j < p; ++j) compensated_add(-columns[j][i] * (gamma[j] / scales[j]), value, error);
        residual[i] = value + error;
    }
}
__global__ void active_defect(const double* const* columns, const double* residual, const double* scales,
    const double* signs, double* out, size_t n, size_t p, double penalty, bool dual) {
    __shared__ double scratch[128];
    for (size_t j = blockIdx.x; j < p; j += gridDim.x) {
        double value = 0, error = 0;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) compensated_add(columns[j][i] * residual[i], value, error);
        const double total = reduce_sum(value + error, scratch) / scales[j] - penalty * signs[j] / scales[j];
        if (!threadIdx.x) out[j] = dual ? -total : total;
    }
}
__global__ void boundary_step(const double* gamma, const double* direction, const double* signs, double* candidate,
    uint64_t* hits, size_t p, bool null_step, Control* control) {
    __shared__ double crossing[128];
    double first = CUDART_INF;
    for (size_t j = threadIdx.x; j < p; j += blockDim.x) if (signs[j] * direction[j] < 0) first = fmin(first, fmax(0.0, -gamma[j] / direction[j]));
    crossing[threadIdx.x] = first; __syncthreads();
    for (unsigned int step = blockDim.x / 2; step; step /= 2) { if (threadIdx.x < step) crossing[threadIdx.x] = fmin(crossing[threadIdx.x], crossing[threadIdx.x + step]); __syncthreads(); }
    const double step = null_step ? crossing[0] : fmin(1.0, crossing[0]);
    for (size_t j = threadIdx.x; j < p; j += blockDim.x) {
        const double ratio = signs[j] * direction[j] < 0 ? -gamma[j] / direction[j] : CUDART_INF;
        hits[j] = ratio <= step + 16 * DBL_EPSILON * fmax(1.0, fabs(step));
        candidate[j] = hits[j] ? 0 : gamma[j] + step * direction[j];
    }
    __syncthreads();
    if (!threadIdx.x) {
        control->step = step; control->null_step = null_step; control->failed = !isfinite(step); control->dropped = 0;
        for (size_t j = 0; j < p; ++j) if (hits[j]) hits[control->dropped++] = j;
    }
}
__global__ void scaled_defect(double* defect, const double* scales, const double* candidate, size_t p, double norm, double penalty) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < p; j += static_cast<size_t>(gridDim.x) * blockDim.x)
        defect[j] = candidate[j] ? defect[j] / (1 + norm + penalty / scales[j]) : 0;
}
__global__ void absolute_coefficients(const double* gamma, const double* scales, double* out, size_t p) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < p; j += static_cast<size_t>(gridDim.x) * blockDim.x) out[j] = fabs(gamma[j] / scales[j]);
}
__global__ void round_coefficients(double* gamma, const double* scales, size_t p) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < p; j += static_cast<size_t>(gridDim.x) * blockDim.x) gamma[j] = static_cast<double>(static_cast<float>(gamma[j] / scales[j])) * scales[j];
}
__global__ void output_coefficients(const double* gamma, const double* scales, double* out, size_t p) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < p; j += static_cast<size_t>(gridDim.x) * blockDim.x) out[j] = gamma[j] / scales[j];
}
__global__ void delete_factor(double* r, double* projected, double* rotations, size_t rows, size_t p, size_t stride, size_t position) {
    if (threadIdx.x) return;
    for (size_t j = position; j + 1 < p; ++j) for (size_t i = 0; i < rows; ++i) r[j * stride + i] = r[(j + 1) * stride + i];
    for (size_t j = position; j + 1 < rows && j + 1 < p; ++j) {
        const double radius = hypot(r[j * stride + j], r[j * stride + j + 1]);
        const double c = radius ? r[j * stride + j] / radius : 1, s = radius ? r[j * stride + j + 1] / radius : 0;
        rotations[2 * j] = c; rotations[2 * j + 1] = s;
        for (size_t k = j; k + 1 < p; ++k) {
            const double first = r[k * stride + j], second = r[k * stride + j + 1];
            r[k * stride + j] = c * first + s * second; r[k * stride + j + 1] = -s * first + c * second;
        }
        const double first = projected[j], second = projected[j + 1];
        projected[j] = c * first + s * second; projected[j + 1] = -s * first + c * second;
    }
}
__global__ void delete_columns(double* q, double* normalized, const double* rotations, size_t n, size_t rows, size_t p, size_t position) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        for (size_t j = position; j + 1 < rows && j + 1 < p; ++j) {
            const double c = rotations[2 * j], s = rotations[2 * j + 1], first = q[j * n + i], second = q[(j + 1) * n + i];
            q[j * n + i] = c * first + s * second; q[(j + 1) * n + i] = -s * first + c * second;
        }
        for (size_t j = position; j + 1 < p; ++j) normalized[j * n + i] = normalized[(j + 1) * n + i];
    }
}
__global__ void delete_state(const double** columns, uint64_t* slots, double* scales, double* signs, double* gamma, size_t p, size_t position) {
    if (threadIdx.x) return;
    for (size_t j = position; j + 1 < p; ++j) { columns[j] = columns[j + 1]; slots[j] = slots[j + 1]; scales[j] = scales[j + 1]; signs[j] = signs[j + 1]; gamma[j] = gamma[j + 1]; }
}
__global__ void triangular_copy(const double* source, double* r, size_t n, size_t p, size_t rows, size_t stride) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < rows * p; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index % rows, j = index / rows;
        r[j * stride + i] = i <= j ? source[j * n + i] : 0;
    }
}
__global__ void transpose_matrix(const double* source, double* out, size_t rows, size_t columns, size_t stride) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < rows * columns; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index % rows, j = index / rows;
        out[i * columns + j] = source[j * stride + i];
    }
}
__global__ void svd_policy(const double* singular, const double* penalty_projection, size_t n, size_t p, size_t rows, Control* control) {
    if (threadIdx.x) return;
    const double threshold = DBL_EPSILON * static_cast<double>(n > p ? n : p) * singular[0];
    size_t rank = 0;
    while (rank < rows && singular[rank] > threshold) ++rank;
    double total = 0, null_norm = 0;
    for (size_t j = 0; j < p; ++j) { const double square = penalty_projection[j] * penalty_projection[j]; total += square; if (j >= rank) null_norm += square; }
    control->rank = rank; control->null_step = sqrt(null_norm) > 64 * DBL_EPSILON * sqrt(total);
}
__global__ void svd_coordinates(const double* singular, const double* projected_y, const double* projected_penalty,
    double* coordinates, size_t p, size_t rank, bool null_step) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < p; j += static_cast<size_t>(gridDim.x) * blockDim.x)
        coordinates[j] = null_step ? (j >= rank ? -projected_penalty[j] : 0)
            : (j < rank ? projected_y[j] / singular[j] - projected_penalty[j] / (singular[j] * singular[j]) : 0);
}
__global__ void pool_correlations(const PoolColumn* pool, const uint64_t* active, const double* residual, double* scores, size_t n, size_t count, size_t support) {
    __shared__ double scratch[128];
    for (size_t j = blockIdx.x; j < count; j += gridDim.x) {
        bool excluded = false;
        for (size_t k = 0; k < support; ++k) excluded |= pool[j].slot == active[k];
        if (excluded) { if (!threadIdx.x) scores[j] = 0; continue; }
        double value = 0, error = 0;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) compensated_add(pool[j].values[i] * residual[i], value, error);
        const double total = reduce_sum(value + error, scratch) / n;
        if (!threadIdx.x) scores[j] = total;
    }
}
__global__ void choose_candidate(const PoolColumn* pool, const double* scores, size_t count, double alpha, Selection* result) {
    __shared__ double best_values[128];
    __shared__ uint64_t best_positions[128];
    double best = 0; uint64_t position = UINT64_MAX;
    for (size_t j = threadIdx.x; j < count; j += blockDim.x) if (fabs(scores[j]) > alpha * (1 + 32 * DBL_EPSILON) && fabs(scores[j]) > best) { best = fabs(scores[j]); position = j; }
    best_values[threadIdx.x] = best; best_positions[threadIdx.x] = position; __syncthreads();
    for (unsigned int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step) {
            const auto other = threadIdx.x + step;
            if (best_values[other] > best_values[threadIdx.x] || (best_values[other] == best_values[threadIdx.x] && best_positions[other] < best_positions[threadIdx.x])) {
                best_values[threadIdx.x] = best_values[other]; best_positions[threadIdx.x] = best_positions[other];
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) { const auto j = best_positions[0]; result->slot = j == UINT64_MAX ? static_cast<size_t>(-1) : pool[j].slot; result->correlation = j == UINT64_MAX ? 0 : scores[j]; }
}

// The factorisation state owns all numeric arrays on the caller's device.
class CUDAActive final : public ActiveSet {
    Backend& b_;
    cudaStream_t stream_;
    cublasHandle_t blas_;
    cusolverDnHandle_t solver_ = nullptr;
    int device_;
    double alpha_;
    size_t capacity_ = 0, rows_ = 0, pool_count_ = 0;
    Array y_, residual_, q_, r_, normalized_, projected_, gamma_, signs_, scales_, pointers_, slots_;
    Array temporary_, direction_, candidate_, trial_, rotations_, hits_, control_, selection_;
    Array qr_copy_, tau_, solver_workspace_, solver_info_, svd_copy_, singular_, left_, right_, transpose_, svd_y_, svd_penalty_;
    Array pool_, scores_;
    std::shared_ptr<void> solver_account_, metadata_, pool_metadata_;
    std::vector<size_t> host_slots_;
    std::vector<PoolColumn> host_pool_;

    size_t n() const { return y_.size; }
    size_t stride() const { return std::min(n(), capacity_); }
    unsigned grid(size_t count) const { return make_cuda_1d_grid(count, 128); }
    void check() { CUDA_CHECK(cudaGetLastError()); }
    void grow(Array& buffer, size_t count) { if (buffer.size < count) buffer = b_.alloc(count); }
    Control control() { Control value{}; b_.download(control_.data, reinterpret_cast<double*>(&value), sizeof(value) / sizeof(double)); return value; }
    void gemv(cublasOperation_t operation, size_t rows, size_t columns, const double* matrix, size_t leading,
        const double* x, double* y, double alpha = 1, double beta = 0) {
        if (!rows || !columns) return;
        blas_check(cublasDgemv(blas_, operation, static_cast<int>(rows), static_cast<int>(columns), &alpha,
            matrix, static_cast<int>(leading), x, 1, &beta, y, 1));
    }
    void solve(double* rhs, bool transpose) {
        blas_check(cublasDtrsv(blas_, CUBLAS_FILL_MODE_UPPER, transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, static_cast<int>(size()), r_.data, static_cast<int>(stride()), rhs, 1));
    }
    void residual_for(const Array& gamma, Array& residual) {
        residual_from_active<<<grid(n()), 128, 0, stream_>>>(y_.data, reinterpret_cast<const double* const*>(pointers_.data), gamma.data,
            scales_.data, residual.data, n(), size()); check();
    }
    void defect(const Array& residual, Array& out, bool dual = false) {
        active_defect<<<static_cast<unsigned>(std::min<size_t>(size(), 65535)), 128, 0, stream_>>>(reinterpret_cast<const double* const*>(pointers_.data),
            residual.data, scales_.data, signs_.data, out.data, n(), size(), n() * alpha_, dual); check();
    }

    void reserve(size_t count);
    void initialise_solver();
    void rebuild();
    bool direction(bool singular);
    void erase(size_t position);
public:
    CUDAActive(Backend& b, const Array& y, double alpha, cudaStream_t stream, cublasHandle_t blas, int device)
        : b_(b), stream_(stream), blas_(blas), device_(device), alpha_(alpha), y_(y), residual_(b.clone(y)), trial_(b.alloc(y.size)),
          control_(b.alloc(sizeof(Control) / sizeof(double))), selection_(b.alloc(sizeof(Selection) / sizeof(double))) {}
    ~CUDAActive() override { if (solver_) cusolverDnDestroy(solver_); }
    size_t size() const override { return host_slots_.size(); }
    const Array& residual() const override { return residual_; }
    void activate(const Column&, size_t, double) override;
    Correction correct(bool force_rebuild = false) override;
    void refresh(bool rounded) override;
    double penalty() override;
    void dual_residual(Array&) override;
    Selection select(const std::vector<Column>&, const std::vector<size_t>&) override;
    void output(std::vector<size_t>&, std::vector<double>&) override;
};

void CUDAActive::reserve(size_t count) {
    if (count <= capacity_) return;
    const size_t next = std::max<size_t>(16, std::max(count, 2 * capacity_));
    if (next > INT_MAX) throw std::out_of_range("Active signature support exceeds cuSOLVER limits");
    auto q = b_.alloc(multiply(n(), std::min(n(), next))), r = b_.alloc(multiply(std::min(n(), next), next));
    auto normalized = b_.alloc(multiply(n(), next));
    b_.fill(r, 0);
    if (rows_) b_.copy(q_.data, q.data, multiply(n(), rows_));
    if (size()) {
        b_.copy(normalized_.data, normalized.data, multiply(n(), size()));
        CUDA_CHECK(cudaMemcpy2DAsync(r.data, std::min(n(), next) * sizeof(double), r_.data, stride() * sizeof(double),
            rows_ * sizeof(double), size(), cudaMemcpyDeviceToDevice, stream_));
    }
    for (Array* vector : {&projected_, &gamma_, &scales_, &signs_, &pointers_, &slots_}) {
        auto enlarged = b_.alloc(next);
        if (size()) b_.copy(vector->data, enlarged.data, vector == &projected_ ? rows_ : size());
        *vector = std::move(enlarged);
    }
    temporary_ = b_.alloc(next + 1); direction_ = b_.alloc(next); candidate_ = b_.alloc(next);
    rotations_ = b_.alloc(2 * next); hits_ = b_.alloc(next);
    auto metadata = b_.account(multiply(next, sizeof(size_t)));
    host_slots_.reserve(next); metadata_ = std::move(metadata);
    q_ = std::move(q); r_ = std::move(r); normalized_ = std::move(normalized); capacity_ = next;
}

void CUDAActive::initialise_solver() {
    if (solver_) return;
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_));
    solver_account_ = b_.account((major >= 9 ? 32 : 4) * 1024 * 1024 * (stream_ ? 1 : 2));
    solver_check(cusolverDnCreate(&solver_));
    solver_check(cusolverDnSetStream(solver_, stream_));
    solver_info_ = b_.alloc(1);
}

void CUDAActive::rebuild() {
    if (!size()) return;
    auto started = Clock::now();
    initialise_solver();
    rows_ = std::min(n(), size());
    grow(qr_copy_, multiply(n(), size())); grow(tau_, rows_);
    b_.copy(normalized_.data, qr_copy_.data, multiply(n(), size()));
    int qr_workspace = 0, q_workspace = 0;
    solver_check(cusolverDnDgeqrf_bufferSize(solver_, static_cast<int>(n()), static_cast<int>(size()), qr_copy_.data, static_cast<int>(n()), &qr_workspace));
    solver_check(cusolverDnDorgqr_bufferSize(solver_, static_cast<int>(n()), static_cast<int>(rows_), static_cast<int>(rows_),
        q_.data, static_cast<int>(n()), tau_.data, &q_workspace));
    grow(solver_workspace_, std::max(qr_workspace, q_workspace));
    solver_check(cusolverDnDgeqrf(solver_, static_cast<int>(n()), static_cast<int>(size()), qr_copy_.data, static_cast<int>(n()), tau_.data,
        solver_workspace_.data, qr_workspace, reinterpret_cast<int*>(solver_info_.data)));
    double status = 0; b_.download(solver_info_.data, &status, 1);
    int info = 0; std::memcpy(&info, &status, sizeof(info));
    if (info) throw std::runtime_error("cuSOLVER QR factorisation failed");
    triangular_copy<<<grid(rows_ * size()), 128, 0, stream_>>>(qr_copy_.data, r_.data, n(), size(), rows_, stride());
    b_.copy(qr_copy_.data, q_.data, multiply(n(), rows_));
    solver_check(cusolverDnDorgqr(solver_, static_cast<int>(n()), static_cast<int>(rows_), static_cast<int>(rows_), q_.data, static_cast<int>(n()),
        tau_.data, solver_workspace_.data, q_workspace, reinterpret_cast<int*>(solver_info_.data)));
    b_.download(solver_info_.data, &status, 1);
    std::memcpy(&info, &status, sizeof(info));
    if (info) throw std::runtime_error("cuSOLVER QR basis construction failed");
    gemv(CUBLAS_OP_T, n(), rows_, q_.data, n(), y_.data, projected_.data);
    ++b_.stats.qr_rebuilds; b_.stats.factorization_seconds += elapsed(started); check();
}

bool CUDAActive::direction(bool singular) {
    const size_t p = size();
    signed_rhs<<<grid(p), 128, 0, stream_>>>(scales_.data, signs_.data, temporary_.data, p, n() * alpha_);
    if (!singular) {
        solve(temporary_.data, true);
        subtract_vectors<<<grid(p), 128, 0, stream_>>>(projected_.data, temporary_.data, direction_.data, p);
        solve(direction_.data, false);
        for (size_t attempt = 0; attempt < 3; ++attempt) {
            residual_for(direction_, trial_); defect(trial_, temporary_);
            solve(temporary_.data, true); solve(temporary_.data, false);
            b_.axpy(1, temporary_.data, direction_.data, p);
        }
        subtract_vectors<<<grid(p), 128, 0, stream_>>>(direction_.data, gamma_.data, direction_.data, p);
        check(); return false;
    }
    auto started = Clock::now();
    initialise_solver(); ++b_.stats.singular_steps;
    grow(svd_copy_, multiply(rows_, p)); grow(singular_, rows_); grow(left_, multiply(rows_, rows_));
    grow(right_, multiply(p, p)); grow(transpose_, multiply(p, p)); grow(svd_y_, p); grow(svd_penalty_, p); grow(tau_, rows_);
    const bool wide = p > rows_;
    if (wide) transpose_matrix<<<grid(rows_ * p), 128, 0, stream_>>>(r_.data, svd_copy_.data, rows_, p, stride());
    else CUDA_CHECK(cudaMemcpy2DAsync(svd_copy_.data, rows_ * sizeof(double), r_.data, stride() * sizeof(double),
        rows_ * sizeof(double), p, cudaMemcpyDeviceToDevice, stream_));
    int workspace = 0;
    solver_check(cusolverDnDgesvd_bufferSize(solver_, static_cast<int>(p), static_cast<int>(rows_), &workspace));
    grow(solver_workspace_, workspace);
    solver_check(cusolverDnDgesvd(solver_, 'A', 'A', static_cast<int>(p), static_cast<int>(rows_), svd_copy_.data, static_cast<int>(p),
        singular_.data, wide ? right_.data : left_.data, static_cast<int>(p), transpose_.data, static_cast<int>(rows_),
        solver_workspace_.data, workspace, tau_.data, reinterpret_cast<int*>(solver_info_.data)));
    double status = 0; b_.download(solver_info_.data, &status, 1);
    int info = 0; std::memcpy(&info, &status, sizeof(info));
    if (info) throw std::runtime_error("cuSOLVER SVD did not converge");
    if (wide) transpose_matrix<<<grid(rows_ * rows_), 128, 0, stream_>>>(transpose_.data, left_.data, rows_, rows_, rows_);
    else transpose_matrix<<<grid(p * p), 128, 0, stream_>>>(transpose_.data, right_.data, p, p, p);
    gemv(CUBLAS_OP_T, p, p, right_.data, p, temporary_.data, svd_penalty_.data);
    gemv(CUBLAS_OP_T, rows_, rows_, left_.data, rows_, projected_.data, svd_y_.data);
    svd_policy<<<1, 1, 0, stream_>>>(singular_.data, svd_penalty_.data, n(), p, rows_, reinterpret_cast<Control*>(control_.data));
    const auto policy = control();
    svd_coordinates<<<grid(p), 128, 0, stream_>>>(singular_.data, svd_y_.data, svd_penalty_.data, temporary_.data, p, policy.rank, policy.null_step != 0);
    gemv(CUBLAS_OP_N, p, p, right_.data, p, temporary_.data, direction_.data);
    if (!policy.null_step) subtract_vectors<<<grid(p), 128, 0, stream_>>>(direction_.data, gamma_.data, direction_.data, p);
    b_.stats.factorization_seconds += elapsed(started); check();
    return policy.null_step != 0;
}

void CUDAActive::activate(const Column& column, size_t slot, double correlation) {
    auto started = Clock::now();
    reserve(size() + 1);
    const size_t p = size();
    const double scale = std::sqrt(column.norm2);
    if (!(scale > 0)) throw std::logic_error("Cannot activate a zero signature column");
    initialise_active<<<grid(n()), 128, 0, stream_>>>(column.values.data, normalized_.data, reinterpret_cast<const double**>(pointers_.data),
        reinterpret_cast<uint64_t*>(slots_.data), scales_.data, signs_.data, gamma_.data, n(), p, slot, scale, std::copysign(1.0, correlation));
    double* next = p < n() ? q_.data + p * n() : trial_.data;
    b_.copy(normalized_.data + p * n(), next, n());
    b_.fill({r_.data + p * stride(), stride(), r_.owner}, 0);
    for (size_t pass = 0; pass < 2 && rows_; ++pass) {
        gemv(CUBLAS_OP_T, n(), rows_, q_.data, n(), next, temporary_.data);
        b_.axpy(1, temporary_.data, r_.data + p * stride(), rows_);
        gemv(CUBLAS_OP_N, n(), rows_, q_.data, n(), temporary_.data, next, -1, 1);
    }
    if (p < n()) {
        blas_check(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_DEVICE));
        const auto norm_status = cublasDnrm2(blas_, static_cast<int>(n()), next, 1, temporary_.data + capacity_);
        cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST); blas_check(norm_status);
        finish_append<<<grid(n()), 128, 0, stream_>>>(q_.data, r_.data, temporary_.data + capacity_, n(), p, stride());
        blas_check(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_DEVICE));
        const auto dot_status = cublasDdot(blas_, static_cast<int>(n()), next, 1, y_.data, 1, projected_.data + p);
        cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST); blas_check(dot_status);
        ++rows_;
    }
    host_slots_.push_back(slot); ++b_.stats.activations; ++b_.stats.qr_appends;
    b_.stats.factorization_seconds += elapsed(started); check();
}

void CUDAActive::erase(size_t position) {
    auto started = Clock::now();
    delete_factor<<<1, 1, 0, stream_>>>(r_.data, projected_.data, rotations_.data, rows_, size(), stride(), position);
    delete_columns<<<grid(n()), 128, 0, stream_>>>(q_.data, normalized_.data, rotations_.data, n(), rows_, size(), position);
    delete_state<<<1, 1, 0, stream_>>>(reinterpret_cast<const double**>(pointers_.data), reinterpret_cast<uint64_t*>(slots_.data),
        scales_.data, signs_.data, gamma_.data, size(), position);
    host_slots_.erase(host_slots_.begin() + position); rows_ = std::min(n(), size());
    ++b_.stats.drops; ++b_.stats.qr_deletions; b_.stats.factorization_seconds += elapsed(started); check();
}

Correction CUDAActive::correct(bool force_rebuild) {
    if (!size()) return Correction::stationary;
    rank_check<<<1, 1, 0, stream_>>>(r_.data, stride(), n(), size(), reinterpret_cast<Control*>(control_.data));
    bool singular = control().failed != 0;
    if (force_rebuild || singular) {
        rebuild();
        rank_check<<<1, 1, 0, stream_>>>(r_.data, stride(), n(), size(), reinterpret_cast<Control*>(control_.data));
        singular = control().failed != 0;
    }
    const double previous = b_.dot(residual_.data, residual_.data, n()) / (2 * n()) + alpha_ * penalty();
    for (size_t attempt = 0; attempt < 2; ++attempt) {
        const bool null_step = direction(singular);
        boundary_step<<<1, 128, 0, stream_>>>(gamma_.data, direction_.data, signs_.data, candidate_.data,
            reinterpret_cast<uint64_t*>(hits_.data), size(), null_step, reinterpret_cast<Control*>(control_.data));
        const auto next = control();
        if (next.failed) return Correction::failed;
        residual_for(candidate_, trial_);
        absolute_coefficients<<<grid(size()), 128, 0, stream_>>>(candidate_.data, scales_.data, temporary_.data, size());
        const double squared = b_.dot(trial_.data, trial_.data, n());
        const double objective = finite(squared / (2 * n()) + alpha_ * b_.absolute_sum({temporary_.data, size(), temporary_.owner}));
        bool accurate = true;
        if (!null_step && next.step == 1) {
            defect(trial_, temporary_);
            scaled_defect<<<grid(size()), 128, 0, stream_>>>(temporary_.data, scales_.data, candidate_.data, size(), std::sqrt(squared), n() * alpha_);
            const double error = b_.dot(temporary_.data, temporary_.data, size());
            const double limit = 64 * DBL_EPSILON * (n() + size());
            accurate = error <= size() * limit * limit;
        }
        if (accurate && objective <= previous + 256 * DBL_EPSILON * std::max(1.0, previous)) {
            b_.copy(candidate_.data, gamma_.data, size()); b_.copy(trial_.data, residual_.data, n());
            if (next.dropped) {
                std::vector<uint64_t> dropped(next.dropped);
                b_.download(hits_.data, reinterpret_cast<double*>(dropped.data()), dropped.size());
                for (size_t j = dropped.size(); j-- > 0;) erase(dropped[j]);
            }
            return next.dropped || next.step < 1 || null_step ? Correction::boundary : Correction::stationary;
        }
        if (attempt == 0) {
            rebuild(); refresh(false);
            rank_check<<<1, 1, 0, stream_>>>(r_.data, stride(), n(), size(), reinterpret_cast<Control*>(control_.data));
            singular = control().failed != 0;
        }
    }
    return Correction::failed;
}

void CUDAActive::refresh(bool rounded) {
    if (rounded && b_.options.single_precision && size()) round_coefficients<<<grid(size()), 128, 0, stream_>>>(gamma_.data, scales_.data, size());
    residual_for(gamma_, residual_);
}

double CUDAActive::penalty() {
    if (!size()) return 0;
    absolute_coefficients<<<grid(size()), 128, 0, stream_>>>(gamma_.data, scales_.data, temporary_.data, size());
    return b_.absolute_sum({temporary_.data, size(), temporary_.owner});
}

void CUDAActive::dual_residual(Array& out) {
    b_.copy(residual_.data, out.data, n());
    if (!size() || rows_ != size()) return;
    rank_check<<<1, 1, 0, stream_>>>(r_.data, stride(), n(), size(), reinterpret_cast<Control*>(control_.data));
    if (control().failed) return;
    for (size_t pass = 0; pass < 5; ++pass) {
        defect(out, temporary_, true); solve(temporary_.data, true);
        gemv(CUBLAS_OP_N, n(), size(), q_.data, n(), temporary_.data, out.data, 1, 1);
    }
    check();
}

Selection CUDAActive::select(const std::vector<Column>& columns, const std::vector<size_t>& pool) {
    if (pool.empty()) return {};
    if (pool_count_ != pool.size()) {
        auto metadata = b_.account(multiply(pool.size(), sizeof(PoolColumn) + sizeof(size_t)));
        std::vector<size_t> sorted = pool;
        std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) { return word_less(columns[a].word, columns[b].word); });
        host_pool_.clear(); host_pool_.reserve(sorted.size());
        for (auto slot : sorted) if (columns[slot].norm2) host_pool_.push_back({columns[slot].values.data, slot});
        grow(pool_, multiply(host_pool_.size(), sizeof(PoolColumn)) / sizeof(double)); grow(scores_, host_pool_.size());
        if (!host_pool_.empty()) CUDA_CHECK(cudaMemcpyAsync(pool_.data, host_pool_.data(), host_pool_.size() * sizeof(PoolColumn), cudaMemcpyHostToDevice, stream_));
        pool_count_ = pool.size(); pool_metadata_ = std::move(metadata);
    }
    if (host_pool_.empty()) return {};
    pool_correlations<<<static_cast<unsigned>(std::min<size_t>(65535, host_pool_.size())), 128, 0, stream_>>>(reinterpret_cast<const PoolColumn*>(pool_.data),
        reinterpret_cast<const uint64_t*>(slots_.data), residual_.data, scores_.data, n(), host_pool_.size(), size());
    choose_candidate<<<1, 128, 0, stream_>>>(reinterpret_cast<const PoolColumn*>(pool_.data), scores_.data, host_pool_.size(), alpha_, reinterpret_cast<Selection*>(selection_.data));
    Selection chosen; b_.download(selection_.data, reinterpret_cast<double*>(&chosen), sizeof(chosen) / sizeof(double));
    finite(chosen.correlation); return chosen;
}

void CUDAActive::output(std::vector<size_t>& slots, std::vector<double>& beta) {
    slots = host_slots_; beta.resize(size());
    if (size()) {
        output_coefficients<<<grid(size()), 128, 0, stream_>>>(gamma_.data, scales_.data, temporary_.data, size());
        b_.download(temporary_.data, beta.data(), size());
        for (auto value : beta) finite(value);
    }
}

} // namespace

std::unique_ptr<ActiveSet> make_cuda_active(Backend& b, const Array& y, double alpha, cudaStream_t stream, cublasHandle_t blas, int device) {
    return std::make_unique<CUDAActive>(b, y, alpha, stream, blas, device);
}
} // namespace siglasso
