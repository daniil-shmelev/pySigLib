// Experimental CPU product-integration solver and benchmark adapter.
// Include the current finite-difference implementation unchanged so both
// methods use the same compiler flags and native timing boundary.
#include "../../siglib/cpsig/cp_branched_sig_kernel.cpp"
#include <chrono>

namespace {
struct PolyWorkspace {
    std::vector<double> a, b, tmp, prefix, stack;
};
thread_local PolyWorkspace poly_ws;
thread_local BranchedSigKernelWorkspace_<double> fd_ws;

// Piecewise polynomial cumulative integration along the first coordinate.
// The transpose uses a suffix sum and is the exact discrete adjoint.
void integrate_x(const double* src, double* dst, size_t nx, size_t width,
                 size_t q, const double* A, const double* w, bool transpose) {
    auto& prefix = poly_ws.prefix;
    prefix.assign(width, 0.0);
    for (size_t step = 0; step < nx; ++step) {
        const size_t i = transpose ? nx - 1 - step : step;
        for (size_t a = 0; a < q; ++a) {
            double* out = dst + (i * q + a) * width;
            const double factor = transpose ? w[a] : 1.0;
            for (size_t j = 0; j < width; ++j) out[j] = factor * prefix[j];
            for (size_t b = 0; b < q; ++b) {
                const double coeff = transpose ? A[b * q + a] : A[a * q + b];
                const double* in = src + (i * q + b) * width;
                for (size_t j = 0; j < width; ++j) out[j] += coeff * in[j];
            }
        }
        for (size_t b = 0; b < q; ++b) {
            const double* in = src + (i * q + b) * width;
            const double factor = transpose ? 1.0 : w[b];
            for (size_t j = 0; j < width; ++j) prefix[j] += factor * in[j];
        }
    }
}

void integrate_y(const double* src, double* dst, size_t height, size_t ny,
                 size_t q, const double* A, const double* w, bool transpose) {
    const size_t width = ny * q;
    for (size_t i = 0; i < height; ++i) {
        double prefix = 0.0;
        for (size_t step = 0; step < ny; ++step) {
            const size_t j = transpose ? ny - 1 - step : step;
            const double* in = src + i * width + j * q;
            double* out = dst + i * width + j * q;
            for (size_t a = 0; a < q; ++a) {
                double val = (transpose ? w[a] : 1.0) * prefix;
                for (size_t b = 0; b < q; ++b)
                    val += (transpose ? A[b * q + a] : A[a * q + b]) * in[b];
                out[a] = val;
            }
            for (size_t b = 0; b < q; ++b) prefix += (transpose ? 1.0 : w[b]) * in[b];
        }
    }
}

void weight_gram(const double* gram, const double* src, double* dst,
                 size_t nx, size_t ny, size_t q) {
    const size_t width = ny * q;
    for (size_t i = 0; i < nx; ++i)
        for (size_t a = 0; a < q; ++a)
            for (size_t j = 0; j < ny; ++j) {
                const size_t offset = (i * q + a) * width + j * q;
                for (size_t b = 0; b < q; ++b)
                    dst[offset + b] = gram[i * ny + j] * src[offset + b];
            }
}

double endpoint(const double* gram, const double* prev, size_t nx,
                size_t ny, size_t q, const double* w) {
    double total = 0.0;
    for (size_t i = 0; i < nx; ++i)
        for (size_t a = 0; a < q; ++a)
            for (size_t j = 0; j < ny; ++j) {
                const size_t offset = (i * q + a) * ny * q + j * q;
                double cell = 0.0;
                for (size_t b = 0; b < q; ++b) cell += w[b] * prev[offset + b];
                total += gram[i * ny + j] * w[a] * cell;
            }
    return std::exp(total);
}

double poly(const double* gram, double* grad, size_t nx, size_t ny,
            size_t depth, size_t q, const double* A, const double* w) {
    if (grad) std::fill_n(grad, nx * ny, 0.0);
    if (depth == 0 || nx == 0 || ny == 0) return 1.0;
    if (depth == 1) {
        double total = 0.0;
        for (size_t k = 0; k < nx * ny; ++k) total += gram[k];
        const double val = std::exp(total);
        if (grad) std::fill_n(grad, nx * ny, val);
        return val;
    }
    const size_t n = nx * ny * q * q;
    auto& ws = poly_ws;
    ws.a.resize(n); ws.b.resize(n); ws.tmp.resize(n);
    if (grad) ws.stack.resize(depth * n);
    double* prev = grad ? ws.stack.data() : ws.a.data();
    std::fill_n(prev, n, 1.0);
    for (size_t m = 1; m < depth; ++m) {
        double* curr = grad ? ws.stack.data() + m * n
                            : (prev == ws.a.data() ? ws.b.data() : ws.a.data());
        weight_gram(gram, prev, curr, nx, ny, q);
        integrate_x(curr, ws.tmp.data(), nx, ny * q, q, A, w, false);
        integrate_y(ws.tmp.data(), curr, nx * q, ny, q, A, w, false);
        for (size_t k = 0; k < n; ++k) curr[k] = std::exp(curr[k]);
        prev = curr;
    }
    const double result = endpoint(gram, prev, nx, ny, q, w);
    if (!grad) return result;

    // Seed derivatives of the final scalar quadrature, avoiding a final grid.
    for (size_t i = 0; i < nx; ++i)
        for (size_t a = 0; a < q; ++a)
            for (size_t j = 0; j < ny; ++j)
                for (size_t b = 0; b < q; ++b) {
                    const size_t k = (i * q + a) * ny * q + j * q + b;
                    const double seed = result * w[a] * w[b];
                    grad[i * ny + j] += seed * prev[k];
                    ws.a[k] = seed * gram[i * ny + j];
                }

    for (size_t m = depth - 1; m > 0; --m) {
        const double* curr = ws.stack.data() + m * n;
        const double* previous = curr - n;
        for (size_t k = 0; k < n; ++k) ws.a[k] *= curr[k];
        integrate_y(ws.a.data(), ws.tmp.data(), nx * q, ny, q, A, w, true);
        integrate_x(ws.tmp.data(), ws.a.data(), nx, ny * q, q, A, w, true);
        for (size_t i = 0; i < nx; ++i)
            for (size_t a = 0; a < q; ++a)
                for (size_t j = 0; j < ny; ++j)
                    for (size_t b = 0; b < q; ++b) {
                        const size_t k = (i * q + a) * ny * q + j * q + b;
                        grad[i * ny + j] += ws.a[k] * previous[k];
                        ws.a[k] *= gram[i * ny + j];
                    }
    }
    return result;
}

double fd(const double* gram, double* grad, size_t nx, size_t ny,
          size_t depth, size_t refinement, bool forward) {
    double result = 0.0;
    if (forward)
        get_branched_sig_kernel_scalar_(gram, nx + 1, ny + 1, depth,
                                       &result, refinement, refinement, fd_ws);
    if (grad) {
        const double seed = 1.0;
        get_branched_sig_kernel_backprop_(gram, grad, &seed,
            static_cast<const double*>(nullptr), nx + 1, ny + 1, depth,
            refinement, refinement, false, fd_ws);
    }
    return result;
}
} // namespace

extern "C" {
void bsk_reset() {
    poly_ws = PolyWorkspace{};
    fd_ws = BranchedSigKernelWorkspace_<double>{};
}

// mode=0: forward; mode=1: backward (reconstructing forward state);
// mode=2: separate forward then backward, reconstructing state for BOTH methods.
double bsk_bench(int method, int mode, const double* gram, double* out,
                 double* grad, uint64_t nx, uint64_t ny, uint64_t depth,
                 uint64_t parameter, const double* A, const double* w,
                 uint64_t repetitions) {
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t rep = 0; rep < repetitions; ++rep) {
        if (method == 0) {
            const double val = fd(gram, mode ? grad : nullptr, nx, ny, depth, parameter, mode != 1);
            if (mode != 1) *out = val;
        } else {
            if (mode != 1) *out = poly(gram, nullptr, nx, ny, depth, parameter, A, w);
            if (mode) {
                const double val = poly(gram, grad, nx, ny, depth, parameter, A, w);
                if (mode != 1) *out = val;
            }
        }
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / repetitions;
}
}
