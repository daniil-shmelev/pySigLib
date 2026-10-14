// Copyright 2026. Licensed under the Apache License, Version 2.0.
#include "cppch.h"
#include "cpsig.h"
#include "macros.h"
#include "cp_dispatch.h"
#include "cache_lifecycle/cp_bch_cache.h"
#include "cache_lifecycle/cp_poly_cache.h"
#include "cache_lifecycle/cp_branched_cache.h"
#include "cache_lifecycle/cp_branched_log_cache.h"
#include "cache_lifecycle/cp_branched_sig_coef_cache.h"


namespace cpsig {
HWY_EXPORT_T(branched_sig_kernel_poly_f, branched_sig_kernel_poly_<float>);
HWY_EXPORT_T(branched_sig_kernel_poly_backprop_f, branched_sig_kernel_poly_backprop_<float>);
HWY_EXPORT_T(branched_sig_kernel_poly_d, branched_sig_kernel_poly_<double>);
HWY_EXPORT_T(branched_sig_kernel_poly_backprop_d, branched_sig_kernel_poly_backprop_<double>);

extern "C" CPSIG_API int branched_sig_kernel_poly_f(const float* gram, float* out, float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_kernel_poly_f)(gram, out, state, batch, dim, l1, l2, depth, order, grid, jobs));
	}
extern "C" CPSIG_API int branched_sig_kernel_poly_backprop_f(const float* gram, float* grad, const float* derivs, const float* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_kernel_poly_backprop_f)(gram, grad, derivs, state, batch, dim, l1, l2, depth, order, grid, jobs));
	}
extern "C" CPSIG_API int branched_sig_kernel_poly_d(const double* gram, double* out, double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_kernel_poly_d)(gram, out, state, batch, dim, l1, l2, depth, order, grid, jobs));
	}
extern "C" CPSIG_API int branched_sig_kernel_poly_backprop_d(const double* gram, double* grad, const double* derivs, const double* state, uint64_t batch, uint64_t dim, uint64_t l1, uint64_t l2, uint64_t depth, uint64_t order, bool grid, int jobs) noexcept {
		SAFE_CALL(HWY_DYNAMIC_DISPATCH_T(branched_sig_kernel_poly_backprop_d)(gram, grad, derivs, state, batch, dim, l1, l2, depth, order, grid, jobs));
	}

}  // namespace cpsig
