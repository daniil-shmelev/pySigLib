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

#include "cppch.h"
#include "cp_simd_config.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cp_polynomial_sig_kernel_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cpsig.h"
#include "cp_polynomial_sig_kernel.h"
#include "macros.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {





template void sig_kernel_poly_<float>(const float* gram, float* out, float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid, int n_jobs);

template void sig_kernel_poly_<double>(const double* gram, double* out, double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid, int n_jobs);

template void sig_kernel_poly_backprop_<float>(const float* gram, float* gram_derivs, const float* output_derivs, const float* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid, int n_jobs);

template void sig_kernel_poly_backprop_<double>(const double* gram, double* gram_derivs, const double* output_derivs, const double* state, uint64_t batch_size, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t order, bool return_grid, int n_jobs);

}
}
HWY_AFTER_NAMESPACE();
