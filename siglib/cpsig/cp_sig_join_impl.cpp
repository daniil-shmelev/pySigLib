/* Copyright 2025 Daniil Shmelev
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
#define HWY_TARGET_INCLUDE "cp_sig_join_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cpsig.h"
#include "cp_sig_join.h"
#include "macros.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template void batch_linear_sig_<float>(const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term, int n_jobs);

template void batch_linear_sig_<double>(const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool scalar_term, int n_jobs);

template void sig_join_<float>(const float* sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs);

template void sig_join_<double>(const double* sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs);

template void batch_sig_join_backprop_<float>(const float* d_out, float* d_sig, float* d_displacement, const float* sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs);

template void batch_sig_join_backprop_<double>(const double* d_out, double* d_sig, double* d_displacement, const double* sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool prepend, bool scalar_term, int n_jobs);

}
}
HWY_AFTER_NAMESPACE();
