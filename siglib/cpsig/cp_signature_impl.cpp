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
#define HWY_TARGET_INCLUDE "cp_signature_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cpsig.h"
#include "cp_signature.h"
#include "macros.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template void signature_<float>(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, bool horner, bool scalar_term, int n_jobs, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template void signature_<double>(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool horner, bool scalar_term, int n_jobs, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template void sig_backprop_<float>(const float* path, float* out, const float* sig_derivs, const float* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, bool scalar_term, int n_jobs, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template void sig_backprop_<double>(const double* path, double* out, const double* sig_derivs, const double* sig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool scalar_term, int n_jobs, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

}
}
HWY_AFTER_NAMESPACE();
