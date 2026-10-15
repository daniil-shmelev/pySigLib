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
#define HWY_TARGET_INCLUDE "cp_branched_signature_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cpsig.h"
#include "cp_branched_signature.h"
#include "cache_lifecycle/cp_branched_cache.h"
#include "macros.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template void branched_signature_<float>(const float* path, float* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time, bool planar, bool scalar_term, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template void branched_signature_<double>(const double* path, double* out, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time, bool planar, bool scalar_term, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template void branched_sig_combine_<float>(const float* bsig1, const float* bsig2, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term);

template void branched_sig_combine_<double>(const double* bsig1, const double* bsig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term);

template void branched_sig_combine_backprop_<float>(const float* bsig1, const float* bsig2, const float* derivs, float* out1, float* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term);

template void branched_sig_combine_backprop_<double>(const double* bsig1, const double* bsig2, const double* derivs, double* out1, double* out2, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term);

template void branched_sig_backprop_<float>(const float* path, float* out, const float* bsig_derivs, const float* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, float end_time, bool planar, bool scalar_term, const float* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

template void branched_sig_backprop_<double>(const double* path, double* out, const double* bsig_derivs, const double* bsig, uint64_t batch_size, uint64_t dimension, uint64_t length, uint64_t max_nodes, int n_jobs, bool time_aug, bool lead_lag, double end_time, bool planar, bool scalar_term, const double* correction, uint64_t correction_len, uint64_t correction_batch_stride, uint64_t correction_segment_stride);

}
}
HWY_AFTER_NAMESPACE();
