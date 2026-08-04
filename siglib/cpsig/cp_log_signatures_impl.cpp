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
#define HWY_TARGET_INCLUDE "cp_log_signatures_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cpsig.h"
#include "cp_log_signature.h"
#include "cp_exp_signature.h"
#include "macros.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template void sig_to_log_sig_<float>(const float* sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template void sig_to_log_sig_<double>(const double* sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template void sig_to_log_sig_backprop_<float>(const float* sig, float* out, const float* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template void sig_to_log_sig_backprop_<double>(const double* sig, double* out, const double* log_sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template void logsig_to_sig_<float>(const float* log_sig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template void logsig_to_sig_<double>(const double* log_sig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template void logsig_to_sig_backprop_<float>(const float* log_sig, float* out, const float* sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

template void logsig_to_sig_backprop_<double>(const double* log_sig, double* out, const double* sig_derivs, uint64_t batch_size, uint64_t dimension, uint64_t degree, bool time_aug, bool lead_lag, LogSigMethod method, bool scalar_term, int n_jobs);

}
}
HWY_AFTER_NAMESPACE();
