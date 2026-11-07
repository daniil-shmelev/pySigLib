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
#define HWY_TARGET_INCLUDE "cp_bch_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cp_bch.h"
#include "cp_log_signature.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template void log_sig_reduce_<float>(const float*, float*, uint64_t, uint64_t, uint64_t, uint64_t, int);
template void log_sig_reduce_<double>(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, int);

template void log_sig_combine_<float>(const float* log_sig1, const float* log_sig2, float* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template void log_sig_combine_<double>(const double* log_sig1, const double* log_sig2, double* out,
		uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template void batch_log_sig_from_path_<float>(const float* path, float* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs);

template void batch_log_sig_from_path_<double>(const double* path, double* out,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs);

template void batch_log_sig_from_path_backprop_<float>(const float* d_out, float* d_path, const float* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs);

template void batch_log_sig_from_path_backprop_<double>(const double* d_out, double* d_path, const double* path,
		uint64_t batch_size, uint64_t length, uint64_t dimension, uint64_t degree, int n_jobs);

template void log_sig_combine_backprop_<float>(const float* d_out, float* d_ls1, float* d_ls2,
		const float* ls1, const float* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template void log_sig_combine_backprop_<double>(const double* d_out, double* d_ls1, double* d_ls2,
		const double* ls1, const double* ls2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template void log_sig_join_<float>(const float* log_sig, const float* displacement, float* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template void log_sig_join_<double>(const double* log_sig, const double* displacement, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template void log_sig_join_backprop_<float>(const float* d_out, float* d_logsig, float* d_displacement, const float* log_sig, const float* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

template void log_sig_join_backprop_<double>(const double* d_out, double* d_logsig, double* d_displacement, const double* log_sig, const double* displacement, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs);

}
}
HWY_AFTER_NAMESPACE();
