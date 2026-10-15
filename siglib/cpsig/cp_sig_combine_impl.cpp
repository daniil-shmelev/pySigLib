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
#define HWY_TARGET_INCLUDE "cp_sig_combine_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "cpsig.h"
#include "cp_sig_combine.h"
#include "macros.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

template void batch_sig_combine_<float>(const float* sig1, const float* sig2, float* out,
        uint64_t batch_size, uint64_t dimension, uint64_t degree,
        bool scalar_term, int n_jobs);

template void batch_sig_combine_<double>(const double* sig1, const double* sig2, double* out,
        uint64_t batch_size, uint64_t dimension, uint64_t degree,
        bool scalar_term, int n_jobs);

template void batch_sig_combine_backprop_<float>(const float* sig_combined_deriv, float* sig1_deriv, float* sig2_deriv,
        const float* sig1, const float* sig2,
        uint64_t batch_size, uint64_t dimension, uint64_t degree,
        bool scalar_term, int n_jobs);

template void batch_sig_combine_backprop_<double>(const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv,
        const double* sig1, const double* sig2,
        uint64_t batch_size, uint64_t dimension, uint64_t degree,
        bool scalar_term, int n_jobs);

}
}
HWY_AFTER_NAMESPACE();
