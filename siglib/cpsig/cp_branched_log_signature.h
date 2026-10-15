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

#include "cp_simd_config.h"
#include "hwy/highway.h"
#if defined(CPSIG_CP_BRANCHED_LOG_SIGNATURE_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_CP_BRANCHED_LOG_SIGNATURE_H_
#undef CPSIG_CP_BRANCHED_LOG_SIGNATURE_H_
#else
#define CPSIG_CP_BRANCHED_LOG_SIGNATURE_H_
#endif
#include "cppch.h"
#include "cache_lifecycle/cp_branched_log_cache.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {


template<std::floating_point T>
void branched_sig_to_log_sig_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	int n_jobs,
	bool planar = false,
	bool scalar_term = true
);


template<std::floating_point T>
void branched_sig_to_log_sig_backprop_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int method,
	int n_jobs,
	bool planar = false,
	bool scalar_term = true
);

}  // namespace HWY_NAMESPACE
}  // namespace cpsig
HWY_AFTER_NAMESPACE();
#endif
