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

#include "cpsig.h"
#include "cp_simd_config.h"
#include "hwy/targets.h"

extern "C" CPSIG_API int64_t cpsig_test_supported_targets() noexcept {
    static const int64_t supported = hwy::SupportedTargets() & HWY_TARGETS;
    return supported;
}

extern "C" CPSIG_API int cpsig_test_set_target(int64_t target) noexcept {
    const int64_t supported = cpsig_test_supported_targets();
    if (target && ((target & supported) != target || (target & (target - 1))))
        return 1;
    hwy::SetSupportedTargetsForTest(target);
    return 0;
}

extern "C" CPSIG_API const char* cpsig_test_target_name(int64_t target) noexcept {
    return hwy::TargetName(target);
}
