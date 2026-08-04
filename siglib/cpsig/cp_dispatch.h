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

#pragma once
#include <concepts>
#include <cstdint>
#include "log_sig_method.h"
#include "cp_simd_config.h"
#include "hwy/highway.h"

namespace cpsig {
#if HWY_TARGETS & HWY_SSE2
namespace N_SSE2 {
#include "cp_kernel_api.h"
}
#endif
#if HWY_TARGETS & HWY_AVX2
namespace N_AVX2 {
#include "cp_kernel_api.h"
}
#endif
#if HWY_TARGETS & HWY_NEON_WITHOUT_AES
namespace N_NEON_WITHOUT_AES {
#include "cp_kernel_api.h"
}
#endif
#if HWY_TARGETS & HWY_SCALAR
namespace N_SCALAR {
#include "cp_kernel_api.h"
}
#endif
}
