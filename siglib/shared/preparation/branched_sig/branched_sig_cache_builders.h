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

#include "branched_sig_cache.h"

namespace branched_sig_cache_detail {
BranchedSigCache build_bck_branched_sig_cache_(
	uint64_t dimension,
	uint64_t max_nodes);
BranchedSigCache build_mkw_branched_sig_cache_(
	uint64_t dimension,
	uint64_t max_nodes);
}  // namespace branched_sig_cache_detail
