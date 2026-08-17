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

#include "branched_sig_cache.h"
#include "branched_sig_cache_builders.h"

BranchedSigCache::BranchedSigCache(
	uint64_t dimension_value,
	uint64_t max_nodes_value,
	bool planar_value
) {
	if (planar_value)
		*this = branched_sig_cache_detail::build_mkw_branched_sig_cache_(
			dimension_value, max_nodes_value);
	else
		*this = branched_sig_cache_detail::build_bck_branched_sig_cache_(
			dimension_value, max_nodes_value);
}
