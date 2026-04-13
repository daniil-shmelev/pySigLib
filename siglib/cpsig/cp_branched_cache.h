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
#include "cppch.h"
#include "../shared/branched_cache.h"
#include "words.h"  // for PairHash

void prepare_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool use_disk = false, bool planar = false);
const BranchedSigCache& get_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool planar = false);
void clear_branched_sig_cache();
