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

#pragma once
#include "cppch.h"
#include "cpsig.h"

// Integer power with overflow detection. Returns 0 on overflow.
uint64_t power(uint64_t base, uint64_t exp) noexcept;

// Populate the level index array: level_index[k] = start offset of level k.
void populate_level_index(uint64_t* level_index, uint64_t dimension, uint64_t degree);

// Length of a branched signature vector (1 + basis size up to max_nodes).
uint64_t branched_sig_length_(uint64_t dimension, uint64_t max_nodes, bool planar = false);
