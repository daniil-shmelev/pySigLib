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
#include "preparation/cache_io.h"

// Shared disk cache infrastructure used by both log_sig_cache and
// cp_branched_cache.  The cache directory and magic number are common
// to all pySigLib on-disk caches.

extern const char* cache_folder_name;

// Thread-safe accessor for the cache directory. Returns a copy under a shared
// lock; lazily initializes to the platform default on first call.
std::filesystem::path get_cache_dir();

void set_default_cache_dir();
void set_cache_dir_(const char* dir);
