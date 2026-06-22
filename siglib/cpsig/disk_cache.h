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

// Shared disk cache infrastructure used by both log_sig_cache and
// cp_branched_cache.  The cache directory and magic number are common
// to all pySigLib on-disk caches.

constexpr uint64_t cache_magic_number = 0x70797369676C6962;  // "pysiglib"

// Upper bound on any single deserialized cache vector. Well above any
// realistic (dim, deg) signature size; blocks gigabyte-scale allocations
// from a corrupt or malicious file in a shared cache dir.
inline constexpr uint64_t MAX_CACHE_VECTOR_SIZE = 1'000'000'000ULL;

extern const char* cache_folder_name;

// Thread-safe accessor for the cache directory. Returns a copy under a shared
// lock; lazily initializes to the platform default on first call.
std::filesystem::path get_cache_dir();

void set_default_cache_dir();
void set_cache_dir_(const char* dir);
void clear_cache_dir_();

void serialize_vector(std::ostream& out, const std::vector<uint64_t>& v);
void deserialize_vector(std::istream& in, std::vector<uint64_t>& out);

// Verify `need` bytes are still available on `in`; throws if not. Leaves
// the stream position unchanged.
inline void check_stream_has_bytes(std::istream& in, uint64_t need, const char* label) {
	const std::streampos here = in.tellg();
	in.seekg(0, std::ios::end);
	const std::streampos end = in.tellg();
	in.seekg(here);
	if (here < 0 || end < 0 || static_cast<uint64_t>(end - here) < need)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
}
