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

#include <algorithm>
#include <cstddef>
#include <cstring>

inline thread_local char pysiglib_last_error_buffer_[4096]{};

inline void clear_pysiglib_last_error() noexcept {
	pysiglib_last_error_buffer_[0] = '\0';
}

inline void set_pysiglib_last_error(const char* message) noexcept {
	if (!message) {
		clear_pysiglib_last_error();
		return;
	}
	const size_t count = std::min(
		std::strlen(message), sizeof(pysiglib_last_error_buffer_) - 1);
	std::memcpy(pysiglib_last_error_buffer_, message, count);
	pysiglib_last_error_buffer_[count] = '\0';
}

inline const char* get_pysiglib_last_error() noexcept {
	return pysiglib_last_error_buffer_;
}
