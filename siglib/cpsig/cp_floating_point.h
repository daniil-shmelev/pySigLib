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

#include "prefix_suffix.h"
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace cpsig {

class CpuSubnormalMode {
public:
	static constexpr unsigned int mask = 0x8040;
	static unsigned int current() noexcept {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
		return _mm_getcsr() & mask;
#else
		return 0;
#endif
	}

	explicit CpuSubnormalMode(unsigned int mode) noexcept : previous(current()), changed(mode != previous) {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
		if (changed) _mm_setcsr((_mm_getcsr() & ~mask) | (mode & mask));
#endif
	}

	~CpuSubnormalMode() {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
		if (changed) _mm_setcsr((_mm_getcsr() & ~mask) | previous);
#endif
	}

	CpuSubnormalMode(const CpuSubnormalMode&) = delete;
	CpuSubnormalMode& operator=(const CpuSubnormalMode&) = delete;

private:
	unsigned int previous;
	bool changed;
};

class SignatureArithmetic {
public:
	SignatureArithmetic() : scope(requested_mode()) {}

private:
	static unsigned int requested_mode() {
		const auto enabled = prefix_suffix::option("PYSIGLIB_CPU_FLUSH_SUBNORMALS", 0, true);
		if (enabled > 1)
			throw std::invalid_argument("PYSIGLIB_CPU_FLUSH_SUBNORMALS must be 0 or 1");
#if !defined(_M_X64) && !defined(__x86_64__) && !defined(_M_IX86) && !defined(__i386__)
		if (enabled)
			throw std::invalid_argument("PYSIGLIB_CPU_FLUSH_SUBNORMALS requires an x86 CPU");
#endif
		return CpuSubnormalMode::current() | (enabled ? CpuSubnormalMode::mask : 0);
	}

	CpuSubnormalMode scope;
};

}  // namespace cpsig
