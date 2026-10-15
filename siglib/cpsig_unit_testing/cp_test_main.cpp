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

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifndef CPSIG_PREBUILT_TESTS
extern "C" int64_t cpsig_test_supported_targets() noexcept;
extern "C" int cpsig_test_set_target(int64_t target) noexcept;
extern "C" const char* cpsig_test_target_name(int64_t target) noexcept;
#endif

int main(int argc, char** argv) {
    const char* requested = std::getenv("CPSIG_TEST_TARGET");
    if (requested && std::strcmp(requested, "auto")) {
#ifdef CPSIG_PREBUILT_TESTS
        std::cerr << "Target forcing is unavailable for a prebuilt library\n";
        return 77;
#else
        int64_t remaining = cpsig_test_supported_targets();
        int64_t selected = 0;
        while (remaining) {
            const int64_t target = remaining & -remaining;
            remaining &= ~target;
            if (!std::strcmp(requested, cpsig_test_target_name(target))) selected = target;
        }
        if (!selected || cpsig_test_set_target(selected)) {
            std::cerr << "Unsupported test target: " << requested << '\n';
            return 77;
        }
#endif
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
