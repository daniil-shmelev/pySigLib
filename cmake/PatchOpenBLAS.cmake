# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

function(replace_kernel_text before after)
    string(FIND "${source}" "${before}" pos)
    if(pos EQUAL -1)
        string(FIND "${source}" "${after}" pos)
        if(pos EQUAL -1)
            message(FATAL_ERROR "Unexpected OpenBLAS kernel layout in ${kernel}")
        endif()
        return()
    endif()
    string(REPLACE "${before}" "${after}" updated "${source}")
    set(source "${updated}" PARENT_SCOPE)
endfunction()

# OpenBLAS 0.3.34 selects panels of up to 512 rows on Zen 4/5.
foreach(name IN ITEMS sgemm_kernel_16x4_skylakex dgemm_kernel_16x2_skylakex)
    set(kernel "${OPENBLAS_SOURCE_DIR}/kernel/x86_64/${name}.S")
    file(READ "${kernel}" original)
    set(source "${original}")
    if(name STREQUAL "sgemm_kernel_16x4_skylakex")
        # One panel of six floats per row. Probe all three Windows stack pages.
        replace_kernel_text("#if defined(OS_WINDOWS)\n#define L_BUFFER_SIZE 8192\n#else\n#define L_BUFFER_SIZE 12288\n#endif"
                            "#define L_BUFFER_SIZE 12288")
        replace_kernel_text("#elif L_BUFFER_SIZE > 12288" "#elif L_BUFFER_SIZE >= 12288")
    else()
        # Two panels of three doubles per row. Probe all six pages, from high to low.
        replace_kernel_text("#define L_BUFFER_SIZE 512*8*4" "#define L_BUFFER_SIZE 512*8*6")
        replace_kernel_text("#define LB2_OFFSET    512*8*2" "#define LB2_OFFSET    512*8*3")
        replace_kernel_text([=[#if   L_BUFFER_SIZE > 16384
#define STACK_TOUCH \
        movl    $0,  4096 * 4(%rsp);]=] [=[#if   L_BUFFER_SIZE == 24576
#define STACK_TOUCH \
        movl    $0,  4096 * 6(%rsp);\
        movl    $0,  4096 * 5(%rsp);\
        movl    $0,  4096 * 4(%rsp);]=])
    endif()
    if(NOT source STREQUAL original)
        file(WRITE "${kernel}" "${source}")
    endif()
endforeach()
