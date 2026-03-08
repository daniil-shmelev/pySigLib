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

// test_app.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#if defined(_WIN32)
    #include <Windows.h>
    #include <strsafe.h>
#else
    #include <stdlib.h>
    #include <stdio.h>
    #include <dlfcn.h>
    #include <float.h>
#endif

#include "dll_funcs.h"
#include "tests.h"


int main(int argc, char* argv[])
{
    std::string dir_path(".");

    if (argc >= 2) {
        dir_path = argv[1];
    }

    load_cpsig(dir_path);
    get_cpsig_fn_ptrs();

    load_cusig(dir_path);
    get_cusig_fn_ptrs();

    //example_batch_signature_d(1000, 6, 100, 6, false, false, true, -1, 10);
    //example_batch_signature_cuda_d(1000, 6, 100, 6, false, false, true, 10); // Min run time: 156ms

    //// ---- Forward pass benchmark: single path, dim=5, len=1000, deg=5 ----
    //example_signature_d(5, 1000, 5);
    //example_signature_cuda_d(5, 1000, 5);

    //// ---- Backprop benchmark: batch=100, dim=6, len=100, deg=6 ----
    //example_batch_sig_backprop_d(100, 6, 100, 6);
    //example_batch_sig_backprop_cuda_d(100, 6, 100, 5);

    example_batch_sig_to_log_sig_d();
    example_batch_sig_to_log_sig_cuda_d();

    example_batch_sig_to_log_sig_backprop_d();
    example_batch_sig_to_log_sig_backprop_cuda_d();

    unload_cpsig();
    unload_cusig();
}