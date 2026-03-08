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

#if defined(_WIN32)
#include <Windows.h>
#include <strsafe.h>
#else
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <float.h>
#endif

#include <string>
#include <limits>

#if defined(_WIN32) && !defined(__GNUC__)
#define CDECL_ __cdecl
#else
#define CDECL_
#endif

void load_cpsig(const std::string&);
void load_cusig(const std::string&);

void unload_cpsig();
void unload_cusig();

void get_cpsig_fn_ptrs();
void get_cusig_fn_ptrs();

using sig_length_fn = uint64_t(CDECL_*)(uint64_t, uint64_t);
using log_sig_length_fn = uint64_t(CDECL_*)(uint64_t, uint64_t);
using signature_d_fn = void(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, double, bool);
using batch_signature_d_fn = void(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double, bool, int);

using signature_f_fn = void(CDECL_*)(const float*, float*, uint64_t, uint64_t, uint64_t, bool, bool, float, bool);
using batch_signature_f_fn = void(CDECL_*)(const float*, float*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, float, bool, int);

using sig_kernel_f_fn = void(CDECL_*)(const float*, float*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
using batch_sig_kernel_f_fn = void(CDECL_*)(const float*, float*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int, bool);

using sig_kernel_d_fn = void(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
using batch_sig_kernel_d_fn = void(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int, bool);

using sig_kernel_cuda_d_fn = void(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);
using batch_sig_kernel_cuda_d_fn = void(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool);

using signature_cuda_f_fn = int(CDECL_*)(const float*, float*, uint64_t, uint64_t, uint64_t, bool, bool, float, bool);
using signature_cuda_d_fn = int(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, double, bool);
using batch_signature_cuda_f_fn = int(CDECL_*)(const float*, float*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, float, bool);
using batch_signature_cuda_d_fn = int(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double, bool);

using sig_coef_d_fn = void(CDECL_*)(const double*, double*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, bool, bool, double, bool);
using batch_sig_coef_d_fn = void(CDECL_*)(const double*, double*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, double, bool, int);

using sig_coef_backprop_d_fn = void(CDECL_*)(const double*, double*, double*, double*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, bool, bool, double);
using batch_sig_coef_backprop_d_fn = void(CDECL_*)(const double*, double*, double*, double*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, double, int);

using sig_coef_cuda_d_fn = int(CDECL_*)(const double*, double*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, bool);
using batch_sig_coef_cuda_d_fn = int(CDECL_*)(const double*, double*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool);

using batch_sig_combine_d_fn = void(CDECL_*)(const double*, const double*, double*, uint64_t, uint64_t, uint64_t, int);
using batch_sig_combine_cuda_d_fn = int(CDECL_*)(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
using batch_sig_combine_backprop_d_fn = int(CDECL_*)(const double*, double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t, int);
using batch_sig_combine_backprop_cuda_d_fn = int(CDECL_*)(const double*, double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t);
using sig_backprop_d_fn = void(CDECL_*)(const double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t, bool, bool, double);
using batch_sig_backprop_d_fn = int(CDECL_*)(const double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double, int);

using sig_backprop_cuda_d_fn = int(CDECL_*)(const double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t, bool, bool, double);
using batch_sig_backprop_cuda_d_fn = int(CDECL_*)(const double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double);

using sig_kernel_backprop_d_fn = void(CDECL_*)(const double*, double*, double, const double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using batch_sig_kernel_backprop_d_fn = void(CDECL_*)(const double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, int);

using sig_kernel_backprop_cuda_d_fn = void(CDECL_*)(const double*, double*, double, const double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using batch_sig_kernel_backprop_cuda_d_fn = void(CDECL_*)(const double*, double*, const double*, const double*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

using set_cache_dir_fn = void(CDECL_*)(const char*);
using prepare_log_sig_fn = void(CDECL_*)(uint64_t, uint64_t, int, bool);
using clear_cache_fn = void(CDECL_*)(bool);

using sig_to_log_sig_d_fn = void(CDECL_*)(const double*, double*, uint64_t, uint64_t, bool, bool, int);
using batch_sig_to_log_sig_d_fn = int(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, bool, bool, int, int);

using prepare_log_sig_cuda_fn = int(CDECL_*)(uint64_t, uint64_t, int);

using sig_to_log_sig_cuda_d_fn = int(CDECL_*)(const double*, double*, uint64_t, uint64_t, int);
using batch_sig_to_log_sig_cuda_d_fn = int(CDECL_*)(const double*, double*, uint64_t, uint64_t, uint64_t, int);

using sig_to_log_sig_backprop_d_fn = int(CDECL_*)(const double*, double*, const double*, uint64_t, uint64_t, bool, bool, int);
using batch_sig_to_log_sig_backprop_d_fn = int(CDECL_*)(const double*, double*, const double*, uint64_t, uint64_t, uint64_t, bool, bool, int, int);

using sig_to_log_sig_backprop_cuda_d_fn = int(CDECL_*)(const double*, double*, const double*, uint64_t, uint64_t, int);
using batch_sig_to_log_sig_backprop_cuda_d_fn = int(CDECL_*)(const double*, double*, const double*, uint64_t, uint64_t, uint64_t, int);

extern HMODULE cpsig;
extern HMODULE cusig;

extern sig_length_fn sig_length;
extern log_sig_length_fn log_sig_length;
extern signature_d_fn signature_d;
extern batch_signature_d_fn batch_signature_d;
extern signature_f_fn signature_f;
extern batch_signature_f_fn batch_signature_f;
extern sig_kernel_f_fn sig_kernel_f;
extern batch_sig_kernel_f_fn batch_sig_kernel_f;
extern sig_kernel_d_fn sig_kernel_d;
extern batch_sig_kernel_d_fn batch_sig_kernel_d;
extern batch_sig_combine_d_fn batch_sig_combine;
extern batch_sig_combine_cuda_d_fn batch_sig_combine_cuda_d;
extern batch_sig_combine_backprop_d_fn batch_sig_combine_backprop_d;
extern batch_sig_combine_backprop_cuda_d_fn batch_sig_combine_backprop_cuda_d;
extern sig_backprop_d_fn sig_backprop_d;
extern batch_sig_backprop_d_fn batch_sig_backprop_d;

extern sig_kernel_cuda_d_fn sig_kernel_cuda_d;
extern batch_sig_kernel_cuda_d_fn batch_sig_kernel_cuda_d;

extern signature_cuda_f_fn signature_cuda_f;
extern signature_cuda_d_fn signature_cuda_d;
extern batch_signature_cuda_f_fn batch_signature_cuda_f;
extern batch_signature_cuda_d_fn batch_signature_cuda_d;

extern sig_coef_d_fn sig_coef_d;
extern batch_sig_coef_d_fn batch_sig_coef_d;

extern sig_coef_backprop_d_fn sig_coef_backprop_d;
extern batch_sig_coef_backprop_d_fn batch_sig_coef_backprop_d;

extern sig_coef_cuda_d_fn sig_coef_cuda_d;
extern batch_sig_coef_cuda_d_fn batch_sig_coef_cuda_d;

extern sig_kernel_backprop_d_fn sig_kernel_backprop;
extern batch_sig_kernel_backprop_d_fn batch_sig_kernel_backprop_d;

extern sig_kernel_backprop_cuda_d_fn sig_kernel_backprop_cuda_d;
extern batch_sig_kernel_backprop_cuda_d_fn batch_sig_kernel_backprop_cuda_d;

extern sig_backprop_cuda_d_fn sig_backprop_cuda_d;
extern batch_sig_backprop_cuda_d_fn batch_sig_backprop_cuda_d;

extern set_cache_dir_fn set_cache_dir;
extern prepare_log_sig_fn prepare_log_sig;
extern clear_cache_fn clear_cache;

extern sig_to_log_sig_d_fn sig_to_log_sig_d;
extern batch_sig_to_log_sig_d_fn batch_sig_to_log_sig_d;

extern prepare_log_sig_cuda_fn prepare_log_sig_cuda;

extern sig_to_log_sig_cuda_d_fn sig_to_log_sig_cuda_d;
extern batch_sig_to_log_sig_cuda_d_fn batch_sig_to_log_sig_cuda_d;

extern sig_to_log_sig_backprop_d_fn sig_to_log_sig_backprop_d;
extern batch_sig_to_log_sig_backprop_d_fn batch_sig_to_log_sig_backprop_d;

extern sig_to_log_sig_backprop_cuda_d_fn sig_to_log_sig_backprop_cuda_d;
extern batch_sig_to_log_sig_backprop_cuda_d_fn batch_sig_to_log_sig_backprop_cuda_d;

#if defined(_WIN32)
#define GET_FN_PTR ::GetProcAddress
#else
#define GET_FN_PTR dlsym
#endif
