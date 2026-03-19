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

#include <iostream>
#include <stdexcept>

#include "dll_funcs.h"

void load_cpsig(const std::string& dir_path) {
    //////////////////////////////////////////////
    // Load cpsig
    //////////////////////////////////////////////
    std::string cpsig_path = dir_path + "\\cpsig.dll";

    std::cout << "Loading cpsig from " << cpsig_path << std::endl;


#if defined(_WIN32) && !defined __GNUC__

    cpsig = ::LoadLibraryA(cpsig_path.c_str());
    if (cpsig == NULL) {
        // failed to load dll
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD dw = ::GetLastError();

        ::FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            dw,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);

        lpDisplayBuf = (LPVOID)::LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 40) * sizeof(TCHAR));

        ::StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf) / sizeof(TCHAR),
            TEXT("Failed with error %d: %s"),
            dw, lpMsgBuf);

        LocalFree(lpMsgBuf);
        LocalFree(lpDisplayBuf);
        throw std::runtime_error("Failed to load cpsig");
    }

#else

    cpsig = dlopen((dir_path + "/libcpsig.so").c_str(), RTLD_LAZY | RTLD_DEEPBIND);
    if (!cpsig) {
        fputs(dlerror(), stderr);
        throw std::runtime_error("Failed to load cpsig");
    }

#endif

}

void load_cusig(const std::string& dir_path) {
    //////////////////////////////////////////////
    // Load cusig
    //////////////////////////////////////////////
    std::string cusig_path = dir_path + "\\cusig.dll";

    std::cout << "Loading cusig from " << cusig_path << std::endl;

    cusig = ::LoadLibraryA(cusig_path.c_str());
    if (cusig == NULL) {
        // failed to load dll
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD dw = GetLastError();

        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            dw,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);

        lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 40) * sizeof(TCHAR));
        StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf) / sizeof(TCHAR),
            TEXT("failed with error %d: %s"),
            dw, lpMsgBuf);
        MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

        LocalFree(lpMsgBuf);
        LocalFree(lpDisplayBuf);
        throw std::runtime_error("Failed to load cusig");
    }

    std::cout << "cusig loaded\n" << std::endl;
}

void unload_cpsig() {
#if defined(_WIN32) && !defined __GNUC__
    if (cpsig)
        ::FreeLibrary(cpsig);
#else
    if (cpsig)
        dlclose(cpsig);
#endif
    cpsig = nullptr;
}

void unload_cusig() {
#if defined(_WIN32) && !defined __GNUC__
    if (cusig)
        ::FreeLibrary(cusig);
#else
    if (cusig)
        dlclose(cusig);
#endif
    cusig = nullptr;
}

//////////////////////////////////////////////
// Getting functions pointers
//////////////////////////////////////////////

#define GET_FN(NAME, LIBNAME) NAME = reinterpret_cast<NAME ## _fn>(GET_FN_PTR(LIBNAME, #NAME)); \
    if (!NAME) throw std::runtime_error("Failed to get the address of function " #NAME " from " #LIBNAME " library." );

HMODULE cpsig = nullptr;
HMODULE cusig = nullptr;

sig_length_fn sig_length = nullptr;
log_sig_length_fn log_sig_length = nullptr;
signature_d_fn signature_d = nullptr;
batch_signature_d_fn batch_signature_d = nullptr;
signature_f_fn signature_f = nullptr;
batch_signature_f_fn batch_signature_f = nullptr;
sig_kernel_f_fn sig_kernel_f = nullptr;
batch_sig_kernel_f_fn batch_sig_kernel_f = nullptr;
sig_kernel_d_fn sig_kernel_d = nullptr;
batch_sig_kernel_d_fn batch_sig_kernel_d = nullptr;
batch_sig_combine_d_fn batch_sig_combine_d = nullptr;
batch_sig_combine_cuda_d_fn batch_sig_combine_cuda_d = nullptr;
batch_sig_combine_backprop_d_fn batch_sig_combine_backprop_d = nullptr;
batch_sig_combine_backprop_cuda_d_fn batch_sig_combine_backprop_cuda_d = nullptr;
sig_backprop_d_fn sig_backprop_d = nullptr;
batch_sig_backprop_d_fn batch_sig_backprop_d = nullptr;

sig_kernel_cuda_d_fn sig_kernel_cuda_d = nullptr;
batch_sig_kernel_cuda_d_fn batch_sig_kernel_cuda_d = nullptr;

sig_coef_d_fn sig_coef_d = nullptr;
batch_sig_coef_d_fn batch_sig_coef_d = nullptr;

sig_coef_backprop_d_fn sig_coef_backprop_d = nullptr;
batch_sig_coef_backprop_d_fn batch_sig_coef_backprop_d = nullptr;

sig_coef_cuda_d_fn sig_coef_cuda_d = nullptr;
batch_sig_coef_cuda_d_fn batch_sig_coef_cuda_d = nullptr;

sig_coef_backprop_cuda_d_fn sig_coef_backprop_cuda_d = nullptr;
batch_sig_coef_backprop_cuda_d_fn batch_sig_coef_backprop_cuda_d = nullptr;

sig_kernel_backprop_d_fn sig_kernel_backprop_d = nullptr;
batch_sig_kernel_backprop_d_fn batch_sig_kernel_backprop_d = nullptr;

sig_kernel_backprop_cuda_d_fn sig_kernel_backprop_cuda_d = nullptr;
batch_sig_kernel_backprop_cuda_d_fn batch_sig_kernel_backprop_cuda_d = nullptr;

signature_cuda_f_fn signature_cuda_f = nullptr;
signature_cuda_d_fn signature_cuda_d = nullptr;
batch_signature_cuda_f_fn batch_signature_cuda_f = nullptr;
batch_signature_cuda_d_fn batch_signature_cuda_d = nullptr;

sig_backprop_cuda_d_fn sig_backprop_cuda_d = nullptr;
batch_sig_backprop_cuda_d_fn batch_sig_backprop_cuda_d = nullptr;

set_cache_dir_fn set_cache_dir = nullptr;
prepare_log_sig_fn prepare_log_sig = nullptr;
clear_cache_fn clear_cache = nullptr;

sig_to_log_sig_d_fn sig_to_log_sig_d = nullptr;
batch_sig_to_log_sig_d_fn batch_sig_to_log_sig_d = nullptr;

prepare_log_sig_cuda_fn prepare_log_sig_cuda = nullptr;

sig_to_log_sig_cuda_d_fn sig_to_log_sig_cuda_d = nullptr;
batch_sig_to_log_sig_cuda_d_fn batch_sig_to_log_sig_cuda_d = nullptr;

sig_to_log_sig_backprop_d_fn sig_to_log_sig_backprop_d = nullptr;
batch_sig_to_log_sig_backprop_d_fn batch_sig_to_log_sig_backprop_d = nullptr;

sig_to_log_sig_backprop_cuda_d_fn sig_to_log_sig_backprop_cuda_d = nullptr;
batch_sig_to_log_sig_backprop_cuda_d_fn batch_sig_to_log_sig_backprop_cuda_d = nullptr;

log_sig_combine_d_fn log_sig_combine_d = nullptr;
batch_log_sig_combine_d_fn batch_log_sig_combine_d = nullptr;


void get_cpsig_fn_ptrs()
{
    GET_FN(sig_length, cpsig);
    GET_FN(log_sig_length, cpsig);
    GET_FN(signature_d, cpsig);
    GET_FN(batch_signature_d, cpsig);
    GET_FN(signature_f, cpsig);
    GET_FN(batch_signature_f, cpsig);
    GET_FN(sig_kernel_d, cpsig);
    GET_FN(batch_sig_kernel_d, cpsig);
    GET_FN(sig_kernel_f, cpsig);
    GET_FN(batch_sig_kernel_f, cpsig);
    GET_FN(batch_sig_combine_d, cpsig);
    GET_FN(sig_backprop_d, cpsig);
    GET_FN(batch_sig_backprop_d, cpsig);
    GET_FN(sig_kernel_backprop_d, cpsig);
    GET_FN(batch_sig_kernel_backprop_d, cpsig);
    GET_FN(prepare_log_sig, cpsig);
    GET_FN(set_cache_dir, cpsig);
    GET_FN(clear_cache, cpsig);
    GET_FN(sig_to_log_sig_d, cpsig);
    GET_FN(batch_sig_to_log_sig_d, cpsig);
    GET_FN(sig_to_log_sig_backprop_d, cpsig);
    GET_FN(batch_sig_to_log_sig_backprop_d, cpsig);
    GET_FN(sig_coef_d, cpsig);
    GET_FN(batch_sig_coef_d, cpsig);
    GET_FN(sig_coef_backprop_d, cpsig);
    GET_FN(batch_sig_coef_backprop_d, cpsig);
    GET_FN(batch_sig_combine_backprop_d, cpsig);
    GET_FN(log_sig_combine_d, cpsig);
    GET_FN(batch_log_sig_combine_d, cpsig);
}

void get_cusig_fn_ptrs()
{
    GET_FN(sig_kernel_cuda_d, cusig);
    GET_FN(batch_sig_kernel_cuda_d, cusig);
    GET_FN(batch_sig_kernel_backprop_cuda_d, cusig);
    GET_FN(signature_cuda_f, cusig);
    GET_FN(signature_cuda_d, cusig);
    GET_FN(batch_signature_cuda_f, cusig);
    GET_FN(batch_signature_cuda_d, cusig);
    GET_FN(sig_backprop_cuda_d, cusig);
    GET_FN(batch_sig_backprop_cuda_d, cusig);
    GET_FN(batch_sig_combine_cuda_d, cusig);
    GET_FN(batch_sig_combine_backprop_cuda_d, cusig);
    GET_FN(prepare_log_sig_cuda, cusig);
    GET_FN(sig_to_log_sig_cuda_d, cusig);
    GET_FN(batch_sig_to_log_sig_cuda_d, cusig);
    GET_FN(sig_to_log_sig_backprop_cuda_d, cusig);
    GET_FN(batch_sig_to_log_sig_backprop_cuda_d, cusig);
    GET_FN(sig_coef_cuda_d, cusig);
    GET_FN(batch_sig_coef_cuda_d, cusig);
    GET_FN(sig_coef_backprop_cuda_d, cusig);
    GET_FN(batch_sig_coef_backprop_cuda_d, cusig);
}