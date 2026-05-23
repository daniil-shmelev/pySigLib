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

#include "xla/ffi/api/ffi.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include "cpsig.h"

#ifdef PYSIGLIB_JAX_WITH_CUDA
#include "cusig.h"
#include <cuda_runtime_api.h>
#endif

namespace ffi = xla::ffi;

namespace {

inline ffi::Error InvalidArgument(const std::string& message) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, message);
}

inline ffi::Error InternalError(const std::string& message) {
    return ffi::Error(ffi::ErrorCode::kInternal, message);
}

inline ffi::Error NativeCallError(const char* fn_name, int err_code) {
    std::ostringstream oss;
    oss << fn_name << " failed with native error code " << err_code;
    return InternalError(oss.str());
}

inline std::string UnsupportedDtypeMessage(ffi::DataType dtype) {
    std::ostringstream oss;
    oss << "only float32 and float64 are supported, got " << dtype;
    return oss.str();
}

struct PathSpec {
    bool is_batch = false;
    std::uint64_t batch_size = 1;
    std::uint64_t length = 0;
    std::uint64_t dimension = 0;
};

template <typename BufferT>
auto BufferDims(BufferT& buffer) {
    return buffer.dimensions();
}

template <typename T>
auto BufferDims(ffi::Result<T>& buffer) {
    return buffer->dimensions();
}

template <typename BufferT>
ffi::DataType BufferElementType(BufferT& buffer) {
    return buffer.element_type();
}

template <typename T>
ffi::DataType BufferElementType(ffi::Result<T>& buffer) {
    return buffer->element_type();
}

template <typename T, typename BufferT>
T* BufferData(BufferT& buffer) {
    return buffer.template typed_data<T>();
}

template <typename T, typename U>
T* BufferData(ffi::Result<U>& buffer) {
    return buffer->template typed_data<T>();
}

inline bool IsSupportedFloatType(ffi::DataType dtype) {
    return dtype == ffi::DataType::F32 || dtype == ffi::DataType::F64;
}

template <typename BufferT>
std::string ValidateFloatBuffer(const char* name, BufferT& buffer) {
    const auto dtype = BufferElementType(buffer);
    if (!IsSupportedFloatType(dtype)) {
        std::ostringstream oss;
        oss << name << " has unsupported dtype " << dtype;
        return oss.str();
    }
    return {};
}

template <typename LhsBuffer, typename RhsBuffer>
std::string ValidateSameFloatDtype(const char* lhs_name, LhsBuffer& lhs, const char* rhs_name, RhsBuffer& rhs) {
    const auto lhs_dtype = BufferElementType(lhs);
    if (!IsSupportedFloatType(lhs_dtype)) {
        std::ostringstream oss;
        oss << lhs_name << " has unsupported dtype " << lhs_dtype;
        return oss.str();
    }

    const auto rhs_dtype = BufferElementType(rhs);
    if (!IsSupportedFloatType(rhs_dtype)) {
        std::ostringstream oss;
        oss << rhs_name << " has unsupported dtype " << rhs_dtype;
        return oss.str();
    }

    if (lhs_dtype != rhs_dtype) {
        std::ostringstream oss;
        oss << lhs_name << " has dtype " << lhs_dtype << " but "
            << rhs_name << " has dtype " << rhs_dtype;
        return oss.str();
    }

    return {};
}

template <typename BufferT>
std::string GetPathSpec(BufferT& path, PathSpec& spec) {
    const auto dims = BufferDims(path);
    if (dims.size() < 2) {
        std::ostringstream oss;
        oss << "path must have at least rank 2, got rank " << dims.size();
        return oss.str();
    }

    spec.length = static_cast<std::uint64_t>(dims[dims.size() - 2]);
    spec.dimension = static_cast<std::uint64_t>(dims[dims.size() - 1]);

    if (dims.size() == 2) {
        spec.is_batch = false;
        spec.batch_size = 1;
    } else {
        spec.is_batch = true;
        spec.batch_size = 1;
        for (size_t i = 0; i < dims.size() - 2; ++i) {
            spec.batch_size *= static_cast<std::uint64_t>(dims[i]);
        }
    }
    return {};
}

template <typename BufferT>
std::string CheckSigOutputShape(BufferT& out, const PathSpec& spec, std::uint64_t sig_len) {
    const auto dims = BufferDims(out);
    if (dims.size() < 1) return "unexpected signature output shape";

    if (static_cast<std::uint64_t>(dims[dims.size() - 1]) != sig_len) {
        return "unexpected signature output shape";
    }

    std::uint64_t leading = 1;
    for (size_t i = 0; i < dims.size() - 1; ++i) {
        leading *= static_cast<std::uint64_t>(dims[i]);
    }
    if (leading != spec.batch_size) {
        return "unexpected signature output shape";
    }
    return {};
}

template <typename BufferT>
std::string CheckGradOutputShape(BufferT& out, const PathSpec& spec) {
    const auto dims = BufferDims(out);
    if (dims.size() < 2) return "unexpected signature backprop output shape";

    if (static_cast<std::uint64_t>(dims[dims.size() - 2]) != spec.length ||
        static_cast<std::uint64_t>(dims[dims.size() - 1]) != spec.dimension) {
        return "unexpected signature backprop output shape";
    }

    std::uint64_t leading = 1;
    for (size_t i = 0; i < dims.size() - 2; ++i) {
        leading *= static_cast<std::uint64_t>(dims[i]);
    }
    if (leading != spec.batch_size) {
        return "unexpected signature backprop output shape";
    }
    return {};
}

inline std::string ValidateArgs(std::int64_t degree, std::int64_t n_jobs, const PathSpec& spec) {
    if (degree < 0) {
        return "degree must be non-negative";
    }
    if (n_jobs == 0) {
        return "n_jobs cannot be 0";
    }
    if (spec.dimension == 0) {
        return "path dimension must be positive";
    }
    return {};
}

inline std::uint64_t AugmentedDimension(std::uint64_t dimension, bool time_aug, bool lead_lag) {
    return (lead_lag ? 2 * dimension : dimension) + (time_aug ? 1 : 0);
}

template <typename BufferT>
std::string GetCorrectionLen(BufferT& correction, std::uint64_t& len) {
    const auto dims = BufferDims(correction);
    if (dims.size() == 1) {
        len = static_cast<std::uint64_t>(dims[0]);
        return {};
    }
    for (const auto dim : dims) {
        if (dim == 0) {
            len = 0;
            return {};
        }
    }
    {
        std::ostringstream oss;
        oss << "correction must have rank 1, got rank " << dims.size();
        return oss.str();
    }
}

template <typename T>
using CpuSigFn = int (*)(const T*, T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool, bool, int, const T*, std::uint64_t) noexcept;

template <typename T>
using CpuSigBackpropFn = int (*)(const T*, T*, const T*, const T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool, int, const T*, std::uint64_t) noexcept;

template <typename T>
struct CpuFns;

template <>
struct CpuFns<float> {
    static constexpr auto sig = signature_f;
    static constexpr auto backprop = sig_backprop_f;
    static constexpr const char* sig_name = "signature_f";
    static constexpr const char* backprop_name = "sig_backprop_f";

    static constexpr auto sig_combine = sig_combine_f;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_f;

    static constexpr auto transform_path = transform_path_f;
    static constexpr auto transform_path_backprop = transform_path_backprop_f;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_f;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_f;

    static constexpr auto log_sig_combine = log_sig_combine_f;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_f;

    static constexpr auto sig_kernel = sig_kernel_f;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_f;

    static constexpr auto bsig = branched_sig_f;
    static constexpr auto bsig_backprop = branched_sig_backprop_f;

    static constexpr auto bsig_combine = branched_sig_combine_f;
    static constexpr auto bsig_combine_backprop = branched_sig_combine_backprop_f;

    static constexpr auto bsig_to_log_sig = branched_sig_to_log_sig_f;
    static constexpr auto bsig_to_log_sig_backprop = branched_sig_to_log_sig_backprop_f;

    static constexpr auto log_sig_from_path = log_sig_from_path_f;
    static constexpr auto log_sig_from_path_backprop = log_sig_from_path_backprop_f;

    static constexpr auto logsig_to_sig = logsig_to_sig_f;
    static constexpr auto logsig_to_sig_backprop = logsig_to_sig_backprop_f;
};

template <>
struct CpuFns<double> {
    static constexpr auto sig = signature_d;
    static constexpr auto backprop = sig_backprop_d;
    static constexpr const char* sig_name = "signature_d";
    static constexpr const char* backprop_name = "sig_backprop_d";

    static constexpr auto sig_combine = sig_combine_d;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_d;

    static constexpr auto transform_path = transform_path_d;
    static constexpr auto transform_path_backprop = transform_path_backprop_d;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_d;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_d;

    static constexpr auto log_sig_combine = log_sig_combine_d;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_d;

    static constexpr auto sig_kernel = sig_kernel_d;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_d;

    static constexpr auto bsig = branched_sig_d;
    static constexpr auto bsig_backprop = branched_sig_backprop_d;

    static constexpr auto bsig_combine = branched_sig_combine_d;
    static constexpr auto bsig_combine_backprop = branched_sig_combine_backprop_d;

    static constexpr auto bsig_to_log_sig = branched_sig_to_log_sig_d;
    static constexpr auto bsig_to_log_sig_backprop = branched_sig_to_log_sig_backprop_d;

    static constexpr auto log_sig_from_path = log_sig_from_path_d;
    static constexpr auto log_sig_from_path_backprop = log_sig_from_path_backprop_d;

    static constexpr auto logsig_to_sig = logsig_to_sig_d;
    static constexpr auto logsig_to_sig_backprop = logsig_to_sig_backprop_d;
};

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
using CudaSigFn = int (*)(const T*, T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool, bool, const T*, std::uint64_t) noexcept;

template <typename T>
using CudaSigBackpropFn = int (*)(const T*, T*, const T*, const T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool, const T*, std::uint64_t) noexcept;

template <typename T>
struct CudaFns;

template <>
struct CudaFns<float> {
    static constexpr auto sig = signature_cuda_f;
    static constexpr auto backprop = sig_backprop_cuda_f;
    static constexpr const char* sig_name = "signature_cuda_f";
    static constexpr const char* backprop_name = "sig_backprop_cuda_f";

    static constexpr auto sig_combine = sig_combine_cuda_f;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_cuda_f;

    static constexpr auto transform_path = transform_path_cuda_f;
    static constexpr auto transform_path_backprop = transform_path_backprop_cuda_f;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_cuda_f;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_cuda_f;

    static constexpr auto log_sig_combine = log_sig_combine_cuda_f;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_cuda_f;

    static constexpr auto sig_kernel = sig_kernel_cuda_f;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_cuda_f;

    static constexpr auto bsig = branched_sig_cuda_f;
    static constexpr auto bsig_backprop = branched_sig_backprop_cuda_f;

    static constexpr auto bsig_combine = branched_sig_combine_cuda_f;
    static constexpr auto bsig_combine_backprop = branched_sig_combine_backprop_cuda_f;

    static constexpr auto bsig_to_log_sig = branched_sig_to_log_sig_cuda_f;
    static constexpr auto bsig_to_log_sig_backprop = branched_sig_to_log_sig_backprop_cuda_f;

    static constexpr auto log_sig_from_path = log_sig_from_path_cuda_f;
    static constexpr auto log_sig_from_path_backprop = log_sig_from_path_backprop_cuda_f;

    static constexpr auto logsig_to_sig = logsig_to_sig_cuda_f;
    static constexpr auto logsig_to_sig_backprop = logsig_to_sig_backprop_cuda_f;
};

template <>
struct CudaFns<double> {
    static constexpr auto sig = signature_cuda_d;
    static constexpr auto backprop = sig_backprop_cuda_d;
    static constexpr const char* sig_name = "signature_cuda_d";
    static constexpr const char* backprop_name = "sig_backprop_cuda_d";

    static constexpr auto sig_combine = sig_combine_cuda_d;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_cuda_d;

    static constexpr auto transform_path = transform_path_cuda_d;
    static constexpr auto transform_path_backprop = transform_path_backprop_cuda_d;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_cuda_d;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_cuda_d;

    static constexpr auto log_sig_combine = log_sig_combine_cuda_d;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_cuda_d;

    static constexpr auto sig_kernel = sig_kernel_cuda_d;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_cuda_d;

    static constexpr auto bsig = branched_sig_cuda_d;
    static constexpr auto bsig_backprop = branched_sig_backprop_cuda_d;

    static constexpr auto bsig_combine = branched_sig_combine_cuda_d;
    static constexpr auto bsig_combine_backprop = branched_sig_combine_backprop_cuda_d;

    static constexpr auto bsig_to_log_sig = branched_sig_to_log_sig_cuda_d;
    static constexpr auto bsig_to_log_sig_backprop = branched_sig_to_log_sig_backprop_cuda_d;

    static constexpr auto log_sig_from_path = log_sig_from_path_cuda_d;
    static constexpr auto log_sig_from_path_backprop = log_sig_from_path_backprop_cuda_d;

    static constexpr auto logsig_to_sig = logsig_to_sig_cuda_d;
    static constexpr auto logsig_to_sig_backprop = logsig_to_sig_backprop_cuda_d;
};
#endif

template <typename F>
ffi::Error DispatchFloatDtype(ffi::DataType dtype, F&& f) {
    switch (dtype) {
        case ffi::DataType::F32:
            return std::forward<F>(f).template operator()<float>();
        case ffi::DataType::F64:
            return std::forward<F>(f).template operator()<double>();
        default:
            return InvalidArgument(UnsupportedDtypeMessage(dtype));
    }
}

template <typename T, typename PathBuffer, typename OutBuffer>
ffi::Error SigCpuImpl(
    CpuSigFn<T> sig_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    bool horner,
    std::int64_t n_jobs,
    PathBuffer& path,
    PathBuffer& correction,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);

    const auto sig_len = sig_length(
        AugmentedDimension(spec.dimension, time_aug, lead_lag),
        static_cast<std::uint64_t>(degree)
    );
    if (sig_len == 0) {
        return InvalidArgument("signature length overflow");
    }
    if (auto msg = CheckSigOutputShape(out, spec, sig_len); !msg.empty()) return InvalidArgument(msg);

    const auto* path_ptr = BufferData<T>(path);
    const auto* correction_ptr = BufferData<T>(correction);
    auto* out_ptr = BufferData<T>(out);

    int err_code = sig_fn(
        path_ptr,
        out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension,
        spec.length,
        static_cast<std::uint64_t>(degree),
        time_aug,
        lead_lag,
        static_cast<T>(end_time),
        horner,
        true,
        static_cast<int>(n_jobs),
        correction_ptr,
        correction_len
    );

    if (err_code != 0) {
        return NativeCallError(fn_name, err_code);
    }

    return ffi::Error::Success();
}

template <typename T, typename PathBuffer, typename SigBuffer, typename CotangentBuffer, typename OutBuffer>
ffi::Error SigBackpropCpuImpl(
    CpuSigBackpropFn<T> sig_backprop_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    PathBuffer& path,
    SigBuffer& sig,
    CotangentBuffer& cotangent,
    PathBuffer& correction,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);

    const auto sig_len = sig_length(
        AugmentedDimension(spec.dimension, time_aug, lead_lag),
        static_cast<std::uint64_t>(degree)
    );
    if (sig_len == 0) {
        return InvalidArgument("signature length overflow");
    }
    if (auto msg = CheckSigOutputShape(sig, spec, sig_len); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = CheckSigOutputShape(cotangent, spec, sig_len); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = CheckGradOutputShape(out, spec); !msg.empty()) return InvalidArgument(msg);

    const auto* path_ptr = BufferData<T>(path);
    const auto* sig_ptr = BufferData<T>(sig);
    const auto* cotangent_ptr = BufferData<T>(cotangent);
    const auto* correction_ptr = BufferData<T>(correction);
    auto* out_ptr = BufferData<T>(out);

    int err_code = sig_backprop_fn(
        path_ptr,
        out_ptr,
        cotangent_ptr,
        sig_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension,
        spec.length,
        static_cast<std::uint64_t>(degree),
        time_aug,
        lead_lag,
        static_cast<T>(end_time),
        true,
        static_cast<int>(n_jobs),
        correction_ptr,
        correction_len
    );

    if (err_code != 0) {
        return NativeCallError(fn_name, err_code);
    }

    return ffi::Error::Success();
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T, typename PathBuffer, typename OutBuffer>
ffi::Error SigCudaImpl(
    cudaStream_t stream,
    CudaSigFn<T> sig_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    bool horner,
    std::int64_t n_jobs,
    PathBuffer& path,
    PathBuffer& correction,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);

    const auto sig_len = sig_length(
        AugmentedDimension(spec.dimension, time_aug, lead_lag),
        static_cast<std::uint64_t>(degree)
    );
    if (sig_len == 0) {
        return InvalidArgument("signature length overflow");
    }
    if (auto msg = CheckSigOutputShape(out, spec, sig_len); !msg.empty()) return InvalidArgument(msg);

    const auto sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) {
        return InternalError(cudaGetErrorString(sync_status));
    }

    const auto* path_ptr = BufferData<T>(path);
    const auto* correction_ptr = BufferData<T>(correction);
    auto* out_ptr = BufferData<T>(out);

    int err_code = sig_fn(
        path_ptr,
        out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension,
        spec.length,
        static_cast<std::uint64_t>(degree),
        time_aug,
        lead_lag,
        static_cast<T>(end_time),
        horner,
        true,
        correction_ptr,
        correction_len
    );

    if (err_code != 0) {
        return NativeCallError(fn_name, err_code);
    }

    return ffi::Error::Success();
}

template <typename T, typename PathBuffer, typename SigBuffer, typename CotangentBuffer, typename OutBuffer>
ffi::Error SigBackpropCudaImpl(
    cudaStream_t stream,
    CudaSigBackpropFn<T> sig_backprop_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    PathBuffer& path,
    SigBuffer& sig,
    CotangentBuffer& cotangent,
    PathBuffer& correction,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);

    const auto sig_len = sig_length(
        AugmentedDimension(spec.dimension, time_aug, lead_lag),
        static_cast<std::uint64_t>(degree)
    );
    if (sig_len == 0) {
        return InvalidArgument("signature length overflow");
    }
    if (auto msg = CheckSigOutputShape(sig, spec, sig_len); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = CheckSigOutputShape(cotangent, spec, sig_len); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = CheckGradOutputShape(out, spec); !msg.empty()) return InvalidArgument(msg);

    const auto sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) {
        return InternalError(cudaGetErrorString(sync_status));
    }

    const auto* path_ptr = BufferData<T>(path);
    const auto* sig_ptr = BufferData<T>(sig);
    const auto* cotangent_ptr = BufferData<T>(cotangent);
    const auto* correction_ptr = BufferData<T>(correction);
    auto* out_ptr = BufferData<T>(out);

    int err_code = sig_backprop_fn(
        path_ptr,
        out_ptr,
        cotangent_ptr,
        sig_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension,
        spec.length,
        static_cast<std::uint64_t>(degree),
        time_aug,
        lead_lag,
        static_cast<T>(end_time),
        true,
        correction_ptr,
        correction_len
    );

    if (err_code != 0) {
        return NativeCallError(fn_name, err_code);
    }

    return ffi::Error::Success();
}
#endif

ffi::Error SigCpu(
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    bool horner,
    std::int64_t n_jobs,
    ffi::AnyBuffer path,
    ffi::AnyBuffer correction,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) {
        return InvalidArgument(msg);
    }
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) {
        return InvalidArgument(msg);
    }

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigCpuImpl<T>(
            CpuFns<T>::sig,
            degree,
            time_aug,
            lead_lag,
            end_time,
            horner,
            n_jobs,
            path,
            correction,
            out,
            CpuFns<T>::sig_name
        );
    });
}

ffi::Error SigBackpropCpu(
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    ffi::AnyBuffer path,
    ffi::AnyBuffer sig,
    ffi::AnyBuffer cotangent,
    ffi::AnyBuffer correction,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "sig", sig); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigBackpropCpuImpl<T>(
            CpuFns<T>::backprop,
            degree,
            time_aug,
            lead_lag,
            end_time,
            n_jobs,
            path,
            sig,
            cotangent,
            correction,
            out,
            CpuFns<T>::backprop_name
        );
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
ffi::Error SigCuda(
    cudaStream_t stream,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    bool horner,
    std::int64_t n_jobs,
    ffi::AnyBuffer path,
    ffi::AnyBuffer correction,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) {
        return InvalidArgument(msg);
    }
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) {
        return InvalidArgument(msg);
    }

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigCudaImpl<T>(
            stream,
            CudaFns<T>::sig,
            degree,
            time_aug,
            lead_lag,
            end_time,
            horner,
            n_jobs,
            path,
            correction,
            out,
            CudaFns<T>::sig_name
        );
    });
}

ffi::Error SigBackpropCuda(
    cudaStream_t stream,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    ffi::AnyBuffer path,
    ffi::AnyBuffer sig,
    ffi::AnyBuffer cotangent,
    ffi::AnyBuffer correction,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "sig", sig); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigBackpropCudaImpl<T>(
            stream,
            CudaFns<T>::backprop,
            degree,
            time_aug,
            lead_lag,
            end_time,
            n_jobs,
            path,
            sig,
            cotangent,
            correction,
            out,
            CudaFns<T>::backprop_name
        );
    });
}
#endif

// ---------------------------------------------------------------------------
// sig_combine
// ---------------------------------------------------------------------------

struct SigSpec {
    bool is_batch = false;
    std::uint64_t batch_size = 1;
    std::uint64_t sig_len = 0;
};

template <typename BufferT>
std::string GetSigSpec(BufferT& buf, SigSpec& spec) {
    const auto dims = BufferDims(buf);
    if (dims.size() < 1) {
        return "signature must have at least rank 1";
    }

    spec.sig_len = static_cast<std::uint64_t>(dims[dims.size() - 1]);

    if (dims.size() == 1) {
        spec.is_batch = false;
        spec.batch_size = 1;
    } else {
        spec.is_batch = true;
        spec.batch_size = 1;
        for (size_t i = 0; i < dims.size() - 1; ++i) {
            spec.batch_size *= static_cast<std::uint64_t>(dims[i]);
        }
    }
    return {};
}

template <typename T>
ffi::Error SigCombineCpuImpl(
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t n_jobs,
    ffi::AnyBuffer& sig1,
    ffi::AnyBuffer& sig2,
    ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig1, spec); !msg.empty()) return InvalidArgument(msg);

    const auto expected_len = sig_length(
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree)
    );
    if (expected_len == 0) return InvalidArgument("signature length overflow");
    if (spec.sig_len != expected_len) return InvalidArgument("sig1 length does not match dimension and degree");

    const auto* sig1_ptr = BufferData<T>(sig1);
    const auto* sig2_ptr = BufferData<T>(sig2);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CpuFns<T>::sig_combine(
        sig1_ptr, sig2_ptr, out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        true,
        static_cast<int>(n_jobs)
    );

    if (err_code != 0) return NativeCallError("sig_combine", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error SigCombineBackpropCpuImpl(
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t n_jobs,
    ffi::AnyBuffer& cotangent,
    ffi::AnyBuffer& sig1,
    ffi::AnyBuffer& sig2,
    ffi::Result<ffi::AnyBuffer>& grad_sig1,
    ffi::Result<ffi::AnyBuffer>& grad_sig2
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig1, spec); !msg.empty()) return InvalidArgument(msg);

    const auto* cot_ptr = BufferData<T>(cotangent);
    const auto* sig1_ptr = BufferData<T>(sig1);
    const auto* sig2_ptr = BufferData<T>(sig2);
    auto* grad1_ptr = BufferData<T>(grad_sig1);
    auto* grad2_ptr = BufferData<T>(grad_sig2);

    int err_code = CpuFns<T>::sig_combine_backprop(
        cot_ptr, grad1_ptr, grad2_ptr, sig1_ptr, sig2_ptr,
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        true,
        static_cast<int>(n_jobs)
    );

    if (err_code != 0) return NativeCallError("sig_combine_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error SigCombineCpu(
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t n_jobs,
    ffi::AnyBuffer sig1,
    ffi::AnyBuffer sig2,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("sig1", sig1, "sig2", sig2); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("sig1", sig1, "out", out); !msg.empty()) return InvalidArgument(msg);

    return DispatchFloatDtype(BufferElementType(sig1), [&]<typename T>() -> ffi::Error {
        return SigCombineCpuImpl<T>(dimension, degree, n_jobs, sig1, sig2, out);
    });
}

ffi::Error SigCombineBackpropCpu(
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t n_jobs,
    ffi::AnyBuffer cotangent,
    ffi::AnyBuffer sig1,
    ffi::AnyBuffer sig2,
    ffi::Result<ffi::AnyBuffer> grad_sig1,
    ffi::Result<ffi::AnyBuffer> grad_sig2
) {
    if (auto msg = ValidateSameFloatDtype("sig1", sig1, "sig2", sig2); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("sig1", sig1, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);

    return DispatchFloatDtype(BufferElementType(sig1), [&]<typename T>() -> ffi::Error {
        return SigCombineBackpropCpuImpl<T>(dimension, degree, n_jobs, cotangent, sig1, sig2, grad_sig1, grad_sig2);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error SigCombineCudaImpl(
    cudaStream_t stream,
    std::int64_t dimension,
    std::int64_t degree,
    ffi::AnyBuffer& sig1,
    ffi::AnyBuffer& sig2,
    ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig1, spec); !msg.empty()) return InvalidArgument(msg);

    const auto sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) return InternalError(cudaGetErrorString(sync_status));

    const auto* sig1_ptr = BufferData<T>(sig1);
    const auto* sig2_ptr = BufferData<T>(sig2);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CudaFns<T>::sig_combine(
        sig1_ptr, sig2_ptr, out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        true
    );

    if (err_code != 0) return NativeCallError("sig_combine_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error SigCombineBackpropCudaImpl(
    cudaStream_t stream,
    std::int64_t dimension,
    std::int64_t degree,
    ffi::AnyBuffer& cotangent,
    ffi::AnyBuffer& sig1,
    ffi::AnyBuffer& sig2,
    ffi::Result<ffi::AnyBuffer>& grad_sig1,
    ffi::Result<ffi::AnyBuffer>& grad_sig2
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig1, spec); !msg.empty()) return InvalidArgument(msg);

    const auto sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) return InternalError(cudaGetErrorString(sync_status));

    const auto* cot_ptr = BufferData<T>(cotangent);
    const auto* sig1_ptr = BufferData<T>(sig1);
    const auto* sig2_ptr = BufferData<T>(sig2);
    auto* grad1_ptr = BufferData<T>(grad_sig1);
    auto* grad2_ptr = BufferData<T>(grad_sig2);

    int err_code = CudaFns<T>::sig_combine_backprop(
        cot_ptr, grad1_ptr, grad2_ptr, sig1_ptr, sig2_ptr,
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        true
    );

    if (err_code != 0) return NativeCallError("sig_combine_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error SigCombineCuda(
    cudaStream_t stream,
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer sig1,
    ffi::AnyBuffer sig2,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("sig1", sig1, "sig2", sig2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(sig1), [&]<typename T>() -> ffi::Error {
        return SigCombineCudaImpl<T>(stream, dimension, degree, sig1, sig2, out);
    });
}

ffi::Error SigCombineBackpropCuda(
    cudaStream_t stream,
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer cotangent,
    ffi::AnyBuffer sig1,
    ffi::AnyBuffer sig2,
    ffi::Result<ffi::AnyBuffer> grad_sig1,
    ffi::Result<ffi::AnyBuffer> grad_sig2
) {
    if (auto msg = ValidateSameFloatDtype("sig1", sig1, "sig2", sig2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(sig1), [&]<typename T>() -> ffi::Error {
        return SigCombineBackpropCudaImpl<T>(stream, dimension, degree, cotangent, sig1, sig2, grad_sig1, grad_sig2);
    });
}
#endif

// ---------------------------------------------------------------------------
// transform_path
// ---------------------------------------------------------------------------

template <typename T>
ffi::Error TransformPathCpuImpl(
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    ffi::AnyBuffer& path,
    ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);

    const auto* path_ptr = BufferData<T>(path);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CpuFns<T>::transform_path(
        path_ptr, out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension, spec.length,
        time_aug, lead_lag, static_cast<T>(end_time),
        static_cast<int>(n_jobs)
    );

    if (err_code != 0) return NativeCallError("transform_path", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error TransformPathBackpropCpuImpl(
    std::int64_t orig_dimension,
    std::int64_t orig_length,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    ffi::AnyBuffer& cotangent,
    ffi::Result<ffi::AnyBuffer>& out
) {
    // cotangent has transformed shape; out has original shape
    // C++ backprop takes original dimension and length
    const auto dims = BufferDims(cotangent);
    std::uint64_t batch_size = 1;
    for (size_t i = 0; i + 2 < dims.size(); ++i) {
        batch_size *= static_cast<std::uint64_t>(dims[i]);
    }

    const auto* cot_ptr = BufferData<T>(cotangent);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CpuFns<T>::transform_path_backprop(
        cot_ptr, out_ptr,
        batch_size,
        static_cast<std::uint64_t>(orig_dimension),
        static_cast<std::uint64_t>(orig_length),
        time_aug, lead_lag, static_cast<T>(end_time),
        static_cast<int>(n_jobs)
    );

    if (err_code != 0) return NativeCallError("transform_path_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error TransformPathCpu(
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    ffi::AnyBuffer path,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("path", path); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return TransformPathCpuImpl<T>(time_aug, lead_lag, end_time, n_jobs, path, out);
    });
}

ffi::Error TransformPathBackpropCpu(
    std::int64_t orig_dimension,
    std::int64_t orig_length,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    ffi::AnyBuffer cotangent,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(cotangent), [&]<typename T>() -> ffi::Error {
        return TransformPathBackpropCpuImpl<T>(orig_dimension, orig_length, time_aug, lead_lag, end_time, n_jobs, cotangent, out);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error TransformPathCudaImpl(
    cudaStream_t stream,
    bool time_aug,
    bool lead_lag,
    double end_time,
    ffi::AnyBuffer& path,
    ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);

    const auto sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) return InternalError(cudaGetErrorString(sync_status));

    const auto* path_ptr = BufferData<T>(path);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CudaFns<T>::transform_path(
        path_ptr, out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension, spec.length,
        time_aug, lead_lag, static_cast<T>(end_time)
    );

    if (err_code != 0) return NativeCallError("transform_path_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error TransformPathBackpropCudaImpl(
    cudaStream_t stream,
    std::int64_t orig_dimension,
    std::int64_t orig_length,
    bool time_aug,
    bool lead_lag,
    double end_time,
    ffi::AnyBuffer& cotangent,
    ffi::Result<ffi::AnyBuffer>& out
) {
    const auto dims = BufferDims(cotangent);
    std::uint64_t batch_size = 1;
    for (size_t i = 0; i + 2 < dims.size(); ++i) {
        batch_size *= static_cast<std::uint64_t>(dims[i]);
    }

    const auto sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) return InternalError(cudaGetErrorString(sync_status));

    const auto* cot_ptr = BufferData<T>(cotangent);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CudaFns<T>::transform_path_backprop(
        cot_ptr, out_ptr,
        batch_size,
        static_cast<std::uint64_t>(orig_dimension),
        static_cast<std::uint64_t>(orig_length),
        time_aug, lead_lag, static_cast<T>(end_time)
    );

    if (err_code != 0) return NativeCallError("transform_path_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error TransformPathCuda(
    cudaStream_t stream,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer path,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("path", path); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return TransformPathCudaImpl<T>(stream, time_aug, lead_lag, end_time, path, out);
    });
}

ffi::Error TransformPathBackpropCuda(
    cudaStream_t stream,
    std::int64_t orig_dimension,
    std::int64_t orig_length,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer cotangent,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(cotangent), [&]<typename T>() -> ffi::Error {
        return TransformPathBackpropCudaImpl<T>(stream, orig_dimension, orig_length, time_aug, lead_lag, end_time, cotangent, out);
    });
}
#endif

// ---------------------------------------------------------------------------
// sig_to_log_sig
// ---------------------------------------------------------------------------

// For sig_to_log_sig, we pass the augmented dimension directly from Python.
// CPU C++ takes time_aug/lead_lag but we pass false/false with the pre-augmented dim.
// CUDA C++ ignores n_jobs.

template <typename T>
ffi::Error SigToLogSigCpuImpl(
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t method,
    std::int64_t n_jobs,
    ffi::AnyBuffer& sig_buf,
    ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig_buf, spec); !msg.empty()) return InvalidArgument(msg);

    const auto* sig_ptr = BufferData<T>(sig_buf);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CpuFns<T>::sig_to_log_sig(
        sig_ptr, out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        false, false,
        static_cast<int>(method),
        true,
        static_cast<int>(n_jobs)
    );

    if (err_code != 0) return NativeCallError("sig_to_log_sig", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error SigToLogSigBackpropCpuImpl(
    std::int64_t dimension,
    std::int64_t degree,
    std::int64_t method,
    std::int64_t n_jobs,
    ffi::AnyBuffer& sig_buf,
    ffi::AnyBuffer& cotangent,
    ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig_buf, spec); !msg.empty()) return InvalidArgument(msg);

    const auto* sig_ptr = BufferData<T>(sig_buf);
    const auto* cot_ptr = BufferData<T>(cotangent);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CpuFns<T>::sig_to_log_sig_backprop(
        sig_ptr, out_ptr, cot_ptr,
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        false, false,
        static_cast<int>(method),
        true,
        static_cast<int>(n_jobs)
    );

    if (err_code != 0) return NativeCallError("sig_to_log_sig_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error SigToLogSigCpu(
    std::int64_t dimension, std::int64_t degree, std::int64_t method, std::int64_t n_jobs,
    ffi::AnyBuffer sig_buf, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("sig", sig_buf); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(sig_buf), [&]<typename T>() -> ffi::Error {
        return SigToLogSigCpuImpl<T>(dimension, degree, method, n_jobs, sig_buf, out);
    });
}

ffi::Error SigToLogSigBackpropCpu(
    std::int64_t dimension, std::int64_t degree, std::int64_t method, std::int64_t n_jobs,
    ffi::AnyBuffer sig_buf, ffi::AnyBuffer cotangent, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("sig", sig_buf, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(sig_buf), [&]<typename T>() -> ffi::Error {
        return SigToLogSigBackpropCpuImpl<T>(dimension, degree, method, n_jobs, sig_buf, cotangent, out);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error SigToLogSigCudaImpl(
    cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    ffi::AnyBuffer& sig_buf, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig_buf, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    int err_code = CudaFns<T>::sig_to_log_sig(BufferData<T>(sig_buf), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(method), true);
    if (err_code != 0) return NativeCallError("sig_to_log_sig_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error SigToLogSigBackpropCudaImpl(
    cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    ffi::AnyBuffer& sig_buf, ffi::AnyBuffer& cotangent, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(sig_buf, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    int err_code = CudaFns<T>::sig_to_log_sig_backprop(BufferData<T>(sig_buf), BufferData<T>(out), BufferData<T>(cotangent),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(method), true);
    if (err_code != 0) return NativeCallError("sig_to_log_sig_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error SigToLogSigCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer sig_buf, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("sig", sig_buf); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(sig_buf), [&]<typename T>() -> ffi::Error {
        return SigToLogSigCudaImpl<T>(stream, dimension, degree, method, sig_buf, out);
    });
}

ffi::Error SigToLogSigBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer sig_buf, ffi::AnyBuffer cotangent, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("sig", sig_buf, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(sig_buf), [&]<typename T>() -> ffi::Error {
        return SigToLogSigBackpropCudaImpl<T>(stream, dimension, degree, method, sig_buf, cotangent, out);
    });
}
#endif

// ---------------------------------------------------------------------------
// log_sig_combine  (mirrors sig_combine pattern)
// ---------------------------------------------------------------------------

template <typename T>
ffi::Error LogSigCombineCpuImpl(
    std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer& ls1, ffi::AnyBuffer& ls2, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(ls1, spec); !msg.empty()) return InvalidArgument(msg);

    int err_code = CpuFns<T>::log_sig_combine(BufferData<T>(ls1), BufferData<T>(ls2), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(n_jobs));
    if (err_code != 0) return NativeCallError("log_sig_combine", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error LogSigCombineBackpropCpuImpl(
    std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer& cotangent, ffi::AnyBuffer& ls1, ffi::AnyBuffer& ls2,
    ffi::Result<ffi::AnyBuffer>& grad1, ffi::Result<ffi::AnyBuffer>& grad2
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(ls1, spec); !msg.empty()) return InvalidArgument(msg);

    int err_code = CpuFns<T>::log_sig_combine_backprop(BufferData<T>(cotangent), BufferData<T>(grad1), BufferData<T>(grad2),
        BufferData<T>(ls1), BufferData<T>(ls2),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(n_jobs));
    if (err_code != 0) return NativeCallError("log_sig_combine_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error LogSigCombineCpu(std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer ls1, ffi::AnyBuffer ls2, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("ls1", ls1, "ls2", ls2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(ls1), [&]<typename T>() -> ffi::Error {
        return LogSigCombineCpuImpl<T>(dimension, degree, n_jobs, ls1, ls2, out);
    });
}

ffi::Error LogSigCombineBackpropCpu(std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer cotangent, ffi::AnyBuffer ls1, ffi::AnyBuffer ls2,
    ffi::Result<ffi::AnyBuffer> grad1, ffi::Result<ffi::AnyBuffer> grad2) {
    if (auto msg = ValidateSameFloatDtype("ls1", ls1, "ls2", ls2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(ls1), [&]<typename T>() -> ffi::Error {
        return LogSigCombineBackpropCpuImpl<T>(dimension, degree, n_jobs, cotangent, ls1, ls2, grad1, grad2);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error LogSigCombineCudaImpl(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    ffi::AnyBuffer& ls1, ffi::AnyBuffer& ls2, ffi::Result<ffi::AnyBuffer>& out) {
    SigSpec spec;
    if (auto msg = GetSigSpec(ls1, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    int err_code = CudaFns<T>::log_sig_combine(BufferData<T>(ls1), BufferData<T>(ls2), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree));
    if (err_code != 0) return NativeCallError("log_sig_combine_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error LogSigCombineBackpropCudaImpl(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    ffi::AnyBuffer& cotangent, ffi::AnyBuffer& ls1, ffi::AnyBuffer& ls2,
    ffi::Result<ffi::AnyBuffer>& grad1, ffi::Result<ffi::AnyBuffer>& grad2) {
    SigSpec spec;
    if (auto msg = GetSigSpec(ls1, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    int err_code = CudaFns<T>::log_sig_combine_backprop(BufferData<T>(cotangent), BufferData<T>(grad1), BufferData<T>(grad2),
        BufferData<T>(ls1), BufferData<T>(ls2),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree));
    if (err_code != 0) return NativeCallError("log_sig_combine_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error LogSigCombineCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer ls1, ffi::AnyBuffer ls2, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("ls1", ls1, "ls2", ls2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(ls1), [&]<typename T>() -> ffi::Error {
        return LogSigCombineCudaImpl<T>(stream, dimension, degree, ls1, ls2, out);
    });
}

ffi::Error LogSigCombineBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer cotangent, ffi::AnyBuffer ls1, ffi::AnyBuffer ls2,
    ffi::Result<ffi::AnyBuffer> grad1, ffi::Result<ffi::AnyBuffer> grad2) {
    if (auto msg = ValidateSameFloatDtype("ls1", ls1, "ls2", ls2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(ls1), [&]<typename T>() -> ffi::Error {
        return LogSigCombineBackpropCudaImpl<T>(stream, dimension, degree, cotangent, ls1, ls2, grad1, grad2);
    });
}
#endif

// ---------------------------------------------------------------------------
// sig_kernel PDE solver
// ---------------------------------------------------------------------------

struct GramSpec {
    bool is_batch = false;
    std::uint64_t batch_size = 1;
    std::uint64_t length1 = 0;
    std::uint64_t length2 = 0;
};

template <typename BufferT>
std::string GetGramSpec(BufferT& buf, GramSpec& spec) {
    const auto dims = BufferDims(buf);
    if (dims.size() < 2) {
        return "gram must have at least rank 2";
    }

    spec.length1 = static_cast<std::uint64_t>(dims[dims.size() - 2]);
    spec.length2 = static_cast<std::uint64_t>(dims[dims.size() - 1]);

    if (dims.size() == 2) {
        spec.is_batch = false;
        spec.batch_size = 1;
    } else {
        spec.is_batch = true;
        spec.batch_size = 1;
        for (size_t i = 0; i < dims.size() - 2; ++i) {
            spec.batch_size *= static_cast<std::uint64_t>(dims[i]);
        }
    }
    return {};
}

template <typename T>
ffi::Error SigKernelPdeCpuImpl(
    std::int64_t dimension, std::int64_t dyadic_order_1, std::int64_t dyadic_order_2,
    bool return_grid, std::int64_t n_jobs,
    ffi::AnyBuffer& gram, ffi::Result<ffi::AnyBuffer>& out
) {
    GramSpec spec;
    if (auto msg = GetGramSpec(gram, spec); !msg.empty()) return InvalidArgument(msg);

    // gram shape is (L1-1, L2-1) from double-diff; C++ wants original lengths
    auto length1 = spec.length1 + 1;
    auto length2 = spec.length2 + 1;

    int err_code = CpuFns<T>::sig_kernel(BufferData<T>(gram), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), length1, length2,
        static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2),
        return_grid, static_cast<int>(n_jobs));
    if (err_code != 0) return NativeCallError("sig_kernel", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error SigKernelPdeBackpropCpuImpl(
    std::int64_t dimension, std::int64_t dyadic_order_1, std::int64_t dyadic_order_2,
    bool return_grid, std::int64_t n_jobs,
    ffi::AnyBuffer& gram, ffi::AnyBuffer& derivs, ffi::AnyBuffer& k_grid,
    ffi::Result<ffi::AnyBuffer>& out
) {
    GramSpec spec;
    if (auto msg = GetGramSpec(gram, spec); !msg.empty()) return InvalidArgument(msg);

    auto length1 = spec.length1 + 1;
    auto length2 = spec.length2 + 1;

    int err_code = CpuFns<T>::sig_kernel_backprop(BufferData<T>(gram), BufferData<T>(out),
        BufferData<T>(derivs), BufferData<T>(k_grid),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), length1, length2,
        static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2),
        return_grid, static_cast<int>(n_jobs));
    if (err_code != 0) return NativeCallError("sig_kernel_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error SigKernelPdeCpu(
    std::int64_t dimension, std::int64_t dyadic_order_1, std::int64_t dyadic_order_2,
    bool return_grid, std::int64_t n_jobs,
    ffi::AnyBuffer gram, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("gram", gram); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(gram), [&]<typename T>() -> ffi::Error {
        return SigKernelPdeCpuImpl<T>(dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs, gram, out);
    });
}

ffi::Error SigKernelPdeBackpropCpu(
    std::int64_t dimension, std::int64_t dyadic_order_1, std::int64_t dyadic_order_2,
    bool return_grid, std::int64_t n_jobs,
    ffi::AnyBuffer gram, ffi::AnyBuffer derivs, ffi::AnyBuffer k_grid,
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("gram", gram); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(gram), [&]<typename T>() -> ffi::Error {
        return SigKernelPdeBackpropCpuImpl<T>(dimension, dyadic_order_1, dyadic_order_2, return_grid, n_jobs, gram, derivs, k_grid, out);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error SigKernelPdeCudaImpl(
    cudaStream_t stream, std::int64_t dimension,
    std::int64_t dyadic_order_1, std::int64_t dyadic_order_2, bool return_grid,
    ffi::AnyBuffer& gram, ffi::Result<ffi::AnyBuffer>& out
) {
    GramSpec spec;
    if (auto msg = GetGramSpec(gram, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    auto length1 = spec.length1 + 1;
    auto length2 = spec.length2 + 1;
    int err_code = CudaFns<T>::sig_kernel(BufferData<T>(gram), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), length1, length2,
        static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2), return_grid);
    if (err_code != 0) return NativeCallError("sig_kernel_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error SigKernelPdeBackpropCudaImpl(
    cudaStream_t stream, std::int64_t dimension,
    std::int64_t dyadic_order_1, std::int64_t dyadic_order_2, bool return_grid,
    ffi::AnyBuffer& gram, ffi::AnyBuffer& derivs, ffi::AnyBuffer& k_grid,
    ffi::Result<ffi::AnyBuffer>& out
) {
    GramSpec spec;
    if (auto msg = GetGramSpec(gram, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    auto length1 = spec.length1 + 1;
    auto length2 = spec.length2 + 1;
    int err_code = CudaFns<T>::sig_kernel_backprop(BufferData<T>(gram), BufferData<T>(out),
        BufferData<T>(derivs), BufferData<T>(k_grid),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), length1, length2,
        static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2), return_grid);
    if (err_code != 0) return NativeCallError("sig_kernel_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error SigKernelPdeCuda(cudaStream_t stream, std::int64_t dimension,
    std::int64_t dyadic_order_1, std::int64_t dyadic_order_2, bool return_grid,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer gram, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("gram", gram); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(gram), [&]<typename T>() -> ffi::Error {
        return SigKernelPdeCudaImpl<T>(stream, dimension, dyadic_order_1, dyadic_order_2, return_grid, gram, out);
    });
}

ffi::Error SigKernelPdeBackpropCuda(cudaStream_t stream, std::int64_t dimension,
    std::int64_t dyadic_order_1, std::int64_t dyadic_order_2, bool return_grid,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer gram, ffi::AnyBuffer derivs, ffi::AnyBuffer k_grid,
    ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("gram", gram); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(gram), [&]<typename T>() -> ffi::Error {
        return SigKernelPdeBackpropCudaImpl<T>(stream, dimension, dyadic_order_1, dyadic_order_2, return_grid, gram, derivs, k_grid, out);
    });
}
#endif

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigCpu,
    SigCpu,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("degree")
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<bool>("horner")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigBackpropCpu,
    SigBackpropCpu,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("degree")
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigCuda,
    SigCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("degree")
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<bool>("horner")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigBackpropCuda,
    SigBackpropCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("degree")
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);
#endif

// sig_combine

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigCombineCpu,
    SigCombineCpu,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigCombineBackpropCpu,
    SigCombineBackpropCpu,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigCombineCuda,
    SigCombineCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibSigCombineBackpropCuda,
    SigCombineBackpropCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);
#endif

// transform_path

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibTransformPathCpu,
    TransformPathCpu,
    ffi::Ffi::Bind()
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibTransformPathBackpropCpu,
    TransformPathBackpropCpu,
    ffi::Ffi::Bind()
        .Attr<std::int64_t>("orig_dimension")
        .Attr<std::int64_t>("orig_length")
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibTransformPathCuda,
    TransformPathCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PySigLibTransformPathBackpropCuda,
    TransformPathBackpropCuda,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("orig_dimension")
        .Attr<std::int64_t>("orig_length")
        .Attr<bool>("time_aug")
        .Attr<bool>("lead_lag")
        .Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);
#endif

// sig_to_log_sig

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigToLogSigCpu, SigToLogSigCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("method").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigToLogSigBackpropCpu, SigToLogSigBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("method").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigToLogSigCuda, SigToLogSigCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("method")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigToLogSigBackpropCuda, SigToLogSigBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("method")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif

// log_sig_combine

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigCombineCpu, LogSigCombineCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigCombineBackpropCpu, LogSigCombineBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigCombineCuda, LogSigCombineCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigCombineBackpropCuda, LogSigCombineBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif

// sig_kernel PDE solver

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigKernelPdeCpu, SigKernelPdeCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("dyadic_order_1").Attr<std::int64_t>("dyadic_order_2")
        .Attr<bool>("return_grid").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigKernelPdeBackpropCpu, SigKernelPdeBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("dyadic_order_1").Attr<std::int64_t>("dyadic_order_2")
        .Attr<bool>("return_grid").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigKernelPdeCuda, SigKernelPdeCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("dyadic_order_1").Attr<std::int64_t>("dyadic_order_2")
        .Attr<bool>("return_grid").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigKernelPdeBackpropCuda, SigKernelPdeBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("dyadic_order_1").Attr<std::int64_t>("dyadic_order_2")
        .Attr<bool>("return_grid").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif

namespace {

// ---------------------------------------------------------------------------
// branched_sig
// ---------------------------------------------------------------------------

template <typename T>
ffi::Error BranchedSigCpuImpl(
    std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer& path, ffi::AnyBuffer& correction, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);

    const auto* path_ptr = BufferData<T>(path);
    const auto* correction_ptr = BufferData<T>(correction);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CpuFns<T>::bsig(
        path_ptr, out_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension, spec.length,
        static_cast<std::uint64_t>(max_nodes), static_cast<int>(n_jobs),
        time_aug, lead_lag, static_cast<T>(end_time), planar, true,
        correction_ptr, correction_len
    );
    if (err_code != 0) return NativeCallError("branched_sig", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error BranchedSigBackpropCpuImpl(
    std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer& path, ffi::AnyBuffer& bsig, ffi::AnyBuffer& cotangent,
    ffi::AnyBuffer& correction, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);

    const auto* path_ptr = BufferData<T>(path);
    const auto* bsig_ptr = BufferData<T>(bsig);
    const auto* cot_ptr = BufferData<T>(cotangent);
    const auto* correction_ptr = BufferData<T>(correction);
    auto* out_ptr = BufferData<T>(out);

    int err_code = CpuFns<T>::bsig_backprop(
        path_ptr, out_ptr, cot_ptr, bsig_ptr,
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension, spec.length,
        static_cast<std::uint64_t>(max_nodes), static_cast<int>(n_jobs),
        time_aug, lead_lag, static_cast<T>(end_time), planar, true,
        correction_ptr, correction_len
    );
    if (err_code != 0) return NativeCallError("branched_sig_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error BranchedSigCpu(
    std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer path, ffi::AnyBuffer correction, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return BranchedSigCpuImpl<T>(max_nodes, time_aug, lead_lag, end_time, n_jobs, planar, path, correction, out);
    });
}

ffi::Error BranchedSigBackpropCpu(
    std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer path, ffi::AnyBuffer bsig, ffi::AnyBuffer cotangent,
    ffi::AnyBuffer correction, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "bsig", bsig); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return BranchedSigBackpropCpuImpl<T>(max_nodes, time_aug, lead_lag, end_time, n_jobs, planar, path, bsig, cotangent, correction, out);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error BranchedSigCudaImpl(
    cudaStream_t stream, std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t /*n_jobs*/, bool planar,
    ffi::AnyBuffer& path, ffi::AnyBuffer& correction, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    int err_code = CudaFns<T>::bsig(BufferData<T>(path), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension, spec.length,
        static_cast<std::uint64_t>(max_nodes), time_aug, lead_lag, static_cast<T>(end_time), planar, true,
        BufferData<T>(correction), correction_len);
    if (err_code != 0) return NativeCallError("branched_sig_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error BranchedSigBackpropCudaImpl(
    cudaStream_t stream, std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t /*n_jobs*/, bool planar,
    ffi::AnyBuffer& path, ffi::AnyBuffer& bsig, ffi::AnyBuffer& cotangent,
    ffi::AnyBuffer& correction, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    std::uint64_t correction_len = 0;
    if (auto msg = GetCorrectionLen(correction, correction_len); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    int err_code = CudaFns<T>::bsig_backprop(BufferData<T>(path), BufferData<T>(out), BufferData<T>(cotangent), BufferData<T>(bsig),
        spec.is_batch ? spec.batch_size : 1,
        spec.dimension, spec.length,
        static_cast<std::uint64_t>(max_nodes), time_aug, lead_lag, static_cast<T>(end_time), planar, true,
        BufferData<T>(correction), correction_len);
    if (err_code != 0) return NativeCallError("branched_sig_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error BranchedSigCuda(
    cudaStream_t stream, std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer path, ffi::AnyBuffer correction, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return BranchedSigCudaImpl<T>(stream, max_nodes, time_aug, lead_lag, end_time, n_jobs, planar, path, correction, out);
    });
}

ffi::Error BranchedSigBackpropCuda(
    cudaStream_t stream, std::int64_t max_nodes, bool time_aug, bool lead_lag, double end_time, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer path, ffi::AnyBuffer bsig, ffi::AnyBuffer cotangent,
    ffi::AnyBuffer correction, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "bsig", bsig); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "correction", correction); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return BranchedSigBackpropCudaImpl<T>(stream, max_nodes, time_aug, lead_lag, end_time, n_jobs, planar, path, bsig, cotangent, correction, out);
    });
}
#endif

// ---------------------------------------------------------------------------
// branched_sig_combine
// ---------------------------------------------------------------------------

template <typename T>
ffi::Error BranchedSigCombineCpuImpl(
    std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer& bsig1, ffi::AnyBuffer& bsig2, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig1, spec); !msg.empty()) return InvalidArgument(msg);

    int err_code = CpuFns<T>::bsig_combine(BufferData<T>(bsig1), BufferData<T>(bsig2), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes), static_cast<int>(n_jobs), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_combine", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error BranchedSigCombineBackpropCpuImpl(
    std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer& cotangent, ffi::AnyBuffer& bsig1, ffi::AnyBuffer& bsig2,
    ffi::Result<ffi::AnyBuffer>& grad1, ffi::Result<ffi::AnyBuffer>& grad2
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig1, spec); !msg.empty()) return InvalidArgument(msg);

    int err_code = CpuFns<T>::bsig_combine_backprop(BufferData<T>(bsig1), BufferData<T>(bsig2), BufferData<T>(cotangent),
        BufferData<T>(grad1), BufferData<T>(grad2),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes), static_cast<int>(n_jobs), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_combine_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error BranchedSigCombineCpu(std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer bsig1, ffi::AnyBuffer bsig2, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("bsig1", bsig1, "bsig2", bsig2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig1), [&]<typename T>() -> ffi::Error {
        return BranchedSigCombineCpuImpl<T>(dimension, max_nodes, n_jobs, planar, bsig1, bsig2, out);
    });
}

ffi::Error BranchedSigCombineBackpropCpu(std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer cotangent, ffi::AnyBuffer bsig1, ffi::AnyBuffer bsig2,
    ffi::Result<ffi::AnyBuffer> grad1, ffi::Result<ffi::AnyBuffer> grad2) {
    if (auto msg = ValidateSameFloatDtype("bsig1", bsig1, "bsig2", bsig2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig1), [&]<typename T>() -> ffi::Error {
        return BranchedSigCombineBackpropCpuImpl<T>(dimension, max_nodes, n_jobs, planar, cotangent, bsig1, bsig2, grad1, grad2);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error BranchedSigCombineCudaImpl(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t /*n_jobs*/, bool planar,
    ffi::AnyBuffer& bsig1, ffi::AnyBuffer& bsig2, ffi::Result<ffi::AnyBuffer>& out) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig1, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    int err_code = CudaFns<T>::bsig_combine(BufferData<T>(bsig1), BufferData<T>(bsig2), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_combine_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error BranchedSigCombineBackpropCudaImpl(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t /*n_jobs*/, bool planar,
    ffi::AnyBuffer& cotangent, ffi::AnyBuffer& bsig1, ffi::AnyBuffer& bsig2,
    ffi::Result<ffi::AnyBuffer>& grad1, ffi::Result<ffi::AnyBuffer>& grad2) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig1, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    int err_code = CudaFns<T>::bsig_combine_backprop(BufferData<T>(bsig1), BufferData<T>(bsig2), BufferData<T>(cotangent),
        BufferData<T>(grad1), BufferData<T>(grad2),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_combine_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error BranchedSigCombineCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer bsig1, ffi::AnyBuffer bsig2, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("bsig1", bsig1, "bsig2", bsig2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig1), [&]<typename T>() -> ffi::Error {
        return BranchedSigCombineCudaImpl<T>(stream, dimension, max_nodes, n_jobs, planar, bsig1, bsig2, out);
    });
}

ffi::Error BranchedSigCombineBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer cotangent, ffi::AnyBuffer bsig1, ffi::AnyBuffer bsig2,
    ffi::Result<ffi::AnyBuffer> grad1, ffi::Result<ffi::AnyBuffer> grad2) {
    if (auto msg = ValidateSameFloatDtype("bsig1", bsig1, "bsig2", bsig2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig1), [&]<typename T>() -> ffi::Error {
        return BranchedSigCombineBackpropCudaImpl<T>(stream, dimension, max_nodes, n_jobs, planar, cotangent, bsig1, bsig2, grad1, grad2);
    });
}
#endif

// ---------------------------------------------------------------------------
// branched_sig_to_log_sig
// ---------------------------------------------------------------------------

template <typename T>
ffi::Error BranchedSigToLogSigCpuImpl(
    std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer& bsig, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig, spec); !msg.empty()) return InvalidArgument(msg);

    int err_code = CpuFns<T>::bsig_to_log_sig(BufferData<T>(bsig), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes),
        static_cast<int>(n_jobs), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_to_log_sig", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error BranchedSigToLogSigBackpropCpuImpl(
    std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer& bsig, ffi::AnyBuffer& cotangent, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig, spec); !msg.empty()) return InvalidArgument(msg);

    int err_code = CpuFns<T>::bsig_to_log_sig_backprop(BufferData<T>(bsig), BufferData<T>(cotangent),
        BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes),
        static_cast<int>(n_jobs), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_to_log_sig_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error BranchedSigToLogSigCpu(std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer bsig, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("bsig", bsig); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig), [&]<typename T>() -> ffi::Error {
        return BranchedSigToLogSigCpuImpl<T>(dimension, max_nodes, n_jobs, planar, bsig, out);
    });
}

ffi::Error BranchedSigToLogSigBackpropCpu(std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer bsig, ffi::AnyBuffer cotangent, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("bsig", bsig, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig), [&]<typename T>() -> ffi::Error {
        return BranchedSigToLogSigBackpropCpuImpl<T>(dimension, max_nodes, n_jobs, planar, bsig, cotangent, out);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error BranchedSigToLogSigCudaImpl(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t /*n_jobs*/, bool planar,
    ffi::AnyBuffer& bsig, ffi::Result<ffi::AnyBuffer>& out) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    int err_code = CudaFns<T>::bsig_to_log_sig(BufferData<T>(bsig), BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_to_log_sig_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error BranchedSigToLogSigBackpropCudaImpl(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t /*n_jobs*/, bool planar,
    ffi::AnyBuffer& bsig, ffi::AnyBuffer& cotangent, ffi::Result<ffi::AnyBuffer>& out) {
    SigSpec spec;
    if (auto msg = GetSigSpec(bsig, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));
    int err_code = CudaFns<T>::bsig_to_log_sig_backprop(BufferData<T>(bsig), BufferData<T>(cotangent),
        BufferData<T>(out),
        spec.is_batch ? spec.batch_size : 1,
        static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(max_nodes), planar, true);
    if (err_code != 0) return NativeCallError("branched_sig_to_log_sig_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error BranchedSigToLogSigCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer bsig, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("bsig", bsig); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig), [&]<typename T>() -> ffi::Error {
        return BranchedSigToLogSigCudaImpl<T>(stream, dimension, max_nodes, n_jobs, planar, bsig, out);
    });
}

ffi::Error BranchedSigToLogSigBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t max_nodes, std::int64_t n_jobs, bool planar,
    ffi::AnyBuffer bsig, ffi::AnyBuffer cotangent, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("bsig", bsig, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(bsig), [&]<typename T>() -> ffi::Error {
        return BranchedSigToLogSigBackpropCudaImpl<T>(stream, dimension, max_nodes, n_jobs, planar, bsig, cotangent, out);
    });
}
#endif

// ---------------------------------------------------------------------------
// log_sig_from_path (method=3)
// ---------------------------------------------------------------------------

template <typename T>
ffi::Error LogSigFromPathCpuImpl(
    std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer& path, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CpuFns<T>::log_sig_from_path(
        BufferData<T>(path), BufferData<T>(out),
        batch, spec.length,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        static_cast<int>(n_jobs)
    );
    if (err_code != 0) return NativeCallError("log_sig_from_path", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error LogSigFromPathBackpropCpuImpl(
    std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer& cotangent, ffi::AnyBuffer& path, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CpuFns<T>::log_sig_from_path_backprop(
        BufferData<T>(cotangent), BufferData<T>(out), BufferData<T>(path),
        batch, spec.length,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        static_cast<int>(n_jobs)
    );
    if (err_code != 0) return NativeCallError("log_sig_from_path_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error LogSigFromPathCpu(
    std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer path, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("path", path); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return LogSigFromPathCpuImpl<T>(dimension, degree, n_jobs, path, out);
    });
}

ffi::Error LogSigFromPathBackpropCpu(
    std::int64_t dimension, std::int64_t degree, std::int64_t n_jobs,
    ffi::AnyBuffer cotangent, ffi::AnyBuffer path, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("path", path); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return LogSigFromPathBackpropCpuImpl<T>(dimension, degree, n_jobs, cotangent, path, out);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error LogSigFromPathCudaImpl(
    cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    ffi::AnyBuffer& path, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CudaFns<T>::log_sig_from_path(
        BufferData<T>(path), BufferData<T>(out),
        batch, spec.length,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree)
    );
    if (err_code != 0) return NativeCallError("log_sig_from_path_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error LogSigFromPathBackpropCudaImpl(
    cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    ffi::AnyBuffer& cotangent, ffi::AnyBuffer& path, ffi::Result<ffi::AnyBuffer>& out
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CudaFns<T>::log_sig_from_path_backprop(
        BufferData<T>(cotangent), BufferData<T>(out), BufferData<T>(path),
        batch, spec.length,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree)
    );
    if (err_code != 0) return NativeCallError("log_sig_from_path_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error LogSigFromPathCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer path, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("path", path); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return LogSigFromPathCudaImpl<T>(stream, dimension, degree, path, out);
    });
}

ffi::Error LogSigFromPathBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer cotangent, ffi::AnyBuffer path, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("path", path); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return LogSigFromPathBackpropCudaImpl<T>(stream, dimension, degree, cotangent, path, out);
    });
}
#endif

}  // namespace

// branched_sig

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigCpu, BranchedSigCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("max_nodes")
        .Attr<bool>("time_aug").Attr<bool>("lead_lag").Attr<double>("end_time").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigBackpropCpu, BranchedSigBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("max_nodes")
        .Attr<bool>("time_aug").Attr<bool>("lead_lag").Attr<double>("end_time").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigCuda, BranchedSigCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("max_nodes").Attr<bool>("time_aug").Attr<bool>("lead_lag").Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigBackpropCuda, BranchedSigBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("max_nodes").Attr<bool>("time_aug").Attr<bool>("lead_lag").Attr<double>("end_time")
        .Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif

// branched_sig_combine

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigCombineCpu, BranchedSigCombineCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigCombineBackpropCpu, BranchedSigCombineBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigCombineCuda, BranchedSigCombineCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigCombineBackpropCuda, BranchedSigCombineBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif

// branched_sig_to_log_sig

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigToLogSigCpu, BranchedSigToLogSigCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigToLogSigBackpropCpu, BranchedSigToLogSigBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigToLogSigCuda, BranchedSigToLogSigCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibBranchedSigToLogSigBackpropCuda, BranchedSigToLogSigBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("max_nodes").Attr<std::int64_t>("n_jobs").Attr<bool>("planar")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif

// log_sig_from_path (method=3)

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigFromPathCpu, LogSigFromPathCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigFromPathBackpropCpu, LogSigFromPathBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigFromPathCuda, LogSigFromPathCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigFromPathBackpropCuda, LogSigFromPathBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif

// logsig_to_sig (tensor exponential)

namespace {

template <typename T>
ffi::Error LogSigToSigCpuImpl(
    std::int64_t dimension, std::int64_t degree, std::int64_t method, std::int64_t n_jobs,
    ffi::AnyBuffer& log_sig, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(log_sig, spec); !msg.empty()) return InvalidArgument(msg);

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CpuFns<T>::logsig_to_sig(
        BufferData<T>(log_sig), BufferData<T>(out),
        batch,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        false, false,
        static_cast<int>(method),
        true,
        static_cast<int>(n_jobs)
    );
    if (err_code != 0) return NativeCallError("logsig_to_sig", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error LogSigToSigBackpropCpuImpl(
    std::int64_t dimension, std::int64_t degree, std::int64_t method, std::int64_t n_jobs,
    ffi::AnyBuffer& log_sig, ffi::AnyBuffer& cotangent, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(log_sig, spec); !msg.empty()) return InvalidArgument(msg);

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CpuFns<T>::logsig_to_sig_backprop(
        BufferData<T>(log_sig), BufferData<T>(out), BufferData<T>(cotangent),
        batch,
        static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree),
        false, false,
        static_cast<int>(method),
        true,
        static_cast<int>(n_jobs)
    );
    if (err_code != 0) return NativeCallError("logsig_to_sig_backprop", err_code);
    return ffi::Error::Success();
}

ffi::Error LogSigToSigCpu(
    std::int64_t dimension, std::int64_t degree, std::int64_t method, std::int64_t n_jobs,
    ffi::AnyBuffer log_sig, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("log_sig", log_sig); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(log_sig), [&]<typename T>() -> ffi::Error {
        return LogSigToSigCpuImpl<T>(dimension, degree, method, n_jobs, log_sig, out);
    });
}

ffi::Error LogSigToSigBackpropCpu(
    std::int64_t dimension, std::int64_t degree, std::int64_t method, std::int64_t n_jobs,
    ffi::AnyBuffer log_sig, ffi::AnyBuffer cotangent, ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateFloatBuffer("log_sig", log_sig); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(log_sig), [&]<typename T>() -> ffi::Error {
        return LogSigToSigBackpropCpuImpl<T>(dimension, degree, method, n_jobs, log_sig, cotangent, out);
    });
}

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
ffi::Error LogSigToSigCudaImpl(
    cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    ffi::AnyBuffer& log_sig, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(log_sig, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CudaFns<T>::logsig_to_sig(
        BufferData<T>(log_sig), BufferData<T>(out),
        batch, static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree), static_cast<int>(method), true);
    if (err_code != 0) return NativeCallError("logsig_to_sig_cuda", err_code);
    return ffi::Error::Success();
}

template <typename T>
ffi::Error LogSigToSigBackpropCudaImpl(
    cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    ffi::AnyBuffer& log_sig, ffi::AnyBuffer& cotangent, ffi::Result<ffi::AnyBuffer>& out
) {
    SigSpec spec;
    if (auto msg = GetSigSpec(log_sig, spec); !msg.empty()) return InvalidArgument(msg);
    auto sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) return InternalError(cudaGetErrorString(sync));

    std::uint64_t batch = spec.is_batch ? spec.batch_size : 1;
    int err_code = CudaFns<T>::logsig_to_sig_backprop(
        BufferData<T>(log_sig), BufferData<T>(out), BufferData<T>(cotangent),
        batch, static_cast<std::uint64_t>(dimension),
        static_cast<std::uint64_t>(degree), static_cast<int>(method), true);
    if (err_code != 0) return NativeCallError("logsig_to_sig_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error LogSigToSigCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer log_sig, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("log_sig", log_sig); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(log_sig), [&]<typename T>() -> ffi::Error {
        return LogSigToSigCudaImpl<T>(stream, dimension, degree, method, log_sig, out);
    });
}

ffi::Error LogSigToSigBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    std::int64_t /*n_jobs*/,
    ffi::AnyBuffer log_sig, ffi::AnyBuffer cotangent, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("log_sig", log_sig); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(log_sig), [&]<typename T>() -> ffi::Error {
        return LogSigToSigBackpropCudaImpl<T>(stream, dimension, degree, method, log_sig, cotangent, out);
    });
}
#endif

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigToSigCpu, LogSigToSigCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("method").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigToSigBackpropCpu, LogSigToSigBackpropCpu,
    ffi::Ffi::Bind().Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree")
        .Attr<std::int64_t>("method").Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

#ifdef PYSIGLIB_JAX_WITH_CUDA
XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigToSigCuda, LogSigToSigCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("method")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigToSigBackpropCuda, LogSigToSigBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("method")
        .Attr<std::int64_t>("n_jobs")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif
