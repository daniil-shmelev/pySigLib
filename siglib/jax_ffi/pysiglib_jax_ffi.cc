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
    if (dims.size() == 2) {
        spec.is_batch = false;
        spec.batch_size = 1;
        spec.length = static_cast<std::uint64_t>(dims[0]);
        spec.dimension = static_cast<std::uint64_t>(dims[1]);
        return {};
    }

    if (dims.size() == 3) {
        spec.is_batch = true;
        spec.batch_size = static_cast<std::uint64_t>(dims[0]);
        spec.length = static_cast<std::uint64_t>(dims[1]);
        spec.dimension = static_cast<std::uint64_t>(dims[2]);
        return {};
    }

    std::ostringstream oss;
    oss << "path must have rank 2 or 3, got rank " << dims.size();
    return oss.str();
}

template <typename BufferT>
std::string CheckSigOutputShape(BufferT& out, const PathSpec& spec, std::uint64_t sig_len) {
    const auto dims = BufferDims(out);

    if (!spec.is_batch && dims.size() == 1 && static_cast<std::uint64_t>(dims[0]) == sig_len) {
        return {};
    }

    if (spec.is_batch && dims.size() == 2 &&
        static_cast<std::uint64_t>(dims[0]) == spec.batch_size &&
        static_cast<std::uint64_t>(dims[1]) == sig_len) {
        return {};
    }

    return "unexpected signature output shape";
}

template <typename BufferT>
std::string CheckGradOutputShape(BufferT& out, const PathSpec& spec) {
    const auto dims = BufferDims(out);

    if (!spec.is_batch && dims.size() == 2 &&
        static_cast<std::uint64_t>(dims[0]) == spec.length &&
        static_cast<std::uint64_t>(dims[1]) == spec.dimension) {
        return {};
    }

    if (spec.is_batch && dims.size() == 3 &&
        static_cast<std::uint64_t>(dims[0]) == spec.batch_size &&
        static_cast<std::uint64_t>(dims[1]) == spec.length &&
        static_cast<std::uint64_t>(dims[2]) == spec.dimension) {
        return {};
    }

    return "unexpected signature backprop output shape";
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

template <typename T>
using CpuSigFn = int (*)(const T*, T*, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool) noexcept;

template <typename T>
using CpuBatchSigFn = int (*)(const T*, T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool, int) noexcept;

template <typename T>
using CpuSigBackpropFn = int (*)(const T*, T*, const T*, const T*, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T) noexcept;

template <typename T>
using CpuBatchSigBackpropFn = int (*)(const T*, T*, const T*, const T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, int) noexcept;

template <typename T>
struct CpuFns;

template <>
struct CpuFns<float> {
    static constexpr auto sig = signature_f;
    static constexpr auto batch_sig = batch_signature_f;
    static constexpr auto backprop = sig_backprop_f;
    static constexpr auto batch_backprop = batch_sig_backprop_f;
    static constexpr const char* sig_name = "signature_f";
    static constexpr const char* backprop_name = "sig_backprop_f";

    static constexpr auto sig_combine = sig_combine_f;
    static constexpr auto batch_sig_combine = batch_sig_combine_f;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_f;
    static constexpr auto batch_sig_combine_backprop = batch_sig_combine_backprop_f;

    static constexpr auto transform_path = transform_path_f;
    static constexpr auto batch_transform_path = batch_transform_path_f;
    static constexpr auto transform_path_backprop = transform_path_backprop_f;
    static constexpr auto batch_transform_path_backprop = batch_transform_path_backprop_f;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_f;
    static constexpr auto batch_sig_to_log_sig = batch_sig_to_log_sig_f;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_f;
    static constexpr auto batch_sig_to_log_sig_backprop = batch_sig_to_log_sig_backprop_f;

    static constexpr auto log_sig_combine = log_sig_combine_f;
    static constexpr auto batch_log_sig_combine = batch_log_sig_combine_f;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_f;
    static constexpr auto batch_log_sig_combine_backprop = batch_log_sig_combine_backprop_f;

    static constexpr auto sig_kernel = sig_kernel_f;
    static constexpr auto batch_sig_kernel = batch_sig_kernel_f;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_f;
    static constexpr auto batch_sig_kernel_backprop = batch_sig_kernel_backprop_f;
};

template <>
struct CpuFns<double> {
    static constexpr auto sig = signature_d;
    static constexpr auto batch_sig = batch_signature_d;
    static constexpr auto backprop = sig_backprop_d;
    static constexpr auto batch_backprop = batch_sig_backprop_d;
    static constexpr const char* sig_name = "signature_d";
    static constexpr const char* backprop_name = "sig_backprop_d";

    static constexpr auto sig_combine = sig_combine_d;
    static constexpr auto batch_sig_combine = batch_sig_combine_d;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_d;
    static constexpr auto batch_sig_combine_backprop = batch_sig_combine_backprop_d;

    static constexpr auto transform_path = transform_path_d;
    static constexpr auto batch_transform_path = batch_transform_path_d;
    static constexpr auto transform_path_backprop = transform_path_backprop_d;
    static constexpr auto batch_transform_path_backprop = batch_transform_path_backprop_d;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_d;
    static constexpr auto batch_sig_to_log_sig = batch_sig_to_log_sig_d;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_d;
    static constexpr auto batch_sig_to_log_sig_backprop = batch_sig_to_log_sig_backprop_d;

    static constexpr auto log_sig_combine = log_sig_combine_d;
    static constexpr auto batch_log_sig_combine = batch_log_sig_combine_d;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_d;
    static constexpr auto batch_log_sig_combine_backprop = batch_log_sig_combine_backprop_d;

    static constexpr auto sig_kernel = sig_kernel_d;
    static constexpr auto batch_sig_kernel = batch_sig_kernel_d;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_d;
    static constexpr auto batch_sig_kernel_backprop = batch_sig_kernel_backprop_d;
};

#ifdef PYSIGLIB_JAX_WITH_CUDA
template <typename T>
using CudaSigFn = int (*)(const T*, T*, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool) noexcept;

template <typename T>
using CudaBatchSigFn = int (*)(const T*, T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T, bool) noexcept;

template <typename T>
using CudaSigBackpropFn = int (*)(const T*, T*, const T*, const T*, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T) noexcept;

template <typename T>
using CudaBatchSigBackpropFn = int (*)(const T*, T*, const T*, const T*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, bool, bool, T) noexcept;

template <typename T>
struct CudaFns;

template <>
struct CudaFns<float> {
    static constexpr auto sig = signature_cuda_f;
    static constexpr auto batch_sig = batch_signature_cuda_f;
    static constexpr auto backprop = sig_backprop_cuda_f;
    static constexpr auto batch_backprop = batch_sig_backprop_cuda_f;
    static constexpr const char* sig_name = "signature_cuda_f";
    static constexpr const char* backprop_name = "sig_backprop_cuda_f";

    static constexpr auto sig_combine = sig_combine_cuda_f;
    static constexpr auto batch_sig_combine = batch_sig_combine_cuda_f;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_cuda_f;
    static constexpr auto batch_sig_combine_backprop = batch_sig_combine_backprop_cuda_f;

    static constexpr auto transform_path = transform_path_cuda_f;
    static constexpr auto batch_transform_path = batch_transform_path_cuda_f;
    static constexpr auto transform_path_backprop = transform_path_backprop_cuda_f;
    static constexpr auto batch_transform_path_backprop = batch_transform_path_backprop_cuda_f;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_cuda_f;
    static constexpr auto batch_sig_to_log_sig = batch_sig_to_log_sig_cuda_f;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_cuda_f;
    static constexpr auto batch_sig_to_log_sig_backprop = batch_sig_to_log_sig_backprop_cuda_f;

    static constexpr auto log_sig_combine = log_sig_combine_cuda_f;
    static constexpr auto batch_log_sig_combine = batch_log_sig_combine_cuda_f;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_cuda_f;
    static constexpr auto batch_log_sig_combine_backprop = batch_log_sig_combine_backprop_cuda_f;

    static constexpr auto sig_kernel = sig_kernel_cuda_f;
    static constexpr auto batch_sig_kernel = batch_sig_kernel_cuda_f;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_cuda_f;
    static constexpr auto batch_sig_kernel_backprop = batch_sig_kernel_backprop_cuda_f;
};

template <>
struct CudaFns<double> {
    static constexpr auto sig = signature_cuda_d;
    static constexpr auto batch_sig = batch_signature_cuda_d;
    static constexpr auto backprop = sig_backprop_cuda_d;
    static constexpr auto batch_backprop = batch_sig_backprop_cuda_d;
    static constexpr const char* sig_name = "signature_cuda_d";
    static constexpr const char* backprop_name = "sig_backprop_cuda_d";

    static constexpr auto sig_combine = sig_combine_cuda_d;
    static constexpr auto batch_sig_combine = batch_sig_combine_cuda_d;
    static constexpr auto sig_combine_backprop = sig_combine_backprop_cuda_d;
    static constexpr auto batch_sig_combine_backprop = batch_sig_combine_backprop_cuda_d;

    static constexpr auto transform_path = transform_path_cuda_d;
    static constexpr auto batch_transform_path = batch_transform_path_cuda_d;
    static constexpr auto transform_path_backprop = transform_path_backprop_cuda_d;
    static constexpr auto batch_transform_path_backprop = batch_transform_path_backprop_cuda_d;

    static constexpr auto sig_to_log_sig = sig_to_log_sig_cuda_d;
    static constexpr auto batch_sig_to_log_sig = batch_sig_to_log_sig_cuda_d;
    static constexpr auto sig_to_log_sig_backprop = sig_to_log_sig_backprop_cuda_d;
    static constexpr auto batch_sig_to_log_sig_backprop = batch_sig_to_log_sig_backprop_cuda_d;

    static constexpr auto log_sig_combine = log_sig_combine_cuda_d;
    static constexpr auto batch_log_sig_combine = batch_log_sig_combine_cuda_d;
    static constexpr auto log_sig_combine_backprop = log_sig_combine_backprop_cuda_d;
    static constexpr auto batch_log_sig_combine_backprop = batch_log_sig_combine_backprop_cuda_d;

    static constexpr auto sig_kernel = sig_kernel_cuda_d;
    static constexpr auto batch_sig_kernel = batch_sig_kernel_cuda_d;
    static constexpr auto sig_kernel_backprop = sig_kernel_backprop_cuda_d;
    static constexpr auto batch_sig_kernel_backprop = batch_sig_kernel_backprop_cuda_d;
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
    CpuBatchSigFn<T> batch_sig_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    bool horner,
    std::int64_t n_jobs,
    PathBuffer& path,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);

    const auto sig_len = sig_length(
        AugmentedDimension(spec.dimension, time_aug, lead_lag),
        static_cast<std::uint64_t>(degree)
    );
    if (sig_len == 0) {
        return InvalidArgument("signature length overflow");
    }
    if (auto msg = CheckSigOutputShape(out, spec, sig_len); !msg.empty()) return InvalidArgument(msg);

    const auto* path_ptr = BufferData<T>(path);
    auto* out_ptr = BufferData<T>(out);

    int err_code = 0;
    if (spec.is_batch) {
        err_code = batch_sig_fn(
            path_ptr,
            out_ptr,
            spec.batch_size,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time),
            horner,
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = sig_fn(
            path_ptr,
            out_ptr,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time),
            horner
        );
    }

    if (err_code != 0) {
        return NativeCallError(fn_name, err_code);
    }

    return ffi::Error::Success();
}

template <typename T, typename PathBuffer, typename SigBuffer, typename CotangentBuffer, typename OutBuffer>
ffi::Error SigBackpropCpuImpl(
    CpuSigBackpropFn<T> sig_backprop_fn,
    CpuBatchSigBackpropFn<T> batch_sig_backprop_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    PathBuffer& path,
    SigBuffer& sig,
    CotangentBuffer& cotangent,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);

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
    auto* out_ptr = BufferData<T>(out);

    int err_code = 0;
    if (spec.is_batch) {
        err_code = batch_sig_backprop_fn(
            path_ptr,
            out_ptr,
            cotangent_ptr,
            sig_ptr,
            spec.batch_size,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time),
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = sig_backprop_fn(
            path_ptr,
            out_ptr,
            cotangent_ptr,
            sig_ptr,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time)
        );
    }

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
    CudaBatchSigFn<T> batch_sig_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    bool horner,
    std::int64_t n_jobs,
    PathBuffer& path,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);

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
    auto* out_ptr = BufferData<T>(out);

    int err_code = 0;
    if (spec.is_batch) {
        err_code = batch_sig_fn(
            path_ptr,
            out_ptr,
            spec.batch_size,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time),
            horner
        );
    } else {
        err_code = sig_fn(
            path_ptr,
            out_ptr,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time),
            horner
        );
    }

    if (err_code != 0) {
        return NativeCallError(fn_name, err_code);
    }

    return ffi::Error::Success();
}

template <typename T, typename PathBuffer, typename SigBuffer, typename CotangentBuffer, typename OutBuffer>
ffi::Error SigBackpropCudaImpl(
    cudaStream_t stream,
    CudaSigBackpropFn<T> sig_backprop_fn,
    CudaBatchSigBackpropFn<T> batch_sig_backprop_fn,
    std::int64_t degree,
    bool time_aug,
    bool lead_lag,
    double end_time,
    std::int64_t n_jobs,
    PathBuffer& path,
    SigBuffer& sig,
    CotangentBuffer& cotangent,
    OutBuffer& out,
    const char* fn_name
) {
    PathSpec spec;
    if (auto msg = GetPathSpec(path, spec); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateArgs(degree, n_jobs, spec); !msg.empty()) return InvalidArgument(msg);

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
    auto* out_ptr = BufferData<T>(out);

    int err_code = 0;
    if (spec.is_batch) {
        err_code = batch_sig_backprop_fn(
            path_ptr,
            out_ptr,
            cotangent_ptr,
            sig_ptr,
            spec.batch_size,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time)
        );
    } else {
        err_code = sig_backprop_fn(
            path_ptr,
            out_ptr,
            cotangent_ptr,
            sig_ptr,
            spec.dimension,
            spec.length,
            static_cast<std::uint64_t>(degree),
            time_aug,
            lead_lag,
            static_cast<T>(end_time)
        );
    }

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
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) {
        return InvalidArgument(msg);
    }

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigCpuImpl<T>(
            CpuFns<T>::sig,
            CpuFns<T>::batch_sig,
            degree,
            time_aug,
            lead_lag,
            end_time,
            horner,
            n_jobs,
            path,
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
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "sig", sig); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigBackpropCpuImpl<T>(
            CpuFns<T>::backprop,
            CpuFns<T>::batch_backprop,
            degree,
            time_aug,
            lead_lag,
            end_time,
            n_jobs,
            path,
            sig,
            cotangent,
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
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) {
        return InvalidArgument(msg);
    }

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigCudaImpl<T>(
            stream,
            CudaFns<T>::sig,
            CudaFns<T>::batch_sig,
            degree,
            time_aug,
            lead_lag,
            end_time,
            horner,
            n_jobs,
            path,
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
    ffi::Result<ffi::AnyBuffer> out
) {
    if (auto msg = ValidateSameFloatDtype("path", path, "sig", sig); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "cotangent", cotangent); !msg.empty()) return InvalidArgument(msg);
    if (auto msg = ValidateSameFloatDtype("path", path, "out", out); !msg.empty()) return InvalidArgument(msg);

    return DispatchFloatDtype(BufferElementType(path), [&]<typename T>() -> ffi::Error {
        return SigBackpropCudaImpl<T>(
            stream,
            CudaFns<T>::backprop,
            CudaFns<T>::batch_backprop,
            degree,
            time_aug,
            lead_lag,
            end_time,
            n_jobs,
            path,
            sig,
            cotangent,
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
    if (dims.size() == 1) {
        spec.is_batch = false;
        spec.batch_size = 1;
        spec.sig_len = static_cast<std::uint64_t>(dims[0]);
        return {};
    }
    if (dims.size() == 2) {
        spec.is_batch = true;
        spec.batch_size = static_cast<std::uint64_t>(dims[0]);
        spec.sig_len = static_cast<std::uint64_t>(dims[1]);
        return {};
    }
    std::ostringstream oss;
    oss << "signature must have rank 1 or 2, got rank " << dims.size();
    return oss.str();
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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CpuFns<T>::batch_sig_combine(
            sig1_ptr, sig2_ptr, out_ptr,
            spec.batch_size,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree),
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = CpuFns<T>::sig_combine(
            sig1_ptr, sig2_ptr, out_ptr,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree)
        );
    }

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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CpuFns<T>::batch_sig_combine_backprop(
            cot_ptr, grad1_ptr, grad2_ptr, sig1_ptr, sig2_ptr,
            spec.batch_size,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree),
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = CpuFns<T>::sig_combine_backprop(
            cot_ptr, grad1_ptr, grad2_ptr, sig1_ptr, sig2_ptr,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree)
        );
    }

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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CudaFns<T>::batch_sig_combine(
            sig1_ptr, sig2_ptr, out_ptr,
            spec.batch_size,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree)
        );
    } else {
        err_code = CudaFns<T>::sig_combine(
            sig1_ptr, sig2_ptr, out_ptr,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree)
        );
    }

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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CudaFns<T>::batch_sig_combine_backprop(
            cot_ptr, grad1_ptr, grad2_ptr, sig1_ptr, sig2_ptr,
            spec.batch_size,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree)
        );
    } else {
        err_code = CudaFns<T>::sig_combine_backprop(
            cot_ptr, grad1_ptr, grad2_ptr, sig1_ptr, sig2_ptr,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree)
        );
    }

    if (err_code != 0) return NativeCallError("sig_combine_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error SigCombineCuda(
    cudaStream_t stream,
    std::int64_t dimension,
    std::int64_t degree,
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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CpuFns<T>::batch_transform_path(
            path_ptr, out_ptr,
            spec.batch_size, spec.dimension, spec.length,
            time_aug, lead_lag, static_cast<T>(end_time),
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = CpuFns<T>::transform_path(
            path_ptr, out_ptr,
            spec.dimension, spec.length,
            time_aug, lead_lag, static_cast<T>(end_time)
        );
    }

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
    bool is_batch = (dims.size() == 3);
    std::uint64_t batch_size = is_batch ? static_cast<std::uint64_t>(dims[0]) : 1;

    const auto* cot_ptr = BufferData<T>(cotangent);
    auto* out_ptr = BufferData<T>(out);

    int err_code = 0;
    if (is_batch) {
        err_code = CpuFns<T>::batch_transform_path_backprop(
            cot_ptr, out_ptr,
            batch_size,
            static_cast<std::uint64_t>(orig_dimension),
            static_cast<std::uint64_t>(orig_length),
            time_aug, lead_lag, static_cast<T>(end_time),
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = CpuFns<T>::transform_path_backprop(
            cot_ptr, out_ptr,
            static_cast<std::uint64_t>(orig_dimension),
            static_cast<std::uint64_t>(orig_length),
            time_aug, lead_lag, static_cast<T>(end_time)
        );
    }

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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CudaFns<T>::batch_transform_path(
            path_ptr, out_ptr,
            spec.batch_size, spec.dimension, spec.length,
            time_aug, lead_lag, static_cast<T>(end_time)
        );
    } else {
        err_code = CudaFns<T>::transform_path(
            path_ptr, out_ptr,
            spec.dimension, spec.length,
            time_aug, lead_lag, static_cast<T>(end_time)
        );
    }

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
    bool is_batch = (dims.size() == 3);
    std::uint64_t batch_size = is_batch ? static_cast<std::uint64_t>(dims[0]) : 1;

    const auto sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) return InternalError(cudaGetErrorString(sync_status));

    const auto* cot_ptr = BufferData<T>(cotangent);
    auto* out_ptr = BufferData<T>(out);

    int err_code = 0;
    if (is_batch) {
        err_code = CudaFns<T>::batch_transform_path_backprop(
            cot_ptr, out_ptr,
            batch_size,
            static_cast<std::uint64_t>(orig_dimension),
            static_cast<std::uint64_t>(orig_length),
            time_aug, lead_lag, static_cast<T>(end_time)
        );
    } else {
        err_code = CudaFns<T>::transform_path_backprop(
            cot_ptr, out_ptr,
            static_cast<std::uint64_t>(orig_dimension),
            static_cast<std::uint64_t>(orig_length),
            time_aug, lead_lag, static_cast<T>(end_time)
        );
    }

    if (err_code != 0) return NativeCallError("transform_path_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error TransformPathCuda(
    cudaStream_t stream,
    bool time_aug,
    bool lead_lag,
    double end_time,
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
// CUDA C++ takes just dimension/degree/method.

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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CpuFns<T>::batch_sig_to_log_sig(
            sig_ptr, out_ptr,
            spec.batch_size,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree),
            false, false,
            static_cast<int>(method),
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = CpuFns<T>::sig_to_log_sig(
            sig_ptr, out_ptr,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree),
            false, false,
            static_cast<int>(method)
        );
    }

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

    int err_code = 0;
    if (spec.is_batch) {
        err_code = CpuFns<T>::batch_sig_to_log_sig_backprop(
            sig_ptr, out_ptr, cot_ptr,
            spec.batch_size,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree),
            false, false,
            static_cast<int>(method),
            static_cast<int>(n_jobs)
        );
    } else {
        err_code = CpuFns<T>::sig_to_log_sig_backprop(
            sig_ptr, out_ptr, cot_ptr,
            static_cast<std::uint64_t>(dimension),
            static_cast<std::uint64_t>(degree),
            false, false,
            static_cast<int>(method)
        );
    }

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

    int err_code = spec.is_batch
        ? CudaFns<T>::batch_sig_to_log_sig(BufferData<T>(sig_buf), BufferData<T>(out),
              spec.batch_size, static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(method))
        : CudaFns<T>::sig_to_log_sig(BufferData<T>(sig_buf), BufferData<T>(out),
              static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(method));
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

    int err_code = spec.is_batch
        ? CudaFns<T>::batch_sig_to_log_sig_backprop(BufferData<T>(sig_buf), BufferData<T>(out), BufferData<T>(cotangent),
              spec.batch_size, static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(method))
        : CudaFns<T>::sig_to_log_sig_backprop(BufferData<T>(sig_buf), BufferData<T>(out), BufferData<T>(cotangent),
              static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(method));
    if (err_code != 0) return NativeCallError("sig_to_log_sig_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error SigToLogSigCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
    ffi::AnyBuffer sig_buf, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("sig", sig_buf); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(sig_buf), [&]<typename T>() -> ffi::Error {
        return SigToLogSigCudaImpl<T>(stream, dimension, degree, method, sig_buf, out);
    });
}

ffi::Error SigToLogSigBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree, std::int64_t method,
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

    int err_code = spec.is_batch
        ? CpuFns<T>::batch_log_sig_combine(BufferData<T>(ls1), BufferData<T>(ls2), BufferData<T>(out),
              spec.batch_size, static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(n_jobs))
        : CpuFns<T>::log_sig_combine(BufferData<T>(ls1), BufferData<T>(ls2), BufferData<T>(out),
              static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree));
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

    int err_code = spec.is_batch
        ? CpuFns<T>::batch_log_sig_combine_backprop(BufferData<T>(cotangent), BufferData<T>(grad1), BufferData<T>(grad2),
              BufferData<T>(ls1), BufferData<T>(ls2), spec.batch_size,
              static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree), static_cast<int>(n_jobs))
        : CpuFns<T>::log_sig_combine_backprop(BufferData<T>(cotangent), BufferData<T>(grad1), BufferData<T>(grad2),
              BufferData<T>(ls1), BufferData<T>(ls2),
              static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree));
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
    int err_code = spec.is_batch
        ? CudaFns<T>::batch_log_sig_combine(BufferData<T>(ls1), BufferData<T>(ls2), BufferData<T>(out),
              spec.batch_size, static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree))
        : CudaFns<T>::log_sig_combine(BufferData<T>(ls1), BufferData<T>(ls2), BufferData<T>(out),
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
    int err_code = spec.is_batch
        ? CudaFns<T>::batch_log_sig_combine_backprop(BufferData<T>(cotangent), BufferData<T>(grad1), BufferData<T>(grad2),
              BufferData<T>(ls1), BufferData<T>(ls2), spec.batch_size,
              static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree))
        : CudaFns<T>::log_sig_combine_backprop(BufferData<T>(cotangent), BufferData<T>(grad1), BufferData<T>(grad2),
              BufferData<T>(ls1), BufferData<T>(ls2),
              static_cast<std::uint64_t>(dimension), static_cast<std::uint64_t>(degree));
    if (err_code != 0) return NativeCallError("log_sig_combine_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error LogSigCombineCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
    ffi::AnyBuffer ls1, ffi::AnyBuffer ls2, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateSameFloatDtype("ls1", ls1, "ls2", ls2); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(ls1), [&]<typename T>() -> ffi::Error {
        return LogSigCombineCudaImpl<T>(stream, dimension, degree, ls1, ls2, out);
    });
}

ffi::Error LogSigCombineBackpropCuda(cudaStream_t stream, std::int64_t dimension, std::int64_t degree,
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
    if (dims.size() == 2) {
        spec.is_batch = false;
        spec.batch_size = 1;
        spec.length1 = static_cast<std::uint64_t>(dims[0]);
        spec.length2 = static_cast<std::uint64_t>(dims[1]);
        return {};
    }
    if (dims.size() == 3) {
        spec.is_batch = true;
        spec.batch_size = static_cast<std::uint64_t>(dims[0]);
        spec.length1 = static_cast<std::uint64_t>(dims[1]);
        spec.length2 = static_cast<std::uint64_t>(dims[2]);
        return {};
    }
    return "gram must have rank 2 or 3";
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

    int err_code = spec.is_batch
        ? CpuFns<T>::batch_sig_kernel(BufferData<T>(gram), BufferData<T>(out),
              spec.batch_size, static_cast<std::uint64_t>(dimension), length1, length2,
              static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2),
              return_grid, static_cast<int>(n_jobs))
        : CpuFns<T>::sig_kernel(BufferData<T>(gram), BufferData<T>(out),
              static_cast<std::uint64_t>(dimension), length1, length2,
              static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2),
              return_grid);
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

    int err_code = spec.is_batch
        ? CpuFns<T>::batch_sig_kernel_backprop(BufferData<T>(gram), BufferData<T>(out),
              BufferData<T>(derivs), BufferData<T>(k_grid),
              spec.batch_size, static_cast<std::uint64_t>(dimension), length1, length2,
              static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2),
              return_grid, static_cast<int>(n_jobs))
        : CpuFns<T>::sig_kernel_backprop(BufferData<T>(gram), BufferData<T>(out),
              BufferData<T>(derivs), BufferData<T>(k_grid),
              static_cast<std::uint64_t>(dimension), length1, length2,
              static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2),
              return_grid);
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
    int err_code = spec.is_batch
        ? CudaFns<T>::batch_sig_kernel(BufferData<T>(gram), BufferData<T>(out),
              spec.batch_size, static_cast<std::uint64_t>(dimension), length1, length2,
              static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2), return_grid)
        : CudaFns<T>::sig_kernel(BufferData<T>(gram), BufferData<T>(out),
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
    int err_code = spec.is_batch
        ? CudaFns<T>::batch_sig_kernel_backprop(BufferData<T>(gram), BufferData<T>(out),
              BufferData<T>(derivs), BufferData<T>(k_grid),
              spec.batch_size, static_cast<std::uint64_t>(dimension), length1, length2,
              static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2), return_grid)
        : CudaFns<T>::sig_kernel_backprop(BufferData<T>(gram), BufferData<T>(out),
              BufferData<T>(derivs), BufferData<T>(k_grid),
              static_cast<std::uint64_t>(dimension), length1, length2,
              static_cast<std::uint64_t>(dyadic_order_1), static_cast<std::uint64_t>(dyadic_order_2), return_grid);
    if (err_code != 0) return NativeCallError("sig_kernel_backprop_cuda", err_code);
    return ffi::Error::Success();
}

ffi::Error SigKernelPdeCuda(cudaStream_t stream, std::int64_t dimension,
    std::int64_t dyadic_order_1, std::int64_t dyadic_order_2, bool return_grid,
    ffi::AnyBuffer gram, ffi::Result<ffi::AnyBuffer> out) {
    if (auto msg = ValidateFloatBuffer("gram", gram); !msg.empty()) return InvalidArgument(msg);
    return DispatchFloatDtype(BufferElementType(gram), [&]<typename T>() -> ffi::Error {
        return SigKernelPdeCudaImpl<T>(stream, dimension, dyadic_order_1, dyadic_order_2, return_grid, gram, out);
    });
}

ffi::Error SigKernelPdeBackpropCuda(cudaStream_t stream, std::int64_t dimension,
    std::int64_t dyadic_order_1, std::int64_t dyadic_order_2, bool return_grid,
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
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigToLogSigBackpropCuda, SigToLogSigBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree").Attr<std::int64_t>("method")
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
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibLogSigCombineBackpropCuda, LogSigCombineBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension").Attr<std::int64_t>("degree")
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
        .Attr<bool>("return_grid")
        .Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(PySigLibSigKernelPdeBackpropCuda, SigKernelPdeBackpropCuda,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<std::int64_t>("dimension")
        .Attr<std::int64_t>("dyadic_order_1").Attr<std::int64_t>("dyadic_order_2")
        .Attr<bool>("return_grid")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>());
#endif
