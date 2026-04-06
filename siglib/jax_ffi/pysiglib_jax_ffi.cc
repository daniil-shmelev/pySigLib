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
};

template <>
struct CpuFns<double> {
    static constexpr auto sig = signature_d;
    static constexpr auto batch_sig = batch_signature_d;
    static constexpr auto backprop = sig_backprop_d;
    static constexpr auto batch_backprop = batch_sig_backprop_d;
    static constexpr const char* sig_name = "signature_d";
    static constexpr const char* backprop_name = "sig_backprop_d";
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
};

template <>
struct CudaFns<double> {
    static constexpr auto sig = signature_cuda_d;
    static constexpr auto batch_sig = batch_signature_cuda_d;
    static constexpr auto backprop = sig_backprop_cuda_d;
    static constexpr auto batch_backprop = batch_sig_backprop_cuda_d;
    static constexpr const char* sig_name = "signature_cuda_d";
    static constexpr const char* backprop_name = "sig_backprop_cuda_d";
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
