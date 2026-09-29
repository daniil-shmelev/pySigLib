# Dependency-free example: Julia -> CUDA runtime allocation -> cusig -> Julia.
# In a Julia package, pass artifact"pysiglib_cuda12" to load_libraries().
module SigLibCUDAExample
using Libdl

function load_libraries(root)
    directory = joinpath(root, "pysiglib_cuda")
    runtime = if Sys.iswindows()
        joinpath(directory, "cudart64_12.dll")
    else
        only(filter(p -> startswith(basename(p), "libcudart-"),
            readdir(joinpath(root, "pysiglib_cuda.libs"); join=true)))
    end
    # Preload cudart explicitly on Windows, where the DLL search path differs.
    cudart = Libdl.dlopen(runtime)
    filename = Sys.iswindows() ? "cusig.dll" : "libcusig.$(Libdl.dlext)"
    cusig = Libdl.dlopen(joinpath(directory, filename))
    return (; cusig, cudart)
end

function check_cuda(libraries, status)
    status == 0 && return
    message = ccall(Libdl.dlsym(libraries.cudart, :cudaGetErrorString), Cstring, (Cint,), status)
    error("CUDA: ", unsafe_string(message))
end

function linear_signature(libraries, displacement::Vector{Float64}, degree::Integer)
    dimension = length(displacement)
    dimension > 0 && degree > 0 || throw(ArgumentError("dimension and degree must be positive"))
    count, level = 1, 1
    for _ in 1:degree
        level = Base.checked_mul(level, dimension)
        count = Base.checked_add(count, level)
    end
    signature = zeros(count)
    device_input, device_output = Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL)
    try
        check_cuda(libraries, ccall(Libdl.dlsym(libraries.cudart, :cudaMalloc), Cint,
            (Ref{Ptr{Cvoid}}, Csize_t), device_input, sizeof(displacement)))
        check_cuda(libraries, ccall(Libdl.dlsym(libraries.cudart, :cudaMalloc), Cint,
            (Ref{Ptr{Cvoid}}, Csize_t), device_output, sizeof(signature)))
        check_cuda(libraries, ccall(Libdl.dlsym(libraries.cudart, :cudaMemcpy), Cint,
            (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Cint), device_input[], displacement, sizeof(displacement), 1))
        status = ccall(Libdl.dlsym(libraries.cusig, :linear_sig_cuda_d), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, UInt64, UInt64, UInt64, Bool),
            device_input[], device_output[], 1, dimension, degree, true)
        if status != 0
            message = ccall(Libdl.dlsym(libraries.cusig, :cusig_last_error_message), Cstring, ())
            error("cusig: ", unsafe_string(message))
        end
        check_cuda(libraries, ccall(Libdl.dlsym(libraries.cudart, :cudaDeviceSynchronize), Cint, ()))
        check_cuda(libraries, ccall(Libdl.dlsym(libraries.cudart, :cudaMemcpy), Cint,
            (Ptr{Cdouble}, Ptr{Cvoid}, Csize_t, Cint), signature, device_output[], sizeof(signature), 2))
        return signature
    finally
        for pointer in (device_input[], device_output[])
            pointer == C_NULL || ccall(Libdl.dlsym(libraries.cudart, :cudaFree), Cint, (Ptr{Cvoid},), pointer)
        end
    end
end
end
