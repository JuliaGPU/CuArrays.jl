module CuArrays

using CUDAapi, CUDAdrv, CUDAnative

using GPUArrays

export CuArray, CuVector, CuMatrix, CuVecOrMat, cu
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND, CUDNN, CUTENSOR

import LinearAlgebra

using Adapt

using Libdl

using Requires


## dependencies

const deps = joinpath(@__DIR__, "..", "deps", "deps.jl")
isfile(deps) || error("CuArrays.jl has not been built, please run Pkg.build(\"CuArrays\").")
include(deps)

"""
    prefix()

Returns the installation prefix directories of the CUDA toolkit in use.
"""
prefix() = toolkit_dirs

"""
    version()

Returns the version of the CUDA toolkit in use.
"""
version() = toolkit_version

"""
    release()

Returns the CUDA release part of the version as returned by [`version`](@ref).
"""
release() = toolkit_release


## source code includes

# core array functionality
include("memory.jl")
include("array.jl")
include("gpuarrays.jl")
include("subarray.jl")
include("utils.jl")

# integrations and specialized functionality
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("accumulate.jl")
include("linalg.jl")
include("nnlib.jl")

# vendor libraries
include("blas/CUBLAS.jl")
include("sparse/CUSPARSE.jl")
include("solver/CUSOLVER.jl")
include("fft/CUFFT.jl")
include("rand/CURAND.jl")
include("dnn/CUDNN.jl")
include("tensor/CUTENSOR.jl")

include("deprecated.jl")


## initialization

const __initialized__ = Ref(false)
functional() = __initialized__[]

export has_cudnn, has_cutensor
has_cudnn() = Libdl.dlopen_e(CUDNN.libcudnn) !== C_NULL
has_cutensor() = Libdl.dlopen_e(CUTENSOR.libcutensor) !== C_NULL

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false")) || precompiling
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional() || !CUDAnative.functional()
        verbose && @warn "CuArrays.jl did not initialize because CUDAdrv.jl or CUDAnative.jl failed to"
        return
    end

    try
        configured || error("CuArrays.jl has not been successfully built, please run Pkg.build(\"CuArrays\").")

        # library dependencies
        CUBLAS.version()
        CUSPARSE.version()
        CUSOLVER.version()
        CUFFT.version()
        CURAND.version()
        # CUDNN and CUTENSOR are optional

        # library compatibility
        cuda = version()
        if has_cutensor()
            cutensor = CUTENSOR.version()
            if cutensor < v"1"
                silent || @warn("CuArrays.jl only supports CUTENSOR 1.0 or higher")
            end

            cutensor_cuda = CUTENSOR.cuda_version()
            if cutensor_cuda.major != cuda.major || cutensor_cuda.minor != cuda.minor
                silent || @warn("You are using CUTENSOR $cutensor for CUDA $cutensor_cuda with CUDA toolkit $cuda; these might be incompatible.")
            end
        end
        if has_cudnn()
            cudnn = CUDNN.version()
            if cudnn < v"7.6"
                silent || @warn("CuArrays.jl only supports CUDNN v7.6 or higher")
            end

            cudnn_cuda = CUDNN.cuda_version()
            if cudnn_cuda.major != cuda.major || cudnn_cuda.minor != cuda.minor
                silent || @warn("You are using CUDNN $cudnn for CUDA $cudnn_cuda with CUDA toolkit $cuda; these might be incompatible.")
            end
        end

        # package integrations
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("forwarddiff.jl")

        __init_memory__()

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent
            if verbose
                @error "CuArrays.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CuArrays.jl failed to initialize and will be unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

end # module
