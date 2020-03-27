# Julia wrapper for header: cublasMg.h
# Automatically generated using Clang.jl


function cublasMgGemm(handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork, streams)
    ccall((:cublasMgGemm, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, cublasOperation_t, cublasOperation_t, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, cudaDataType_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{cudaStream_t}), handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork, streams)
end

function cublasMgGemmWorkspace(handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork)
    ccall((:cublasMgGemmWorkspace, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, cublasOperation_t, cublasOperation_t, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, Ptr{Cvoid}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, cudaLibMgMatrixDesc_t, Ptr{Ptr{Cvoid}}, Ptr{Int64}, cudaDataType_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}), handle, transA, transB, alpha, descA, A, llda, descB, B, lldb, beta, descC, C, lldc, descD, D, lldd, computeType, workspace, lwork)
end

function cublasMgCreate(handle)
    ccall((:cublasMgCreate, libcublasmg()), cublasStatus_t, (Ptr{cublasMgHandle_t},), handle)
end

function cublasMgDestroy(handle)
    ccall((:cublasMgDestroy, libcublasmg()), cublasStatus_t, (cublasMgHandle_t,), handle)
end

function cublasMgDeviceSelect(handle, nbDevices, deviceIds)
    ccall((:cublasMgDeviceSelect, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, Cint, Ptr{Cint}), handle, nbDevices, deviceIds)
end

function cublasMgDeviceCount(handle, nbDevices)
    ccall((:cublasMgDeviceCount, libcublasmg()), cublasStatus_t, (cublasMgHandle_t, Ptr{Cint}), handle, nbDevices)
end

function cublasMgGetVersion()
    ccall((:cublasMgGetVersion, libcublasmg()), Csize_t, ())
end

function cublasMgGetCudartVersion()
    ccall((:cublasMgGetCudartVersion, libcublasmg()), Csize_t, ())
end
