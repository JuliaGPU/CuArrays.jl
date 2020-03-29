function cublasMgCreate() handle = Ref{cublasMgHandle_t}(C_NULL)
    cublasMgCreate(handle)
    return handle[]
end


#=function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::CuVecOrMat,
                  B::CuVecOrMat,
                  beta::Number,
                  C::CuVecOrMat,
                  grid::CudaLibMGGrid)

    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    descA = CudaLibMGDescriptor(A, grid)
    descB = CudaLibMGDescriptor(B, grid)
    descC = CudaLibMGDescriptor(C, grid)
    descD = descC
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [alpha], descA, A, lda, descB, B, ldb, [beta], descC, C, ldc, descD, D, ldd, cudaDataType(eltype(D)), workspace, lwork)
    cublasMgGemm(mg_handle(), cutransA, cutransB, [alpha], descA, A, lda, descB, B, ldb, [beta], descC, C, ldc, descD, D, ldd, cudaDataType(eltype(D)), workspace, lwork[], streams)
    return D
end
=#

function allocateBuffers(grid, n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, streams, row_block_size, col_block_size, desc, D::Matrix)
    buffers  = Vector{CuMatrix{eltype(D)}}(undef, num_devices)
    numRows  = Vector{Int64}(undef, num_devices)
    numCols  = Vector{Int64}(undef, num_devices)
    typesize = sizeof(eltype(D))
    cudaLibMgGetLocalMatrixDimensions(desc, numRows, numCols)
    current_dev = device()
    llds = Vector{Int64}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid[1:num_devices])
        device!(dev)
        llds[di] = numRows[di]
        buffers[di] = CuMatrix{eltype(D)}(undef, numRows[di], numCols[di])
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 1))
        buffers[di] = CuArray(D[row_inds, col_inds])
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        synchronize(streams[di])
    end
    return buffers, llds
end

function returnBuffers(grid, n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, streams, row_block_size, col_block_size, desc, dDs, D::Matrix)
    buffers  = Vector{CuMatrix{eltype(D)}}(undef, num_devices)
    numRows  = Vector{Int64}(undef, num_devices)
    numCols  = Vector{Int64}(undef, num_devices)
    typesize = sizeof(eltype(D))
    cudaLibMgGetLocalMatrixDimensions(desc, numRows, numCols)
    current_dev = device()
    sub_Ds = Vector{Matrix}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 1))
        sub_Ds[di] = Array( dDs[di] )
        D[row_inds, col_inds] = sub_Ds[di]
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
    end
    device!(current_dev)
    return D
end
# out of device move the memory myself
function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Matrix,
                  B::Matrix,
                  beta::Number,
                  C::Matrix;
                  devs=[0])
    device!(devs[1])
    CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs), devs)
    dev_rows = 2
    dev_cols = 1
    #devs    = filter(dev->occursin("V100-PCIE-32GB", name(dev)), collect(CUDAdrv.devices()))[1:(dev_rows*dev_cols)]
    #devs = [CUDAdrv.device()]
    @show devs 
    grid = Ref{cudaLibMgGrid_t}(C_NULL)
    cudaLibMgCreateDeviceGrid(grid, dev_rows, dev_cols, devs, CUDALIBMG.CUDALIBMG_GRID_MAPPING_ROW_MAJOR)
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    descA    = CudaLibMGDescriptor(A, grid[], rowblocks=div(size(A, 1), dev_rows), colblocks=div(size(A, 2), dev_cols))
    descB    = CudaLibMGDescriptor(B, grid[], rowblocks=div(size(B, 1), dev_rows), colblocks=div(size(B, 2), dev_cols))
    descC    = CudaLibMGDescriptor(C, grid[], rowblocks=div(size(C, 1), dev_rows), colblocks=div(size(C, 2), dev_cols))
    ndevs    = length(devs)
    streams = Vector{CuStream}(undef, ndevs)
    dA, ldas = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), descA, A)
    dB, ldbs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(B, 1), dev_rows), div(size(B, 2), dev_cols), descB, B)
    dC, ldcs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, C)
    lwork     = Vector{Int64}(undef, ndevs)
    workspace = Vector{CuVector{eltype(C)}}(undef, ndevs)
    device!(devs[1])
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [alpha], descA, dA, ldas, descB, dB, ldbs, [beta], descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork)
    synchronize()
    # set up workspaces and streams
    for (di, dev) in enumerate(devs)
        device!(dev)
        @show di, lwork[di]
        workspace[di] = CuVector{eltype(C)}(undef, div(lwork[di], sizeof(eltype(C))))
        synchronize(streams[di])
        synchronize()
    end
    device!(devs[1])
    synchronize()
    cublasMgGemm(mg_handle(), cutransA, cutransB, eltype(C)[alpha], descA, dA, ldas, descB, dB, ldbs, eltype(C)[beta], descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork, streams)
    #=synchronize()
    for (di, dev) in enumerate(devs)
        device!(dev)
        synchronize(streams[di])
        #unsafe_free!(workspace[di])
    end
    device!(devs[1])
    C = returnBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, dC, C)=#
    return C
end

# out-of-device - CPU arrays must be host-managed in UVM!
#=function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Matrix,
                  B::Matrix,
                  beta::Number,
                  C::Matrix)
    devs    = filter(dev->occursin("V100", name(dev)), collect(CUDAdrv.devices()))[1:1]
    cublasMgDeviceSelect(mg_handle(), length(devs), devs)
    grid = Ref{cudaLibMgGrid_t}(C_NULL)
    cudaLibMgCreateDeviceGrid(grid, 1, 1, [-1], CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    descA    = CudaLibMGDescriptor(A, grid[])
    descB    = CudaLibMGDescriptor(B, grid[])
    descC    = CudaLibMGDescriptor(C, grid[])

    ndevs     = length(devs)
    lwork     = Vector{Int64}(undef, ndevs)
    workspace = Vector{CuVector{eltype(C)}}(undef, ndevs)

    #A_buf = CUDAdrv.Mem.alloc(CUDAdrv.Mem.HostBuffer, length(A)*sizeof(eltype(A)))
    #B_buf = CUDAdrv.Mem.alloc(CUDAdrv.Mem.HostBuffer, length(B)*sizeof(eltype(B)))
    #C_buf = CUDAdrv.Mem.alloc(CUDAdrv.Mem.HostBuffer, length(C)*sizeof(eltype(C)))
    #A_arr = unsafe_wrap(Array{eltype(A),2}, convert(Ptr{eltype(A)}, A_buf), size(A); own=true)
    #B_arr = unsafe_wrap(Array{eltype(B),2}, convert(Ptr{eltype(B)}, B_buf), size(B); own=true)
    #C_arr = unsafe_wrap(Array{eltype(C),2}, convert(Ptr{eltype(C)}, C_buf), size(C); own=true)
    #Base.unsafe_copyto!(pointer(A_arr), pointer(A), length(A))
    #Base.unsafe_copyto!(pointer(B_arr), pointer(B), length(B))
    #Base.unsafe_copyto!(pointer(C_arr), pointer(C), length(C))
    A_arr = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(A), length(A)*sizeof(eltype(A)))
    B_arr = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(B), length(B)*sizeof(eltype(B)))
    C_arr = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(C), length(B)*sizeof(eltype(C)))
    ldcc  = Int64[ldc]
    #=C_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    C_arr[1] = C_buf
    B_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    B_arr[1] = B_buf
    A_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    A_arr[1] = A_buf=#

    C_ref_arr = [C_arr] 
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [alpha], descA, [A_arr], [lda], descB, [B_arr], [ldb], [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork)
    
    # set up workspaces and streams
    streams = Vector{CuStream}(undef, ndevs)
    for (di, dev) in enumerate(devs)
        device!(dev)
        @show di, lwork[di]
        workspace[di] = CuVector{eltype(C)}(undef, Int(lwork[di]))
        streams[di] = CuStream()
        synchronize()
    end
    cublasMgGemm(mg_handle(), cutransA, cutransB, [alpha], descA, [A_arr], [lda], descB, [B_arr], [ldb], [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork, streams)
    for (di, dev) in enumerate(devs)
        device!(dev)
        synchronize(streams[di])
        unsafe_free!(workspace[di])
    end
    copyto!(C, C_ref_arr[1])
    return C
end
=#
