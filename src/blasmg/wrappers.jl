function cublasMgCreate() handle = Ref{cublasMgHandle_t}(C_NULL)
    cublasMgCreate(handle)
    return handle[]
end


function allocateBuffers(grid, n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, streams, row_block_size, col_block_size, desc, D::Matrix)
    buffers  = Vector{CuMatrix{eltype(D)}}(undef, num_devices)
    numRows  = Vector{Int64}(undef, num_devices)
    numCols  = Vector{Int64}(undef, num_devices)
    typesize = sizeof(eltype(D))
    cudaLibMgGetLocalMatrixDimensions(desc, numRows, numCols)
    llds = Vector{Int64}(undef, num_devices)
    println("input to allocator:")
    display(D)
    println()
    for (di, dev) in enumerate(deviceIdsGrid[1:num_devices])
        device!(dev)
        llds[di] = numRows[di]
        buffers[di] = CuMatrix{eltype(D)}(undef, numRows[di], numCols[di])
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 1))
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        buffers[di] = CuArray(D[row_inds, col_inds])
        @show di, dev, size(buffers[di])
        display(D[row_inds, col_inds])
        println()
        synchronize(streams[di])
    end
    device!(0)
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
    device!(0)
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
    #device!(devs[1])
    device!(0)
    CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs), devs)
    dev_rows = 1
    dev_cols = 1
    #devs    = filter(dev->occursin("V100-PCIE-32GB", name(dev)), collect(CUDAdrv.devices()))[1:(dev_rows*dev_cols)]
    #devs = [CUDAdrv.device()]
    @show devs 
    grid = Ref{cudaLibMgGrid_t}(C_NULL)
    cudaLibMgCreateDeviceGrid(grid, dev_rows, dev_cols, devs, CUDALIBMG.CUDALIBMG_GRID_MAPPING_ROW_MAJOR)
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    #descA    = CudaLibMGDescriptor(A, grid[], rowblocks=div(size(A, 1), dev_rows), colblocks=div(size(A, 2), dev_cols))
    #descB    = CudaLibMGDescriptor(B, grid[], rowblocks=div(size(B, 1), dev_rows), colblocks=div(size(B, 2), dev_cols))
    #descC    = CudaLibMGDescriptor(C, grid[], rowblocks=div(size(C, 1), dev_rows), colblocks=div(size(C, 2), dev_cols))
    #descA    = CudaLibMGDescriptor(A, grid[], rowblocks=2048, colblocks=2048)
    #descB    = CudaLibMGDescriptor(B, grid[], rowblocks=2048, colblocks=2048)
    #descC    = CudaLibMGDescriptor(C, grid[], rowblocks=2048, colblocks=2048)
    descA    = CudaLibMGDescriptor(A, grid[], rowblocks=1, colblocks=1)
    descB    = CudaLibMGDescriptor(B, grid[], rowblocks=1, colblocks=1)
    descC    = CudaLibMGDescriptor(C, grid[], rowblocks=1, colblocks=1)
    ndevs    = length(devs)
    streams = Vector{CuStream}(undef, ndevs)
    dA, ldas = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), descA, A)
    dB, ldbs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(B, 1), dev_rows), div(size(B, 2), dev_cols), descB, B)
    dC, ldcs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, C)
    lwork     = Vector{Int64}(undef, ndevs)
    workspace = Vector{CuVector{eltype(C)}}(undef, ndevs)
    device!(0)
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [convert(eltype(C), alpha)], descA, dA, ldas, descB, dB, ldbs, [convert(eltype(C), beta)], descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork)
    # set up workspaces and streams
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di] = CuVector{eltype(C)}(undef, div(lwork[di], sizeof(eltype(C))))
        synchronize(streams[di])
    end
    device!(0)
    @show ldas
    @show ldbs
    @show ldcs

    flush(stdout)
    cublasMgGemm(mg_handle(), cutransA, cutransB, [alpha], descA, dA, ldas, descB, dB, ldbs, [beta], descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork, streams)
    println("DONE GEMM")
    flush(stdout)
    sychronize()
    for (di, dev) in enumerate(devs)
        device!(dev)
        println("switched to dev $dev, isassigned stream: $(isassigned(streams, di)), is busy: $(query(streams[di]))")
        flush(stdout)
        synchronize(streams[di])
        #unsafe_free!(workspace[di])
    end
    println("DONE STREAM SYNC")
    flush(stdout)
    device!(0)
    C = returnBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, dC, C)
    return C
end

# out-of-device - CPU arrays must be host-managed in UVM!
#=function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Matrix,
                  B::Matrix,
                  beta::Number,
                  C::Matrix;
                  devs=[0])
    grid = Ref{cudaLibMgGrid_t}(C_NULL)
    cudaLibMgCreateDeviceGrid(grid, 1, 1, [-1], CUDALIBMG_GRID_MAPPING_ROW_MAJOR)
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
    synchronize()
    C_ref_arr = [pointer(C)]
    A_ref_arr = [pointer(A)]
    B_ref_arr = [pointer(B)]
    cublasMgDeviceSelect(mg_handle(), length(devs), devs)
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, [lda], descB, B_ref_arr, [ldb], [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork)
    # set up workspaces and streams
    streams = Vector{CuStream}(undef, ndevs)
    for (di, dev) in enumerate(devs)
        device!(dev)
        @show di, lwork[di]
        workspace[di] = CuVector{eltype(C)}(undef, div(lwork[di], sizeof(eltype(C))))
        streams[di] = CuStream()
        synchronize(streams[di])
    end
    device!(0)
    cublasMgGemm(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, [lda], descB, B_ref_arr, [ldb], [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork, streams)
    for (di, dev) in enumerate(devs)
        device!(dev)
        synchronize(streams[di])
    end
    device!(0)
    copyto!(C, C_ref_arr[1])
    return C
end
=#
