function cublasMgCreate()
    handle = Ref{cublasMgHandle_t}()
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
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        llds[di]    = numRows[di]
        buffers[di] = CuMatrix{eltype(D)}(undef, numRows[di], numCols[di])
        dev_row     = mod(di - 1, n_row_devs) + 1
        dev_col     = div(di - 1, n_row_devs) + 1
        row_inds    = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds    = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 1))
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        cpu_buf = D[row_inds, col_inds]
        println()
        println("IN ALLOCATOR:")
        @show di, dev, dev_row, dev_col, numRows[di], numCols[di], sizeof(cpu_buf)
        println()
        flush(stdout)
        buffers[di] = CuArray(cpu_buf)
        #synchronize(streams[di])
    end
    device!(deviceIdsGrid[1])
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
    device!(deviceIdsGrid[1])
    return D
end
# out of device move the memory myself
#=function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Matrix,
                  B::Matrix,
                  beta::Number,
                  C::Matrix;
                  devs=[0], dev_rows=1, dev_cols=1)
    device!(devs[1])
    @show devs 
    grid = Ref{cudaLibMgGrid_t}(0)
    cudaLibMgCreateDeviceGrid(grid, dev_rows, dev_cols, devs, CUDALIBMG.CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    descA    = CudaLibMGDescriptor(A, grid[], rowblocks=div(size(A, 1), dev_rows), colblocks=div(size(A, 2), dev_cols))
    descB    = CudaLibMGDescriptor(B, grid[], rowblocks=div(size(B, 1), dev_rows), colblocks=div(size(B, 2), dev_cols))
    descC    = CudaLibMGDescriptor(C, grid[], rowblocks=div(size(C, 1), dev_rows), colblocks=div(size(C, 2), dev_cols))
    ndevs    = length(devs)
    streams  = Vector{CuStream}(undef, ndevs)
    dA, ldas = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), descA, A)
    dB, ldbs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(B, 1), dev_rows), div(size(B, 2), dev_cols), descB, B)
    dC, ldcs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, C)
    lwork     = Vector{Int}(undef, ndevs)
    workspace = Vector{CuVector{eltype(C)}}(undef, ndevs)
    device!(devs[1])
    alpha_arr = [alpha]
    beta_arr  = [beta]
    GC.@preserve descA descB descC dA dB dC alpha_arr beta_arr workspace lwork A B C ldas ldbs ldcs begin cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, alpha_arr, descA, dA, ldas, descB, dB, ldbs, beta_arr, descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork); synchronize() end
    # set up workspaces and streams
    for (di, dev) in enumerate(devs)
        device!(dev)
        synchronize()
        println("IN WORKSPACE")
        @show dev, device(), lwork[di]
        flush(stdout)
        workspace[di] = CuVector{eltype(C)}(undef, div(convert(Int, lwork[di]), sizeof(eltype(C))))
        synchronize()
    end
    device!(devs[1])
    println("BEGIN GEMM")
    flush(stdout)
    synchronize()
    GC.@preserve descA descB descC dA dB dC alpha_arr beta_arr workspace lwork A B C ldas ldbs ldcs streams begin cublasMgGemm(mg_handle(), cutransA, cutransB, alpha_arr, descA, dA, ldas, descB, dB, ldbs, beta_arr, descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork, streams); synchronize() end
    println("DONE GEMM")
    flush(stdout)
    for (di, dev) in enumerate(devs)
        device!(dev)
        synchronize(streams[di])
    end
    println("DONE STREAM SYNC")
    flush(stdout)
    device!(devs[1])
    C = returnBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, dC, C)
    return C
end=#
# out-of-device - CPU arrays must be host-managed in UVM!
function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Vector,
                  B::Vector,
                  beta::Number,
                  C::Vector; devs=[0])
    grid = CudaLibMGGrid(1, 1, [-1], CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    #lda = Int64(max(1,stride(A,2)))
    #ldb = Int64(max(1,stride(B,2)))
    #ldc = Int64(max(1,stride(C,2)))
    lda = 8192
    ldb = 8192
    ldc = 8192
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    descA    = CudaLibMGDescriptor(A, grid, rowblocks=8192, colblocks=8192)
    descB    = CudaLibMGDescriptor(B, grid, rowblocks=8192, colblocks=8192)
    descC    = CudaLibMGDescriptor(C, grid, rowblocks=8192, colblocks=8192)
    ndevs    = length(devs)
    C_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    B_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    A_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    C_ref_arr = Vector{Ptr}(undef, ndevs)
    B_ref_arr = Vector{Ptr}(undef, ndevs)
    A_ref_arr = Vector{Ptr}(undef, ndevs)
    device!(devs[1])
    for (di, dev) in enumerate(devs)
        device!(dev)
        A_arr[di] = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(A), length(A)*sizeof(eltype(A)))
        B_arr[di] = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(B), length(B)*sizeof(eltype(B)))
        C_arr[di] = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(C), length(B)*sizeof(eltype(C)))
        A_ref_arr[di] = A_arr[di].ptr 
        B_ref_arr[di] = B_arr[di].ptr
        C_ref_arr[di] = C_arr[di].ptr
    end
    println("Set up buffers")
    flush(stdout)
    device!(devs[1])
    
    ldcc      = Int64[ldc]
    lwork     = Vector{Csize_t}(undef, ndevs)
    workspace = Vector{CuVector}(undef, ndevs)
    
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, [lda], descB, B_ref_arr, [ldb], [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork)
    # set up workspaces and streams
    streams = Vector{CuStream}(undef, ndevs)
    for (di, dev) in enumerate(devs)
        device!(dev)
        @show di, lwork[di]
        flush(stdout)
        #buf = CUDAdrv.Mem.alloc(CUDAdrv.Mem.DeviceBuffer, lwork[di])
        workspace[di] = CuVector{eltype(C)}(undef, div(Int(lwork[di]), sizeof(eltype(C))))
        streams[di]   = CuStream()
        synchronize(streams[di])
    end
    device!(devs[1])
    println("Begin gemm")
    flush(stdout)
    
    GC.@preserve descA descB descC A_ref_arr A_arr B_ref_arr B_arr C_ref_arr C_arr workspace lwork A B C begin cublasMgGemm(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, [lda], descB, B_ref_arr, [ldb], [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork, streams); synchronize() end
    for (di, dev) in enumerate(devs)
        device!(dev)
        CUDAdrv.Mem.unregister(A_arr[di])
        CUDAdrv.Mem.unregister(B_arr[di])
        CUDAdrv.Mem.unregister(C_arr[di])
        synchronize(streams[di])
    end
    println("Done stream sync")
    flush(stdout)
    device!(devs[1])
    return C
end
