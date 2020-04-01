function cublasMgCreate()
    handle = Ref{cublasMgHandle_t}()
    cublasMgCreate(handle)
    return handle[]
end

function allocateBuffers(grid, n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, streams, row_block_size, col_block_size, desc, D, Dsize)
    buffers  = Vector{CuVector{eltype(D)}}(undef, num_devices)
    numRows  = Vector{Int64}(undef, num_devices)
    numCols  = Vector{Int64}(undef, num_devices)
    typesize = sizeof(eltype(D))
    cudaLibMgGetLocalMatrixDimensions(desc, numRows, numCols)
    llds = Vector{Int64}(undef, num_devices)
    Dinds = LinearIndices((1:Dsize[1], 1:Dsize[2]))
    sub_Ds = Vector{Vector}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        llds[di]    = numRows[di]
        buffers[di] = CuVector{eltype(D)}(undef, numRows[di]*numCols[di])
        dev_row     = mod(di - 1, n_row_devs) + 1
        dev_col     = div(di - 1, n_row_devs) + 1
        row_inds    = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, Dsize[1])
        col_inds    = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, Dsize[2])
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        cpu_buf = vec(D[Dinds[row_inds, col_inds]])
        println()
        println("IN ALLOCATOR:")
        @show di, dev, dev_row, dev_col, numRows[di], numCols[di], sizeof(cpu_buf)
        println()
        flush(stdout)
        buffers[di] = CuArray(cpu_buf)
        #synchronize(streams[di])
        synchronize()
    end
    device!(deviceIdsGrid[1])
    return buffers, llds
end

function returnBuffers(grid, n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, streams, row_block_size, col_block_size, desc, dDs, D, Dsize)
    buffers  = Vector{CuVector{eltype(D)}}(undef, num_devices)
    numRows  = Vector{Int64}(undef, num_devices)
    numCols  = Vector{Int64}(undef, num_devices)
    typesize = sizeof(eltype(D))
    cudaLibMgGetLocalMatrixDimensions(desc, numRows, numCols)
    current_dev = device()
    sub_Ds = Vector{Vector}(undef, num_devices)
    Dinds = LinearIndices((1:Dsize[1], 1:Dsize[2]))
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
        synchronize()
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 1))
        Dsubinds = LinearIndices((row_inds, col_inds)) 
        D[vec(Dsubinds)] = vec(collect(dDs[di]))
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
                  A::Vector, dimsA::Tuple{Int, Int},
                  B::Vector, dimsB::Tuple{Int, Int},
                  beta::Number,
                  C::Vector, dimsC::Tuple{Int, Int};
                  devs=[0], dev_rows=1, dev_cols=1)
    device!(devs[1])
    grid = Ref{cudaLibMgGrid_t}(0)
    C = zeros(eltype(A), dimsC[1]*dimsC[2])
    cudaLibMgCreateDeviceGrid(grid, dev_rows, dev_cols, devs, CUDALIBMG.CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    lda = dimsA[1]
    ldb = dimsB[1]
    ldc = dimsC[1]
    descA    = CudaLibMGDescriptor(A, grid[], rowblocks=div(dimsA[1], dev_rows), colblocks=div(dimsA[2], dev_cols))
    descB    = CudaLibMGDescriptor(B, grid[], rowblocks=div(dimsB[1], dev_rows), colblocks=div(dimsB[2], dev_cols))
    descC    = CudaLibMGDescriptor(C, grid[], rowblocks=div(dimsC[1], dev_rows), colblocks=div(dimsC[2], dev_cols))
    ndevs    = length(devs)
    streams  = Vector{CuStream}(undef, ndevs)
    dA, ldas = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(dimsA[1], dev_rows), div(dimsA[2], dev_cols), descA, A, dimsA)
    dB, ldbs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(dimsB[1], dev_rows), div(dimsB[2], dev_cols), descB, B, dimsB)
    dC, ldcs = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(dimsC[1], dev_rows), div(dimsC[2], dev_cols), descC, C, dimsC)
    lwork     = Vector{Int}(undef, ndevs)
    workspace = Vector{CuVector{eltype(C)}}(undef, ndevs)
    device!(devs[1])
    alpha_arr = Ref(alpha)
    beta_arr  = Ref(beta)
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, alpha_arr, descA, dA, ldas, descB, dB, ldbs, beta_arr, descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork)
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
    @show pointer(dC[1])
    @show pointer(dC)
    flush(stdout)
    
    synchronize()
    #GC.@preserve descA descB descC dA dB dC alpha_arr beta_arr workspace lwork A B C ldas ldbs ldcs streams begin cublasMgGemm(mg_handle(), cutransA, cutransB, alpha_arr, descA, dA, ldas, descB, dB, ldbs, beta_arr, descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork, streams); synchronize() end
    cublasMgGemm(mg_handle(), cutransA, cutransB, alpha_arr, descA, dA, ldas, descB, dB, ldbs, beta_arr, descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace, lwork, streams)
    hC = collect(dC[1])
    @show hC[1]
    println("DONE GEMM")
    @show pointer(dC[1])
    @show pointer(dC)
    flush(stdout)
    for (di, dev) in enumerate(devs)
        device!(dev)
        synchronize(streams[di])
        synchronize()
    end
    println("DONE STREAM SYNC")
    flush(stdout)
    device!(devs[1])
    C = returnBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, dC, C, dimsC)
    return C
end=#
# out-of-device - CPU arrays must be host-managed in UVM!
function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Vector, dimsA::Tuple{Int, Int},
                  B::Vector, dimsB::Tuple{Int, Int},
                  beta::Number,
                  C::Vector, dimsC::Tuple{Int, Int}; devs=[0])
    device!(devs[1])
    grid = CudaLibMGGrid(Int32(1), Int32(1), [Int32(-1)], CUDALIBMG_GRID_MAPPING_ROW_MAJOR)
    lda = dimsA[1]
    ldb = dimsB[1]
    ldc = dimsC[1]
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    descA    = CudaLibMGDescriptor(A, grid, rowblocks=dimsA[1], colblocks=dimsA[2])
    descB    = CudaLibMGDescriptor(B, grid, rowblocks=dimsB[1], colblocks=dimsB[2])
    descC    = CudaLibMGDescriptor(C, grid, rowblocks=dimsC[1], colblocks=dimsC[2])
    ndevs    = length(devs)
    C_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    B_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    A_arr = Vector{CUDAdrv.Mem.HostBuffer}(undef, ndevs)
    C_ref_arr = Vector{Ptr}(undef, ndevs)
    B_ref_arr = Vector{Ptr}(undef, ndevs)
    A_ref_arr = Vector{Ptr}(undef, ndevs)
    lwork     = Vector{Csize_t}(undef, ndevs)
    workspace = Vector{CuVector}(undef, ndevs)
    streams   = Vector{CuStream}(undef, ndevs)
    GC.@preserve descA descB descC A_ref_arr A_arr B_ref_arr B_arr C_ref_arr C_arr workspace lwork A B C streams begin
        for (di, dev) in enumerate(devs)
            A_arr[di] = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(A), length(A)*sizeof(eltype(A)), CUDAdrv.Mem.HOSTREGISTER_DEVICEMAP)
            B_arr[di] = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(B), length(B)*sizeof(eltype(B)), CUDAdrv.Mem.HOSTREGISTER_DEVICEMAP)
            C_arr[di] = CUDAdrv.Mem.register(CUDAdrv.Mem.HostBuffer, pointer(C), length(C)*sizeof(eltype(C)), CUDAdrv.Mem.HOSTREGISTER_DEVICEMAP)
            A_wrapper = unsafe_wrap(Array, convert(Ptr{eltype(A)}, A_arr[di]), length(A))
            B_wrapper = unsafe_wrap(Array, convert(Ptr{eltype(B)}, B_arr[di]), length(B))
            C_wrapper = unsafe_wrap(Array, convert(Ptr{eltype(C)}, C_arr[di]), length(C))
            A_ref_arr[di] = pointer(A)
            B_ref_arr[di] = pointer(B)
            C_ref_arr[di] = pointer(C)
            synchronize()
        end
        device!(devs[1])
        
        ldcc      = Ref(ldc)
        ldaa      = Ref(lda)
        ldbb      = Ref(ldb)
        cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, ldaa, descB, B_ref_arr, ldbb, [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork)
        # set up workspaces and streams
        for (di, dev) in enumerate(devs)
            device!(dev)
            workspace[di] = CuVector{eltype(C)}(undef, div(Int(lwork[di]), sizeof(eltype(C))))
            streams[di]   = CuStream()
            synchronize(streams[di])
            synchronize()
        end
        device!(devs[1])
        cublasMgGemm(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, ldaa, descB, B_ref_arr, ldbb, [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace, lwork, streams)
        for (di, dev) in enumerate(devs)
            device!(dev)
            synchronize(streams[di])
            synchronize()
            @show C[1]
            CUDAdrv.Mem.unregister(A_arr[di])
            CUDAdrv.Mem.unregister(B_arr[di])
            CUDAdrv.Mem.unregister(C_arr[di])
        end
        println("Done stream sync")
        flush(stdout)
        device!(devs[1])
    end
    @show C[1]
    return C
end

