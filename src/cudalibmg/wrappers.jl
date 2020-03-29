
mutable struct CudaLibMGDescriptor
    desc::cudaLibMgMatrixDesc_t

    function CudaLibMGDescriptor(a, grid; rowblocks = size(a, 1), colblocks = size(a, 2), eltype = eltype(a) )
        desc = Ref{cudaLibMgMatrixDesc_t}()
        cudaLibMgCreateMatrixDesc(desc, size(a, 1), size(a, 2), rowblocks, colblocks, cudaDataType(eltype), grid)
        return new(desc[])
    end
end

Base.cconvert(::Type{cudaLibMgMatrixDesc_t}, obj::CudaLibMGDescriptor) = obj.desc
