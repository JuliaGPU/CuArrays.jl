using LinearAlgebra

@testset "CUBLASMB" begin

using CuArrays.CUBLASMG
using CUDAdrv

@show name.(collect(CUDAdrv.devices()))
voltas    = filter(dev->occursin("V100-PCIE-32GB", name(dev)), collect(CUDAdrv.devices()))
CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(voltas), voltas)
m = 2^1 * length(voltas)
n = 2^1 * length(voltas)
k = 2^1 * length(voltas)

@testset "element type $elty" for elty in [Float32, Float64]
    alpha = convert(elty,2)
    beta = convert(elty,3)
    @testset "Level 3" begin
        A = rand(elty,m,k)
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        @testset "gemm!" begin
            d_C = copy(C)
            d_C = CUBLASMG.mg_gemm!('N','N',alpha,A,B,beta,d_C)
            # compare
            for dev in voltas
                device!(dev)
                CUDAdrv.synchronize()
            end
            h_C = (alpha*A)*B + beta*C
            @test d_C ≈ h_C
        end
    end
end # elty

end # cublasmg testset
