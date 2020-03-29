using LinearAlgebra

@testset "CUBLASMB" begin

using CuArrays.CUBLASMG
using CUDAdrv

voltas    = filter(dev->occursin("V100-PCIE", name(dev)), collect(CUDAdrv.devices()))
#voltas = [CUDAdrv.device()]
@show voltas
voltas = voltas[2:2]
#CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), 1, [0])
#m = 2^1 * length(voltas)
#n = 2^1 * length(voltas)
#k = 2^1 * length(voltas)
#m = 8192
#n = 8192
#k = 8192

m = 2
n = 2
k = 2

@testset "element type $elty" for elty in [Float64]
    alpha = convert(elty,1.1)
    beta = convert(elty,0.0)
    @testset "Level 3" begin
        A = rand(elty,m,k)
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        @testset "gemm!" begin
            d_C = copy(C)
            d_C = CUBLASMG.mg_gemm!('N','N',alpha,A,B,beta,d_C, devs=voltas)
            # compare
            #=for dev in voltas
                device!(dev)
                CUDAdrv.synchronize()
            end=#
            h_C = (alpha*A)*B + beta*C
            @test d_C ≈ h_C
        end
    end
end # elty

end # cublasmg testset
