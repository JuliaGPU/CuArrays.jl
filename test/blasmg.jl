using LinearAlgebra
@testset "CUBLASMB" begin

using CuArrays.CUBLASMG
using CUDAdrv

voltas    = filter(dev->occursin("V100", name(dev)), collect(CUDAdrv.devices()))
pascals    = filter(dev->occursin("P100-PCIE", name(dev)), collect(CUDAdrv.devices()))
#CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(pascals), pascals)
m = 8192
n = div(8192, 2)
k = 8192*2
@testset "element type $elty" for elty in [Float32, Float64]
    alpha = convert(elty,1.1)
    beta  = convert(elty,0.0)
    devs = voltas[1:1]
    CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs), devs)
    @testset "mg_gemm!" begin
        C = zeros(elty, m, n)
        A = rand(elty, k, m)
        B = rand(elty, k, n)
        d_C = copy(C)
        d_C = CUBLASMG.mg_gemm!('T','N',alpha,A,B,beta,d_C, devs=devs)
        # compare
        h_C = alpha*transpose(A)*B + beta*C
        @test d_C ≈ h_C
        C = zeros(elty, m, n)
        A = rand(elty, m, k)
        B = rand(elty, n, k)
        d_C = copy(C)
        d_C = CUBLASMG.mg_gemm!('N','T',alpha,A,B,beta,d_C, devs=devs)
        # compare
        h_C = alpha*A*transpose(B) + beta*C
        @test d_C ≈ h_C

        C = zeros(elty, m, n)
        A = rand(elty, m, k)
        B = rand(elty, k, n)
        d_C = copy(C)
        d_C = CUBLASMG.mg_gemm!('N','N',alpha,A,B,beta,d_C, devs=devs)
        # compare
        h_C = alpha*A*B + beta*C
        @test d_C ≈ h_C
    end
    # can only set devices once per handle
    CUBLASMG.cublasMgDestroy(CUBLASMG.mg_handle())
    fill!(CUBLASMG.thread_handles, nothing)
    devs = voltas[1:4]
    CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs), devs)
    @testset "mg_gemm_gpu!" begin
        C = zeros(elty, m, n)
        A = rand(elty, k, m)
        B = rand(elty, k, n)
        h_C = alpha*transpose(A)*B + beta*C
        d_C = copy(C)
        d_C = CUBLASMG.mg_gemm_gpu!('T','N',alpha,A,B,beta,d_C, devs=devs, dev_rows=2, dev_cols=2)
        @test d_C ≈ h_C
        C = zeros(elty, m, n)
        A = rand(elty, m, k)
        B = rand(elty, n, k)
        h_C = alpha*A*transpose(B) + beta*C
        d_C = copy(C)
        d_C = CUBLASMG.mg_gemm_gpu!('N','T',alpha,A,B,beta,d_C, devs=devs, dev_rows=2, dev_cols=2)
        @test d_C ≈ h_C

        C = zeros(elty, m, n)
        A = rand(elty, m, k)
        B = rand(elty, k, n)
        h_C = alpha*A*B + beta*C
        d_C = copy(C)
        d_C = CUBLASMG.mg_gemm_gpu!('N','N',alpha,A,B,beta,d_C, devs=devs, dev_rows=2, dev_cols=2)
        @test d_C ≈ h_C
    end
end # elty

end # cublasmg testset
