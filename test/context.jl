using CuArrays, Test

W = rand(5, 5)
b = rand(5)

@test cuda(() -> W*b) ≈ W*b
