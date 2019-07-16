using CuArrays, Test

W = rand(5, 5)
b = rand(5)

@test cuda() do
  W*b
end ≈ W*b
