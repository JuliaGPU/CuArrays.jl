using IRTools: isexpr, IR, @dynamo, postwalk
using IRTools: meta, Pipe, finish, Variable, self
using MacroTools: @forward

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize
import Base.Broadcast.Broadcasted

# TODO use a WeakKeyDict
struct CUDACtx
  array_bank::IdDict{Array,CuArray}
end

CUDACtx() = CUDACtx(IdDict{Array,CuArray}())

# Display fns for debugging, remove before committing
function Base.summary(io::IO, c::CUDACtx)
  print(io, "IR Context for CUDA ")
  summary(io, c.array_bank)
end

function Base.show(io::IO, c::CUDACtx)
  print(io, "IR Context for CUDA ")
  display(c.array_bank)
end

@forward CUDACtx.array_bank Base.getindex, Base.iterate,
			Base.setindex!, Base.empty!,
			Base.length,
			Base.first, Base.last, Base.haskey

function _resize!(a::Array, sz::NTuple{<:Any,Integer})
  ccall(:jl_array_grow_end, Cvoid, (Any, UInt), a, prod(sz))
  ptr = convert(Ptr{Csize_t},pointer_from_objref(a))
  for i = 1:length(sz)
    unsafe_store!(ptr+8*(i+2), sz[i])
  end
  return a
end

function refill!(a::Array, b::CuArray)
  _resize!(a, size(b))
  copy!(a, b)
end

function cache(cx, x::CuArray{T,N})::Array{T,N} where {T,N}
  cpu = Array{T,N}(undef, ntuple(_->0,N))
  cx[cpu] = x
  return cpu
end

for f in (:+, :-, :*, :/)
  @eval function (c::CUDACtx)(::typeof($f), a::AbstractArray, b::AbstractArray)
    ga = get_cached(array_bank, a)
    gb = get_cached(array_bank, b)
    # cache(array_bank, $f(ga, gb))
    $f(ga, gb)
  end
end

function get_cached(array_bank, arr::Array{T,N})::CuArray{T,N} where {T,N}
  haskey(array_bank, arr) ?
    array_bank[arr] :
    (array_bank[arr] = CuArray(arr))
end

function (c::CUDACtx)(::typeof(broadcasted), f, args...)
  gargs = map(x -> get_cached(array_bank, x), args)
  broadcasted(f, gargs...)
end

function (c::CUDACtx)(::typeof(getproperty), o, s::Symbol)
  getproperty(o, s) |> get_cached
end

function (c::CUDACtx)(::typeof(broadcast), f, args...)
  gargs = map(x -> get_cached(array_bank, x), args)
  broadcast(f, gargs...)
end

function (c::CUDACtx)(::typeof(getfield), o, s::Symbol)
  getfield(o, s) |> get_cached
end

function wrap_cuize(f)
  @eval function (c::CUDACtx)(::typeof($f), args...)
    gargs = map(get_cached, args)
    $f(gargs...) # use `cache`
  end
end

wrap_cuize.((sum, similar, ))

function (c::CUDACtx)(::typeof(reshape), arr, args...)
  reshape(get_cached(arr), args...)
end

@dynamo function (c::CUDACtx)(meta...)
  meta == nothing && return
  ir = IR(meta...)
  ir == nothing && return

  pr = Pipe(ir)
  for (v,st) in pr
    isexpr(st.expr, :call) || continue
    ex = st.expr

    pr[v] = Expr(:call, self, ex.args...)

  end
  return finish(pr)
end

get_cached(array_bank, t::Union{Type, UnitRange, Function, Broadcasted, Symbol, Module, Nothing, Missing, Ptr, CuPtr, T}) where {T <: Real} = t
get_cached(array_bank, t::Union{Tuple, NamedTuple}) = map(get_cached, t)

get_cached(array_bank, x::CuArray) = x

function get_cached(x::T) where T
  T <: Array && return get_cached(array_bank, x)
  isstructtype(T) && return x # get_cached(obs, x)
  get_cached(array_bank, x)
end

"""
  Disable `CUDACtx` for a function
"""
function noop_pass(f)
  @eval (c::CUDACtx)(::typeof($f), args...) = $f(args...)
end

noop_pass.((materialize, get_cached, NNlib.check_spdf,
	))

for f in names(NNlib)
  getfield(NNlib, f) isa Function || continue
  @eval function (c::CUDACtx)(::typeof($f), args...)
    gargs = map(get_cached, args)
    # cache(array_bank, $f(gargs...))
    $f(gargs...)
  end
end

# Hold all the arrays related to the op
# TODO: this should be a context
# BitArray and friends would like an AbstractArray construct
# const array_bank = IdDict{Array,CuArray}()

const array_bank = CUDACtx()

"""
  Creates a `cuda` context within which we travel
  through the entire callstack to find matrix/vector
  operations and try to offload them to a GPU.

  Example:
  ```
  cuda() do
    # do something
  end
  ```
"""
function cuda(f)
  out = array_bank(f)
  for (x, cx) in array_bank
    length(x) == length(cx) && continue
    refill!(x, cx)
  end
  # empty!(array_bank)
  return out
end
