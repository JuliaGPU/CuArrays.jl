using IRTools: isexpr, IR, @dynamo
using IRTools: meta, Pipe, finish

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize
import Base.Broadcast.Broadcasted


# Hold all the arrays related to the op
# array_bank = WeakKeyDict{AbstractArray, AbstractArray}()
# array_bank = WeakKeyDict()
array_bank = IdDict{AbstractArray, AbstractArray}()

# Hold all the results related to the op, but permanently
__context__ = IdDict()

for f in (:+, :-, :*, :/)
  q = quote
      function cuize(::typeof($f), a::AbstractArray, b::AbstractArray)
          ga = get_cached(array_bank, a)
          gb = get_cached(array_bank, b)
          c = $f(ga, gb)
      end
    end
  eval(q)
end

function get_cached(array_bank, arr::AbstractArray)
  # CuArrays can come up when you have outputs/ movements before ops
  arr isa CuArray && return arr
  arr isa TrackedArray && arr.data isa CuArray && return arr

  haskey(array_bank, arr) ?
    array_bank[arr] :
    cache(array_bank, arr)
end

function cache(array_bank::IdDict{T,T}, arr::AbstractArray) where T <: AbstractArray
  array_bank[arr] = cu(arr)
end

cuize(::typeof(materialize), bc) = materialize(bc)

function cuize(::typeof(broadcasted), f, args...)
  gargs = map(x -> get_cached(array_bank, x), args)
  broadcasted(f, gargs...)
end

function cuize(::typeof(broadcast), f, args...)
  gargs = map(x -> get_cached(array_bank, x), args)
  broadcast(f, gargs...)
end

@dynamo function cuize(meta...)
  meta == nothing && return
  ir = IR(meta...)
  ir == nothing && return
  pr = Pipe(ir)
  for (v,st) in pr
    isexpr(st.expr, :call) || continue
    ex = st.expr
    pr[v] = Expr(:call, GlobalRef(Main, :cuize), st.expr.args...)
  end
  return finish(pr)
end

###################################################################

# Makes things work, but breaks continuity
# Gets called after every line of IR, make it stop
function cuize(f::T, arg1, args...) where T 
  q = quote
    function cuize(::typeof($f), arg1, args...)
      garg1 = get_cached(arg1)
      gargs = map(get_cached, args)
      gf = get_cached($f)
      c = gf(garg1, gargs...)
    end
  end
  eval(q)
  c = invoke(cuize, Tuple{typeof(f), typeof(arg1), typeof.(args)...}, (f, arg1, args...))
end

function children(x::T, fs = fieldnames(T)) where T
  map(f -> get_cached(array_bank, getproperty(x, f)), fs)
end

mapchildren(x::T) where T = @eval $(Symbol(T.name))($(children(x))...)

function get_cached(__context__, x::T) where T
  haskey(__context__, x) && return __context__[x]
  x isa CuArray && return x
  x isa TrackedArray && x.data isa CuArray && return x

  __context__[x] = mapchildren(x)
end

get_cached(array_bank, t::Union{Type, Function, Broadcasted, T}) where {T <: Real} = t
get_cached(array_bank, t::Union{Tuple,NamedTuple}) = map(get_cached, t)

function get_cached(x::T) where T
  T <: AbstractArray && return get_cached(array_bank, x)
  isstructtype(T) && return get_cached(__context__, x)
  get_cached(array_bank, x)
end
