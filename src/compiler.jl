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

function get_cached(array_bank::IdDict{T,T}, arr::AbstractArray) where T <: AbstractArray
  # CuArrays can come up when you have outputs/ movements before ops
  arr isa CuArray && return arr
  arr isa TrackedArray && arr.data isa CuArray && return arr

  haskey(array_bank, arr) ?
    array_bank[arr] :
    cache(array_bank, arr)
end

function cache(array_bank::IdDict{T,T}, arr::AbstractArray{<:Real}) where T <: AbstractArray

  array_bank[arr] = cu(arr)
end

cache(array_bank::IdDict{T,T}, arr::AbstractArray) where T <: AbstractArray = map(get_cached, arr)
# function cache(array_bank::IdDict, s::K) where K <: AbstractSet}
#   t = K()
#   for p in s
#     push!(t, get_cached(p))
#   end
#   t
# end

function cuable(st)
  args = st.expr.args

  # @show typeof.(args)
  # @show tellmethetype.(args)
  flag = true
  for x in args
    if x isa GlobalRef && x.mod == NNlib && x.name == :DenseConvDims
      flag = false
      break
    end
  end
  # map(x -> x isa GlobalRef ? @show x.name : x, args)
  flag
end

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
    # if cuable(st)
      pr[v] = Expr(:call, GlobalRef(Main, :cuize), st.expr.args...)
    # else
    #   @show "trying to write wrapper"
    #   pr[v] = Expr(:call, GlobalRef(Main, :fcuize), st.expr.args...)
    #   @show "written wrapper"
    # end
  end
  return finish(pr)
end

cuize(::typeof(setindex!), ::Tuple, args...) = tuple(args...)

###################################################################

# Makes things work, but breaks continuity and recursion
# Gets called after every line of IR, make it stop
# Basically a horrible hack around getting constructors to work
# function cuize(f, arg1, args...)
#   # @show f
#   # if Symbol(f) == :DenseConvDims

#     q = quote
#       function cuize(::typeof($f), arg1, args...)
#         garg1 = get_cached(arg1)
#         gargs = map(get_cached, args)
#         gf = get_cached($f)
#         c = gf(garg1, gargs...)
#       end
#     end
#     eval(q)
#   # end

#   c = invoke(cuize, Tuple{typeof(f), typeof(arg1), typeof.(args)...}, (f, arg1, args...))
# end

function children(x::T, fs = fieldnames(T)) where T
  map(f -> get_cached(getproperty(x, f)), fs) # get_cached -> get_cached(array_bank, getproperty...)
end

children(x::Tuple) = map(get_cached, x)
# function children(s::T) where T <:AbstractSet
#   t = T()
#   for p in s
#     push!(t, get_cached(p))
#   end
#   t
# end


mapchildren(x::T) where T = @eval $(Symbol(T.name))($(children(x))...)
# mapchildren(x::T) where T<:AbstractSet = children(x)

# function get_cached(__context__, x::T) where T
#   haskey(__context__, x) && return __context__[x]
#   x isa CuArray && return x
#   x isa TrackedArray && x.data isa CuArray && return x

#   __context__[x] = mapchildren(x)
# end

get_cached(array_bank, t::Union{Type, Function, Broadcasted, Symbol, T}) where {T <: Real} = t
get_cached(array_bank, t::Union{Tuple, NamedTuple}) = map(get_cached, t)

function get_cached(x::T) where T
  # T <: AbstractArray && return get_cached(array_bank, x)
  # isstructtype(T) && return get_cached(__context__, x)
  get_cached(array_bank, x)
end

function noop_pass(f, args...)
  @eval cuize(::typeof($f), args...) = $f(args...)
end

noop_pass.((getproperty, materialize, ))

@generated function tellmethetype(x)
  x
end
