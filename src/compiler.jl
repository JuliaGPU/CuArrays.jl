using IRTools: isexpr, IR, @dynamo
using IRTools: meta, Pipe, finish, Variable

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize
import Base.Broadcast.Broadcasted


# Hold all the arrays related to the op
# array_bank = WeakKeyDict{AbstractArray, AbstractArray}()
# array_bank = WeakKeyDict()
array_bank = IdDict{AbstractArray, AbstractArray}()

# Hold all the objects related to the op
obs = IdDict()

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
# function cache(array_bank::IdDict, s::K) where K <: AbstractSet
#   t = K()
#   for p in s
#     push!(t, get_cached(p))
#   end
#   t
# end

function cuize(::typeof(broadcasted), f, args...)
  gargs = map(x -> get_cached(array_bank, x), args)
  Main.broadcasted(f, gargs...)
end

function cuize(::typeof(getproperty), o, s::Symbol)
  getproperty(o, s) |> get_cached
end

# function cuize(::typeof(getproperty), o, s::Symbol)
#   getproperty(get_cached(o), s)
# end

function cuize(::typeof(broadcast), f, args...)
  @show f
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

    pr[v] = Expr(:call, GlobalRef(Main, :cuize), ex.args...)

  end
  return finish(pr)
end

cuize(::typeof(setindex!), ::Tuple, args...) = tuple(args...)

###################################################################

function children(x::T, fs = fieldnames(T)) where T
  (; zip(fs, map(f -> get_cached(getproperty(x, f)), fs))...)
end

children(x::Tuple) = map(get_cached, x)
# function children(s::T) where T <:AbstractSet
#   t = T()
#   for p in s
#     push!(t, get_cached(p))
#   end
#   t
# end


# mapchildren(x::T) where T = @eval $(Symbol(T.name))($(children(x))...)
mapchildren(x::T) where T = children(x)
# mapchildren(x::T) where T<:AbstractSet = children(x)

function get_cached(obs, x::T) where T
  haskey(obs, x) && return obs[x]
  x isa CuArray && return x
  x isa TrackedArray && x.data isa CuArray && return x

  obs[x] = mapchildren(x)
end

get_cached(array_bank, t::Union{Type, Function, Broadcasted, Symbol, Module, Nothing, Missing, Ptr, T}) where {T <: Real} = t
get_cached(array_bank, t::Union{Tuple, NamedTuple}) = map(get_cached, t)

function get_cached(x::T) where T
  T <: AbstractArray && return get_cached(array_bank, x)
  isstructtype(T) && return x # get_cached(obs, x)
  get_cached(array_bank, x)
end

"""
  Disable `cuize` for a function
"""
function noop_pass(f, args...)
  @eval cuize(::typeof($f), args...) = $f(args...)
end

noop_pass.((materialize, ))

cuize(::typeof(get_cached), args...) = get_cached(args...)

function makechildren(T::Type, nt::NamedTuple)
  eval(nameof(T))(nt...)
end


# Functions called inside `cuize` aren't executed as part of the context
# So any assumptions made inside the context (`getproperty`, for eg) will
# not Hold
# Thus we need the actual objects when trying to call the objects, as opposed to
# continuing inside the context where we can pick fields up from a NamedTuple
# Without this limitation, we can avoid caching the structs themselves
function cuize(::typeof(invoke), f::T, types, args...) where T
  gf = f isa Function ? f : makechildren(T, get_cached(obs, f))
  invoke(gf, types, map(get_cached, args)...)
end