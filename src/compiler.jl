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

# function cache(__context__, f, args...)
#   q = quote
#     function cuize(::typeof($f), args...)
#       if haskey(__context__, ($f, args...))
#         __context__[($f, args...)]
#       else
#         gargs = map(x -> get_cached(array_bank, x), args)
#         c = $f(gargs...)
#         __context__[($f, args...)] = c
#         end
#       end
#   end
#   eval(q)
# end

# function get_cached end

# function cuize(f, args...)
#   q = quote
#     function cuize(::typeof($f), args...)
#       gargs = map(x -> get_cached(array_bank, x), args)
#       c = $f(gargs...)
#     end
#   end
#   eval(q)
#   # cuize(f, args...)
# end

for f in (:+, :-, :*, :/)
  q = quote
      function cuize(::typeof($f), a::AbstractArray, b::AbstractArray)
        # @show length(a)
        # ga = get_cached(array_bank, a)
        # gb = get_cached(array_bank, b)

        # c = $f(ga, gb)
        # __context__[c] = c
        # if haskey(__context__, ($f, a, b))
          # __context__[($f, a, b)]
        # else
        # @timeit to "cuize" begin
          ga = get_cached(array_bank, a)
          gb = get_cached(array_bank, b)
          c = $f(ga, gb)
          # __context__[($f, a, b)] = c
        # end
      end
    end
  eval(q)
end

function get_cached(array_bank, arr::AbstractArray)
  # CuArrays can come up when you have outputs/ movements before ops
  arr isa CuArray && return arr

  # Broadcasted objects are new everytime they're generated, ignore them
  arr isa Broadcasted && return arr

  arr isa TrackedArray && arr.data isa CuArray && return arr

  haskey(array_bank, arr) ?
    array_bank[arr] :
    cache(array_bank, arr)
end

# get_cached(x::AbstractArray) = get_cached(array_bank, parent(x))

# get_cached(array_bank, arr::TrackedArray) = get_cached(array_bank, Tracker.data(arr))
# get_cached(array_bank, arr::TrackedArray) = get_cached(Tracker.data(arr))

function cache(array_bank, arr::AbstractArray)
  array_bank[arr] = cu(arr)
end

# using `Array` instead of `cpu` here works but
# causes tracking issues with TrackedArray
# cuize(::typeof(*), a, b) = cpu(cu(a) * cu(b))
# cuize(::typeof(+), a, b) = cpu(cu(a) + cu(b))
# cuize(::typeof(-), a, b) = cpu(cu(a) - cu(b))
# cuize(::typeof(/), a, b) = cpu(cu(a) / cu(b))
cuize(::typeof(materialize), bc) = materialize(bc)

# function cuize(::typeof(getproperty), args...)
#   # @show args
#   getproperty(args...)
# end

# function cuize(::typeof(getfield), args...)
#   # @info "in getfield: $(typeof.(args))"
#   getfield(args...)
#   # try
#   #   getfield(args...)
#   # catch e
#   #   @show typeof.(args)
#   #   throw()
#   # end
# end

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

# function cuize(f, args...)
#   T = Tuple{typeof(f), typeof.(args)...}
#   q = quote
#     function cuize(::typeof($f), args...)
#       # @show typeof.(args)
#       gargs = map(x -> get_cached(array_bank, x), args)
#       # @show typeof.(gargs)
#       # @timeit to "gf" gf = get_cached(array_bank, $f)
#       gf = get_cached(array_bank, $f)
#       # @show typeof($f)
#       c = gf(gargs...)
#     end
#   end
#   eval(q)

#   c = invoke(cuize, Tuple{typeof(f), typeof.(args)...}, (f, args...))
# end

function children(x::T, fs = fieldnames(T)) where T
  map(f -> get_cached(array_bank, getproperty(x, f)), fs)
end

mapchildren(x::T) where T = @eval $(Symbol(T.name))($(children(x))...)

function get_cached(array_bank, x)

  x isa Broadcasted && return x
  x isa Type && return x
  x isa Function && return x

  haskey(__context__, x) && return __context__[x]

  __context__[x] = mapchildren(x)
end

get_cached(array_bank, t::Tuple) = t
