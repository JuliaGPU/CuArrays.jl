using IRTools: isexpr, IR, @dynamo
using IRTools: meta, Pipe, finish

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize


# Hold all the arrays related to the op
# array_bank = WeakKeyDict{AbstractArray, AbstractArray}()
# array_bank = WeakKeyDict()
array_bank = IdDict()

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


for f in (:+, :-, :*, :/)
  q = quote
      function cuize(::typeof($f), a, b)
        # @show length(a)
        # ga = get_cached(array_bank, a)
        # gb = get_cached(array_bank, b)

        # c = $f(ga, gb)
        # __context__[c] = c
        # if haskey(__context__, ($f, a, b))
          # __context__[($f, a, b)]
        # else
          ga = get_cached(array_bank, a)
          gb = get_cached(array_bank, b)
          c = $f(ga, gb)
          # __context__[($f, a, b)] = c
        # end
      end
    end
  eval(q)
end

function get_cached(array_bank, arr)
  haskey(array_bank, arr) ?
    array_bank[arr] :
    cache(array_bank, arr)

end

function cache(array_bank, arr)
  array_bank[arr] = cu(arr)
end

# using `Array` instead of `cpu` here works but
# causes tracking issues with TrackedArray
# cuize(::typeof(*), a, b) = cpu(cu(a) * cu(b))
# cuize(::typeof(+), a, b) = cpu(cu(a) + cu(b))
# cuize(::typeof(-), a, b) = cpu(cu(a) - cu(b))
# cuize(::typeof(/), a, b) = cpu(cu(a) / cu(b))
cuize(::typeof(materialize), bc) = materialize(bc)

function cuize(::typeof(broadcasted), f, a...)
  b = map(cu, a)
  broadcasted(f, b...)
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
