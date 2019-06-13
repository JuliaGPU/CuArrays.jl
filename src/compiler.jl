using IRTools: isexpr, IR, @dynamo
using IRTools: meta, Pipe, finish

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize

# array_bank = IdDict()
__context__ = IdDict()
# __context__ = WeakKeyDict()


for f in (:+, :-, :*, :/)
  q = quote
      function cuize(::typeof($f), a, b)
        a = get_cached(__context__, a)
        # @show typeof(a)
        b = get_cached(__context__, b)

        c = $f(a, b)
        get_cached(__context__, c)
      end
    end
  eval(q)
end

function get_cached(__context___, arr)
  @show "here"
  haskey(__context__, arr) ?
    __context___[arr] :
    cache(__context___, arr)

end

function cache(__context___, arr)
  __context___[arr] = cu(arr)
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
