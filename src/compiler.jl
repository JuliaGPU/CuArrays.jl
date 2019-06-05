using IRTools: isexpr, IR, @dynamo
using IRTools: meta, Pipe, finish

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize

# using `Array` instead of `cpu` here works but
# causes tracking issues with TrackedArray
cuize(::typeof(*), a, b) = cpu(cu(a) * cu(b))
cuize(::typeof(+), a, b) = cpu(cu(a) + cu(b))
cuize(::typeof(-), a, b) = cpu(cu(a) - cu(b))
cuize(::typeof(/), a, b) = cpu(cu(a) / cu(b))
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
