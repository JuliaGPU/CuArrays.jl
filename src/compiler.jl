using IRTools: isexpr, IR, Variable, @dynamo, @code_ir, prewalk
using IRTools: xcall, prewalk, postwalk, insertafter!, exprtype
using IRTools: Argument, meta, arg, NewVariable
using IRTools: Pipe, stmt, var, finish
using IRTools
import Adapt, GPUArrays

# [4] TODO: clean iscuable fn

function iscuable(x)
  x isa NewVariable && return false
  x isa Variable && return true
  x isa Argument && return true
  x isa Symbol && return false
  x isa QuoteNode && return false
  x isa GlobalRef && x.mod ∈ (CuArrays, Core, Adapt, GPUArrays) && return false

  x isa Expr && x.args[1] isa GlobalRef && x.args[1].mod ∈ (CuArrays, Core, Adapt, GPUArrays) && return false
  x isa Expr && x.args[1] ∈ (:Base, :CuArrays, :Core, :Adapt)   && return false
  x isa Expr && x.args[1] == :cu && return false
  x isa Expr && length(x.args) == 1 && return false
  x isa Expr && x.args[1] isa Variable && return false
  x isa Expr && x.args[2] isa QuoteNode && return true

  x isa GlobalRef && x.mod ∉ (Base, Core) || return true
  isconst(x.mod, x.name) || return true
  x = getfield(x.mod, x.name)
  !(x isa Type || sizeof(x) == 0)
end


function traverse_and_insert!(ir, v, ex)
  ir[v] = postwalk(ex) do x
    iscuable(x) || return x
    insert!(ir, v, stmt(Expr(:call, GlobalRef(CuArrays, :cu), x)))
  end
end

∉(a,b) = !in(a,b)

@dynamo function cuize(meta)
  meta == nothing && return
  ir = IR(meta)
  ir == nothing && return
  pr = Pipe(ir)
  pr == nothing && return

  for (v,st) in pr
    ex = st.expr
    ex = traverse_and_insert!(pr, v, ex)

    # Comment this to make it work
    # pr[v] = Expr(:call, GlobalRef(Main, :cuize), st.expr.args...)
  end

   return finish(pr)
end



