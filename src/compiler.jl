using IRTools: isexpr, IR, @dynamo
using IRTools: meta, Pipe, finish, Variable

import Base.Broadcast.broadcasted
import Base.Broadcast.materialize
import Base.Broadcast.Broadcasted


# Hold all the arrays related to the op
# array_bank = WeakKeyDict{AbstractArray, AbstractArray}()
# array_bank = WeakKeyDict()
array_bank = IdDict{AbstractArray, AbstractArray}()
# array_bank = IdDict()

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
  # @show f
  # @show typeof.(args)
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
  gargs = map(x -> get_cached(array_bank, x), args)
  broadcast(f, gargs...)
end

# function cuize(::typeof(reshape), arr::AbstractArray, args...; kwargs...)
#   @show "in reshape"
#   reshape(get_cached(arr), args...; kwargs...)
# end

# cuize(::typeof(reshape), arr::Base.OneTo, args...; kwargs...) = reshape(get_cached(collect(arr)), args...; kwargs...)

# function h!(ir)
#      args = IRTools.arguments(ir)

#      for a in args
#          pushfirst!(ir, :(get_cached($a)))
#          # IRTools.deletearg!(ir, a)
#          # ir[a] = :(get_cached($a))
#      end
#      ir
#  end

@dynamo function cuize(meta...)
  meta == nothing && return
  ir = IR(meta...)
  # @show ir
  ir == nothing && return
  # args = IRTools.arguments(ir)
  # ir = IRTools.postwalk(ir) do x
  #   x in args && return Expr(:call, GlobalRef(Main, :get_cached), x)
  #   return x
  # end

  pr = Pipe(ir)
  for (v,st) in pr
    isexpr(st.expr, :call) || continue
    ex = st.expr

    # ex = IRTools.postwalk(ex) do x
    #   x in args && reutrn :(get_cached(x))
    #   return x
    # end
    # @show ex
    # ex isa Nothing && continue
    # ex = IRTools.postwalk(ex) do x

    #   i = findall(y -> y == x, ex.args)
    #   if length(i) > 0
    #     i = i[1]
    #     temp = Expr(:call, GlobalRef(Main, :get_cached), ex.args[i])
    #     ex.args[i] = GlobalRef(Main, :temp)
    #   end
    #     # temp = Expr(:call, GlobalRef(Main, :get_cached), arg)
    #     # GlobalRef(Main, :temp)
    # end
    # ex = IRTools.postwalk(ex) do x
    #   x isa GlobalRef && x in ex.args && return IRTools.xcall(Main, :get_cached, x)
    #   x
    # end


    pr[v] = Expr(:call, GlobalRef(Main, :cuize), ex.args...)

  end
  return finish(pr)
end


# function cuize(::typeof(reshape), args...)
#   reshape(map(get_cached, args)...)
# end

cuize(::typeof(setindex!), ::Tuple, args...) = tuple(args...)

###################################################################

function children(x::T, fs = fieldnames(T)) where T
  # map(f -> get_cached(getproperty(x, f)), fs) # get_cached -> get_cached(array_bank, getproperty...)
  # q = quote
  #   function cuize(c::$(nameof(T)), args...)
  #     gargs = map(get_cached, args)
  #     g_c = $(nameof(T))(get_cached(c)...)
  #     g_c(gargs...)
  #   end
  # end
  # eval(q)
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

# This allows capturing calls that have at least one argument
# Use case handled is that a lot of calls to `Main.getfield(Symbol("#xx#xx"))`
# like cases also get covered with just `args...` which would be nice to avoid
# CANT DO THIS - OVERLOADS CUIZE AND BREAKS RECURSION
# function cuize(f::Function, xs, args...)
#   gxs = get_cached(xs)
#   f(gxs, map(get_cached, args)...)
# end
# function cuize(o, xs, args...)
#   gxs = get_cached(xs)
#   o(gxs, map(get_cached, args)...)
# end


cuize(::typeof(get_cached), args...) = get_cached(args...)


###################################################

# Adding get_cached calls in dynamo - 
# Before making Pipe - core dumped; julia crashed
# Inside Pipe - Bad compiler errors argextype ones
# Need to find a way to catch arguments
# 

# cuize(a) = a()


# function set_cuize(f::T) where T
#   # isstructtype(typeof(f))
#   @show T.name
#   q = quote
#     function cuize(::$(T.name), args...)
#       gargs = map(get_cached, args)
#       cuize() do
#         ($f::$(T.name))(gargs...)
#       end
#     end
#   end
#   @show q
#   # eval(q)
# end

# function modifyex(ex)
#   @assert IRTools.isexpr(ex, :call)
#   args = ex.args
#   fs = Symbol[args[1]]
#   for a in args[2:end]
#     if a isa Expr
#       @assert IRTools.isexpr(a, :call)
#       push!(fs, a.args[1])
#     end
#   end
#   mex = IRTools.postwalk(ex) do x
#     x isa Expr && return x
#     if x in fs
#       return x
#     else
#       return :(get_cached($x))
#     end
#   end
#   mex
# end

# macro cuize(ex)
#            mex = modifyex(ex)
#            return :(cuize() do
#                $mex
#            end)
#        end



function makechildren(T::Type, nt::NamedTuple)
  eval(nameof(T))(nt...)
end
# function cuize(f::Function, xs, args...)
#            # @show f
#            gxs = get_cached(xs)
#            gargs = map(get_cached, args)
#            f(gxs, gargs...)
#        end


# Functions called inside `cuize` aren't executed as part of the context
# So any assumptions made inside the context (`getproperty`, for eg) will
# not Hold
# Thus we need the actual objects when trying to call the objects, as opposed to
# continuing inside the context where we can pick fields up from a NamedTuple
# Without this limitation, we can avoid caching the structs themselves
function cuize(::typeof(invoke), f::T, types, args...) where T
                  @timeit to "bad stuff" gf = f isa Function ? f : makechildren(T, get_cached(obs, f))
                  invoke(gf, types, map(get_cached, args)...)
              end



# Same problem as recursing; works for basic stuff
# if objects are cached as objects
# function cuize(f::Function, xs, args...)
#            # @show f
#            gxs = get_cached(xs)
#            gargs = map(get_cached, args)
#            f(gxs, gargs...)
#        end




# function get_cached(__context__::Function, xs)
#   __context__() do
#     xs
#   end
# end