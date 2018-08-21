export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUDNNError) = print(io, "CUDNNError(code $(err.code), $(err.msg))")

function CUDNNError(status::cudnnStatus_t)
    msg = unsafe_string(cudnnGetErrorString(status))
    return CUDNNError(status, msg)
end

macro check(handler::Expr, ex::Expr)
    quote
        local status::cudnnStatus_t
        status = $(esc(ex))
        if status != CUDNN_STATUS_SUCCESS
            $(esc(handler))(status)
        end
        nothing
    end
end

macro check(ex::Expr)
    quote
        @check($(esc(ex))) do status
            throw(CUDNNError(status))
        end
    end
end
