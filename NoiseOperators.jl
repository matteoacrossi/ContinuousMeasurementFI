#=
Functions for constructing noise operators
=#
using QuantumOptics
import Base.LinAlg.trace

"""
    σ_j(j, n, direction)

Return a sparse matrix representing the Pauli operator on the ``j``-th of ``n`` spins in the computational basis.

`direction` must be `(:x, :y, :z)`
"""
function σ_j(direction::Symbol, j::Int, n::Int)
    if direction ∉ (:x, :y, :z)
        throw(ArgumentError("Direction must be :x, :y, :z"))
    end

    if j > n
        throw(ArgumentError("j must be less or equal than n"))
    end

    b = SpinBasis(1//2)
    operator = eval(Symbol("sigma" * String(direction)))

    return tensor(
            vcat([identityoperator(b) for i = 1:j-1],
                [operator(b)],
                [identityoperator(b) for i = j + 1:n])...).data
end

"""
    σ(direction, n)

Return a sparse matrix representing the collective noise operator
``\\σ_d = \\sum \\σ^{(d)}_j`` where d is the
`direction` and must be one of `(:x, :y, :z)`

# Examples
```jldoctest
julia> full(σ(:x, 1))
2×2 Array{Complex{Float64},2}:
 0.0+0.0im  0.5+0.0im
 0.5+0.0im  0.0+0.0im
```
"""
function σ(direction::Symbol, n::Int)
    σ = spzeros(2^n, 2^n)
    for i = 1:n
        σ += σ_j(direction, i, n)
    end
    return σ
end

function sup_pre(A)
    return kron(eye(A), A)
end

function sup_post(A)
    return kron(transpose(A), eye(A))
end

function trace(A::AbstractArray{T,1}) where T
    N = Int(sqrt(length(A)))
    return trace(reshape(A, (N,N)))
end
