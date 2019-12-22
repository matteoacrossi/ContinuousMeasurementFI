#=
Functions for constructing noise operators
=#
using SparseArrays
using LinearAlgebra
using LinearMaps
"""
    σ_j(j, n, direction)

Return a sparse matrix representing the Pauli operator on the ``j``-th of ``n`` spins in the computational basis.

`direction` must be `(:x, :y, :z)`
"""
function σ_j(direction::Symbol, j::Int, n::Int)
    @assert direction ∈ (:x, :y, :z) "Direction must be :x, :y, :z"
    @assert j <= n "j must be less or equal than n"

    sigma = Dict(:x => sparse([0im 1.; 1. 0.] ),
             :y => sparse([0. -1im; 1im 0.]),
             :z => sparse([1. 0im; 0. -1.]))
    if n == 1
        return sigma[direction]
    else
        return kron(
            vcat([SparseMatrixCSC{ComplexF64}(I, 2, 2) for i = j + 1:n],
                [sigma[direction]],
                [SparseMatrixCSC{ComplexF64}(I, 2, 2) for i = 1:j-1])...)
    end
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

"""
    trace(A)

Return the trace of a vectorized operator A
"""
function trace(A::AbstractArray{T,1}) where T

    N = Int(sqrt(length(A)))
    return tr(reshape(A, (N,N)))
end

"""
    sup_pre(A)

    Superoperator formed from pre-multiplication by operator A.

    Effectively evaluate the Kronecker product I ⊗ A
"""
function sup_pre(A::LinearMap)
    unit = one(eltype(A))
    return kron(LinearMaps.UniformScalingMap(unit, size(A)), A)
end

"""
    sup_post(A)

    Superoperator formed from post-multiplication by operator A.

    Effectively evaluate the Kronecker product A.T ⊗ I
"""
function sup_post(A::LinearMap)
    unit = one(eltype(A))
    return kron(transpose(A), LinearMaps.UniformScalingMap(unit, size(A)))
end

"""
    sup_pre_post(A, B)

    Superoperator formed from A * . * B

    Effectively evaluate the Kronecker product B* ⊗ A
"""
function sup_pre_post(A::LinearMap, B::LinearMap)
    return kron(transpose(B), A)
end

"""
    sup_pre_post(A)

    Superoperator formed from A * . * A†

    Effectively evaluate the Kronecker product A* ⊗ A
"""
function sup_pre_post(A::LinearMap)
    return kron(transpose(adjoint(A)), A)
end
