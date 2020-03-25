#=
Functions for constructing noise operators
=#
using SparseArrays
using LinearAlgebra
import SparseArrays.getcolptr
using BlockDiagonalMatrices

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
function sup_pre(A)
    return blkdiag(A, size(A, 1))
end


"""
    sup_post(A)

    Superoperator formed from post-multiplication by operator A†.

    Effectively evaluate the Kronecker product A* ⊗ I
"""
function sup_post(A)
    return kron(conj(A), I + zero(A))
end

"""
    sup_pre_post(A, B)

    Superoperator formed from A * . * B†

    Effectively evaluate the Kronecker product B* ⊗ A
"""
function sup_pre_post(A, B)
    return kron(conj(B), A)
end

"""
    sup_pre_post(A)

    Superoperator formed from A * . * A†

    Effectively evaluate the Kronecker product A* ⊗ A
"""
function sup_pre_post(A)
    return kron(conj(A), A)
end

function blkdiag(X::SparseMatrixCSC{Tv, Ti}, num) where {Tv, Ti<:Integer}
    mX = size(X, 1)
    nX = size(X, 2)
    m = num * size(X, 1)
    n = num * size(X, 2)

    nnzX = nnz(X)
    nnz_res = nnzX * num
    colptr = Vector{Ti}(undef, n+1)
    rowval = Vector{Ti}(undef, nnz_res)
    nzval = repeat(X.nzval, num)

    @inbounds @simd for i = 1 : num
         @simd for j = 1 : nX + 1
            colptr[(i - 1) * nX + j] = X.colptr[j] + (i-1) * nnzX
        end
         @simd for j = 1 : nnzX
            rowval[(i - 1) * nnzX + j] = X.rowval[j] + (i - 1) * (mX)
        end
    end
    colptr[n+1] = num * nnzX + 1
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

# function blkdiag!(Sup::SparseMatrixCSC{Tv, Ti}, X::SparseMatrixCSC{Tv, Ti}, num) where {Tv, Ti<:Integer}
#     copy!(Sup.nzval, repeat(X.nzval, num))
#     return Sup
# end

"""
Non-allocating update of the matrix Sup = I ⊗ A.

ATTENTION!!! It assumes the position of the non-zero elements
does not change!
USE WITH CARE!!!!!
"""
function fast_sup_pre!(Sup::SparseMatrixCSC{Tv, Ti}, A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti<:Integer}
    num = size(A, 1)
    nnzX = nnz(A)
    @inbounds @simd for i = 1 : num
         @simd for j = 1 : nnzX
            Sup.nzval[(i - 1) * nnzX + j] = A.nzval[j]
        end
    end
end

"""
Non-allocating update of the matrix Sup = A' ⊗ I.

ATTENTION!!! It assumes the position of the non-zero elements
does not change!
USE WITH CARE!!!!!
"""
function fast_sup_post!(Sup::SparseMatrixCSC{T1, S1}, A::SparseMatrixCSC{T1,S1}) where {T1, S1}
    n = size(A, 1)
    col = 1

    @inbounds for j = 1 : n
        startA = getcolptr(A)[j]
        stopA = getcolptr(A)[j+1] - 1
        lA = stopA - startA + 1
        for i = 1:n
            ptr_range = Sup.colptr[col]
            col += 1
            for ptrA = startA : stopA
                Sup.nzval[ptr_range] = nonzeros(A)[ptrA]'
                ptr_range += 1
            end
        end
    end
    return Sup
end


struct BlockIndices{T}
    b::Array{T, 1}
    i::Array{T, 1}
    j::Array{T, 1}
end


"""
Defines a superoperator in terms of its action on a square density matrix.

A superoperator maps density matrices into density matrices.
"""
struct SuperOperator{Tv, Ti <: Integer}
    size::Tuple{Int64, Int64}
    rowind::BlockIndices{Ti}
    colind::BlockIndices{Ti}
    values::Array{Tv}

    """
        SuperOperator(A)

    Construct the Superoperator acting on a N-dimensional Hilbert space from a 2N × 2N
    sparse superoperator matrix.
    """
    function SuperOperator{Tv, Ti}(A::SparseMatrixCSC{Tv, Ti}) where Tv where Ti <: Integer
        N = Int(sqrt(size(A, 1)))
        row, col, val = findnz(A)
        rowind = _block_ij(row, N)
        colind = _block_ij(col, N)

        new((N,N), rowind, colind, val)
    end
end

SuperOperator(A::SparseMatrixCSC{Tv, Ti}) where Ti <: Integer where Tv = SuperOperator{Tv, Ti}(A)


"""
    apply_superop!(C, A, B)

Apply the `SuperOperator` `A` to `BlockDiagonal` matrix `B` and stores the result in `C`
"""
function apply_superop!(C::BlockDiagonal, A::SuperOperator{Tv, Ti}, B::BlockDiagonal) where Ti <: Integer where Tv
    for b in blocks(C)
        fill!(b, zero(eltype(b)))
    end
    val::Array{Tv, 1} = A.values

    @simd for i = 1 : length(val)
        if A.colind.b[i] > 0
            tmp = val[i] * B.blocks[A.colind.b[i]][A.colind.i[i], A.colind.j[i]]
            if tmp != 0.0
                C.blocks[A.rowind.b[i]][A.rowind.i[i], A.rowind.j[i]] += tmp
            end
        end
    end
    return C
end


"""
    bi, newi, newj = _block_ij(indices, N)

Given the `indices` of the row or column of a superoperator of size `N`, transforms them to block indices
for the action to a density matrix.
"""
function _block_ij(indices, N)
    Nj = nspins(N)
    bs = block_sizes(Nj)

    # We add a leading 1 and a trailing element larger than the
    # total size to avoid out-of-bounds problems
    blockindices = vcat(cumsum(vcat(1, bs[1:end-1])), N+1)

    # Tranform from superoperator index to matrix i,j indices
    newi = (indices .- 1) .% N .+ 1
    newj = (indices .- 1) .÷ N .+ 1

    # Store the output block indices
    bi = similar(newi)

    for i in eachindex(newi)
        # Find the index of the block corresponding to the i index
        tmp = searchsortedfirst(blockindices, newi[i], lt=<=) - 1

        # Check that the j index corresponds to the same block
        if blockindices[tmp] <= newj[i] < blockindices[tmp + 1]
            bi[i] = tmp
            newi[i] -= blockindices[bi[i]] - 1
            newj[i] -= blockindices[bi[i]] - 1
        else # Otherwise, ignore it (I don't remember why)
            bi[i] = 0
            newi[i] = 0
            newj[i] = 0
        end
    end

    return BlockIndices{Int64}(bi, newi, newj)
end



""" nspins(size)

Obtain the number of spins from the size of a matrix in the
Dicke basis
"""
function nspins(size)
    if sqrt(size) == floor(sqrt(size)) # Nspin even
         return 2 * (Int(sqrt(size)) - 1)
     else # Nspin odd
         return -2 + Int(sqrt(1 + 4 * size))
     end
 end

""" blockdiagonal(M; [dense=false])

Converts a matrix into a BlockDiagonal matrix, by default preserving the matrix type (e.g. sparse, dense).
If dense=true, force the blocks to have a dense structure
"""
function blockdiagonal(M::T; dense=false) where T <: AbstractArray
    N = nspins(size(M, 1))

    blocksizes = block_sizes(N)

    views = Array{T}(undef, length(blocksizes))
    startidx = 1
    for i in 1:length(blocksizes)
        range = startidx:(startidx + blocksizes[i] - 1)
        views[i] = view(M, range, range)
        startidx += blocksizes[i]
    end
    if dense # Force dense blocks
        return BlockDiagonal(Matrix.(views))
    else
        return BlockDiagonal(views)
    end
end