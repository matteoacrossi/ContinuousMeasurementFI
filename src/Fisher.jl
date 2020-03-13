using ZChop

"""
    QFI(ρ, dρ [, abstol])

Numerically evaluate the quantum Fisher information for the matrix ρ given its derivative dρ wrt the parameter

This function is the implementation of Eq. (13) in Paris, Int. J. Quantum Inform. 7, 125 (2009).

# Arguments
    * `ρ`:  Density matrix
    * `dhro`: Derivative wrt the parameter to be estimated
    * `abstol = 1e-5`: tolerance in the denominator of the formula
"""
function QFI(ρ, dρ; abstol = 1e-5)

    # Get the eigenvalues and eigenvectors of the density matrix
    # We enforce its Hermiticity so that the algorithm is more efficient and returns real values

    eigval, eigvec = eigen(Hermitian(Matrix(zchop(ρ, 1e-10))))

    dim = length(eigval)
    tmpvec = Array{eltype(ρ)}(undef, size(ρ, 1))
    res = 0.
    tmp = 0.
    for m = 1:dim
        mul!(tmpvec, dρ, view(eigvec, :, m))
        for n = 1:dim
            tmp = eigval[n] + eigval[m]
            if tmp > abstol
                res += 2 * (1. / tmp) * abs2(dot(view(eigvec, :, n), tmpvec))
            end
        end
    end
    return res
end

function QFI!(ρ, dρ; abstol = 1e-5)
    # Get the eigenvalues and eigenvectors of the density matrix
    # We enforce its Hermiticity so that the algorithm is more efficient and returns real values
    eigval, eigvec = eigen!(Hermitian(ρ))
    dim = length(eigval)
    tmpvec = Array{eltype(ρ)}(undef, size(ρ, 1))
    res = 0.
    tmp = 0.
    for m = 1:dim
        mul!(tmpvec, dρ, view(eigvec, :, m))
        for n = 1:dim
            tmp = eigval[n] + eigval[m]
            if tmp > abstol
                res += 2 * (1. / tmp) * abs2(dot(view(eigvec, :, n), tmpvec))
            end
        end
    end
    return res
end

"""
    QFI(ρ::BlockDiagonal, dρ::BlockDiagonal, N [, abstol])

Numerically evaluate the quantum Fisher information for the matrix ρ given its derivative dρ wrt the parameter,
for the special case of a block-diagonal matrix in the Dicke basis.

This function is the implementation of Eq. (13) in Paris, Int. J. Quantum Inform. 7, 125 (2009).

# Arguments
    * `ρ`:  Density matrix
    * `dhro`: Derivative wrt the parameter to be estimated
    * `N`: number of spins
    * `abstol = 1e-5`: tolerance in the denominator of the formula
"""
function QFI(ρ::BlockDiagonal, dρ::BlockDiagonal; abstol = 1e-5)
    qfi = 0.
    for i in 1:nblocks(ρ)
        qfi += QFI(ρ.blocks[i], dρ.blocks[i])
    end
    return qfi
end

function QFI!(ρ::BlockDiagonal, dρ::BlockDiagonal; abstol = 1e-5)
    qfi = 0.
    for i in 1:nblocks(ρ)
        qfi += QFI!(ρ.blocks[i], dρ.blocks[i])
    end
    return qfi
end

function QFI(state::State, abstol = 1e-5)
    QFI(state.ρ, state.dρ)
end

function QFI!(state::State, abstol = 1e-5)
    copy!(state._tmp1, state.ρ)
    QFI!(state._tmp1, state.dρ)
end