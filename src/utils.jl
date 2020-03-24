using ZChop
function squeezing_param(N, ΔJ1, J2m, J3m)
    """
        ξ2 = squeezing_param(N, ΔJ1, J2m, J3m)

    Returns the squeezing parameter defined, e.g.,
    in Phys. Rev. A 65, 061801 (2002), Eq. (1).
    """
    return (J2m .^2 + J3m .^2) ./ (N * ΔJ1)
end

"""
    density(M)

Returns the density of a sparse matrix
"""
function density(s::SparseMatrixCSC)
    return length(s.nzval) / (s.n * s.m)
end

function Unconditional_QFI_Dicke(Nj::Int64, Tfinal::Real, dt::Real;
    κ::Real = 1.,                    # Independent noise strength
    κcoll::Real = 1.,                # Collective noise strength
    ω::Real = 0.0                   # Frequency of the Hamiltonian
    )
    return Eff_QFI_HD_Dicke(Nj, 1, Tfinal, dt; κ=κ, κcoll = κcoll, ω=ω, η=0.0)
end

# Specialize zchop! to BlockDiagonal
function ZChop.zchop!(A::BlockDiagonal)
    for b in blocks(A)
         zchop!(b)
     end
 end