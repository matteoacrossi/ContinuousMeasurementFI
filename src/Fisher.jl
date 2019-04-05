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
    
    eigval, eigvec = eigen(Hermitian(Matrix(zchop(ρ,1e-10))))
    
    dim = length(eigval)
    res = real(2*sum( [( (eigval[n] + eigval[m] > abstol) ?
             (1. / (eigval[n] + eigval[m])) * abs(eigvec[:,n]' * dρ * eigvec[:,m])^2 
             : 0.) for n=1:dim, m=1:dim]))
    
    return res
end
