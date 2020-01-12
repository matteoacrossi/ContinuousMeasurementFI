using SparseArrays

"""
    tosparse(obj) -> SparseMatrixCSC

Returns the sparse matrix corresponding to the PyObject obj.
obj can be a scipy.sparse matrix or a qutip Qobj
"""
function tosparse(obj::PyObject)
    # This is just a wrapper to the ugly py"..." syntax
    (I, J, V, m, n) = pystuff.sparse_to_ijv(obj)
    return sparse(I, J, V, m, n)
end

"""
    liouvillian(N) -> SparseMatrixCSC

Return a dephasing Liouvillian for N spins in the Dicke basis
"""
function liouvillian(N::T) where T<:Integer
    sys = piqs.Dicke(N)
    sys.dephasing = 4.

    return tosparse(sys.liouvillian())
end

"""
    css(N) -> SparseMatrixCSC

Return a coherent spin state in the Dicke basis for N spins
"""
function css(N::T) where T<:Integer
    return tosparse(piqs.css(N))
end

"""
    jspin(N) -> Tuple{SparseMatrixCSC}

Return the tuple (Jx, Jy, Jz) of the spin operators in the
Dicke basis. The returned operators are sparse arrays.
"""
function jspin(N::T) where T<:Integer
    return tosparse.(piqs.jspin(N))
end

"""
    j_min(N)

Return the minimum value of j for N spins
"""
function j_min(N::T) where T<:Integer
    if N % 2 == 0
        return 0
    else
        return 0.5
    end
end

"""
    j_vals(N)

Returns a list of values of j in decreasing order
"""
function j_vals(N::T) where T<:Integer
    j = (N/2):-1:j_min(N)
    return j
end

"""
    block_sizes(N)

Return a list of block sizes for the density matrix of a N-spin
system in the Dicke basis.
"""
function block_sizes(N::T) where T<:Integer
   return Int.(2*j_vals(N) .+ 1)
end
