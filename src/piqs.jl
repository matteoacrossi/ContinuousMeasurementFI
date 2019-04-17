using SparseArrays

"""
    tosparse(obj)

Returns the sparse matrix corresponding to the PyObject obj. 
obj can be a scipy.sparse matrix or a qutip Qobj 
"""
function tosparse(obj::PyObject)
    # This is just a wrapper to the ugly py"..." syntax
    (I, J, V, m, n) = py"sparse_to_ijv"(obj)
    return sparse(I, J, V, m, n)
end

function css(N)
    return tosparse(piqs.css(N))
end
