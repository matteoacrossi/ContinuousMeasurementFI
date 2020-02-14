import scipy.sparse as sp
import numpy as np
import qutip

def sparse_to_ijv(sparsematrix):
    # If a Qobj, get the sparse representation
    if isinstance(sparsematrix, qutip.Qobj):
        sparsematrix = sparsematrix.data

    # Get the shape of the sparse matrix
    (m, n) = sparsematrix.shape

    # Get the row and column incides of the nonzero elements
    I, J = sparsematrix.nonzero()

    # Get the vector of values
    V = np.array([sparsematrix[i, j] for (i, j) in zip(I, J)])

    # Convert to 1-based indexing
    I += 1
    J += 1
    return (I, J, V, m, n)