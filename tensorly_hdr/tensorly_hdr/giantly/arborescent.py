import numpy as np
from juliacall import Main as jl
import pathlib

path = str(pathlib.Path(__file__).parent.resolve())
print(path)

# Load the arborescent.jl file, will be run upon importing
jl.include(path+"/arborescent.jl")
jl.include(path+"/activeset.jl")


# Define a helper function to convert numpy arrays to Julia arrays
def to_julia_array(np_array):
    return jl.Array(np_array)
# TODO: compare my syntax with 
# jl.matrix_func(juliacall.convert(jl.Array[jl.Float64, 2], my_array))


def ksparse_nnls(UtM, UtU, MtM, V, k, sum_to_one=False):
    """Compute a solution to the k sparse Nonnegative Least Squares problem in matrix format.
    Uses a columnwise loop over the arborescent algorithm.
    Everything is implemented in julia, arrays are cast as julia/numpy arrays before/after computation.
    The problem has the form $\|M - UV\|_F^2$ such that $\|H[:,q]\|_0\leq k$ for all columns $q$.
    The provided solution is a global minimizer.
    Since the problem is NP-hard, this can be quite costly when dimensions grow (especially for small sparsity level k).

    Parameters
    ----------
    UtM : numpy 2d array
        mixing matrix times input matrix
    UtU : numpy 2d array
        mixing gram matrix
    MtM : numpy 1d array
        input gram matrix (only diagonal values !!)
    V   : numpy 2d array
        initial coefficients, r \times n
    k : int
        sparsity level columnwise
    sum_to_one : bool, optional
        if the columns of H should sum to one, by default False

    Returns
    -------
    numpy 2d array
        the estimated H coefficient matrix with k sparse columns.
    """
    UtM_jl = to_julia_array(UtM)
    UtU_jl = to_julia_array(UtU)
    MtM_jl = to_julia_array(MtM) 
    V_jl = to_julia_array(V)
    jl.ksparse_mnnls(UtM_jl, UtU_jl, MtM_jl, V_jl, k, sumtoone=sum_to_one)
    return np.array(V_jl)


# Define the Python wrapper function for the arborescent function
def arborescent(AtA, Atb, btb, k, return_pareto=False, return_nb_nodes=False, sum_to_one=False):
    """
    Sparse NNLS solver for one vector
    Wrapper for the arborescent function defined in Julia (from arborescent.jl)
    
    Parameters:
    - AtA: Array (2D) for matrix AtA in the algorithm
    - Atb: Array (1D) for vector Atb in the algorithm
    - btb: Float for the btb scalar in the algorithm
    - k: Integer for parameter k
    - return_pareto: Boolean to indicate if pareto front should be returned
    - return_nb_nodes: Boolean to indicate if number of nodes should be returned
    - sum_to_one: Boolean to indicate if the solution should sum to one

    Returns:
    - The result of the arborescent function as a tuple or single result
    """
    
    AtA_jl = to_julia_array(AtA)
    Atb_jl = to_julia_array(Atb)
    
    print("Calling Julia arborescent from Giant")
    result = jl.arborescent(
        AtA_jl,
        Atb_jl,
        btb,
        k,
        returnpareto=return_pareto,
        returnnbnodes=return_nb_nodes,
        sumtoone=sum_to_one
    )
 
    # Convert the result to a numpy array (or arrays if it’s a tuple)
    if isinstance(result, tuple):
        res_array, _ = result
        return np.array(res_array), _
    else:
        return np.array(result)
    

# Define the Python wrapper function for the bab function
def bab(AtA, Atb, btb, k, support, prevx, bze, bestx, bestresid, paretofront, nbnodes, sum_to_one=False):
    """
    Wrapper for the bab function defined in Julia (from arborescent.jl).
    
    Parameters:
    - AtA: Array (2D) for matrix AtA in the algorithm
    - Atb: Array (1D) for vector Atb in the algorithm
    - btb: Float for the btb scalar in the algorithm
    - k: Integer for parameter k
    - support: List of indices for the support set
    - prevx: Array (1D) for the previous solution vector
    - bze: Integer for the biggest zeroed entry index
    - bestx: Array (1D) for the current best solution vector
    - bestresid: Float for the current best residual
    - paretofront: List of BabNode objects to store Pareto front
    - nbnodes: List of one integer to store the number of nodes explored
    - sum_to_one: Boolean to indicate if the solution should sum to one

    Returns:
    - A tuple of (bestx, bestresid)
    """
    res_bestx, res_bestid = jl.bab(
        AtA,
        Atb,
        btb,
        k,
        support,
        prevx,
        bze,
        bestx,
        bestresid,
        paretofront,
        nbnodes,
        sumtoone=sum_to_one
    )
    return np.array(res_bestx), res_bestid