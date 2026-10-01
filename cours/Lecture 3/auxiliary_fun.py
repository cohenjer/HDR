import numpy as np
import tensorly as tl

# A quick translation of separable NMF from Giant.jl


def spa(X, r, tol=1e-8, normalize=False, verbose=True):
    """Computes pure spectra and pure pixels from nonnegative matrix X, using the successive  projection algorithm (SPA).

    Parameters
    ----------
    X : numpy 2d array
        input matrix on which to compute separable nonnegative NMF
    r : int
        number of spectra to extract from pure pixels
    tol : float, optional
        tolerance for stopping inner iterations, by default 1e-8
    normalize : bool, optional
        normalize the input on the simplex, by default false
    verbose : bool, optional
        whether to print progress, by default true

    Returns
    -------
    list
        indices of selected pure pixels
    numpy 2d array
        the estimated pure spectra
    numpy 2d array
        the estimated abundances
    """
    # Get dimensions
    m, n = tl.shape(X)
    X = tl.copy(X)  # copy in local scope to avoid modifying input
    
    # Optionally normalize so that columns of X sum to one
    if normalize:
        for j in range(n):
            X[:, j] = X[:, j]/tl.sum(tl.abs(X[:, j]))

    # Init
    # Set of selected indices
    K = [0 for i in range(r)]
    # Norm of columns of input X
    normX0 = tl.norm(X, axis=0)**2
    R = tl.copy(X)
    # Max of the columns norm
    nXmax = tl.max(normX0)
    # Init residual
    normR = tl.copy(normX0)

    # SPA loop
    i = 0
    while i < r and tl.sqrt(tl.max(normR)/nXmax) > tol:
        if verbose:
            print(i, K)
        # Select column of X with largest l2-norm
        a = tl.max(normR)
        # Check ties up to 1e-6 precision
        b = np.argwhere((a - normR) / a <= 1e-6)
        if tl.ndim(b) > 1:
            # b should be 1d array, reduce to 1d
            b = tl.reshape(b, (-1,))
        # In case of a tie, select column with largest norm of the input matrix
        d = np.argmax(normX0[b])
        b = b[d]
        # Save index of selected column, and column itself
        K[i] = int(b)
        U = X[:, K[:i+1]]  # can be optimimed by pre-allocations
       
        # Update residual coefficient
        R = R - tl.tenalg.outer([U[:, -1], U[:, -1].T@R])/tl.norm(U[:, -1])**2
        normR = tl.norm(R, axis=0)
        
        # Update residual (correct?)
        #for j in range(i-1):  # ??i or i-1
            #U[:, i] = U[:, i] - U[:, j] * np.dot(U[:, j], U[:, i])
        #U[:, i] = U[:, i]/np.linalg.norm(U[:, i])
        #normR = normR - (X.T @ U[:, i]) ** 2 # TODO BUGGED (?)

        # Increment iterator
        i += 1

    #if len(np.unique(K)) == len(K):
    #    H = hals_nnls(U.T@X, U.T@U, n_iter_max=100, tol=1e-8)
    #else:
    #    H = None
    #    print("There are duplicates in K, cannot estimate H")
    if verbose:
        print(f"Returning {K} as estimated pure pixel indices")

    return K, U

