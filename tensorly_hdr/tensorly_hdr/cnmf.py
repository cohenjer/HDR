from math import e
import numpy as np
from scipy.special import kl_div

# from numba import jit
import matplotlib.pyplot as plt


def Dkl(x, y):
    return np.sum(kl_div(x, y))


def beta_divergence(beta: float, x: np.array, y: np.array):
    # special case KL
    nrow = np.shape(x)[0]
    ncol = np.shape(x)[1]
    d = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            # if y[i,j]<1e-8:
            #     print('Warning')
            if beta == 0:
                d[i, j] = x[i, j] / y[i, j] - np.log(x[i, j] / y[i, j]) - 1
            elif beta == 1:
                d[i, j] = x[i, j] * np.log(x[i, j] / y[i, j]) - x[i, j] + y[i, j]
            else:
                d[i, j] = (
                    (
                        x[i, j] ** beta
                        + (beta - 1) * y[i, j] ** beta
                        - beta * x[i, j] * y[i, j] ** (beta - 1)
                    )
                    / (beta - 1)
                    / beta
                )
    return np.sum(d)


def convolutive_MM(
    X: np.array, r: int, itmax: int, beta: float, T: int, e: float, W0=None, H0=None
):
    """
    Algorithm MM for cnmf
    --------------------------------
    :param X: spectrogram after STFT
    :param r: factorization rank
    :param itmax: limit number of iteration
    :param beta: beta in beta divergence
    :param T: dictionary number
    :param e: relative err tolerance
    :param W0: initialization of W
    :param H0: initialization of H
    :return: matrix W, matrix H, total iteration number, an array of objective value in each iteration
    """
    nrow = np.shape(X)[0]
    ncol = np.shape(X)[1]
    if W0 is None or H0 is None:
        # random initialisation of W and H
        # W is a 3 dimensions array
        W = np.array([np.random.rand(nrow, r)] * T)
        H = np.random.rand(r, ncol)
    if W0 is not None:
        W = np.copy(W0)
    if H0 is not None:
        H = np.copy(H0)
    n_iter = 0
    # set value of gamma
    if beta < -1:
        gamma = (2 - beta) ** (-1)
    elif beta > 2:
        gamma = (beta - 1) ** (-1)
    else:
        gamma = 1

    err_int = beta_divergence(beta, X, sum(np.dot(W[t], shift(H, t)) for t in range(T)))
    obj1 = 0
    all_err = [err_int]

    while n_iter < itmax:
        # update H
        A = sum(np.dot(W[t], shift(H, t)) for t in range(T))
        for n in range(ncol):
            if n < ncol - T:
                num = sum(
                    np.dot(
                        W[n_prime - n].T, X[:, n_prime] * (A[:, n_prime]) ** (beta - 2)
                    )
                    for n_prime in range(n, n + T)
                )
                denom = sum(
                    np.dot(W[n_prime - n].T, A[:, n_prime] ** (beta - 1))
                    for n_prime in range(n, n + T)
                )
            else:
                num = sum(
                    np.dot(
                        W[n_prime - n].T, X[:, n_prime] * (A[:, n_prime]) ** (beta - 2)
                    )
                    for n_prime in range(n, ncol)
                )
                denom = sum(
                    np.dot(W[n_prime - n].T, A[:, n_prime] ** (beta - 1))
                    for n_prime in range(n, ncol)
                )
            H[:, n] = H[:, n] * (num / denom) ** gamma

        # update W
        for t in range(T):
            A = sum(np.dot(W[t], shift(H, t)) for t in range(T))
            W[t] = (
                W[t]
                * (
                    np.dot((A ** (beta - 2)) * X, shift(H, t).T)
                    / np.dot(A ** (beta - 1), shift(H, t).T)
                )
                ** gamma
            )

        obj = beta_divergence(beta, X, sum(np.dot(W[t], shift(H, t)) for t in range(T)))
        all_err.append(obj)
        print(f"Iteration: {n_iter}, loss: {obj}")

        # renormalization
        W, H = renormalization(W, H, T)
        if abs(obj - obj1) / err_int < e:
            break
        obj1 = obj
        n_iter = n_iter + 1
        # print("objective value: ", obj)
    return W, H, n_iter, all_err


def shift(H, t):
    """
    Right shift matrix H by t columns
    ------------------------------
    :param H: activation matrix H
    :param t: shift number
    :return: matrix H after shift
    """
    if H.ndim == 1:
        H_shift = np.roll(H, t)
        if t > 0:
            H_shift[:t] = 0
        return H_shift
    s = np.shape(H)
    # print("shape s ", s)
    H_shift = np.zeros(shape=s)
    H_shift[:, t : s[1]] = H[:, 0 : (s[1] - t)]
    return H_shift


def shift_m(h, T, rank=1):
    """
    Creates a matrix with shifted versions of h in its rows
    """
    if h.ndim == 1:
        s = np.shape(h)[0]
        H_shift = np.zeros((T, s))
        for i in range(T):
            H_shift[i, :] = np.copy(np.roll(h, i))
            H_shift[i, :i] = 0
        return H_shift
    s = np.shape(h)[1]
    for r in range(rank):
        # Repeat the same procedure, stacking the outputs for each row of H
        H_r = np.zeros((T, s))
        for i in range(T):
            H_r[i, :] = np.copy(np.roll(h[r, :], i))
            H_r[i, :i] = 0
        if r == 0:
            H_shift = H_r
        else:
            H_shift = np.vstack((H_shift, H_r))
    return H_shift


# def average_shift(Hshift):
# """
# Average the shifted versions in H to get back h
# """
# T = np.shape(Hshift)[0]
# s = np.shape(Hshift)[1]
# h = np.zeros(s)
# count = np.zeros(s)
# for i in range(T):
## roll is circular, we want zero padding
# shifth = np.roll(Hshift[i, :], -i)
# shifth[-i:0] = 0
# h += shifth
# count[i:] += 1
# h = h / count
# return h


def renormalization(W, H, T):
    """
    Normalize W and H by norm L1 of W(k)
    --------------------
    :param W: dictionary W in present iteration
    :param H: activation matrix in present iteration
    :param T: Total number of dictionary
    :return: W and H after normalization
    """

    r = np.shape(H)[0]
    lam = np.zeros(r)
    for k in range(r):
        lam[k] = np.sum(W[:, :, k])
    for t in range(T):
        W[t] = np.dot(W[t], np.diag(1 / lam))
    H = np.dot(np.diag(lam), H)
    return W, H


# Specializations for AMT
def convolutive_regression(
    X: np.array,
    W: np.array,
    itmax=100,
    tol=1e-8,
    eps=1e-8,
    verbose=True,
    print_it=10,
):
    """
    Algorithm MM for cnmf fitting only H, KL divergence, MM2 from Fagot et al.
    Initialization is important, in praticular if the initial patterns are shifted from origin (max intensity is not as tau=0), this helps with border effects...
    Only computes H update.
    --------------------------------
    :param X: spectrogram after STFT
    :param W: tensor with each template stacked on the third dimension.
    :param itmax: limit number of iteration
    :param e: relative err tolerance
    :param W0: initialization of W
    :param H0: initialization of H
    :return: matrix W, vector H, an array of objective value in each iteration
    """
    nrow = np.shape(X)[0]
    ncol = np.shape(X)[1]
    T = np.shape(W)[1]
    rank = np.shape(W)[2]
    H = np.random.rand(rank, ncol)
    n_iter = 1

    Ht = shift_m(H, T, rank=rank)  # flat
    A = np.reshape(W, [nrow, T*rank], order="F") @ Ht
    err_int = Dkl(X, A)
    if verbose:
        print(f"Iteration initial, loss: {err_int}")
    obj1 = 0
    all_err = [err_int]

    # Update H
    Wsum = np.sum(W, axis=(0, 1))
    Wsumpartial = []
    for i in range(T):
        Wsumpartial.append(np.sum(W[:, : T - i, :], axis=(0, 1)))
            
    while n_iter <= itmax:

        XonA = X / A
        for n in range(ncol):
            if n < ncol - T:  # all columns of W
                num = np.sum(W * (XonA[:, n : n + T:, None]), axis=(0, 1))
                H[:, n] = np.maximum(H[:, n] * (num / Wsum), eps)
            else:
                num = np.sum(W[:, : (ncol - n)] * (XonA[:, n:ncol, None]), axis=(0, 1))
                H[:, n] = np.maximum(H[:, n] * (num / Wsumpartial[n - (ncol - T)]), eps)
        Ht = shift_m(H, T, rank=rank)
        A = np.reshape(W, [nrow, T*rank], order="F") @ Ht

        if (n_iter % print_it) == 0:
            obj = Dkl(X, A)
            if verbose:
                print(f"Iteration: {n_iter}, loss: {obj}")
            all_err.append(obj)

            if abs(obj - obj1) / err_int < tol:
                break
            obj1 = obj

        n_iter = n_iter + 1

    # renormalization so that max of W is one
    # Wnorm = np.max(W)
    # W = W / Wnorm
    # h = h * Wnorm
    return H, all_err


#def rank_one_convolutive_nmf(
    #X: np.array,
    #itmax=100,
    #T=5,
    #tol=1e-8,
    #eps=1e-8,
    #n_iter_inner=5,
    #W0=None,
    #H0=None,
    #verbose=True,
    #print_it=10,
    #update_w=True,
#):
    #"""
    #Algorithm MM for cnmf, KL divergence and rank-one case, MM2 from Fagot et al.
    #Initialization is important, in praticular if the initial patterns are shifted from origin (max intensity is not as tau=0), this helps with border effects...
    #--------------------------------
    #:param X: spectrogram after STFT
    #:param itmax: limit number of iteration
    #:param T: dictionary number
    #:param e: relative err tolerance
    #:param W0: initialization of W
    #:param H0: initialization of H
    #:return: matrix W, vector H, an array of objective value in each iteration
    #"""
    #nrow = np.shape(X)[0]
    #ncol = np.shape(X)[1]
    #if W0 is None and H0 is None:
        ## Init by rank-one KL NMF
        ## wkl = np.sum(X, axis=1)/np.sqrt(np.sum(X))
        ## h = np.sum(X, axis=0)/np.sqrt(np.sum(X))
        ## small init
        ## W = eps*np.ones((nrow, T))
        ## W[:, 0] = wkl
        ## purenote init
        #t0 = np.argmax(np.sum(X, axis=0))
        #h = eps * np.ones(ncol)
        ## h = np.ones(ncol)
        #if T >= 5:
            #W = X[:, t0 - 3 : t0 + T - 3]  # hopefully not on the border
            #h[t0 - 3] = 1
        #else:
            #W = X[:, t0 : t0 + T]  # hopefully not on the border

    #else:
        #h = np.copy(H0)
        #W = np.copy(W0)
    #n_iter = 1

    #H = shift_m(h, T)  # flat
    #A = W @ H
    #err_int = Dkl(X, A)
    #if verbose:
        #print(f"Iteration initial, loss: {err_int}")
    #obj1 = 0
    #all_err = [err_int]

    #while n_iter <= itmax:

        ## Update H
        #if update_w or n_iter == 1:
            #Wsum = np.sum(W)
            #Wsumpartial = []
            #for i in range(T):
                #Wsumpartial.append(np.sum(W[:, : T - i]))

        #for _ in range(n_iter_inner):
            ## A = W@H
            #XonA = X / A
            #for n in range(ncol):
                #if n < ncol - T:  # all columns of W
                    #num = np.sum(W * (XonA[:, n : n + T]))
                    #h[n] = np.maximum(h[n] * (num / Wsum), eps)
                    ## Version MM1
                    ## H = shift_m(h, T)
                    ## An = W@H[:,n:n+T]
                    ## XonA[:, n:n+T] = X[:, n:n+T]/An
                #else:
                    #num = np.sum(W[:, : (ncol - n)] * (XonA[:, n:ncol]))
                    #h[n] = np.maximum(h[n] * (num / Wsumpartial[n - (ncol - T)]), eps)
                    ## Version MM1
                    ## H = shift_m(h, T)
                    ## An = W@H[:, n:ncol]
                    ## XonA[:, n:ncol] = X[:, n:ncol]/An
            ## Version MM2
            #H = shift_m(h, T)
            #A = W @ H
        ## plt.imshow(20*np.log10(A))
        ## plt.show()

        ## update W
        ## A = W@H
        #if update_w:
            #Hsum = np.sum(H.T, axis=0)[None, :]
            #for _ in range(n_iter_inner):
                #W = np.maximum(W * (np.dot(X / A, H.T) / Hsum), eps)
                #A = W @ H  # outside the loop in Fagot et al.

        #if (n_iter % print_it) == 0:
            #obj = Dkl(X, A)
            #if verbose:
                #print(f"Iteration: {n_iter}, loss: {obj}")
            #all_err.append(obj)

            #if abs(obj - obj1) / err_int < tol:
                #break
            #obj1 = obj

        #n_iter = n_iter + 1

    ## renormalization so that max of W is one
    ## Wnorm = np.max(W)
    ## W = W / Wnorm
    ## h = h * Wnorm
    #return W, h, all_err



def convolutive_nmf(
    X: np.array,
    rank=None,
    T=5,
    itmax=100,
    n_iter_inner=5,
    tol=1e-8,
    eps=1e-8,
    init="rankone",
    verbose=True,
    print_it=10,
    normalize_output=True
):
    """
    Algorithm MM for cnmf fitting only H, KL divergence, MM2 from Fagot et al.
    Initialization is important, in praticular if the initial patterns are shifted from origin (max intensity is not as tau=0), this helps with border effects...
    --------------------------------
    :param X: spectrogram after STFT
    :param W: tensor with each template stacked on the third dimension.
    :param itmax: limit number of iteration
    :param n_iter_inner: number of inner iterations
    :param e: relative err tolerance
    :param W0: initialization of W
    :param H0: initialization of H
    :param normalize_output: sets the max of each template in W to one and scales rows of H accordingly
    :return: matrix W, vector H, an array of objective value in each iteration
    """
    nrow = np.shape(X)[0]
    ncol = np.shape(X)[1]
    if init == "rankone":
        # Init by rank-one KL NMF
        wkl = np.sum(X, axis=1)/np.sqrt(np.sum(X))
        h = np.sum(X, axis=0)/np.sqrt(np.sum(X))
        # normalization like spectrogram (max one)
        norm_wkl = np.max(wkl)
        wkl = wkl / norm_wkl
        h = h * norm_wkl
        # small init
        W = np.random.rand(nrow, T, rank)*1e-3
        W[:, 0, 0] = wkl
        H = eps*np.ones((rank, ncol))
        H[0, :] = h
    elif init == "separable":
        # purenote init
        t0 = np.argmax(np.sum(X, axis=0))  # spectrogram normalized to one
        H = 1e-3*np.random.rand(rank, ncol)
        W = 1e-3*np.random.rand(nrow, T, rank)
        if T >= 5:
            W[:, :, 0] = X[:, t0 - 3 : t0 + T - 3]  # hopefully not on the border
            H[0, t0 - 3] = 1
        else:
            W[:, :, 0] = X[:, t0 : t0 + T]  # hopefully not on the border
    else:
        # random init, needs rescaling
        W = np.random.rand(nrow, T, rank)
        H = np.random.rand(rank, ncol)

    Ht = shift_m(H, T, rank=rank)  # flat
    A = np.reshape(W, [nrow, T*rank], order="F") @ Ht
    err_int = Dkl(X, A)
    if verbose:
        print(f"Iteration initial, loss: {err_int}")
    obj1 = 0
    all_err = [err_int]

    n_iter = 1
            
    while n_iter <= itmax:

        # Update H
        Wsum = np.sum(W, axis=(0, 1))
        Wsumpartial = []
        for i in range(T):
            Wsumpartial.append(np.sum(W[:, : T - i, :], axis=(0, 1)))
            
        for _ in range(n_iter_inner):
            XonA = X / A
            for n in range(ncol):
                if n < ncol - T:  # all columns of W
                    num = np.sum(W * (XonA[:, n : n + T:, None]), axis=(0, 1))
                    H[:, n] = np.maximum(H[:, n] * (num / Wsum), eps)
                else:
                    num = np.sum(W[:, : (ncol - n)] * (XonA[:, n:ncol, None]), axis=(0, 1))
                    H[:, n] = np.maximum(H[:, n] * (num / Wsumpartial[n - (ncol - T)]), eps)
            Ht = shift_m(H, T, rank=rank)
            A = np.reshape(W, [nrow, T*rank], order="F") @ Ht

        # Update W
        # A = W@H
        Hsum = np.sum(Ht.T, axis=0)[None, :]
        Wunfold = np.reshape(W, [nrow, T*rank], order="F")
        for _ in range(n_iter_inner):
            Wunfold = np.maximum(Wunfold * (np.dot(X / A, Ht.T) / Hsum), eps)
            A = Wunfold @ Ht
        W = np.reshape(Wunfold, [nrow, T, rank], order="F")
                
        if (n_iter % print_it) == 0:
            obj = Dkl(X, A)
            if verbose:
                print(f"Iteration: {n_iter}, loss: {obj}")
            all_err.append(obj)

            if abs(obj - obj1) / err_int < tol:
                break
            obj1 = obj

        n_iter = n_iter + 1
        
    if normalize_output:
        # renormalization so that max of W is one
        Wnorms = np.max(W, axis=(0,1))
        W = W / Wnorms[None, None, :]
        H = H * Wnorms[:, None]

    return W, H, all_err