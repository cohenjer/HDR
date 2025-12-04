import numpy as np
from scipy.special import kl_div
import tensorly as tl  # for backend compatibility
import torch

import time


def compute_error(V, WH):
    """
    Elementwise Kullback Leibler divergence

    Parameters
    ----------
    V : 2darray
        input data, left hand side of KL
    WH : 2d array
        right hand side of KL
    ind0 : boolean 2d array, optional
        table with True where V is not small, by default None
    ind1 : boolean 2d array, optional
        table with False where V is almost 0, by default None

    Returns
    -------
    float
        elementwise KL divergence

    """
    return tl.sum(kl_div(V,WH))    

    
def Lee_Seung_KL(V,  Wini, Hini, nb_inner=10, NbIter=10000, epsilon=1e-8, tol=1e-7, legacy=False, verbose=False, print_it=100, delta=np.inf):
    
    """
    The goal of this method is to factorize (approximately) the nonnegative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminize [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
    
    References:  
        [1] Daniel D. Lee and H. Sebastian Seung.  Learning the parts of objects by nonnegative matrix factorization.
        Nature, 1999
        [2]   Daniel D. Lee and H. Sebastian Seung. Algorithms for nonnegative matrix factorization. In
        Advances in Neural Information Processing Systems. MIT Press, 2001   
    
    Parameters
    ----------
    V : MxN array 
        observation matrix that is Vorig + B where B represents to the noise.
    W0 : MxR array
        matrix with all entries are nonnegative.
    H0 : RxN array
        matrix with all entries are nonnegative.
    NbIter : int
        the maximum number of iterations.
    NbIter_inner: int
        number of inner loops
    print_it: int
        if verbose is true, sets the number of iterations between each print.
        default: 100
    delta: float
        relative change between first and next inner iterations that should be reached to stop inner iterations dynamically.
        A good value empirically: 0.4
        default: np.inf (no dynamic stopping)

    Returns
    -------
    err : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        nonnegative estimated matrix.
    W : MxR array
        nonnegative estimated matrix.

    """
    toc = [0]
    tic = time.perf_counter()

    if verbose:
        print("\n------Lee_Sung_KL running------")

    W = Wini.copy()
    H = Hini.copy()    
    WH = W.dot(H)
    crit = [compute_error(V, WH)]
    cnt = []
  
    if legacy:
        epsilon = 0
     
    for k in range(NbIter):
                
        # FIXED H ESTIMATE W
        sumH = (np.sum(H, axis=1)[None, :])
        inner_change_0 = 1
        inner_change_l = np.inf
        for l in range(nb_inner):
            deltaW = np.maximum(W * (((V/WH).dot(H.T))/sumH-1), epsilon-W)
            W = W + deltaW
            WH = W.dot(H) 
            if k > 0:
                if l == 0:
                    inner_change_0 = np.linalg.norm(deltaW)**2
                else:
                    inner_change_l = np.linalg.norm(deltaW)**2
                if inner_change_l < delta*inner_change_0:
                    break
        cnt.append(l+1)

        # FIXED W ESTIMATE H
        
        sumW = np.sum(W, axis=0)[:, None]
        inner_change_0 = 1
        inner_change_l = np.inf
        for l in range(nb_inner):
            deltaH = np.maximum(H * ((W.T@(V/WH))/sumW-1), epsilon-H)
            H = H + deltaH
            WH = W.dot(H)
            if k > 0:
                if l == 0:
                    inner_change_0 = np.linalg.norm(deltaH)**2
                else:
                    inner_change_l = np.linalg.norm(deltaH)**2
                if inner_change_l < delta*inner_change_0:
                    break
        cnt.append(l+1)   
 
        
        # compute the error 
        crit.append(compute_error(V, WH))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k % print_it == 0:
                print("Loss at iteration {}: {}".format(k+1, crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        if tol:
            if (crit[k] <= tol):
                if verbose:
                    print("Loss at iteration {}: {}".format(k+1, crit[-1]))
                return crit, W, H, tol, cnt
        
    if verbose:
        print("Loss at iteration {}: {}".format(k+1, crit[-1]))
    return crit, W, H, toc, cnt
    
    
def Lee_Seung_KL_regression(V, W, Hini, nb_inner=10, NbIter=10000, epsilon=1e-8, tol=1e-7, legacy=False, verbose=False, print_it=100, delta=np.inf):
    
    """
    The goal of this method is to regress (approximately) the nonnegative (entry-wise) matrix V by W i.e find H such that
    V = WH + N where N represents to the noise --> It leads to find H in miminize [ V log (V/WH) - V + WH ] s.t. H >= 0
    
    
    References:  
        [1] Daniel D. Lee and H. Sebastian Seung.  Learning the parts of objects by nonnegative matrix factorization.
        Nature, 1999
        [2]   Daniel D. Lee and H. Sebastian Seung. Algorithms for nonnegative matrix factorization. In
        Advances in Neural Information Processing Systems. MIT Press, 2001   
    
    Parameters
    ----------
    V : MxN array 
        observation matrix that is Vorig + B where B represents to the noise.
    W : MxR array
        matrix with all entries are nonnegative.
    H0 : RxN array
        matrix with all entries are nonnegative.
    NbIter : int
        the maximum number of iterations.
    NbIter_inner: int
        number of inner loops
    print_it: int
        if verbose is true, sets the number of iterations between each print.
        default: 100
    delta: float
        relative change between first and next inner iterations that should be reached to stop inner iterations dynamically.
        A good value empirically: 0.4
        default: np.inf (no dynamic stopping)

    Returns
    -------
    err : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        nonnegative estimated matrix.
    W : MxR array
        nonnegative estimated matrix.

    """
    toc = [0]
    tic = time.perf_counter()

    if verbose:
        print("\n------Lee_Sung_KL running------")

    H = tl.copy(Hini) 
    WH = W@H
    crit = [compute_error(V, WH)]
    if verbose:
        print(f"Loss at initialization: {crit[0]}")
    cnt = []
  
    if legacy:
        epsilon = 0
     
    sumW = tl.sum(W, axis=0)
    for k in range(NbIter):
        inner_change_0 = 1
        inner_change_l = tl.inf
        for l in range(nb_inner):
            deltaH = tl.maximum(H * ((W.T@(V/WH))/sumW-1), epsilon-H)
            H = H + deltaH
            WH = W@H
            if k > 0:
                if l == 0:
                    inner_change_0 = tl.norm(deltaH)**2
                else:
                    inner_change_l = tl.norm(deltaH)**2
                if inner_change_l < delta*inner_change_0:
                    break
        cnt.append(l+1)   
 
        # compute the error 
        crit.append(compute_error(V, WH))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k % print_it == 0:
                print("Loss at iteration {}: {}".format(k+1, crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        if tol:
            if (crit[k] <= tol):
                if verbose:
                    print("Loss at iteration {}: {}".format(k+1, crit[-1]))
                return crit, W, H, tol, cnt
        
    if verbose:
        print("Loss at iteration {}: {}".format(k+1, crit[-1]))
    return crit, H, toc, cnt
    
    
def MU_SinglePixel(Y, H, A0, W0, lmbd=None, maxA=None, niter=1000, n_iter_inner=20, eps=1e-8, verbose=True, print_it=10):
    """
    Multiplicative Update algorithm for Nonnegative Matrix Factorization with Kullback-Leibler divergence.
    The model is Y ~ P(\alpha WAH^T), where Y is the observation matrix, W is the endmember matrix, A is the abundance matrix to be estimated, H is a positive Hadamard matrix, and N is noise. Parameter alpha accounts for the number of photons, the higher the less noisy.
    
    The optimization problem solved is:
        min_{A,W} D_KL(Y/alpha || WAH^T) + lambda * (||A||_1 + ||W||_1)
    subject to A >= 0, W >= 0
    
    Parameters
    ----------
    A0 : torch.Tensor
        Initial abundance matrix (PxM)
    W : torch.Tensor
        Endmember matrix (BxP)
    Y : torch.Tensor
        Data matrix, should be normalized by photon count (BxN)
    H : torch.Tensor
        Observation matrix, typically a positive Hadamard matrix (MxN)
    lmbd : list or float or None, optional
        Regularization parameter for the l1 norm on W and A, by default 1
        If a list is provided, it should contain two elements: [lambda_W, lambda_A]
        Otherwise the same lambda is used for both W and A.
    niter : int, optional
        Number of iterations, by default 1000
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-8
    Atrue : torch.Tensor, optional
        Ground truth abundance matrix for oracle metrics, by default None
    print_it : int, optional
        Frequency of printing metrics, by default 10
    verbose : bool, optional
        Whether to print metrics during iterations, by default True
    model : measurement class or None, optional
        Model to provide fast inference with model.forward(), by default None
    """
    #M, N = H.shape
    M = Y.shape[1]
    B, P = W0.shape
    A = torch.clone(A0)
    W = torch.clone(W0)
    sumH = torch.ones((B, M))@H  # (H.T @ 1).T
    #sumH = torch.sum(H, axis=0)
    costs = []
    
    # Handle lambda parameter
    if lmbd is None:
        lmbd = 0.0
    if isinstance(lmbd, (list, tuple)):
        lmbd_W, lmbd_A = lmbd
    else:
        lmbd_W = lmbd
        lmbd_A = lmbd

    for k in range(niter):

        WtsumH = W.T@sumH
        for _ in range(n_iter_inner):
            A = A * ((W.T@(Y/(W@(A@H.T))))@H) / (WtsumH + lmbd_A)
            if maxA is not None:
                A = torch.clamp(A, min=eps, max=maxA)
            else:
                A = torch.clamp(A, min=eps)

        AH = A@H.T
        AHtsum = torch.sum(AH, axis=1)[None, :]
        for _ in range(n_iter_inner):
            W = W * ((Y/(W@AH))@AH.T) / (AHtsum + lmbd_W)
            W = torch.clamp(W, min=eps)
            
        if k % print_it == 0:
            WAHt = W@AH
            c = compute_error(Y, WAHt) + lmbd_A*torch.sum(A) + lmbd_W*torch.sum(W)
            costs.append(c.cpu().detach().numpy())

            if verbose:
                print(f"Iteration {k}, Cost: {c.cpu().detach().numpy()}")
                #print('Erreur :', e.cpu().detach().numpy())
                #print('PSNR :', torch.mean(p))

    return W, A, costs


def MU_SinglePixel_fast(Y, forward, adjoint, A0, W0, lmbd=None, maxA=None, niter=1000, n_iter_inner=20, eps=1e-8, verbose=True, print_it=10):
    """
    Multiplicative Update algorithm for Nonnegative Matrix Factorization with Kullback-Leibler divergence.
    The model is Y ~ P(\alpha WAH^T), where Y is the observation matrix, W is the endmember matrix, A is the abundance matrix to be estimated, H is a positive Hadamard matrix, and N is noise. Parameter alpha accounts for the number of photons, the higher the less noisy.
    
    The optimization problem solved is:
        min_{A,W} D_KL(Y/alpha || WAH^T) + lambda * (||A||_1 + ||W||_1)
    subject to A >= 0, W >= 0
    
    Parameters
    ----------
    A0 : torch.Tensor
        Initial abundance matrix (PxM)
    W : torch.Tensor
        Endmember matrix (BxP)
    Y : torch.Tensor
        Data matrix, should be normalized by photon count (BxN)
    forward : function
        Computes efficiently x@H.T where H is the observation matrix
    adjoint : function
        Computes efficiently y@H where H is the observation matrix
    lmbd : list or float or None, optional
        Regularization parameter for the l1 norm on W and A, by default 1
        If a list is provided, it should contain two elements: [lambda_W, lambda_A]
        Otherwise the same lambda is used for both W and A.
    niter : int, optional
        Number of iterations, by default 1000
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-8
    Atrue : torch.Tensor, optional
        Ground truth abundance matrix for oracle metrics, by default None
    print_it : int, optional
        Frequency of printing metrics, by default 10
    verbose : bool, optional
        Whether to print metrics during iterations, by default True
    model : measurement class or None, optional
        Model to provide fast inference with model.forward(), by default None
    """
    #M, N = H.shape
    B, M = Y.shape
    B, _ = W0.shape
    A = torch.clone(A0)
    W = torch.clone(W0)
    sumH = adjoint(torch.ones((B, M))) # (H.T @ 1).T
    costs = []
    
    # Handle lambda parameter
    if lmbd is None:
        lmbd = 0.0
    if isinstance(lmbd, (list, tuple)):
        lmbd_W, lmbd_A = lmbd
    else:
        lmbd_W = lmbd
        lmbd_A = lmbd

    for k in range(niter):

        WtsumH = W.T@sumH
        for _ in range(n_iter_inner):
            A = A * adjoint(W.T@(Y/(W@(forward(A).T)))) / (WtsumH + lmbd_A)
            if maxA is not None:
                A = torch.clamp(A, min=eps, max=maxA)
            else:
                A = torch.clamp(A, min=eps)

        AH = forward(A).T
        AHtsum = torch.sum(AH, axis=1)[None, :]
        for _ in range(n_iter_inner):
            W = W * ((Y/(W@AH))@AH.T) / (AHtsum + lmbd_W)
            W = torch.clamp(W, min=eps)
            
        if k % print_it == 0:
            WAHt = W@AH
            c = compute_error(Y, WAHt) + lmbd_A*torch.sum(A) + lmbd_W*torch.sum(W)
            costs.append(c.cpu().detach().numpy())

            if verbose:
                print(f"Iteration {k}, Cost: {c.cpu().detach().numpy()}")
                #print('Erreur :', e.cpu().detach().numpy())
                #print('PSNR :', torch.mean(p))

    return W, A, costs