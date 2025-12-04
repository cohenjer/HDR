import tensorly as tl
import numpy as np
from dlra.utils import gen_BSplines

# Note: numpy/tensorly stores images with ys in axis=0, and xs in axis=1
# This is a source of infinite confusion!


def convert_to_pixel(pixlist, imshape_x):
    '''
    Converts the list of pixel index to two lists of positions of these pixels in the (x,y) plane.
    Vectorization in tensorly is row-first, so the indexes vary faster along the x axis (axis=1). 
    Therefore only the x dimension of the image is needed to warp the index and compute (x,y) indices.
    '''
    Kx = []
    Ky = []
    for pix in pixlist:
        Kx.append(pix % imshape_x)
        Ky.append(pix // imshape_x)
    return Kx, Ky


def convert_to_index(Kx, Ky, imshape_x):
    '''
    Takes two input lists Kx and Ky containing positions in the (x,y) plane of pixels, and converts them into a single list of indices in the unfolded image.
    Unfolding (vectorization) is supposed row-first as in Tensorly, therefore we only need to know the number of pixels along the x axis, imshape_x.
    '''
    K = []
    for i in range(len(Kx)):
        K.append(Kx[i]+Ky[i]*imshape_x)
    return K


def convert_to_pixel_from_patches(Kd, zx, zy):
    '''
    Takes a list of pairs (patch index, column index inside the patch) Kd, and lists of positions of the patches along x, zx, and y, zy, to compute the position of
    the columns inside in patch in the global image as pixels (Kx[i], Ky[i]).
    '''
    Kx = []
    Ky = []
    for i, idx in Kd:
        dzx = zx[i][1]-zx[i][0]
        Kx.append(idx % dzx + zx[i][0])
        Ky.append(idx // dzx + zy[i][0])
    return Kx, Ky
        

def sparsify(M, s=0.5, epsilon=0):
    """Adds zeroes in matrix M in order to have a ratio s of nnzeroes/nnentries.

    Parameters
    ----------
    M : 2darray
        The input numpy array
    s : float, optional
        the sparsity ratio (0 for fully sparse, 1 for density of the original array), by default 0.5
    """    
    vecM = M.flatten()
    # use quantiles
    val = np.quantile(vecM, 1-s)
    # put zeros in M
    M[M<val]=epsilon
    return M

def generate_splines_DLRA(n1, width, deg):
    D1 = gen_BSplines(n1, width[0], deg[0], shift=0)
    D2 = gen_BSplines(n1, width[0], deg[0], shift=6)
    D3 = gen_BSplines(n1, width[0], deg[0], shift=12)
    D4 = gen_BSplines(n1, width[0], deg[0], shift=18)
    D5 = gen_BSplines(n1, width[0], deg[0], shift=24)
    D6 = gen_BSplines(n1, width[1], deg[1], shift=1)
    D7 = gen_BSplines(n1, width[1], deg[1], shift=3)
    D8 = gen_BSplines(n1, width[1], deg[1], shift=5)
    D9 = gen_BSplines(n1, width[1], deg[1], shift=7)
    D10 = gen_BSplines(n1, width[1], deg[1], shift=9)
    D11 = gen_BSplines(n1, width[2], deg[2], shift=0)
    D = np.concatenate((D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11), axis=1)
    D = D/np.linalg.norm(D,axis=0)
    
    return D


def permute_spectra(W):
    """Permutes the columns of W to have them sorted by increasing frequency of the maximum value in each column.

    Parameters
    ----------
    W : 2darray
        The input numpy array (spectra)

    Returns
    -------
    W_perm : 2darray
        The permuted array
    perm : list
        The permutation applied to the columns
    """
    max_indices = np.argmax(W, axis=0)
    perm = np.argsort(max_indices)
    return perm