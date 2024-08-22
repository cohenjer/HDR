import tensorly as tl
import numpy as np
from tensorly.cp_tensor import cp_norm


def get_pseudo_inverse(cp_e, r, mode, T):
    pseudo_inverse = tl.ones((r, r), **tl.context(T))
    for j, factor in enumerate(cp_e[1]):
        if j != mode:
            pseudo_inverse = pseudo_inverse * tl.dot(
                tl.transpose(factor), factor
            )
    return pseudo_inverse

def err_calc_simple(cp_e, mttkrp, norm_tensor):
    factors_norm = cp_norm(cp_e)
    iprod = tl.sum(tl.sum(mttkrp * tl.conj(cp_e[1][-1]), axis=0))
    unnorml_rec_error = tl.sqrt(
        tl.abs(norm_tensor + factors_norm**2 - 2 * iprod)
    )
    return unnorml_rec_error