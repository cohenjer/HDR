import tensorly as tl
import numpy as np
import copy


def scale_factors_fro(
    tensor,
    data,
    ridge_coefficients,
    sparsity_coefficients,
    format_tensor="cp",
    nonnegative=False,
):
    """
    Optimally scale Tucker tensor [G;A,B,C] in

    :math: `min_x \|data - x^{n_modes} [G;A_1,A_2,A_3]\|_F^2 + \sum_i sparsity_coefficients_i \|A_i\|_1 + \sum_j ridge_coefficients_j \|A_j\|_2^2`

    This avoids a scaling problem when starting the separation algorithm, which may lead to zero-locking.
    The problem is solved by finding the positive roots of a polynomial.

    Works with any number of modes and both CP and Tucker, as specified by the `format` input. For "tucker" format, sparsity and ridge have an additional final value for the core reg.

    note: sparsity works only under nonnegativity constraints
    note: comment on nonnegative keyword
    """
    print("TODO USE THE FONCTION FROM TENSORLY HDR")
    factors = copy.deepcopy(tensor[1])
    if format_tensor == "tucker":
        factors.append(tensor[0])
    n_modes = len(factors)
    l1regs = [
        sparsity_coefficients[i] * tl.sum(tl.abs(factors[i])) for i in range(n_modes)
    ]
    l2regs = [ridge_coefficients[i] * tl.norm(factors[i]) ** 2 for i in range(n_modes)]
    # We define a polynomial
    # a x^{2n_modes} + b x^{n_modes} + c x^{2} + d x^{1}
    # and find the roots of its derivative, compute the value at each one, and return the optimal scale x and scaled factors.
    a = (tensor.norm() ** 2) / 2
    b = -tl.sum(data * tensor.to_tensor())
    c = sum(l2regs)
    d = sum(l1regs)
    poly = [0 for i in range(2 * n_modes + 1)]
    poly[1] = d
    poly[2] = c
    poly[n_modes] = b
    poly[2 * n_modes] = a
    poly.reverse()
    grad_poly = [0 for i in range(2 * n_modes)]
    grad_poly[0] = d
    grad_poly[1] = 2 * c
    grad_poly[n_modes - 1] = n_modes * b
    grad_poly[2 * n_modes - 1] = 2 * n_modes * a
    grad_poly.reverse()
    roots = np.roots(grad_poly)
    current_best = np.Inf
    best_x = 0
    for sol in roots:
        if sol.imag < 1e-16:
            sol = sol.real
            if sol > 0 or not nonnegative:
                val = np.polyval(poly, sol)
                if val < current_best:
                    current_best = val
                    best_x = sol
    if current_best == np.Inf:
        print("No solution to optimal scaling")
        return tensor, None

    # We have the optimal scale
    for i in range(n_modes):
        factors[i] *= best_x

    if format_tensor == "tucker":
        return tl.tucker_tensor.TuckerTensor((factors[-1], factors[:-1])), best_x
    return tl.cp_tensor.CPTensor((None, factors)), best_x