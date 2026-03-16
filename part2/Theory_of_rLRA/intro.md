# Summary

Despite their widespread use in the signal processing and machine learning communities, regularized LRA models remain rather poorly understood to this day. In particular, there are two kinds of questions in the literature for which there are no clear answers.

- **Identifiability**: given an exact factorization model, e.g. $M=X_1X_2$, and a set of constraints $\mathcal{C}_i$ such that $X_i\in\mathcal{C}_i$, are solutions to the factorization problem under these constraints essentially unique?

- **Solution characterization**: given a constrained or regularized optimization problem, for instance

$$
    \argmin{X_1,X_2} \|M - X_1X_2^T\|_F^2 + g_1(X_1) + g_2(X_2)
$$ 
for some regularization functions $g_1$ and $g_2$, what are the properties satisfied by the solutions of this problem? 

These two questions are related at first glance but differ fundamentally. The identifiability of rLRA models is a cornerstone of source separation techniques, as it allows one to give a physical meaning to the estimated factor matrices. Characterizing the solutions to an optimization problem amounts to ensuring that certain properties hold for the estimated factor matrices, regardless of whether the solution is unique. For instance, in some cases, the user cares only about the sparsity level of the factor matrices or their orthogonality.

Identifiability of rLRA has received significant attention, in particular for matrix factorization models such as dictionary learning, NMF, or tensor decomposition models such as nonnegative CPD and, more recently, nonnegative Tucker decomposition, see [](../../part1/lra.md). My contribution to the identifiability question has been to provide deterministic sufficient conditions for the [identifiability of complete dictionary learning](./DL_identifiability.md), improving upon other works and correcting mistakes in the literature. I also show how to [compute complete DL](./DL_identifiability_code.ipynb) in practice with Tensorly.

Solution characterization, on the other hand, has not been considered much in the literature, in particular in contrast to the significant body of work that concerns regularized regression problems such as ridge regression or LASSO {cite}`foucart2013introduction`. For instance, it is still unclear whether solutions to a sparse NMF problem

$$
\argmin{X_1\geq 0,\; X_2\geq 0} \|M - X_1X_2^T\|_F^2 + \mu \left( \|X_1\|_1 + \|X_2\|_1 \right)
$$

are indeed sparse or not, and under which conditions. This might be due to the relative difficulty of studying the solutions of these problems. Indeed, the same problem with a single factor admits a closed-form solution via the soft-thresholding operator, whereas the sparse NMF problem with two factors is non-convex, non-smooth, and NP-hard. As a first step towards understanding the properties of the solution or rLRA, together with Valentin Leplat, we showed that, for a large class of rLRA problems, explicit regularization, such as $\ell_1$-$\ell_1$, leads to implicit regularization due to the scale-invariance of rLRA models. Our contributions are summarized in [](./HRSI_theory.md), with a discussion of the algorithmic aspects in [](./HRSI_algorithm.ipynb).


Because rLRA solutions have ambiguous properties, in particular for homogeneous regularization such as the $\ell_1$ norm, I studied, in another line of work, dictionary learning and dictionary-based LRA under cardinality constraints. Cardinality constraints impose sparsity explicitly on the solution. I have studied in particular the identifiability and the conception of dedicated algorithms for a class of models that includes the following dictionary-based NMF

$$
\argmin{X_2\geq 0,\; S\in\{0,1\},\; \|S[:,q]\|_0\leq 1} \|M - DSX_2^T\|_F^2 
$$

where $D$ is a known dictionary of template components. These contributions are summarized in [](./DLRA.ipynb).
