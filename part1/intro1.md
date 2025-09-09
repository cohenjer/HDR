# Background

```{tableofcontents}
```

Donner une idée très large spectre de ce qu'on fait en LRA/optim etc, comme une intro d'ANR.

Low-rank approximations are workhorse methods in several unsupervised machine learning tasks such as dimensionality reduction [cite Tucker, SVD, PCA] and blind source separation [CPD, NMF, sourcesep]. The most prominent LRA method in machine learning is probably Principal Component Analysis, which numerically boils down to computing a singular value decomposition, and is therefore efficiently computed. 

[Illustration of PCA?]

Formally, for matrix data, a (real) low-rank approximation problem is an optimization problem of the form

$$ \min_{U\in\mathbb{R}^{m\times r},V\in\mathbb{R}^{r\times n}} f(Y, UV) $$
where $r$ is the rank of the approximation $UV\approx Y$, with $r\ll m,n$, and $f$ is a cost function, typically $f(Y, X) = \|Y-X\|_F^2$.

```{margin}
We present the case of matrix factorization for simplicity here, but the same logic applies to higher order factorizations.
```

In unsupervised machine learning tasks such as blind source separation, users often seek to give a physical meaning to the factor matrices $U$ and $V$. In fact, blind source separation can be seen as an inverse problem, where both the mixing matrix $U$ and the sources $V$ are unknown. In this context, there exists a true pair $(U_0, V_0)$ that the user seeks to recover. A necessary condition to correctly interpret a reconstructed pair $(U^\ast,V^\ast)$ is the uniqueness of this reconstructed solution. Note that LRA as defined above is never unique, since any product $UV$ can also be written $UPP^{-1}V$ for an invertible matrix $P$ of size $r\times r$.

An approach widely used in signal processing and machine learning to restrict the set of solutions of an inverse problem is to add prior information on the unknowns in the form of constraints or regularizations. Regularized LRA can be formulated as 

$$ \min_{U\in \mathbb{R}^{m\times r},V\in\mathbb{R}^{r\times n}} f(Y, UV) + g_{U}(U) + g_{V}(V) $$
where $g_U$ and $g_V$ are regularizations promoting specific solutions. Depending on the choice of regularizations, the solutions may now be unique, and have specific properties such as smoothness, sparsity or nonnegativity.

```{note}
Regularizations apply typically on each factor $U$ and $V$ independently. Indeed the main identifiability issue in LRA is the rotation ambiguity $UV = UPP^{-1}V$. A regularization $g_{UV}(UV)$ would not discriminate between two solutions identical up to permutation and therefore is not enough to ensure uniqueness of rLRA solutions.
```

PCA is a constrained LRA model: factors are imposed to be orthognal matrices. This allows to obtain a model with unique factors (under mild conditions [ref]), but orthogonality may not satisfied by the ground-truth factors $U^*$ and $V^*$. On the other hand, Nonnegative Matrix Factorization (NMF), obtained by setting $g_U$ and $g_V$ to characteristic functions of the nonnegative orthant, can also be unique [ref], In many applications such as spectral unmixing, elementwise nonnegativity is a natural assumption, which makes NMF particularly suited as a source separation/pattern mining model.

[Example of NMF ?]


Few words on optim
