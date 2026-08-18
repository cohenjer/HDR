---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Identifiability of complete dictionary learning

:::{admonition} Reference
:class: tip
{cite:p}`Cohen2019Identifiability` J. E. Cohen, N. Gillis, "Identifiability of Complete Dictionary Learning", SIAM Journal of Mathematics on Data Science, vol.1 issue 3, pp. 518-536, 2019. [pdf](https://arxiv.org/pdf/1808.08765)`
:::

A core issue in source separation models is ensuring that a user can uniquely identify the output of a separation algorithm with the ground-truth sources. We say that a model is identifiable if it features this uniqueness property. To illustrate what a lack of identifiability means, let a nonnegative matrix $X$ be generated as the product of nonnegative matrices $WH$. If an NMF $X=\tilde{W}\tilde{H}^T$ is computed, in general, the estimated sources $\tilde{W}, \tilde{H}$ are not essentially the same as the ground truth sources $W,H$ (*i.e.*, up to permutation and scaling ambiguities), see [](../../part1/lra.md). In what follows, we summarize our contribution {cite:p}`Cohen2019Identifiability`, in which we study the identifiability of cDL. Note that this work has been extended for sparse NMF {cite:p}`abdolaliDualSimplexVolume2024,abdolaliSimplexStructuredMatrixFactorization2021`.

cDL is a sparse matrix factorization model, where $n$ vectors $M[:,i]$ of dimensions $p$ stacked in a matrix $M\in\mathbb{R}^{p\times n}$ are decomposed as a sparse combinations with at most $k$ nonzero coefficients of $r$ vectors $D[:,q]$ called atoms and stacked in a dictionary matrix $D\in\mathbb{R}^{p\times r}$. cDL can be formalized as a matrix factorization problem, 

$$ \text{Find } D\in\mathbb{R}^{p\times r} \text{ and } B\in\mathbb{R}^{r\times n} \text{ such that } M = DB \; \text{and } \forall i \leq n,~ \|B[:,i]\|_0\leq k, $$ 

where matrix $B$ is the coefficients matrix. We consider the blind setup where both $D$ and $B$ are unknown. In short, the goal of cDL is to estimate the dictionary $D$ and the coefficients matrix $B$ solely from the knowledge of $X$, the number of atoms $r$, and the sparsity level $k$. In what follows, we set the dimensions such that $p\leq r$ and suppose that $D$ has full column rank. This means that the dictionary is not over-complete (which is a usual setup of interest, but harder to study), thus the name cDL. In this setup, without loss of generality in the noiseless setting, we can assume $p=r$ and will do so in the following.

## Identifying $r-1$ dimensional facets

Our identifiability result relies on the geometric interpretation of cDL. cDL can be seen as a subspace clustering problem, since the data points $M[:,i]$ all lie on the union of subspaces of dimension at most $k$. Indeed, denoting $S_i$ the support of column $B[:,i]$, any data point writes as a linear combination of at most $k$ atoms 

$$M[:,i] = \sum_{q\leq k} D[:,S_i[q]]B[S_i[q],i].$$

Since $B$ is unknown, so is the support $S_i$ but we still know that $M[:,i]$ must live in one of the subspaces spanned by any $k$ columns of $D$, therefore for any $i\leq n$, 

$$M[:,i]\in\cup_{\#S\leq k} \text{col}\left(D[:,S]\right).$$

Informally, our theorem states the following. Suppose there exists a cDL factorisation $M=DB$. Then for any sparsity value $k$, the dictionary atoms define $r-1$-dimensional subspaces, called facets, $F_i = \text{col}(D[:,-i])$, that contain the data points. These facets can be used to uniquely recover the atoms in dictionary $D$ up to scalings using 

$$D[:,i] = \cap_{j\leq i}F_i .$$

If the sparsity level $k$ is smaller than $r-1$, these facets might seem oversized tools for studying cDL, since the data points $M[:,i]$ then belong to $k<r-1$-dimensional subspaces. It turns out they are just the right objects to study for identifiability. Indeed, if a unique set of facets can be collected that covers the data points, then $D$ is identifiable (which is not true for lower-dimensional subspaces). The proof idea is straightforward: count how many data points with full dimensionality lie on each facet, and verify that there are enough to uniquely characterize each facet. We can optimize this count using the fact that data points $M[:,i]$ lie on at least $r-k$ facets $F_i$ simultaneously. The minimal number of points we need to identify a single facet is obtained by saturating all facets with as many points as possible, while no facet is identifiable (which happens when a facet contains $r-1$ points). This leads to the following result.

````{prf:theorem} Identifiability of cDL
:label: thm_identif_cDL

A cDL $M = DB$ is essentially unique if on each facet $F_i$ there are $\lfloor \frac{r(r-2)}{r-k} \rfloor +1$ data
points with spark $r$.

````

This is maybe better understood with a visualisation. First, note that we may normalize the dictionary $D$ columnwise without loss of generality, because of scale invariance in the product $DB$. Therefore, we may suppose that atoms live on the simplex. Furthermore, we can normalize the columns of $M$ by the $\ell_1$ norm, and it can be shown {cite:p}`gillisNonnegativeMatrixFactorization2020` that this amounts to normalizing the columns of $B$ by the $\ell_1$ norm as well. In other words, for an $r$-dimensional cDL problem, we can assume that the data points and the atoms are located on the $r-1$ dimensional simplex. This means we can visualize the case $r=3$ in two dimensions, which is convenient. In the figures below, we set $k=2$, and the white circles and squares are two sets of correct dictionary atoms $D[:,i]$ that may be used to write the data point $M[:,i]$ (black dots) as a 2-sparse combination of these atoms.

```{figure} ../../Figures/ex1.*
---
width: 600px
align: center
name: DL_identifiability_ex1
---
Two points per facet.
```

```{figure} ../../Figures/ex3.*
---
width: 600px
align: center
name: DL_identifiability_ex3
---
Four points per facet.
```

```{figure} ../../Figures/ex2.*
---
width: 600px
align: center
name: DL_identifiability_ex2
---
Three points per facet, not identifiable.
```


The first case is not identifiable because there are only two points per facet, while in the second example, four points per facet ensure identifiability. This is proven by {prf:ref}`thm_identif_cDL` since in three dimensions with $k=2$ and $r=3$, $\frac{r(r-2)}{r-k}$ amounts to 3. In the third case, we were unlucky; generically, there should be a single solution, but we carefully chose a setup in which the data points lie at the intersection of two triangles.

## Relation to existing results

Our result complements the existing literature in several ways:
- It contradicts an existing result from Georgiev et. al. {cite:p}`georgievSparseComponentAnalysis2005`. These authors incorrectly assumed that facets can be uniquely identified using $r$ data points (when $k=r-1$), but we provide a counterexample in {numref}`DL_identifiability_ex2`. When $r=3$ and $k=2$, we need in fact $r+1$ data points on the facets.
- It is fully deterministic, therefore it holds with slightly more generality than many existing results based on a Bayesian description of the model, see for instance {cite:p}`gribonvalSparseSpuriousDictionary2015` for an overview.


## Computation of complete Dictionary Learning with tensorly

It is possible to compute a cDL with tensorly using the constrained CP decomposition function. Indeed, one may compute a solution to the cDL problem by solving the optimization problem

$$ \min_{D,B} \|M - DB\|_F^2 + \eta_{\|B[:,i]\|_0\leq k}. $$

Tensorly implements an AO algorithm where each block (here the dictionary $D$ and the coefficients $X$ respectively) are updated using a few iterations of Alternating Descent Method of Multipliers (ADMM). This algorithm was proposed as a flexible optimization framework for constrained matrix and tensor problems by Huang, Sidiropoulos and Liavas {cite:p}`huangFlexibleEfficientAlgorithmic2016` and implemented in tensorly in collaboration with Caglayan Tuna in [PR#284](https://github.com/tensorly/tensorly/pull/284).

This algorithm is demonstrated below on a simulated example. Convergence speed and runtime can be improved using dedicated software and algorithms.

```{code-cell} ipython3
:tags: []

import numpy as np
import tensorly as tl
from tensorly.decomposition._constrained_cp import constrained_parafac
from tensorly.cp_tensor import cp_permute_factors

# ------------- Parameters ---------------
k = 3 # true number of nonzeros in columns of X
kest = 3 # number of nonzeros in the computed solution Xe
noise = 0 # how much noise in the data
rank = 8 # rank of the factorization (number of atoms)
sig = 0.02 # how far from the solution we initialize
oversampling = 2.0 # oversampling factor, <1. has lots of chances to fail

# Getting dimensions from Theorem 1 bound
bound_theorem1 = (np.floor(rank*(rank-2)/(rank-k))+1)*(rank/(rank-k))
print(f"Theorem 13 ensures identifiability with more than {bound_theorem1} data samples")
dims = [rank, int(np.floor(oversampling*bound_theorem1))]
print(f"There are {dims[1]} data samples used in this experiment")

# ------------ Data generation -----------
D = np.random.randn(dims[0],rank)
D = D/tl.norm(D,axis=0)
X = np.random.randn(rank,dims[1])
# sparsify X, Bernouilli Gaussian model
for i in range(X.shape[1]):
    X[:-k,i] = 0
    np.random.shuffle(X[:,i])

M = D@X
Mnoise = M + noise*np.random.randn(*M.shape)

# Init close to solution
D0 = D+sig*np.random.randn(*D.shape)
X0 = X+sig*np.random.randn(*X.shape)
init = (None,[D0,X0.T])

# ------- Decomposition with tensorly -----
out, err = constrained_parafac(Mnoise, rank, hard_sparsity_rowwise={1:kest}, verbose=False, init=init, n_iter_max=500, return_errors=True, tol_outer=0)
print(f"Initial cost was {err[0]}, Final cost is {err[-1]}, {len(err)} iterations were used")
# postprocess estimate by permuting the components optimally
try:
    out, _ = cp_permute_factors((None,[D,X.T]), out)
except:
    print("no permutation can be computed because a zero component is present in the true or estimated coefficients")
Xe = out[1][1].T
De = out[1][0]

# ----------- Error metrics --------------
# Computing True False Positives, True False Negatives
tp = tl.sum((X!=0) & (Xe!=0))
fp = tl.sum((X==0) & (Xe!=0))
fn = tl.sum((X!=0) & (Xe==0)) # always equal to fp if k is chosen optimally
tn = tl.sum((X==0) & (Xe==0))
# Precision, Recall, Accuracy, Fmetric
precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = (tp+tn)/(tp+tn+fn+fp)
fmet = 2*precision*recall/(precision+recall)
print(f"Support estimation: Precision {precision:.2f}, Recall {recall:.2f}, F metric {fmet:.2f}, Accuracy {accuracy:.2f}")

# Estimation of D and X
Denorms = tl.norm(De, axis=0)
De = De/Denorms
Xe = (Denorms*Xe.T).T
rmse_D = tl.norm(D-De)/tl.sqrt(tl.prod(D.shape))
rmse_X = tl.norm(X-Xe)/tl.sqrt(tl.prod(X.shape))
print(f"Relative error on the dictionary: {rmse_D}, on the sparse factor: {rmse_X}")
```

Feel free to play with the parameters. One may observe for instance that as soon as the initial factors are chosen too far from the true solution, most of the time of support is poorly estimated. Similarly, the true support is only recovered in small noise regimes. These two properties, respectively local convergence and robustness, have been studied in the literature, see for instance {cite:p}`liangSimpleAlternatingMinimization2022` and {cite:p}`gribonvalSparseSpuriousDictionary2015` and references therein. However the bounds on the initial error and noise levels are in general hard to compute explicitly.

We can illustrate the previously presented result on identifiability with this simulation. The data points are located uniformly on each facet with sparsity exactly $k$. Each facets requires $\lfloor \frac{r(r-2)}{ r-k }\rfloor +1$ points at least and there are $r-k$ facets. Each point lives in $k$ facets. Therefore we need more than $(\lfloor \frac{r(r-2)}{ r-k } \rfloor+1)\frac{r}{r-k}$ data points to satisfy {prf:ref}`thm_identif_cDL`.

In the above simulation, setting the noise level to zero, if there are fewer points than the bound of {prf:ref}`thm_identif_cDL`, the dictionary and the sparse factors are never recovered because of a lack of identifiability. This is more visible when starting midly far from the true solution. Indeed when the model is not identifiable, the true solution is still a minimizer of the problem, but there are generally infinitely many solutions which may be obtained instead. Try running the simulation several times with the following parameter set:

```python

    k = 3 # true number of nonzeros in columns of X
    kest = 3 # number of nonzeros in the computed solution Xe
    noise = 0 # how much noise in the data
    rank = 8 # rank of the factorization (number of atoms)
    sig = 0.03 # how far from the solution we initialize
    oversampling = .8 # oversampling factor, <1. has lots of chances to fail
```

and observe that the RMSE are generally not small, even when perfect support estimation occurs.

When more samples are drawn, because of the stochastic nature of the repartition of samples on the facets, it is possible that the model is still not identifiable, but as the number of points grows this becomes less likely. The algorithm still falls in local minimizers, but sometimes it finds a solution very close to the true, global solution. Try running the simulation with the same parameters but more samples:

```python

    oversampling = 2.

```

and notice how the RMSE on the dictionary and factor are often smaller then in the undersampled regime, and sometimes close to machine precision.
