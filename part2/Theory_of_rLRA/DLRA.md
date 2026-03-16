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

# Dictionary-based Low-Rank Approximations

:::{admonition} Reference
:class: tip

{cite:p}`cohenDictionarybasedTensorCanonical2018` J. E. Cohen, N. Gillis, "Dictionary-based Tensor Canonical Polyadic Decomposition", IEEE Transactions on Signal Processing, vol.66 issue 7, pp. 1876-1889 2018. [pdf](https://arxiv.org/pdf/1704.00541.pdf)

{cite:p}`cohenSpectralUnmixingMultiple2018` J. E. Cohen, N. Gillis, "Spectral Unmixing with Multiple Dictionaries", IEEE Geoscience and Remote Sensing Letters, vol.15 issue 2, pp. 187-191, 2018. [pdf](https://arxiv.org/pdf/1711.02883.pdf)

{cite:p}`cohenDictionaryBasedLowRankApproximations2022` J. E. Cohen, "Dictionary-based low-rank approximation and the mixed sparse coding problem", Frontiers in Applied Mathematics and Statistics, 2022. [frontiers](https://www.frontiersin.org/article/10.3389/fams.2022.801650) [arxiv](http://arxiv.org/abs/2111.12399) [codes](https://github.com/cohenjer/dlra)

{cite:p}`cohenComputingProximalOperator` J. E. Cohen, "Computing the proximal operator of the l1 induced matrix norm", [arxiv 2005.06804](https://arxiv.org/abs/2005.06804), 2020

**Note:** the publication in the Frontiers journal was in a special issue with editors that I know and trust, with the intention to start a new journal dedicated to tensors. The review process was comparable to that of a medium-level publication venue. I would not repeat the experiment, and the editors have since stopped collaboration with Frontiers.
:::

When decomposing matrices and tensors, a reasonable assumption is that the factor matrices, although unknown, are generated from a known set of templates. Let us provide two typical examples.
- Factors contain smooth components, say the first factor matrix $A$. A simple way to account for smoothness in a decomposition is to fix a set of smooth splines stored as columns of a dictionary matrix $D$, and assume that each component writes as the sum of a few splines. Then each column of factor $A$ writes as a sparse combination of columns of the splines dictionary, $A=DX$ with $\|X[:,i]\|_0\leq k$ for each atom $i$, where the integer $k$ is the number of smooth splines that can add up to build each component. [TODO illustration?] 
- The components are contained in a large dictionary $D$, but it is not known which column of $D\in\mathbb{R}^{m\times d}$ must be picked exactly. This can be challenging, in particular when the number of atoms $d$ is much larger than the decomposition rank $r$. Again in the case of factor matrix $A$, this translates into a relationship $A[:,i]=D[:,\sigma_i]$ for all column index $i$, where $sigma$ is an injection of $[1,d]$ into $[1,r]$. Equivalently, one may write $A=DX$ with $\|X[:,i]\|_0=1$ for all atoms $i$ and $X\in\{0,1\}^{m\times d}$. This may happen in applications such as spectral unmixing, where the components in the spectral mode represent reflectance spectra that are typically known, whereas the components present in the data may be unknown. The decomposition problem is to unmix the sources and identify their components using their spectral signatures.

A mathematical formulation that encompasses both these problems, using the Frobenius norm as a metric, is therefore the following:

$$ \min_{B\in\Omega_B,\; X\in\Omega_X,\; \|X[:,i]\|_0\leq k\; \forall i\leq r} \|Y - DXB^T \|_F^2 $$

where $k$ is the known sparsity level of columns of matrix $X$. The minimization over the matrix $B$ is over a set $\Omega_B$ that we may adapt so that the above problem covers both tensor decompositions and matrix decompositions. For instance, by imposing that $B$ is the Khatri-Rao product of two factor matrices, in the case of order-three CP decomposition. Similarly, the constraint set $\Omega_X$ is used to impose additional constraints on the matrix $X$, such as elementwise nonnegativity. This problem is referred to as the Dictionary-based Low-Rank Approximation (DLRA) in the following.

One may observe that this minimization problem, in general, is very difficult. It is a generalization of sparse coding with $B$ set to the identity and $r=1$, so it is NP-hard. Moreover, it generalizes matrix factorization problems such as nonnegative matrix factorization when $D$ is the identity, $k=m$, and the constraint sets are the nonnegative orthants. Even if we know the support of the solution $X$, *i.e.*, the positions of the nonzeros in the matrix $X$, one still needs to estimate both the nonzeros in $X$ and the right-hand-side matrix $B$, which is challenging [TODO cite Gribonval].

Therefore, to better understand the nature of the solutions to DLRA and to design efficient heuristics to solve this problem, we have studied two particular cases in two different publications:
- The special case of [1-sparse DLRA](./onesparseDLRA.ipynb) when $k=1$ {cite}`cohenDictionarybasedTensorCanonical2018`
- The subproblem of [estimating factor matrix](./MixedSparseCoding.ipynb) $X$ with fixed $B$ {cite}`cohenDictionaryBasedLowRankApproximations2022`, and its usage in solving the DLRA problem with alternating optimization.
[A third contribution](./multiple_dictionaries.ipynb) {cite}`cohenSpectralUnmixingMultiple2018` has also been proposed in the context of spectral unmixing under the pure pixels assumption.

These first two problems are studied in the following subsections. Python packages for running experiments are located in two dedicated repositories, respectively [dlra](https://github.com/cohenjer/dlra) and [dlraos-essentials](https://github.com/cohenjer/dlraos-essentials).

To get a taste of the kind of final result one may obtain using DLRA, here are examples of outputs of DLRA algorithms that we proposed for each of the examples introduced above. These experiments are explained in detail in the relevant subsections.

[TODO reuse figures from next section smooth CP and multidico NMF]