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
# Identifiability of complete Dictionary Learning

:::{admonition} Reference
:class: tip
{cite}`Cohen2019Identifiability` J. E. Cohen, N. Gillis, "Identifiability of Complete Dictionary Learning", SIAM Journal of Mathematics on Data Science, vol.1 issue 3, pp. 518-536, 2019. [pdf](https://arxiv.org/pdf/1808.08765)`
:::

A core issue in source separation models is to ensure that a user can uniquely identify the output of a separation algorithm with the ground truth sources. We say that a model is identifiable if it features this uniqueness property. To illustrate what a lack of identifiability means, let a nonnegative matrix $X$ generated as the product of nonnegative matrices $WH$. If an NMF $X=\tilde{W}\tilde{H}$ is computed, in general the estimated sources $\tilde{W}, \tilde{H}$ are not essentially the same as the ground truth sources $W,H$ (i.e. up to permutation and scaling ambiguities). (TODO link to relevant section of intro?)

This issue is shared by many matrix and tensor decomposition models. In what follows, we summarize our contribution {cite}`Cohen2019Identifiability` where we study the identifiability of complete Dictionary Learning (cDL). Note that this work has been extended for sparse nonnegative matrix factorization {cite}`abdolaliDualSimplexVolume2024,abdolaliSimplexStructuredMatrixFactorization2021`.

cDL is a sparse matrix factorization model, where $n$ vectors $M[:,i]$ of dimensions $p$ stacked in a matrix $M\in\mathbb{R}^{p\times n}$ are decomposed as a sparse combinations (at most $k$ nonzero coefficients) of $r$ vectors $D[:,q]$ called atoms and stacked in a dictionary matrix $D\in\mathbb{R}^{p\times r}$:

$$ M = DB$$ 

where $\|B[:,i]\|_0\leq k$ for all column $B[:,i]$ of the coefficients matrix $B\in\mathbb{R}^{r\times n}$. We consider the blind setup where both $D$ and $B$ are unknown. In short, the goal of cDL is to estimate the dictionary $D$ and the coefficients matrix $B$ solely from the knowledge of $X$, the number of atoms $r$ and the sparsity level $k$. In what follows, we set the dimensions such that $p\leq r$ and suppose that $D$ has full column rank. This means that the dictionary is not over-complete (which is a usual setup of interest, but harder to study), thus the name complete DL. In this setup, without loss of generality in the noiseless setting we can assume $p=r$ and will do so in the following.

## Identifying $r-1$ dimensional facets

Our identifiability result relies on the geometric interpretation of cDL. cDL can be seen indeed as a subspace clustering problem, since the data points $M[:,i]$ all lie on the union of subspaces of dimension at most $k$. Indeed, denoting $S_i$ the support of column $B[:,i]$, any data point writes as a linear combination of at most $k$ atoms 

$$M[:,i] = \sum_{q\leq k} D[:,S[q]]B[S[q],i].$$

Since $B$ is unknown, so is the support $S_i$ but we still know that $M[:,i]$ must live in one of the subspaces spanned by any $k$ columns of $D$, therefore for any $i\leq n$, $M[:,i]\in\cup_{|S|\leq k} \text{span}\left(D[:,S]\right)$.

Informally, our theorem states the following. Suppose there exist a cDL factorisation $M=DB$. Then the dictionary atoms define $r-1$ dimensional subspaces called facets $F_i = \text{span}(D[:,-i])$ that contain the data points. These facets can be used to uniquely recover the atoms in dictionary $D$ up to scalings (TODO def $[-i]$) using 

$$D[:,i] = \cap_{j\leq i}F_i .$$

If the sparsity level $k$ is smaller than $r-1$, these facets might seem over-sized tools to study cDL since in that case data points $M[:,i]$ rather belong to $k<r-1$ dimensional subspaces. It turns out they are just the right objects to study for identifiability. Indeed if a unique set of facets may be collected that covers the data points, then $D$ is identifiable (which is not true for smaller dimensional subspaces). The proof idea is then straightforward: count how many data points with full dimensionality live on each facets, and check that there are enough data points to uniquely characterize each facet. We can optimize this count using the fact that data points $M[:,i]$ lie on at least $r-k$ facets $F_i$ simultaneously. The minimal number of points we need to identify a single facet is obtained by saturating all facets with as many points as possible while no facet is identifiable (which happens when a facet contains $r-1$ points). This leads to the following result.

````{prf:theorem} Identifiability of cDL
:label: thm_identif_cDL

An LRSCA decomposition $M = DB$ is essentially unique if on each facet $F_i$ there are $\lfloor \frac{r(r-2)}{r-k} \rfloor +1$ data
points with spark $r$.

````

This is maybe better understood with a visualisation. First, note that we may normalize the dictionary $D$ columnwise without loss of generality, because of scale invariance in the product $DB$. Therefore we may suppose that atoms live on the simplex. Furthermore, we can normalize the columns of $M$ with the $\ell_1$ norm, and it can be shown {cite}`gillisNonnegativeMatrixFactorization2020` that this amount to normalizing the columns of $B$ also by the $\ell_1$ norm. In other words, for an $r$-dimensional cDL problem, we can assume that the data points and the atoms are located on the $r-1$ dimensional simplex. This means we can visualize the case $r=3$ in two dimensions which is convenient. In the figures below, we set $k=2$, and the white circles and squares are two sets of correct dictionary atoms $D[:,i]$ that may be used to write the data point $M[:,i]$ (black dots) as a 2-sparse combination of these atoms.

```{figure} ../Figures/ex1.*
---
width: 600px
align: center
name: DL_identifiability_ex1
---
Two points per facet.
```

```{figure} ../Figures/ex3.*
---
width: 600px
align: center
name: DL_identifiability_ex3
---
Four points per facet.
```

```{figure} ../Figures/ex2.*
---
width: 600px
align: center
name: DL_identifiability_ex2
---
Three points per facet, not identifiable.
```


The first case is not identifiable because there are only two points per facets, while in the second example, four points per facets ensure identifiability. This is proven by {prf:ref}`thm_identif_cDL` since in three dimensions with $k=2$ and $r=3$, $\frac{r(r-2)}{r-k}$ amounts to 3. In the third case we got unlucky; generically there should be a single solution, but we carefully chose a setup where the data points are located at the intersection of two triangles.

## Relation to existing results

Our result complements the existing literature in several ways:
- It contradicts an existing results from Georgiev et. al. {cite}`georgievSparseComponentAnalysis2005`. These authors incorrectly assumed that facets can be uniquely identified using $r$ data points (when $k=r-1$) but we provide a counter example in Figure {ref}`DL_identifiability_ex2`. When $r=3$ and $k=2$ we need in fact $r+1$ data points on the facets.
- It is fully deterministic, therefore it holds with slightly more generality than many existing results based on a Bayesian description of the model, see for instance {cite}`gribonvalSparseSpuriousDictionary2015` for an overview.

In the next section, we compute cDL using Tensorly!