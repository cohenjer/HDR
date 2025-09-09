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
(sec:CMTF)=
# Constrained Coupled Matrix and Tensor Factorization

:::{admonition} Reference
:class: tip
{cite}`cabralfariasJointDecompositionsFlexible2015`
{cite}`cohenNonnegativePARAFAC2Flexible2018`
{cite}`schenkerOptimizationFrameworkRegularized2021`
{cite}`schenkerFlexibleOptimizationFramework2021`
{cite}`roaldPARAFAC2AOADMMConstraints2021`
{cite}`roaldAOADMMApproachConstraining2022a`
:::



## Background on joint factorization models

[Figures for each model in tikz ?]

Joint factorization models are collections of matrix or tensor factorization problems where some of the variables are related explicitly. There are at least two motivations for considering this family of problems.
- **Multimodal acquisitions:** Several dataset are acquired on the same phenomenon but with different modalities, e.g. EEG+FMRI, NMR+LCMS+?, occulometry+EEG {cite}`rivetModelingTimeWarping2016`. Joint factorization techniques allow to extract the latent information from each dataset while making use of the shared information between all dataset to reduce estimation error and enhance uniqueness {cite}`` [todo] and therefore intepretability.
- **Alternative formulations of tensor decompositions:** Joint factorization models are useful for rewriting the classical tensor models [such as CP decomposition](subsec:joint-diagonalisation) and proposing new extended models such as PARAFAC2 {cite}`Harshman1972PARAFAC2` {cite}`Kiers1999PARAFAC2`, Shift/Conv NMF/CP [ref Morten], PARATUCK2 [ref Konstantin] and so on. These formulations are also useful for identifiability proofs since they relate tensor decompositions with matrix low-rank approximation problems.

Let us describe a few important joint factorization models, namely [CPD](subsec:joint-diagonalisation), [CMTF](subsec:cmtf-direct-coupling) and [PARAFAC2](subsec:parafac2-and-variants).

(subsec:joint-diagonalisation)=
### Joint diagonalisation

CP decomposition is a particular case of coupled matrix factorization. Indeed, if a tensor $T\in\mathbb{R}^{n_1\times n_2\times n_3}$ follows a rank $r$ CP decomposition with factors $A,B,C$, then for each slice index $k\leq n_3$

$$
T[:,:,k] = A\text{Diag}(C[:,k])B^T.
$$

All the slice $T[:,:,k]$ are therefore jointly diagonalized with left and right bases $A$ and $B$. However orthogonality is not imposed. There is therefore a strong connection between CP decomposition and joint diagonalisation methods such as GSVD, SOBI {cite}`belouchraniBlindSourceSeparation1997`, JBSS {cite}`` [Lahat Jutten 2018]  

(subsec:cmtf-direct-coupling)=
### CMTF (direct coupling)

One of the most well known examples of joint decompositions is coined Coupled Matrix and Tensor Factorization (CMTF) {cite}`` [acar kolda]. For the particular case of a coupled matrix and tensor on a single mode, where the matrix is low-rank and the tensor has low CP-rank, CMTF may be formulated as the optimization problem

$$
\argmin{A,B,C,D} \|T - I_r \times_1 A \times_2 B \times_3 C \|_F^2 + \|M-AD^T\|_F^2.
$$

- The first factor of the low-rank approximations of the tensor and the matrix are the same. This means that the same underlying patterns are sought in both dataset along the first mode.
- This formulation does not allow for sharing only a subset of components, but variants of CMTF can handle this case in practice {cite}`` [ACMTF].
- The two terms in the cost function should be balanced in practice, based on the SNR of each dataset {cite}`cohenJointTensorCompression2016`.

(subsec:parafac2-and-variants)=
### Parafac2 and variants

Starting from the joint diagonalization formulation of the CPD, one may observe that CPD imposes the coupled slices/matrices $T[:,:,k]$ to have the same row and column factors. In some applications it could be interesting to relax this assumption and suppose rather than only a single mode, say the first one, has a shared factor. Then the coupled factorization model becomes

$$
T[:,:,k] = A\text{Diag}(C[k,:])B_k^T
$$

where each slice $T[:,:,k]$ has a different factor matrix $B_k\in\mathbb{R}^{n_{2,k}\times r}$. The issue with this relaxed formulation is that it is equivalent to an unconstrained matrix factorization of the stacked slices,

$$
\left[T[:,:,1], \ldots, T[:,:,n_3] \right] 
    &= A \left[\text{Diag}(C[1,:])B_1^T, \ldots, \text{Diag}(C[n_3,:])B_{n_3}^T \right], \\
  T_{[1]}  &= A \tilde{B}.
$$

The factor matrix $\tilde{B}$ is virtually unconstrained as any matrix can be written as a stacked matrix of products. This observation means that to obtain a multiway decomposition that does not reduce to unconstrained low-rank matrix approximation, further constraints must be applied on the factor matrices $B_k$. There are several ways to do so, for instance imposing some shift-invariance with respect to the slice index (Shift-Parafac {cite} [todo]), or imposing further low-rank structure on the factors (PARATUCK2) {cite} [todo]. A popular extension of CPD based on this construction is the PARAFAC2 model, obtained by imposing that the cross product of the $B_k$ matrices is constant,

$$
\forall k\leq n_3, \;B_k^TB_k = \Delta^T\Delta
$$
with $\Delta$ of size $r$ by $r$.

There are several reasons for assuming that the correlation matrices of the second-mode factors are constant across slices. A simple explanation may be obtained by observing that this constraint is equivalent to the reparameterization 

$$
B_k = P_k \Delta
$$

with $P_k$ a left-orthogonal matrix. Therefore, all the slices have the same rank $r$ factor matrix, transformed from slice to slice by a rotation matrix. This kind of orthogonal linear coupling between slices allows for several simple transformations across slices like shifts or diffeomorphisms {cite}`cohenCurveRegisteredCoupled2018`. The PARAFAC2 model has therefore been used extensively in chemometrics applications where such transformations occur, in particular LCMS and GCMS data. [link todo ?] [refs]

## General formulation: Linearly-Coupled Constrained CMTF

My main contribution on joint factorization models was to consider a generalization of CMTF that allows for more complicated coupled relationships. If CMTF supposes that the same factor matrix can be extracted from several dataset, Linearly-Coupled CMTF assumes rather than the shared components are linked through a linear relationship. For the particular case of a joint matrix and tensor decomposition, LC-CMTF can be formalized as the optimization problem

$$
\argmin{A_1, B, C, A_2, D, A} \|T - I_r \times_1 A_1 \times_2 B \times_3 C \|_F^2 + \|M-A_2D^T\|_F^2. \\
\text{s.t.}~ \vec{A_1} = H_1 \vec{A} \text{ and } \vec{A_2} = H_2 \vec{A}
$$

for known linear coupling matrices $H_1$ and $H_2$ and a shared, unknown latent factor matrix $A$ which size may differ from $A_1$ and $A_2$.

This model is flexible enough to express several interesting coupling scenarios: [TODO explain more]
- Partially shared components [Eqs et Figures Carla]
- PARAFAC2 with known rotation matrices
- unaligned data

Necessary condition for identifiability of $\vec{A}$: $[H_1; H_2]$ invertible.

The section discuss the design of algorithms for [regularized PARAFAC2](./NNParafac2.ipynb). 