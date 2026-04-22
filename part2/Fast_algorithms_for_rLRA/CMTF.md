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
{cite:p}`cabralfariasExploringMultimodalData2016` R. Cabral Farias, J. E. Cohen, and P. Comon, "Exploring multimodal data fusion through
joint decompositions with flexible couplings," IEEE Transactions on Signal Processing, vol. 64, pp. 4830-4844, September 2016. [pdf](https://jeremy-e-cohen.jimdofree.com/publications/%20https:/hal.archives-ouvertes.fr/hal-01158082)

{cite:p}`cohenNonnegativePARAFAC2Flexible2018` J. E. Cohen, R. Bro, "Nonnegative PARAFAC2, a flexible coupling approach", LVA/ICA 2018, [arXiv:1802.05035](https://arxiv.org/abs/1802.05035)

{cite:p}`schenkerOptimizationFrameworkRegularized2021` C. Schenker, J. E. Cohen, E. Acar, "An optimization framework for regularized linearly
coupled matrix-tensor factorization", EUSIPCO2020, 2021 [pdf](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0000985.pdf)

{cite:p}`schenkerFlexibleOptimizationFramework2021` C. Schenker, J. E. Cohen, E. Acar, ``A Flexible Optimization Framework for Regularized Matrix-Tensor Factorizations with Linear Couplings'', IEEE Journal on Selected Topics in Signal Processing, 2020. [arxiv](https://arxiv.org/pdf/2007.09605.pdf) [IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9298877) [Supplementary Materials](https://github.com/AOADMM-DataFusionFramework/Supplementary-Materials) [Matlab code](https://github.com/AOADMM-DataFusionFramework/Matlab-Code)

{cite:p}`roaldPARAFAC2AOADMMConstraints2021` M. Roald, C. Schenker, J. E. Cohen, E. Acar, "PARAFAC2 AO-ADMM: Constraints in all modes", EUSIPCO2021, [arxiv](https://arxiv.org/pdf/2102.02087.pdf)

{cite:p}`roaldAOADMMApproachConstraining2022a` M. Roald, C. Schenker, V. D. Calhoun, T. Adali, R. Bro, J. E. Cohen, E. Acar, "An AO-ADMM approach to constraining PARAFAC2 on all modes", SIAM Journal of Mathematics on Data Science, 2022. [arxiv](https://arxiv.org/abs/2110.01278) [code](https://github.com/MarieRoald/PARAFAC2-AOADMM-SIMODS) [SIMODS](https://epubs.siam.org/doi/10.1137/21M1450033)

{cite:p}`chatzisDCMFLearningInterpretable2025` C. Chatzis, C. Schenker, J. E. Cohen, E. Acar, "DCMF: Learning Interpretable Evolving Patterns from Temporal Multiway Data" EUSIPCO 2025. [arxiv](https://arxiv.org/abs/2502.19367) 
:::



## Background on joint factorization models

Joint factorization models are collections of matrix or tensor factorization problems in which some of the variables are explicitly related. There are at least two motivations for considering this family of problems.
- **Multimodal acquisitions:** Several datasets are acquired informing on the same phenomenon but with different modalities, *e.g.*, EEG+FMRI, NMR+LCMS {cite:p}`Acar2009Unsupervised`, occulometry+EEG {cite:p}`rivetModelingTimeWarping2016`. Joint factorization techniques allow the extraction of latent information from each dataset while leveraging the shared information across datasets to reduce estimation error and enhance uniqueness {cite:p}`Sorensen2015Coupleda`, thereby improving interpretability.
- **Alternative formulations of tensor decompositions:** Joint factorization models are useful for rewriting the classical tensor models [such as CP decomposition](subsec:joint-diagonalisation) and proposing new extended models such as PARAFAC2 {cite:p}`Harshman1972PARAFAC2` {cite:p}`Kiers1999PARAFAC2`, Shift/Conv NMF/CP, PARATUCK2 {cite:p}`usevichApprocheAlgebriquePour2025` and so on. These formulations are also useful for identifiability proofs, as they relate tensor decompositions to matrix low-rank approximation problems.

Let us describe a few important joint factorization models, namely [CPD](subsec:joint-diagonalisation), [CMTF](subsec:cmtf-direct-coupling), and [PARAFAC2](subsec:parafac2-and-variants).

(subsec:joint-diagonalisation)=
### Joint diagonalisation

[CP decomposition](../../part1/lra.md#cp-decomposition) is a particular case of coupled matrix factorization. Indeed, if a tensor $T\in\mathbb{R}^{n_1\times n_2\times n_3}$ follows a rank $r$ CP decomposition with factors $A,B,C$, then for each slice index $k\leq n_3$

$$
T[:,:,k] = A\text{Diag}(C[:,k])B^T.
$$

All the slices $T[:,:,k]$ are therefore jointly diagonalized with left and right bases $A$ and $B$. There is therefore a strong connection between CP decomposition and joint diagonalisation methods such as GSVD, SOBI {cite:p}`belouchraniBlindSourceSeparation1997`, JBSS {cite:p}`Lahat2018New`.

(subsec:cmtf-direct-coupling)=
### CMTF (direct coupling)

One of the most well-known examples of joint decompositions is coined Coupled Matrix and Tensor Factorization (CMTF) {cite:p}`acarUnderstandingDataFusion2013`. CMTF couples a matrix and a tensor on a single mode, where the matrix $Y$ is low-rank, and the tensor $T$ has low CP-rank. CMTF may be formulated as an optimization problem

$$
\argmin{A,B,C,D} \|T - I_r \times_1 A \times_2 B \times_3 C \|_F^2 + \|Y-AD^T\|_F^2.
$$
The first factor of the low-rank approximations of the tensor and the matrix is the same. This means that the same underlying patterns are sought in both datasets along the first mode. In practice, the two terms in the cost function should be balanced based on the SNR of each dataset {cite:p}`cohenJointTensorCompression2016`.

CMTF assumes that the parameter matrix $A$ is shared across measurement modalities. Therefore, CMTF lacks flexibility in many applications, such as genomics and chemometrics, where some components cannot be observed by certain modalities due to physical constraints. It is then necessary to modify the CMTF model to share only a subset of components. This variant of CMTF is referred to as ACMTF {cite:p}`Acar2017ACMTF`. Another extension of CMTF assumes that the two modalities share a parameter matrix through a known linear relationship. This allows modeling possible variations in the sampling rates by linear interpolation {cite:p}`cabralfariasExploringMultimodalData2016`, or coupling only the derivatives of the components {cite:p}`rivetMultimodalApproachEstimate2015`. This work is [discussed below](#general-formulation-linearly-coupled-constrained-cmtf).

(subsec:parafac2-and-variants)=
### PARAFAC2 and variants

Starting from the joint diagonalization formulation of the CPD, one may observe that CPD imposes the coupled slices/matrices $T[:,:,k]$ to have the same row and column factors. In some applications, it could be interesting to relax this assumption and suppose that, rather than only a single mode, say the first one, has a shared factor. Then the coupled factorization model becomes

$$
T[:,:,k] = A\text{Diag}(C[k,:])B_k^T
$$

where each slice $T[:,:,k]$ has a different factor matrix $B_k\in\mathbb{R}^{n_{2,k}\times r}$. The issue with this relaxed formulation is that it is equivalent to an unconstrained matrix factorization of the stacked slices,

$$
\left[T[:,:,1], \ldots, T[:,:,n_3] \right] 
    &= A \left[\text{Diag}(C[1,:])B_1^T, \ldots, \text{Diag}(C[n_3,:])B_{n_3}^T \right], \\
  T_{[1]}  &= A \tilde{B}^T.
$$

The factor matrix $\tilde{B}$ is virtually unconstrained, as any matrix can be written as a stacked matrix of products. This observation means that to obtain a multiway decomposition that does not reduce to unconstrained low-rank matrix approximation, further constraints must be applied on the parameter matrices $B_k$. There are several ways to do so, for instance, imposing some shift-invariance with respect to the slice index (Shift-PARAFAC) {cite:p}`harshmanShiftedFactorAnalysis2003`, or imposing further low-rank structure on the factors (PARATUCK2) {cite:p}`usevichApprocheAlgebriquePour2025`. A popular extension of CPD based on this construction is the PARAFAC2 model, obtained by imposing that the cross product of the $B_k$ matrices is constant,

$$
\forall k\leq n_3, \;B_k^TB_k = \Delta^T\Delta
$$
with $\Delta$ of size $r$ by $r$.

There are several reasons for assuming that the correlation matrices of the second-mode factors are constant across slices. A simple explanation may be obtained by observing that this constraint is equivalent to the reparameterization 

$$
B_k = P_k \Delta
$$

with $P_k$ a left-orthogonal matrix. Therefore, all the slices have the same rank $r$ factor matrix, transformed from slice to slice by a rotation matrix. Orthogonal linear coupling between slices models many linear transformations, such as circular shifts or diffeomorphisms {cite:p}`cohenCurveRegisteredCoupled2018`. The PARAFAC2 model has therefore been used extensively in chemometrics applications where such transformations occur, in particular LCMS and GCMS data {cite:p}`cohenNonnegativePARAFAC2Flexible2018`. The computation of PARAFAC2 with nonnegativity constraints is a topic I have worked on, detailed in [](./NNParafac2.md).

## General formulation: Linearly-Coupled Constrained CMTF

My earliest contribution on joint factorization models was to consider a generalization of CMTF that allows for more complicated coupled relationships {cite:p}`cabralfariasExploringMultimodalData2016`. If CMTF supposes that the same factor matrix can be extracted from several datasets, Linearly-Coupled CMTF (LC-CMTF) assumes that the shared components are linked through a linear relationship. For the particular case of a joint matrix and tensor decomposition, LC-CMTF can be formalized as the optimization problem

$$
\argmin{A_1, B, C, A_2, D, A} \|T - I_r \times_1 A_1 \times_2 B \times_3 C \|_F^2 + \|M-A_2D^T\|_F^2. \\
\text{s.t.}~ \vec{A_1} = H_1 \vec{A} \text{ and } \vec{A_2} = H_2 \vec{A}
$$
for known linear coupling matrices $H_1$ and $H_2$ and a shared, unknown latent factor matrix $A$ which size may differ from $A_1$ and $A_2$.

This model is flexible enough to express several realistic scenarios, such as partially shared components and unaligned data. Two particular linear couplings often encoutered are $[A_2; A_2] = [H_1; H_2]A$ (allows for resampling on a common grid) and $[A_1, A_2] = A[H_1,H_2]$. The latent variable $A$ is introduced to enable the model to easily extend to more than two coupled tensors. In general, a necessary condition for identifiability of $\vec{A}$ is that the matrix $[H_1; H_2]$ is invertible.

In collaboration with Carla Schenker and Evrim Acar {cite:p}`schenkerFlexibleOptimizationFramework2021`, we formalized the linear couplings more clearly and proposed an algorithm based on [Alternating Optimization](../../part1/AlternatingOptimization.md), where each subproblem is solved by the ADMM algorithms {cite:p}`huangFlexibleEfficientAlgorithmic2016`. ADMM is chosen for its flexibility, as we can also impose constraints on the parameter matrices, including the coupled ones. It is also possible to use other loss functions than the Frobenius norm. A MATLAB implementation is available [here](https://github.com/AOADMM-DataFusionFramework/Matlab-Code). Sadly, there are no available Python implementations, so I cannot show examples of the AO-ADMM algorithm in action in this manuscript.

### Other related works

There are a couple of related published works of mine that are worth mentioning here.

#### Temporal-aware CMTF

An extension of the [regularized PARAFAC2 work](../Fast_algorithms_for_rLRA/NNParafac2.ipynb) includes modeling temporal dynamics. A simple way to model the evolution over time of a multivariate vector is through a linear dynamical system, which essentially acts as a linear coupling across slices of a tensor of data. In collaboration with Christos Chatzis, we have studied the results of the time-varying CMTF model and proposed an algorithm to estimate its parameters when the linear dynamical system is known.


#### EEG artifact removal using ocular measurements

In a collaboration with Bertrand Rivet and Rodrigo Cabral-Farias, we used eye-movement signals recorded with an oculometer to remove artifacts in EEG signals {cite:p}`cohenCurveRegisteredCoupled2018`. The signals were acquired by Emmanuelle Kristensen. The goal was to detect eye movements and remove EEG peaks synchronized with them. To identify eye saccades, the data need to be aligned in the temporal domain. The problem may be formulated as a coupled matrix factorization problem, where the sacades are identified by low-rank factorization, and the coupled sacades are related by a time warping.

Instead of using PARAFAC2, which implicitly models time warping, we explicitly model time warping via a diffeomorphism. Our contribution was to propose an algorithm to estimate the parameters of the diffeomorphism along with the low-rank parameter matrices. The low-rank parameter matrices are estimated as in CMTF, while the diffeomorphism parameters are one-dimensional and estimated by binary search. This work was a proof of concept, but the algorithm is too unstable to use in practice, and our earlier work (to which I contributed modestly) using non-parametric coupling with dynamic time warping proved more convenient in the long run {cite:p}`rivetModelingTimeWarping2016`.

