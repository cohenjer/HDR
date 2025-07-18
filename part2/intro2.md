---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Contributions to regularized low-rank approximations
subtitle: from theory to applications

Since the start of my PhD thesis in 2013, I have been focusing on advancing the current state of knowledge for regularized Low-Rank Approximations (rLRA). My works cover a diverse list of topics ranging from understanding the properties of the solutions of rLRA to proposing new applications for these models. A significant section of my work is however dedicated to algorithm design to compute solutions to rLRA efficiently in various setups. Below is a short introduction to what is rLRA and why regularizations are useful, but the reader should refer to Part 1 [todo link] of this book for more background on LRA.

TODO: move this to foreword/intro and here only introduce my work ?

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

[block?]
In the last ten years, I have focused my research on LRA models as tools for extracting meaningful information out of matrices and tensors, using regularizations to enhance interpetability. These regularizations often include nonnegativity, which has become my specialty. There is a significant body of litterature on these models, covering variants of rLRA alongside indentifiability theory (NMF [], Canonical Polaydic Decompostion [], Tucker Decomposition [], Parafac2 [] to name a few), algorithms to compute rLRA using in particular rather recent developments on non-smooth and non-convex optimization [refs], and a myriad of applications in chemometrics [], neuroscience [], statistical inference [anankudmar], spectral unmixing for remote sensing [] or microscopy imaging [], music information retrieval [], telecommunications [], psychometry [] \ldots.

Despite the significant amount of existing works on rLRA, their use by practitionners is often quite difficult. Indeed there are many exciting theoretical and practical issues that remain to be dealt with:
- Optimization: multi-block, nonconvex, --> convergence ?
- Available side information (semi-supervised) --> how to account for it; 
- Multimodality
- Modeling: nature of the solutions? guarantees ?
- For Tensors: software tools, large scale contractions, GPU support and so on


My work in the last ten years has been dedicated to proposing (partial) solutions to these issues.
I have grouped my contributions in three sections, dealing respectively with theoretical contributions, applications-oriented contributions and algorithmic-focused contributions. However in most of these works, all three aspects (theory, algorithms, applications) are intertwined, so this is no a strict segmentation of my work.

FOR THE EVAL A VAGUE --> summarize
TODO: fill in with links
## Theory of rLRA contributions

## Applications of rLRA contributions

## Algorithms for existing rLRA models contributions

## Others
Lever arm
Birkov
Okular-EEG
Temporal-aware CMTF

## Table of contents (necessary?)

a refaire à la fin.

%```{toctree}
%:glob:
%:maxdepth: 2

%Theory_of_rLRA/intro
%Fast_algorithms_for_rLRA/intro
%Applications_of_rLRA/intro
%```

This tells you things that are in chapter 2 (contributions, core of HDR)