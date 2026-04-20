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


# Summary of HDR contents


## Some personal context

When I was six years old and starting to learn to play the piano, I had fun guessing the notes my tutor was playing while turning my back to the piano. This was a pleasant but easy game as I discovered that I could hear the pitch class of individual notes as they were played. I thought everyone else could also hear these names and, therefore, transcribe music easily. It was only a few years later, when joining my first music school, that I discovered I was mistaken and that music transcription is actually really hard. For everyone, in fact, as even with perfect pitch, transcribing polyphonic instruments becomes a compromise between instantaneous recognition and pattern matching with template harmonies stored in one's memory.

As I grew older, I began to enjoy the broader concepts of sound and music processing, the Fourier transform, and their connections to how humans perceive sound, as well as the symbolic and theoretical description of music. It was with great interest that I realized, by the end of my PhD thesis, that many questions in the field of music information retrieval, which encompasses most of these concepts, could be addressed at least partially using a core concept I had been working on: low-rank approximations of matrices and tensors. In particular, the design of regularizations and constraints was key to achieving reasonable performance, and I felt that, with my skill set, I could improve on the existing state of the art in that domain, particularly in automatic music transcription.

It turns out that low-rank approximations, which are the main topic of this manuscript, are quite heavily beaten in practice by deep learning approaches when it comes to transcribing piano pieces. However, this anecdote shows that low-rank approximations are ubiquitous and critical to unsupervised machine learning. While some researchers specialize in one specific application of low-rank approximations, I am rather interested in the applied mathematical side of low-rank approximations, in particular in modeling and training aspects. I am convinced that low-rank approximation-based learning algorithms, in many applications such as automatic music transcription, can still be improved in light of the new results obtained over the last decade, a very modest part of which I am responsible for. It would bring me great joy to see anyone design a state-of-the-art automatic transcription algorithm that relies heavily on low-rank approximations, combined, for instance, with deep learning, or maybe for rare instruments with scarce training data. I firmly believe that this objective is within reach in the coming years.

I write this "Habilitation à Diriger des Recherches" manuscript with the hope of demonstrating that regularized low-rank approximations is a rich topic with many exciting open questions. I also hope that fellow researchers who are willing to read through all this content will learn about some lesser-known results, tricks, and algorithmic details that can help them in their work on regularized low-rank approximations.

## Why Regularized Low-rank approximations

Low-rank approximations are workhorse methods in several unsupervised machine learning tasks such as dimensionality reduction {cite:p}`Tucker1966Some, Golub1989Matrix, hotelling1992relations` and blind source separation {cite:p}`harshman1970foundations, Paatero1997weighted, Lee1999Learning`. The most prominent LRA method in machine learning is probably Principal Component Analysis, which numerically boils down to computing a singular value decomposition, and is therefore efficiently computed.

```{figure} ../Figures/Pca.png
---
width: 700px
align: center
name: fig:pca
---
Illustration of Principal Component Analysis. The centered data matrix $M$ lies in a low-dimensional subspace $col(U)$. If the orthogonal matrix $U$ is computed from the data matrix $M$, then the columns of $U$ are the (unscaled) principal components. A low-rank approximation is obtained by projecting the data points on a subset of the principal components, here on the first component.
```

Formally, for matrix data, a (real) low-rank approximation problem is an optimization problem of the form

$$ \min_{U\in\mathbb{R}^{m\times r},V\in\mathbb{R}^{n\times r}} f(Y, UV^T) $$
where integer $r$ is the rank of the approximation $UV^T\approx Y$, with $r\ll m,n$, and $f$ is a cost function, typically $f(Y, X) = \|Y-X\|_F^2$.

```{margin}
We present the case of matrix factorization here for simplicity, but the same logic applies to higher-order factorizations.
```

In unsupervised machine learning tasks such as blind source separation, users often seek to give a physical meaning to the factor matrices $U$ and $V$. In fact, blind source separation can be viewed as an inverse problem, in which both the mixing matrix $U$ and the sources $V$ are unknown. In this context, there exists a true pair $(U_0, V_0)$ that the user seeks to recover. A necessary condition to correctly interpret a reconstructed pair is the uniqueness of this reconstructed solution. Note that LRA, as defined above, is never unique, since any product $UV$ can also be written $UPP^{-1}V$ for an invertible matrix $P$ of size $r\times r$.

A widely used approach in signal processing and machine learning to restrict the set of solutions to an inverse problem is to incorporate prior information on the parameters in the form of constraints or regularizations. Regularized LRA can be formulated as 

$$ \min_{U\in \mathbb{R}^{m\times r},V\in\mathbb{R}^{r\times n}} f(Y, UV) + g_{U}(U) + g_{V}(V) $$
where $g_U$ and $g_V$ are regularizations promoting certain properties in solutions, such as smoothness, sparsity, or nonnegativity. Depending on the choice of regularizations, the solutions may then be unique.

```{note}
Regularizations typically apply to each factor $U$ and $V$ independently. Indeed, the main identifiability issue in LRA is the rotation ambiguity $UV^T = UP\left(VP\right)^T$ for an orthogonal matrix $P$. A regularization $g(UV^T)$ would not discriminate between two solutions identical up to permutation and therefore is not enough to ensure the uniqueness of rLRA solutions.
```

```{margin}
Essential uniqueness is the uniqueness up to permutations and scaling ambiguities inherent to LRA models.
```

PCA is a constrained LRA model: factors are imposed to be orthogonal matrices. This allows us to obtain a model with essentially unique factors (under the mild condition that singular values must be distinct {cite:p}`Golub1989Matrix`), but orthogonality may not be satisfied by the ground-truth factors $U^*$ and $V^*$. On the other hand, Nonnegative Matrix Factorization (NMF), obtained by setting $g_U$ and $g_V$ to characteristic functions of the nonnegative orthant, can also be unique {cite:p}`gillisNonnegativeMatrixFactorization2020`. In many applications, such as spectral unmixing, elementwise nonnegativity is a natural assumption, which makes NMF particularly suited as a source separation/pattern mining model. A typical example of NMF usage is in spectral unmixing for remote sensing, as illustrated in Figure [TODO below]

```{figure} ../Figures/Hsi_nmf.png
---
width: 700px
align: center
name: fig:HSI_nmf
---
Nonnegative Matrix Factorization performs spectral unmixing on hyperspectral images, identifying essential spectra that are present in the scene, as well as their proportion in each pixel stored in abundance maps. The images are vectorized in the data matrix and the abundance maps.
```

A natural extension of matrix LRA is tensor LRA, which applies to input data with more than two dimensions. Tensor LRA has several interesting properties, including, in the case of the Canonical Polyadic Decomposition (CPD), identifiability without regularization {cite:p}`Kruskal1977Three`. In practice, however, regularization, and in particular nonnegativity, is often useful to improve the interpretability of the estimated parameters from tensor LRA. Chapter [](../part1/lra.md) describes known results on matrix and tensor LRA in more detail.

```{figure} ../Figures/Cpd_simple.png
---
width: 700px
align: center
name: fig:cpd_simple
---
Canonical Polyadic Decomposition (CPD) of rank $r$ of a data tensor $T$. The CPD can be viewed as a multilinear diagonalization technique, with bases $A$, $B$ and $C$ respectively spanning the columns, rows and fibers of the tensor.
```

Once an LRA model has been chosen for a particular application, as in most machine learning problems, the model parameters need to be estimated so that the model fits the data. Optimization problems encountered in regularized LRA problems are often continuous but non-smooth, for instance, due to the nonnegativity constraint, and non-convex due to the presence of the product of variables in LRA. Convexity, however, often holds when the cost function is considered only with respect to one of the parameter matrices. Consider the NNLS problem with Frobenius data fitting

$$ \min_{U\in \mathbb{R}_+^{m\times r}} \|Y -  UV^T\|_F^2, $$
which is essentially NMF, where the matrix $V$ is fixed; the cost with respect to $U$ is convex. Therefore, for the applied mathematician working on regularized LRA, it is important to be familiar with both non-smooth constrained optimization and multi-convex optimization problems, often solved with alternating optimization strategies. The specific case of nonnegatively-constrained regressions is of critical importance to this manuscript and is covered in depth in [](../part1/nnls.md), while a summary of known results regarding alternating optimization strategies is provided in [](../part1/AlternatingOptimization.md).

```{figure} ../Figures/nnlsU.png
---
width: 700px
align: center
name: fig:nnlsU
---
Graphical representation of the Nonnegative Least Squares (NNLS) problem. NNLS consists in the orthogonal projection of the data vector $y$ on the cone $\cp{U}$ spanned by the nonnegative linear combinations of the columns of matrix $U$.
```

There is a significant body of literature on regularized LRA (rLRA) with a myriad of applications, for instance in chemometrics {cite:p}`Bro1996Multiway`, neuroscience {cite:p}`cichockiTensorDecompositionsNew`, statistical inference {cite:p}`Anandkumar2015When`, spectral unmixing for remote sensing {cite:p}`Bioucas-Dias2012Hyperspectral` or microscopy imaging {cite:p}`harigaJointReconstructionSpectral2024`, music information retrieval {cite:p}`smaragdis2003non`, telecommunications {cite:p}`Sidiropoulos2000Parallel`, psychometry {cite:p}`Harshman1972PARAFAC2` and more. There is no dedicated section to reviewing all these applications in this manuscript. Datasets from hyperspectral imaging and music information retrieval are used to illustrate the various contributions, and my contributions to these applications are detailed in [a separate chapter](../part2/Applications_of_rLRA/intro.md).

## Contributions to regularized low-rank approximations

Since the start of my PhD thesis in 2013,  I have focused my research on LRA models as tools for extracting meaningful information out of matrices and tensors, using regularizations to enhance interpretability. These regularizations often include nonnegativity, which has become my specialty. My work covers a diverse range of topics, from understanding the properties of solutions to rLRA to proposing new applications for these models. A significant portion of my work is dedicated to algorithm design for efficiently computing solutions to rLRA across various setups.

### Challenges in rLRA

Despite the large body of existing work on rLRA, its use by practitioners is often quite difficult. Indeed, there are many exciting theoretical and practical issues that remain to be dealt with:
- **Optimization problems** stemming from fitting rLRA models are multi-block, non-convex, and often non-smooth. Recent advances in numerical optimization, including acceleration, majorization-minimization, generalized projection operators, and alternating optimization, can all be leveraged to improve upon the state-of-the-art. It is also unclear, in practice, which optimization algorithm to use depending on the context.
- **Side information** is often available alongside the matrix or tensor data, which transforms the unsupervised rLRA learning problem into a semi-supervised or fully supervised problem. This information can take diverse forms: a known library of templates for the unknown sources, a downstream task with available training data, deep priors for the unknown sources, such as denoisers, sparsity in well-known bases, sparsity level of the sources, among others. Accounting for this side information in rLRA is often not straightforward, both from a modeling and from a training perspective.
- **Multimodality** has emerged as an important topic in source separation, where the same phenomenon is observed through different sensing devices. A typical example would be acquiring brain activity through EEG and fMRI jointly. The joint processing of the acquired dataset can be performed by several rLRA. The research questions on joint rLRA typically revolve around the coupling model for the various datasets, and the design of flexible algorithms that allow solving a wide range of multimodal source separation problems, in particular when the datasets have different dynamics, noise levels, and even different fields (such as floats vs count vs categorical data).
- The rLRA models themselves still have elusive properties. An important example is nonnegative Tucker decomposition, for which **identifiability** is unclear in many cases, despite recent advances on the topic {cite:p}`sahaIdentifiabilityNonnegativeTucker2025`. Generally, a user needs guarantees about the nature of rLRA solutions, and such theoretical guarantees are, in general, very challenging to obtain, especially when the hypotheses must be verifiable in practice.
- The **scientific software ecosystem** is a dimension often overlooked but critical in practice. Available software for tensor decomposition is largely bloated; see this survey, which lists over 70 tensor packages scattered across the internet {cite:p}`psarrasLandscapeSoftwareTensor2022`. However, software packages dedicated to, or capable of handling at least partially, rLRA are scarce. For tensor decompositions, many packages focus on the decomposition itself but do not make use of efficient large-scale contractions on CPU or GPU that are crucial for up-scaling {cite:p}`smithTensormatrixProductsCompressed2015,liInputadaptiveInplaceApproach2015,g.a.smithOptEinsumPython2018`. Another critical issue is the actual implementation of rLRA algorithms, that requires non-trivial caching of expensive operations and speed-momery trade-offs that are research topics by themselves {cite:p}`kayaHighPerformanceParallel2016`.

### Improving on theory, algorithms, and applications

My work has been dedicated to proposing (partial) solutions to these issues. I have grouped my contributions into three parts: theoretical contributions, application-oriented contributions, and algorithmic-focused contributions. However, in most of these works, all three aspects (theory, algorithms, applications) are intertwined, so this is not a strict segmentation of my work.

There is no dedicated chapter in this manuscript to software development. Instead, simple implementations of the methods under scrutiny are either provided inline, imported from Tensorly, which I have co-developed since 2019, or imported from a local set of methods developed specifically for this manuscript. Software development is an important part of scientific contributions in rLRA and, more generally, in applied mathematics. In my opinion, it is important to show the exact code being run for experiments and demos so that readers can be more convinced of the results' exactitude.

The three main sections of this manuscript are
- [Theory of rLRA contributions](../part2/Theory_of_rLRA/intro.md), where I summarize my contributions to the analysis of several sparse models: [Dictionary-based LRA](../part2/Theory_of_rLRA/DL_identifiability.md), [sparse nonnegative least-squares](../part2/Theory_of_rLRA/sparse_nnls.ipynb). I also study [the impact of scaling ambiguity on rLRA solutions](../part2/Theory_of_rLRA/HRSI_theory.md), which can induce unexpected group-sparsity at the component level.
- [Algorithms for existing rLRA models contributions](../part2/Fast_algorithms_for_rLRA/intro.md), where I summarize my contributions to algorithm design for rLRA. My contributions concern [flexible algorithms for multimodal rLRA](../part2/Fast_algorithms_for_rLRA/CMTF.md), [heuristic inertial acceleration for alternating optimization algorithms](../part2/Fast_algorithms_for_rLRA/inertial_BCD.ipynb), and more recently, [the design of tight surrogates](../part2/Fast_algorithms_for_rLRA/mSOM.md) for nonnegative optimization problems, including NMF in non-Euclidean settings. I have also worked on data-driven regularization for LRA, in particular on [unrolling multiplicative updates](../part2/Fast_algorithms_for_rLRA/UnrolledNMF.md) for NMF. Finally, I revisited a work from my PhD thesis on [projected least squares to compute nonnegative LRA](../part2/Fast_algorithms_for_rLRA/proco-als.ipynb).
- [Applications of rLRA contributions](../part2/Applications_of_rLRA/intro.md). My expertise in terms of applications of rLRA is geared towards both [spectral imaging](../part2/Applications_of_rLRA/single_pixel.ipynb) and music information retrieval, in particular [automatic transcription](../part2/Applications_of_rLRA/AMT.md). My interest in spectral imaging stems from the fact that LRA is intimately linked to additive mixtures, which accurately describe mixtures in spectral acquisition techniques such as fluorescence spectroscopy or hyperspectral remote sensing. Being a semi-professional pianist and singer, I have a personal interest in music that drives my research in music information retrieval. Other applications, mostly related to biomedical imaging, are more punctual collaborations and are described below.

More details on each topic, including a detailed summary of the contributions of each part, are provided respectively in the introductions of [Theory of rLRA](../part2/Theory_of_rLRA/intro.md), [Algorithms for rLRA](../part2/Fast_algorithms_for_rLRA/intro.md), and [Applications of rLRA](../part2/Applications_of_rLRA/intro.md).

### Other works
Several of my works are not discussed in this manuscript because they would otherwise make it too long. These works are often targeted toward specific applications or require specific mathematical tools that I want to avoid introducing here. Below is a short paragraph for each of these works.

#### Music segmentation

Maybe the most impactful work that I chose not to discuss in this manuscript is related to the PhD thesis of Axel Marmoret {cite:p}`marmoret2020uncovering,marmoretNonnegativeTuckerDecomposition2022, marmoretUnsupervisedMachineLearning2022`. Low-dimensional models such as Tucker decomposition and auto-encoders are used to compress information from a recording of a full song {cite:p}`smithNonnegativeTensorFactorization2018`. This compression highlights the similarities and dissimilarities between bars in the song, which then helps a dedicated dynamic program solver to perform an automatic segmentation of the song. Despite the method being unsupervised (a supervised beat-tracking algorithm is still used to cut the song into bars), the results are very encouraging if the parameters of the compressed models can be chosen optimally. This choice is difficult in practice, which limits the method's performance.

```{figure} ../Figures/NNTucker_Music.png
---
width: 700px
align: center
name: fig:NNTucker_music
---
Nonnegative Tucker decomposition can be used to extract meaningful patterns from a Time-Frequency-Bar tensor, obtained from an audio recording of a song by slicing the audio barwise and applying individual short-time Fourier transforms. The bar activation patterns are then used as low-dimensional features to perform automatic segmentation on an auto-similarity matrix, revealing the hidden structure of the song.
```

This work was rather visible in the music information retrieval community when released. It has launched the academic career of Axel Marmoret, who is today working, among other areas, on LRA in conjunction with deep learning, with applications to music information retrieval. His PhD manuscript describes our contributions in detail.

### Birkhoff-von Neumann decomposition of stochastic matrices

The Birkhoff-von Neumann (BvN) decomposition of a stochastic matrix is the problem of writing a stochastic matrix $M$ as a minimal number of permutation matrices $P_i$,

$$ \text{Find the smallest } k\in\mathbb{N} \text{ such that } M = \sum_{i=1}^{k} \alpha_i P_i, $$
where $\alpha_i$ are nonnegative coefficients; both coefficients and permutation matrices have to be estimated from the input matrix. This long-standing NP-hard problem can be tackled by a variety of existing heuristics. In collaboration with Bora Ucar and Damien Lesens, we proposed viewing BvN as a sparse coding problem and adapting existing sparse coding solvers, such as orthogonal matching pursuit, to compute solutions {cite:p}`lesensOrthogonalMatchingPursuitBased2024`. The contribution is to link the selection of the best atom (here, the best permutation matrix) to several graph matching algorithms, making the selection both fast and effective. Our method is competitive with the state-of-the-art. In a follow-up work driven by Damien Lesens, we extended BvN algorithms to the symmetric case {cite:p}`lesensAlgorithmsSymmetricBirkhoffVon2026`. We are the first to propose an implementable algorithm to solve symmetric BvN. 

These works fall slightly outside the realm of rLRA. My role for the second contribution on symmetric BvN, in particular, was rather minor. Therefore, I chose not to discuss this work further. Damien Lesens is now pursuing a PhD co-supervised with Bora Ucar on optimization algorithms for nonnegative LRA, see the [Perspectives](../part3/Perspectives.md).

### Sparse separable NMF

Separable NMF is a regularized variant of NMF with nice identifiability and algorithmic properties, see [](../part1/lra.md#regularized-nmf). In a joint work with Nicolas Nadisic, Arnaud Vandaele, and Nicolas Gillis, we studied a slightly more general model called sparse separable NMF, which assumes that the data points are sparse convex combinations of a few data vectors {cite:p}`nadisicSparseSeparableNonnegative2021`. An algorithm is proposed ressembling orthogonal matching pursuit that identifies the template components from the data, and compute sparse regressions for each data point using a previously [proposed sparse NNLS algorithm](../part2/Theory_of_rLRA/sparse_nnls.ipynb). We show, however, that the problem is in general NP-hard, unlike separable NMF.

### Optical Biopsy NMF

A contribution closely related to [](../part2/Applications_of_rLRA/Single_pixel_spectral_imaging.md) performs segmentation of cancerous cells from videos of the brain during surgery using separable NMF {cite:p}`careddaSeparableSpectralUnmixing2023`, in collaboration with Bruno Montcel, Charly Carrera, and others. For this work, my contribution is to propose a robust variant of the well-known direct algorithm for rank-two exact separable NMF in the Frobenius norm. The components estimated from the rank-two NMF are then used to estimate the spatial oxygenated blood flows, which help locate the glioma.

### Lever arm tensor decomposition

In a collaboration with Raphaël Dumas and Joris Claude, we studied a set of lever arm measurements from leg muscles acquired on patients with knee injuries during walking {cite:p}`claudeConstrainedTensorDecomposition2024`. This study revealed synergies among leg muscles. These muscle activations can be clustered to summarize the complex action of the human body during walking into a few descriptors. My contribution to this work was to help design the constrained tensor decomposition model. The data tensor is quite difficult to decompose, and running unconstrained CP decomposition yields inconsistent results across runs; using sparsity and nonnegativity priors helped reduce this variability and improve the interpretability of results.
Joris Claude is now pursuing a PhD in biomechanical systems.


## Perspectives

### Research objectives

% Can be used for ANR summary

There have been at least two sources of frustration in my research work so far. First, I have always been very interested in the applied mathematics aspects of signal processing, and many of my current contributions are based on heuristics or lack a clean theoretical motivation. Second, numerical optimization is the bread and butter of rLRA. There is a large body of literature on numerical optimization that I am still unfamiliar with. I have the impression that contributions to numerical optimization, with the practical challenges of rLRA and its applications in mind, are both within reach and potentially impactful for the rest of the community. Therefore, in the coming years, my main objective is to dedicate more research time to understanding the intricacies of numerical optimization and to propose theoretically motivated algorithms both for convex and non-convex problems.

An optimization problem that I believe is particularly interesting is the so-called NNKL problem described in [](../part1/nnls.md), namely, solving linear regressions under the Kullback-Leibler divergence loss. For an input nonnegative data vector $y\in\mathbb{R}_+^{m}$ and a linear observation matrix $A\in\mathbb{R}_+^{m\times n}$, NNKL can be formulated as

$$ \argmin{x\geq 0} \KL{y, Wx},$$
where $\KL{y,z} = \sum_{i} y[i]\log(\frac{y[i]}{x[i]}) + x[i] - y[i] $ is the Kullback-Leibler divergence. This optimization problem is quite challenging for at least two reasons:
- The cost function is not Lipschitz-smooth at zero. Lipschitz-continuity is a key property of cost functions in most convergence proofs of first-order methods. In practice, choosing a step-size for first-order methods can be challenging.
- When $y[i]$ is significantly smaller than $x[i]$, the loss is almost linear (the logarithmic term vanishes). This means that the cost function is not strongly convex, another important property that guarantees the practical speed of first-order methods.

[Figure KL taken from NNLS]

The optimization community has dedicated significant effort to proposing dedicated algorithms to solve problems such as NNKL. [Multiplicatives Updates](../part1/nnls.md#multiplicatives-updates) is such an algorithm that works well in practice. However, few methods can solve constrained variants of NNKL. In the current state of signal processing, where data-driven methods such as plug-and-play and unrolling algorithms yield significant performance gains, there is a need for algorithms that solve such non-smooth, non-Lipschitz-smooth, non-convex problems. I also plan to work on improving the state-of-the-art for solving NNKL and related problems, as I believe that within the frameworks of majorization-minimization and second-order approximation, significant gains can be made. We have already started to propose [better approximations of the cost](../part2/Fast_algorithms_for_rLRA/mSOM.md), but more work is required in this direction to find faster algorithms, robust to data and parameter sparsity, that can scale to large datasets. 

Finally, there is a dire need for a community-oriented optimization benchmark in rLRA. As mentioned above, the number of software libraries for performing LRA is extremely large; however, to this day, as far as I am concerned, it remains unclear which optimization method should be used in which context, and which implementation is the best on the market. I will start making efforts towards such benchmarking tools, starting with NMF. [The benchmark structure](https://github.com/benchopt/benchmark_nmf) has already been completed using the benchopt framework, in collaboration with Cassio Fraga-Dantas, and we are working with several authors, in particular Damien Lesens, on [filling up the benchmark with algorithms and datasets](https://github.com/DamienLesens/benchmark_nmf_kl).

The perspectives of my research on NNKL, and more generally Poison-distributed problems, are further detailed in [](../part3/KarpCoi.md).

A related, important line of work for the future focuses on single-pixel imaging. Perspectives on this topic include
- a joint reconstruction and unmixing of single pixel images with deep priors (such as plug-and-play and unrolling). This is the topic of the end of Serena Hariga's PhD thesis.
- better algorithms for Poisson noise problems, as discussed above.
- a deeper understanding of Poisson-Gaussian equivalences, and estimation under Poisson-Gaussian noise. Anna Jezierska and co-authors have already worked on this problem in the context of inverse problems {cite:p}`chouzenouxConvexApproachImage2015`, but I believe there could be additional discoveries to be made. In particular, known results about Poisson-Gaussian equivalences in high-count settings, although standard in the optics and statistics community {cite:p}`curtisSimpleFormulaDistortions1975,seifertMaximumlikelihoodEstimationPtychography2023`, are not easily summarized or put in use in a numerical optimization context. This work could be performed in collaboration with Valentin Debarnot, recently recruited to CREATIS.

These topics are also discussed in [](../part3/KarpCoi.md). Other interesting topics regarding single pixel imaging are discussed in [](../part3/others.md). A first topic of interest is the [design of the acquisition operator](../part3/others#single-pixel-imaging-and-compressive-acquisition), putting the theory of compressive sensing to the test. A second topic is the derivation of [separable NMF heuristics for inverse problems](../part3/others.md#inverse-problems-and-separable-nmf). The low-rank data matrix is not observed directly, therefore one may not simply pick columns from it as traditionally done in separable NMF. Finally, a last topic of interest is Borgen plots {cite:p}`neymeyrSetSolutionsNonnegative2018`, and more generally, algorithmic tools for studying uniqueness of solutions in inverse problems. Borgen plots show all possible solutions of NMF graphically for rank three, but their generalization to higher dimensions and ranks, as well as to other inverse problems {cite:p}`munierMLEReliableSource2025,sawallCalculationLowerUpper2022`, remains an open problem.


### About the evolution of research on rLRA

In the long run, with the rise of deep learning and the successes of end-to-end black-box modeling, coupled with the increasing ease of collecting large training datasets or generating synthetic ones, one may fear that unsupervised machine learning techniques, and in particular LRA, could become obsolete. My honest answer to this observation is that there is indeed a risk that LRA models, and more generally physics-inspired models, may not be required to achieve state-of-the-art results across a wide range of applications. Looking at the current state of things, however, there are at least two reasons to hope for a future in signal processing and AI in which LRA still plays an important role.
1. Physics-driven AI is currently a promising research direction to "open the black box". By using physical models, for instance, as forward operators, the performance of deep learning systems may not improve, but can be more robust to distribution shifts [paper Nicolas noise level etc]. Unrolling optimization algorithms provides a means to guide the design of neural networks motivated by traditional optimization algorithms. Plug-and-play methods enable the use of black-box models within physics-driven iterative algorithms. Although the resulting algorithms can still work in mysterious ways.
2. LRA is a fundamental tool in numerical linear algebra, signal processing, and machine learning. Many researchers are already working on using LRA to improve the design or training of deep learning architectures. A famous example of this is LORA {cite:p}`huLoRALowRankAdaptation2021b`, but other works have considered more involved usage of LRA models in deep learning {cite:p}`kossaifiTensorRegressionNetworksa, borsoiLowRankTensorDecompositions2025`.
Another more personal view on this matter is that rLRA can always be used as a trustworthy baseline in the future, even if more advanced models may significantly outperform it. In applications such as [automatic transcription](../part2/Applications_of_rLRA/AMT.md), it can be refined in multiple ways so that its performance is close to that of deep learning {cite:p}`marmoret2020uncovering`. If the current generation can produce efficient and flexible software for rLRA, it is reasonable to assume that, in applications where rLRA performs well, it will be used as an unsupervised baseline for the foreseeable future. The manuscript and the codes furnished along the way are, hopefully, one small step towards this goal.
