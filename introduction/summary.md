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


## Some personnal context

When I was six years old and starting to learn playing the piano, I had fun guessing the notes being played by my tutor while turning my back to the piano. This was a pleasant but easy game as I discovered that I could hear the pitch class of individual notes as they are played. I thought everyone else could also hear these names and therefore transcribe music easily. It was only a few years later, when joining my first music school, that I discovered I was mistaken, and that music transcription is actually really hard. For everyone in fact, as even with perfect pitch, transcribing polyphonic instruments becomes a compromise between instantaneous recognition and pattern matching with template harmonies store in one's memory.

As I grew older, I started to enjoy more generally the concept of sound and music processing, the Fourier transform and all its connections with the way human ear sounds, but also the symbolic and theoretical description of music. It was with great interest that I realized by the end of my PhD thesis that many questions in the field of music information retrieval, which encompasses most of these concepts, can be attacked at least partially with a core concept I had been working on, low-rank approximations of matrices and tensors. In particular, the design of regularizations and constraints was key to obtain reasonable performance, and I felt that with my skillset, I could improve on the existing state-of-the-art in that domain, and in particular in automatic music transcription.

It turns out that low-rank approximations, which is the main topic of this manuscript, are quite heavily beaten in practice by deep learning approaches when it comes to transcribing piano pieces. However, this annecdote shows that low-rank approximations are ubiquitus and of critical importance in unsupervised machine learning. While some researchers specialize in one specific application of low-rank approximations, I am rather interested in the applied mathematical side of low-rank approximations, in particular in modeling and training aspects. I am convinced that low-rank approximations based learning algorithms, in many applications such as automatic music transcription, can still be improved in the light of all the new results obtained in the last decade, a very modest part of which I am responsible for. It would bring me great joy to see anyone design a state-of-the-art automatic transcription algorithm that relies heavily on low-rank approximations, combined for instance with deep learning, or maybe in specific cases or rare instruments with scarce training data. I firmly believe that this objective is within reach in the comming years.

I write this "Habilitation à Diriger des Recherches" manuscript with the hope to demonstrate that regularized low-rank approximations is a rich topic with many exciting open questions. I also hope that fellow researchers who are willing to read through all this content will learn about some lesser known results, tricks and algorithmic details that can help them in their work on regularized low-rank approximations.

## Why Regularized Low-rank approximations

Low-rank approximations are workhorse methods in several unsupervised machine learning tasks such as dimensionality reduction {cite:p}`Tucker1966Some,Golub1989Matrix, hotelling1992relations` and blind source separation {cite:p}`harshman1970foundations, Paatero1997weighted, Lee1999Learning`. The most prominent LRA method in machine learning is probably Principal Component Analysis, which numerically boils down to computing a singular value decomposition, and is therefore efficiently computed.

[Illustration of PCA?]

Formally, for matrix data, a (real) low-rank approximation problem is an optimization problem of the form

$$ \min_{U\in\mathbb{R}^{m\times r},V\in\mathbb{R}^{n\times r}} f(Y, UV^T) $$
where integer $r$ is the rank of the approximation $UV^T\approx Y$, with $r\ll m,n$, and $f$ is a cost function, typically $f(Y, X) = \|Y-X\|_F^2$.

```{margin}
We present the case of matrix factorization for simplicity here, but the same logic applies to higher order factorizations.
```

In unsupervised machine learning tasks such as blind source separation, users often seek to give a physical meaning to the factor matrices $U$ and $V$. In fact, blind source separation can be seen as an inverse problem, where both the mixing matrix $U$ and the sources $V$ are unknown. In this context, there exists a true pair $(U_0, V_0)$ that the user seeks to recover. A necessary condition to correctly interpret a reconstructed pair is the uniqueness of this reconstructed solution. Note that LRA as defined above is never unique, since any product $UV$ can also be written $UPP^{-1}V$ for an invertible matrix $P$ of size $r\times r$.

An approach widely used in signal processing and machine learning to restrict the set of solutions of an inverse problem is to add prior information on the parameters in the form of constraints or regularizations. Regularized LRA can be formulated as 

$$ \min_{U\in \mathbb{R}^{m\times r},V\in\mathbb{R}^{r\times n}} f(Y, UV) + g_{U}(U) + g_{V}(V) $$
where $g_U$ and $g_V$ are regularizations promoting certain properties in solutions, such as smoothness, sparsity or nonnegativity. Depending on the choice of regularizations, the solutions may then be unique.

```{note}
Regularizations apply typically on each factor $U$ and $V$ independently. Indeed the main identifiability issue in LRA is the rotation ambiguity $UV^T = UP\left(VP\right)^T$ for an orthogonal matrix $P$. A regularization $g(UV^T)$ would not discriminate between two solutions identical up to permutation and therefore is not enough to ensure uniqueness of rLRA solutions.
```

```{margin}
Essential uniqueness is the uniqueness up to permutations and scaling ambiguities inherent to LRA models.
```

PCA is a constrained LRA model: factors are imposed to be orthognal matrices. This allows to obtain a model with essentially unique factors (under the mild condition that singular values must be distinct {cite:p}`Golub1989Matrix`), but orthogonality may not satisfied by the ground-truth factors $U^*$ and $V^*$. On the other hand, Nonnegative Matrix Factorization (NMF), obtained by setting $g_U$ and $g_V$ to characteristic functions of the nonnegative orthant, can also be unique {cite:p}`gillisNonnegativeMatrixFactorization2020` . In many applications such as spectral unmixing, elementwise nonnegativity is a natural assumption, which makes NMF particularly suited as a source separation/pattern mining model. A typical example of NMF usage is in spectral unmixing for remote sensing, as illustrated in Figure [TODO below]

[Example of NMF HSI]

A natural extension of matrix LRA is tensor LRA, where the input data has more than two dimensions. Tensor LRA have a number of interesting properties, including, in the case of the Canonical Polyadic Decomposition (CPD), identifiability without the need for regularization {cite:p}`Kruskal1977Three`. In practice however, regularization, and in particular nonnegativity, is often useful to improve the interpretability of the estimated parameters from tensor LRA. Chapter [](../part1/lra.md) describes in more details known results regarding matrix and tensor LRA.

[CPD figure]

Once an LRA model has been chosen as a model for a particular application, as in most machine learning problems, the model parameters need to be estimated so that the model fits the data. Optimization problems encountered in regularized LRA problems are often continuous but non-smooth, for instance due to the nonnegativity constraint, and non-convex due to the presence of product of variables in LRA. Convexity, however, often holds when the cost function is considered only with respect to one of the parameter matrix. Consider the NNLS problem with Frobenius data fitting

$$ \min_{U\in \mathbb{R}_+^{m\times r}} \|Y -  UV^T\|_F^2, $$
which is essentially NMF where matrix $V$ is fixed, the cost with respect to $U$ is convex. Therefore, for the applied mathematician working on regularized LRA, it is important to be both familiar with non-smooth constrained optimization, and with multi-convex optimization problems often solved with alternating optimization strategies. The specific case of nonnegatively-constrained regressions is of critical importance to this manuscript and is covered in depth in [](../part1/nnls.md), while a summary of known results regarding alternating optimization strategies is provided in [](../part1/AlternatingOptimization.md).

[NMF figure if not above]

There is a significant body of literature on regularized LRA (rLRA) with a myriad of applications, for instance in chemometrics {cite:p}`Bro1996Multiway`, neuroscience {cite:p}`cichockiTensorDecompositionsNew`, statistical inference {cite:p}`Anandkumar2015When`, spectral unmixing for remote sensing {cite:p}`Bioucas-Dias2012Hyperspectral` or microscopy imaging {cite:p}`harigaJointReconstructionSpectral2024`, music information retrieval {cite:p}`smaragdis2003non`, telecommunications {cite:p}`Sidiropoulos2000Parallel`, psychometry {cite:p}`Harshman1972PARAFAC2` and more. There is no dedicated section to reviewing these applications in this manuscript, rather dataset from hyperspectral imaging and music information retrieval are used as illustration for the various contributions.

## Contributions to regularized low-rank approximations

Since the start of my PhD thesis in 2013,  I have focused my research on LRA models as tools for extracting meaningful information out of matrices and tensors, using regularizations to enhance interpetability. These regularizations often include nonnegativity, which has become my specialty. My works cover a diverse list of topics ranging from understanding the properties of the solutions of rLRA to proposing new applications for these models. A significant section of my work is dedicated to algorithm design to compute solutions to rLRA efficiently in various setups.

### Challenges in rLRA

Despite the large amount of existing works on rLRA, their use by practitionners is often quite difficult. Indeed there are many exciting theoretical and practical issues that remain to be dealt with:
- **Optimization problems** stemming from fitting rLRA models are multi-block, non-convex and often non-smooth. Recent advances in numerical optimization regarding acceleration, majorization-minimization, generalized projection operators and alternating optimization can all be leveraged to improve upon the state-of-the-art. It is also unclear, in practice, which optimization algorithm to use depending on the context.
- **Side information** is often available alongside the matrix or tensor data, that transforms the unsupervised rLRA learning problem into a semi-supervised or fully supervised problem. This information can take diverse forms: a known library of templates for the unknown sources, a downstream task with available training data, deep priors for the unknown sources such as denoisers, sparsity in well-known bases, sparsity level of the sources, among others. Acounting for this side information in rLRA is often not straightforward, both from a modeling and from a training perspective.
- **Multimodality** has emerged as an important topic in source separation, where the same phenomenon is observed through different sensing devices. A typical exemple would be acquiring brain activity through EEG and fMRI jointly. The joint processing of the acquired dataset can be performed by several rLRA. The research questions on joint rLRA typically revolve around the coupling model for the various dataset, and the design of flexible algorithms that allow to solve a wide range of multimodal source separation problems, in particular when the dataset have different dynamics, noise levels and even different fields (such as floats vs count vs categorical data).
- The rLRA models themselves still have elusive properties. An important example is nonnegative Tucker decomposition, for which **identifiability** is unclear in many cases, despite recent advances on the topic {cite:p}`sahaIdentifiabilityNonnegativeTucker2025`. Generally, a user needs guarantees on the nature of the solutions of rLRA, and such theoretical guarantees are, in general, very challenging to achieve, even more so if the hypotheses must be verifiable in practice.
- The **scientific software ecosystem** is a dimension often overlooked but critical in practice. Available software for tensor decomposition is essentially bloated, see this survey that lists over 70 tensor packages scattered over the internet {cite:p}`psarrasLandscapeSoftwareTensor2022`. However, software packages dedicated to, or able to handle at least partially, rLRA are scarce. For tensor decompositions, many packages focus on the decomposition itself but do not make use of efficient large-scale contractions on CPU or GPU that are crucial for up-scaling {cite:p}`smithTensormatrixProductsCompressed2015,liInputadaptiveInplaceApproach2015,g.a.smithOptEinsumPython2018`. Another critical issue is the actual implementation of rLRA algorithms, that requires non-trivial caching of expensive operations and speed-momery trade-offs that are research topics by themselves {cite:p}`kayaHighPerformanceParallel2016`.

### Improving on theory, algorithms and applications

My work has been dedicated to proposing (partial) solutions to these issues. I have grouped my contributions in three parts, dealing respectively with theoretical contributions, applications-oriented contributions and algorithmic-focused contributions. However in most of these works, all three aspects (theory, algorithms, applications) are intertwined, so this is no a strict segmentation of my work.

There is no dedicated chapter in this manuscript to software development. Instead, simple implementations of methods under scrutiny are either provided inline, imported from Tensorly, which is a package I have co-developping since 2019, or imported from a local set of methods specially developped for this manuscript. Software development is an important part of scientific contributions in the field of rLRA and applied mathematics in general, and in my opinion it is important to show the exact code being run for experiments and demos so that the reader can be better convinced of the exactitude of the results.

The three main sections of this manuscript are
- [Theory of rLRA contributions](../part2/Theory_of_rLRA/intro.md), where I summarize my contributions to the analysis of several sparse models: [Dictionary-based LRA](../part2/Theory_of_rLRA/DL_identifiability.md), [sparse nonnegative least-squares](../part2/Theory_of_rLRA/sparse_nnls.ipynb). I also study [the impact of scaling ambiguity on rLRA solutions](../part2/Theory_of_rLRA/HRSI_theory.md), that can induce unexpected group-sparsity at the component level.
- [Algorithms for existing rLRA models contributions TODO](../part2/Fast_algorithms_for_rLRA/intro.md), where I summarize my contributions to algorithm design for rLRA. My contributions concern [flexible algorithms for multimodal rLRA](../part2/Fast_algorithms_for_rLRA/CMTF.md), [heuristic inertial acceleration for alternating optimization algorithms](../part2/Fast_algorithms_for_rLRA/inertial_BCD.ipynb), and more recently, [the design of tight surrogates](../part2/Fast_algorithms_for_rLRA/mSOM.md) for nonnegative optimization problems including NMF in non-Euclidean settings. I have also worked on data-driven regularization for LRA, in particular on [unroling multiplicative updates](../part2/Fast_algorithms_for_rLRA/UnrolledNMF.md) for NMF. Finnaly I revisited a work from my PhD thesis on [projected least squares to compute nonnegative LRA](../part2/Fast_algorithms_for_rLRA/proco-als.ipynb).
- [Applications of rLRA contributions](../part2/Applications_of_rLRA/intro.md). My expertise in terms of applications of rLRA is geared towards both [spectral imaging](../part2/Applications_of_rLRA/single_pixel.ipynb) and music information retrieval, in particular [automatic transcription](../part2/Applications_of_rLRA/AMT.md). My interest for spectral imaging resides in the fact that LRA is intimely linked with additive mixtures that accurately describe mixtures in spectral acquisition techniques such as fluorescence spectroscopy or hyperspectral remote sensing. Being a semi-professional pianist and singer, I have a personnal interest in music that drives my research in music information retrieval. Other applications, mostly related to biomedical imaging, are more punctual collaborations and are described below.

More details on each topic, including a detailled summary of the contributions of each part, are provided respectively in the introductions of [Theory of rLRA](../part2/Theory_of_rLRA/intro.md), [Algorithms for rLRA](../part2/Fast_algorithms_for_rLRA/intro.md) and [Applications of rLRA](../part2/Applications_of_rLRA/intro.md).

### Others works
There are a number of my works that are not discussed in this manuscript, that would be otherwise even longer. These work are often targetted toward specific applications or require specific mathematical tools that I want to avoid introducing here. Below is a short paragraph for each of these works.

#### Music segmentation

Maybe the most impactful work that I chose to not discuss in this manuscript is related to the PhD thesis of Axel Marmoret {cite:p}`marmoret2020uncovering,marmoretNonnegativeTuckerDecomposition2022, marmoretUnsupervisedMachineLearning2022`. Low-dimensional models such as Tucker decomposition, but also auto-encoders, are used to compress the information from a recording of a full song {cite:p}`smithNonnegativeTensorFactorization2018`. This compression highlights the similarities and dissimilarities between bars in the song, which then helps a dedicated dynamic program solver to perform an automatic segmentation of the song. Despite the method being unsupervised (a supervised beat-tracking algorithm is still used to cut the song into bars), the results are very encouraging if the parameters of the compressed models can be chosen optimally. This choice is difficult in practice, which limits the performance of the method.

[Figure déjà prête]

This work was rather visible in the music information retrieval community when released. It has launched the academic career of Axel Marmoret, who is working today, among others, on LRA in conjunction with deep learning with applications to music information retrieval. His PhD manuscript desribes our contributions precisely and extensively.

### Birkhoff-von Neumann decomposition of stochastic matrices

The Birkhoff-von Neumann (BvN) decomposition of a stochastic matrix is the problem of writing a stochatic matric $M$ as a minimal number of permutation matrices $P_i$,

$$ \text{Find the smallest } k\in\mathbb{N} \text{ such that } M = \sum_{i=1}^{k} \alpha_i P_i, $$
where $\alpha_i$ are nonnegative coefficients; both coefficients and permutations matrices have to be estimated from the input matrix. This long standing NP-hard problem can be tackled by a variety of existing heuristics. In a collaboration with Bora Ucar and Damien Lesens, we proposed to view BvN as a sparse coding problem and adapt existing sparse coding solvers such as orthogonal matching pursuit to compute solutions {cite:p}`lesensOrthogonalMatchingPursuitBased2024`. The contribution is to link the selection of the best atom, here the selection of the best permutation matrix, to several graph matching algorithms so that the selection is both fast and effective. Our method is competitive with the state-of-the-art. In a follow-up work driven by Damien Lesens, we extended BvN algorithms to the symmetric case {cite:p}`lesensAlgorithmsSymmetricBirkhoffVon2026`. We are the first to propose an implementable algorithms to solve symmetric BvN. 

These works fall slightly outside the realm of rLRA. My role for the second contribution on symmetric BvN in particular was rather minor. Therefore I chose to not discuss further this work. Damien Lesens is now pursuing a PhD co-supervised with Bora Ucar on optimization algorithms for nonnegative LRA, see the [Perspectives](../part3/Perspectives.md).

### Sparse separable NMF

Separable NMF is a regularized variant of NMF with nice indentifiability and algorithmic properties, see [](../part1/lra.md#regularized-nmf). In a joint with Nicolas Nadisic, Arnaud Vandaele and Nicolas Gillis, we studied a slightly more general model called sparse separable NMF, which assumes that the data point are sparse convex combinations of a few data vectors {cite:p}`nadisicSparseSeparableNonnegative2021`. An algorithm is proposed ressembling orthogonal matching pursuit that identifies the template components from the data, and compute sparse regressions for each data point using a previously [proposed sparse NNLS algorithm](../part2/Theory_of_rLRA/sparse_nnls.ipynb). We show however that the problem is in general NP-hard, unlike separable NMF.

### Optical Biopsy NMF

A contribution closely related to [](../part2/Applications_of_rLRA/Single_pixel_spectral_imaging.md) performs segmentation of cancerous cells from videos of the brain during surgery using separable NMF {cite:p}`careddaSeparableSpectralUnmixing2023`, in collaboration with Bruno Montcel, Charly Carrera and others. For this work, my contribution is to propose a robust variant of the well-known direct algorithm for rank-two exact separable NMF in Frobenius norm. The components estimated from the rank-two NMF are then used to estimate the spatial oxygenated blood flows, that helps locate the glioma.

### Lever arm tensor decomposition

In a collaboration with Raphaël Dumas and Joris Claude, we studied a set of lever arm measurements from leg muscles acquired on patients with knee injuries during walking {cite:p}`claudeConstrainedTensorDecomposition2024`. This study revealed synnergies between various muscles in the leg. These muscles activations can be clustered to summarize the complex action of the human body during walking to a few descriptors. My contribution to this work was to help with the design of the constrained tensor decomposition model. The data tensor is quite difficult to decompose, and running unconstrained CP decomposition yields inconsistent results across runs; using sparsity and nonnegativity priors helped reduce this variability and improve results interpretability.
Joris Claude is now pursuying a PhD on biomecanical systems.


## Perspectives

### Research objectives

% Can be used for ANR summary

There have been at least two sources of frustration in my research work up to now. First, I have always been very interested in the applied mathematics aspects of signal processing, and many of my current contributions are based on heuristics or lack a clean theoretical motivation. Second, numerical optimization is the bread and butter of rLRA. There is a large body of litterature on numerical optimization that I am still unfamiliar with. I have the impression that contributions to numerical optimization, having in mind the practical challenges of rLRA and its applications, are both within reach and potentially impacful for the rest of the community. Therefore in the comming years, my main objective is to dedicate more research time to understanding the intricacies of numerical optimization, and to propose theoretically motivated algorithms both for convex and non-convex problems.

An optimization problem which I believe is particularly interesting is the so-called NNKL problem described in [](../part1/nnls.md), namely solving linear regressions under Kullback-Leibler divergence loss. For an input nonnegative data vector $y\in\mathbb{R}_+^{m}$ and a linear observation matrix $A\in\mathbb{R}_+^{m\times n}$, NNKL can be formulated as

$$ \argmin{x\geq 0} \KL{y, Wx},$$
where $\KL{y,z} = \sum_{i} y[i]\log(\frac{y[i]}{x[i]}) + x[i] - y[i] $ is the Kullback-Leibler divergence. This optimization problem is quite challenging for at least two reasons:
- The cost function is not Lipschitz-smooth at zero. Lipschitz-continuity is a key property of cost functions in most convergence proofs for first-order methods. In practice, the choice of a step-size for first-order methods can be challenging.
- When $y[i]$ is significantly smaller than $x[i]$, the loss is almost linear (the logarithmic term vanishes). This means that the cost is not strongly convex, another important property to guarantee the practical speed of first-order methods.

[Figure KL take from NNLS]

The optimization community has dedicated significant effort to propose dedicated algorithms to solve problems such as NNKL. [Multiplicatives Updates](../part1/nnls.md#multiplicatives-updates) is such an algorithm, that works well in practice. However, few method are able to solve constrained variants of NNKL. In the current state of signal processing, were data-driven methods such as plug-and-play and unrolling algorithms allow for significant performance gains, there is a need for algorithms solving such non-smooth, non Lipschitz-smooth, non-convex problems. I also plan to work on improving the state-of-the-art for solving NNKL and related problems, as I believe that within the frameworks of majorization-minimization and second-order approximation are significant gains to be made. We have already started to propose [better approximations of the cost](../part2/Fast_algorithms_for_rLRA/mSOM.md), but more work is required in this direction to find faster algorithms, robust to data and parameter sparsity, that can scale to large dataset. 

Finally, there is a dire need for community-oriented optimization benchmark in rLRA. As mentionned above, the number of software libraires to perform LRA is extremely large; however, it is still unclear to this day, as far as I am concerned, which optimization method should be used in which context, and which implementation is the best at the moment on the market. I will start making efforts towards such benchmarking tools starting with NMF. [The benchmark structure](https://github.com/benchopt/benchmark_nmf) has already been completed using the benchopt framework, in collaboration with Cassio Fraga-Dantas, and we are working with several authors, including Damien Lesens, on [filling up the benchmark with algorithms and dataset](https://github.com/DamienLesens/benchmark_nmf_kl).

The perspectives of my research on NNKL, and more generally Poison-distributed problems, are further detailed in [](../part3/KarpCoi.md).

A related important line of work for the future revolves around single-pixel imaging. Perspectives on this topic include
- a joint reconstruction and unmixing of single pixel images with deep priors (such as plug-and-play and unrolling). This is the topic of the end of Serena Hariga's PhD thesis.
- better algorithms for Poisson noise problems, as discussed above.
- a deeper understanding of Poisson-Gaussian equivalences, and estimation under Poisson-Gaussian noise. Anna Jezierska and co-authors have already worked on this problem in the context of inverse problems {cite:p}`chouzenouxConvexApproachImage2015`, but I believe there could be additional discoveries to be made. In particular, known results about Poisson-Gaussian equivalences in high-count settings, although standard in the optics and statistics community {cite:p}`curtisSimpleFormulaDistortions1975,seifertMaximumlikelihoodEstimationPtychography2023`, are not easily summarized or put in use in a numerical optimization context. This work could be performed in collaboration with Valentin Debarnot, recently recruted in CREATIS.

These topics are discussed also in [](../part3/KarpCoi.md). Other interesting topics regarding single pixel imaging are discussed in [](../part3/others.md). A first topic of interest is the [design of the acquisition operator](../part3/others#single-pixel-imaging-and-compressive-acquisition), putting the theory of compressive to the test. A second topic is the derivation of [separable NMF heuristics for inverse problems](../part3/others.md#inverse-problems-and-separable-nmf). The low-rank data matrix is not observed directly, therefore one may not simply pick columns from it as traditionnaly done in separable NMF. Finally, a last topic of interest are Borgen plots {cite:p}`neymeyrSetSolutionsNonnegative2018`, and more generally algorithmic tools for studying uniqueness of solutions in ionverse problems. Borgen plots show all possible solutions of NMF graphically for rank three, but their generalization to higher dimensions and ranks, as well as to other inverse problems {cite:p}`munierMLEReliableSource2025,sawallCalculationLowerUpper2022`, remains an open problem.


### About the evolution of research on rLRA

In the long run, with the rise of deep learning and the successes of end-to-end black box modeling coupled with the increasing ease of collecting large training dataset or generating synthetic ones, one may fear that unsupervised machine learning techniques, and in particular LRA, could become obsolete. My honest answer to this observation is that indeed, there is a risk that LRA models, and more generally physics-inspired models, may not be required to obtain state-of-the-art results in a large spectrum of applications. Looking at the current state of things however, there are at least two reasons why one may hope for a future in signal processing and AI where LRA still play an important role.
1. Physics-driven AI is currently a promising research direction to "open the black box". By making use of physical models, for instance in the form of forward operators, performance of deep learning systems may not improve but can be more robust to distribution shifts [paper nicolas noise level etc ?]. Unrolling optimization algorithms provides a mean to guide the design of neural networks motivated by traditionnal optimization algorithms. Plug-and-play methods allow for using black-box models inside physics-driven iterative algorithms. although the resulting algorithms can still work in mysterious ways.
2. LRA is a fundamental tool in numerical linear algebra, signal processing and machine learning. Many researchers are already working on using LRA to improve the design or training of deep learning architectures. A famous example of this is LORA {cite:p}`huLoRALowRankAdaptation2021b`, but other works have considered more involved usage of LRA models in deep learning {cite:p}`kossaifiTensorRegressionNetworksa, borsoiLowRankTensorDecompositions2025`.
Another more personnal view on this matter is that rLRA can always be used as a trustable baseline in the future, even if more advanced models may significantly outperform it. In applications such as [automatic transcription](../part2/Applications_of_rLRA/AMT.md), it can be refined in multiple ways so that its performance is close to that of deep learning {cite:p}`marmoret2020uncovering`. If the current generation is able to produce efficient and flexible software for rLRA, it is reasonable to assume that in applications where rLRA works rather well, it will be used as an unsupervised baseline for the forseeable future. The manuscript are the codes furnished along the way are, hopefully, one small step towards this goal.