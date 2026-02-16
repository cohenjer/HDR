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

# Numerical Optimization for KL-based Regularized Inverse Problems

The primary focus of my research over the past decade has been on regularized low-rank approximation models (rLRA). rLRA is a discipline at the intersection of inverse problems, statistics, numerical optimization, and machine learning, with diverse applications such as music information retrieval and spectral imaging that require specialized expertise. While I have always enjoyed embracing the diversity of the mathematical tools required to make contributions on rLRA, in the comming years I want to spend time focussing especially on numerical optimization, with potential contributions outside the scope of rLRA. I enjoy the idea that in numerical optimization, there is often a clear problem-solving objective, such as cost minimization or speed maximization, that allows us to compare methods against each other. I also enjoy the mathematical framework of both smooth and discrete optimization problems and algorithms. Numerical optimization provides an opiniated view of the world as many tasks can be expressed in this framework. On the mathematical side, theoretical results and proofs are both intuitive and rigorously enunciated.

```{margin}
KARP-COI stands for Kullback-Leibler divergence And Regularized inverse Problems Call for faster OptimIzation algorithms.
```

A family of problems that I want to study in particular, related to nonnegativity of the parameters and data, is Kullback-Leibler-divergence (KL-divergence) based estimation, such as [NN-KL](../part1/nnls.md). KL-divergence is a rich loss function to study because it lacks Lischitz-smoothness at zero (its gradient tends to infinity), but it is also asymptotically flat towards infinity. This makes the design of global strategies, such as gradient descent with fixed stepsize or global majorization-minimization, rather difficult. In the following, I explain why the KL-divergence appears in inverse problems, what the hypotheses and challenges of my research project are, and its positioning in the state-of-the-art. I then detail the various contributions I have in mind for the comming years with identified collaborators. This research project has been named KARP-COI in the context of an ANR submission.

Other, smaller-scale aspects to my research project are detailed in [a separate document](./others.md). This chapter contains redundant information on the basic mathematical tools described in this HDR manuscript to ensure that it can be read independently, to some degree.

## Scientific context

Kullback-Leibler divergence (KL divergence), a fundamental measure of similarity between probability distributions in machine learning, naturally arises from [Poisson maximum likelihood models](../part1/nnls.md) [6], but also provides a robust measure of discrepancies compared to the Euclidean​ norm. This robustness is particularly relevant for audio spectrograms that exhibit large dynamic ranges, and for low signal to noise ratio counting processes that constitute the future of many existing computational imaging devices with reduced acquisition time. KL-divergence between two vectors $y$ (the data) and $Ax$ (the unknown) in $\R{n}_+$ is defined as

$$ \KL{y, Ax} = \sum_{i=1}^{n} y[i]\log\left(\frac{y[i]}{Ax[i]}\right) + Ax[i] - y[i]. $$


% ANR add figure of KL and technical details

Inverse problems aim at estimating physically meaningful parameters (images, audio representations, sources) from incomplete and noisy data. In this research project, I focus on two problems, Nonnegative regression problems (NN-KL) and Nonnegative matrix and tensor factorization problems (NMF-KL), see the [dedicated chapter on NN-KL](../part1/nnls.md), and the figure [TODO]. NN-KL reconstructs or denoises images in many computational imaging applications such as tomography, Compton camera, cryoEM, and [single-pixel imagers](../part2/Applications_of_rLRA/single_pixel.ipynb) [7, B].  NMF-KL is a linear dimensionality reduction technique similar to Principal Component Analysis, where nonnegativity ensures an interpretable part-based representation [1, A, D]. NMF-KL is also a blind extension of NN-KL and has been used extensively to analyse spectrograms found in music information retrieval tasks, in particular to perform [Automatic Music Transcription](../part2/Applications_of_rLRA/AMT.md) that translates music recordings into MIDI files (numerical music sheets) [F]. NN-KL and NMF-KL problems are generally ill-posed: there are too few measurements or too much noise to uniquely and precisely recover the unknowns, which compromises interpretability. Regularization is thus essential. It can be introduced via handcrafted priors (e.g., sparsity, smoothness), via pre-trained priors using Plug-and-Play (PnP) denoisers, or [Unrolled algorithms](../part2/Fast_algorithms_for_rLRA/UnrolledNMF.md) that exploit training datasets [4, 8, A].

% ANR add explanation on application to audio, and to SPI spectral unmixing
% Add summary figure updated

### Hypotheses and challenges
I hypothesize that it is possible to design algorithms that are both fast and provably convergent for KL-based regularized inverse problems. Purely from a smooth numerical optimization perspective, the main difficulties are
1. The lack of Lipschitz continuity at zero of the KL-divergence, which causes instabilities with sparse data or low counts when using first-order methods [1]. This explains why most existing approaches rely on second-order information, typically through separable quadratic surrogates derived from Hessian approximations.
2. The asymptotic linearity of the KL-divergence (when the term $Ax[i]$ dominates the cost$), which makes any gradient-based algorithm hard to use. The cost is not strongly convex, and the Hessian has near-zero singular values. Both these issues are known to imply slower convergence. 
3. The presence of nonnegativity constraints, as well as other regularizations, possibly based on deep-learning. 
Our challenge is twofold. First, to optimize the design of algorithms for NN-KL and NMF-KL, striking the right balance between accuracy and per-iteration cost. I plan to rely on local (instead of global) approximations of the cost function and tools from linear programming. Local majorants are expected to work well because they do not need to diverge at zero, but the assymptotic flatness, in particular for sparse data, must be dealt with carefully. Second, to extend these strategies to include data-driven regularizations, which fall within variable metric forward–backward methods and remain difficult to implement for both handcrafted and data-driven priors [5]. Beyond algorithm design, the project will unify contributions scattered across numerical optimization, source separation, and computational imaging, producing a benchmark and a toolbox for KL-based inverse problems. The methodology will be validated on two applications where the consortium has strong expertise: single-pixel imaging and automatic music transcription [A, F]

### Positioning with respect to the state of the art
Multiplicative Updates (MU), or Maximum-Likelihood Expectation-Maximization in computational imaging, remain the historical baseline for KL-based problems. MU is simple to implement and makes use of second-order information in the form of preconditionning. It is a baseline that can be slow, unstable with sparse data, and difficult to adapt with regularizations. However, it is not so easily beaten, especially in the context of NMF-KL [TODO ref Gillis]. MU has a wide number of variants, some of which have been developed in the computational imaging community and involve block-coordinate updates [OSEM Fessler TODO 6, 7], while others come from the signal processing community and focus, for instance, on all-at-once updates for wider classes of loss functions [ref Fevotte] or high-order tensor decompositions [MU gen ref]. Most algorithms for NN-KL and NMF-KL, including MU, fall within the majorization-minimization framework, where the cost is globally majorized by a simpler function, often separable with respect to each parameter, and then this majorant is minimized efficiently. In this project, we will make use of local approximations, which do not belong to the majorization-minimization framework.

```{margin}
The topic of linking MU and mirror descent is under active research in the team TOMORADIO.
```

Classical constrained formulations lack a general framework for KL-based problems. MU can be applied, after non-trivial modifications, for a limited set of priors such as $\ell_p$ norms and TV regularization. Data-driven approaches (Plug-and-Play, Unrolled NMF) are promising to reach state-of-the-art performance while ensuring interpretability of the model, yet their KL-based theoretical grounding is weak. In particular, PnP algorithms for KL divergence rely on a loose Bregman divergence, Burg’s entropy, resulting in poor convergence speed [2, 8]. Burg's entropy is used in proximal mirror descent to build a global majorant of the cost, but it can be observed on the following simple synthetic example that MU is typically much faster than Burg's entropy. The reason why Burg's entropy is used despite its loose majoration of the KL-divergence is that is unclear how to relate the majorant leading to MU or other existing faster algorithms with proximal mirror gradient descent, and therefore obtain a clean setup for non-Euclidean data-driven algorithms. In this research project, I hypothesize that such links can be obtained, and that other frameworks than mirror gradient descent may be used for convergent data-driven non-Euclidean algorithms.

```{code-cell}ipython3
from matplotlib.pylab import f
import numpy as np

# Let's compare MU and Bregman mirror descent on a simple Poisson regression problem

# Hyperparameters
m, n = 100, 20  # number of samples and features
alpha = 100 # Poisson signal level

hgt = np.maximum(np.random.randn(n),0)
W = np.random.rand(m,n)
y = np.random.poisson(alpha * W @ hgt)

def loss(h):  # KL divergence
    return np.sum(W@h-y) + np.sum(y * np.log((y+1e-10)/(W @ h)))

def MU(y, W, h, itermax=1000, epsilon=1e-10):
    Wtsum = W.T @ np.ones_like(y)  # suboptimal but ok
    hout = np.copy(h)
    loss_val = [loss(hout)]
    for i in range(itermax):
        hout = np.maximum(epsilon, hout * (W.T @ (y / (W @ hout ))) / (Wtsum))
        loss_val.append(loss(hout))
    return hout, loss_val

def BurgMU(y, W, h, itermax=1000, epsilon=1e-10):  # NoLIPS by Bauschke et al
    Wtsum = W.T @ np.ones_like(y)
    lamb = 1/2/np.sum(y)
    hout = np.copy(h)
    loss_val = [loss(hout)]
    for i in range(itermax):
        hout = np.maximum(epsilon, hout / ( 1 + lamb* hout * ( Wtsum - W.T @ (y / (W @ hout )))))
        loss_val.append(loss(hout))
    return hout, loss_val 


hmu, loss_mu = MU(y, W, alpha*np.ones(n), itermax=5000)
hburg, loss_burg = BurgMU(y, W, alpha*np.ones(n), itermax=5000)


# Printing the final estimation error
print(f"Final loss MU: {loss_mu[-1]}, Burg MU: {loss_burg[-1]}")
print(f"Estimation error MU: {np.linalg.norm(hmu - alpha*hgt)/np.linalg.norm(alpha*hgt)}, Burg MU: {np.linalg.norm(hburg - alpha*hgt)/np.linalg.norm(alpha*hgt)}")

```

```{code-cell}ipython3
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.semilogy(loss_mu, label='MU')
plt.semilogy(loss_burg, label='Burg MU')
plt.title('Loss convergence for Poisson regression NN-KL')
plt.xlabel('Iteration')
plt.ylabel('Loss (KL divergence)')
plt.legend()
plt.show()
```

Finally, existing algorithms are designed for small, dense matrices, while the audio application also leads to sparse, large matrices. In summary, no existing method simultaneously achieves speed, robustness to data sparsity, and theoretical guarantees, leaving a clear gap that I aim to fill.


## Methodology and scientific coverage
The research project is divided into four complementary subtasks to achieve the global objective of conceiving faster regularized algorithms for KL-based inverse problems, see Figure above [TODO]. The project is grounded in the numerical optimization community, but with strong interactions with the machine learning and signal processing communities.


### NMF toolbox and benchmark
As of 2026, it is far from straightforward to navigate the world of NMF algorithms for researchers and end-users alike. On the side of models, there are several loss functions to choose from depending on the data type and the noise distribution, and various regularizations can be considered. But even for a barebone model such as NMF-KL, the literature does not clearly inform on which algorithm to use in which context. Choices to be made have several layers: should the algorithm be alternating over the two parameter matrices, or update both simultaneously? Should one use a majorization-minimization algorithm such as MU, primal-dual algorithms, mirror descent with Burg entropy, or a second-order inspired algorithm such as [the proposed mSOM](../part2/Fast_algorithms_for_rLRA/mSOM.md)? Are extrapolation and momentum needed? The actual performance gap between these methods depends heavily on the application at hand. In particular, in the context of NMF-KL for document mining or audio processing, data sparsity and data size can be critical, and the actual implementation of algorithms (sparsity handling, efficient caching of operations) matters. 

I believe that performing meaningful research on NMF-KL algorithms requires first taking a step back and analyzing precisely the current state-of-the-art. In particular, I plan to build a benchmark for NMF models, starting with NMF-KL, that includes as many algorithms from the literature as possible, and datasets publicly available from applications in document mining, music information retrieval, hyperspectral imaging in remote sensing and microscopy, computer vision, and chemometrics. The backbone of the benchmark was already developed in 2022 in collaboration with Cassio Fraga-Dantas, relying on Benchopt [TODO ref], but needs to be enhanced with algorithms and a dataset.

Alongside the development of this benchmark, the implementation of existing algorithms will be made available in a dedicated NMF toolbox in Python, along with other useful tools such as separable NMF algorithms and [Borgen plots](../part3/others.md#borgen-plots-revisited) discussed later in the perspectives, and an integration of efficient implementations in lower level language of important routines such as [active-set and HALS](../part1/nnls.md). Available software packages in Python for NMF include scikit-learn, which implements only multiplicative updates for NMF-KL, and Nimfa, a toolbox no longer maintained that focuses on model diversity (including many Bayesian and graph-regularized variants of NMF) rather than efficient algorithms for a few targeted models. The proposed NMF toolbox may rely on Tensorly to be backend-agnostic (allowing lower-level operations to be performed transparently by numpy, pytorch, jax...). The architecture of the package is not yet decided, and it may also be built on top of only PyTorch for simplicity and auto-differentiation support, or in Julia for better integration of efficient sparse-oriented routines such as [sparse NNLS](../part2/Theory_of_rLRA/sparse_nnls.ipynb).

**Collaborators**: The benchmark is already under construction by Damien Lesens (ENS Lyon, LIP) in the context of his PhD thesis co-supervised with Bora Ucar (Inria Lyon, LIP). It will be completed by me and all supervised students who work on problems connected with the KL-divergence. Collaborators from Telecom Paris (Christophe Kervazo, Mathieu Fontaine, Roland Badeau) and IMT Atlantique (Quyen Pham) are also expected to participate. If this benchmark finds echo in the NMF community, other research groups may join the initiative, such as the group of Nicolas Gillis in Mons that lacks such a unified software plateform.

%All partners contribute to the creation of a dedicated toolbox to compute NN-KL and NMF-KL using Tensorly as a backend. A concurrent toolbox exists (nimfa) that focuses on variants of NMF and is not maintained. All methods are evaluated and compared on a dedicated benchmark using Benchopt [12]. The benchmark was already initiated by the project coordinator, but needs to be enhanced with more algorithms and datasets. WP1 also relies on the available manpower and support of Benchopt. Benchmarking dataset will include various applications including chemometrics, audio source separation, text mining, and hyperspectral unmixing.

### New numerical optimization tools for faster NN-KL and NMF-KL

NN-KL is a rather old optimization problem, and both researchers and users (in particular from computational optics) have proposed a consequent number of ideas to design fast and accurate algorithms. One key scientific lever has been the design of global majorants of the KL-divergence, in direct relationship with the Expectation Maximization framework [refs Fessler, etc.]. In this context, it may seem unrealistic to assert that the participants of this research project will propose faster algorithms for NN-KL. NMF-KL has been slightly less studied since NMF is far less commonly used than linear regression, but the same observation could be made. 

This research project is built on the idea that some important research directions have not been explored regarding optimization with the KL-divergence. First, there are relatively few methods that do not rely on global Majorization Minimization (MM). While MM is useful to design convergent algorithms, the geometry of the KL-divergence, which diverges at zero but is asymptotically linear, makes it difficult to design majorants that are both global and tight. Popular majorants such as those used in [MU](../part1/nnls.md) are also separable. Recent works for regularized NN-KL and NMF-KL based on mirror gradient descent make use of Burg's entropy to build a majorant, which is worse than the MU majorant as discussed above. One promising research direction is to resort to **local separable majorants using second-order information**. In an ongoing work (TODO update review status) with Mai Quyen Pham and Thierry Chonavel, we show that we can design local majorants of the cost by building diagonal majorants of the Hessian matrix iteratively, of the form

$$M(u,x) = \text{Diag}\left(\frac{H(x)u}{u}\right),$$
where $H(x)$ is the Hessian matrix at point $x$ and $u$ is a vector that represents a diagonal matrix from this class. For any positive $u$, it can be shown that $M(u,x) - H(x)$ is symmetric and definite positive. In other words, we can build a local majorant of the cost with curvature given by the matrix $M(u,x)$. Diagonal approximations of the Hessian matrix allow for decoupling the parameters in the minimization of the majorant, which is important to make the implementation scalable. Our work opens many perspectives, as we study a single class of local majorants, and pick a particular item in that class that can be obtained in closed form. We plan in particular to either find even better local majorants by solving small-scale optimization problems. We also plan to extend this strategy to all-at-once algorithms [TODO ref fevotte] and explore similar ideas within the framework of mirror descent.

Second, KL-divergence has a linear term $\sum_{i,j} A[i,j]x[j] - \sum_{i} y[i]$, that links NN-KL with linear programming. More precisely, it is easy to see on the KKT conditions that at optimality, it must hold that this sum cancels out. This allows us to replace the linear term in the KL-divergence minimization problem with a linearly constrained problem

$$\argmin{1^Ty = 1^TAx} -\sum_{i} y[i]\log((Ax)[i]).$$
While this is maybe not so important for dense dataset, we hypothesize that this can be used to build efficient algorithms for sparse NN-KL and sparse NMF-KL problems. Indeed, if most entries in the data vector $y$ are null, the problem is similar to an interior point method for linear programming with a log-barrier function. It is therefore tempting to associate NN-KL for sparse data with a linear program that could be used to initialize other algorithms. It is also interesting to explore the use of the Sinkhorn algorithm to enforce this scaling inside of any alternating algorithm for NMF-KL, since these marginal constraints are in general not satisfied along the iterations. Finally, second-order methods will be considered since the Hessian matrix should be highly structured, and linear constraints can be incorporated. Most algorithms on the market simply ignore the possibility that the data matrix and/or the parameters can be extremely large and extremely sparse. While most algorithm can be easily adapted to handle sparse data, some are not so easy to adapt, such as primal dual algorithms, where the dual variables are generally dense.

**Collaborators**: The design of local majorants is the purpose of a long-term collaboration with Mai Quyen Pham (IMT Atlantique). We plan to continue working on this topic together by recruiting a PhD student. The study of links between KL-divergence optimization and linear programming is the topic of the PhD thesis of Damien Lesens, co-supervised with Bora Ucar. The topic of optimization for KL-divergence and more generally Poisson-distributed data is currently under scrutiny by several research groups, including Voichita Maxim in my team at CREATIS, Lucas Calatroni (I3S, Nice and University of Genoa, Italy), Emilie Chouzenoux (Inria Saclay), Marceylo Pereyra (Heriott-Watt University, Edinburgh), Nicolas Gillis (UMONS, Belgium),  and Shota Takahashi (University of Tokyo). This opens the way for new international collaborations. Research on nonnegative KL-divergence problems may also generalize beyond the realm of NMF to quadratic programming [ref TODO] and any nonnegative einsum factorization [ref TODO].


### KL-based inverse problems for computational imaging

Computational imaging has a long history of handling Poisson noise in inverse problems, probably more so than any other field within the scope of signal processing. Historically, Poisson noise was handled by a variance-stabilizing transformation (since Poisson noise variance depends on the signal), such as the Anscombe transform, that transforms the data entrywise such that the noise distribution becomes Gaussian with uniform variance. Variance-stabilizing methods have the advantage of being fast at inference, since the forward and inverse transformations are cheap, and algorithms solving inverse problems with Gaussian noise are generally significantly faster than their KL-divergence counterparts. The Anscombe transform and other similar techniques, however, suffer in the presence of low-count data. KL-divergence is the loss function that appears when computing the maximum likelihood estimator for inverse problems with Poisson noise. While it does not introduce statistical errors like variance-stabilizing transformations, it leads to significantly more expensive algorithms.

In modern signal processing and machine learning algorithms for inverse problems, we build on top of the classical estimation algorithms, such as solvers for NN-KL, with prior information provided by deep models. The general idea is that a neural network can be trained on denoising problems to provide information on the distribution of clean data, which in turns allows to build efficient priors for other inverse problems. The Plug-and-play (PnP) framework [TODO ref] simply plugs a pretrained network in place of a proximal (projection) operator in iterative proximal gradient algorithms, while other methods like RED [TODO ref] use differentiable architectures plugged in the cost function.

Something important to acknowledge is that, while these data-driven techniques allow to significantly improve the estimation performance of algorithms in inverse problems, they inherit from the ups and downs of classical methods from which they are built. In this research project, I believe that while fellow researchers have focused on improving the data-driven aspect of data-driven algorithms for  Poisson inverse problems, based for instance on generalized Tweedie's formula [ref TODO], **the interplay between deep regularizers, classical methods including variance stabilizing transformations, and specific computational imaging applications is under-explored**. There is probably no need for data-driven KL-divergence-based algorithms for high-count data when the Anscombe transform can be applied without issue. There is in contrast a definitive need for a more detailled analysis of when to use heavy computational tools such as PnP with complex NN-KL solvers. Nevertheless, part of this project is also to develop such data-driven Poisson algorithms; this is ongoing work within the PhD thesis of Serena Hariga [TODO ref Gretsi].

We will target our contributions towards the computational imaging modalities developed at CREATIS, in particular [spectral single pixel imaging](../part2/Applications_of_rLRA/single_pixel.ipynb) (SPI). SPI is interesting because the Poison noise level depends on the acquisition technique and on the output wavelength. The energy of incomming photos is split between spectral bands, and some spectral bands recieve significantly fewer photons than others. This kind of data is perfect for analyzing the effect of signal noise on estimation performance. Moreover, the spectral data require unmixing, which can be performed efficiently with NN-KL or NMF-KL in the blind case, inside the reconstruction problem. In the blind case, the problem of estimating sources and abundances in SPI can be formulated as

$$ \argmin{U\geq 0,\; V\geq 0} \KL{Y, HUV^T} + g_1(U) + g_2(V)  $$
where $H$ is the acquisition matrix (such as Hadamard patterns), and $g_i$ are deep or classical regularizations required to ensure interpretability of the results and proper image reconstruction. A particularity of SPI is that the noise can be modeled as a **mixture of Poisson and Gaussian noise**, for which the exact MLE is not known in closed form and is often approximated using variance-stabilizing transformations [TODO ref]. Other interesting modalities include the spectral CT scanner, compton camera, and cryo-EM [TODO refs help!]

Algorithms developed in the second research direction will also be good candidates for extensions to the data-driven framework. Many authors focus on Burg entropy for building data-driven algorithms with Poisson noise [TODO ref], but better results should be achieved by considering better majorants of the KL-divergence, at the cost of more difficult convergence proofs.

**Collaborations**: Single Pixel imaging is the topic of an ongoing, fruitful collaboration with Nicolas Ducros (INSA Lyon, CREATIS). The PhD thesis of Serena Harriga, co-supervised with Prof. Ducros, that started in 2023, is concerned with primal-dual data-driven algorithms for SPI. At the national level, joint spectral unmixing and inverse reconstruction in the context of Poisson noise is one of the topics discussed in an ANR project under construction with HORIBA, a spectrometer manufacturer, also in collaboration with the team DyNaChem of the LASIRE (Lille). 

### Automatic transcription using NMF-KL, neural architectures, and unsupervised learning
%(quite big change from ANR v1)

NMF has a twenty-year long history of usage within music information retrieval. One of its first use cases, which has remained an application of choice for NMF, is Automatic Music Transcription (AMT) [ref Smaragdis TODO]. NMF has shown promising performance for AMT, in particular when KL-divergence (and other $beta$-divergences) are used as loss functions. It has been outperformed by deep networks trained end-to-end for almost a decade [ref Onset Frames].

Deep learning for AMT leverages training samples. The raw comparison with NMF, which has been used as an unsupervised model in AMT, is unfair. However, it is not clear at all how NMF could benefit from training samples. Within this research project, there are two possible routes that both seem promising.

First, one may use the framework of [unrolling](../part2/Fast_algorithms_for_rLRA/UnrolledNMF.md) to train parts of NMF for AMT. With Christophe Kervazo, we have recently proposed unrolled MU updates, with application to spectral unmixing. Our preliminary experiments on unrolled NMF for AMT show that this application is quite challenging because audio data size, sparsity, and dynamics fail current approaches. It should be possible to define a suitable unrolled state-of-the-art algorithm, a training procedure, and compare symmetric unrolling (both W and H) to older approaches that unroll H as a function of W [4].

Second, training deep networks for AMT, and in fact for most music information retrieval tasks, is costly in terms of the quantity of training data required for training. While training data are available for some instruments like piano thanks to the existence of acoustic instruments with activation sensors (Yamaha Disklavier), for rare instruments, it is unlikely that such a database can be acquired in the comming decades. Therefore, there is still room for unsupervised learning algorithms for AMT. NMF models have seen several refinements in recent years regarding identifiability (minimum volume, separability, and, of course, better NMF-KL algorithms) that can be helpful in their fruitful application to a source separation problem like AMT. This project goes further by making use of both these recent NMF developement, and Differential Digital Signal Processing, a state-of-the-art framework for unsupervised learning with deep learning for audio. DDSP essentially training encoder networks to play software synthetisers and reproduce unlabeled instruments recordings, given the knowledge of pitch class and sound amplitude [TODO ref]. NMF could be trained in conjunction with a DDSP model as a generalized pitch detection algorithm. 
 
Research on AMT is by nature translational, and therefore it should be a priority to produces an open source software with a friendly user-interface and a python backend that can be used to transcribe multi-instruments professional recordings, provided that the research is fruitful enough to reach satisfactory performance.

**Collaboration**: I started working on AMT with Nancy Bertin in Rennes, who is currently on leave from CNRS, and with Axel Marmoret, my former PhD student. Axel is still working on AMT but because I want him to be as independent as possible, I refrain from collaborating with him on this topic. Since 2023, we have started to collaborate with Christophe Kervazo and Mathieu Fontaine (Telecom Paris) on [unrolled NMF](../part2/Fast_algorithms_for_rLRA/UnrolledNMF.md) for AMT. We are currently in the process of looking for motivated PhD candidates and funding.
