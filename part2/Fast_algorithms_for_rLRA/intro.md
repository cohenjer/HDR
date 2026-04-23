# Summary

## Optimization challenges in rLRA
Once a rLRA model has been designed for a targeted application, the computation of the rLRA parameters, often called training, parameter estimation, or model fitting, boils down to solving an optimization of the form

$$ \argmin{\forall q\leq d,\;x_q\in\mathbb{R}^{n_q}} f(x_1,x_2,\ldots,x_d) + \sum_q g_q(x_q)$$
with function $f$ a data-fitting term, and $g_q$ mode-wise regularization.

This particular family of problems lies at the intersection of two separate research fields in numerical optimization:
- Multiblock optimization, which is concerned with problems of the form 
  
  $$
    \argmin{\forall q\leq d,\;x_q\in\mathbb{R}^{n_q}} f(x_1,x_2,\ldots,x_d).
  $$

  Alternating optimization and Block-Coordinate Descent algorithms are frameworks usually employed to solve these problems, when the cost function $f$ has "nice" block-wise properties (convexity, smoothness, closed-form minimization...). The current state-of-the-art on alternating optimization and block-coordinate descent is summarized in [](../../part1/AlternatingOptimization.md). Algorithms that update all the parameters simultaneously can also be efficient, but are not explored in this manuscript {cite:p}`Acar2011Scalable,marminJointMajorizationMinimizationNonnegative2023a, takahashiMajorizationMinimizationBregmanProximal2025`.
- Non-smooth optimization, encountered when regularizations $g_q$ are non-smooth. Typical examples in signal processing and machine learning are the $\ell_1$ norm, the $\ell_0$ "norm", or nonnegativity constraints. 

Non-smooth optimization has received a lot of attention in the signal processing optimization community over the last twenty years, as sparse approximations have become more prominent and have required better optimization tools. Among the most significant mathematical tools for non-smooth first-order optimization is the proximity operator, a generalization of projections onto convex sets. The proximity operator is defined for a convex, proper, closed (lower-semi-continuous) function $g$ as 

$$ \text{prox}_{\lambda g}(x) = \argmin{u\in\mathbb{R}^n} g(u) + \frac{1}{2}\|u - x \|_2^2 $$
where $\lambda$ is a positive scaling parameter. The proximity operator of many classic regularizations is known in closed form or can be efficiently computed, a [list is available online](http://proximity-operator.net/). The proximity operator of the characteristic function of a convex set, which penalizes to $+\infty$ any vector outside that set, is exactly the projection on that convex set. The particular case of nonnegativity constraints has been studied extensively and is summarized in [](../../part1/nnls.md).

A third field of research in optimization, sometimes encountered when fitting rLRA models, concerns cost functions that are not Lipschitz-smooth. A typical example is the Kullback-Leibler divergence, which plays a central role in nonnegative low-rank approximations. First-order algorithms are hard to derive in this context, see the discussion in [](../../part1/nnls.md#nonnegative-kullback-leibler-regression-nnkl).

Because rLRA leads to optimization problems that mix multiblock, non-smooth, and sometimes non-Lipschitz-smooth optimization, fitting rLRA models efficiently is often challenging.

## Contributions

My contributions in the last ten years on numerical optimization have been geared towards two goals:
- Making optimization algorithms for rLRA faster (for a fixed computational power).
- Making optimization algorithms for rLRA more flexible.

### Faster algorithms for rLRA

Faster optimization algorithms are often the focus of numerical optimization research, as they reduce energy and time consumption for end users. In applications where the rLRA model must be fitted in real time, training speed is a key factor that can even drive model choice, at the cost of increased reconstruction error. A typical example is separable NMF, which can be fitted with polynomial-time algorithms but yields higher errors than fully blind NMF.

I have several contributions in this direction. First, in the context of nonnegative LRA, in collaboration with Mai Quyen Pham, we proposed a tight second-order approximation technique coined [median Second-Order Majorant](../Fast_algorithms_for_rLRA/mSOM.md) (mSOM) that is similar to multiplicative updates in theory, but has provable linear convergence and speeds up convergence in practice {cite:p}`phamSecondOrderMajorantAlgorithm2025`. This contribution is important in my opinion since it proposes a novel idea for constructing local majorants of loss functions with nonnegative Hessian matrices, that can be developed in several ways we have yet to explore; see the discussion in [](../../part3/KarpCoi.md). It is also a step towards algorithmic design for computational imaging, a family of applications that I grew interested in after my mutation to CREATIS, see [](../Applications_of_rLRA/Single_pixel_spectral_imaging.md).

Second, in a collaboration with Christophe Kervazo, we have studied unrolled algorithms for NMF. The key idea in unrolling is to leverage training pairs of inputs and outputs from an optimization algorithm to tune certain parameters in the model, or hyperparameters in the algorithm itself. The gradient of a supervision loss is computed after the optimization algorithm has converged, and any first-order method can then be used to upgrade these parameters. Our contribution is to [unroll the multiplicative updates algorithm for NMF](../Fast_algorithms_for_rLRA/UnrolledNMF.md). This is challenging because multiplicative updates have no hyperparameters and because of NMF's multiblock nature. While earlier works had considered unrolling the updates for a single block, we unroll the updates for both blocks by introducing trainable masked matrices. Unrolled multiplicative updates are not only more precise when trained properly, but they are also significantly faster to converge than multiplicative updates.

Third, in a collaboration with Andersen Man Shun Ang, we proposed a [heuristic extrapolation strategy for alternating optimization](../Fast_algorithms_for_rLRA/inertial_BCD.md). After each block update, the estimated parameters are extrapolated from the current and previous iterates. The actual implementation includes two important details: restarting and pairing sequences. Restarting cancels any extrapolation if it increases the cost, thereby guaranteeing that the cost decreases at each iteration. Pairing sequences are used in analogy to Nesterov's fast gradient. While this work was of interest to me at that time, more recent works have been proposed that perform similar extrapolation strategies but with convergence guarantees {cite:p}`hienBlockMajorizationMinimization2025`.

Fourth, in collaboration with Nicolas Nadisic, Arnaud Vandaele, and Nicolas Gillis, we studied branch-and-bound algorithms to compute the exact solution to medium-scale [sparse NNLS problems](../Fast_algorithms_for_rLRA/sparse_nnls.md). These problems typically arise in source separation, where each mixture at each measurement involves only a small number of spectra. Solving sparse NNLS exactly as fast as possible using combinatorial techniques is useful to design further algorithms such as [sparse separable NMF](../../introduction/summary.md#sparse-separable-nmf).

In my PhD thesis, I was already concerned with fast algorithms for nonnegative tensor decomposition. Because my understanding of this problem has evolved significantly since then, I also included a section analysing one of my first contributions, [projected-constrained alternating least squares](../Fast_algorithms_for_rLRA/proco-als.md).

### Flexible algorithms for rLRA

From my experience, end users of rLRA algorithms and software often try various constraints, ranks, data preprocessing, and postprocessing before settling on a setup they feel comfortable with. Therefore, designing algorithms for rLRA that support a wide range of constraints, shapes, and data types is crucial in practice. My contribution to this goal is within the context of multimodality, in which rLRA is applied to multiple matrices or tensors simultaneously. Leveraging earlier work on ADMM for rLRA {cite:p}`Huang2016flexible`, in a series of collaborations with Rasmus Bro, Evrim Acar, Carla Schenker, and Marie Roald, we proposed an ADMM-based algorithm for coupled factorizations, which also allows for flexible coupling designs between the multimodal dataset. The main concepts in coupled factorizations are described in [](../Fast_algorithms_for_rLRA/CMTF.md), while a more specific work on nonnegative PARAFAC2 is detailed in [](../Fast_algorithms_for_rLRA/NNParafac2.md). A follow-up contribution on an ADMM-based algorithm for PARAFAC2 with more general constraints is succinctly discussed in [](../Fast_algorithms_for_rLRA/NNParafac2.md) and merges ideas from these two works {cite:p}`roaldAOADMMApproachConstraining2022a`.
