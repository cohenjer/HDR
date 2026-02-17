# Summary

## Optimization challenges in rLRA
Once a Regularized LRA model has been designed for a targeted application, the computation of the rLRA parameters, often called training, parameter estimation or model fitting, boils down to solving an optimization of the form

$$ \argmin{\forall q\leq d,\;x_q\in\mathbb{R}^{n_q}} f(x_1,x_2,\ldots,x_d) + \sum_q g_q(x_q)$$
with function $f$ a data-fitting term, and $g_q$ mode-wise regularization.

This particular family of problems lies at the intersection of two separate reseach fields in numerical optimization:
- Multiblock optimization, that is concerned with problems of the form
$  \argmin{\forall q\leq d,\;x_q\in\mathbb{R}^{n_q}} f(x_1,x_2,\ldots,x_d).$
 Alternating optimization and Block-Coordinate Descent algorithms are frameworks usually employed to solve these problems, when the cost function $f$ has "nice" block-wise properties (convexity, smoothness, closed-form expression...). The current state-of-the-art on alternating optimization and block-coordinate descent is summarized in [](../../part1/AlternatingOptimization.md). Algorithms that update all the parameters simultaneously can also be efficient, but are not explored in this manuscript [ref Fevotte all at once, ref Takahashi, ref Evrim CMTF].
- Non-smooth optimization, encoutered when regularizations $g_q$ are non-smooth. Typical examples in signal processing and machine learning are the $\ell_1$ norm, the $\ell_0$ norm or nonnegativity constraints. 

Non-smooth optimization has recieved a lot of attention in the optimization community specialized in signal processing in the last twenty years, as sparse approximations were becoming more prominent and required better optimization tools. Among the most significant mathematical tool for non-smooth first-order optimization is the proximity operator, a generalization of projections onto convex sets. The proximity operator is defined for a convex, proper closed (lower-semi-continuous) function $g$ as 

$$ \text{prox}_{\lambda g}(x) = \argmin{u\in\mathbb{R}^n} g(u) + \frac{1}{2}\|u - x \|_2^2 $$
where $\lambda$ is a positive scaling parameter. The proximity operator of many classic regularizations is known in closed form or can be efficiently computed, a [list is available online](http://proximity-operator.net/). The proximity operator of the caracteristic function of a convex set, that penalized to $+\infty$ any vector outside that set, is exactly the projection on that convex set. The particular case of nonnegativity constraints has been studied extensively and is summarized in [](../../part1/nnls.md).

A third field of research in optimization sometimes encoutered when fitting rLRA models concerns cost functions that are not Lipschitz-smooth. A typical example is the Kullback-Leibler divergence, that plays a central role in nonnegative low-rank approximations. First-order algorithms are hard to dervied in this context, see the discussion in [](../../part1/nnls.md#nonnegative-kullback-leibler-regression-nnkl).

Because rLRA leads to optimization problems mixing multiblock, non-smooth and sometimes non-Lipschitz-smooth optimization, fitting rLRA models efficiently is often a challenging task.

## Contributions

My contributions in the last ten years on numerical optimization have been geared towards two goals:
- Making optimization algorithms for rLRA faster (for a fixed computational power).
- Making optimization algorithms for rLRA more flexible.

### Faster algorithms for rLRA

Faster optimization algorithms is often the target of numerical optimization research, as it allows to reduce energy and time consumption for the end user. In applications where the rLRA model must be fitted in real-time, training speed is a key component that can even drive the model choice at the cost of increased reconstruction error. A typical example is separable NMF that can be fitted with polynomial-time algorithms, but typically provides higher errors than fully blind NMF.

I have several contributions in this direction. First, in the context of nonnegative LRA, in collaboration with Mai Quyen Pham, proposed a tight second-order approximation technique coined [median Second-Order Majorant](../Fast_algorithms_for_rLRA/mSOM.md) (mSOM) that is similar to multiplicative updates in theory, but has provable linear convergence and speeds up convergence in practice [TODO ref]. This contribution is important in my opinion since it build on a novel idea for building local majorant of loss functions with nonnegative Hessian matrices, and this idea can be developped in several ways we have yet to explore, see the discussion in [](../../part3/KarpCoi.md). It is also a step towards algorithmic design for computational imaging, a family of applications that I grew interested in after my mutation to CREATIS, see [](../Applications_of_rLRA/Single_pixel_spectral_imaging.md).

Second, in a collaboration with Christophe Kervazo, we have studied unrolled algorithms for NMF. The key idea in unrolling is to leverage training pairs of inputs and outputs of an optimization algorithm in order to tune certain parameters in the model, or hyperparameters in the algorithm itself. The gradient of a supervision loss is computed after the optimization algorithm has converged, and any first-order method can then be used to upgrade these parameters. Our contribution is to [unroll the multiplicative updates algorithm for NMF](../Fast_algorithms_for_rLRA/UnrolledNMF.md). This is challenging because multiplicative updates have no hyperparameters, and because of the multiblock nature of NMF. While earlier works had considered unrolling the updates for a single block, we unroll the updates for both blocks by introducing trainable masked matrices. Unrolled multiplicative updates are not only more precise when trained properly, they are also significantly faster to converge than multiplicative updates.

Third, in a collaboration with Andersen Man Shun Ang, we proposed a [heuristic extrapolation strategy for alternating optimization](../Fast_algorithms_for_rLRA/inertial_BCD.ipynb). After each block update, the estimated parameters are extrapolated based on the current and previous iterate. The actual implementation includes two important details: restarting and pairing sequences. Restarting cancels any extrapolation if it increases the cost, which allows to guarantee that the cost decreases at each iteration. Pairing sequences are used in analogy to Nesterov's fast gradient. While this work was of interest to me at that time, more recent works have been proposed that perform similar extrapolation strategies but with convergence guarantees [ref hien TODO].

Fourth, in collaboration with Nicolas Nadisic, Arnaud Vandaele and Nicolas Gillis, we studied branch and bound algorithms to compute the exact solution of medium-scale [sparse NNLS problems](../Fast_algorithms_for_rLRA/sparse_nnls.ipynb). These sparse NNLS problem are typically encountered in source separation problems where the mixtures at each measurement only involve a small number of spectra. Solving sparse NNLS exactly as fast as possible using combinatorial techniques is useful to design further algorithms such as [sparse separable NMF](../../introduction/summary.md#sparse-separable-nmf).

In my PhD thesis, I was already concerned with fast algorithms for nonnegative tensor decomposition. Because my understanding of this problem has evolved significantly since then, I also included a section analysing one of my first contribution, [projected-constrained alternating least squares](../Fast_algorithms_for_rLRA/proco-als.ipynb).

### Flexible algorithms for rLRA

From my experience, end users of rLRA algorithms and softwares will often try various constraints, ranks, data-preprocessing or post-processing, before achieving a setup they feel confortable with. Therefore, designing convergent algorithms for rLRA that allow for a large range of constraints, shapes and data types is of crucial important in practice. My contribution towards this goal is within the context of multimodality, when employing rLRA on several matrices or tensors simultaneously. Leveraging earlier work on ADMM for rLRA [TODO ref Sidi], in series of collaborations with Rasmus Bro, Evrim Acar, Carla Schenker and Marie Roald, we proposed an ADMM-based algorithm for coupled factorizations, that also allows for flexible coupling designs between the multimodal dataset. The main concepts in coupled factorizations are described in [](../Fast_algorithms_for_rLRA/CMTF.md), while a more specific work on nonnegative PARAFAC2 is detailled in [](../Fast_algorithms_for_rLRA/NNParafac2.ipynb). A follow-up contribution on an ADMM based algorithm for PARAFAC2 with more general constraints is not described in this manuscript but relies on merging ideas from these two works [TODO cite].