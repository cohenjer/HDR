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


# Alternating Optimization

All the practical problems considered in the manuscript rely on solving an optimization problem that involve $d$ different blocks, of the form

$$ \argmin{x_1\in\mathcal{X}_1,~\ldots~,x_d\in\mathcal{X}_d} f(x_1,x_2,\ldots,x_n) $$

where $f:\mathcal{X}_1\times \ldots \times \mathcal{X}_{d} \mapsto \mathbb{R}_+$ is a positive cost function and $\mathcal{X}_k$ are vector spaces embedded in $\mathbb{R}^{n_k}$. Indeed, low-rank approximation models are multifactor models; each block $x_i$ may represent a factor matrix in a low-rank model. For instance, solving a rank $r$ approximate nonnegative matrix factorization problem in the presence of Gaussian noise may result in the following two-block optimization problem

$$ \argmin{X_1\in \mathbb{R}_+^{n_1\times r},~X_2\in \mathbb{R}_+^{r\times n_2}} \|Y - X_1X_2 \|_F^2 $$

for a known data matrix $Y\in\mathbb{R}^{n_1\times n_2}$. In this example, and in fact in most problems involving LRA, the cost has interesting blockwise properties. Here the cost is not convex because of the product of variables, but it is convex, and often strongly convex in practice, with respect to each block. This observation hints towards the design of a class of convergent algorithms to compute solutions to LRA problems that update each block sequentially.

This section provides a short and incomplete overview of existing results regarding methods that alternatively update the blocks of variables. These methods are clustered in two groups: Alternating Optimizations methods (AO) where the cost is exactly minimized with respect to each block alternatively, and Block Coordinate Descent methods (BCD) where the cost decreases with respect to each block alternatively. BCD methods considered here are all, in fact, particular cases of Alternating Maximization Minization algorithms that we also shortly discuss. The litterature on both AO and BCD is both old and vast. I will mostly present recent results of practical interest for the rest of the manuscript, but an interested reader should find in these recent references many ackowledgments of previous works, especially regarding AO, that were conducted in the seventiees. This section is also similar to, and heavily inspired from, chapters 11 and 14 from the book of Amir Beck {cite}`Beck2017first`, and to Section 8.1.3 in the book of Nicolas Gillis {cite}`gillisNonnegativeMatrixFactorization2020`. The book of Beck also constitutes a great introduction to convex optimization in general, which can be completed by the book of Dimitri Bertsekas {cite}`Bertsekas1999Nonlinear`.

We build the rest of this section as such. First, general definitions and common assumptions are quickly recalled. Then AO and BCD are introduced sequentially, going from the more general results to the more specific ones.

## General assumptions

define important assumptions and notions

### Defs

do ou pas ?? je dis do pas

#### For singleblock functions
- Continuously differentiable (C1)
- directional differentiability (only positive mouvements, therefore includes l1)
- Kurdika-Lojiasievich (KLo)
- Global min vs Local min vs Point Critique

#### For multiblock functions
- coordinate-wise minimum
- block-Lipschitz
- block-convex
  
### Summary of assumptions and convergence results

```{margin}
 Block Lischitz-smoothness does not imply global Lipschitz-smoothness for nonconvex functions.
```

Assume the splitting $$ f(x) = f_0(x) + r_i(x_i). $$
- (A1): $\mathcal{X}_k$ are closed convex sets.
- (A2): $f$ or $f_0$ has bounded level sets $I(z)=\{x | f(x) \leq z\}$.
- (A3): $f$ or $f_0$ is proper, closed, and continuous on its domain.
- (A4): $r_i$ is proper, convex, lower semi-continuous.
- (A5): $f$ or $f_0$ is globally Lipschitz-smooth
- (A6): $f$ or $f_0$ is differentiable in the interior of its definition domain.
- (A7): $f$ or $f_0$ is continuously differentiable.
- (A8): $f_0$ is block Lipschitz-smooth
- (A9): $f$, $f_0$ or $r_i$ admits directional differentials defined at $x$ along direction $d$ as $$f'(x,d) = \underset{\lambda \to 0}{\lim\inf}\frac{f(x\lambda d) - f(x)}{\lambda}$$ (notice that $\lambda$ is positive). This includes non-smooth functions such as the $\ell_1$ norm.
- (A10): The block updates $\argmin{x_k\in\mathcal{X}_k} f(x_1,\ldots,x_k,\ldots,x_d)$ has a unique minimizer.
- (A11): $f$ is convex.
- (A12): $f$ is block quasi-convex

We use the notation (A9r) to denote, for instance, assumption (A9) applied on the maps $r_i$. Table xxTODO summarizes the various convergence results for AO and BCD with their assumptions.

| Reference  | Assumptions  | Convergence | 
|---|---|---|
| {prf:ref}`th:AO`  | A3f, A2f, A10f  | limit points are coordinate-wise minimum  |
| {prf:ref}`th:AO2`  |   |   |  
| {prf:ref}`th:AO3`  |   |   | 
| {prf:ref}`th:AO4`  |   |   | 
| {prf:ref}`th:AO5`  |   |   | 
| {prf:ref}`th:AO6`  |   |   | 
| {prf:ref}`th:sum`  |   |   | 
| {prf:ref}`th:bsum1`  |   |   | 
| {prf:ref}`th:bsum2`  |   |   | 


(sec:mm)=
### Majorization Minimization principle
TODO
Stay general, cite Kenneth Lange SIAM book on MM or TSP paper Song.
MM and its relation with the expectation maximization framework is also discussed in [](./nnls.md).

## Alternating Optimization

Alternating Optimization (AO), sometimes also called exact block-coordinate descent, minimizes the cost function $f$ sequentially for each block of variable, essentially performing cyclic updates

$$
  x^{(i+1)}_{k} = \argmin{x_{k}\in\mathcal{X}_k} f(x_1^{(i+1)},\ldots, x^{(i+1)}_{k-1}, x_k, x^{(i)}_{k+1}, \ldots, x^{(i)}_d ) 
$$ (eq:AOupdate)

at iteration $i+1$ and for the $k$th block. It is immediate to observe that because the cost $f$ is positive, and because each AO update diminishes the cost, the sequence of cost function values is always decreasing. Therefore the AO algorithm converges, in cost, towards a positive value. Interesting questions therefore are not to know if the AO algorithm converges in cost, but rather
- does AO converge in cost to a stationary point, and at what rate ?
- are the sequences of iterates $\{x^{(i)}_k\}_{k\leq d}$ converging towards a stationary point in the non-convex case, or the global minimum in the convex case, and at what rate ?

A famous counter-example was found by Powell in 1973 {cite}`powellSearchDirectionsMinimization1973`. According to Beck{cite}`beck2017first`, this counter-example occurs because the minimization subproblems with respect to each block do not have a unique solution (assumption xx). We will see below that most AO convergence guarantees rely on this assumption. One may note the similarity with the assumptions for the [convergence of MM](#majorization-minimization-principle) to stationary points. 
For $d=2$ blocks, singe-block MM and AO are in fact equivalent {cite}`sunMajorizationMinimizationAlgorithmsSignal2017`. Denoting $x_2(x_1) = \argmin{x_2} f(x_1,x_2)$, we get that $f(x_1,x_2(x^{(i)}_1))\geq f(x_1, x_2(x_1))$. Two blocks AO is exactly the computation of, first, $x_2(x_1^{(i)})$, and then the minimization of the majorant $f(x_1,x_2(x_1^(i)))$. This majoration can be loose for arbitrary functions without inter-block regularity, which can intuitively explain the slow convergence discussed in [](#convergence-rate-of-ao).
%[WRONG] This can be understood intuitively as the block updates of AO are block MM updates where the majorant is constant over all but one block. This majorization can be loose for arbitrary convex functions without inter-block regularity, which can intuitively explain the theoretical slow convergence discussed in [](#convergence-rate-of-ao).

Another immediate issue with AO is that coordinate-wise minima are not necessarily stationary points. Beck also shows this is wrong even when the cost is convex. The root of the problem is that the Fermat rule $0\in\partial f$ does not decompose into separate conditions for each block, unlike the gradient for differentiable functions: it is always true that $\nabla f = 0$ is equivalent to $\frac{\partial}{\partial x_i} f = 0 $, but for non-differentiable functions, $0\in\partial f$ only implies $0\in\partial f_k$ for all $k\leq d$ while the converse is not true. This explains why most works on AO that consider non-smooth costs introduce additive separable non-smooth regularizations $r_i$ to a differentiable multi-block cost $f_0$,

$$ f(x_1,\ldots, x_d) = f_0(x_1,\ldots,x_d) + \sum_{k=1}^{d} r_k(x_k) $$

where $f_0$ is proper, continuously differentiable, and functions $r_k$ are proper, convex, lower-semicontinuous. [Merge this with assumptions sec ?].

A typical example where the constraints are non-smooth and non-separable are linear couplings constraints between variables; AO for these problems is bound to get stuck in a coordinate-wise minimum which can be irrelevant for the problem at hand.

### Different flavours of convergence of AO iterates

```{margin}
The book has been edited three times, in 1995, 1999 and 2016.
```

Probably the most well-known result on the convergence of AO is due to Dimitry Berstekas and was published in his book Nonlinear Programming in 1995 {cite}`Bertsekas1999Nonlinear`. There have been a number of iterations of his result, in particular to differentiate between convergence to coordinate-wise minima and stationary points. Below is a summary of Bertsekas's result and its variations.


```{prf:theorem} Convergence of AO, non-convex non-smooth
:label: th:AO
Assume function $f$ is (A1) closed, proper and continuous over its domain. Assume that (Axx) each block update {eq}`eq:AOupdate` has a unique solution. Assume also that (Axx) the level sets of function $f$ are bounded. Then the sequence of AO iterates is bounded, and any limit point is a coordinate-wise minimum.


From {cite}`Bertsekas1999Nonlinear,beck2017first`
```

As discussed above, the convergence to coordinate-wise minima is of little use for non-smooth functions, since they are not necesarily stationary points. Therefore while this first result does not require differentiability, in practice it should be used with caution in a non-differentiable setup; hence the splitting of smooth and non-smooth terms in this second theorem.

```{prf:theorem} Convergence of AO, non-convex smooth + separable convex non-smooth
:label: th:AO2
Assume function $f$ is (A1) closed, proper and continuous over its domain. Assume that (Axx) each block update {eq}`eq:AOupdate` has a unique solution. Assume also that (Axx) the level sets of function $f$ are bounded.

If moreover (Axx) function $f$ decomposes as $f(x) = f_0(x) + \sum_{i\leq d} r_i(x_i)$ with $r_i$ proper, closed, continuous over its domain and convex, and $f_0$ is a differentiable function on the interior of (Axx) a Cartesian product of closed convex sets, then any limit point of the AO iterates is a stationary point. 

From {cite}`beck2017first`
```

There are two further versions of this result. First, one may want to relax assumption (Axx) that the block updates have unique solutions; this assumption can indeed be too strong in practical cases where convergence is yet observed. This assumption can be easily relaxed if the cost $f$ is convex.

```{prf:theorem} Convergence of AO, convex smooth + separable convex non-smooth
:label: th:AO3
Assume function $f$ is (A1) closed, proper and continuous over its domain. Assume also that (Axx) the level sets of function $f$ are bounded.

If moreover (Axx) function $f$ decomposes as $f(x) = f_0(x) + \sum_{i\leq d} r_i(x_i)$ with $r_i$ proper, closed, continuous over its domain and convex, and $f_0$ is a continuously differentiable **convex** function, then any limit point of the AO iterates is a stationary point. 

From {cite}`beck2017first`
```

A second version that applies in the non-convex case discards the splitting and the boundness of the level sets assumptions. To obtain convergence to a stationary point, differentiability is required. The level set assumption is replaced by the requirement that along the block updates, the cost is non-increasing, a property ressembling pseudo-convexity often used in earlier AO convergence proofs. This is actually the original result by Bertsekas.

```{prf:theorem} Convergence of AO, non-convex smooth 
:label: th:AO4
Assume function $f$ is (A1) closed, proper and (Axx) continuously differentiable over its domain, (Axx) which is a Cartesian product of closed convex sets. Assume that (Axx) each block update {eq}`eq:AOupdate` has a unique solution. 

Assume futher that the cost is non-increasing in the interval 

$$
  \left[(x^{(i+1)}_1,\ldots,x^{(i+1)}_{k-1}, x^{(i)}_k, x^{(i)}_{k+1}, \ldots, x^{(i)}_d), (x^{(i+1)}_1,\ldots, x^{(i+1)}_{k-1}, x^{(i+1)}_k, x^{(i)}_{k+1}, \ldots, x^{(i)}_d)\right].
$$

Then any limit point of the AO iterates is a stationary point. 

From {cite}`Bertsekas1999Nonlinear`
```

It is important to note that the order of the AO block updates are not important in these results, as long as there exist an integer $K$ such that each block is updated at least once in every $K$ AO iterations {cite}`gillisNonnegativeMatrixFactorization2020`.

### Convergence rate of AO

The only derivation of the convergence rate for the AO algorithm with arbitrarily many blocks that I am aware of is found in the book of Amir Beck {cite}`beck2017first`, in the case where $f$ follows assumption (Axx). It shows that AO in general convergence sublinearly in the convex, Lipschitz-smooth case; AO has the same assymptotic convergence speed as first-order methods. Moreover, the convergence rate depends linearly on the Lipschitz constant $L_{f_0}$ of the whole function $f_0$, and therefore on the worst Lipschitz constant over all blocks. In practice, AO can be much faster than all-at-once optimization such as gradient descent performed over $\mathcal{X}_1 \times \ldots \times \mathcal{X}_d$, see the discussion in {ref}`sec:practical_issues_AO`.

### AO with $d=2$ blocks

An important special case is AO with two blocks. The convergence of AO with two blocks has been studied in the seminal paper of Grippo and Scriandrone {cite}`Grippo2000convergence`. 

```{prf:theorem} Convergence of AO, $d=2$, non-convex smooth
:label: th:AO5
Assume (Axx) function $f$ is continuously differentiable, and assume (Axx) sets $\mathcal{X}_k$ are closed convex sets. Then the limit points of the AO iterates are stationary points.

```

Notice how the assumptions for the two blocks case are milder than in the general case. Compared to {prf:ref}`th:AO4`, the two-block case requires neither the uniqueness of block updates nor the monotonicity of the cost along the update path. The two blocks case is useful when considering the convergence of alternating algorithms for nonnegative matrix factorization.


```{note}
The convergence rate of AO with two block is still sublinear under convexity and separable non-smoothness assumptions, but the constant is now proportional to the **smallest** Lipschitz constant of $f_0$ for the two blocks, a sharp improvement with respect to the $d>2$ blocks case {cite}`beck2017first`. Moreover, the function $f_0$ does not need to be Lipschitz-smooth and may be differentiable only on an open set.
```

Grippo and Scriandrone also provide a convergence result for $d>2$ blocks based on block-wise strict quasiconvexity of the cost for $d-2$ blocks and continuous differentiability of the cost. This result requires both quasiconvexity and differentiability of the cost, and is therefore weaker than the variants introduced above.

```{prf:theorem} Convergence of AO, block strictly quasiconvex, smooth
:label: th:AO6
Assume (Axx) function $f$ is continuously differentiable, and assume (Axx) sets $\mathcal{X}_k$ are closed convex sets. Further assume that (Axx) function $f$ is blockwise strictly quasiconvex with respect to $d-2$ blocks. Then the limit points of the AO iterates are stationary points.
```

Because this result does not require the uniqueness of the block updates and does not require convexity with respect to all blocks, it can sometimes be applied in contexts where {prf:ref}`th:AO2` cannot.  

(sec:bcd)=
## Block-coordinate descent

```{margin}
As for AO, in BCD, the blocks can be visited in any order as long as each block is visited at least once every $K$ iterations for a fixed integer $K$. In particular, each block can be updated several times. If a block is updated many times, a BCD algorithm may approximaly behave like an AO algorithm, see the discussion on the practical consequences of this observation in [](#practical-issues).
```

AO is a simple framework for multiblock optimization problems, but its convergence guarantees can be hard to satisfy. In particular, computing exactly the block updates and ensuring the uniqueness of the update can be challenging. The convergence rate is also sublinear, and one may hope to find faster alternating algorithms. Block-Coordinate Descent (BCD) is another algorithmic framework where the blocks are updates sequentially (in any order), but each block updates may simply reduce the cost function value. The next paragraphs introduce useful BCD frameworks, that are all based on the MM principle. 

#### Successive upper-bound minimization
by Tom (Zhi-Quan) Zuo

(move beginning to MM ?)

The Successive Upper-bound Minimization framework (SUM), proposed by Razaviyayn, Hong and Luo in 2013 {cite}`razaviyaynUnifiedConvergenceAnalysis2013`, is an extension of the MM principle, where the majorant must be continuous, and have the same first-order derivative as the cost at the majoration point in all directions. This requirement is rather weak in practice as many MM algorithms rely on Taylor expensions to build the majorants, in which case this tangent condition holds.

The SUM framework is then very simple. Consider the following algorithm.
1. At a given point $x^{(i)}$, build a continuous majorant $u(y,x^{(i)})$ of the cost $f$, tight and tangent.
2. Compute $x^{(i+1)}$ as a minimizer of the majorant $u(y,x^{(i)})$ over $y$.
3. Repeat 1. and 2. until convergence.

The convergence of SUM is surprinsingly simply and powerful to study MM algorithms.

```{prf:theorem} Convergence of SUM, non-convex non-smooth
:label: th:sum

Assume (Axx) the cost function $f$ admits directional derivatives at all points, and that (Axx) the set $\mathcal{X}$ is closed and convex. Further assume that the majorant $u$ in SUM satisfies the majoration conditions, namely it is tight, tangent in all directions and upper-bounds the cost at all points. Then every limit point of the SUM iterates is a stationary point. If, further, the level set $\{x| f(x)\leq f(x^{(0)}) \}$ is compact, then the iterates converge to the set of stationary points.

From {cite}`razaviyaynUnifiedConvergenceAnalysis2013`.
```

A typical use of the SUM framework is when the cost decomposes as $f(x) = f_0(x) + r(x)$, with $f$ continuously differentiable and $r$ a proximable regularization with direction derivatives such as the $\ell_1$ norm. Then SUM where $f$ is majorized by a second-order Taylor expansion of $f_0$ with an isotropic quadratic term is exactly a proximal gradient descent algorithm. SUM also covers the multiplicative updates algorithm discussed in [](nnls.md#multiplicatives-updates). In general, the convergence rate of SUM is therefore at best sublinear.

### The block successive upper-bound minimization framework

```{margin}
While PALM is arguably more well-known in the French numerical optimization community, in my opinion it offers little more than the BSUM framework. BSUM is more general than PALM. When the majorants in BSUM are block-wise second-order isotropic Taylor expansions of the differentiable part of the block-wise Lipschitz-smooth cost, BSUM is exactly PALM. The convergence guarantees of BSUM to coordinate-wise minima (and stationary points with regularity assumptions) are satisfied since the majorants are strongly convex and the updates have unique solutions. PALM leverages the Kurdyka-Łojasiewicz property and assumes a splitting as continuoustly differentiable plus serapable non-smooth terms to guarantee convergence towards a critical point without directional derivability or regularity assumptions. The proof techniques in SUM and BSUM are also arguably simpler than in PALM, these frameworks are thus more ammenable to an introductory course on alternating optimization techniques.
```

The extension of SUM to a multiblock function is called Block SUM (BSUM) and is a generic framework that encompases many other BCD algorithms such as PALM, under the hypothesis that the cost admits directional derivatives. BSUM is exactly the application of SUM sequentially over all blocks, in any order. Compared to SUM, BSUM requires additional assumptions for convergence, which come in two flavors. First, one may assume that the majorants are quasi-convex, and that the block updates have unique solutions.

```{margin}
Regular in this context means that the Fermat rule implies stationarity.
```

```{prf:theorem} Convergence of BSUM, quasi-convex non-smooth
:label: th:bsum1

Assume (Axx) the cost function $f$ admits directional derivatives at all points, and that (Axx) the sets $\mathcal{X}_k$ are closed and convex. Further assume that the majorants $u$ in BSUM satisfies the majoration conditions for all blocks, namely the majorants are tight, tangent in all directions and upper-bound the cost at all points. Additionally, assume the marjorants for each block are quasi-convex, and that the block updates have unique solutions.

Then every limit point of the BSUM iterates is a coordinate-wise minimum. If, further, the cost is regular at all these limit points, then the limit points of the BSUM iterates are stationary points.

From {cite}`razaviyaynUnifiedConvergenceAnalysis2013`.
```

A second version of this convergence result avoids the quasi-convexity assumption, and shows the convergence of the iterates towards the set of stationary points rather than the limit points of the iterates. It relies on the compacity of the set $\{x| f(x)\leq f(x^{(0)}) \}$.

```{prf:theorem} Convergence of BSUM, non-convex non-smooth
:label: th:bsum2

Assume (Axx) the cost function $f$ admits directional derivatives at all points, and that (Axx) the sets $\mathcal{X}_k$ are closed and convex. Further assume that the majorants $u$ in BSUM satisfies the majoration conditions for all blocks, namely the majorants are tight, tangent in all directions and upper-bound the cost at all points. Additionally, assume the level set $\{x| f(x)\leq f(x^{(0)}) \}$ is compact, and that the block updates have unique solutions for at least $d-1$ blocks.
Further, assume the cost is regular at all the stationary points. 

Then the BSUM iterates converge to the set of stationary points.

From {cite}`razaviyaynUnifiedConvergenceAnalysis2013`.
```

It is important to note that, similarly to AO, the BSUM framework assumes that the constraint sets are separated. In particular, BSUM cannot handle linear couplings between blocks of variables, and in fact BSUM, like AO, is bound to fail in this setup.

The convergence rate of BSUM was studied in the convex case {cite}`hongIterationComplexityAnalysis2017`. BSUM, PALM and other similar first-order BCD algorithms have, in general, sublinear convergence rates, see also chapter 11 of the book of Beck {cite}`beck2017first`.

%Convergence rates derived later in 2017 but in the convex, non-smooth case.

%- BSUM CVG rate
%First deposited on arxiv in 2013 !! accepted in 2017
%Similar results as Shefi and Teboulle 2016, convergence rate of PALM.

%BSUM but in the non-smooth case **BUT convex**, also addresses Block minimization (AO)
%- Convergence results (Hong 2017) with ADMM
%  - BSUM sublinear convergence $1/r$, if each subproblem is strongly convex
%  - accelerated BSUM two block: $1/r^2$ convergence rate, no strong convexity and one block need not be gradient L-smooth

%#### PALM and variants
%- PALM, APGD (special cases), Beck p331 theorème 11.14 pour BPGD non convexe cvg vers point critique, 11.18 pour linear rate en cvx

%#### SGD ?
%- Stochastic gradient descent


%## Frameworks
%PALM: ref with beck and tetruashvii 2013 but special case, PALM 2014 Bolte Teboulle Shefi, linear rate from Teboulle and Shefi in 2016

#### BCD with extrapolation
An important research direction to speed up BCD algorithms is to leverage extrapolation of the iterates, inspired by the seminal work of Nesterov on the fast gradient algorithm {cite}`Nesterov1983method`. Going into the details of all the proposed extrapolated BCD methods would be cumbersome, therefore here is a list of a few useful works. These methods typically also have sublinear rates, but improved from $\mathcal{O}(i^{-1})$ to $\mathcal{O}(i^{-2})$ where $i$ is the iteration index.

- Alternating Progximal Gradient Descent (APGD) {cite}`xu2013block` is a general framework for alternating algorithms, that covers AO and PALM with extrapolation. APGD assumes that the cost is the sum of a term strongly convex (for AO) or Lipschitz smooth (for PALM with extrapolation) with respect to each block and separable convex nonsmooth regularizations. The originality is the general conditions with assymptotic rates, and the extrapolation for PALM.
- iPALM is an extension of PALM with extrapolation {cite}`Pock2016`. It is similar to APGD but relaxes a condition on the monotonicity of the cost along the update path.
- TITAN {cite}`hienInertialBlockMajorization2023` is essentially the BSUM framework with extrapolation. TITAN combines ingredients from PALM, such as the Kurdyka-Łojasiewicz property, with heavy-ball acceleration. A downside of TITAN is that it introduces, among others, the nearly sufficiently decreasing property that may not be satisfied when the majorants are not strongly convex. This condition can be relaxed provided other assumption on the extrapolation parameters, making use of the framework of mirror gradient descent to include the extrapolation term in the majorant definition {cite}`hienBlockMajorizationMinimization2025`.

(sec:practical_issues_AO)=
## Practical issues

From the convergence properties summarized above, it is far from obvious wether AO or BCD should be preferred in practice. What's more, BCD and AO are optimization frameworks, but their actual implementation can greatly impact their performance. For BCD in particular, the order at which the blocks are visited, and the number of times each block is updated before switching to a different block, is an important topic.

### BCD vs AO vs approximate AO

Let us discuss the concrete example of NMF with Frobenius loss, see [](./lra.md) and [](./nnls.md) for details on the NMF model and classical solvers. The solver we consider is [HALS](./nnls.md#hals-nnls-only). Denoting $W$ and $H$ the two factors of the NMF $Y\approx WH^T$, HALS can be seen as a BCD algorithm where the blocks are the columns of matrices $W$ and $H$. Each block update is computed in closed form. 

The HALS updates for the columns of matrix $H$ involve computing the quantities $W^TW$ and $W^TY$. In fact these operations are, for high-dimensional settings, the computational bottleneck of HALS. Because these quantities are independent of matrix $H$, it is more efficient in terms of computation cost to update the columns of matrix $H$ several times, sequentially, before switching to matrix $W$. The number of times $n_{inner}$ the columns of matrix $H$ (and similarly for $W$) are updated is thus an hyperparameter of HALS. The outer iterations are incremented once both matrices have been updated.

However if the number of inner iterations $n_{inner}$ is set to a large number, the HALS algorithm is essentially an AO algorithm, solving alternatively the nonnegative least squares problem for matrices $W$ and $H$. One may wonder if setting the number of inner iterations $n_{iter}$ to a low number leads to a sharper decrease of the cost **per outer iteration**, meaning that BCD has an advantage over AO in terms of convergence speed, and if this translates into a computation time advantage as well. Let us test the performance of HALS with various inner iterations $n_{iter}$ numerically in a noiseless setting. The HALS implementation is taken from tensorly.

```{code-cell} ipython3
import tensorly as tl
import numpy as np
from tensorly.solvers import hals_nnls
from time import perf_counter

def BCD_hals(Y,Winit,Hinit,n_inner,n_outer):
    W = np.copy(Winit)
    H = np.copy(Hinit)
    start = perf_counter()
    loss = []
    time = []
    
    for it in range(n_outer):
        # Update H
        UtM = W.T @ Y
        UtU = W.T @ W
        H = hals_nnls(UtM, UtU, V=H.T, n_iter_max=n_inner, tol=0).T

        # Update W
        VtM = H.T @ Y.T
        VtV = H.T @ H
        W = hals_nnls(VtM, VtV, V=W.T, n_iter_max=n_inner, tol=0).T

        loss.append(np.linalg.norm(Y - W@H.T,'fro')**2)
        time.append(perf_counter() - start)
    return W, H, loss, time
    

# Generate toy synthetic nonnegative tensor data
np.random.seed(0)
shape = [2000, 50]
rank = 3
Wtrue, Htrue = [np.random.rand(s, rank) for s in shape]
Y = Wtrue @ Htrue.T
Winit = np.random.rand(shape[0], rank)
Hinit = np.random.rand(shape[1], rank)
results = {}

for n_inner in [1, 2, 3, 5, 10, 20, 100]:
    West, Hest, loss, time_ = BCD_hals(Y, Winit=Winit, Hinit=Hinit, n_inner=n_inner, n_outer=300)
    results[n_inner] = (loss, time_)

```

```{code-cell} ipython3
:tags: [hide-input]

# Iteration plot
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
for n_inner, (loss, time_) in results.items():
    plt.semilogy(loss, label=f"n_inner={n_inner}")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("BCD-HALS Convergence for different n_inner")
plt.legend()
plt.grid()
plt.show()

# Time plot
plt.figure(figsize=(6,6))
for n_inner, (loss, time_) in results.items():
    plt.loglog(time_, loss, label=f"n_inner={n_inner}")
plt.xlabel("Time (s)")
plt.ylabel("Loss")
plt.title("BCD-HALS Convergence for different n_inner")
plt.legend()
plt.grid()
plt.show()
```

We may observe that
- per iteration, setting a large number of inner iterations leads to lower reconstruction error;
- with respect to time however, with the data matrix of size $2000$ by $50$, using one to five inner iterations leads to faster runtime to reach a given precision.
In this exemple, therefore, AO is a better strategy than BCD, but its implementation is costly (solving the NNLS problems exactly requires many inner iterations of HALS), and therefore BCD should be used instead.

Note that this simulation is seeded. Running this experiment several times with different dataset and initializations leads to different conclusions regarding the optimal number of inner iterations, both with respect to iterations and time. The dimensions of the input matrix is also an important factor, and for larger dataset (which make this notebook slow to run and the whole manuscript therefore too slow to compile), setting the number of inner iterations higher can be beneficial. It can also be beneficial to run a different number of inner iterations for each matrix when their dimensions are vastly different, see for instance {cite}`Gillis2012Accelerated`.

### About continuous differentiability
In this manuscript, we are mostly concerned with the squared Frobenius norm to measure discrepencies between the LRA model and the dataset. However we also sometimes use $\beta$-divergences, and KL-divergence in particular. It is then important to note that KL-divergence is not differentiable at zero, and that therefore some of the results introduced above do not apply for the convergence of AO and BCD. Theorem {prf:ref}`th:AO2`, on the contrary, does apply to KL-divergence loss since differentiability is only requires in the interior of the domain of the smooth part of the cost.


### About compact sets for the block variables and LRA
One may note that many of the AO and BCD convergence results rely on assumption (Axx) that the sets $\mathcal{X}_k$ are closed and compact. In the context of LRA, because of scaling ambiguity, the block variables which are typically the factors of the LRA model do not belong to compact sets (these sets are not bounded). This problem is adressed in the book of Gillis on NMF p267, his solution applies to most LRA models {cite}`gillisNonnegativeMatrixFactorization2020`. In a nutshell, one may simply constraint the norms of the blocks with a large enough constant that depends on the data and the initialization. The trick also applies to guarantee that the level set $\{x| f(x)\leq f(x^{(0)}) \}$ is compact as long as the cost is coercive.

%## notes

%- block-lipschitz does not imply global lipschitz in general, only true with cvx functions.
%- BCD (type BSUM) fails in constrained problems !!!!! e.g. with linear constraints (comme j'avais eu avec Bora). Une ref: M. V. Solodov, “On the convergence of constrained parallel variable distribution algorithms,”  SIAM J. on Optimization, vol. 8, no. 1, pp. 187–196, 1998. Une sol: BSUM-M (fusion de ADMM et BSUM) par Luo et al, 2014.
%- AO can fail to converge, there is a famous Powell counterexample with $f(x,y,z) = -(xy+yz+xz)+ (|x|-1)_+^2 + (|y|-1)_+^2 + (|z|-1)_+^2$ given in Wright review in 2015, also detailled in Beck book. He gives a nice figure. It shows AO alone in nncvx can fail.
%- AO can also fail because not all coordinate-wise minima are stationary points. Counterexamples in Beck book, even with convex cost. This can be avoided in two-blocks, or with other assumptions as in Grippo Scriandrone.
  %- Analysis of Powell example by Grippo Scriandrone: "Nonconvergence is due to the fact that the limit points associated to consecutive partial updates are distinct because of the fact that the function is not componentwise strictly quasiconvex; on the other hand, as the function is not pseudoconvex, the limit points of the sequence {x k } are not critical points."