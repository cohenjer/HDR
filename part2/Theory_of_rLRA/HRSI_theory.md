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
(sec:HRSItheory)=
# Implicit regularization in rLRA

:::{admonition} Reference
:class: tip
{cite:p}`cohenEfficientAlgorithmsRegularized2025` J. E. Cohen, V. Leplat, "Efficient Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models", SIAM Journal of Mathematics on Data Science, 2025 [[arxiv]](https://arxiv.org/abs/2403.18517)
:::

Consider the ridge NMF problem with Frobenius loss:

(eq:ridge-NMF)=
$$ \argmin{X_1\geq 0, X_2\geq 0} \|M - X_1X_2^T\|_F^2 + \lambda_1 \|X_1\|_F^2 + \lambda_2 \|X_2\|_F^2. $$
with $M\in\mathbb{R}_{+}^{m_1\times m_2}$ and the nonnegative rank is $r$.

Such a ridge-regularized model could be preferred by a user on the basis of ridge-regression: avoiding overfitting (in the case of NMF, this would translate as choosing a particular solution with low variance) or improving the numerical stability of the optimization algorithm (the optimization problem is strictly bi-convex when $\lambda_i>0$). 

It turns out that, maybe surprisingly at first, the ridge-regularized NMF problem is equivalent (in the sense that the minimizers are related by a trivial mapping) to a variant of the nuclear-norm low-rank sensing problem with nonnegativity constraints,

$$ \argmin{X_1\geq 0, X_2\geq 0} \|M -  X_1X_2^T\|_F^2 + \frac{\sqrt{\lambda_1\lambda_2}}{2} \sum_{q=1}^{r} \|X_1[:,q]\otimes X_2[:,q]\|_F $$

This second formulation is insightful in many ways.
  - The regularization in the second, implicit formulation, takes the form of a group-sparse regularization of the rank-one terms in NMF. Without nonnegativity constraints, this constraint is exactly the nuclear norm of the low-rank matrix $X_1X_2^T$. The ridge penalty on each factor matrix, therefore, implies rank-minimization implicitly.
  - Tuning each parameter $\lambda_1$ and $\lambda_2$ individually yields counterintuitive results. For instance, when $\lambda_1$ is much larger than $\lambda_2$, the regularization on the first factor $X_1$ is not stronger than on the second factor $X_2$. Only the product of the two regularization hyperparameters governs the regularization intensity.

In the work conducted with Valentin Leplat {cite:p}`cohenEfficientAlgorithmsRegularized2025`, we generalize this observation for a large class of regularized LRA models, namely Homogeneous Regularized Scale-Invariant models (HRSI), that account for many LRA models regularized with homogeneous positive-definite regularizations such as any $\ell_p$ norm. The implicit rank-selection effect observed for ridge regularization is shown to be due to the scale-ambiguity of LRA models, and therefore pertains to most $\ell_p$ norms. We provide the [main theoretical result](sec:theory) below, and provide more examples, namely $\ell_1$-$\ell_1$ and $\ell_1$-$\ell_2$ regularizations.

We present numerical simulations in the next section, along with algorithm-oriented concerns, such as leveraging scale-invariance to optimally balance the factors to minimize regularization, which provably allows one to escape the so-called "scaling swamp" {cite:p}`Papalexakis2013From`.

(sec:theory)=
## Scale invariance main result

While the scale of the factor matrices leaves the low-rank approximation intact, namely $X_1X_2^T$ and $X_1\Lambda \Lambda^{-1}X_2^T$ are equivalent low-rank representations for any nonzero diagonal matrix $\Lambda$, implicit regularization in LRA essentially occurs because the regularizations are not invariant with respect to such a balancing of columns and rows of the factor matrices. Solving for the optimal scales in rLRA modifies the penalization. Further, the optimal scales are determined by the homogeneity degree of the regularizations and the value of the regularization hyperparameters, and essentially balance the rank-one components.

In what follows, we first introduce the general framework of HRSI, and then derive the main result that relates scale-invariance with implicit regularization.

### Homogeneous Regularized Scale-Invariant models
Consider the following optimization problem

```{math}
:label: eq:hrsi-pb
 \argmin{\forall i\leq n,\; X_i\in\mathbb{R}^{m_i\times r}}f(\{X_i\}_{i\leq n}) + \sum_{i=1}^{n} \mu_{i} \sum_{q=1}^{r} g_i(X_i[:,q]), 
``` 

```{margin}
Coercivity is satisfied by the regularization functions considered in this section.
```

where $f$ is a continuous map from the Cartesian product $\times_{i=1}^{n} \mathbb{R}^{m_i\times r} \cap \text{dom}(g_i)$ to $\mathbb{R}_+$, $\{\mu_i\}_{i\leq n}$ is a set of positive regularization hyperparameters, and $\{g_i\}_{i\leq n}$ is a set of lower semicontinous regularization maps from $\mathbb{R}^{m_i}$ to $\mathbb{R}_+$. We assume that the total cost is coercive, ensuring the existence of a minimizer. Furthermore, we assume the following assumptions hold:
- **A1**: The function $f$ is invariant to columnwise scaling of the parameter matrices. For any sequence of positive diagonal matrices $\{\Lambda_i\}_{i\leq n}$,
  
    $$  f(\{X_i\Lambda_i\}_{i\leq n}) = f(\{X_i\}_{i\leq n})\;\;\text{if}\; \prod_{i=1}^{n}\Lambda_i = I_r,
    \; \Lambda_i>0. $$

    In other words, scaling any two columns $X_{i_1}[:,q]$ and $X_{i_2}[:,q]$ inversely proportionally has no effect on the value of $f$. 
- **A2**: Each regularization function $g_i$ is positive and homogeneous of degree $p_i$, that is for all $i\leq n$ there exists $p_i>0$ such that for any $\lambda > 0$ and any $x\in \mathbb{R}^{m_i}$,

    $$        g_i(\lambda x) = \lambda^{p_i} g_i(x). $$

    This property holds in particular for any $\ell_p^p$ norm.
```{margin}
Assumption **A3** limits the framework's applicability by excluding positive homogeneous regularizations such as Total Variation.
```
- **A3**: Each function $g_i$ is positive-definite, meaning that for any $i\leq n$, $g_i(x)=0$ if and only if $x$ is null.
 
We refer to the optimization problem {eq}`eq:hrsi-pb` with assumptions **A1**, **A2**, and **A3**, the Homogeneous Regularized Scale-Invariant problem (HRSI). 

HRSI is quite general and encompasses many instances of regularized LRA problems. Specifically, [ridge NMF](eq:ridge-NMF) is an HRSI problem with $f(\{X_i\}_{i\leq n})= \|M - X_1X_2^T\|_F^2$, $g_1(x) = \|x\|^2_2 + \eta_{\mathbb{R}_+^{m_1}}(x)$ and $g_2(x) = \|x\|^2_2 + \eta_{\mathbb{R}_+^{m_2}}(x)$.

(sec:theorem-HRSI)=
### Solutions to HRSI also solve an implicit regularized problem, and are balanced

We proved the following result in {cite:p}`cohenEfficientAlgorithmsRegularized2025`, relying on an identity for the geometric mean.
````{prf:theorem} HRSI solutions characterization and implicit equivalent formulation
:label: th:HRSI
If $\mu_i>0$ for all $i$, any solution $\{X_i^\ast\}_{i\leq n}$ to the HRSI problem {eq}`eq:hrsi-pb` under assumptions **A1**, **A2** and **A3** satisfies 

$$p_i\mu_ig_i(X_i^{\ast}[:,q]) = \beta_q$$

for all $i\leq n$ and $q\leq r$, where 

$$ \beta_q = \left(\prod_{i\leq n} \left(p_i\mu_ig_i(X^\ast_i[:,q])\right)^{\frac{1}{p_i}}\right)^{\frac{1}{\sum_{i\leq n} \frac{1}{p_i}}} . $$

Moreover, the solutions to the HRSI problem are, up to scaling, solutions to the implicit HRSI problem

$$   \argmin{\forall i\leq n,\; X_i\in\mathbb{R}^{m_i\times r}}f(\{X_i\}_{i\leq n}) + \tilde{\mu} \sum_{q=1}^{r}  \left(\prod_{i=1}^{n} g_i(X_{i}[:,q])^{\frac{1}{p_i}}\right)^{\frac{1}{\sum_{i=1}^{n} \frac{1}{p_i}}}, $$
    
where $\tilde{\mu} = \left(\prod_{i=1}^{n}(p_i\mu_i)^{\frac{1}{p_i}}
    \right)^{\frac{1}{\sum_{i=1}^{n} \frac{1}{p_i}}}\left(\sum_{i=1}^{n}\frac{1}{p_i}\right).$ 
````

From this theorem, we observe that
- Only the product of regularization hyperparameters impacts the regularity of the solution.
- The sum of columnwise regularization on the factor matrices in the explicit formulation becomes a product of regularizations in the implicit formulation, yielding group-sparsity on the rank-one components.
- At optimality, the scales of the components must satisfy the balancing condition $p_i\mu_ig_i(X_i^{\ast}[:,q])=\beta_q$. This means, in particular, that LRA models regularized with homogeneous functions do not suffer from scale ambiguity.


## Special cases with $\ell_1$ and $\ell_2$ regularizations

To make our main result {prf:ref}`th:HRSI` more explicit, let us instantiate the implicit HRSI formulation and the optimal scales for a couple of classical problems in rLRA. Based on the observation that only the product of regularization hyperparameters matters, the same regularization hyperparameter is used for all factor matrices.

### Ridge-regularized nonnegative CPD

Define the ridge nCPD formally as the optimization problem

$$ \argmin{X_i\geq 0}  \| \mathcal{T} - \mathcal{I}_r \times_1 X_1 \times_2 X_2 \times_3 X_3 \|^2_F +  \mu \left(\|X_1\|_F^2 + \|X_2\|_F^2 + \|X_3\|_F^2\right). $$

The implicit HRSI problem shows that ridge-regularization yields a componentwise group-sparsity implicit regularization,

$$ \argmin{\{\mathcal{L}_q\}_{q\leq r},~\text{rank}(\mathcal{L}_q)\leq 1}  \|\mathcal{T} -\sum_{q=1}^{r} \mathcal{L}_q \|_F^2 + 3\mu \sum_{q=1}^{r} \|\mathcal{L}_q\|_F^{\frac{2}{3}} $$

where tensors $\mathcal{L}_q$ are rank-one tensors $X_1[:,q]\otimes X_2[:,q] \otimes X_3[:,q]$. We experimentally validate in [the experiments subsection](#balanced-and-non-euclidean-algorithms-for-regularized-lra) that tuning the regularization hyperparameter $\mu$ indeed helps us to select the rank of the nCPD factorization automatically. Moreover, the optimal solution of ridge nCPD verifies the balancing equation

$$ 1 = \sqrt{2} \|X^\ast_i[:,q]\|_2^{-\frac{1}{3}}\prod_{j\neq i}\|X^\ast_j[:,q]\|_2^{\frac{2}{3}}  .$$

This balancing identity will be used in [the experiments subsection](#balanced-and-non-euclidean-algorithms-for-regularized-lra) to refine the iterations of an iterative optimization algorithm to solve ridge nCPD.

### Sparse NMF L1-L1

Another interesting case of an implicit regularization effect concerns the (double) sparse NMF model discussed in {cite:p}`Papalexakis2013From`,

$$\argmin{X_1\geq 0, X_2\geq 0} \|M - X_1X_2^T\|_F^2 + \mu \left( \|X_1\|_1 + \|X_2\|_1 \right). $$

Intuitively, a practitioner using sparse NMF with $\ell_1$ penalizations on both factors would expect sparse entries in both matrices, leveraging the usual behavior of the $\ell_1$ norm in regression problems. The implicit formulation 

$$ \argmin{L_q\in\mathbb{R}_+^{m_1\times m_2},\; \text{rank}(L_q)\leq 1} \|M - \sum_{q=1}^{r}L_q\|_F^2 + 2\sqrt{\mu_1\mu_2}\sum_{q=1}^{r} \sqrt{\|L_q\|_1}$$

shows that, while both factors are indeed penalized to be sparse, there is another group-sparse effect that prunes entire components. The problem with this formulation is that both effects (elementwise sparsity and componentwise sparsity) are not controlled individually but rather by the same regularization hyperparameter. This yields unexpected component pruning as shown in the simulation below with mixtures of Gaussians. It is also unclear from the implicit formulation whether both factors will be sparse elementwise.



```{code-cell} ipython3
import numpy as np
import tensorly as tl
import matplotlib
import matplotlib.pyplot as plt

# Generating toy image
n1 = 30
n2 = 40
r = 3
sig = 0.2  # 0.15 # 0.05

# Components will be Gaussian distributions, here is the macro to generate them
def gauss(x, m, sig, thresh=0.1):
    # max is 1
    out = np.exp(-(x-m)**2/sig**2)
    out[out < thresh] = 0
    return out

# Generating the data as a mixture of separable Gaussians
x = np.linspace(0, n1-1, n1) # 1D abcisse
y = np.linspace(0, n2-1, n2)
W = np.zeros([n1, r])
W[:,0] = gauss(x, 5, 3)
W[:,1] = gauss(x, 15, 5)
W[:,2] = gauss(x, 25, 10)
H = np.zeros([n2, r])
H[:, 0] = gauss(y, 10, 5)
H[:, 1] = gauss(y, 20, 6)
H[:, 2] = gauss(y, 25, 5)
trueNMF = (None, [W, H])

Y = W@H.T  # NMF model
rng = np.random.default_rng(1246)
Yn = Y + sig*rng.random((n1, n2))  # corruption with gaussian noise

# initialization 
W0 = 0*W + 0.5*rng.random(W.shape)
H0 = 0*H + 0.5*rng.random(H.shape)

# Storing output factors for various regularization in dictionaries
We = dict()
He = dict()

# Chose the grid values for the hyperparameter 
lambset = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.75, 0.78, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5, 5.2, 6, 20]

import copy
for lamb in lambset:
    # Computing the sparse NMF with tensorly (the algorithm is HALS, which is the state of the art for this problem)
    out = tl.decomposition.non_negative_parafac_hals(Yn, r, sparsity_coefficients=[lamb, lamb], init=copy.deepcopy((None, [W0, H0])))
    # Estimated factors
    We[lamb] = out[1][0]
    He[lamb] = out[1][1]
    # Set estimates as new initialization for the next iteration (warm start)
    W0 = We[lamb]
    H0 = He[lamb]
    
    # rescale We so that when a column of W is 0, so is the corresponding column of H
    norms = tl.max(We[lamb], axis=0)
    for q in range(r):
        if norms[q] > 1e-8:
            We[lamb][:, q] = We[lamb][:, q]/norms[q]
            He[lamb][:, q] = He[lamb][:, q]*norms[q]
        else:
            We[lamb][:, q] = We[lamb][:, q]*0
            He[lamb][:, q] = He[lamb][:, q]*0
```

```{code-cell} ipython3
:tags: [hide-cell]
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, Slider, CustomJS, CustomJSTickFormatter
from bokeh.layouts import column
output_notebook()
```

```{code-cell} ipython3
:tags: [hide-input]


font = {'size'   : 16}
matplotlib.rc('font', **font)

# Initial data for lambda = 0
initial_idx = 0
initial_lamb = lambset[initial_idx]
sources = [ColumnDataSource(data=dict(x=x, y=We[initial_lamb][:, i])) for i in range(r)]

# Figure setup
p = figure(title=f"Regularization = {initial_lamb}", width=700, height=400,
           x_axis_label="X", y_axis_label="We[:,q]")

# --- Set up Bokeh plot ---
component_names = [f"component {i+1}" for i in range(r)]
colors = ["blue", "green", "red"]
lines = []
for i in range(r):
    line = p.line('x', 'y', source=sources[i], line_width=2,
                  line_color=colors[i % len(colors)],
                  legend_label=component_names[i])
    lines.append(line)

p.legend.title = "Components"
p.legend.location = "top_right"


# Flatten the data to send to JavaScript
We_js = [[[float(We[lam][i, j]) for i in range(len(x))] for j in range(r)]
                for lam in lambset]


# --- Slider (index-based) and label display ---
slider = Slider(start=0, end=len(lambset)-1, step=1, value=initial_idx, title="Regularization")

# Format ticks and slider value display using CustomJSTickFormatter
slider.format = CustomJSTickFormatter(code=f"""
    const labels = {lambset};
    return labels[tick];
""")


callback = CustomJS(args=dict(sources=sources,
                              all_data=We_js,
                              title=p.title,
                              lambset=lambset),
                              #label=lambda_label),
    code="""
    const idx = cb_obj.value;
    const lam_value = lambset[idx];
    for (let i = 0; i < sources.length; i++) {
        sources[i].data['y'] = all_data[idx][i];
        sources[i].change.emit();
    }
    title.text = "Regularization = " + lam_value;
""")

slider.js_on_change('value', callback)

# --- Display ---
show(column(p, slider))
```

Observe in particular that the sparsity level of the components does not evolve smoothly with the regularization hyperparameter. Components vanish brutally at $\mu=0.75$ and $\mu=5.2$.


## The sparse NMF using $\ell_1$-$\ell_2$ regularization

The explicit $\ell_1$-$\ell_2$ sparse NMF model writes

$$\argmin{X_1\geq 0, X_2\geq 0} \|M - X_1X_2^T\|_F^2 + \mu \left( \|X_1\|_1 + \|X_2\|^2_2 \right). $$

The implicit HRSI formulation of sparse NMF can be easily derived:

$$ \argmin{L_q\in\mathbb{R}_+^{m_1\times m_2},\; \text{rank}(L_q)\leq 1} \|M - \sum_{q=1}^{r}L_q\|_F^2 + \frac{3}{2^\frac{2}{3}}\mu\sum_{q=1}^{r} \|L_q\|^{\frac{2}{3}}_{1,2}.$$

where the mixed matrix norm is defined as $\|L_q\|_{1,2} = \sqrt{\sum_{j} \left( \sum_i L_q[i,j] \right)^2} $ and we used the identity 

$$ \|x\otimes y\|_{1,2} = \|x\|_1\|y\|_2 $$ 

that holds for any two vectors $x$ and $y$. One may observe that the implicit regularization acts as a group-LASSO norm  on the rows of the rank-one components, effectively enforcing sparsity elementwise in the factor matrix $X_1$. The group-LASSO effect on the rank-one components still appears, however, in the implicit regularization, and similarly to the $\ell_1$-$\ell_1$ case, the two effects may occur simultaneously.

It is still open to fully characterize the link between the regularized formulation and the constrained formulation

$$\argmin{X_1\geq 0, X_2\geq 0} \|M - X_1X_2^T\|_F^2 + \mu \|X_1\|_1  \text{ s.t. } \|X_2[:,q]\|_2 = 1 \; \forall q\leq r. $$

Interestingly, Marmin, Goulard, and Févotte show equivalence between the constrained formulation and the implicit HRSI formulation of sparse NMF, hinting towards the equivalence between the three formulations {cite:p}`marminMajorizationMinimizationSparseNonnegative2023`. 


## Balanced and non-Euclidean algorithms for regularized LRA

In our work with Valentin Leplat, in addition to the theoretical analysis of HRSI discussed in [](./HRSI_theory.md), we tackle a more practical problem. We observe, both numerically and theoretically, the existence of scaling swamps that considerably slow the convergence of alternating algorithms. Can the balancing equation $p_i\mu_ig_i(X_i^{\ast}[:,q]) = \beta_q$ be leveraged to escape these swamps?

In what follows, we first show the existence of scaling swamps in a toy problem numerically ({cite:p}`cohenEfficientAlgorithmsRegularized2025` formally proves the sublinear convergence rate of ALS for this problem). A simulation then shows the rank-selection capabilities of ridge nCPD, along with the importance of balancing.

### Scaling swamps

In the tensor decomposition literature, swamps refer to many consecutive iterations of optimization algorithms in which the cost function remains nearly constant while the parameters change significantly. This phenomenon occurs for many algorithms and is conjectured to be due to the ill-posedness of tensor LRA {cite:p}`hillar:2013:tensor.problems.nphard`, {cite:p}`Comon2009Tensor`, {cite:p}`mohlenkampDynamicsSwampsCanonical2019`, {cite:p}`vermeylenReducingSwampBehavior2025a`.

Another kind of swamp was observed by Papalexakis and Sidiropoulos {cite:p}`Papalexakis2016Tensors` for $\ell_1$-$\ell_1$ sparse NMF. It is called a scaling-swamp because, as in traditional swamps, the cost improves very slowly over many iterations. Unlike in swamps from the tensor decomposition literature, however, the cost does not decrease any faster after a certain number of iterations, and scaling swamps also occur in matrix factorization problems such as NMF, even though the low-rank manifold is closed.

We can construct a simple numerical example in one dimension in which the ALS algorithm is provably slow. Consider the loss function

$$ f(x_1,x_2) = (y - x_1x_2)^2 + \lambda (x_1^2 + x_2^2). $$ (eq:toy-pb)

where $x_1$ and $x_2$ are real values and the regularization hyperparameter $\lambda$ and the data $y$ are positive real values.

Using a perturbation analysis, it can be observed that the ALS algorithm, whose updates at iteration $k+1$ are given by

$$
    x_1^{(k+1)} = \frac{x_2^{(k)}y}{{x^2_2}^{(k)}+\lambda} \; \; \text{and} \; \; 
        x_2^{(k+1)} = \frac{x_1^{(k+1)}y}{{x^2_1}^{(k+1)}+\lambda} .
$$

has a convergence rate for the iterates asymptotically close to $1-4\lambda/y$, which is arbitrarily close to one (sublinear convergence) for small regularizations. The following numerical simulation shows that scaling swamps are more pronounced when regularization is small relative to the data magnitude.

```{code-cell}ipython3
:tags: []

from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt

# A barebone implementation of the ALS (Alternating Least Squares) algorithm
def ALS(y,x1,x2, lamb, itermax=200):
    loss = [(y - x1*x2)**2 + lamb * (x1**2 + x2**2)]
    x1_store = [x1]
    x2_store = [x2]
    for k in range(itermax):
        x1 = x2*y/(x2**2+lamb)
        x2 = x1*y/(x1**2+lamb)
        loss.append((y - x1*x2)**2 + lamb * (x1**2 + x2**2))
        x1_store.append(x1)
        x2_store.append(x2)
    return x1, x2, loss, x1_store, x2_store

# Example usage, you can play with the values of y and lamb !
y = 1
lamb = 0.001

# Initialization
x1 = 0.2
x2 = 5

# Prints and Plots
x1, x2, loss, x1store, x2store = ALS(y, x1, x2, lamb)
print("After 200 iterations:")
print(f"x1: {x1}, x2: {x2}, Optimal values: {np.sqrt(y-lamb)}")
print(f"Loss: {loss[-1]}, Optimal loss: {(y-np.sqrt(y-lamb))**2+ lamb * 2 * (y-lamb)}")

```

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x1store, label='x1')
axes[0].plot(x2store, label='x2')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Values')
axes[0].set_title('Values of x1 and x2 over iterations')
axes[0].legend()
axes[1].semilogy(loss, label='Loss')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss over iterations')   
plt.show()

```

Interestingly, the toy problem {eq}`eq:toy-pb` has a simple closed form solution that can be obtained by optimally balancing the two scalar values $x_1$ and $x_2$ as defined in {prf:ref}`th:HRSI`, namely

$$
|x_1| = |x_2|.
$$

Then the optimal solution is simply (up to sign ambiguities) $x_1=x_2=\sqrt{y-\lambda}$.

This suggests that balancing the estimates optimally before, after, or within an optimization algorithm solving HRSI could help avoid the scaling swamp phenomenon. We explore here only the balancing of initialization and/or outputs of an algorithm for simplicity.

Balancing initial or final estimates of an algorithm is a straightforward operation using {prf:ref}`th:HRSI`. First, compute the columnwise geometric mean

$$ \beta_q = \left(\prod_{i\leq n} \left(p_i\mu_ig_i(X^\ast_i[:,q])\right)^{\frac{1}{p_i}}\right)^{\frac{1}{\sum_{i\leq n} \frac{1}{p_i}}} . $$

Then scale all columns such that their regularization is exactly $\frac{1}{p_i\mu_i}\beta_q$. For the problem of ridge-regularized nCPD, this amounts to performing for all $i$ in $\{1,2,3\}$ the balancing procedure

$$ X_i[:,q] \leftarrow \left( \prod_{j\leq 3}\|X_j[:,q]\|_2^{\frac{1}{3}} \right) \frac{X_i[:,q]}{\|X_i[:,q]\|_2}. $$ 

An example barebone implementation is shown below.

```{code-cell} ipython3
:tags: []

import matplotlib.pyplot as plt 
import tensorly as tl
from copy import deepcopy
from tensorly.solvers.penalizations import scale_factors_fro

def optimal_balancing(factors):
    rank = factors[0].shape[1]
    beta = []
    for q in range(rank):
        beta.append(tl.prod([tl.norm(factor[:,q]) ** (1 / 3) for factor in factors]))
    balanced_factors = []
    for i in range(len(factors)):
        balanced_factor = factors[i].copy()
        for q in range(rank):
            balanced_factor[:, q] =  factors[i][:, q] * beta[q] / tl.norm(factors[i][:, q])
        balanced_factors.append(balanced_factor)
    return balanced_factors
```

The simulation below illustrates the importance of balancing initial (and sometimes final) estimates. In {cite:p}`cohenEfficientAlgorithmsRegularized2025`, we also show that balancing at every outer iteration in the HALS algorithm further improves the convergence speed empirically.
%, but this simulation only performs balancing at the first and last iteration.

Recall the ridge-regularized nCPD problem

$$
 \argmin{X_i\geq 0}  \| \mathcal{T} - \mathcal{I}_r \times_1 X_1 \times_2 X_2 \times_3 X_3 \|^2_F + \mu \left(\|X_1\|_F^2 + \|X_2\|_F^2 + \|X_3\|_F^2\right). 
$$

The reader can play around with the regularization value: high values lead to more component pruning, but the effect of balancing is less pronounced; low values lead to no component pruning, but the importance of balancing is more visible, even after the optimization algorithm has converged.

Also note that balancing only scales factors proportionally to one another. To fully scale the factors, accounting for the data fitting term and therefore starting the algorithm at the best scaled position, it can be useful to scale the initial guess by solving the polynomial minimization problem 

$$
    \argmin{\lambda\geq 0} \| \mathcal{T} - \lambda^3 \mathcal{I}_r \times_1 X_1 \times_2 X_2 \times_3 X_3 \|^2_F +  \mu\lambda^2 \left(\|X_1\|_F^2 + \|X_2\|_F^2 + \|X_3\|_F^2\right),
$$

which amounts to evaluating the cost at all the positive roots of the polynomial

$$
    P(\lambda) = - 6\lambda^2 \left\langle \mathcal{T}, \mathcal{I}_r \times_1 X_1 \times_2 X_2 \times_3 X_3  \right\rangle + 6\lambda^5 \|\mathcal{I}_r \times_1 X_1 \times_2 X_2 \times_3 X_3  \|_F^2 \\ + 2\mu\lambda \left(\|X_1\|_F^2 + \|X_2\|_F^2 + \|X_3\|_F^2\right).
$$

```{code-cell} ipython3
:tags: []

# Example usage of the optimal_balancing function
rank = 3
dims = [10, 11, 12]
ndims = len(dims)
noise = 0.1
itermax = 100

# Regularization hyperparameter
ridge_reg = 0.1

# Create random factors for a 3-way tensor decomposition
np.random.seed(22)  # For reproducibility   
true_factors = [tl.tensor(np.random.rand(dims[i], rank)) for i in range(ndims)]

# Creating the data (from factors or balanced factors, it leads to the same result)
CPtensor = tl.cp_tensor.CPTensor(([10**(-i) for i in range(ndims)], true_factors)) 
data = CPtensor.to_tensor() + noise * tl.tensor(np.random.randn(*dims))  # Adding some noise

# Initialization
rank_e = rank + 3  # overestimating rank for initialization
init = [10**(-i) * tl.tensor(np.random.rand(dims[i], rank_e)) for i in range(ndims)]
CPinit = tl.cp_tensor.CPTensor((None, init)) 

# Optimal scaling of the initialization
t_init_unscaled = CPinit.to_tensor()
CPinit_scaled, scale = scale_factors_fro(CPinit, data, [ridge_reg]*ndims, [0]*ndims, nonnegative=True)
init_scaled = CPinit_scaled.factors

# Balancing the unscaled initialization factors
balanced_init = optimal_balancing(init)
CPinit_balanced = tl.cp_tensor.CPTensor((None, balanced_init)) 

# Balancing the scaled initialization factors
balanced_scaled_init = optimal_balancing(init_scaled)
CPinit_scaled_balanced = tl.cp_tensor.CPTensor((None, balanced_scaled_init)) 
```

```{code-cell} ipython3
:tags: [hide-input]

# Check that factor norms are balanced after balancing
print("Optimal scaling factors:", scale)
print("Initial factors:")
for i, factor in enumerate(CPinit.factors):
    print(f"Factor {i}: shape {factor.shape}, norm {tl.norm(factor)}")
# same for scaled factors
print("\nScaled factors:")
for i, factor in enumerate(CPinit_scaled.factors):
    print(f"Factor {i}: shape {factor.shape}, norm {tl.norm(factor)}")
# same for balanced factors
print("\nBalanced factors:")
for i, factor in enumerate(CPinit_balanced.factors):
    print(f"Factor {i}: shape {factor.shape}, norm {tl.norm(factor)}")
# same for balanced scaled factors
print("\nBalanced scaled factors:")
for i, factor in enumerate(CPinit_scaled_balanced.factors):
    print(f"Factor {i}: shape {factor.shape}, norm {tl.norm(factor)}")
```

```{code-cell} ipython3
:tags: []

# Fetching the loss values with callback
callback_loss = []
def callback(factors, unnormalized_rec_errors):
    loss = (unnormalized_rec_errors**2)/2 + sum([ridge_reg*tl.norm(factors[1][i])**2 for i in range(3)])
    callback_loss.append(loss)
    
# Running nonnegative CP decomposition with various balancing/scaling setups
from tensorly.decomposition import non_negative_parafac_hals

CPe = non_negative_parafac_hals(data, rank_e, n_iter_max=itermax, tol=0, init=deepcopy(CPinit), verbose=False, ridge_coefficients=ridge_reg, callback=callback)
loss = np.copy(callback_loss)

callback_loss = []
CPe_scaled = non_negative_parafac_hals(data, rank_e, n_iter_max=itermax, tol=0, init=deepcopy(CPinit_scaled), verbose=False, ridge_coefficients=ridge_reg, callback=callback)
loss_scaled = np.copy(callback_loss)

callback_loss = []
CPe_balanced = non_negative_parafac_hals(data, rank_e, n_iter_max=itermax, tol=0, init=deepcopy(CPinit_balanced), verbose=False, ridge_coefficients=ridge_reg, callback=callback)
loss_balanced = np.copy(callback_loss)

callback_loss = []
CPe_scaled_balanced = non_negative_parafac_hals(data, rank_e, n_iter_max=itermax, tol=0, init=deepcopy(CPinit_scaled_balanced), verbose=False, ridge_coefficients=ridge_reg, callback=callback)
loss_scaled_balanced = callback_loss
```

```{code-cell} ipython3
:tags: [hide-input]

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.semilogy(loss, '-.', label='Loss without balancing')
plt.semilogy(loss_scaled, '--', label='Loss with scaling')
plt.semilogy(loss_balanced, label='Loss with balancing')
plt.semilogy(loss_scaled_balanced, label='Loss with balancing and scaling')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Values Over Iterations')
plt.legend(['Without scaling or balancing', 'With scaling', 'With balancing', 'With scaling + balancing'])
plt.grid()
plt.show()
```

We can also look at the loss function with and without balancing of the final estimates to observe that balancing reduces the final error slightly, and the effect is more prononced for the output of algorithms that did not scale and balance the initial guess.

```{code-cell} ipython3
:tags: []

# Balancing the final estimated factors 
CPe_fbalance = tl.cp_tensor.CPTensor((None, optimal_balancing(CPe.factors)))
CPe_scaled_fbalance = tl.cp_tensor.CPTensor((None, optimal_balancing(CPe_scaled.factors)))
CPe_balanced_fbalance = tl.cp_tensor.CPTensor((None, optimal_balancing(CPe_balanced.factors)))
CPe_scaled_balanced_fbalance = tl.cp_tensor.CPTensor((None, optimal_balancing(CPe_scaled_balanced.factors)))

def loss(data, CPestim, ridge_reg):
    regs = sum([ridge_reg * tl.norm(factor) ** 2 for factor in CPestim.factors])
    return 0.5 * (tl.norm(data - CPestim.to_tensor()) ** 2) + regs

```

```{code-cell} ipython3
:tags: [hide-input]

# Printing the final loss values before and after rebalancing
print("Final Loss values before | after rebalancing:")
print("Without balancing or scaling:")
print(f"{loss(data, CPe, ridge_reg)} | {loss(data, CPe_fbalance, ridge_reg)}")
print("With scaling:")
print(f"{loss(data, CPe_scaled, ridge_reg)} | {loss(data, CPe_scaled_fbalance, ridge_reg)}")
print("With balancing:")
print(f"{loss(data, CPe_balanced, ridge_reg)} | {loss(data, CPe_balanced_fbalance, ridge_reg)}")
print("With scaling and balancing:")
print(f"{loss(data, CPe_scaled_balanced, ridge_reg)} | {loss(data, CPe_scaled_balanced_fbalance, ridge_reg)}")

```





