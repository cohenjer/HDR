(subsec:nnls)=
# Nonnegative Regressions: NNLS and NNKL

Section basée en partie sur le cours à iTWIST 2020, et sur des travaux plus récents autour de MU.

## NNLS

$ f\left(y,W[i,:]x\right)= \|y - Wx\|_2^2$ sizes m x n

(subsec:nnls-kl)=
## NNKL

Attention KL et likelihood différent à cst près, le préciser qqepart.

$ f\left(y,W[i,:]x\right) = \sum_i \KL{y[i],W[i,:]x}$

(block:scaling-NNKL)=
```{admonition} Scaling NNKL
say something about scaling with KKT ?
```

also say separable $f\left(y,z\right) = \sum_i f\left(y[i],z[i]\right)$. Convex wrt second variable

## Algorithms for NNLS and NNKL


| alg name |  LS | KL | fast ? | Convergence cost | 
| ---------| ----|----|--------|------       |
| AS | yes | no | with good initialization, yes | finite number of (exponentially many) steps |
| HALS | yes | no | yes | yes |
| MU | yes | yes | fast for KL only | yes |

Others: PGD, NeNMF (nesterov), SN (KL hien)
more exotic: primal-dual for KL (idée Nelly)


## Active-set

La flemme d'expliquer, cf le cours doctoral

## HALS

Expliquer simplement

## Multiplicatives Updates

One of the most well known iterative solver for general nonnegative least squares problems (both quadratic and KL although it is primarily used for the latter) is the Multiplicative Updates (MU) algorithm. This history of MU is complex: is has been proposed independently under various names in the literature. The oldest reference of MU that I am aware of is probably the so-called Richardson-Lucy iteration {cite}`ref`[todo] which is specialized for convolutive models and stems from the computational imaging community. MU is also sometimes refered to as the Maximum-Likelihood-Expectation-Maximization algorithm (ML-EM) {cite}`ref`[todo] as it can be formulated as a particular case of the generic EM framework {cite}`ref`[ref thibaut], see more details in [ref] below. MU was popularized in the source-separation community as an algorithm to solve NMF by Lee and Seung in several seminal papers {cite}`ref`[todo] and later generalized by Fevotte and Idier for a wider class of loss functions.

```{margin}
It is in fact interesting to note that the Richardson-Lucy/ ML-EM algorithm has known many developments and improvements over the years in parallel to the developements proposed in the source separation/numerical optimization literature. A few interesting reads are a constrained version of Richardson-Lucy called one-step late {cite}`ref`[todo] that amounts to MM with linearized regularizations, the general formulation of EM for likelihoods in the exponential family as an mirror gradient descent which allows to derive a generic linear convergence rate [todo] and the design of TV-regularized and Plug-and-Play variants with convergence guarantees {cite}`ref`[thibaut].
```

The general equations for the MU are, for the NNLS problem,

$$ 
x^{(k+1)} = x^{(k)} \frac{W^Ty}{W^TWx^{(k)}},
$$ (eq:MU-NNLS)

and for the NNKL problem,

$$ 
x^{(k+1)} = x^{(k)} \frac{W^T\frac{y}{Wx^{(k)}}}{W1_n}.
$$ (eq:MU-NNKL)

These updates can be derived from several frameworks, as already underlined for the quadratic case in the book of Gillis on NMF {cite}`ref` [todo]. However for NNKL, it is important to notice that MU is a special case of the EM algorithm; this fact is known to experts in NMF but not to practitioners in computational imaging where EM for NNKL was derived originally. Below I therefore try to give a short explanation on how to obtain MU from these various frameworks and the implications of these derivations. 

### As a gradient ratio
A simple way to obtain MU is by noting that the multiplicative term is exactly the ratio between the negative and positive parts of the gradient. The gradients are easily computed as 

$$
\nabla_x f\left(y,Wx\right) = 
\begin{cases}
   2 W^TWx -  2W^Ty & \text{ for NNLS }  \\
   W^T\frac{y}{Wx} -  W^T1_n & \text{ for NNKL }  .
\end{cases}
$$

Note that the terms $W^TWx, W^Ty, W^T\frac{y}{Wx}$ and $W^T1_n$ are all nonnegative. Denoting $\nabla_x^{+}$ the positive terms in the gradient computation and similarly for $\nabla_x^{-}$, we may write rewrite the MU updates as

$$
x^{(k+1)} = x^{(k)}\frac{ \nabla_x^- f\left(y,x\right) }{\nabla_x^+ f\left(y,x\right)}.
$$

This interpretation of MU is, in my opinion, useful mostly to quickly remember the updates. While Gillis {cite}`ref`[todo] provides an interpretation of the gradient ratio related to the KKT conditions (namely, the ratio should get closer to one to satisfy the KKT conditions), deriving convergence results from this formulation is not straightforward. It is also unclear how this update rule will behave for regularized NNLS/NNKL problems, see [ref below TODO].

### As a preconditionned gradient descent algorithm

MU for NNLS can be cast exactly as a preconditioned descent algorithm with a diagonal preconditioner designed to upper-bound the true Hessian. The full description of this formulation is deferred to the [second part of the manuscript](../part2/Fast_algorithms_for_rLRA/mSOM.md) since it deeply connects with a contribution regarding fast algorithms for NNLS and NNKL.

### As a majorization-minimization algorithm

```{margin}
$\beta$-divergences are a family of separable divergences that contain in particular the Euclidean distance ($\beta=2$) and the (symmeterised) Kullback-Leibler divergence ($\beta=1$). The general formula, prolonged by continuity for $\beta$ in $\{0,1\}$ is given by

$$
\mathcal{D}_{\beta} = \frac{1}{\beta(\beta-1)}\left( x^\beta + (\beta-1) y^\beta - \beta x y^{\beta-1} \right).
$$

```

The MU algorithm was introduced in the source separation and machine learning communities by Lee and Seung in 1999 {cite}`Lee1999Learning` as a majorization minimization algorithm. We follow in this paragraph the derivations of Fevotte and Idier {cite}`fevotte2011algorithms` that work for the more general class of $\beta$-divergences. For simplicity we restrain here to the case of a convex loss function for $\beta\in[1,2]$. We show in the next paragraph that for the particular case of KL-divergence, the MM derivations fall in the Expectation Maximization (EM) framework.

The main idea of MM is to fix a current iterate $x^{(k)}$, build a global majorant of the cost $\xi(x,x^{(k)})\leq f\left(y,x\right)$ tight and tangent to the cost at $x^{(k)}$, and then minimize this cost.
[insert figure]
To obtain the MU algorithm, one may use the convexity inequality for the loss function, also called the Jensen inequality in this context:

$$
\sum_{i\leq m} f\left(y[i],\sum_{j\leq n}\lambda_{i,j} W[i,j]x[j] \right) \leq \sum_{i\leq m} \lambda_{i,j} f\left(y[i], W[i,j]x[j]\right)
$$

for nonnegative coefficients $\lambda_i$ that sum to one and any positive vector $x\in\mathbb{R}^{n}_+$. The trick to obtaining MU using MM is to chose the specific values 

$$
 \lambda_{i,j} = \frac{W[i,j]x^{(k)}[j]}{\sum_{j\leq n} W[i,j]x^{(k)}[j]} = \frac{W[i,j]x^{(k)}[j]}{\hat{y}[i]}.
$$

The estimated observations at the current iterate $\hat{y} = Wx^{(k)}$ have been introduced to simplify the presentation. The lambda parameters are introduced in the summand of the second argument of the loss, setting $\sum_j W[i,j] x[i,j] = \sum_j W[i,j]x[j] \frac{\lambda_{i,j}}{\lambda_{i,j}}$. Applying the convexity inequality after observing that $\frac{W[i,j]}{\lambda_{i,j}}=\frac{\hat{y}[i]}{x^{(k)}[j]}$ yields a majorant of the cost

$$
\xi(x,x^{(k)}) = \sum_{i\leq m,~j\leq n} \frac{W[i,j]x^{(k)}[j]}{\hat{y}[i]} f\left(y[i], \frac{x[j]}{x^{(k)}[j]}\hat{y}[i]\right).
$$

This majorant is separable and convex, we can find its minimizer by setting the derivative with respect to each $x[j]$ to zero. Denoting $\frac{\partial}{\partial x_2}f$ the partial derivative of function $f$ with respect to its second argument, the MM algorithm updates $x$ by solving

$$
\forall j\leq n, \quad 0 = \sum_{i\leq m} W[i,j] \frac{\partial}{\partial x_2} f\left(y[i], \frac{x[j]}{x^{(k)}[j]}\hat{y}[i]\right).
$$ (eq:stationary-point)

For the Euclidean loss and KL-divergence respectively, 

$$
 \frac{\partial}{\partial x_2} f\left(y[i], \frac{x[j]}{x^{(k)}[j]}\hat{y}[i]\right) =   \frac{x[j]}{x^{(k)}[j]}\hat{y}[i] - y[i], \\
 \frac{\partial}{\partial x_2} f\left(y[i], \frac{x[j]}{x^{(k)}[j]}\hat{y}[i]\right) = \frac{y[i]x^{(k)}[j]}{\hat{y}[i]x[j]} - 1,
$$ 

both of which, when plugged in the stationary point equation {eq}`eq:stationary-point`, boil down to the MU updates {eq}`eq:MU-NNLS` and {eq}`eq:MU-NNKL`.

The MM interpretation of MU is useful because it automatically guarantees that MU iterations always decrease the cost. Moreover, for NNLS with positive initialization, MU falls within the scope of the SUM framework described in sec[TODO], that guarantees convergence of the cost to a stationary point. However, additional hypotheses are required to guarantee that the limit point of the cost and the iterates are respectively stationary points are global minimizers of the cost function in the NNKL problem. These assumptions are summarized in {cite}`gillisNonnegativeMatrixFactorization2020`, and revolve around the fact that Lipschitz continuity of the cost is required to avoid arbitrarily small improvements of the iterates for a given cost decrease. 

```{margin}
The zero-locking phenomenon is a numerical instability problem that occurs when values in vector $x$ are numerical so close to zero that the computer stores actual zeros. Once a value in $x$ is zero, it can never increase again in MU due to the elementwise multiplications at each iterations. This can prevent convergence in practice, and cause numerical instabilities caused by division by zero.
```

(block:pos_cstr)=
```{admonition} Positivity constraints

A workaround for the NNKL problem, that also guarantees the positivity required for convergence in NNLS, is to impose positivity constraints by introducing $\epsilon>0$ such that $x\geq \epsilon$. Positivity avoids the non-Lipschitz smoothness of KL-divergence at zero, and avoid the zero-locking phenomenon {cite}`ref`[Takahashi]. The MU iterates then become

$$ 
x^{(k+1)} = \max\left( x^{(k)} \frac{W^Ty}{W^TWx^{(k)}}, \epsilon\right),
$$

for the NNLS problem, and

$$ 
x^{(k+1)} = \max\left(x^{(k)} \frac{W^T\frac{y}{Wx^{(k)}}}{W^T1_n},\epsilon\right).
$$

for the NNKL problem, where the maximum is applied elementwise. The underlying algorithm here can be thought of as a generalized proximal gradient or preconditionned forward-backward algorithm, see for instance the Variable Metric forward-backward framework for more details {cite}`ref`[Repetti Chouzenoux].

```

### As an EM algorithm
In the MM description of MU, the particular choice for the parameters $\lambda_{i,j}$, which is crucial to obtain the updates, could seem somewhat arbitrary. It turns out that using the EM framework for the NNKL problem, one can make sense of this choice. Below we derive MU from EM in a way that slightly differs earlier references from computational tomography {cite}`ref`[todo] that I find personally hard to follow. 

A statistical description of the NNKL problem is required to write the EM algorithm. We suppose that the data samples $y[i]$ are sampled independently from random variables $Y[i]$ following Poisson distributions conditionally to the knowledge of unknown parameters $x$, that is

$$
\forall i\leq m, \quad Y[i] ~|~ x \sim \mathcal{P}\left(\sum_{j\leq n} W[i,j] x[j] \right).
$$

The maximum-likelihood estimator in this setting finds the parameters $x$ that maximize (minimizes) the (negative) log-probability of the observations $y$,

$$
\argmin{x\in\mathbb{R}^{n}} - \sum_{i\leq m} \log p(Y[i]=y[i] ~|~ x),
$$

where the summation over $i$ is due to the independence of the random variables in $Y$. For Poisson distribution, one may observe that the log-likelihood is the KL divergence up to constant terms with respect to $x$:

$$
\log p(Y[i] = y[i] | x[i]) =  y[i]\log\sum_{j\leq n}W[i,j]x[j] - \sum_{j\leq n}W[i,j]x[j] - \log y[i]!.
$$

Therefore, computing the ML estimator amounts to solving the NNKL problem.

#### Basics of the EM algorithm

EM is a particular case of majorization-minimization algorithm that applies to the log-likelihood function. There are several ways to understand EM besides the usual textbook presentation (including MM, proximal point algorithms, alternating optimization), see for instance the discussion in {cite}`kunstnerHomeomorphicInvarianceEMNonAsymptotic2021`. The MM point of view connects EM with other optimization algorithms quite naturally, let us derive EM within the MM framework.

```{margin}
The EM algorithm is here presented with discrete probabilities for simplicity, but the general formulation for continuous probability densities is obtained in the same fashion.
```

EM can be obtained using combining two techniques: marginalization with respect to latent variables chosen by the user and Jensen/convexity inequality (similarly to the derivations of MU withing the MM framework) to approximate the . Introducing latent random variable(s) $Z$ taking values $z$ in the set $\mathcal{Z}$ and the observation vector $y$, using the total probability law, the log-likelihood writes

$$
   \log p(y ~|~ x) = \log \sum_{z \in \mathcal{Z}} p(y,z ~|~ x).
$$

These latent variables can represent, of instance, intermediate values in the optimization problem (see the application of EM to NNKL below), or missing observations. A core idea of EM (and more generally variational Bayesian estimation) is to leverage Jensen inequality to build a lower bound of $\log p(y ~|~ x)$ by introducing another well-chosen distribution. EM is an iterative algorithm where this distribution is chosen, at iteration k, as $p(z ~|~ y, x^{(k)})$, which is morally the distribution of the latent variables given the observed data and the current estimates of the unknown parameters. In many cases this conditional distribution can be derived from the statistical description of the problem. More precisely, the log-likelihood is lower-bounded as follows:

$$
   \log p(y ~|~ x) &= \log \sum_{z \in \mathcal{Z}} p(y,z ~|~ x) \frac{p(z ~|~ y, x^{(k)})}{p(z ~|~ y, x^{(k)})}, \\
                   &\geq \sum_{z \in\mathcal{Z}} p(z ~|~ y, x^{(k)}) \log  \frac{p(y,z ~|~ x)}{p(z ~|~ y,x^{(k)})}, \\
                   &\geq \mathbb{E}_{Z | y, x^{k}}\left[ \log p(y,z ~|~ x) \right] -  \underbrace{\mathbb{E}_{Z | y, x^{k}}\left[\log p(z ~|~ y, x^{(k)}) \right]}_{\text{constant w.r.t. }x}.
$$

One can also easily show that this low-bound is tight for $x=x^{(k)}$ and that the first order derivates match. To minimize the negative log-likelihood, the EM algorithm therefore minimizes $ \mathbb{E}_{z | y, x^{k}}\left[ -\log p(y,z ~|~ x) \right] $. There are again many other equivalent formulations of the functional minimized by EM, one that works well for source separation problems is to introduce back the conditional likelihood and prior information on the latent variables:

$$
\xi(x, x^{(k)}) = \mathbb{E}_{Z | y, x^{k}}\left[- \log p(y ~|~ z, x) - \log p(z ~|~ x) \right].
$$ (eq:EM)

Thus far the EM algorithm is rather high-level and abstract. One personnal difficulty to applying EM has always been to make sense of the probabilities and the conditional expectation in {eq}`eq:EM`. For some problems deriving these quantities can in fact be very challenging. Hopefully this is not the case for NNKL, and the EM algorithm is derived in the next section.

```{admonition} Relationship between EM, MU and mirror descent

Bauschke, Bolte and Teboulle have proposed to solve NNKL using (proximal) mirror descent {cite}`ref`[Bauschke Bolte]. They make use of a particular choice of potential, Burg's entropy $-\sum_i\log(x[i])$, and show that KL divergence is smooth relative to this choice with constant $\frac{1}{\|y\|_1}$. In other words KL divergence can be upper-bounded using a first-order approximation and the Bregman divergence $\mathcal{D}_h$ obtained with Burg's entropy

$ \KL{y,Wx} \leq \KL{y,Wx^{(k)}} + \langle \nabla_x\KL{y,Wx^{(k)}}, x - x^{(k)} \rangle + \frac{1}{2\|y\|_1}\mathcal{D}_{h}(x,x^{(k)}),$

that yields, after simpler derivations found in {cite}`ref`[Bauschke]

$ \KL{y,Wx} \leq \KL{y,Wx^{(k)}} - \langle W^T\frac{y}{Wx^{(k)}}, x - x^{(k)} \rangle + \frac{1}{2\|y\|_1} \sum_{j\leq n} \frac{x[j] - x^{(k)}[j]}{x^{(k)}[j]} - \log\frac{x[j]-x^{(k)}[j]}{x^{(k)}[j]}.$

Minimizing this upper bound leads to unusual multiplicative updates 

$ x = x \frac{1}{1+\frac{1}{\|y\|_1}x \odot \left(W^T1_n - W^T\frac{y}{Wx}\right)}. $

Since these updates allow for using (Bregman) proximal operators while ensuring convergence, they have been used in regularized NNKL problems {cite}`ref`[Hurault]. However it is unclear how these updates compare to the classical MU {eq}`eq:MU-NNKL` and its convergence rate is unknown.

Recently, a formal link was established between the EM algorithm for exponential family [ref kunstner 2022] and mirror descent with Bregman divergences. It allows in particular to derive linear convergence rates for EM, and therefore for the usual MU in the NNKL problem. This also shows that other choices of potential than Burg entropy lead to interesting updates rules for NNKL. This was brought to my attention by [Thibaut Modrzyk](https://github.com/Tmodrzyk) who is currently investigating this observation. The relationship between MU and mirror gradient descent was also partially discussed in [Hien]. Linear convergence for NNKL explains why the MU algorithm is an efficient algorithm to solve NNKL.

```

#### MU is an instance of EM for NNKL

To apply EM to NNKL, on top of introducing the Poisson distribution on the observed variables $Y[i]$, we introduce latent variables that will allow to derive the EM algorithm efficiently. The literature {cite}`Dempster`[aussi Shepp Vardi] informs us to define independent latent variables

$$
Z[i,j] \sim \mathcal{P}(W[i,j]x[j]),
$$

such that $\sum_{j\leq n} Z[i,j] = Y[i]$. In source separation, random variables $Z[i,j]$ model (parts of) the various components in an additive mixture and therefore bear physical meaning.

We can now expand the upper-bound $\xi$. First notice that the prior distribution of the latent variables $p(z ~|~ x)$ is separable into the product $\prod_{i,j} p(z[i,j] ~|~ x[j])$ since all latent variables are mutually independent, and the random variable $Z[i,j]$ depends only on the parameter $x[j]$. Therefore,

$$
\xi(x, x^{(k)}) = \sum_{i\leq m} \mathbb{E}_{Z[i,:] | y[i], x^{k}}\left[- \log p(y[i] ~|~ z[i,:], x) - \sum_{j\leq n}\log p(z[i,j] ~|~ x[j]) \right].
$$

The conditional likelihood $p(Y[i] ~|~ z[i,:], x)$ has a particular shape. Because we defined the latent variables $Z$ such that $Y[i] = \sum_j Z[i,j]$, conditionned on $Z$, $Y$ is deterministic. To simplify, we thereafter treat this first term in the majorant as a constraint imposing that the latent variables indeed sum up to the observations, $y[i] = \sum_j Z[i,j]$ (otherwise the logarithm goes to $-\infty$), and ignore it in the definition of the cost. Therefore,

$$
\xi(x, x^{(k)}) &= \sum_{i\leq m, j\leq n} \mathbb{E}_{Z[i,:] | y[i], x^{k}}\left[- \log p(z[i,j] ~|~ x[j]) \right] \text{ such that } \forall i\leq m,~\sum_j Z[i,j] = y[i] \\
                &= \sum_{i\leq m, j\leq n} \mathbb{E}_{Z[i,:] | \sum_{j}Z[i,j], x^{k}}\left[- \log p(z[i,j] ~|~ x[j]) \right].
$$

After expending the priors,

$$
- \log p(z[i,j] ~|~ x[j]) = W[i,j]x[j] - z[i,j]\log W[i,j]x[j] + \log z[i,j]!.
$$

the upper bound simplifies into

$$
\xi(x, x^{(k)}) = \sum_{i\leq m, j\leq n} W[i,j]x[j] - \mathbb{E}_{Z[i,:] | y[i], x^{k}}\left[z[i,j]\right]\log W[i,j]x[j] + \mathbb{E}_{Z[i,:] | y[i], x^{k}}[\log z[i,j]!].
$$

The last term involving the factorial of the latent variables is constant with respect to the parameters $x$ and can be ignored. What remains to compute to obtain a simpler form for the majorant $\xi$ is the expected value of $Z[i,j]$ conditioned to the knowledge of the sum of other Poisson variables $\sum_j Z[i,j]$ and the parameter values $x^{k}$ of the Poisson laws, $\mathbb{E}_{Z[i,:] |y[i], x^{k}}\left[z[i,j]\right]$ with $y[i]=\sum_j Z[i,j]$. 

The following Lemma allows to conclude, its proof is simple and available for instance in {cite}``[Oosten14].

%```{margin}
%This Lemma also explains the Wiener filtering procedure in audio signal to reconstruct the phase when sources have been estimated using magnitude spectrogramms, see for instance {cite}`ref`[todo ???]. 
%```

```{prf:lemma} Distribution of Poisson variables conditionned by their sum
:label: lemma_sum_poisson
Let $Z_1, Z_2$ be two random variables distributed respectively as $\mathcal{P}(\lambda_1)$ and $\mathcal{P}(\lambda_2)$. Then the conditionned random variable $Z_1 | Z_1+Z_2, \lambda_1,\lambda_2$ follows a Binomial distribution $\text{Bin}(Z_1+Z_2,\frac{\lambda_1}{\lambda_1+\lambda_2})$.
```

Applying {prf:ref}`lemma_sum_poisson` with $Z_1 = Z[i,j]$ and $Z_2 = y[i] - Z[i,j]$ yields

$$
\mathbb{E}_{Z[i,:] | y[i], x^{k}}\left[z[i,j]\right] = y[i]\frac{W[i,j]x^{(k)}[j]}{\sum_{j} W[i,j]x[j]}=  y[i]\frac{W[i,j]x^{(k)}[j]}{\hat{y}[i]}.
$$

Notice that we recover the weights of the MM formulation of MU, $y[i]\lambda_{i,j} = \mathbb{E}_{Z[i,:] | y[i], x^{k}}\left[z[i,j]\right]$, which provides a nice interpretation of the ad-hoc choice for these parameters in the MM framework for NNKL.

We now observe that the majorant $xi$ is equal up to constant terms to the majorant obtained with the MM framework, and the gradients of the two convex separable majorant match:

$$
\xi(x, x^{(k)}) &= \sum_{i\leq m, j\leq n} W[i,j]x[j] - y[i]\frac{W[i,j]x^{(k)}[j]}{\hat{y}[i]}. \log x[j] + \text{cst(x)}, \\
\frac{\partial}{\partial x[j]}\xi(x, x^{(k)}) &= \sum_{i\leq m} W[i,j] - \frac{x^{(k)}[j]}{x[j]}\sum_{i\leq m}\frac{W[i,j]y[i]}{\hat{y}[i]}.
$$

Therefore the EM algorithm for NNKL is exactly MU.

#### MU may not be an instance of EM for NNLS

Importantly, while the EM algorithm can in principle be used to derive update rules for the NNLS problem given reasonable additional statistical assumptions on the variance of the variables $z[i]$, the update rules obtained with these assumptions distributions do not match the MU updates {eq}`eq:MU-NNLS`. Instead, they are another particular case of diagonally preconditioned gradient descent, see for instance {cite}`ref`[Van Oosten 2014].


```{admonition} Equivalence of the MU formulations breaks with l1 and l2 regularizations
:class: tip

If additive $\ell_1$ and $\ell_2$ regularizations terms $\lambda_1 \|x\|_1 + \frac{1}{2}\lambda_2 \|x\|_2^2$ are added to the cost, the MU updates using the gradient ratio for both NNLS and NNKL would write

$$
x^{(k+1)} = x^{(k)}\frac{ \nabla_x^- f\left(y,x\right) }{\nabla_x^+ f\left(y,x\right) + \lambda_1 + \lambda_2 x^{(k)}}.
$$

since the gradients of the regularizations are always positive.
For instance for NNKL, MU obtained from the gradient ratio heuristic are

$$
x^{(k+1)} =  x^{(k)} \frac{W^T\frac{y}{Wx^{(k)}}}{W1_n + \lambda_1 + \lambda_2 x}.
$$


However using the MM framework, the impact of the regularizations on the update rules change for NNLS and NNKL. Note that a more complete analysis is provided in {cite}`ref`[TODO valou]. The updates within the MM framework are given by solving

$$
\forall j\leq n, \quad 0 = \sum_{i\leq m} W[i,j] \frac{\partial}{\partial x_2} f\left(y[i], \frac{x[j]}{x^{(k)}[j]}\hat{y}[i]\right) + \lambda_1 + \lambda_2 x[j].
$$

Injecting the gradients of the cost function yields

$$
\text{for NNLS:}\quad & \forall j\leq n, \quad 0 = \sum_{i\leq m} W[i,j] \left(\frac{x[j]}{x^{(k)}[j]}\hat{y}[i] - y[i]\right) + \lambda_1 + \lambda_2 x[j],\\
\text{for NNKL:}\quad &  \forall j\leq n, \quad 0 = \sum_{i\leq m} W[i,j] \left( \frac{y[i]x^{(k)}[j]}{\hat{y}[i]x[j]} - 1\right)  + \lambda_1 + \lambda_2 x[j].
$$

and the resulting MU updates for the NNLs problem are

$$ 
x^{(k+1)} = \max\left( x^{(k)} \frac{W^Ty - \lambda_1}{W^TWx^{(k)} + \lambda_2}, \epsilon\right).
$$

Note the difference with the gradient ratio heuristic that places the $\ell_1$ regularization hyperparameter in the denominator. Moreover, these MU updates with $\lambda_1\geq 0$ can in principle become negative or zero, therefore positivity constraints should be enforced as discussed [above](block:pos_cstr).

For the NNKL problem, one needs to compute the positive roots of the second-order polynomials

$$ 
\lambda_2 x[j]^2 + \left(\sum_{i\leq m} W[i,j] + \lambda_1 \right) x[j] + x^{(k)}[j]  \sum_{i\leq m}W[i,j]\frac{y[i]}{\hat{y}[i]}.
$$

```
 
