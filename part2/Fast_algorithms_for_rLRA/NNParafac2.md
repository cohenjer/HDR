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

# Nonnegative PARAFAC2

:::{admonition} Reference
:class: tip
{cite:p}`cohenNonnegativePARAFAC2Flexible2018` J. E. Cohen, R. Bro, "Nonnegative PARAFAC2, a flexible coupling approach", LVA/ICA 2018, [arXiv:1802.05035](https://arxiv.org/abs/1802.05035)

{cite:p}`roaldPARAFAC2AOADMMConstraints2021` M. Roald, C. Schenker, J. E. Cohen, E. Acar, "PARAFAC2 AO-ADMM: Constraints in all modes", EUSIPCO2021, [arxiv](https://arxiv.org/pdf/2102.02087.pdf)

{cite:p}`roaldAOADMMApproachConstraining2022a` M. Roald, C. Schenker, V. D. Calhoun, T. Adali, R. Bro, J. E. Cohen, E. Acar, "An AO-ADMM approach to constraining PARAFAC2 on all modes", SIAM Journal of Mathematics on Data Science, 2022. [arxiv](https://arxiv.org/abs/2110.01278) [code](https://github.com/MarieRoald/PARAFAC2-AOADMM-SIMODS) [SIMODS](https://epubs.siam.org/doi/10.1137/21M1450033)
:::

## Why Nonnegative PARAFAC2

The PARAFAC2 model is often used in applications such as Liquid/Gas Chromatography Mass Spectroscopy (LCMS and GC-MS, respectively) to extract mass spectra and time elution profiles for various experimental setups. The mixture of interest is injected into a long tube where the various chemical compounds glide at different speeds, and therefore reach the end of the tube at different times. By measuring the mass spectrum at the output of the tube with a mass spectrometer, one may acquire spectra at various time instants. One acquisition, therefore, yields a data matrix with intensities collected at various coordinates (mass-to-charge (m/z) x time). Repeating this acquisition several times for different chemical compounds yields a tensor dataset, with the third dimension indexing the experiments. An example GCMS dataset is available below, with different wine samples from various producers measured individually. The full dataset is described in {cite:p}`ballabioClassificationGCMSMeasurements2008`.

%Also nice example usage with others constraints by Rivet and Magbonde 2023 (finger strength of climbers)

```{code-cell}ipython3
:tags: [hide-input, hide-output]

import matcouply as mcl
import tensorly as tl
from tensorly_hdr.load_eem import load_gcms_interval
import matplotlib.pyplot as plt
import numpy as np
from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Slider, ColumnDataSource, CustomJS
from bokeh.palettes import Category10

output_notebook()

# Experiment (long time) x time x m/z
dataset, tensor, time_steps = load_gcms_interval()
tensor = tensor[:52,:,:]  # two full acquisitions of the wine along the industrial process

```

```{code-cell}ipython3
:tags: [hide-input]

# Initial indices
xp_index = 1
mz_index = 23
time_index = 12
tensor_shape = tensor.shape  # (xp, time, mz)

# Colors
palette = Category10[7]
colors3 = [palette[i % len(palette)] for i in range(tensor_shape[0])]

# Plot 1: multiple curves (one per m/z)
ys1 = [tensor[l,:,:] for l in range(tensor_shape[0])]
source1 = ColumnDataSource(data=dict(image=[ys1[time_index]]))
p1 = figure(height=250, width=350, title="Elution profiles vs m/z", y_axis_label="Time_index", x_axis_label="m/z index")
p1.image(image='image', x=0, y=0, dw=tensor_shape[2], dh=tensor_shape[1], palette="Viridis256", source=source1)

# Plot 2: image
ys2 = [tensor[:,:,l] for l in range(tensor_shape[2])]
source2 = ColumnDataSource(data=dict(image=[ys2[mz_index]]))
p2 = figure(height=250, width=350, title="Elution profiles vs xp index", y_axis_label="Experiment index", x_axis_label="Time index")
p2.image(image='image', x=0, y=0, dw=tensor_shape[1], dh=tensor_shape[0], palette="Viridis256", source=source2)

# Plot 3: curve at xp_index over time for a fixed m/z
xs3 = [time_steps.tolist()] * tensor_shape[0]
ys3 = [[tensor[i, :, j] for i in range(tensor_shape[0])] for j in range(tensor_shape[2])]
source3 = ColumnDataSource(data=dict(x=xs3, y=ys3[mz_index], color=colors3))
p3 = figure(height=250, width=350, title="Elution profile across xp_index", x_axis_label="Time")
p3.multi_line(xs='x', ys='y', line_color='color', source=source3)

# Plot 4: m/z spectrum
xs4 = np.arange(tensor_shape[2])
ys4 = [[tensor[i, j, :] for j in range(tensor_shape[1])] for i in range(tensor_shape[0])]
source4 = ColumnDataSource(data=dict(x=xs4, y=ys4[xp_index][time_index]))
p4 = figure(height=250, width=350, title="Mass over charge spectrum", x_axis_label="m/z index", y_axis_label="Intensity")
p4.line('x', 'y', source=source4)

# Sliders
slider_xp = Slider(start=0, end=tensor_shape[0]-1, value=xp_index, step=1, title="xp_index")
slider_mz = Slider(start=0, end=tensor_shape[2]-1, value=mz_index, step=1, title="mz_index")
slider_time = Slider(start=0, end=tensor_shape[1]-1, value=time_index, step=1, title="time_index")

# JavaScript callback
callback = CustomJS(args=dict(
    source1=source1,
    source2=source2,
    source3=source3,
    source4=source4,
    slider_xp=slider_xp,
    slider_mz=slider_mz,
    slider_time=slider_time,
    ys1=ys1,
    ys2=ys2,
    xs3=xs3,
    ys3=ys3,
    xs4=xs4,
    ys4=ys4,
    colors3=colors3
), code="""
    const xp = slider_xp.value;
    const mz = slider_mz.value;
    const t = slider_time.value;

    // Plot 1: all m/z curves at xp
    source1.data.image[0] = ys1[xp];
    source1.change.emit();

    // Plot 2: image at mz
    source2.data.image[0] = ys2[mz];
    source2.change.emit();

    // Plot 3: curve at xp over time
    source3.data = { x: xs3, y: ys3[mz], color: colors3 };

    // Plot 4: m/z spectrum
    source4.data = { x: xs4, y: ys4[xp][t] };
""")

slider_xp.js_on_change('value', callback)
slider_mz.js_on_change('value', callback)
slider_time.js_on_change('value', callback)

layout = column(row(slider_xp, slider_mz), slider_time, row(p1, p2), row(p3, p4),)
show(layout)

```

One interesting feature of this dataset, visible in the top-right plot, is that the time elution profiles shift slightly with the experiment index.

Formally, given a collection of mass spectra over time $X[:,:,k]$ where $k\leq K$ stands for the experiment index, supposing elution profiles and mass spectra can be extracted from each experiment using a low-rank approximation, and further supposing that these components are shared across all experiments, estimating these components amounts to computing

$$
    X[:,:,k] = AD_kB^T,
$$

that is by computing a CPD from the data tensor $X$. The factors bear physical meaning (mass spectra, elution time profiles, and experiment contribution); it is reasonable to impose nonnegativity on all factors.

However, in practice, as observed in this dataset, the injection of chemical compounds into the chromatography system is done manually, and the elution time varies due to other experimental conditions, such as temperature. Therefore, the time profiles of components $B$ are slightly modified across experiments indexed by $k$. As discussed in {ref}`subsec:parafac2-and-variants`, further constraints are required to link the resulting $B_k$ matrices, and PARAFAC2 ensure in particular that 
- The columns of factors $B_k$ are transformed similarly (the same transformation matrix $P_k$ applies to each column $B[:,q]$).
- The cross product of the $B_k$ factors is constant.

A technical difficulty in applying PARAFAC2 to GCMS data is designing an optimization algorithm that allows for nonnegativity constraints on the $B_k$ matrices. Indeed, the classical algorithm to fit PARAFAC2 proposed by Kiers and Bro {cite:p}`Kiers1999PARAFAC2` relies on the reparameterization $B_k = P_kB$. It can be summarized as follows:
1. Compute the optimal $P_k$ matrices by solving $K$ orthogonal Procrustes problems.
2. Compute the CP decomposition of the tensor obtained by stacking $X[:,:,k]P_k$.
3. Loop the first two steps until convergence

The matrices $B_k$ do not appear explicitly in this algorithm, and therefore, we need to rethink the optimization procedure to include these constraints.

## Formalisation of the Nonnegative PARAFAC2 problem

In our first work with Rasmus Bro, we proposed to use a flexible constraint formulation for the Nonnegative PARAFAC2 {cite:p}`cohenNonnegativePARAFAC2Flexible2018`. While this work had a good impact on the chemometrics community, it relies on a relaxation of the PARAFAC2 constraint. It is, however, possible to use the AO-ADMM framework discussed in {ref}`LC-CMTF` to design an alternating iterative algorithm that solves the nonnegative PARAFAC2 problem. The core idea is to introduce a "PARAFAC2 constraint" on the matrices $B_k$ that imposes that their cross products are constant. A collection of matrices $\{B_k\}_{k\leq K}$ satisfies the PARAFAC2 constraint if for all $k\leq K$, $B_k^TB_k$ is constant. In that case we write that $\{B_k\}_{k\leq K}\in\mathcal{C}_{\text{P2}}$.

Nonnegative PARAFAC2 then boils down to the optimization problem

$$
    \argmin{A\geq 0,~B_k \geq 0,~C\geq 0 } \|X[:,:,k] - A\text{Diag}(C[k,:])B_k^T \|_F^2 \text{  s.t.  } \{B_k\}_{k\leq K}\in\mathcal{C}_{\text{P2}}.
$$

This problem can be handled by AO-ADMM with splitting, provided we can project onto the set of sets of matrices satisfying the PARAFAC2 constraint. We explore this projection next.

## Projection on the PARAFAC2 constraint

A full understanding of the projection under the PARAFAC2 constraint has yet to be reached, and it remains an active research topic. It connects with Riemmanian geometry and optimization, but has not been studied in this literature as such, to the best of my knowledge.

In the work conducted with Marie Roald and co-authors, we used an empirical approach to this projection, relying on a few mathematical observations. First, notice that if a set $\{ B_k\}_{k\leq K}$ satisfies the PARAFAC2 constraint, it can be represented as points on the orbit 

$$
\{PB ~|~ P^TP=I  \} = \mathcal{O}(B).
$$

Projection on the PARAFAC2 constraint therefore means finding a matrix $B$ such that the distance $d(B, B_k)$ between each point $B_k$ and the orbit $\mathcal{O}(B)$ is as small as possible:

$$
 \argmin{B} \sum_{k\leq K} d(B,B_k).
$$

The distance $d$ we propose to use is the Euclidean distance to the closest point $P_kB$ on $\mathcal{O}(B)$ to $B_k$, that is

$$
d(B,B_k) = \argmin{P_k \text{ orthogonal}} \|B_k - P_kB\|_F^2.
$$

Projection on the PARAFAC2 constraint, therefore, boils down to solving

$$
\argmin{B, P_k \text{ orthogonal }} \sum_{k\leq K} \|B_k - P_kB \|_F^2.
$$ (eq:P2proj)

From the point of view of algorithm design, we seek a procedure to compute exact solutions to the problem {eq}`eq:P2proj` and plug them into the AO-ADMM algorithm. Maybe surprisingly, the following simple alternating algorithm typically converges in a few iterations:
1. Compute orthogonal matrices $P_k$ solving $K$ orthogonal Procrustes problems.
2. Compute the geodesic mean $B$ as $\argmin{B} \sum_{k} \|P_k^TB_k-B\|_F^2 = \frac{1}{K}\sum_{k} P_k^TB_k $. 

We can observe the procedure's fast convergence empirically in the experiment below.

```{code-cell}ipython3

# Hyperparameters
dims = [10,10,5]  # Dimensions of the tensor X
rank = 3  # Rank of the factorization

# Generate a collection of (nonnegative) matrices Bk randomly
Bks = [np.random.rand(dims[1],rank) for _ in range(dims[2])]

def project_parafac2(Bks, itermax=10):
    B = sum(Bks) / len(Bks)  # Initial guess as the mean of the input matrices
    cost = [sum([np.linalg.norm(Bks[i] - B)**2 for i in range(len(Bks))])]  # Initial cost
    Pks = [np.eye(0) for _ in range(len(Bks))]  # Initialize Pk matrices
    
    for _ in range(itermax):
        # Solve for Pk rotation matrices
        for i in range(len(Bks)):
            # Project each matrix Bk onto the PARAFAC2 constraints
            U,s,V = np.linalg.svd(Bks[i]@B.T)
            Pks[i] = U @ V
            
        # Update B matrix as the mean
        B = sum([Pks[i].T @ Bks[i] for i in range(len(Bks))]) / len(Bks)
        
        # Recompute the cost
        cost.append(sum([np.linalg.norm(Bks[i] - Pks[i] @ B)**2 for i in range(len(Bks))]))
        
        print(f"Iteration {_+1}, Cost: {cost[-1]:.8f}")
        
    return B, Pks, cost

out = project_parafac2(Bks, itermax=5)

```

The convergence speed of this alternating optimization algorithm is suspiciously fast, and we suspect that there may exist a similar procedure that converges in a single iteration, or at least a theoretical analysis guaranteeing the fast convergence of this heuristic projection. What is known for now is that this alternating algorithm converges to the solution of the problem, as a two-block alternating optimization with exact and unique solutions to each subproblem, see [](../../part1/AlternatingOptimization.md).

## Nonnegative PARAFAC2 for GCMS data

We may now compute PARAFAC2 with and without nonnegativity constraints on the time elution mode. The AO-ADMM implementation of PARAFAC2 with constraints is found in the [matcouply package](https://github.com/MarieRoald/matcouply){cite}`roaldMatCoupLyLearningCoupled2023`, developed by [Marie Roald](https://github.com/MarieRoald) at Simula (Norway) during her PhD thesis. For PARAFAC2 without nonnegativity on the second mode (but with nonnegativity on the first mode to avoid sign ambiguities), we use the implementation available in tensorly, which uses reparameterization and relies on the CP decomposition.

```{code-cell}ipython3

import matcouply as mcl
from matcouply.decomposition import parafac2_aoadmm
from tensorly.decomposition import parafac2

# With nonnegativity constraint
cmf_nn, diagnostics_nn = parafac2_aoadmm(
    tensor, 3, n_iter_max=200, non_negative=True, verbose=False, return_errors=True, tol=1e-9, random_state=5
)
weightsnn, (Ann, B_isnn, Cnn) = cmf_nn

# tensorly unconstrained PARAFAC2
pf2, rec_err = parafac2(
    tensor, 3, n_iter_max=200, return_errors=True, nn_modes=[0], random_state=0, tol=1e-9, verbose=False
)
weights, (A, B, C), P_is = pf2
B_is = [P_i @ B for P_i in P_is]

```


```{code-cell}ipython3
:tags: [hide-input]

# Showing error plot
plt.figure(figsize=(4, 3))
plt.title("Convergence of the PARAFAC2 decomposition")
plt.xlabel("Iteration")
plt.ylabel("Error (log scale)") 
plt.semilogy(rec_err, label= "No constraints")
plt.semilogy(diagnostics_nn[0], label="Nonnegative")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input, hide-output]

from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider, CustomJS

output_notebook()
```

```{code-cell} ipython3
:tags: [hide-input]

# We normalise the components to make them easier to compare
A_norm = np.linalg.norm(A, axis=0, keepdims=True)
C_norm = np.linalg.norm(C, axis=0, keepdims=True)
A = A / A_norm
B_is = [np.abs(B_i * A_norm * C_norm) for B_i in B_is]
C = C / C_norm
# Also Nonnegative output
Ann_norm = np.linalg.norm(Ann, axis=0, keepdims=True)
Cnn_norm = np.linalg.norm(Cnn, axis=0, keepdims=True)
Ann = Ann / Ann_norm
Bnn_is = [Bnn_i * Ann_norm * Cnn_norm for Bnn_i in B_isnn]
Cnn = Cnn / Cnn_norm

# Sample data: list of matrices (each 100 x 3)
n_B = len(B_is)
n_points = B_is[0].shape[0]
n_modes = B_is[0].shape[1]
colors = ["black", "magenta", "blue"]

# Flatten B_is into a JS-friendly dictionary
data_dict = {}
for i in range(len(B_is)):
    for j in range(n_modes):
        data_dict[f"B{i}"] = [B_is[i][:, j] for j in range(n_modes)]
        data_dict[f"B{i}nn"] = [Bnn_is[i][:, j] for j in range(n_modes)]

# Initial source
xs = [time_steps.tolist()]*3
source = ColumnDataSource(data=dict(
    x=xs,
    y=data_dict['B0'],  # Use the first B matrix for initial display
    color=colors
))
source2 = ColumnDataSource(data=dict(
    x=xs,
    y=data_dict['B0nn'],
    color=colors
))

# Plot
p = figure(height=300, width=350, title="Without nonnegativity on B mode (abs value)", x_axis_label="Time", y_axis_label="Intensity")
p.multi_line(xs='x', ys='y', source=source, line_color='color')

p2 = figure(height=300, width=350, title="With nonnegativity on B mode", x_axis_label="Time", y_axis_label="Intensity")
p2.multi_line(xs='x', ys='y',  source=source2, line_color='color')

# Slider
slider = Slider(start=0, end=n_B - 1, value=0, step=1, title="Experiment index")

# Callback
callback = CustomJS(args=dict(source=source, source2=source2, slider=slider, xs=xs, data=data_dict, n_modes=n_modes, n_points=n_points, colors=colors
                              ), code="""
    const i = slider.value;
    source.data = { x: xs, y: data[`B${i}`], color: colors };
    source.change.emit();
    source2.data = { x: xs, y: data[`B${i}nn`], color: colors };
    source2.change.emit();
""")

slider.js_on_change("value", callback)

layout = column(row(p, p2), slider)
show(layout)
```

One may observe that the nonnegative PARAFAC2 model can recover elution peaks that overlap significantly with the baseline. Without nonnegativity constraints, the extracted elution profiles are somewhat mixed, and the baseline is found in all three components. 

In a collaboration with Marie Roald, Carla Schenker, and Evrim Acar (SimulaMet, Oslo), we have extended the proposed algorithms for nonnegative PARAFAC2 to other constraints {cite:p}`roaldAOADMMApproachConstraining2022a`. The algorithm works similarly by projecting onto the PARAFAC2 constraint and applying the proximity operator of the constraints to the $B_k$ matrices, after splitting.

