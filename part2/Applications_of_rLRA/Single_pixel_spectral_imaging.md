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

# Single pixel hyperspectral image reconstruction and unmixing

:::{admonition} Reference
:class: tip
{cite:p}`harigaJointReconstructionSpectral2024` S. Hariga, J. E. Cohen and N. Ducros, "Joint Reconstruction and Spectral Unmixing from Single-Pixel Acquisitions", EUPSICO 2024. [hal](https://hal.science/hal-04539349)

{cite:p}`harigaApprochePlugandplayPour2025` S. Hariga, A. Repetti, N. Ducros, J. E. Cohen,  "Approche plug-and-play pour la reconstruction des cartes d’abondance en imagerie hyperspectrale mono-pixel" GRETSI 2025 [hal](https://hal.science/hal-05235066v1/document)

{cite:p}`ducrosIntroductionSinglePixelImaging2024` N. Ducros, J. E. Cohen, L. Mahieu-Williame, "Freeform Hadamard imaging: Back to the roots of computational optics", under review, [hal](https://hal.science/hal-05337760v1)
:::

## Single-pixel hyperspectral imaging principle

Spectral cameras are invaluable tools for acquiring detailed spectral information. They function by spatially spreading an incoming light flux across its constituent wavelengths. This spectral-spatial (Fourier) transformation implies that, by design, it is difficult to acquire spectral images with high spatial and spectral resolution. This is particularly true if the imaging device must remain affordable and lightweight.

Although the idea of compressive acquisitions in computational optics precedes the development of compressive sensing {cite:p}`nelsonHadamardSpectroscopy1970c`, Duarte and co-authors proposed the single-pixel camera, which uses spatial multiplexing to feed focalized masked images into a high-resolution point spectrometer {cite:p}`duarteSinglepixelImagingCompressive2008, Duarte2012Kronecker`. In the setup available at CREATIS, the compressed light flux is the input of a spectrometer. The hyperspectral cube containing both the spatial information (images) and the spectral information (spectra) is then reconstructed by solving an inverse problem, following a long tradition of reconstruction imaging techniques in the computational imaging community {cite:p}`BenetiMartins_2023`.

```{figure} ../../Figures/schema_spc.png
---
width: 650px
align: center
name: singlepixel_camera
---
Representation of the single pixel camera [credits: Séréna Hariga].
```



Let us formalize the problem. Let $Y\in\mathbb{R}_+^{m\times p}$ be the acquired matrix, with $m$ the number of measurements or patterns, and $p$ the number of spectral bands measured by the spectrometer. Each acquisition, say the $k$-th, is obtained by summing the pixels of a masked image $a_k \ast_{1,2} X_t$ where $a_k\in\{0,1\}^{n_1\times n_2}$ is a binary mask and $X_t\in\mathbb{R}_+^{n_1\times n_2\times p}$ is the unknown hypespectral cube to reconstruct, with $n_1\times n_2$ pixels. The elementwise product $\ast_{1,2}$ indicates that the mask is applied on each slice $X_t[:,:,i]$ of the hypercube. It can be convenient to vectorize the spatial dimension and consider the vector patterns $A[k,:]\in\{0,1\}^{n_1n_2}$ as well as the unknown matrix $X\in\mathbb{R}_+^{n_1n_2\times p}$. The noiseless acquisition is therefore modeled linearly as $Y = AX$. 

The noise model for single-pixel imagers is standardized as a Poisson-Gaussian mixture. For the sake of simplicity, and since Gaussian noise is typically weak in these acquisitions, we consider the noise to be pure Poisson. Poisson noise arises because the spectrometer counts photons hitting its electronic sensors, and counting errors/fluctuations are well modeled by the Poisson distribution. The model then writes

$$ Y \sim \mathcal{P}(\alpha AX) $$

where $\alpha$ is the average photon count that can be interpreted as the signal strength. The signal to noise ratio can be shown to be proportional $\sqrt{\alpha}$: the higher $\alpha$ the less noisy the acquisition. We consider that the hypercube $X$ has normalized intensities in $[0,1]$.

The reconstruction of the hypercube $X$, knowing the measurements $Y$ and the acquisition matrix $A$, is an inverse problem. It is ill-posed in the sense that infinitely many measurements are required to obtain a perfect reconstruction, because of Poisson noise. Furthermore, if the number of patterns is strictly smaller than the number of pixels $n:=n_1n_2$, the linear system $Y=AX$ has infinitely many solutions. For both these reasons, regularization is essential in single-pixel image reconstruction. The [classical approach](#reconstruction-wavelength-by-wavelength-existing-work) is to reconstruct each slice of the hypercube separately, wavelength by wavelength, using state-of-the-art image reconstruction algorithms and priors. In the PhD thesis of Serena Hariga, co-supervised with Nicolas Ducros, we have studied the [joint reconstruction and unmixing](#joint-reconstruction-and-unmixing-eusipco) of the hypercube, using spectral regularization, as well as [spectral unmixing directly from the measurements](#unmixing-with-known-spectra-gretsi) with spatial priors.

## Reconstruction wavelength by wavelength (existing work)

### Hadamard patterns

Before considering a baseline reconstruction method, it is necessary to discuss more about the choice of the acquisition matrix $A$. There are both practical and theoretical constraints that guide the choice of $A$. 
- Practically, the patterns are applied to the image by leveraging a micromirror matrix (DMD). The image is focused on the DMD, which has small mirrors corresponding to pixels that can point in two different directions. All the mirrors that point towards the lens that targets the spectrometer reflect light from the observed object and contribute to the measured spectrum. The other mirror effectively blocks all the incoming light. Therefore, we may assume that the observation matrix is binary. Contrary to other imaging modalities, such as magnetic resonance imaging, the patterns can be changed at will for each acquisition.
- Theoretically, if the patterns could be negative in $[-1, 1]$, the "optimal" acquisition patterns are known to be Hadamard matrices {cite:p}`nelsonHadamardSpectroscopy1970c`. Optimality here is meant in the sense of statistical minimal linear reconstruction error, *i.e.* minimal estimation variance. In particular, Hadamard patterns are shown to outperform raster scan, which observes each pixel individually and therefore does not perform spatial multiplexing. The theoretical gain of Hadamard matrices versus raster scan is known as the Felgett advantage.

:::{admonition} Why are Hadamard patterns "optimal"

In 1970, Nelson and Fredman introduced the concept of Hadamard spectrocopy {cite:p}`nelsonHadamardSpectroscopy1970c`. In their seminal work, they prove that Hadamard patterns are an "optimal" choice for the acquisition basis. Let us clarify in what sense this optimality is meant, and under which assumptions.

Nelson and Fredman are concerned with linear reconstructions of acquisitions $y=Ax + e$, where matrix $A$ is either the identity matrix (raster scan) or a complete Hadamard basis. The noise $e$ is supposed to be additive, i.i.d. with finite variance $\sigma$, but the distribution of $n$ is not necessarily Gaussian. 

Linear reconstructions are of the form $\hat{x} = By$ with matrix $B$ defined as the pseudo-inverse of matrix $A$. This choice for the reconstruction corresponds to the least-squares estimation of the unknown $x$; it is the maximum-likelihood estimator under Gaussian noise. If the noise is not Gaussian, its statistical meaning is more ambiguous.

The quantity that Nelson and Fredman consider for measuring the quality of the reconstruction is the Root Mean Squared Error (RMSE), defined as

$$ \mathbb{E}\left[ \| x - \hat{x} \|^2 \right]. $$

RMSE is exactly the average estimation error in $\ell_2$ norm, and therefore is a reasonable error metric for Gaussian noise. Because the noise $e$ is i.i.d., the RMSE is easily derived as

$$ \text{RMSE}(B) = \sigma\text{Diag}(\sqrt{B^2\mathbb{1}_n}), $$

or for the $i$-th measurement only, 

$$ \text{RMSE}_i(B) = \sigma\sqrt{\sum_{j} B[i,j]^2}. $$

The Felgett advantage is obtained when comparing the RMSE of Hadamard patterns and raster scan. The ratio of the $\text{RMSE}_i$ values, denoted by $F_i$, is simply 

$$F_i = \sqrt{\sum_{j} B[i,j]^2} = \|B[i,:]\|_2.$$

For raster scan the rows of matrix $B^2$ sum to one, while for Hadamard patterns $B=\frac{1}{N}A^T$, which leads to 

$$ F_i =: F = \sqrt{n}.$$

A reasonable question is whether there exist better choices than Hadamard matrices for maximizing this ratio. Interestingly, Nelson and Fredman show that the ratio $F$ is bounded by $\sqrt{n}$. Indeed, by the Cauchy-Schwarz inequality,

$$ 1 = \sum_{j} B[i,j]A[i,j] \leq  \underbrace{\|B[i,:]\|_2}_{F_i^{-1}} \|A[i,:]\|_2. $$

Therefore, it holds that $F_i\leq \|A[i,:]\|_2 \leq \sqrt{n}$ where the last inequality assumes that the measurement operator $A$ has all values between $-1$ and $1$. Hadamard matrices are thus MSE-optimal.

The optimality of the Hadamard patterns, however, is meant under many unrealistic assumptions:
- The measurement operator $A$ has values in $[-1,1]$. The bound on the magnitude is reasonable for single-pixel imaging, but a DMD can only implement nonnegative entries. Neslon and Fredman discussed this issue and introduced binary $S$-matrices, which have a Felgett advantage of $\frac{n+1}{2\sqrt{n}}$, therefore quasi-optimal.
- The noise is additive. In single-pixel imaging, the noise model is Poisson-Gaussian, so additivity is not a valid assumption.
- The reconstruction is linear. Most state-of-the-art reconstruction algorithms employ more sophisticated reconstruction strategies.
- The error is measured with the RMSE, which is not a robust or statistically meaningful error metric for Poisson-Gaussian noise.

There are, therefore, ample research avenues for improving upon the (nonnegative) Hadamard patterns design. Revisiting the MSE gains for Poisson-Gaussian noise is one of my [research perspectives](../../part3/KarpCoi.md#kl-based-inverse-problems-for-computational-imaging).
:::

A practical solution for implementing the DMD of Hadamard patterns is to acquire the positive and negative parts sequentially and concatenate the measurements into a single vector. If the image has $n$ pixels and $n$ Hadamard patterns to be acquired, the number of measurements is then $m:=2n-1$ (one pattern is full of zeros and ignored), and the data vector $y$ has size $m$. Doing so preserves the Poisson distribution. Another usual post-processing method computes the difference of the positive and negative acquisitions, resulting in a simulated Hadamard acquisition, but the noise distribution is modified. Formally, we define 

$$ A = \left[ \begin{array}{c} [H]^+ \\ [H]^- \end{array}\right] $$

where $H$ here denotes a Hadamard matrix, and $[x]^- \geq 0$ is the negative part of a vector or matrix $x$. Matrix $[H]^-$ contains a row of zeroes that is usually removed. 


```{figure} ../../Figures/hadamard.gif
---
width: 350px
align: center
name: hadamard
---
Examples of Hadamard patterns stored in the rows of matrix $H$. Ones are shown in white, and minus ones (zeroes in practice) in black. The patterns are orthogonal and have the same number of black and white pixels, except the first two (all white, all black).
```

In short, we suppose that measurements $Y$ are stored in a matrix of size $m\times p$, and assume the model $Y\sim \mathcal{P}\left(\alpha AX\right)$ with $A$ of size $m\times n$ a matrix with entries in $\{0,1\}$.

Hadamard matrices have both a simple form for the pseudo-inverse and a cheap matrix-vector product {cite:p}`magoarouChasingButterfliesSearch2015`. It is natural to question whether one can apply, and invert, the split operator $A$ at a similar cost, and we show here that both operations can be performed efficiently when $A$ is the split Hadamard matrix.

First, the product $A^Ty$ can be obtained using the fast Hadamard transform. Indeed, $[H]^{\pm} = \frac{1}{2} \left[H \pm 1\right]$. Second, the pseudo-inverse $A^{\dagger}$ admits a closed-form expression, see section 3.5.4 in {cite:p}`Harwit_1979`.

$$ A^\dagger = \frac{2}{n} \left( I_n - \frac{1}{n+1} \mathbb{1}_{n\times n} \right) A^T $$

where $\mathbb{1}_{n,n}$ is a matrix of ones, and its contraction is implemented as a summation. This is obtained by noting that

$$ A^TA = \frac{n}{2} \left( I_n +\mathbb{1}_{n\times n} \right) $$

and that $ \left( I_n - \frac{1}{n+1}\mathbb{1}_{n\times n} \right)\left( I_n +\mathbb{1}_{n\times n} \right)  = I_n $.

```{note} Inversion of the positive Hadamard matrix

Most matrices obtained from the Hadamard matrix admit fast inversion algorithms. The methodology, exemplified on matrix $[H]^+$, is as follows.

Matrix $[H]^+$ is left-invertible, and by noticing that

$$ \frac{1}{n}H^T[H]^+ = \frac{1}{2}I + \frac{1}{2} e_1 \mathbb{1}^T,\; e_1 = [1,0,\ldots,0]^T  $$

we get that for any vector $x\in\mathbb{R}^{n}$,

$$ \frac{1}{n}H^T[H]^+ x = \left[x[1] + \sum_{j>1} \frac{x[j]}{2} , \frac{x[2]}{2}, \ldots, \frac{x[n]}{2} \right]^T, $$

and vector $x$ is recovered as

$$ x = \frac{1}{n} \left(2I_n - e_1 \mathbb{1}^T\right)H^T [H]^+ x.$$ 

We see that the pseudo-inverse of $[H]^+$ is computed as

$$ \frac{1}{n} \left(2I_n - e_1 \mathbb{1}^T\right)H^T = \frac{1}{n} \left( 2H^T - e_1e_1^T\right) $$

which can be efficiently contracted with any vector $x$ given a fast Hadamard transform algorithm.

```

We can easily simulate the acquisition with the help of the Python package spyrit, developed at CREATIS {cite:p}`Abascal_2025`.

```{code-cell}ipython3
import torch, torchvision
from spyrit.core.meas import HadamSplit2d
from spyrit.core.noise import Poisson
import matplotlib.pyplot as plt
import tensorly as tl
tl.set_backend("pytorch")

# Cat obtained from https://www.kaggle.com/datasets/mahmudulhaqueshawon/cat-image (small)
x = torch.sum(torchvision.io.read_image("../../tensorly_hdr/dataset/cat.jpg"), axis=0)
x = x/torch.max(x)  # normalize to [0,1], this is realistic

plt.imshow(x, cmap='gray')
plt.colorbar()
plt.title("A 64x64 cat image")
plt.show()
```

We define the measurement operator and the noise model, then use fast transforms to efficiently simulate the acquisition. Note that while the mathematical discussion in this section is focused on 1D vectors $x$, we apply single-pixel imaging in the context of 2D monochromatic images (or 3D hyperspectral cubes when spectra are considered). The Hadamard and split Hadamard operators can be defined in 2D using separability.

We can also visualize the measurements as coefficients in the Hadamard basis, which can be seen as a wavelet transform. This is more meaningful here to display the coefficients corresponding to the split acquisitions, $[H]^+$ and $[H]^-$.

```{code-cell}ipython3
from matplotlib.colors import LogNorm

alpha=1e2
meas_op = HadamSplit2d(64, noise_model=Poisson(alpha))  # this creates the split measurement operator A, and defines the model parameters (noise level)
y = meas_op.forward(x)  # This sampled y~P(\alpha Ax)
y = y/alpha  # rescale to get back the right intensity scale
print(f"We can check that the shape of y, {y.shape[0]}, is 64^2*2=8192, as expected for a split Hadamard of size 64x64, where the row of zero in A is not removed")

y_pos = y[0::2]
y_neg = y[1::2]  # Note: y[1] is zero

f, axs = plt.subplots(1, 2, figsize=(12, 5))
# increase fontsize of the title
axs[0].set_title(r"$\frac{1}{\alpha}H^+x$", fontsize=22)
# In logscale to better see small values
im = axs[0].imshow(y_pos.reshape(64, 64), cmap="gray", norm=LogNorm())
# add a colorbar, that fits the size of the image, and with fixed range
plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
axs[1].set_title(r"$\frac{1}{\alpha}H^-x$", fontsize=22)
im = axs[1].imshow(y_neg.reshape(64, 64), cmap="gray", norm=LogNorm())
plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
# remove axis for the both plots
axs[0].axis('off')
axs[1].axis('off')
plt.show()
```



### Reconstruction

As long as the spectral dimension is ignored and reconstruction, that is, recovering the true image $x$ from the measurements $y$, is computed wavelength by wavelength, there are a variety of well-known available methods that are listed thereafter.

#### Pseudo-inverse reconstruction

The simplest reconstruction algorithm performs a Least Squares (LS) reconstruction,

$$ \hat{X} = \frac{1}{\alpha}A^{\dagger} Y = \argmin{X\in\mathbb{R}^{n\times p}} \|\frac{1}{\alpha}Y - AX \|_F^2. $$


This reconstruction has several advantages.
- It is unbiased.
- It is reasonably cheap to perform as long as the observation matrix $A$ is not too large. Recall that the pseudo-inverse of matrix $A$ is known in closed form, and its contraction with a vector or matrix leverages fast Hadamard transforms.

In practice, the spyrit package that we rely on does not yet implement these fast transforms for the split Hadamard measurements without further processing, so the pseudo-inverse computation is quite slow. The solvers I use are not compatible with fast Hadamard transforms and are also quite slow.

```{code-cell}ipython3
# Reconstruction with pseudo-inverse
x_rec = torch.linalg.lstsq(meas_op.A, y).solution.reshape(64, 64)
print(f"The pseudo-inverse reconstruction has {torch.sum(x_rec<0)} negative entries")
# Bonus: we can use an nnls solver, also very slow without fast operators
from tensorly.solvers.nnls import hals_nnls
Aty = meas_op.adjoint(y)[:,None]
AtA = meas_op.A.T@meas_op.A
x_rec_nnls = hals_nnls(Aty, AtA, n_iter_max=20, tol=0).reshape(64, 64)
```
```{code-cell}ipython3
:tags: [hide-input]
plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1, 2, 2)
plt.imshow(x_rec_nnls, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.title("Reconstructed image using NNLS")
plt.subplot(1, 2, 1)
plt.imshow(x_rec, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.title("Reconstructed image using pseudo-inverse")
plt.show()
```

Note that the pseudo-inverse reconstruction gives essentially the same result as the NNLS solver. The two reconstructions are not the same, however, if we did not have to split the acquisitions and could acquire $Hx$ directly, then they would be the same. Indeed, $H$ is an invertible matrix and therefore $x^\ast=H^Ty$ minimizes the loss $\|y - Hx\|_2^2$. The same solution would also minimize any positive loss $g(x)$ equal to zero at $x^\ast$, which includes the maximum-likelihood reconstruction discussed next. This observation suggests that the choice of reconstruction method may not significantly affect the results, and that the pseudo-inverse is a good compromise among speed, simplicity, and performance.

We can also show the error map for the pseudo-inverse reconstruction, to observe that the residual is not evidently spatially correlated, despite the misfit between the noise model and the reconstruction loss. 

```{code-cell}ipython3
plt.figure(figsize=(8,6))
plt.imshow(torch.abs(x - x_rec), cmap='gray')
plt.colorbar()
plt.axis('off')
plt.title("Error map for pseudo-inverse")
plt.show()
```

This can be explained by the fact that entries of $\alpha Ax$ are large and concentrated around the same value $\frac{1}{2}\alpha \sum_{i\leq n}x[i]$ because, except for the first row of $A$ and the row of zero, all rows have exactly half their values set to 1, and the other half to zero. The measurements are therefore essentially bootstraped mean evaluations of the image. This has two consequences:
- The Poisson distrubtions for each $y_i$ are well approximated by Gaussian distributions of mean $\alpha A[i,:]x$ and variance $\alpha A[i,:]x$.
- The variance is essentially the same for all measurements, except $y[0]$ (corresponding to the row of ones) and $y[1]$ (corresponding to the row of zeroes, and that could be ignored). We denote $\sigma=\frac{1}{2}\alpha \sum_{i\leq n}x[i]$ the empirical variance estimation.
If we consider the model $y \sim \mathcal{N}(\alpha Ax, \sigma)$, which, as discussed above, should correctly approximate the true Poisson distribution, in particular for large $\alpha$, then the pseudo-inverse is the maximum-likelihood estimator of the image. For Gaussian problems, it is known that the residuals are uncorrelated with the image.

```{code-cell}ipython3
:tags: [hide-input]
plt.figure(figsize=(8,6))
plt.hist(y)
plt.title("Histogram of the measurements")
plt.xlabel("Measurement value")
plt.ylabel("Number of occurrences")
plt.show()
```

#### Maximum-Likelihood reconstruction

The Maximum-Likelihood (ML) estimator for Poisson noise is given by the minimizer of the Kullback-Leibler divergence

$$ \hat{X} = \argmin{X\geq 0} \KL{\frac{1}{\alpha} Y, AX}, $$

which can be approximately computed using *e.g.* the [MU algorithm](../../part1/nnls.md#multiplicatives-updates). Below is an example of ML reconstruction.

```{code-cell}ipython3
# Reco avec MU
from tensorly_hdr.nmf_kl import Lee_Seung_KL_regression
# remove the second element in y and the row of zeros to avoid division by zero
yr = torch.cat([y[0:1], y[2:]])
Ar = torch.cat([meas_op.A[0:1,:], meas_op.A[2:,:]], dim=0)
crit, x_mu, time, _ = Lee_Seung_KL_regression(yr, Ar, Hini=x_rec_nnls.reshape(64**2), epsilon=1e-8, NbIter=150, tol=1e-6, verbose=True, print_it=50)
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(x_mu.reshape(64,64), cmap='gray')
plt.colorbar(fraction=0.046, pad=0.04)
plt.title("Reconstructed image using MU NMF KL")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(torch.abs(x - x_mu.reshape(64,64)), cmap='gray')
plt.title("Absolute error")
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
```
Arguably, in this context, the complication induced by using the Poisson ML estimator instead of the pseudo-inverse may not be worth the gain in estimation performance. However, the ML estimator can be easily generalized to measurements that are not Hadamard-based, such as a raster scan that acquires each pixel sequentially. We therefore work with this estimator in the following.

Both the LS and the ML reconstruction are computed in parallel across wavelengths. Therefore, it is expected that incoherent reconstruction artifacts will be found in the reconstructed spectra. The goal of the works presented subsequently is to use known spectra or spectral priors to improve spectral reconstruction.

### Undersampled reconstruction

```{margin}
It is possible to interpret the Hadamard transform as a binary equivalent of the Fourier transform and to associate with each pattern a notion of spatial frequency. In this context, the low-frequency patterns are typically crucial for reconstructing the acquired image, while the high-frequency coefficients can help reconstruct finer details but are also more susceptible to noise.
```

The acquisition of the data matrix $Y$ is performed in parallel along the wavelength dimension, but sequentially along the spatial dimension: Hadamard patterns are loaded onto the DMD one after the other. Each acquisition lasts for a pre-determined time $\Delta t$; the higher $\Delta t$, the more signal is acquired. Therefore, to reduce the total acquisition time, one may either reduce $\Delta t$, which increases the noise level for all acquisitions, or acquire only a subset of the Hadamard patterns. This second choice allows for the acquisition of the most important patterns with a good signal-to-noise ratio. However, the reconstruction using the left pseudo-inverse is no longer available.

One may use the right pseudo-inverse instead of the left one, which is known to amount to solving a problem of the form

$$ \argmin{Y = AX} \|X\|_F^2. $$

Prior information about the image is typically incorporated into the model to ensure identifiability. Classical priors include:
- Nonnegativity
- Sparsity in wavelet bases
- Sparsity in finite differences, also called Total Variation.

Denoting $g(X)$ the negative log-prior, reconstruction in the Maximum *a posteriori* sense is then performed by minimizing

$$ \KL{Y, AX} + \lambda g(X), $$

while the regularized least-squares estimator is obtained by minimizing

$$ \|Y - AX\|_F^2 + \lambda g(X). $$

We can illustrate the reconstruction by right pseudo-inverse and the effect of the nonnegativity prior with an NNLS solver, in which case $g(X)$ is the characteristic function of the nonnegative orthant.

```{code-cell}ipython
# NNLS with subsampling
yr = torch.cat([y[0:1], y[2:]])
Ar = torch.cat([meas_op.A[0:1,:], meas_op.A[2:,:]], dim=0)
# Removing the second half of measurements
yr = yr[:len(Aty)//2]
Ar = Ar[:len(AtA)//2, :]
# Right pseudo-inverse
x_rec_ls_sub = torch.linalg.pinv(Ar)@yr
x_rec_ls_sub = x_rec_ls_sub.reshape(64,64)
# MU initialized with the pseudo-inverse
crit, x_mu_sub, time, _ = Lee_Seung_KL_regression(yr, Ar, Hini=torch.abs(x_rec_ls_sub.reshape(64**2)), epsilon=1e-8, NbIter=40, tol=1e-6, verbose=True, print_it=20)
x_mu_sub = x_mu_sub.reshape(64,64)
```

```{code-cell}ipython3
:tags: [hide-input]
plt.figure(figsize=(8,6))
plt.subplot(1,3,1)
plt.imshow(x_rec_ls_sub, cmap='gray')
plt.title("Pseudo-inverse subsampled")
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(1,3,2)
plt.imshow(x_mu_sub, cmap='gray')
plt.title("MU subsampled")
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(1,3,3)
plt.imshow(torch.abs(x - x_mu_sub), cmap='gray')
plt.title("Absolute error")
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
```

Again, in this setup, nonnegativity alone is essentially useless, since the subsampled pseudo-inverse reconstruction is already nonnegative. This means that the reconstruction error for both the regularized least-squares and the MAP is null at $\hat{X} = A^{\dagger}y$; both reconstructions may provide the exact same estimate. Moreover, we see on the error map that the original image is poorly estimated around the edges. This observation is consistent with the absence of high-frequency content. The image details cannot be reconstructed solely from the measurements. 

## Unmixing and reconstruction with known spectra and non-smooth priors

### Joint reconstruction and unmixing principle

[As discussed in the introduction of this section](./intro.md#spectral-mixing-and-unmixing), when considering hyperspectral images, it is reasonable to assume that the spectrum at each pixel is a linear combination of so-called endmembers, which are the spectra of pure materials present in the scene. When there are $r$ different endmembers and $r$ is smaller than the number of pixels $n$, and the number of spectral bands $p$, a simple but powerful model to represent the hyperspectral image is NMF,

```{margin}
The notations $U$ and $V$ are used to avoid confusion with the Hadamard matrix $H$ and the observation matrix $A$. 
```

$$ X = UV^T, $$

where the nonnegative matrix $V\in\mathbb{R}_+^{p\times r}$ contains the endmembers columnwise, and the nonnegative matrix $U\in\mathbb{R}_+^{n\times r}$ contains the abundance maps, that is, the proportion of each material at each pixel. Because the illumination of the scene is not homogeneous spatially, and spectra may vary depending on other physical parameters such as the geometry of the scene, we refrain from assuming that the abundances sum to one pixel-wise, but we assume that they are bounded by one element-wise.

In preliminary works, we have assumed that the endmembers $V$ are known. The acquisition model then becomes

$$ Y \sim \mathcal{P}\left(\alpha AUV^T\right), $$ (eq:joint-reco-unmixing-model)


and our goal is to design a state-of-the-art reconstruction algorithm that estimates the abundances $U$, therefore performing jointly the reconstruction of the hypercube and its spectral unmixing.

### Joint reconstruction and unmixing in one step

In {cite:p}`harigaJointReconstructionSpectral2024`, a simple strategy to jointly reconstruct and unmix a single-pixel image was proposed, computing the maximum likelihood estimator from Equation {eq}`eq:joint-reco-unmixing-model` with an alternating optimization algorithm, in this case alternating Multiplicative Updates (MU). The update rule for the spectra $V$ is the same as detailled in [](../../part1/nnls.md#multiplicatives-updates), and for $U$ it can be obtained simply by 1) flattening the variables, 2) computing the classical MU rule and 3) folding the output into a matrix (see also {cite:p}`fevotte2011algorithms`). This yields after simplification

$$
 U^{(k+1)} = U^{(k)} \ast \frac{A^T\frac{Y}{AUV^T}V}{A^T\mathbb{1}_{m,p}V}.
$$

The code for the algorithm is slightly too complex to show here and is available in the tensorly_hdr package. A $\ell_1$ regularization is added on both factors, which adds a constant term in the denominator of the update and helps control the dynamics of the factors, see [](../Theory_of_rLRA/HRSI_theory.md).

Since the optimization problem 

$$
\argmin{U\geq 0, V\geq 0} \KL{Y, AUV^T} + \lambda \left( \|U\|_1 + \|V\|_1 \right)
$$

is jointly non-convex, initialization is important. A viable optimization for simple unmixing problems is to first perform the reconstruction, for instance, solving an NNLS problem, then use a pure-pixel selection algorithm such as SNPA {cite}`gillisSuccessiveNonnegativeProjection2014` to initialize $V$.

This procedure is illustrated below on, first, a synthetic exemple, and then on a real dataset acquired in CREATIS.

#### Synthetic experiment

A dataset is generated in which two spectra have disjoint supports and lie on the boundary of the nonnegative orthant. This way, the NMF decomposition may be unique. Moreover, the abundances are designed to include pixels that are almost pure. No noise is added. We can observe that the reconstruction is good, but not perfect; this is mostly due to the non-uniqueness of the NMF model (which would require sparse abundances).

```{code-cell}ipython3

# Separation on synthetic dataset (Eusipco + HCERES)
import numpy as np
from tensorly_hdr.nmf_kl import MU_SinglePixel
from tensorly_hdr.sep_nmf import snpa

W = torch.tensor([[1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1]], dtype=torch.float32).T
W = W/torch.sum(W, dim=0, keepdim=True)  # normalize columns
n = 32
At = torch.rand(2, n**2)
At = At/tl.sum(At,axis=0)
At[0,0]=0.99
At[1,1]=0.99
At[0,1]=0.01
At[1,0]=0.01
# Measurement operator is Ar, noiseless
Xtrue = W@At
alpha=1e2
meas_op = HadamSplit2d(n, noise_model=Poisson(alpha))  # this creates the split measurement operator A, and defines the model parameters (noise level)
Ar = torch.cat([meas_op.A[0:1,:], meas_op.A[2:,:]], dim=0) # TODO: adapt to use forward with wavelengths
Y = torch.poisson(alpha*Xtrue@Ar.T)
#Y = meas_op.forward(Xtrue)  # This sampled y~P(\alpha Ax)
Y = Y/alpha  # rescale to get back the right intensity scale

# Init pinv+spa
#X_rec = torch.linalg.lstsq(Ar, Y.T).solution.T
X_rec = hals_nnls(Ar.T@Y.T, Ar.T@Ar, n_iter_max=10).T
_, W0, A0 = snpa(X_rec, 2)

# Reconstruction with NMF 
W_est, A_est, crit = MU_SinglePixel(Y, Ar, tl.abs(A0), tl.abs(W0), lmbd=0, maxA=None, niter=300, n_iter_inner=20, eps=1e-8, verbose=True, print_it=100)
# Normalization of W_est and A_est
sum_W_est = torch.sum(W_est, dim=0, keepdim=True)
W_est = W_est/sum_W_est
A_est = A_est*sum_W_est.T

```

```{code-cell}ipython3
:tags: [hide-input]
# Plot true W and esimated W and init W0
# adapt these lines to have 2 line plots for each matrix with the same color
plt.figure(figsize=(12,4))
plt.plot(W[:,0].cpu().numpy(), 'r')
plt.plot(W0[:,0].cpu().numpy(), 'b')
plt.plot(W_est[:,0].cpu().numpy(), 'k')
plt.xlabel("Simulated wavelength")
plt.legend(['True W', 'Init W0', 'Estimated W'])
plt.plot(W[:,1].cpu().numpy(), 'r')
plt.plot(W0[:,1].cpu().numpy(), 'b')
plt.plot(W_est[:,1].cpu().numpy(), 'k')
plt.legend(['True W', 'Init W0', 'Estimated W'])
plt.show()
```

#### Real data reconstruction and unmixing

An acquisition was performed in the laboratory, with a cat image layered with two color filters, red and green. There is a thin spatial overlap between the two filters, which induces subtractive mixing; therefore, the hyperspectral image will be rank-two plus nonlinear mixing terms plus noise. However, the abundance maps are mostly disjoint, and the spectral bands where the green and red filters are intense also mostly do not overlap. Therefore, the separation problem is rather simple and amenable to separable NMF algorithms.

We first load the data and the metadata and process them to permute the acquisitions correctly. Then we remove the dark current, which is essentially the mean of the Gaussian noise in the Poisson-Gaussian mixture. We chose to ignore the Gaussian noise and assume that the only source of measurement variation is the Poisson distribution. However, Gaussian noise is not zero-centered, and the induced bias must be corrected. It can be estimated from the measurement acquired with the first row of the negative patterns $H^{-}$, filled with zeroes. The data is then clipped and normalized to lie in the interval $[0,1]$.

```{code-cell}ipython3
# Loading
import json
import ast

# Fetching the spectral measurements
data = np.load("../../tensorly_hdr/dataset/obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1_spectraldata.npz", allow_pickle=True)
Ymeas = data["spectral_data"]

# Fetching metadata
file = open("../../tensorly_hdr/dataset/obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1_metadata.json", "r")
json_metadata = json.load(file)[4]
file.close()

# replace "np.int32(" with an empty string and ")" with an empty string
tmp = json_metadata["patterns"]
tmp = tmp.replace("np.int32(", "").replace(")", "")
patterns = ast.literal_eval(tmp)  # the list (of list of) of pattern indices (evaluation because stored as text)
wavelengths = ast.literal_eval(json_metadata["wavelengths"])

# Permutation of measurements, that are acquired in an experiment-specific order
from spyrit.misc import sampling as samp
img_size = 64
acq_size = img_size
Ord_acq = (-np.array(patterns)[::2] // 2).reshape((acq_size, acq_size))
Ord_rec = torch.ones(img_size, img_size)
Perm_rec = samp.Permutation_Matrix(Ord_rec)
Perm_acq = samp.Permutation_Matrix(Ord_acq).T
Ymeas = samp.reorder(Ymeas, Perm_acq, Perm_rec)


# Post-processing of the measurements
Y = torch.tensor(Ymeas, dtype=torch.float32).T
del Ymeas

# Unbiased by removing the dark current, estimated as the minimum value of marginals of Y (better?)
dc = tl.sum(Y[:,1])/Y.shape[0] # average of dc over all wavelengths
print(f"Estimated dark current to remove: {dc}")
Y = Y - dc
Y = tl.clip(Y, 0, tl.max(Y))
# Normalization to [0,1]
Y = Y / tl.max(Y)

# Showing the measurements after dark current removal
plt.imshow(Y.cpu().numpy(), aspect='auto', cmap='gray')
plt.title("Measurements after dark current removal")
plt.xlabel("Pattern index")
plt.ylabel("Wavelength index")
plt.colorbar()
plt.show()
```

We then follow the reconstruction procedure described earlier. We remove some elements from the data matrix: the measurement with the row of zeros, which, apart from the dark current estimation, is uninformative, and some spectral bands that acquired no signal; here, the first 16 bands. The remaining spectral bands are binned into groups of $8$ to reduce the dimension of the problem due to runtime issues. We first show here the pseudo-inverse reconstruction.

```{code-cell}ipython3
# Reconstruction with pseudo-inverse
from spyrit.core.meas import HadamSplit2d
import spyrit.misc.sampling as samp

# Patterns are acquired in order Acq, compared to order nat
# Patterns are stored in order Rec in spyrit
acq_size = img_size
meas_op = HadamSplit2d(img_size)
A = meas_op.A  # noiseless operator
Anz = torch.cat([A[0:1,:], A[2:,:]], dim=0)
print(Anz.shape)

# Remove row of zero and remove 16 first bands that are null
nbremove = 16
Ynz = torch.cat([Y[nbremove:,0:1], Y[nbremove:,2:]], dim=1)
wavelengths_nz = wavelengths[nbremove:]
print(f"Measurements shape after removing zero rows and first 16 bands: {Ynz.shape}")
# five bands binning
bin_size = 8
Ynz_binned = tl.zeros((Ynz.shape[0]//bin_size, Ynz.shape[1]))
wavelengths_binned = []
for i in range(0, Ynz.shape[0], bin_size):
    if i+bin_size <= Ynz.shape[0]:
        Ynz_binned[i//bin_size,:] = tl.mean(Ynz[i:i+bin_size,:], axis=0)
        wavelengths_binned.append(np.mean(wavelengths_nz[i:i+bin_size]))
    else:
        Ynz_binned[i//bin_size,:] = tl.mean(Ynz[i:,:], axis=0)
        wavelengths_binned.append(np.mean(wavelengths_nz[i:]))
Ynz = Ynz_binned
wavelengths_nz = wavelengths_binned
print(f"Binned measurements shape: {Ynz.shape}")

# define custom forward and adjoint forward functions
def forward(x):
    # x@Anz.T or Anz@x
    # x is of shape (B, N**2)  # batch first
    # reshape to (B, N, N)
    temp = x.reshape((x.shape[0], img_size, img_size))  # Whyyyyyyy >????
    temp = meas_op.forward(temp).T  # shape (M, B)
    # Removing the zero row of A changes the forward A@X
    return torch.cat([temp[0:1,:], temp[2:,:]], dim=0)
def adjoint(y):
    # Also implemented as a contraction in spyrit
    return y@Anz

# Pseudo-inverse reconstruction, the NNLS is quite slow here
X_rec = torch.linalg.lstsq(Anz, Ynz.T).solution.T

# Show cat reconstructed with all wavelengths
plt.figure(figsize=(8,6))
plt.imshow(np.rot90(torch.sum(X_rec, dim=0).reshape(64,64), 2), cmap='gray')
plt.title("Reconstructed image at all wavelengths")
plt.colorbar()
plt.axis('off')

plt.show()
```

We now perform the pure-pixel selection and reconstruct/unmix jointly with an alternating MU algorithm. Observe how the abundance maps are well estimated with both methods, but the spectra are slightly smoother with the alternating MU algorithm. The third component (and in fact, most other components if the rank is increased) models the border of the overlap zone between the filters, which is intense because at the center of the image, and is not well modeled by a linear combination of the two filter spectra.

```{code-cell}ipython3

# Init pinv+spa
rank = 3
Kset, W0, A0 = snpa(X_rec, rank, verbose=True)

# Reconstruction with NMF
from tensorly_hdr.nmf_kl import MU_SinglePixel_fast
lmbd = 1e-4
W_est, A_est, crit = MU_SinglePixel_fast(Ynz, forward, adjoint, tl.abs(A0), tl.abs(W0), lmbd=lmbd, maxA=1, niter=120, n_iter_inner=20, eps=1e-8, verbose=True, print_it=40)  # regularization just for implicit scaling
print("Computation done")
```

```{code-cell}ipython3
:tags: [hide-input]
# Plot the initial and estimated spectra
Anorms = [torch.max(A_est[i,:]) for i in range(rank)]
A0norms = [torch.max(A0[i,:]) for i in range(rank)]

# show hypercube at some wavelengths and some spectra
plt.figure(figsize=(10,10))
for i in range(rank):
    plt.subplot(rank,3,3*i+1)
    plt.imshow(np.rot90(A_est[i,:].reshape(64,64), 2), cmap='gray')
    plt.title(f"Abundance map AMU comp. {i+1}")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.subplot(rank,3,3*i+2)
    plt.plot(wavelengths_nz, A0norms[i]*W0[:,i].cpu().numpy())
    plt.plot(wavelengths_nz, Anorms[i]*W_est[:,i].cpu().numpy())
    plt.title(f"Spectrum comp. {i+1}")
    plt.xlabel("Wavelength index")
    plt.ylabel("Intensity") 
    plt.legend(["pinv+SNPA", "AMU"])
    plt.subplot(rank,3,3*i+3)
    plt.imshow(np.rot90(A0[i,:].reshape(64,64), 2), cmap='gray')
    plt.title(f"Abundance map init comp. {i+1}")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
plt.show()


# Superimpose the four estimated abundance maps in an RBG image with four different colors
n=img_size
abundance_map_rgb = torch.zeros((n, n, 3))
# First component in red
abundance_map_rgb[:,:,0] += A_est[0,:].reshape(n,n)
# Second component in green
abundance_map_rgb[:,:,1] += A_est[1,:].reshape(n,n)
# Third component in blue
abundance_map_rgb[:,:,2] += A_est[2,:].reshape(n,n)
if rank>3:
    # Fourth component in yellow (red+green)
    abundance_map_rgb[:,:,0] += A_est[3,:].reshape(n,n)
    abundance_map_rgb[:,:,1] += A_est[3,:].reshape(n,n)
    if rank > 4:
        # use magenta for the fifth component (red+blue)
        abundance_map_rgb[:,:,0] += A_est[4,:].reshape(n,n)
        abundance_map_rgb[:,:,2] += A_est[4,:].reshape(n,n)
        if rank > 5:
            # use cyan for the sixth component (green+blue)
            abundance_map_rgb[:,:,1] += A_est[5,:].reshape(n,n)
            abundance_map_rgb[:,:,2] += A_est[5,:].reshape(n,n)
# Clip values to [0,1]
abundance_map_rgb = torch.clamp(abundance_map_rgb, 0, 1)
plt.imshow(np.rot90(abundance_map_rgb.cpu().numpy(), k=2))
plt.title("Superimposed estimated abundance maps")
# add color legends
plt.legend(['Component 1 (Red)', 'Component 2 (Green)', 'Component 3 (Blue)', 'Component 4 (Yellow)', 'Component 5 (Magenta)', 'Component 6 (Cyan)'][:rank], loc='upper right')
plt.axis('off')
plt.show()
```
