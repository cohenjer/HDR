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

# Automatic Music Transcription

:::{admonition} Reference
:class: tip
{cite}`wuSemiSupervisedConvolutiveNMF2022` TODO
:::

## Nonnegative matrix factorization for automatic music transcription

Spectrogramms of piano recordings contain the spectral information of notes played in the recording along time.  An interesting property of the spectrogram of a single piano note recording is that it is typically well approximated by a rank-one matrix. For instance on a recorded piano A4 (440Hz) from the dataset MAPS [ref], the best rank-one approximation of the magnitude spectrogram looks similar to the magnitude spectrogram, in particular when using Kullback-Leibler divergence as a loss function [ref rank one, ref Axel].

```{code-cell} ipython3
from cmath import phase
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import scipy.signal as signal

A4_wav, sr = sf.read('../../tensorly_hdr/dataset/MAPS_ISOL_LG_M_S0_M69_ENSTDkAm.wav')
A4_wav = A4_wav / np.max(np.abs(A4_wav)) # audio normalization
A4_wav = A4_wav[:,0] # use only one channel if stereo

# Compute spectrogram
frequencies, times, Sxx = signal.stft(A4_wav, fs=sr, nperseg=4096, nfft=4096, noverlap=4096 - 882)
half_freqs = len(frequencies) // 2
frequencies = frequencies[:half_freqs]
Sxx = Sxx[:half_freqs, :] # keep only lower half
Sphase = np.angle(Sxx)
Sxx_dB = 20 * np.log10(np.abs(Sxx) + 1e-10)  # convert to dB scale
Sxx_abs = np.abs(Sxx)

# Best rank-one approximation of magnitude spectrogram in Frobenius sense
U,s,V = np.linalg.svd(Sxx_abs)
r1_approx = s[0]*np.outer(U[:,0],V[0,:])
# in dB
r1_approx_dB = 20 * np.log10(r1_approx + 1e-10)
# Same in KL sense with NMF
ukl = np.sum(Sxx_abs, axis=1)/np.sqrt(np.sum(Sxx_abs))
vkl = np.sum(Sxx_abs, axis=0)/np.sqrt(np.sum(Sxx_abs))
r1_approx_kl = np.outer(ukl, vkl)

r1_approx_kl_dB = 20 * np.log10(r1_approx_kl + 1e-10)

```

```{code-cell} ipython3
:tags: [hide-input]

# Plotting rank-one approximation
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(r1_approx_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
plt.title('Rank-One Approximation')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
plt.subplot(1, 3, 2)
plt.imshow(r1_approx_kl_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
plt.title('Rank-One Approximation KL')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
# colorbar with the same min and max as previous
plt.clim(np.min(r1_approx_dB), np.max(r1_approx_dB))
plt.colorbar(label='Intensity [dB]')
plt.subplot(1, 3, 3)
plt.imshow(Sxx_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
plt.title('Original Magnitude Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()

```
Magnitude spectrogramms are rather well approximated by rank-one matrices. This suggests the use of low-rank NMF to analyse recordings containing multiple notes. Assuming a linear additive mixture of time-signals however does not lead to a low-rank model exactly. In practice this approximation is reasonable and we shall make use of it, see [encadré below] for more details.

### Simplified model and low-rank spectrograms
Assume two notes are played simultaneously by the piano, with time signals $s_1(t)$ and $s_2(t)$. We focus on a single time window in the STFT. We assume linear mixing, and record the sum of the two signals $s(t) = s_1(t) + s_2(t)$. The Fourier transform of the summed signal $\mathcal{F}[s](\nu)$, by linearity of the Fourier transform, is exactly $ \mathcal{F}[s_1](\nu) + \mathcal{F}[s_2](\nu) $.

As we saw in [](./intro.md), the phase of the spectrogram is not low-rank even for a single note, therefore we only want to compute the magnitude spectrogram $\left| \mathcal{F}[s] (\nu)\right| $, where the modulus is taken elementwise. However if we assume that $\left|\mathcal{F}[s_{i}](\nu)\right|$ are rank-one matrices, we find that 

$$ \left| \mathcal{F}[s](\nu) \right|^2  = \left| \mathcal{F}[s_1](\nu) + \mathcal{F}[s_2](\nu) \right|^2 = \left|\mathcal{F}[s_{1}](\nu)\right|^2 + \left|\mathcal{F}[s_{2}](\nu)\right|^2 + 2\mathcal{Re}\left(\mathcal{F}[s_1]^\ast(\nu) \mathcal{F}[s_2](\nu)\right), $$

and by Cauchy-Schwartz, the spectrogram computed from the recording is rank-two if and only if $ \mathcal{F}[s_1](\nu) = \lambda_\nu \mathcal{F}[s_2](\nu) $ for each $\nu$ and $\lambda_nu\geq 0$. This condition is equivalent to assuming that the two signals are in phase, in other words $\mathcal{Im}\left(\mathcal{F}[s_1]^\ast(\nu) \mathcal{F}[s_2](\nu)\right)=0$.

A particular case of the equality condition in Cauchy-Schwartz is obtained when the two frequency spectra have disjoint support, in which case $\lambda_\nu$ is null. In other words, if the notes $s_1(t)$ and $s_2(t)$ have no partials in common, then the spectrogram of their additive mixture in the temporal domain should remain low-rank, and one can hope to recover each rank-one spectrogram by performing a rank-two NMF. This assumption is related to time-frequency masking, a technique that has been used for decades in audio source separation and remained used at the start of the era of deep learning {cite}`araki30YearsSource2025`. 

The disjoint support hypothesis is useful, but incorrect in practice. For instance two notes separated by an octave have similar spectra, but the colinearity condition will be significantly violated. Not only are the fundamental frequencies and partials different in amplitude, the phase of each signal can also change. This can be observed on the MAPS dataset below with notes A4 and A3. Observe that the imaginary part of the cross product is in particular nonzero at locations where the sum of the modulus is far from the modulus of the sum, and that this corresponds to partials of interest. This implies that methods based on low-rank approximations of spectrograms are particularly prone to octave errors.

```{code-cell} ipython3
:tags: [hide-input]

from cmath import phase
from re import A
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import scipy.signal as signal

A3_wav, sr = sf.read('../../tensorly_hdr/dataset/MAPS_ISOL_LG_F_S1_M57_ENSTDkAm.wav')
A4_wav, sr = sf.read('../../tensorly_hdr/dataset/MAPS_ISOL_LG_M_S0_M69_ENSTDkAm.wav')
A4_wav = A4_wav / np.max(np.abs(A4_wav)) # audio normalization
A4_wav = A4_wav[:,0] # use only one channel if stereo
A3_wav = A3_wav / np.max(np.abs(A3_wav)) # audio normalization
A3_wav = A3_wav[:,0] # use only one channel if stereo
# Use short time window (rectangular window for simplicity)
t_1 = 2*sr
t_2 = 3*sr
A4_segment = A4_wav[t_1:t_2]
A3_segment = A3_wav[t_1:t_2]

# Compute 1D Fourier transform
frequencies_1D = np.fft.fftfreq(len(A4_segment), d=1/sr)
fourier_1D_A4 = np.fft.fft(A4_segment)
fourier_1D_A3 = np.fft.fft(A3_segment)
magnitude_1D_A4 = np.abs(fourier_1D_A4)
phase_1D_A4 = np.angle(fourier_1D_A4)
magnitude_1D_A3 = np.abs(fourier_1D_A3)
phase_1D_A3 = np.angle(fourier_1D_A3)

# Compute sum of magnitude and magnitude of sum
fourier_sum = fourier_1D_A4 + fourier_1D_A3
magnitude_sum = np.abs(fourier_sum)
fourier_magnitude_sum = magnitude_1D_A4 + magnitude_1D_A3
# Plot results
# use same y-axis limits for better comparison
ymin = min(np.min(magnitude_1D_A4), np.min(magnitude_1D_A3))
ymax = max(np.max(magnitude_1D_A4), np.max(magnitude_1D_A3))
plt.figure(figsize=(9, 3))
plt.subplot(1, 4, 1)
plt.title('Magnitude Spectrum of A4')
plt.plot(frequencies_1D, magnitude_1D_A4, color='blue')
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.ylim(ymin, ymax)
plt.subplot(1, 4, 2)
plt.title('Magnitude Spectrum of A3')
plt.plot(frequencies_1D, magnitude_1D_A3, color='orange')
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.ylim(ymin, ymax)
plt.subplot(1, 4, 3)
plt.title('Residual error')
plt.plot(frequencies_1D, np.abs(magnitude_sum - fourier_magnitude_sum), color='green')
#plt.plot(frequencies_1D, fourier_magnitude_sum, label='Sum of Magnitudes', color='red', linestyle='--')
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.ylim(ymin, ymax)

# The residual is due to the phase differences between the two signals
residual = np.conjugate(fourier_1D_A4)*fourier_1D_A3
im_res = np.imag(residual)
plt.subplot(1, 4, 4)
plt.title('Imaginary Part of Residual')
plt.plot(frequencies_1D, im_res, color='brown')
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Phase of each signal
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.title('Phase Spectrum of A4')
plt.plot(frequencies_1D, phase_1D_A4, color='blue')
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.subplot(1, 2, 2)
plt.title('Phase Spectrum of A3')
plt.plot(frequencies_1D, phase_1D_A3, color='orange')
plt.xlim(0, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.tight_layout()
plt.show()
```

```{Note}
AMT entered the [MIREX competition](https://www.music-ir.org/mirex/wiki/MIREX_HOME) as polyphonic transcription in 2024 (in fact it was just piano, not guitar, organs, synthetisers or others), even though it was already a research topic in early 2000s. Single-instrument transcription is still an open problem, and there is still much to do in multi-instrument setting, performing source-separation jointly with AMT. 
```

### NMF for AMT: useful but limited

```{margin}
Spectrograms have a few hyperparameters that should be fixed, most importantly the time window size and type, and the amount of overlap for the windows. In the following, we use parameters found in the litterature [ref AD], hann windows, 4096 samples per window and 4096 - 882 overlapping samples. This leads to a time resolution of about 20ms (the sample rate is usually 44kHz) with spectra computed over 93ms, and a frequency resolution of about 10.7Hz which is a reasonable time-frequency resolution trade-off for piano transcription.
```

Given a spectrogram $Y$ of an audio recording, based on the observation that individual notes have approximately rank-one spectrograms, and assuming both linear mixture and marginally overlapping support of the spectrograms, one may hope to recover each individual note spectrogram with NMF. In fact if the supports of the spectrograms of each note do not overlap, NMF is separable (each row of the data spectrogram belongs to exactly one note specrogram) and therefore the unique NMF solution can be recovered by any reasonable NMF algorithm. On top of estimating each individual spectrogram, these spectrograms are factorized into a temporal activation, telling when each note is played and at which intensity, and a frequency spectrum containing the fundamental and partials characteristic of the note. A simple post-processing step thus involves identifying the fundamental frequency for each rank-one NMF component, and thresholding the time-activation {cite}`vincentHarmonicInharmonicNonnegative2008a`. Although more elaborate strategies could be designed, this pipeline is essentially what has been used in the AMT literature to post-process NMF factors.

```{figure} ../../Figures/partition.png
---
width: 900px
align: center
name: partition
---
Symbolic notation for the recorded audio sample. In green, the first six seconds with isolated notes, and in red the second part up to eight seconds with chords.
```

We can try this on a simple recording from the MAPS dataset {cite}`emiyaMAPSAPianoDatabase2010` with many isolated notes and few chords. A first example below is performed on a song with only isolated notes. Notice how good the reconstruction is. There are six notes played in the first six seconds of the recording. We used a rank seven NMF. One of the component models the hammer action; this can be seen by the spectral signature which is not comb-shaped, and the time activation following each played note except the high F which has a softer attack. 

```{code-cell} ipython3
# Load song
song, sr = sf.read('../../tensorly_hdr/dataset/MAPS_MUS-muss_1_ENSTDkAm.wav')
song = song / np.max(np.abs(song)) # audio normalization
song = song[:,0] # use only one channel if stereo
# The song has six different single notes until 6s, then chords
# We cut the song after 6s (rectangular window for simplicity)
t_1 = 0*sr
t_2 = 6*sr
song = song[t_1:t_2]

# Compute spectrogram
frequencies, times, Sxx = signal.stft(song, fs=sr, nperseg=4096, nfft=4096, noverlap=4096 - 882)
half_freqs = len(frequencies) // 2
frequencies = frequencies[:half_freqs]
Sxx = Sxx[:half_freqs, :] # keep only lower half
Sxx_dB = 20 * np.log10(np.abs(Sxx) + 1e-10)  # convert to dB scale
Sxx_abs = np.abs(Sxx)

# NMF with KL divergence (function taken from paper with Quyen, using alternating MU, code from tensorly_hdr)
# Could use also spa or snpa on the rows
from tensorly_hdr.nmf_kl import Lee_Seung_KL
rank = 7
W_init = np.abs(np.random.randn(Sxx.shape[0], rank))
H_init = np.abs(np.random.randn(Sxx.shape[1], rank)).T
crit, W_kl, H_kl, toc, cnt = Lee_Seung_KL(Sxx_abs, W_init, H_init, NbIter=20, verbose=False, print_it=20)
# Normalize W columnwise and put the norm in H
W_kl_norms = np.max(W_kl, axis=0)
W_kl = W_kl / W_kl_norms
H_kl = H_kl.T * W_kl_norms

# Permuting spectra in increasing fundamental frequency
from tensorly_hdr.image_utils import permute_spectra
perm = permute_spectra(W_kl)
W_kl = W_kl[:, perm]
H_kl = H_kl[:, perm]

```

```{margin}

The components are ordered by increasing fundamental frequency. The notes played in the recording are, in order, G3 F3 Bb3 C4 F4 D4 C4 F4 D4 Bb3 C4 G3 F3. We can see that the components match that order.

```

```{code-cell} ipython3
:tags: [hide-input]

notes = ["F3", "G3", "Bb3", "Hammer", "C4", "D4", "F4"]
# Plots
plt.figure(figsize=(10, 6))
for i in range(rank):
    plt.subplot(rank, 2, 2*i + 1)
    plt.plot(frequencies, W_kl[:, i])
    if i==0:
        plt.title(f'KL Components Frequency Profile')
    if i==rank-1:
        plt.xlabel('Frequency (Hz)')
    
    plt.subplot(rank, 2, 2*i + 2)
    plt.plot(times, H_kl[:, i])
    plt.ylabel(notes[i])
    plt.ylim(0, np.max(H_kl))
    if i==0:
        plt.title(f'KL Components Time Activation')
    if i==rank-1:
        plt.xlabel('Time (s)')

# showing the KL rank-1 approximations in dB
plt.figure(figsize=(10, 12))
# Original recording spectrogram
plt.subplot(rank//2+1, 2, 1)
plt.imshow(Sxx_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
plt.title('Original Spectrogram (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
for i in range(rank):
    plt.subplot(rank//2+1, 2, i + 2)
    rank_1_kl_approx = 20*np.log10(np.outer(W_kl[:, i], H_kl[:, i]) + 1e-10)
    plt.imshow(rank_1_kl_approx, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
    # all use same colomap limits for better comparison
    plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
    plt.title(f'KL Rank-1 Approximation {notes[i]} (dB)')
    plt.ylabel('Frequency (Hz)')
    if i==rank-1:
        plt.xlabel('Time (s)')
    plt.colorbar(label='Intensity [dB]')
plt.tight_layout()        

plt.show()
```

While this is promising, we knew here the correct number of notes in the recording. Also the rank-one spectrograms are not well located in time when analyzed in logarithmic scale. For instance the spetrogram of C4 has small but nonzero values when all the other notes are played. Adding the next few seconds make things much harder to analyse. The audio contains two chords, with five notes each, for a total of 14 different notes played. We see in the experiment below that NMF-KL does not identify all the notes, and rather introduces components to refine its approximation of the notes already identified.


```{code-cell} ipython3
:tags: [hide-input]

# Load song
song, sr = sf.read('../../tensorly_hdr/dataset/MAPS_MUS-muss_1_ENSTDkAm.wav')
song = song / np.max(np.abs(song)) # audio normalization
song = song[:,0] # use only one channel if stereo
# Use short time window (rectangular window for simplicity)
t_1 = 0*sr
t_2 = 8*sr
song = song[t_1:t_2]
# The song has six different single notes until 6s, then chords

# Compute spectrogram
frequencies, times, Sxx = signal.stft(song, fs=sr, nperseg=4096, nfft=4096, noverlap=4096 - 882) #8192
half_freqs = len(frequencies) // 2
frequencies = frequencies[:half_freqs]
Sxx = Sxx[:half_freqs, :] # keep only lower half
Sxx_dB = 20 * np.log10(np.abs(Sxx) + 1e-10)  # convert to dB scale
Sxx_abs = np.abs(Sxx)

# NMF with KL divergence (function taken from paper with Quyen, using alternating MU, code from tensorly_hdr)
# Could use also spa or snpa on the rows
from tensorly_hdr.nmf_kl import Lee_Seung_KL
rank = 15
W_init = np.abs(np.random.randn(Sxx.shape[0], rank))
H_init = np.abs(np.random.randn(Sxx.shape[1], rank)).T
crit, W_kl, H_kl, toc, cnt = Lee_Seung_KL(Sxx_abs, W_init, H_init, NbIter=20, verbose=False, print_it=20)
# Normalize W columnwise and put the norm in H
W_kl_norms = np.max(W_kl, axis=0)
W_kl = W_kl / W_kl_norms
H_kl = H_kl.T * W_kl_norms

# Permuting spectra in increasing fundamental frequency
perm = permute_spectra(W_kl)
W_kl = W_kl[:, perm]
H_kl = H_kl[:, perm]

# Plots
plt.figure(figsize=(10, 6))
for i in range(rank):
    plt.subplot(rank, 2, 2*i + 1)
    plt.plot(frequencies, W_kl[:, i])
    if i==0:
        plt.title(f'KL Components Frequency Profile')
    if i==rank-1:
        plt.xlabel('Frequency (Hz)')
    
    plt.subplot(rank, 2, 2*i + 2)
    plt.plot(times, H_kl[:, i])
    plt.ylim(0, np.max(H_kl))
    if i==0:
        plt.title(f'KL Components Time Activation')
    if i==rank-1:
        plt.xlabel('Time (s)')

plt.show()
```

This suggests a few problems with NMF:
- The rank-one hypothesis for isolated notes is not good enough. Therefore the model tends to use several components to model a single note. This makes the choice of the rank harder, and the post-processing also more challenging.
- We have little guarantee that the output spectra are indeed interpretable as comb-like spectra with a clear fundamental frequency. We clearly see in the above example that the components with smallest fundamental frequency are hardly interpretable. Their time activation is also far from the standard attack-hold-sustain-decay pattern. In other words, the uniqueness of NMF is unclear as soon as many notes overlap in time and frequency, which is usually the case with chords in tonal music.

Tentative improvements to blind NMF for AMT involve parameterizing the time activations to actually follow and attack-decay pattern {cite}`chengAttackDecayModel2016`. We used a different route and utilized the so-called Convolutive NMF with supervision.

## Convolutive pretrained NMF for improved piano transcriptions

In {cite}`wuSemiSupervisedConvolutiveNMF2022`, we proposed to address both issues: the modeling errors of spectrograms of single notes are reduced by introducing convolutive rank-one components, while the interpretability is ensured by pretraining the dictionary $W$ on isolated notes recordings, thus switching from an unsupervised setup to a (weakly)-supervised framework. Let us first discuss convolutive NMF.

### Convolutive NMF, a better model for notes spectrograms

As mentionned in [](./intro.md), magnitude spectrograms of single piano notes are not exactly rank-one matrices. While it is rather accurate that a single activation vector can describe the dynamics of the note along time, the description of a single note as a fixed frequency spectrum is wrong. The spectrum at the attack contains transients that rapidly dissipate. It therefore may differ significantly from the spectrum measured when the note is sustained. Ideally to obtain a linear model we could split attack and decay/sustain spectra as proposed in {cite}`chengAttackDecayModel2016`. We proposed rather to model each single note spectrogram $Y[f,t]$ as the convolution of a small time-frequency matrix with a time activation:

$$
 \hat{Y}[f,t] = \sum_{\tau=0}^{T-1} W[f,\tau] h[t+\tau],
$$ (eq:rank1CNMF)

where $T$ is the size of the convolution window. This rank-one convolutive model can also be seen as a constrained rank $T$ NMF if we consider the Toeplitz matrix $\tilde{H}$ obtained by stacking rows of $h$ shifted by one:

$$ \hat{Y} = W \tilde{H}. $$

Convolutive NMF is obtained by modeling a nonnegative matrix as the sum of rank-one convolutive terms as defined in {eq}`eq:rank1CNMF`, that now depend on the component index $q$. It was proposed originally in the context of speech separation {cite}`smaragdisConvolutiveSpeechBases2006`, and writes

$$ \hat{Y} = \sum_{q=1}^{r} \sum_{\tau=0}^{T-1} W[f,\tau,q] H[q,t-\tau]. $$

The figure below compares NMF and CNMF on a visual example.

```{figure} ../../Figures/CNMFvsNMF.png
---
width: 900px
align: center
name: CNMF
---
Illustration of the CNMF model compared to NMF.
```

Computing CNMF is typically done in an alternating fashion. Here we do not really need the full update for the tensor $W$ in CNMF since we will only work with rank-one CNMF at training time, see [](#a-cnmf-dictionary-of-pure-notes-with-rank-one-cnmf). The update for $H$ provided in the original CNMF paper is a heuristic, but it was later refined using majorization-minimization techniques {cite}`fagotMajorizationminimizationAlgorithmsConvolutive2019b` similar to what we describe in [](../../part1/nnls.md). In terms of optimization, multiplicative update rules for CNMF are obtained essentially with the same majorization techniques as introduced in [](../../part1/nnls.md). Indeed, the MU algorithm relies on a majorization that removes the dependence of each variable with the others, and CNMF in that aspect has the same structure as NMF. The updates for tensor $W^{(k)}$ and matrix $H^{(k)}$ and iteration $k$ can be written in matrix format as follows (following the MM2 algorithm from {cite}`fagotMajorizationminimizationAlgorithmsConvolutive2019b`):

$$ W^{(k+1)} = W^{(k)}\ast \frac{\frac{Y}{\hat{X}^{(k)}}{\tilde{H}^{(k)}}^T }{\mathbf{1}\otimes\sum_{t} \tilde{H}^{(k)}[:,t]} \text{ and }  H^{(k+1)}[:,t] = H^{(k)}[:,t] \ast \frac{  W^{(k)} \times_{1,2} \frac{Y}{\hat{Y}}[:,t:t+T]}{\sum_{f,\tau} W^{(k)}[f,\tau,:] \mathbf{1}_{t+\tau\leq n}},   $$

where $\hat{Y}\in\mathbb{R}^{m\times n}$ is the reconstructed dataset, matrix $\tilde{H}$ is computed from matrix $H$ by shifting each row $T$ times and concatenating the results, resulting in a block-Toeplitz matrix. Recall that $\times_{1,2}$ denotes the tensor contraction on modes 1 and 2, and here corresponds to the broadcasted inner product of slices of $\frac{Y}{\hat{Y}}$ with each slice of tensor $W$ along the third mode. Zero-padding can be considered to handle the border of the input spectrogram. 

Similarly to NMF, CNMF is a non-convex problem. Personal observations lead me to believe that CNMF is significantly harder however, with many non-trivial local minima, and therefore initialization is crucial in CNMF. 

### A CNMF dictionary of pure notes with rank-one CNMF

Armed with the iterative algorithm we just described, we can decompose the spectrogram of a single note, say A4, with rank-one CNMF. The template tensor $W$, which has a single slice for rank-one CNMF and is therefore essentially a matrix, is initialized by copying the slice of the input spectrogram $Y[:,t_0-3:t_0-3+T]$ where $t_0$ is the index of the maximum column in $ell_1$ norm, and $T$ is set to 10. This corresponds to selecting the most intense 0.2 seconds of the input spectrogram. This way we ensure that the CNMF template is interpretable as a part of the spectrogram of A4. Random initialization may lead to smaller loss function at convergence but results are harder to interpret. Separable CNMF has been studied in the litterature [ref nicolas].

```{code-cell} ipython3
A4_wav, sr = sf.read('../../tensorly_hdr/dataset/MAPS_ISOL_LG_M_S0_M69_ENSTDkAm.wav')
A4_wav = A4_wav / np.max(np.abs(A4_wav)) # audio normalization
A4_wav = A4_wav[:,0] # use only one channel if stereo
# Work on the first few seconds to avoid learning the recording noise around 10s
A4_wav = A4_wav[sr//3:sr*4]
# Compute spectrogram up to 10kHz
frequencies, times, Sxx = signal.stft(A4_wav, fs=sr, nperseg=4096, nfft=4096, noverlap=4096 - 882) #8192
# Cut for speed
half_freqs = len(frequencies) // 2
Sxx_dB = 20 * np.log10(np.abs(Sxx[:half_freqs,:]))  # convert to dB scale for display
Sxx_abs = np.abs(Sxx[:half_freqs,:])
# normalization
Sxx_abs = Sxx_abs / np.max(Sxx_abs)

# Testing rank-1 CNMF
T = 20
from tensorly_hdr.cnmf import convolutive_nmf
W_r1, H_r1, err_r1 = convolutive_nmf(Sxx_abs, rank=1, T=T, itmax=10, eps=1e-16, tol=0, print_it=1, n_iter_inner=10, init="separable", verbose=False)
W_cnmf=W_r1[:, :, 0]
h_cnmf = H_r1[0, :]
```

```{code-cell} ipython3
:tags: [hide-input]

from tensorly_hdr.cnmf import shift_m
import matplotlib.pyplot as plt
# normalization
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
# plot W as a spectrogram (tall)
W_cnmf_db = 20 * np.log10(W_cnmf)
plt.imshow(W_cnmf_db, aspect='auto', interpolation='none', origin='lower', extent=[0, T, frequencies[0], frequencies[half_freqs]])
plt.colorbar(label='Intensity [dB]')
plt.title('CNMF template')
plt.xlabel('Convolution Index')
plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
plt.ylabel('Frequency (Hz)')
plt.subplot(2, 2, 2)
plt.plot(h_cnmf)
plt.title('Activation component')
plt.xlabel('Time Frames')
# Reconstructed rank-1 approximation
Sxx_cnmf_approx = W_cnmf @ shift_m(h_cnmf, T)
Sxx_cnmf_approx_dB = 20 * np.log10(Sxx_cnmf_approx)
plt.subplot(2, 2, 3)
plt.imshow(Sxx_cnmf_approx_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[half_freqs]], interpolation='none')
plt.title('Reconstructed A4 (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
# same limit as data spectrograms for better comparison
plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
plt.subplot(2, 2, 4)
# data spectrogram
plt.imshow(Sxx_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[half_freqs]], interpolation='none')
plt.title('Recorded A4 (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()

```
[TODO: move spectrograms of rank1 notes here, c'est trop pour l'intro]

Observe that the time activation profile is much sparse than with the best rank-one NMF approximation, which is expected since the template now contains short-time information. The template however has some issues. It captures correctly the attack spectrum with a comb-shaped spectrum and many harmonics, but the high-pitch harmonics should decrease in intensity. The content of the template after the attack above 4kHz is therefore not realistic.


This problem appears probably because the rank-one model is modeling echoes observed in the signal. To alleviate this issue, one can compute a rank-two CNMF with the second template initialized randomly. Observe how this second component captures noise and high-frequency decay spectra, effectively cleaning the first component. In particular the second component does not model the attack at all. For simplicity and computational speed reasons, we will work with rank-one CNMF approximations of single notes, as done in regular NMF.

```{code-cell} ipython3
W_r2, H_r2, err_r2 = convolutive_nmf(Sxx_abs, rank=2, T=T, itmax=10, eps=1e-16, tol=0, print_it=1, n_iter_inner=10, init="separable", verbose=False)
```

```{code-cell} ipython3
:tags: [hide-input]

# Same code for rank-2 CNMF
plt.figure(figsize=(10, 6))
plt.subplot(3, 3, 1)
W_r2_db = 20 * np.log10(W_r2[:,:,0])
plt.imshow(W_r2_db, aspect='auto', interpolation='none', origin='lower', extent=[0, T, frequencies[0], frequencies[half_freqs]])
plt.colorbar(label='Intensity [dB]')
plt.title('Template 1')
plt.xlabel('Convolution Index')
plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
plt.ylabel('Frequency (Hz)')
# Plot activations
plt.subplot(3, 3, 2)
plt.plot(H_r2[0,:])
plt.title('Activation 1')
plt.xlabel('Time Frames')
# Show second component as well on the same figure
# Rank-one spectrogram
Sxx_r1_comp =  W_r2[:,:,0] @ shift_m(H_r2[0,:], T)
Sxx_r1_comp_dB = 20 * np.log10(Sxx_r1_comp)
plt.subplot(3, 3, 3)
plt.imshow(Sxx_r1_comp_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[half_freqs]], interpolation='none')
plt.title('Component 1 (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
# Template 2
plt.subplot(3, 3, 4)
W_r2_db_2 = 20 * np.log10(W_r2[:,:,1])
plt.imshow(W_r2_db_2, aspect='auto', interpolation='none', origin='lower', extent=[0, T, frequencies[0], frequencies[half_freqs]])
plt.colorbar(label='Intensity [dB]')
plt.xlabel('Convolution Index')
plt.title('Template 2')
plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
plt.ylabel('Frequency (Hz)')
plt.subplot(3, 3, 5)
plt.plot(H_r2[1,:])
plt.title('Activation 2')
plt.xlabel('Time Frames')
# Rank-two spectrogram
Sxx_r2_comp =  W_r2[:,:,1] @ shift_m(H_r2[1,:], T)
Sxx_r2_comp_dB = 20 * np.log10(Sxx_r2_comp)
plt.subplot(3, 3, 6)
plt.imshow(Sxx_r2_comp_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[half_freqs]], interpolation='none')
plt.title('Component 2 (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
# Reconstructed rank-2 approximation
Sxx_r2_approx = W_r2[:,:,0] @ shift_m(H_r2[0,:], T) + W_r2[:,:,1] @ shift_m(H_r2[1,:], T)
Sxx_r2_approx_dB = 20 * np.log10(Sxx_r2_approx)
plt.subplot(3, 3, 7)
plt.imshow(Sxx_r2_approx_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[half_freqs]], interpolation='none')
plt.title('Reconstructed A4 (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
# same limit as data spectrograms for better comparison
plt.clim(np.min(Sxx_dB), np.max(Sxx_dB))
plt.subplot(3, 3, 8)
# data spectrogram
plt.imshow(Sxx_dB, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[half_freqs]], interpolation='none')  
plt.title('Recorded A4 (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()

```

### Transcription example on MAPS

In this document we simply provide a toy example of AMT with pretrained templates. Our goal is to transcribe the same audio recording as above with isolated notes and two chords. We know the notes utilized in the recording, and MAPS has all single notes recorded individually. Therefore we can perform rank-one CNMF on each note recording, and then transcribe the recorded music. 

```{code-cell} ipython
import os
import re
import numpy as np
import soundfile as sf
from scipy import signal
from tensorly_hdr.cnmf import convolutive_nmf

# Directory with your .wav files
data_dir = '../../tensorly_hdr/dataset/notes_to_learn'
T = 10
rank = len(os.listdir(data_dir))  # 14

# Templates list
Wlist = []
names = []
display = ""

# Learning the templates
# perform the code below but with every file in ../../tensorly_hdr/dataset/notes_to_learn
for filename in os.listdir(data_dir):
    midi_num = re.search(r'M(\d+)_', filename).group(1)     
    display += f"Processing MIDI note: {midi_num} \n"
    names.append(midi_num)
    filepath = os.path.join(data_dir, filename)
    A4_wav, sr = sf.read(filepath)
    A4_wav = A4_wav / np.max(np.abs(A4_wav)) # audio normalization
    A4_wav = A4_wav[:,0] # use only one channel if stereo
    # Work on the first few seconds to avoid learning the recording noise around 10s
    A4_wav = A4_wav[0:sr*4]

    # Compute spectrogram up to 10kHz
    frequencies, times, Sxx = signal.stft(A4_wav, fs=sr, nperseg=4096, nfft=4096, noverlap=4096 - 882) #8192
    # Cutting last few time frames (40ms) because there are artifacts (probably from the windowing above)
    Sxx = Sxx[:,:-2]
    times = times[:-2]
    half_freqs = len(frequencies) // 2
    Sxx_dB = 20 * np.log10(np.abs(Sxx[:half_freqs,:]))  # convert to dB scale
    Sxx_abs = np.abs(Sxx[:half_freqs,:])
    # normalization
    Sxx_abs = Sxx_abs / np.max(Sxx_abs)

    # rank-1 CNMF
    # Only init
    W_r1, H_r1, err_r1 = convolutive_nmf(Sxx_abs, rank=1, T=T, itmax=10, eps=1e-16, tol=0, print_it=10, n_iter_inner=10, init="separable", verbose=False)
    Wlist.append(W_r1[:, :, 0])
    h_cnmf = H_r1[0, :]
    display += f"Reconstruction error: {err_r1[-1]} \n"
print(display)

# Post-process Wlist into a tensor
W = np.zeros((Wlist[0].shape[0], Wlist[0].shape[1], rank))
for r in range(rank):
    W[:, :, r] = Wlist[r]
    
# Permute by increasing MIDI name
perm = np.argsort(names)
names = np.sort(names)
W = W[:,:,perm]
names_music = ["G2", "A2", "F3", "G3", "A3", "Bb3", "C4", "D4", "F4", "G4", "Bb4", "C5", "D5", "F5"]
```

Now that the training is performed, we can transcribe the audio recording!

```{code-cell} ipython3
:tags: [hide-input]
# Load song
song, sr = sf.read('../../tensorly_hdr/dataset/MAPS_MUS-muss_1_ENSTDkAm.wav')
song = song / np.max(np.abs(song)) # audio normalization
song = song[:,0] # use only one channel if stereo
# Use short time window (rectangular window for simplicity)
t_1 = 0*sr
t_2 = 8*sr
song = song[t_1:t_2]
# The song has six different single notes until 6s, then two chords with five notes each
# There are 14 different notes. Hammer noise should be in the TF templates.
rank = 14

# Compute spectrogram
frequencies, times, Sxx = signal.stft(song, fs=sr, nperseg=4096, nfft=4096, noverlap=4096 - 882) #8192
# Cutting last few time frames (40ms) because there are artifacts (probably from the windowing above)
Sxx = Sxx[:,:-2]
times = times[:-2]
half_freqs = len(frequencies) // 2
frequencies = frequencies[:half_freqs]
Sxx = Sxx[:half_freqs, :] # keep only lower half
Sxx_dB = 20 * np.log10(np.abs(Sxx) + 1e-10)  # convert to dB scale
Sxx_abs = np.abs(Sxx)
```
```{code-cell} ipython3
from tensorly_hdr.cnmf import convolutive_regression
H_tr, err = convolutive_regression(Sxx_abs, W, itmax=50, verbose=True, print_it=25)
```

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import gridspec

fig = plt.figure(figsize=(8, 12), constrained_layout=True)
gs = gridspec.GridSpec(rank, 3, figure=fig, width_ratios=[0.2, 1.4, 1.4])

for r in range(rank):
    # --- Left column: Template ---
    ax1 = fig.add_subplot(gs[r, 0])
    im = ax1.imshow(
        20 * np.log10(W[:, :, r] + 1e-16),
        aspect='auto',
        interpolation='none',
        origin='lower',
        extent=[0, T, frequencies[0], frequencies[-1]]
    )
    im.set_clim(np.min(Sxx_dB), np.max(Sxx_dB))

    # remove ticks and labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel(f"{names_music[r]}", fontsize=9)

    if r == 0:
        ax1.set_title("Template", fontsize=10)

    # --- Right column: Activation ---
    ax2 = fig.add_subplot(gs[r, 1])
    ax2.plot(times, H_tr[r, :])
    if r == 0:
        ax2.set_title("Activation", fontsize=10)

    # Truncate y-ticks to integers

    ax3 = fig.add_subplot(gs[r, 2])
    # Reconstructed component
    comp = W[:, :, r] @ shift_m(H_tr[r, :], T=T)
    im = ax3.imshow(
        20*np.log10(comp), 
        aspect='auto',
        interpolation='none',
        origin='lower',
        extent=[times[0], times[-1], frequencies[0], frequencies[-1]]
    )
    if r == 0:
        ax3.set_title("Components", fontsize=10)
    im.set_clim(np.min(Sxx_dB), np.max(Sxx_dB))
    ax3.set_yticks([])

plt.show()
```

The song is now well transcribed. In particular the last two chords have the right five notes, played simultaneously. The chords activations are still imperfect, but this is expected because of the non-linear mixing artifacts and the frequency similarity between octaves. We also see that all components slightly activate on each attack. This might be due to the hammer action sound which is the same for all notes and has not been taken into account separately.

The full model, weakly supervised on all piano notes, has performance close to that of fully supervised networks but is not robust to distribution shift (e.g. changing the piano or the recording conditions) {cite}`wuSemiSupervisedConvolutiveNMF2022`. Improving these aspects, and the training procedure, is part of my research project [link].

