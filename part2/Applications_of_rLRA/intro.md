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

(part:applications)=
# Summary

Regularizations in LRA are necessarily driven by the properties of the low-rank factors one seeks to recover from the computation of LRA. Therefore, by understanding deeper the applications of LRA in signal processing, one is able to derive both interesting fundamental problems related to LRA, and ideas on how to design regularizations for LRA and algorithms to compute rLRA. Conversely, in some applications such as medical imaging, having theoretical guaranties on the quality of the reconstructed factors in rLRA is important because interpretation mistakes on the outputs of these methods may have significant impact. In the optical biopsy example detailed in [](TODO), rLRA outputs allow a surgeon to decide whether or not to remove brain cells during a surgery. Removing too few may lead to cancer recurrence, but removing too much may harm critical brain functionalities.

In my work, I have focused on a few specific modalities in the hope to gain expert knowledge in these fields and make actually useful practical contributions based on rLRA. The first modality is [spectral imaging](#spectral-imaging-and-rlra), both in remote sensing and in microscopy, and the second is music information retrieval, in particular [automatic music transcription](#automatic-music-transcription). Around the end of my PhD I also worked on chemometrics dataset including fluorescence spectroscopy {cite}`cohenCorrectingInnerFilter2016`, chromatography and nuclear magnetic resonance, which share similarities with spectral imaging when it comes to signal processing.

## Spectral imaging and rLRA
 
### Basics of spectral representation of light

```{note}
Sorry for the rant on spectral images :) Also I am not a physicist so the discussion below is probably fairly inaccurate, but this is my current understanding of spectral imaging.
```

In everyday life, color helps distinguishing different objects and materials in space. Color in fact carries a lot of information on our environment: one would probably never eat a light blue apple because this unusual color is a sign that the apple may contain unwanted chemical components. Tree leaves are typically orange-ish when they dry and are about to fall, but typically green-ish in spring and summer. It is therefore a very natural, and ancient, idea to exploit color information in science to obtain information on an observable system. Color is used in a variety of measurements systems, including color photography. 

From a mathematical and informatics point of view, defining color formally is, maybe surpisingly, a large and complex research domain. For the sake of simplicity, let us assume that colors can be represented in a three-dimensional system with basis vectors red, blue, and green. Any two-dimensional $m$ by $n$ color image is, in this simplistic model, a tensor of size $m\times n \times 3$. 

From a physical perspective, color is not a physical property of matter. It relates to the human perception (and therefore changes for each individual) of a physical property called wavelength, that characterizes the distance traveled by an electromagnetic wave in a single cycle. More precisely, any wave can be decomposed in the spectral domain as the sum of sinusoidal waves with a fixed wavelength. This is exactly the Fourier transform ubiquitous in signal processing. Any light emitted or reflected by a source of interest has a specific spectrum that represented the amplitude of the individual additive sine waves, and which contains information about the source, just like color. In fact color, as described in the simplified RGB format, is a compressed description of the full light spectrum, where spectral coefficients are added in three blobs respectively named red, green and blue [figure]. This is mostly due to human perspection: while human ears excel at identifying the Fourier coefficients (the amplitude of each additive sinusoidal wave), the human eye is much less sensitive; cones in the human eye that are sensitive to color naturally perform this dimensionality reduction. 

This fact has a few interesting consequences. First, the human eye is blind to any wavelength outside the visible range, limiting our ability to detect spectral information in the infrared or ultraviolet spectral bands. Second, two objects of the same color can have very different spectra. Therefore the human eye is a poor spectral sensor. Hopefully, it is possible to accurately measure light spectra of a single light flux using prisms, that have the property to spread light beams spatially depending on the wavelength. In other words, prisms perform, in some way, a Fourier transform of an incoming light wave and output the individual sinusoidal light waves with pure wavelengths separated spatially. It is then possible to measure the spectrum by essentially taking a black and white picture of the prism output. Devices that measure light spectra, using prisms or other separation means, are called spectrometers.

[figure wavelength] 

[figure with visible range and full spectrum]

### Spectral mixing and unmixing


We are taught in high school that colors obey two sets of rules for mixing: additive and substractive. Additive mixture works for light, and is typically described in the RBG coordinate format: red + blue lights yield magenta light, red + blue + green lights yield white light. This principle is ubiquitous in everyday life since color screen rely on this concept to generate colors by combining red, blue and green lights generated by tiny light sources for each pixel. Substractive synthesis applies for instance to paint, typically in the CMY format: cyan + magenta paints yield blue paint, cyan + magenta + yellow paints yields black paint. The coexistence of these two models can be slightly confusing: when is a mixture additive or substractive? Is it possible to design additive mixtures of paints and substractive mixtures for light beams?

A way to explain additive and substractive mixtures which I find useful is to relate them to the physical process underlying the mixture, and to the full spectral description of color. Additive mixture is a matter of perception. Various light beams with different spectra hit the human eye and are seen as a single light source by conjunction of the lens that focuses the input light on the cones integrating the contribution of each light source. In the case of a spectrometer, an optical lens can be used to produce a similar additive acquisition. Addition synthesis is therefore not a physical modification of the wavelengths of a light wave, but rather the superposition, typically spatial, of different light waves. In constrast, substractive mixture is a direct modification of the spectrum of a light wave that removes a part of that spectrum. A key concept to understand substractive mixture is the concept of filtering: when a light wave hits an object, it is partially absorbed and partially reflected. The intensity of reflected light depends on the wavelength, therefore the reflected light has a different spectra than the input light, filtered by the object spectral response. What we usually call the color of an object is in fact the projection in RBG space of the resulting spectrum of "white" light after it has been reflected by the item. The absorption spectrum of an object is the spectral filter by which the incoming light is multiplied to obtain the reflected light spectrum. Substractive mixture is then simply the consecutive filtering of a light source by several items, where spectral filters are multiplied. When mixing paints, the chemical compounds in each paint are intimely mixed, and the paint after mixture essentially filters lights jointly for all paints.

[Figure si j'ai la foi]

In scientific imaging, both additive and substractive mixtures are usually encountered. The important point is that additive mixture is usually perfectly well-suited for linear models, while substractive mixture leads to non-linear models. Two examples of additive mixture help illustrate this fact.

a. Fluorescence Spectroscopy in chemometrics. A mixture of several fluorophores is observed with a spectrometer. Each fluorophore emits a light wave with a specific spectrum. Since the fluorophores are loosely mixed inside the sample, and each individually emmit a fluorescence signal, the measured spectrum is simply the sum of the fluorescence spectra of each component. The light emitted by the fluorophores depends on the wavelength of the light exciting the sample; using a laser excitation with a controlled wavelength leads to a series of spectra acquisitions, stored into a matrix (fluorescence excitation emission matrix) $Y$ with $n$ columns corresponding to excitation wavelengths and $m$ rows containing the measured additive mixture of fluorescence spectra. Additive mixture in this context also related to the Beer-Lambert law [ref thèse], and translates into an (approximate) low-rank NMF

$$ Y = WH^T $$

where $W$ is a matrix containing the spectral of each individual fluorophore in the chemical mixture, and $H$ contains the amplitude of each fluorophore response to the excitation wavelength. Applying NMF to the data matrix $Y$, or nonnegative tensor factorization to several such measurement matrices, can in principle recover the individual fluorescence spectra, essentially performing spectral unmixing.

b. The linear mixing model in remote sensing makes the hypothesis that materials on an observed scene are spatially distributed and non-overlapping. The scene is cut into pixels by the camera, and each pixel may therefore contain several materials [see figure TODO] with proportions given by the portion of the pixel covered by each material. The spectra acquisition is then the additive mixture of the reflectance spectra (the spectrum of ambient light, essentially white, filtered by each material). For a single pixel $Y[:,i]$ of the acquired spectral image $Y$ with $m$ spectral wavelengths (or spectral bands if spectra are acquired in a compressed spectral representation) and $n$ pixels, the additive mixture of $K$ materials simply translates into a linear model

$$ Y[:,i] = \sum_{k=1}^{K} W[i,k] H^T[:,k];\; Y=WH^T $$

where $W$ contains columnwise the spectra of each material in the scene (supposing they are consistent over the whole image), and H contains the proportions of each material in each pixel columnwise, also called abundances. Interestingly, extensions of the linear mixing model that account for multiple reflexions typically involve products of matrix $W$ with itself; this is coherent with the substractive mixture model where the ambient light is filtered consecutively by several materials [ref Dobigeon].

To conclude this introduction to spectral unmixing, we can now answer the abvove question: is it possible to additively mix paints to produce white. The answer is nuanced. It is impossible to mix paint to produce pure white. Mixing paints intimately will result in a filtering effect that negates the ambient light spectrum in all wavelengths, leading to a black color. However, we can produce grey by juxtaposing paints on a surface and looking from afar. Spatial juxtaposition will result in additive filtering of the filtered white light, as in the linear mixing model. If three paints red, blue and green are used in equal proportions, the resulting spectrum is the sum of blue, red and green light but with reduced intensity, and the object will appear gray. Screens are able to produce white because they can emit red, blue and green spectra at full intensity, which is not possible with reflectance spectra.

### Several applications of spectral unmixing

Spectral unmixing, as hinted above, separates components spectra from several acquisitions of additive mixtures with varying mixture conditions. A popular example of spectral unmixing in the signal processing community is found in remote sensing, and I have often motivated rather theoretical works with such images (TODO links to pages). Since my arrival in CREATIS in 2022 however, my work has been focused on optical systems that allow to perform optical biopsy. Optical biopsy is a non-invasive technique that allows to gain spectral information on tissues (brain cells for instance) during a live surgery without actually requiring to extract a tissue sample. The idea is to design lightweight optics systems with high-end processing pipelines to acquire spectral images, and then use both spatial and spectral information to deduce information on the tissue nature (e.g. which brain cells are cancerous or not). Most of my work on this topic has been dedicated to the joint reconstruction and spectral unmixing for the single-pixel spectral camera [](./Single_pixel_spectral_imaging.md), but I have also contributed to a system working with RBG videos [](./Optical_Biopsy_separable_nmf.ipynb).


## Automatic music transcription

```{margin}
MIDI (Musical Instrument Digital Interface) is a numeric standard used to store and exchange symbolic music notation between numerical instruments and computers. In the context of music notation, it contains for each played note its pitch (in the Western system from C0 to A7), its onset (the activation time with respect to the song start), its offset, and its velocity (nuance) in 8 bits. While MIDI representations are in many regards much more precise than usual musical notations found on music sheets, it is often insuficient to describe the full interaction of the artist with the instrument: velocity may be time-dependent, pitch can be bent or microtonal, and activation may not be instantaneous.
```

Music Information Retrieval (MIR) is a collection of music-oriented machine learning tasks, ranging from tempo detection to automatic music composition. The tasks that compose MIR are always evolving (see for instance the [MIREX competition](https://www.music-ir.org/mirex/wiki/MIREX_HOME) that lists old and new tasks yearly). Among MIR tasks, Automatic Music Transcription (AMT) is a challenging task that aims at converting an audio recording into a MIDI file. Polyphonic instruments are typically the most challenging to transcribe, but singing voice or wind/brass instruments also have inherent difficulties despite being monophonic [ref generique AMT]. In what follows, let us focus mostly on piano transcription.

Among existing AMT algorithms, many rely on a time-frequency representation of the audio signal. The rationale is that for a single note played on the piano corresponds a comb-shaped spectrum in the Fourier domain composed of the fundamental frequency (440Hz for A4) and all harmonics. These harmonics, and in particular their relative intensity and position, define the tone of the piano. Harmonic instruments are characterized by the existence of this comb-shaped spectrum, while inharmonic instruments such as drums do not produce comb-shaped spectra. This is easily observed on a single note recording, here extracted from the MAPS database [ref].

```{code-cell} ipython3
from cmath import phase
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import scipy.signal as signal

A4_wav, sr = sf.read('../../tensorly_hdr/dataset/MAPS_ISOL_LG_F_S1_M57_ENSTDkAm.wav')
A4_wav = A4_wav / np.max(np.abs(A4_wav)) # audio normalization
A4_wav = A4_wav[:,0] # use only one channel if stereo

# Compute 1D Fourier transform
frequencies_1D = np.fft.fftfreq(len(A4_wav), d=1/sr)
fourier_1D = np.fft.fft(A4_wav)
magnitude_1D = np.abs(fourier_1D)
phase_1D = np.angle(fourier_1D)

# Plot 1D Fourier transform
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(frequencies_1D, 20*np.log10(magnitude_1D))
plt.title('1D Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.xlim(0, sr/2)
plt.subplot(2, 2, 2)
plt.plot(frequencies_1D, phase_1D)
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
#plt.xlim(0, sr/2)

# Compute spectrogram
frequencies, times, Sxx = signal.stft(A4_wav, fs=sr, nperseg=4096, nfft=4096, noverlap=4096 - 882)
Sxx_dB = 20 * np.log10(np.abs(Sxx) + 1e-10)  # convert to dB scale
Sxx_abs = np.abs(Sxx)
Sphase = np.angle(Sxx)
#Sphase_smoothed = signal.medfilt(Sphase, kernel_size=(25, 5))
half_freqs = len(frequencies) // 2

# Plot power spectrogram (up to 10kHz)
plt.subplot(2, 2, 3)
#plt.imshow(np.abs(Sxx), aspect='auto')
# use correct extent to display time and frequency axes
plt.imshow(Sxx_dB[:half_freqs,:], aspect='auto', origin='lower', extent=[times.min(), times.max(), frequencies.min(), frequencies[half_freqs]])
plt.title('Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity [dB]')

# Phase spectrogram
plt.subplot(2, 2, 4)
plt.imshow(Sphase[:half_freqs,:], aspect='auto', origin='lower', extent=[times.min(), times.max(), frequencies.min(), frequencies[half_freqs]])
plt.title('Phase Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Phase (radians)')
plt.tight_layout()
plt.show()
```
We show the spectrum of the full audio recording (about 15s) in magnitude as well as the phase (which is arguably hard to interpret as such). However over the 15s of recording, there are times where the note is not played, or when the timbre have changed. In particular, high frequencies tend to dissipate faster than low frequencies. Therefore, in audio signal processing, the spectrogram (magnitude and phase) that contains spectra for small time windows of the full signal is incredibly useful. We see above that the magnitude spectrogram of a single note contains rich information: one sees the comb-shape spectra fading over time, the hammer action when the note is played around 0.5s, and some noise around 12s. The phase spectrogram works similarly and, while it is harder to read, it contains a lot of information. In particular we can decifer the various partials of the comb. For AMT, we have enough information in the magnitude spectrogram and typically discard the phase spectrogram.

An interesting property of the spectrogram of a single note is that it is well approximated by a rank-one matrix. This suggests the use of a low-rank model to analyse more complex signals and perform AMT. NMF for AMT however as several issues: the rank-one hypothesis is too strong, the source mixture is non-linear and there are too few garantees for NMF to be unique. I adressed these issues partially in my work on weakly-supervised convolutive NMF for AMT, my contributions are detailled in [Automatic Music Transcription](./AMT.md). 

## Music structure estimation

We can also use spectrograms of full songs to detect similarities between bars. I will not detail this contribution in this manuscript, it was already detailed in length in Axel Marmoret's PhD manuscript [ref]. There are also available tutorials in the BarMusComp toolbox [link](https://gitlab.imt-atlantique.fr/a23marmo/barmuscomp) and the more recent Autosimilarity Segmentation toolbox [link](https://gitlab.imt-atlantique.fr/a23marmo/autosimilarity_segmentation) [todo refs]. The main methdological tool to enhance the similarity detection is the Nonnegative Tucker Factorization of the tensor spectrogramm obtained by stacking spectrogramms of each bar of a song [refs Nieto et nous]. See also [the related paragraph in the HDR summary](../../introduction/summary.md#music-segmentation).
