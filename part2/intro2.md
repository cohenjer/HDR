---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Contributions to regularized low-rank approximations


Since the start of my PhD thesis in 2013, I have been focusing on advancing the current state of knowledge for regularized Low-Rank Approximations (rLRA). My works cover a diverse list of topics ranging from understanding the properties of the solutions of rLRA to proposing new applications for these models. A significant section of my work is however dedicated to algorithm design to compute solutions to rLRA efficiently in various setups. The reader should refer to part [todo] of the manuscript for a quick technical introduction to the topics discussed thereafter.

[block?]
In the last ten years, I have focused my research on LRA models as tools for extracting meaningful information out of matrices and tensors, using regularizations to enhance interpetability. These regularizations often include nonnegativity, which has become my specialty. There is a significant body of literature on these models, covering variants of rLRA alongside indentifiability theory (NMF [], Canonical Polaydic Decompostion [], Tucker Decomposition [], Parafac2 [] to name a few), algorithms to compute rLRA using in particular rather recent developments on non-smooth and non-convex optimization [refs], and a myriad of applications in chemometrics [], neuroscience [], statistical inference [anankudmar], spectral unmixing for remote sensing [] or microscopy imaging [], music information retrieval [], telecommunications [], psychometry [] \ldots.

Despite the significant amount of existing works on rLRA, their use by practitionners is often quite difficult. Indeed there are many exciting theoretical and practical issues that remain to be dealt with:
- Optimization: multi-block, nonconvex, --> convergence ?
- Available side information (semi-supervised) --> how to account for it; 
- Multimodality
- Modeling: nature of the solutions? guarantees ?
- For Tensors: software tools, large scale contractions, GPU support and so on

My work has been dedicated to proposing (partial) solutions to these issues.
I have grouped my contributions in three sections, dealing respectively with theoretical contributions, applications-oriented contributions and algorithmic-focused contributions. However in most of these works, all three aspects (theory, algorithms, applications) are intertwined, so this is no a strict segmentation of my work.

FOR THE EVAL A VAGUE --> summarize
TODO: fill in with links
## Theory of rLRA contributions

## Applications of rLRA contributions

## Algorithms for existing rLRA models contributions

## Others
Lever arm
Birkov
Okular-EEG
Temporal-aware CMTF

## Table of contents (necessary?)

a refaire à la fin.

%```{toctree}
%:glob:
%:maxdepth: 2

%Theory_of_rLRA/intro
%Fast_algorithms_for_rLRA/intro
%Applications_of_rLRA/intro
%```

This tells you things that are in chapter 2 (contributions, core of HDR)