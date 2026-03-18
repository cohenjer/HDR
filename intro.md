# Manuscript Home

[Figure moi]

Welcome to [Jeremy Cohen](https://jeremy-e-cohen.jimdofree.com/)'s "Habilitation à Diriger des Recherches" (HDR) manuscript. This document is a synthesis of all the work I have done since my PhD thesis, and took almost two years to complete. The main topic under study in this manuscript is Low-Rank Approximations, and alongside my personal contributions, I summarize the current state-of-the-art to the best of my knowledge.

The book contains code snippets with barebone implementations of algorithms, numerical simulations and visualisations. This was done so that most results shown in the manuscript are easily reproduced by the reader. The book is available in .pdf, but it is advised to read the content from this website to get the best interface experience. The philosophy of the book design is further discussed in [](./Howtouse/howtoread.md) [TODO].

The scientific content of the book is summarized in the [](./intro/introduction.md). Here is the table of content for quick access.

TODOlist:
- ajout refs dans les .ipynb, il reste aussi des TODOs pas vu au 1er tour car ipynb
- Figures list: (main / inkscape peut aussi être excalidraw ? faire un test pour genre projection NNLS avec les 2/3 sols)
  - NMF en HSI [summary.md] --> main ou inkscape ? voir aussi LMM
  - CPD [summary.md] [lra.md] --> main ?
  - rank-one sum / factors CP [lra.md] --> main ?
  - NMF cone dim 3 [lra.md] --> tikz ou python ou inkscape
  - NMF non uniqueness rang 2 [lra.md]
  - NMF rang 3 dim 4 vue projective, unique (1) et nonunique (2) [lra.md]
  - NMF rank 2 separable special case [lra.md]
  - Tucker [lra.md] --> main
  - NNLS projection [nnls.md, lra.md] --> main
  - Fonction global Lipschitz / stepsize etc [nnls.md] --> main
  - MM principle [nnls.md] --> main
  - steps of active set [nnls.md] --> main/inkscape
  - HALS ? [nnls.md] --> main
  - second order >0 solution [nnls.md] --> main
  - Linear mixing model [parts2/applis/intro.md] --> inkscape
  - Patterns Hadamard [single_pixel_spectral_imaging.md] --> python, cf aussi papier freeform
  - Branch and Bound illustration with graph pruning (gif?) [sparse_nnls.md] --> main et inkscape pour les maths
  - illustration smooth basis [DLRA.md] (code ?) --> python
  - Figure projet de recherche [KarpCoi.md] --> inkscape
  - Music segmentation avec AS barwise [summary.md] --> main ou inkscape ?
  - KL cost [nnls.md ?, summary.md] --> python ?
  - AO vs BCD en terme de majoration du coût, en 3d [AO.md] --> python ?
  - ACP [lra.md] --> main ou inkscape
  - Spectre visible vs RBG [part2/applis/intro.md] --> inkscape
  - Wavelength, visible range... [part2/applis.intro.md] ?
  - Explication principe [single_pixel_spectral_imaging.md] --> inkscape reprendre Séréna ou Nicolas
  - CMTF models [CMTF.md] --> main
  - HOSVD compression [proco-als.md] --> main
  - projection cone dual [proco-als.md] (utile?) --> main
  - Borgen plot illustration [others.md] --> inkscape inspiration site web
  - Acquisition principle for dataset hyperspectral microscopy [others.md] --> main ou inkscape
- Figures design
    - Tikz si maths, main si principe, code python si simple
    - Geometrie NMF book Gillis, 20min max at figure time
    - Figure inline integration with glue and environments
- Relecture
    - Relecture deja faite du rapport CNRS
    - Cohérence style (we, I, forme passive...)
    - Coherence between parts, intro, perspectives.
    - Acronyms, vocabulary and math notations list (NNKL or NN-KL)
    - Broken links
- Binder build
- Precise Version and package dependence
- Latex build fork ?
- How to use (0.25d)


```{tableofcontents}
```

Many pages contain code executed upon the book compilation, below is their status.

```{nb-exec-table}
```
