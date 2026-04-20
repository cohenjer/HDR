# Manuscript Home

[Figure moi]

Welcome to [Jeremy Cohen](https://jeremy-e-cohen.jimdofree.com/)'s "Habilitation à Diriger des Recherches" (HDR) manuscript. This document is a synthesis of all the work I have done since my PhD thesis, and took almost two years to complete. The main topic under study in this manuscript is Low-Rank Approximations, and alongside my personal contributions, I summarize the current state-of-the-art to the best of my knowledge.

The book contains code snippets with barebone implementations of algorithms, numerical simulations and visualisations. This was done so that most results shown in the manuscript are easily reproduced by the reader. The book is available in .pdf, but it is advised to read the content from this website to get the best interface experience. The philosophy of the book design is further discussed in [](./Howtouse/howtoread.md) [TODO].

The scientific content of the book is summarized in the [](./intro/introduction.md). Here is the table of content for quick access.

TODOlist:
- ajout refs dans les .ipynb, il reste aussi des TODOs pas vu au 1er tour car ipynb
- Figures list: (main / inkscape, peut aussi être excalidraw ? faire un test pour genre projection NNLS avec les 2/3 sols)
  - --> export from notebook boox en vector pdf max resolution; import avec Cairo !! Color, scale Aetc. Save as svg, export as png. Move files to HDR, modify subsequently from there, remove from filetransfer.
  - DONE NMF en HSI [summary.md] --> main ou inkscape ? voir aussi LMM
  - DONE CPD [summary.md] 
  - DONE ACP [lra.md] --> main ou inkscape
  - DONE CPD [summary.md] [lra.md] --> main ?
  - DONE rank-one sum / factors CP [lra.md] --> main ?
  - DONE NNLS projection [nnls.md, lra.md] --> main
  - DONE NMF cone dim 3 [lra.md] --> tikz ou python ou inkscape
  - DONE NMF non uniqueness rang 2 [lra.md]
  - DONE NMF rang 2 dim 3 vue projective, unique (1) et nonunique (2) [lra.md]
  - DONE NMF rank 2 et 3 separable special case [lra.md]
  - DONE Tucker [lra.md] --> main
  - DONE Fonction global Lipschitz / stepsize etc [nnls.md] --> main
  - DONE MM principle [nnls.md] --> main
  - DONE HALS ? [nnls.md] --> main
  - DONE second order >0 solution [nnls.md] --> main
  - DONE Linear mixing model [parts2/applis/intro.md] --> inkscape
  - DONE Spectre visible vs RBG [part2/applis/intro.md] --> inkscape
  - DONE Wavelength, visible range... [part2/applis.intro.md] ?
  - DONE Borgen plot illustration [others.md] --> inkscape inspiration site web
  - DONE Acquisition principle for unmixing dataset hyperspectral microscopy [others.md] --> main ou inkscape
  - DONE Figure projet de recherche [KarpCoi.md] --> inkscape
  - DONE Music segmentation avec AS barwise [summary.md] --> main ou inkscape ?
  - DONE HOSVD compression [proco-als.md] --> main
  - DONE KL cost in 1d and 2d  [nnls.md ?, summary.md] --> python ?
  - DONE Patterns Hadamard [single_pixel_spectral_imaging.md] --> python, cf aussi papier freeform
  - DONE Explication principe [single_pixel_spectral_imaging.md] --> inkscape reprendre Séréna ou Nicolas
  - steps of active set [nnls.md] --> main/inkscape
  - Branch and Bound illustration with graph pruning (gif?) [sparse_nnls.md] --> main et inkscape pour les maths
  - illustration smooth basis [DLRA.md] (code ?) --> python
  - AO vs BCD en terme de majoration du coût, en 3d [AO.md] --> python ?
- Figure inline integration with glue and environments
- Relecture
    - Relecture deja faite du rapport CNRS
    - Cohérence style (we, I, forme passive...)
    - Coherence between parts, intro, perspectives.
    - Acronyms, vocabulary and math notations list (NNKL or NN-KL)
    - Broken links
- Fix code dependencies again
- Binder build
- Precise Version and package dependence
- Latex build fork ?
- How to use (0.25d)

- Maybe later:
  - Figures
    - CMTF models [CMTF.md] --> main
    - PARAFAC2 projections
    - projection cone dual [proco-als.md] (utile?) --> main


```{tableofcontents}
```

Many pages contain code executed upon the book compilation, below is their status.

```{nb-exec-table}
```
