# Manuscript Home

```{figure} Figures/Moi_2022_cartoonish.jpg
---
width: 300px
align: right
name: fig:moi
---
```

Welcome to [Jeremy Cohen](https://jeremy-e-cohen.jimdofree.com/)'s "Habilitation à Diriger des Recherches" (HDR) manuscript. This document is a synthesis of the work I have done since my PhD thesis, and took almost two years to complete. The main topic under study in this manuscript is Low-Rank Approximations, and alongside my personal contributions, I introduce relevant existing results and tools from numerical optimization, having in mind to reuse this material for teaching.

The book contains code snippets with barebone implementations of algorithms, numerical simulations and visualisations. This was done so that most results shown in the manuscript are easily reproduced by the reader. The book is available in .pdf, but it is advised to read the content from this website to get the best interface experience. The philosophy of the book design is further discussed in [](./Howtouse/howtoread.md).

The scientific content of the book is summarized in the [](./introduction/summary.md). Here is the table of content for quick access.

TODOlist:
- Relecture
    - Acronyms, vocabulary and math notations list (NNKL or NN-KL)
    - Cohérence style (we, I (only where relevant), forme passive...)
    - Coherence between parts, intro, perspectives.
- Fix code dependencies again
  ------- Envoi 1 --------- (web)
- Figure inline integration with glue and environments
- Binder build
- Precise Version and package dependence
- Latex build fork ?
- Photo plus fun accueil
  ------- Envoi 2 (pdf) ----

%- Maybe later:
  %- bar instead of line plots in HRSI alg
  %- Figures
    %- CMTF models [CMTF.md] --> main
    %- PARAFAC2 projections
    %- projection cone dual [proco-als.md] (utile?) --> main
    %- rank r matrix factorization [lra.md] --> main
  %- Local installation instructions ?


%## Note to self about figures:
%On peut cacher le print avec plt.close(). Sinon pour le dev, utiliser remove-output dans les tags de la cellule de code; glue permet de garder en mémoire des variables, on peut alors cacher les cellules où on fait ca. [TODO remove]

```{tableofcontents}
```

Many pages contain code executed upon compilation of the book for publication, below is their status.

```{nb-exec-table}
```
