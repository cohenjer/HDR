# Manuscript Home


Welcome to [Jeremy Cohen](https://jeremy-e-cohen.jimdofree.com/)'s "Habilitation à Diriger des Recherches" (HDR) manuscript. This document is a synthesis of the work I have done since my PhD thesis, and took almost two years to complete. The main scientific topic of this manuscript is Low-Rank Approximations, and alongside my personal contributions, I introduce relevant existing results and tools from numerical optimization, having in mind to reuse this material for teaching.

```{figure} Figures/Moi_lofi_ChatGPT.png
---
width: 300px
align: right
name: fig:moi
---
(Generated with ChatGPT)
```

The book contains code snippets with barebone implementations of algorithms, numerical simulations and visualisations. This was done so that most results shown in the manuscript are easily reproduced by the reader. The book is available in .pdf upon request, but it is advised to read the content from this website to get the best interface experience. The philosophy of the book design is further discussed in [](./Howtouse/howtoread.md).

The scientific content of the book is summarized in the [](./introduction/summary.md). Here is the table of content for quick access.
 
```{Admonition} Use of AI tools
Except for the picture above, generative AI has been used in this manuscript only for grammar and style checking, and generate javascript codes for interactive plots.
```

%TODOlist:
%- Latex build fork ?
  %- Moving margin notes
  %- Gif hadamard
  %- taille figures notamment AMT
  %- pas de proof index
  %------- Envoi 2 (pdf) ----

% Before soutenance:
  %- better bibliography
  %- Binder build
  %- Precise Version and package dependence

%- Maybe later:
  %- Figure inline integration with glue and environments
  %- Local installation instructions ?
  %- Better proofread project
  %- precise pages in books refs
  %- Figures
    %- CMTF models [CMTF.md] --> main
    %- PARAFAC2 projections
    %- projection cone dual [proco-als.md] (utile?) --> main
    %- rank r matrix factorization [lra.md] --> main


%## Note to self about figures:
%On peut cacher le print avec plt.close(). Sinon pour le dev, utiliser remove-output dans les tags de la cellule de code; glue permet de garder en mémoire des variables, on peut alors cacher les cellules où on fait ca. [TODO remove]

%```{tableofcontents}
%```


```{bibliography}
:style: alpha
```

Many pages contain code executed upon compilation of the book for publication, below is their status.

```{nb-exec-table}
```