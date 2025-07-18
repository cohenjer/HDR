# A tentative structure for the book

(label1)=

Note: could be at the root page, remove part 0

## Intro
- How to read this book (intro)
  - Book philosophy 
    - Contents: Not deep on theory, see Hackbush for it; also not deep on practical implementation of tensor routines, see Tammys book for it. This is the middle layer book! Algorithm design require specific knowledge, but with knowledge from both pure theory and practical aspects.
    - use code seemlessly in the text instead of length descriptions or imprecise pseudo-code
    - reproducibility of all experiments
    - making this HDR useful, easy access to all/doc for tensorly
  - Online tools: reviewing, code running (hands-on)
  - Tensorly package, data links

## Chapter 1
- Prerequisites with links (numerical optimization, linear algebra and system solving) (Also Tammy's book)
- A family of algorithms: NNLS, LASSO/IHT, ADMM 
- Something about alternating algorithms convergence (BSUM, Bertsekas, other??)
- Matrix and Tensor algebra and low-rank approximations preliminaries with tensorly
  - Tensor product and multiway arrays
  - Abstract and practical tensor operations: contractions from tensor networks to n-mode products. Avoid matricization!!
  - Matrix and Tensor decompositions: separability/rank-one, matrix/CP/multilinear ranks, interpretability/identifiability
  - Inference algorithms: the ALS example (cf paper)
  - Example applications in ML (Tensorly or Caglayan Dataset)

## Chapter 2
- Contributions outline (detailled, e.g. the contents of the eval vague, with hyperlinks)
- Constrained decompositions contributions
  - Why constraints (matrix case e.g. NMF, Tucker). For CP, convincing example?
  - My work focuses on nonnegativity and sparsity. Some important considerations: prox, constrained data vs constrained factors, constrained rank
  - Contributions
    - Nonnegative LRA for Matrix/Tensor
      - PROCO-ALS
    - Sparse LRA
      - DL identifiability
      - Dictionary-based decompositions (semi-supervised LRA)
    - Sparse and Nonnegative
      - Sparse l0 NNLS/NMF (algorithms?)
      - Sparse separable NMF
      - Implicit regularization
- Applications-focused contributions (put first to motivate the rest ??)
  - Music
    - Automatic transcription
    - Segmentation
  - Hyperspectral imaging
    - Tensor for angle, time or patch data
    - single pixel imaging
  - Chemometrics
    - NLFD
  - Others? (neuro, 2 papers !)
- Algorithmic focused contributions
  - Accelerated AO
  - Improving MU
    - FastMU
    - Unrolled MU
  - Coupled factorization algorithms
    - constrained flex
    - constrained Parafac2
- Software contributions
  - Tensorly
    - HALS implementation
    - ADMM
    - GCP
    - ...
  - Shootout
  - Benchopt NMF benchmark
  - Codes from papers
  
## Chapter 3
- Perspectives
  - Optimization for LRA with KL and regularization (most important)
  - Sparse decompositions with optimized kernels
  - Better sparse tensor decomposition software support/interface for end-users outside matlab

## Example
refering to section above is as ez as [this](label1). Question: cross ref across documents?

TODO dont use too many ## headers as they will diminish the ## levels of all subsequent files