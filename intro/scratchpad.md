# A tentative structure for the book

(label1)=

TODO dont use too many ## headers as they will diminish the ## levels of all subsequent files

- Prerequisites with links (numerical optimization, linear algebra and system solving) (Also Tammy's book)
- Something about alternating algorithms convergence (BSUM, Bertsekas, other??)
- Tensor preliminaries with tensorly
  - Tensor product and multiway arrays
  - Abstract and practical tensor operations: contractions from tensor networks to n-mode products. Avoid matricization!!
  - Matrix and Tensor decompositions: separability/rank-one, matrix/CP/multilinear ranks, interpretability/identifiability
  - Inference algorithms: the ALS example (cf paper)
  - Example applications in ML (Tensorly or Caglayan Dataset)
- Constrained decompositions contributions
  - Why constraints (matrix case e.g. NMF, Tucker). For CP, convincing example?
  - My work focuses on nonnegativity and sparsity. Some important considerations: prox, constrained data vs constrained factors, constrained rank
  - A family of algorithms: NNLS, LASSO/IHT, ADMM 
  - Contributions
    - Nonnegative LRA for Matrix/Tensor
      - PROCO-ALS
    - Sparse LRA
      - DL identifiability
      - Dictionary-based decompositions (semi-supervised LRA)
    - Sparse and Nonnegative
      - Sparse l0 NNLS/NMF
      - Sparse separable NMF
      - Implicit regularization
- Applications-focused contributions
  - Music
    - Automatic transcription
    - Segmentation
  - Hyperspectral imaging
    - Tensor for angle, time or patch data
    - single pixel imaging
  - Chemometrics
    - NLFD
  - Others?
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
  - Codes from papers
- Perspectives
  - Sparse decompositions with optimized kernels
  - Better sparse tensor decomposition software support/interface for end-users outside matlab

## Example
refering to section above is as ez as [this](label1). Question: cross ref across documents?
