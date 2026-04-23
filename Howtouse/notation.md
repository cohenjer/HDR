# Notation

- Matrices are typically represented by uppercase letters, such as $M$, while vectors and scalars are denoted by lowercase letters, like $x$.
- Scalar values are often indicated using lowercase Greek $\lambda$. 
- The transpose of a matrix or vector is indicated with a superscript $M^T$.
- Slicing of tensors is performed as in python. For instance, the entry at row $i$ and column $j$ of a matrix $M$ is expressed h as $M[i,j]$. To remove the $j$th column of a matrix $M$, we write $M[:,-j]$. To consider all columns and rows up to indices $i$ and $j$, we write $M[:i,:j]$.
- Notation $\mathbb{R}_+$ denotes the set of nonnegative real numbers, and constraints like $x \geq 0$ indicate elementwise nonnegativity.
- The Frobenius norm is denoted as $\|M\|_F$.
- The support of a vector $x$, or the set of indices of its nonzero elements, is denoted by $S(x)$, or simply $S$ when clear from context.
- The Kronecker product is symbolized by $\otimes_K$ to avoid confusion with the generic tensor product $\otimes$. 
- The gradient of a function $f$ is written as $\nabla f$, and its Hessian as $\nabla^2 f$.
- The (left) Moore-Penrose pseudo-inverse of a matrix $M$ is represented by $M^\dagger$. 
- The Kullback-Leibler divergence between two nonnegative reals $y$ and $z$ is denoted as $\KL{y, z}$. 
- The spark of a matrix $M$, or the smallest number of linearly dependent columns in $M$, is written as $\text{spark}(W)$. 
- The column space, or span, of a matrix $U$ is the set of any vector that can be written as $Ux$ and is denoted as $\text{col}(U)$. The positive span of matrix $U$, obtained when $x$ is elementwise nonnegative, is denoted by $\cp(U)$.
- The set of linear maps acting on vectors from a subspace $E$ is denoted as $\mathcal{L}(E)$.

## Tensor notation
The unfolding of a tensor $T$ along mode $i$, denoted $T_{[i]}$, is obtained by row-first vectorization of all modes but $i$. The Khatri-Rao product of two matrices $A$ and $B$ with the same number $r$ of columns is the columnwise Kronecker product,

$$ A\odot B = \left[A[:,1]\otimes_K B[:,1], \ldots, A[:,r]\otimes_K B[:,r] \right]. $$
The multiway product can be defined for three matrices $A$, $B$, $C$ multiplied on modes one, two and three with a tensor $G$ as

$$\left(A\times_1 B\times_2 C\times_3 G\right)[i,j,k] = \sum_{r_1,r_2,r_3}A[i,r_1]B[j,r_2]C[k,r_3]G[r_1,r_2,r_3].$$
It can be implemented with tensor contractions, or with matrix-matrix products with unfoldings and reshapes involved.

The [CP decomposition](../part1/lra.md) is represented using the Kruskal notation $\llbracket A,B,C\rrbracket$.