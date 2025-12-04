import numpy as np
from tensorly_hdr.sep_nmf import spa, snpa
from scipy.io import loadmat
import matplotlib.pyplot as plt

n = 20
m = 300
r = 4
noise = 0.01
U = np.random.rand(n, r)
U = U/np.sum(U, axis=0)
V = np.random.rand(m, r)
V = V/np.sum(V, axis=0)
V = V.T  # fat matrix
for i in range(r):
    V[:, i] = 0
    V[i, i] = 1

X = U@V + noise * np.random.randn(n,m)

K2, U2, V2 = snpa(X, r)
K1, U1, V1 = spa(X, r)

# torch test
import tensorly as tl
import torch
tl.set_backend('pytorch')

U = torch.rand(n, r)
U = U/tl.sum(U, axis=0)
V = torch.rand(m, r)
V = V/tl.sum(V, axis=0)
V = V.T  # fat matrix

for i in range(r):
    V[:, i] = 0
    V[i, i] = 1

X = U@V + noise * torch.randn(n,m)

K2, U2, V2 = snpa(X, r)
K1, U1, V1 = spa(X, r)


# Test on urban
rank = 4 
data = 1.0*loadmat('../dataset/Urban.mat')['A'].T  # no Urban
normalize = True
out = spa(data, rank, tol=1e-8, normalize=normalize)
#out = snpa(data, 6, normalize=normalize)
_, W, H = out
fit = np.linalg.norm(data-W@H)**2/np.linalg.norm(data)**2

plt.plot(W)

plt.figure()
for i in range(rank):
    plt.subplot(1, rank, i+1)
    image = np.reshape(H[i, :], [307, 307]).T
    plt.imshow(image)
    