import numpy as np
from tensorly_hdr.giantly.arborescent import arborescent, ksparse_nnls

# Testing arborescent
n = 12
k = 4
x = np.random.rand(n)
x[k:] = 0
A = np.random.randn(10*n,n)
y = A@x + 0.01*np.random.randn(10*n)

AtA = A.T@A
Atb = A.T@y
btb = np.linalg.norm(y)**2

out = arborescent(AtA=AtA, Atb=Atb, btb=btb, k=k, return_nb_nodes=True)

print(x)
print(out)

# Testing mnnls, loop over arborescent
n = 1000
r = 5
k = 2
m = 101
H = np.random.rand(n, r)
W = np.random.rand(m, r)
H[:, k:] = 0
Y = W@H.T + 0.0001*np.random.randn(m, n)
He = np.random.rand(n, r)
YtY = np.linalg.norm(Y, axis=0)**2

out = ksparse_nnls(W.T@Y, W.T@W, YtY, He.T, k).T

print(H[:5, :])
print(out[:5, :])
