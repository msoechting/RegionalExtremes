import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import random
from math import sqrt
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances, f1_score
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split

from time import time

import cvxpy as cp


# Kernel-PCA
def K_PCA(X, kernel, n_components):

    # Kernel matrix
    K = pairwise_kernels(X, metric=kernel, gamma=15)

    # Centered kernel matrix
    N = K.shape[0]
    K_tild = np.dot(
        np.dot((np.identity(N) - (1 / N) * np.ones(N)), K),
        (np.identity(N) - (1 / N) * np.ones(N)),
    )  # centred K

    # Eignevalues & Eigenvectors
    sigma = np.cov(np.transpose(K_tild))
    eigenvector, eigenvalue = np.linalg.eig(sigma)

    # Top k eigenvectors
    eigvals, eigvecs = eigh(K_tild)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    X_pca = np.column_stack([eigvecs[:, i] for i in range(n_components)])

    return X_pca


X_kpca = K_PCA(moonX, "rbf", 2)
y = moonY

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color="red", marker="^", alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color="blue", marker="o", alpha=0.5)

ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)), color="red", marker="^", alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)), color="blue", marker="o", alpha=0.5)

ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")

ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC1")
# ax[0].set_aspect('equal') #distances between points are represented accurately on screen
plt.ylim(-2, 2)

plt.tight_layout()
plt.show()
