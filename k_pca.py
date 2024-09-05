import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import eigh


# Kernel-PCA
class K_PCA:
    def __init__(
        self,
        n_components: int,
        kernel="rbf",
    ):
        self.n_components = n_components
        self.kernel = kernel

    def fit_transform(X):
        # Kernel matrix
        K = pairwise_kernels(X, metric=self.kernel, gamma=15)

        # Centered kernel matrix
        N = K.shape[0]
        K_tild = np.dot(
            np.dot((np.identity(N) - (1 / N) * np.ones(N)), K),
            (np.identity(N) - (1 / N) * np.ones(N)),
        )

        # Eignevalues & Eigenvectors
        # sigma = np.cov(np.transpose(K_tild))
        # eigenvector, eigenvalue = np.linalg.eig(sigma)

        # Top k eigenvectors
        eigvals, eigvecs = eigh(K_tild)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
        X_pca = np.column_stack([eigvecs[:, i] for i in range(self.n_components)])

        return X_pca
