import numpy as np

__all__ = ["kcca"]


def kcca(H1, H2, outdim_size):
    """
    An implementation of Kernel Canonical Correlation Analysis (KCCA)
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o)
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o)

    # Center the kernel matrices
    H1_centered = H1 - np.mean(H1, axis=0)
    H2_centered = H2 - np.mean(H2, axis=0)

    # Gaussian kernel function
    def gaussian_kernel(X, Y, sigma=1.0):
        pairwise_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-pairwise_dists / (2 * sigma ** 2))

    # Compute kernel matrices
    K1 = gaussian_kernel(H1_centered, H1_centered)
    K2 = gaussian_kernel(H2_centered, H2_centered)

    # Center the kernel matrices
    N = K1.shape[0]
    one_n = np.ones((N, N)) / N
    K1 -= one_n.dot(K1) - K1.dot(one_n) + one_n.dot(K1).dot(one_n)
    K2 -= one_n.dot(K2) - K2.dot(one_n) + one_n.dot(K2).dot(one_n)

    # Compute eigenvectors
    eigvals, U = np.linalg.eigh(K1)
    eigvals, V = np.linalg.eigh(K2)

    # Compute transformation matrices
    A = np.dot(U, np.dot(np.diag(1. / np.sqrt(eigvals + r1)), U.T)[:outdim_size, :])
    B = np.dot(V, np.dot(np.diag(1. / np.sqrt(eigvals + r2)), V.T)[:outdim_size, :])

    return A, B, mean1, mean2
