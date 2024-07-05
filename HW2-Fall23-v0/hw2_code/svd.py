import numpy as np


class SVD(object):
    def __init__(self):
        pass

    def svd(self, data):
        """
        Do SVD. You could use numpy SVD.
        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V^T: (D,D) numpy array
        """
        U, S, Vt = np.linalg.svd(data, full_matrices=True)
        return U, S, Vt

    def rebuild_svd(self, U, S, V, k):
        """
        Rebuild SVD by k componments.

        Args:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V: (D,D) numpy array
                k: int corresponding to number of components

        Return:
                data_rebuild: (N,D) numpy array

        Hint: numpy.matmul may be helpful for reconstruction.
        """
        rank = min(U.shape[1], V.shape[0])
        if k > rank:
            print("Warning: k is greater than the rank of the input matrix. Setting k to the maximum valid value.")
            k = rank

        S_k = np.diag(S[:k])
        U_k = U[:, :k]
        Vt_k = V[:k, :]
        data_rebuild = np.matmul(U_k, np.matmul(S_k, Vt_k))

        return data_rebuild

    def compression_ratio(self, data, k): 
        """
        Compute the compression ratio: (num stored values in compressed)/(num stored values in original)

        Args:
                data: (N, D) TF-IDF features for the data.
                k: int corresponding to number of components

        Return:
                compression_ratio: float of proportion of storage used
        """
        U, S, Vt = self.svd(data)
        num_original_values = data.shape[0] * data.shape[1]
        num_compressed_values = (U.shape[0] * k) + k + (Vt.shape[1] * k)
        compression_ratio = num_compressed_values / num_original_values

        return compression_ratio

    def recovered_variance_proportion(self, S, k):  
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
                S: (min(N,D), ) numpy array
                k: int, rank of approximation

        Return:
                recovered_var: float corresponding to proportion of recovered variance
        """
        total_variance = np.sum(S**2)
        variance_recovered = np.sum(S[:k]**2)
        recovered_var = variance_recovered / total_variance

        return recovered_var