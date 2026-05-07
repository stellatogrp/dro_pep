"""
Custom Interpolation Canonicalizer.

This module provides a canonicalizer that takes PEP data directly (not via PEPit)
for use with autodiff-compatible pipelines where constraint matrices are computed
as JAX arrays.
"""

import numpy as np
import scipy.sparse as spa
from sklearn.cluster import KMeans


class CustomInterpCanonicalizer(object):
    """
    Canonicalizer that accepts pre-computed PEP constraint matrices directly.
    
    This class does not rely on PEPit's internal functions, allowing the constraint
    matrices (A_vals, b_vals, etc.) to be computed externally (e.g., via JAX for
    autodifferentiation).
    
    Args:
        pep_data: Tuple containing the PEP problem data:
            (A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
            - A_obj: Objective matrix (dimG x dimG)
            - b_obj: Objective vector (dimF,)
            - A_vals: List of constraint matrices, each (dimG x dimG)
            - b_vals: List of constraint vectors, each (dimF,)
            - c_vals: List of constraint constants (scalars)
            - PSD_A_vals: List of PSD constraint matrices (can be empty [])
            - PSD_b_vals: List of PSD constraint vectors (can be empty [])
            - PSD_c_vals: List of PSD constraint constants (can be empty [])
            - PSD_shapes: List of PSD matrix shapes (can be empty [])
        samples: List of (G, F) tuples representing sampled Gram matrices
        measure: Risk measure type ('expectation' or 'cvar')
        wrapper: Solver wrapper type ('cvxpy' or 'clarabel')
        precond: Whether to apply preconditioning (default: True)
        precond_type: Type of preconditioning ('average', 'max', 'min')
        mro_clusters: Number of MRO clusters (optional)
    """

    def __init__(self, pep_data, samples, measure, wrapper, 
                 precond=True, precond_type='average', mro_clusters=None):
        # Unpack pep_data tuple
        (A_obj, b_obj, A_vals, b_vals, c_vals, 
         PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = pep_data
        
        # Store provided constraint matrices directly
        self.A_obj = np.asarray(A_obj)
        self.b_obj = np.asarray(b_obj)
        self.A_vals = np.asarray(A_vals) if len(A_vals) > 0 else np.array([])
        self.b_vals = np.asarray(b_vals) if len(b_vals) > 0 else np.array([])
        self.c_vals = np.asarray(c_vals) if len(c_vals) > 0 else np.array([])
        self.PSD_A_vals = [np.asarray(a) for a in PSD_A_vals]
        self.PSD_b_vals = [np.asarray(b) for b in PSD_b_vals]
        self.PSD_c_vals = [np.asarray(c) for c in PSD_c_vals]
        self.PSD_shapes = PSD_shapes
        
        # Store other parameters
        self.samples = samples
        self.measure = measure
        self.wrapper = wrapper
        self.precond = precond
        self.precond_type = precond_type
        
        # Set preconditioning based on samples and A_obj/b_obj
        self.set_preconditioner()
        
        # Handle MRO clustering if specified
        if mro_clusters is None:
            self.samples_to_use = samples
        else:
            self.cluster_centers, self.cluster_labels, self.kmeans_obj = \
                self.clustered_sample_centers(samples, mro_clusters)
            self.samples_to_use = self.cluster_centers
        self.full_samples = samples

    def set_preconditioner(self):
        """Set preconditioning factors based on samples."""
        self.preconditioner = (np.ones((self.A_obj.shape[0],)), np.ones(self.b_obj.shape))
        
        if self.precond and len(self.samples) > 0:
            if self.precond_type == 'average':
                avg_sample = (
                    np.average([np.sqrt(np.diag(sample[0])) for sample in self.samples], axis=0),
                    np.average([sample[1] for sample in self.samples], axis=0)
                )
                self.preconditioner = (1/avg_sample[0], 1/np.sqrt(avg_sample[1]))
            elif self.precond_type == 'max':
                max_sample = (
                    np.max([np.sqrt(np.diag(sample[0])) for sample in self.samples], axis=0),
                    np.max([sample[1] for sample in self.samples], axis=0)
                )
                self.preconditioner = (1/max_sample[0], 1/np.sqrt(max_sample[1]))
            elif self.precond_type == 'min':
                min_sample = (
                    np.min([np.sqrt(np.diag(sample[0])) for sample in self.samples], axis=0),
                    np.min([sample[1] for sample in self.samples], axis=0)
                )
                self.preconditioner = (1/min_sample[0], 1/np.sqrt(min_sample[1]))
            else:
                raise ValueError(f'{self.precond_type} is invalid precond_type')
            
            # Handle NaN/inf and scale
            new_precond_0 = np.nan_to_num(
                self.preconditioner[0], nan=1.0, posinf=1.0, neginf=1.0
            ) * self.A_obj.shape[0]
            new_precond_1 = np.nan_to_num(
                self.preconditioner[1], nan=1.0, posinf=1.0, neginf=1.0
            ) * np.sqrt(self.b_obj.shape[0])
            self.preconditioner = (new_precond_0, new_precond_1)
        
        self.precond_inv = (1 / self.preconditioner[0], 1 / self.preconditioner[1])

    def setup_problem(self):
        raise NotImplementedError
    
    def set_eps(self, eps):
        raise NotImplementedError
    
    def solve(self):
        raise NotImplementedError
    
    def set_params(self, eps=0.1, alpha=0.1):
        raise NotImplementedError

    def clustered_sample_centers(self, samples, num_clusters, rng_seed=0):
        """Cluster samples using KMeans and return cluster centers."""
        print(f'clustering into {num_clusters} clusters')
        
        F_idx_cutoff = samples[0][1].shape[0]
        precondG, precondF = self.preconditioner
        precondinvG, precondinvF = self.precond_inv
        
        precondG = np.diag(precondG)
        precondF = np.diag(precondF)
        precondinvG = np.diag(precondinvG)
        precondinvF = np.diag(precondinvF)
        
        X = []
        for samp in samples:
            G, F = samp
            X.append(np.hstack([symm_vectorize(precondG @ G @ precondG), precondF @ F @ precondF]))
        X = np.array(X)
        
        clusters = KMeans(n_clusters=num_clusters, random_state=rng_seed).fit(X)
        cluster_centers = clusters.cluster_centers_
        
        kmeans_obj = 1 / len(samples) * np.sqrt(clusters.inertia_)
        n = X.shape[1]
        
        out_centers = []
        for j in range(num_clusters):
            curr_cluster = X[j]
            currG_vec = curr_cluster[:n-F_idx_cutoff]
            currF = curr_cluster[n-F_idx_cutoff:]
            out_centers.append((
                precondinvG @ symm_vec_to_mat(currG_vec) @ precondinvG,
                precondinvF @ currF @ precondinvF
            ))
        
        return out_centers, clusters.labels_, kmeans_obj

    def extract_solution(self):
        """Extract solution from solved problem (to be implemented by subclasses)."""
        raise NotImplementedError

    def extract_dro_feas_sol_from_mro(self, eps=0.1, alpha=0.1):
        """Extract DRO feasible solution from MRO solution."""
        if self.measure == 'expectation':
            return self.extract_feas_sol_mro_expectation(eps)
        elif self.measure == 'cvar':
            return self.extract_feas_sol_mro_cvar(eps, alpha)

    def extract_feas_sol_mro_expectation(self, eps):
        """Extract feasible solution for expectation measure."""
        mro_sol = self.extract_solution()
        full_samples = self.full_samples
        
        lambd = mro_sol['lambda']
        y = mro_sol['y']
        Gz = mro_sol['Gz']
        Fz = mro_sol['Fz']
        H = mro_sol['H']
        c = self.c_vals
        
        m_psd = len(self.PSD_c_vals)
        mro_obj = 0
        
        for i in range(len(full_samples)):
            Gz_i, Fz_i = full_samples[i]
            cluster_label = self.cluster_labels[i]
            Gz_k = Gz[cluster_label]
            Fz_k = Fz[cluster_label]
            y_k = y[cluster_label]
            s_i = -c @ y_k - np.trace(Gz_k @ Gz_i) - Fz_i @ Fz_k
            for m in range(m_psd):
                s_i -= np.trace(self.PSD_c_vals[m] @ H[cluster_label][m])
            mro_obj += s_i
        
        return lambd * eps + 1 / len(full_samples) * mro_obj

    def extract_feas_sol_mro_cvar(self, eps, alpha):
        """Extract feasible solution for CVaR measure."""
        mro_sol = self.extract_solution()
        full_samples = self.full_samples
        
        lambd = mro_sol['lambda']
        t = mro_sol['t']
        y1 = mro_sol['y1']
        y2 = mro_sol['y2']
        Gz1 = mro_sol['Gz1']
        Gz2 = mro_sol['Gz2']
        Fz1 = mro_sol['Fz1']
        Fz2 = mro_sol['Fz2']
        H1 = mro_sol['H1']
        H2 = mro_sol['H2']
        
        c = self.c_vals
        m_psd = len(self.PSD_c_vals)
        mro_obj = 0
        
        for i in range(len(full_samples)):
            Gz_i, Fz_i = full_samples[i]
            cluster_label = self.cluster_labels[i]
            
            Gz1_k = Gz1[cluster_label]
            Fz1_k = Fz1[cluster_label]
            y1_k = y1[cluster_label]
            
            Gz2_k = Gz2[cluster_label]
            Fz2_k = Fz2[cluster_label]
            y2_k = y2[cluster_label]
            
            s1_i = t - c @ y1_k - np.trace(Gz1_k @ Gz_i) - Fz_i @ Fz1_k
            for m in range(m_psd):
                s1_i -= np.trace(self.PSD_c_vals[m] @ H1[cluster_label][m])
            
            s2_i = -(1 / alpha - 1) * t - c @ y2_k - np.trace(Gz2_k @ Gz_i) - Fz_i @ Fz2_k
            for m in range(m_psd):
                s2_i -= np.trace(self.PSD_c_vals[m] @ H2[cluster_label][m])
            
            mro_obj += max(s1_i, s2_i)
        
        return lambd * eps + 1 / len(full_samples) * mro_obj


def symm_vectorize(A, scale_factor=np.sqrt(2)):
    """Convert symmetric matrix to vector form."""
    n = A.shape[0]
    rows, cols = np.tril_indices(n)
    
    A_vec = A[rows, cols]
    off_diag_mask = rows != cols
    A_vec[off_diag_mask] *= scale_factor
    
    return A_vec


def symm_vec_to_mat(v, scale_factor=np.sqrt(2)):
    """Convert vector form back to symmetric matrix."""
    vec_shape = v.shape[0]
    n = int((np.sqrt(8 * vec_shape + 1) - 1) / 2)
    S = np.zeros((n, n))
    
    S[np.tril_indices(n)] = v / scale_factor
    S = S + S.T
    S[range(n), range(n)] *= scale_factor / 2
    
    return S
