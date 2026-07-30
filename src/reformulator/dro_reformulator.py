import clarabel
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa

from .canonicalizers.clarabel_canonicalizer import ClarabelCanonicalizer
from .canonicalizers.cvxpy_canonicalizer import CvxpyCanonicalizer

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


VALID_MEASURES = [
    'expectation',
    'cvar',
]

VALID_WRAPPERS = [
    'cvxpy',
    'clarabel',
]


class DROReformulator(object):
    """
    DRO reformulator that constructs and solves the DRO optimization problem.
    
    Args:
        pep_data: Tuple containing PEP constraint matrices:
            (A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
        samples: List of (G, F) tuples representing sampled Gram matrices
        measure: Risk measure ('expectation' or 'cvar')
        wrapper: Solver wrapper ('cvxpy' or 'clarabel')
        precond: Whether to apply preconditioning
        precond_type: Type of preconditioning ('average', 'max', 'min')
        mro_clusters: Number of MRO clusters (optional)
    """

    def __init__(self, pep_data, samples, measure, wrapper, precond=True, precond_type='average', mro_clusters=None):
        self.pep_data = pep_data
        self.samples = samples

        if measure not in VALID_MEASURES:
            raise NotImplementedError(f'{measure} not a valid measure')

        self.measure = measure
        self.mro_clusters = mro_clusters

        if wrapper == 'cvxpy':
            self.canon = CvxpyCanonicalizer(pep_data, samples, measure, wrapper, precond=precond, precond_type=precond_type, mro_clusters=mro_clusters)
        elif wrapper == 'clarabel':
            self.canon = ClarabelCanonicalizer(pep_data, samples, measure, wrapper, precond=precond, precond_type=precond_type, mro_clusters=mro_clusters)
        else:
            raise NotImplementedError(f'wrapper {wrapper} not implemented')

        self.canon.setup_problem()

    def set_params(self, **kwargs):
        self.canon.set_params(**kwargs)

    def solve(self):
        return self.canon.solve()

    def set_single_eps_val(self, eps):
        self.canon.set_eps(eps)

    def set_single_alpha_eps_val(self, alpha, eps):
        self.canon.set_eps_alpha_value(alpha, eps)

    def extract_solution(self):
        return self.canon.extract_solution()

    def extract_mro_diff(self):
        if self.mro_clusters is None:
            return 0
        return self.canon.extract_mro_diff()

    def extract_dro_feas_sol_from_mro(self, **kwargs):
        return self.canon.extract_dro_feas_sol_from_mro(**kwargs)
