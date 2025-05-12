import clarabel
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa

from .canonicalizers.clarabel_canonicalizer import ClarabelCanonicalizer
from .canonicalizers.cvxpy_canonicalizer import CvxpyCanonicalizer
from PEPit.tools.expressions_to_matrices import expression_to_matrices

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

    def __init__(self, pep_problem, samples, measure, wrapper, precond=True, mro_clusters=None):
        self.pep_problem = pep_problem
        self.samples = samples

        if measure not in VALID_MEASURES:
            raise NotImplementedError(f'{measure} not a valid measure')

        if pep_problem.objective is None:
            raise AssertionError('pep problem needs to be solved to extract data')

        self.measure = measure
        self.mro_clusters = mro_clusters

        if wrapper == 'cvxpy':
            self.canon = CvxpyCanonicalizer(pep_problem, samples, measure, wrapper, precond=precond, mro_clusters=mro_clusters)
        elif wrapper == 'clarabel':
            self.canon = ClarabelCanonicalizer(pep_problem, samples, measure, wrapper, precond=precond, mro_clusters=mro_clusters)
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
