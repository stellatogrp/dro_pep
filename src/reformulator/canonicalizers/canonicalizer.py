import numpy as np
import scipy.sparse as spa

from PEPit.tools.expressions_to_matrices import expression_to_matrices


class Canonicalizer(object):

    def __init__(self, pep_problem, samples, measure, wrapper):
        self.pep_problem = pep_problem
        self.samples = samples
        self.measure = measure
        self.wrapper = wrapper

        self.extract_pep_data()

    def extract_pep_data(self):
        problem = self.pep_problem
        A_obj, b_obj, _ = expression_to_matrices(problem.objective)
        self.A_obj = A_obj
        self.b_obj = b_obj

        A_vals = []
        b_vals = []
        c_vals = []
        for constr in problem._list_of_constraints_sent_to_wrapper:
            A_cons, b_cons, c_cons = expression_to_matrices(constr.expression)

            A_vals.append(A_cons)
            b_vals.append(b_cons)
            c_vals.append(c_cons)

            if constr.equality_or_inequality == 'equality':
                # raise NotImplementedError # TODO add the extra constraints for the double sided inequalities
                A_vals.append(-A_cons)
                b_vals.append(-b_cons)
                c_vals.append(-c_cons)
        self.A_vals = np.array(A_vals)
        self.b_vals = np.array(b_vals)
        self.c_vals = np.array(c_vals)

        assert self.A_vals.shape[0] == self.b_vals.shape[0] == self.c_vals.shape[0]    

    def setup_problem(self):
        raise NotImplementedError

    def set_eps(self, eps):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError
