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

        PSD_A_vals = []
        PSD_b_vals = []
        PSD_c_vals = []
        PSD_shapes = []

        for psd in problem._list_of_psd_sent_to_wrapper:
            psd_expr = psd.matrix_of_expressions
            psd_shape = psd_expr.shape
            PSD_shapes.append(psd_shape)

            PSD_A_val, PSD_b_val, PSD_c_val = [], [], []
            for i in range(psd_shape[0]) :
                PSD_A_row, PSD_b_row, PSD_c_row = [], [], []
                for j in range(psd_shape[1]) :
                    PSD_A_cons, PSD_b_cons, PSD_c_cons = expression_to_matrices(psd_expr[i,j])
                    PSD_A_row.append(PSD_A_cons)
                    PSD_b_row.append(PSD_b_cons[:-1])
                    PSD_c_row.append(PSD_c_cons)
                PSD_A_val.append(PSD_A_row)
                PSD_b_val.append(PSD_b_row)
                PSD_c_val.append(PSD_c_row)
            PSD_A_vals.append(np.array(PSD_A_val))
            PSD_b_vals.append(np.array(PSD_b_val))
            PSD_c_vals.append(np.array(PSD_c_val))
        self.PSD_A_vals = PSD_A_vals
        self.PSD_b_vals = PSD_b_vals
        self.PSD_c_vals = PSD_c_vals
        self.PSD_shapes = PSD_shapes

        assert self.A_vals.shape[0] == self.b_vals.shape[0] == self.c_vals.shape[0]    

    def setup_problem(self):
        raise NotImplementedError

    def set_eps(self, eps):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError
