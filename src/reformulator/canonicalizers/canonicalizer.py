import numpy as np
import scipy.sparse as spa

from PEPit.tools.expressions_to_matrices import expression_to_matrices
from sklearn.cluster import KMeans


class Canonicalizer(object):

    def __init__(self, pep_problem, samples, measure, wrapper, precond=True, mro_clusters=None):
        self.pep_problem = pep_problem
        self.samples = samples
        self.measure = measure
        self.wrapper = wrapper
        self.precond = precond

        if mro_clusters is None:
            self.samples_to_use = samples
        else:
            self.cluster_centers, self.cluster_labels, self.kmeans_obj = self.clustered_sample_centers(samples, mro_clusters)
            self.samples_to_use = self.cluster_centers
            self.full_samples = samples
        self.extract_pep_data()

    def extract_pep_data(self):
        problem = self.pep_problem
        A_obj, b_obj, _ = expression_to_matrices(problem._list_of_constraints_sent_to_wrapper[0].expression)

        self.A_obj = - A_obj
        self.b_obj = - b_obj[:-1]

        self.preconditioner = ( np.ones((self.A_obj.shape[0],)), np.ones(self.b_obj.shape) )
        if self.precond :
            avg_sample = (np.average([np.sqrt(np.diag(sample[0])) for sample in self.samples_to_use], axis=0), np.average([sample[1] for sample in self.samples_to_use], axis=0))
            self.preconditioner = ( 1/avg_sample[0], 1/np.sqrt(avg_sample[1]) )
            self.preconditioner[0][0] = 1.0 # avoid divide-by-zero error from g(x_star) = 0
            self.preconditioner[1][0] = 1.0 # avoid divide-by-zero error from x_star = 0

        A_vals = []
        b_vals = []
        c_vals = []
        for constr in problem._list_of_constraints_sent_to_wrapper[1:]:
            A_cons, b_cons, c_cons = expression_to_matrices(constr.expression)

            A_vals.append(A_cons)
            b_vals.append(b_cons[:-1])
            c_vals.append(c_cons)

            if constr.equality_or_inequality == 'equality':
                # raise NotImplementedError # TODO add the extra constraints for the double sided inequalities
                A_vals.append(-A_cons)
                b_vals.append(-b_cons[:-1])
                c_vals.append(-c_cons)
        self.A_vals = np.array(A_vals)
        self.b_vals = np.array(b_vals)
        self.c_vals = np.array(c_vals)

        # TODO: PSD constraint for quadratic interpolation
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

    def set_params(self, eps=0.1, alpha=0.1):
        raise NotImplementedError

    def clustered_sample_centers(self, samples, num_clusters, rng_seed=0):
        print(f'clustering into {num_clusters} clusters')

        F_idx_cutoff = samples[0][1].shape[0]

        X = []
        for samp in samples:
            G, F = samp
            X.append(np.hstack([symm_vectorize(G), F]))
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
            out_centers.append((symm_vec_to_mat(currG_vec), currF))
        return out_centers, clusters.labels_, kmeans_obj

    def extract_mro_diff(self):
        if self.measure == 'expectation':
            return self.extract_expectation_mro_diff()
        elif self.measure == 'cvar':
            return self.extract_cvar_mro_diff()
    
    def extract_expectation_mro_diff(self):
        pass

    def extract_cvar_mro_diff(self):
        solution = self.extract_solution()
        cluster_labels = self.cluster_labels
        cluster_centers = self.cluster_centers
        full_samples = self.full_samples

        # avg_G = np.average([sample[0] for sample in self.samples], axis=0)
        # avg_F = np.average([sample[1] for sample in self.samples], axis=0)

        # slacks = [( (sample[0]-avg_G, sample[1]-avg_F), \
        #             np.array([-np.trace(Am@(sample[0]-avg_G))-bm.T@(sample[1]-avg_F) for (Am, bm) in zip(self.A_vals, self.b_vals)]), \
        #             [np.array([[np.trace(Ap[i,j]@(sample[0]-avg_G))+bp[i,j].T@(sample[1]-avg_F) for j in range(Ap.shape[1])] for i in range(Ap.shape[0])]) for (Ap, bp) in zip(self.PSD_A_vals, self.PSD_b_vals)] ) \
        #               for k, sample in enumerate(full_samples)]

        slacks = [( (sample[0]-cluster_centers[cluster_labels[k]][0], sample[1]-cluster_centers[cluster_labels[k]][1]), \
                    np.array([-np.trace(Am@(sample[0]-cluster_centers[cluster_labels[k]][0]))-bm.T@(sample[1]-cluster_centers[cluster_labels[k]][1]) for (Am, bm) in zip(self.A_vals, self.b_vals)]), \
                    [np.array([[np.trace(Ap[i,j]@(sample[0]-cluster_centers[cluster_labels[k]][0]))+bp[i,j].T@(sample[1]-cluster_centers[cluster_labels[k]][1]) for j in range(Ap.shape[1])] for i in range(Ap.shape[0])]) for (Ap, bp) in zip(self.PSD_A_vals, self.PSD_b_vals)] ) \
                      for k, sample in enumerate(full_samples)]

        # print(slacks)

        y_list = [solution['y1'], solution['y2']]
        Gz_list = [solution['Gz1'], solution['Gz2']]
        Fz_list = [solution['Fz1'], solution['Fz2']]
        Gz_psd_list = [solution['H1'], solution['H2']]

        # TODO: check if the final part of the slack calculation actually uses then entire Gz_list or if just the label

        for Gz_psd in Gz_psd_list:
            print(len(Gz_psd))

        out = 0
        for i, (diff, LGF, H_list) in enumerate(slacks):
            label = cluster_labels[i]
            temp = [ - np.trace(Gz[label]@diff[0]) - Fz[label].T@diff[1] - LGF.T@y[label] - np.sum([np.trace(G_psd@H) for (G_psd, H) in zip(Gz_psd, H_list)]) for (y, Gz, Fz, Gz_psd) in zip(y_list, Gz_list, Fz_list, Gz_psd_list) ]
            out += np.max(temp)
        
        # for (diff, LGF, H_list) in slacks :
        #     temp = [ - np.trace(Gz@diff[0]) - np.array(Fz).T@diff[1] - LGF.T@y - np.sum([cp.trace(G_psd@H) for (G_psd, H) in zip(Gz_psd, H_list)]) for (y, Gz, Fz, Gz_psd) in zip(y_list, Gz_list, Fz_list, Gz_psd_list) ]
        #     out += np.max(np.vstack(temp))

        return 1 / len(full_samples) * out


def symm_vectorize(A, scale_factor=np.sqrt(2)):
    n = A.shape[0]
    rows, cols = np.tril_indices(n)  # tril is correct, need upper triangle in column order

    A_vec = A[rows, cols]
    off_diag_mask = rows != cols
    A_vec[off_diag_mask] *= scale_factor

    return A_vec


def symm_vec_to_mat(v, scale_factor=np.sqrt(2)):
    vec_shape = v.shape[0]
    n = int((np.sqrt(8 * vec_shape + 1) - 1) / 2)
    S = np.zeros((n, n))
    
    S[np.tril_indices(n)] = v / scale_factor # tril is again correct here

    S = S + S.T
    S[range(n), range(n)] *= scale_factor / 2
    return S
