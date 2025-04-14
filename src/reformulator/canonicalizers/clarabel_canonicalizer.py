import clarabel
import numpy as np
import scipy.sparse as spa

from .canonicalizer import Canonicalizer


class ClarabelCanonicalizer(Canonicalizer):

    def __init__(self, pep_problem, samples, measure, wrapper, precond=True):
        super().__init__(pep_problem, samples, measure, wrapper, precond=precond)

    def setup_problem(self):
        if self.measure == 'expectation':
            self.setup_expectation_problem()
        if self.measure == 'cvar':
            self.setup_cvar_problem()

    def setup_expectation_problem(self):
        N = len(self.samples)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)

        H_mat_dims = []
        H_vec_dims = []
        H_rel_offsets = [0]

        C_symvecs = []
        d_vecs = []

        for m_psd in range(M_psd):
            H_shape = self.PSD_shapes[0]
            dim = H_shape[0]
            svec_dim = int(dim * (dim + 1) / 2)
            H_mat_dims.append(dim)
            H_vec_dims.append(svec_dim)

            # the offsets array tells us how far to look in the x vector to skip the previous m_psd vars
            H_rel_offsets.append(H_rel_offsets[-1] + N * svec_dim)

            rows, cols = get_triangular_idx(dim)

            curr_C_symvecs = []
            curr_d_vecs = []

            for r, c in zip(rows, cols):
                if r == c:
                    curr_C_symvecs.append(symm_vectorize(self.PSD_A_vals[m_psd][r][c], np.sqrt(2.)))
                    curr_d_vecs.append(self.PSD_b_vals[m_psd][r][c])
                else:
                    curr_C_symvecs.append(2 * symm_vectorize(self.PSD_A_vals[m_psd][r][c], np.sqrt(2.)))
                    curr_d_vecs.append(2 * self.PSD_b_vals[m_psd][r][c])
            curr_C_full = np.array(curr_C_symvecs)
            curr_d_full = np.array(curr_d_vecs)

            C_symvecs.append(curr_C_full.T)
            d_vecs.append(curr_d_full.T)

        H_vec_sum = sum(H_vec_dims)

        V = self.b_obj.shape[0]
        S_mat = self.A_obj.shape[0]
        S_vec = int(S_mat * (S_mat + 1) / 2)

        x_dim = 1 + N * (1 + M + V + S_vec + H_vec_sum)

        lambd_idx = 0
        lambd_offset = 1

        s_start = lambd_offset
        def s_idx(i):
            return lambd_offset + i
        s_offset = s_idx(N)
        # print(s_offset)

        y_start = s_offset
        def y_idx(i, j):
            return s_offset + i * M + j 
        y_offset = y_idx(N, 0)
        # print(y_offset)

        Fz_start = y_offset
        def Fz_idx(i, j):
            return y_offset + i * V + j
        Fz_offset = Fz_idx(N, 0)
        # print(Fz_offset)

        Gz_start = Fz_offset
        def Gz_idx(i, j):
            return Fz_offset + i * S_vec + j
        Gz_offset = Gz_idx(N, 0)
        # print(Gz_offset)

        def H_idx(m_psd, i, j):
            if i == 0 and j == 0:
                return Gz_offset + H_rel_offsets[m_psd]
            else:
                return Gz_offset + H_rel_offsets[m_psd] + i * H_vec_dims[m_psd] + j
        H_full_offset = H_idx(M_psd, 0, 0)

        assert H_full_offset == x_dim

        c = self.c_vals

        A = []
        b = []
        cones = []
        # constraints: -c^T yi - Tr(G_sample @ Gz_i) - F_sample @ Fz_i - s_i <= 0
        epi_constr = np.zeros((N, x_dim))
        for i in range(N):
            G_sample, F_sample = self.samples[i]
            y_start, y_end = y_idx(i, 0), y_idx(i, M)
            epi_constr[i, y_start: y_end] = -c
            epi_constr[i, s_idx(i)] = -1

            Fz_start, Fz_end = Fz_idx(i, 0), Fz_idx(i, V)
            epi_constr[i, Fz_start: Fz_end] = -F_sample

            Gz_start, Gz_end = Gz_idx(i, 0), Gz_idx(i, S_vec)
            epi_constr[i, Gz_start: Gz_end] = -symm_vectorize(G_sample, 2)

        A.append(spa.csc_matrix(epi_constr))
        b.append(np.zeros(N))
        # cones.append(clarabel.NonnegativeConeT(N)) # coalescing with next nonneg cone

        # constraints: yi >= 0
        y_nonneg = np.zeros((N * M, x_dim))
        y_start = y_idx(0, 0)
        y_end = y_idx(N-1, M)
        y_nonneg[0: N*M, y_start: y_end] = -np.eye(N * M)

        A.append(spa.csc_matrix(y_nonneg))
        b.append(np.zeros(N * M))
        # cones.append(clarabel.NonnegativeConeT(N * M))
        cones.append(clarabel.NonnegativeConeT(N + N * M)) # coalesce from above ineq constraints

        # constraints: H >> 0
        for m_psd in range(M_psd):
            H_mat = H_mat_dims[m_psd]
            H_vec = H_vec_dims[m_psd]
            for i in range(N):
                H_start, H_end = H_idx(m_psd, i, 0), H_idx(m_psd, i, H_vec)
                H_psd_constr = np.zeros((H_vec, x_dim))
                scaledI = scaled_off_triangles(np.ones((H_mat, H_mat)), np.sqrt(2.))
                H_psd_constr[:, H_start: H_end] = -scaledI

                A.append(spa.csc_matrix(H_psd_constr))
                b.append(np.zeros(H_vec))
            cones += [clarabel.PSDTriangleConeT(H_mat) for _ in range(N)]

        # constraints: -B^Ty_i + sum_{m_psd} \sum_{k, l} (H_i)_{k, l} d_{k, l} + Fz_i = -b_obj
        Bm_full = np.array(self.b_vals)
        Bm_T = Bm_full.T

        yB_lhs = []
        yB_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((V, x_dim))
            y_start, y_end = y_idx(i, 0), y_idx(i, M)
            curr_lhs[:, y_start: y_end] = -Bm_T

            for m_psd in range(M_psd):
                H_vec = H_vec_dims[m_psd]
                H_start, H_end = H_idx(m_psd, i, 0), H_idx(m_psd, i, H_vec)
                curr_lhs[:, H_start: H_end] = d_vecs[m_psd]

            Fz_start, Fz_end = Fz_idx(i, 0), Fz_idx(i, V)
            curr_lhs[:, Fz_start: Fz_end] = np.eye(V)

            yB_lhs.append(spa.csc_matrix(curr_lhs))
            yB_rhs.append(-self.b_obj)
        
        yB_lhs = spa.vstack(yB_lhs)
        yB_rhs = np.hstack(yB_rhs)

        A.append(yB_lhs)
        b.append(yB_rhs)
        cones.append(clarabel.ZeroConeT(V * N))

        # PSD constraints
        Aobj_svec = symm_vectorize(self.A_obj, np.sqrt(2.))
        Am_svec = [symm_vectorize(self.A_vals[m], np.sqrt(2.)) for m in range(M)]
        Am_full = np.array(Am_svec)
        Am_T = Am_full.T

        yA_lhs = []
        yA_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((S_vec, x_dim))
            y_start, y_end = y_idx(i, 0), y_idx(i, M)
            curr_lhs[:, y_start: y_end] = -Am_T

            for m_psd in range(M_psd):
                H_vec = H_vec_dims[m_psd]
                H_start, H_end = H_idx(m_psd, i, 0), H_idx(m_psd, i, H_vec)
                curr_lhs[:, H_start: H_end] = C_symvecs[m_psd]

            Gz_start, Gz_end = Gz_idx(i, 0), Gz_idx(i, S_vec)
            scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.))
            curr_lhs[:, Gz_start: Gz_end] = scaledI # need to scale the off-triangles

            yA_lhs.append(spa.csc_matrix(curr_lhs))
            yA_rhs.append(-Aobj_svec)

        yA_lhs = spa.vstack(yA_lhs)
        yA_rhs = np.hstack(yA_rhs)

        A.append(yA_lhs)
        b.append(yA_rhs)
        cones += [clarabel.PSDTriangleConeT(S_mat) for _ in range(N)]

        # SOCP constraints
        G_precond_vec = self.preconditioner[0]
        F_precond = self.preconditioner[1]

        F_precond_sq = F_precond ** 2
        scaled_G_vec_outer_prod = np.outer(G_precond_vec, G_precond_vec)
        scaledG_mult = scaled_off_triangles(scaled_G_vec_outer_prod, np.sqrt(2.))

        socp_lhs = []
        socp_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((1 + V + S_vec, x_dim))
            Fz_start, Fz_end = Fz_idx(i, 0), Fz_idx(i, V)
            Gz_start, Gz_end = Gz_idx(i, 0), Gz_idx(i, S_vec)

            # scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.))
            # curr_lhs[0, lambd_idx] = -1
            # curr_lhs[1: V + 1, Fz_start: Fz_end] = -np.eye(V)
            # curr_lhs[V + 1:, Gz_start: Gz_end] = -scaledI

            curr_lhs[0, lambd_idx] = -1
            curr_lhs[1: V + 1, Fz_start: Fz_end] = -np.diag(F_precond_sq)
            curr_lhs[V + 1:, Gz_start: Gz_end] = -scaledG_mult

            socp_lhs.append(spa.csc_matrix(curr_lhs))
            socp_rhs.append(np.zeros(1 + V + S_vec))
        
        socp_lhs = spa.vstack(socp_lhs)
        socp_rhs = np.hstack(socp_rhs)

        A.append(socp_lhs)
        b.append(socp_rhs)
        cones += [clarabel.SecondOrderConeT(1 + V + S_vec) for _ in range(N)]

        A = spa.vstack(A)
        b = np.hstack(b)

        self.A = A
        self.b = b
        self.cones = cones
        self.P = spa.csc_matrix((x_dim, x_dim))
        self.s_idx_func = s_idx
        self.y_idx_func = y_idx
        self.Fz_idx_func = Fz_idx
        self.Gz_idx_func = Gz_idx
        self.H_idx_func = H_idx

    def setup_cvar_problem(self):
        # sets up the part of the cvar problem that does not depend on alpha
        N = len(self.samples)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)

        H_mat_dims = []
        H_vec_dims = []
        H_rel_offsets = [0]

        # print(self.PSD_A_vals[0].shape)
        # print(self.PSD_b_vals[0].shape)
        # print(self.PSD_shapes[0])

        C_symvecs = []
        d_vecs = []

        for m_psd in range(M_psd):
            H_shape = self.PSD_shapes[0]
            dim = H_shape[0]
            svec_dim = int(dim * (dim + 1) / 2)
            H_mat_dims.append(dim)
            H_vec_dims.append(svec_dim)

            # the offsets array tells us how far to look in the x vector to skip the previous m_psd vars
            H_rel_offsets.append(H_rel_offsets[-1] + N * svec_dim)

            rows, cols = get_triangular_idx(dim)

            curr_C_symvecs = []
            curr_d_vecs = []

            for r, c in zip(rows, cols):
                if r == c:
                    curr_C_symvecs.append(symm_vectorize(self.PSD_A_vals[m_psd][r][c], np.sqrt(2.)))
                    curr_d_vecs.append(self.PSD_b_vals[m_psd][r][c])
                else:
                    curr_C_symvecs.append(2 * symm_vectorize(self.PSD_A_vals[m_psd][r][c], np.sqrt(2.)))
                    curr_d_vecs.append(2 * self.PSD_b_vals[m_psd][r][c])
            curr_C_full = np.array(curr_C_symvecs)
            curr_d_full = np.array(curr_d_vecs)

            C_symvecs.append(curr_C_full.T)
            d_vecs.append(curr_d_full.T)

        self.C_symvecs = C_symvecs
        self.d_vecs = d_vecs

        H_vec_sum = sum(H_vec_dims)

        V = self.b_obj.shape[0]
        S_mat = self.A_obj.shape[0]
        S_vec = int(S_mat * (S_mat + 1) / 2)

        # print('N, M, V, S_vec, H_vec_sum:', N, M, V, S_vec, H_vec_sum)

        x_dim = 2 + N * (1 + 2 * (M + V + S_vec + H_vec_sum))
        # print('x_dim:', x_dim)

        lambd_idx = 0
        # lambd_offset = 1

        t_idx = 1
        t_offset = 2

        s_start = t_offset
        def s_idx(i):
            return s_start + i
        s_offset = s_idx(N)

        y1_start = s_offset
        def y1_idx(i, j):
            return y1_start + i * M + j
        y1_offset = y1_idx(N, 0)

        y2_start = y1_offset
        def y2_idx(i, j):
            return y2_start + i * M + j
        y2_offset = y2_idx(N, 0)

        fz1_start = y2_offset
        def fz1_idx(i, j):
            return fz1_start + i * V + j
        fz1_offset = fz1_idx(N, 0)

        fz2_start = fz1_offset
        def fz2_idx(i, j):
            return fz2_start + i * V + j
        fz2_offset = fz2_idx(N, 0)

        Gz1_start = fz2_offset
        def Gz1_idx(i, j):
            return Gz1_start + i * S_vec + j
        Gz1_offset = Gz1_idx(N, 0)

        Gz2_start = Gz1_offset
        def Gz2_idx(i, j):
            return Gz2_start + i * S_vec + j
        Gz2_offset = Gz2_idx(N, 0)

        def H1_idx(m_psd, i, j):
            if i == 0 and j == 0:
                return Gz2_offset + H_rel_offsets[m_psd]
            else:
                return Gz2_offset + H_rel_offsets[m_psd] + i * H_vec_dims[m_psd] + j
        H1_offset = H1_idx(M_psd, 0, 0)

        def H2_idx(m_psd, i, j):
            if i == 0 and j == 0:
                return H1_offset + H_rel_offsets[m_psd]
            else:
                return H1_offset + H_rel_offsets[m_psd] + i * H_vec_dims[m_psd] + j
        H2_offset = H2_idx(M_psd, 0, 0)

        assert H2_offset == x_dim

        c = self.c_vals
        A = []
        b = []
        cones = []

        # constraints: y1 >= 0, y2 >= 0
        y1_nonneg = np.zeros((N * M, x_dim))
        y1_idx_start = y1_idx(0, 0)
        y1_idx_end = y1_idx(N-1, M)
        y1_nonneg[0: N*M, y1_idx_start: y1_idx_end] = -np.eye(N * M)

        y2_nonneg = np.zeros((N * M, x_dim))
        y2_idx_start = y2_idx(0, 0)
        y2_idx_end = y2_idx(N-1, M)
        y2_nonneg[0: N*M, y2_idx_start: y2_idx_end] = -np.eye(N * M)

        A += [spa.csc_matrix(y1_nonneg), spa.csc_matrix(y2_nonneg)]
        b += [np.zeros(2 * N * M)]
        cones += [clarabel.NonnegativeConeT(2 * N * M)]

        # constraints: H1 >> 0, H2 >> 0
        for m_psd in range(M_psd):
            H_mat = H_mat_dims[m_psd]
            H_vec = H_vec_dims[m_psd]
            for i in range(N):
                H1_start, H1_end = H1_idx(m_psd, i, 0), H1_idx(m_psd, i, H_vec)
                H1_psd_constr = np.zeros((H_vec, x_dim))
                scaledI = scaled_off_triangles(np.ones((H_mat, H_mat)), np.sqrt(2.))
                H1_psd_constr[:, H1_start: H1_end] = -scaledI

                H2_start, H2_end = H2_idx(m_psd, i, 0), H2_idx(m_psd, i, H_vec)
                H2_psd_constr = np.zeros((H_vec, x_dim))
                H2_psd_constr[:, H2_start: H2_end] = -scaledI

                A += [spa.csc_matrix(H1_psd_constr), spa.csc_matrix(H2_psd_constr)]
                b.append(np.zeros(2 * H_vec))
            cones += [clarabel.PSDTriangleConeT(H_mat) for _ in range(2 * N)]

        # SOCP constraints
        G_precond_vec = self.preconditioner[0]
        F_precond = self.preconditioner[1]

        F_precond_sq = F_precond ** 2
        scaled_G_vec_outer_prod = np.outer(G_precond_vec, G_precond_vec)
        scaledG_mult = scaled_off_triangles(scaled_G_vec_outer_prod, np.sqrt(2.))

        socp_lhs = []
        socp_rhs = []
        scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.))
        for i in range(N):
            # Gz1, fz1
            curr_lhs = np.zeros((1 + V + S_vec, x_dim))
            fz1_idx_start, fz1_idx_end = fz1_idx(i, 0), fz1_idx(i, V)
            Gz1_idx_start, Gz1_idx_end = Gz1_idx(i, 0), Gz1_idx(i, S_vec)

            curr_lhs[0, lambd_idx] = -1
            # curr_lhs[1: V+1, fz1_idx_start: fz1_idx_end] = -np.eye(V)
            # curr_lhs[V+1:, Gz1_idx_start: Gz1_idx_end] = -scaledI
            curr_lhs[1: V+1, fz1_idx_start: fz1_idx_end] = -np.diag(F_precond_sq)
            curr_lhs[V+1:, Gz1_idx_start: Gz1_idx_end] = -scaledG_mult

            socp_lhs.append(spa.csc_matrix(curr_lhs))
            socp_rhs.append(np.zeros(1 + V + S_vec))

            # Gz2, fz2
            curr_lhs = np.zeros((1 + V + S_vec, x_dim))
            fz2_idx_start, fz2_idx_end = fz2_idx(i, 0), fz2_idx(i, V)
            Gz2_idx_start, Gz2_idx_end = Gz2_idx(i, 0), Gz2_idx(i, S_vec)

            curr_lhs[0, lambd_idx] = -1
            # curr_lhs[1:V+1, fz2_idx_start: fz2_idx_end] = -np.eye(V)
            # curr_lhs[V+1:, Gz2_idx_start: Gz2_idx_end] = -scaledI
            curr_lhs[1:V+1, fz2_idx_start: fz2_idx_end] = -np.diag(F_precond_sq)
            curr_lhs[V+1:, Gz2_idx_start: Gz2_idx_end] = -scaledG_mult

            socp_lhs.append(spa.csc_matrix(curr_lhs))
            socp_rhs.append(np.zeros(1 + V + S_vec))

        socp_lhs = spa.vstack(socp_lhs)
        socp_rhs = np.hstack(socp_rhs)

        A.append(socp_lhs)
        b.append(socp_rhs)
        cones += [clarabel.SecondOrderConeT(1 + V + S_vec) for _ in range(2 * N)]

        # constraints: -B^Ty_i + Fz1_i = 0
        Bm_full = np.array(self.b_vals)
        Bm_T = Bm_full.T

        yB_lhs = []
        yB_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((V, x_dim))
            y1_idx_start, y1_idx_end = y1_idx(i, 0), y1_idx(i, M)
            curr_lhs[:, y1_idx_start: y1_idx_end] = -Bm_T

            for m_psd in range(M_psd):
                H_vec = H_vec_dims[m_psd]
                H1_start, H1_end = H1_idx(m_psd, i, 0), H1_idx(m_psd, i, H_vec)
                curr_lhs[:, H1_start: H1_end] = d_vecs[m_psd]

            fz1_idx_start, fz1_idx_end = fz1_idx(i, 0), fz1_idx(i, V)
            curr_lhs[:, fz1_idx_start: fz1_idx_end] = np.eye(V)

            yB_lhs.append(spa.csc_matrix(curr_lhs))
            yB_rhs.append(np.zeros(V))
        
        yB_lhs = spa.vstack(yB_lhs)
        yB_rhs = np.hstack(yB_rhs)

        A.append(yB_lhs)
        b.append(yB_rhs)
        cones.append(clarabel.ZeroConeT(V * N))

        # constraints: -A^\star y_1i + Gz1_i << 0
        Am_svec = [symm_vectorize(self.A_vals[m], np.sqrt(2.)) for m in range(M)]
        Am_full = np.array(Am_svec)
        Am_T = Am_full.T

        yA_lhs = []
        yA_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((S_vec, x_dim))
            y1_idx_start, y1_idx_end = y1_idx(i, 0), y1_idx(i, M)
            curr_lhs[:, y1_idx_start: y1_idx_end] = -Am_T

            for m_psd in range(M_psd):
                H_vec = H_vec_dims[m_psd]
                H1_start, H1_end = H1_idx(m_psd, i, 0), H1_idx(m_psd, i, H_vec)
                curr_lhs[:, H1_start: H1_end] = C_symvecs[m_psd]

            Gz1_idx_start, Gz1_idx_end = Gz1_idx(i, 0), Gz1_idx(i, S_vec)
            curr_lhs[:, Gz1_idx_start: Gz1_idx_end] = scaledI

            yA_lhs.append(spa.csc_matrix(curr_lhs))
            yA_rhs.append(np.zeros(S_vec))

        yA_lhs = spa.vstack(yA_lhs)
        yA_rhs = np.hstack(yA_rhs)

        A.append(yA_lhs)
        b.append(yA_rhs)
        cones += [clarabel.PSDTriangleConeT(S_mat) for _ in range(N)]

        # constraints: t - c^T y1 - Tr(G_sample @ Gz1_i) - F_sample @ Fz1_i - s_i <= 0
        epi1_constr = np.zeros((N, x_dim))
        for i in range(N):
            G_sample, F_sample = self.samples[i]
            epi1_constr[i, t_idx] = 1
            y1_idx_start, y1_idx_end = y1_idx(i, 0), y1_idx(i, M)
            epi1_constr[i, y1_idx_start: y1_idx_end] = -c
            epi1_constr[i, s_idx(i)] = -1

            fz1_idx_start, fz1_idx_end = fz1_idx(i, 0), fz1_idx(i, V)
            epi1_constr[i, fz1_idx_start: fz1_idx_end] = -F_sample

            Gz1_idx_start, Gz1_idx_end = Gz1_idx(i, 0), Gz1_idx(i, S_vec)
            epi1_constr[i, Gz1_idx_start: Gz1_idx_end] = -symm_vectorize(G_sample, 2)

        A.append(spa.csc_matrix(epi1_constr))
        b.append(np.zeros(N))
        cones.append(clarabel.NonnegativeConeT(N))

        A = spa.vstack(A)
        b = np.hstack(b)

        self.x_dim = x_dim
        self.H_vec_dims = H_vec_dims
        self.A_fixed = A
        self.b_fixed = b
        self.cones_fixed = cones
        self.P = spa.csc_matrix((x_dim, x_dim))

        # self.A = A
        # self.b = b
        # self.cones = cones
        # self.P = spa.csc_matrix((x_dim, x_dim))
        self.lambd_idx = lambd_idx
        self.t_idx = t_idx
        self.s_idx_func = s_idx

        self.y1_idx_func = y1_idx
        self.Fz1_idx_func = fz1_idx
        self.Gz1_idx_func = Gz1_idx
        self.H1_idx_func = H1_idx

        self.y2_idx_func = y2_idx
        self.Fz2_idx_func = fz2_idx
        self.Gz2_idx_func = Gz2_idx
        self.H2_idx_func = H2_idx

    def set_params(self, eps=0.1, alpha=0.1):
        if self.measure == 'expectation':
            self.set_expectation_eps(eps)
        if self.measure == 'cvar':
            self.set_cvar_alpha_eps(eps, alpha)

    def set_expectation_eps(self, eps):
        x_dim = self.P.shape[0]
        N = len(self.samples)
        q = np.zeros(x_dim)
        q[0] = eps
        q[self.s_idx_func(0): self.s_idx_func(N)] = 1 / N

        self.q = q

    def set_cvar_alpha_eps(self, eps, alpha):
        alpha_inv = 1 / alpha
        N = len(self.samples)
        M = len(self.A_vals)
        M_psd = len(self.PSD_A_vals)

        V = self.b_obj.shape[0]
        S_mat = self.A_obj.shape[0]
        S_vec = int(S_mat * (S_mat + 1) / 2)

        H_vec_dims = self.H_vec_dims

        d_vecs = self.d_vecs
        C_symvecs = self.C_symvecs

        Bm_full = np.array(self.b_vals)
        Bm_T = Bm_full.T

        Am_svec = [symm_vectorize(self.A_vals[m], np.sqrt(2.)) for m in range(M)]
        Am_full = np.array(Am_svec)
        Am_T = Am_full.T

        c = self.c_vals
        P = self.P
        A = []
        b = []
        cones = []

        scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.))

        x_dim = self.x_dim
        lambd_idx = self.lambd_idx
        t_idx = self.t_idx

        s_idx = self.s_idx_func
        y1_idx = self.y1_idx_func
        fz1_idx = self.Fz1_idx_func
        Gz1_idx = self.Gz1_idx_func
        H1_idx = self.H1_idx_func

        y2_idx = self.y2_idx_func
        fz2_idx = self.Fz2_idx_func
        Gz2_idx = self.Gz2_idx_func
        H2_idx = self.H2_idx_func

        # everything from here on out depends on alpha
        # constraints: -(alpha_inv - 1) t - c^T y2 - Tr(G_sample @ Gz2_i) - F_sample @ Fz2_i - s_i <= 0
        epi2_constr = np.zeros((N, x_dim))
        for i in range(N):
            G_sample, F_sample = self.samples[i]
            epi2_constr[i, t_idx] = -(alpha_inv - 1)
            y2_idx_start, y2_idx_end = y2_idx(i, 0), y2_idx(i, M)
            epi2_constr[i, y2_idx_start: y2_idx_end] = -c
            epi2_constr[i, s_idx(i)] = -1

            fz2_idx_start, fz2_idx_end = fz2_idx(i, 0), fz2_idx(i, V)
            epi2_constr[i, fz2_idx_start: fz2_idx_end] = -F_sample

            Gz2_idx_start, Gz2_idx_end = Gz2_idx(i, 0), Gz2_idx(i, S_vec)
            epi2_constr[i, Gz2_idx_start: Gz2_idx_end] = -symm_vectorize(G_sample, 2)

        A.append(spa.csc_matrix(epi2_constr))
        b.append(np.zeros(N))
        cones.append(clarabel.NonnegativeConeT(N))

        # constraints: -B^Ty_2i + Fz2_i = -alpha_inv * b_obj
        yB_lhs = []
        yB_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((V, x_dim))
            y2_idx_start, y2_idx_end = y2_idx(i, 0), y2_idx(i, M)
            curr_lhs[:, y2_idx_start: y2_idx_end] = -Bm_T

            for m_psd in range(M_psd):
                H_vec = H_vec_dims[m_psd]
                H2_start, H2_end = H2_idx(m_psd, i, 0), H2_idx(m_psd, i, H_vec)
                curr_lhs[:, H2_start: H2_end] = d_vecs[m_psd]

            fz2_idx_start, fz2_idx_end = fz2_idx(i, 0), fz2_idx(i, V)
            curr_lhs[:, fz2_idx_start: fz2_idx_end] = np.eye(V)

            yB_lhs.append(spa.csc_matrix(curr_lhs))
            yB_rhs.append(-alpha_inv * self.b_obj)
        
        yB_lhs = spa.vstack(yB_lhs)
        yB_rhs = np.hstack(yB_rhs)

        A.append(yB_lhs)
        b.append(yB_rhs)
        cones.append(clarabel.ZeroConeT(V * N))

        # constraints: -A^\star y_2i + Gz2+i << -alpha_inv * A_obj
        Aobj_svec = symm_vectorize(self.A_obj, np.sqrt(2.))

        yA_lhs = []
        yA_rhs = []
        for i in range(N):
            curr_lhs = np.zeros((S_vec, x_dim))
            y2_idx_start, y2_idx_end = y2_idx(i, 0), y2_idx(i, M)
            curr_lhs[:, y2_idx_start: y2_idx_end] = -Am_T

            for m_psd in range(M_psd):
                H_vec = H_vec_dims[m_psd]
                H2_start, H2_end = H2_idx(m_psd, i, 0), H2_idx(m_psd, i, H_vec)
                curr_lhs[:, H2_start: H2_end] = C_symvecs[m_psd]

            Gz2_idx_start, Gz2_idx_end = Gz2_idx(i, 0), Gz2_idx(i, S_vec)
            curr_lhs[:, Gz2_idx_start: Gz2_idx_end] = scaledI

            yA_lhs.append(spa.csc_matrix(curr_lhs))
            yA_rhs.append(-alpha_inv * Aobj_svec)

        yA_lhs = spa.vstack(yA_lhs)
        yA_rhs = np.hstack(yA_rhs)

        A.append(yA_lhs)
        b.append(yA_rhs)
        cones += [clarabel.PSDTriangleConeT(S_mat) for _ in range(N)]

        A = spa.vstack(A)
        b = np.hstack(b)

        print(self.cones_fixed, cones)
        exit(0)


    def solve(self):
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(self.P, self.q, self.A, self.b, self.cones, settings)
        solution = solver.solve()
        out = {
            'obj': solution.obj_val,
            'solvetime': solution.solve_time,
        }
        return out


def symm_vectorize(A, scale_factor):
    n = A.shape[0]
    rows, cols = np.tril_indices(n)  # tril is correct, need upper triangle in column order

    A_vec = A[rows, cols]
    off_diag_mask = rows != cols
    A_vec[off_diag_mask] *= scale_factor

    return A_vec


def scaled_off_triangles(A, scale_factor):
    Avec = symm_vectorize(A, scale_factor)
    return np.diag(Avec)


def get_triangular_idx(n):
    rows, cols = np.tril_indices(n)
    return cols, rows  # flip the order to get the upper triangle in column order
