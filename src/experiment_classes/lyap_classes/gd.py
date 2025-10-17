import numpy as np
import cvxpy as cp


def smooth_strongly_convex(repX, repG, repF, mu=0.0, L=np.inf, varG=None, varF=None) :
    
    assert mu <= L, "mu must be less than or equal to L"
    assert len(repX) == len(repG) == len(repF), "constraint on same number of points"
    n_points = len(repX) - 1 # last point is the optimal point

    idx_list = []
    A_list, b_list = [], []

    for i in range(n_points+1) :
        for j in range(n_points+1) :
            if i != j:
                xi, xj = repX[i, :], repX[j, :]
                gi, gj = repG[i, :], repG[j, :]
                fi, fj = repF[i, :], repF[j, :]

                Ai = (1 / 2) * np.outer(gj, xi - xj) + (1 / 2) * np.outer(xi - xj, gj)
                Ai += 1 / 2 / (1 - (mu / L)) * (
                    (1 / L) * np.outer(gi - gj, gi - gj)
                    + mu * np.outer(xi - xj, xi - xj)
                    - (mu / L) * np.outer(gi - gj, xi - xj)
                    - (mu / L) * np.outer(xi - xj, gi - gj)
                )
                bi = (fj - fi)

                idx_list.append((i, j))
                A_list.append(Ai)
                b_list.append(bi)

    constraints = None
    if varG is not None and varF is not None:
        constraints = []
        for (Am, bm) in zip(A_list, b_list) :
            constraints += [cp.trace(Am @ varG) + bm.T @ varF <= 0]
    
    return idx_list, A_list, b_list, constraints


# F_{mu,L} interpolation conditions
def smooth_strongly_convex_gd(repX, repG, repF, mu=0.0, L=np.inf, varG=None, varF=None) :
    
    assert mu <= L, "mu must be less than or equal to L"
    assert len(repX) == len(repG) == len(repF), "constraint on same number of points"
    n_points = len(repX) - 1

    idx_list = []
    A_list, b_list = [], []

    for i in range(n_points) :
        for j in range(n_points) : # ignore (k,s)-interpolation conditions
            # if i != j:
            # if j == i-1 or j == i+1 : # only consider (k,k-1) and (k,k+1) interpolation conditions
            if j == i+1 : # only consider (k,k+1) interpolation conditions
                xi, xj = repX[i, :], repX[j, :]
                gi, gj = repG[i, :], repG[j, :]
                fi, fj = repF[i, :], repF[j, :]

                Ai = (1 / 2) * np.outer(gj, xi - xj) + (1 / 2) * np.outer(xi - xj, gj)
                Ai += 1 / 2 / (1 - (mu / L)) * (
                    (1 / L) * np.outer(gi - gj, gi - gj)
                    + mu * np.outer(xi - xj, xi - xj)
                    - (mu / L) * np.outer(gi - gj, xi - xj)
                    - (mu / L) * np.outer(xi - xj, gi - gj)
                )
                bi = (fj - fi)

                idx_list.append((i, j))
                A_list.append(Ai)
                b_list.append(bi)

    s = n_points
    for j in range(n_points) : # only consider (s,j) interpolation conditions
        xs, xj = repX[s, :], repX[j, :]
        gs, gj = repG[s, :], repG[j, :]
        fs, fj = repF[s, :], repF[j, :]

        As = (1 / 2) * np.outer(gj, xs - xj) + (1 / 2) * np.outer(xs - xj, gj)
        As += 1 / 2 / (1 - (mu / L)) * (
            (1 / L) * np.outer(gs - gj, gs - gj)
            + mu * np.outer(xs - xj, xs - xj)
            - (mu / L) * np.outer(gs - gj, xs - xj)
            - (mu / L) * np.outer(xs - xj, gs - gj)
        )
        bs = (fj - fs)

        idx_list.append((s, j))
        A_list.append(As)
        b_list.append(bs)

    constraints = None
    if varG is not None and varF is not None:
        constraints = []
        for (Am, bm) in zip(A_list, b_list) :
            constraints += [cp.trace(Am @ varG) + bm.T @ varF <= 0]
    
    return idx_list, A_list, b_list, constraints


def gd_lyap(mu, L, eta, n_points, samples, dro_eps, cvar_alpha=0.1):
    dimG = n_points + 2 # [x0-xs, g0, ..., gn]
    dimF = n_points + 1 # [f0-fs, ..., fn-fs]
    
    # Algorithm iterates representation
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x = eyeG[0, :]  # x0
    g = eyeG[1, :]  # g0
    f = eyeF[0, :]  # f0
    xs = 0.0 * x
    gs = 0.0 * g
    fs = 0.0 * f

    repX, repG, repF = [], [], []
    repX.append(x)      # x0
    repG.append(g)      # g0
    repF.append(f)      # f0

    for i in range(1, n_points+1):
        x = x - eta * g     # xi
        g = eyeG[i+1, :]    # gi
        f = eyeF[i, :]      # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)

    assert len(repX) == len(repG) == len(repF) == n_points + 1, "constraint on same number of points"

    repX.append(xs)
    repG.append(gs)
    repF.append(fs)
    s = len(repX) - 1   # index to the optimal point

    # Define the interpolation conditions
    # idx_list, A_list, b_list, _ = smooth_strongly_convex(np.array(repX), np.array(repG), np.array(repF), mu, L)
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)
    print(idx_list)

    # Initial condition: <A0, G> + b0.T @ F + c0 <= 0
    # A0 = np.outer(repX[0] - xs, repX[0] - xs)
    # b0 = 0.0 * fs
    # # c0 = - 1.0
    # c0 = - 400

    # Primal objective function (performance metric)
    Aobj = np.outer(repX[n_points] - xs, repX[n_points] - xs)
    # Aobj = 0.0 * Aobj

    bobj = repF[n_points] - fs
    bobj = 0.0 * bobj

    # Corresponding dual variables to primal constriants
    # lmbd = cp.Variable(len(A_list), nonneg=True)    # Interpolation conditions
    lambd = cp.Variable(nonneg=True)
    t = cp.Variable()

    N = len(samples)
    MAT_SIZE = 3
    VEC_SIZE = 2
    alpha_inv = 1 / cvar_alpha

    s = cp.Variable(N)

    X = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    Y = [cp.Variable(bobj.shape) for _ in range(N)]
    Xtilde = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    Ytilde = [cp.Variable(bobj.shape) for _ in range(N)]

    y = [cp.Variable(len(A_list), nonneg=True) for _ in range(N)]
    ytilde = [cp.Variable(len(A_list), nonneg=True) for _ in range(N)]

    Q_mat = cp.Variable(Aobj.shape, symmetric=True)
    Q_vec = cp.Variable(bobj.shape)

    Qplus_mat = cp.Variable(Aobj.shape, symmetric=True)
    Qplus_vec = cp.Variable(bobj.shape)

    rho = cp.Parameter()

    constraints = [1 / N * cp.sum(s) <= 0]

    # Lyapunov function ansatz
    Q_param_mat = cp.Variable((2, 2), symmetric=True)
    Q_param_vec = cp.Variable()
    constraints += [
        Q_mat == np.vstack([repX[0]-repX[-1], repG[0]]).T @ Q_param_mat @ np.vstack([repX[0]-repX[-1], repG[0]]),
        Q_vec == Q_param_vec * (repF[0] - repF[-1]),
        Qplus_mat == np.vstack([repX[1]-repX[-1], repG[1]]).T @ Q_param_mat @ np.vstack([repX[1]-repX[-1], repG[1]]),
        Qplus_vec == Q_param_vec * (repF[1] - repF[-1]),
    ] # had to use index '-1' instead of 's' as it overlaps with cvxpy variable 's'

    for i in range(N):
        Gi, Fi = samples[i]
        normalizer = Gi[0, 0]
        Gi = Gi / normalizer
        Fi = Fi / normalizer
        
        S_star_yA = 0
        S_star_yb = 0
        S_star_ytildeA = 0
        S_star_ytildeb = 0

        for j in range(len(A_list)):
            S_star_yA += y[i][j] * A_list[j]
            S_star_yb += y[i][j] * b_list[j]

            S_star_ytildeA = ytilde[i][j] * A_list[j]
            S_star_ytildeb = ytilde[i][j] * b_list[j]
        
        constraints += [
            (1 - alpha_inv) * t - (cp.trace(X[i] @ Gi) + Y[i] @ Fi) + lambd * dro_eps <= s[i],
            t - (cp.trace(Xtilde[i] @ Gi) + Ytilde[i] @ Fi) + lambd * dro_eps <= s[i],
            cp.SOC(lambd, cp.hstack([cp.vec(X[i], order='C'), Y[i]])),
            cp.SOC(lambd, cp.hstack([cp.vec(Xtilde[i], order='C'), Ytilde[i]])),
            # Q_mat >> Aobj,
            # Q_vec >= bobj,
            Q_mat == Aobj,
            Q_vec == bobj,
        ]

        constraints += [
            S_star_yA - X[i] - alpha_inv * (Qplus_mat - rho * Q_mat) >> 0,
            S_star_yb - Y[i] - alpha_inv * (Qplus_vec - rho * Q_vec) == 0,
            S_star_ytildeA - Xtilde[i] >> 0,
            S_star_ytildeb - Ytilde[i] == 0,
        ]

    # OLD VERSION BELOW
    # X0 = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    # Y0 = [cp.Variable(bobj.shape) for _ in range(N)]
    # X1 = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    # Y1 = [cp.Variable(bobj.shape) for _ in range(N)]
    # X2 = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    # Y2 = [cp.Variable(bobj.shape) for _ in range(N)]
    # X3 = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    # Y3 = [cp.Variable(bobj.shape) for _ in range(N)]

    # y0 =[cp.Variable(len(A_list), nonneg=True) for _ in range(N)]
    # y1 =[cp.Variable(len(A_list), nonneg=True) for _ in range(N)]
    # y2 =[cp.Variable(len(A_list), nonneg=True) for _ in range(N)]
    # y3 =[cp.Variable(len(A_list), nonneg=True) for _ in range(N)]

    # Q0_mat = cp.Variable(Aobj.shape, symmetric=True)
    # Q0_vec = cp.Variable(bobj.shape)

    # Q1_mat = cp.Variable(Aobj.shape, symmetric=True)
    # Q1_vec = cp.Variable(bobj.shape)

    # R = cp.Variable(Aobj.shape, symmetric=True)
    # r = cp.Variable(bobj.shape, nonneg=True)
    # # r = cp.Variable(bobj.shape)

    # rho = cp.Parameter()

    # constraints = [1 / N * cp.sum(s) <= 0, R >> 0]

    # for i in range(N):
    #     Gi, Fi = samples[i]

    #     S_star_y0A = 0
    #     S_star_y0b = 0
    #     S_star_y1A = 0
    #     S_star_y1b = 0
    #     S_star_y2A = 0
    #     S_star_y2b = 0
    #     S_star_y3A = 0
    #     S_star_y3b = 0

    #     for j in range(len(A_list)):
    #         S_star_y0A += y0[i][j] * A_list[j]
    #         S_star_y0b += y0[i][j] * b_list[j]

    #         S_star_y1A += y1[i][j] * A_list[j]
    #         S_star_y1b += y1[i][j] * b_list[j]

    #         S_star_y2A += y2[i][j] * A_list[j]
    #         S_star_y2b += y2[i][j] * b_list[j]

    #         S_star_y3A += y3[i][j] * A_list[j]
    #         S_star_y3b += y3[i][j] * b_list[j]

    #     constraints += [
    #         t - (cp.trace(X0[i] @ Gi) + Y0[i] @ Fi) + lambd * dro_eps <= s[i],
    #         (1 - alpha_inv) * t - (cp.trace(X1[i] @ Gi) + Y1[i] @ Fi) + lambd * dro_eps <= s[i],
    #         (1 - alpha_inv) * t - (cp.trace(X2[i] @ Gi) + Y2[i] @ Fi) + lambd * dro_eps <= s[i],
    #         (1 - alpha_inv) * t - (cp.trace(X3[i] @ Gi) + Y3[i] @ Fi) + lambd * dro_eps <= s[i],
    #         cp.SOC(lambd, cp.hstack([cp.vec(X0[i], order='C'), Y0[i]])),
    #         cp.SOC(lambd, cp.hstack([cp.vec(X1[i], order='C'), Y1[i]])),
    #         cp.SOC(lambd, cp.hstack([cp.vec(X2[i], order='C'), Y2[i]])),
    #         cp.SOC(lambd, cp.hstack([cp.vec(X3[i], order='C'), Y3[i]]))
    #     ]

    #     constraints += [
    #         S_star_y0A - X0[i] >> 0,
    #         S_star_y0b - Y0[i] == 0,
    #         S_star_y1A - X1[i] - alpha_inv * (Aobj - Q1_mat) >> 0,
    #         S_star_y1b - Y1[i] - alpha_inv * (bobj - Q1_vec) == 0,
    #         S_star_y2A - X2[i] - alpha_inv * (Q1_mat - rho * Q0_mat + R) >> 0,
    #         S_star_y2b - Y2[i] - alpha_inv * (Q1_vec - rho * Q0_vec + r) == 0,
    #         S_star_y3A - X3[i] - alpha_inv * (Q0_mat - A0) >> 0,
    #         S_star_y3b - Y3[i] - alpha_inv * (Q0_vec - b0) == 0,
    #     ]

    binary_search_iters = 10
    rho_lo = 0
    rho_hi = 1
    for i in range(binary_search_iters):
        mid = (rho_lo + rho_hi) / 2
        rho.value = mid
        obj = cp.Minimize(0)
        prob = cp.Problem(obj, constraints)
        # res = prob.solve(solver=cp.MOSEK, verbose=False)
        res = prob.solve(solver=cp.CLARABEL, verbose=False)
        print('i:', i, 'rho:', rho.value, 'res:', res)

        # print(Q0_mat.value, Q0_vec.value)
        # print(Q1_mat.value, Q1_vec.value)
        print(Q_mat.value, Q_vec.value)
        print(Qplus_mat.value, Qplus_vec.value)

        if res == 0.0:
            print('decreasing rho')
            rho_hi = mid
        else:
            print('increasing rho')
            rho_lo = mid

    return rho_hi

def gd_lyap_nobisect(mu, L, eta, n_points, samples, dro_eps, cvar_alpha=0.1):
    dimG = n_points + 2 # [x0-xs, g0, ..., gn]
    dimF = n_points + 1 # [f0-fs, ..., fn-fs]
    
    # Algorithm iterates representation
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x = eyeG[0, :]  # x0
    g = eyeG[1, :]  # g0
    f = eyeF[0, :]  # f0
    xs = 0.0 * x
    gs = 0.0 * g
    fs = 0.0 * f

    repX, repG, repF = [], [], []
    repX.append(x)      # x0
    repG.append(g)      # g0
    repF.append(f)      # f0

    for i in range(1, n_points+1):
        x = x - eta * g     # xi
        g = eyeG[i+1, :]    # gi
        f = eyeF[i, :]      # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)

    assert len(repX) == len(repG) == len(repF) == n_points + 1, "constraint on same number of points"

    repX.append(xs)
    repG.append(gs)
    repF.append(fs)
    s = len(repX) - 1   # index to the optimal point

    # Define the interpolation conditions
    # idx_list, A_list, b_list, _ = smooth_strongly_convex(np.array(repX), np.array(repG), np.array(repF), mu, L)
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)
    print(idx_list)

    # Primal objective function (performance metric)
    # Aobj = np.outer(repX[n_points] - xs, repX[n_points] - xs)
    # # Aobj = 0.0 * Aobj
    # bobj = repF[n_points] - fs
    # bobj = 0.0 * bobj

    Aobj = np.outer(repX[0] - xs, repX[0] - xs)
    bobj = repF[0] - fs
    
    Aobj = 0.0 * Aobj
    # bobj = 0.0 * bobj

    Aobj_plus = np.outer(repX[n_points] - xs, repX[n_points] - xs)
    bobj_plus = repF[n_points] - fs

    Aobj_plus = 0.0 * Aobj_plus
    # bobj_plus = 0.0 * bobj_plus

    # print(Aobj, bobj)
    # print(Aobj_plus, bobj_plus)

    # Corresponding dual variables to primal constriants
    # lmbd = cp.Variable(len(A_list), nonneg=True)    # Interpolation conditions
    lambd = cp.Variable(nonneg=True)
    t = cp.Variable()

    N = len(samples)
    MAT_SIZE = 3
    VEC_SIZE = 2
    alpha_inv = 1 / cvar_alpha

    s = cp.Variable(N)

    X = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    Y = [cp.Variable(bobj.shape) for _ in range(N)]
    Xtilde = [cp.Variable(Aobj.shape, symmetric=True) for _ in range(N)]
    Ytilde = [cp.Variable(bobj.shape) for _ in range(N)]

    y = [cp.Variable(len(A_list), nonneg=True) for _ in range(N)]
    ytilde = [cp.Variable(len(A_list), nonneg=True) for _ in range(N)]

    # Q_mat = cp.Variable(Aobj.shape, symmetric=True)
    # Q_vec = cp.Variable(bobj.shape)

    # Qplus_mat = cp.Variable(Aobj.shape, symmetric=True)
    # Qplus_vec = cp.Variable(bobj.shape)

    # rho = cp.Parameter()
    rho = cp.Variable()

    constraints = [1 / N * cp.sum(s) <= 0]

    for i in range(N):
        Gi, Fi = samples[i]
        normalizer = Gi[0, 0]
        Gi = Gi / normalizer
        Fi = Fi / normalizer
        
        S_star_yA = 0
        S_star_yb = 0
        S_star_ytildeA = 0
        S_star_ytildeb = 0

        for j in range(len(A_list)):
            S_star_yA += y[i][j] * A_list[j]
            S_star_yb += y[i][j] * b_list[j]

            S_star_ytildeA = ytilde[i][j] * A_list[j]
            S_star_ytildeb = ytilde[i][j] * b_list[j]

        constraints += [
            (1 - alpha_inv) * t - (cp.trace(X[i] @ Gi) + Y[i] @ Fi) + lambd * dro_eps <= s[i],
            t - (cp.trace(Xtilde[i] @ Gi) + Ytilde[i] @ Fi) + lambd * dro_eps <= s[i],
            cp.SOC(lambd, cp.hstack([cp.vec(X[i], order='C'), Y[i]])),
            cp.SOC(lambd, cp.hstack([cp.vec(Xtilde[i], order='C'), Ytilde[i]])),
            S_star_yA - X[i] - alpha_inv * (Aobj_plus - rho * Aobj) >> 0,
            S_star_yb - Y[i] - alpha_inv * (bobj_plus - rho * bobj) == 0,
            S_star_ytildeA - Xtilde[i] >> 0,
            S_star_ytildeb - Ytilde[i] == 0,
        ]

    obj = cp.Minimize(rho)
    prob = cp.Problem(obj, constraints)
    # res = prob.solve(solver=cp.MOSEK, verbose=False)
    res = prob.solve(solver=cp.CLARABEL, verbose=False)
    print('rho:', rho.value)

    return rho.value
