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


def gd_lyap(mu, L, eta, n_points, samples, dro_eps):
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
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)

    # Initial condition: <A0, G> + b0.T @ F + c0 <= 0
    A0 = np.outer(repX[0] - xs, repX[0] - xs)
    b0 = 0.0 * fs
    # c0 = - 1.0

    # Primal objective function (performance metric)
    Aobj = np.outer(repX[n_points] - xs, repX[n_points] - xs)
    Aobj = 0.0 * Aobj
    bobj = repF[n_points] - fs
    # bobj = 0.0 * bobj

    # Corresponding dual variables to primal constriants
    # lmbd = cp.Variable(len(A_list), nonneg=True)    # Interpolation conditions
    tau = cp.Variable()                             # Initial condition
    lambd = cp.Variable()
