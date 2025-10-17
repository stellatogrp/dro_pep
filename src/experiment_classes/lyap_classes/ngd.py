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


def gd_lyap_nobisect(mu, L, eta, n_points, samples, dro_eps, cvar_alpha=0.1):
    pass