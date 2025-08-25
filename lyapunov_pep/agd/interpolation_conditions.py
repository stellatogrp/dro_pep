import numpy as np
import cvxpy as cp


# F_{mu,L} interpolation conditions
def smooth_strongly_convex(repX, repG, repF, mu=0.0, L=np.inf, varG=None, varF=None) :
    
    assert mu <= L, "mu must be less than or equal to L"
    assert len(repX) == len(repG) == len(repF), "constraint on same number of points"
    n_points = len(repX)

    idx_list = []
    A_list, b_list = [], []

    for i in range(n_points) :
        for j in range(n_points) :
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
def smooth_strongly_convex_agd(repX, repG, repF, mu=0.0, L=np.inf, varG=None, varF=None) :
    
    assert mu <= L, "mu must be less than or equal to L"
    assert len(repX) == len(repG) == len(repF), "constraint on same number of points"
    n_points = len(repX)

    idx_list = []
    A_list, b_list = [], []

    for i in range(n_points-1) :
        for j in range(n_points-1) : # ignore (k,s)-interpolation conditions
            # if i != j:
            # if j == i-1 or j == i+1 : # only consider (k,k-1) and (k,k+1) interpolation conditions
            # if j == i+1 : # only consider (k,k+1) interpolation conditions
            if j == i+1 or j == i + 2 : # only consider (k,k+1) interpolation conditions
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

    i = n_points-1
    for j in range(n_points-1) :
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


if __name__ == "__main__":
    print("test")