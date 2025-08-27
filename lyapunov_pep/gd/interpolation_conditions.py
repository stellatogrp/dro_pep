import numpy as np
import cvxpy as cp


# F_{mu,L} interpolation conditions
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


def smooth_strongly_convex_agd(repX, repGX, repFX, repY, repGY, repFY, mu=0.0, L=np.inf, varG=None, varF=None) :
    
    assert mu <= L, "mu must be less than or equal to L"
    assert len(repX) == len(repGX) == len(repFX), "constraint on same number of points"
    assert len(repY) == len(repGY) == len(repFY), "constraint on same number of points"
    n_points = len(repX)

    idx_list = []
    A_list, b_list = [], []

    def Ab(xi, gi, fi, xj, gj, fj) :
        A = (1 / 2) * np.outer(gj, xi - xj) + (1 / 2) * np.outer(xi - xj, gj) \
            + 1 / 2 / (1 - (mu / L)) * (
                (1 / L) * np.outer(gi - gj, gi - gj)
                + mu * np.outer(xi - xj, xi - xj)
                - (mu / L) * np.outer(gi - gj, xi - xj)
                - (mu / L) * np.outer(xi - xj, gi - gj)
            )
        b = (fj - fi)
        return (A, b)

    # interpolation conditions for x
    for i in range(n_points) :
        for j in range(n_points) :
            if i != j :
                xi, xj =  repX[i, :],  repX[j, :]
                gi, gj = repGX[i, :], repGX[j, :]
                fi, fj = repFX[i, :], repFX[j, :]

                idx_list.append((0, 0, i, j))
                (Ai, bi) = Ab(xi, gi, fi, xj, gj, fj)
                A_list.append(Ai)
                b_list.append(bi)

    # interpolation conditions for y
    for iy in range(n_points) :
        for jy in range(n_points) :
            if iy != jy :
                yi, yj =  repY[iy, :],  repY[jy, :]
                gi, gj = repGY[iy, :], repGY[jy, :]
                fi, fj = repFY[iy, :], repFY[jy, :]

                idx_list.append((1, 1, iy, jy))
                (Ai, bi) = Ab(yi, gi, fi, yj, gj, fj)
                A_list.append(Ai)
                b_list.append(bi)
    
    # interpolation conditions for (x, y) and (y, x) pairs:
    for i in range(n_points-1):
        for j in range(n_points-1):
            # (x, y) case
            xi, yj =  repX[i, :],  repY[j, :]
            gi, gj = repGX[i, :], repGY[j, :]
            fi, fj = repFX[i, :], repFY[j, :]

            idx_list.append((0, 1, i, j))
            (Ai, bi) = Ab(xi, gi, fi, yj, gj, fj)
            A_list.append(Ai)
            b_list.append(bi)

            # (y, x) case
            yi, xj =  repY[i, :],  repX[j, :]
            gi, gj = repGY[i, :], repGX[j, :]
            fi, fj = repFY[i, :], repFX[j, :]

            idx_list.append((1, 0, i, j))
            (Ai, bi) = Ab(yi, gi, fi, xj, gj, fj)
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