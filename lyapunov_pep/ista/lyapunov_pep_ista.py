import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from copy import copy
# from gd.interpolation_conditions import smooth_strongly_convex, smooth_strongly_convex_gd
# from gd.sample_generation import sample_generation

from argparse import ArgumentParser


def interpolation_inequalities_mu_L(repX, repG, repF, mu=0.0, L=np.inf, varG=None, varF=None):
    
    assert mu <= L, "mu must be less than or equal to L"
    # print(len(repX), len(repG), len(repF))
    assert len(repX) == len(repG) == len(repF), "constraint on same number of points"

    n_points = len(repX)

    idx_list = []
    A_list, b_list = [], []

    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                pass
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


def interpolation_inequalities_convex(repX, repG, repF, varG=None, varF=None):
    assert mu <= L, "mu must be less than or equal to L"
    # print(len(repX), len(repG), len(repF))
    assert len(repX) == len(repG) == len(repF), "constraint on same number of points"

    n_points = len(repX)
    idx_list = []
    A_list, b_list = [], []

    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            xi, xj = repX[i, :], repX[j, :]
            gi, gj = repG[i, :], repG[j, :]
            fi, fj = repF[i, :], repF[j, :]

            Ai = (1 / 2) * np.outer(gj, xi - xj) + (1 / 2) * np.outer(xi - xj, gj)
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


def solve_ista_pep_primal(mu, L, eta, n_points):

    K = n_points

    # G = [xs, x0, df(x0), ..., df(x_K-1), df(x_K), dg(x1), ..., dg(xK), dfs, dgs] -> dim: 2K + 5
    # F = [fs, f0, ..., fK, gs, g0, ..., gK] -> dim: 2K + 4

    # reduced version:
    # G = [x0, df(x0), ..., df(x_K), dg(x1), ..., dg(xK), dfs] -> dim: 2K + 3
    # F = [f0, ..., fK, g1, ..., gK, fs] -> dim: 2K + 2

    dimG = 2 * K + 3  # 1 + K + K + 1
    dimF = 2 * K + 2  # 1 + (K+1) + (K+1)

    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    dfs_idx = dimG - 1
    def df_idx(k):
        return k + 1
    
    def dg_idx(k):
        # return K + k + 1  # 1 + (K + 1) + (k - 1)
        return df_idx(K) + k

    fs_idx = dimF - 1
    def func_f_idx(k):
        return k

    def func_g_idx(k):
        # return (K + 1) + k # (K + 2) + k - 1
        return func_f_idx(K) + k

    repX, repF_grad, repF_funcval, repG_grad, repG_funcval = [], [], [], [], []
    x = eyeG[0, :] # x0
    # y = x # y0
    dfx = eyeG[df_idx(0), :] # df(x0)
    fx = eyeF[func_f_idx(0), :] # f(x0)

    # dgx = eyeG[dg_idx(0), :] # dg(x0)
    # gx = eyeF[func_g_idx(0), :] # g(x0)

    xs = 0.0 * x
    dfs = eyeG[dfs_idx, :]
    dgs = - dfs

    fs = eyeF[fs_idx, :]
    gs = - fs

    repX.append(x)
    repF_grad.append(dfx)
    repF_funcval.append(fx)

    for k in range(1, K+1):
        dgx = eyeG[dg_idx(k), :]
        gx = eyeF[func_g_idx(k), :]

        y = x - eta * dfx
        x = y - eta * dgx

        dfx = eyeG[df_idx(k), :]
        fx = eyeF[func_f_idx(k), :]

        repX.append(x)
        repF_grad.append(dfx)
        repF_funcval.append(fx)
        
        repG_grad.append(dgx)
        repG_funcval.append(gx)

    assert len(repX) == len(repF_grad) == len(repF_funcval) == K + 1

    assert len(repX) - 1 == len(repG_grad) == len(repG_funcval) == K
    
    repX.append(xs)
    repF_grad.append(dfs)
    repF_funcval.append(fs)
    repG_grad.append(dgs)
    repG_funcval.append(gs)

    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)

    constraints = [ G >> 0 ]  # G must be positive semidefinite
    constraints += [ F >= 0 ]  # F must be non-negative

    constraints += interpolation_inequalities_mu_L(np.array(repX), np.array(repF_grad), np.array(repF_funcval), mu=mu, L=L, varG=G, varF=F)[-1]
    constraints += interpolation_inequalities_convex(np.array(repX[1:]), np.array(repG_grad), np.array(repG_funcval), varG=G, varF=F)[-1]

    constraints += [ ((repX[0] - xs) @ G @ (repX[0] - xs).T) <= 1 ]

    objective = (repF_funcval[K] + repG_funcval[K-1] - (fs + gs)) @ F

    problem = cp.Problem(cp.Maximize(objective), constraints)

    problem.solve(solver=cp.CLARABEL, verbose=False)

    return problem.status, problem.value, {'G': G.value, 'F': F.value}


if __name__ == "__main__":
    parser = ArgumentParser(description="Gradient Descent PEP Parameters")
    parser.add_argument('--mu', type=float, default=0.0, help='Strong convexity parameter')
    parser.add_argument('--L', type=float, default=1.0, help='Lipschitz constant for the gradient')
    parser.add_argument('--eta', type=float, default=1.0, help='Step size')
    parser.add_argument('--iter_K', type=int, default=10, help='Number of iterations')
    parser.add_argument('--eps', type=float, default=1e-2, help='Wasserstein eps-ball radius (for DRO-PEP)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    mu = args.mu
    L = args.L
    eta = args.eta
    iter_K = args.iter_K
    eps = args.eps
    verbose = args.verbose

    print("GD with mu =", mu, ", L =", L, ", eta =", eta, ", up to iteration =", iter_K)

    primal_obj, dual_obj = [], []

    print("Solve [Primal PEP]")
    for n_points in range(1, iter_K + 1) :
        primal_stat, primal_val, primal_var = solve_ista_pep_primal(mu, L, eta, n_points)
        print("[Primal PEP at iteration", n_points, "] Status:", primal_stat, ", Value:", primal_val)
        primal_obj.append(primal_val)

        if verbose == True :
            print("G =", primal_var['G'])
            print("F =", primal_var['F'])
