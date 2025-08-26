import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from copy import copy
# from lyapunov_pep.interpolation_conditions import smooth_strongly_convex, smooth_strongly_convex_gd
# from lyapunov_pep.sample_generation import sample_generation

from interpolation_conditions import smooth_strongly_convex, smooth_strongly_convex_gd
from sample_generation import sample_generation

from argparse import ArgumentParser

def solve_gd_pep_primal(mu, L, eta, n_points) :
    """
    Solve the performance estimation problem for gradient descent on F_{mu,L}.
        x^{i} = x^{i-1} - eta * grad F(x^{i-1}) for i=1,...,n_points

    Args:
        mu: Strong convexity parameter
        L: Lipschitz constant for the gradient
        eta: Step size
        n_points: Number of algorithm iterations  

    Returns:
        dict: Contains 'status' and 'value' from the optimization problem
    """

    dimG = n_points + 2 # [x0, g0, ..., gn]
    dimF = n_points + 1 # [f0, ..., fn]
    
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
    repX.append(x)
    repG.append(g)
    repF.append(f)

    for i in range(1, n_points+1):
        x = x - eta * g   # xi
        g = eyeG[i+1, :]        # gi
        f = eyeF[i, :]          # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)

    assert len(repX) == len(repG) == len(repF) == n_points + 1, "constraint on same number of points"

    # Define variables
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)

    constraints = [ G >> 0 ]  # G must be positive semidefinite
    constraints += [ F >= 0 ]  # F must be non-negative

    # Define the interpolation conditions
    repX.append(xs)
    repG.append(gs)
    repF.append(fs)

    constraints += smooth_strongly_convex(np.array(repX), np.array(repG), np.array(repF), mu, L, varG=G, varF=F)[-1]
    
    # Initial condition
    constraints += [ ((repX[0] - xs) @ G @ (repX[0] - xs).T) <= 1 ]

    # Objective function (performance metric)
    objective = (repF[n_points] - fs) @ F
    # objective = (repX[n_points] - xs) @ G @ (repX[n_points] - xs).T

    # Formulate the optimization problem
    problem = cp.Problem(cp.Maximize(objective), constraints)

    # Solve the problem
    # problem.solve(solver=cp.MOSEK, verbose=False)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    return problem.status, problem.value, {'G': G.value, 'F': F.value}


def solve_gd_pep_dual(mu, L, eta, n_points) :
    
    dimG = n_points + 2 # [x0, g0, ..., gn]
    dimF = n_points + 1 # [f0, ..., fn]
    
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
    repX.append(x)
    repG.append(g)
    repF.append(f)

    for i in range(1, n_points+1):
        x = x - eta * g   # xi
        g = eyeG[i+1, :]        # gi
        f = eyeF[i, :]          # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)

    assert len(repX) == len(repG) == len(repF) == n_points + 1, "constraint on same number of points"

    # Define the interpolation conditions
    repX.append(xs)
    repG.append(gs)
    repF.append(fs)

    # Define dual variables
    constraints = []  # S must be positive semidefinite

    # Interpolation conditions: <Ai, G> + bi.T @ F <= 0
    # idx_list, A_list, b_list, _ = smooth_strongly_convex(np.array(repX), np.array(repG), np.array(repF), mu, L)
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)

    # Initial condition: <A0, G> + b0.T @ F + c0 <= 0
    A0 = np.outer(repX[0] - xs, repX[0] - xs)
    b0 = 0.0 * fs
    c0 = - 1.0

    # Corresponding dual variables to primal constriants
    lmbd = cp.Variable(len(A_list), nonneg=True)    # Interpolation conditions
    tau = cp.Variable()                             # Initial condition

    # Dual PEP constraints
    PSD_dual = tau * A0
    LIN_dual = tau * b0
    for (ij, (Aij, bij)) in enumerate(zip(A_list, b_list)) :
        PSD_dual += lmbd[ij] * Aij
        LIN_dual += lmbd[ij] * bij

    # Primal objective function (performance metric)
    Aobj = np.outer(repX[n_points] - xs, repX[n_points] - xs)
    Aobj = 0.0 * Aobj
    bobj = repF[n_points] - fs
    # bobj = 0.0 * bobj

    PSD_dual -= Aobj
    LIN_dual -= bobj

    # Add dual constraints
    constraints += [ PSD_dual >> 0 ]  # PSD constraint
    constraints += [ LIN_dual == 0 ]  # Linear constraint

    # Solve the dual PEP problem
    problem = cp.Problem(cp.Minimize(tau), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    return problem.status, problem.value, {'idx': idx_list, 'lmbd': lmbd.value, 'tau': tau.value, 'S': PSD_dual.value}


def lyap_search_for_gd(mu, L, eta, n_points) :
    
    dimG = n_points + 2 # [x0, g0, ..., gn]
    dimF = n_points + 1 # [f0, ..., fn]
    
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

    # Define the interpolation conditions
    repX.append(xs)
    repG.append(gs)
    repF.append(fs)

    # Obtain primal- and dual- PEP data
    _, primal_obj, primal_data = solve_gd_pep_primal(mu, L, eta, n_points)
    _, dual_obj, dual_data = solve_gd_pep_dual(mu, L, eta, n_points)

    G, F = primal_data['G'], primal_data['F']
    lmbd, tau, S = dual_data['lmbd'], dual_data['tau'], dual_data['S']

    ##### Lyapunov PEP Definition
    
    # Lyapunov function definition
    Vk_G = cp.Variable((n_points+1, dimG, dimG))
    Vk_F = cp.Variable((n_points+1, dimF))
    constraints = [Vk_G[k,:,:] == Vk_G[k,:,:].T for k in range(n_points+1)]

    # Lyapunov function ansatz
    Vk_A = cp.Variable((n_points+1, 3))
    Vk_B = cp.Variable((n_points+1))

    for k in range(n_points+1) :
        constraints += [ Vk_G[k,:,:] == \
                            # Vk_A[k,0] * np.outer(repG[k], repG[k]) \
                            Vk_A[k,1] * np.outer(repX[k] - xs, repX[k] - xs) \
                            + Vk_A[k,2] * (1/2) * (np.outer(repX[k] - xs, repG[k]) + np.outer(repG[k], repX[k] - xs)),
                         Vk_F[k,:] == Vk_B[k] * repF[k]]

    # Initial condition
    V_initial_G = np.outer(repX[0] - xs, repX[0] - xs)
    V_initial_F = 0.0 * (repF[0] - fs)

    # Objective function
    tau = cp.Variable(nonneg=True)
    V_obj_G = tau * np.outer(repX[n_points] - xs, repX[n_points] - xs)
    V_obj_G = 0.0 * V_obj_G
    V_obj_F = tau * (repF[n_points] - fs) # We maximize tau
    # V_obj_F = 0.0 * V_obj_F

    # Lyapunov PEP definition
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)

    # [1] V0 <= V_initial : (s, 0) interpolation condition
    lmbd0 = cp.Variable(nonneg=True)
    constraints += [
        V_initial_G - Vk_G[0,:,:] + lmbd0 * A_list[-n_points-1] >> 0,
        V_initial_F - Vk_F[0,:] + lmbd0 * b_list[-n_points-1] == 0
        ]
    if verbose == True :
        print("00", idx_list[-n_points-1])

    # [2] V{k+1} <= V{k} for k=0,...,n_points-1 : (k, k+1), (s, k), (s, k+1) interpolation conditions
    if n_points > 0 :
        lmbd = cp.Variable((n_points, 3), nonneg=True)  # 3 dual variables for each k
        for k in range(n_points) :
            constraints += [
                Vk_G[k,:,:] - Vk_G[k+1,:,:] \
                    + lmbd[k, 0] * A_list[k] \
                    + lmbd[k, 1] * A_list[k-n_points-1] \
                    # + lmbd[k, 2] * A_list[k-n_points] \
                        >> 0,
                Vk_F[k,:] - Vk_F[k+1,:] \
                    + lmbd[k, 0] * b_list[k] \
                    + lmbd[k, 1] * b_list[k-n_points-1] \
                    # + lmbd[k, 2] * b_list[k-n_points] \
                        == 0
                ]
            if verbose == True :
                print(k, idx_list[k], idx_list[k-n_points-1], idx_list[k-n_points])

    # [3] V_obj <= V{n_points} : (s, n_points) interpolation condition
    lmbdK = cp.Variable(nonneg=True)
    constraints += [
        Vk_G[n_points,:,:] - V_obj_G \
            + lmbdK * A_list[-1] >> 0,
        Vk_F[n_points,:] - V_obj_F \
            + lmbdK * b_list[-1] == 0
    ]
    if verbose == True :
        print(idx_list[-1])

    problem = cp.Problem(cp.Maximize(tau), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    return problem.status, problem.value, {
        'Vk_G': Vk_G.value, 'Vk_F': Vk_F.value,
        'Vk_A': Vk_A.value, 'Vk_B': Vk_B.value,
        'lmbd0': lmbd0.value, 'lmbd': lmbd.value, 'lmbdK': lmbdK.value
    }


def dro_lyap_search_for_gd(mu, L, eta, n_points, eps_val=1e-4) :

    dimG = n_points + 2 # [x0, g0, ..., gn]
    dimF = n_points + 1 # [f0, ..., fn]
    
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

    # Define the interpolation conditions
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)

    # Generate samples
    _, _, avgG, avgF = sample_generation(iter_K=n_points, N=1, mu=mu, L=L)
    assert avgG.shape == (dimG, dimG), "avgG shape"
    assert avgF.shape == (dimF,), "avgF shape"

    ##### Lyapunov DRO-PEP Definition
    eps = cp.Parameter(nonneg=True, value=eps_val)  # Wasserstein eps-ball radius
    
    # Lyapunov function definition
    Vk_G = cp.Variable((n_points+1, dimG, dimG))
    Vk_F = cp.Variable((n_points+1, dimF))
    constraints = [Vk_G[k,:,:] == Vk_G[k,:,:].T for k in range(n_points+1)]

    # Lyapunov function ansatz
    Vk_A = cp.Variable((n_points+1, 3))
    Vk_B = cp.Variable((n_points+1))

    for k in range(n_points+1) :
        constraints += [ Vk_G[k,:,:] == \
                            # Vk_A[k,0] * np.outer(repG[k], repG[k]) \
                            Vk_A[k,1] * np.outer(repX[k] - xs, repX[k] - xs) \
                            + Vk_A[k,2] * (1/2) * (np.outer(repX[k] - xs, repG[k]) + np.outer(repG[k], repX[k] - xs)) ]
        constraints += [ Vk_F[k,:] == Vk_B[k] * repF[k]]  # F must be non-negative

    # Initial condition
    V_initial_G = np.outer(repX[0] - xs, repX[0] - xs)
    V_initial_F = 0.0 * (repF[0] - fs)

    # Objective function
    tau = cp.Variable(nonneg=True)
    V_obj_G = tau * np.outer(repX[n_points] - xs, repX[n_points] - xs)
    V_obj_G = 0.0 * V_obj_G
    V_obj_F = tau * (repF[n_points] - fs) # We maximize tau
    # V_obj_F = 0.0 * V_obj_F

    # Sample rate
    sample_rate = avgF[n_points] / avgG[0, 0]

    # [1] V0 <= V_initial : (s, 0) interpolation condition
    lmbd0 = cp.Variable(nonneg=True)
    t0 = cp.Variable()
    X0 = cp.Variable((dimG, dimG), symmetric=True)
    Y0 = cp.Variable(dimF)
    constraints += [
        V_initial_G - Vk_G[0,:,:] + lmbd0 * A_list[-n_points-1] + X0 >> 0,
        V_initial_F - Vk_F[0,:] + lmbd0 * b_list[-n_points-1] + Y0 == 0,
        eps*t0 + cp.trace(avgG@X0) + avgF.T@Y0 <= 0,
        cp.SOC(t0, cp.hstack([cp.vec(X0, order='F'), Y0]))  # SOC constraint
        ]

    # [2] V{k+1} <= V{k} for k=0,...,n_points-1 : (k, k+1), (s, k), (s, k+1) interpolation conditions
    if n_points > 0 :
        lmbd = cp.Variable((n_points, 3), nonneg=True)  # 3 dual variables for each k

        t = cp.Variable(n_points)  # 3 dual variables for each k
        X = cp.Variable((n_points, dimG, dimG))
        Y = cp.Variable((n_points, dimF))
        constraints += [X[k,:,:] == X[k,:,:].T for k in range(n_points) ]

        # Xparam = cp.Variable((n_points, 6))
        # Yparam = cp.Variable((n_points, 2))
        # for k in range(n_points) :
        #     constraints += [ X[k,:,:] == Xparam[k,0] * np.outer(repG[k], repG[k]) \
        #                         + Xparam[k,1] * np.outer(repX[k] - xs, repX[k] - xs) \
        #                         + Xparam[k,2] * (1/2) * (np.outer(repX[k] - xs, repG[k]) + np.outer(repG[k], repX[k] - xs))
        #                         + Xparam[k,3] * (1/2) * (np.outer(repX[k] - xs, repG[k+1]) + np.outer(repG[k+1], repX[k] - xs)) \
        #                         + Xparam[k,4] * (1/2) * (np.outer(repG[k+1], repG[k]) + np.outer(repG[k], repG[k+1])) \
        #                         + Xparam[k,5] * np.outer(repG[k+1], repG[k+1]) ]
        #     constraints += [ Y[k,:] == Yparam[k,0] * repF[k] + Yparam[k,1] * repF[k+1] ]

        for k in range(n_points) :
            constraints += [
                Vk_G[k,:,:] - Vk_G[k+1,:,:] \
                    + lmbd[k, 0] * A_list[k] \
                    + lmbd[k, 1] * A_list[k-n_points-1] \
                    # + lmbd[k, 2] * A_list[k-n_points] \
                    + X[k,:,:] >> 0,
                Vk_F[k,:] - Vk_F[k+1,:] \
                    + lmbd[k, 0] * b_list[k] \
                    + lmbd[k, 1] * b_list[k-n_points-1] \
                    # + lmbd[k, 2] * b_list[k-n_points] \
                    + Y[k, :] == 0,
                eps*t[k] + cp.trace(avgG@X[k,:,:]) + avgF.T@Y[k,:] <= 0,
                cp.SOC(t[k], cp.hstack([cp.vec(X[k,:,:], order='F'), Y[k,:]]))  # SOC constraint
                ]

    # [3] V_obj <= V{n_points} : (s, n_points) interpolation condition
    lmbdK = cp.Variable(nonneg=True)

    tK = cp.Variable()
    XK = cp.Variable((dimG, dimG), symmetric=True)
    YK = cp.Variable(dimF)

    constraints += [
        Vk_G[n_points,:,:] - V_obj_G \
            + lmbdK * A_list[-1] \
            + XK >> 0,
        Vk_F[n_points,:] - V_obj_F \
            + lmbdK * b_list[-1] \
            + YK == 0,
        eps*tK + cp.trace(avgG@XK) + avgF.T@YK <= 0,
        cp.SOC(tK, cp.hstack([cp.vec(XK, order='F'), YK]))  # SOC constraint
    ]

    problem = cp.Problem(cp.Maximize(tau), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    return problem.status, problem.value, {
            'Vk_G': Vk_G.value, 'Vk_F': Vk_F.value,
            'Vk_A': Vk_A.value, 'Vk_B': Vk_B.value,
            'lmbd0': lmbd0.value, 'lmbd': lmbd.value, 'lmbdK': lmbdK.value,
            't0': t0.value, 't': t.value, 'tK': tK.value,
            'X0': X0.value, 'X': X.value, 'XK': XK.value,
            'Y0': Y0.value, 'Y': Y.value, 'YK': YK.value,
            'sample_rate': sample_rate
        }


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
    for n_points in range(1, iter_K) :
        primal_stat, primal_val, primal_var = solve_gd_pep_primal(mu, L, eta, n_points)
        print("[Primal PEP at iteration", n_points, "] Status:", primal_stat, ", Value:", primal_val)
        primal_obj.append(primal_val)

        if verbose == True :
            print("G =", primal_var['G'])
            print("F =", primal_var['F'])

    print("\nSolve [Dual PEP]")
    last_idx_list = None
    lmbd = None
    for n_points in range(1, iter_K) :
        dual_stat, dual_val, dual_var = solve_gd_pep_dual(mu, L, eta, n_points)
        print("[Dual PEP at iteration", n_points, "] Status:", dual_stat, ", Value:", dual_val)
        dual_obj.append(dual_val)

        if verbose == True :
            print("tau =", dual_var['tau'])
            print("S =", dual_var['S'])
            for m in range(len(idx_list)) :
                print("\tdual lamdas :")
                idx_list = dual_var['idx']
                lmbd = dual_var['lmbd']
                (i, j) = idx_list[m]
                print(f"\t\tlmbd[{i if i < n_points+1 else "s"}, {j if j < n_points+1 else "s"}] = {lmbd[m]}")
        
        if n_points == iter_K - 1 :
            last_idx_list = copy(dual_var['idx'])
            lmbd = copy(dual_var['lmbd'])

    if last_idx_list is not None :
        print("\tlast dual lamdas :")
        for m in range(len(last_idx_list)) :
            lmbd = dual_var['lmbd']
            (i, j) = last_idx_list[m]
            print(f"\t\tlmbd[{i if i < n_points+1 else "s"}, {j if j < n_points+1 else "s"}] = {lmbd[m]}")

    # Lyapunov search PEP
    lyap_obj = []
    print("\nSolve [Lyapunov PEP]")
    for n_points in range(1, iter_K) :
        lyap_stat, rate_inv, lyap_var = lyap_search_for_gd(mu, L, eta, n_points)
        lyap_obj.append(1.0 / rate_inv if rate_inv > 0 else np.inf)
        print("[Lyapunov PEP at iteration", n_points, "] Status:", lyap_stat, ", rate_inv:", rate_inv)

        if n_points == iter_K - 1 :
            for (Ak, Bk, lmbdk) in zip(lyap_var['Vk_A'], lyap_var['Vk_B'], lyap_var['lmbd']) :
                print("\tGk = ", Ak)
                print("\tFk = ", Bk)
                print("\tlamda =", lmbdk)
                print()
            print("\tlmbd0 =", lyap_var['lmbd0'])
            print("\tlmbdK =", lyap_var['lmbdK'])


    # Lyapunov search DRO-PEP
    dro_lyap_obj = []
    sample_rate = []
    print("\nSolve [Lyapunov DRO-PEP]")
    for n_points in range(1, iter_K) :
        dro_lyap_stat, dro_rate_inv, dro_lyap_var = dro_lyap_search_for_gd(mu, L, eta, n_points, eps_val=eps)
        dro_lyap_obj.append(1.0 / dro_rate_inv if dro_rate_inv > 0 else np.inf)
        sample_rate.append(dro_lyap_var['sample_rate'])
        print("[ Lyapunov DRO-PEP at iteration", n_points, "with eps:", eps, "] Status:", dro_lyap_stat, ", rate_inv:", dro_rate_inv)

        if n_points == iter_K - 1 :
            for (Ak, Bk, lmbdk) in zip(dro_lyap_var['Vk_A'], dro_lyap_var['Vk_B'], dro_lyap_var['lmbd']) :
                print("\tAk =", Ak)
                print("\tBk =", Bk)
                print("\tlamda =", lmbdk)
                print()
            print("\tlmbd0 =", dro_lyap_var['lmbd0'])
            print("\tlmbdK =", dro_lyap_var['lmbdK'])
    
    # Plotting results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot: Primal and Dual PEP
    axs[0].plot(range(1,iter_K), primal_obj, 'o-', label='Primal PEP')
    axs[0].plot(range(1,iter_K), dual_obj, 's--', label='Dual PEP')
    axs[0].plot(range(1,iter_K), lyap_obj, 'd:', label='Lyapunov PEP')
    axs[0].set_xlabel('Number of iterations $K$')
    axs[0].set_ylabel('PEP value')
    axs[0].legend()
    axs[0].set_title('Primal/Dual PEP')
    axs[0].grid()
    # axs[0].set_xscale('log')
    # axs[0].set_yscale('log')

    # Second subplot: Lyapunov PEP and DRO-PEP
    axs[1].plot(range(1,iter_K), sample_rate, '--', label='Sample Rate', linewidth=0.5, color='red')
    axs[1].plot(range(1,iter_K), lyap_obj, 'd:', label='Lyapunov PEP')
    axs[1].plot(range(1,iter_K), dro_lyap_obj, 'x-.', label=f'Lyapunov DRO-PEP ($\\epsilon$={eps})')
    axs[1].set_xlabel('Number of iterations $K$')
    axs[1].set_ylabel('PEP value')
    axs[1].legend()
    axs[1].set_title('Lyapunov PEPs')
    axs[1].grid()
    # axs[1].set_xscale('log')
    # axs[1].set_yscale('log')

    plt.tight_layout()
    # plt.savefig('lyapunov_pep/results/gradient_descent_pep_results.pdf', dpi=300)
    plt.savefig('results/gradient_descent_pep_results.pdf', dpi=300)

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    # plt.savefig('lyapunov_pep/results/gradient_descent_pep_results_loglog.pdf', dpi=300)
    plt.savefig('results/gradient_descent_pep_results_loglog.pdf', dpi=300)

    plt.show()
