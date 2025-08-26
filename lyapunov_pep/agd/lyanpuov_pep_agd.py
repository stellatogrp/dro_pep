import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from copy import copy
from interpolation_conditions import smooth_strongly_convex, smooth_strongly_convex_agd
from sample_generation import sample_generation
from argparse import ArgumentParser

def solve_agd_pep_primal(mu, L, eta, n_points) :
    """
    Solve the performance estimation problem for gradient descent on F_{mu,L}.
        y^{i} = x^{i-1} - eta * grad F(x^{i-1})
        x^{i} = y^{i} + (i-1) / (i+2) * (y^{i} - y^{i-1}) for i=1,...,n_points

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

    y = x
    xs = 0.0 * x
    gs = 0.0 * g
    fs = 0.0 * f

    repX, repG, repF = [], [], []
    repX.append(x)
    repG.append(g)
    repF.append(f)

    for i in range(1, n_points+1):
        y_prev = y

        # y = x - t * g(x)
        # x = y + (k-1)/(k+2)*(y - y_prev)

        y = x - eta * g
        x = y + (i - 1) / (i + 2) * (y - y_prev) # xi

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


def solve_agd_pep_dual(mu, L, eta, n_points) :

    dimG = n_points + 2 # [x0, g0, ..., gn]
    dimF = n_points + 1 # [f0, ..., fn]

    # Algorithm iterates representation
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x = eyeG[0, :]  # x0
    g = eyeG[1, :]  # g0
    f = eyeF[0, :]  # f0

    y = x
    xs = 0.0 * x
    gs = 0.0 * g
    fs = 0.0 * f

    repX, repG, repF = [], [], []
    repX.append(x)
    repG.append(g)
    repF.append(f)

    for i in range(1, n_points+1):
        y_prev = y
        y = x - eta * g
        x = y + (i - 1) / (i + 2) * (y - y_prev) # xi

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
    idx_list, A_list, b_list, _ = smooth_strongly_convex_agd(np.array(repX), np.array(repG), np.array(repF), mu, L)

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


def lyap_search_for_agd(mu, L, eta, n_points) :

    dimG = n_points + 2 # [x0, g0, ..., gn]
    dimF = n_points + 1 # [f0, ..., fn]

    # Algorithm iterates representation
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x = eyeG[0, :]  # x0
    g = eyeG[1, :]  # g0
    f = eyeF[0, :]  # f0

    y = x
    xs = 0.0 * x
    gs = 0.0 * g
    fs = 0.0 * f

    repX, repG, repF = [], [], []
    repX.append(x)
    repG.append(g)
    repF.append(f)

    for i in range(1, n_points+1):
        y_prev = y
        y = x - eta * g
        x = y + (i - 1) / (i + 2) * (y - y_prev) # xi

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

    # Obtain primal- and dual- PEP data
    _, primal_obj, primal_data = solve_agd_pep_primal(mu, L, eta, n_points)
    _, dual_obj, dual_data = solve_agd_pep_dual(mu, L, eta, n_points)

    G, F = primal_data['G'], primal_data['F']
    lmbd, tau, S = dual_data['lmbd'], dual_data['tau'], dual_data['S']

    ##### Lyapunov PEP Definition

    # Lyapunov function definition
    Vk_G = cp.Variable((n_points+1, dimG, dimG))
    Vk_F = cp.Variable((n_points+1, dimF))
    constraints = [Vk_G[k,:,:] == Vk_G[k,:,:].T for k in range(n_points+1)]

    # Lyapunov function ansatz
    Vk_A = cp.Variable((n_points+1, 6))  # (k, k+1, k+2, star) choose 2
    Vk_B = cp.Variable((n_points+1))

    exit(0)

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
        primal_stat, primal_val, primal_var = solve_agd_pep_primal(mu, L, eta, n_points)
        print("[Primal PEP at iteration", n_points, "] Status:", primal_stat, ", Value:", primal_val)
        primal_obj.append(primal_val)

        if verbose == True :
            print("G =", primal_var['G'])
            print("F =", primal_var['F'])

    print("\nSolve [Dual PEP]")
    last_idx_list = None
    lmbd = None
    for n_points in range(1, iter_K) :
        dual_stat, dual_val, dual_var = solve_agd_pep_dual(mu, L, eta, n_points)
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
        lyap_stat, rate_inv, lyap_var = lyap_search_for_agd(mu, L, eta, n_points)
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

    # SECOND PLOT HERE

    plt.tight_layout()
    # plt.savefig('lyapunov_pep/results/gradient_descent_pep_results.pdf', dpi=300)
    plt.savefig('results/gd_pep_results.pdf', dpi=300)

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    # plt.savefig('lyapunov_pep/results/gradient_descent_pep_results_loglog.pdf', dpi=300)
    plt.savefig('results/agd_pep_results_loglog.pdf', dpi=300)

    plt.show()
