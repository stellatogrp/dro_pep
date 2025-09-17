import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from copy import copy
from gd.interpolation_conditions import smooth_strongly_convex, smooth_strongly_convex_gd, smooth_strongly_convex_agd
from gd.sample_generation import sample_generation_agd
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

    dimG = 2 * n_points + 2 # [x0-xs,    gx0,    gx1,    gy1, ...,    gxn,    gyn]
    dimF = 2 * n_points + 1 # [       fx0-fs, fx1-fs, fy1-fs, ..., fxn-fs, fyn-fs]
    
    # Algorithm iterates representation
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x  = eyeG[0, :]  # x0
    gx = eyeG[1, :]  # g at x0
    fx = eyeF[0, :]  # f at x0

    y = copy(x)
    gy = copy(gx)
    fy = copy(fx)

    xs = 0.0 * x
    gs = 0.0 * gx
    fs = 0.0 * fx

    repX, repGX, repFX = [], [], []
    repX.append(x)
    repGX.append(gx)
    repFX.append(fx)

    repY, repGY, repFY = [], [], []
    repY.append(y)
    repGY.append(gy)
    repFY.append(fy)

    for i in range(1, n_points+1):
        y_prev = copy(y)

        y = x - eta * gx     # yi (primary sequence)
        x = y + (i - 1) / (i + 2) * (y - y_prev)    # xi (secondary sequence)

        gx = eyeG[  2*i, :] # gxi
        fx = eyeF[2*i-1, :] # fxi

        gy = eyeG[2*i+1, :] # gyi
        fy = eyeF[  2*i, :] # fyi
    
        repX.append(x)
        repGX.append(gx)
        repFX.append(fx)

        repY.append(y)
        repGY.append(gy)
        repFY.append(fy)

    assert len(repX) == len(repY) == n_points + 1, "constraint on same number of points"

    # Define variables
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)

    constraints = [ G >> 0 ]  # G must be positive semidefinite
    constraints += [ F >= 0 ]  # F must be non-negative

    # Define the interpolation conditions
    repX.append(xs)
    repGX.append(gs)
    repFX.append(fs)
    
    repY.append(xs)
    repGY.append(gs)
    repFY.append(fs)

    s = len(repX) - 1   # index to the optimal point

    # Full PEP: use the interpolation conditions for all pairs (i, j)
    constraints += smooth_strongly_convex_agd(
        np.array(repX), np.array(repGX), np.array(repFX),
        np.array(repY), np.array(repGY), np.array(repFY),
        mu, L, varG=G, varF=F
    )[-1]
    
    # Initial condition
    constraints += [ ((repX[0] - repX[s]) @ G @ (repX[0] - repX[s]).T) <= 1 ]

    # Objective function (performance metric)
    # objective = (repFY[n_points] - repFY[s]) @ F
    objective = (repFX[n_points] - repFX[s]) @ F
    # objective = (repX[n_points] - repX[s]) @ G @ (repX[n_points] - repX[s]).T

    # Formulate the optimization problem
    problem = cp.Problem(cp.Maximize(objective), constraints)

    # Solve the problem
    # problem.solve(solver=cp.MOSEK, verbose=False)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    return problem.status, problem.value, {'G': G.value, 'F': F.value}


# If you are to analyze the convergence of primary sequence x{k}
def solve_agd_pep_dual_full(mu, L, eta, n_points) :

    dimG = 2 * n_points + 2 # [x0-xs,    gx0,    gx1,    gy1, ...,    gxn,    gyn]
    dimF = 2 * n_points + 1 # [       fx0-fs, fx1-fs, fy1-fs, ..., fxn-fs, fyn-fs]
    
    # Algorithm iterates representation
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x  = eyeG[0, :]  # x0
    gx = eyeG[1, :]  # g at x0
    fx = eyeF[0, :]  # f at x0

    y = x
    gy = gx
    fy = fx

    xs = 0.0 * x
    gs = 0.0 * gx
    fs = 0.0 * fx

    repX, repGX, repFX = [], [], []
    repX.append(x)
    repGX.append(gx)
    repFX.append(fx)

    repY, repGY, repFY = [], [], []
    repY.append(y)
    repGY.append(gy)
    repFY.append(fy)

    for i in range(1, n_points+1):
        y_prev = y

        y = x - eta * gx     # yi (primary sequence)
        x = y + (i - 1) / (i + 2) * (y - y_prev)    # xi (secondary sequence)

        gx = eyeG[  2*i, :] # gxi
        fx = eyeF[2*i-1, :] # fxi

        gy = eyeG[2*i+1, :] # gyi
        fy = eyeF[  2*i, :] # fyi
    
        repX.append(x)
        repGX.append(gx)
        repFX.append(fx)

        repY.append(y)
        repGY.append(gy)
        repFY.append(fy)

    assert len(repX) == len(repY) == n_points + 1, "constraint on same number of points"

    # Define the interpolation conditions
    repX.append(xs)
    repGX.append(gs)
    repFX.append(fs)
    
    repY.append(xs)
    repGY.append(gs)
    repFY.append(fs)

    s = len(repX) - 1   # index to the optimal point

    # Define dual variables
    constraints = []

    # Interpolation conditions: <Ai, G> + bi.T @ F <= 0
    idx_list, A_list, b_list, _ = smooth_strongly_convex_agd(
        np.array(repX), np.array(repGX), np.array(repFX),
        np.array(repY), np.array(repGY), np.array(repFY),
        mu, L
    )

    # Initial condition: <A0, G> + b0.T @ F + c0 <= 0
    A0 = np.outer(repX[0] - repX[s], repX[0] - repX[s])
    b0 = 0.0 * repFX[s]
    c0 = - 1.0

    # Corresponding dual variables to primal constriants
    lmbd = cp.Variable(len(A_list), nonneg=True)    # Interpolation conditions
    tau  = cp.Variable()                            # (performance metric) <= tau * (initial condition)

    # Dual PEP constraints
    PSD_dual = tau * A0
    LIN_dual = tau * b0
    for (ij, (Aij, bij)) in enumerate(zip(A_list, b_list)) :
        PSD_dual += lmbd[ij] * Aij
        LIN_dual += lmbd[ij] * bij
    
    # Restrict the set of interpolation conditions - (ex) lmbd[idx_list.index((0, 0, i, s))] == 0
    # constraints += [ ]

    # Primal objective function (performance metric)
    Aobj = np.outer(repX[n_points] - repX[s], repX[n_points] - repX[s])
    Aobj = 0.0 * Aobj
    bobj = (repFY[n_points] - repFY[s])
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

def solve_agd_pep_dual(mu, L, eta, n_points) :

    dimG = n_points + 2 # [x0-xs,    gx0,    gx1, ...,    gxn]
    dimF = n_points + 1 # [       fx0-fs, fx1-fs, ..., fxn-fs]
    
    # Algorithm iterates representation
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x  = eyeG[0, :]  # x0
    gx = eyeG[1, :]  # g at x0
    fx = eyeF[0, :]  # f at x0
    y = x

    xs = 0.0 * x
    gs = 0.0 * gx
    fs = 0.0 * fx

    repX, repG, repF = [], [], []
    repX.append(x)
    repG.append(gx)
    repF.append(fx)

    for i in range(1, n_points+1):
        y_prev = y

        y = x - eta * gx     # yi (primary sequence)
        x = y + (i - 1) / (i + 2) * (y - y_prev)    # xi (secondary sequence)

        gx = eyeG[i+1, :] # gxi
        fx = eyeF[  i, :] # fxi
    
        repX.append(x)
        repG.append(gx)
        repF.append(fx)

    assert len(repX) == len(repG) == len(repF) == n_points + 1, "constraint on same number of points"

    # Define the interpolation conditions
    repX.append(xs)
    repG.append(gs)
    repF.append(fs)
    s = len(repX) - 1   # index to the optimal point

    # Define dual variables
    constraints = []

    # Interpolation conditions: <Ai, G> + bi.T @ F <= 0
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(
        np.array(repX), np.array(repG), np.array(repF),
        mu, L
    )

    # Initial condition: <A0, G> + b0.T @ F + c0 <= 0
    A0 = np.outer(repX[0] - repX[s], repX[0] - repX[s])
    b0 = 0.0 * (repF[0] - repF[s])
    c0 = - 1.0

    # Corresponding dual variables to primal constriants
    lmbd = cp.Variable(len(A_list), nonneg=True)    # Interpolation conditions
    tau  = cp.Variable()                            # (performance metric) <= tau * (initial condition)

    # Dual PEP constraints
    PSD_dual = tau * A0
    LIN_dual = tau * b0
    for (ij, (Aij, bij)) in enumerate(zip(A_list, b_list)) :
        PSD_dual += lmbd[ij] * Aij
        LIN_dual += lmbd[ij] * bij
    
    # Restrict the set of interpolation conditions - (ex) lmbd[idx_list.index((0, 0, i, s))] == 0
    # constraints += [ ]

    # Primal objective function (performance metric)
    Aobj = np.outer(repX[n_points] - repX[s], repX[n_points] - repX[s])
    Aobj = 0.0 * Aobj
    bobj = (repF[n_points] - repF[s])
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

    dimG = n_points + 2 # [x0-xs,    gx0,    gx1, ...,    gxn]
    dimF = n_points + 1 # [       fx0-fs, fx1-fs, ..., fxn-fs]

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
    
    z = x
    repZ = [z]

    for i in range(1, n_points+1):
        y_prev = y
        y = x - eta * g
        x = y + (i - 1) / (i + 2) * (y - y_prev) # xi

        g = eyeG[i+1, :]        # gi
        f = eyeF[i, :]          # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)

        z = y + (i + 2) / 2 * (x - y)
        repZ.append(z)

    assert len(repX) == len(repG) == len(repF) == n_points + 1, "constraint on same number of points"

    # Define the interpolation conditions
    repX.append(xs)
    repG.append(gs)
    repF.append(fs)
    s = len(repX) - 1   # index to the optimal point
    assert s == n_points+1, "s definition wrong"

    repZ.append(xs)

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
    # Vk_A = cp.Variable((n_points+1, 6))
    Vk_A = []
    Vk_B = cp.Variable((n_points+1, 2))

    for k in range(n_points+1) :
        # if k+2 <= dimG-1 :
        #     constraints += [ Vk_G[k,k+2:,:] == 0 ] # ignore later iterates

        Ak = cp.Variable((3, 3), symmetric=True) # [xk-xs, zk-xs, gk]
        Pk = np.vstack([repX[k]-repX[s], repZ[k]-repZ[s], repG[k]])
        constraints += [
            Vk_G[k,:,:] == Pk.T @ Ak @ Pk,
            Ak[2,2] == 0,
        ]
        Vk_A.append(Ak)

        constraints += [ Vk_F[k,:] == Vk_B[k,0] * (repF[k] - repF[s]) ]
    
    # Initial condition
    V_initial_G = np.outer(repX[0] - repX[s], repX[0] - repX[s])
    V_initial_F = 0.0 * (repF[0] - repF[s])

    # Objective function
    tau = cp.Variable(nonneg=True)
    V_obj_G = tau * np.outer(repX[n_points] - repX[s], repX[n_points] - repX[s])
    V_obj_G = 0.0 * V_obj_G
    V_obj_F = tau * (repF[n_points] - repF[s]) # We maximize tau
    # V_obj_F = 0.0 * V_obj_F

    # Lyapunov PEP definition
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)

    # [1] V0 <= V_initial : (s, 0) interpolation condition
    lmbd0 = cp.Variable(nonneg=True)
    constraints += [
        V_initial_G - Vk_G[0,:,:] + lmbd0 * A_list[idx_list.index((s,0))] >> 0,
        V_initial_F - Vk_F[0,:] + lmbd0 * b_list[idx_list.index((s,0))] == 0
        ]
    if verbose == True :
        print("00", idx_list[idx_list.index((s,0))])

    # [2] V{k+1} <= V{k} for k=0,...,n_points-1 : (k, k+1), (s, k), (s, k+1) interpolation conditions
    if n_points > 0 :
        lmbd = cp.Variable((n_points, 3), nonneg=True)  # 3 dual variables for each k
        for k in range(n_points) :
            constraints += [
                Vk_G[k,:,:] - Vk_G[k+1,:,:] \
                    + lmbd[k, 0] * A_list[idx_list.index((k,k+1))] \
                    + lmbd[k, 1] * A_list[idx_list.index((s,k))] \
                    # + lmbd[k, 2] * A_list[idx_list.index((s,k+1))] \
                        >> 0,
                Vk_F[k,:] - Vk_F[k+1,:] \
                    + lmbd[k, 0] * b_list[idx_list.index((k,k+1))] \
                    + lmbd[k, 1] * b_list[idx_list.index((s,k))] \
                    # + lmbd[k, 2] * b_list[idx_list.index((s,k+1))] \
                        == 0
                ]
            if verbose == True :
                print(k, idx_list[idx_list.index((k,k+1))], idx_list[idx_list.index((s,k))], idx_list[idx_list.index((s,k+1))])

    # [3] V_obj <= V{n_points} : (s, n_points) interpolation condition
    lmbdK = cp.Variable(nonneg=True)
    constraints += [
        Vk_G[n_points,:,:] - V_obj_G \
            + lmbdK * A_list[idx_list.index((s,n_points))] >> 0,
        Vk_F[n_points,:] - V_obj_F \
            + lmbdK * b_list[idx_list.index((s,n_points))] == 0
    ]
    if verbose == True :
        print(idx_list[idx_list.index((s,n_points))])

    problem = cp.Problem(cp.Maximize(tau), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    return problem.status, problem.value, {
        'Vk_G': Vk_G.value, 'Vk_F': Vk_F.value,
        # 'Vk_A': Vk_A.value,
        'Vk_A': [Ak.value for Ak in Vk_A],
        'Vk_B': Vk_B.value,
        'lmbd0': lmbd0.value, 'lmbd': lmbd.value, 'lmbdK': lmbdK.value
    }


def dro_lyap_search_for_agd(mu, L, eta, n_points, eps_val=1e-4) :

    dimG = n_points + 2 # [x0-xs,    gx0,    gx1, ...,    gxn]
    dimF = n_points + 1 # [       fx0-fs, fx1-fs, ..., fxn-fs]

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
    
    z = x
    repZ = [z]

    for i in range(1, n_points+1):
        y_prev = y
        y = x - eta * g
        x = y + (i - 1) / (i + 2) * (y - y_prev) # xi

        g = eyeG[i+1, :]        # gi
        f = eyeF[i, :]          # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)

        z = y + (i + 2) / 2 * (x - y)
        repZ.append(z)

    assert len(repX) == len(repG) == len(repF) == n_points + 1, "constraint on same number of points"

    # Define the interpolation conditions
    repX.append(xs)
    repG.append(gs)
    repF.append(fs)
    s = len(repX) - 1   # index to the optimal point
    assert s == n_points+1, "s definition wrong"

    repZ.append(xs)

    # Obtain primal- and dual- PEP data
    _, primal_obj, primal_data = solve_agd_pep_primal(mu, L, eta, n_points)
    _, dual_obj, dual_data = solve_agd_pep_dual(mu, L, eta, n_points)

    G, F = primal_data['G'], primal_data['F']
    lmbd, tau, S = dual_data['lmbd'], dual_data['tau'], dual_data['S']

    # Generate samples
    _, _, avgG, avgF = sample_generation_agd(iter_K=n_points, N=100, mu=mu, L=L)
    assert avgG.shape == (dimG, dimG), "avgG shape"
    assert avgF.shape == (dimF,), "avgF shape"

    ##### Lyapunov DRO-PEP Definition

    # Lyapunov function definition
    Vk_G = cp.Variable((n_points+1, dimG, dimG))
    Vk_F = cp.Variable((n_points+1, dimF))
    constraints = [Vk_G[k,:,:] == Vk_G[k,:,:].T for k in range(n_points+1)]

    # Lyapunov function ansatz
    # Vk_A = cp.Variable((n_points+1, 6))
    Vk_A = []
    Vk_B = cp.Variable((n_points+1, 2))

    for k in range(n_points+1) :
        # if k+2 <= dimG-1 :
        #     constraints += [ Vk_G[k,k+2:,:] == 0 ] # ignore later iterates

        Ak = cp.Variable((3, 3), symmetric=True) # [xk-xs, zk-xs, gk]
        Pk = np.vstack([repX[k]-repX[s], repZ[k]-repZ[s], repG[k]])
        constraints += [
            Vk_G[k,:,:] == Pk.T @ Ak @ Pk,
            Ak[2,2] == 0,
        ]
        Vk_A.append(Ak)

        constraints += [ Vk_F[k,:] == Vk_B[k,0] * (repF[k] - repF[s]) ]
    
    # Initial condition
    V_initial_G = np.outer(repX[0] - repX[s], repX[0] - repX[s])
    V_initial_F = 0.0 * (repF[0] - repF[s])

    # Objective function
    tau = cp.Variable(nonneg=True)
    V_obj_G = tau * np.outer(repX[n_points] - repX[s], repX[n_points] - repX[s])
    V_obj_G = 0.0 * V_obj_G
    V_obj_F = tau * (repF[n_points] - repF[s]) # We maximize tau
    # V_obj_F = 0.0 * V_obj_F

    # Lyapunov PEP definition
    idx_list, A_list, b_list, _ = smooth_strongly_convex_gd(np.array(repX), np.array(repG), np.array(repF), mu, L)

    # [1] V0 <= V_initial : (s, 0) interpolation condition
    lmbd0 = cp.Variable(nonneg=True)
    t0 = cp.Variable()
    X0 = cp.Variable((dimG, dimG), symmetric=True)
    Y0 = cp.Variable(dimF)
    P0 = np.vstack([repX[0]-repX[s], repG[0]])
    constraints += [
        V_initial_G - Vk_G[0,:,:] + lmbd0 * A_list[idx_list.index((s,0))] + X0 >> 0,
        V_initial_F - Vk_F[0,:] + lmbd0 * b_list[idx_list.index((s,0))] + Y0 == 0,
        X0 == P0.T @ cp.Variable((2, 2), symmetric=True) @ P0,
        Y0 == cp.Variable() * (repF[0] - repF[s]),
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

        for k in range(n_points) :
            Pk1 = np.vstack([repX[k]-repX[s], repZ[k]-repZ[s], repG[k], repG[k+1]])
            constraints += [
                Vk_G[k,:,:] - Vk_G[k+1,:,:] \
                    + lmbd[k, 0] * A_list[idx_list.index((k,k+1))] \
                    + lmbd[k, 1] * A_list[idx_list.index((s,k))] \
                    # + lmbd[k, 2] * A_list[idx_list.index((s,k+1))] \
                    + X[k,:,:] \
                        >> 0,
                Vk_F[k,:] - Vk_F[k+1,:] \
                    + lmbd[k, 0] * b_list[idx_list.index((k,k+1))] \
                    + lmbd[k, 1] * b_list[idx_list.index((s,k))] \
                    # + lmbd[k, 2] * b_list[idx_list.index((s,k+1))] \
                    + Y[k, :] \
                        == 0,
                X[k,:,:] == Pk1.T @ cp.Variable((4, 4), symmetric=True) @ Pk1,
                Y[k, :] == cp.Variable() * (repF[k] - repF[s]) + cp.Variable() * (repF[k+1] - repF[s]),
                eps*t[k] + cp.trace(avgG@X[k,:,:]) + avgF.T@Y[k,:] <= 0,
                cp.SOC(t[k], cp.hstack([cp.vec(X[k,:,:], order='F'), Y[k,:]]))  # SOC constraint
            ]

    # [3] V_obj <= V{n_points} : (s, n_points) interpolation condition
    lmbdK = cp.Variable(nonneg=True)
    tK = cp.Variable()
    XK = cp.Variable((dimG, dimG), symmetric=True)
    YK = cp.Variable(dimF)
    PK = np.vstack([repX[n_points]-repX[s], repZ[n_points]-repZ[s], repG[n_points]])

    constraints += [
        Vk_G[n_points,:,:] - V_obj_G \
            + lmbdK * A_list[idx_list.index((s,n_points))] \
            + XK >> 0,
        Vk_F[n_points,:] - V_obj_F \
            + lmbdK * b_list[idx_list.index((s,n_points))] \
            + YK == 0,
        XK == PK.T @ cp.Variable((3, 3), symmetric=True) @ Pk,
        YK == cp.Variable() * (repF[n_points] - repF[s]),
        eps*tK + cp.trace(avgG@XK) + avgF.T@YK <= 0,
        cp.SOC(tK, cp.hstack([cp.vec(XK, order='F'), YK]))  # SOC constraint
    ]

    problem = cp.Problem(cp.Maximize(tau), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    sample_rate = ((repF[n_points] - repF[s]).T@avgF) / np.trace(V_initial_G.T@avgG)
    return problem.status, problem.value, {
        'Vk_G': Vk_G.value, 'Vk_F': Vk_F.value,
        'Vk_A': [Ak.value for Ak in Vk_A],
        'Vk_B': Vk_B.value,
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

    # print("Solve [Primal PEP]")
    # for n_points in range(1, iter_K) :
    #     primal_stat, primal_val, primal_var = solve_agd_pep_primal(mu, L, eta, n_points)
    #     print("[Primal PEP at iteration", n_points, "] Status:", primal_stat, ", Value:", primal_val)
    #     primal_obj.append(primal_val)

    #     if verbose == True :
    #         print("G =", primal_var['G'])
    #         print("F =", primal_var['F'])

    print("\nSolve [Dual PEP]")
    last_idx_list = None
    lmbd = None
    for n_points in range(1, iter_K) :
        dual_stat, dual_val, dual_var = solve_agd_pep_dual(mu, L, eta, n_points)
        print("[Dual PEP at iteration", n_points, "] Status:", dual_stat, ", Value:", dual_val)
        dual_obj.append(dual_val)
        s = n_points + 1

        if verbose == True :
            print("tau =", dual_var['tau'])
            print("S =", dual_var['S'])
            for m in range(len(idx_list)) :
                print("\tdual lamdas :")
                idx_list = dual_var['idx']
                lmbd = dual_var['lmbd']
                if len(idx_list[m]) == 2 :
                    (i, j) = idx_list[m]
                    print(f"\t\tlmbd[{i if i < s else "s"}, {j if j < s else "s"}] = {lmbd[m]}")
                else :
                    (xy1, xy2, i, j) = idx_list[m]
                    print(f"\t\tlmbd[{xy1}, {xy2}, {i if i < s else "s"}, {j if j < s else "s"}] = {lmbd[m]}")
        
        if n_points == iter_K - 1 :
            last_idx_list = copy(dual_var['idx'])
            lmbd = copy(dual_var['lmbd'])

    # Lyapunov search PEP
    lyap_obj = []
    print("\nSolve [Lyapunov PEP]")
    for n_points in range(1, iter_K) :
        lyap_stat, rate_inv, lyap_var = lyap_search_for_agd(mu, L, eta, n_points)
        lyap_obj.append(1.0 / rate_inv if rate_inv > 0 else np.inf)
        print("[Lyapunov PEP at iteration", n_points, "] Status:", lyap_stat, ", rate_inv:", rate_inv)

        if n_points == iter_K - 1 :
            for (Ak, Bk, lmbdk) in zip(lyap_var['Vk_A'], lyap_var['Vk_B'], lyap_var['lmbd']) :
            # for (Ak, Bk, lmbdk) in zip(lyap_var['Vk_A'], lyap_var['Vk_B'], lyap_var['lmbd']) :
                print("\tGk = ", Ak, np.linalg.eigvals(Ak))
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
        dro_lyap_stat, dro_rate_inv, dro_lyap_var = dro_lyap_search_for_agd(mu, L, eta, n_points, eps_val=eps)
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
    fig.suptitle("Smooth convex minimization: accelerated GD")

    # First subplot: Primal and Dual PEP
    axs[0].plot(range(1,iter_K), 2/(np.arange(1,iter_K)**2+7*np.arange(1,iter_K)+4), '*--', label='$2/(K^2+7K+4)$ secondary')   # secondary
    # axs[0].plot(range(1,iter_K), 2/(np.arange(1+2,iter_K+2)*np.arange(1+3,iter_K+3)), '^--', label='$2/(k+2)(k+3)$ primary')    # primary
    # axs[0].plot(range(1,iter_K), primal_obj, 'o-', label='Primal PEP')
    axs[0].plot(range(1,iter_K), dual_obj, 's--', label='Dual PEP')
    axs[0].plot(range(1,iter_K), lyap_obj, 'd:', label='Lyapunov PEP')
    axs[0].set_xlabel('Number of iterations $K$')
    axs[0].set_ylabel('$\\frac{f(x^K) - f^\\star}{\\|x^0-x^\\star\\|^2}$')
    axs[0].legend()
    axs[0].set_title('Primal/Dual PEP')
    axs[0].grid()

    # Second subplot: Lyapunov PEP and DRO-PEP
    axs[1].plot(range(1,iter_K), 2/(np.arange(1,iter_K)**2+7*np.arange(1,iter_K)+4), '*--', label='$2/(K^2+7K+4)$ secondary')   # secondary
    # axs[1].plot(range(1,iter_K), 2/(np.arange(1+2,iter_K+2)*np.arange(1+3,iter_K+3)), '^--', label='$2/(k+2)(k+3)$ primary')    # primary
    axs[1].plot(range(1,iter_K), lyap_obj, 'o-', label='worst-case')
    axs[1].plot(range(1,iter_K), sample_rate, '--', label='Sample Rate', linewidth=0.5, color='red')
    axs[1].plot(range(1,iter_K), dro_lyap_obj, 's--', label=f'DRO-PEP (eps={eps})')
    axs[1].set_xlabel('Number of iterations $K$')
    axs[1].set_ylabel('$\\frac{f(x^K) - f^\\star}{\\|x^0-x^\\star\\|^2}$')
    axs[1].legend()
    axs[1].set_title('Lyapunov PEP/DRO-PEP')
    axs[1].grid()

    # plt.tight_layout()
    plt.savefig('results/agd_pep_results.pdf', dpi=300)

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    plt.savefig('results/agd_pep_results_loglog.pdf', dpi=300)

    plt.show()
