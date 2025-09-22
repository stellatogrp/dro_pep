import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from copy import copy
from interpolation_conditions import smooth_strongly_convex, smooth_strongly_convex_gd
from sample_generation import sample_generation

d = 10


def ve(k, d):
    I = np.eye(d)
    return I[:, k:k+1]

def inner_product(GF1, GF2):
    G1, F1 = GF1
    G2, F2 = GF2

    assert G1.shape == G2.shape, "not same shape for G"
    assert F1.shape == F2.shape, "not same shape for F"

    return cp.trace(G1.T@G2) + cp.sum(F1.T@F2)

def gradient_descent(x, g, mu=0.0, L=1.0):
    return x - (1/L)*g

def iterate_representation(K, mu=0.0, L=1.0, algorithm=gradient_descent):
    eyeG = np.eye(K+2)
    eyeF = np.eye(K+1)

    x = eyeG[0, :]  # x0
    g = eyeG[1, :]  # g0
    f = eyeF[0, :]  # f0

    repX, repG, repF = [], [], []
    repX.append(x)
    repG.append(g)
    repF.append(f)

    # gradient descent
    for i in range(1, K+1):
        # xi-update
        x = algorithm(x, g, mu=mu, L=L)

        g = eyeG[i+1, :]    # gi
        f = eyeF[i, :]      # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)

    assert len(repX) == len(repG) == len(repF) == K + 1, "constraint on same number of points"

    repX.append(0.0*x)  # xs
    repG.append(0.0*g)  # gs
    repF.append(0.0*f)  # fs

    return np.array(repX), np.array(repG), np.array(repF)

def soc_constraint(t, X, Y, precond=None) : # precond = (precondG, precondF) : vectors, not matrix
    if precond is None:
        G_precond = np.ones(X.shape[0])
        F_precond = np.ones(Y.shape[0])
    else:
        G_precond = np.reciprocal(precond[0])
        F_precond = np.reciprocal(precond[1])
    
    return cp.SOC(t, cp.hstack([cp.vec(np.diag(G_precond).T@X@np.diag(G_precond), order='F'), cp.multiply(F_precond, Y)]))

def ansatz(k, Q, q) :
    ansatz_param = cp.Variable()
    ansatz_constr = [
        Q == ansatz_param * np.outer(ve(k, d), ve(k, d)),
        q == ansatz_param * ve(k, d)
    ]
    return ansatz_param, ansatz_constr

"""
    Input parameters:
        K : interation number
        N : number of samples
        mu : strong convexity parameter
        L : smoothness parameter
        interpolation : function that outputs interpolation condition (A_m, b_m)
        algorithm : x{k+1}-update given x{k} and g{k}
"""
def multiple_sample_lyapunov(K=5, N=100, mu=0.0, L=1.0, interpolation=smooth_strongly_convex_gd, algorithm=gradient_descent) :

    dimG, dimF = K+2, K+1

    # choose algorithm
    repX, repG, repF = iterate_representation(K, mu=mu, L=L, algorithm=algorithm)
    s = len(repX) - 1

    # interpolation condition: reference as `A_list[idx_list.index((i,j))]' for pair (i,j)
    idx_list, A_list, b_list, _ = interpolation(repX, repG, repF, mu=mu, L=L)

    # initial condition and objective
    Aobj, bobj = np.zeros((dimG, dimG)), (repF[K] - repF[s])
    A0, b0 = np.outer(repX[0] - repX[s], repX[0] - repX[s]), np.zeros((dimF, 1))

    # generate samples
    sampleG, sampleF, avgG, avgF = sample_generation(iter_K=K, N=N, mu=mu, L=L, algorithm=algorithm)
    average_tau = (np.trace(avgG.T@Aobj) + avgF.T@bobj) / (np.trace(avgG.T@A0) + avgF.T@b0)
    precond = (np.eye(dimG), np.ones((dimF, 1)))

    ##### Define tau-minimization problem #####
    dimG = K + 2
    dimF = K + 1

    # problem parameters
    eps = cp.Parameter(value=0.1)
    alpha = cp.Parameter(value=0.5) # 1.0 for expectation

    # optimization variables
    tau = cp.Variable()
    lmbd = cp.Variable(nonneg=True)
    X, Y = cp.Variable((K+2, dimG, dimG)), cp.Variable((K+2, dimF))  # universal over samples
    # X, Y = cp.Variable((K+2, N, dimG, dimG)), cp.Variable((K+2, N, dimF, 1))
    Q, q = cp.Variable((K+3, dimG, dimG)), cp.Variable((K+3, dimF))  # universal over samples
    y = cp.Variable((K+2, 3), nonneg=True)

    # constraints
    constr = []

    # Lyapunov ansatz
    constr += [
        Q[-1,:,:] == Aobj,  # Q[K+2]
        q[-1,:] == bobj,    # q[K+2]
        Q[0,:,:] == tau * A0,
        q[0,:] == tau * b0,
    ]
    constr += [ Q[k+1,:,:] == Q[k+1,:,:].T for k in range(K+1) ]    # symmetric Qk

    print(["1 " for _ in zip(X, Y, y)])

    # for (k1, (Xk, Yk, yk)) in enumerate(zip(X, Y, y)) :
        # k1 = 0,...,K+1
    for k1 in range(K+2) :
        k = k1 - 1  # -1,0,...,K

        Xk, Yk, yk = X[k1,:,:], Y[k1,:], y[k1,:]
        print("k1 =", k1, "/", K+1)

        # SOC constraint
        constr += [ soc_constraint(lmbd, Xk, Yk) ]

        for (Gi, Fi) in zip(sampleG, sampleF) :

            # inequality constraint
            constr += [ lmbd * eps - alpha * inner_product((Xk, Yk), (Gi, Fi)) <= 0 ]

        # psd and equality constraints
        if k == -1 :
            constr += [
                Q[0,:,:] - Q[1,:,:] - Xk \
                    + yk[0] * A_list[idx_list.index((s,0))]
                        >> 0,
                q[0,:] - q[1,:] - Yk \
                    + yk[0] * b_list[idx_list.index((s,0))]
                        == 0,
            ]
        elif k < K :
            constr += [
                Q[k1,:,:] - Q[k1+1,:,:] - Xk \
                    + yk[0] * A_list[idx_list.index((s,k))] \
                    + yk[1] * A_list[idx_list.index((k,k+1))] \
                    + yk[2] * A_list[idx_list.index((s,k+1))] \
                        >> 0,
                q[k1,:] - q[k1+1,:] - Yk \
                    + yk[0] * b_list[idx_list.index((s,k))] \
                    + yk[1] * b_list[idx_list.index((k,k+1))] \
                    + yk[2] * b_list[idx_list.index((s,k+1))] \
                        == 0,
            ]
        else : # k = K, k1 = K+1
            constr += [
                Q[k1,:,:] - Q[k1+1,:,:] - Xk \
                    + yk[0] * A_list[idx_list.index((s,K))] \
                        >> 0,
                q[k1,:] - q[k1+1,:] - Yk \
                    + yk[0] * b_list[idx_list.index((s,K))] \
                        == 0,
            ]

    # define problem
    obj = cp.Minimize(tau)
    prob = cp.Problem(objective=obj, constraints=constr)
    prob.solve(solver=cp.CLARABEL, verbose=True)

    # return solver status
    print(prob.status, prob.value, average_tau)
    return prob.status, prob.value



def multiple_sample_one_step_fixed_lyapunov(
    K=10,   # iteration number to observe up to
    N=100,  # sample number
    mu=0.0, # strong convexity parameter
    L=1.0,  # smoothness parameter
    eps_val=1e-1,
    alpha_val=1.0,
    interpolation=smooth_strongly_convex_gd,
    algorithm=gradient_descent,
    precond=False,   # True to use preconditioner
) :
    
    dimG, dimF = K+2, K+1

    # algorithm representation
    repX, repG, repF = iterate_representation(K, mu=mu, L=L, algorithm=algorithm)
    star = len(repX) - 1

    # sample generation
    sampleG, sampleF, _, _ = sample_generation(iter_K=K, N=N, mu=mu, L=L, algorithm=algorithm)

    # sample trimming: obtain sample G for k-th and (k+1)-th iterate -> [xk-xs, gk, g{k+1}]
    sampleG_trim, sampleF_trim = [], []
    for (Gi, Fi) in zip(sampleG, sampleF) :
        k = np.random.randint(K)  # random iterate number k
        Xk = np.vstack([repX[k] - repX[star], repG[k], repG[k+1]]).T    # shape: (K+2, 3)
        xk = np.array([repF[k] - repF[star], repF[k+1] - repF[star]]).T # shape: (K+2, 2)

        # extract partial information
        Gi_trim = Xk.T@Gi@Xk
        Fi_trim = xk.T@Fi

        # include scaling: [(G, F) \in \fclass && t > 0] ==> [t(G, F) \in \fclass]
        Gi_trim, Fi_trim = Gi_trim / Gi_trim[0,0], Fi_trim / Gi_trim[0,0]
        sampleG_trim.append(Gi_trim)
        sampleF_trim.append(Fi_trim)
    
    # reduced to [x0-xs, g0, g1] sample
    sampleG = np.array(sampleG_trim)
    sampleF = np.array(sampleF_trim)
    
    K = 1
    dimG, dimF = K+2, K+1

    # define preconditioner
    precondG = np.ones(dimG)
    precondF = np.ones(dimF)

    if precond :
        # D_G, D_F value as (K+2)- and (K+1)-vector -> multiplied on (G, F)
        precondG = (1/dimG) / np.average([np.sqrt(np.diag(Gi)) for Gi in sampleG], axis=0)
        precondF = (1/dimF) / np.average(sampleF, axis=0)

        assert precondG.shape == np.diag(sampleG[0]).shape, "precondG shape mismatch"
        assert precondF.shape == sampleF[0].shape, "precondF shape mismatch"
    
    # reduce K for one-step update: K = 1
    repX, repG, repF = iterate_representation(K, mu=mu, L=L, algorithm=algorithm)
    star = len(repX) - 1

    # objective function definition
    Aobj, bobj = np.outer(repX[1] - repX[star], repX[1] - repX[star]), np.zeros((dimF, 1))

    # initial condition definition
    A0, b0 = np.outer(repX[0] - repX[star], repX[0] - repX[star]), np.zeros((dimF, 1))

    # interpolation condition: reference as `A_list[idx_list.index((i,j))]' for pair (i,j)
    idx_list, A_list, b_list, _ = interpolation(repX, repG, repF, mu=mu, L=L)
    M = len(idx_list)

    # check if the samples are valid:
    for i, (Gi, Fi) in enumerate(zip(sampleG, sampleF)) :
        check_list = np.array([inner_product((Am, bm), (Gi, Fi)).value for (Am, bm) in zip(A_list, b_list)])
        if np.sum(check_list > 0.0) > 0 :
            print(i, np.sum(check_list > 0.0))
    
    # define adjoint operator Sstar
    def Sstar(y) :
        assert y.shape[0] == M

        SstarA, Sstarb = 0, 0
        for m in range(M) :
            SstarA += - y[m]*A_list[m]
            Sstarb += - y[m]*b_list[m]
        
        return SstarA, Sstarb

    # define lyapunov pep problem
    dro_eps = cp.Parameter(value=1e-2)

    # specify problem parameter
    dro_eps.value = eps_val
    dro_alpha = alpha_val

    # define optimization variable
    rho = cp.Variable()     # contraction factor
    s = cp.Variable(N)
    t = cp.Variable()
    lmbda = cp.Variable(nonneg=True)
    y = cp.Variable((2, M), nonneg=True)
    X, Y = cp.Variable((2, dimG, dimG)), cp.Variable((2, dimF))
    constraint = [ X[j,:,:] == X[j,:,:].T for j in range(2)]

    # lyapunov ansatz: V(G, F) = |xk-xs|^2
    Q0, q0 = A0, b0
    Q1, q1 = Aobj, bobj
    
    # residual term: subject to change
    R, r = cp.Variable((dimG, dimG), symmetric=True), cp.Variable((1, dimF))
    eta = cp.Variable(nonneg=True)
    constraint += [ R == 0 ]
    constraint += [ r >= eta * (repF[0] - repF[star]) ]
    # constraint += [ r >= eta * repF[1] - repF[star] ]
 
    # coupling constraint
    constraint += [ (1/N) * cp.sum(s) <= 0 ]
    
    # sample-dependent constraints
    for i in range(N) :
        Gi, Fi = sampleG[i], sampleF[i]
        constraint += [ (1-1/dro_alpha)*t - inner_product((X[1], Y[1]), (Gi, Fi)) + lmbda*dro_eps.value <= s[i] ]
        if dro_alpha < 1.0 :
            constraint += [ t - inner_product((X[0], Y[0]), (Gi, Fi)) + lmbda*dro_eps.value <= s[i] ]

    # if y, X, Y are universal over i=1:N
    if dro_alpha < 1.0 : # cvar
        SstarA, Sstarb = Sstar(y[0])
        constraint += [
            - SstarA - X[0] >> 0,
            - Sstarb - Y[0] == 0,
        ]
    
    # rho*Q0 is bilinear: set Q0, q0 as fixed ones (distance-to-solution)
    SstarA, Sstarb = Sstar(y[1])
    constraint += [
        - SstarA - X[1] + (1/dro_alpha) * (rho*Q0 - Q1 - R) >> 0,
        - Sstarb - Y[1] + (1/dro_alpha) * (rho*q0 - q1 - r) == 0,
    ]

    constraint += [ soc_constraint(lmbda, X[j], Y[j], precond=(precondG, precondF)) for j in range(2) ]

    problem = cp.Problem(cp.Minimize(rho), constraints=constraint)
    # problem.solve(solver=cp.CLARABEL, verbose=False)
    problem.solve(solver=cp.MOSEK, verbose=False)


    # sample objective rho for comparison
    sample_rhos = np.array([
        (np.trace(Gi.T@Aobj)+Fi.T@bobj) / (np.trace(Gi.T@A0)+Fi.T@b0) for (Gi, Fi) in zip(sampleG_trim, sampleF_trim)
    ])
    alpha_cvar = cp.Variable(N, nonneg=True)
    prob_cvar = cp.Problem(
        cp.Maximize( (1/N) * cp.sum([alpha_cvar[i]*sample_rhos[i] for i in range(N)]) ),
        [ (1/N) * cp.sum(alpha_cvar) <= 1, alpha_cvar <= (1/dro_alpha) ]
        )
    prob_cvar.solve(solver=cp.MOSEK, verbose=False)
    sample_cvar = prob_cvar.value

    # output solver result
    print(f"solver_status: {problem.status}, eps_val: {dro_eps.value}, alpha: {dro_alpha}")
    print(f"sample_cvar_rho = {sample_cvar} <= dro_rho = {rho.value} <= 1.0")
    print(f"eta: {eta.value} >= 0.0")



def multiple_sample_one_step_lyapunov_search(
    K=10,   # iteration number to observe up to
    N=100,  # sample number
    mu=0.0, # strong convexity parameter
    L=1.0,  # smoothness parameter
    interpolation=smooth_strongly_convex_gd,
    algorithm=gradient_descent,
    precond=False,   # True to use preconditioner
    residual=False
) :
    
    dimG, dimF = K+2, K+1

    # algorithm representation
    repX, repG, repF = iterate_representation(K, mu=mu, L=L, algorithm=algorithm)
    star = len(repX) - 1

    # sample generation
    sampleG, sampleF, _, _ = sample_generation(iter_K=K, N=N, mu=mu, L=L, algorithm=algorithm)

    # sample trimming: obtain sample G for k-th and (k+1)-th iterate -> [xk-xs, gk, g{k+1}]
    sampleG_trim, sampleF_trim = [], []
    for (Gi, Fi) in zip(sampleG, sampleF) :
        k = np.random.randint(K)  # random iterate number k
        Xk = np.vstack([repX[k] - repX[star], repG[k], repG[k+1]]).T    # shape: (K+2, 3)
        xk = np.array([repF[k] - repF[star], repF[k+1] - repF[star]]).T # shape: (K+2, 2)

        # extract partial information
        Gi_trim = Xk.T@Gi@Xk
        Fi_trim = xk.T@Fi

        # include scaling: [(G, F) \in \fclass && t > 0] ==> [t(G, F) \in \fclass]
        Gi_trim, Fi_trim = Gi_trim / Gi_trim[0,0], Fi_trim / Gi_trim[0,0]
        sampleG_trim.append(Gi_trim)
        sampleF_trim.append(Fi_trim)
    
    # reduced to [x0-xs, g0, g1] sample
    sampleG = np.array(sampleG_trim)
    sampleF = np.array(sampleF_trim)
    
    K = 1
    dimG, dimF = K+2, K+1

    # define preconditioner
    precondG = np.ones(dimG)
    precondF = np.ones(dimF)

    if precond :
        # D_G, D_F value as (K+2)- and (K+1)-vector -> multiplied on (G, F)
        precondG = (1/dimG) / np.average([np.sqrt(np.diag(Gi)) for Gi in sampleG], axis=0)
        precondF = (1/dimF) / np.average(sampleF, axis=0)

        assert precondG.shape == np.diag(sampleG[0]).shape, "precondG shape mismatch"
        assert precondF.shape == sampleF[0].shape, "precondF shape mismatch"
    
    # reduce K for one-step update: K = 1
    repX, repG, repF = iterate_representation(K, mu=mu, L=L, algorithm=algorithm)
    star = len(repX) - 1

    # objective function definition
    Aobj, bobj = np.outer(repX[1] - repX[star], repX[1] - repX[star]), np.zeros((dimF, 1))

    # initial condition definition
    A0, b0 = np.outer(repX[0] - repX[star], repX[0] - repX[star]), np.zeros((dimF, 1))

    # interpolation condition: reference as `A_list[idx_list.index((i,j))]' for pair (i,j)
    idx_list, A_list, b_list, _ = interpolation(repX, repG, repF, mu=mu, L=L)
    M = len(idx_list)

    # check if the samples are valid:
    for i, (Gi, Fi) in enumerate(zip(sampleG, sampleF)) :
        check_list = np.array([inner_product((Am, bm), (Gi, Fi)).value for (Am, bm) in zip(A_list, b_list)])
        if np.sum(check_list > 0.0) > 0 :
            print(i, np.sum(check_list > 0.0))
    
    # define adjoint operator Sstar
    def Sstar(y) :
        assert y.shape[0] == M

        SstarA, Sstarb = 0, 0
        for m in range(M) :
            SstarA += - y[m]*A_list[m]
            Sstarb += - y[m]*b_list[m]
        
        return SstarA, Sstarb

    # define lyapunov pep problem
    dro_eps = cp.Parameter(value=1e-2)
    dro_alpha_inv = cp.Parameter(value=1.0/0.1)
    rho = cp.Parameter(value=100.0)

    # rho = cp.Variable()
    s = cp.Variable(N)
    t = cp.Variable()
    lmbda = cp.Variable(nonneg=True)
    y = cp.Variable((4, M), nonneg=True)
    X, Y = cp.Variable((4, dimG, dimG)), cp.Variable((4, dimF))
    constraint = [ X[i,:,:] == X[i,:,:].T for i in range(4) ]

    # lyapunov ansatz parameter: V(G, F) = <(Q, q), (G, F)>
    Q, q = cp.Variable((2, 2), symmetric=True), cp.Variable()
    
    X0 = np.vstack([repX[0] - repX[star], repG[0]]) # shape: (2, K+2)
    x0 = np.array([repF[0] - repF[star]])           # shape: (1, K+2)
    X1 = np.vstack([repX[1] - repX[star], repG[0]]) # shape: (2, K+2)
    x1 = np.array([repF[1] - repF[star]])           # shape: (1, K+2)

    Q0, q0 = X0.T@Q@X0, q*x0
    Q1, q1 = X1.T@Q@X1, q*x1
    
    # residual term
    R = np.zeros((dimG, dimG))
    r = np.zeros((1, dimF))

    if residual :
        R, r = cp.Variable((dimG, dimG), symmetric=True), cp.Variable((1, dimF))
        eta = cp.Variable(nonneg=True)
        constraint += [ R == 0]
        constraint += [ r >= eta * (repF[0] - repF[star]) ]
        # constraint += [ r >= eta * repF[1] - repF[star] ]
 
    constraint += [ (1/N) * cp.sum(s) <= 0 ]
    
    # sample-dependent constraints
    for i in range(N) :
        Gi, Fi = sampleG[i], sampleF[i]

        constraint += [ t - inner_product((X[0], Y[0]), (Gi, Fi)) + lmbda*dro_eps <= s[i] ]
        constraint += [ (1-dro_alpha_inv)*t - inner_product((X[1], Y[1]), (Gi, Fi)) + lmbda*dro_eps <= s[i] ]
        constraint += [ (1-dro_alpha_inv)*t - inner_product((X[2], Y[2]), (Gi, Fi)) + lmbda*dro_eps <= s[i] ]
        constraint += [ (1-dro_alpha_inv)*t - inner_product((X[3], Y[3]), (Gi, Fi)) + lmbda*dro_eps <= s[i] ]

    # if y, X, Y are universal over i=1:N
    SstarA, Sstarb = Sstar(y[0])
    constraint += [
        - SstarA - X[0] >> 0,
        - Sstarb - Y[0] == 0,
        ]
    
    SstarA, Sstarb = Sstar(y[1])
    constraint += [
        - SstarA - X[1] + dro_alpha_inv * (Q1 - Aobj) >> 0,
        - Sstarb - Y[1] + dro_alpha_inv * (q1 - bobj) == 0,
        ]
    
    # rho*Q0 is bilinear: set Q0, q0 as fixed ones (distance-to-solution)
    SstarA, Sstarb = Sstar(y[2])
    constraint += [
        - SstarA - X[2] + dro_alpha_inv * (rho*Q0 - Q1 - R) >> 0,
        - Sstarb - Y[2] + dro_alpha_inv * (rho*q0 - q1 - r) == 0,
        ]
    
    SstarA, Sstarb = Sstar(y[3])
    constraint += [
        - SstarA - X[3] + dro_alpha_inv * (A0 - Q0) >> 0,
        - Sstarb - Y[3] + dro_alpha_inv * (b0 - q0) == 0,
        ]

    # constraint += [ soc_constraint(lmbda, X[j], Y[j], precond=(precondG, precondF)) for j in range(4) ]
    constraint += [ soc_constraint(lmbda, X[j], Y[j]) for j in range(4) ]

    problem = cp.Problem(cp.Minimize(rho), constraints=constraint)
    rho.value = 1.0
    dro_eps.value = 1e0
    dro_alpha_inv = 1.0 / 1.0

    # problem.solve(solver=cp.CLARABEL, verbose=False)
    problem.solve(solver=cp.MOSEK, verbose=True)


    # sample objective rho for comparison
    sample_rho = [
        (np.trace(Gi.T@Aobj)+Fi.T@bobj) / (np.trace(Gi.T@A0)+Fi.T@b0) for (Gi, Fi) in zip(sampleG_trim, sampleF_trim)
    ]
    average_rho = np.average(sample_rho) # for comparison

    # output solver result
    print(f"solver_status: {problem.status}")
    print(f"sample_rho: {average_rho} <= 1.0")
    print(f"dro_rho: {rho.value} <= 1.0")
    print(f"Q, q: {Q.value},\n {q.value}\n")







if __name__ == "__main__" :
    # multiple_sample_lyapunov(K=3, N=1, mu=0.0, L=1.0)
    multiple_sample_one_step_fixed_lyapunov(K=20, N=100, eps_val=1e-1, alpha_val=0.25, precond=True)

