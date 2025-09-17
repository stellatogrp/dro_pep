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

def soc_constraint(t, X, Y, precond=None) :
    if precond is None:
        G_precond = np.ones(X.shape[0])
        F_precond = np.ones(Y.shape[0])
    else:
        G_precond = np.reciprocal(np.diag(precond[0]))
        F_precond = np.reciprocal(precond[1])
    
    # return cp.SOC(t, cp.hstack(
    #         [cp.vec( np.outer(G_precond, G_precond)@X, order='F' ), cp.multiply(F_precond**2, Y)]
    #     ))
    return cp.SOC(t, cp.hstack([cp.vec(X, order='F'), Y]))


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
    # obj = cp.Maximize(tau)
    obj = cp.Minimize(tau)
    prob = cp.Problem(objective=obj, constraints=constr)
    prob.solve(solver=cp.CLARABEL, verbose=True)

    # return solver status
    print(prob.status, prob.value, average_tau)
    return prob.status, prob.value












if __name__ == "__main__" :
    multiple_sample_lyapunov(K=3, N=1, mu=0.0, L=1.0)
    pass

