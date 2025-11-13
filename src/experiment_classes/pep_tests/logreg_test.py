import cvxpy as cp
import numpy as np
import pandas as pd
import logging
import time
from tqdm import trange
from sklearn.datasets import load_breast_cancer
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from ucimlrepo import fetch_ucirepo

log = logging.getLogger(__name__)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


class LogReg(object):

    def __init__(self, n=100, N=1000, p=0.3, delta=0.1, R=1, seed=10, A_std=3, eps_std=0.01):
        np.random.seed(seed)
        self.beta = np.random.uniform(low=-1, high=1, size=(n,))
        self.beta_mask = np.random.binomial(1, p, size=(n,))
        self.beta = np.multiply(self.beta, self.beta_mask)
        # print(self.beta)

        self.x0 = np.zeros(n)
        self.x0[0] = R
        self.R = R
        self.dim = n
        self.delta = delta

        self.A = np.random.normal(size=(N, n), scale=A_std)
        # print(self.A)
        Abeta_noise = self.A @ self.beta + eps_std * np.random.normal(scale=eps_std, size=(N,))
        # print(Abeta_noise)
        self.b = np.where(Abeta_noise > 0, 1, 0)
        # print(self.b)

        self.solve_optimal_values()
        self.compute_mu_L()

    def solve_optimal_values(self):
        X, y = self.A, self.b
        m, n = X.shape
        beta = cp.Variable(n)
        log_likelihood = cp.sum(
            cp.multiply(y, X @ beta) - cp.logistic(X @ beta)
        )
        obj = - 1 / m * log_likelihood + 0.5 * self.delta * cp.sum_squares(beta)
        problem = cp.Problem(cp.Minimize(obj))
        problem.solve()

        self.x_opt = beta.value
        self.f_opt = problem.value

    def compute_mu_L(self):
        X = self.A
        m = X.shape[0]

        XTX_eigvals = np.real(np.linalg.eigvals(X.T @ X))
        lambd_max = np.max(XTX_eigvals)
        L = lambd_max / (4 * m) + self.delta
        mu = self.delta
        
        self.mu, self.L = mu, L

    def f(self, z):
        X, y = self.A, self.b
        m = X.shape[0]
        z = z + self.x_opt
        log_likeli = np.sum(np.multiply(y, X @ z) - np.logaddexp(0, X @ z))
        return - 1 / m * log_likeli + 0.5 * self.delta * z.T @ z - self.f_opt

    def grad(self, z):
        X, y = self.A, self.b
        m = X.shape[0]
        z = z + self.x_opt
        return 1 / m * X.T @ (sigmoid(X @ z) - y) + self.delta * z

    def sample_init_point(self):
        return sample_x0_centered_disk(self.dim, self.R)


class LogRegCredit(object):

    def __init__(self, sample_frac=0.5, delta=0.1, R=1):
        self.delta = delta

        # full_X, full_y = load_breast_cancer(return_X_y=True)
        default_of_credit_card_clients = fetch_ucirepo(id=350)
        X = default_of_credit_card_clients.data.features 
        y = default_of_credit_card_clients.data.targets 
        full_X = X.to_numpy()
        full_y = y.to_numpy().reshape(-1,)

        self.full_X = full_X
        self.full_y = full_y

        self.samp_X, self.samp_y = self.sample_normalized(sample_frac=sample_frac)
        self.solve_optimal_values()
        self.compute_mu_L()

        self.x0 = np.zeros(self.samp_X.shape[1])
        self.x0[0] = R
        self.R = R
        self.dim = self.samp_X.shape[1]

    def sample(self, sample_frac=0.8):
        X = self.full_X
        y = self.full_y

        sample_size = int(0.8 * X.shape[0])
        idx = np.random.choice(X.shape[0], size=sample_size, replace=False)

        return X[idx], y[idx]

    def sample_normalized(self, sample_frac=0.8):
        X_samp, y_samp = self.sample(sample_frac=sample_frac)
        means = X_samp.mean(axis=0)
        std_devs = X_samp.std(axis=0)

        X_samp_normalized = (X_samp - means) / std_devs
        ones_column = np.ones((X_samp.shape[0], 1))
        X_samp_with_ones = np.hstack((X_samp_normalized, ones_column))
        return X_samp_with_ones, y_samp

    def solve_optimal_values(self):
        X, y = self.samp_X, self.samp_y
        m, n = X.shape
        beta = cp.Variable(n)
        log_likelihood = cp.sum(
            cp.multiply(y, X @ beta) - cp.logistic(X @ beta)
        )
        obj = - 1 / m * log_likelihood + 0.5 * self.delta * cp.sum_squares(beta)
        problem = cp.Problem(cp.Minimize(obj))
        problem.solve()

        self.x_opt = beta.value
        self.f_opt = problem.value

    def compute_mu_L(self):
        X = self.samp_X
        m = X.shape[0]

        XTX_eigvals = np.real(np.linalg.eigvals(X.T @ X))
        lambd_max = np.max(XTX_eigvals)
        L = lambd_max / (4 * m) + self.delta
        mu = self.delta
        
        self.mu, self.L = mu, L

    def f(self, z):
        X, y = self.samp_X, self.samp_y
        m = X.shape[0]
        z = z + self.x_opt
        log_likeli = np.sum(np.multiply(y, X @ z) - np.logaddexp(0, X @ z))
        return - 1 / m * log_likeli + 0.5 * self.delta * z.T @ z - self.f_opt

    def grad(self, z):
        X, y = self.samp_X, self.samp_y
        m = X.shape[0]
        z = z + self.x_opt
        return 1 / m * X.T @ (sigmoid(X @ z) - y) + self.delta * z

    def sample_init_point(self):
        return sample_x0_centered_disk(self.dim, self.R)


def main():
    np.random.seed(0)

    delta = 0.1

    lr = LogReg(delta=delta)
    # X, y = lr.sample_normalized()
    # X, y = lr.samp_X, lr.samp_y
    X, y = lr.A, lr.b
    print(lr.mu, lr.L)

    # exit(0)

    m, n = X.shape
    beta = cp.Variable(n)
    log_likelihood = cp.sum(
        cp.multiply(y, X @ beta) - cp.logistic(X @ beta)
    )
    obj = - 1 / m * log_likelihood + 0.5 * delta * cp.sum_squares(beta)
    problem = cp.Problem(cp.Minimize(obj))
    problem.solve()
    print(beta.value)

    def f(z):
        # log_likeli = np.sum(np.multiply(y, X @ z) - np.log1p(np.exp(X @ z)))
        log_likeli = np.sum(np.multiply(y, X @ z) - np.logaddexp(0, X @ z))
        return - 1 / m * log_likeli + 0.5 * delta * z.T @ z

    def grad(z):
        return 1 / m * X.T @ (sigmoid(X @ z) - y) + delta * z

    beta_k = np.zeros(n)
    fp_resids = []
    for _ in range(1000):
        beta_new = beta_k - 0.1 * grad(beta_k)
        fp_resids.append(np.linalg.norm(beta_new - beta_k))
        beta_k = beta_new
    print(beta_k)
    # print(fp_resids)
    XTX_eigvals = np.real(np.linalg.eigvals(X.T @ X))
    lambd_max = np.max(XTX_eigvals)
    L = lambd_max / (4 * m) + delta
    print(delta, L)

    print(problem.value)
    print(f(beta.value))

    # f(x) -> f(x + x^\star) - f^\star

    print('--------')

    beta_k = np.ones(n)
    for _ in range(1000):
        beta_k = beta_k - 0.1 * lr.grad(beta_k)
    print(beta_k)
    print(lr.f(beta_k))
    print(lr.grad(beta_k))

    exit(0)

    default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
    # data (as pandas dataframes) 
    X = default_of_credit_card_clients.data.features 
    y = default_of_credit_card_clients.data.targets 
    X = X.to_numpy()
    y = y.to_numpy().reshape(-1,)
    print(X.shape, y.shape)
    print(X)
    print(y)

if __name__ == '__main__':
    main()
