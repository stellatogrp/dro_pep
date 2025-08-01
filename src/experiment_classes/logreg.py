import cvxpy as cp
import numpy as np
from sklearn.datasets import load_breast_cancer


class LogReg(object):

    def __init__(self):
        full_X, full_y = load_breast_cancer(return_X_y=True)
        # print(full_X)
        # print(full_y)
        # print(full_X.shape, full_y.shape)
        self.full_X = full_X
        self.full_y = full_y

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


def main():
    np.random.seed(0)
    lr = LogReg()

    X, y = lr.sample_normalized()
    # print(X_samp, y_samp)
    # print(X_samp.shape, y_samp.shape)

    delta = 0.1

    m, n = X.shape
    beta = cp.Variable(n)
    log_likelihood = cp.sum(
        cp.multiply(y, X @ beta) - cp.logistic(X @ beta)
    )
    obj = - 1/ m * log_likelihood + 0.5 * delta * cp.sum_squares(beta)
    problem = cp.Problem(cp.Minimize(obj))
    problem.solve()
    print(beta.value)

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def f(z):
        log_likeli = np.sum(np.multiply(y, X @ z) - np.log1p(np.exp(X @ z)))
        return - 1/ m * log_likeli + 0.5 * delta * z.T @ z

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
    print(L)

    print(problem.value)
    print(f(beta.value))


if __name__ == '__main__':
    main()
