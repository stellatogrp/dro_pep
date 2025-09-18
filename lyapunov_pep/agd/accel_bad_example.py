import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(5)
    n = 300
    mu = 1

    # P = np.array([
    #     [1, 0, 0],
    #     [0, 2, 0],
    #     [0, 0, 3],
    # ])
    offset = 1.0

    P = np.diag(mu + offset * np.random.uniform(size=(n,)))

    def f(z):
        return .5 * z.T @ P @ z 
    
    def grad_f(z):
        return P @ z

    # L = np.max(np.linalg.eigvals(P))
    L = mu + offset
    eta = 1 / L
    K = 30

    zk = 10 * np.ones(n)
    GD_vals = [f(zk)]
    for _ in range(K):
        zk = zk - eta * grad_f(zk)
        GD_vals.append(f(zk))

    zk = 10 * np.ones(n)
    AGD_vals = [f(zk)]
    y_prev = zk
    for k in range(1, K+1):
        yk = zk - eta * grad_f(zk)
        zk = yk + (k - 1) / (k + 2) * (yk - y_prev)

        AGD_vals.append(f(zk))
        y_prev = yk

    print(GD_vals)
    print(AGD_vals)

    fix, ax = plt.subplots()
    plt.plot(range(K+1), GD_vals, label='GD')
    plt.plot(range(K+1), AGD_vals, label='AGD')

    plt.legend()
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
