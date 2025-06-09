import numpy as np
import pandas as pd
import logging
import scipy as sp
import time
from tqdm import trange

from .utils import generate_P_bounded_mu_L, gradient_descent, generate_trajectories

log = logging.getLogger(__name__)


def generate_A(cfg):
    np.random.seed(cfg.seed)
    A = np.random.normal(scale=1/cfg.m, size=(cfg.m, cfg.n))

    A_mask = np.random.binomial(1, p=cfg.p_A_nonzero, size=(cfg.m, cfg.n))

    A = np.multiply(A, A_mask)

    A = A / np.linalg.norm(A, axis=0)
    return A


def generate_b_samples(cfg, A, gamma, strong_cvx=True):
    rads = []
    opt_bounds = []

    ATA = A.T @ A
    lu, piv = sp.linalg.lu_factor(ATA)

    # x_samp = np.random.normal(size=(cfg.n,))
    # x_mask = np.random.binomial(1, p=cfg.p_xsamp_nonzero, size=(cfg.n,))
    # x_samp = np.multiply(x_samp, x_mask)
    # b = A @ x_samp + cfg.noise_eps * np.random.normal(size=(cfg.m, ))
    # x0 = sp.linalg.lu_solve((lu, piv), A.T @ b)

    x0 = np.zeros(cfg.n)

    for _ in trange(cfg.N):
        x_samp = np.random.normal(size=(cfg.n,))
        x_mask = np.random.binomial(1, p=cfg.p_xsamp_nonzero, size=(cfg.n,))
        x_samp = np.multiply(x_samp, x_mask)

        b = A @ x_samp + cfg.noise_eps * np.random.normal(size=(cfg.m, ))
        
        x_star = sp.linalg.lu_solve((lu, piv), A.T @ b)

        f_star = .5 * np.linalg.norm(A @ x_star - b) ** 2
        def f(x):
            return .5 * np.linalg.norm(A @ x - b) ** 2 - f_star

        # x0 = np.zeros(cfg.n)
        rads.append(np.linalg.norm(x0 - x_star))

        opt_bounds_i = []
        xk = x0
        for _ in range(cfg.K_max):
            xk = xk - gamma * A.T @ (A @ xk - b)
            if strong_cvx:
                opt_bounds_i.append(np.linalg.norm(xk - x_star))
            else:
                opt_bounds_i.append(f(xk) - f_star)
        opt_bounds.append(opt_bounds_i)
    
    R_max = np.max(rads)
    log.info(f'max rad: {R_max}')

    return R_max, opt_bounds


def lstsq_samples(cfg):
    log.info(cfg)
    
    A = generate_A(cfg)
    log.info(A)
    
    eigvals = np.linalg.eigvals(A.T @ A)
    mu = np.real(np.min(eigvals))
    L = np.real(np.max(eigvals))
    log.info(f'mu, L: {mu, L}')
    if cfg.gamma == 'opt':
        gamma = 2 / (mu + L)
    else:
        gamma = cfg.gamma / L
    log.info(f'gamma: {gamma}')

    if cfg.m >= cfg.n:
        R_max, opt_dists = generate_b_samples(cfg, A, gamma)
        opt_dists = np.array(opt_dists)
        log.info(opt_dists)
        df = pd.DataFrame(opt_dists)
        df.to_csv('samples.csv', header=None, index=None)

        # theory_bounds = []
        tau = (L - mu) / (L + mu)

        theory_bounds = [(tau ** k) * R_max for k in range(1, cfg.K_max + 1)]
        df = pd.DataFrame(theory_bounds)
        df.to_csv('theory.csv', header=None, index=None)
    else:
        R_max, opt_objs = generate_b_samples(cfg, A, gamma, strong_cvx=False)
