"""Logistic regression certification experiment (DRO-PEP) on real data.

Unregularized (delta = 0, smooth convex, mu = 0) logistic regression on
subsample instances of a LIBSVM dataset (see logreg_data.py). Mirrors the
Lasso experiment: three stages (samples / pep / dro) sharing deterministic
family constants L, R calibrated from cfg.seed.in_sample reference instances,
with the modern custom-interpolation DRO path (Clarabel) and cross-validated
epsilon support via sample_summary_dist.csv.

Algorithms: gradient descent ('grad_desc') and Nesterov's fast gradient
method ('nesterov_fgm'), step size eta/L, started at x0 = 0 (so the shifted
initial point is z0 = -x_opt and ||x0 - x*|| <= R by calibration).
"""
import diffcp_patch  # noqa: F401  # COO->CSC fix for diffcp/clarabel
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import trange

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction

from reformulator.dro_reformulator import DROReformulator
from learning.pep_constructions import (
    construct_gd_pep_data, construct_fgm_pep_data, pep_data_to_numpy,
)
from learning.trajectories import logreg_gd_trajectories, logreg_fgm_trajectories
from learning.acceleration_stepsizes import jax_get_nesterov_fgm_beta_sequence

from .utils import build_eps_vals
from .lasso import summarize_per_k, plot_sample_summary
from .logreg_data import load_dataset, compute_L_R, sample_instance_batch

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


# =============================================================================
# Numpy objective/gradient and simulation (mirrors utils' PEP conventions)
# =============================================================================

def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def logreg_gap(A, b, x, f_opt, delta):
    """f(x) - f_opt for one instance."""
    m = A.shape[0]
    Ax = A @ x
    log_likeli = np.sum(b * Ax - np.logaddexp(0, Ax))
    val = -1 / m * log_likeli + 0.5 * delta * float(x @ x)
    return val - f_opt


def logreg_grad(A, b, x, delta):
    m = A.shape[0]
    return 1 / m * A.T @ (_sigmoid(A @ x) - b) + delta * x


def simulate_alg(cfg, A, b, x_opt, f_opt, t):
    """Run cfg.alg from x0 = 0 and return the metric [m(x_k), k=1..K_max],
    where m is cfg.pep_obj ('obj_val' -> f(x_k)-f*, 'grad_sq_norm' ->
    ||grad f(x_k)||^2).

    Iterate conventions match construct_gd_pep_data / construct_fgm_pep_data
    (and the PEPit baseline below), so all three stages bound the same quantity.
    """
    K_max = cfg.K_max
    delta = cfg.delta
    d = A.shape[1]
    x = np.zeros(d)
    gaps = []

    def metric(xk):
        if cfg.pep_obj == 'obj_val':
            return logreg_gap(A, b, xk, f_opt, delta)
        elif cfg.pep_obj == 'grad_sq_norm':
            g = logreg_grad(A, b, xk, delta)
            return float(g @ g)
        raise ValueError(f"unsupported pep_obj '{cfg.pep_obj}'")

    if cfg.alg == 'grad_desc':
        for _ in range(K_max):
            x = x - t * logreg_grad(A, b, x, delta)
            gaps.append(metric(x))
    elif cfg.alg == 'nesterov_fgm':
        # FGM in the same form as construct_fgm_pep_data and
        # logreg_fgm_trajectories: gradients at y_k, objective at
        # x_{k+1} = y_k - t g(y_k), y_{k+1} = x_{k+1} + beta_k (x_{k+1} - x_k).
        # Beta recursion identical to jax_get_nesterov_fgm_beta_sequence.
        q = cfg.delta / cfg.L
        A_seq = [0.0, 1.0 / (1.0 - q)]
        y = x.copy()
        x_curr = x.copy()
        for k in range(K_max):
            A_next_num = 2 * A_seq[k + 1] + 1 + np.sqrt(
                1 + 4 * A_seq[k + 1] + 4 * q * A_seq[k + 1] ** 2)
            A_seq.append(A_next_num / (2 * (1 - q)))
            beta_num = (A_seq[k + 2] - A_seq[k + 1]) * (A_seq[k + 1] * (1 - q) - A_seq[k] - 1)
            beta_den = A_seq[k + 2] * (2 * q * A_seq[k + 1] + 1) - q * A_seq[k + 1] ** 2
            beta_k = beta_num / beta_den

            x_new = y - t * logreg_grad(A, b, y, delta)
            y = x_new + beta_k * (x_new - x_curr)
            x_curr = x_new
            gaps.append(metric(x_new))
    else:
        raise NotImplementedError(f"unsupported alg '{cfg.alg}'")

    return gaps


# =============================================================================
# Stage 1: samples
# =============================================================================

def logreg_samples(cfg):
    log.info(cfg)
    A_full, b_full = load_dataset(cfg.dataset, intercept=cfg.intercept,
                                  standardize=cfg.get('standardize', False))
    compute_L_R(cfg, A_full, b_full)   # sets cfg.L, cfg.R (consistent across stages)
    t = cfg.eta / cfg.L
    log.info(f'L: {cfg.L}, R: {cfg.R}, step size: {t}')

    alpha_vals = list(cfg.alpha_vals)
    n_repeats = cfg.cross_val_repeats

    # The repeats must not redraw the in-sample set, or one of them silently
    # becomes in-sample and the "cross-validation" coverage is optimistic.
    if 0 <= cfg.seed.in_sample - cfg.seed.out_of_sample < n_repeats:
        raise ValueError(
            f'seed.in_sample={cfg.seed.in_sample} falls inside the repeat range '
            f'[{cfg.seed.out_of_sample}, {cfg.seed.out_of_sample + n_repeats}); '
            f'move seed.out_of_sample so the two streams stay disjoint')

    dist = []
    for j in trange(n_repeats):
        rng = np.random.default_rng(cfg.seed.out_of_sample + j)
        instances, _ = sample_instance_batch(
            rng, A_full, b_full, cfg, int(cfg.N), desc=f'repeat {j}')

        df = []
        for i, (A, b, x_opt, f_opt, L_i) in enumerate(instances):
            gaps = simulate_alg(cfg, A, b, x_opt, f_opt, t)
            for k in range(1, cfg.K_max + 1):
                df.append(pd.Series({'i': i, 'K': k, 'obj_val': gaps[k - 1]}))

        df_to_save = pd.DataFrame(df)
        summary = summarize_per_k(df_to_save, alpha_vals, 'obj_val', cfg.K_max)

        if j == 0:
            # Repeat-0 artifacts reproduce the legacy single-run outputs.
            df_to_save.to_csv(cfg.sample_fname, index=False)
            summary.to_csv('sample_summary.csv', index=False)
            log.info(summary)
            plot_sample_summary(summary, cfg)

        dist.append(summary.assign(repeat=j))

    dist_df = pd.concat(dist, ignore_index=True)
    dist_df = dist_df[['repeat'] + [c for c in dist_df.columns if c != 'repeat']]
    dist_df.to_csv('sample_summary_dist.csv', index=False)


# =============================================================================
# Stage 2: worst-case PEP baseline (PEPit)
# =============================================================================

def logreg_pep(cfg):
    log.info(cfg)
    A_full, b_full = load_dataset(cfg.dataset, intercept=cfg.intercept,
                                  standardize=cfg.get('standardize', False))
    compute_L_R(cfg, A_full, b_full)   # sets cfg.L, cfg.R (consistent across stages)

    res = []
    for k in range(cfg.K_min, cfg.K_max + 1):
        tau, solvetime = logreg_pep_subproblem(cfg, cfg.delta, cfg.L, k, cfg.pep_obj)
        res.append(pd.Series({
            'K': k,
            'obj': cfg.pep_obj,
            'val': tau,
            'solvetime': solvetime,
        }))
        df = pd.DataFrame(res)
        df.to_csv(cfg.pep_fname, index=False)


def logreg_pep_subproblem(cfg, mu, L, k, obj, return_problem=False):
    """Worst-case PEP for GD / FGM over mu-strongly convex L-smooth functions.

    Iterate conventions match construct_gd_pep_data / construct_fgm_pep_data
    (FGM: gradients at y_k, objective at x_K = y_{K-1} - t g(y_{K-1})), so the
    DRO bound converges to this PEP value as eps grows.
    """
    problem = PEP()
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= cfg.R ** 2)

    t = cfg.eta / cfg.L
    if cfg.alg == 'grad_desc':
        x = x0
        for _ in range(k):
            x = x - t * func.gradient(x)
        x_last = x
    elif cfg.alg == 'nesterov_fgm':
        beta = np.asarray(jax_get_nesterov_fgm_beta_sequence(mu, L, k))
        y = x0
        x_curr = x0
        for j in range(k):
            x_new = y - t * func.gradient(y)
            y = x_new + float(beta[j]) * (x_new - x_curr)
            x_curr = x_new
        x_last = x_curr
    else:
        raise NotImplementedError(f"unsupported alg '{cfg.alg}'")

    if obj == 'obj_val':
        problem.set_performance_metric(func(x_last) - fs)
    elif obj == 'grad_sq_norm':
        problem.set_performance_metric(func.gradient(x_last) ** 2)
    elif obj == 'opt_dist_sq_norm':
        problem.set_performance_metric((x_last - xs) ** 2)
    else:
        raise ValueError(f"unknown pep objective '{obj}'")

    if return_problem:
        return problem

    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
    solvetime = problem.wrapper.prob.solver_stats.solve_time
    log.info(f'pepit_tau at K={k}: {pepit_tau} (solvetime {solvetime})')
    return pepit_tau, solvetime


# =============================================================================
# Stage 3: DRO-PEP (modern custom-interpolation path, Clarabel)
# =============================================================================

def logreg_dro(cfg):
    log.info(cfg)
    A_full, b_full = load_dataset(cfg.dataset, intercept=cfg.intercept,
                                  standardize=cfg.get('standardize', False))
    L, R, ref_instances = compute_L_R(cfg, A_full, b_full)
    mu = float(cfg.delta)
    t = cfg.eta / L
    log.info(f'L: {L}, mu: {mu}, R: {R}, step size: {t}')

    if cfg.alg not in ('grad_desc', 'nesterov_fgm'):
        raise NotImplementedError(
            f"DRO custom PEP construction supports 'grad_desc'/'nesterov_fgm', got '{cfg.alg}'")

    if cfg.dro_obj == 'expectation':
        N = cfg.training.expectation_N
        num_clusters = cfg.num_clusters.expectation
        measure = 'expectation'
    elif cfg.dro_obj == 'cvar':
        N = cfg.training.cvar_N
        num_clusters = cfg.num_clusters.cvar
        measure = 'cvar'
    else:
        raise ValueError(f"invalid dro_obj '{cfg.dro_obj}'")

    if N > len(ref_instances):
        raise ValueError(
            f'training N={N} exceeds the {len(ref_instances)} calibration instances '
            f'(cfg.N); raise cfg.N so L/R calibration covers the in-sample set')
    instances = ref_instances[:N]

    eps_vals = build_eps_vals(cfg)
    alpha_vals = list(cfg.alpha_vals)
    # alpha only affects the cvar objective; expectation ignores it (single pass).
    alphas_to_run = alpha_vals if measure == 'cvar' else [alpha_vals[0]]

    res = []
    sample_df_list = []
    for k in range(cfg.K_min, cfg.K_max + 1):
        t_vec = jnp.full(k, t)
        if cfg.alg == 'grad_desc':
            stp = (t_vec,)
            pep_data = pep_data_to_numpy(construct_gd_pep_data(
                t_vec, mu, L, R, k, cfg.pep_obj, composition_type='final'))
            traj_fn = logreg_gd_trajectories
        else:  # nesterov_fgm
            beta = jax_get_nesterov_fgm_beta_sequence(mu, L, k)
            stp = (t_vec, beta)
            pep_data = pep_data_to_numpy(construct_fgm_pep_data(
                t_vec, beta, mu, L, R, k, cfg.pep_obj, composition_type='final'))
            traj_fn = logreg_fgm_trajectories

        A_obj_np, b_obj_np = pep_data[0], pep_data[1]

        samples = []
        for i, (A, b, x_opt, f_opt, L_i) in enumerate(instances):
            z0 = -jnp.asarray(x_opt)   # x0 = 0 in original coordinates
            G, F = traj_fn(stp, jnp.asarray(A), jnp.asarray(b), z0,
                           jnp.asarray(x_opt), f_opt, cfg.delta, k,
                           return_Gram_representation=True)
            G, F = np.asarray(G), np.asarray(F)
            samples.append((G, F))
            emp = float(np.trace(A_obj_np @ G) + b_obj_np @ F)  # PEP objective on the sample
            sample_df_list.append(pd.Series({'i': i, 'K': k, 'obj_val': emp}))
        pd.DataFrame(sample_df_list).to_csv('samples.csv', index=False)

        DR = DROReformulator(
            pep_data,
            samples,
            measure,
            'clarabel',
            precond=cfg.precond,
            precond_type=cfg.precond_type,
            mro_clusters=num_clusters,
        )
        log.info(f'----dro reformulator built at k={k}----')
        for eps_idx, eps in enumerate(eps_vals):
            for alpha_idx, alpha in enumerate(alphas_to_run):
                DR.set_params(eps=eps, alpha=alpha)
                out = DR.solve()
                if num_clusters is not None:
                    dro_feas = DR.extract_dro_feas_sol_from_mro(eps=eps, alpha=alpha)
                else:
                    dro_feas = out['obj']

                res.append(pd.Series({
                    'K': k,
                    'eps_idx': eps_idx,
                    'eps': eps,
                    'alpha_idx': alpha_idx,
                    'alpha': alpha,
                    'mro_sol': out['obj'],
                    'solvetime': out['solvetime'],
                    'dro_feas_sol': dro_feas,
                }))

                df = pd.DataFrame(res)
                df.to_csv(cfg.dro_fname, index=False)


# =============================================================================
# Legacy entry point (kept importable for run_lyap_experiment.py)
# =============================================================================

def logreg_lyap(cfg):
    raise NotImplementedError(
        'logreg_lyap was retired in the real-data certification rewrite; '
        'see git history (pre bs37/logreg_cert_realdata) for the old version')
