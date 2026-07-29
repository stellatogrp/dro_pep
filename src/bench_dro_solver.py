"""Benchmark solver backends on a single logreg DRO subproblem.

Builds the exact (alg, K, measure) reformulation that logreg_dro solves and
times one solve at a fixed (eps, alpha), reporting build time, solve time,
objective, and peak RSS. Modes:

  clarabel        production path (ClarabelCanonicalizer, direct conic build)
  cvxpy-clarabel  same reformulation through CVXPY, Clarabel core
  cvxpy-mosek     same reformulation through CVXPY, MOSEK core
                  (threads set from --threads / SLURM_CPUS_PER_TASK)

Run each mode as its own process so peak RSS is attributable:

  python bench_dro_solver.py --mode cvxpy-mosek --alg nesterov_fgm \
      --K 24 --measure cvar --eps 1e-4 --alpha 0.01

Extra MOSEK parameters pass through as repeatable
--mosek-param MSK_IPAR_XXX=value pairs.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


def peak_rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS
    scale = 1024 ** 2 if sys.platform.startswith('linux') else 1024 ** 3
    return rss / scale


def parse_mosek_param(kv):
    key, val = kv.split('=', 1)
    for cast in (int, float):
        try:
            return key, cast(val)
        except ValueError:
            continue
    return key, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['clarabel', 'cvxpy-clarabel', 'cvxpy-mosek'])
    p.add_argument('--alg', default='nesterov_fgm',
                   choices=['grad_desc', 'nesterov_fgm'])
    p.add_argument('--K', type=int, default=24)
    p.add_argument('--measure', default='cvar',
                   choices=['expectation', 'cvar'])
    p.add_argument('--eps', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--eta', type=float, default=None,
                   help='step-size multiplier override (default: config)')
    p.add_argument('--threads', type=int, default=None,
                   help='solver threads (default: SLURM_CPUS_PER_TASK or all cores)')
    p.add_argument('--mosek-param', action='append', default=[],
                   metavar='KEY=VALUE')
    p.add_argument('--out', default=None, help='JSON output path')
    args = p.parse_args()

    threads = args.threads
    if threads is None:
        threads = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))

    from hydra import initialize, compose
    overrides = [f'alg={args.alg}', f'dro_obj={args.measure}',
                 f'K_min={args.K}', f'K_max={args.K}']
    if args.eta is not None:
        overrides.append(f'eta={args.eta}')
    with initialize(version_base='1.2', config_path='configs'):
        cfg = compose(config_name='logreg', overrides=overrides)

    from experiment_classes.logreg_data import load_dataset, compute_L_R
    from learning.pep_constructions import (
        construct_gd_pep_data, construct_fgm_pep_data, pep_data_to_numpy)
    from learning.trajectories import (
        logreg_gd_trajectories, logreg_fgm_trajectories)
    from learning.acceleration_stepsizes import (
        jax_get_nesterov_fgm_beta_sequence)
    from reformulator.dro_reformulator import DROReformulator

    t0 = time.perf_counter()
    A_full, b_full = load_dataset(cfg.dataset, intercept=cfg.intercept)
    L, R, ref_instances = compute_L_R(cfg, A_full, b_full)
    mu = float(cfg.delta)
    t = cfg.eta / L
    N = cfg.training.expectation_N if args.measure == 'expectation' \
        else cfg.training.cvar_N
    instances = ref_instances[:N]

    k = args.K
    t_vec = jnp.full(k, t)
    if args.alg == 'grad_desc':
        stp = (t_vec,)
        pep_data = pep_data_to_numpy(construct_gd_pep_data(
            t_vec, mu, L, R, k, cfg.pep_obj, composition_type='final'))
        traj_fn = logreg_gd_trajectories
    else:
        beta = jax_get_nesterov_fgm_beta_sequence(mu, L, k)
        stp = (t_vec, beta)
        pep_data = pep_data_to_numpy(construct_fgm_pep_data(
            t_vec, beta, mu, L, R, k, cfg.pep_obj, composition_type='final'))
        traj_fn = logreg_fgm_trajectories

    samples = []
    for A, b, x_opt, f_opt, _L_i in instances:
        z0 = -jnp.asarray(x_opt)
        G, F = traj_fn(stp, jnp.asarray(A), jnp.asarray(b), z0,
                       jnp.asarray(x_opt), f_opt, cfg.delta, k,
                       return_Gram_representation=True)
        samples.append((np.asarray(G), np.asarray(F)))
    data_s = time.perf_counter() - t0

    wrapper = 'clarabel' if args.mode == 'clarabel' else 'cvxpy'
    t0 = time.perf_counter()
    DR = DROReformulator(
        pep_data, samples, args.measure, wrapper,
        precond=cfg.precond, precond_type=cfg.precond_type,
        mro_clusters=None)
    build_s = time.perf_counter() - t0

    if args.mode == 'cvxpy-clarabel':
        import cvxpy as cp
        DR.canon.set_solver(cp.CLARABEL, verbose=True)
    elif args.mode == 'cvxpy-mosek':
        import cvxpy as cp
        mosek_params = {'MSK_IPAR_NUM_THREADS': threads}
        mosek_params.update(dict(parse_mosek_param(kv)
                                 for kv in args.mosek_param))
        DR.canon.set_solver(cp.MOSEK, mosek_params=mosek_params,
                            verbose=True, ignore_dpp=True)

    DR.set_params(eps=args.eps, alpha=args.alpha)
    t0 = time.perf_counter()
    out = DR.solve()
    solve_wall_s = time.perf_counter() - t0

    result = {
        'mode': args.mode, 'alg': args.alg, 'K': k,
        'measure': args.measure, 'eps': args.eps, 'alpha': args.alpha,
        'eta': float(cfg.eta), 'N': int(N), 'threads': threads,
        'obj': float(out['obj']),
        'solver_reported_s': float(out['solvetime']),
        'solve_wall_s': solve_wall_s,
        'build_s': build_s,
        'data_s': data_s,
        'peak_rss_gb': peak_rss_gb(),
    }
    print('BENCH_RESULT ' + json.dumps(result))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
