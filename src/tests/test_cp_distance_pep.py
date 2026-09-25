"""Chambolle-Pock PEP with the squared-distance metric ||x_K - x*||^2 + ||y_K - y*||^2.

Checks that the new ``metric='dist'`` objective of construct_chambolle_pock_pep_data
  1. matches an independent PEPit construction (linear operator ||K|| <= M);
  2. is bounded, and respects the explicit M-dependent bound for stable steps:
     for constant tau, sigma with tau*sigma*M^2 < 1 and theta = 1, PDHG is
     nonexpansive in the P-norm, P = [[I/tau, -K^T], [-K, I/sigma]], whose
     eigenvalues lie in [a - r, a + r] with a = (1/tau + 1/sigma)/2 and
     r = sqrt(((1/tau - 1/sigma)/2)^2 + M^2). Hence
         dist_K <= (a + r) / (a - r) * R^2;
  3. is bounded for ANY stepsizes by an explicit M-dependent recursion: with
     a = ||x - x*||, b = ||y - y*||, prox nonexpansiveness and ||K|| <= M give
         a' <= a + tau M b,   ||xbar - x*|| <= (1 + theta) a' + theta a,
         b' <= b + sigma M ||xbar - x*||,
     a nonnegative linear map T_k, so dist_K <= ||T_K ... T_1||_2^2 R^2. The
     worst case grows geometrically once tau sigma M^2 > 1; for strongly
     unstable steps it exceeds what Clarabel can resolve (a numerical, not a
     mathematical, limit -- PEPit's SDP fails at the same points);
  4. scales as R^2;
  5. upper-bounds the distance of real LP CP trajectories started in the R-ball.

Usage (from src/):
    pytest tests/test_cp_distance_pep.py -v -s
"""
import glob
from pathlib import Path

import cvxpy as cp
import jax
import numpy as np
import pandas as pd
import pytest

jax.config.update('jax_enable_x64', True)

from learning.pep_constructions import construct_chambolle_pock_pep_data  # noqa: E402
from tests.test_chambolle_pock_interpolation import generate_lp, run_cp_on_lp, solve_lp  # noqa: E402


def solve_dist_pep(tau, sigma, theta, M, R, K, composition_type='final', metric='dist'):
    """Assemble and solve the distance PEP SDP with Clarabel; returns (value, status)."""
    (A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A, PSD_b, PSD_c, shapes) = [
        x if isinstance(x, list) else np.asarray(x) if not isinstance(x, (int,)) else x
        for x in construct_chambolle_pock_pep_data(
            tau=tau, sigma=sigma, theta=theta, M=M, R=R, K_max=K,
            composition_type=composition_type, metric=metric)]
    dimG, dimF = A_obj.shape[0], b_obj.shape[0]
    G = cp.Variable((dimG, dimG), PSD=True)
    F = cp.Variable(dimF)
    cons = [cp.trace(np.asarray(A_vals[i]) @ G) + np.asarray(b_vals[i]) @ F + float(c_vals[i]) <= 0
            for i in range(A_vals.shape[0])]
    for A_psd, b_psd, c_psd in zip(PSD_A, PSD_b, PSD_c):
        A_psd, b_psd = np.asarray(A_psd), np.asarray(b_psd)
        n = A_psd.shape[0]
        H = np.asarray(c_psd) + sum(A_psd[:, :, i, j] * G[i, j] for i in range(dimG) for j in range(dimG)
                                    if np.any(A_psd[:, :, i, j]))
        if dimF:
            H = H + sum(b_psd[:, :, k] * F[k] for k in range(dimF) if np.any(b_psd[:, :, k]))
        cons.append((H + H.T) / 2 >> 0) if n > 1 else cons.append(H >= 0)
    prob = cp.Problem(cp.Maximize(cp.trace(np.asarray(A_obj) @ G) + np.asarray(b_obj) @ F), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value, prob.status


def pepit_dist(tau, sigma, theta, K, M=1.0, primal_only=False):
    """Independent PEPit worst case of ||x_K - x_s||^2 + ||u_K - u_s||^2 (Euclidean IC, R = 1)."""
    from PEPit import PEP
    from PEPit.functions import ConvexFunction
    from PEPit.operators import LinearOperator
    from PEPit.primitive_steps import proximal_step
    problem = PEP()
    f1 = problem.declare_function(ConvexFunction)
    h = problem.declare_function(ConvexFunction)
    Kop = problem.declare_function(LinearOperator, L=M)
    xs, us = problem.set_initial_point(), problem.set_initial_point()
    f1.add_point((xs, -Kop.T.gradient(us), f1.value(xs)))
    h.add_point((us, Kop.gradient(xs), h.value(us)))
    x0, u0 = problem.set_initial_point(), problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 + (u0 - us) ** 2 <= 1)
    x, u = x0, u0
    for k in range(K):
        t, s, th = tau[k], sigma[k], theta[k]
        x_new, _, _ = proximal_step(x - t * Kop.T.gradient(u), f1, t)
        x_bar = x_new + th * (x_new - x)
        u_new, _, _ = proximal_step(u + s * Kop.gradient(x_bar), h, s)
        x, u = x_new, u_new
    problem.set_performance_metric((x - xs) ** 2 if primal_only else (x - xs) ** 2 + (u - us) ** 2)
    return problem.solve(verbose=0, wrapper='cvxpy')


def stable_bound(tau, sigma, M):
    a = 0.5 * (1 / tau + 1 / sigma)
    r = np.sqrt((0.5 * (1 / tau - 1 / sigma)) ** 2 + M ** 2)
    assert a - r > 0, "bound needs tau * sigma * M^2 < 1"
    return (a + r) / (a - r)


vec = lambda v, K: np.full(K, float(v))


# ------------------------------------------------------------------ 1. PEPit
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("steps", [(0.5, 0.5, 1.0), (0.9, 0.3, 0.6), (1.5, 1.2, 1.0)])
def test_matches_pepit(K, steps):
    t, s, th = steps
    ours, status = solve_dist_pep(vec(t, K), vec(s, K), vec(th, K), 1.0, 1.0, K)
    ref = pepit_dist(vec(t, K), vec(s, K), vec(th, K), K, M=1.0)
    print(f"K={K} steps={steps}: ours {ours:.6f} ({status}) PEPit {ref:.6f}")
    assert status in ("optimal", "optimal_inaccurate")
    assert ours == pytest.approx(ref, rel=2e-3, abs=1e-5)


# ------------------------------------------------------------------ 2. stable-step bound (uses M)
@pytest.mark.parametrize("M", [1.0, 4.94])
@pytest.mark.parametrize("K", [1, 3, 5])
@pytest.mark.parametrize("frac", [0.25, 0.9])       # tau * sigma * M^2 = frac
def test_stable_steps_respect_M_bound(M, K, frac):
    tau = sigma = np.sqrt(frac) / M
    val, status = solve_dist_pep(vec(tau, K), vec(sigma, K), vec(1.0, K), M, 1.0, K)
    bnd = stable_bound(tau, sigma, M)
    print(f"M={M} K={K} tau*sigma*M^2={frac}: PEP {val:.5f} <= bound {bnd:.5f}")
    assert status in ("optimal", "optimal_inaccurate") and np.isfinite(val)
    assert val <= bnd * (1 + 1e-4) + 1e-6


# ------------------------------------------------------------------ 3. explicit M-dependent bound, any steps
def recursion_bound(tau, sigma, theta, M):
    """||T_K ... T_1||_2^2 for the nonnegative distance recursion (see module docstring)."""
    T = np.eye(2)
    for t, s_, th in zip(tau, sigma, theta):
        Tx = np.array([[1.0, t * M], [0.0, 1.0]])               # (a, b) -> (a', b)
        xbar = np.array([(1 + th) + th, (1 + th) * t * M])      # coefficient row of ||xbar - x*|| in (a, b)
        Tk = np.vstack([Tx[0], [sigma_M := s_ * M * xbar[0], 1.0 + s_ * M * xbar[1]]])
        T = Tk @ T
    return float(np.linalg.norm(T, 2) ** 2)


@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("M", [1.0, 2.0])
def test_bounded_by_M_recursion_for_any_steps(K, M):
    rng = np.random.default_rng(10 * K + int(M))
    for trial in range(3):
        tau, sigma, theta = rng.uniform(0.2, 1.5, K) / M, rng.uniform(0.2, 1.5, K) / M, rng.uniform(0.0, 2.0, K)
        v, status = solve_dist_pep(tau, sigma, theta, M, 1.0, K)
        bnd = recursion_bound(tau, sigma, theta, M)
        print(f"M={M} K={K} trial {trial}: max tau*sigma*M^2 {np.max(tau * sigma) * M ** 2:.2f} | "
              f"PEP {v:.4f} ({status}) <= M-recursion bound {bnd:.4f}")
        assert status in ("optimal", "optimal_inaccurate") and np.isfinite(v)
        assert v <= bnd * (1 + 1e-4) + 1e-6


def test_worst_case_grows_with_M():
    K = 3
    tau, sigma, theta = vec(0.6, K), vec(0.6, K), vec(1.0, K)
    vals = [solve_dist_pep(tau, sigma, theta, M, 1.0, K)[0] for M in (0.5, 1.0, 1.5, 2.0)]
    print("M = 0.5, 1, 1.5, 2 ->", np.round(vals, 4).tolist())
    assert all(a <= b * (1 + 1e-4) + 1e-6 for a, b in zip(vals, vals[1:]))


# ------------------------------------------------------------------ 4. R^2 scaling
def test_scales_with_R_squared():
    K = 3
    t, s, th = vec(0.7, K), vec(0.4, K), vec(0.8, K)
    v1, _ = solve_dist_pep(t, s, th, 2.0, 1.0, K)
    v3, _ = solve_dist_pep(t, s, th, 2.0, 3.0, K)
    assert v3 == pytest.approx(9.0 * v1, rel=1e-3)


# ------------------------------------------------------------------ 5. LP trajectories are bounded by the PEP
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("K", [1, 3])
def test_lp_trajectory_distance_below_pep(seed, K):
    A, b, c, _ = generate_lp(m=3, n=5, seed=seed)
    xs, ys = solve_lp(A, b, c)
    M = float(np.linalg.norm(A, 2))
    rng = np.random.default_rng(seed + 7)
    dx, dy = rng.standard_normal(5), rng.standard_normal(3)
    dx = np.where(xs + dx < 0, -xs, dx)
    sc = 1.0 / np.sqrt((dx @ dx + dy @ dy) * 1.01)
    x0 = np.maximum(xs + sc * dx, 0.0); y0 = ys + sc * dy
    R = np.sqrt(np.sum((x0 - xs) ** 2) + np.sum((y0 - ys) ** 2))
    tau, sigma, theta = 0.9 / M, 0.9 / M, 1.0
    xi, yi, *_ = run_cp_on_lp(A, b, c, x0, y0, tau=tau, sigma=sigma, theta=theta, K_max=K)
    dist = np.sum((xi[K] - xs) ** 2) + np.sum((yi[K] - ys) ** 2)
    val, _ = solve_dist_pep(vec(tau, K), vec(sigma, K), vec(theta, K), M, R, K)
    print(f"seed={seed} K={K}: LP distance {dist:.5f} <= PEP {val:.5f}")
    assert dist <= val * (1 + 1e-4) + 1e-6


# ------------------------------------------------------------------ 6. the learned PDLP schedules at the real M
def _learned(run):
    ex = Path(__file__).resolve().parent.parent / "iclr_data_outputs/explore_warmstart/runs"
    p = glob.glob(str(ex / run / "c/*/PDLP/*/*/learn_dro_outputs/K_5/progress.csv"))
    if not p:
        pytest.skip(f"learned schedule {run} not available locally")
    d = pd.read_csv(p[0]); b = int(d.validation_loss.idxmin())
    return [d.loc[b, [f"{q}_{k}" for k in range(5)]].to_numpy(float) for q in ("tau", "sigma", "theta")]


@pytest.mark.parametrize("run,K", [("cons_dr_eps300", 1), ("cons_dr_eps300", 3), ("cons_dr_eps300", 5),
                                   ("cons_l2o", 1), ("cons_l2o", 2)])
def test_learned_schedules_match_pepit_and_bound(run, K):
    t, s_, th = (a[:K] for a in _learned(run))
    M = 4.9377
    v, status = solve_dist_pep(t, s_, th, M, 1.0, K)
    ref = pepit_dist(t, s_, th, K, M=M)
    bnd = recursion_bound(t, s_, th, M)
    print(f"{run} K={K}: PEP {v:.4f} ({status}) PEPit {ref:.4f} bound {bnd:.4g}")
    assert v == pytest.approx(ref, rel=1e-3)
    assert v <= bnd * (1 + 1e-4)


@pytest.mark.xfail(reason="bounded (see recursion bound) but the worst case is too large for Clarabel/PEPit "
                          "numerics once the L2O schedule's unstable steps compound (K>=3 at M=4.94)", strict=False)
def test_l2o_schedule_K3_numerics():
    t, s_, th = (a[:3] for a in _learned("cons_l2o"))
    v, status = solve_dist_pep(t, s_, th, 4.9377, 1.0, 3)
    assert status in ("optimal", "optimal_inaccurate")


# ------------------------------------------------------------------ 7. primal-only distance ||x_K - x*||^2
@pytest.mark.parametrize("K", [1, 2, 3])
@pytest.mark.parametrize("steps", [(0.5, 0.5, 1.0), (0.9, 0.3, 0.6), (1.5, 1.2, 1.0)])
def test_primal_distance_matches_pepit_and_is_below_full(K, steps):
    t, s_, th = vec(steps[0], K), vec(steps[1], K), vec(steps[2], K)
    vp, status = solve_dist_pep(t, s_, th, 1.0, 1.0, K, metric='pdist')
    ref = pepit_dist(t, s_, th, K, M=1.0, primal_only=True)
    vf, _ = solve_dist_pep(t, s_, th, 1.0, 1.0, K, metric='dist')
    print(f"K={K} steps={steps}: primal PEP {vp:.6f} PEPit {ref:.6f} <= full {vf:.6f} <= recursion {recursion_bound(t, s_, th, 1.0):.4g}")
    assert status in ("optimal", "optimal_inaccurate")
    assert vp == pytest.approx(ref, rel=2e-3, abs=1e-5)
    assert vp <= vf * (1 + 1e-4) + 1e-6


@pytest.mark.parametrize("run,K", [("cons_dr_eps300", 5), ("cons_l2o", 2)])
def test_primal_distance_learned_schedules(run, K):
    t, s_, th = (a[:K] for a in _learned(run))
    v, status = solve_dist_pep(t, s_, th, 4.9377, 1.0, K, metric='pdist')
    ref = pepit_dist(t, s_, th, K, M=4.9377, primal_only=True)
    print(f"{run} K={K}: primal PEP {v:.4f} ({status}) PEPit {ref:.4f}")
    assert v == pytest.approx(ref, rel=1e-3)
