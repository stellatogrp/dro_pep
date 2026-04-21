"""
End-to-end comparison: our construct_chambolle_pock_pep_data SDP solve vs PEPit.

Two complementary assertions:
  1. Upper bound: our SDP at (tau, sigma, theta, L_M) should match PEPit's worst
     case within rtol=0.1.
  2. Achievability: a real LP CP trajectory's observed duality gap at iterate K
     should satisfy gap_observed <= pep_value + eps.

Before Phase 2 fixes: (1) fails because our SDP is unbounded. After: both match.
"""
import pytest
import numpy as np
import cvxpy as cp
import jax

jax.config.update('jax_enable_x64', True)

from learning.pep_constructions import construct_chambolle_pock_pep_data
from tests.test_chambolle_pock_pepit_reference import (
    pepit_linop_reference,
    CANONICAL_TAU, CANONICAL_SIGMA, CANONICAL_THETA, CANONICAL_L_M, CANONICAL_R,
)
from tests.test_chambolle_pock_interpolation import (
    generate_lp, solve_lp, run_cp_on_lp,
)


def solve_cp_pep_sdp(tau, sigma, theta, M, R, K_max, verbose=False,
                     composition_type='final', decay_rate=0.9):
    """Assemble and solve our PEP SDP with CLARABEL. Return (value, status)."""
    pep_data = construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=M, R=R, K_max=K_max,
        composition_type=composition_type, decay_rate=decay_rate,
    )
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = pep_data

    A_obj = np.asarray(A_obj); b_obj = np.asarray(b_obj)
    A_vals = np.asarray(A_vals); b_vals = np.asarray(b_vals); c_vals = np.asarray(c_vals)
    PSD_A_vals = [np.asarray(a) for a in PSD_A_vals]
    PSD_b_vals = [np.asarray(a) for a in PSD_b_vals]
    PSD_c_vals = [np.asarray(a) for a in PSD_c_vals]

    dimG = A_obj.shape[0]
    dimF = b_obj.shape[0]

    G = cp.Variable((dimG, dimG), PSD=True, name="G")
    F = cp.Variable(dimF, name="F")

    constraints = []
    for i in range(A_vals.shape[0]):
        constraints.append(cp.trace(A_vals[i] @ G) + b_vals[i] @ F + c_vals[i] <= 0)

    for idx in range(len(PSD_A_vals)):
        A_psd = PSD_A_vals[idx]; b_psd = PSD_b_vals[idx]; c_psd = PSD_c_vals[idx]
        size_H = PSD_shapes[idx]
        H = c_psd.copy()
        for i in range(dimG):
            for j in range(dimG):
                H = H + A_psd[:, :, i, j] * G[i, j]
        for k in range(dimF):
            H = H + b_psd[:, :, k] * F[k]
        constraints.append(H >> 0)

    objective = cp.Maximize(cp.trace(A_obj @ G) + b_obj @ F)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=verbose)
    except Exception as e:
        return None, f"exception: {e}"
    return prob.value, prob.status


@pytest.mark.parametrize('K_max', [1, 3, 5])
def test_pep_matches_pepit(K_max):
    """Our PEP SDP value should match PEPit's reference within rtol=0.1."""
    pepit_val = pepit_linop_reference(K_max)
    pep_val, status = solve_cp_pep_sdp(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        M=CANONICAL_L_M, R=CANONICAL_R, K_max=K_max,
    )
    print(f"\n=== PEP vs PEPit (K={K_max}) ===")
    print(f"  PEPit reference: {pepit_val!r}")
    print(f"  Our PEP value:   {pep_val!r}")
    print(f"  Status:          {status}")

    assert pep_val is not None, "Solver raised an exception"
    assert status in ['optimal', 'optimal_inaccurate'], \
        f"Unexpected solver status: {status}"
    assert np.isfinite(pep_val), f"Non-finite PEP value: {pep_val}"
    assert np.isclose(pep_val, pepit_val, rtol=0.1), \
        f"PEP {pep_val:.4e} does not match PEPit {pepit_val:.4e} (rtol=0.1)"


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1])
def test_lp_trajectory_gap_bounded_by_pep(K_max, seed):
    """A concrete LP CP trajectory's gap at iterate K must be <= PEP worst case.

    Requires the LP trajectory to satisfy the IC (Euclidean: ||dx||^2 + ||dy||^2 <= R^2).
    We scale the trajectory's initial point to sit inside the unit Euclidean ball.
    """
    m, n = 3, 5
    A, b, c, x_feas = generate_lp(m=m, n=n, seed=seed)
    xs, ys = solve_lp(A, b, c)

    # Initial point chosen to SATISFY the IC (||dx||^2 + ||dy||^2 <= 1) so comparison is fair.
    rng = np.random.default_rng(seed + 100)
    dx = rng.standard_normal(n) * 0.1
    dy = rng.standard_normal(m) * 0.1
    # Ensure x0 = xs + dx is non-negative.
    dx = np.where(xs + dx < 0, -xs, dx)  # Clip dx so x0 >= 0
    x0 = xs + dx
    y0 = ys + dy

    # Compute Euclidean norm and rescale dx, dy so ||dx||^2 + ||dy||^2 <= 1.
    euc_sq = dx @ dx + dy @ dy
    if euc_sq > 0:
        scale = 1.0 / np.sqrt(euc_sq * 1.01)   # within ball with 1% margin
        dx = dx * scale; dy = dy * scale
        dx = np.where(xs + dx < 0, -xs + 1e-6, dx)  # Still non-negative
        x0 = xs + dx; y0 = ys + dy

    xs_iters, ys_iters, gf1_iters, gh_iters, w_iters, z_iters = run_cp_on_lp(
        A, b, c, x0, y0,
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        K_max=K_max,
    )

    x_K = xs_iters[K_max]; y_K = ys_iters[K_max]
    gap = (c @ x_K + ys @ (b - A @ x_K)) - (c @ xs + y_K @ (b - A @ xs))

    pep_val, status = solve_cp_pep_sdp(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        M=CANONICAL_L_M, R=CANONICAL_R, K_max=K_max,
    )

    print(f"\n=== LP gap vs PEP (K={K_max}, seed={seed}) ===")
    print(f"  Observed LP gap: {gap!r}")
    print(f"  PEP worst case:  {pep_val!r} (status={status})")

    assert pep_val is not None and np.isfinite(pep_val), \
        f"PEP unbounded/failed: {pep_val} ({status})"
    assert gap <= pep_val + 1e-6, \
        f"LP trajectory gap {gap:.6e} exceeds PEP worst case {pep_val:.6e}"
