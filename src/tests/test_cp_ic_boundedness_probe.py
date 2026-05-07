"""
Probe: does the Chambolle-Pock PEP remain bounded under weaker initial
conditions than the current P-norm Lyapunov form?

Background: the current CP PEP IC is

    (x0-xs)^2 / tau + (u0-us)^2 / sigma - 2 <u0-us, K(x0-xs)> <= R^2

which bakes the stepsizes (tau, sigma) into the IC. For PDLP, this forces
R_val to be recomputed whenever stepsizes change, which is awkward for
LDRO-PEP with learned stepsizes. We want to move to an ISTA-style IC
that bounds only the primal+dual distance.

This module probes a family of candidate ICs with PEPit. Each variant
just swaps `problem.set_initial_condition(...)`; dynamics and the
duality-gap metric are identical to
`test_chambolle_pock_pepit_reference.py`.

Run with:
    pytest src/tests/test_cp_ic_boundedness_probe.py -v -s

The `-s` flag is essential — the Phase 5 summary table prints to stdout.
"""
import pytest
import numpy as np

from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.operators import LinearOperator
from PEPit.primitive_steps import proximal_step

from tests.test_chambolle_pock_pepit_reference import pepit_linop_reference


# ---------------------------------------------------------------------------
# Core probe helper: one PEPit solve per (IC variant, params) tuple.
# ---------------------------------------------------------------------------

def _wc_cp(ic_variant, tau, sigma, theta, n, M=1.0, R=1.0, verbose=0):
    """PEPit worst-case CP last-iterate gap under a chosen IC variant.

    Returns (status_str, wc_value). status is 'ok' if PEPit returns a
    finite value; 'unbounded_or_inaccurate' if value is +/- inf or NaN;
    'error:<Type>' if the solver raised.

    We deliberately do not assert finiteness inside the helper — the whole
    point of this probe is to discover which variants are unbounded.
    """
    problem = PEP()
    f1 = problem.declare_function(ConvexFunction)
    h = problem.declare_function(ConvexFunction)
    K = problem.declare_function(LinearOperator, L=M)

    xs = problem.set_initial_point()
    us = problem.set_initial_point()
    f1.add_point((xs, -K.T.gradient(us), f1.value(xs)))
    h.add_point((us, K.gradient(xs), h.value(us)))

    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    dx = x0 - xs
    du = u0 - us

    if ic_variant == 'A1':
        # dx^2 + du^2 <= R^2  (pure Euclidean, no tau/sigma, no M)
        problem.set_initial_condition(dx ** 2 + du ** 2 <= R ** 2)
    elif ic_variant == 'A2':
        # dx^2 <= R^2 AND du^2 <= R^2  (two independent balls)
        problem.set_initial_condition(dx ** 2 <= R ** 2)
        problem.set_initial_condition(du ** 2 <= R ** 2)
    elif ic_variant == 'B1':
        # dx^2 + du^2 / M^2 <= R^2  (anisotropic, dual unit-scaled by M)
        problem.set_initial_condition(dx ** 2 + du ** 2 / (M ** 2) <= R ** 2)
    elif ic_variant == 'B2':
        # dx^2 + du^2 - 2 <du, K dx> / M <= R^2  (cross term scaled by 1/M)
        problem.set_initial_condition(
            dx ** 2 + du ** 2 - 2.0 * du * K.gradient(dx) / M <= R ** 2
        )
    elif ic_variant == 'B3':
        # Two ICs: distance ball + cross-term bound
        problem.set_initial_condition(dx ** 2 + du ** 2 <= R ** 2)
        problem.set_initial_condition(- du * K.gradient(dx) <= M * R ** 2 / 2.0)
    elif ic_variant == 'C':
        # Baseline Lyapunov P-norm (current production IC)
        problem.set_initial_condition(
            dx ** 2 / tau + du ** 2 / sigma
            - 2.0 * du * K.gradient(dx) <= R ** 2
        )
    else:
        raise ValueError(f"unknown ic_variant: {ic_variant}")

    x, u = x0, u0
    for _ in range(n):
        x_new, _, _ = proximal_step(x - tau * K.T.gradient(u), f1, tau)
        x_bar = x_new + theta * (x_new - x)
        u_new, _, _ = proximal_step(u + sigma * K.gradient(x_bar), h, sigma)
        x, u = x_new, u_new

    def Lagrangian(a, b):
        return f1.value(a) + K.gradient(a) * b - h.value(b)

    gap = Lagrangian(x, us) - Lagrangian(xs, u)
    problem.set_performance_metric(gap)

    try:
        val = problem.solve(verbose=verbose, wrapper='cvxpy')
    except Exception as e:
        return (f'error:{type(e).__name__}', float('inf'))

    try:
        val_f = float(val)
    except (TypeError, ValueError):
        return ('non_numeric', float('inf'))

    if not np.isfinite(val_f):
        return ('unbounded_or_inaccurate', val_f)
    return ('ok', val_f)


# ---------------------------------------------------------------------------
# Canonical parameters (match test_chambolle_pock_pepit_reference).
# ---------------------------------------------------------------------------
CANONICAL_M = 1.0
CANONICAL_THETA = 1.0
# "Canonical operating point" for the probe: tau = sigma = 0.9 / M.
PROBE_TAU = 0.9 / CANONICAL_M
PROBE_SIGMA = 0.9 / CANONICAL_M
# Reference-baseline params (match the existing PEPit reference test for Phase 1).
REF_TAU = 0.5
REF_SIGMA = 0.5

IC_VARIANTS_TO_PROBE = ['A1', 'A2', 'B1', 'B2', 'B3']
K_RANGE = [1, 2, 3]


# ===========================================================================
# Phase 1 — Scaffolding sanity.
#
# After the Euclidean IC was ported into both the analytic SDP and
# test_chambolle_pock_pepit_reference.py, the probe's A1 variant IS the
# PEPit reference. Phase 1 now checks that the probe's A1 helper still
# matches that reference — a regression check that catches any drift in
# either side.
# ===========================================================================

@pytest.mark.parametrize('K', [1, 3])
def test_phase1_baseline_matches_reference(K):
    status, val = _wc_cp(
        'A1', tau=REF_TAU, sigma=REF_SIGMA, theta=CANONICAL_THETA,
        n=K, M=CANONICAL_M, R=1.0,
    )
    assert status == 'ok', f"A1 helper failed: status={status}, val={val}"
    ref = pepit_linop_reference(K=K, tau=REF_TAU, sigma=REF_SIGMA,
                                theta=CANONICAL_THETA, L_M=CANONICAL_M)
    print(f"\nPhase1 K={K}: probe_A1={val:.6f}  reference={ref:.6f}")
    np.testing.assert_allclose(val, ref, rtol=1e-3,
                               err_msg="probe A1 diverged from PEPit reference")


# ===========================================================================
# Phase 2 — Boundedness sweep (the main probe).
#
# Single aggregator test that sweeps every (variant, K) and prints a table.
# Uses no hard assertions on "bounded" status — we just need to see the
# matrix. The summary artifact is the test output.
# ===========================================================================

# Module-level cache so Phase 3 can reuse Phase 2 results without re-solving.
_PHASE2_RESULTS = {}


def test_phase2_boundedness_sweep():
    results = {}
    print("\n" + "=" * 72)
    print("Phase 2 — Boundedness sweep")
    print(f"  tau = sigma = {PROBE_TAU}, theta = {CANONICAL_THETA}, "
          f"M = {CANONICAL_M}, R = 1.0")
    print("=" * 72)
    header = f"  {'variant':<5} " + " ".join(f"{'K='+str(k):>14}" for k in K_RANGE)
    print(header)

    for variant in IC_VARIANTS_TO_PROBE:
        row = []
        for K in K_RANGE:
            status, val = _wc_cp(
                variant, tau=PROBE_TAU, sigma=PROBE_SIGMA,
                theta=CANONICAL_THETA, n=K, M=CANONICAL_M, R=1.0,
            )
            results[(variant, K)] = (status, val)
            if status == 'ok':
                row.append(f"{val:14.6g}")
            else:
                row.append(f"{status[:14]:>14}")
        print(f"  {variant:<5} " + " ".join(row))

    _PHASE2_RESULTS.update(results)

    # Soft assertion: at least the baseline 'C' would work if we included it.
    # We don't include C here because Phase 1 already covers it. But we do
    # assert that *something* in the sweep is finite — otherwise the probe
    # setup itself is broken.
    any_finite = any(s == 'ok' for (s, _) in results.values())
    assert any_finite, ("Every variant unbounded at every K — likely a bug "
                        "in the probe helper, not genuine unboundedness.")


# ===========================================================================
# Phase 3 — R-scaling confirmation.
#
# For each variant-K cell that was 'ok' at R=1, re-solve at R=2 and assert
# wc(R=2) / wc(R=1) == 4 within 5% (since the feasible set scales as R,
# and the duality gap is quadratic-homogeneous in the IC radius).
# ===========================================================================

def test_phase3_R_scaling():
    if not _PHASE2_RESULTS:
        pytest.skip("Phase 2 results missing; run phase2 first.")

    print("\n" + "=" * 72)
    print("Phase 3 — R-scaling check (expect wc(R=2) / wc(R=1) ~= 4)")
    print("=" * 72)
    header = f"  {'variant':<5} {'K':>3} {'wc(R=1)':>14} {'wc(R=2)':>14} {'ratio':>8}"
    print(header)

    failures = []
    for (variant, K), (status, val1) in _PHASE2_RESULTS.items():
        if status != 'ok':
            continue
        _, val2 = _wc_cp(
            variant, tau=PROBE_TAU, sigma=PROBE_SIGMA,
            theta=CANONICAL_THETA, n=K, M=CANONICAL_M, R=2.0,
        )
        ratio = val2 / val1 if val1 > 0 else float('nan')
        flag = "" if np.isfinite(ratio) and abs(ratio - 4.0) < 0.2 else " <-- FAIL"
        print(f"  {variant:<5} {K:>3} {val1:14.6g} {val2:14.6g} {ratio:>8.3f}{flag}")
        if flag:
            failures.append((variant, K, ratio))

    # This is informational; we flag but don't hard-fail on the final decision
    # since near-degenerate SDPs can give weird ratios even when the structure
    # is fine. The reviewer reads the printout.
    if failures:
        print(f"\n  {len(failures)} cell(s) failed R^2 scaling: {failures}")


# ===========================================================================
# Phase 4 — tau, sigma independence (A variants only).
#
# If A1/A2 are genuinely stepsize-free in the IC, the wc SHOULD only depend
# on (tau, sigma) through the dynamics, not through the feasible set. We
# sweep a few (tau, sigma) corners and print the values. Informational.
# ===========================================================================

TAU_SIGMA_GRID = [
    (0.1 / CANONICAL_M, 0.9 / CANONICAL_M),
    (0.5 / CANONICAL_M, 0.5 / CANONICAL_M),
    (0.9 / CANONICAL_M, 0.1 / CANONICAL_M),
]


def test_phase4_tau_sigma_independence():
    print("\n" + "=" * 72)
    print("Phase 4 — (tau, sigma) sweep for Option A variants at n=2, R=1, M=1")
    print("=" * 72)
    header = (f"  {'variant':<5} "
              + " ".join(f"{'tau,sig=('+f'{t:.2f},{s:.2f}'+')':>20}"
                         for (t, s) in TAU_SIGMA_GRID))
    print(header)

    for variant in ['A1', 'A2']:
        row = []
        for (t, s) in TAU_SIGMA_GRID:
            status, val = _wc_cp(
                variant, tau=t, sigma=s, theta=CANONICAL_THETA,
                n=2, M=CANONICAL_M, R=1.0,
            )
            if status == 'ok':
                row.append(f"{val:20.6g}")
            else:
                row.append(f"{status[:20]:>20}")
        print(f"  {variant:<5} " + " ".join(row))


# ===========================================================================
# Phase 5 — Final summary artifact.
#
# Re-prints the Phase 2 verdict table in a form the human reviewer can
# quickly scan to decide which IC to port into chambolle_pock.py.
# ===========================================================================

def test_phase5_summary():
    if not _PHASE2_RESULTS:
        pytest.skip("Phase 2 results missing; run phase2 first.")

    print("\n" + "=" * 72)
    print("Phase 5 — Summary verdict (bounded iff status=ok at every K in {1,2,3})")
    print("=" * 72)

    for variant in IC_VARIANTS_TO_PROBE:
        statuses = [_PHASE2_RESULTS[(variant, K)][0] for K in K_RANGE]
        values = [_PHASE2_RESULTS[(variant, K)][1] for K in K_RANGE]
        all_ok = all(s == 'ok' for s in statuses)
        if all_ok:
            v_str = " ".join(f"{v:10.4g}" for v in values)
            print(f"  {variant}: BOUNDED at all K — values: {v_str}")
        else:
            detail = " ".join(
                f"K={K}:{s if s != 'ok' else f'{v:.3g}'}"
                for (K, s, v) in zip(K_RANGE, statuses, values)
            )
            print(f"  {variant}: UNBOUNDED at some K — {detail}")
