"""
Weighted-composition SDP solve for Chambolle-Pock vs PEPit.

Parametric comparison across K ∈ {1, 3, 5} and decay_rate ∈ {0.7, 0.9}. After
the Phase 1 fix (last-iterate) plus Phase 2 refactor (compose_objective
dispatch), our construction should match the PEPit weighted-gap reference to
within rtol=0.1.

Also locks down the sanity identity that at K=1 the weighted sum is a single
normalized term (w_1 = 1), so it must equal the final-iterate SDP value.
"""
import pytest
import numpy as np
import jax

jax.config.update('jax_enable_x64', True)

from tests.test_chambolle_pock_pepit_reference import (
    pepit_linop_weighted_reference,
    CANONICAL_TAU, CANONICAL_SIGMA, CANONICAL_THETA, CANONICAL_L_M, CANONICAL_R,
)
from tests.test_chambolle_pock_pep_vs_pepit import solve_cp_pep_sdp


@pytest.mark.parametrize('K', [1, 3, 5])
@pytest.mark.parametrize('decay_rate', [0.7, 0.9])
def test_weighted_pep_matches_pepit(K, decay_rate):
    """Our weighted-composition SDP value matches PEPit within rtol=0.1."""
    pepit_val = pepit_linop_weighted_reference(
        K=K,
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        L_M=CANONICAL_L_M, decay_rate=decay_rate,
    )
    pep_val, status = solve_cp_pep_sdp(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        M=CANONICAL_L_M, R=CANONICAL_R, K_max=K,
        composition_type='weighted', decay_rate=decay_rate,
    )
    print(f"\n=== CP weighted (K={K}, decay={decay_rate}) ===")
    print(f"  PEPit reference: {pepit_val!r}")
    print(f"  Our PEP value:   {pep_val!r}")
    print(f"  Status:          {status}")

    assert pep_val is not None, "Solver raised an exception"
    assert status in ['optimal', 'optimal_inaccurate'], \
        f"Unexpected solver status: {status}"
    assert np.isfinite(pep_val), f"Non-finite PEP value: {pep_val}"
    assert np.isclose(pep_val, pepit_val, rtol=0.1), \
        f"PEP {pep_val:.4e} does not match PEPit {pepit_val:.4e}"


def test_weighted_K1_equals_final_K1():
    """At K=1 weighted reduces to a single-term normalized sum; should equal final."""
    final_val, status_final = solve_cp_pep_sdp(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        M=CANONICAL_L_M, R=CANONICAL_R, K_max=1,
        composition_type='final',
    )
    weighted_val, status_weighted = solve_cp_pep_sdp(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        M=CANONICAL_L_M, R=CANONICAL_R, K_max=1,
        composition_type='weighted', decay_rate=0.9,
    )
    print(f"\nfinal K=1:    {final_val!r} ({status_final})")
    print(f"weighted K=1: {weighted_val!r} ({status_weighted})")
    assert final_val is not None and weighted_val is not None
    assert np.isclose(final_val, weighted_val, rtol=1e-5), \
        f"weighted K=1 {weighted_val:.6e} != final K=1 {final_val:.6e}"
