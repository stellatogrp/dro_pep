"""
Unit tests for Silver stepsize schedules.

Run with: pytest tests/test_silver_stepsizes.py -v
"""
import pytest
import numpy as np
from learning_experiment_classes.silver_stepsizes import (
    get_nonstrongly_convex_silver_stepsizes,
    get_strongly_convex_silver_stepsizes,
    compute_shifted_2adics,
    generate_yz_sequences,
)


# ============================================================================
# Tests: compute_shifted_2adics
# ============================================================================

class TestComputeShifted2Adics:
    """Tests for compute_shifted_2adics function."""
    
    def test_first_8_values(self):
        """Test 2-adic valuations for k=1,...,8."""
        # v_2(1)=0, v_2(2)=1, v_2(3)=0, v_2(4)=2, v_2(5)=0, v_2(6)=1, v_2(7)=0, v_2(8)=3
        expected = np.array([0, 1, 0, 2, 0, 1, 0, 3])
        result = compute_shifted_2adics(8)
        np.testing.assert_array_equal(result, expected)
    
    def test_length(self):
        """Test that output length matches K."""
        for K in [1, 4, 8, 16]:
            result = compute_shifted_2adics(K)
            assert len(result) == K


# ============================================================================
# Tests: Non-Strongly Convex Silver Stepsizes
# ============================================================================

class TestNonStronglyConvexSilverStepsizes:
    """Tests for non-strongly convex Silver step sizes (mu=0)."""
    
    def test_length(self):
        """Test that output length matches K."""
        for K in [1, 3, 7, 15, 31]:
            result = get_nonstrongly_convex_silver_stepsizes(K, L=1)
            assert len(result) == K
    
    def test_first_step_size(self):
        """First step size should be (1 + rho^(-1)) / L."""
        rho = 1 + np.sqrt(2)
        expected_first = (1 + rho ** (-1))  # For k=1: (1 & -1).bit_length()-2 = -1
        result = get_nonstrongly_convex_silver_stepsizes(1, L=1)
        assert np.isclose(result[0], expected_first, rtol=1e-10)
    
    def test_L_scaling(self):
        """Step sizes should scale inversely with L."""
        K = 7
        result_L1 = get_nonstrongly_convex_silver_stepsizes(K, L=1)
        result_L2 = get_nonstrongly_convex_silver_stepsizes(K, L=2)
        for s1, s2 in zip(result_L1, result_L2):
            assert np.isclose(s1, 2 * s2, rtol=1e-10)
    
    def test_positive_stepsizes(self):
        """All step sizes should be positive."""
        for K in [1, 3, 7, 15]:
            result = get_nonstrongly_convex_silver_stepsizes(K, L=1)
            assert all(s > 0 for s in result)


# ============================================================================
# Tests: generate_yz_sequences
# ============================================================================

class TestGenerateYZSequences:
    """Tests for generate_yz_sequences function."""
    
    def test_initial_values(self):
        """y_1 and z_1 should both equal 1/kappa."""
        kappa = 10
        y_vals, z_vals = generate_yz_sequences(kappa, t=3)
        assert np.isclose(y_vals[1], 1 / kappa, rtol=1e-10)
        assert np.isclose(z_vals[1], 1 / kappa, rtol=1e-10)
    
    def test_keys_are_powers_of_2(self):
        """Check that y and z sequences have keys that are powers of 2."""
        kappa = 10
        t = 4
        y_vals, z_vals = generate_yz_sequences(kappa, t)
        expected_keys = {2 ** i for i in range(t + 1)}
        assert set(y_vals.keys()) == expected_keys
        assert set(z_vals.keys()) == expected_keys


# ============================================================================
# Tests: Strongly Convex Silver Stepsizes
# ============================================================================

class TestStronglyConvexSilverStepsizes:
    """Tests for strongly convex Silver step sizes (mu > 0)."""
    
    def test_mu_zero_fallback(self):
        """When mu=0, should fall back to non-strongly convex formula."""
        K = 7
        result_sc = get_strongly_convex_silver_stepsizes(K, mu=0, L=1)
        result_nsc = get_nonstrongly_convex_silver_stepsizes(K, L=1)
        for s1, s2 in zip(result_sc, result_nsc):
            assert np.isclose(s1, s2, rtol=1e-10)
    
    def test_length_power_of_2(self):
        """Test that output length matches K for powers of 2."""
        for K in [2, 4, 8, 16]:
            result = get_strongly_convex_silver_stepsizes(K, mu=1, L=10)
            assert len(result) == K
    
    def test_positive_stepsizes(self):
        """All step sizes should be positive."""
        for K in [2, 4, 8]:
            result = get_strongly_convex_silver_stepsizes(K, mu=1, L=10)
            assert all(s > 0 for s in result)
    
    def test_stepsizes_bounded(self):
        """Step sizes should be bounded between mu and L parameters."""
        K = 8
        mu, L = 1, 10
        result = get_strongly_convex_silver_stepsizes(K, mu=mu, L=L)
        for s in result:
            assert s > 0
            assert s < 2 * L  # Very loose upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
