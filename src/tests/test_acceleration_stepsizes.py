"""
Unit tests for acceleration step size schedules.

Run with: pytest tests/test_acceleration_stepsizes.py -v
"""
import pytest
import numpy as np
from learning_experiment_classes.acceleration_stepsizes import (
    get_nesterov_fgm_beta_sequence,
    jax_get_nesterov_fgm_beta_sequence,
)


class TestNesterovFGMBetaSequence:
    """Tests for Nesterov FGM beta sequence functions."""
    
    @pytest.mark.parametrize("mu,L,K_max", [
        (1, 10, 4),
        (1, 10, 8),
        (0.5, 5, 4),
        (2, 20, 6),
        (0, 1, 4),  # Non-strongly convex case
    ])
    def test_numpy_jax_consistency(self, mu, L, K_max):
        """NumPy and JAX versions should return matching values."""
        np_result = get_nesterov_fgm_beta_sequence(mu, L, K_max)
        jax_result = jax_get_nesterov_fgm_beta_sequence(mu, L, K_max)
        np.testing.assert_allclose(np_result, np.array(jax_result), rtol=1e-5)
    
    def test_length(self):
        """Output length should match K_max."""
        for K_max in [2, 4, 8, 16]:
            result = get_nesterov_fgm_beta_sequence(1, 10, K_max)
            assert len(result) == K_max
    
    def test_first_beta_is_zero(self):
        """First beta value should be 0 (A[0]=0 causes numerator to be 0)."""
        result = get_nesterov_fgm_beta_sequence(1, 10, 4)
        assert np.isclose(result[0], 0.0, atol=1e-10)
    
    def test_beta_values_bounded(self):
        """Beta values should be in [0, 1)."""
        for mu, L in [(1, 10), (0.5, 5), (2, 20)]:
            result = get_nesterov_fgm_beta_sequence(mu, L, 8)
            assert all(b >= 0 for b in result)
            assert all(b < 1 for b in result)
    
    def test_beta_increasing(self):
        """Beta values should generally increase (after first)."""
        result = get_nesterov_fgm_beta_sequence(1, 10, 8)
        # Check that values after the first are non-decreasing
        for i in range(1, len(result) - 1):
            assert result[i+1] >= result[i] - 1e-10  # Allow small numerical tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
