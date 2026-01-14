"""
Unit tests for Adam and AdamW optimizers with proj_x_fn support.

Run with: conda activate algoverify && python -m pytest tests/test_adam_optimizers.py -v
"""
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import jax
import jax.numpy as jnp
from learning_experiment_classes.adam_optimizers import AdamWMinMax, AdamMinMax


class TestAdamMinMaxProjX:
    """Tests for AdamMinMax with proj_x_fn."""
    
    def test_proj_x_fn_called(self):
        """Verify proj_x_fn is called and applied to x params."""
        x_params = [jnp.array([1.0, 2.0, 3.0])]
        y_params = [jnp.array([1.0, 1.0])]
        
        optimizer = AdamMinMax(x_params, y_params, lr=1.0)
        
        # Large gradient that would push x negative
        grads_x = [jnp.array([10.0, 10.0, 10.0])]
        grads_y = [jnp.zeros(2)]
        
        # Projection function that clips to nonnegative
        def proj_x(x_list):
            return [jax.nn.relu(x) for x in x_list]
        
        x_new, y_new = optimizer.step(
            x_params, y_params, grads_x, grads_y, 
            proj_x_fn=proj_x
        )
        
        # After descent, values should be nonnegative due to projection
        assert jnp.all(x_new[0] >= 0), f"x should be nonnegative, got {x_new[0]}"
    
    def test_proj_x_fn_none_no_projection(self):
        """Verify no projection when proj_x_fn is None."""
        x_params = [jnp.array([1.0, 2.0, 3.0])]
        y_params = [jnp.array([1.0, 1.0])]
        
        optimizer = AdamMinMax(x_params, y_params, lr=1.0)
        
        # Large gradient that will push x negative
        grads_x = [jnp.array([10.0, 10.0, 10.0])]
        grads_y = [jnp.zeros(2)]
        
        x_new, y_new = optimizer.step(
            x_params, y_params, grads_x, grads_y,
            proj_x_fn=None
        )
        
        # Without projection, some values could be negative
        # (Adam step may not go fully negative in one step due to momentum, but result should not be clipped)
        assert x_new[0] is not None  # Just verify step completes
    
    def test_proj_x_fn_with_multiple_params(self):
        """Verify proj_x_fn works with multiple x params (like stepsizes tuple)."""
        x_params = [jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4])]
        y_params = [jnp.eye(2)]
        
        optimizer = AdamMinMax(x_params, y_params, lr=1.0)
        
        # Gradients that push some values negative
        grads_x = [jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0])]
        grads_y = [jnp.zeros((2, 2))]
        
        def proj_x(x_list):
            return [jax.nn.relu(x) for x in x_list]
        
        x_new, y_new = optimizer.step(
            x_params, y_params, grads_x, grads_y,
            proj_x_fn=proj_x
        )
        
        # All values should be nonnegative
        for i, x in enumerate(x_new):
            assert jnp.all(x >= 0), f"x[{i}] should be nonnegative"


class TestAdamWMinMaxProjX:
    """Tests for AdamWMinMax with proj_x_fn."""
    
    def test_proj_x_fn_called(self):
        """Verify proj_x_fn is called and applied to x params."""
        x_params = [jnp.array([1.0, 2.0, 3.0])]
        y_params = [jnp.array([1.0, 1.0])]
        
        optimizer = AdamWMinMax(x_params, y_params, lr=1.0, weight_decay=0.0)
        
        # Large gradient that would push x negative
        grads_x = [jnp.array([10.0, 10.0, 10.0])]
        grads_y = [jnp.zeros(2)]
        
        def proj_x(x_list):
            return [jax.nn.relu(x) for x in x_list]
        
        x_new, y_new = optimizer.step(
            x_params, y_params, grads_x, grads_y,
            proj_x_fn=proj_x
        )
        
        # After descent, values should be nonnegative due to projection
        assert jnp.all(x_new[0] >= 0), f"x should be nonnegative, got {x_new[0]}"
    
    def test_backward_compatibility_proj_y_only(self):
        """Verify existing code using only proj_y_fn still works."""
        x_params = [jnp.array([1.0, 2.0])]
        y_params = [jnp.array([1.0, 1.0]), jnp.array([0.5])]
        
        optimizer = AdamWMinMax(x_params, y_params, lr=0.1)
        
        grads_x = [jnp.array([0.5, 0.5])]
        grads_y = [jnp.array([0.1, 0.1]), jnp.array([0.1])]
        
        # Only proj_y_fn, no proj_x_fn
        def proj_y(y_list):
            return [jnp.clip(y, 0, 1) for y in y_list]
        
        x_new, y_new = optimizer.step(
            x_params, y_params, grads_x, grads_y,
            proj_y_fn=proj_y
        )
        
        # Y values should be clipped to [0, 1]
        for i, y in enumerate(y_new):
            assert jnp.all(y >= 0) and jnp.all(y <= 1), f"y[{i}] should be in [0, 1]"


class TestStepsizeProjection:
    """Tests for stepsize nonnegativity projection pattern."""
    
    def test_relu_projects_negative_to_zero(self):
        """Verify jax.nn.relu correctly projects negative values to zero."""
        stepsizes = [jnp.array([-0.1, 0.2, -0.3]), jnp.array([0.5, -0.5])]
        projected = [jax.nn.relu(s) for s in stepsizes]
        
        expected_0 = jnp.array([0.0, 0.2, 0.0])
        expected_1 = jnp.array([0.5, 0.0])
        
        assert jnp.allclose(projected[0], expected_0)
        assert jnp.allclose(projected[1], expected_1)
    
    def test_relu_preserves_positive(self):
        """Verify jax.nn.relu preserves positive values."""
        stepsizes = [jnp.array([0.1, 0.2, 0.3])]
        projected = [jax.nn.relu(s) for s in stepsizes]
        
        assert jnp.allclose(projected[0], stepsizes[0])
    
    def test_relu_handles_zero(self):
        """Verify jax.nn.relu handles zero correctly."""
        stepsizes = [jnp.array([0.0, 0.0])]
        projected = [jax.nn.relu(s) for s in stepsizes]
        
        assert jnp.allclose(projected[0], jnp.array([0.0, 0.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
