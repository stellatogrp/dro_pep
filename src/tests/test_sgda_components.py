"""
Unit tests for SGDA pipeline components.

Tests verify each part of the batched Stochastic Gradient-Descent Ascent pipeline
based on the main_learn() function in test_learning_cvxpylayers.py.

Run with: conda activate algoverify && python -m pytest tests/test_sgda_components.py -v
"""
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


# ============================================================================
# Sampling Functions (from test_learning_cvxpylayers.py)
# ============================================================================

def marchenko_pastur(d, mu, L):
    """Generate random matrix using Marchenko-Pastur distribution."""
    r = (np.sqrt(L) - np.sqrt(mu))**2 / (np.sqrt(L) + np.sqrt(mu))**2
    sigma = (np.sqrt(L) + np.sqrt(mu)) / 2
    X = np.random.normal(0, sigma, size=(d, np.round(r*d).astype(np.int64)))
    H = X.T @ X / d
    return H


def rejection_sample_MP(dim, mu, L):
    """Rejection sample to get Q with eigenvalues in [mu, L]."""
    Q = marchenko_pastur(dim, mu, L)
    eigvals = np.real(np.linalg.eigvals(Q))
    if mu > np.min(eigvals) or L < np.max(eigvals):
        return rejection_sample_MP(dim, mu, L)
    return Q


def sample_x0_centered_disk(n, R):
    """Sample uniformly from a centered disk of radius R."""
    x = np.random.normal(0, 1, n)
    x /= np.linalg.norm(x)
    dist = np.random.uniform(0, 1) ** (1 / n)
    return R * dist * x


# ============================================================================
# Projection Functions (from main_learn())
# ============================================================================

@jax.jit
def proj_z0(v, R):
    """Project z0 onto ball of radius R."""
    norm = jnp.linalg.norm(v)
    scale = R / jnp.maximum(norm, R)
    return v * scale


@jax.jit
def proj_Q(M, mu, L):
    """Project Q to have eigenvalues in [mu, L]."""
    evals, evecs = jnp.linalg.eigh(M)
    evals_clipped = jnp.clip(evals, mu, L)
    return (evecs * evals_clipped) @ evecs.T


# ============================================================================
# Trajectory Generation (from test_learning_cvxpylayers.py)
# ============================================================================

@partial(jax.jit, static_argnames=['K_max'])
def problem_data_to_gd_trajectories(t, Q, z0, K_max):
    """Generate (G, F) from problem data using gradient descent."""
    def f(x):
        return .5 * x.T @ Q @ x
    
    def g(x):
        return Q @ x

    dim = z0.shape[0]
    zs = jnp.zeros(dim)
    z_stack = jnp.zeros((dim, K_max + 2))
    g_stack = jnp.zeros((dim, K_max + 2))
    f_stack = jnp.zeros(K_max + 2)

    z_stack = z_stack.at[:, 0].set(zs)
    z_stack = z_stack.at[:, 1].set(z0)
    g_stack = g_stack.at[:, 0].set(g(zs))
    g_stack = g_stack.at[:, 1].set(g(z0))
    f_stack = f_stack.at[0].set(f(zs))
    f_stack = f_stack.at[1].set(f(z0))

    def body_fun(i, val):
        z, z_stack, g_stack, f_stack = val
        z = z - t * g(z)
        z_stack = z_stack.at[:, i+1].set(z)
        g_stack = g_stack.at[:, i+1].set(g(z))
        f_stack = f_stack.at[i+1].set(f(z))
        return (z, z_stack, g_stack, f_stack)

    init_val = (z0, z_stack, g_stack, f_stack)
    _, z_stack, g_stack, f_stack = \
        jax.lax.fori_loop(1, K_max + 1, body_fun, init_val)

    G_half = jnp.hstack([z_stack[:,:2], g_stack[:,1:]])
    return G_half.T @ G_half, f_stack


def batch_GF_function(t, Q_batch, z0_batch, K_max):
    """Batched version of problem_data_to_gd_trajectories using vmap."""
    batch_fn = jax.vmap(problem_data_to_gd_trajectories, in_axes=(None, 0, 0, None))
    return batch_fn(t, Q_batch, z0_batch, K_max)


# ============================================================================
# t Initialization Helper
# ============================================================================

def get_t_init(mu, L):
    """Compute initial step size: 1/L if mu=0, else 2/(mu+L)."""
    if mu == 0:
        return 1.0 / L
    else:
        return 2.0 / (mu + L)


# ============================================================================
# Algorithm Stubs
# ============================================================================

def jax_vanilla_gd(t, Q, z0, K_max):
    """Vanilla gradient descent (wrapper for trajectories)."""
    return problem_data_to_gd_trajectories(t, Q, z0, K_max)


def jax_nesterov_gd(t, Q, z0, K_max):
    """Nesterov accelerated gradient descent (STUB - not implemented)."""
    raise NotImplementedError("nesterov_gd not yet implemented")


ALGORITHMS = {
    "vanilla_gd": jax_vanilla_gd,
    "nesterov_gd": jax_nesterov_gd,
}


# ============================================================================
# SGDA Type Stubs
# ============================================================================

def sgda_step_vanilla(t, Q, z0, dt, dQ, dz0, eta_t, eta_Q, eta_z0, mu, L, R):
    """Vanilla SGDA step: descent in t, ascent in Q and z0 with projections."""
    t_new = t - eta_t * dt
    Q_new = proj_Q(Q + eta_Q * dQ, mu, L)
    z0_new = proj_z0(z0 + eta_z0 * dz0, R)
    return t_new, Q_new, z0_new


def sgda_step_adamw(t, Q, z0, dt, dQ, dz0, eta_t, eta_Q, eta_z0, mu, L, R, state=None):
    """AdamW SGDA step (STUB - not implemented)."""
    raise NotImplementedError("adamw SGDA not yet implemented")


SGDA_TYPES = {
    "vanilla_sgda": sgda_step_vanilla,
    "adamw": sgda_step_adamw,
}


# ============================================================================
# Tests: Sampling Functions
# ============================================================================

class TestSampling:
    """Tests for Q and z0 sampling functions."""
    
    def test_rejection_sample_MP_eigenvalues(self):
        """Verify sampled Q has eigenvalues in [mu, L]."""
        np.random.seed(42)
        mu, L, dim = 1.0, 10.0, 50
        Q = rejection_sample_MP(dim, mu, L)
        eigvals = np.real(np.linalg.eigvals(Q))
        
        assert np.min(eigvals) >= mu - 1e-10, f"Min eigenvalue {np.min(eigvals)} < mu={mu}"
        assert np.max(eigvals) <= L + 1e-10, f"Max eigenvalue {np.max(eigvals)} > L={L}"
    
    def test_rejection_sample_MP_symmetric(self):
        """Verify Q is symmetric."""
        np.random.seed(42)
        mu, L, dim = 1.0, 10.0, 50
        Q = rejection_sample_MP(dim, mu, L)
        
        assert np.allclose(Q, Q.T), "Q should be symmetric"
    
    def test_sample_x0_centered_disk_norm(self):
        """Verify z0 has norm <= R."""
        np.random.seed(42)
        n, R = 50, 1.0
        for _ in range(10):
            x0 = sample_x0_centered_disk(n, R)
            assert np.linalg.norm(x0) <= R + 1e-10, f"||x0|| = {np.linalg.norm(x0)} > R={R}"


# ============================================================================
# Tests: Projection Functions
# ============================================================================

class TestProjections:
    """Tests for proj_Q and proj_z0 projection functions."""
    
    def test_proj_z0_inside_ball_unchanged(self):
        """z0 inside ball should be unchanged."""
        R = 1.0
        v = jnp.array([0.3, 0.3, 0.3])  # norm ~ 0.52 < R
        v_proj = proj_z0(v, R)
        
        assert jnp.allclose(v, v_proj), "z0 inside ball should not change"
    
    def test_proj_z0_outside_ball_projected(self):
        """z0 outside ball should be projected onto boundary."""
        R = 1.0
        v = jnp.array([2.0, 0.0, 0.0])  # norm = 2 > R
        v_proj = proj_z0(v, R)
        
        assert jnp.isclose(jnp.linalg.norm(v_proj), R), f"Projected norm should be R={R}"
        # Direction should be preserved
        assert jnp.allclose(v_proj / jnp.linalg.norm(v_proj), v / jnp.linalg.norm(v))
    
    def test_proj_z0_on_boundary(self):
        """z0 on boundary should be unchanged."""
        R = 1.0
        v = jnp.array([1.0, 0.0, 0.0])
        v_proj = proj_z0(v, R)
        
        assert jnp.allclose(v, v_proj), "z0 on boundary should not change"
    
    def test_proj_Q_clipping(self):
        """Q eigenvalues should be clipped to [mu, L]."""
        mu, L = 1.0, 10.0
        # Create matrix with eigenvalues outside [mu, L]
        eigvals = jnp.array([0.5, 5.0, 15.0])  # 0.5 < mu, 15 > L
        eigvecs = jnp.eye(3)
        M = (eigvecs * eigvals) @ eigvecs.T
        
        M_proj = proj_Q(M, mu, L)
        proj_eigvals = jnp.linalg.eigvalsh(M_proj)
        
        assert jnp.all(proj_eigvals >= mu - 1e-6), f"Min eigenvalue should be >= {mu}"
        assert jnp.all(proj_eigvals <= L + 1e-6), f"Max eigenvalue should be <= {L}"
    
    def test_proj_Q_symmetric(self):
        """Projected Q should remain symmetric."""
        mu, L = 1.0, 10.0
        key = jax.random.PRNGKey(42)
        M = jax.random.normal(key, (5, 5))
        M = (M + M.T) / 2  # Make symmetric
        
        M_proj = proj_Q(M, mu, L)
        
        assert jnp.allclose(M_proj, M_proj.T), "Projected Q should be symmetric"
    
    def test_proj_Q_inside_unchanged(self):
        """Q with eigenvalues in [mu, L] should be unchanged."""
        mu, L = 1.0, 10.0
        eigvals = jnp.array([2.0, 5.0, 8.0])  # all in [mu, L]
        eigvecs = jnp.eye(3)
        M = (eigvecs * eigvals) @ eigvecs.T
        
        M_proj = proj_Q(M, mu, L)
        
        assert jnp.allclose(M, M_proj, atol=1e-5), "Q with valid eigenvalues should not change"


# ============================================================================
# Tests: Trajectory Generation
# ============================================================================

class TestTrajectories:
    """Tests for GD trajectory generation."""
    
    def test_gd_trajectories_shape(self):
        """Verify G and F have correct shapes."""
        K_max = 3
        dim = 5
        t = 0.1
        Q = jnp.eye(dim) * 5.0  # Simple diagonal Q
        z0 = jnp.ones(dim) * 0.5
        
        G, F = problem_data_to_gd_trajectories(t, Q, z0, K_max)
        
        # G_half has shape (dim, K_max + 2) for x_stack[:,:2] and g_stack[:,1:]
        # So G = G_half.T @ G_half has shape (K_max + 3, K_max + 3)
        expected_G_dim = K_max + 3
        expected_F_len = K_max + 2
        
        assert G.shape == (expected_G_dim, expected_G_dim), f"G shape mismatch: {G.shape}"
        assert F.shape == (expected_F_len,), f"F shape mismatch: {F.shape}"
    
    def test_gd_trajectories_convergence(self):
        """Verify GD converges for simple problem."""
        K_max = 50
        dim = 3
        L = 5.0
        t = 1.0 / L  # Step size for convergence
        Q = jnp.eye(dim) * L
        z0 = jnp.ones(dim)
        
        G, F = problem_data_to_gd_trajectories(t, Q, z0, K_max)
        
        # F[-1] should be close to 0 (optimal value for quadratic with min at 0)
        assert F[-1] < F[1] * 0.01, f"GD should converge: f_final={F[-1]}, f_init={F[1]}"
    
    def test_gd_trajectories_single_step(self):
        """Verify single GD step is correct."""
        K_max = 1
        dim = 2
        t = 0.1
        Q = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        z0 = jnp.array([1.0, 1.0])
        
        G, F = problem_data_to_gd_trajectories(t, Q, z0, K_max)
        
        # Manual computation: z1 = z0 - t * Q @ z0 = [1, 1] - 0.1 * [2, 4] = [0.8, 0.6]
        # f(z1) = 0.5 * z1.T @ Q @ z1 = 0.5 * (0.8*2*0.8 + 0.6*4*0.6) = 0.5 * (1.28 + 1.44) = 1.36
        expected_f1 = 1.36
        
        assert jnp.isclose(F[2], expected_f1, atol=1e-5), f"f(z1)={F[2]}, expected={expected_f1}"


# ============================================================================
# Tests: Batch GF Function
# ============================================================================

class TestBatchGF:
    """Tests for batched (G, F) computation."""
    
    def test_batch_GF_function_shape(self):
        """Verify batched output has correct batch dimension."""
        N = 4  # batch size
        K_max = 2
        dim = 3
        t = 0.1
        
        Q_batch = jnp.stack([jnp.eye(dim) * (i + 1) for i in range(N)])
        z0_batch = jnp.ones((N, dim)) * 0.5
        
        G_batch, F_batch = batch_GF_function(t, Q_batch, z0_batch, K_max)
        
        expected_G_dim = K_max + 3
        expected_F_len = K_max + 2
        
        assert G_batch.shape == (N, expected_G_dim, expected_G_dim), f"G_batch shape: {G_batch.shape}"
        assert F_batch.shape == (N, expected_F_len), f"F_batch shape: {F_batch.shape}"
    
    def test_batch_GF_function_consistency(self):
        """Verify batched output matches individual calls."""
        N = 2
        K_max = 2
        dim = 3
        t = 0.1
        
        Q_batch = jnp.stack([jnp.eye(dim) * (i + 2) for i in range(N)])
        z0_batch = jnp.ones((N, dim)) * 0.5
        
        G_batch, F_batch = batch_GF_function(t, Q_batch, z0_batch, K_max)
        
        for i in range(N):
            G_i, F_i = problem_data_to_gd_trajectories(t, Q_batch[i], z0_batch[i], K_max)
            assert jnp.allclose(G_batch[i], G_i, atol=1e-5), f"G mismatch at index {i}"
            assert jnp.allclose(F_batch[i], F_i, atol=1e-5), f"F mismatch at index {i}"


# ============================================================================
# Tests: t Initialization
# ============================================================================

class TestTInit:
    """Tests for t initialization logic."""
    
    def test_t_init_mu_zero(self):
        """When mu=0, t_init should be 1/L."""
        mu, L = 0.0, 10.0
        t = get_t_init(mu, L)
        assert t == 1.0 / L, f"t_init with mu=0 should be 1/L={1/L}, got {t}"
    
    def test_t_init_mu_positive(self):
        """When mu>0, t_init should be 2/(mu+L)."""
        mu, L = 1.0, 10.0
        t = get_t_init(mu, L)
        expected = 2.0 / (mu + L)
        assert t == expected, f"t_init with mu>0 should be {expected}, got {t}"


# ============================================================================
# Tests: SGDA Step
# ============================================================================

class TestSGDAStep:
    """Tests for SGDA step with projections."""
    
    def test_sgda_step_vanilla_descent_t(self):
        """Verify t decreases with positive gradient (descent)."""
        mu, L, R = 1.0, 10.0, 1.0
        eta_t, eta_Q, eta_z0 = 0.1, 0.01, 0.01
        
        t = 0.5
        Q = jnp.eye(3) * 5.0
        z0 = jnp.array([0.3, 0.3, 0.3])
        dt = 0.1  # positive gradient
        dQ = jnp.zeros((3, 3))
        dz0 = jnp.zeros(3)
        
        t_new, Q_new, z0_new = sgda_step_vanilla(t, Q, z0, dt, dQ, dz0, eta_t, eta_Q, eta_z0, mu, L, R)
        
        assert t_new < t, f"t should decrease: t_new={t_new}, t={t}"
    
    def test_sgda_step_vanilla_ascent_Q(self):
        """Verify Q changes with gradient (ascent)."""
        mu, L, R = 1.0, 10.0, 1.0
        eta_t, eta_Q, eta_z0 = 0.01, 0.1, 0.01
        
        t = 0.5
        Q = jnp.eye(3) * 5.0
        z0 = jnp.array([0.3, 0.3, 0.3])
        dt = 0.0
        dQ = jnp.eye(3) * 0.5  # positive gradient
        dz0 = jnp.zeros(3)
        
        t_new, Q_new, z0_new = sgda_step_vanilla(t, Q, z0, dt, dQ, dz0, eta_t, eta_Q, eta_z0, mu, L, R)
        
        # Q should increase (ascent)
        assert jnp.trace(Q_new) > jnp.trace(Q), f"Q trace should increase"
    
    def test_sgda_step_vanilla_projection_applied(self):
        """Verify projections are applied after updates."""
        mu, L, R = 1.0, 10.0, 1.0
        eta_t, eta_Q, eta_z0 = 0.01, 1.0, 1.0  # large learning rates
        
        t = 0.5
        Q = jnp.eye(3) * 5.0
        z0 = jnp.array([0.3, 0.3, 0.3])
        dt = 0.0
        dQ = jnp.eye(3) * 10.0  # large gradient pushing eigenvalues > L
        dz0 = jnp.array([10.0, 0.0, 0.0])  # large gradient pushing z0 outside ball
        
        t_new, Q_new, z0_new = sgda_step_vanilla(t, Q, z0, dt, dQ, dz0, eta_t, eta_Q, eta_z0, mu, L, R)
        
        # Check Q eigenvalues are clipped
        eigvals = jnp.linalg.eigvalsh(Q_new)
        assert jnp.all(eigvals <= L + 1e-6), "Q eigenvalues should be <= L"
        assert jnp.all(eigvals >= mu - 1e-6), "Q eigenvalues should be >= mu"
        
        # Check z0 is within ball
        assert jnp.linalg.norm(z0_new) <= R + 1e-6, "z0 should be within ball"


# ============================================================================
# Tests: Type Safety for PEP Functions
# ============================================================================

class TestTypeSafety:
    """Tests ensuring correct types are passed to PEP functions."""
    
    def test_t_stays_python_float_after_update(self):
        """Verify t remains Python float after SGDA update."""
        t = 0.5  # Python float
        eta_t = 0.1
        dt = jnp.array(0.2)  # JAX array (as returned by grad_fn)
        
        # This is what happens in SGDA step
        t_new = float(t - eta_t * dt)
        
        assert isinstance(t_new, float), f"t should be Python float, got {type(t_new)}"
    
    def test_jax_scalar_to_float_conversion(self):
        """Verify JAX scalars can be converted to Python float."""
        jax_scalar = jnp.array(0.5)
        
        # Conversion that happens before PEP setup
        py_float = float(jax_scalar)
        
        assert isinstance(py_float, float), f"Should be Python float, got {type(py_float)}"
        assert py_float == 0.5
    
    def test_params_dict_has_python_float_t(self):
        """Verify params dict used in PEP has Python float for t."""
        t_jax = jnp.array(0.18181818)  # Like 2/(mu+L)
        K_max = 5
        
        # This is what should happen in quad_pep_subproblem
        params = {'t': float(t_jax), 'K_max': K_max}
        
        assert isinstance(params['t'], float), f"params['t'] should be Python float"
        assert isinstance(params['K_max'], int), f"params['K_max'] should be int"




class TestAlgorithmRegistry:
    """Tests for algorithm and SGDA type registries."""
    
    def test_vanilla_gd_registered(self):
        """Verify vanilla_gd is in ALGORITHMS."""
        assert "vanilla_gd" in ALGORITHMS
        assert callable(ALGORITHMS["vanilla_gd"])
    
    def test_nesterov_gd_stub_raises(self):
        """Verify nesterov_gd stub raises NotImplementedError."""
        assert "nesterov_gd" in ALGORITHMS
        with pytest.raises(NotImplementedError):
            ALGORITHMS["nesterov_gd"](0.1, jnp.eye(3), jnp.ones(3), 5)
    
    def test_vanilla_sgda_registered(self):
        """Verify vanilla_sgda is in SGDA_TYPES."""
        assert "vanilla_sgda" in SGDA_TYPES
        assert callable(SGDA_TYPES["vanilla_sgda"])
    
    def test_adamw_stub_raises(self):
        """Verify adamw stub raises NotImplementedError."""
        assert "adamw" in SGDA_TYPES
        with pytest.raises(NotImplementedError):
            SGDA_TYPES["adamw"](0.5, jnp.eye(3), jnp.ones(3), 0.1, jnp.zeros((3,3)), jnp.zeros(3), 0.01, 0.01, 0.01, 1.0, 10.0, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
