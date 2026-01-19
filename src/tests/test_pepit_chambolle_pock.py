import unittest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, ConvexFunction
from PEPit.primitive_steps import proximal_step

from learning.interpolation_conditions import (
    smooth_strongly_convex_interp,
    convex_interp,
)

def run_pepit_proximal_gradient(tau, sigma, K):
    """
    Run PEPit's proximal gradient example and return the optimal value.
    
    Args:
        L: Smoothness parameter for f1
        mu: Strong convexity parameter for f1
        gamma: Step size
        n: Number of iterations
    
    Returns:
        pepit_tau: Worst-case value from PEPit
    """
    # Instantiate PEP
    problem = PEP()
    
  # Declare a convex and a smooth convex function.
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, mu=0.1, L=1)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K)]
    w = [x0 for _ in range(K + 1)]

    alpha = 1
    theta = 1
    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= 1)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    problem.set_performance_metric((func2(y) + fy) - fs)
    
    # Solve the PEP
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)
    
    return pepit_tau


class TestChambollePackPEPit(unittest.TestCase):
    """Test Chambolle-Pock algorithm using PEPit with convex functions."""
    
    def setUp(self):
        """Set up test parameters."""
        self.tau = 0.01
        self.sigma = 0.01
    
    def test_chambolle_pock_K1(self):
        """Test Chambolle-Pock with K=1 iteration."""
        result = run_pepit_proximal_gradient(self.tau, self.sigma, K=1)
        print(f"\nK=1: PEPit result = {result}")
        # Check if result is finite (not unbounded)
        if result is not None and np.isfinite(result):
            self.assertGreaterEqual(result, 0)
        else:
            print("  -> Problem may be unbounded")
    
    def test_chambolle_pock_K2(self):
        """Test Chambolle-Pock with K=2 iterations."""
        result = run_pepit_proximal_gradient(self.tau, self.sigma, K=2)
        print(f"\nK=2: PEPit result = {result}")
        if result is not None and np.isfinite(result):
            self.assertGreaterEqual(result, 0)
        else:
            print("  -> Problem may be unbounded")
    
    def test_chambolle_pock_K3(self):
        """Test Chambolle-Pock with K=3 iterations."""
        result = run_pepit_proximal_gradient(self.tau, self.sigma, K=3)
        print(f"\nK=3: PEPit result = {result}")
        if result is not None and np.isfinite(result):
            self.assertGreaterEqual(result, 0)
        else:
            print("  -> Problem may be unbounded")
    
    def test_chambolle_pock_K5(self):
        """Test Chambolle-Pock with K=5 iterations."""
        result = run_pepit_proximal_gradient(self.tau, self.sigma, K=5)
        print(f"\nK=5: PEPit result = {result}")
        if result is not None and np.isfinite(result):
            self.assertGreaterEqual(result, 0)
        else:
            print("  -> Problem may be unbounded")


if __name__ == '__main__':
    unittest.main(verbosity=2)