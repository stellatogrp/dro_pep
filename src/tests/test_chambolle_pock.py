from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction, ConvexIndicatorFunction, SmoothConvexLipschitzFunction
from PEPit.primitive_steps import proximal_step


def wc_chambolle_pock(tau, sigma, theta, n, verbose=1):
    """
    Worst-case analysis of the Chambolle-Pock algorithm (PDHG) with L=Id.
    
    Args:
        tau (float): Primal step size
        sigma (float): Dual step size
        theta (float): Extrapolation parameter (typically 1.0)
        n (int): Number of iterations
    """
    
    # 1. Instantiate the PEPit problem
    problem = PEP()

    L = 1.0
    # 2. Define the functions f1 and f2
    # The problem is min_x { f1(x) + f2(x) }
    # f1 = problem.declare_function(ConvexFunction)
    # f2 = problem.declare_function(ConvexFunction)

    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=0.1, L=L)
    f2 = problem.declare_function(SmoothStronglyConvexFunction, mu=0.1, L=L)

    # 3. Define the optimal point (saddle point setup)
    # For worst-case analysis of PDHG, we often look at the duality gap or distance to solution.
    # Here we set up a reference optimal point xs.
    func = f1 + f2
    xs = func.stationary_point()
    fs = func(xs)
    
    # 4. Initialize the algorithm
    # x0 is the starting primal point, y0 is the starting dual point
    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()
    
    # Initial condition: bounded distance to solution (standard setup)
    # You might want to bound both primal and dual distances.
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    problem.set_initial_condition((y0 - xs) ** 2 <= 1)
    
    # 5. Run the Algorithm Loop
    x = x0
    y = y0
    x_prev = x0 

    for _ in range(n):
        # --- Primal Step ---
        # x_{k+1} = prox_{tau * f1} (x_k - tau * y_k)
        x_new, _, _ = proximal_step(x - tau * y, f1, tau)
        
        # --- Extrapolation Step ---
        # x_bar_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)
        x_bar = x_new + theta * (x_new - x)
        
        # --- Dual Step ---
        # y_{k+1} = prox_{sigma * f2^*} (y_k + sigma * x_bar)
        # We use Moreau Identity: prox_{s*f*}(v) = v - s * prox_{f/s}(v/s)
        # Here s = sigma.
        z = y + sigma * x_bar
        
        # Compute prox_{f2/sigma}(z/sigma)
        # Note: proximal_step(u, f, gamma) computes argmin f(w) + 1/(2*gamma)||w-u||^2
        # We need argmin (1/sigma)*f2(w) + 1/2||w - z/sigma||^2
        # This is equivalent to proximal_step with gamma = 1/sigma
        p, _, _ = proximal_step(z / sigma, f2, 1 / sigma)
        
        y_new = z - sigma * p
        
        # Update variables for next iteration
        x_prev = x
        x = x_new
        y = y_new

    # 6. Define Performance Metric
    # Example: Distance of the last iterate to the optimal solution
    problem.set_performance_metric((x - xs)**2)
    # problem.set_performance_metric(func(y) - fs)

    # 7. Solve the PEP
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=verbose)
    
    return pepit_tau

# Example usage:
# tau * sigma * L^2 <= 1 (with L=1, theta=1) -> tau * sigma <= 1
if __name__ == "__main__":
    wc = wc_chambolle_pock(tau=0.1, sigma=0.1, theta=1.0, n=2)
    print(f"Worst-case performance: {wc}")