from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.operators import LinearOperator
from PEPit.primitive_steps import proximal_step

def wc_chambolle_pock_averaged_iterate(tau, sigma, theta, n, L_M=1., verbose=1):
    """
    Worst-case analysis of Chambolle-Pock (PDHG) for the last iterate.
    
    Metric: Distance to the saddle point (Lyapunov function).
    Formulation: min_x max_y L(x, u) = f1(x) + <M@x, u> - h(u)

    Convention: y = M@x, v = M.T@u
    """
    problem = PEP()

    # 1. Define the Lagrangian Components
    # We reformulate min_x (f1(x) + f2(M@x)) as min_x max_u (f1(x) + <M@x,u> - f2*(u))
    # Let h(u) = f2*(u). 
    f1 = problem.declare_function(ConvexFunction)
    h  = problem.declare_function(ConvexFunction) 
    M = problem.declare_function(LinearOperator, L=L_M)

    # 2. Define the Optimal Saddle Point (xs, us)
    # The optimality conditions for the Lagrangian L(x, u) are:
    #   0 \in \partial_x L(xs, us)  =>  -M.T @ us \in \partial f1(xs)
    #   0 \in \partial_u L(xs, us)  =>   M @ xs \in \partial h(us)
    xs = problem.set_initial_point()
    us = problem.set_initial_point()
    
    # Enforce these conditions in PEPit
    f1.add_point((xs, - M.T.gradient(us), f1.value(xs)))
    h.add_point((us, M.gradient(xs), h.value(us)))

    # 3. Initialize the Algorithm
    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    
    # Constrain initial distance to the saddle point
    # We use the standard Euclidean norm for simplicity, though PDHG 
    # is naturally contractive in the norm ||z||_M where M depends on tau/sigma.
    initial_term = (x0 - xs)**2 / tau + (u0 - us)**2 / sigma - 2 * (u0 - us) * M.gradient(x0 - xs)
    problem.set_initial_condition(initial_term  <= 1)

    # 4. Run the Algorithm (No Averaging)
    x = x0
    u = u0

    x_avg = 0.0*x0
    u_avg = 0.0*u0

    for k in range(n):
        # --- Primal Step ---
        # x_{k+1} = prox_{tau f1}(x_k - tau * M.T@u_k)
        x_new, _, _ = proximal_step(x - tau * M.T.gradient(u), f1, tau)
        
        # --- Extrapolation ---
        # x_bar_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)
        x_bar = x_new + theta * (x_new - x)

        # --- Dual Step ---
        # u_{k+1} = prox_{sigma f2^*}(u_k + sigma * M@x_bar)
        # Note: prox_{sigma f2^*} is exactly prox_{sigma h}
        u_new, _, _ = proximal_step(u + sigma * M.gradient(x_bar), h, sigma)

        # Averaged iterates
        x_avg = ( k * x_avg + x_new ) / (k+1)
        u_avg = ( k * u_avg + u_new ) / (k+1)

        # Update
        x = x_new
        u = u_new

    def Lagrangian(x, u) :
        return f1.value(x) + M.gradient(x) * u - h.value(u)

    L_primal_view = Lagrangian(x_avg, us)
    L_dual_view   = Lagrangian(xs, u_avg)
    gap = L_primal_view - L_dual_view
    
    problem.set_performance_metric(gap)
    pepit_result = problem.solve(verbose=verbose)
    
    return pepit_result

if __name__ == "__main__":
    # Standard PDHG condition: tau * sigma * L_M ** 2 <= 1 (often < 1 for strict convergence)
    wc_dist = wc_chambolle_pock_averaged_iterate(tau=1.0, sigma=1.0, theta=1.0, n=3, L_M=1.0)
    print(f"Worst-case Duality Gap on Averaged Iterate: {wc_dist}")
