from PEPit import PEP
from PEPit.functions import ConvexFunction, ConvexLipschitzFunction
from PEPit.operators import LinearOperator
from PEPit.primitive_steps import proximal_step

def wc_chambolle_pock_last_iterate(tau, sigma, theta, n, M=1, verbose=1):
    """
    Worst-case analysis of Chambolle-Pock (PDHG) for the last iterate.
    
    Metric: Distance to the saddle point (Lyapunov function).
    Formulation: min_x max_y L(x, y) = f1(x) + M * <x, y> - h(y)
    """
    problem = PEP()

    # 1. Define the Lagrangian Components
    # We reformulate min_x (f1(x) + f2(x)) as min_x max_y (f1(x) + M * <x,y> - f2*(y))
    # Let h(y) = f2*(y). 
    f1 = problem.declare_function(ConvexFunction)
    h  = problem.declare_function(ConvexFunction) 

    # 2. Define the Optimal Saddle Point (xs, ys)
    # The optimality conditions for the Lagrangian L(x, y) are:
    #   0 \in \partial_x L(xs, ys)  =>  -M * ys \in \partial f1(xs)
    #   0 \in \partial_y L(xs, ys)  =>   M * xs \in \partial h(ys)
    xs = problem.set_initial_point()
    ys = problem.set_initial_point()
    
    # Enforce these conditions in PEPit
    # f1.add_point(xs, g=-ys, f=f1.value(xs))
    # h.add_point(ys, g=xs,  f=h.value(ys))
    f1.add_point((xs, -M * ys, f1.value(xs)))
    h.add_point((ys, M * xs, h.value(ys)))

    # 3. Initialize the Algorithm
    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()
    
    # Constrain initial distance to the saddle point
    # We use the standard Euclidean norm for simplicity, though PDHG 
    # is naturally contractive in the norm ||z||_M where M depends on tau/sigma.
    problem.set_initial_condition((x0 - xs)**2 + (y0 - ys)**2 <= 1)

    # 4. Run the Algorithm (No Averaging)
    x = x0
    y = y0

    for k in range(n):
        # --- Primal Step ---
        # x_{k+1} = prox_{tau f1}(x_k - tau * y_k)
        x_new, _, _ = proximal_step(x - tau * M * y, f1, tau)
        
        # --- Extrapolation ---
        # x_bar_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)
        x_bar = x_new + theta * (x_new - x)
        
        # --- Dual Step ---
        # y_{k+1} = prox_{sigma f2^*}(y_k + sigma * x_bar)
        # Note: prox_{sigma f2^*} is exactly prox_{sigma h}
        y_new, _, _ = proximal_step(y + sigma * M * x_bar, h, sigma)

        # Update
        x = x_new
        y = y_new

    L_primal_view = f1.value(x) + M * (x * ys) - h.value(ys)
    
    # Term 2: L(xs, y_avg) = f1(xs) + <xs, y_avg> - h(y_avg)
    L_dual_view   = f1.value(xs) + M * (xs * y) - h.value(y)
    
    gap = L_primal_view - L_dual_view
    
    problem.set_performance_metric(gap)

    pepit_result = problem.solve(verbose=verbose)
    
    return pepit_result


def wc_chambolle_pock_last_iterate_linop(tau, sigma, theta, n, L_M=1, verbose=1):
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

    # x_avg = 0.0*x0
    # u_avg = 0.0*u0

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
        # x_avg = ( k * x_avg + x_new ) / (k+1)
        # u_avg = ( k * u_avg + u_new ) / (k+1)

        # Update
        x = x_new
        u = u_new

    def Lagrangian(x, u) :
        return f1.value(x) + M.gradient(x) * u - h.value(u)

    L_primal_view = Lagrangian(x, us)
    L_dual_view   = Lagrangian(xs, u)
    gap = L_primal_view - L_dual_view
    
    problem.set_performance_metric(gap)
    pepit_result = problem.solve(verbose=verbose)
    
    return pepit_result


if __name__ == "__main__":
    # Example: N=1 step. If result <= 1, the algorithm is non-expansive (stable).
    # If result < 1, it is strictly contractive.
    # Standard PDHG condition: tau * sigma * M ** 2 <= 1 (often < 1 for strict convergence)
    # wc_dist = wc_chambolle_pock_last_iterate(tau=0.01, sigma=0.01, theta=1.0, n=15, M=10)
    # print(f"Worst-case Squared Distance to Saddle Point (NO LINOP): {wc_dist}")

    wc_dist = wc_chambolle_pock_last_iterate_linop(tau=0.01, sigma=0.01, theta=1.0, n=1, L_M=10)
    print(f"Worst-case Squared Distance to Saddle Point (LINOP): {wc_dist}")
