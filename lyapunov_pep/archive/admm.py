import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def interp_conditions(G, F, XM, GM, FM, L, mu, n_points):
    constraints = []
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                xi, xj = XM[i, :], XM[j, :]
                gi, gj = GM[i, :], GM[j, :]
                fi, fj = FM[i, :], FM[j, :]

                # import ipdb
                #
                # ipdb.set_trace()

                if L != np.inf:
                    term1 = (fj - fi) @ F
                    term2 = gj @ G @ (xi - xj).T
                    term3 = (1 / (2 * L)) * (gi - gj) @ G @ (gi - gj).T
                    term4 = (
                        mu
                        / (2 * (1 - (mu / L)))
                        * (xi - xj - (1 / L) * (gi - gj))
                        @ G
                        @ (xi - xj - (1 / L) * (gi - gj)).T
                    ) if mu < L else 0.0
                    constraints += [0 >= term1 + term2 + term3 + term4]
                else:
                    term1 = (fj - fi) @ F
                    term2 = gj @ G @ (xi - xj).T
                    term3 = (mu / 2) * (xi - xj) @ G @ (xi - xj).T
                    constraints += [0 >= term1 + term2 + term3]

    return constraints


def solve_admm_pep(rho_list, Lf=1, mf=0.1, Lg=4, mg=0.2):
    """
    Solve the performance estimation problem for ADMM with given step sizes.

    Args:
        rho_list: List of step sizes, one per iteration
        Lf: Lipschitz constant for function f (default: 1)
        mf: Strong convexity parameter for function f (default: 0.1)
        Lg: Lipschitz constant for function g (default: 4)
        mg: Strong convexity parameter for function g (default: 0.2)

    Returns:
        dict: Contains 'status' and 'value' from the optimization problem
    """
    # Infer number of iterations from step sizes
    N = len(rho_list)

    # Convert to numpy array for easier indexing
    rho = np.array(rho_list)

    # Note: dual variable is y = u * t

    # Formulate PEP problem
    # P = [x0 | z0 | y0 | xs | zs | us | df(x0) | df(x_N-1) | df(z0) | ... | df(z_N-1) | dfs]
    # F = [f0]
    # G = P'P

    n_var = 3  # number of variables
    dimG = n_var + n_var + 2 * N + 1
    dimF = 2 * (N + 1)  # once for each x and z + optimal value
    n_points = 1 + N  # number of points to interpolate
    slices = {
        "w0": slice(0, n_var),
        "ws": slice(n_var, 2 * n_var),
        "df": slice(2 * n_var, 2 * n_var + N),
        "dg": slice(2 * n_var + N, 2 * n_var + 2 * N),
        "dfs": 2 * n_var + 2 * N,
        "dgs": 2 * n_var + 2 * N + 1,
        "f": slice(0, N),
        "g": slice(N, 2 * N),
        "fs": 2 * N,
        "gs": 2 * N + 1,
    }

    # Variables
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)

    # Constraints
    constraints = [G >> 0]  # PSD constraint

    # Define selector matrices and call them directly
    # to create problem
    # e.g., true_x0 = P @ x0.T

    # Initial iterates
    x0 = np.zeros((1, dimG))
    x0[0, 0] = 1
    z0 = np.zeros((1, dimG))
    z0[0, 1] = 1
    y0 = np.zeros((1, dimG))
    y0[0, 2] = 1

    # Optimal solution
    xs = np.zeros((1, dimG))
    xs[0, slices["ws"]][0] = 1
    zs = xs
    ys = np.zeros((1, dimG))
    ys[0, slices["ws"]][2] = 1

    # subgradients
    df = np.zeros((N, dimG))
    df[:, slices["df"]] = np.eye(N)
    dfs = np.zeros((1, dimG))
    dfs[0, slices["dfs"]] = 1
    dg = np.zeros((N, dimG))
    dg[:, slices["dg"]] = np.eye(N)

    # optimality conditions 0 \in \partial f(x^\star) + \partial g(z^\star)
    ys = -dfs
    dgs = ys

    # Functions values
    f = np.zeros((N, dimF))
    f[:, slices["f"]] = np.eye(N)
    fs = np.zeros((1, dimF))
    g = np.zeros((N, dimF))
    g[:, slices["g"]] = np.eye(N)
    gs = np.zeros((1, dimF))

    # Optimal values
    fs[0, slices["f"]] = 1
    gs[0, slices["g"]] = 1

    """Algorithm"""
    xk = np.zeros((N + 1, dimG))
    xk[0, :] = x0
    zk = np.zeros((N + 1, dimG))
    zk[0, :] = z0
    yk = np.zeros((N + 1, dimG))
    yk[0, :] = y0

    for i in range(N):
        rho_i = rho[i]  # Use step size for iteration i
        xk[i + 1, :] = zk[i, :] - 1 / rho_i * yk[i, :] - rho_i * df[i, :]
        zk[i + 1, :] = xk[i + 1, :] + 1 / rho_i * yk[i, :] - rho_i * dg[i, :]
        yk[i + 1, :] = yk[i, :] + rho_i * (xk[i + 1, :] - zk[i + 1, :])

    """Interpolation conditions"""
    # variables, gradients, and values
    XF = np.vstack([xs, xk[1:, :]])
    # XF = np.vstack([xs, xk])
    GF = np.vstack([dfs, df])
    FF = np.vstack([fs, f])

    # variables, gradients, and values
    XG = np.vstack([zs, zk[1:, :]])
    # XG = np.vstack([zs, zk])
    GG = np.vstack([dgs, dg])
    FG = np.vstack([gs, g])

    # Q: why do iterates start at 1 instead of 0?
    # A: they are useless.  You can start from 0 but you need to also have gradients and function values

    """Constraints"""
    # Initial distance constraints
    # constraints += [(x0 - xs) @ G @ (x0 - xs).T <= 1]
    constraints += [(z0 - zs) @ G @ (z0 - zs).T <= 1]
    constraints += [(y0 - ys) @ G @ (y0 - ys).T <= 1]

    # Interpolation conditions
    constraints += interp_conditions(G, F, XF, GF, FF, Lf, mf, n_points)
    constraints += interp_conditions(G, F, XG, GG, FG, Lg, mg, n_points)

    """Objective function"""
    # objective = (xk[-1, :] - xs) @ G @ (xk[-1, :] - xs).T + (xk[-1, :] - xs) @ G @ (xk[-1, :] - xs).T

    # duality gap L(xk, zk, ys) - L(xs, zs, yk)
    objective = (f[-1] + g[-1] - fs - gs) @ F + ys @ G @ (zk[-1, :] - xk[-1, :]).T
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    return {"status": problem.status, "value": problem.value}


def create_stepsize_heatmap(rho1_range, rho2_range, filename="admm_stepsize_heatmap.pdf", **kwargs):
    """
    Create a heatmap showing the performance estimation values for different combinations
    of two step sizes in ADMM.

    Args:
        rho1_range: Array-like, range of values for first step size
        rho2_range: Array-like, range of values for second step size
        filename: String, output filename for PDF
        **kwargs: Additional arguments passed to solve_admm_pep

    Returns:
        tuple: (rho1_grid, rho2_grid, values_grid) containing the meshgrid and values
    """
    rho1_range = np.array(rho1_range)
    rho2_range = np.array(rho2_range)

    # Create meshgrid
    rho1_grid, rho2_grid = np.meshgrid(rho1_range, rho2_range)
    values_grid = np.zeros_like(rho1_grid)

    # Evaluate PEP for each combination
    total_combinations = len(rho1_range) * len(rho2_range)
    completed = 0

    for i, rho1 in enumerate(rho1_range):
        for j, rho2 in enumerate(rho2_range):
            try:
                result = solve_admm_pep([rho1, rho2], **kwargs)
                if result["status"] == "optimal":
                    values_grid[j, i] = result["value"]
                else:
                    values_grid[j, i] = np.nan
            except Exception:
                values_grid[j, i] = np.nan

            # Update progress
            completed += 1
            progress = completed / total_combinations * 100
            bar_length = 50
            filled_length = int(bar_length * completed // total_combinations)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(
                f"\rProgress: |{bar}| {progress:.1f}% ({completed}/{total_combinations})",
                end="",
                flush=True,
            )

    print()  # New line after progress bar

    # Create heatmap with logarithmic scale
    plt.figure(figsize=(10, 8))

    # Filter out non-positive values for log scale
    valid_values = values_grid[~np.isnan(values_grid)]
    if len(valid_values) > 0 and np.min(valid_values) > 0:
        # Use logarithmic normalization
        im = plt.imshow(
            values_grid,
            extent=[rho1_range.min(), rho1_range.max(), rho2_range.min(), rho2_range.max()],
            origin="lower",
            cmap="viridis",
            aspect="auto",
            norm=LogNorm(vmin=np.nanmin(valid_values), vmax=np.nanmax(valid_values)),
        )
    else:
        # Fallback to linear scale if logarithmic isn't suitable
        im = plt.imshow(
            values_grid,
            extent=[rho1_range.min(), rho1_range.max(), rho2_range.min(), rho2_range.max()],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )

    plt.colorbar(im, label="PEP Objective Value")
    plt.xlabel("Step size rho (iteration 1)")
    plt.ylabel("Step size rho (iteration 2)")
    plt.title("ADMM Performance Estimation Problem\nObjective Value vs Step Sizes")

    # Save to PDF
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Heatmap saved to {filename}")
    return rho1_grid, rho2_grid, values_grid


if __name__ == "__main__":
    # Test individual step size combinations
    print("Testing solve_admm_pep function:")
    rho_list = [10, 10]
    result = solve_admm_pep(rho_list, Lf=1.0, mf=1.0, Lg=1.0, mg=0.1)
    print(f"Step sizes {rho_list}: Status = {result['status']}, Value = {result['value']:.6f}")

    # # Test heatmap generation
    # print("\nGenerating heatmap:")
    # rho1_range = np.linspace(0.01, 10.0, 30)
    # rho2_range = np.linspace(0.01, 10.0, 30)
    #
    # grid1, grid2, values = create_stepsize_heatmap(
    #     rho1_range, rho2_range, filename="admm_stepsize_heatmap.pdf"
    # )
    #
    # print("Heatmap completed:")
    # print(f"  Grid size: {values.shape}")
    # print(f"  Optimal solutions: {np.sum(~np.isnan(values))}/{values.size}")
    # print(f"  Value range: {np.nanmin(values):.4f} to {np.nanmax(values):.4f}")
