
import jax
import jax.numpy as jnp
import numpy as np
from learning.trajectories_pdhg import problem_data_to_pdhg_trajectories
from learning.pep_constructions import construct_chambolle_pock_pep_data

def test_gap_consistency():
    jax.config.update("jax_enable_x64", True)
    
    # Create a simple toy LP: min x s.t. x >= 1
    # c = [1], G = [-1] (so -x <= -1 => x >= 1). wait standard form Gx >= h.
    # so G=[1], h=[1]. 
    # x_opt = 1. y_opt (dual for Gx>=h) ?
    # L(x, y) = x + y(1 - x) = x + y - yx = x(1-y) + y.
    # min_x L: requires 1-y=0 => y=1.
    # max_y L: requires 1-x <= 0 => x>=1.
    # so x*=1, y*=1. L* = 1.
    
    c = jnp.array([1.0])
    G = jnp.array([[1.0]])
    h = jnp.array([1.0])
    # No equality constraints
    A_eq = jnp.zeros((0, 1))
    b_eq = jnp.zeros((0,))
    
    K = jnp.vstack([G, A_eq])
    q = jnp.concatenate([h, b_eq])
    
    l = jnp.array([-10.0]) # Loose bounds
    u = jnp.array([10.0])
    
    x_opt = jnp.array([1.0])
    y_opt = jnp.array([1.0])
    f_opt = 1.0
    
    m1 = 1
    K_max = 2
    
    # Step sizes satisfying tau * sigma * ||K||^2 < 1. 
    # ||K|| = 1. 
    # tau=0.5, sigma=0.5.
    stepsizes = (0.5, 0.5, 1.0)
    
    x0 = jnp.array([0.0])
    y0 = jnp.array([0.0])
    
    # Case 1: M_global = ||K|| = 1.0 (Matched)
    print("\n--- CASE 1: Matched M ---")
    G, F = problem_data_to_pdhg_trajectories(
        stepsizes, c, K, q, l, u, x0, y0, x_opt, y_opt, f_opt,
        K_max=K_max, m1=m1, return_Gram_representation=True
    )
    
    M_global = 1.0
    R = 100.0
    pep_data = construct_chambolle_pock_pep_data(0.5, 0.5, 1.0, M_global, R, K_max)
    A_obj, b_obj = pep_data[0], pep_data[1]
    
    # Compute Gap value: Tr(A_obj @ G) + b_obj @ F
    # Note: b_obj matches F structure?
    # F in trajectory: [F1, F_h]
    # b_obj in PEP: [fK_f1, fK_h] which selects the last elements.
    # The term is actually just b_obj @ F ? No, b_obj are coefficients?
    # In pep_construction: b_obj = concatenate([fK_f1, fK_h])
    # These are unit vectors selecting the K-th value.
    # Wait, b_obj in PEP construction (line 198) returns `fK_f1` which is a VECTOR (dimF1 or dimF_h).
    # Ah, construct_pep returns A_obj (matrix), b_obj (vector).
    # The objective is Tr(A_obj G) + b_obj^T F.
    
    gap_val = jnp.trace(A_obj @ G) + jnp.dot(b_obj, F)
    print(f"Gap Value (Matched M=1.0): {gap_val}")
    
    # Case 2: M_global = 10.0 (Mismatched)
    # The trajectory generation uses local M (calculated inside) which is 1.0.
    # But PEP uses M_global = 10.0.
    print("\n--- CASE 2: Mismatched M (Global=10.0, Local=1.0) ---")
    M_global_mismatch = 10.0
    pep_data_mismatch = construct_chambolle_pock_pep_data(0.5, 0.5, 1.0, M_global_mismatch, R, K_max)
    A_obj_m, b_obj_m = pep_data_mismatch[0], pep_data_mismatch[1]
    
    gap_val_mismatch = jnp.trace(A_obj_m @ G) + jnp.dot(b_obj_m, F)
    print(f"Gap Value (Mismatched M=10.0): {gap_val_mismatch}")
    
    # Case 3: Mismatched M but Passing M Explicitly
    print("\n--- CASE 3: Mismatched M (Global=10.0) with Explicit Pay Passing ---")
    G_fixed, F_fixed = problem_data_to_pdhg_trajectories(
        stepsizes, c, K, q, l, u, x0, y0, x_opt, y_opt, f_opt,
        K_max=K_max, m1=m1, return_Gram_representation=True, M=M_global_mismatch
    )
    
    gap_val_fixed = jnp.trace(A_obj_m @ G_fixed) + jnp.dot(b_obj_m, F_fixed)
    print(f"Gap Value (Fixed via M arg): {gap_val_fixed}")

if __name__ == "__main__":
    test_gap_consistency()
