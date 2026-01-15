"""
Silver stepsize schedules for gradient descent.

Based on the Silver Stepsize Schedule paper, which provides optimal step sizes
for both strongly convex and non-strongly convex functions.
"""
import numpy as np


def compute_shifted_2adics(K):
    """
    Compute the shifted 2-adic valuations for indices 1 to K.
    
    The 2-adic valuation v_2(k) of an integer k is the largest power of 2 that divides k.
    This function returns v_2(k) for k = 1, ..., K.
    """
    two_adics = np.zeros(K, dtype=int)
    for k in range(1, K + 1):
        # (k & -k) gives the lowest set bit, bit_length()-1 gives its position
        two_adics[k - 1] = (k & -k).bit_length() - 1
    return two_adics


def get_nonstrongly_convex_silver_stepsizes(K, L=1):
    """
    Get Silver step sizes for non-strongly convex functions (mu=0).
    
    Args:
        K: Number of iterations
        L: Lipschitz constant of the gradient
        
    Returns:
        List of step sizes of length K
    """
    rho = 1 + np.sqrt(2)
    silver_steps = [1 + rho ** ((k & -k).bit_length() - 2) for k in range(1, K + 1)]
    return [alpha / L for alpha in silver_steps]


def compute_silver_idx(kappa, K):
    """
    Compute the index values for Silver step sizes.
    
    These indices determine which y or z value to use for each step.
    """
    two_adics = compute_shifted_2adics(K)
    idx_vals = np.power(2, two_adics)
    last_pow2 = int(np.floor(np.log2(K)))
    idx_vals[(2 ** last_pow2) - 1] /= 2
    return idx_vals


def generate_yz_sequences(kappa, t):
    """
    Generate the y and z sequences for strongly convex Silver step sizes.
    
    Args:
        kappa: Condition number L/mu
        t: log_2(K) where K is the number of iterations
        
    Returns:
        y_vals: Dictionary mapping powers of 2 to y values
        z_vals: Dictionary mapping powers of 2 to z values
    """
    y_vals = {1: 1 / kappa}
    z_vals = {1: 1 / kappa}

    # Generate z sequences
    for i in range(1, t + 1):
        K = 2 ** i
        z_ihalf = z_vals[int(K / 2)]
        xi = 1 - z_ihalf
        z_i = z_ihalf * (xi + np.sqrt(1 + xi ** 2))
        z_vals[K] = z_i

    # Generate y sequences
    for i in range(1, t + 1):
        K = 2 ** i
        zK = z_vals[K]
        zKhalf = z_vals[int(K // 2)]
        yK = zK - 2 * (zKhalf - zKhalf ** 2)
        y_vals[K] = yK
        
    return y_vals, z_vals


def get_strongly_convex_silver_stepsizes(K, mu=0, L=1):
    """
    Get Silver step sizes for strongly convex functions.
    
    Args:
        K: Number of iterations (must be a power of 2 for mu > 0)
        mu: Strong convexity parameter
        L: Lipschitz constant of the gradient
        
    Returns:
        Array of step sizes of length K
    """
    if mu == 0:
        return get_nonstrongly_convex_silver_stepsizes(K, L=L)
    
    kappa = L / mu
    # Assume K is a power of 2
    idx_vals = compute_silver_idx(kappa, K)
    y_vals, z_vals = generate_yz_sequences(kappa, int(np.log2(K)))

    def psi(t):
        return (1 + kappa * t) / (1 + t)

    silver_steps = []
    for i in range(idx_vals.shape[0] - 1):
        idx = int(idx_vals[i])
        silver_steps.append(psi(y_vals[idx]))
    silver_steps.append(psi(z_vals[int(idx_vals[-1])]))

    return np.array(silver_steps) / L