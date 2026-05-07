"""
Monkey-patch for diffcp to fix type compatibility issues with _diffcp.M_operator.

Fixes:
1. COO -> CSC matrix format for Q
2. numpy array -> Python float for w argument

Usage:
    import diffcp_patch  # Apply patch
    # ... rest of your code
"""
import numpy as np
import diffcp._diffcp as _diffcp

# Store original M_operator
_original_M_operator = _diffcp.M_operator


def _patched_M_operator(Q, cones_parsed, u, v, w):
    """
    Patched version that:
    1. Converts Q to CSC format if needed
    2. Converts w to Python float if it's a numpy array
    """
    # Convert Q to CSC if it's not already CSC
    if hasattr(Q, 'tocsc'):
        Q = Q.tocsc()
    
    # Convert w to Python float if it's an array
    if isinstance(w, np.ndarray):
        w = float(w.item()) if w.size == 1 else float(w[0])
    elif hasattr(w, 'item'):
        w = float(w.item())
    
    return _original_M_operator(Q, cones_parsed, u, v, w)


# Apply the patch
_diffcp.M_operator = _patched_M_operator

print("[diffcp_patch] Applied COO->CSC and array->float fixes for M_operator")
