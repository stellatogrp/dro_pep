"""
Monkey-patch for diffcp to fix type compatibility issues with _diffcp.M_operator.

Fixes:
1. COO -> CSC matrix format for Q
2. numpy array -> Python float for w argument
3. Clarabel -> SCS inverse permutation of PSD blocks of (y, s) for cones of
   size n >= 4 (diffcp 1.1.9 applies the forward permutation's inverse)

Usage:
    import diffcp_patch  # Apply patch
    # ... rest of your code
"""
import os

import numpy as np
import diffcp._diffcp as _diffcp
import diffcp.cone_program as _cone_program

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


def _fixed_inverse_permute_psd_solution(y, s, n, row_offset):
    """Map a PSD block of (y, s) from Clarabel (triu) back to SCS (tril) order.

    ``permute_psd_rows`` sends SCS row k to Clarabel row P[k], with
    P = argsort(tril_multi_index). The inverse is therefore y_scs = y_clarabel[P].
    diffcp 1.1.9 indexes with argsort(triu_multi_index) = P^{-1} instead. That is
    the same map only when P is an involution, i.e. n <= 3. For n >= 4 the
    returned y, s are scrambled (KKT residuals O(1) while the objective is
    right), and everything built from them is wrong: the adjoint derivative,
    and hence every ldro-pep gradient. See tests/test_sparse_sdp_layer_grad.py.
    """
    tril_rows, tril_cols = np.tril_indices(n)
    P = np.argsort(np.ravel_multi_index((tril_cols, tril_rows), (n, n)))
    new_y = np.copy(y)
    new_s = np.copy(s)
    new_y[row_offset:row_offset + len(P)] = y[row_offset + P]
    new_s[row_offset:row_offset + len(P)] = s[row_offset + P]
    return new_y, new_s


# Apply the patch.
#
# The escape hatch exists because diffcp upstream keeps fixing this same
# territory -- notably PR #77, the Clarabel PSD-cone permutation for MULTIPLE
# cones, which is exactly our layout ('s': [S_mat] * N). Whether this patch is
# still needed, or now double-corrects, has to be answerable against any given
# diffcp build without editing the six modules that import this one:
#
#   DRO_PEP_NO_DIFFCP_PATCH=1 python run_learning_experiment.py ...
if os.environ.get('DRO_PEP_NO_DIFFCP_PATCH') == '1':
    print("[diffcp_patch] NOT applied (DRO_PEP_NO_DIFFCP_PATCH=1)")
else:
    _diffcp.M_operator = _patched_M_operator
    # solve_internal looks this name up in the module globals at call time.
    _cone_program.inverse_permute_psd_solution = _fixed_inverse_permute_psd_solution
    print("[diffcp_patch] Applied COO->CSC and array->float fixes for M_operator, "
          "and the Clarabel->SCS PSD inverse permutation fix")
