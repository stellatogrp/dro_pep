"""Base types and utilities for PEP construction."""
import jax.numpy as jnp
from typing import Tuple, List

PEPData = Tuple[
    jnp.ndarray,  # A_obj
    jnp.ndarray,  # b_obj
    jnp.ndarray,  # A_vals
    jnp.ndarray,  # b_vals
    jnp.ndarray,  # c_vals
    List[jnp.ndarray],  # PSD_A_vals
    List[jnp.ndarray],  # PSD_b_vals
    List[jnp.ndarray],  # PSD_c_vals
    List[int],  # PSD_shapes
]
