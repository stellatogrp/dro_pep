"""Training result data structures."""

from dataclasses import dataclass
from typing import List
import jax.numpy as jnp

# Type alias for stepsizes (e.g., (t,) for GD, (t, beta) for FGM)
Stepsizes = tuple[jnp.ndarray, ...]


@dataclass
class TrainingResult:
    """Result from a training run.

    Attributes:
        stepsizes: Final learned stepsizes after training
        stepsizes_history: Full history of stepsizes including initialization
        losses: Training loss value at each iteration
        times: Wall-clock time in seconds for each iteration
    """
    stepsizes: Stepsizes
    stepsizes_history: List[Stepsizes]
    losses: List[float]
    times: List[float]
