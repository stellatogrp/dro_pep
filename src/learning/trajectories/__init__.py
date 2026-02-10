"""Trajectory computation functions for various optimization algorithms."""

from .gd_fgm import (
    compute_preconditioner_from_samples,
    problem_data_to_gd_trajectories,
    problem_data_to_nesterov_fgm_trajectories,
    dro_pep_obj_jax,
    problem_data_to_pep_obj,
)
from .ista_fista import (
    soft_threshold_jax,
    problem_data_to_ista_trajectories,
    problem_data_to_fista_trajectories,
)
from .logreg_gd_fgm import (
    logreg_f_shifted,
    logreg_grad_shifted,
    logreg_gd_trajectories,
    logreg_fgm_trajectories,
    logreg_pep_obj,
    create_logreg_traj_fn_gd,
    create_logreg_traj_fn_fgm,
)
from .pdhg import (
    proj_box,
    proj_nonneg_first_m1,
    problem_data_to_pdhg_trajectories,
)
