"""PEP construction functions for various algorithms."""

from .gd_fgm import (
    construct_gd_pep_data,
    construct_fgm_pep_data,
    pep_data_to_numpy,
)
from .ista_fista import (
    construct_ista_pep_data,
    construct_fista_pep_data,
    ista_pep_data_to_numpy,
)
from .chambolle_pock import (
    construct_chambolle_pock_pep_data,
    chambolle_pock_pep_data_to_numpy,
)
from .interpolation_conditions import (
    smooth_strongly_convex_interp,
    smooth_strongly_convex_interp_consecutive,
    convex_interp,
    smooth_strongly_convex_proximal_gradient_interp,
)
from .loss_compositions import compose_objective, compose_final, compose_weighted
from .base import PEPData
