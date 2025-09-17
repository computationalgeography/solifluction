# Import main modules so they are available at package level
from . import (
    boundary_condition,
    config,
    derivatives_discretization,
    heat_transfer,
    interpolation,
    io_data_process,
    layer,
    mass_conservation,
    momentum,
    phase_detect,
    solifluction,
    viscosity_calc,
    vof,
)
from .version import __version__

# from .derivatives_discretization import second_derivatives_in_y


# This is optional but __all__ provides an access for "from soli3d import *"
__all__ = [
    "__version__",
    "boundary_condition",
    "config",
    "derivatives_discretization",
    "heat_transfer",
    "interpolation",
    "io_data_process",
    "layer",
    "mass_conservation",
    "momentum",
    "phase_detect",
    "solifluction",
    "viscosity_calc",
    "vof",
]
