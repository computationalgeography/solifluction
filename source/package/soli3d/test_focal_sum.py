#!/usr/bin/env python
# import os
# import os.path
# import sys
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# import docopt
import lue.framework as lfr
import numpy as np

# from pympler import asizeof
# from .derivatives_discretization import dx_upwind, second_derivatives_in_y
# from .heat_transfer import compute_temperature_1D_in_y
# from .interpolation import interpolate_temperature
from .io_data_process import read_config_file, read_tif_info_from_gdal
from .kernels import kernel_im1_j

# from .layer import Layer
# from .momentum import momentum_ux  # momentum_ux_steady_state
# from .phase_detect import phase_detect_from_temperature
# from .viscosity_calc import viscosity_exp_temp
# from .vof import calculate_total_h, h_mesh_assign, mass_conservation_2D_vof

# from source.boundary_condition import boundary_set


# from input_output import write

Shape = tuple[int, int]


def initialize_focal_sum(
    array_shape: Shape,
    partition_shape: Shape,
) -> Any:  # raster,

    initial_raster = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=5.0,
        partition_shape=partition_shape,
    )

    # ----------------- initial layer information  -------------

    num_rows, num_cols = array_shape

    print("start to run solifluction_simulate")

    # initial_u_x_lue = lfr.create_array(
    #     array_shape,
    #     dtype=np.float64,
    #     fill_value=1e-10,
    #     partition_shape=partition_shape,
    # )

    return initial_raster


def simulate_focal_sum(
    array_shape: Shape,
    partition_shape: Shape,
    initial_raster: Any,
    dt_global_model: float,
    dx: float,
    inner_iteration_threshold: int,
    model_total_iteration: int,
) -> tuple[Any, int]:

    time: float = 0

    time = time + dt_global_model

    local_inner_iteration: int = 0

    raster = initial_raster

    while local_inner_iteration < inner_iteration_threshold:

        raster = (raster - lfr.focal_sum(raster, kernel_im1_j)) / dx

        local_inner_iteration = local_inner_iteration + 1

        model_total_iteration = model_total_iteration + 1

    return raster, model_total_iteration


class Focal(lfr.Model):

    def __init__(self, array_shape: Shape, partition_shape: Shape, results_path: Path):
        super().__init__()
        self.array_shape = array_shape
        self.partition_shape = partition_shape
        self.results_path = results_path

    # def save_generation(self, generation: Generation, generation_id: int) -> None:

    #     lfr.to_gdal(generation, f"{self.generation_path}-{generation_id}.tif")

    def initialize(
        self,
    ) -> None:

        initial_raster = initialize_focal_sum(
            self.array_shape,
            self.partition_shape,
        )

        self.raster = initial_raster

    def simulate(self, iteration: int) -> Any:
        # self.generation = next_generation(self.generation)
        # self.save_generation(self.generation, iteration)

        print("iteration inside simulate -------", iteration)

        self.raster, self.model_total_iteration = simulate_focal_sum(
            self.array_shape,
            self.partition_shape,
            self.raster,
            self.dt_global_model,
            self.dx,
            self.inner_iteration_threshold,
            self.model_total_iteration,
        )

        return self.raster.future()


@lfr.runtime_scope
def focal_sum(
    *,
    array_shape: Shape,
    partition_shape: Shape,
    number_of_iterations: int,
    inner_iteration_threshold: int,
    results_pathname: Path,
    dx: float,
    dt_global_model: float,
    write_intervals_time: float,
) -> None:

    model = Focal(
        array_shape=array_shape,
        partition_shape=partition_shape,
        results_path=results_pathname,
    )

    model.dt_global_model = dt_global_model
    model.write_intervals_time = write_intervals_time
    model.dx = dx

    model_total_iteration_initial: int = 0
    model.model_total_iteration = model_total_iteration_initial
    model.u_x_tem_time = []
    model.number_of_iterations = number_of_iterations
    model.inner_iteration_threshold = inner_iteration_threshold

    lfr.run_deterministic(
        model,
        lfr.DefaultProgressor(),
        nr_time_steps=number_of_iterations,
        rate_limit=5,
    )


def main() -> None:

    # if len(sys.argv) > 1:
    #     param_path = Path(sys.argv[1]).resolve()
    # else:
    #     param_path = Path("param.txt").resolve()

    # if not param_path.is_file():
    #     print(f"Parameter file does not exist: {param_path}")
    #     sys.exit(1)

    # Find the first argument that looks like a file (not starting with "--")
    param_path = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            param_path = Path(arg).resolve()
            break

    if param_path is None:
        param_path = Path("param.txt").resolve()

    if not param_path.is_file():
        print(f"Parameter file does not exist: {param_path}")
        sys.exit(1)

    print("sys.argv =", sys.argv)

    # -----  read input variables from param.txx ---------------

    (
        number_of_iterations,
        dt_momentum,
        momentum_iteration_threshold,
        dt_global_model,
        dt_heat_transfer,
        dt_mass_conservation,
        write_intervals_time,
        partition_shape_size,
        h_mesh_step_value,
        mu_value,
        density_value,
        k_conductivity_value,
        rho_c_heat_value,
        h_total_initial_file_name,
        heat_transfer_warmup,
        heat_transfer_warmup_iteration,
        days_temperature_file,
        temps_temperature_file,
        results_pathname,
        slope_radian,
        permafrost,
    ) = read_config_file(param_path)

    # ---------------------  initial information --------------------

    dx, dz, array_shape, max_h_total = read_tif_info_from_gdal(
        h_total_initial_file_name
    )
    # num_rows, num_cols = array_shape

    partition_shape: tuple[int, int] = 2 * (partition_shape_size,)

    # ---------------------  initial information --------------------

    focal_sum(
        array_shape=array_shape,
        partition_shape=partition_shape,
        number_of_iterations=number_of_iterations,
        inner_iteration_threshold=momentum_iteration_threshold,
        results_pathname=results_pathname,
        dx=dx,
        dt_global_model=dt_global_model,
        write_intervals_time=write_intervals_time,
    )
