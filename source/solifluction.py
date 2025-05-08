#!/usr/bin/env python
import os
import os.path
import sys
from typing import Any

import docopt
import lue.framework as lfr
import numpy as np
from osgeo import gdal

from source.derivatives_discretization import second_derivatives_in_y
from source.heat_transfer import compute_temperature_1D_in_y
from source.io_data_process import (
    convert_numpy_to_lue,
    create_zero_numpy_array,
    default_boundary_type,
    read_run_setup,
)
from source.layer import Layer
from source.momentum import momentum_ux
from source.vof import h_mesh_assign

# from source.boundary_condition import boundary_set


# from input_output import write


@lfr.runtime_scope  # type: ignore[misc]
def solifluction_simulate(
    dx: float,
    dz: float,
    num_cols: int,
    num_rows: int,
    num_layers: int,
    initial_layer_variables: Layer,
    h_total_initial: Any,
    dt_momentum: float,
    dt_heat_transfer: float,
    time_end_simulation: float,
    heat_transfer_warmup: bool,
    heat_transfer_warmup_iteration: int,
    T_bed: Any,
    T_surface: Any,
    d2u_x_dy2_initial: Any,
    h_mesh_step_value: float,
    nu_x: float,
    nu_z: float,
    partition_shape: tuple[int, int],
    results_pathname: str,
) -> None:

    array_shape: tuple[int, int] = (num_rows, num_cols)
    zero_lue: Any = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=0.0,
        partition_shape=partition_shape,
    )

    layer_list: list[Layer] = []

    # Assign bed layer properties
    layer_list.append(initial_layer_variables)

    # Assign internal layers properties
    for _ in range(1, num_layers):
        layer_list.append(initial_layer_variables)

    # Assign surface layer properties
    layer_list.append(initial_layer_variables)

    # ---------------- boundary condition set --------------------------

    # Currently, homogeneous Neumann boundary conditions (∂(.)/∂n = 0) are applied to
    # both u and h.
    # This corresponds to assuming zero normal gradients at the boundaries
    # (e.g., fully developed flow at the outlet).
    # These conditions can be modified if necessary.

    boundary_type_numpy_default, boundary_loc_numpy_default = default_boundary_type(
        num_cols, num_rows, boundary_in_first_last_row_col=False
    )

    Dirichlet_boundary_value_numpy = create_zero_numpy_array(
        num_cols, num_rows, 0, np.float64
    )
    Neumann_boundary_value_numpy = create_zero_numpy_array(
        num_cols, num_rows, 0, np.float64
    )

    Dirichlet_boundary_value_numpy[[0, -1], :] = 0  # -999
    Dirichlet_boundary_value_numpy[:, [0, -1]] = 0  # -999

    boundary_loc = convert_numpy_to_lue(
        boundary_loc_numpy_default, partition_shape=partition_shape
    )

    boundary_type = convert_numpy_to_lue(
        boundary_type_numpy_default, partition_shape=partition_shape
    )

    Dirichlet_boundary_value = convert_numpy_to_lue(
        Dirichlet_boundary_value_numpy, partition_shape=partition_shape
    )

    Neumann_boundary_value = convert_numpy_to_lue(
        Neumann_boundary_value_numpy, partition_shape=partition_shape
    )

    # ---------------- End: boundary condition set --------------------------

    h_mesh_assign(h_total_initial, num_layers, h_mesh_step_value, layer_list)

    if heat_transfer_warmup:

        for _ in range(1, heat_transfer_warmup_iteration):

            layer_list[0].T = T_bed
            layer_list[num_layers - 1].T = T_surface

            # T_numpy_bed = lfr.to_numpy(layer_list[0].T)
            # T_numpy_surf = lfr.to_numpy(layer_list[num_layers - 1].T)

            for layer_id in range(1, num_layers - 1):
                layer_list[layer_id].T = compute_temperature_1D_in_y(
                    layer_list[layer_id].k_conductivity_heat,
                    layer_list[layer_id + 1].k_conductivity_heat,
                    layer_list[layer_id - 1].k_conductivity_heat,
                    layer_list[layer_id].rho_c_heat,
                    layer_list[layer_id].T,
                    layer_list[layer_id + 1].T,
                    layer_list[layer_id - 1].T,
                    dt_heat_transfer,
                    layer_list[layer_id].h_mesh,
                    layer_list[layer_id - 1].h_mesh,
                    compute_flag,
                    precomputed_value,
                )

    time = 0
    local_momentum_time = 0
    local_mass_conservation_time = 0
    local_heat_transfer_time = 0

    d2u_x_dy2 = d2u_x_dy2_initial

    dt_mass_conservation: float = dt_momentum

    dt_min = min(dt_momentum, dt_heat_transfer)

    while time < time_end_simulation:

        time = time + dt_min

        if local_momentum_time >= dt_momentum:

            for layer_id in range(1, num_layers):

                rhs = g_sin + (
                    (layer_list[layer_id].mu_soil / layer_list[layer_id].density_soil)
                    * d2u_x_dy2[layer_id]
                )

                layer_list[layer_id].u_x = momentum_ux(
                    layer_list[layer_id].u_x,
                    phase_state_lue,
                    dx,
                    dz,
                    dt,
                    layer_list[layer_id].u_x,
                    zero_array_lue,
                    0.0,
                    0.0,
                    rhs,
                    h_mesh,
                    boundary_loc,
                    boundary_type,
                    Dirichlet_boundary_value_lue,
                    Neumann_boundary_value_lue,
                )

            for layer_id in range(0, num_layers):
                if layer_id == 0:  # bed layer
                    d2u_x_dy2[0] = second_derivatives_in_y(
                        Layer_list[1].u_x,
                        Layer_list[2].u_x,
                        Layer_list[0].u_x,
                        h_mesh,
                        h_mesh,
                    )

                elif layer_id == num_layers - 1:  # surface layer
                    d2u_x_dy2[-1] = second_derivatives_in_y(
                        Layer_list[num_layers - 2].u_x,
                        Layer_list[num_layers - 1].u_x,
                        Layer_list[num_layers - 3].u_x,
                        h_mesh,
                        h_mesh,
                    )

                else:  # internal layers
                    d2u_x_dy2[layer_id] = second_derivatives_in_y(
                        Layer_list[layer_id].u_x,
                        Layer_list[layer_id + 1].u_x,
                        Layer_list[layer_id - 1].u_x,
                        h_mesh,
                        h_mesh,
                    )

            local_momentum_time = 0

        if local_mass_conservation_time >= dt_mass_conservation:

            layer_list[layer_id].h_mesh = mass_conservation_2D_vof(
                layer_list[layer_id].h_mesh,
                layer_list[layer_id].u_x,
                layer_list[layer_id].u_z,
                dt,
                dx,
                dz,
                boundary_loc,
                boundary_type,
                Dirichlet_boundary_value,
                Neumann_boundary_value,
            )

            local_mass_conservation_time = 0

        # compute temperatures in internal layers

        if local_heat_transfer_time >= dt_heat_transfer:

            layer_list[0].T = T_bed_lue
            layer_list[num_layers - 1].T = T_surface_lue

            T_result[0] = T_bed_value
            T_result[num_layers - 1] = T_surface_value

            T_numpy_bed = lfr.to_numpy(Layer_list[0].T)
            T_numpy_surf = lfr.to_numpy(Layer_list[num_layers - 1].T)
            print("T_numpy_bed: \n", T_numpy_bed)
            print("T_numpy_surf: \n", T_numpy_surf)

            for layer_id in range(1, num_layers - 1):
                layer_list[layer_id].T = compute_temperature_1D_in_y(
                    layer_list[layer_id].k_conductivity_heat,
                    layer_list[layer_id + 1].k_conductivity_heat,
                    layer_list[layer_id - 1].k_conductivity_heat,
                    layer_list[layer_id].rho_c_heat,
                    layer_list[layer_id].T,
                    layer_list[layer_id + 1].T,
                    layer_list[layer_id - 1].T,
                    dt,
                    layer_list[layer_id].h_mesh,
                    layer_list[layer_id - 1].h_mesh,
                    compute_flag,
                    precomputed_value,
                )

            local_heat_transfer_time = 0


usage = f"""\
Solifluction simulation

Usage:
    {os.path.basename(sys.argv[0])} <run_setup_file>

Options:
    <run_setup_file>         simulation setup file which defines the input variables
"""


def main():

    argv = list(sys.argv[1:])
    arguments = docopt.docopt(usage, argv)

    run_setup_file = arguments["<run_setup_file>"]
    assert not os.path.splitext(run_setup_file)[1]

    # rho_s = 2650
    # porosity = 0.4

    input_variables = read_run_setup("run_setup_file")

    dt_momentum: float = input_variables["time_step_momentum"]
    dt_heat_transfer: float = input_variables["time_step_heat_transfer"]
    dt_mass_conservation: float = input_variables["time_step_mass_conservation"]
    partition_shape_size: int = input_variables["partition_shape_size"]
    h_mesh_step_value: float = input_variables["initial_layer_size"]
    mu_value: float = input_variables["uniform_mu"]
    density_value: float = input_variables["uniform_density"]
    k_conductivity_value: float = input_variables["uniform_k_conductivity"]
    rho_c_heat_value: float = input_variables["uniform_rho_c_heat"]
    h_total_initial_file: str = input_variables["h_total_initial_file"]
    dt_momentum: float = input_variables["dt_momentum"]
    dt_heat_transfer: float = input_variables["dt_heat_transfer"]
    time_end_simulation: float = input_variables["time_end_simulation"]

    heat_transfer_warmup: bool = input_variables.get("heat_transfer_warmup", False)
    heat_transfer_warmup_iteration: int = input_variables.get(
        "heat_transfer_warmup_iteration", 200
    )

    # ---------------------  initial information --------------------

    dataset = gdal.Open(h_total_initial_file)
    num_cols = dataset.RasterXSize
    num_rows = dataset.RasterYSize

    print(f"num_cols: {num_cols}")
    print(f"num_rows: {num_rows}")

    geotransform = dataset.GetGeoTransform()

    dx = geotransform[1]  # Pixel width (Δx)
    dy = abs(
        geotransform[5]
    )  # Pixel height (Δy) — take abs because it may be negative (north-up images)

    print(f"Pixel size dx: {dx}")
    print(f"Pixel size dy: {dy}")

    raster_array = dataset.ReadAsArray()

    max_h_total = np.max(raster_array)
    print(f"max_h_total: {max_h_total}")

    array_shape = (
        num_rows,
        num_cols,
    )
    partition_shape: tuple[int, int] = 2 * (partition_shape_size,)

    bed_depth_elevation = 0  # it can be any value

    num_layers: int = int(
        np.round(max_h_total - bed_depth_elevation) / h_mesh_step_value + 1
    )

    h_total_initial = lfr.from_gdal(
        h_total_initial_file, partition_shape=partition_shape
    )

    zero_array_lue = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=0.0,
        partition_shape=partition_shape,
    )

    mu_array_lue = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=mu_value,
        partition_shape=partition_shape,
    )

    density_array_lue = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=density_value,
        partition_shape=partition_shape,
    )

    k_conductivity_array_lue = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=k_conductivity_value,
        partition_shape=partition_shape,
    )

    rho_c_heat_array_lue = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=rho_c_heat_value,
        partition_shape=partition_shape,
    )

    T_bed = zero_array_lue
    # T_surface =

    initial_u_x = zero_array_lue
    initial_u_z = zero_array_lue
    initial_temperature = zero_array_lue
    initial_h_mesh = zero_array_lue  # this will be updated by "h_mesh_assign" function
    initial_mu_soil = mu_array_lue
    initial_density_soil = density_array_lue
    initial_phase_state = zero_array_lue
    initial_k_conductivity_heat = k_conductivity_array_lue
    initial_rho_c_heat = rho_c_heat_array_lue
    initial_vegetation_vol_fraction = zero_array_lue

    initial_layer_variables: Layer = Layer(
        initial_u_x,
        initial_u_z,
        initial_temperature,
        initial_h_mesh,
        initial_mu_soil,
        initial_density_soil,
        initial_phase_state,
        initial_k_conductivity_heat,
        initial_rho_c_heat,
        initial_vegetation_vol_fraction,
    )

    d2u_x_dy2_initial = []

    for _ in range(num_layers):
        d2u_x_dy2_initial.append(zero_array_lue)

    # ---------------------  initial information --------------------

    solifluction_simulate(
        dx,
        dy,
        num_cols,
        num_rows,
        num_layers,
        initial_layer_variables,
        h_total_initial,
        dt_momentum,
        dt_heat_transfer,
        time_end_simulation,
        heat_transfer_warmup,
        heat_transfer_warmup_iteration,
        T_bed,
        T_surface,
        d2u_x_dy2_initial,
        dt,
        T_surf_file,
        T_initial_file,
        h_total_file,
        mu_soil_initial_file,
        mu_soil_surf_file,
        U_x_surf_file,
        U_x_initial_file,
        gama_soil_surf_file,
        gama_soil_initial_file,
        phase_state_surf_file,
        phase_state_initial_file,
        thermal_diffusivity_coeff_surf_file,
        vegetation_vol_fraction_surf_file,
        vegetation_vol_fraction_initial_file,
        nu_x,
        nu_y,
        nr_time_steps,
        partition_shape,
        results_pathname,
    )


if __name__ == "__main__":
    main()
