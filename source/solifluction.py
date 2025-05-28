#!/usr/bin/env python
# import os
# import os.path
# import sys
from typing import Any

# import docopt
import lue.framework as lfr
import numpy as np

from source.derivatives_discretization import second_derivatives_in_y
from source.heat_transfer import compute_temperature_1D_in_y
from source.interpolation import interpolate_temperature
from source.io_data_process import (
    convert_numpy_to_lue,
    create_zero_numpy_array,
    default_boundary_type,
    initiate_layers_variables,
    write_tif_file,
)
from source.layer import Layer
from source.momentum import momentum_ux
from source.phase_detect import phase_detect_from_temperature
from source.vof import calculate_total_h, h_mesh_assign, mass_conservation_2D_vof

# from source.boundary_condition import boundary_set


# from input_output import write


@lfr.runtime_scope  # type: ignore[misc]
def solifluction_simulate(
    dx: float,
    dz: float,
    num_cols: int,
    num_rows: int,
    max_h_total: float,
    bed_depth_elevation: float,
    h_total_initial_file: str,
    mu_value: float,
    density_value: float,
    k_conductivity_value: float,
    rho_c_heat_value: float,
    dt_momentum: float,
    dt_mass_conservation: float,
    dt_heat_transfer: float,
    time_end_simulation: float,
    heat_transfer_warmup: bool,
    heat_transfer_warmup_iteration: int,
    h_mesh_step_value: float,
    g_sin: float,
    nu_x: float,
    nu_z: float,
    days_temperature_file: list[float],
    temps_temperature_file: list[float],
    partition_shape: tuple[int, int],
    results_pathname: str,
) -> None:

    # ----------------- initial layer information  -------------

    array_shape: tuple[int, int] = (num_rows, num_cols)

    (
        initial_layer_variables,
        d2u_x_dy2_initial,
        h_total_initial,
        num_layers,
        temperature_bed,
    ) = initiate_layers_variables(
        max_h_total,
        bed_depth_elevation,
        h_mesh_step_value,
        array_shape,
        partition_shape,
        h_total_initial_file,
        mu_value,
        density_value,
        k_conductivity_value,
        rho_c_heat_value,
    )

    # ----------------- initial layer information  -------------

    print("start to run solifluction_simulate")

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
    # both u and h. (It is the default boundary condition)
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

    Dirichlet_boundary_value_numpy[[0, -1], :] = 0  # -999    # These are virtual layers
    # NOT boundaries (boundaries are imposed on layer inner)
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

            surface_temperature = temps_temperature_file[0]

            layer_list[0].T = temperature_bed
            layer_list[num_layers - 1].T = surface_temperature

            for layer_id in range(1, num_layers - 1):

                compute_flag_temperature = lfr.where(
                    layer_list[layer_id].h_mesh > 0, 1, 0
                )

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
                    compute_flag_temperature,
                    surface_temperature,
                )

    time: float = 0
    local_momentum_time: float = 0
    local_mass_conservation_time: float = 0
    local_heat_transfer_time: float = 0

    iteration_write_h_total: int = 0

    d2u_x_dy2 = d2u_x_dy2_initial

    dt_min: float = min(dt_momentum, dt_heat_transfer)

    while time < time_end_simulation:

        time = time + dt_min

        print("time: ", time)

        if local_momentum_time >= dt_momentum:

            print("momentum_ux in run")

            for layer_id in range(1, num_layers):

                rhs = g_sin + (
                    (layer_list[layer_id].mu_soil / layer_list[layer_id].density_soil)
                    * d2u_x_dy2[layer_id]
                )

                phase_state = phase_detect_from_temperature(layer_list[layer_id].T)

                layer_list[layer_id].u_x = momentum_ux(
                    layer_list[layer_id].u_x,
                    phase_state,
                    dx,
                    dz,
                    dt_momentum,
                    layer_list[layer_id].u_x,
                    layer_list[layer_id].u_z,
                    nu_x,
                    nu_z,
                    rhs,
                    layer_list[layer_id].h_mesh,
                    boundary_loc,
                    boundary_type,
                    Dirichlet_boundary_value,
                    Neumann_boundary_value,
                )

            for layer_id in range(0, num_layers):
                if layer_id == 0:  # bed layer
                    d2u_x_dy2[0] = second_derivatives_in_y(
                        layer_list[1].u_x,
                        layer_list[2].u_x,
                        layer_list[0].u_x,
                        layer_list[1].h_mesh,
                        layer_list[0].h_mesh,
                    )

                elif layer_id == num_layers - 1:  # surface layer
                    d2u_x_dy2[-1] = second_derivatives_in_y(
                        layer_list[num_layers - 2].u_x,
                        layer_list[num_layers - 1].u_x,
                        layer_list[num_layers - 3].u_x,
                        layer_list[num_layers - 1].h_mesh,
                        layer_list[num_layers - 2].h_mesh,
                    )

                else:  # internal layers
                    d2u_x_dy2[layer_id] = second_derivatives_in_y(
                        layer_list[layer_id].u_x,
                        layer_list[layer_id + 1].u_x,
                        layer_list[layer_id - 1].u_x,
                        layer_list[layer_id].h_mesh,
                        layer_list[layer_id - 1].h_mesh,
                    )

            local_momentum_time = 0

        if local_mass_conservation_time >= dt_mass_conservation:

            print("mass_conservation_2D_vof in run")

            layer_list[layer_id].h_mesh = mass_conservation_2D_vof(
                layer_list[layer_id].h_mesh,
                layer_list[layer_id].u_x,
                layer_list[layer_id].u_z,
                dt_mass_conservation,
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

            print("compute_temperature_1D_in_y in run")

            surface_temperature = interpolate_temperature(
                time, days_temperature_file, temps_temperature_file
            )

            layer_list[0].T = temperature_bed
            layer_list[num_layers - 1].T = surface_temperature

            for layer_id in range(1, num_layers - 1):

                compute_flag_temperature = lfr.where(
                    layer_list[layer_id].h_mesh > 0, 1, 0
                )

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
                    compute_flag_temperature,
                    surface_temperature,
                )

            local_heat_transfer_time = 0

        h_total_simulated_lue = calculate_total_h(layer_list)

        write_tif_file(
            h_total_simulated_lue, "h_total", iteration_write_h_total, results_pathname
        )

        iteration_write_h_total = iteration_write_h_total + 1

        print("simulation iteration_write_h_total: ", iteration_write_h_total)
