#!/usr/bin/env python
# import os
# import os.path
# import sys
from typing import Any

# import docopt
import lue.framework as lfr
import numpy as np

from source.derivatives_discretization import dx_upwind, second_derivatives_in_y
from source.heat_transfer import compute_temperature_1D_in_y
from source.interpolation import interpolate_temperature
from source.io_data_process import (
    convert_numpy_to_lue,
    create_zero_numpy_array,
    default_boundary_type,
    initiate_layers_variables,
    write_tif_file,
)
from source.momentum import momentum_ux
from source.phase_detect import phase_detect_from_temperature
from source.vof import calculate_total_h, h_mesh_assign, mass_conservation_2D_vof

# from source.boundary_condition import boundary_set


# from input_output import write


def is_phi_steady(phi_previous: Any, phi_current: Any, tol: float = 1e-4) -> bool:

    relative_error = lfr.sum(lfr.abs(phi_current - phi_previous)) / lfr.sum(
        lfr.abs(phi_previous)
    )
    if relative_error < tol:
        return True
    return False


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
    momentum_iteration_threshold: int,
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

    input("enter to continue ...")

    print("size of domain (num_rows, num_cols): ", array_shape)

    (
        layer_list,
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

    phase_state_initial = lfr.create_array(
        array_shape,
        dtype=np.uint8,
        fill_value=1,
        partition_shape=partition_shape,
    )

    # ----------------- initial layer information  -------------

    print("start to run solifluction_simulate")

    print(
        "type of initial_layer_variables[2].h_mesh:",
        type(layer_list[2].h_mesh),
    )

    print(
        "type of initial_layer_variables[2].mu_soil:",
        type(layer_list[2].mu_soil),
    )

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

    # print(
    #     " Before function h_mesh_assign type of h_total_initial:",
    #     type(h_total_initial),
    # )

    h_mesh_list = h_mesh_assign(
        h_total_initial, num_layers, np.float64(h_mesh_step_value)
    )

    h_total = h_total_initial

    for i in range(num_layers):
        layer_list[i].h_mesh = h_mesh_list[i]

        # h_mesh_list_numpy = lfr.to_numpy(h_mesh_list[i])
        # print(
        #     "h_mesh_list_numpy[50,50]",
        #     h_mesh_list_numpy[50, 50],
        #     "layer_id",
        #     i,
        # )

        # h_mesh_numpy_initial = lfr.to_numpy(layer_list[i].h_mesh)
        # print(
        #     "h_mesh_numpy_initial layer_list[i].h_mesh[50,50]",
        #     h_mesh_numpy_initial[50, 50],
        #     "layer_id",
        #     i,
        # )

    # h_mesh_assign_0(h_total_initial, num_layers, 4.5, layer_list)

    # for i in range(num_layers):

    #     h_mesh_numpy_before_heat_transfer_warmup = lfr.to_numpy(layer_list[i].h_mesh)
    #     print(
    #         "h_mesh_numpy_before_heat_transfer_warmup.h_mesh[10,10]",
    #         h_mesh_numpy_before_heat_transfer_warmup[10, 10],
    #         "layer_id",
    #         i,
    #     )

    # for i in range(num_layers):

    #     h_mesh_list_numpy_before_heat_transfer_warmup = lfr.to_numpy(h_mesh_list[i])
    #     print(
    #         "h_mesh_list_numpy_before_heat_transfer_warmup.h_mesh[50,50]",
    #         h_mesh_list_numpy_before_heat_transfer_warmup[50, 50],
    #         "layer_id",
    #         i,
    #     )

    if heat_transfer_warmup:

        for _ in range(1, heat_transfer_warmup_iteration):

            surface_temperature = temps_temperature_file[0]

            surface_temperature_lue = lfr.create_array(
                array_shape,
                dtype=np.float64,
                fill_value=surface_temperature,
                partition_shape=partition_shape,
            )

            layer_list[0].T = temperature_bed
            layer_list[num_layers - 1].T = surface_temperature_lue

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
                    surface_temperature,
                )

    time: float = 0
    local_momentum_time: float = 0
    local_mass_conservation_time: float = 0
    local_heat_transfer_time: float = 0

    iteration_write_h_total: int = 0

    d2u_x_dy2 = d2u_x_dy2_initial

    # dt_min: float = min(dt_momentum, dt_heat_transfer)
    # dt_max: float = max(dt_momentum, dt_heat_transfer, dt_mass_conservation)
    dt_max = dt_mass_conservation  # dt_mass_conservation is considered as maximum time step for simulation

    # for i in range(num_layers):

    #     h_mesh_numpy_before_loop = lfr.to_numpy(layer_list[i].h_mesh)
    #     print(
    #         "before loop layer_list[i].h_mesh[50,50]",
    #         h_mesh_numpy_before_loop[50, 50],
    #         "layer_id",
    #         i,
    #     )

    # layer_u_x_0_numpy = lfr.to_numpy(layer_list[0].u_x)
    # print(
    #     "layer_u_x_0_numpy initial [50,50]",
    #     layer_u_x_0_numpy[50, 50],
    #     "layer_id",
    #     0,
    # )

    heat_transfer_iteration_threshold: int = momentum_iteration_threshold

    while time < time_end_simulation:

        time = time + dt_max

        local_momentum_iteration: int = 0
        local_heat_transfer_iteration: int = 0

        # --------------- compute temperatures in internal layers -------------------------------

        while (
            local_heat_transfer_iteration < heat_transfer_iteration_threshold
        ):  # if abs(time - local_heat_transfer_time) >= dt_heat_transfer:

            print("compute_temperature_1D_in_y in run")

            surface_temperature = interpolate_temperature(
                time, days_temperature_file, temps_temperature_file
            )

            surface_temperature_lue = lfr.create_array(
                array_shape,
                dtype=np.float64,
                fill_value=surface_temperature,
                partition_shape=partition_shape,
            )

            layer_list[0].T = temperature_bed
            layer_list[num_layers - 1].T = surface_temperature_lue

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
                    surface_temperature,
                )

                # print(
                #     "type of layer_list[layer_id].T before to numpy :",
                #     type(layer_list[layer_id].T),
                # )

            for layer_id in range(0, num_layers):

                layer_T_numpy = lfr.to_numpy(layer_list[layer_id].T)
                print(
                    "layer_T_numpy[50,50]",
                    layer_T_numpy[50, 50],
                    "layer_id",
                    layer_id,
                )

            # local_heat_transfer_time = local_heat_transfer_time + dt_heat_transfer

            local_heat_transfer_iteration = local_heat_transfer_iteration + 1

            print("local_heat_transfer_iteration :", local_heat_transfer_iteration)

        # --------------- End: compute temperatures in internal layers -----------------

        # --------------- compute momentum u_x in internal layers ----------------------

        for local_momentum_iteration in range(1, momentum_iteration_threshold + 1):
            # while (
            #     local_momentum_iteration < momentum_iteration_threshold
            # ):  # if abs(time - local_momentum_time) >= dt_momentum:

            print("momentum_ux in run")

            for layer_id in range(1, num_layers):

                gama_prim_surface: float = (2610 - 1000) * 9.81

                rhs = (
                    g_sin
                    + (
                        (
                            layer_list[layer_id].mu_soil
                            / layer_list[layer_id].density_soil
                        )
                        * d2u_x_dy2[layer_id]
                    )
                    - (
                        (gama_prim_surface / layer_list[layer_id].density_soil)
                        * dx_upwind(h_total, dx, layer_list[layer_id].u_x)
                    )
                )

                # NOTE: rhs cannot be negative as the flow cannot be to the upstream
                rhs = lfr.where(
                    (rhs > 0),
                    rhs,
                    0,
                )

                # print("type of layer_list[layer_id].T:", type(layer_list[layer_id].T))

                phase_state = phase_detect_from_temperature(layer_list[layer_id].T)

                # phase_state_numpy = lfr.to_numpy(phase_state)
                # print(
                #     "phase_state_numpy[50,50]",
                #     phase_state_numpy[50, 50],
                #     "layer_id",
                #     layer_id,
                # )

                # h_mesh_numpy = lfr.to_numpy(layer_list[layer_id].h_mesh)
                # print(
                #     "h_mesh_numpy[50,50]",
                #     h_mesh_numpy[50, 50],
                #     "layer_id",
                #     layer_id,
                # )

                # print(
                #     "IS SAME OBJECT:",
                #     layer_list[layer_id].u_x is layer_list[0].u_x,
                #     "layer_id:",
                #     layer_id,
                # )

                # print("Before:", id(layer_list[layer_id].u_x))

                layer_list[layer_id].u_x = momentum_ux(
                    layer_list[layer_id].u_x,
                    phase_state,  # phase_state_initial,
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

                # print("After:", id(layer_list[layer_id].u_x))

                rhs_numpy = lfr.to_numpy(rhs)
                print(
                    "rhs_numpy[10,10]",
                    rhs_numpy[10, 10],
                    "layer_id",
                    layer_id,
                )

                # if layer_id == 0:  # bed layer

                #     # print("type of layer_list[1].u_x:", type(layer_list[1].u_x))
                #     # print("type of layer_list[1].h_mesh:", type(layer_list[1].h_mesh))

                #     d2u_x_dy2[0] = second_derivatives_in_y(
                #         layer_list[1].u_x,
                #         layer_list[2].u_x,
                #         layer_list[0].u_x,
                #         layer_list[1].h_mesh,
                #         layer_list[0].h_mesh,
                #     )

                # if layer_id == num_layers - 1:  # surface layer
                #     d2u_x_dy2[-1] = second_derivatives_in_y(
                #         layer_list[num_layers - 2].u_x,
                #         layer_list[num_layers - 1].u_x,
                #         layer_list[num_layers - 3].u_x,
                #         layer_list[num_layers - 1].h_mesh,
                #         layer_list[num_layers - 2].h_mesh,
                #     )

                # else:  # internal layers
                #     d2u_x_dy2[layer_id] = second_derivatives_in_y(
                #         layer_list[layer_id].u_x,
                #         layer_list[layer_id + 1].u_x,
                #         layer_list[layer_id - 1].u_x,
                #         layer_list[layer_id].h_mesh,
                #         layer_list[layer_id - 1].h_mesh,
                #     )

                # d2u_x_dy2_numpy = lfr.to_numpy(d2u_x_dy2[layer_id])
                # print(
                #     "d2u_x_dy2_numpy[50,50]",
                #     d2u_x_dy2_numpy[50, 50],
                #     "layer_id",
                #     layer_id,
                # )

                # input("Enter to continue ...")

            for layer_id in range(0, num_layers):
                if layer_id == 0:  # bed layer

                    # print("type of layer_list[1].u_x:", type(layer_list[1].u_x))
                    # print("type of layer_list[1].h_mesh:", type(layer_list[1].h_mesh))

                    d2u_x_dy2[0] = second_derivatives_in_y(
                        layer_list[1].u_x,
                        layer_list[2].u_x,
                        layer_list[0].u_x,
                        layer_list[1].h_mesh,
                        layer_list[0].h_mesh,
                    )

                elif layer_id == num_layers - 1:  # surface layer
                    d2u_x_dy2[num_layers - 1] = second_derivatives_in_y(
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

                d2u_x_dy2_numpy = lfr.to_numpy(d2u_x_dy2[layer_id])
                print(
                    "d2u_x_dy2_numpy[50,50]",
                    d2u_x_dy2_numpy[50, 50],
                    "layer_id",
                    layer_id,
                )

            for layer_id in range(0, num_layers):

                layer_u_x_numpy = lfr.to_numpy(layer_list[layer_id].u_x)
                print(
                    "layer_u_x_numpy[10,10]",
                    layer_u_x_numpy[10, 10],
                    "layer_id",
                    layer_id,
                )

            for layer_id in range(0, num_layers):

                layer_h_mesh_numpy = lfr.to_numpy(layer_list[layer_id].h_mesh)
                print(
                    "layer_h_mesh_numpy[10,10]",
                    layer_h_mesh_numpy[10, 10],
                    "layer_id",
                    layer_id,
                )

            # local_momentum_time = local_momentum_time + dt_momentum
            # local_momentum_iteration = local_momentum_iteration + 1

            print("local_momentum_iteration: ", local_momentum_iteration)

            # if local_momentum_iteration > momentum_iteration_threshold:

            #     time = (
            #         time - (momentum_iteration_threshold * dt_momentum)
            #     ) + dt_mass_conservation

        print("time inside momentum u_x", time)
        input("enter to continue ...")

        # --------------- End: compute momentum u_x in internal layers -----------------

        # --------------- compute VOF -------------------------------

        if abs(time - local_mass_conservation_time) >= dt_mass_conservation:

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

            local_mass_conservation_time = (
                local_mass_conservation_time + dt_mass_conservation
            )

        h_total = calculate_total_h(layer_list)

        # --------------- End: compute VOF -------------------------------

        eps_write_intervals: float = 1e-6

        write_intervals_time = dt_mass_conservation

        if time % write_intervals_time < eps_write_intervals:

            write_tif_file(
                h_total,
                "h_total",
                iteration_write_h_total,
                results_pathname,
            )

            for layer_id in range(0, num_layers):

                write_tif_file(
                    layer_list[layer_id].u_x,
                    f"u_x_l_{layer_id}_t",
                    iteration_write_h_total,
                    results_pathname,
                )

                write_tif_file(
                    layer_list[layer_id].T,
                    f"temp_l_{layer_id}_t",
                    iteration_write_h_total,
                    results_pathname,
                )

            iteration_write_h_total = iteration_write_h_total + 1

            print("simulation iteration_write_h_total: ", iteration_write_h_total)
