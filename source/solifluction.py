#!/usr/bin/env python
# import os
# import os.path
# import sys
import sys
from pathlib import Path
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
    save_u_x_tem_time,
    write_tif_file,
)
from source.layer import Layer
from source.momentum import momentum_ux
from source.phase_detect import phase_detect_from_temperature
from source.viscosity_calc import viscosity_exp_temp
from source.vof import calculate_total_h, h_mesh_assign, mass_conservation_2D_vof

# from source.boundary_condition import boundary_set


# from input_output import write

Shape = tuple[int, int]

print("sys.argv inside code source/solifluction.py =", sys.argv)


def is_phi_steady(phi_previous: Any, phi_current: Any, tol: float = 1e-4) -> bool:

    relative_error = lfr.sum(lfr.abs(phi_current - phi_previous)) / lfr.sum(
        lfr.abs(phi_previous)
    )
    if relative_error < tol:
        return True
    return False


def initialize_solifluction(
    array_shape: Shape,
    partition_shape: Shape,
    max_h_total: float,
    bed_depth_elevation: float,
    h_mesh_step_value: float,
    h_total_initial_file_name: str,
    mu_value: float,
    density_value: float,
    k_conductivity_value: float,
    rho_c_heat_value: float,
    temps_temperature_file: list[float],
) -> tuple[
    list[Layer],  # layer_list,
    list[Any],  # d2u_x_dy2_initial,
    Any,  # h_total_initial,
    int,  # num_layers,
    Any,  # temperature_bed,
    Any,  # boundary_loc,
    Any,  # boundary_type,
    Any,  # dirichlet_boundary_value,
    Any,  # neumann_boundary_value,
    Any,  # phase_state_initial
]:

    # self.generation = initialize_generation(
    #     array_shape=self.array_shape, partition_shape=self.partition_shape
    # )

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
        h_total_initial_file_name,
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

    num_rows, num_cols = array_shape

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

    dirichlet_boundary_value_numpy = create_zero_numpy_array(
        num_cols, num_rows, 0, np.float64
    )
    neumann_boundary_value_numpy = create_zero_numpy_array(
        num_cols, num_rows, 0, np.float64
    )

    dirichlet_boundary_value_numpy[[0, -1], :] = 0  # -999    # These are virtual layers
    # NOT boundaries (boundaries are imposed on layer inner)
    dirichlet_boundary_value_numpy[:, [0, -1]] = 0  # -999

    boundary_loc = convert_numpy_to_lue(
        boundary_loc_numpy_default, partition_shape=partition_shape
    )

    boundary_type = convert_numpy_to_lue(
        boundary_type_numpy_default, partition_shape=partition_shape
    )

    dirichlet_boundary_value = convert_numpy_to_lue(
        dirichlet_boundary_value_numpy, partition_shape=partition_shape
    )

    neumann_boundary_value = convert_numpy_to_lue(
        neumann_boundary_value_numpy, partition_shape=partition_shape
    )

    # ---------------- End: boundary condition set --------------------------

    # print(
    #     " Before function h_mesh_assign type of h_total_initial:",
    #     type(h_total_initial),
    # )

    h_mesh_list = h_mesh_assign(
        h_total_initial, num_layers, np.float64(h_mesh_step_value)
    )

    # h_total = h_total_initial

    for i in range(num_layers):
        layer_list[i].h_mesh = h_mesh_list[i]

    surface_temperature = temps_temperature_file[0]

    surface_temperature_lue = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=surface_temperature,
        partition_shape=partition_shape,
    )

    layer_list[0].T = temperature_bed

    for layer_id in range(1, num_layers):

        layer_list[layer_id].T = (layer_id / num_layers) * surface_temperature_lue

    return (
        layer_list,
        d2u_x_dy2_initial,
        h_total_initial,
        num_layers,
        temperature_bed,
        boundary_loc,
        boundary_type,
        dirichlet_boundary_value,
        neumann_boundary_value,
        phase_state_initial,
    )


def simulate_solifluction(
    array_shape: Shape,
    partition_shape: Shape,
    layer_list: list[Layer],
    d2u_x_dy2: list[Any],
    h_total: Any,
    num_layers: int,
    dt_global_model: float,
    dt_mass_conservation: float,
    dx: float,
    dz: float,
    dt_momentum: float,
    dt_heat_transfer: float,
    momentum_iteration_threshold: int,
    slope_radian: float,
    nu_x: float,
    nu_z: float,
    temperature_bed: Any,
    boundary_loc: Any,
    boundary_type: Any,
    dirichlet_boundary_value: Any,
    neumann_boundary_value: Any,
    surface_temperature: float,
    model_total_iteration: int,
    surface_density: float,
) -> tuple[list[Layer], int]:

    # if heat_transfer_warmup:

    #     for _ in range(1, heat_transfer_warmup_iteration):

    #         # surface_temperature = temps_temperature_file[0]

    #         # surface_temperature_lue = lfr.create_array(
    #         #     array_shape,
    #         #     dtype=np.float64,
    #         #     fill_value=surface_temperature,
    #         #     partition_shape=partition_shape,
    #         # )

    #         # layer_list[0].T = temperature_bed
    #         # layer_list[num_layers - 1].T = surface_temperature_lue

    #         for layer_id in range(1, num_layers - 1):

    #             layer_list[layer_id].T = compute_temperature_1D_in_y(
    #                 layer_list[layer_id].k_conductivity_heat,
    #                 layer_list[layer_id + 1].k_conductivity_heat,
    #                 layer_list[layer_id - 1].k_conductivity_heat,
    #                 layer_list[layer_id].rho_c_heat,
    #                 layer_list[layer_id].T,
    #                 layer_list[layer_id + 1].T,
    #                 layer_list[layer_id - 1].T,
    #                 dt_heat_transfer,
    #                 layer_list[layer_id].h_mesh,
    #                 layer_list[layer_id - 1].h_mesh,
    #                 surface_temperature,
    #             )

    time: float = 0
    # local_momentum_time: float = 0
    local_mass_conservation_time: float = 0
    # local_heat_transfer_time: float = 0

    # iteration_write_h_total: int = 0

    # d2u_x_dy2 = d2u_x_dy2_initial

    # dt_min: float = min(dt_momentum, dt_heat_transfer)
    # dt_max: float = max(dt_momentum, dt_heat_transfer, dt_mass_conservation)
    # dt_max = dt_mass_conservation  # dt_mass_conservation is considered as maximum time step for simulation

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

    # time_iteration: int = 0

    # while time < time_end_simulation:

    time = time + dt_global_model

    local_momentum_iteration: int = 0
    local_heat_transfer_iteration: int = 0

    g_sin = np.sin(slope_radian) * 9.81

    # print("time inside simulate_solifluction: ", time)

    # time_iteration = time_iteration + 1
    # print("time iteration: ", time_iteration)

    # --------------- compute temperatures in internal layers --------------------------

    while (
        local_heat_transfer_iteration < heat_transfer_iteration_threshold
    ):  # if abs(time - local_heat_transfer_time) >= dt_heat_transfer:

        # print("compute_temperature_1D_in_y in run")

        # surface_temperature = interpolate_temperature(
        #     time, days_temperature_file, temps_temperature_file
        # )

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

        # for layer_id in range(0, num_layers):

        #     layer_T_numpy = lfr.to_numpy(layer_list[layer_id].T)
        #     print(
        #         "layer_T_numpy[50,50]",
        #         layer_T_numpy[50, 50],
        #         "layer_id",
        #         layer_id,
        #     )

        # local_heat_transfer_time = local_heat_transfer_time + dt_heat_transfer

        local_heat_transfer_iteration = local_heat_transfer_iteration + 1

        # print("local_heat_transfer_iteration :", local_heat_transfer_iteration)

    # --------------- End: compute temperatures in internal layers -----------------

    # --------------- compute momentum u_x in internal layers ----------------------

    while local_momentum_iteration < momentum_iteration_threshold:
        # while (
        #     local_momentum_iteration < momentum_iteration_threshold
        # ):  # if abs(time - local_momentum_time) >= dt_momentum:

        # print("momentum_ux in run")

        for layer_id in range(1, num_layers):

            # a, b: 2e13, 0.4605 (T_data = ([0, 5]), mu_data([2e13, 2e12]))
            # a, b: 2e13, 0.9210 (T_data = ([0, 5]), mu_data([2e13, 2e11]))

            layer_list[layer_id].mu_soil = viscosity_exp_temp(
                layer_list[layer_id].T, 2e13, 0.9210
            )

            # gama_prim_surface: float = (2610 - 1000) * 9.81
            gama_prim_surface: float = (
                (surface_density / np.cos(slope_radian)) - 1000
            ) * 9.81

            rhs = (
                g_sin
                + (
                    (layer_list[layer_id].mu_soil / layer_list[layer_id].density_soil)
                    * d2u_x_dy2[layer_id]
                )
                - (
                    (gama_prim_surface / layer_list[layer_id].density_soil)
                    * dx_upwind(h_total, dx, layer_list[layer_id].u_x)
                )
            )

            # print("g_sin: ", g_sin)

            # d2u_x_dy2_numpy = lfr.to_numpy(d2u_x_dy2[layer_id])

            # print("d2u_x_dy2: ", d2u_x_dy2_numpy[50, 50])

            # print(
            #     "dx_upwind: ",
            #     lfr.to_numpy(dx_upwind(h_total, dx, layer_list[layer_id].u_x))[50, 50],
            # )

            # print("rhs:", lfr.to_numpy(rhs))

            # input("Enter to continue ...")

            # # NOTE: rhs cannot be negative as the fluid cannot flow to the upstream
            # rhs = lfr.where(
            #     (rhs > 0),
            #     rhs,
            #     0,
            # )

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
                dirichlet_boundary_value,
                neumann_boundary_value,
            )

            # layer_list[layer_id].u_x = momentum_ux_steady_state(
            #     layer_list[layer_id].u_x,
            #     phase_state,  # phase_state_initial,
            #     dx,
            #     dz,
            #     layer_list[layer_id].u_x,
            #     layer_list[layer_id].u_z,
            #     nu_x,
            #     nu_z,
            #     rhs,
            #     layer_list[layer_id].h_mesh,
            #     boundary_loc,
            #     boundary_type,
            #     dirichlet_boundary_value,
            #     neumann_boundary_value,
            # )

            # print("After:", id(layer_list[layer_id].u_x))

            # rhs_numpy = lfr.to_numpy(rhs)
            # print(
            #     "rhs_numpy[10,10]",
            #     rhs_numpy[10, 10],
            #     "layer_id",
            #     layer_id,
            # )

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

            # d2u_x_dy2_numpy = lfr.to_numpy(d2u_x_dy2[layer_id])
            # print(
            #     "d2u_x_dy2_numpy[50,50]",
            #     d2u_x_dy2_numpy[50, 50],
            #     "layer_id",
            #     layer_id,
            # )

        # for layer_id in range(0, num_layers):

        #     layer_u_x_numpy = lfr.to_numpy(layer_list[layer_id].u_x)
        #     print(
        #         "layer_u_x_numpy[10,10]",
        #         layer_u_x_numpy[10, 10],
        #         "layer_id",
        #         layer_id,
        #     )

        #     print(
        #         "layer_u_x_numpy[10,10] (cm/year)",
        #         layer_u_x_numpy[10, 10] * 3600 * 24 * 365 * 100,
        #         "layer_id",
        #         layer_id,
        #     )

        # for layer_id in range(0, num_layers):

        #     layer_h_mesh_numpy = lfr.to_numpy(layer_list[layer_id].h_mesh)
        #     print(
        #         "layer_h_mesh_numpy[10,10]",
        #         layer_h_mesh_numpy[10, 10],
        #         "layer_id",
        #         layer_id,
        #     )

        # local_momentum_time = local_momentum_time + dt_momentum
        local_momentum_iteration = local_momentum_iteration + 1

        model_total_iteration = model_total_iteration + 1

        # print("local_momentum_iteration: ", local_momentum_iteration)

        # if local_momentum_iteration > momentum_iteration_threshold:

        #     time = (
        #         time - (momentum_iteration_threshold * dt_momentum)
        #     ) + dt_mass_conservation

    # print("time inside momentum u_x", time)
    # input("enter to continue ...")

    # --------------- End: compute momentum u_x in internal layers -----------------

    # --------------- compute VOF -------------------------------

    if abs(time - local_mass_conservation_time) >= dt_mass_conservation:

        # print("mass_conservation_2D_vof in run")

        layer_list[layer_id].h_mesh = mass_conservation_2D_vof(
            layer_list[layer_id].h_mesh,
            layer_list[layer_id].u_x,
            layer_list[layer_id].u_z,
            dt_mass_conservation,
            dx,
            dz,
            boundary_loc,
            boundary_type,
            dirichlet_boundary_value,
            neumann_boundary_value,
        )

        local_mass_conservation_time = (
            local_mass_conservation_time + dt_mass_conservation
        )

    h_total = calculate_total_h(layer_list)

    # --------------- End: compute VOF -------------------------------

    # eps_write_intervals: float = 1e-6

    # if model_total_iteration % 50 == 0:

    #     for layer_id in range(0, num_layers):

    #         print("----- inside simulate_solifluction ---------")
    #         print("model_total_iteration: ", model_total_iteration)

    #         layer_u_x_numpy = lfr.to_numpy(layer_list[layer_id].u_x)
    #         print(
    #             "layer_u_x_numpy[10,10]",
    #             layer_u_x_numpy[10, 10],
    #             "layer_id",
    #             layer_id,
    #         )

    #         print(
    #             "layer_u_x_numpy[10,10] (cm/year)",
    #             layer_u_x_numpy[10, 10] * 3600 * 24 * 365 * 100,
    #             "layer_id",
    #             layer_id,
    #         )

    #         print(
    #             "layer_u_x_numpy[50,50] (cm/year)",
    #             layer_u_x_numpy[50, 50] * 3600 * 24 * 365 * 100,
    #             "layer_id",
    #             layer_id,
    #         )

    #     for layer_id in range(0, num_layers):

    #         layer_h_mesh_numpy = lfr.to_numpy(layer_list[layer_id].h_mesh)
    #         print(
    #             "layer_h_mesh_numpy[10,10]",
    #             layer_h_mesh_numpy[10, 10],
    #             "layer_id",
    #             layer_id,
    #         )

    #     for layer_id in range(0, num_layers):

    #         layer_T_numpy = lfr.to_numpy(layer_list[layer_id].T)
    #         print(
    #             "layer_T_numpy[50,50]",
    #             layer_T_numpy[50, 50],
    #             "layer_id",
    #             layer_id,
    #         )

    # save_tif_file(
    #     h_total,
    #     "h_total",
    #     time,
    #     results_path,
    # )

    # for layer_id in range(0, self.num_layers):

    #     save_tif_file(
    #         self.layer_list[layer_id].u_x,
    #         f"u_x_l_{layer_id}_itr_{iteration}_t",
    #         time,
    #         self.results_path,
    #     )

    #     save_tif_file(
    #         self.layer_list[layer_id].T,
    #         f"temp_l_{layer_id}_itr_{iteration}_t",
    #         time,
    #         self.results_path,
    #     )

    # print("-----------write inside simulate_solifluction------------", "time: ", time)
    # input("enter to continue ...")

    # write_intervals_time = dt_mass_conservation

    # if time % write_intervals_time < eps_write_intervals:

    #     for layer_id in range(0, num_layers):

    #         layer_u_x_numpy = lfr.to_numpy(layer_list[layer_id].u_x)
    #         print(
    #             "layer_u_x_numpy[10,10]",
    #             layer_u_x_numpy[10, 10],
    #             "layer_id",
    #             layer_id,
    #         )

    #         print(
    #             "layer_u_x_numpy[10,10] (cm/year)",
    #             layer_u_x_numpy[10, 10] * 3600 * 24 * 365 * 100,
    #             "layer_id",
    #             layer_id,
    #         )

    #         print(
    #             "layer_u_x_numpy[50,50] (cm/year)",
    #             layer_u_x_numpy[50, 50] * 3600 * 24 * 365 * 100,
    #             "layer_id",
    #             layer_id,
    #         )

    #     for layer_id in range(0, num_layers):

    #         layer_h_mesh_numpy = lfr.to_numpy(layer_list[layer_id].h_mesh)
    #         print(
    #             "layer_h_mesh_numpy[10,10]",
    #             layer_h_mesh_numpy[10, 10],
    #             "layer_id",
    #             layer_id,
    #         )

    #     for layer_id in range(0, num_layers):

    #         layer_T_numpy = lfr.to_numpy(layer_list[layer_id].T)
    #         print(
    #             "layer_T_numpy[50,50]",
    #             layer_T_numpy[50, 50],
    #             "layer_id",
    #             layer_id,
    #         )

    #     write_tif_file(
    #         h_total,
    #         "h_total",
    #         time,
    #         results_pathname,
    #     )

    #     for layer_id in range(0, num_layers):

    #         write_tif_file(
    #             layer_list[layer_id].u_x,
    #             f"u_x_l_{layer_id}_t",
    #             time,
    #             results_pathname,
    #         )

    #         write_tif_file(
    #             layer_list[layer_id].T,
    #             f"temp_l_{layer_id}_t",
    #             time,
    #             results_pathname,
    #         )

    #     print("-----------------write---------------", "time: ", time)
    #     input("enter to continue ...")

    # iteration_write_h_total = iteration_write_h_total + 1
    # print("simulation iteration_write_h_total: ", iteration_write_h_total)

    return layer_list, model_total_iteration


class Solifluction(lfr.Model):

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

        (
            self.layer_list,
            d2u_x_dy2_initial,
            h_total_initial,
            self.num_layers,
            self.temperature_bed,
            self.boundary_loc,
            self.boundary_type,
            self.dirichlet_boundary_value,
            self.neumann_boundary_value,
            self.phase_state_initial,
        ) = initialize_solifluction(
            self.array_shape,
            self.partition_shape,
            self.max_h_total,
            self.bed_depth_elevation,
            self.h_mesh_step_value,
            self.h_total_initial_file_name,
            self.mu_value,
            self.density_value,
            self.k_conductivity_value,
            self.rho_c_heat_value,
            self.temps_temperature_file,
        )

        self.h_total = h_total_initial
        self.d2u_x_dy2 = d2u_x_dy2_initial

    def simulate(self, iteration: int) -> Any:
        # self.generation = next_generation(self.generation)
        # self.save_generation(self.generation, iteration)

        print("iteration inside simulate -------", iteration)

        # ----- interpolate surface temperature from file

        time: float = iteration * self.dt_global_model

        surface_temperature: float = interpolate_temperature(
            time, self.days_temperature_file, self.temps_temperature_file
        )

        # ----- End: interpolate surface temperature from file

        self.layer_list, self.model_total_iteration = simulate_solifluction(
            self.array_shape,
            self.partition_shape,
            self.layer_list,
            self.d2u_x_dy2,
            self.h_total,
            self.num_layers,
            self.dt_global_model,
            self.dt_mass_conservation,
            self.dx,
            self.dz,
            self.dt_momentum,
            self.dt_heat_transfer,
            self.momentum_iteration_threshold,
            self.slope_radian,
            self.nu_x,
            self.nu_z,
            self.temperature_bed,
            self.boundary_loc,
            self.boundary_type,
            self.dirichlet_boundary_value,
            self.neumann_boundary_value,
            surface_temperature,
            self.model_total_iteration,
            self.density_value,
        )

        if iteration % self.write_intervals_time == 0:

            for layer_id in range(0, self.num_layers):

                layer_u_x_numpy = lfr.to_numpy(self.layer_list[layer_id].u_x)
                print(
                    "layer_u_x_numpy[10,10]",
                    layer_u_x_numpy[10, 10],
                    "layer_id",
                    layer_id,
                )

                print(
                    "layer_u_x_numpy[10,10] (cm/year)",
                    layer_u_x_numpy[10, 10] * 3600 * 24 * 365 * 100,
                    "layer_id",
                    layer_id,
                )

                print(
                    "layer_u_x_numpy[50,50] (cm/year)",
                    layer_u_x_numpy[50, 50] * 3600 * 24 * 365 * 100,
                    "layer_id",
                    layer_id,
                )

            for layer_id in range(0, self.num_layers):

                layer_h_mesh_numpy = lfr.to_numpy(self.layer_list[layer_id].h_mesh)
                print(
                    "layer_h_mesh_numpy[10,10]",
                    layer_h_mesh_numpy[10, 10],
                    "layer_id",
                    layer_id,
                )

            for layer_id in range(0, self.num_layers):

                layer_T_numpy = lfr.to_numpy(self.layer_list[layer_id].T)
                print(
                    "layer_T_numpy[50,50]",
                    layer_T_numpy[50, 50],
                    "layer_id",
                    layer_id,
                )

            layer_u_x_numpy_surf = (
                lfr.to_numpy(self.layer_list[self.num_layers - 1].u_x)
                * 3600
                * 24
                * 365
                * 100
            )

            print(
                "layer_u_x_numpy_surf[50,50]: ",
                layer_u_x_numpy_surf[50, 50],
            )

            self.u_x_tem_time.append(
                [time / 86400, surface_temperature, layer_u_x_numpy_surf[50, 50]]
            )

            print("u_x_tem_time: ", self.u_x_tem_time)

            # save_tif_file(
            #     self.h_total,
            #     "h_total",
            #     time,
            #     self.results_path,
            # )

            write_tif_file(
                self.h_total,
                "h_total",
                iteration,
                self.results_path,
            )

            # for layer_id in range(0, self.num_layers):

            #     save_tif_file(
            #         self.layer_list[layer_id].u_x,
            #         f"u_x_l_{layer_id}_itr_{iteration}_t",
            #         time,
            #         self.results_path,
            #     )

            #     save_tif_file(
            #         self.layer_list[layer_id].T,
            #         f"temp_l_{layer_id}_itr_{iteration}_t",
            #         time,
            #         self.results_path,
            #     )

            for layer_id in range(0, self.num_layers):

                write_tif_file(
                    self.layer_list[layer_id].u_x,
                    f"u_x_l_{layer_id}_itr",
                    iteration,
                    self.results_path,
                )

                write_tif_file(
                    self.layer_list[layer_id].u_x,
                    f"temp_l_{layer_id}_itr",
                    iteration,
                    self.results_path,
                )

            print(
                "-----------------write---------------",
                "time: ",
                time,
                "iteration",
                iteration,
            )
            # input("enter to continue ...")

        if iteration == self.number_of_iterations:

            save_u_x_tem_time(
                self.u_x_tem_time, f"{self.results_path}/u_x_tem_time.csv"
            )

        return self.layer_list[1].u_x.future()


@lfr.runtime_scope
def solifluction(
    *,
    array_shape: Shape,
    partition_shape: Shape,
    number_of_iterations: int,
    results_pathname: Path,
    dx: float,
    dz: float,
    dt_momentum: float,
    momentum_iteration_threshold: int,
    dt_global_model: float,
    dt_heat_transfer: float,
    dt_mass_conservation: float,
    time_end_simulation: float,
    write_intervals_time: float,
    max_h_total: float,
    bed_depth_elevation: float,
    h_mesh_step_value: float,
    h_total_initial_file_name: str,
    mu_value: float,
    density_value: float,
    k_conductivity_value: float,
    rho_c_heat_value: float,
    slope_radian: float,
    nu_x: float,
    nu_z: float,
    days_temperature_file: list[float],
    temps_temperature_file: list[float],
) -> None:

    model = Solifluction(
        array_shape=array_shape,
        partition_shape=partition_shape,
        results_path=results_pathname,
    )

    model.dt_global_model = dt_global_model
    model.dt_momentum = dt_momentum
    model.momentum_iteration_threshold = momentum_iteration_threshold
    model.dt_heat_transfer = dt_heat_transfer
    model.dt_mass_conservation = dt_mass_conservation
    model.time_end_simulation = time_end_simulation
    model.write_intervals_time = write_intervals_time
    model.dx = dx
    model.dz = dz
    model.slope_radian = slope_radian
    model.nu_x = nu_x
    model.nu_z = nu_z
    model.max_h_total = max_h_total
    model.bed_depth_elevation = bed_depth_elevation
    model.h_mesh_step_value = h_mesh_step_value
    model.h_total_initial_file_name = h_total_initial_file_name
    model.mu_value = mu_value
    model.density_value = density_value
    model.k_conductivity_value = k_conductivity_value
    model.rho_c_heat_value = rho_c_heat_value
    model.days_temperature_file = days_temperature_file
    model.temps_temperature_file = temps_temperature_file

    model_total_iteration_initial: int = 0
    model.model_total_iteration = model_total_iteration_initial
    model.u_x_tem_time = []
    model.number_of_iterations = number_of_iterations

    lfr.run_deterministic(
        model, lfr.DefaultProgressor(), nr_time_steps=number_of_iterations, rate_limit=3
    )
