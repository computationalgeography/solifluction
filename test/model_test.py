import os
import unittest

import lue.framework as lfr
import matplotlib.pyplot as plt
import numpy as np

from source.boundary_condition import boundary_set
from source.derivatives_discretization import second_derivatives_in_y
from source.heat_transfer import compute_temperature_1D_in_y
from source.io_data_process import (
    convert_numpy_to_lue,
    create_zero_numpy_array,
    default_boundary_type,
)
from source.layer import Layer
from source.momentum import momentum_ux
from source.vof import calculate_total_h, h_mesh_assign, mass_conservation_2D_vof


def h_exact_calculate(h, ux, uz, time_iteration, dt, dx, dz, value_boundary):
    # x ---> matrix or raster columns
    # z ---> matrix or raster rows

    time = time_iteration * dt

    shift_mesh_col = (ux * time) / dx
    shift_mesh_row = (uz * time) / dz

    h_exact = np.zeros_like(h)

    for row in range(h.shape[0]):
        for col in range(h.shape[1]):
            shifted_row = round(row - shift_mesh_row)
            shifted_col = round(col - shift_mesh_col)

            if 0 <= shifted_row < h.shape[0] and 0 <= shifted_col < h.shape[1]:
                h_exact[row, col] = h[shifted_row, shifted_col]
            else:
                h_exact[row, col] = value_boundary

    return h_exact


def boundary_set_with_numpy(
    phi,
    boundary_loc,
    boundary_type,
    Dirichlet_boundary_value,
    Neumann_boundary_value,
    dx,
    dz,
):
    phi_original = phi

    phi = np.where(
        (boundary_loc & (boundary_type == 0)), Dirichlet_boundary_value, phi_original
    )

    for num_row in range(0, phi.shape[0]):
        for num_col in range(0, phi.shape[1]):
            if boundary_loc[num_row, num_col] & (
                (boundary_type[num_row, num_col] == 1)
                | (boundary_type[num_row, num_col] == 5)
                | (boundary_type[num_row, num_col] == 6)
            ):
                phi[num_row, num_col] = phi[num_row, num_col + 1] - (
                    dx * Neumann_boundary_value[num_row, num_col]
                )

    for num_row in range(0, phi.shape[0]):
        for num_col in range(0, phi.shape[1]):
            if boundary_loc[num_row, num_col] & (boundary_type[num_row, num_col] == 2):
                phi[num_row, num_col] = phi[num_row - 1, num_col] - (
                    dz * Neumann_boundary_value[num_row, num_col]
                )

    for num_row in range(0, phi.shape[0]):
        for num_col in range(0, phi.shape[1]):
            if boundary_loc[num_row, num_col] & (
                (boundary_type[num_row, num_col] == 3)
                | (boundary_type[num_row, num_col] == 7)
                | (boundary_type[num_row, num_col] == 8)
            ):
                phi[num_row, num_col] = phi[num_row, num_col - 1] + (
                    dx * Neumann_boundary_value[num_row, num_col]
                )

    for num_row in range(0, phi.shape[0]):
        for num_col in range(0, phi.shape[1]):
            if boundary_loc[num_row, num_col] & (boundary_type[num_row, num_col] == 4):
                # print("num_row, num_col: ", num_row, num_col)
                # print("num_row + 1, num_col: ", num_row + 1, num_col)
                # print("boundary_type: \n", boundary_type)

                phi[num_row, num_col] = phi[num_row + 1, num_col] + (
                    dz * Neumann_boundary_value[num_row, num_col]
                )

    return phi


def exact_velocity_uniform_laminal_flow(g_sin, mu, rho_density, h_layer, num_layers):
    nu = mu / rho_density
    h_total = (num_layers - 1) * h_layer

    u = np.zeros(num_layers, dtype=np.float64)

    for i in range(1, num_layers):
        y = i * h_layer
        u[i] = ((-g_sin / (2 * nu)) * (y**2)) + ((g_sin / nu) * h_total * y)

    return u


def exact_heat_transfer_steady(h_total, T1, T2, num_layers):
    """Compute the exact steady-state solution for 1D heat conduction."""
    # Steady-state solution is independent of initial condition
    x = np.linspace(0, h_total, num_layers)  # Creates num_layers points between 0 and L

    # Compute the temperature at each x position
    return T1 + (T2 - T1) * x / h_total


# def exact_heat_transfer_unsteady(
#     L, T1, T2, T_initial, x, Thermal_diffusivity, time, N_terms=50, Nx=100
# ):
#     """Compute the analytical solution for 1D transient heat conduction.
#     T1, T2 are boundary values"""
#     T_s = T1 + (T2 - T1) * x / L  # Steady-state solution
#     T0 = T_initial(x, L)  # Initial condition passed as a function

#     # Compute Fourier coefficients
#     Bn = np.array(
#         [
#             (2 / L) * np.trapz((T0 - T_s) * np.sin(n * np.pi * x / L), x)
#             for n in range(1, N_terms + 1)
#         ]
#     )

#     # Compute transient solution
#     T_transient = np.zeros_like(x)
#     for n in range(1, N_terms + 1):
#         T_transient += (
#             Bn[n - 1]
#             * np.exp(-Thermal_diffusivity * (n * np.pi / L) ** 2 * time)
#             * np.sin(n * np.pi * x / L)
#         )

#     T = T_s + T_transient

#     return T


# def update_lue_data(lue_data):
#     lue_data += 1


# def update_numpy_data(numpy_data):
#     numpy_data += 1


class TestPackage(unittest.TestCase):

    @lfr.runtime_scope
    def test_mass_conservation_2D_vof(self) -> None:
        num_cols: int = 100  # x direction
        num_rows: int = 100  # z direction

        dt = 0.1
        dx = 1
        dz = 1

        u_x_value = 1
        u_z_value = 0

        time_iteration = 500  # int(time_end / dt)

        partition_shape = 2 * (20,)

        # Initial

        h_numpy = create_zero_numpy_array(
            num_cols,
            num_rows,
            0,
            np.float64,
        )

        u_x_numpy = create_zero_numpy_array(
            num_cols,
            num_rows,
            0,
            np.float64,
        )

        u_z_numpy = create_zero_numpy_array(
            num_cols,
            num_rows,
            0,
            np.float64,
        )

        boundary_type_numpy_default, boundary_loc_numpy_default = default_boundary_type(
            num_cols, num_rows, boundary_in_first_last_row_col=False
        )

        Dirichlet_boundary_value_numpy = create_zero_numpy_array(
            num_cols, num_rows, 0, np.float64
        )
        Neumann_boundary_value_numpy = create_zero_numpy_array(
            num_cols, num_rows, 0, np.float64
        )

        Dirichlet_boundary_value_numpy[[0, -1], :] = 5  # 0  # -999
        Dirichlet_boundary_value_numpy[:, [0, -1]] = 5  # 0  # -999

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

        h_numpy[:, :] = 5

        h_square = 10

        h_numpy[40:61, 10:31] = h_square

        u_x_numpy[:, :] = u_x_value
        u_z_numpy[:, :] = u_z_value

        h_lue = convert_numpy_to_lue(h_numpy, partition_shape)
        u_x_lue = convert_numpy_to_lue(u_x_numpy, partition_shape)
        u_z_lue = convert_numpy_to_lue(u_z_numpy, partition_shape)

        h_exact_numpy = h_exact_calculate(
            h_numpy,
            ux=u_x_value,
            uz=u_z_value,
            time_iteration=time_iteration,
            dt=dt,
            dx=dx,
            dz=dz,
            value_boundary=5,
        )

        h_exact_lue = convert_numpy_to_lue(h_exact_numpy, partition_shape)

        for _ in range(1, time_iteration + 1):
            h_lue = mass_conservation_2D_vof(
                h_lue,
                u_x_lue,
                u_z_lue,
                dt,
                dx,
                dz,
                boundary_loc,
                boundary_type,
                Dirichlet_boundary_value,
                Neumann_boundary_value,
            )

        error_matrix = lfr.abs(h_exact_lue - h_lue)

        error_matrix_numpy = lfr.to_numpy(error_matrix)

        time = time_iteration * dt
        shift_mesh_col = round((u_x_value * time) / dx)
        shift_mesh_row = round((u_z_value * time) / dz)
        h_lue_result_to_numpy = lfr.to_numpy(h_lue)
        h_result_center = h_lue_result_to_numpy[
            50 + shift_mesh_row, 20 + shift_mesh_col
        ]

        max_relative_error = np.max(error_matrix_numpy) / h_square

        print("max error: ", max_relative_error)

        center_box_error = abs(h_result_center - h_square) / h_square

        print("h_exact_center: ", h_square, "h_result_center: ", h_result_center)

        print("box center error: ", center_box_error)

        folder_path = "test/test_mass_conservation_2D_vof"
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        plt.contourf(lfr.to_numpy(h_lue))
        plt.colorbar()
        plt.title("Numerical results for h")
        plt.savefig(os.path.join(folder_path, "h_numerical.png"))
        plt.close()

        plt.contourf(h_exact_numpy)
        plt.colorbar()
        plt.title("h_exact")
        plt.savefig(os.path.join(folder_path, "h_exact.png"))
        plt.close()

        plt.contourf(error_matrix_numpy)
        plt.colorbar()
        plt.title("error_matrix")
        plt.savefig(os.path.join(folder_path, "error_matrix.png"))
        plt.close()

        error_threshold_max = 0.3
        error_threshold_center = 0.1

        self.assertLess(max_relative_error, error_threshold_max)
        self.assertLess(center_box_error, error_threshold_center)

        print(f"Check figures in {folder_path} comparing exact and numerical results")

    @lfr.runtime_scope
    def test_second_derivatives_in_y(self) -> None:
        """Test second derivatives (central,forward, and backward) for different
        types of functions."""

        partition_shape = 2 * (20,)

        layer_variable_ref = lfr.from_gdal(
            "test/h_test.tif", partition_shape=partition_shape
        )

        array_shape = layer_variable_ref.shape

        h_mesh_100 = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=100,
            partition_shape=partition_shape,
        )
        h_mesh_500 = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=500,
            partition_shape=partition_shape,
        )

        # layers order
        # --------------top
        # ------- layer 3
        # ------- layer 2
        # ------- layer 1
        # --------------bottom

        test_cases = [
            {
                # ay^2+by+c (a=2, b=5, c=layer_variable_2) ; for y=0 c=center_variable
                "name": "central_quadratic",
                "dy_layers_up": h_mesh_100,
                "dy_layers_down": h_mesh_500,
                "exact_d2var_dy2": 2 * 2,
                "layer_variable_2": layer_variable_ref,
                "layer_variable_3": 2 * (100**2) + 5 * 100 + layer_variable_ref,
                "layer_variable_1": 2 * ((-500) ** 2) + 5 * (-500) + layer_variable_ref,
            },
            {
                # c*exp(ay) (c=layer_variable_2, a=1.5); for y=0  c=center_variable
                "name": "central_exponential",
                "dy_layers_up": 0.02,
                "dy_layers_down": 0.01,
                "exact_d2var_dy2": 1 * layer_variable_ref,
                "layer_variable_2": layer_variable_ref,
                "layer_variable_3": layer_variable_ref * lfr.exp(1 * 0.02),
                "layer_variable_1": layer_variable_ref * lfr.exp(1 * -0.01),
            },
            {
                # ay^2+by+c (a=2, b=5, c=layer_variable_1); for y=0 c=center_variable
                "name": "forward_quadratic",
                "dy_layers_up": h_mesh_100,
                "dy_layers_down": h_mesh_500,
                "exact_d2var_dy2": 2 * 2,
                "layer_variable_1": layer_variable_ref,
                "layer_variable_2": 2 * (500**2) + 5 * 500 + layer_variable_ref,
                "layer_variable_3": 2 * ((500 + 100) ** 2)
                + (5 * (500 + 100))
                + layer_variable_ref,
            },
            {
                # ay^2+by+c (a=2, b=5, c=center_variable) ; for y=0 c=center_variable
                "name": "backward_quadratic",
                "dy_layers_up": 100,
                "dy_layers_down": 500,
                "exact_d2var_dy2": 2 * 2,
                "layer_variable_1": 2 * ((-500 - 100) ** 2)
                + (5 * (-500 - 100))
                + layer_variable_ref,
                "layer_variable_2": 2 * ((-100) ** 2)
                + 5 * (-100)
                + layer_variable_ref,  # a*layer_variable_up^2+b*layer_variable_up+c)
                "layer_variable_3": layer_variable_ref,
            },
        ]

        for case in test_cases:
            with self.subTest(function=case["name"]):
                d2var_dy2 = second_derivatives_in_y(
                    case["layer_variable_2"],
                    case["layer_variable_3"],
                    case["layer_variable_1"],
                    case["dy_layers_up"],
                    case["dy_layers_down"],
                )

                exact_value = case["exact_d2var_dy2"]

                error_matrix = lfr.abs(d2var_dy2 - exact_value)
                error_matrix_numpy = lfr.to_numpy(error_matrix)

                print(f"[{case['name']}] max error: {np.max(error_matrix_numpy)} \n")

                error_threshold = 0.05
                self.assertLess(np.max(error_matrix_numpy), error_threshold)

    @lfr.runtime_scope
    def test_momentum_ux(self) -> None:
        time = 0
        dt = 0.01  # 0.5  # 0.01  # 0.0005  # 0.01  # 0.1  # 1
        nr_time_steps = 300  # 10   #  300  # 100  # 400  # 500  # 200  # 50

        num_layers = 10  # 15  # 20  # 10  # 5

        mu = (
            10**4
        )  # ( 10**4 ok test)    #10**2  # 10**4 (ok test)  # 1000  # 10**-2  # 0
        density_soil = 1000  # 2650

        h_mesh_layer = (
            0.5  # 0.25  # 1  # 0.25  # S 0.25  # 0.5  # 0.5  # 1  # 0.1  # 20
        )

        num_cols: int = 200  # x direction size for layers' raster
        num_rows: int = 100  # z direction size for layers' raster

        dx = 1
        dz = 1

        # dh_dx = zero_array_lue

        g_sin = 9.81 * np.sin(np.pi / 6)

        u_exact = exact_velocity_uniform_laminal_flow(
            g_sin, mu, density_soil, h_mesh_layer, num_layers
        )

        ux_result = np.zeros(num_layers, dtype=np.float64)

        array_shape = (
            num_rows,
            num_cols,
        )
        partition_shape = 2 * (20,)

        # Initial condition definition

        # velocity boundary condition is Dirichlet and in all boundaries (first and last rows and columns) set to zero

        # boundary_type_numpy_default, boundary_loc_numpy_default = default_boundary_type(
        #     num_cols, num_rows, boundary_in_first_last_row_col=True
        # )
        # # set Dirichlet boundary type in all boundaries
        # boundary_type_numpy_default[:, :] = 0

        boundary_type_numpy_default, boundary_loc_numpy_default = default_boundary_type(
            num_cols, num_rows, boundary_in_first_last_row_col=False
        )

        Dirichlet_boundary_value_numpy = create_zero_numpy_array(
            num_cols, num_rows, 0, np.float64
        )
        Neumann_boundary_value_numpy = create_zero_numpy_array(
            num_cols, num_rows, 0, np.float64
        )

        Dirichlet_boundary_value_numpy[[0, -1], :] = -999
        Dirichlet_boundary_value_numpy[:, [0, -1]] = -999

        boundary_loc = convert_numpy_to_lue(
            boundary_loc_numpy_default, partition_shape=partition_shape
        )

        boundary_type = convert_numpy_to_lue(
            boundary_type_numpy_default, partition_shape=partition_shape
        )

        Dirichlet_boundary_value_lue = convert_numpy_to_lue(
            Dirichlet_boundary_value_numpy, partition_shape=partition_shape
        )

        Neumann_boundary_value_lue = convert_numpy_to_lue(
            Neumann_boundary_value_numpy, partition_shape=partition_shape
        )

        print("boundary_type_numpy_default: \n", boundary_type_numpy_default)
        print("boundary_loc_numpy_default: \n", boundary_loc_numpy_default)

        print(" boundary_loc.shape : ", boundary_loc.shape)

        zero_array_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=0.0,
            partition_shape=partition_shape,
        )

        # phase_state: 0 solid  --> (frozen soil), 1 --> (fluid or unfrozen),
        # for now vegetation is ignored in phase_state but it is considered in
        # vegetation_vol_fraction
        # In this test phase_state is 1 --> (fluid or unfrozen)

        phase_state_lue = lfr.create_array(
            array_shape,
            dtype=np.uint8,
            fill_value=1,
            partition_shape=partition_shape,
        )

        mu_array_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=mu,
            partition_shape=partition_shape,
        )

        density_soil_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=density_soil,
            partition_shape=partition_shape,
        )

        h_mesh = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=h_mesh_layer,
            partition_shape=partition_shape,
        )

        # End: Initial condition definition

        # instantiate Layer objects for all layers

        Layer_list: list[Layer] = []

        d2u_x_dy2 = []

        for _ in range(num_layers):
            d2u_x_dy2.append(zero_array_lue)

        # NOTE: number of layers is 0 to "num_layers" for bed layer to surface layer

        # Assign bed layer properties

        Layer_list.append(
            Layer(
                zero_array_lue,
                zero_array_lue,
                None,
                h_mesh,
                mu_array_lue,
                density_soil_lue,
                phase_state_lue,
                None,
                None,
                None,
            )
        )

        # Assign internal layers properties

        for i in range(1, num_layers):
            Layer_list.append(
                Layer(
                    zero_array_lue,
                    zero_array_lue,
                    None,
                    h_mesh,
                    mu_array_lue,
                    density_soil_lue,
                    phase_state_lue,
                    None,
                    None,
                    None,
                )
            )

        # Assign surface layer properties

        Layer_list.append(
            Layer(
                zero_array_lue,
                zero_array_lue,
                None,
                h_mesh,
                mu_array_lue,
                density_soil_lue,
                phase_state_lue,
                None,
                None,
                None,
            )
        )

        # End: instantiate Layer objects for all layers

        for time_step in range(1, nr_time_steps + 1):

            time: float = time + dt

            # velocity at bed layer (layer_id = 0) is zero
            for layer_id in range(1, num_layers):
                # calculate du2_dy2 for the right hand side of momentum (velocity) equation

                # rhs = g_sin - ((mu_array_lue / density_soil_lue) * d2u_x_dy2)

                rhs = g_sin + ((mu_array_lue / density_soil_lue) * d2u_x_dy2[layer_id])

                # rhs = g_sin

                # rhs_numpy = lfr.to_numpy(rhs)

                # print("rhs_numpy: \n", rhs_numpy)

                Layer_list[layer_id].u_x = momentum_ux(
                    Layer_list[layer_id].u_x,
                    phase_state_lue,
                    dx,
                    dz,
                    dt,
                    Layer_list[layer_id].u_x,
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

                layer_u_x_numpy = lfr.to_numpy(Layer_list[layer_id].u_x)

                ux_result[layer_id] = layer_u_x_numpy[50, 100]

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

                else:

                    d2u_x_dy2[layer_id] = second_derivatives_in_y(
                        Layer_list[layer_id].u_x,
                        Layer_list[layer_id + 1].u_x,
                        Layer_list[layer_id - 1].u_x,
                        h_mesh,
                        h_mesh,
                    )

            # CFL: float = (ux_result[-1] * dt) / dx
            # print("CFL: ", CFL)
            # print("time_step: ", time_step)

        folder_path = "test/test_momentum_ux"
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        plt.plot(
            ux_result,
            np.arange(0, h_mesh_layer * num_layers, h_mesh_layer),
            "b",
            label="Calculated numerical velocity (ux_result)",
        )

        plt.plot(
            u_exact,
            np.arange(0, h_mesh_layer * num_layers, h_mesh_layer),
            "r",
            label="Exact velocity (u_exact)",
        )

        plt.xlabel("Velocity")
        plt.ylabel("height")
        plt.legend()
        plt.savefig(os.path.join(folder_path, "compare_numerical_exact_ux.png"))
        plt.close()

        error_threshold_max = 0.3
        l2_error_threshold = 0.2

        max_relative_error: float = np.max(abs(u_exact - ux_result)) / np.max(
            abs(u_exact)
        )
        l2_relative_error: float = np.linalg.norm(u_exact - ux_result) / np.linalg.norm(
            u_exact
        )

        print("max_relative_error: ", max_relative_error)
        print("l2_relative_error: ", l2_relative_error)

        self.assertLess(max_relative_error, error_threshold_max)
        self.assertLess(l2_relative_error, l2_error_threshold)

        print(f"Check figures in {folder_path} comparing exact and numerical results")

    @lfr.runtime_scope
    def test_boundary_set(self):
        num_cols: int = 200  # x direction size for layers' raster
        num_rows: int = 100  # z direction size for layers' raster

        dx = 1
        dz = 1

        array_shape = (num_rows, num_cols)
        partition_shape = 2 * (20,)

        Dirichlet_boundary_value_numpy = create_zero_numpy_array(
            num_cols, num_rows, 0, np.float64
        )
        Neumann_boundary_value_numpy = create_zero_numpy_array(
            num_cols, num_rows, 0, np.float64
        )

        Dirichlet_boundary_value = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=0.0,
            partition_shape=partition_shape,
        )

        Neumann_boundary_value = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=0.0,
            partition_shape=partition_shape,
        )

        # boundary_type_numpy_default is the default boundary type.
        # It is overwritten only if Dirichlet condition (type = 0) is considered

        boundary_type_numpy_default, boundary_loc_numpy_default = default_boundary_type(
            num_cols, num_rows
        )

        boundary_type_numpy_test_Dirichlet = boundary_type_numpy_default
        boundary_type_numpy_test_Dirichlet[1:-1, -2] = 0

        (
            boundary_type_numpy_default_first_last,
            boundary_loc_numpy_default_first_last,
        ) = default_boundary_type(
            num_cols, num_rows, boundary_in_first_last_row_col=True
        )

        boundary_type_numpy_default_first_last[:, :] = 0

        phi_with_nan_numpy = np.random.uniform(0, 100, (num_rows, num_cols)).astype(
            np.float64
        )  # Ensure float64 type
        phi_with_nan_numpy[[0, -1], :] = np.nan
        phi_with_nan_numpy[:, [0, -1]] = np.nan

        phi_with_nan = convert_numpy_to_lue(
            phi_with_nan_numpy, partition_shape=partition_shape
        )

        test_cases = [
            {
                "name": "all_Neumann_second_mesh_layer",
                "phi": lfr.uniform(
                    array_shape,
                    dtype=np.float64,
                    min_value=0,
                    max_value=10,
                    partition_shape=partition_shape,
                ),
                "boundary_loc_numpy": boundary_loc_numpy_default,
                "boundary_type_numpy": boundary_type_numpy_default,
            },
            {
                "name": "Neumann_and_Dirichlet",
                "phi": lfr.uniform(
                    array_shape,
                    dtype=np.float64,
                    min_value=0,
                    max_value=10,
                    partition_shape=partition_shape,
                ),
                "boundary_loc_numpy": boundary_loc_numpy_default,
                "boundary_type_numpy": boundary_type_numpy_test_Dirichlet,
            },
            {
                "name": "all_Neumann_first_mesh_layer_with_nan",
                "phi": phi_with_nan,
                "boundary_loc_numpy": boundary_loc_numpy_default_first_last,
                "boundary_type_numpy": boundary_type_numpy_default_first_last,
            },
        ]

        for case in test_cases:
            with self.subTest(case["name"]):
                phi = case["phi"]

                boundary_loc_numpy = case["boundary_loc_numpy"]
                boundary_type_numpy = case["boundary_type_numpy"]

                boundary_loc = convert_numpy_to_lue(
                    boundary_loc_numpy, partition_shape=partition_shape
                )
                boundary_type = convert_numpy_to_lue(
                    boundary_type_numpy, partition_shape=partition_shape
                )

                phi_numpy = lfr.to_numpy(phi)

                phi = boundary_set(
                    phi,
                    boundary_loc,
                    boundary_type,
                    Dirichlet_boundary_value,
                    Neumann_boundary_value,
                    dx,
                    dz,
                )

                phi_expect_numpy = boundary_set_with_numpy(
                    phi_numpy,
                    boundary_loc_numpy,
                    boundary_type_numpy,
                    Dirichlet_boundary_value_numpy,
                    Neumann_boundary_value_numpy,
                    dx,
                    dz,
                )

                phi_lue_to_numpy = lfr.to_numpy(phi)

                error_matrix_numpy = np.abs(phi_lue_to_numpy - phi_expect_numpy)
                error_threshold = 10**-8

                print("phi_numpy - original phi: \n", phi_numpy)
                print("phi_lue_to_numpy - original \n", phi_lue_to_numpy)
                print("error_matrix_numpy: \n", error_matrix_numpy)
                print("boundary_loc_numpy: \n", boundary_loc_numpy)
                print("boundary_type_numpy: \n", boundary_type_numpy)

                self.assertLess(np.max(error_matrix_numpy), error_threshold)

    @lfr.runtime_scope
    def test_heat_transfer(self):
        # for this test uniform thermal_diffusivity is considered
        # uniform h_mesh is considered for layers

        num_cols: int = 100  # x direction size for layers' raster
        num_rows: int = 100  # z direction size for layers' raster

        array_shape = (num_rows, num_cols)
        partition_shape = 2 * (20,)

        num_layers = 6

        T_surface_value: float = 10.0
        T_bed_value: float = 2.0

        h_mesh_layer_value = 1.5

        dt = 0.1

        k_conductivity_heat_value: float = 5.0
        rho_c_heat_value: float = 2.0

        T_result = np.zeros(num_layers, dtype=np.float64)
        h_total = (num_layers - 1) * h_mesh_layer_value

        h_mesh = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=h_mesh_layer_value,
            partition_shape=partition_shape,
        )

        zero_array_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=0.0,
            partition_shape=partition_shape,
        )

        T_surface_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=T_surface_value,
            partition_shape=partition_shape,
        )

        T_bed_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=T_bed_value,
            partition_shape=partition_shape,
        )

        k_conductivity_heat = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=k_conductivity_heat_value,
            partition_shape=partition_shape,
        )

        rho_c_heat = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=rho_c_heat_value,
            partition_shape=partition_shape,
        )

        compute_flag = lfr.create_array(
            array_shape,
            dtype=np.uint8,
            fill_value=1,
            partition_shape=partition_shape,
        )

        precomputed_value = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=-99,
            partition_shape=partition_shape,
        )

        Layer_list = []

        # Assign bed layer properties
        Layer_list.append(
            Layer(
                None,
                None,
                zero_array_lue,
                h_mesh,
                None,
                None,
                None,
                k_conductivity_heat,
                rho_c_heat,
                None,
            )
        )

        # Assign internal layers properties
        for _ in range(1, num_layers):
            Layer_list.append(
                Layer(
                    None,
                    None,
                    zero_array_lue,
                    h_mesh,
                    None,
                    None,
                    None,
                    k_conductivity_heat,
                    rho_c_heat,
                    None,
                )
            )

        # Assign surface layer properties
        Layer_list.append(
            Layer(
                None,
                None,
                zero_array_lue,
                h_mesh,
                None,
                None,
                None,
                k_conductivity_heat,
                rho_c_heat,
                None,
            )
        )

        # Test Case 1: steady_state

        with self.subTest(
            msg="Testing steady problem and with uniform thermal diffusivity"
        ):

            T_exact = exact_heat_transfer_steady(
                h_total, T_bed_value, T_surface_value, num_layers
            )

            num_time_iteration = 100

            for time_iter in range(1, num_time_iteration + 1):
                # set boundary (bed and surface temperatures)
                Layer_list[0].T = T_bed_lue
                Layer_list[num_layers - 1].T = T_surface_lue

                T_result[0] = T_bed_value
                T_result[num_layers - 1] = T_surface_value

                T_numpy_bed = lfr.to_numpy(Layer_list[0].T)
                T_numpy_surf = lfr.to_numpy(Layer_list[num_layers - 1].T)
                print("T_numpy_bed: \n", T_numpy_bed)
                print("T_numpy_surf: \n", T_numpy_surf)

                # compute temperatures in internal layers
                for layer_id in range(1, num_layers - 1):
                    Layer_list[layer_id].T = compute_temperature_1D_in_y(
                        Layer_list[layer_id].k_conductivity_heat,
                        Layer_list[layer_id + 1].k_conductivity_heat,
                        Layer_list[layer_id - 1].k_conductivity_heat,
                        Layer_list[layer_id].rho_c_heat,
                        Layer_list[layer_id].T,
                        Layer_list[layer_id + 1].T,
                        Layer_list[layer_id - 1].T,
                        dt,
                        Layer_list[layer_id].h_mesh,
                        Layer_list[layer_id - 1].h_mesh,
                        compute_flag,
                        precomputed_value,
                    )

                    print("layer_id: ", layer_id)
                    T_numpy = lfr.to_numpy(Layer_list[layer_id].T)
                    CFL = (T_numpy[50, 50] * dt) / h_mesh_layer_value
                    print("CFL: ", CFL)
                    print("T_numpy: \n", T_numpy)
                    print("T_exact: ", T_exact)

                print("time_iter: ", time_iter)

            input("Enter key to continue ...")

            for layer_id in range(0, num_layers):
                layer_T_numpy = lfr.to_numpy(Layer_list[layer_id].T)

                T_result[layer_id] = layer_T_numpy[50, 50]

            T_exact = exact_heat_transfer_steady(
                h_total, T_bed_value, T_surface_value, num_layers
            )

            plt.plot(
                T_result,
                np.arange(0, h_total + h_mesh_layer_value, h_mesh_layer_value),
                "bd--",
                label="Calculated temperature (T_result)",
            )
            plt.plot(
                T_exact,
                np.arange(0, h_total + h_mesh_layer_value, h_mesh_layer_value),
                "r",
                label="Exact temperature (T_exact)",
            )
            plt.xlabel("Temperature")
            plt.ylabel("height")
            plt.legend()
            plt.show()

            error_threshold = 0.05

            error_matrix = abs((T_exact - T_result) / T_exact)

            print("numerical simulation max error is: \n", np.max(error_matrix))

            self.assertLess(np.max(error_matrix), error_threshold)

    @lfr.runtime_scope
    def test_h_mesh_assign(self):

        num_cols: int = 10  # x direction size for layers' raster
        num_rows: int = 10  # z direction size for layers' raster

        array_shape = (num_rows, num_cols)
        partition_shape = 2 * (2,)

        zero_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=0.0,
            partition_shape=partition_shape,
        )

        h_mesh_step_value = 10.0

        h_total_numpy = np.zeros(array_shape, dtype=np.float64)

        for i_rows in range(0, num_rows):
            for i_cols in range(0, num_cols):

                h_total_numpy[i_rows, i_cols] = (
                    12.55 + (num_cols - i_cols - 1) * h_mesh_step_value
                )

        num_layers: int = int(np.round(np.max(h_total_numpy) / h_mesh_step_value) + 1)

        print("num_layers: ", num_layers)

        layer_list = []

        # Assign bed layer properties
        layer_list.append(
            Layer(
                None,
                None,
                None,
                zero_lue,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )

        # Assign internal layers properties
        for _ in range(1, num_layers):
            layer_list.append(
                Layer(
                    None,
                    None,
                    None,
                    zero_lue,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )

        # Assign surface layer properties
        layer_list.append(
            Layer(
                None,
                None,
                None,
                zero_lue,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )

        print(
            "layer_list[3].h_mesh for initial array: ",
            layer_list[3].h_mesh.dtype,
        )

        h_total_lue = convert_numpy_to_lue(h_total_numpy, partition_shape)

        plt.contourf(h_total_numpy)
        plt.colorbar()
        plt.show()

        h_mesh_assign(h_total_lue, num_layers, h_mesh_step_value, layer_list)

        layer_0_h_mesh_numpy = lfr.to_numpy(layer_list[0].h_mesh)
        layer_1_h_mesh_numpy = lfr.to_numpy(layer_list[1].h_mesh)
        layer_2_h_mesh_numpy = lfr.to_numpy(layer_list[2].h_mesh)
        layer_3_h_mesh_numpy = lfr.to_numpy(layer_list[3].h_mesh)
        layer_4_h_mesh_numpy = lfr.to_numpy(layer_list[4].h_mesh)
        layer_5_h_mesh_numpy = lfr.to_numpy(layer_list[5].h_mesh)
        layer_10_h_mesh_numpy = lfr.to_numpy(layer_list[10].h_mesh)
        h_total_lue_numpy = lfr.to_numpy(h_total_lue)

        exact_h_mesh_layer_3 = [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.55, 0.0, 0.0],
        ]

        plt.contourf(layer_3_h_mesh_numpy)
        plt.colorbar()
        plt.show()
        plt.title("layer_3_h_mesh_numpy")

        plt.contourf(exact_h_mesh_layer_3)
        plt.colorbar()
        plt.show()
        plt.title("exact_h_mesh_layer_3")

        print(
            "layer_0_h_mesh_numpy: \n",
            layer_0_h_mesh_numpy,
        )
        print(
            "layer_1_h_mesh_numpy: \n",
            layer_1_h_mesh_numpy,
        )
        print(
            "layer_2_h_mesh_numpy: \n",
            layer_2_h_mesh_numpy,
        )
        print(
            "layer_3_h_mesh_numpy: \n",
            layer_3_h_mesh_numpy,
        )
        print(
            "layer_4_h_mesh_numpy: \n",
            layer_4_h_mesh_numpy,
        )
        print(
            "layer_5_h_mesh_numpy: \n",
            layer_5_h_mesh_numpy,
        )

        print(
            "h_total_lue_numpy: \n",
            h_total_lue_numpy,
        )

        print(
            "layer_10_h_mesh_numpy: \n",
            layer_10_h_mesh_numpy,
        )

        error_threshold = 10e-8

        error_matrix = abs(exact_h_mesh_layer_3 - layer_3_h_mesh_numpy)

        print("simulation max error is: \n", np.max(error_matrix))

        print("error_matrix: ", error_matrix)

        self.assertLess(np.max(error_matrix), error_threshold)

        h_total_retrieved = calculate_total_h(layer_list)
        h_total_retrieved_numpy = lfr.to_numpy(h_total_retrieved)

        print("h_total_retrieved_numpy: ", h_total_retrieved_numpy)

        self.assertLess(
            np.max(abs(h_total_numpy - h_total_retrieved_numpy)), error_threshold
        )

    @lfr.runtime_scope
    def test_vof(self):

        num_cols: int = 100  # x direction size for layers' raster
        num_rows: int = 100  # z direction size for layers' raster

        array_shape = (num_rows, num_cols)
        partition_shape = 2 * (20,)

        nr_time_steps = 100
        dt = 0.05
        dx = 1
        dz = 1

        zero_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=0.0,
            partition_shape=partition_shape,
        )

        # create a hill topography data

        hill_center = (50, 30)  # (50, 20)  # (50, 50)

        h_total_initial_numpy = np.zeros(array_shape, dtype=np.float64)

        sigma_hill = 10  # 5
        amplitude_hill = 100

        for i_rows in range(num_rows):
            for i_cols in range(num_cols):
                h_total_initial_numpy[i_rows, i_cols] = amplitude_hill * np.exp(
                    -((i_cols - hill_center[1]) ** 2 + (i_rows - hill_center[0]) ** 2)
                    / (2 * sigma_hill**2)
                )

        h_mesh_step_value = 8

        h_total_initial_lue = convert_numpy_to_lue(
            h_total_initial_numpy, partition_shape
        )

        num_layers: int = int(
            np.round(np.max(h_total_initial_numpy) / h_mesh_step_value) + 1
        )

        print("num_layers: ", num_layers)

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

        print("boundary_type_numpy_default: \n", boundary_type_numpy_default)
        print("boundary_loc_numpy_default: \n", boundary_loc_numpy_default)

        print(" boundary_loc.shape : ", boundary_loc.shape)

        u_x_value: float = 2
        u_z_value: float = 0

        u_x_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=u_x_value,
            partition_shape=partition_shape,
        )

        u_z_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=u_z_value,
            partition_shape=partition_shape,
        )

        layer_list = []

        # Assign bed layer properties
        layer_list.append(
            Layer(
                u_x_lue,
                u_z_lue,
                None,
                zero_lue,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )

        # Assign internal layers properties
        for _ in range(1, num_layers):
            layer_list.append(
                Layer(
                    u_x_lue,
                    u_z_lue,
                    None,
                    zero_lue,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )

        # Assign surface layer properties
        layer_list.append(
            Layer(
                u_x_lue,
                u_z_lue,
                None,
                zero_lue,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )

        plt.contourf(h_total_initial_numpy)
        plt.colorbar()
        plt.title("h_total_initial_numpy")
        plt.show()

        h_mesh_assign(h_total_initial_lue, num_layers, h_mesh_step_value, layer_list)

        layer_5_h_mesh_initial_numpy = lfr.to_numpy(layer_list[5].h_mesh)

        plt.contourf(layer_5_h_mesh_initial_numpy)
        plt.colorbar()
        plt.title("layer_5_h_mesh_initial_numpy")
        plt.show()

        time: float = 0

        for _ in range(1, nr_time_steps + 1):
            time = time + dt

            for layer_id in range(1, num_layers):

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

        h_total_simulated_lue = calculate_total_h(layer_list)

        h_total_simulated_lue_numpy = lfr.to_numpy(h_total_simulated_lue)

        plt.contourf(h_total_simulated_lue_numpy)
        plt.colorbar()
        plt.title("h_total_simulated_lue_numpy")
        plt.show()

        layer_5_h_mesh_numpy = lfr.to_numpy(layer_list[5].h_mesh)

        plt.contourf(layer_5_h_mesh_numpy)
        plt.colorbar()
        plt.title("layer_5_h_mesh_numpy")
        plt.show()

        shift_mesh_num_hill = (
            int(np.round((u_z_value * time) / dz)),
            int(np.round((u_x_value * time) / dx)),
        )

        h_total_exact_numpy = h_exact_calculate(
            h=h_total_initial_numpy,
            ux=u_x_value,
            uz=u_z_value,
            time_iteration=nr_time_steps,
            dt=dt,
            dx=dx,
            dz=dz,
            value_boundary=0,
        )

        h_layer_5_exact_numpy = h_exact_calculate(
            h=layer_5_h_mesh_initial_numpy,
            ux=u_x_value,
            uz=u_z_value,
            time_iteration=nr_time_steps,
            dt=dt,
            dx=dx,
            dz=dz,
            value_boundary=0,
        )

        plt.contourf(h_total_exact_numpy)
        plt.colorbar()
        plt.title("h_total_exact_numpy")
        plt.show()

        plt.contourf(h_layer_5_exact_numpy)
        plt.colorbar()
        plt.title("h_layer_5_exact_numpy")
        plt.show()

        error_matrix_h_total = np.abs(h_total_exact_numpy - h_total_simulated_lue_numpy)

        max_relative_error_h_total = np.max(error_matrix_h_total) / amplitude_hill

        print("max_relative_error_h_total: ", max_relative_error_h_total)

        error_matrix_h_layer_5 = np.abs(h_layer_5_exact_numpy - layer_5_h_mesh_numpy)

        max_relative_error_h_layer_5 = (
            np.max(error_matrix_h_layer_5) / h_mesh_step_value
        )

        plt.contourf(error_matrix_h_layer_5)
        plt.colorbar()
        plt.title("error_matrix_h_layer_5")
        plt.show()

        print("max_relative_error_h_layer_5: ", max_relative_error_h_layer_5)

        center_hill_error_h_total = (
            abs(
                h_total_simulated_lue_numpy[
                    (
                        hill_center[0] + shift_mesh_num_hill[0],
                        hill_center[1] + shift_mesh_num_hill[1],
                    )
                ]
                - amplitude_hill
            )
            / amplitude_hill
        )

        print("center_hill_error_h_total: ", center_hill_error_h_total)

        print(
            "center_h_total_simulated: ",
            h_total_simulated_lue_numpy[
                (
                    hill_center[0] + shift_mesh_num_hill[0],
                    hill_center[1] + shift_mesh_num_hill[1],
                )
            ],
            "exact center h_total: ",
            amplitude_hill,
        )

        center_hill_error_layer_5 = (
            abs(
                layer_5_h_mesh_numpy[
                    (
                        hill_center[0] + shift_mesh_num_hill[0],
                        hill_center[1] + shift_mesh_num_hill[1],
                    )
                ]
                - h_mesh_step_value
            )
            / h_mesh_step_value
        )

        print(
            "center_layer_5_simulated: ",
            layer_5_h_mesh_numpy[
                (
                    hill_center[0] + shift_mesh_num_hill[0],
                    hill_center[1] + shift_mesh_num_hill[1],
                )
            ],
            "exact_center_layer_5: ",
            h_mesh_step_value,
        )

        print("center_hill_error_layer_5: ", center_hill_error_layer_5)

        error_threshold_max = 0.2
        error_threshold_center = 0.1

        self.assertLess(max_relative_error_h_total, error_threshold_max)
        # self.assertLess(max_relative_error_h_layer_5, error_threshold_max)
        self.assertLess(center_hill_error_h_total, error_threshold_center)
        # self.assertLess(center_hill_error_layer_5, error_threshold_center)


if __name__ == "__main__":
    unittest.main()
