import unittest

import lue.framework as lfr
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from source.io_data_process import convert_numpy_to_lue, create_zero_numpy_array, write
from source.solifluction import (
    Layer,
    mass_conservation_2D,
    momentum_ux,
    second_derivatives_in_y,
)

# def plot_contour(data_array, title: str):
#     plt.contour(data_array, levels=10, cmap="viridis")
#     plt.colorbar()
#     plt.title(title)
#     plt.show()


def plot_gdal_contours(array):
    # Open the raster file
    ds = gdal.Open(array)

    # Read the data as an array
    data = ds.GetRasterBand(1).ReadAsArray()

    # Plot the contour
    plt.figure(figsize=(5, 5))
    contour = plt.contour(data, levels=10, cmap="viridis")
    plt.colorbar(contour)
    plt.title("Contour Plot of Raster Data")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()


class TestPackage(unittest.TestCase):

    @lfr.runtime_scope
    def test_mass_conservation_2D_test(self):

        partition_shape = 2 * (20,)

        h_mesh = lfr.from_gdal("test/h_test.tif", partition_shape=partition_shape)
        u_x_mesh = lfr.from_gdal("test/u_test_1.tif", partition_shape=partition_shape)
        u_z_mesh = lfr.from_gdal("test/u_test_0.tif", partition_shape=partition_shape)
        h_mesh_exact = lfr.from_gdal(
            "test/h_exact_ux1_uz0.tif", partition_shape=partition_shape
        )

        boundary_loc = lfr.from_gdal(
            "test/boundary_loc.tif", partition_shape=partition_shape
        )

        boundary_value = h_mesh  # initial h is set to be boundary_value

        dt = 0.1
        dx = 1
        dz = 1

        # time_end = 10
        time_iteration = 1000  # int(time_end / dt)
        plot_it: int = 100

        for it_time in range(1, time_iteration + 1):

            h_mesh, flux_x_upstream, net_flux = mass_conservation_2D(
                h_mesh, u_x_mesh, u_z_mesh, dt, dx, dz, boundary_loc, boundary_value
            )

            if it_time % plot_it == 0:
                write(h_mesh, "test", "h_results_test", it_time)
                write(flux_x_upstream, "test", "flux_x_upstream", it_time)
                write(net_flux, "test", "net_flux", it_time)

        error_matrix = lfr.abs(h_mesh_exact - h_mesh)
        written = lfr.to_gdal(
            error_matrix, "test/error_matrix_mass_conservation_2D_test.tif"
        )
        written.wait()

        # error = lfr.maximum(lfr.abs(h_mesh_exact - h_mesh))
        # print("error: ", error.future())

        error_matrix_numpy = lfr.to_numpy(error_matrix)
        print("max error: ", np.max(error_matrix_numpy))
        error_threshold = 0.05
        self.assertLess(max(error_matrix_numpy), error_threshold)

    """
    @lfr.runtime_scope
    def test_second_derivatives_in_y(self):

        # test second derivatives of function ay^2+by+c

        partition_shape = 2 * (20,)

        dy_layers_up = 100  # 2.5
        dy_layers_down = 500  # 5  # 2.5

        b = 5
        a = 2

        exact_d2var_dy2 = 2 * a

        layer_variable_center = lfr.from_gdal(
            "test/h_test.tif", partition_shape=partition_shape
        )  # c in ay^2+by+c

        layer_variable_down = (
            (a * (dy_layers_down**2)) + (b * -dy_layers_down) + layer_variable_center
        )
        layer_variable_up = (
            (a * (dy_layers_up**2)) + (b * dy_layers_up) + layer_variable_center
        )

        d2var_dy2 = second_derivatives_in_y(
            layer_variable_center,
            layer_variable_up,
            layer_variable_down,
            dy_layers_up,
            dy_layers_down,
        )

        error_matrix = lfr.abs(d2var_dy2 - exact_d2var_dy2)

        written = lfr.to_gdal(
            error_matrix, "test/error_matrix_second_derivatives_in_y.tif"
        )
        written.wait()

        d2var_dy2_numpy = lfr.to_numpy(d2var_dy2)
        print("d2var_dy2_numpy: ", d2var_dy2_numpy)

        error_matrix_numpy = lfr.to_numpy(error_matrix)
        print(error_matrix_numpy)
        print("max error: ", np.max(error_matrix_numpy))
        print(f"max error: {np.max(error_matrix_numpy):.15f}")
        print("error_matrix_numpy [50,50]: ", error_matrix_numpy[50, 50])

        error_threshold = 1
        self.assertLess(np.max(error_matrix_numpy), error_threshold)

    """

    @lfr.runtime_scope
    def test_second_derivatives_in_y(self):
        """Test second derivatives (central,forward, and backward) for different types of functions."""

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
    def test_momentum_ux(self):

        time = 0
        dt = 1
        nr_time_steps = 10
        num_layers = 5

        mu = 10**-2
        density_soil = 2650

        h_mesh_layer = 20

        num_cols: int = 200  # x direction size for layers' raster
        num_rows: int = 100  # z direction size for layers' raster

        dx = 1
        dz = 1

        array_shape = (
            num_rows,
            num_cols,
        )
        partition_shape = 2 * (20,)

        # Initial condition definition
        zero_numpy_array = create_zero_numpy_array(num_cols, num_rows, 0, np.uint8)
        boundary_loc_numpy = zero_numpy_array
        boundary_loc_numpy[0, :] = 1
        boundary_loc_numpy[-1, :] = 1
        boundary_loc_numpy[:, 0] = 1
        boundary_loc_numpy[:, -1] = 1
        boundary_type_numpy = zero_numpy_array
        boundary_type_numpy[:, 0] = 1
        boundary_type_numpy[0, :] = 4
        boundary_type_numpy[:, -1] = 3
        boundary_type_numpy[-1, :] = 2

        boundary_loc = convert_numpy_to_lue(
            boundary_loc_numpy, partition_shape=partition_shape
        )

        boundary_type = convert_numpy_to_lue(
            boundary_type_numpy, partition_shape=partition_shape
        )

        print(" boundary_loc.shape : ", boundary_loc.shape)

        Dirichlet_boundary_value = zero_numpy_array
        Neumann_boundary_value = zero_numpy_array

        zero_array_lue = lfr.create_array(
            array_shape,
            dtype=np.float64,
            fill_value=0.0,
            partition_shape=partition_shape,
        )

        Dirichlet_boundary_value = zero_array_lue
        Neumann_boundary_value = zero_array_lue

        # phase_state: 0 solid  --> (frozen soil), 1 --> (fluid or unfrozen), now vegetation is ignored in phase_state but it is considered in vegetation_vol_fraction
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

        # dh_dx = zero_array_lue

        g_sin = 9.81 * np.sin(np.pi / 6)

        # End: Initial condition definition

        # instantiate Layer objects for all layers

        Layer_list = []

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
            )
        )

        # End: instantiate Layer objects for all layers

        for time_step in range(1, nr_time_steps + 1):

            time = time + dt

            # velocity at bed layer (layer_id = 0) is zero
            for layer_id in range(1, num_layers):

                # calculate du2_dy2 for the right hand side of momentum (velocity) equation

                if layer_id == 0:  # bed layer

                    d2u_x_dy2 = second_derivatives_in_y(
                        Layer_list[1].u_x,
                        Layer_list[2].u_x,
                        Layer_list[0].u_x,
                        h_mesh,
                        h_mesh,
                    )

                elif layer_id == num_layers - 1:  # surface layer

                    d2u_x_dy2 = second_derivatives_in_y(
                        Layer_list[num_layers - 2].u_x,
                        Layer_list[num_layers - 1].u_x,
                        Layer_list[num_layers - 3].u_x,
                        h_mesh,
                        h_mesh,
                    )

                else:

                    print("layer_id :", layer_id)
                    print(
                        "Layer_list[layer_id].u_x.dtype: ",
                        Layer_list[layer_id].u_x.dtype,
                    )
                    print(
                        "Layer_list[layer_id + 1].u_x.dtype: ",
                        Layer_list[layer_id + 1].u_x.dtype,
                    )
                    print(
                        "Layer_list[layer_id - 1].u_x.dtype: ",
                        Layer_list[layer_id - 1].u_x.dtype,
                    )

                    d2u_x_dy2 = second_derivatives_in_y(
                        Layer_list[layer_id].u_x,
                        Layer_list[layer_id + 1].u_x,
                        Layer_list[layer_id - 1].u_x,
                        h_mesh,
                        h_mesh,
                    )

                # momentum in x direction for velocity calculation

                rhs = g_sin - ((mu_array_lue / density_soil_lue) * d2u_x_dy2)

                Layer_list[layer_id].u_x, phi_internal = momentum_ux(
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
                    Dirichlet_boundary_value,
                    Neumann_boundary_value,
                )

                print("layer_id: ", layer_id)

                # print(
                #     "Layer_list[layer_id].u_x.dtype: ",
                #     Layer_list[layer_id].u_x.dtype,
                # )

                # plot_contour(rhs, "rhs")

                # numpy_u_x = lfr.to_numpy(Layer_list[layer_id].u_x)
                # plot_contour(numpy_u_x, f"layre_{layer_id}")

                write(rhs, "test", "rhs", 0)
                plot_gdal_contours("rhs-0.tif")
                input("Press Enter to continue ...")

            write(Layer_list[0].u_x, "test", "u_x_layer", 0)
            write(Layer_list[1].u_x, "test", "u_x_layer", 1)
            write(Layer_list[2].u_x, "test", "u_x_layer", 2)
            write(Layer_list[3].u_x, "test", "u_x_layer", 3)
            write(Layer_list[4].u_x, "test", "u_x_layer", 4)
            write(phi_internal, "test", "phi_internal", 4)

            print("time_step: ", time_step)

        # write(Layer_list[0].u_x, "test", "u_x_layer_", 0)
        # write(Layer_list[1].u_x, "test", "u_x_layer_", 1)
        # write(Layer_list[2].u_x, "test", "u_x_layer_", 2)
        # write(Layer_list[3].u_x, "test", "u_x_layer_", 3)
        # write(Layer_list[4].u_x, "test", "u_x_layer_", 4)


if __name__ == "__main__":
    unittest.main()
