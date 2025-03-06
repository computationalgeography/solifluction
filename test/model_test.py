import os
import os.path
import unittest

import lue.framework as lfr
import numpy as np

from source.solifluction import mass_conservation_2D, second_derivatives_in_y


def write_lue(lue_array, write_pathname_directory, file_name):
    if not os.path.exists(write_pathname_directory):
        os.makedirs(write_pathname_directory)

    full_pathname_exact = os.path.join(write_pathname_directory, file_name)
    lfr.to_gdal(lue_array, f"{full_pathname_exact}.tif")


def write(lue_array, write_pathname_directory, file_name, iteration):
    if not os.path.exists(write_pathname_directory):
        os.makedirs(write_pathname_directory)

    full_pathname_exact = os.path.join(write_pathname_directory, file_name)
    lfr.to_gdal(lue_array, "{}-{}.tif".format(full_pathname_exact, iteration))


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

        # layers order
        # --------------top
        # ------- layer 3
        # ------- layer 2
        # ------- layer 1
        # -------------bottom

        test_cases = [
            {
                # ay^2+by+c (a=2, b=5, c=layer_variable_2) ; for y=0 c=center_variable
                "name": "central_quadratic",
                "dy_layers_up": 100,
                "dy_layers_down": 500,
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
                "dy_layers_up": 100,
                "dy_layers_down": 500,
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
        pass


if __name__ == "__main__":
    unittest.main()
