#!/usr/bin/env python

import os
import os.path
from typing import Any

import lue.framework as lfr
import numpy as np
from numpy.typing import NDArray

# from typing import Type


def create_zero_numpy_array(
    num_grid_col: int,
    num_grid_row: int,
    num_virtual_layer: int = 1,
    dtype: np.dtype = np.float64,  # Default dtype set to float64
) -> NDArray[np.float64]:

    # num_grid_col : x direction
    # num_grid_row: z direction

    # default value for num_virtual_layer = int(1)
    num_grid_col = num_grid_col + int(2 * num_virtual_layer)
    num_grid_row = num_grid_row + int(2 * num_virtual_layer)

    array_numpy: NDArray[np.float64] = np.zeros(
        (num_grid_row, num_grid_col), dtype=dtype
    )

    print("---- create_zero_numpy_array is done ---------")

    return array_numpy


def convert_numpy_to_lue(
    numpy_array: NDArray[Any], partition_shape: tuple[int, int]
) -> Any:
    lue_array: Any = lfr.from_numpy(numpy_array, partition_shape=partition_shape)

    print("---- convert_numpy_to_lue is done ---------")

    return lue_array


def convert_lue_to_gdal(file_name, raster):

    written = lfr.to_gdal(raster, file_name)
    written.wait()


def read_lue_from_gdal(gdal_array, partition_shape):

    lue_array = lfr.from_gdal(gdal_array, partition_shape=partition_shape)

    return lue_array


def write(lue_array, write_pathname_directory, file_name, iteration):
    if not os.path.exists(write_pathname_directory):
        os.makedirs(write_pathname_directory)

    full_pathname_exact = os.path.join(write_pathname_directory, file_name)
    written = lfr.to_gdal(lue_array, "{}-{}.tif".format(full_pathname_exact, iteration))
    written.wait()


def default_boundary_type(
    num_cols: int, num_rows: int, boundary_in_first_last_row_col: bool = False
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    # boundary_in_first_last_row_col if the first and last rows and
    # columns are considered as boundaries this switch to True

    boundary_loc_numpy: NDArray[np.uint8] = create_zero_numpy_array(
        num_cols, num_rows, 0, np.uint8
    )
    boundary_type_numpy_default: NDArray[np.uint8] = create_zero_numpy_array(
        num_cols, num_rows, 0, np.uint8
    )

    if boundary_in_first_last_row_col:

        boundary_loc_numpy[0, :] = 1  # first row
        boundary_loc_numpy[-1, :] = 1  # last row
        boundary_loc_numpy[:, 0] = 1  # first column
        boundary_loc_numpy[:, -1] = 1  # last column

        # boundary_type_numpy_default is the default boundary type.
        # It is overwritten only if Dirichlet condition (type = 0) is considered

        boundary_type_numpy_default[:, 0] = 1
        boundary_type_numpy_default[0, :] = 4
        boundary_type_numpy_default[:, -1] = 3
        boundary_type_numpy_default[-1, :] = 2
        boundary_type_numpy_default[0, 0] = 5
        boundary_type_numpy_default[-1, 0] = 6
        boundary_type_numpy_default[-1, -1] = 7
        boundary_type_numpy_default[0, -1] = 8

    else:

        # boundary_type_numpy_default:
        # [[0 0 0 ... 0 0 0]
        # [0 5 4 ... 4 8 0]
        # [0 1 0 ... 0 3 0]
        # ...
        # [0 1 0 ... 0 3 0]
        # [0 6 2 ... 2 7 0]
        # [0 0 0 ... 0 0 0]]
        # boundary_loc_numpy_default:
        # [[1 1 1 ... 1 1 1]
        # [1 1 1 ... 1 1 1]
        # [1 1 0 ... 0 1 1]
        # ...
        # [1 1 0 ... 0 1 1]
        # [1 1 1 ... 1 1 1]
        # [1 1 1 ... 1 1 1]]

        boundary_loc_numpy = create_zero_numpy_array(num_cols, num_rows, 0, np.uint8)
        boundary_loc_numpy[1, 1:-1] = 1  # second row
        boundary_loc_numpy[-2, 1:-1] = 1  # Second-to-last row
        boundary_loc_numpy[1:-1, 1] = 1  # second column
        boundary_loc_numpy[1:-1, -2] = 1  # Second-to-last column

        # boundary_type_numpy_default is the default boundary type.
        # It is overwritten only if Dirichlet condition (type = 0) is considered

        boundary_type_numpy_default = create_zero_numpy_array(
            num_cols, num_rows, 0, np.uint8
        )
        boundary_type_numpy_default[1:-1, 1] = 1
        boundary_type_numpy_default[1, 1:-1] = 4
        boundary_type_numpy_default[1:-1, -2] = 3
        boundary_type_numpy_default[-2, 1:-1] = 2
        boundary_type_numpy_default[1, 1] = 5
        boundary_type_numpy_default[-2, 1] = 6
        boundary_type_numpy_default[-2, -2] = 7
        boundary_type_numpy_default[1, -2] = 8

        boundary_loc_numpy[[0, -1], :] = 1
        boundary_loc_numpy[:, [0, -1]] = 1

        # boundary type is zero on virtual outer layers (first and last rows and columns)
        # boundary_type_numpy_default[[0, -1], :] = 0
        # boundary_type_numpy_default[:, [0, -1]] = 0

    return boundary_type_numpy_default, boundary_loc_numpy


def read_run_setup(file_path: str) -> dict:
    variables: dict = {}

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file {file_path} not found.")

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace

            # Ignore empty lines or comments
            if not line or line.startswith("#"):
                continue

            # Split the line at '=' and handle assignment
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()  # Remove any extra whitespace around the key
                value = value.strip()  # Remove extra whitespace around value

                # Handle known types
                if key == "dt":
                    variables[key] = float(value)  # Convert dt to float
                elif key == "u_xfile":
                    variables[key] = value  # File path remains as string
                # Add more type checks here if needed

                # Add any default handling for other variables as needed.
                else:
                    variables[key] = value  # Store as string by default

    return variables
