#!/usr/bin/env python

import os
import os.path
from typing import TYPE_CHECKING, Any

import lue.framework as lfr
import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray
from osgeo import gdal

if TYPE_CHECKING:
    from configparser import ConfigParser

from pathlib import Path

from source.config import load_config
from source.layer import Layer

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


# def read_run_setup(file_path: str) -> dict:
#     variables: dict = {}

#     # Check if file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Configuration file {file_path} not found.")

#     with open(file_path, "r") as file:
#         for line in file:
#             line = line.strip()  # Remove leading/trailing whitespace

#             # Ignore empty lines or comments
#             if not line or line.startswith("#"):
#                 continue

#             # Split the line at '=' and handle assignment
#             if "=" in line:
#                 key, value = line.split("=", 1)
#                 key = key.strip()  # Remove any extra whitespace around the key
#                 value = value.strip()  # Remove extra whitespace around value

#                 # Handle known types
#                 if key == "dt":
#                     variables[key] = float(value)  # Convert dt to float
#                 elif key == "u_xfile":
#                     variables[key] = value  # File path remains as string
#                 # Add more type checks here if needed

#                 # Add any default handling for other variables as needed.
#                 else:
#                     variables[key] = value  # Store as string by default

#     return variables


def load_daily_temperatures(csv_path: str) -> tuple[list[float], list[float]]:
    df = pd.read_csv(csv_path)
    days = df["day"].values  # Day number (0, 1, 2, ...)
    temps = df["temperature"].values  # temperature value
    return days, temps


def read_config_file(param_path: Path) -> tuple[
    int,  # number_of_iterations
    float,  # dt_momentum
    int,  # momentum_iteration_threshold
    float,  # dt_global_model
    float,  # dt_heat_transfer
    float,  # dt_mass_conservation
    float,  # time_end_simulation
    float,  # write_intervals_time
    int,  # partition_shape_size
    float,  # h_mesh_step_value
    float,  # mu_value
    float,  # density_value
    float,  # k_conductivity_value
    float,  # rho_c_heat_value
    str,  # h_total_initial_file
    bool,  # heat_transfer_warmup
    int,  # heat_transfer_warmup_iteration
    list[float],  # days_temperature_file
    list[float],  # temps_temperature_file
    str,  # results_pathname
    float,  # slope_radian
]:

    input_variables: ConfigParser = load_config(param_path)

    def clean_float(val: str) -> float:
        return float(val.split("#")[0].strip())

    def clean_int(val: str) -> int:
        return int(val.split("#")[0].strip())

    dt_momentum = clean_float(input_variables["simulation"]["dt_momentum"])
    dt_heat_transfer = clean_float(input_variables["simulation"]["dt_heat_transfer"])
    dt_mass_conservation = clean_float(
        input_variables["simulation"]["dt_mass_conservation"]
    )
    dt_global_model = clean_float(input_variables["simulation"]["dt_global_model"])

    number_of_iterations = clean_int(
        input_variables["simulation"]["number_of_iterations"]
    )

    momentum_iteration_threshold = clean_int(
        input_variables["simulation"]["momentum_iteration_threshold"]
    )

    time_end_simulation = clean_float(
        input_variables["simulation"]["time_end_simulation"]
    )

    write_intervals_time = clean_float(
        input_variables["simulation"]["write_intervals_time"]
    )

    partition_shape_size = clean_int(input_variables["grid"]["partition_shape_size"])
    h_mesh_step_value = clean_float(input_variables["grid"]["initial_layer_size"])
    h_mesh_step_value = np.float64(h_mesh_step_value)

    mu_value = clean_float(input_variables["material"]["uniform_mu"])
    density_value = clean_float(input_variables["material"]["uniform_density"])
    k_conductivity_value = clean_float(
        input_variables["material"]["uniform_k_conductivity"]
    )
    rho_c_heat_value = clean_float(input_variables["material"]["uniform_rho_c_heat"])

    h_total_initial_file = input_variables["initial_condition"][
        "h_total_initial_file"
    ].strip()

    heat_transfer_warmup = input_variables.getboolean(
        "simulation", "heat_transfer_warmup", fallback=False
    )
    heat_transfer_warmup_iteration = input_variables.getint(
        "simulation", "heat_transfer_warmup_iteration", fallback=200
    )

    try:
        temperature_file_csv = input_variables["material"]["temperature_file"].strip()
        days_temperature_file, temps_temperature_file = load_daily_temperatures(
            temperature_file_csv
        )
    except (ValueError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error loading temperature file: {e}")
        exit(1)

    try:
        results_pathname = input_variables["simulation"]["results_path"].strip()
    except KeyError as err:
        raise ValueError(f"Missing results path: {err}") from err

    slope_radian = clean_float(input_variables["simulation"]["alfa_slope"])
    # g_sin = np.sin(slope_radian) * 9.81

    return (
        number_of_iterations,
        dt_momentum,
        momentum_iteration_threshold,
        dt_global_model,
        dt_heat_transfer,
        dt_mass_conservation,
        time_end_simulation,
        write_intervals_time,
        partition_shape_size,
        h_mesh_step_value,
        mu_value,
        density_value,
        k_conductivity_value,
        rho_c_heat_value,
        h_total_initial_file,
        heat_transfer_warmup,
        heat_transfer_warmup_iteration,
        days_temperature_file,
        temps_temperature_file,
        results_pathname,
        slope_radian,
    )


def read_tif_info_from_gdal(
    tif_file: str,
) -> tuple[float, float, tuple[int, int], float]:

    print("read_tif_info_from_gdal in run")

    dataset = gdal.Open(tif_file)
    num_cols = dataset.RasterXSize
    num_rows = dataset.RasterYSize

    print(f"num_cols: {num_cols}")
    print(f"num_rows: {num_rows}")

    geotransform = dataset.GetGeoTransform()

    dx = geotransform[1]  # Pixel width (Δx)
    dz = abs(
        geotransform[5]
    )  # Pixel height (Δz) — take abs because it may be negative (north-up images)

    print(f"Pixel size dx: {dx}")
    print(f"Pixel size dy: {dz}")

    raster_array = dataset.ReadAsArray()

    max_h_total = np.max(raster_array)
    print(f"max_h_total: {max_h_total}")

    array_shape = (
        num_rows,
        num_cols,
    )

    return dx, dz, array_shape, max_h_total


def initiate_layers_variables(
    max_h_total: float,
    bed_depth_elevation: float,
    h_mesh_step_value: float,
    array_shape: tuple[int, int],
    partition_shape: tuple[int, int],
    h_total_initial_file: str,
    mu_value: float,
    density_value: float,
    k_conductivity_value: float,
    rho_c_heat_value: float,
) -> tuple[list[Layer], list[Any], Any, int, Any]:

    print("initiate_initial_layers_variables is in run")

    num_layers: int = int(
        np.ceil((max_h_total - bed_depth_elevation) / h_mesh_step_value)
        # np.ceil(max_h_total - bed_depth_elevation) / h_mesh_step_value + 1
    )

    print("max_h_total: ", max_h_total)
    print("h_mesh_step_value: ", h_mesh_step_value)
    print("bed_depth_elevation: ", bed_depth_elevation)

    print("num_layers: ", num_layers)

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

    temperature_bed = zero_array_lue
    # temperature_surface_initial = zero_array_lue

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

    # initial_layer_variables: Layer = Layer(
    #     initial_u_x,
    #     initial_u_z,
    #     initial_temperature,
    #     initial_h_mesh,
    #     initial_mu_soil,
    #     initial_density_soil,
    #     initial_phase_state,
    #     initial_k_conductivity_heat,
    #     initial_rho_c_heat,
    #     initial_vegetation_vol_fraction,
    # )

    # NOTE: number of layers is 0 to "num_layers" for bed layer to surface layer

    # Assign bed layer properties

    layer_list: list[Layer] = []

    layer_list.append(
        Layer(
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
    )

    # Assign internal layers properties

    for i in range(1, num_layers):
        layer_list.append(
            Layer(
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
        )

    # Assign surface layer properties

    layer_list.append(
        Layer(
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
    )

    d2u_x_dy2_initial = []

    for _ in range(num_layers):
        d2u_x_dy2_initial.append(zero_array_lue)

    return (
        layer_list,
        d2u_x_dy2_initial,
        h_total_initial,
        num_layers,
        temperature_bed,
    )


def write_tif_file(
    array: Any, output_file_name: str, time_label: float, output_folder_path: str
) -> None:
    """Write a lue array to a .tif file using lfr.to_gdal."""

    os.makedirs(output_folder_path, exist_ok=True)

    output_path = os.path.join(
        output_folder_path, f"{output_file_name}-{time_label}.tif"
    )

    lfr.to_gdal(array, output_path)


# def save_tif_file(
#     array: Any, output_file_name: str, time_label: float, output_folder_path: Path
# ) -> None:
#     """Write a lue array to a .tif file using lfr.to_gdal."""

#     os.makedirs(output_folder_path, exist_ok=True)

#     output_path: str = os.path.join(
#         output_folder_path, f"{output_file_name}-{time_label}.tif"
#     )

#     lfr.to_gdal(array, output_path)


def save_u_x_tem_time(u_x_tem_time, filename="u_x_tem_time.csv") -> None:
    """
    Save simulation results surface u_x and temperature in time to CSV file.

    Parameters
    ----------
    filename : str
        Output CSV file name.
    """
    data = np.array(u_x_tem_time)

    print("Data shape:", data.shape)
    print("Data type:", data.dtype)

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="time,surface_temp,u_x_50_50",
        comments="",
        fmt="%.10e",
    )
