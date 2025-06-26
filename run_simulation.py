import sys
from pathlib import Path

from source.io_data_process import read_config_file, read_tif_info_from_gdal
from source.solifluction import solifluction_simulate


def main() -> None:

    if len(sys.argv) > 1:
        param_path = Path(sys.argv[1]).resolve()
    else:
        param_path = Path("param.txt").resolve()

    if not param_path.is_file():
        print(f"Parameter file does not exist: {param_path}")
        sys.exit(1)

    # -----  read input variables from param.txx ---------------

    (
        dt_momentum,
        momentum_iteration_threshold,
        dt_heat_transfer,
        dt_mass_conservation,
        time_end_simulation,
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
        g_sin,
    ) = read_config_file(param_path)

    # ---------------------  initial information --------------------

    dx, dz, array_shape, max_h_total = read_tif_info_from_gdal(h_total_initial_file)
    num_rows, num_cols = array_shape

    partition_shape: tuple[int, int] = 2 * (partition_shape_size,)

    bed_depth_elevation = 0  # it can be any value

    nu_x: float = 0
    nu_z: float = 0
    # this is viscosity in raster dimension (dx,dz)
    # for instance in momentum equation coefficient in
    # nu_x *d^2u/dx^2 and nu_z *d^2u/dz^2
    # diffusion term effect in gravity direction (y) (d^2u/dy^2)
    #  is considered in rhs of momentum function

    # ---------------------  initial information --------------------

    solifluction_simulate(
        dx,
        dz,
        num_cols,
        num_rows,
        max_h_total,
        bed_depth_elevation,
        h_total_initial_file,
        mu_value,
        density_value,
        k_conductivity_value,
        rho_c_heat_value,
        dt_momentum,
        dt_mass_conservation,
        dt_heat_transfer,
        momentum_iteration_threshold,
        time_end_simulation,
        heat_transfer_warmup,
        heat_transfer_warmup_iteration,
        h_mesh_step_value,
        g_sin,
        nu_x,
        nu_z,
        days_temperature_file,
        temps_temperature_file,
        partition_shape,
        results_pathname,
    )


if __name__ == "__main__":
    main()
