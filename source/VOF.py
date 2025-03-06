import lue.framework as lfr
import numpy as np
from layer import Layer


def mass_conservation(variable, u_mesh, dt, dx):

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    flux_upstream = lfr.focal_sum((u_mesh * variable), kernel_im1_j)

    variable = variable - ((dt / dx) * (flux_upstream - (u_mesh * variable)))

    return variable


def mass_conservation_2D(
    var, u_x_mesh, u_z_mesh, dt, dx, dz, boundary_loc, boundary_value
) -> lfr.PartitionedArray<uint64, 2>:

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    # kernel_i_jm1   i, j-1
    kernel_i_jm1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # kernel_ip1_j   i+1, j
    kernel_ip1_j = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # kernel_i_jp1   i, j+1
    kernel_i_jp1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    # Upwind first order method

    flux_x_upstream = lfr.where(
        u_x_mesh >= 0,
        lfr.focal_sum((u_x_mesh * var), kernel_im1_j),
        lfr.focal_sum((u_x_mesh * var), kernel_ip1_j),
    )
    flux_z_upstream = lfr.where(
        u_z_mesh >= 0,
        lfr.focal_sum((u_z_mesh * var), kernel_i_jm1),
        lfr.focal_sum((u_z_mesh * var), kernel_i_jp1),
    )

    var_internal = (
        var
        + ((dt / dx) * (flux_x_upstream - (u_x_mesh * var)))
        + ((dt / dz) * (flux_z_upstream - (u_z_mesh * var)))
    )

    """ # Averaged flux or central method
    flux_x_upstream = lfr.where(
        u_x_mesh >= 0,
        lfr.focal_sum((u_x_mesh * var), kernel_im1_j),
        lfr.focal_sum((u_x_mesh * var), kernel_ip1_j),
    )
    flux_z_upstream = lfr.where(
        u_z_mesh >= 0,
        lfr.focal_sum((u_z_mesh * var), kernel_i_jm1),
        lfr.focal_sum((u_z_mesh * var), kernel_i_jp1),
    )

    flux_x_downstream = lfr.where(
        u_x_mesh >= 0,
        lfr.focal_sum((u_x_mesh * var), kernel_ip1_j),
        lfr.focal_sum((u_x_mesh * var), kernel_im1_j),
    )
    flux_z_downstream = lfr.where(
        u_z_mesh >= 0,
        lfr.focal_sum((u_z_mesh * var), kernel_i_jp1),
        lfr.focal_sum((u_z_mesh * var), kernel_i_jm1),
    )

    var_internal = (
        var
        + ((dt / (2 * dx)) * (flux_x_upstream - flux_x_downstream))
        + ((dt / (2 * dz)) * (flux_z_upstream - flux_z_downstream))
    )
    """

    var = lfr.where(
        boundary_loc,
        boundary_value,
        var_internal,
    )

    return var
