import lue.framework as lfr
import numpy as np

from source.boundary_condition import boundary_set


def mass_conservation_2D_vof(
    phi,
    u_x_mesh,
    u_z_mesh,
    dt,
    dx,
    dz,
    boundary_loc,
    boundary_type,
    Dirichlet_boundary_value,
    Neumann_boundary_value,
):
    """
    This function solves 2D mass conservation in each layer.


    """

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    # kernel_i_jm1   i, j-1    # check this ???
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

    # kernel_i_jp1   i, j+1      # check this ???
    kernel_i_jp1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    # Upwind first order method
    # It is assumed that var>=0

    flux_x_upstream = lfr.where(
        u_x_mesh >= 0,
        lfr.focal_sum((u_x_mesh * phi), kernel_im1_j),
        lfr.focal_sum((u_x_mesh * phi), kernel_ip1_j),
    )
    flux_z_upstream = lfr.where(
        u_z_mesh >= 0,
        lfr.focal_sum((u_z_mesh * phi), kernel_i_jm1),
        lfr.focal_sum((u_z_mesh * phi), kernel_i_jp1),
    )

    # var_internal = (
    #     var
    #     + ((dt / dx) * (flux_x_upstream - (u_x_mesh * var)))
    #     + ((dt / dz) * (flux_z_upstream - (u_z_mesh * var)))
    # )

    phi_internal = (
        phi
        + ((dt / dx) * (lfr.abs(flux_x_upstream) - (lfr.abs(u_x_mesh) * phi)))
        + ((dt / dz) * (lfr.abs(flux_z_upstream) - (lfr.abs(u_z_mesh) * phi)))
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

    # net_flux = flux_x_upstream - (u_x_mesh * phi)
    # return phi, flux_x_upstream, net_flux

    phi = boundary_set(
        phi_internal,
        boundary_loc,
        boundary_type,
        Dirichlet_boundary_value,
        Neumann_boundary_value,
        dx,
        dz,
    )

    return phi


def calculate_total_h(layer_list):

    h_total = layer_list[0].h_mesh

    for i in range(1, len(layer_list)):
        h_total = h_total + layer_list[i].h_mesh

    return h_total


# source function h_mesh_assign
# def h_mesh_assign(h_total, num_layers, prespecified_uniform_h_mesh_value, layer_list):

#     h_total_remain = h_total

#     layer_list[0].h_mesh = lfr.where(
#         h_total_remain < prespecified_uniform_h_mesh_value,
#         h_total_remain,
#         prespecified_uniform_h_mesh_value,
#     )

#     h_total_remain = h_total_remain - prespecified_uniform_h_mesh_value

#     h_total_remain = lfr.where(
#         h_total_remain < 0,
#         0,
#         h_total_remain,
#     )

#     h_mesh_numpy = lfr.to_numpy(layer_list[0].h_mesh)

#     h_total_remain_numpy = lfr.to_numpy(h_total_remain)

#     print(
#         "h_mesh_numpy_0: \n",
#         h_mesh_numpy,
#     )

#     print(
#         "h_total_remain_numpy_0: \n",
#         h_total_remain_numpy,
#     )

#     # input(" enter key to continue ...")

#     for i in range(1, num_layers):

#         layer_list[i].h_mesh = lfr.where(
#             h_total_remain < prespecified_uniform_h_mesh_value,
#             h_total_remain,
#             prespecified_uniform_h_mesh_value,
#         )

#         h_total_remain = h_total_remain - prespecified_uniform_h_mesh_value

#         h_total_remain = lfr.where(
#             h_total_remain < 0,
#             0,
#             h_total_remain,
#         )

#         h_mesh_numpy = lfr.to_numpy(layer_list[i].h_mesh)

#         h_total_remain_numpy = lfr.to_numpy(h_total_remain)

#         print(
#             f"h_mesh_numpy_{i}: \n",
#             h_mesh_numpy,
#         )

#         print(
#             f"h_total_remain_numpy_{i}: \n",
#             h_total_remain_numpy,
#         )

#         # input(" enter key to continue ...")


# # def h_mesh_assign_1(remained_h_total_to_assign, prespecified_uniform_h_mesh_value):

# #     remained_h_total_to_assign = (
# #         remained_h_total_to_assign - prespecified_uniform_h_mesh_value
# #     )

# #     h_mesh = lfr.where(
# #         remained_h_total_to_assign < 0,
# #         remained_h_total_to_assign + prespecified_uniform_h_mesh_value,
# #         prespecified_uniform_h_mesh_value,
# #     )

# #     return h_mesh, remained_h_total_to_assign

# tested old version
# def h_mesh_assign(h_total, num_layers, prespecified_uniform_h_mesh_value, layer_list):

#     h_total_remain = h_total

#     layer_list[0].h_mesh = lfr.where(
#         h_total_remain < prespecified_uniform_h_mesh_value,
#         h_total_remain,
#         prespecified_uniform_h_mesh_value,
#     )

#     h_total_remain = h_total_remain - prespecified_uniform_h_mesh_value

#     h_total_remain = lfr.where(
#         h_total_remain < 0,
#         0,
#         h_total_remain,
#     )

#     for i in range(1, num_layers):

#         layer_list[i].h_mesh = lfr.where(
#             h_total_remain < prespecified_uniform_h_mesh_value,
#             h_total_remain,
#             prespecified_uniform_h_mesh_value,
#         )

#         h_total_remain = h_total_remain - prespecified_uniform_h_mesh_value

#         h_total_remain = lfr.where(
#             h_total_remain < 0,
#             0,
#             h_total_remain,
#         )


def h_mesh_assign(h_total, num_layers, prespecified_uniform_h_mesh_value):

    h_total_remain = h_total
    h_mesh_list = []

    h_mesh = lfr.where(
        h_total_remain < prespecified_uniform_h_mesh_value,
        h_total_remain,
        prespecified_uniform_h_mesh_value,
    )
    h_mesh_list.append(h_mesh)

    h_total_remain = h_total_remain - prespecified_uniform_h_mesh_value
    h_total_remain = lfr.where(h_total_remain < 0, 0, h_total_remain)

    for _ in range(1, num_layers):
        h_mesh = lfr.where(
            h_total_remain < prespecified_uniform_h_mesh_value,
            h_total_remain,
            prespecified_uniform_h_mesh_value,
        )
        h_mesh_list.append(h_mesh)

        h_total_remain = h_total_remain - prespecified_uniform_h_mesh_value
        h_total_remain = lfr.where(h_total_remain < 0, 0, h_total_remain)

    return h_mesh_list
