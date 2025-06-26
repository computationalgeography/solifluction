from typing import Any

import lue.framework as lfr
import numpy as np
from numpy.typing import NDArray

import source.derivatives_discretization as fdm
from source.boundary_condition import boundary_set

# Eq: d_phi/d_t + (u_x * d_phi/d_x) + (u_z * d_phi/d_z) + (nu_x * d2_phi/d_x2)
#     + (nu_z * d2_phi/d_z2) = rhs

# NOTE: boundary_type = (0 for Dirichlet and 1,2,3,4,5,6,7,8 for Neumann)

#     Dirichlet_boundary_type = 0
#
#     5---------------------boundary_type=4---------------------8
#     |                                                         |
#     |                                                         |
# boundary_type=1           Neumann_boundary                boundary_type=3
#     |                                                         |
#     |                                                         |
#     |                                                         |
#     6---------------------boundary_type=2---------------------7

"""
# momentum function is the general form of momentum solver but it has not been tested.
# It can be considered in the future. momentum_x function works well for now.

def momentum(
    phi,
    phase_state,
    dx: float,
    dz: float,
    dt: float,
    lue_u_x,
    lue_u_z,
    nu_x: float,
    nu_z: float,
    rhs,
    h_mesh,
    boundary_loc,
    boundary_type,
    Dirichlet_boundary_value,
    Neumann_boundary_value,
):

    phi_internal = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_upwind(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_upwind(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_central(phi, dx))
        - ((nu_z * dt) * fdm.d2z_central(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_1 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_forward(phi, dx))
        - ((lue_u_z * dt) * fdm.dz_upwind(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_forward(phi, dx))
        - ((nu_z * dt) * fdm.d2z_central(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_2 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_upwind(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_forward(phi, dz))
        - ((nu_x * dt) * fdm.d2x_central(phi, dx))
        - ((nu_z * dt) * fdm.d2z_forward(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_3 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_backward(phi, dx))
        - ((lue_u_z * dt) * fdm.dz_upwind(phi, dz, lue_u_z))
        - ((nu_x * dt) * fdm.d2x_backward(phi, dx))
        - ((nu_z * dt) * fdm.d2z_central(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_4 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_upwind(phi, dx, lue_u_x))
        - ((lue_u_z * dt) * fdm.dz_backward(phi, dz))
        - ((nu_x * dt) * fdm.d2x_central(phi, dx))
        - ((nu_z * dt) * fdm.d2z_backward(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_5 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_forward(phi, dx))
        - ((lue_u_z * dt) * fdm.dz_backward(phi, dz))
        - ((nu_x * dt) * fdm.d2x_forward(phi, dx))
        - ((nu_z * dt) * fdm.d2z_backward(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_6 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_forward(phi, dx))
        - ((lue_u_z * dt) * fdm.dz_forward(phi, dz))
        - ((nu_x * dt) * fdm.d2x_forward(phi, dx))
        - ((nu_z * dt) * fdm.d2z_forward(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_7 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_backward(phi, dx))
        - ((lue_u_z * dt) * fdm.dz_forward(phi, dz))
        - ((nu_x * dt) * fdm.d2x_backward(phi, dx))
        - ((nu_z * dt) * fdm.d2z_forward(phi, dz))
        + rhs,
        0,
    )

    phi_boundary_8 = lfr.where(
        (phase_state != 0) & (h_mesh > 0),
        phi
        - ((lue_u_x * dt) * fdm.dx_backward(phi, dx))
        - ((lue_u_z * dt) * fdm.dz_backward(phi, dz))
        - ((nu_x * dt) * fdm.d2x_backward(phi, dx))
        - ((nu_z * dt) * fdm.d2z_backward(phi, dz))
        + rhs,
        0,
    )

    # phase_state: 0 solid  --> (frozen soil), 1 --> (fluid or unfrozen),
    # now vegetation is ignored in phase_state but it is
    # considered in vegetation_vol_fraction

    phi = lfr.where(
        boundary_type == 1,
        phi_boundary_1,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 2,
        phi_boundary_2,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 3,
        phi_boundary_3,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 4,
        phi_boundary_4,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 5,
        phi_boundary_5,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 6,
        phi_boundary_6,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 7,
        phi_boundary_7,
        phi_internal,
    )

    phi = lfr.where(
        boundary_type == 8,
        phi_boundary_8,
        phi_internal,
    )

    phi = boundary_set(
        phi,
        boundary_loc,
        boundary_type,
        Dirichlet_boundary_value,
        Neumann_boundary_value,
        dx,
        dz,
    )

    return phi, phi_internal

    """


def momentum_ux(
    phi: Any,
    phase_state: Any,
    dx: float,
    dz: float,
    dt: float,
    lue_u_x: Any,
    lue_u_z: Any,
    nu_x: float,
    nu_z: float,
    rhs: Any,
    h_mesh: Any,
    boundary_loc: Any,
    boundary_type: Any,
    Dirichlet_boundary_value: Any,  # noqa: N803
    Neumann_boundary_value: Any,  # noqa: N803
) -> Any:

    # lue_u_x : phi
    # lue_u_y : 0
    # nu_x : +(mu_mesh/gama_soil_mesh)
    # rhs_g_sin : g*sin(alfa)
    # gama_prime_mesh: (gama_soil/cos(alfa))-(gama_water*cos(alfa))
    # rhs : g*sin(alfa) - ((gama_soil/cos(alfa))-(gama_water*cos(alfa)))*dh/dx

    # kernel_i_j = np.array(
    #     [
    #         [0, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 0],
    #     ],
    #     dtype=np.uint8,
    # )

    # kernel_im1_j   i-1, j
    kernel_im1_j: NDArray[np.uint8] = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    """
    # kernel_i_jm1   i, j-1
    kernel_i_jm1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    """

    # kernel_i_jm1   i, j-1
    kernel_i_jm1: NDArray[np.uint8] = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # kernel_ip1_j   i+1, j
    kernel_ip1_j: NDArray[np.uint8] = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    """
    # kernel_i_jp1   i, j+1
    kernel_i_jp1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    """

    # kernel_i_jp1   i, j+1
    kernel_i_jp1: NDArray[np.uint8] = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    # coeff_map_i_j = (
    #     1
    #     + (-(dt / dx) * lfr.abs(lue_u_x))
    #     + (-(dt / dy) * lfr.abs(lue_u_y))
    #     + (2 * (dt / (dx**2)) * nu_x)
    #     + (2 * (dt / (dy**2)) * nu_y)
    # )

    coeff_map_i_j: Any = (
        1  # check this
        + (-(dt / dx) * lfr.abs(lue_u_x))
        + (-(dt / dz) * lfr.abs(lue_u_z))
        + (2 * (dt / (dx**2)) * nu_x)
        + (2 * (dt / (dz**2)) * nu_z)
    )

    coeff_map_im1_j: Any = lfr.where(
        lue_u_x >= 0,
        ((dt / dx) * lfr.abs(lue_u_x)) + (-(dt / (dx**2)) * nu_x),
        (-(dt / (dx**2)) * nu_x),
    )

    coeff_map_ip1_j: Any = lfr.where(
        lue_u_x < 0,
        ((dt / dx) * lfr.abs(lue_u_x)) + (-(dt / (dx**2)) * nu_x),
        (-(dt / (dx**2)) * nu_x),
    )

    coeff_map_i_jm1: Any = lfr.where(
        lue_u_z >= 0,
        ((dt / dz) * lfr.abs(lue_u_z)) + (-(dt / (dz**2)) * nu_z),
        (-(dt / (dz**2)) * nu_z),
    )

    coeff_map_i_jp1: Any = lfr.where(
        lue_u_z < 0,
        ((dt / dz) * lfr.abs(lue_u_z)) + (-(dt / (dz**2)) * nu_z),
        (-(dt / (dz**2)) * nu_z),
    )

    # NOTE: coeff_<.> should be implemented on boundaries, for instance on boundary_tpe
    # 1 (left boundary) phi_i-1,j is located outside of domain and we need
    # forward discretization
    # For now this implementation is ignored as we impose certain boundary conditions
    # on the boundaries which overwrite phi on the boundaries but in the future
    # this should be considered and for each boundary use exclusive discretization

    phi_all_internal_domain_i_j: Any = (
        (coeff_map_i_j * phi)  # (coeff_map_i_j * lfr.focal_sum(solution, kernel_i_j))
        + (coeff_map_im1_j * lfr.focal_sum(phi, kernel_im1_j))
        + (coeff_map_i_jm1 * lfr.focal_sum(phi, kernel_i_jm1))
        + (coeff_map_ip1_j * lfr.focal_sum(phi, kernel_ip1_j))
        + (coeff_map_i_jp1 * lfr.focal_sum(phi, kernel_i_jp1))
        + (dt * rhs)
    )

    # phase_state: 0 solid  --> (frozen soil), 1 --> (fluid or unfrozen), now vegetation
    # is ignored in phase_state but it is considered in vegetation_vol_fraction

    phi_internal: Any = lfr.where(
        (phase_state != 0) & (h_mesh > 0),  # fluid or unfrozen
        phi_all_internal_domain_i_j,
        0,
    )

    phi = boundary_set(
        phi_internal,
        boundary_loc,
        boundary_type,
        Dirichlet_boundary_value,
        Neumann_boundary_value,
        dx,
        dz,
    )

    # ----------------

    # phi_u_x_numpy = lfr.to_numpy(phi)
    # print(
    #     "inside momentum_ux function phi_u_x_numpy[10,10]",
    #     phi_u_x_numpy[10, 10],
    # )

    # rhs_numpy = lfr.to_numpy(rhs)
    # print(
    #     "inside momentum_ux function rhs_numpy[10,10]",
    #     rhs_numpy[10, 10],
    # )

    return phi
