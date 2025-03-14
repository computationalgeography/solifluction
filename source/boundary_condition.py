# This module sets up the boundary conditions based on their type: Dirichlet and Neumann boundaries.
# For Neumann boundaries, only the values normal to the boundary are implemented

import lue.framework as lfr
import numpy as np


def boundary_set(
    phi,
    boundary_loc,
    boundary_type,
    Dirichlet_boundary_value,
    Neumann_boundary_value,
    dx,
    dz,
):

    # NOTE: boundary_type = (0 for Dirichlet and 1,2,3,4 for Neumann)

    #     Dirichlet_boundary_type = 0
    #
    #     ---------------------boundary_type=4----------------------
    #     |                                                         |
    #     |                                                         |
    # boundary_type=1           Neumann_boundary                boundary_type=3
    #     |                                                         |
    #     |                                                         |
    #     |                                                         |
    #     ---------------------boundary_type=2----------------------

    # Dirichlet boundary imposing

    phi = lfr.where(
        ((boundary_loc == 1) & (boundary_type == 0)),
        Dirichlet_boundary_value,
        phi,
    )

    # kernel_im1_j   i-1, j
    kernel_im1_j = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # kernel_i_jm1   i, j-1    # Check it. It is changed compared to advection-diffusion  test model.
    kernel_i_jm1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
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

    # kernel_i_jp1   i, j+1     # Check it. It is changed compared to advection-diffusion  test model.
    kernel_i_jp1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    # Neumann boundary imposing

    # phi = lfr.where(
    #     ((boundary_loc == 1) & (boundary_type == 1)),
    #     lfr.focal_sum(phi, kernel_ip1_j) - (dx * Neumann_boundary_value),
    #     lfr.where(
    #         ((boundary_loc == 1) & (boundary_type == 2)),
    #         lfr.focal_sum(phi, kernel_i_jp1) - (dz * Neumann_boundary_value),
    #         lfr.where(
    #             ((boundary_loc == 1) & (boundary_type == 3)),
    #             lfr.focal_sum(phi, kernel_im1_j) + (dx * Neumann_boundary_value),
    #             lfr.where(
    #                 ((boundary_loc == 1) & (boundary_type == 4)),
    #                 lfr.focal_sum(phi, kernel_i_jm1) + (dz * Neumann_boundary_value),
    #                 phi,
    #             ),
    #         ),
    #     ),
    # )

    phi = lfr.where(
        ((boundary_loc == 1) & (boundary_type == 1)),
        lfr.focal_sum(phi, kernel_ip1_j) - (dx * Neumann_boundary_value),
        phi,
    )

    phi = lfr.where(
        ((boundary_loc == 1) & (boundary_type == 2)),
        lfr.focal_sum(phi, kernel_i_jp1) - (dz * Neumann_boundary_value),
        phi,
    )

    phi = lfr.where(
        ((boundary_loc == 1) & (boundary_type == 3)),
        lfr.focal_sum(phi, kernel_im1_j) + (dx * Neumann_boundary_value),
        phi,
    )

    phi = lfr.where(
        ((boundary_loc == 1) & (boundary_type == 4)),
        lfr.focal_sum(phi, kernel_i_jm1) + (dz * Neumann_boundary_value),
        phi,
    )

    return phi
