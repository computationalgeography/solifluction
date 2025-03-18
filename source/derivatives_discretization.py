# Discretize the derivatives
import lue.framework as lfr
import numpy as np

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


def dx_forward(phi, dx):

    return (lfr.focal_sum(phi, kernel_ip1_j) - phi) / dx


def dx_backward(phi, dx):

    return (phi - lfr.focal_sum(phi, kernel_im1_j)) / dx


def dx_central(phi, dx):

    return (lfr.focal_sum(phi, kernel_ip1_j) - lfr.focal_sum(phi, kernel_im1_j)) / (
        2 * dx
    )


def d2x_forward(phi, dx):

    return (
        lfr.focal_sum(lfr.focal_sum(phi, kernel_ip1_j), kernel_ip1_j)
        - (2 * lfr.focal_sum(phi, kernel_ip1_j))
        + phi
    ) / (dx**2)


def d2x_backward(phi, dx):

    return (
        lfr.focal_sum(lfr.focal_sum(phi, kernel_im1_j), kernel_im1_j)
        - (2 * lfr.focal_sum(phi, kernel_im1_j))
        + phi
    ) / (dx**2)


def d2x_central(phi, dx):

    return (
        lfr.focal_sum(phi, kernel_ip1_j) - (2 * phi) + lfr.focal_sum(phi, kernel_im1_j)
    ) / (dx**2)


def dz_forward(phi, dz):

    return (lfr.focal_sum(phi, kernel_i_jp1) - phi) / dz


def dz_backward(phi, dz):

    return (phi - lfr.focal_sum(phi, kernel_i_jm1)) / dz


def dz_central(phi, dz):

    return (lfr.focal_sum(phi, kernel_i_jp1) - lfr.focal_sum(phi, kernel_i_jm1)) / (
        2 * dz
    )


def d2z_forward(phi, dz):

    return (
        lfr.focal_sum(lfr.focal_sum(phi, kernel_i_jp1), kernel_i_jp1)
        - (2 * lfr.focal_sum(phi, kernel_i_jp1))
        + phi
    ) / (dz**2)


def d2z_backward(phi, dz):

    return (
        lfr.focal_sum(lfr.focal_sum(phi, kernel_i_jm1), kernel_i_jm1)
        - (2 * lfr.focal_sum(phi, kernel_i_jm1))
        + phi
    ) / (dz**2)


def d2z_central(phi, dz):

    return (
        lfr.focal_sum(phi, kernel_i_jp1) - (2 * phi) + lfr.focal_sum(phi, kernel_i_jm1)
    ) / (dz**2)


def dx_upwind(phi, dx, ux):

    return lfr.where(ux >= 0, dx_backward(phi, dx), dx_forward(phi, dx))


def dz_upwind(phi, dz, uz):

    return lfr.where(uz >= 0, dz_backward(phi, dz), dz_forward(phi, dz))
