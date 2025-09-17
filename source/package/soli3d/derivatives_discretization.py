# Discretize the derivatives
from typing import Any

import lue.framework as lfr
import numpy as np

# Part 1: compute derivatives in x-z plan (uniform grid mesh. dx and dz are uniform)

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


def dx_forward(phi: Any, dx: float) -> Any:

    return (lfr.focal_sum(phi, kernel_ip1_j) - phi) / dx


def dx_backward(phi: Any, dx: float) -> Any:

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


def dx_upwind(phi: Any, dx: float, ux: Any) -> Any:

    return lfr.where(ux >= 0, dx_backward(phi, dx), dx_forward(phi, dx))


def dz_upwind(phi, dz, uz):

    return lfr.where(uz >= 0, dz_backward(phi, dz), dz_forward(phi, dz))


# part 2: calculate derivatives in y direction


# Second derivatives with non-uniform layer distance in y
def second_derivatives_in_y(
    layer_variable_center: Any,
    layer_variable_up: Any,
    layer_variable_down: Any,
    dy_layers_up: Any,
    dy_layers_down: Any,
) -> Any:
    """
    Calculate second derivatives for a discretization of non-uniform layer
    distances in y.

    - `{.}_center` represents the center layer.
    - `{.}_up` is the upper layer.
    - `{.}_down` is the lower layer.

    Layer order:
        - -------------- (top surface)
        - ------- layer `{.}_up`
        - ------- layer `{.}_center`
        - ------- layer `{.}_down`
        - -------------- (bottom bed)

    Note:
        For central difference (used on internal layers, which excludes boundary layers
        like the bed and surface layers), `{.}_center`, `{.}_up`, and `{.}_down`
        correspond to the center, upper, and lower layers, respectively.

        For backward difference (used on the surface layer), `layer_variable_up`
        represents the surface layer information, `layer_variable_center` corresponds
        to the layer one layer below (`surface_layer_index - 1`), and
        `layer_variable_down` represents the layer
        two layers below (`surface_layer_index - 2`).

        For forward difference (used on the bed layer), `layer_variable_down` is the bed
        layer information, `layer_variable_center` is one layer above
        (`bed_layer_index + 1`), and `layer_variable_up` is two layers
        above (`bed_layer_index + 2`).
    """

    # layer_variable_center:center variable with FDM coeff -2 , layer_variable_up
    # and layer_variable_down around variables with FDM coeff 1
    # NOTE: for surface layer (backward) layer_variable_center is one layer below, and
    # layer_variable_up and layer_variable_down are surface layer and two layers below

    # dy_layers is averaged BUT certainly it is WRONG formula for discretization

    return (2 / (dy_layers_up + dy_layers_down)) * (
        ((layer_variable_up - layer_variable_center) / dy_layers_up)
        - ((layer_variable_center - layer_variable_down) / dy_layers_down)
    )


def dy_forward(
    layer_variable_center,
    layer_variable_forward,
    dy,
):

    return (layer_variable_forward - layer_variable_center) / dy


def dy_backward(
    layer_variable_center,
    layer_variable_backward,
    dy,
):

    return (layer_variable_center - layer_variable_backward) / dy


def dy_upwind(
    layer_variable_center,
    layer_variable_up,
    layer_variable_down,
    dy_up,
    dy_down,
    adv_coeff,
):

    # First derivative discretization for convection term (adv_coeff * d_phi/dy) in Eq like:
    # [d_phi/dt + (adv_coeff * d_phi/dy)  - (diff_coeff * d2_phi/dy2) = rhs]

    return lfr.where(
        adv_coeff >= 0,
        dy_backward(layer_variable_center, layer_variable_down, dy_down),
        dy_forward(layer_variable_center, layer_variable_up, dy_up),
    )
