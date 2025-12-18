import numpy as np
from numpy.typing import NDArray

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


# NOTE: -------------------------- the kernels in second version look at kernel_i_jm1 and kernel_i_jp1 -----


# # kernel_im1_j   i-1, j
# kernel_im1_j = np.array(
#     [
#         [0, 0, 0],
#         [1, 0, 0],
#         [0, 0, 0],
#     ],
#     dtype=np.uint8,
# )

# # ????? kernel_i_jm1   i, j-1    # Check it. It is changed compared to advection-diffusion test model.
# kernel_i_jm1 = np.array(
#     [
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 1, 0],
#     ],
#     dtype=np.uint8,
# )

# # kernel_ip1_j   i+1, j
# kernel_ip1_j = np.array(
#     [
#         [0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 0],
#     ],
#     dtype=np.uint8,
# )

# # ????? kernel_i_jp1   i, j+1     # Check it. It is changed compared to advection-diffusion test model.
# kernel_i_jp1 = np.array(
#     [
#         [0, 1, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#     ],
#     dtype=np.uint8,
# )

# ---- NOTE kernel_i_jm1_boundary and kernel_i_jp1_boundary
# are different kernel_i_jm1 and kernel_i_jp1 -------------


# kernel_im1_j   i-1, j
kernel_im1_j_boundary: NDArray[np.uint8] = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)

# kernel_i_jm1   i, j-1    # Check it. It is changed compared to advection-diffusion  test model.
kernel_i_jm1_boundary: NDArray[np.uint8] = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

# kernel_ip1_j   i+1, j
kernel_ip1_j_boundary: NDArray[np.uint8] = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)

# kernel_i_jp1   i, j+1     # Check it. It is changed compared to advection-diffusion  test model.
kernel_i_jp1_boundary: NDArray[np.uint8] = np.array(
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)
