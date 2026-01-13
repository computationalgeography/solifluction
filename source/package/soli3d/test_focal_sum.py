#!/usr/bin/env python

import sys
from pathlib import Path
from typing import Any

import lue.framework as lfr
import numpy as np

from .kernels import kernel_im1_j

Shape = tuple[int, int]


def initialize_focal_sum(
    array_shape: Shape,
    partition_shape: Shape,
) -> Any:  # raster,

    initial_raster = lfr.create_array(
        array_shape,
        dtype=np.float64,
        fill_value=5.0,
        partition_shape=partition_shape,
    )

    # ----------------- initial layer information  -------------

    print("start to run initial focal sum")

    return initial_raster


def simulate_focal_sum(
    initial_raster: Any,
    dx: float,
    inner_iteration_threshold: int,
) -> Any:

    local_inner_iteration: int = 0

    raster = initial_raster

    while local_inner_iteration < inner_iteration_threshold:

        raster = (raster - lfr.focal_sum(raster, kernel_im1_j)) / dx

        local_inner_iteration = local_inner_iteration + 1

    return raster


class Focal(lfr.Model):

    def __init__(self, array_shape: Shape, partition_shape: Shape):
        super().__init__()
        self.array_shape = array_shape
        self.partition_shape = partition_shape

    def initialize(
        self,
    ) -> None:

        initial_raster = initialize_focal_sum(
            self.array_shape,
            self.partition_shape,
        )

        self.raster = initial_raster

    def simulate(self, iteration: int) -> Any:

        print("iteration inside simulate -------", iteration)

        self.raster = simulate_focal_sum(
            self.raster,
            self.dx,
            self.inner_iteration_threshold,
        )

        return self.raster.future()


@lfr.runtime_scope
def focal_sum(
    *,
    array_shape: Shape,
    partition_shape: Shape,
    number_of_global_iterations: int,
    inner_iteration_threshold: int,
    dx: float,
) -> None:

    model = Focal(
        array_shape=array_shape,
        partition_shape=partition_shape,
    )

    model.dx = dx
    model.inner_iteration_threshold = inner_iteration_threshold

    lfr.run_deterministic(
        model,
        lfr.DefaultProgressor(),
        nr_time_steps=number_of_global_iterations,
        rate_limit=5,
    )


def main() -> None:

    num_args = 5

    # remove any HPX flags from the argument list
    argv = [arg for arg in sys.argv[1:] if not arg.startswith("--hpx")]

    if len(argv) != num_args:
        raise RuntimeError(
            "Usage: python run_test_focal_sum.py array_size partition_size "
            "number_of_global_iterations inner_iteration_threshold dx"
        )

    array_size = int(argv[0])
    partition_shape_size = int(argv[1])
    number_of_global_iterations = int(argv[2])
    inner_iteration_threshold = int(argv[3])
    dx = float(argv[4])

    print("sys.argv =", sys.argv)

    array_shape: tuple[int, int] = 2 * (array_size,)

    partition_shape: tuple[int, int] = 2 * (partition_shape_size,)

    # ---------------------  initial information --------------------

    focal_sum(
        array_shape=array_shape,
        partition_shape=partition_shape,
        number_of_global_iterations=number_of_global_iterations,
        inner_iteration_threshold=inner_iteration_threshold,
        dx=dx,
    )
