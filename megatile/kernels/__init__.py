################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Kernels module - All @T.macro implementations
#
################################################################################

from .task_context import (
    TILE_READY_SIGNAL,
    TaskBaseInfo,
    TensorDesc,
    wait_deps,
    release_tile,
    task_scoredboard_start,
    compute_scoreboard_offset,
)

from .linear import (
    tile_wise_matmul_compute,
    linear_task_compute,
)

from .norm import (
    rmsnorm_task_compute,
)

from .activation import (
    act_mul_up_tile_compute,
    silu_mul_up_task_compute,
)

__all__ = [
    # Task context
    "TILE_READY_SIGNAL",
    "TaskBaseInfo",
    "TensorDesc",
    "wait_deps",
    "release_tile",
    "task_scoredboard_start",
    "compute_scoreboard_offset",
    # Linear
    "tile_wise_matmul_compute",
    "linear_task_compute",
    # Norm
    "rmsnorm_task_compute",
    # Activation
    "act_mul_up_tile_compute",
    "silu_mul_up_task_compute",
]
