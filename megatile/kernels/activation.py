################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Activation Kernels
#
# Reference: triton_dist/mega_triton_kernel/kernels/activation.py
#
################################################################################
import tilelang.language as T
from .task_context import release_tile, task_scoredboard_start


@T.macro
def act_mul_up_tile_compute(
    gate: T.Buffer,
    up: T.Buffer,
    output: T.Buffer,
    tile_id: T.int32,
    M: T.int32,
    N: T.int32,
    BLOCK_SIZE_M: T.int32,
    BLOCK_SIZE_N: T.int32,
):
    """
    SiLU(gate) * up tile compute.
    Aligned with Triton's act_mul_up_tile_compute.
    """
    num_pid_n = T.ceildiv(N, BLOCK_SIZE_N)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    for m in T.serial(BLOCK_SIZE_M):
        for n in T.serial(BLOCK_SIZE_N):
            row = start_m + m
            col = start_n + n
            if row < M and col < N:
                g = gate[row, col]
                u = up[row, col]
                sigmoid_g = 1.0 / (1.0 + T.exp(-g))
                output[row, col] = g * sigmoid_g * u


@T.macro
def silu_mul_up_task_compute(
    gate: T.Buffer,
    up: T.Buffer,
    output: T.Buffer,
    scoreboard_flat: T.Buffer,
    tile_id: T.int32,
    layer_id: T.int32,
    task_id: T.int32,
    M: T.int32,
    N: T.int32,
    BLOCK_SIZE_M: T.int32,
    BLOCK_SIZE_N: T.int32,
    MAX_TASK_ID: T.int32,
    MAX_NUM_TASK_PER_OP: T.int32,
):
    """
    SiLU mul task.
    Aligned with Triton's silu_mul_up_task_compute(task_base_info, scoreboard, ...).
    
    Note: In TileLang, we pass parameters separately instead of using TaskBaseInfo/Scoreboard objects.
    """
    act_mul_up_tile_compute(gate, up, output, tile_id, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    # Aligned with Triton: scoreboard.release_tile(task_base_info, tile_id)
    sb_offset = task_scoredboard_start(scoreboard_flat, layer_id, task_id, MAX_TASK_ID, MAX_NUM_TASK_PER_OP)
    release_tile(scoreboard_flat, sb_offset + tile_id)
