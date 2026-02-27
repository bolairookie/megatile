################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Normalization Kernels
#
# Reference: triton_dist/mega_triton_kernel/kernels/norm.py
#
################################################################################
import tilelang.language as T
from .task_context import release_tile, task_scoredboard_start


@T.macro
def rmsnorm_task_compute(
    x: T.Buffer,
    weight: T.Buffer,
    y: T.Buffer,
    scoreboard_flat: T.Buffer,
    tile_id: T.int32,
    layer_id: T.int32,
    task_id: T.int32,
    N: T.int32,
    RMS_EPS: T.float32,
    BLOCK_SIZE_N: T.int32,
    MAX_TASK_ID: T.int32,
    MAX_NUM_TASK_PER_OP: T.int32,
):
    """
    RMSNorm task.
    Aligned with Triton's rmsnorm_task_compute(task_base_info, scoreboard, ...).
    
    Note: In TileLang, we pass parameters separately instead of using TaskBaseInfo/Scoreboard objects.
    """
    row = tile_id
    
    sq_sum = T.alloc_fragment((1,), T.float32)
    T.clear(sq_sum)
    
    for col in T.serial(N):
        val = x[row, col]
        sq_sum[0] += val * val
    
    rms_val = T.sqrt(sq_sum[0] / N + RMS_EPS)
    
    for col in T.serial(N):
        y[row, col] = x[row, col] / rms_val * weight[col]
    
    # Aligned with Triton: scoreboard.release_tile(task_base_info, tile_id)
    sb_offset = task_scoredboard_start(scoreboard_flat, layer_id, task_id, MAX_TASK_ID, MAX_NUM_TASK_PER_OP)
    release_tile(scoreboard_flat, sb_offset + tile_id)
