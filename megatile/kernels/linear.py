################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Linear (GEMM) Kernel
#
# Reference: triton_dist/mega_triton_kernel/kernels/linear.py
#
################################################################################
import tilelang.language as T
from .task_context import release_tile, task_scoredboard_start, TILE_READY_SIGNAL


@T.macro
def tile_wise_matmul_compute(
    A: T.Buffer,
    B: T.Buffer,
    C: T.Buffer,
    tile_id: T.int32,
    M: T.int32,
    N: T.int32,
    K: T.int32,
    BLOCK_SIZE_M: T.int32,
    BLOCK_SIZE_N: T.int32,
    BLOCK_SIZE_K: T.int32,
    NUM_STAGES: T.int32,
):
    """
    Tile-wise matmul: C[tile] = A @ B^T
    Aligned with Triton's tile_wise_matmul_compute.
    """
    num_pid_n = T.ceildiv(N, BLOCK_SIZE_N)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    A_shared = T.alloc_shared((BLOCK_SIZE_M, BLOCK_SIZE_K), T.bfloat16)
    B_shared = T.alloc_shared((BLOCK_SIZE_N, BLOCK_SIZE_K), T.bfloat16)
    C_local = T.alloc_fragment((BLOCK_SIZE_M, BLOCK_SIZE_N), T.float32)
    T.clear(C_local)
    
    for k in T.Pipelined(T.ceildiv(K, BLOCK_SIZE_K), num_stages=NUM_STAGES):
        T.copy(A[start_m, k * BLOCK_SIZE_K], A_shared)
        T.copy(B[start_n, k * BLOCK_SIZE_K], B_shared)
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)
    
    T.copy(C_local, C[start_m, start_n])


@T.macro
def linear_task_compute(
    A: T.Buffer,
    B: T.Buffer,
    C: T.Buffer,
    scoreboard_flat: T.Buffer,
    tile_id: T.int32,
    layer_id: T.int32,
    task_id: T.int32,
    M: T.int32,
    N: T.int32,
    K: T.int32,
    BLOCK_SIZE_M: T.int32,
    BLOCK_SIZE_N: T.int32,
    BLOCK_SIZE_K: T.int32,
    NUM_STAGES: T.int32,
    MAX_TASK_ID: T.int32,
    MAX_NUM_TASK_PER_OP: T.int32,
):
    """
    Linear task: compute + release_tile.
    Aligned with Triton's linear_task_compute(task_base_info, scoreboard, ...).
    
    Note: In TileLang, we pass parameters separately instead of using TaskBaseInfo/Scoreboard objects.
    """
    tile_wise_matmul_compute(A, B, C, tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES)
    # Aligned with Triton: scoreboard.release_tile(task_base_info, tile_id)
    sb_offset = task_scoredboard_start(scoreboard_flat, layer_id, task_id, MAX_TASK_ID, MAX_NUM_TASK_PER_OP)
    release_tile(scoreboard_flat, sb_offset + tile_id)
