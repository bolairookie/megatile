################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Task Context - Aligned with Triton-distributed
#
# Reference: triton_dist/mega_triton_kernel/kernels/task_context.py
#
# KEY WORKAROUND:
#   All atomic ops use address_of which requires BufferLoadNode.
#   Solution: Use fixed max range + condition filters, and atomic_add(x, 0) to read.
#
################################################################################
import tilelang.language as T
from dataclasses import dataclass

TILE_READY_SIGNAL = 1
MAX_NUM_TENSOR_DIMS = 4
INT_PER_TENSOR = MAX_NUM_TENSOR_DIMS + 2  # data_ptr_idx, placeholder, dim0, dim1, ...


# Work queue offsets (aligned with Triton-distributed)
TASK_TYPE_OFFSET = 0
LAYER_ID_OFFSET = 1
TASK_ID_OFFSET = 2
TILE_ID_OFFSET = 3
DEPS_START_OFFSET = 4
DEPS_END_OFFSET = 5
IO_TENSORS_OFFSET = 6


@dataclass
class TensorDesc:
    """
    Tensor descriptor - encapsulates tensor metadata.
    Aligned with Triton-distributed's TensorDesc.
    
    In TileLang, we store tensor index in work_queue instead of pointer,
    so TensorDesc stores the index and shape information.
    """
    tensor_idx: T.int32  # Index into all_tensors array
    work_queue: T.Buffer  # Work queue slice for this task
    io_tensors_ptr_offset: T.int32  # Offset in work_queue where tensor descriptors start
    INT_PER_TENSOR: T.int32
    
    @T.macro
    def data_ptr(self, all_tensors: T.Buffer, dtype: T.dtype) -> T.Buffer:
        """
        Get data pointer for this tensor.
        
        Args:
            all_tensors: Tuple of all tensor buffers
            dtype: Element dtype
        
        Returns:
            Buffer pointing to the tensor data
        """
        # In TileLang, we pass tensors as explicit arguments
        # So we need to index into the all_tensors tuple
        # This is a limitation - we can't dynamically index into a tuple
        # For now, we'll need to pass tensors explicitly in codegen
        # This is a placeholder - actual implementation depends on how we pass tensors
        pass
    
    @T.macro
    def size(self, dim_idx: T.int32) -> T.int32:
        """
        Get size of a dimension.
        
        Args:
            dim_idx: Dimension index (0-based)
        
        Returns:
            Size of the dimension
        """
        # Read from work_queue: io_tensors_ptr_offset + tensor_idx * INT_PER_TENSOR + 2 + dim_idx
        base = self.io_tensors_ptr_offset + self.tensor_idx * self.INT_PER_TENSOR
        return self.work_queue[base + 2 + dim_idx]  # +2 for data_ptr_idx and placeholder


@dataclass
class TaskBaseInfo:
    """
    Task base information - encapsulates task metadata.
    Aligned with Triton-distributed's TaskBaseInfo.
    """
    io_tensors_ptr_offset: T.int32  # Offset in work_queue where tensor descriptors start
    task_type: T.int32
    layer_id: T.int32
    task_id: T.int32
    tile_id_or_start: T.int32
    depend_entry_start: T.int32
    depend_entry_end: T.int32
    work_queue: T.Buffer  # Work queue slice for this task
    MAX_NUM_TENSOR_DIMS: T.int32
    INT_PER_TENSOR: T.int32
    
    @T.macro
    def get_tensor(self, idx: T.int32) -> TensorDesc:
        """
        Get tensor descriptor for input/output tensor.
        
        Args:
            idx: Tensor index (0 for first input, etc.)
        
        Returns:
            TensorDesc for the tensor
        """
        tensor_idx = self.work_queue[self.io_tensors_ptr_offset + idx * self.INT_PER_TENSOR]
        return TensorDesc(
            tensor_idx=tensor_idx,
            work_queue=self.work_queue,
            io_tensors_ptr_offset=self.io_tensors_ptr_offset,
            INT_PER_TENSOR=self.INT_PER_TENSOR,
        )


@T.macro
def wait_deps(scoreboard_table, task_deps_ptr, entry_start, entry_end, num_warps):
    """
    Wait for dependencies, aligned with Triton-distributed.
    
    Aligned with Triton's Scoreboard.wait_deps(self, task_base_info: TaskBaseInfo).
    
    Args:
        scoreboard_table: Scoreboard table buffer
        task_deps_ptr: Task dependencies pointer (2D: [entry, 0/1] for l/r)
        entry_start: Start entry index
        entry_end: End entry index (exclusive)
        num_warps: Number of warps (for stride in loop)
    """
    # Don't exit early, otherwise it will cause function inlining to fail.
    WARP_SIZE = 32
    thread_idx = T.get_thread_binding(0)
    lane_id = thread_idx % WARP_SIZE
    warp_id = thread_idx // WARP_SIZE
    
    # Aligned with Triton: for t in range(entry_start + warp_id, entry_end, num_warps)
    t = T.alloc_var(T.int32)
    t = entry_start + warp_id
    while t < entry_end:
        # Aligned with Triton: l = ld(self.task_deps_ptr + t * self.INT_PER_DEPS + 0)
        l = task_deps_ptr[t, 0]
        # Aligned with Triton: r = ld(self.task_deps_ptr + t * self.INT_PER_DEPS + 1)
        r = task_deps_ptr[t, 1]
        num_signals = r - l
        # Aligned with Triton: sb_wait_base_ptr = self.scoreboard_table + l
        # Note: We can't use pointer arithmetic, so we use l + i directly in indexing
        
        # Aligned with Triton: for i in range(lane_id, num_signals, WARP_SIZE)
        # Aligned with Triton: while ld(sb_wait_base_ptr + i, scope="gpu", semantic="acquire") != self.TILE_READY_SIGNAL: pass
        i = T.alloc_var(T.int32)
        i = lane_id
        while i < num_signals:
            ready = T.alloc_var(T.int32)
            ready = 0
            while ready == 0:
                val = T.atomic_load(scoreboard_table[T.cast(l + i, T.int32)])
                if val == TILE_READY_SIGNAL:
                    ready = 1
            i = i + WARP_SIZE
        
        t = t + num_warps
    
    # Aligned with Triton: __syncthreads()
    T.sync_threads()


@T.macro
def release_tile(scoreboard_flat: T.Buffer, offset: T.int32):
    """
    Release a tile by setting its signal in the scoreboard.
    
    Aligned with Triton's Scoreboard.release_tile(self, task_base_info: TaskBaseInfo, tile_id).
    """
    # Aligned with Triton: __syncthreads()  # ensure that `store` on all threads is finish
    T.sync_threads()
    thread_idx = T.get_thread_binding(0)
    # Aligned with Triton: if thread_idx == 0: st(sb_set_base_ptr + tile_id, self.TILE_READY_SIGNAL, "gpu", "release")
    if thread_idx == 0:
        T.atomic_store(scoreboard_flat[offset], TILE_READY_SIGNAL, memory_order="release")
    # Aligned with Triton: __syncthreads()  # avoid divergence
    T.sync_threads()


@T.macro
def task_scoredboard_start(scoreboard_table: T.Buffer, layer_id: T.int32, task_id: T.int32, MAX_TASK_ID: T.int32, MAX_NUM_TASK_PER_OP: T.int32) -> T.int32:
    """
    Compute scoreboard start offset for a task.
    
    Aligned with Triton's Scoreboard.task_scoredboard_start(self, task_base_info: TaskBaseInfo).
    Note: Triton uses "scoredboard" (typo) instead of "scoreboard".
    """
    # Aligned with Triton: sb_layer_offset = task_base_info.layer_id * self.MAX_TASK_ID * self.MAX_NUM_TASK_PER_OP
    sb_layer_offset = layer_id * MAX_TASK_ID * MAX_NUM_TASK_PER_OP
    # Aligned with Triton: sb_set_base_ptr = self.scoreboard_table + sb_layer_offset + task_base_info.task_id * self.MAX_NUM_TASK_PER_OP
    sb_set_base_ptr = sb_layer_offset + task_id * MAX_NUM_TASK_PER_OP
    return sb_set_base_ptr


def compute_scoreboard_offset(
    layer_id: int,
    task_id: int,
    max_task_id: int,
    max_tiles_per_op: int,
) -> int:
    """
    HOST-side utility to compute base offset in flat scoreboard.
    """
    return layer_id * max_task_id * max_tiles_per_op + task_id * max_tiles_per_op
