#!/usr/bin/env python3
"""
Test ModelBuilder - Aligned with Triton-distributed style.
TileLang testing framework style.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tilelang.testing
import torch

from megatile import ModelBuilder
from megatile.core.task_base import TaskDependency


@tilelang.testing.requires_cuda
def test_single_linear():
    """Test single linear layer with ModelBuilder."""
    M, K, N = 64, 128, 64
    
    # Create tensors
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    C_ref = A @ B.T
    
    # Build model
    builder = ModelBuilder(num_warps=4)
    builder.make_linear(A, B, C, layer_id=0)
    builder.compile()
    builder.run()
    torch.cuda.synchronize()
    
    tilelang.testing.torch_assert_close(C, C_ref, atol=0.1, rtol=0.1)


@tilelang.testing.requires_cuda
def test_two_linear_layers():
    """Test two sequential linear layers (serial execution)."""
    M1, K1, N1 = 32, 128, 64
    M2, K2, N2 = 32, 64, 64
    
    # Create tensors
    A = torch.randn(M1, K1, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(N1, K1, dtype=torch.bfloat16, device='cuda')
    C = torch.zeros(M1, N1, dtype=torch.bfloat16, device='cuda')
    D = torch.randn(N2, K2, dtype=torch.bfloat16, device='cuda')
    E = torch.zeros(M2, N2, dtype=torch.bfloat16, device='cuda')
    
    # Reference
    C_ref = A @ B.T
    E_ref = C_ref @ D.T
    
    # Build model
    builder = ModelBuilder(num_warps=4)
    builder.make_linear(A, B, C, layer_id=0)
    builder.make_linear(C, D, E, layer_id=1)  # Layer 1 depends on Layer 0's output C
    builder.compile()
    builder.run()
    torch.cuda.synchronize()
    
    tilelang.testing.torch_assert_close(C, C_ref, atol=0.1, rtol=0.1)
    tilelang.testing.torch_assert_close(E, E_ref, atol=0.5, rtol=0.5)


@tilelang.testing.requires_cuda
def test_parallel_linear_layers():
    """Test two parallel linear layers (no dependencies)."""
    M1, K1, N1 = 32, 128, 64
    M2, K2, N2 = 32, 128, 64
    
    # Create tensors for two independent linear layers
    A1 = torch.randn(M1, K1, dtype=torch.bfloat16, device='cuda')
    B1 = torch.randn(N1, K1, dtype=torch.bfloat16, device='cuda')
    C1 = torch.zeros(M1, N1, dtype=torch.bfloat16, device='cuda')
    
    A2 = torch.randn(M2, K2, dtype=torch.bfloat16, device='cuda')
    B2 = torch.randn(N2, K2, dtype=torch.bfloat16, device='cuda')
    C2 = torch.zeros(M2, N2, dtype=torch.bfloat16, device='cuda')
    
    # Reference
    C1_ref = A1 @ B1.T
    C2_ref = A2 @ B2.T
    
    # Build model with two independent layers (same layer_id, different task_id)
    builder = ModelBuilder(num_warps=4)
    builder.make_linear(A1, B1, C1, layer_id=0)  # Task 0
    builder.make_linear(A2, B2, C2, layer_id=0)  # Task 1, same layer_id but independent
    builder.compile()
    builder.run()
    torch.cuda.synchronize()
    
    tilelang.testing.torch_assert_close(C1, C1_ref, atol=0.1, rtol=0.1)
    tilelang.testing.torch_assert_close(C2, C2_ref, atol=0.1, rtol=0.1)


@tilelang.testing.requires_cuda
def test_fork_join_linear_layers():
    """
    Test fork-join pattern: A and B parallel, C waits for both A and B.
    
    This test verifies that C3 correctly waits for both C1 and C2 to complete
    before executing. The dependency is set manually to ensure C3 depends on both.
    """
    # Use very small sizes
    # Note: Shared memory issue occurs with 3 tasks even with tiny matrices
    # This is because TileLang may accumulate shared memory for all task types at compile time
    M1, K1, N1 = 4, 8, 8
    M2, K2, N2 = 4, 8, 8
    M3, K3, N3 = 4, 8, 4  # C3 takes C1 (4x8) as input
    
    # Create tensors
    # A and B are independent (parallel)
    A1 = torch.randn(M1, K1, dtype=torch.bfloat16, device='cuda')
    B1 = torch.randn(N1, K1, dtype=torch.bfloat16, device='cuda')
    C1 = torch.zeros(M1, N1, dtype=torch.bfloat16, device='cuda')
    
    A2 = torch.randn(M2, K2, dtype=torch.bfloat16, device='cuda')
    B2 = torch.randn(N2, K2, dtype=torch.bfloat16, device='cuda')
    C2 = torch.zeros(M2, N2, dtype=torch.bfloat16, device='cuda')
    
    # C depends on both C1 and C2 (join)
    B3 = torch.randn(N3, K3, dtype=torch.bfloat16, device='cuda')
    C3 = torch.zeros(M3, N3, dtype=torch.bfloat16, device='cuda')
    
    # Reference
    C1_ref = A1 @ B1.T
    C2_ref = A2 @ B2.T
    # C3 depends on C1 (waits for both A and B to complete, but uses C1)
    C3_ref = C1_ref @ B3.T
    
    # Build model: A and B parallel, C waits for both
    # Use smaller block sizes to reduce shared memory usage
    # Note: BLOCK_SIZE_M must be divisible by 16 for TileLang
    builder = ModelBuilder(num_warps=2)
    # Use smaller block sizes for this test to avoid shared memory limit
    # BLOCK_SIZE_M=16 (min required), BLOCK_SIZE_N=16 (reduced from 128), BLOCK_SIZE_K=16 (reduced from 128), NUM_STAGES=1
    builder.make_linear(A1, B1, C1, layer_id=0, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, NUM_STAGES=1)
    builder.make_linear(A2, B2, C2, layer_id=0, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, NUM_STAGES=1)
    
    # Get tasks from builder to create dependencies
    # Tasks are added in order, so tasks_a is first, tasks_b is second
    tasks_a = [t for t in builder.megakernel_tasks if t.layer_id == 0 and t.task_id == 0]
    tasks_b = [t for t in builder.megakernel_tasks if t.layer_id == 0 and t.task_id == 1]
    
    # Save dependencies for C1 and C2
    dep_c1 = TaskDependency(
        layer_id=tasks_a[0].layer_id,
        task_id=tasks_a[0].task_id,
        start_tiles=0,
        end_tiles=tasks_a[0].num_tiles,
    )
    dep_c2 = TaskDependency(
        layer_id=tasks_b[0].layer_id,
        task_id=tasks_b[0].task_id,
        start_tiles=0,
        end_tiles=tasks_b[0].num_tiles,
    )
    
    # Manually set C3's dependency to wait for both C1 and C2
    builder.last_dependency = [dep_c1, dep_c2]
    builder.make_linear(C1, B3, C3, layer_id=1, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, NUM_STAGES=1)  # Task C (waits for both A and B)
    
    # Verify dependency is correctly set
    tasks_c = [t for t in builder.megakernel_tasks if t.layer_id == 1]
    assert isinstance(tasks_c[0].dependency, list), "Task C dependency should be a list"
    assert len(tasks_c[0].dependency) == 2, f"Task C should have 2 dependencies, got {len(tasks_c[0].dependency)}"
    
    builder.compile()
    builder.run()
    torch.cuda.synchronize()
    
    tilelang.testing.torch_assert_close(C1, C1_ref, atol=0.1, rtol=0.1)
    tilelang.testing.torch_assert_close(C2, C2_ref, atol=0.1, rtol=0.1)
    tilelang.testing.torch_assert_close(C3, C3_ref, atol=0.5, rtol=0.5)


if __name__ == "__main__":
    tilelang.testing.main()

