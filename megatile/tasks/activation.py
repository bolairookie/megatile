################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Activation Task - Aligned with Triton-distributed
#
################################################################################
import torch
from typing import Any, Dict, List
import dataclasses
from dataclasses import dataclass

from ..core.task_base import TaskBase, TaskDependency, DeviceProp, InputDependencyDesc, OutputTilingDesc
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase
from .utils import cdiv


@dataclass
class SiLUMulUpConfig(ConfigBase):
    BLOCK_SIZE_M: int = 8  # Aligned with Triton
    BLOCK_SIZE_N: int = 128


@dataclass
class SiLUMulUpTask(TaskBase):
    config: SiLUMulUpConfig


def silu_mul_up_config_factory(**kwargs) -> SiLUMulUpConfig:
    return dataclasses.replace(SiLUMulUpConfig(), **kwargs)


def codegen_silu_mul_up(task: SiLUMulUpTask) -> str:
    """Generate code for SiLU mul up task."""
    config = task.config
    fc1_out = task.io_tensors[0][0]
    M, N2 = fc1_out.shape
    N = N2 // 2
    
    code = f'''silu_mul_up_task_compute(
    gate, up, output, scoreboard, tile_id, layer_id, task_id,
    {M}, {N}, {config.BLOCK_SIZE_M}, {config.BLOCK_SIZE_N}
)'''
    return code


@registry.register_task(
    op_type="silu_mul_up",
    task_cls=SiLUMulUpTask,
    config_factory=silu_mul_up_config_factory,
    codegen_func=codegen_silu_mul_up,
)
class SiLUMulUpTaskBuilder(TaskBuilderBase):
    
    @classmethod
    def get_problem_size(cls, io_tensors: List[List[torch.Tensor]], extra_params: Dict[str, Any]):
        fc1_out = io_tensors[0][0]
        M, N2 = fc1_out.shape
        return (M, N2 // 2)
    
    @classmethod
    def _build_tasks_impl(
        cls,
        device_prop: DeviceProp,
        layer_id: int,
        dependency: TaskDependency,
        io_tensors,
        extra_params,
    ) -> List[TaskBase]:
        kernel_config = cls.create_config()
        task_id = cls.get_task_id(layer_id)
        
        M, N = cls.get_problem_size(io_tensors, extra_params)
        BLOCK_SIZE_M = kernel_config.BLOCK_SIZE_M
        BLOCK_SIZE_N = kernel_config.BLOCK_SIZE_N
        
        num_tiles_m = cdiv(M, BLOCK_SIZE_M)
        num_tiles_n = cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_tiles_m * num_tiles_n
        
        x = io_tensors[0][0]
        y = io_tensors[1][0]
        num_sm = device_prop.NUM_SMS
        tasks = []
        cls.log(
            f"SiLUMulUp Task: M = {M}, N = {N}, num_tiles = {num_tiles}, num_sm = {num_sm}, tile_wise = True"
        )
        for tm in range(num_tiles_m):
            for tn in range(num_tiles_n):
                tile_id = tm * num_tiles_n + tn
                bm = min(BLOCK_SIZE_M, M - tm * BLOCK_SIZE_M)
                bn = min(BLOCK_SIZE_N, N - tn * BLOCK_SIZE_N)
                # Aligned with Triton: InputDependencyDesc and OutputTilingDesc
                x_desc = InputDependencyDesc(x, require_full=False,
                                             start_indices=(tm * BLOCK_SIZE_M, tn * BLOCK_SIZE_N),
                                             data_sizes=(bm, bn))
                y_desc = OutputTilingDesc(
                    start_indices=(tm * BLOCK_SIZE_M, tn * BLOCK_SIZE_N),
                    tile_sizes=(BLOCK_SIZE_M, BLOCK_SIZE_N)
                )
                inputs_dep = {x: x_desc}
                outs_tile_mapping = {y: y_desc}
                tasks.append(
                    cls._create_task(
                        layer_id, task_id, tile_id, num_tiles,
                        kernel_config, dependency, io_tensors,
                        extra_params, inputs_dep, outs_tile_mapping
                    )
                )
        
        return tasks
    
    @classmethod
    def build_tasks(
        cls,
        device_prop: DeviceProp,
        layer_id: int,
        dependency: TaskDependency,
        io_tensors: List[List[torch.Tensor]],
        extra_params: Dict[str, Any],
    ) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params)

