################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Norm Task - Aligned with Triton-distributed
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


@dataclass
class RMSNormConfig(ConfigBase):
    BLOCK_SIZE_N: int = 1024
    RMS_EPS: float = 1e-6


@dataclass
class RMSNormTask(TaskBase):
    config: RMSNormConfig


def rmsnorm_config_factory(**kwargs) -> RMSNormConfig:
    return dataclasses.replace(RMSNormConfig(), **kwargs)


def codegen_rmsnorm(task: RMSNormTask) -> str:
    """Generate code for RMSNorm task."""
    config = task.config
    x = task.io_tensors[0][0]
    M, N = x.shape
    
    code = f'''rmsnorm_task_compute(
    x, weight, y, scoreboard, tile_id, layer_id, task_id,
    {N}, {config.RMS_EPS}, {config.BLOCK_SIZE_N}
)'''
    return code


@registry.register_task(
    op_type="rms_norm",
    task_cls=RMSNormTask,
    config_factory=rmsnorm_config_factory,
    codegen_func=codegen_rmsnorm,
)
class RMSNormTaskBuilder(TaskBuilderBase):
    
    @classmethod
    def get_problem_size(cls, io_tensors: List[List[torch.Tensor]], extra_params: Dict[str, Any]):
        x = io_tensors[0][0]
        M, N = x.shape
        return (M, N)
    
    @classmethod
    def _build_tasks_impl(
        cls,
        device_prop: DeviceProp,
        layer_id: int,
        dependency: TaskDependency,
        io_tensors,
        extra_params,
        tile_wise=True,
    ) -> List[TaskBase]:
        # Aligned with Triton: use build_tile_desc-like logic
        input, weight = io_tensors[0]
        output = io_tensors[1][0]
        num_tiles = output.numel() // output.shape[-1]
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config(**extra_params)
        cls.log(f"RMS Norm Task: num_tiles = {num_tiles}")
        tasks = []
        tile_size = output.shape[-1]
        BLOCK_SIZE_N = kernel_config.BLOCK_SIZE_N
        
        # Aligned with Triton: build tile descriptions
        for i in range(num_tiles):
            # Calculate start_indices and data_sizes for input/output
            # Similar to build_tile_desc but simplified for 2D case
            row = i
            in_start_indices = (row, 0)
            in_data_sizes = (1, min(tile_size, input.shape[-1] - 0))  # return_valid_size=True
            out_start_indices = (row, 0)
            out_data_sizes = (1, tile_size)
            
            input_desc = InputDependencyDesc(input, require_full=False,
                                             start_indices=in_start_indices,
                                             data_sizes=in_data_sizes)
            weight_desc = InputDependencyDesc(weight, require_full=True)
            out_desc = OutputTilingDesc(
                start_indices=out_start_indices,
                tile_sizes=out_data_sizes
            )
            inputs_dep = {input: input_desc, weight: weight_desc}
            outs_tile_mapping = {output: out_desc}
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params,
                                 inputs_dep, outs_tile_mapping))
        
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

