################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
################################################################################
import torch
from typing import Any, Dict, List
import dataclasses
from dataclasses import dataclass

from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc, DeviceProp
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase
from .utils import cdiv


@dataclass
class LinearConfig(ConfigBase):
    BLOCK_SIZE_M: int = 16
    BLOCK_SIZE_N: int = 128
    BLOCK_SIZE_K: int = 128
    NUM_STAGES: int = 2  # Reduced from 4 to avoid shared memory limit


@dataclass
class LinearTask(TaskBase):
    config: LinearConfig


@dataclass
class MLPFC1Config(LinearConfig):
    pass


@dataclass
class MLPFC1Task(LinearTask):
    config: MLPFC1Config


@dataclass
class MLPFC2Config(LinearConfig):
    pass


@dataclass
class MLPFC2Task(LinearTask):
    config: MLPFC2Config


def linear_config_factory(**kwargs) -> LinearConfig:
    return dataclasses.replace(LinearConfig(), **kwargs)


def mlp_fc1_config_factory(**kwargs) -> MLPFC1Config:
    default = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 128,
        'NUM_STAGES': 6,
    }
    default.update(kwargs)
    return MLPFC1Config(**default)


def mlp_fc2_config_factory(**kwargs) -> MLPFC2Config:
    default = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 256,
        'NUM_STAGES': 6,
    }
    default.update(kwargs)
    return MLPFC2Config(**default)


def codegen_linear(task: LinearTask, tensor_to_index: dict = None) -> str:
    """Generate code for linear task."""
    config = task.config
    a, b = task.io_tensors[0]
    c = task.io_tensors[1][0]
    M, K = a.shape
    N = b.shape[0]
    
    # Get tensor variable names from tensor_to_index mapping
    if tensor_to_index is None:
        # Fallback: use A, B, C (will fail at runtime, but allows codegen to work)
        a_var = "A"
        b_var = "B"
        c_var = "C"
    else:
        a_idx = tensor_to_index.get(a, 0)
        b_idx = tensor_to_index.get(b, 1)
        c_idx = tensor_to_index.get(c, 2)
        a_var = f"tensor_{a_idx}"
        b_var = f"tensor_{b_idx}"
        c_var = f"tensor_{c_idx}"
    
    code = f'''tile_wise_matmul_compute(
    {a_var}, {b_var}, {c_var}, tile_id, {M}, {N}, {K},
    {config.BLOCK_SIZE_M}, {config.BLOCK_SIZE_N}, {config.BLOCK_SIZE_K}, {config.NUM_STAGES}
)'''
    return code


def codegen_mlp_fc1(task: MLPFC1Task, tensor_to_index: dict = None) -> str:
    return codegen_linear(task, tensor_to_index)


def codegen_mlp_fc2(task: MLPFC2Task, tensor_to_index: dict = None) -> str:
    return codegen_linear(task, tensor_to_index)


@registry.register_task(op_type="linear", task_cls=LinearTask, config_factory=linear_config_factory,
                        codegen_func=codegen_linear)
class LinearTaskBuilder(TaskBuilderBase):
    
    @classmethod
    def get_problem_size(cls, io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]):
        a, b = io_tensors[0]
        M, K = a.shape
        N, K = b.shape
        return (M, N, K)
    
    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True, config_args={}) -> List[TaskBase]:
        assert tile_wise == True  # noqa: E712
        # Extract config_kwargs from extra_params if present
        if extra_params and "config_kwargs" in extra_params:
            config_args = {**config_args, **extra_params["config_kwargs"]}
        kernel_config = cls.create_config(**config_args)
        task_id = cls.get_task_id(layer_id)
        
        BLOCK_SIZE_M = kernel_config.BLOCK_SIZE_M
        BLOCK_SIZE_N = kernel_config.BLOCK_SIZE_N
        M, N, K = cls.get_problem_size(io_tensors, extra_params)
        
        num_tiles_m = cdiv(M, BLOCK_SIZE_M)
        num_tiles_n = cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_tiles_m * num_tiles_n
        
        x, w = io_tensors[0]
        y = io_tensors[1][0]
        
        num_sm = device_prop.NUM_SMS
        tasks = []
        cls.log(
            f"Linear Task: M = {M}, N = {N}, K = {K}, num_tiles = {num_tiles}, num_sm = {num_sm}, tile_wise = {tile_wise}, dependency = {dependency}, BLOCK_SIZE_M ={BLOCK_SIZE_M}, BLOCK_SIZE_N = {BLOCK_SIZE_N}"
        )
        
        for tm in range(num_tiles_m):
            for tn in range(num_tiles_n):
                tile_id = tm * num_tiles_n + tn
                bm = min(BLOCK_SIZE_M, M - tm * BLOCK_SIZE_M)
                bn = min(BLOCK_SIZE_N, N - tn * BLOCK_SIZE_N)
                
                x_desc = InputDependencyDesc(x, require_full=False, start_indices=(tm * BLOCK_SIZE_M, 0),
                                             data_sizes=(bm, K))
                w_desc = InputDependencyDesc(w, require_full=False, start_indices=(tn * BLOCK_SIZE_N, 0),
                                             data_sizes=(bn, K))
                y_desc = OutputTilingDesc(
                    start_indices=(tm * BLOCK_SIZE_M, tn * BLOCK_SIZE_N),
                    tile_sizes=(BLOCK_SIZE_M, BLOCK_SIZE_N)
                )
                
                inputs_dep = {x: x_desc, w: w_desc}
                outs_tile_mapping = {y: y_desc}
                
                tasks.append(
                    cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                     extra_params, inputs_dep, outs_tile_mapping))
        
        return tasks
    
    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params)


@registry.register_task(op_type="mlp_fc1", task_cls=MLPFC1Task, config_factory=mlp_fc1_config_factory,
                        codegen_func=codegen_mlp_fc1)
class MLPFC1TaskBuilder(LinearTaskBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True)


@registry.register_task(op_type="mlp_fc2", task_cls=MLPFC2Task, config_factory=mlp_fc2_config_factory,
                        codegen_func=codegen_mlp_fc2)
class MLPFC2TaskBuilder(LinearTaskBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True)
