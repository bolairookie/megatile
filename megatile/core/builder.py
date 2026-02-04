################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Task Builder Base - Aligned with Triton-distributed
#
################################################################################
from typing import List, Any, Dict, Callable, Union
from abc import ABC
from .task_base import TaskBase, DeviceProp, TaskDependency, TaskIDManager, OutputTilingDesc, InputDependencyDesc
from .config import ConfigBase
import torch


class TaskBuilderBase(ABC):
    # set by registry
    _create_config: Callable = None
    TASK_CLS: TaskBase = None

    @classmethod
    def log(cls, msg: str, level: str = "debug"):
        if level == "debug":
            pass
        else:
            print(f"[{cls.__name__}] {msg}")

    @classmethod
    def get_task_id(cls, layer_id: int) -> int:
        return TaskIDManager.get_task_id(layer_id)

    @classmethod
    def _create_task(
        cls,
        layer_id: int,
        task_id: int,
        tile_id_or_start: int,
        num_tiles: int,
        config: ConfigBase,
        dependency: Union[TaskDependency, List[TaskDependency]],
        io_tensors: List[torch.Tensor],
        extra_params: Dict[str, Any],
        inputs_dep: Dict[torch.Tensor, InputDependencyDesc] = None,
        outs_tile_mapping: Dict[torch.Tensor, OutputTilingDesc] = None,
    ):
        if isinstance(dependency, TaskDependency):
            dependency = [dependency]
        task_cls = cls.get_task_cls()
        return task_cls(
            layer_id=layer_id,
            task_id=task_id,
            tile_id_or_start=tile_id_or_start,
            num_tiles=num_tiles,
            config=config,
            dependency=dependency,
            io_tensors=io_tensors,
            extra_params=extra_params,
            inputs_dep={} if inputs_dep is None else inputs_dep,
            outs_tile_mapping={} if outs_tile_mapping is None else outs_tile_mapping,
        )

    @classmethod
    def get_problem_size(cls, io_tensors: List[torch.Tensor], extra_params: Dict[str, Any]):
        raise NotImplementedError

    @classmethod
    def create_config(cls, **kwargs) -> ConfigBase:
        if cls._create_config is None:
            raise RuntimeError("Config factory not initialized. Ensure the task is registered.")
        return cls._create_config(**kwargs)

    @classmethod
    def get_task_cls(cls) -> TaskBase:
        if cls.TASK_CLS is None:
            raise RuntimeError(f"task cls not initialized for {cls.__name__}")
        return cls.TASK_CLS

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List['torch.Tensor'], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params)
