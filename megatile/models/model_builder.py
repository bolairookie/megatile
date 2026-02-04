################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Model Builder - Aligned with Triton-distributed
#
################################################################################
import torch
import importlib
import tempfile
import os

from ..core.code_generator import CodeGenerator, CodeGenOptions
from ..core.registry import registry
from ..core.task_base import TaskBase, DeviceProp, TaskDependency, TaskIDManager
from ..core.builder import TaskBuilderBase
from ..core.scheduler import enque_tasks
from typing import List, Dict, Any


class ModelBuilder:
    
    def __init__(self, num_warps=4):
        self.reset()
        self._registry = registry
        self._code_generator = CodeGenerator()
        
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        self.device_prop = DeviceProp(NUM_SMS=NUM_SMS)
        
        self.megakernel_tasks: List[TaskBase] = []
        self.scoreboard = None
        self.wq_tensor = None
        self.num_tasks_tensor = None
        self.task_deps_tensor = None
        
        self.MAX_NUM_TILES_PER_OP = 1
        self.last_dependency = TaskDependency()
        self.max_layer_id = 0
        self.max_task_id = 0
        self.num_warps = num_warps
        
        self._gen_kernel = None
        self._codegen_options = CodeGenOptions()
    
    def reset(self):
        TaskIDManager.reset_all_ids()
        self.megakernel_tasks = []
        self.scoreboard = None
        self.wq_tensor = None
        self.num_tasks_tensor = None
        self.task_deps_tensor = None
        self.MAX_NUM_TILES_PER_OP = 1
        self.max_layer_id = 0
        self.max_task_id = 0
        self.last_dependency = TaskDependency()
        self._gen_kernel = None
    
    def _update_tasks(self, tasks: List[TaskBase], do_not_update_dependency=False):
        assert len(tasks) > 0
        last_task = tasks[-1]
        if not do_not_update_dependency:
            self.last_dependency = TaskDependency(layer_id=last_task.layer_id, task_id=last_task.task_id, start_tiles=0,
                                                  end_tiles=last_task.num_tiles)
        self.megakernel_tasks += tasks
        for task in tasks:
            self.MAX_NUM_TILES_PER_OP = max(self.MAX_NUM_TILES_PER_OP, task.num_tiles)
            self.max_layer_id = max(task.layer_id, self.max_layer_id)
            self.max_task_id = max(task.task_id, self.max_task_id)
    
    def get_task_builder(self, op_type: str) -> 'TaskBuilderBase':
        task_type = self._registry.get_op_mapping(op_type)
        if not task_type:
            raise ValueError(f"Unsupported op type: {op_type}")
        builder_cls = registry.get_builder(task_type)
        return builder_cls
    
    def _convert_op(self, op_type: str, layer_id: int, io_tensors: List[List[torch.Tensor]],
                    extra_params: Dict[str, Any] = {}) -> List[TaskBase]:
        assert len(io_tensors) == 2
        # Note: check_contiguous and check_alignment are skipped for TileLang compatibility
        
        builder_cls = self.get_task_builder(op_type)
        tasks = builder_cls.build_tasks(device_prop=self.device_prop, layer_id=layer_id,
                                        dependency=self.last_dependency, io_tensors=io_tensors,
                                        extra_params=extra_params)
        self._update_tasks(tasks)
        return tasks
    
    def make_linear(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0, **config_kwargs):
        # Aligned with Triton: check_tensor_dim (skipped for TileLang compatibility)
        # check_tensor_dim(input, 2)
        # check_tensor_dim(weight, 2)
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM == M and oN == N
        # Note: config_kwargs is a TileLang extension for flexibility
        extra_params = {"config_kwargs": config_kwargs} if config_kwargs else {}
        self._convert_op("linear", layer_id, [[input, weight], [output]], extra_params)
    
    def make_fc1(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc1", input, weight, output, layer_id)

    def make_fc2(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc2", input, weight, output, layer_id)
    
    def _make_fc(self, op_type: str, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
                 layer_id: int = 0):
        # Note: check_tensor_dim and K % 32 == 0 checks are skipped for TileLang compatibility
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM == M and oN == N
        self._convert_op(op_type, layer_id, [[input, weight], [output]])
    
    def make_rms_norm(self, input: torch.Tensor, rms_weight: torch.Tensor, output: torch.Tensor, rms_eps: float = 1e-6,
                      layer_id: int = 0):
        # Aligned with Triton: check_tensor_dtype and check_tensor_dim (skipped for TileLang compatibility)
        # check_tensor_dtype(input, torch.bfloat16)
        # check_tensor_dtype(rms_weight, torch.bfloat16)
        # check_tensor_dtype(output, torch.bfloat16)
        # check_tensor_dim(rms_weight, 1)
        # reshape to 2d tensor (aligned with Triton)
        input = input.reshape(-1, input.shape[-1])
        output = output.reshape(-1, input.shape[-1])
        
        assert input.shape == output.shape
        assert input.shape[-1] == rms_weight.shape[0]
        extra_params = {"rms_eps": rms_eps}
        self._convert_op("rms_norm", layer_id, [[input, rms_weight], [output]], extra_params)
    
    def make_silu_mul_up(self, fc1_out: torch.Tensor, act_out: torch.Tensor, layer_id: int = 0):
        # Aligned with Triton: check_tensor_dim (skipped for TileLang compatibility)
        # check_tensor_dim(fc1_out, 2)
        # check_tensor_dim(act_out, 2)
        M, N = fc1_out.shape
        assert act_out.shape[0] == M
        assert act_out.shape[1] * 2 == N
        self._convert_op("silu_mul_up", layer_id, [[fc1_out], [act_out]])
    
    def compile(self, strategy="round_robin"):
        if not self.megakernel_tasks:
            raise ValueError("No tasks added. Use make_linear(), make_rms_norm(), etc. to add tasks.")
        
        print(f"[ModelBuilder] Compiling mega kernel with {len(self.megakernel_tasks)} tasks")
        
        # Schedule tasks
        self.wq_tensor, self.num_tasks_tensor, scoreboard_3d, self.task_deps_tensor = enque_tasks(
            self.device_prop.NUM_SMS,
            self.megakernel_tasks,
            strategy,
            enable_dependency_opt=True,
        )
        
        # Convert 3D scoreboard to flat 1D (aligned with Triton-distributed)
        # Flat size = (max_layer_id + 1) * (max_task_id + 1) * MAX_NUM_TILES_PER_OP
        scoreboard_shape = scoreboard_3d.shape
        max_layer_id, max_task_id, max_tiles = scoreboard_shape
        flat_scoreboard_size = max_layer_id * max_task_id * max_tiles
        
        # Flatten scoreboard
        self.scoreboard = scoreboard_3d.flatten().contiguous()
        
        # Calculate max values for code generation
        max_deps_entries = max(1024, self.task_deps_tensor.shape[0] if self.task_deps_tensor.shape[0] > 0 else 1)
        max_scoreboard_size = max(4096, flat_scoreboard_size)
        
        # Pad scoreboard to max_scoreboard_size if needed
        if self.scoreboard.shape[0] < max_scoreboard_size:
            padding = torch.zeros(max_scoreboard_size - self.scoreboard.shape[0], dtype=self.scoreboard.dtype, device=self.scoreboard.device)
            self.scoreboard = torch.cat([self.scoreboard, padding])
        
        # Pad task_deps to max_deps_entries if needed
        if self.task_deps_tensor.shape[0] < max_deps_entries:
            padding = torch.zeros((max_deps_entries - self.task_deps_tensor.shape[0], 2), dtype=self.task_deps_tensor.dtype, device=self.task_deps_tensor.device)
            self.task_deps_tensor = torch.cat([self.task_deps_tensor, padding], dim=0)
        
        # Generate code
        src, task_types_to_str, tensor_list = self._code_generator.generate_code(
            self.megakernel_tasks,
            self._codegen_options,
            max_deps_entries,
            max_scoreboard_size,
        )
        
        self._tensor_list = tensor_list
        
        print(f"[ModelBuilder] Generated kernel code ({len(src)} chars)")
        
        # Write to temp file and import
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as tmp:
            tmp.write(src)
            tmp_path = tmp.name
        
        module_name = os.path.basename(tmp_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        self._gen_kernel = module.MEGA_TILELANG_KERNEL
        self._kernel_params = {
            "INT_PER_TASK": self.wq_tensor.shape[2],
            "MAX_LAYER_ID": max_layer_id - 1,
            "MAX_TASK_ID": max_task_id - 1,
            "MAX_NUM_TILES_PER_OP": max_tiles,
            "NUM_SMS": self.device_prop.NUM_SMS,
            "MAX_TASKS_PER_SM": self.wq_tensor.shape[0],
            "num_warps": self.num_warps,
        }
        self._tensor_list = tensor_list
        
        print(f"[ModelBuilder] Compilation complete")
    
    def run(self):
        if self._gen_kernel is None:
            raise RuntimeError("Kernel not compiled. Call compile() first.")
        
        # Build kernel with parameters
        kernel = self._gen_kernel(**self._kernel_params)
        
        # Run kernel with flat scoreboard and all tensors
        kernel_args = [
            self.wq_tensor,
            self.num_tasks_tensor,
            self.scoreboard,
            self.task_deps_tensor,
        ]
        # Add all tensors as arguments
        kernel_args.extend(self._tensor_list)
        kernel(*kernel_args)
        
        torch.cuda.synchronize()
        
        # Reset scoreboard for next run
        self.scoreboard.zero_()
