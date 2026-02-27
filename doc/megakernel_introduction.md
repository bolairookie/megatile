# Triton-Distributed Mega Kernel 架构与原理深度分析

## 目录

1. [概述](#概述)
2. [核心设计理念](#核心设计理念)
3. [架构组件详解](#架构组件详解)
4. [执行流程分析](#执行流程分析)
5. [依赖管理与同步机制](#依赖管理与同步机制)
6. [任务调度策略](#任务调度策略)
7. [代码生成机制](#代码生成机制)
8. [与 Megatile 的对比](#与-megatile-的对比)
9. [性能优化要点](#性能优化要点)
10. [总结](#总结)

---

## 1. 概述

### 1.1 什么是 Mega Kernel

**Mega Kernel** 是一种将多个独立的计算任务（kernels）合并到单个 GPU kernel 中执行的架构模式。与传统的多 kernel 启动方式不同，mega kernel 通过：

- **统一的任务队列管理**：所有任务在编译时或运行时被组织到工作队列中
- **动态任务调度**：每个 SM（Streaming Multiprocessor）从队列中动态获取任务执行
- **依赖感知执行**：通过 scoreboard 机制确保任务间的数据依赖关系
- **细粒度同步**：使用 release-acquire 内存模型实现 tile 级别的同步

### 1.2 核心优势

1. **减少 Kernel Launch Overhead**
   - 传统方式：每个操作启动一个 kernel，产生多次 kernel launch 开销
   - Mega Kernel：一次 launch，内部动态调度，大幅减少开销

2. **更好的资源利用率**
   - 任务可以在不同 SM 间动态分配
   - 避免某些 SM 空闲而其他 SM 过载

3. **细粒度依赖管理**
   - Tile 级别的依赖跟踪（而非整个 kernel）
   - 允许更早的并行执行

4. **统一的代码生成**
   - 所有任务类型在同一个 kernel 中，便于优化和调试

---

## 2. 核心设计理念

### 2.1 任务抽象（Task Abstraction）

每个计算操作被抽象为一个或多个 **Task**：

```python
@dataclass
class TaskBase:
    layer_id: int          # 层标识（用于依赖管理）
    task_id: int           # 任务标识（同一层内的不同任务）
    tile_id_or_start: int  # Tile ID 或起始位置
    num_tiles: int         # Tile 数量
    config: ConfigBase     # Kernel 配置（BLOCK_SIZE, NUM_STAGES 等）
    dependency: List[TaskDependency]  # 依赖关系
    io_tensors: List[List[torch.Tensor]]  # 输入输出张量
```

**关键设计点**：
- **Tile-wise 分解**：大矩阵运算被分解为多个 tile，每个 tile 是一个独立任务
- **依赖描述**：通过 `TaskDependency` 描述对哪些 tile 有依赖
- **配置参数化**：每个任务可以有不同的 `BLOCK_SIZE` 等配置

### 2.2 工作队列（Work Queue）

工作队列是任务调度的核心数据结构：

```
work_queues: [MAX_TASKS_PER_SM, NUM_SMS, INT_PER_TASK]
```

**结构说明**：
- **第一维**：每个 SM 的任务队列（最多 `MAX_TASKS_PER_SM` 个任务）
- **第二维**：SM 索引（`NUM_SMS` 个 SM）
- **第三维**：每个任务的编码信息（`INT_PER_TASK` 个整数）

**任务编码格式**：
```
[task_type, layer_id, task_id, tile_id, deps_start, deps_end, io_tensors...]
```

### 2.3 Scoreboard 机制

Scoreboard 是一个 3D 张量，用于跟踪每个 tile 的完成状态：

```
scoreboard: [max_layer_id + 1, max_task_id + 1, MAX_NUM_TILES_PER_OP]
```

**工作原理**：
- **生产者**：任务完成后，设置 `scoreboard[layer_id][task_id][tile_id] = TILE_READY_SIGNAL`
- **消费者**：任务开始前，检查所有依赖的 tile 是否都已完成
- **同步语义**：使用 release-acquire 内存模型确保可见性

---

## 3. 架构组件详解

### 3.1 TaskBaseInfo（任务基础信息）

Triton 使用聚合类（`@tl.core._aggregate`）来表示任务信息：

```python
@tl.core._aggregate
class TaskBaseInfo:
    io_tensors_ptr: tl.tensor      # 指向 IO 张量描述符的指针
    task_type: tl.tensor           # 任务类型 ID
    layer_id: tl.tensor            # 层 ID
    task_id: tl.tensor             # 任务 ID
    tile_id_or_start: tl.tensor    # Tile ID
    depend_entry_start: tl.tensor  # 依赖条目起始索引
    depend_entry_end: tl.tensor    # 依赖条目结束索引
    MAX_NUM_TENSOR_DIMS: tl.constexpr
```

**设计优势**：
- **紧凑表示**：所有信息打包在一个聚合对象中
- **指针传递**：通过指针访问，减少参数传递开销
- **类型安全**：编译时类型检查

### 3.2 TensorDesc（张量描述符）

用于在 kernel 中表示张量信息：

```python
@tl.core._aggregate
class TensorDesc:
    base_ptr: tl.tensor  # 指向 (data_ptr, shape[0], shape[1], ...) 的指针
    
    def data_ptr(self, dtype):
        # 从 base_ptr 加载 data_ptr
        buf_ptr = self.base_ptr.to(tl.pointer_type(tl.uint64))
        data_ptr = tl.load(buf_ptr).to(tl.pointer_type(dtype))
        return tl.multiple_of(data_ptr, 16)  # 对齐要求
    
    def size(self, i, multiple=1):
        # 从 base_ptr + i + 2 加载 shape[i]
        dim = tl.load(self.base_ptr + i + 2)
        return dim.to(tl.int32)
```

**关键点**：
- **动态形状**：运行时从内存中读取 shape 信息
- **对齐保证**：`multiple_of` 确保内存对齐，提升性能
- **灵活索引**：支持不同维度的张量

### 3.3 Scoreboard（记分板）

Scoreboard 类封装了依赖等待和 tile 释放的逻辑：

```python
@tl.core._aggregate
class Scoreboard:
    scoreboard_table: tl.tensor      # 3D scoreboard 的扁平化指针
    task_deps_ptr: tl.tensor         # 依赖条目数组指针
    INT_PER_DEPS: tl.constexpr       # 每个依赖条目的整数数量
    MAX_TASK_ID: tl.constexpr
    MAX_NUM_TASK_PER_OP: tl.constexpr
    TILE_READY_SIGNAL: tl.constexpr  # 通常为 1
    NUM_THREADS: tl.constexpr
    WARP_SIZE: tl.constexpr = 32
```

**核心方法**：

1. **wait_deps**：等待所有依赖的 tile 完成
   ```python
   def wait_deps(self, task_base_info: TaskBaseInfo):
       # 遍历所有依赖条目
       for t in range(entry_start, entry_end):
           l, r = load_dependency_range(t)  # [l, r) 是需要等待的 tile 范围
           num_signals = r - l
           # 每个 warp 负责一部分 tile 的等待
           for i in range(lane_id, num_signals, WARP_SIZE):
               while atomic_load(scoreboard_table[l + i], "acquire") != TILE_READY_SIGNAL:
                   pass  # 忙等待
       __syncthreads()  # 确保所有线程都完成等待
   ```

2. **release_tile**：释放一个 tile，通知依赖它的任务
   ```python
   def release_tile(self, task_base_info: TaskBaseInfo, tile_id):
       sb_offset = task_scoredboard_start(task_base_info)
       __syncthreads()  # 确保所有计算完成
       if thread_idx == 0:
           atomic_store(scoreboard_table[sb_offset + tile_id], 
                       TILE_READY_SIGNAL, "release")
       __syncthreads()  # 避免线程分歧
   ```

### 3.4 依赖编码（Dependency Encoding）

依赖关系通过 `task_deps_ptr` 数组编码：

```
task_deps_ptr: [num_deps_entries, 2]
每个条目: [l, r] 表示 scoreboard 中的 tile 范围 [l, r)
```

**编码过程**：
1. 收集所有任务的依赖关系
2. 将依赖转换为 scoreboard 中的偏移量范围
3. 存储为 `(layer_id, task_id, start_tiles, end_tiles)` 的区间

**示例**：
- Task A 依赖 Task B 的 tile [0, 4)
- Task B 的 scoreboard 偏移：`layer_B * MAX_TASK_ID * MAX_TILES + task_B * MAX_TILES`
- 依赖条目：`[offset + 0, offset + 4)`

---

## 4. 执行流程分析

### 4.1 编译时流程

```
用户代码
  ↓
ModelBuilder.make_linear/make_norm/... (添加任务)
  ↓
ModelBuilder.compile()
  ├─> Scheduler.enque_tasks() (任务调度)
  │   ├─> 按 SM 分配任务
  │   ├─> 计算依赖关系
  │   └─> 生成 work_queues 和 task_deps
  ├─> CodeGenerator.generate_code() (代码生成)
  │   ├─> 为每个任务类型生成 dispatch 代码
  │   └─> 生成 MEGA_TRITON_KERNEL
  └─> 编译生成的 kernel
```

### 4.2 运行时流程

```
MEGA_TRITON_KERNEL 启动
  ↓
每个 SM 执行：
  ├─> 1. 获取任务数量：num_tasks = num_tasks_per_sm[sm_id]
  ├─> 2. 初始化：cur_task_idx = 0
  ├─> 3. 循环处理任务：
  │   ├─> 3.1 FETCH_TASK: 从 work_queues 加载任务信息
  │   ├─> 3.2 构建 TaskBaseInfo
  │   ├─> 3.3 scoreboard.wait_deps(): 等待依赖完成
  │   ├─> 3.4 根据 task_type 执行对应的任务 kernel
  │   │   ├─> linear_task_compute()
  │   │   ├─> rmsnorm_task_compute()
  │   │   └─> ...
  │   └─> 3.5 scoreboard.release_tile(): 标记 tile 完成
  └─> 4. 处理下一个任务
```

### 4.3 任务执行示例（Linear Task）

```python
@triton.jit
def linear_task_compute(task_base_info: TaskBaseInfo, scoreboard: Scoreboard, ...):
    # 1. 从 TaskBaseInfo 获取张量信息
    input = task_base_info.get_tensor(0)
    weight = task_base_info.get_tensor(1)
    output = task_base_info.get_tensor(2)
    
    # 2. 获取张量指针和形状
    M = input.size(0)
    K = input.size(1, ALIGNMENT_K)
    N = weight.size(0)
    a_ptr = input.data_ptr(tl.bfloat16)
    b_ptr = weight.data_ptr(tl.bfloat16)
    c_ptr = output.data_ptr(tl.bfloat16)
    
    # 3. 执行 tile-wise matmul
    tile_id = task_base_info.tile_id_or_start
    tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, ...)
    
    # 4. 释放 tile，通知依赖它的任务
    scoreboard.release_tile(task_base_info, tile_id)
```

---

## 5. 依赖管理与同步机制

### 5.1 Release-Acquire 内存模型

Mega Kernel 使用 **Release-Acquire** 内存模型确保依赖关系的正确性：

**Release（释放）语义**：
- `atomic_store(scoreboard[tile_id], TILE_READY_SIGNAL, "release")`
- 确保所有**之前**的内存写入在 store 之前完成
- 其他线程通过 acquire load 可以看到这些写入

**Acquire（获取）语义**：
- `atomic_load(scoreboard[tile_id], "acquire")`
- 确保所有**之后**的内存读取在 load 之后执行
- 可以看到对应 release store 之前的所有写入

**为什么需要原子操作**：
- 普通 load/store 没有内存屏障，可能导致：
  - 数据竞争（data race）
  - 可见性问题（visibility）
  - 死锁（依赖等待永远不满足）

### 5.2 依赖等待的优化

**Warp-level 并行等待**：
```python
lane_id = thread_idx % WARP_SIZE
warp_id = thread_idx // WARP_SIZE
for t in range(entry_start + warp_id, entry_end, num_warps()):
    # 每个 warp 处理一部分依赖条目
    l, r = load_dependency_range(t)
    for i in range(lane_id, num_signals, WARP_SIZE):
        # 每个 lane 等待一个 tile
        while atomic_load(...) != TILE_READY_SIGNAL:
            pass
```

**优势**：
- 减少线程分歧（divergence）
- 更好的内存访问模式（coalesced）
- 利用 warp 的 SIMD 特性

### 5.3 依赖优化（Dependency Optimization）

通过 `Graph` 类进行依赖优化：

```python
class Graph:
    def to_tasks(self):
        # 合并相同 producer 的依赖
        # 减少依赖条目的数量
        # 优化任务执行顺序
```

**优化策略**：
1. **依赖合并**：如果多个任务依赖同一个 producer，合并依赖条目
2. **区间压缩**：将连续的 tile 依赖合并为区间
3. **早期执行**：尽可能早地执行没有依赖的任务

---

## 6. 任务调度策略

### 6.1 静态调度（Static Scheduling）

**Round-Robin 策略**：
- 任务按顺序分配给 SM
- SM 0 处理任务 0, 4, 8, ...
- SM 1 处理任务 1, 5, 9, ...
- 简单、可预测，但可能负载不均

**Zig-Zag 策略**：
- 任务在 SM 间交替分配
- 更好的负载均衡

### 6.2 运行时调度（Runtime Scheduling）

通过 `enable_runtime_scheduler=True` 启用：

```python
# 使用原子操作动态分配任务
cur_task_idx = tl.atomic_add(work_queue_start, 1, "release")
```

**优势**：
- 动态负载均衡
- 适应不同任务的计算时间差异

**劣势**：
- 额外的原子操作开销
- 可能影响缓存局部性

### 6.3 任务预取（Task Prefetching）

通过 `enalbe_task_prefetch=True` 启用：

```python
# 在处理当前任务时，预取下一个任务
nxt_task_idx = cur_task_idx + 1
if nxt_task_idx < num_tasks:
    nxt_task_base_info = FETCH_TASK(work_queues, nxt_task_idx, ...)
```

**优势**：
- 隐藏内存访问延迟
- 提升流水线效率

---

## 7. 代码生成机制

### 7.1 任务分发代码生成

CodeGenerator 为每个任务类型生成条件分支：

```python
if task_type == LINEAR_TASK_TYPE:
    linear_task_compute(task_base_info, scoreboard, ...)
elif task_type == RMSNORM_TASK_TYPE:
    rmsnorm_task_compute(task_base_info, scoreboard, ...)
elif task_type == SILU_MUL_UP_TASK_TYPE:
    silu_mul_up_task_compute(task_base_info, scoreboard, ...)
```

**优化**：
- 按任务类型频率排序（高频任务在前）
- 减少平均分支预测失败

### 7.2 任务编码生成

每个任务通过 `encoding_with_deps` 方法编码：

```python
def encoding_with_deps(self, l, r) -> Tuple[int]:
    """
    task_type | layer_id | task_id | tile_id_or_start | 
    dependency(l, r) | io_tensors | extra_params
    """
    entrys = [
        self.get_task_type_id(),
        self.layer_id,
        self.task_id,
        self.tile_id_or_start,
        l, r,  # 依赖范围在 task_deps 中的索引
    ]
    entrys += self.io_to_tuple()  # 张量信息
    entrys += self.extra_params_to_tuple()  # 额外参数
    return tuple(entrys)
```

**张量编码**：
```python
def io_to_tuple(self):
    # 每个张量编码为: (ptr_low, ptr_high, shape[0], shape[1], ...)
    # 确保对齐（长度为偶数）
    for tensor in all_tensors:
        data_ptr = tensor.data_ptr()
        ptr_high = (data_ptr >> 32) & 0xFFFFFFFF
        ptr_low = data_ptr & 0xFFFFFFFF
        shape = list(tensor.shape) + [1] * (MAX_NUM_TENSOR_DIMS - len(shape))
        tensor_tuple = (ptr_low, ptr_high) + tuple(shape)
        io_tuple += tensor_tuple
    return io_tuple
```

---

## 8. 与 Megatile 的对比

### 8.1 架构差异

| 特性 | Triton-Distributed | Megatile (TileLang) |
|------|-------------------|---------------------|
| **任务信息传递** | TaskBaseInfo 聚合类 | 单独参数传递 |
| **张量描述** | TensorDesc (指针) | 直接 Buffer 参数 |
| **Scoreboard** | Scoreboard 类 | 扁平 Buffer + 辅助函数 |
| **代码生成** | Triton JIT | TileLang JIT |
| **内存模型** | Release-Acquire | Release-Acquire |

### 8.2 适配原因

**TileLang 的限制**：
1. **无法动态索引 tuple**：Triton 使用 `TaskBaseInfo.get_tensor(idx)`，TileLang 需要显式参数
2. **聚合类支持**：Triton 有 `@tl.core._aggregate`，TileLang 使用 dataclass
3. **指针操作**：Triton 支持指针算术，TileLang 使用索引

**功能等价性**：
- 虽然实现方式不同，但功能完全等价
- 依赖管理、同步机制、调度策略都保持一致

### 8.3 对齐状态

✅ **已完全对齐**：
- 任务抽象（TaskBase）
- 依赖管理（TaskDependency）
- Scoreboard 机制（wait_deps, release_tile）
- 调度策略（Round-Robin）
- 代码生成流程

✅ **合理适配**：
- 参数传递方式（单独参数 vs 聚合类）
- 张量访问方式（Buffer vs TensorDesc）
- 代码生成后端（TileLang vs Triton）

---

## 9. 性能优化要点

### 9.1 内存访问优化

1. **对齐要求**：
   - 张量 data_ptr 必须 16 字节对齐
   - 使用 `multiple_of` 提示编译器

2. **合并访问**：
   - Work queue 访问：每个 SM 访问连续内存
   - Scoreboard 访问：Warp-level 合并

3. **缓存友好**：
   - 任务信息紧凑编码
   - 减少间接访问

### 9.2 计算优化

1. **Tile 大小调优**：
   - `BLOCK_SIZE_M/N/K` 影响：
     - 共享内存使用
     - 寄存器压力
     - 计算效率
   - 需要根据问题规模和 GPU 型号调优

2. **Pipeline 深度**：
   - `NUM_STAGES` 影响内存带宽利用
   - 更多 stages = 更好的流水线，但需要更多共享内存

3. **任务粒度**：
   - 更小的 tile = 更细的依赖，但更多 overhead
   - 更大的 tile = 更粗的依赖，但更好的计算效率

### 9.3 同步优化

1. **减少同步点**：
   - 合并多个 tile 的释放
   - 批量等待依赖

2. **Warp 同步**：
   - 使用 `__syncthreads()` 而非全局同步
   - 利用 warp shuffle 减少共享内存访问

---

## 10. 总结

### 10.1 核心创新

1. **统一任务抽象**：将不同操作统一为 Task，便于管理和调度
2. **细粒度依赖**：Tile 级别的依赖跟踪，允许更早的并行
3. **动态调度**：运行时任务分配，适应负载变化
4. **Release-Acquire 同步**：确保依赖关系的正确性

### 10.2 适用场景

**适合**：
- 多个小到中等规模的 kernel
- 有复杂依赖关系的计算图
- 需要减少 kernel launch 开销的场景

**不适合**：
- 单个大规模 kernel（传统方式更简单）
- 没有依赖关系的独立操作（可能增加复杂度）

### 10.3 未来方向

1. **自动调优**：BLOCK_SIZE 等参数的自动选择
2. **更智能的调度**：基于任务历史的动态调度
3. **依赖优化**：更激进的依赖合并和优化
4. **多 GPU 支持**：跨 GPU 的任务调度

---

## 附录：关键数据结构

### A.1 TaskBase 编码格式

```
[task_type, layer_id, task_id, tile_id, deps_start, deps_end, 
 io_tensor_0_ptr_low, io_tensor_0_ptr_high, io_tensor_0_shape[0], io_tensor_0_shape[1], ...,
 io_tensor_1_ptr_low, io_tensor_1_ptr_high, io_tensor_1_shape[0], io_tensor_1_shape[1], ...,
 ...]
```

### A.2 Scoreboard 布局

```
scoreboard[layer_id][task_id][tile_id] = TILE_READY_SIGNAL (1) or 0

扁平化偏移计算：
offset = layer_id * (MAX_TASK_ID + 1) * MAX_NUM_TILES_PER_OP +
         task_id * MAX_NUM_TILES_PER_OP +
         tile_id
```

### A.3 依赖条目格式

```
task_deps_ptr[entry_idx] = [l, r]

l, r 是 scoreboard 扁平化后的偏移量
表示需要等待的 tile 范围 [l, r)
```

---
