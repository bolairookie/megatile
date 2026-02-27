# Megatile

Mega kernel implementation based on [TileLang](https://github.com/tile-ai/tilelang). 
Inspired by [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)'s `mega_triton_kernel`.

## What it does

Fuses multiple GPU kernels into a single mega kernel using TileLang.

## Install

```bash
pip install tilelang
pip install -e .
```

## Example

```python
import torch
from megatile import ModelBuilder

builder = ModelBuilder(num_warps=4)

input = torch.randn(1024, 512, dtype=torch.bfloat16, device="cuda").contiguous()
weight = torch.randn(1024, 512, dtype=torch.bfloat16, device="cuda").contiguous()
output = torch.empty(1024, 1024, dtype=torch.bfloat16, device="cuda").contiguous()

builder.make_linear(input, weight, output, layer_id=0)
builder.compile()
builder.run()
```

## Structure

- `core/` - task management, scheduling, code generation
- `tasks/` - task builders (e.g., linear)
- `kernels/` - TileLang kernel implementations
- `models/` - ModelBuilder API

## Credits

This project is based on [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)'s `mega_triton_kernel` implementation. 
Thanks to the ByteDance Seed team for their excellent work.
