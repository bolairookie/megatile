################################################################################
#
# Copyright (c) 2025 Megatile Contributors
#
# Tasks module - Aligned with Triton-distributed
#
################################################################################

from .linear import (
    LinearTask,
    LinearConfig,
    LinearTaskBuilder,
    MLPFC1Task,
    MLPFC1Config,
    MLPFC1TaskBuilder,
    MLPFC2Task,
    MLPFC2Config,
    MLPFC2TaskBuilder,
)

from .norm import (
    RMSNormTask,
    RMSNormConfig,
    RMSNormTaskBuilder,
)

from .activation import (
    SiLUMulUpTask,
    SiLUMulUpConfig,
    SiLUMulUpTaskBuilder,
)

__all__ = [
    # Linear
    "LinearTask", "LinearConfig", "LinearTaskBuilder",
    "MLPFC1Task", "MLPFC1Config", "MLPFC1TaskBuilder",
    "MLPFC2Task", "MLPFC2Config", "MLPFC2TaskBuilder",
    # Norm
    "RMSNormTask", "RMSNormConfig", "RMSNormTaskBuilder",
    # Activation
    "SiLUMulUpTask", "SiLUMulUpConfig", "SiLUMulUpTaskBuilder",
]
