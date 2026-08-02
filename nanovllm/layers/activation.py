# ============================================================================
# SiluAndMul —— SiLU 门控激活函数
# ============================================================================
# 用于 MLP 的 gate_up 输出:把 [N, 2*intermediate] 拆成两半 gate / up,
# 输出 silu(gate) * up。SiLU = x * sigmoid(x)。
# @torch.compile 在 sm75 上仅产生警告(inductor 不支持 bf16),不影响正确性。
# ============================================================================
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入: x [N, 2*intermediate]; 输出: [N, intermediate]"""
        x, y = x.chunk(2, -1)       # gate, up  各 [N, intermediate]
        return F.silu(x) * y        # silu(gate) * up
