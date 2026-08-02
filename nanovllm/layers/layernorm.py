# ============================================================================
# RMSNorm —— Root Mean Square Normalization(含残差融合)
# ============================================================================
# RMSNorm: x' = x / sqrt(mean(x^2) + eps) * weight
# 比 LayerNorm 少减均值,计算量略小,常用于现代 LLM(Qwen/Llama)。
#
# 两个前向变体:
#   rms_forward(x):             纯 RMSNorm
#   add_rms_forward(x, residual): 残差相加 + RMSNorm 融合
#     先 x = x + residual,再 RMSNorm,返回 (normed, new_residual)
#     new_residual = x + residual(融合后避免再读一次显存)
#
# forward 据 residual 是否为 None 分派。
# @torch.compile 在 sm75 上仅产生警告,不影响正确性。
# ============================================================================
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))   # 可学习的缩放参数

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """纯 RMSNorm。输入/输出: x [..., hidden_size]"""
        orig_dtype = x.dtype
        x = x.float()                                          # 转 float32 计算保精度
        var = x.pow(2).mean(dim=-1, keepdim=True)             # 均方
        x.mul_(torch.rsqrt(var + self.eps))                   # 归一化
        x = x.to(orig_dtype).mul_(self.weight)                # 转回原 dtype 并乘权重
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """残差相加 + RMSNorm 融合。

        输入: x [..., H](当前层输出), residual [..., H](之前的残差)
        输出: (normed [..., H], new_residual [..., H])
              new_residual = x + residual(供下一层残差用)
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())                  # 残差相加
        residual = x.to(orig_dtype)                           # 更新残差(转回原 dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """据 residual 是否为 None 分派到 rms_forward 或 add_rms_forward。"""
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
