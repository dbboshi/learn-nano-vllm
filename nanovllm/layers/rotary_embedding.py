# ============================================================================
# RotaryEmbedding —— 旋转位置编码 (RoPE)
# ============================================================================
# RoPE 通过对 q/k 做旋转矩阵乘法来注入位置信息,优势:
#   - 相对位置编码(无需学习)
#   - 外推性好(可扩展到训练时未见的长度)
#
# cos_sin_cache 形状: [max_position, 1, head_dim]
#   由 cos/sin 各 [max_position, head_dim/2] 拼接而成
#   (inv_freq 长度 = head_dim/2,经 einsum 得 [max_pos, head_dim/2] 的 freqs)
# 前向时按 positions 索引取出,chunk(2) 拆回 cos/sin。
#
# apply_rotary_emb 把 x 拿成两半 x1/x2,做旋转:
#   y1 = x1*cos - x2*sin
#   y2 = x2*cos + x1*sin
#   return cat(y1, y2)
#
# @torch.compile 在 sm75 上仅产生警告,不影响正确性。
# get_rope 用 lru_cache 缓存,同一组参数只构造一个实例。
# ============================================================================
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对 x 应用 RoPE 旋转。

    输入:
        x:   [..., head_dim]
        cos: [..., head_dim/2]  (broadcast 到 x 的前几维)
        sin: [..., head_dim/2]
    输出: 旋转后的 x,同形状
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)    # 拆成前后两半,各 [..., head_dim/2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """预计算 cos_sin_cache。

        输入:
            head_size:           每个 head 的维度
            rotary_dim:          旋转维度(本实现中 == head_size)
            max_position_embeddings: 最大位置数
            base:                频率基数(默认 10000,Qwen3 用 1000000)
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # inv_freq: [rotary_dim/2]  频率倒数,指数衰减
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # t: [max_position]  位置序列
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # freqs: [max_position, rotary_dim/2]  位置×频率
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # cache: [max_position, 1, rotary_dim]  (cos/sin 各 rotary_dim/2,拼接后 rotary_dim)
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按 positions 取 cos/sin 表,对 q/k 应用旋转。

        输入:
            positions: [N]  各 token 的绝对位置
            query:     [N, H, head_dim]
            key:       [N, Hkv, head_dim]
        输出: 旋转后的 (query, key),同形状
        """
        cos_sin = self.cos_sin_cache[positions]              # [N, 1, head_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)                  # 各 [N, 1, head_dim/2]
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """工厂函数 + lru_cache:同一组参数全局只构造一个 RotaryEmbedding 实例。"""
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
