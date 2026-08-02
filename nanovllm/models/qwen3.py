# ============================================================================
# Qwen3 模型结构 —— DecoderLayer 堆叠
# ============================================================================
# Qwen3ForCausalLM = Qwen3Model(Embed + N×DecoderLayer + Norm) + ParallelLMHead
#
# 每层 DecoderLayer:
#   hidden_states → input_layernorm → self_attn(QKV→RoPE→Attention→O) → +residual
#                 → post_attention_layernorm → MLP(gate_up→SiLU→down) → +residual
#
# Qwen3 特性:
#   - GQA (Grouped Query Attention): num_kv_heads < num_heads
#   - QK-Norm: 当 attention_bias=False 时对 q/k 做 RMSNorm(仅 Qwen3 有,config 控制)
#   - RoPE: 旋转位置编码
#   - tie_word_embeddings: 可选,lm_head 与 embed_tokens 共享权重
#
# packed_modules_mapping: 告诉 loader.py 如何把 HF 的 q_proj/k_proj/v_proj
#   合并装进 qkv_proj,gate_proj/up_proj 合并装进 gate_up_proj。
# ============================================================================
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 注意力层:QKV 投影 → RoPE → Attention → O 投影。

    含 GQA(分组查询)和可选的 QK-Norm(当 qkv_bias=False 时)。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size          # 本卡的 q heads
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size    # 本卡的 kv heads
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5                      # attention scale
        self.qkv_bias = qkv_bias

        # QKV 合并列并行(无通信),O 行并行(all_reduce)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # QK-Norm: 仅当 qkv_bias=False 时启用(Qwen3 特性)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """注意力前向。

        输入:
            positions:     [N] 各 token 绝对位置
            hidden_states: [N, hidden_size]
        输出: [N, hidden_size]
        """
        qkv = self.qkv_proj(hidden_states)                        # [N, q_size+2*kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)             # [N, H, D]
        k = k.view(-1, self.num_kv_heads, self.head_dim)          # [N, Hkv, D]
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)                                    # QK-Norm
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)                   # RoPE 旋转
        o = self.attn(q, k, v)                                    # PagedAttention
        output = self.o_proj(o.flatten(1, -1))                    # [N, hidden_size]
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 MLP:gate_up 列并行 → SiLU 门控 → down 行并行。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,                              # gate + up 合并
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """输入: x [N, hidden_size]; 输出: [N, hidden_size]"""
        gate_up = self.gate_up_proj(x)       # [N, 2*intermediate]
        x = self.act_fn(gate_up)             # [N, intermediate] (silu(gate)*up)
        x = self.down_proj(x)                # [N, hidden_size]
        return x


class Qwen3DecoderLayer(nn.Module):
    """单层 Decoder:自注意力 + MLP,带残差连接 + RMSNorm(融合)。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """一层前向(含残差融合)。

        输入:
            positions:     [N]
            hidden_states: [N, hidden_size]
            residual:      [N, hidden_size] 或 None(第一层时为 None)
        输出: (hidden_states, residual) 供下一层用
        """
        if residual is None:
            # 第一层:无残差,直接 RMSNorm
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层:残差相加 + RMSNorm 融合
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 模型主体:Embed + N 层 DecoderLayer + 最终 Norm。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """模型前向(不含 LM Head)。

        输入:
            input_ids: [N] (prefill) 或 [B] (decode)
            positions: [N] 或 [B]
        输出: hidden_states [N, hidden_size] 或 [B, hidden_size]
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)     # 最终 Norm + 残差融合
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 因果语言模型:Model + LM Head。

    packed_modules_mapping: 指导 loader.py 把 HF 权重名映射到合并后的参数:
      q_proj  → qkv_proj 的 "q" 分片
      k_proj  → qkv_proj 的 "k" 分片
      v_proj  → qkv_proj 的 "v" 分片
      gate_proj → gate_up_proj 的 0 分片
      up_proj   → gate_up_proj 的 1 分片
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 绑定权重:lm_head 与 embed_tokens 共享同一份参数
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向,返回 hidden_states(不含 logits)。"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算 logits(委托给 ParallelLMHead)。

        输入: hidden_states [N, hidden_size] 或 [B, hidden_size]
        输出: logits [num_seqs, vocab] (prefill) 或 [B, vocab] (decode)
        """
        return self.lm_head(hidden_states)
