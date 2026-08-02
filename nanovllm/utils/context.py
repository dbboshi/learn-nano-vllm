# ============================================================================
# Context —— 进程级全局上下文
# ============================================================================
# 让 Attention / LMHead 等算子无需改签名就能拿到调度信息:
#   is_prefill, cu_seqlens_q/k, slot_mapping, context_lens, block_tables
#
# set_context 在 ModelRunner.prepare_prefill/decode 中写入,
# get_context 在 Attention.forward / ParallelLMHead.forward 中读取,
# reset_context 在 run 末尾清空。
#
# 这是 vLLM 风格的设计:避免在前向签名里层层透传调度参数。
# ============================================================================
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """单步推理的全局上下文(每步 run 前由 set_context 设置)。

    字段(按 prefill/decode 使用的不同):
        is_prefill:    True=prefill 阶段, False=decode 阶段
        cu_seqlens_q:  [num_seqs+1] 累积 query 序列长度(prefill 用,定位每条 seq 边界)
        cu_seqlens_k:  [num_seqs+1] 累积 key 序列长度(prefill 用,有前缀缓存时 > cu_seqlens_q)
        max_seqlen_q:  本步最长 query 序列长度(FA 路径用)
        max_seqlen_k:  本步最长 key 序列长度(FA 路径用)
        slot_mapping:  [N] 每个 token 写入 k_cache 的扁平槽位 = block_id*block_size+offset
                       (-1 表示跳过,CUDA graph 占位用)
        context_lens:  [B] 每序列当前长度(decode 用)
        block_tables:  [num_seqs, max_blocks] 逻辑→物理块映射(有前缀缓存/decode 时用)
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

# 进程级全局变量(单例)
_CONTEXT = Context()

def get_context():
    """获取当前上下文(Attention/LMHead 等算子调用)。"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """设置当前步的上下文(ModelRunner.prepare_* 调用)。"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """清空上下文(run 末尾调用,防止泄漏到下一步)。"""
    global _CONTEXT
    _CONTEXT = Context()
