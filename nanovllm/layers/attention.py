import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context

# ============================================================================
# Attention 后端运行时分发
# ============================================================================
# FlashAttention 2 (flash_attn) 需要 Ampere(sm>=80)及以上 GPU。RTX 2060 是 sm75
# (Turing),FA2 无法运行。本文件在运行时按以下条件二选一:
#   - sm>=80 且 flash_attn / triton 均可导入 → 【原版 FA 路径】(保持原仓库代码不变)
#       flash_attn_varlen_func / flash_attn_with_kvcache + triton store_kvcache 内核
#   - 否则(sm<80 或缺少依赖) → 【SDPA 路径】
#       PyTorch scaled_dot_product_attention + index_copy_ 写 cache
# SDPA 在 sm75 上所有融合后端均不可用,自动回退 math 后端(慢但正确)。
# 两条路径互斥;sm75 上当前已跑通的功能完全不变。
# ============================================================================

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _FA_IMPORT_OK = True
except Exception:
    _FA_IMPORT_OK = False
    flash_attn_varlen_func = flash_attn_with_kvcache = None

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

_USE_FA = None  # 决策缓存,None 表示尚未决定


def _use_flash_attn() -> bool:
    """运行时判定是否走原版 FlashAttention 路径(仅决策一次并缓存)。"""
    global _USE_FA
    if _USE_FA is None:
        _USE_FA = (
            _FA_IMPORT_OK
            and _HAS_TRITON
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
        )
    return _USE_FA


# ----------------------------------------------------------------------------
# FA 路径:写 KV cache(原版 triton 内核,逐字保留)
# ----------------------------------------------------------------------------
if _HAS_TRITON:
    @triton.jit
    def _store_kvcache_kernel_fa(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


    def _store_kvcache_fa(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        _store_kvcache_kernel_fa[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ----------------------------------------------------------------------------
# SDPA 路径:写 KV cache(纯 PyTorch index_copy_,无 triton/MSVC 依赖)
# ----------------------------------------------------------------------------
def _store_kvcache_sdpa(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # key/value: [N, num_kv_heads, head_dim]; k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    # slot_mapping: [N], flat slot index = block_id*block_size + offset; -1 means skip.
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    k_flat = k_cache.reshape(-1, D)       # [num_blocks*block_size, D]
    v_flat = v_cache.reshape(-1, D)
    key_flat = key.reshape(N, D)
    value_flat = value.reshape(N, D)
    valid = slot_mapping != -1
    slots = slot_mapping[valid].long()
    k_flat.index_copy_(0, slots, key_flat[valid])
    v_flat.index_copy_(0, slots, value_flat[valid])


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """写 KV cache:FA 路径用 triton 内核,SDPA 路径用 index_copy_。"""
    if _use_flash_attn():
        _store_kvcache_fa(key, value, k_cache, v_cache, slot_mapping)
    else:
        _store_kvcache_sdpa(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):
    """PagedAttention 注意力算子(整个引擎最关键的模块)。

    两条运行时路径(由 _use_flash_attn() 决策):
      - FA 路径(sm>=80 且 flash_attn/triton 可用):原仓库逻辑,flash_attn 融合内核
      - SDPA 路径(sm<80 或缺依赖):PyTorch scaled_dot_product_attention

    三种注意力模式:
      - _prefill_packed:       普通 packed prefill(无前缀缓存),逐 seq SDPA + is_causal
      - _prefill_prefix_cache: 有前缀缓存/分块 prefill,从 paged cache gather k/v + 因果 mask
      - _decode:               批量 decode,每序列 1 个 query,padding 到等长 + mask

    KV cache 形状: [num_blocks, block_size, num_kv_heads, head_dim]
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """构造 Attention。

        输入:
            num_heads:    本卡的 q heads 数(已按 TP 切分)
            head_dim:     每个 head 的维度
            scale:        attention scale = head_dim ** -0.5
            num_kv_heads: 本卡的 kv heads 数(GQA 时 < num_heads)
        k_cache/v_cache 初始为空 tensor,allocate_kv_cache 后由 ModelRunner 挂载切片。
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])     # 占位,后由 ModelRunner 挂载

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """注意力前向。

        输入:
            q: [N, H, D]    N=本步 token 总数, H=num_heads, D=head_dim
            k: [N, Hkv, D]  Hkv=num_kv_heads (GQA 时 Hkv < H)
            v: [N, Hkv, D]
        输出:
            o: [N, H, D]    attention 输出(与 q 同形状)

        流程:
          1. store_kvcache: 把新算的 k/v 写入分页 cache 的对应槽位
          2. 据 is_prefill + block_tables 选三条路径之一计算注意力
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 1. 写 KV cache(warmup 时 k_cache 为空,跳过)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if _use_flash_attn():
            # ============ 原版 FA 路径(保持原仓库代码不变) ============
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache: k/v live in paged cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:    # decode
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            return o

        # ============ SDPA 路径(sm<80 或缺 FA 依赖) ============
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache / 分块 prefill
                o = self._prefill_prefix_cache(q, k_cache, v_cache, context)
            else:
                o = self._prefill_packed(q, k, v, context)
        else:    # decode
            o = self._decode(q, k_cache, v_cache, context)
        return o

    # ------------------------------------------------------------------
    # SDPA 实现(仅 sm<80 / 缺 FA 依赖时使用)
    # ------------------------------------------------------------------
    def _prefill_packed(self, q, k, v, context):
        """普通 packed prefill(无前缀缓存/分块):逐 seq SDPA + is_causal=True。

        输入:
            q: [N, H, D], k/v: [N, Hkv, D]
            cu_seqlens_q == cu_seqlens_k(无前缀缓存时 q/k 等长)
        输出: o [N, H, D]

        逐 seq 循环:每条 seq 独立做 causal SDPA,最后 cat 回去。
        (FA 路径用 flash_attn_varlen_func 一次融合处理所有 seq,更快)
        """
        # q: [N, H, D], k/v: [N, Hkv, D]; cu_seqlens_q == cu_seqlens_k (no prefix cache)
        cu = context.cu_seqlens_q
        outs = []
        for i in range(cu.shape[0] - 1):
            s = int(cu[i].item())
            e = int(cu[i + 1].item())
            qi = q[s:e].transpose(0, 1).unsqueeze(0)    # [1, H, L, D]
            ki = k[s:e].transpose(0, 1).unsqueeze(0)    # [1, Hkv, L, D]
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=self.scale, enable_gqa=True)
            outs.append(oi.squeeze(0).transpose(0, 1))    # [L, H, D]
        return torch.cat(outs, dim=0)

    def _prefill_prefix_cache(self, q, k_cache, v_cache, context):
        # q: [N, H, D], k/v: [N, Hkv, D]; cu_seqlens_q == cu_seqlens_k (no prefix cache)
        cu = context.cu_seqlens_q
        outs = []
        for i in range(cu.shape[0] - 1):
            s = int(cu[i].item())
            e = int(cu[i + 1].item())
            qi = q[s:e].transpose(0, 1).unsqueeze(0)    # [1, H, L, D]
            ki = k[s:e].transpose(0, 1).unsqueeze(0)    # [1, Hkv, L, D]
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True, scale=self.scale, enable_gqa=True)
            outs.append(oi.squeeze(0).transpose(0, 1))    # [L, H, D]
        return torch.cat(outs, dim=0)

    def _prefill_prefix_cache(self, q, k_cache, v_cache, context):
        """有前缀缓存/分块 prefill:从 paged cache gather k/v + 构造因果 mask。

        输入:
            q: [N, H, D] (新 token)
            k_cache/v_cache: [num_blocks, block_size, Hkv, D]
            cu_seqlens_q/cu_seqlens_k: 累积序列长度(k 比 q 长,含已缓存前缀)
            block_tables: [num_seqs, max_blocks]
        输出: o [N, H, D]

        对每条 seq:
          - q 覆盖绝对位置 [start, start+Lq), start = num_cached_tokens = Lk - Lq
          - k/v 从 paged cache 按 block_table gather,覆盖 [0, Lk)
          - 构造因果 mask:query j (位置 start+j) attend 到 k 列 [0, start+j]
          - 注意:mask 用 per-seq 局部坐标 (Lk-Lq),不是 cu_k 跨 seq 累计偏移
            (这是本地适配修复的关键,原版 FA 用 block_table 内置因果)
        """
        # q: [N, H, D] (new tokens); k/v gathered from paged cache via block_table.
        # For seq i: q covers absolute positions [start, start+Lq), k/v cover [0, Lk) with Lk=start+Lq.
        cu_q = context.cu_seqlens_q
        cu_k = context.cu_seqlens_k
        block_tables = context.block_tables    # [num_seqs, max_blocks]
        block_size = k_cache.shape[1]
        outs = []
        for i in range(cu_q.shape[0] - 1):
            qs, qe = int(cu_q[i].item()), int(cu_q[i + 1].item())
            ks, ke = int(cu_k[i].item()), int(cu_k[i + 1].item())
            Lq = qe - qs
            Lk = ke - ks
            nb = (Lk + block_size - 1) // block_size
            phys = block_tables[i, :nb]
            k_seq = k_cache[phys].reshape(-1, self.num_kv_heads, self.head_dim)[:Lk]    # [Lk, Hkv, D]
            v_seq = v_cache[phys].reshape(-1, self.num_kv_heads, self.head_dim)[:Lk]
            qi = q[qs:qe].transpose(0, 1).unsqueeze(0)    # [1, H, Lq, D]
            ki = k_seq.transpose(0, 1).unsqueeze(0)        # [1, Hkv, Lk, D]
            vi = v_seq.transpose(0, 1).unsqueeze(0)
            # q token j is at within-seq position (start + j), start = Lk - Lq (= num_cached_tokens);
            # it attends to k cols [0, start+j]. k_seq is per-seq local, so use start, not cu_k offset.
            rows = torch.arange(Lq, device=q.device) + (Lk - Lq)
            cols = torch.arange(Lk, device=q.device)
            mask = cols.unsqueeze(0) <= rows.unsqueeze(1)    # [Lq, Lk]
            oi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask.view(1, 1, Lq, Lk), scale=self.scale, enable_gqa=True)
            outs.append(oi.squeeze(0).transpose(0, 1))    # [Lq, H, D]
        return torch.cat(outs, dim=0)

    def _decode(self, q, k_cache, v_cache, context):
        """批量 decode:每序列 1 个 query,gather k/v padding 到等长,带 mask 一次算完。

        输入:
            q: [B, H, D]  B=batch_size,每序列 1 个 token
            k_cache/v_cache: [num_blocks, block_size, Hkv, D]
            context_lens: [B]  每序列当前长度(含正在处理的 token)
            block_tables: [B, max_blocks]
        输出: o [B, H, D]

        流程:
          1. 按各 seq 的 context_lens 算需要的 block 数,gather 出 k/v
          2. padding 到 max_L(最长序列)→ [B, max_L, Hkv, D]
          3. 构造 mask:cols < context_lens 的位置有效(屏蔽 padding)
          4. SDPA 一次算完所有 seq(无需逐 seq 循环)
        """
        # q: [B, H, D] (one token per seq); gather k/v per seq from paged cache.
        context_lens = context.context_lens    # [B]
        block_tables = context.block_tables    # [B, max_blocks]
        block_size = k_cache.shape[1]
        B = q.shape[0]
        max_L = int(context_lens.max().item())
        max_nb = (max_L + block_size - 1) // block_size
        phys = block_tables[:, :max_nb]                          # [B, max_nb]
        k_pad = k_cache[phys].reshape(B, max_nb * block_size, self.num_kv_heads, self.head_dim)[:, :max_L]
        v_pad = v_cache[phys].reshape(B, max_nb * block_size, self.num_kv_heads, self.head_dim)[:, :max_L]
        qi = q.unsqueeze(2)                    # [B, H, 1, D]
        ki = k_pad.transpose(1, 2)             # [B, Hkv, max_L, D]
        vi = v_pad.transpose(1, 2)
        cols = torch.arange(max_L, device=q.device)
        mask = cols.unsqueeze(0) < context_lens.unsqueeze(1)    # [B, max_L]
        o = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask.view(B, 1, 1, max_L), scale=self.scale, enable_gqa=True)
        return o.squeeze(2)    # [B, H, D]
