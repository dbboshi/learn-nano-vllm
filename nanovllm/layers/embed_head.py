# ============================================================================
# Embedding / LM Head —— 词嵌入与输出头(TP 并行)
# ============================================================================
# VocabParallelEmbedding: 词表按 tp 切分,各卡持 vocab/tp 个 token 的 embedding。
#   forward 时先把 input id 限制到本卡范围,查表后 all_reduce 聚合。
#
# ParallelLMHead: 继承 VocabParallelEmbedding,用于计算 logits。
#   - prefill 时只取每条序列最后一个 token 的 hidden 算 logits(省算力):
#       last_indices = cu_seqlens_q[1:] - 1
#   - tp>1 时用 gather 把各卡 logits 拼回 rank0 采样
# ============================================================================
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """词表并行 embedding。

        输入:
            num_embeddings: 词表大小(总 token 数)
            embedding_dim:  每个 token 的 embedding 维度
        """
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 权重: [vocab/tp, embedding_dim]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重 [vocab, dim] 中取本 rank 的词表分片。"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """查表。

        输入: x [N] token id 列表
        输出: y [N, embedding_dim]
        tp>1 时:先把 id 限制到本卡范围(范围外的置 0),查表后 mask 再 all_reduce。
        """
        if self.tp_size > 1:
            # 标记哪些 id 在本卡范围内
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)     # 范围外的 id 变成 0(查表不会越界)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y                  # 范围外的置 0
            dist.all_reduce(y)                         # 聚合各卡结果
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """计算 logits。

        输入: x [N, hidden_size] (prefill) 或 [B, hidden_size] (decode)
        输出: logits [num_seqs, vocab] (prefill, 只取每条末位) 或 [B, vocab] (decode)

        prefill 优化:只取每条序列最后一个 token 的 hidden 算 logits(cu_seqlens_q[1:]-1),
                    避免对所有 token 算 LM head(浪费算力)。
        tp>1 时:用 gather 把各卡 logits 拼回 rank0 采样。
        """
        context = get_context()
        if context.is_prefill:
            # 只取每条序列最后一个 token 的 hidden(prefill 只需生成下一个 token)
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)             # [num_seqs, vocab/tp] 或 [B, vocab/tp]
        if self.tp_size > 1:
            # 各卡 logits 拼回 rank0
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
