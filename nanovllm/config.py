import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """引擎全局配置,贯穿 Scheduler / ModelRunner / BlockManager。

    字段:
        model:                    模型目录路径(含 safetensors 权重与 HF config)
        max_num_batched_tokens:   单步 prefill 最多处理的 token 数(chunked prefill 的块大小上限)
        max_num_seqs:             单步最多并行的序列数(continuous batching 的 batch 上限)
        max_model_len:            单条序列最大长度(prompt + 生成)
        gpu_memory_utilization:   GPU 显存使用比例(0~1),KV cache 池据此分配
        tensor_parallel_size:     张量并行卡数(TP)
        enforce_eager:            True=禁用 CUDA graph 走 eager 前向;
                                  本地 SDPA 后端必须 True(非 graph-safe)
        hf_config:                从 model 目录读取的 HF AutoConfig
        eos:                      EOS token id(构造后由 tokenizer 填充)
        kvcache_block_size:       分页 KV cache 的 block 大小(必须 256 的倍数)
        num_kvcache_blocks:       KV cache 块数(warmup 后由 ModelRunner 填充,初始 -1)
    """

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # 基本校验
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0          # block 大小必须是 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8          # TP 卡数限制
        # 读取 HF 配置(含 num_hidden_layers / num_attention_heads / dtype 等)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 模型最大长度不超过 HF config 的 max_position_embeddings
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
