# ============================================================================
# ModelRunner —— GPU 执行层
# ============================================================================
# 职责: 把调度器选出的 seqs 变成 GPU 上的张量(input_ids/positions/slot_mapping/
#       cu_seqlens/block_tables),跑模型前向,采样输出 token。
#
# 关键流程:
#   __init__  → 初始化 TP 进程组、建模型、加载权重、warmup、分配 KV cache 池
#   run       → 一次推理:prepare_prefill/decode → run_model → sampler
#   prepare_* → 构造张量并 set_context(全局上下文传给 Attention/LMHead)
#
# TP 多进程通信:
#   tp>1 时 rank0 通过共享内存(SharedMemory)广播方法名+参数给所有 worker,
#   各 rank 各自执行同一 run,实现 TP 同步。
#
# 本地适配(RTX 2060 / Windows):
#   - enforce_eager=False 直接抛 NotImplementedError(SDPA 非 graph-safe)
#   - NCCL 不可用时回退 gloo(Windows 无 NCCL,但 tp=1 可用)
#   - TCPStore(use_libuv=False)(Windows 无 libuv)
#   - tp=1 动态选端口(避免多实例端口冲突)
# ============================================================================
import os
import pickle
import socket
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """构造 ModelRunner。

        输入:
            config: 全局 Config
            rank:   本进程的 TP rank(0=主进程, >0=worker 子进程)
            event:  rank0 持有 list[Event](通知各 worker);
                    worker 持有单个 Event(等待 rank0 通知)
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        if not self.enforce_eager:
            # The SDPA-based attention uses data-dependent dynamic shapes
            # (context_lens.max() in _decode) and paged-cache gather, which are not
            # CUDA-graph compatible. The original flash_attn_with_kvcache kernel was
            # graph-safe; this rewrite is not. Fail fast instead of silently producing NaN.
            raise NotImplementedError(
                "enforce_eager=False (CUDA graph) is not supported with the SDPA attention "
                "backend on this environment; use enforce_eager=True."
            )

        backend = "nccl" if dist.is_nccl_available() else "gloo"    # Windows has no NCCL; gloo works for tp=1
        # Windows torch is built without libuv, so default TCPStore(use_libuv=True) fails.
        # For tp=1 pick a free port dynamically to avoid collisions across instances/runs;
        # for tp>1 all ranks must agree, so use NANOVLLM_PORT env var (default 2333).
        if self.world_size == 1:
            _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _s.bind(("localhost", 0))
            port = _s.getsockname()[1]
            _s.close()
        else:
            port = int(os.environ.get("NANOVLLM_PORT", "2333"))
        store = dist.TCPStore("localhost", port, self.world_size, rank == 0, use_libuv=False)
        dist.init_process_group(backend=backend, store=store, rank=rank, world_size=self.world_size)
        torch.cuda.set_device(rank)
        # 临时切换默认 dtype/device,让模型构造时直接在 GPU 上分配正确 dtype 的权重
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)             # 从 safetensors 装载权重
        self.sampler = Sampler()
        self.warmup_model()                               # warmup 测峰值显存
        self.allocate_kv_cache()                          # 据峰值把剩余显存切成分页 KV 池
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # tp>1 时建立共享内存通信:rank0 创建,worker 连接后进入 loop 阻塞等指令
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """退出清理:关闭共享内存、销毁进程组。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool              # 本地走不到(enforce_eager=True)
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """worker(rank>0)的主循环:阻塞等待 rank0 的指令并执行。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """worker 从共享内存读取 rank0 广播的 (method_name, args)。"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """rank0 把 (method_name, args) pickle 后写入共享内存,通知所有 worker。"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """调用自身方法;tp>1 时 rank0 先广播给 worker 再本地执行。"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """用最大规模跑一次 prefill 测峰值显存,为 KV cache 分配留出余量。

        规模: seq_len = min(max_num_batched_tokens, max_model_len),
               num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)
        用全 0 token 构造 dummy 序列,跑一次 prefill(不采样)。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据 warmup 测得的峰值显存,把剩余空间全分给分页 KV cache 池。

        公式: 可用 = total * gpu_mem_util - (used - current) - peak
               其中 (used - current) 是非 PyTorch 占用,peak 是模型峰值
        KV cache 形状: [2, num_layers, num_blocks, block_size, num_kv_heads_per_tp, head_dim]
                       第 0 维=k/v,第 1 维=层号
        分配后把各层的 k_cache/v_cache 切片挂到对应 Attention 模块上。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size     # TP 切分后的 kv heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 单个 block 占用的字节数: 2(k+v) * num_layers * block_size * num_kv_heads * head_dim * dtype
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 统一分配大 tensor,再给每层 Attention 挂切片(避免碎片)
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]     # [num_blocks, block_size, num_kv_heads, head_dim]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把各 seq 的 block_table 拼成等宽 int32 张量(不足的补 -1)。

        输入: seqs (每条 seq.block_table 是 list[int])
        输出: block_tables: [num_seqs, max_blocks] int32, 在 cuda 上
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """构造 prefill 输入张量并写入全局 Context。

        对每条 seq:
          - 取 [start=num_cached_tokens, end=start+num_scheduled_tokens) 的 token
          - positions = range(start, end)
          - cu_seqlens_q/k: vLLM 风格累积序列长度(用于在拼接 batch 张量里定位每条序列边界)
          - slot_mapping: 每个新 token 写入 k_cache 的扁平槽位 = block_id*block_size+offset
            (逐 block 计算,处理跨 block 边界的情况)

        若 cu_seqlens_k[-1] > cu_seqlens_q[-1](有前缀缓存/分块),需 block_tables 做 gather。

        输出: input_ids [N], positions [N] (N=本步总 token 数)
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens           # 前缀已算过的长度
            seqlen_q = seq.num_scheduled_tokens     # 本步要算的新 token 数
            end = start + seqlen_q
            seqlen_k = end                           # k 长度 = 已缓存 + 本步新算
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:                  # warmup 无 block_table
                continue
            # 为本步覆盖到的 block 计算 slot_mapping(写到 k_cache 的哪个槽)
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:                 # 第一个 block 可能从中间开始
                    slot_start += start % self.block_size
                if i != end_block - 1:               # 非最后一个 block:整块
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:                                # 最后一个 block:可能只到中间
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:     # 有前缀缓存/分块 → 需要 block_tables 做 gather
            block_tables = self.prepare_block_tables(seqs)
        # 打包成 cuda 张量(pin_memory + non_blocking 加速 H2D)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """构造 decode 输入张量并写入全局 Context。

        每条 seq 各出 1 个 token(seq.last_token),位置 = len(seq)-1,
        slot = 最后一个 block 的最后一个槽位。
        context_lens = 每条 seq 当前长度(含正在处理的 token)。

        输出: input_ids [B], positions [B] (B=batch_size=seqs 数)
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)        # 上一步生成的 token
            positions.append(len(seq) - 1)           # 该 token 的绝对位置
            context_lens.append(len(seq))            # 当前序列长度
            # 该 token 的 K/V 写入最后一个 block 的最后一个槽
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """收集各 seq 的 temperature,打包成 cuda 张量(仅 rank0 采样)。"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """模型前向 + 计算 logits。

        输入:
            input_ids: [N] (prefill) 或 [B] (decode)
            positions: [N] 或 [B]
            is_prefill: 是否 prefill 阶段
        输出: logits [N, vocab] (prefill) 或 [B, vocab] (decode)

        分两条路径:
          - eager 路径(is_prefill / enforce_eager / decode batch>512):直接前向
          - CUDA graph 路径(本地已禁用,enforce_eager=True 时走不到)
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # ===== CUDA graph 路径(本地 SDPA 后端不可用,enforce_eager=True 时走不到) =====
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """一次完整推理:构造输入 → 前向 → 采样。

        输入: seqs(调度器选出), is_prefill
        输出: 采样出的 token_id 列表(仅 rank0;worker 返回 None)
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获 CUDA graph(本地 SDPA 后端不可用,此方法不会被调用)。

        为不同 batch size 预先捕获 graph,decode 时直接 replay 避免 kernel launch 开销。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
