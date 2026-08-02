# ============================================================================
# Scheduler —— continuous batching 调度器
# ============================================================================
# 每步决定:本步跑 prefill 还是 decode?选哪些 seq?每个 seq 算多少 token?
# 策略:prefill 优先(有 waiting 就先 prefill),无 waiting 才 decode。
# 支持的关键技术:
#   - continuous batching: 每步动态重组 batch,新请求随时插入
#   - chunked prefill:    超 prompt 的部分按 max_num_batched_tokens 分块,
#                          仅允许第一条序列分块(避免多序列同时分块死锁)
#   - preempt(抢占):     decode 时显存不足,把最后进入的序列踢回 waiting 重算
# ============================================================================
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     # 待 prefill 的队列(FIFO)
        self.running: deque[Sequence] = deque()     # 已 prefill 完、正在 decode 的队列

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """把新请求加入 waiting 队尾。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """调度一步,返回 (本步要跑的 seqs, is_prefill)。

        策略:
          1. 先尝试 prefill:从 waiting 队首取 seq,检查前缀缓存命中,
             分配 block,截断 num_scheduled_tokens 到 remaining(chunked prefill)。
             只允许第一条 seq 分块(后续 seq 要么整条进、要么等下一步)。
          2. 若 waiting 为空或有 seq 不可调度,转 decode:
             从 running 取每条 seq 各出 1 token;显存不足则抢占队尾序列。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ===== prefill 阶段:尽量把 waiting 中的序列塞进 batch =====
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 新序列:先查前缀缓存命中几个块
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break                       # 显存不够,等
                # 需要新算的 token 数 = 总 token - 命中的缓存 token
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已分配过(分块 prefill 的后续块):算剩余未处理的
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 仅允许第一条序列分块(避免多序列同时分块造成死锁)
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # chunked prefill 关键:本步只算 min(剩余, remaining) 个 token
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 如果本步把该序列的 prefill 算完了,移到 running
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # ===== decode 阶段:每条 running 序列各出 1 个 token =====
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 显存不足时抢占:把队尾序列踢回 waiting 释放块
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # can_append 通过后才 may_append(可能分配新块)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把本轮调度的序列放回 running 队首(保持顺序)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占:把序列踢回 waiting,释放其 block_table 占用的物理块。

        被抢占的序列下次会重新 prefill(已缓存的 block 可能被复用)。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """执行后更新:登记前缀缓存、推进进度、追加 token、判断完成。

        输入:
            seqs:       本步调度的序列列表
            token_ids:  采样出的 token id(每条 seq 对应一个)
            is_prefill: 本步是否为 prefill
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)         # 把新算好的完整 block 登记进前缀缓存
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # prefill 还没算完的序列不采样新 token(等下一步继续 prefill)
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # 判断是否完成:命中 EOS(且不 ignore_eos)或达到 max_tokens
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
