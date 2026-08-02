# ============================================================================
# Sequence —— 单条请求的状态机
# ============================================================================
# 记录一条推理请求的全部状态:token 列表、block_table(逻辑→物理块映射)、
# 调度进度(num_cached_tokens / num_scheduled_tokens)、采样参数等。
#
# 状态流转:
#   WAITING  --(prefill 完成: num_cached==num_tokens)--> RUNNING
#   RUNNING  --(显存不足被抢占 preempt)-->              WAITING
#   RUNNING  --(命中 EOS 或达 max_tokens)-->            FINISHED → 释放 block_table
#
# __getstate__/__setstate__ 用于 TP 多进程间的 pickle 传输:
#   prefill 时传完整 token_ids(decode 阶段只需要 last_token + 元数据)。
# ============================================================================
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列生命周期状态。"""
    WAITING = auto()    # 等待 prefill(或被抢占后重新等待)
    RUNNING = auto()    # prefill 完成,正在 decode
    FINISHED = auto()   # 生成完毕,可释放资源


class Sequence:
    # 类属性:所有 Sequence 共享的 block 大小(由 Config 在引擎构造时设置)
    block_size = 256
    # 全局自增 seq_id 生成器
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """构造一条请求。

        输入:
            token_ids:      prompt 的 token id 列表(会被 copy 一份防外部修改)
            sampling_params: 采样参数
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)             # 完整序列(prompt + 已生成的 token)
        self.last_token = token_ids[-1]              # 最后一个 token(decode 输入用)
        self.num_tokens = len(self.token_ids)        # 当前序列总长度
        self.num_prompt_tokens = len(token_ids)      # prompt 长度(不含生成)
        self.num_cached_tokens = 0                   # 已算过/已缓存的前缀长度(prefill 推进)
        self.num_scheduled_tokens = 0                # 本步要算的 token 数
        self.is_prefill = True                       # 当前是否处于 prefill 阶段
        self.block_table = []                        # 逻辑块→物理块 id 映射列表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        """支持 seq[start:end] 切片取 token_ids 子段。"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数(不含 prompt)。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """当前序列占用的逻辑块数(向上取整)。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中已写入的 token 数(1~block_size)。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """取第 i 个逻辑块对应的 token_ids 列表。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """追加一个新生成的 token(decode 每步调用)。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """pickle 序列化:TP 多进程传输时压缩数据量。

        prefill 阶段传完整 token_ids(子进程需要);
        decode 阶段只传 last_token(节省 IPC 带宽)。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """pickle 反序列化:恢复序列状态。"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []          # decode 阶段不需要完整 token_ids
            self.last_token = last_state
