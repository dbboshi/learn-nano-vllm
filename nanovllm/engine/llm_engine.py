# ============================================================================
# LLMEngine —— 引擎编排层主循环
# ============================================================================
# 职责: 管理 TP 子进程、加载 tokenizer、驱动 Scheduler + ModelRunner 的
#       收请求 → 调度 → 前向 → 采样 → 输出 循环。
# 与 vLLM 的 LLMEngine 角色一致,但极简化:无 async、无 RPC server。
# ============================================================================
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """构造引擎。

        流程:
          1. 从 kwargs 中筛出 Config 字段,构造 Config
          2. 设置 Sequence.block_size (类属性,全局共享)
          3. tp>1 时 fork (tp-1) 个子进程,每个跑一个 ModelRunner(rank>0)
          4. 主进程建 ModelRunner(rank=0):含 warmup + KV cache 分配
          5. 加载 tokenizer、记录 eos、建 Scheduler
          6. 注册 atexit 退出清理
        """
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []          # TP 子进程列表(rank>0)
        self.events = []      # 与各子进程通信的 Event(rank0 用它通知子进程读共享内存)
        ctx = mp.get_context("spawn")
        # fork rank 1..tp-1 的子进程,各自执行 ModelRunner.__init__(会进入 loop 阻塞等指令)
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 主进程(rank0)的 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """退出清理:通知 ModelRunner 退出、等待子进程结束。"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """把一条 prompt 加入调度器等待队列。

        输入:
            prompt:         str 会先经 tokenizer.encode 转为 token id 列表;
                            list[int] 直接使用
            sampling_params: 该请求的采样参数
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """推进一步(一次 prefill 或一次 decode)。

        输出: (已完成序列的 [(seq_id, token_ids)], num_tokens)
              num_tokens>0 表示 prefill(本步处理的 token 数);
              num_tokens<0 表示 decode(负的 batch 大小,用于统计吞吐)
        """
        seqs, is_prefill = self.scheduler.schedule()                     # 调度器决定本步跑谁
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)     # GPU 前向+采样
        self.scheduler.postprocess(seqs, token_ids, is_prefill)         # 更新状态/释放块/追加 token
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """所有序列是否都已跑完(waiting 和 running 均空)。"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """批量生成接口(同步阻塞)。

        输入:
            prompts:        list[str] 或 list[list[int]](已编码的 token id)
            sampling_params: 单条 SamplingParams 或逐条 list
            use_tqdm:       是否显示进度条
        输出:
            list[{"text": str, "token_ids": list[int]}],顺序与 prompts 一致

        内部循环:每步调用 step(),prefill 优先于 decode,
        先完成的序列先释放 block,不影响其他序列继续跑(continuous batching)。
        """
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 1) 入队所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 2) 引擎主循环
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # 注意:这里显示的是「最后一步」的瞬时吞吐,不是累计平均
            if num_tokens > 0:                    # prefill 步
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:                                  # decode 步
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:       # 收集已完成的序列
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 3) 按 seq_id 排序并解码为文本
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
