from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """采样参数(每条请求独立一份)。

    属性:
        temperature: 温度系数,必须 > 1e-10。本引擎用 Gumbel-max 采样,
                     不支持 greedy(temperature=0),否则除零。
        max_tokens:  单条请求最多生成的 token 数(不含 prompt)。
        ignore_eos:  True 时即使采样到 EOS 也不停止(bench 基准测试用)。
    """

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # Gumbel-max 采样中 probs / Exp(1) 会除以温度,必须为正
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
