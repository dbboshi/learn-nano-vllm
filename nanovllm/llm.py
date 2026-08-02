# LLM 入口类:仅继承 LLMEngine,不添加任何额外逻辑
# 这样设计的目的是让用户侧 API (nanovllm.LLM) 与引擎实现解耦,
# 与 vLLM 的 from vllm import LLM 风格保持一致
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """nano-vllm 对外暴露的推理引擎入口,全部逻辑在 LLMEngine 中实现。"""
    pass
