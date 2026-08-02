# Learning notes of Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

Nano-vLLM 提供两种安装方式。

### 方式一:通过 `pyproject.toml` 安装(原版,需 Ampere / sm>=80 GPU)

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

> 该方式按 `pyproject.toml` 安装全部依赖,包含 `flash-attn` 与 `triton`,
> 二者仅支持 sm>=80 的 GPU;在老 GPU(如 RTX 2060 / Turing sm75)上会安装失败。

### 方式二:通过 `requirements.txt` 安装(推荐老 GPU / Windows 用户)

自当前版本(commit `b1b57e2`,本地老 GPU 适配)起,注意力层([`nanovllm/layers/attention.py`](nanovllm/layers/attention.py))新增 SDPA
(PyTorch `scaled_dot_product_attention`)回退后端:运行时若检测到 GPU 算力 < 8.0 或缺少
`flash-attn` / `triton`,会自动走 SDPA 路径,功能与 FA2 路径一致(速度略慢)。
因此老 GPU 用户可跳过 `flash-attn` / `triton`,仅安装必需依赖:

```bash
# 1. 克隆仓库
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm

# 2. (可选)新建 conda 环境(Python 要求 >=3.10,<3.13)
conda create -n nanovllm python=3.10 -y
conda activate nanovllm

# 3. 安装必需依赖(不含 flash-attn / triton,适配 sm<80 老 GPU)
pip install -r requirements.txt

# 4. 仅当你的 GPU 为 Ampere(sm>=80)且需要 FA2 加速时,再装可选依赖:
#    pip install "triton>=3.0.0" flash-attn
#    (Windows 上官方 triton 不可用,改用: pip install triton-windows)
```

**两种方式依赖对照**(参考 `pyproject.toml` 与 conda `nanovllm` 环境实测版本):

| 依赖包          | pyproject.toml | requirements.txt(必需) | 可选(FA2 加速) | 本机 nanovllm 实测版本，可运行 |
|-----------------|----------------|------------------------|-----------------|------------------------|
| torch           | >=2.4.0        | >=2.4.0                | —               | 2.6.0+cu126            |
| transformers    | >=4.51.0       | >=4.51.0               | —               | 5.14.1                 |
| xxhash          | *              | *                      | —               | 3.8.1                  |
| triton          | >=3.0.0        | —                      | >=3.0.0         | triton-windows 3.2.0   |
| flash-attn      | *              | —                      | *               | 2.7.4                  |

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)