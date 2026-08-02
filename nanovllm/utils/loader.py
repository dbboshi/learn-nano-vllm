# ============================================================================
# loader.py —— 从 safetensors 装载权重
# ============================================================================
# 遍历模型目录下的 *.safetensors,按权重名逐个装载。
#
# 关键:处理 packed(合并)模块
#   模型类(如 Qwen3ForCausalLM)定义 packed_modules_mapping,例如:
#     {"q_proj": ("qkv_proj", "q"), "gate_proj": ("gate_up_proj", 0), ...}
#   当遇到 HF 权重名含 "q_proj" 时:
#     1. 把名字里的 "q_proj" 替换成 "qkv_proj"(合并后的参数名)
#     2. 取出对应的 shard_id="q"
#     3. 调用该参数的 weight_loader(param, loaded_weight, shard_id)
#        weight_loader 会把 HF 的 q_proj 权重装进 qkv_proj 的 q 分片
#
# 非 packed 权重直接调 weight_loader(或 default_weight_loader)复制。
# ============================================================================
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认装载:直接 copy(用于非 TP 切分的普通参数)。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从 path 目录下的 safetensors 文件装载权重到 model。

    输入:
        model: 已构造的模型(权重已按 TP 切分好形状,等待填值)
        path:  模型目录(含 *.safetensors)
    流程:
        遍历每个 safetensors 文件的每个权重名:
          - 若匹配 packed_modules_mapping:替换名字 + 取 shard_id + 调 weight_loader(param, w, shard_id)
          - 否则:直接调 weight_loader(param, w) 或 default_weight_loader
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否是 packed 模块的子权重
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)     # q_proj → qkv_proj
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 非 packed 权重:直接复制
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
