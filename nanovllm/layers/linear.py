# ============================================================================
# Linear 层 —— Tensor Parallel (TP) 并行线性层
# ============================================================================
# TP 的两种切分方式:
#
#   列并行 ColumnParallelLinear: 权重按输出维切分,各卡持不同输出列
#     - forward: 各卡独立算 F.linear(x, W_shard),无需通信
#     - 用于: QKV / gate_up_proj
#
#   行并行 RowParallelLinear: 权重按输入维切分,各卡持不同输入行
#     - forward: 各卡算部分和,末尾 all_reduce 求和
#     - 用于: o_proj / down_proj
#
# 组合使用: ColumnParallel → (无通信) → RowParallel → all_reduce
#   这样一次 TP 通信可覆盖两个线性层(Attention 的 QKV→O、MLP 的 gate_up→down)。
#
# 权重装载:
#   - weight_loader 方法挂在 Parameter 上,loader.py 调用时按 tp_rank 切片
#   - QKVParallelLinear / MergedColumnParallelLinear 把多个 HF 权重(q/k/v 或 gate/up)
#     合并进一个大矩阵,通过 packed_modules_mapping + shard_id 装载到对应分片
# ============================================================================
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除断言,返回 numerator // denominator。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """基类:创建权重 [output_size, input_size](vLLM 风格,权重转置存储)。

        tp_dim: None=复制, 0=列并行(切输出维), 1=行并行(切输入维)
        """
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader      # 挂载装载函数
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """复制型线性层(各卡权重相同,无 TP 切分)。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行:权重按输出维(output_size)切分,各卡持 output_size/tp_size 列。

    forward 无需通信(各卡独立算自己那部分输出)。
    weight_loader 按 tp_rank 从完整权重中 narrow 出对应分片。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)   # tp_dim=0(切输出维)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重 [output_size, input_size] 中取本 rank 的分片 [output_size/tp, input_size]。"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [..., input_size]; 输出: [..., output_size/tp_size]
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并列并行:把多个权重(如 gate_proj + up_proj)合并成一个大矩阵。

    output_sizes: 各子权重的大小列表(如 [intermediate, intermediate])
    装载时通过 loaded_shard_id 指定装进哪个子分片。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """把某个子权重(如 gate_proj)装进合并矩阵的对应分片。

        loaded_shard_id: 0=gate, 1=up(对应 output_sizes 索引)
        """
        param_data = param.data
        # 该子权重在合并矩阵中的偏移(已按 tp 切分)
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中取本 rank 的分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV 合并列并行:把 q_proj/k_proj/v_proj 三个权重合并成一个大矩阵。

    输出布局: [q_shard | k_shard | v_shard]
    其中 q_shard = num_heads * head_size / tp
         k_shard = v_shard = num_kv_heads * head_size / tp (GQA 时 kv < q)
    装载时通过 loaded_shard_id ("q"/"k"/"v") 指定装进哪个子分片。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)            # 本卡的 q heads
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)      # 本卡的 kv heads
        # 合并输出大小 = (q_heads + 2*kv_heads) * head_size
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """把 q_proj/k_proj/v_proj 装进合并矩阵的对应分片。

        loaded_shard_id: "q" / "k" / "v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:    # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行:权重按输入维(input_size)切分,各卡持 input_size/tp_size 行。

    forward: 各卡算部分和 [batch, output_size],末尾 all_reduce 求和。
    bias 只在 rank0 加(避免 all_reduce 后重复加)。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)   # tp_dim=1(切输入维)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重 [output_size, input_size] 中取本 rank 的输入分片。"""
        param_data = param.data
        if param_data.ndim == 1:          # bias 是 1D,直接复制
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [..., input_size/tp_size]; 输出: [..., output_size]
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)            # 各卡部分和求和
        return y
