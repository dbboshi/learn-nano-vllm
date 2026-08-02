# ============================================================================
# Sampler —— Gumbel-max 采样
# ============================================================================
# 用 Gumbel-max 技巧等价于按概率采样:
#   argmax( probs / Exp(1) ) 等价于按 probs 概率分布采样一个 token
# 优点:全程 tensor 操作,无需 Python 循环,可 batch 采样。
#
# 注意:temperature 必须 > 1e-10(SamplingParams 断言),否则除零。
#       因此本引擎不支持 greedy(temperature=0)。
# @torch.compile 在 sm75 上仅产生警告,不影响正确性。
# ============================================================================
import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """采样。

        输入:
            logits:     [B, vocab]  各 token 的原始 logit
            temperatures: [B]        每条 seq 的温度
        输出:
            sample_tokens: [B]  采样出的 token_id
        """
        # 1. 温度缩放:logits / T(高温→更随机,低温→更确定)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 2. softmax 得概率
        probs = torch.softmax(logits, dim=-1)
        # 3. Gumbel-max:probs / Exp(1) 后 argmax,等价于按 probs 采样
        #    clamp_min_(1e-10) 防止 Exp(1) 采样到 0 导致除零
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
