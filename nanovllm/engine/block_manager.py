# ============================================================================
# BlockManager —— 分页 KV Cache 管理 + 前缀缓存
# ============================================================================
# PagedAttention 的核心:KV cache 不按序列连续分配,而是切成固定大小的 block,
# 按需从池中取物理块,用 block_table 做逻辑→物理映射。
#
# 前缀缓存:对每个完整 block 计算链式哈希
#   hash(block_i) = xxh64( hash(block_{i-1}) || token_ids_i )
# 存进 hash_to_block_id。新序列若前缀哈希命中且 token 完全匹配,就复用已有物理块
# (ref_count 引用计数),免去重算。
#
# k_cache / v_cache 形状: [num_blocks, block_size, num_kv_heads, head_dim]
# slot_mapping: 每个新 token 写入的扁平槽位 = block_id * block_size + offset
# ============================================================================
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """单个物理块的元数据(K/V tensor 不在此,在 ModelRunner 的 kv_cache 中)。"""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0          # 引用计数(>1 表示被多个序列共享/前缀缓存命中)
        self.hash = -1              # 该 block 内容的链式哈希值(-1=未登记)
        self.token_ids = []         # 该 block 对应的 token_ids(用于哈希命中后校验)

    def update(self, hash: int, token_ids: list[int]):
        """prefill 后登记哈希与 token_ids。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """块被重新分配时重置(保留 ref_count=1)。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()    # 哈希值→物理块 id(前缀缓存查表)
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算一个 block 的链式哈希。

        输入:
            token_ids: 该 block 的 token id 列表
            prefix:    前一个 block 的哈希值(-1 表示这是第一个 block)
        输出: xxh64 哈希整数
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))     # 把前一个 block 的哈希拼进来(链式)
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲池取一个物理块,返回其 block_id。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 如果该块之前登记过哈希,先从哈希表中移除(内容即将改变)
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """释放一个物理块回空闲池。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """检查新序列能否分配,并返回前缀缓存命中的块数。

        输出: num_cached_blocks(命中的前缀块数); -1 表示显存不足
        注意:最后一个不完整 block 不参与缓存(只有完整 block 才哈希登记)。
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):             # 最后一个 block 跳过(可能不完整)
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)          # 链式哈希
            block_id = self.hash_to_block_id.get(h, -1)
            # 哈希未命中或 token 不符(哈希碰撞)→ 停止
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 已被别人用着的块→共享(ref_count++),无需新增块
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1                                    # 剩余空闲块不够
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为序列分配物理块:命中的前缀块复用,其余从空闲池取。

        输入:
            seq:                待分配的序列(必须无 block_table)
            num_cached_blocks:  can_allocate 返回的命中块数
        副作用: 填充 seq.block_table,设置 seq.num_cached_tokens
        """
        assert not seq.block_table
        h = -1
        # 复用命中的前缀块
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1                     # 共享:引用计数+1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 分配剩余的新块
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有物理块(ref_count 减到 0 才真正回收)。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """decode 时检查能否追加 1 个 token(可能需要 1 个新块)。

        len(seq) % block_size == 1 时表示新 token 落在新块的第一个位置,需要 1 个空闲块。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 时按需追加新块。

        当 len(seq) % block_size == 1(新 token 是某新块的第一格)时分配一个新块。
        """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """prefill 后调用:把本步新算的完整 block 计算哈希并登记,供后续序列复用。

        只登记 [start, end) 范围内完整的 block(不含最后一个可能未满的块)。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return                                      # 没有新的完整 block
        # 取前一个 block 的哈希作为链式起点
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
