"""Phase 2: block-paged KV cache.

The reference engine re-runs a full forward pass per token because it keeps no
cache at all. This module keeps one, but refuses to store it as a contiguous
per-sequence tensor: that layout forces every sequence to reserve room for the
longest completion it *might* produce, and a server sized that way runs out of
memory long before it runs out of compute.

Instead the cache is a pool of fixed-size blocks. A sequence holds a block
table -- an ordered list of physical block ids -- and grows it one block at a
time as it decodes. Memory is committed to a sequence only once it is actually
used, and freeing a sequence returns whole blocks to the pool with no
compaction and no fragmentation.

Attention itself is still the model's own implementation: each step gathers the
sequence's blocks into the contiguous tensors transformers expects. Phase 4's
Triton kernel is what removes that gather by attending over the block table
directly. The gather is the cost of staying correct in the meantime, and the
equivalence tests exist to keep it honest.
"""

from __future__ import annotations

import logging

import torch
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_BLOCKS = 512


class OutOfBlocksError(RuntimeError):
    """The block pool is exhausted.

    Raised rather than silently evicting: the scheduler above decides whether
    to queue the request or shed it, and it cannot make that call if the cache
    quietly drops another sequence's tokens.
    """


class BlockAllocator:
    """A free list over a fixed pool of block ids.

    Blocks are interchangeable, so allocation order does not matter and a
    LIFO free list keeps recently-touched blocks hot.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self._num_blocks = num_blocks
        self._free: list[int] = list(reversed(range(num_blocks)))

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self._num_blocks - len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError("block pool exhausted")
        return self._free.pop()

    def free(self, block_ids: list[int]) -> None:
        # Guard against a double free returning the same id to the pool twice,
        # which would later hand one block to two different sequences.
        for block_id in block_ids:
            if block_id in self._free:
                raise ValueError(f"block {block_id} freed twice")
        self._free.extend(block_ids)


class PagedKVCache:
    """Per-layer key/value storage addressed through per-sequence block tables.

    Storage is allocated lazily on first write, when the model's real head
    count and head dimension are known from the tensors themselves. Reading
    them off the config instead would get grouped-query models wrong, since
    their KV head count differs from their attention head count.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_layers = num_layers
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.allocator = BlockAllocator(num_blocks)

        self._keys: list[torch.Tensor] | None = None
        self._values: list[torch.Tensor] | None = None
        self._tables: dict[int, list[int]] = {}
        self._lengths: dict[int, int] = {}

    # -- sequence lifecycle -------------------------------------------------

    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already present")
        self._tables[seq_id] = []
        self._lengths[seq_id] = 0

    def free_sequence(self, seq_id: int) -> None:
        """Return every block held by `seq_id`. Idempotent."""
        table = self._tables.pop(seq_id, None)
        if table is None:
            return
        self._lengths.pop(seq_id, None)
        self.allocator.free(table)

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._tables

    def length(self, seq_id: int) -> int:
        return self._lengths[seq_id]

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._tables[seq_id])

    def blocks_needed(self, seq_id: int, new_tokens: int) -> int:
        """Blocks that appending `new_tokens` would have to allocate."""
        current = self._lengths[seq_id]
        have = len(self._tables[seq_id]) * self.block_size
        deficit = max(0, current + new_tokens - have)
        return (deficit + self.block_size - 1) // self.block_size

    # -- storage ------------------------------------------------------------

    def _ensure_storage(self, keys: torch.Tensor) -> None:
        if self._keys is not None:
            return
        _, num_heads, _, head_dim = keys.shape
        shape = (self.allocator.num_blocks, num_heads, self.block_size, head_dim)
        self._keys = [
            torch.zeros(shape, device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        self._values = [
            torch.zeros(shape, device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        logger.info(
            "paged cache: %d blocks x %d tokens, %d heads, head_dim %d",
            self.allocator.num_blocks,
            self.block_size,
            num_heads,
            head_dim,
        )

    def append(
        self, seq_id: int, layer_kv: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Append one step's key/value tensors for every layer.

        Args:
            seq_id: sequence to extend.
            layer_kv: per-layer (keys, values), each (1, heads, new_tokens, dim),
                covering only the positions not already stored.
        """
        if not layer_kv:
            return
        self._ensure_storage(layer_kv[0][0])

        new_tokens = layer_kv[0][0].shape[2]
        start = self._lengths[seq_id]
        table = self._tables[seq_id]

        # Commit blocks only for the tokens actually being written.
        while len(table) * self.block_size < start + new_tokens:
            table.append(self.allocator.allocate())

        # Copy span-wise: one slice per block touched, not one per token.
        written = 0
        while written < new_tokens:
            position = start + written
            block_id = table[position // self.block_size]
            offset = position % self.block_size
            span = min(self.block_size - offset, new_tokens - written)

            for layer, (keys, values) in enumerate(layer_kv):
                self._keys[layer][block_id, :, offset : offset + span, :] = keys[
                    0, :, written : written + span, :
                ]
                self._values[layer][block_id, :, offset : offset + span, :] = values[
                    0, :, written : written + span, :
                ]
            written += span

        self._lengths[seq_id] = start + new_tokens

    def truncate(self, seq_id: int, new_length: int) -> None:
        """Drop everything past `new_length`, releasing blocks it freed up.

        Speculative decoding writes tokens it may not keep: the target model
        verifies a draft in one pass, and every rejected token's keys and
        values have to disappear before the next step, or the sequence carries
        state for tokens it never emitted.
        """
        if new_length < 0 or new_length > self._lengths[seq_id]:
            raise ValueError(f"cannot truncate sequence {seq_id} to {new_length}")
        table = self._tables[seq_id]
        keep = (new_length + self.block_size - 1) // self.block_size
        if keep < len(table):
            self.allocator.free(table[keep:])
            del table[keep:]
        self._lengths[seq_id] = new_length

    def gather(self, seq_id: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Rebuild contiguous per-layer (keys, values) for `seq_id`.

        Returns tensors shaped (1, heads, length, dim) -- what the model's
        attention expects, reassembled from possibly non-adjacent blocks.
        """
        length = self._lengths[seq_id]
        table = self._tables[seq_id]
        if length == 0 or self._keys is None:
            return []

        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.num_layers):
            keys = torch.cat([self._keys[layer][b] for b in table], dim=1)
            values = torch.cat([self._values[layer][b] for b in table], dim=1)
            out.append(
                (
                    keys[:, :length, :].unsqueeze(0),
                    values[:, :length, :].unsqueeze(0),
                )
            )
        return out

    def to_cache(self, seq_id: int) -> DynamicCache:
        """Gather `seq_id` into the cache object transformers consumes."""
        return DynamicCache(self.gather(seq_id))

    @property
    def utilisation(self) -> float:
        return self.allocator.used_blocks / self.allocator.num_blocks
