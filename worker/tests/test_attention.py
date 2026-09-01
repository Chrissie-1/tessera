"""Phase 4: paged attention.

Two layers of checking. The torch implementation is held to plain dense
attention over the same logical sequence, which is what pins the block-table
indirection: a shuffled block table must produce exactly the answer a
contiguous sequence would. The Triton kernel is then held to the torch
implementation, so the kernel is only ever asked to reproduce something already
known to be right.

The GPU tests are marked `gpu` and skip without CUDA, so the indexing logic
still gets covered on a CPU-only machine.
"""

from __future__ import annotations

import math

import pytest
import torch

from tessera_worker.attention import (
    HAS_TRITON,
    paged_attention,
    paged_attention_torch,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton")


def dense_attention(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Textbook attention over contiguous keys and values.

    Args:
        query: (num_heads, head_dim)
        keys: (num_heads, seq_len, head_dim)
        values: (num_heads, seq_len, head_dim)
    """
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("hd,htd->ht", query.float(), keys.float()) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,htd->hd", weights, values.float())


def scatter_into_blocks(
    keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
    block_table: list[int],
    num_blocks: int,
):
    """Lay a logical sequence out into a block pool at the given physical ids."""
    num_heads, seq_len, head_dim = keys.shape
    key_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)
    value_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim)

    for position in range(seq_len):
        physical = block_table[position // block_size]
        offset = position % block_size
        key_cache[physical, :, offset, :] = keys[:, position, :]
        value_cache[physical, :, offset, :] = values[:, position, :]

    return key_cache, value_cache


def make_case(num_heads=4, seq_len=10, head_dim=8, block_size=4, shuffle=False):
    torch.manual_seed(seq_len * 1000 + head_dim)
    query = torch.randn(num_heads, head_dim)
    keys = torch.randn(num_heads, seq_len, head_dim)
    values = torch.randn(num_heads, seq_len, head_dim)

    used = (seq_len + block_size - 1) // block_size
    num_blocks = used + 6
    # When shuffled, physical ids are deliberately out of order and
    # non-adjacent -- what a real pool looks like after churn.
    table = list(range(num_blocks))[::-1][:used] if shuffle else list(range(used))

    key_cache, value_cache = scatter_into_blocks(
        keys, values, block_size, table, num_blocks
    )
    return query, keys, values, key_cache, value_cache, torch.tensor(table)


# -- torch implementation ---------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 3, 4, 5, 16, 33])
def test_matches_dense_attention(seq_len):
    query, keys, values, key_cache, value_cache, table = make_case(seq_len=seq_len)

    actual = paged_attention_torch(query, key_cache, value_cache, table, seq_len)
    expected = dense_attention(query, keys, values)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_shuffled_block_table_matches_dense():
    """The indirection is the whole point: physical order must not matter."""
    seq_len = 14
    query, keys, values, key_cache, value_cache, table = make_case(
        seq_len=seq_len, shuffle=True
    )

    actual = paged_attention_torch(query, key_cache, value_cache, table, seq_len)
    expected = dense_attention(query, keys, values)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_ignores_the_unwritten_tail_of_the_last_block():
    """A partially filled block is allocated but not yet real."""
    seq_len = 6
    block_size = 4
    query, keys, values, key_cache, value_cache, table = make_case(
        seq_len=seq_len, block_size=block_size
    )

    # Poison the slots past seq_len; a correct implementation never reads them.
    key_cache[table[-1], :, seq_len % block_size :, :] = 1e4
    value_cache[table[-1], :, seq_len % block_size :, :] = 1e4

    actual = paged_attention_torch(query, key_cache, value_cache, table, seq_len)
    expected = dense_attention(query, keys, values)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_uniform_keys_give_uniform_attention():
    num_heads, head_dim, seq_len, block_size = 2, 4, 8, 4
    query = torch.ones(num_heads, head_dim)
    keys = torch.ones(num_heads, seq_len, head_dim)
    values = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)
    values = values.expand(num_heads, seq_len, head_dim).contiguous()

    key_cache, value_cache = scatter_into_blocks(
        keys, values, block_size, [0, 1], num_blocks=2
    )
    actual = paged_attention_torch(
        query, key_cache, value_cache, torch.tensor([0, 1]), seq_len
    )

    # Equal scores mean the output is the mean of the values.
    assert torch.allclose(actual, torch.full_like(actual, values.mean()), atol=1e-5)


def test_rejects_empty_sequence():
    query, _, _, key_cache, value_cache, table = make_case()

    with pytest.raises(ValueError):
        paged_attention_torch(query, key_cache, value_cache, table, 0)


def test_dispatch_uses_torch_on_cpu():
    seq_len = 9
    query, keys, values, key_cache, value_cache, table = make_case(seq_len=seq_len)

    actual = paged_attention(query, key_cache, value_cache, table, seq_len)

    assert torch.allclose(actual, dense_attention(query, keys, values), atol=1e-5)


# -- Triton kernel ----------------------------------------------------------


@requires_triton
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [1, 7, 16, 40, 129])
def test_kernel_matches_torch(seq_len):
    from tessera_worker.attention import paged_attention_triton

    query, _, _, key_cache, value_cache, table = make_case(
        num_heads=4, seq_len=seq_len, head_dim=32, block_size=16
    )
    expected = paged_attention_torch(query, key_cache, value_cache, table, seq_len)

    actual = paged_attention_triton(
        query.cuda(), key_cache.cuda(), value_cache.cuda(), table.cuda(), seq_len
    )

    assert torch.allclose(actual.cpu(), expected, atol=1e-4)


@requires_triton
@requires_cuda
@pytest.mark.gpu
def test_kernel_handles_a_shuffled_block_table():
    from tessera_worker.attention import paged_attention_triton

    seq_len = 50
    query, _, _, key_cache, value_cache, table = make_case(
        num_heads=4, seq_len=seq_len, head_dim=32, block_size=16, shuffle=True
    )
    expected = paged_attention_torch(query, key_cache, value_cache, table, seq_len)

    actual = paged_attention_triton(
        query.cuda(), key_cache.cuda(), value_cache.cuda(), table.cuda(), seq_len
    )

    assert torch.allclose(actual.cpu(), expected, atol=1e-4)


@requires_triton
@requires_cuda
@pytest.mark.gpu
def test_kernel_ignores_the_unwritten_tail():
    from tessera_worker.attention import paged_attention_triton

    seq_len, block_size = 20, 16
    query, _, _, key_cache, value_cache, table = make_case(
        num_heads=4, seq_len=seq_len, head_dim=32, block_size=block_size
    )
    expected = paged_attention_torch(query, key_cache, value_cache, table, seq_len)

    key_cache[table[-1], :, seq_len % block_size :, :] = 1e4
    value_cache[table[-1], :, seq_len % block_size :, :] = 1e4

    actual = paged_attention_triton(
        query.cuda(), key_cache.cuda(), value_cache.cuda(), table.cuda(), seq_len
    )

    assert torch.allclose(actual.cpu(), expected, atol=1e-4)


@requires_triton
@requires_cuda
@pytest.mark.gpu
def test_dispatch_uses_the_kernel_on_cuda():
    seq_len = 24
    query, _, _, key_cache, value_cache, table = make_case(
        num_heads=4, seq_len=seq_len, head_dim=32, block_size=16
    )
    expected = paged_attention_torch(query, key_cache, value_cache, table, seq_len)

    actual = paged_attention(
        query.cuda(), key_cache.cuda(), value_cache.cuda(), table.cuda(), seq_len
    )

    assert torch.allclose(actual.cpu(), expected, atol=1e-4)
