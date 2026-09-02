"""Decoding a model that is not GPT-2, through every engine.

`test_config.py` proves the architecture helpers read a Llama or Mistral
config correctly. That is a claim about attribute names. This file makes the
claim the README could not previously make: that a non-GPT-2 model is actually
decoded, end to end, by the paged engine, the continuous batcher and the
speculative engine, and that each of them emits the reference engine's tokens.

Three families are covered, chosen for what they break rather than for
variety:

* Llama -- rotary positions instead of GPT-2's learned position embeddings,
  which is what makes the batcher's explicit `position_ids` load-bearing.
* Mistral -- the same, plus genuine grouped-query attention (four query heads
  onto two KV heads), which is the case the paged kernel used to refuse.
* GPT-NeoX -- rotary over only part of the head dimension, and a parallel
  attention/MLP residual, so "RoPE model" is not a single shape.

The models are the hub's own tiny random test checkpoints: one to two million
parameters, a few megabytes, no authentication. They are kept in the default
test run deliberately. A second architecture that CI does not execute is
exactly the situation this file exists to end.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tessera_worker import attention_hook
from tessera_worker.batching import ContinuousBatcher, Request
from tessera_worker.config import WorkerConfig, max_positions, num_layers
from tessera_worker.model import ReferenceEngine
from tessera_worker.paged import PagedKVCache
from tessera_worker.paged_engine import PagedEngine
from tessera_worker.sampling import SamplingParams
from tessera_worker.serving import BatchedEngine
from tessera_worker.speculative import SpeculativeEngine

# Tiny randomly-initialised checkpoints published for exactly this purpose.
MODELS = {
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
}

# Ragged on purpose: the batcher left-pads the shorter caches to the longest,
# and padding that is not masked is the failure this set is shaped to catch.
PROMPTS = {
    "capital": "The capital of France is",
    "short": "Hello",
    "long": "Numbers: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
}


@pytest.fixture(scope="session", params=sorted(MODELS), ids=sorted(MODELS))
def arch_config(request) -> WorkerConfig:
    """A worker config per architecture, sized small enough to churn blocks."""
    return WorkerConfig(
        model_name=MODELS[request.param],
        device="cpu",
        dtype=torch.float32,
        grpc_port=0,
        http_port=0,
        max_tokens_cap=32,
        num_blocks=128,
        block_size=4,
    )


# Session-scoped because loading dominates the runtime of every test here,
# and none of these engines carries state between requests.


@pytest.fixture(scope="session")
def arch_reference(arch_config) -> ReferenceEngine:
    return ReferenceEngine(arch_config)


@pytest.fixture(scope="session")
def arch_paged(arch_config) -> PagedEngine:
    return PagedEngine(arch_config)


@pytest.fixture(scope="session")
def arch_speculative(arch_config) -> SpeculativeEngine:
    return SpeculativeEngine(arch_config, lookahead=3)


# -- the architecture actually loads and is read correctly ------------------


def test_the_model_loads_and_reports_its_shape(arch_reference):
    config = arch_reference.model.config

    assert num_layers(config) >= 1
    assert max_positions(config) > 1
    assert arch_reference.max_position_embeddings == max_positions(config)


def test_a_grouped_query_model_is_among_the_architectures_covered():
    """The point of the Mistral case, asserted rather than assumed.

    If the hub checkpoint ever changes to plain multi-head attention this
    file would keep passing while silently no longer testing grouping, so the
    property is pinned here where the failure names itself.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODELS["mistral"])

    assert config.num_key_value_heads < config.num_attention_heads
    assert config.num_attention_heads % config.num_key_value_heads == 0


def test_the_cache_is_sized_from_the_key_tensors_not_the_head_count(arch_paged):
    """Grouped-query models store KV heads, of which there are fewer.

    `_ensure_storage` reads the shape off the first key tensor written. This
    checks it landed on the KV head count rather than the attention head
    count -- sizing off the config would over-allocate on every GQA model and
    then index it wrong.
    """
    arch_paged.generate("The capital of France is", max_tokens=2)
    config = arch_paged.model.config
    expected = (
        getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    )

    assert arch_paged.cache.layer_keys(0).shape[1] == expected


# -- paged decode -----------------------------------------------------------


@pytest.mark.parametrize("prompt", sorted(PROMPTS), ids=sorted(PROMPTS))
def test_paged_matches_reference(arch_reference, arch_paged, prompt):
    reference = arch_reference.generate(PROMPTS[prompt], max_tokens=10)
    actual = arch_paged.generate(PROMPTS[prompt], max_tokens=10)

    assert actual.token_ids == reference.token_ids
    assert actual.finish_reason == reference.finish_reason


def test_paged_seeded_sampling_matches_reference(arch_reference, arch_paged):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=1234)

    reference = arch_reference.generate("Once upon a time", max_tokens=8, params=params)
    actual = arch_paged.generate("Once upon a time", max_tokens=8, params=params)

    assert actual.token_ids == reference.token_ids


def test_decode_reaches_the_paged_kernel_and_never_gathers(
    arch_reference, arch_paged, monkeypatch
):
    """Equivalence alone would also be satisfied by falling back everywhere.

    Grouped-query attention used to be deferred to the model's own attention,
    and `decode_step` hands the model no past when the hook is installed, so
    that deferral silently attended over a one-token context. Counting the
    dispatcher is what tells the two apart.
    """
    assert arch_paged.paged_attention_enabled

    calls = []
    wrapped = attention_hook.paged_attention
    monkeypatch.setattr(
        attention_hook,
        "paged_attention",
        lambda *args: (calls.append(args[4]), wrapped(*args))[1],
    )
    gathers = []
    original = PagedKVCache.gather
    monkeypatch.setattr(
        PagedKVCache,
        "gather",
        lambda self, seq_id: (gathers.append(seq_id), original(self, seq_id))[1],
    )

    prompt = PROMPTS["capital"]
    actual = arch_paged.generate(prompt, max_tokens=6)

    layers = num_layers(arch_paged.model.config)
    steps = actual.completion_tokens - (1 if actual.finish_reason == "length" else 0)
    assert steps > 0
    assert len(calls) == layers * steps
    assert gathers == []
    assert actual.token_ids == arch_reference.generate(prompt, max_tokens=6).token_ids


def test_gathering_and_the_kernel_agree(arch_reference, arch_config, monkeypatch):
    """The fallback path has to reach the same answer, on this model too."""
    import tessera_worker.paged_engine as paged_engine

    monkeypatch.setattr(paged_engine, "enable_paged_attention", lambda _model: False)
    fallback = paged_engine.PagedEngine(arch_config)

    assert not fallback.paged_attention_enabled

    prompt = PROMPTS["capital"]
    assert (
        fallback.generate(prompt, max_tokens=10).token_ids
        == arch_reference.generate(prompt, max_tokens=10).token_ids
    )


# -- continuous batching ----------------------------------------------------


def test_ragged_batch_matches_unbatched(arch_reference, arch_paged):
    """Prompts of three different lengths through one padded forward pass.

    Rotary models take their positions from `position_ids`, which the batcher
    supplies, but the left padding still has to be masked out of the keys. It
    was not: transformers builds no mask for an unrecognised attention
    implementation, and installing the paged hook made the model's
    implementation unrecognised. Only the longest row -- the one with no
    padding -- came out right.
    """
    expected = {
        name: arch_reference.generate(prompt, max_tokens=8).token_ids
        for name, prompt in PROMPTS.items()
    }

    batcher = ContinuousBatcher(arch_paged, max_batch_size=4)
    for name, prompt in PROMPTS.items():
        batcher.submit(Request(request_id=name, prompt=prompt, max_tokens=8))
    actual = batcher.run_to_completion()

    assert actual == expected


def test_batching_leaks_no_blocks(arch_paged):
    before = arch_paged.cache.allocator.free_blocks

    batcher = ContinuousBatcher(arch_paged, max_batch_size=2)
    for name, prompt in PROMPTS.items():
        batcher.submit(Request(request_id=name, prompt=prompt, max_tokens=6))
    batcher.run_to_completion()

    assert arch_paged.cache.allocator.free_blocks == before


def test_batched_seeded_sampling_matches_unbatched(arch_reference, arch_paged):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=7)
    prompt = PROMPTS["capital"]
    expected = arch_reference.generate(prompt, max_tokens=6, params=params).token_ids

    batcher = ContinuousBatcher(arch_paged, max_batch_size=2)
    batcher.submit(Request(request_id="a", prompt=prompt, max_tokens=6, params=params))
    actual = batcher.run_to_completion()

    assert actual["a"] == expected


def test_concurrent_requests_through_the_serving_engine_match_reference(
    arch_reference, arch_config
):
    """What the gRPC layer actually calls: threads sharing one scheduler.

    `ContinuousBatcher` is driven directly by the tests above; this is the
    same scheduler behind its background thread, reached the way production
    reaches it. The batch composition is then decided by arrival order rather
    than by the test, which is the point.
    """
    expected = {
        name: arch_reference.generate(prompt, max_tokens=8).token_ids
        for name, prompt in PROMPTS.items()
    }

    with (
        BatchedEngine(arch_config, max_batch_size=4) as batched,
        ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool,
    ):
        futures = {
            name: pool.submit(batched.generate, prompt, 8)
            for name, prompt in PROMPTS.items()
        }
        actual = {name: f.result().token_ids for name, f in futures.items()}

    assert actual == expected


# -- speculative decoding ---------------------------------------------------


@pytest.mark.parametrize("prompt", sorted(PROMPTS), ids=sorted(PROMPTS))
def test_speculative_matches_reference(arch_reference, arch_speculative, prompt):
    """Verification attends to many query positions over a longer cache.

    That shape is deferred to the model's own attention, so it is the other
    half of the mask regression: without a mask the causal cut lands at the
    top left of the score matrix instead of the bottom right, and every token
    after the first is wrong.
    """
    reference = arch_reference.generate(PROMPTS[prompt], max_tokens=10)
    actual = arch_speculative.generate(PROMPTS[prompt], max_tokens=10)

    assert actual.token_ids == reference.token_ids
    assert actual.finish_reason == reference.finish_reason


def test_speculative_accepts_its_own_perfect_drafter(arch_speculative):
    """The drafter is the engine itself, so proposals should nearly all hold.

    A collapsed acceptance rate would mean verification and drafting disagree
    about the model -- the symptom the mask bug produced before it was fixed.
    """
    arch_speculative.proposed_tokens = 0
    arch_speculative.accepted_tokens = 0
    arch_speculative.generate(PROMPTS["capital"], max_tokens=10)

    assert arch_speculative.proposed_tokens > 0
    assert arch_speculative.acceptance_rate > 0.9
