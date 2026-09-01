"""gRPC servicer behaviour.

The servicer is the production entry point, so these tests pin the request
validation, the error codes the gateway routes on, and the response mapping.
The servicer is exercised directly against a fake context rather than over a
real socket: what matters here is the translation layer, not gRPC transport.
"""

from __future__ import annotations

from dataclasses import dataclass

import grpc
import pytest

from tessera_worker.config import WorkerConfig
from tessera_worker.generated import inference_pb2
from tessera_worker.model import GenerationResult
from tessera_worker.server import InferenceServicer


class AbortedError(Exception):
    """Raised by FakeContext.abort, mirroring real gRPC abort semantics."""

    def __init__(self, code, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


class FakeContext:
    def abort(self, code, details):
        raise AbortedError(code, details)


@dataclass
class FakeHandle:
    """The subset of EngineHandle the servicer actually uses."""

    config: WorkerConfig
    engine: object
    ready: bool = True


class ExplodingEngine:
    def generate(self, **_):
        raise RuntimeError("CUDA out of memory")


@pytest.fixture
def servicer(config, engine):
    return InferenceServicer(FakeHandle(config=config, engine=engine))


def request(**overrides):
    payload = {"prompt": "The capital of France is", "max_tokens": 4}
    payload.update(overrides)
    return inference_pb2.GenerateRequest(**payload)


def abort_code(servicer, req):
    with pytest.raises(AbortedError) as excinfo:
        servicer.Generate(req, FakeContext())
    return excinfo.value.code


def test_generate_returns_full_response(servicer):
    response = servicer.Generate(request(request_id="req-1"), FakeContext())

    assert response.finished is True
    assert response.request_id == "req-1"
    assert response.prompt_tokens > 0
    assert response.completion_tokens <= 4
    assert response.finish_reason in {"length", "stop"}
    assert isinstance(response.text, str)


def test_request_id_is_generated_when_absent(servicer):
    response = servicer.Generate(request(), FakeContext())

    assert response.request_id


def test_streaming_is_rejected_until_phase_2(servicer):
    assert abort_code(servicer, request(stream=True)) == (
        grpc.StatusCode.INVALID_ARGUMENT
    )


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_non_positive_max_tokens_is_rejected(servicer, max_tokens):
    assert abort_code(servicer, request(max_tokens=max_tokens)) == (
        grpc.StatusCode.INVALID_ARGUMENT
    )


def test_max_tokens_above_cap_is_rejected(servicer, config):
    assert abort_code(servicer, request(max_tokens=config.max_tokens_cap + 1)) == (
        grpc.StatusCode.INVALID_ARGUMENT
    )


def test_max_tokens_at_cap_is_allowed(servicer, config):
    response = servicer.Generate(
        request(max_tokens=config.max_tokens_cap), FakeContext()
    )

    assert response.completion_tokens <= config.max_tokens_cap


def test_greedy_generation_is_deterministic(servicer):
    first = servicer.Generate(request(temperature=0.0), FakeContext())
    second = servicer.Generate(request(temperature=0.0), FakeContext())

    assert first.text == second.text


def test_seeded_sampling_is_reproducible(servicer):
    req = request(temperature=1.0, top_p=0.9, seed=42)

    first = servicer.Generate(req, FakeContext())
    second = servicer.Generate(req, FakeContext())

    assert first.text == second.text


def test_unset_top_p_does_not_empty_the_nucleus(servicer):
    """proto3 defaults top_p to 0.0, which must be read as "no filtering"."""
    response = servicer.Generate(request(temperature=1.0, top_p=0.0), FakeContext())

    assert response.completion_tokens > 0


def test_engine_failure_becomes_internal(config):
    servicer = InferenceServicer(FakeHandle(config=config, engine=ExplodingEngine()))

    assert abort_code(servicer, request()) == grpc.StatusCode.INTERNAL


def test_in_flight_is_released_after_failure(config):
    servicer = InferenceServicer(FakeHandle(config=config, engine=ExplodingEngine()))

    with pytest.raises(AbortedError):
        servicer.Generate(request(), FakeContext())

    assert servicer.in_flight == 0


def test_in_flight_is_released_after_success(servicer):
    servicer.Generate(request(), FakeContext())

    assert servicer.in_flight == 0


def test_in_flight_is_visible_during_generation(config):
    seen = []

    class ObservingEngine:
        def generate(self, **_):
            seen.append(servicer.in_flight)
            return GenerationResult(text="", prompt_tokens=1, completion_tokens=1)

    servicer = InferenceServicer(FakeHandle(config=config, engine=ObservingEngine()))
    servicer.Generate(request(), FakeContext())

    assert seen == [1]


def test_health_reports_engine_state(servicer, config):
    response = servicer.Health(inference_pb2.HealthRequest(), FakeContext())

    assert response.ready is True
    assert response.model == config.model_name
    assert response.device == config.device
    assert response.in_flight == 0


def test_health_reports_not_ready_before_load(config, engine):
    servicer = InferenceServicer(FakeHandle(config=config, engine=engine, ready=False))

    assert servicer.Health(inference_pb2.HealthRequest(), FakeContext()).ready is False
