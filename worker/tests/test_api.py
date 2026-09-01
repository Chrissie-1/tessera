"""HTTP dev surface.

The FastAPI app is a debugging convenience, not the production path, so these
tests cover request validation and the response contract rather than model
behaviour -- that lives in test_model.py.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from tessera_worker.api import create_app


@pytest.fixture(scope="module")
def client(config):
    with TestClient(create_app(config)) as test_client:
        yield test_client


def test_health_reports_ready_after_startup(client, config):
    body = client.get("/health").json()

    assert body["ready"] is True
    assert body["model"] == config.model_name
    assert body["device"] == config.device
    assert body["backend"] == "reference"


def test_completion_returns_full_contract(client):
    response = client.post(
        "/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 4}
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["text"], str)
    assert body["prompt_tokens"] > 0
    assert body["completion_tokens"] <= 4
    assert body["finish_reason"] in {"length", "stop"}
    assert body["latency_ms"] >= 0.0


def test_greedy_completion_is_deterministic(client):
    payload = {"prompt": "Hello world", "max_tokens": 6, "temperature": 0.0}

    first = client.post("/v1/completions", json=payload).json()
    second = client.post("/v1/completions", json=payload).json()

    assert first["text"] == second["text"]


def test_max_tokens_above_cap_is_rejected(client, config):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "max_tokens": config.max_tokens_cap + 1},
    )

    assert response.status_code == 422
    assert str(config.max_tokens_cap) in response.json()["detail"]


def test_max_tokens_at_cap_is_allowed(client, config):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "max_tokens": config.max_tokens_cap},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "hi", "max_tokens": 0},
        {"prompt": "hi", "max_tokens": -1},
        {"prompt": "hi", "top_p": 0.0},
        {"prompt": "hi", "top_p": 1.5},
        {"prompt": "hi", "temperature": -0.1},
        {"max_tokens": 4},
    ],
    ids=["zero", "negative", "top_p_zero", "top_p_above_one", "cold", "no_prompt"],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/v1/completions", json=payload).status_code == 422


def test_defaults_apply_when_only_prompt_is_given(client):
    response = client.post("/v1/completions", json={"prompt": "Once upon a time"})

    assert response.status_code == 200
    # max_tokens defaults to 16.
    assert response.json()["completion_tokens"] <= 16


def test_empty_prompt_is_accepted(client):
    response = client.post("/v1/completions", json={"prompt": "", "max_tokens": 2})

    assert response.status_code == 200
    assert response.json()["prompt_tokens"] >= 1


# -- streaming ---------------------------------------------------------------
#
# The engines have produced tokens one at a time since Phase 2, but this
# wrapper only ever exposed the accumulated result. These cover the SSE
# surface, and the case where the configured backend cannot stream at all.


@pytest.fixture(scope="module")
def streaming_client(config):
    """A client on the paged backend, which implements `iter_generate`."""
    previous = os.environ.get("TESSERA_BACKEND")
    os.environ["TESSERA_BACKEND"] = "paged"
    try:
        with TestClient(create_app(config)) as test_client:
            yield test_client
    finally:
        if previous is None:
            os.environ.pop("TESSERA_BACKEND", None)
        else:
            os.environ["TESSERA_BACKEND"] = previous


def _events(response) -> list[str]:
    """The `data:` payloads of an SSE body, in order."""
    return [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_streaming_is_refused_when_the_backend_cannot_stream(client):
    """The reference engine has no iter_generate; say so rather than crash."""
    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "max_tokens": 4, "stream": True},
    )

    assert response.status_code == 400
    assert "cannot stream" in response.json()["detail"]


def test_streaming_returns_an_event_stream(streaming_client):
    response = streaming_client.post(
        "/v1/completions",
        json={"prompt": "The capital of France is", "max_tokens": 4, "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_stream_terminates_with_done(streaming_client):
    response = streaming_client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 4, "stream": True},
    )

    events = _events(response)
    assert events, "expected at least one event"
    assert events[-1] == "[DONE]"


def test_stream_chunks_carry_one_token_delta_each(streaming_client):
    response = streaming_client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 3, "stream": True},
    )

    chunks = [json.loads(e) for e in _events(response) if e != "[DONE]"]
    assert chunks, "expected token chunks"
    for chunk in chunks:
        assert chunk["object"] == "text_completion.chunk"
        assert chunk["choices"][0]["index"] == 0
        assert isinstance(chunk["choices"][0]["text"], str)

    # Only the terminal chunk may report why generation stopped.
    reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert all(r is None for r in reasons[:-1])
    assert reasons[-1] in {"length", "stop"}


def test_streamed_text_matches_the_unary_completion(streaming_client):
    """Streaming must be a different delivery of the same tokens, not different
    tokens: concatenating the deltas has to reproduce the unary body exactly."""
    payload = {"prompt": "The capital of France is", "max_tokens": 6}

    unary = streaming_client.post("/v1/completions", json=payload).json()
    streamed = streaming_client.post(
        "/v1/completions", json={**payload, "stream": True}
    )

    deltas = [
        json.loads(e)["choices"][0]["text"] for e in _events(streamed) if e != "[DONE]"
    ]
    assert "".join(deltas) == unary["text"]
