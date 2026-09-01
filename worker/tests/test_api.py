"""HTTP dev surface.

The FastAPI app is a debugging convenience, not the production path, so these
tests cover request validation and the response contract rather than model
behaviour -- that lives in test_model.py.
"""

from __future__ import annotations

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
