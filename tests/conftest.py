"""Shared fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import pytest
from llm_kit import LLMClient

ENV_VARS = ("OPENROUTER_API_KEY", "MODEL_PRIMARY", "MODEL_FALLBACKS", "EVAL_MODEL", "LLM_LOG_PATH")


class FakeSDK:
    """Stands in for the openai SDK client that LLMClient wraps.

    `replies` are returned in order, one per chat call; each one is a string or a
    callable taking the request kwargs. Every request is recorded in `calls`.
    """

    def __init__(self, replies: Iterable[str | Callable[[dict[str, Any]], str]]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        if len(self.calls) >= len(self._replies):
            raise AssertionError("FakeSDK ran out of replies")
        reply = self._replies[len(self.calls)]
        self.calls.append(kwargs)
        text = reply(kwargs) if callable(reply) else reply
        message = SimpleNamespace(content=text)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(model=kwargs["model"], choices=[choice], usage=None, error=None)

    def last_prompt(self) -> str:
        return self.calls[-1]["messages"][-1]["content"]


def fake_client(*replies: str | Callable[[dict[str, Any]], str]) -> tuple[LLMClient, FakeSDK]:
    sdk = FakeSDK(replies)
    client = LLMClient(
        api_key="test-key",
        model_primary="fake/model",
        model_fallbacks=(),
        enable_log=False,
        sleep=lambda _seconds: None,
        client=sdk,
    )
    return client, sdk


def order_json(**overrides: Any) -> str:
    """A valid Order reply, as a model would send it."""
    order = {
        "customer_name": "Ana Souza",
        "items": [{"name": "pão de queijo", "quantity": 12, "unit_price": 1.5}],
        "delivery_date": "2026-11-13",
        "address": "Rua das Flores, 10, Centro, Campinas",
        "notes": None,
        "confianca": {
            "customer_name": 0.95,
            "items": 0.9,
            "delivery_date": 0.8,
            "address": 0.9,
            "notes": 0.9,
        },
        "confianca_geral": 0.88,
    }
    order.update(overrides)
    return json.dumps(order, ensure_ascii=False)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Run each test from an empty dir with no settings; restore os.environ afterwards.

    load_dotenv writes to os.environ directly, which monkeypatch can't undo.
    """
    saved = dict(os.environ)
    for name in ENV_VARS:
        os.environ.pop(name, None)
    monkeypatch.chdir(tmp_path)
    yield
    os.environ.clear()
    os.environ.update(saved)
