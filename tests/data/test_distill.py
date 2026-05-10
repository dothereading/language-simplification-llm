"""Tests for the Teacher distillation client.

The Teacher wraps the OpenRouter chat-completions API and handles retries.
The OpenAI client is mocked so no network/API key is needed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

distill = pytest.importorskip("langsimp.data.distill", reason="distill.py not implemented yet (RED)")


def _fake_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _fake_client(side_effect):
    """Build a stand-in for AsyncOpenAI exposing chat.completions.create()."""
    create = AsyncMock(side_effect=side_effect)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ), create


class TestTeacherSimplify:
    async def test_returns_text_on_success(self):
        client, create = _fake_client([_fake_response("simplified")])
        teacher = distill.Teacher(client=client, model="m", max_retries=3, temperature=0.4)
        out = await teacher.simplify("system", "complex text")
        assert out == "simplified"
        assert create.await_count == 1

    async def test_strips_whitespace_from_response(self):
        client, _ = _fake_client([_fake_response("  hello  \n")])
        teacher = distill.Teacher(client=client, model="m", max_retries=1, temperature=0.0)
        assert await teacher.simplify("s", "u") == "hello"

    async def test_retries_on_exception_then_succeeds(self):
        client, create = _fake_client([
            RuntimeError("boom"),
            _fake_response("ok"),
        ])
        teacher = distill.Teacher(client=client, model="m", max_retries=3, temperature=0.0)
        out = await teacher.simplify("s", "u")
        assert out == "ok"
        assert create.await_count == 2

    async def test_returns_none_after_max_retries(self):
        client, create = _fake_client([RuntimeError("boom")] * 5)
        teacher = distill.Teacher(client=client, model="m", max_retries=2, temperature=0.0)
        out = await teacher.simplify("s", "u")
        assert out is None
        assert create.await_count == 2

    async def test_empty_response_counts_as_failure(self):
        client, create = _fake_client([_fake_response(""), _fake_response("real")])
        teacher = distill.Teacher(client=client, model="m", max_retries=2, temperature=0.0)
        out = await teacher.simplify("s", "u")
        assert out == "real"
        assert create.await_count == 2

    async def test_passes_system_and_user_messages(self):
        client, create = _fake_client([_fake_response("x")])
        teacher = distill.Teacher(client=client, model="m", max_retries=1, temperature=0.5)
        await teacher.simplify("SYSTEM", "USER")
        kwargs = create.await_args.kwargs
        assert kwargs["model"] == "m"
        assert kwargs["temperature"] == 0.5
        msgs = kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "SYSTEM"}
        assert msgs[1] == {"role": "user", "content": "USER"}
