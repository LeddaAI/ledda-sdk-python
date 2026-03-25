"""Tests for the Ledda sync and async clients with mocked HTTP."""

import time

import httpx
import pytest

from ledda import (
    AsyncLedda,
    Ledda,
    LeddaConnectionError,
    Prompt,
    PromptNotDeployedError,
    PromptNotFoundError,
)

MOCK_RESPONSE = {
    "name": "welcome",
    "label": "production",
    "version": 5,
    "messages": [{"role": "system", "content": "Hello {{name}}"}],
    "config": {"model": "claude-sonnet-4-20250514", "temperature": 0.7},
    "variables": ["name"],
    "ruleId": None,
    "ruleName": None,
}

FB = {"messages": [{"role": "system", "content": "You are helpful."}]}


# -- Sync client tests --


class TestLeddaSync:
    def test_basic_fetch(self, httpx_mock):
        httpx_mock.add_response(
            json=MOCK_RESPONSE,
            headers={"etag": '"v5"'},
        )
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = client.get_prompt("welcome", fallback=FB)

        assert prompt.version == 5
        assert prompt.is_fallback is False
        assert prompt.source == "api"
        assert prompt.messages[0]["content"] == "Hello {{name}}"

    def test_api_down_returns_fallback(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = client.get_prompt("welcome", fallback=FB)

        assert prompt.is_fallback is True
        assert prompt.source == "fallback"
        assert prompt.messages == [{"role": "system", "content": "You are helpful."}]

    def test_fallback_with_config(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("fail"))
        fb = {"messages": [{"role": "system", "content": "Custom fallback"}], "config": {"model": "gpt-4"}}
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = client.get_prompt("welcome", fallback=fb)

        assert prompt.messages == [{"role": "system", "content": "Custom fallback"}]
        assert prompt.config == {"model": "gpt-4"}

    def test_fallback_multi_message(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("fail"))
        fb = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = client.get_prompt("welcome", fallback=fb)

        assert len(prompt.messages) == 2
        assert prompt.messages[1]["role"] == "user"
        assert prompt.is_fallback is True

    def test_cache_hit(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE, headers={"etag": '"v5"'})
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            p1 = client.get_prompt("welcome", fallback=FB)
            p2 = client.get_prompt("welcome", fallback=FB)

        assert p1.source == "api"
        assert p2.source == "cache"
        # Only one HTTP request was made
        assert len(httpx_mock.get_requests()) == 1

    def test_prompt_not_found_returns_fallback(self, httpx_mock):
        httpx_mock.add_response(
            json={"error": "PROMPT_NOT_FOUND", "message": "Prompt 'nope' not found"},
        )
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = client.get_prompt("nope", fallback=FB)

        assert prompt.is_fallback is True

    def test_label_not_found_returns_fallback(self, httpx_mock):
        httpx_mock.add_response(
            json={
                "error": "LABEL_NOT_FOUND",
                "message": "Prompt 'welcome' exists but has no 'production' environment",
            },
        )
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = client.get_prompt("welcome", fallback=FB)

        assert prompt.is_fallback is True

    def test_strict_mode_prompt_not_found(self, httpx_mock):
        httpx_mock.add_response(
            json={"error": "PROMPT_NOT_FOUND", "message": "Not found"},
        )
        with Ledda(
            api_key="ldda_ak_test", base_url="https://test.ledda.ai", strict_mode=True
        ) as client:
            with pytest.raises(PromptNotFoundError):
                client.get_prompt("nope", fallback=FB)

    def test_strict_mode_label_not_found(self, httpx_mock):
        httpx_mock.add_response(
            json={"error": "LABEL_NOT_FOUND", "message": "Not deployed"},
        )
        with Ledda(
            api_key="ldda_ak_test", base_url="https://test.ledda.ai", strict_mode=True
        ) as client:
            with pytest.raises(PromptNotDeployedError):
                client.get_prompt("welcome", fallback=FB)

    def test_strict_mode_connection_error(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("fail"))
        with Ledda(
            api_key="ldda_ak_test", base_url="https://test.ledda.ai", strict_mode=True
        ) as client:
            with pytest.raises(LeddaConnectionError):
                client.get_prompt("welcome", fallback=FB)

    def test_compile_after_fetch(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE)
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = client.get_prompt("welcome", fallback=FB)
            compiled = prompt.compile(name="Alice")

        assert compiled.messages[0]["content"] == "Hello Alice"
        assert prompt.messages[0]["content"] == "Hello {{name}}"  # original unchanged

    def test_validate_prompt_true(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE)
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            assert client.validate_prompt("welcome") is True

    def test_validate_prompt_false(self, httpx_mock):
        httpx_mock.add_response(
            json={"error": "PROMPT_NOT_FOUND", "message": "Not found"},
        )
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            assert client.validate_prompt("nope") is False

    def test_validate_prompt_connection_error(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("fail"))
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            assert client.validate_prompt("welcome") is False

    def test_prefetch(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE, headers={"etag": '"v5"'})
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            client.prefetch(["welcome"])
            # Next get should be cache hit
            prompt = client.get_prompt("welcome", fallback=FB)

        assert prompt.source == "cache"
        assert len(httpx_mock.get_requests()) == 1

    def test_default_label(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE)
        with Ledda(
            api_key="ldda_ak_test",
            base_url="https://test.ledda.ai",
            default_label="staging",
        ) as client:
            client.get_prompt("welcome", fallback=FB)

        request = httpx_mock.get_requests()[0]
        import json

        body = json.loads(request.content)
        assert body["label"] == "staging"

    def test_cache_stats(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE)
        with Ledda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            client.get_prompt("welcome", fallback=FB)
            client.get_prompt("welcome", fallback=FB)
            stats = client.cache_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


# -- Async client tests --


class TestAsyncLedda:
    @pytest.mark.asyncio
    async def test_basic_fetch(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE, headers={"etag": '"v5"'})
        async with AsyncLedda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = await client.get_prompt("welcome", fallback=FB)

        assert prompt.version == 5
        assert prompt.is_fallback is False

    @pytest.mark.asyncio
    async def test_api_down_returns_fallback(self, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("fail"))
        async with AsyncLedda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            prompt = await client.get_prompt("welcome", fallback=FB)

        assert prompt.is_fallback is True
        assert prompt.source == "fallback"

    @pytest.mark.asyncio
    async def test_cache_hit(self, httpx_mock):
        httpx_mock.add_response(json=MOCK_RESPONSE)
        async with AsyncLedda(api_key="ldda_ak_test", base_url="https://test.ledda.ai") as client:
            p1 = await client.get_prompt("welcome", fallback=FB)
            p2 = await client.get_prompt("welcome", fallback=FB)

        assert p1.source == "api"
        assert p2.source == "cache"

    @pytest.mark.asyncio
    async def test_strict_mode(self, httpx_mock):
        httpx_mock.add_response(
            json={"error": "PROMPT_NOT_FOUND", "message": "Not found"},
        )
        async with AsyncLedda(
            api_key="ldda_ak_test", base_url="https://test.ledda.ai", strict_mode=True
        ) as client:
            with pytest.raises(PromptNotFoundError):
                await client.get_prompt("nope", fallback=FB)
