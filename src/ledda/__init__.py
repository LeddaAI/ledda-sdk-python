"""Ledda Python SDK — resilient prompt management for LLM applications.

Usage:
    from ledda import Ledda

    ledda = Ledda(api_key="ldda_ak_...")
    prompt = ledda.get_prompt("welcome", fallback={
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    })
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from ledda._cache import LRUCache
from ledda._client import AsyncHttpClient, SyncHttpClient
from ledda._exceptions import (
    LeddaConnectionError,
    LeddaError,
    PromptNotDeployedError,
    PromptNotFoundError,
)
from ledda._types import Fallback, Prompt

__all__ = [
    "Ledda",
    "AsyncLedda",
    "Prompt",
    "LeddaError",
    "PromptNotFoundError",
    "PromptNotDeployedError",
    "LeddaConnectionError",
]

__version__ = "0.0.1"

logger = logging.getLogger("ledda")

_DEFAULT_BASE_URL = "https://api.ledda.ai"


class Ledda:
    """Synchronous Ledda client.

    Designed for the critical path: never raises in default mode, always
    returns usable prompt content via cache or fallback.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        cache_ttl: float = 60.0,
        cache_max_size: int = 1000,
        default_label: str = "production",
        enable_otel_injection: bool = False,
        strict_mode: bool = False,
        debug: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._default_label = default_label
        self._enable_otel = enable_otel_injection
        self._strict_mode = strict_mode
        self._debug = debug

        self._cache = LRUCache(max_size=cache_max_size, default_ttl=cache_ttl)
        self._http = SyncHttpClient(base_url=base_url, api_key=api_key)
        # Single-thread pool for background stale refreshes — bounded, non-blocking
        self._refresh_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ledda-refresh")

        if debug:
            logging.getLogger("ledda").setLevel(logging.DEBUG)

    # -- Context manager --

    def __enter__(self) -> Ledda:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._refresh_pool.shutdown(wait=False)
        self._http.close()

    # -- Core API --

    def get_prompt(
        self,
        name: str,
        *,
        fallback: Fallback,
        label: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        routing_key: Optional[str] = None,
        cache_ttl: Optional[float] = None,
    ) -> Prompt:
        """Fetch a prompt by name. Always returns a usable Prompt.

        In default mode, this never raises. On any failure it returns
        the fallback content. Use strict_mode=True to get exceptions.
        """
        resolved_label = label or self._default_label
        cache_key = (name, resolved_label)

        try:
            # Check cache first
            entry, is_fresh = self._cache.get(cache_key, ttl_override=cache_ttl)

            if entry is not None and is_fresh:
                self._log_debug("Cache hit (fresh): %s/%s", name, resolved_label)
                prompt = Prompt._from_api_response(entry.data, source="cache")
                self._inject_otel(prompt)
                return prompt

            if entry is not None:
                # Stale cache — serve immediately, refresh in background
                self._log_debug("Cache hit (stale, refreshing): %s/%s", name, resolved_label)
                self._trigger_background_refresh(
                    name, resolved_label, attributes, routing_key, entry.etag
                )
                prompt = Prompt._from_api_response(entry.data, source="cache")
                self._inject_otel(prompt)
                return prompt

            # Cache miss — synchronous fetch
            self._log_debug("Cache miss, fetching: %s/%s", name, resolved_label)
            data, etag, error_code, error_msg = self._http.resolve(
                name, resolved_label, attributes, routing_key
            )

            if data is not None:
                self._cache.put(cache_key, data, etag)
                prompt = Prompt._from_api_response(data)
                self._inject_otel(prompt)
                return prompt

            # API returned an error
            return self._handle_error(
                name, resolved_label, fallback, error_code, error_msg
            )

        except LeddaError:
            raise
        except Exception as exc:
            logger.warning("Unexpected error in Ledda SDK: %s", exc)
            if self._strict_mode:
                raise LeddaConnectionError(str(exc)) from exc
            return Prompt._from_fallback(fallback, name=name, label=resolved_label)

    def prefetch(self, names: Sequence[str], label: Optional[str] = None) -> None:
        """Pre-warm the cache for a list of prompt names."""
        resolved_label = label or self._default_label
        for name in names:
            cache_key = (name, resolved_label)
            entry, _ = self._cache.get(cache_key)
            if entry is not None:
                continue
            data, etag, _, _ = self._http.resolve(name, resolved_label)
            if data is not None:
                self._cache.put(cache_key, data, etag)
                self._log_debug("Prefetched: %s/%s", name, resolved_label)

    def validate_prompt(self, name: str, label: Optional[str] = None) -> bool:
        """Check if a prompt exists and is deployed. Does not raise."""
        resolved_label = label or self._default_label
        try:
            data, _, error_code, _ = self._http.resolve(name, resolved_label)
            return data is not None
        except Exception:
            return False

    def cache_stats(self) -> Dict[str, int]:
        return self._cache.get_stats_dict()

    # -- Private helpers --

    def _handle_error(
        self,
        name: str,
        label: str,
        fallback: Fallback,
        error_code: Optional[str],
        error_msg: Optional[str],
    ) -> Prompt:
        if error_code == "PROMPT_NOT_FOUND":
            logger.warning("Prompt '%s' not found", name)
            if self._strict_mode:
                raise PromptNotFoundError(name, error_msg or "")
        elif error_code == "LABEL_NOT_FOUND":
            msg = f"Prompt '{name}' exists but has no '{label}' environment. Deploy it first."
            logger.warning(msg)
            if self._strict_mode:
                raise PromptNotDeployedError(name, label, error_msg or msg)
        elif error_code == "CONNECTION_ERROR":
            logger.warning("Could not connect to Ledda API: %s", error_msg)
            if self._strict_mode:
                raise LeddaConnectionError(error_msg or "")
        else:
            logger.warning("Ledda API error (%s): %s", error_code, error_msg)
            if self._strict_mode:
                raise LeddaError(error_msg or f"API error: {error_code}")

        return Prompt._from_fallback(fallback, name=name, label=label)

    def _trigger_background_refresh(
        self,
        name: str,
        label: str,
        attributes: Optional[Dict[str, str]],
        routing_key: Optional[str],
        etag: Optional[str],
    ) -> None:
        """Submit a non-blocking refresh to the thread pool."""
        try:
            self._refresh_pool.submit(
                self._do_refresh, name, label, attributes, routing_key, etag
            )
        except RuntimeError:
            # Pool shut down — ignore
            pass

    def _do_refresh(
        self,
        name: str,
        label: str,
        attributes: Optional[Dict[str, str]],
        routing_key: Optional[str],
        etag: Optional[str],
    ) -> None:
        cache_key = (name, label)
        data, new_etag, error_code, _ = self._http.resolve(
            name, label, attributes, routing_key, if_none_match=etag
        )

        if error_code == "NOT_MODIFIED":
            self._cache.touch(cache_key)
            self._log_debug("Background refresh: 304 Not Modified for %s/%s", name, label)
        elif data is not None:
            self._cache.put(cache_key, data, new_etag)
            self._log_debug("Background refresh: updated %s/%s", name, label)
        else:
            self._cache.record_refresh_failure()
            self._log_debug("Background refresh failed for %s/%s", name, label)

    def _inject_otel(self, prompt: Prompt) -> None:
        if not self._enable_otel:
            return
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.is_recording():
                for k, v in prompt.span_attributes.items():
                    span.set_attribute(k, v)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("OTel injection failed: %s", exc)

    def _log_debug(self, msg: str, *args: Any) -> None:
        if self._debug:
            logger.debug(msg, *args)


class AsyncLedda:
    """Async Ledda client for FastAPI, async frameworks.

    Same resilience guarantees as the sync client.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        cache_ttl: float = 60.0,
        cache_max_size: int = 1000,
        default_label: str = "production",
        enable_otel_injection: bool = False,
        strict_mode: bool = False,
        debug: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._default_label = default_label
        self._enable_otel = enable_otel_injection
        self._strict_mode = strict_mode
        self._debug = debug

        self._cache = LRUCache(max_size=cache_max_size, default_ttl=cache_ttl)
        self._http = AsyncHttpClient(base_url=base_url, api_key=api_key)

        if debug:
            logging.getLogger("ledda").setLevel(logging.DEBUG)

    # -- Context manager --

    async def __aenter__(self) -> AsyncLedda:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.close()

    # -- Core API --

    async def get_prompt(
        self,
        name: str,
        *,
        fallback: Fallback,
        label: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        routing_key: Optional[str] = None,
        cache_ttl: Optional[float] = None,
    ) -> Prompt:
        """Fetch a prompt by name. Always returns a usable Prompt."""
        resolved_label = label or self._default_label
        cache_key = (name, resolved_label)

        try:
            entry, is_fresh = self._cache.get(cache_key, ttl_override=cache_ttl)

            if entry is not None and is_fresh:
                self._log_debug("Cache hit (fresh): %s/%s", name, resolved_label)
                prompt = Prompt._from_api_response(entry.data, source="cache")
                self._inject_otel(prompt)
                return prompt

            if entry is not None:
                self._log_debug("Cache hit (stale, refreshing): %s/%s", name, resolved_label)
                # Fire-and-forget background refresh via asyncio task
                import asyncio

                asyncio.ensure_future(
                    self._do_refresh(
                        name, resolved_label, attributes, routing_key, entry.etag
                    )
                )
                prompt = Prompt._from_api_response(entry.data, source="cache")
                self._inject_otel(prompt)
                return prompt

            self._log_debug("Cache miss, fetching: %s/%s", name, resolved_label)
            data, etag, error_code, error_msg = await self._http.resolve(
                name, resolved_label, attributes, routing_key
            )

            if data is not None:
                self._cache.put(cache_key, data, etag)
                prompt = Prompt._from_api_response(data)
                self._inject_otel(prompt)
                return prompt

            return self._handle_error(
                name, resolved_label, fallback, error_code, error_msg
            )

        except LeddaError:
            raise
        except Exception as exc:
            logger.warning("Unexpected error in Ledda SDK: %s", exc)
            if self._strict_mode:
                raise LeddaConnectionError(str(exc)) from exc
            return Prompt._from_fallback(fallback, name=name, label=resolved_label)

    async def prefetch(self, names: Sequence[str], label: Optional[str] = None) -> None:
        """Pre-warm the cache for a list of prompt names."""
        resolved_label = label or self._default_label
        for name in names:
            cache_key = (name, resolved_label)
            entry, _ = self._cache.get(cache_key)
            if entry is not None:
                continue
            data, etag, _, _ = await self._http.resolve(name, resolved_label)
            if data is not None:
                self._cache.put(cache_key, data, etag)

    async def validate_prompt(self, name: str, label: Optional[str] = None) -> bool:
        resolved_label = label or self._default_label
        try:
            data, _, _, _ = await self._http.resolve(name, resolved_label)
            return data is not None
        except Exception:
            return False

    def cache_stats(self) -> Dict[str, int]:
        return self._cache.get_stats_dict()

    # -- Private helpers --

    def _handle_error(
        self,
        name: str,
        label: str,
        fallback: Fallback,
        error_code: Optional[str],
        error_msg: Optional[str],
    ) -> Prompt:
        if error_code == "PROMPT_NOT_FOUND":
            logger.warning("Prompt '%s' not found", name)
            if self._strict_mode:
                raise PromptNotFoundError(name, error_msg or "")
        elif error_code == "LABEL_NOT_FOUND":
            msg = f"Prompt '{name}' exists but has no '{label}' environment. Deploy it first."
            logger.warning(msg)
            if self._strict_mode:
                raise PromptNotDeployedError(name, label, error_msg or msg)
        elif error_code == "CONNECTION_ERROR":
            logger.warning("Could not connect to Ledda API: %s", error_msg)
            if self._strict_mode:
                raise LeddaConnectionError(error_msg or "")
        else:
            logger.warning("Ledda API error (%s): %s", error_code, error_msg)
            if self._strict_mode:
                raise LeddaError(error_msg or f"API error: {error_code}")

        return Prompt._from_fallback(fallback, name=name, label=label)

    async def _do_refresh(
        self,
        name: str,
        label: str,
        attributes: Optional[Dict[str, str]],
        routing_key: Optional[str],
        etag: Optional[str],
    ) -> None:
        cache_key = (name, label)
        try:
            data, new_etag, error_code, _ = await self._http.resolve(
                name, label, attributes, routing_key, if_none_match=etag
            )
            if error_code == "NOT_MODIFIED":
                self._cache.touch(cache_key)
            elif data is not None:
                self._cache.put(cache_key, data, new_etag)
            else:
                self._cache.record_refresh_failure()
        except Exception:
            self._cache.record_refresh_failure()

    def _inject_otel(self, prompt: Prompt) -> None:
        if not self._enable_otel:
            return
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.is_recording():
                for k, v in prompt.span_attributes.items():
                    span.set_attribute(k, v)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("OTel injection failed: %s", exc)

    def _log_debug(self, msg: str, *args: Any) -> None:
        if self._debug:
            logger.debug(msg, *args)
