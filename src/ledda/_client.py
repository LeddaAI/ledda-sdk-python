"""HTTP client layer for the Ledda API.

Wraps httpx with timeouts, ETag support, and error handling.
Never raises — returns (data, etag, error_code, error_message) tuples.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import httpx

logger = logging.getLogger("ledda")

_TIMEOUT = 3.0  # seconds — hard cap to stay off the critical path
_RESOLVE_PATH = "/v1/prompts/resolve"

# Return type: (response_data | None, etag | None, error_code | None, error_msg | None)
ResolveResult = Tuple[
    Optional[Dict[str, Any]],
    Optional[str],
    Optional[str],
    Optional[str],
]


def _build_resolve_body(
    name: str,
    label: str,
    attributes: Optional[Dict[str, str]],
    routing_key: Optional[str],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"name": name, "label": label}
    if attributes:
        body["attributes"] = attributes
    if routing_key is not None:
        body["routingKey"] = routing_key
    return body


def _parse_response(resp: httpx.Response) -> ResolveResult:
    if resp.status_code == 304:
        return None, None, "NOT_MODIFIED", None

    data = resp.json()
    etag = resp.headers.get("etag")

    if "error" in data:
        return None, None, data["error"], data.get("message")

    return data, etag, None, None


class SyncHttpClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )

    def resolve(
        self,
        name: str,
        label: str,
        attributes: Optional[Dict[str, str]] = None,
        routing_key: Optional[str] = None,
        if_none_match: Optional[str] = None,
    ) -> ResolveResult:
        headers: Dict[str, str] = {}
        if if_none_match:
            headers["If-None-Match"] = if_none_match

        try:
            resp = self._client.post(
                _RESOLVE_PATH,
                json=_build_resolve_body(name, label, attributes, routing_key),
                headers=headers,
            )
            return _parse_response(resp)
        except Exception as exc:
            logger.debug("Ledda API request failed: %s", exc)
            return None, None, "CONNECTION_ERROR", str(exc)

    def close(self) -> None:
        self._client.close()


class AsyncHttpClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )

    async def resolve(
        self,
        name: str,
        label: str,
        attributes: Optional[Dict[str, str]] = None,
        routing_key: Optional[str] = None,
        if_none_match: Optional[str] = None,
    ) -> ResolveResult:
        headers: Dict[str, str] = {}
        if if_none_match:
            headers["If-None-Match"] = if_none_match

        try:
            resp = await self._client.post(
                _RESOLVE_PATH,
                json=_build_resolve_body(name, label, attributes, routing_key),
                headers=headers,
            )
            return _parse_response(resp)
        except Exception as exc:
            logger.debug("Ledda API request failed: %s", exc)
            return None, None, "CONNECTION_ERROR", str(exc)

    async def close(self) -> None:
        await self._client.aclose()
