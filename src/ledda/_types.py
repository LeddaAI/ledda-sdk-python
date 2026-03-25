from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, TypedDict


class MessageDict(TypedDict):
    role: str
    content: str


class Fallback(TypedDict, total=False):
    """Fallback prompt shape — must contain a ``messages`` list.

    ``config`` is optional and defaults to ``{}``.
    """

    messages: List[MessageDict]
    config: Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Prompt:
    """Resolved prompt returned by Ledda SDK.

    This object is immutable. Use .compile() to create a new Prompt with
    template variables replaced.
    """

    messages: List[MessageDict]
    config: Dict[str, Any]
    version: int
    name: str
    label: str
    variables: List[str]
    rule_id: Optional[str]
    rule_name: Optional[str]
    is_fallback: bool
    source: str  # "api" | "cache" | "fallback"

    @property
    def span_attributes(self) -> Dict[str, Any]:
        """OTel span attributes for manual injection."""
        attrs: Dict[str, Any] = {
            "ledda.prompt.name": self.name,
            "ledda.prompt.version": self.version,
            "ledda.prompt.label": self.label,
            "ledda.prompt.source": self.source,
        }
        if self.rule_id is not None:
            attrs["ledda.prompt.rule_id"] = self.rule_id
        return attrs

    def compile(self, *, strict: bool = False, **variables: str) -> Prompt:
        """Replace {{var}} placeholders in message contents.

        Returns a new Prompt — does not mutate this one.

        Args:
            strict: If True, raises ValueError for unknown or missing variables.
            **variables: Variable name → value mappings.
        """
        from ledda._compile import compile_prompt

        return compile_prompt(self, variables, strict=strict)

    @staticmethod
    def _from_fallback(
        fallback: Fallback,
        *,
        name: str,
        label: str,
    ) -> Prompt:
        """Build a Prompt from a customer-provided fallback."""
        return Prompt(
            messages=fallback.get("messages", []),
            config=fallback.get("config", {}),
            version=0,
            name=name,
            label=label,
            variables=[],
            rule_id=None,
            rule_name=None,
            is_fallback=True,
            source="fallback",
        )

    @staticmethod
    def _from_api_response(
        data: Dict[str, Any],
        *,
        source: str = "api",
    ) -> Prompt:
        """Build a Prompt from the API response JSON."""
        return Prompt(
            messages=data.get("messages", []),
            config=data.get("config", {}),
            version=data.get("version", 0),
            name=data.get("name", ""),
            label=data.get("label", ""),
            variables=data.get("variables", []),
            rule_id=data.get("ruleId"),
            rule_name=data.get("ruleName"),
            is_fallback=False,
            source=source,
        )
