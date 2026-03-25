"""Template variable compilation for prompt messages."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from ledda._types import MessageDict, Prompt

logger = logging.getLogger("ledda")

_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def compile_prompt(prompt: Prompt, variables: Dict[str, str], *, strict: bool = False) -> Prompt:
    """Replace {{var}} placeholders in all message contents.

    Returns a new Prompt — does not mutate the original.
    """
    import dataclasses

    known_vars = set(prompt.variables) if prompt.variables else _extract_variables(prompt.messages)
    provided = set(variables.keys())

    if strict:
        unknown = provided - known_vars
        if unknown:
            raise ValueError(
                f"Unknown variables: {unknown}. Expected: {known_vars}"
            )
        missing = known_vars - provided
        if missing:
            raise ValueError(
                f"Missing variables: {missing}. Expected: {known_vars}"
            )

    missing = known_vars - provided
    if missing:
        logger.warning("Unresolved variables: %s", ", ".join(sorted(missing)))

    compiled_messages = [_compile_message(m, variables) for m in prompt.messages]
    return dataclasses.replace(prompt, messages=compiled_messages)


def _compile_message(message: MessageDict, variables: Dict[str, str]) -> MessageDict:
    content = message["content"]
    for var_name, var_value in variables.items():
        content = content.replace("{{" + var_name + "}}", var_value)
    return {"role": message["role"], "content": content}


def _extract_variables(messages: List[MessageDict]) -> Set[str]:
    """Extract {{var}} names from messages (fallback when variables list is empty)."""
    found: Set[str] = set()
    for m in messages:
        found.update(_VAR_PATTERN.findall(m.get("content", "")))
    return found
