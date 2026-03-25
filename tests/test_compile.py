import logging

import pytest

from ledda._types import Prompt


def _make_prompt(content: str, variables: list = None) -> Prompt:
    return Prompt(
        messages=[{"role": "system", "content": content}],
        config={},
        version=1,
        name="test",
        label="production",
        variables=variables or [],
        rule_id=None,
        rule_name=None,
        is_fallback=False,
        source="api",
    )


def test_basic_compile():
    p = _make_prompt("Hello {{name}}", ["name"])
    compiled = p.compile(name="Alice")
    assert compiled.messages[0]["content"] == "Hello Alice"
    # Original unchanged
    assert p.messages[0]["content"] == "Hello {{name}}"


def test_multiple_variables():
    p = _make_prompt("{{greeting}} {{name}}", ["greeting", "name"])
    compiled = p.compile(greeting="Hi", name="Bob")
    assert compiled.messages[0]["content"] == "Hi Bob"


def test_missing_variable_warns(caplog):
    p = _make_prompt("Hello {{name}}, role is {{role}}", ["name", "role"])
    with caplog.at_level(logging.WARNING, logger="ledda"):
        compiled = p.compile(name="Alice")
    assert "role" in caplog.text
    assert compiled.messages[0]["content"] == "Hello Alice, role is {{role}}"


def test_strict_unknown_variable():
    p = _make_prompt("Hello {{name}}", ["name"])
    with pytest.raises(ValueError, match="Unknown variables"):
        p.compile(strict=True, name="Alice", typo="oops")


def test_strict_missing_variable():
    p = _make_prompt("Hello {{name}}", ["name"])
    with pytest.raises(ValueError, match="Missing variables"):
        p.compile(strict=True)


def test_no_variables():
    p = _make_prompt("No variables here")
    compiled = p.compile()
    assert compiled.messages[0]["content"] == "No variables here"
