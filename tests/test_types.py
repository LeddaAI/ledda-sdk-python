from ledda._types import Fallback, Prompt


FB: Fallback = {"messages": [{"role": "system", "content": "You are helpful."}]}


def test_from_fallback():
    p = Prompt._from_fallback(FB, name="test", label="production")
    assert p.is_fallback is True
    assert p.source == "fallback"
    assert p.version == 0
    assert p.messages == [{"role": "system", "content": "You are helpful."}]


def test_from_fallback_with_config():
    fb: Fallback = {
        "messages": [{"role": "system", "content": "Hello"}],
        "config": {"model": "gpt-4"},
    }
    p = Prompt._from_fallback(fb, name="test", label="production")
    assert p.config == {"model": "gpt-4"}


def test_from_fallback_multi_message():
    fb: Fallback = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "{{question}}"},
        ],
    }
    p = Prompt._from_fallback(fb, name="test", label="production")
    assert len(p.messages) == 2
    assert p.messages[1]["role"] == "user"


def test_from_api_response():
    data = {
        "name": "welcome",
        "label": "production",
        "version": 5,
        "messages": [{"role": "system", "content": "Hello"}],
        "config": {"model": "claude-sonnet-4-20250514"},
        "variables": ["name"],
        "ruleId": "abc-123",
        "ruleName": "enterprise",
    }
    p = Prompt._from_api_response(data)
    assert p.version == 5
    assert p.rule_id == "abc-123"
    assert p.is_fallback is False
    assert p.source == "api"


def test_span_attributes():
    p = Prompt(
        messages=[],
        config={},
        version=3,
        name="test",
        label="staging",
        variables=[],
        rule_id="rule-1",
        rule_name="beta",
        is_fallback=False,
        source="cache",
    )
    attrs = p.span_attributes
    assert attrs["ledda.prompt.name"] == "test"
    assert attrs["ledda.prompt.version"] == 3
    assert attrs["ledda.prompt.label"] == "staging"
    assert attrs["ledda.prompt.rule_id"] == "rule-1"
    assert attrs["ledda.prompt.source"] == "cache"


def test_span_attributes_no_rule():
    p = Prompt._from_fallback(FB, name="x", label="prod")
    attrs = p.span_attributes
    assert "ledda.prompt.rule_id" not in attrs
