# ledda

Python SDK for [Ledda](https://ledda.ai) — resilient prompt management for LLM applications.

## Install

```bash
pip install ledda
```

## Quick Start

```python
from ledda import Ledda

ledda = Ledda(api_key="ldda_ak_...")

prompt = ledda.get_prompt("welcome",
    fallback={
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    },
)

# Use the prompt with your LLM
response = openai.chat.completions.create(
    messages=prompt.messages,
    **prompt.config,
)
```

## Features

- **Never stalls your app** — 3-second timeout, automatic fallback on any failure
- **Built-in caching** — bounded LRU cache with configurable TTL, stale-while-revalidate
- **Sync + Async** — `Ledda` for sync code, `AsyncLedda` for FastAPI/async frameworks
- **Template variables** — `{{var}}` placeholders with `.compile(var="value")`
- **Strict mode** — raises exceptions in dev/test to catch misconfigurations early
- **OpenTelemetry** — optional span attribute injection for trace linking

## Usage

### Basic

```python
from ledda import Ledda

# The SDK never raises in default mode — always returns usable content
ledda = Ledda(api_key="ldda_ak_...")

prompt = ledda.get_prompt("welcome",
    fallback={                                     # used when API is unreachable
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    },
    label="production",                            # environment (default: "production")
    attributes={"tenant_id": "acme"},              # for routing rules
    routing_key="user-12345",                      # for A/B test bucketing
)

print(prompt.messages)    # [{"role": "system", "content": "..."}]
print(prompt.config)      # {"model": "claude-sonnet-4-20250514", "temperature": 0.7}
print(prompt.version)     # 5
print(prompt.is_fallback) # False
print(prompt.source)      # "api" | "cache" | "fallback"
```

The fallback must be a dict with a `messages` list — the same shape as a prompt. This ensures your code works identically whether using a live prompt or a fallback.

```python
# Multi-message fallback with config
prompt = ledda.get_prompt("welcome",
    fallback={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{question}}"},
        ],
        "config": {"model": "claude-sonnet-4-20250514", "temperature": 0.7},
    },
)
```

### Template Variables

```python
prompt = ledda.get_prompt("welcome",
    fallback={
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    },
)

compiled = prompt.compile(
    persona="a friendly onboarding assistant",
    task="account setup",
)
# compiled.messages → [{"role": "system", "content": "You are a friendly onboarding assistant..."}]
```

### Async (FastAPI)

```python
from ledda import AsyncLedda

ledda = AsyncLedda(api_key="ldda_ak_...")

@app.post("/chat")
async def chat():
    prompt = await ledda.get_prompt("welcome", fallback={
        "messages": [{"role": "system", "content": "You are helpful."}],
    })
    # ...
```

### Cache Warming

```python
# Pre-fetch on startup to avoid cold-start latency
ledda.prefetch(["welcome", "summarizer", "classifier"])
```

### Strict Mode (Development)

```python
from ledda import Ledda, PromptNotFoundError

ledda = Ledda(api_key="ldda_ak_...", strict_mode=True)

try:
    prompt = ledda.get_prompt("typo-name", fallback={
        "messages": [{"role": "system", "content": "fallback"}],
    })
except PromptNotFoundError:
    print("Prompt doesn't exist — fix the name!")
```

### CI/CD Validation

```python
assert ledda.validate_prompt("welcome", label="production")
```

### OpenTelemetry Integration

```python
# Auto-inject span attributes
ledda = Ledda(api_key="ldda_ak_...", enable_otel_injection=True)

# Or manually
prompt = ledda.get_prompt("welcome", fallback={
    "messages": [{"role": "system", "content": "fallback"}],
})
span.set_attributes(prompt.span_attributes)
```

### Context Managers

```python
# Sync
with Ledda(api_key="ldda_ak_...") as ledda:
    prompt = ledda.get_prompt("welcome", fallback={
        "messages": [{"role": "system", "content": "fallback"}],
    })

# Async
async with AsyncLedda(api_key="ldda_ak_...") as ledda:
    prompt = await ledda.get_prompt("welcome", fallback={
        "messages": [{"role": "system", "content": "fallback"}],
    })
```

## Resilience

The SDK is designed so that Ledda downtime never becomes your downtime:

| Scenario | Behavior |
|----------|----------|
| API reachable | Returns prompt, caches it |
| API down, cache warm | Returns cached prompt (even if stale) |
| API down, cache cold | Returns your fallback |
| Network timeout (3s) | Returns cached or fallback |

In default mode, `get_prompt()` **never raises an exception**. Every failure path returns usable content.

## Configuration

```python
ledda = Ledda(
    api_key="ldda_ak_...",
    base_url="https://edge.ledda.ai",  # default
    cache_ttl=60,                       # seconds, default 60
    cache_max_size=1000,                # max cached prompts, default 1000
    default_label="production",         # default environment
    enable_otel_injection=False,        # auto-inject OTel span attributes
    strict_mode=False,                  # raise on errors (for dev/test)
    debug=False,                        # verbose logging
)
```

## License

MIT
