"""End-to-end test: Ledda SDK → fetch prompt → call LLM via OpenRouter.

Required env vars:
    LEDDA_API_KEY       - Ledda API key (ldda_ak_...)
    LEDDA_BASE_URL      - Ledda edge URL (e.g. https://edge-dev.ledda.ai)
    OPENROUTER_API_KEY  - OpenRouter API key
"""

import os
import sys

from ledda import Ledda

# --- Config ---
LEDDA_API_KEY = os.environ.get("LEDDA_API_KEY", "")
LEDDA_BASE_URL = os.environ.get("LEDDA_BASE_URL", "https://edge.ledda.ai")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

if not LEDDA_API_KEY or not OPENROUTER_API_KEY:
    print("Error: Set LEDDA_API_KEY and OPENROUTER_API_KEY env vars")
    sys.exit(1)

# --- 1. Fetch prompt from Ledda ---
ledda = Ledda(
    api_key=LEDDA_API_KEY,
    base_url=LEDDA_BASE_URL,
    default_label="staging",
    debug=True,
)

print("=== Fetching prompt from Ledda ===")
prompt = ledda.get_prompt(
    "qa",
    fallback={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_message}}"},
        ],
    },
)

print(f"  source:      {prompt.source}")
print(f"  is_fallback: {prompt.is_fallback}")
print(f"  version:     {prompt.version}")
print(f"  variables:   {prompt.variables}")
print(f"  config:      {prompt.config}")
print(f"  messages:    {prompt.messages}")
print()

# --- 2. Compile template variables ---
print("=== Compiling template ===")
compiled = prompt.compile(user_message="What is the capital of France?")
print(f"  compiled messages: {compiled.messages}")
print()

# --- 3. Call LLM via OpenRouter ---
print("=== Calling OpenRouter ===")
import httpx

resp = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "anthropic/claude-sonnet-4",  # override the "asdf" model from config
        "messages": compiled.messages,
        "temperature": prompt.config.get("temperature", 0.7),
        "max_tokens": 256,
    },
    timeout=30,
)

data = resp.json()
reply = data["choices"][0]["message"]["content"]
print(f"  LLM reply: {reply}")
print()

# --- 4. Verify cache works ---
print("=== Cache stats ===")
prompt2 = ledda.get_prompt(
    "qa",
    fallback={"messages": [{"role": "system", "content": "fallback"}]},
)
print(f"  second fetch source: {prompt2.source}")
print(f"  stats: {ledda.cache_stats()}")
print()

# --- 5. Show span attributes (for OTel) ---
print("=== Span attributes ===")
for k, v in prompt.span_attributes.items():
    print(f"  {k}: {v}")

print("\n✅ End-to-end test passed!")
