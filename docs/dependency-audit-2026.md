# Dependency Audit — May 2026

Baseline check before Sprint 2. Records what's installed, what's current on PyPI, and what needed fixing.

## Core Dependencies (stable, no action needed)

| Package | Installed | Latest | Breaking? |
|---------|-----------|--------|-----------|
| click | 8.4.1 | 8.4.x | No |
| rich | 15.0.0 | 15.x | No |
| pytest | 9.0.3 | 9.x | No |
| pytest-asyncio | 1.3.0 | 1.3.x | No |
| pytest-cov | 7.1.0 | 7.x | No |
| ruff | 0.15.14 | 0.15.x | No |
| pyright | 1.1.409 | 1.1.x | No |
| build | 1.5.0 | 1.x | No |

## Optional Dependencies

| Package | Was installed | Installed for audit | Latest | Breaking? |
|---------|--------------|--------------------:|--------|-----------|
| langchain-core | No | 1.4.0 | 1.4.x | See below |
| langgraph | No | 1.2.2 | 1.2.x | See below |
| anthropic | No | 0.104.1 | 0.104.x | No |
| openai | No | 2.38.0 | 2.38.x | Minor (v2 is stable) |
| openai-agents | No | Not installed | 0.2.x | Pre-1.0, weekly breaking |

## LangChain / LangGraph (1.0 migration)

LangChain 1.0 shipped Oct 2025. `langchain-core` is now at 1.4.0.

**Verified compatible:**
- `from langchain_core.callbacks import BaseCallbackHandler` — works.
- `from langchain_core.outputs import LLMResult` — works.
- `on_chat_model_start`, `on_llm_end`, `on_tool_start`, `on_tool_end` — all present with compatible signatures.
- `LLMResult` still has `generations` and `llm_output` fields.
- LangGraph `CompiledStateGraph` still has `ainvoke`, `invoke`, and `.nodes`.
- Callback config dict `{"callbacks": [handler]}` still works.

**No code changes needed** for the LangGraph collector.

## Anthropic SDK (v0.104.1)

- `AsyncAnthropic` exists, `messages.create(model=..., max_tokens=..., messages=[...])` interface unchanged.
- Response still has `response.content[0].text`.
- **Model IDs changed**: dateless format starting with 4.6 generation (`claude-sonnet-4-6` not `claude-sonnet-4-6-20250514`).
- Old models retired: `claude-3-5-sonnet-20241022` (Jan 2026), `claude-3-5-haiku-20241022` (Feb 2026).
- `claude-opus-4-20250514` and `claude-sonnet-4-20250514` retiring June 15, 2026.

**Action**: Update default model in generator.py, update pricing table.

## OpenAI SDK (v2.38.0)

- `AsyncOpenAI` exists, `chat.completions.create(model=..., messages=[...])` interface unchanged.
- Response still has `response.choices[0].message.content`.
- GPT-4o and GPT-4o-mini still available (no deprecation date announced).
- New models: GPT-4.1 family, GPT-5.4 family, GPT-5.5, o3, o4-mini.
- Deprecated: o1, o1-mini, gpt-4-turbo.

**Action**: Update pricing table with new models.

## Pricing Table

Completely refreshed with web-verified pricing (May 2026). Changes from prior version:
- Removed: `gemini-2.0-flash` (deprecated, shutting down June 1, 2026), `gemini-2.5-flash-lite`, `mistral-medium-3.5`, `mistral-small-3.1`.
- Added: `llama-4-scout` ($0.10/$0.40), `deepseek-chat` ($0.14/$0.28), `deepseek-reasoner` ($0.55/$2.19).
- Renamed: `mistral-medium-3.5` → `mistral-large-latest` ($2.00/$6.00), `mistral-small-3.1` → `mistral-small-latest` ($0.10/$0.30).
- Tier changes: `o3` moved from mid → frontier, `o4-mini` moved from fast → mid, `gemini-2.5-flash` moved from mid → fast.
- Added aliases: `mistral-large`, `mistral-small`, `deepseek`.
- Provider detection in generator.py updated to recognize `o4-*` model prefixes.

## pyproject.toml

Updated version bounds for optional deps:
- `langgraph>=1.0` (was `>=0.2`)
- `langchain-core>=1.0` (was `>=0.3`)
- `openai-agents>=0.2` (was `>=0.0.1`)
