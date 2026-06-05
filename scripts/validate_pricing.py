#!/usr/bin/env python3
"""Send one request per backtesting-relevant model and report computed costs.

Run manually (not via pytest) — requires API keys and makes real network calls.
Total cost: ~$0.01 across all providers.

Usage:
    python scripts/validate_pricing.py

Environment variables (set the ones for providers you want to validate):
    ANTHROPIC_API_KEY    — Anthropic (Claude models)
    OPENAI_API_KEY       — OpenAI (GPT models)
    DASHSCOPE_API_KEY    — Qwen / Alibaba Cloud
    DEEPSEEK_API_KEY     — DeepSeek
    GOOGLE_API_KEY       — Google (Gemini models)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcost.pricing.tables import MODEL_PRICING, calculate_cost

_PER_MILLION = 1_000_000

PROMPT = "What is 2+2? Reply with exactly one word."

# Models used in backtesting workflows, mapped to (provider, api_model_string).
BACKTESTING_MODELS: list[tuple[str, str, str]] = [
    # (pricing_table_name, provider, api_model_string)
    ("claude-haiku-4-5", "anthropic", "claude-haiku-4-5-20241022"),
    ("claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6-20250514"),
    ("claude-opus-4-7", "anthropic", "claude-opus-4-7-20250514"),
    ("gpt-4.1-nano", "openai", "gpt-4.1-nano"),
    ("gpt-4.1", "openai", "gpt-4.1"),
    ("gemini-2.5-flash", "google", "gemini-2.5-flash"),
    ("qwen-turbo", "qwen", "qwen-turbo"),
    ("qwen3.6-plus", "qwen", "qwen3.6-plus"),
    ("deepseek-v4-flash", "deepseek", "deepseek-chat"),
]

ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _send_anthropic(api_model: str) -> dict | None:
    try:
        import anthropic
    except ImportError:
        print("  SKIP: anthropic SDK not installed")
        return None
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=api_model,
        max_tokens=10,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "raw_usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


def _send_openai(api_model: str) -> dict | None:
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP: openai SDK not installed")
        return None
    client = OpenAI()
    resp = client.chat.completions.create(
        model=api_model,
        max_tokens=10,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "raw_usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }


def _send_deepseek(api_model: str) -> dict | None:
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP: openai SDK not installed (needed for DeepSeek)")
        return None
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=api_model,
        max_tokens=10,
        messages=[{"role": "user", "content": PROMPT}],
    )
    raw = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }
    if hasattr(resp.usage, "prompt_cache_hit_tokens"):
        raw["prompt_cache_hit_tokens"] = resp.usage.prompt_cache_hit_tokens
    if hasattr(resp.usage, "prompt_cache_miss_tokens"):
        raw["prompt_cache_miss_tokens"] = resp.usage.prompt_cache_miss_tokens
    return {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "raw_usage": raw,
    }


def _send_qwen(api_model: str) -> dict | None:
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP: openai SDK not installed (needed for Qwen)")
        return None
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=api_model,
        max_tokens=10,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "raw_usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }


def _send_google(api_model: str) -> dict | None:
    try:
        import google.generativeai as genai
    except ImportError:
        print("  SKIP: google-generativeai SDK not installed")
        return None
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(api_model)
    resp = model.generate_content(PROMPT, generation_config={"max_output_tokens": 10})
    um = resp.usage_metadata
    return {
        "input_tokens": um.prompt_token_count,
        "output_tokens": um.candidates_token_count,
        "raw_usage": {
            "prompt_token_count": um.prompt_token_count,
            "candidates_token_count": um.candidates_token_count,
            "total_token_count": um.total_token_count,
        },
    }


_SENDERS = {
    "anthropic": _send_anthropic,
    "openai": _send_openai,
    "deepseek": _send_deepseek,
    "qwen": _send_qwen,
    "google": _send_google,
}


def main() -> None:
    timestamp = datetime.now(UTC).isoformat()
    results: list[dict] = []
    skipped_providers: set[str] = set()
    validated_count = 0

    print(f"Pricing Validation — {timestamp}")
    print("=" * 60)
    print()

    for table_name, provider, api_model in BACKTESTING_MODELS:
        env_key = ENV_KEYS[provider]
        if not os.environ.get(env_key):
            if provider not in skipped_providers:
                print(f"SKIP: {provider} — {env_key} not set")
                skipped_providers.add(provider)
            continue

        print(f"  {table_name} ({api_model})...", end=" ", flush=True)
        try:
            sender = _SENDERS[provider]
            usage = sender(api_model)
            if usage is None:
                continue

            inp = usage["input_tokens"]
            out = usage["output_tokens"]
            cost = calculate_cost(table_name, inp, out)
            per_m = MODEL_PRICING[table_name]

            print(f"OK — {inp} in / {out} out → ${cost:.6f}")
            results.append(
                {
                    "model": table_name,
                    "api_model": api_model,
                    "provider": provider,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "computed_cost": cost,
                    "input_price_per_mtok": per_m[0],
                    "output_price_per_mtok": per_m[1],
                    "raw_usage": usage["raw_usage"],
                }
            )
            validated_count += 1
        except Exception as e:
            print(f"ERROR — {e}")
            results.append(
                {
                    "model": table_name,
                    "api_model": api_model,
                    "provider": provider,
                    "error": str(e),
                }
            )

    # --- Write report ---
    report_path = Path(__file__).parent / "pricing_validation_report.md"
    lines = [
        "# Pricing Table Validation Report",
        "",
        f"Date: {timestamp}",
        "",
        ("| Model | Input Toks | Output Toks | Cost | In $/MTok | Out $/MTok | Status |"),
        ("|-------|-----------|------------|------|----------|-----------|--------|"),
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model']} | — | — | — | — | — | ❌ ERROR: {r['error'][:40]} |")
        else:
            lines.append(
                f"| {r['model']} | {r['input_tokens']} | {r['output_tokens']} "
                f"| ${r['computed_cost']:.6f} | ${r['input_price_per_mtok']:.2f} "
                f"| ${r['output_price_per_mtok']:.2f} | ⏳ MANUAL CHECK |"
            )

    lines.extend(
        [
            "",
            "## Manual Verification Steps",
            "",
            "For each model above:",
            "1. Log into the provider's billing dashboard",
            f"2. Find the charge for the request sent at {timestamp}",
            '3. Compare the dashboard charge to the "Computed Cost" column',
            "4. If they match (within rounding tolerance): mark ✅",
            "5. If they differ: the pricing table has an error — update tables.py",
            "",
            "## Notes",
            "- All prices are per million tokens (MTok)",
            "- Computed cost = (input_toks / 1M × in_price) + (output_toks / 1M × out_price)",
            "- Small requests (~25 tokens) may show $0.00 on billing dashboards due to rounding",
            "- If dashboard shows $0.00, verify the per-token rate instead of the total charge",
            "",
            "## Raw Usage Data",
            "",
            "```json",
            json.dumps([r for r in results if "error" not in r], indent=2),
            "```",
        ]
    )

    report_path.write_text("\n".join(lines))

    # --- Summary ---
    providers_validated = len({r["provider"] for r in results if "error" not in r})
    total_cost = sum(r.get("computed_cost", 0) for r in results)

    print()
    print(f"Validated: {validated_count} models across {providers_validated} providers")
    if skipped_providers:
        skips = ", ".join(f"{p} — {ENV_KEYS[p]} not set" for p in sorted(skipped_providers))
        print(f"Skipped: {len(skipped_providers)} provider(s) ({skips})")
    print(f"Total API cost: ~${total_cost:.4f}")
    print()
    print(f"Report saved to: {report_path}")
    print("ACTION REQUIRED: Manually verify computed costs against billing dashboards.")


if __name__ == "__main__":
    main()
