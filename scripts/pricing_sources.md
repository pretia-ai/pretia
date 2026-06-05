# Pricing Table Sources

Per-million-token pricing in `agentcost/pricing/tables.py` is sourced from
the vendor pages below. Verify these URLs before each release — providers
change pricing without notice.

| Provider | Source URL | Last Verified |
|----------|-----------|---------------|
| Anthropic | https://docs.anthropic.com/en/docs/about-claude/pricing | 2026-05-30 |
| OpenAI | https://openai.com/api/pricing/ | 2026-05-30 |
| Google (Gemini) | https://ai.google.dev/gemini-api/docs/pricing | 2026-05-30 |
| Qwen (Alibaba) | https://help.aliyun.com/zh/model-studio/billing | 2026-05-30 |
| DeepSeek | https://api-docs.deepseek.com/quick_start/pricing | 2026-05-30 |
| Meta Llama (Together AI) | https://www.together.ai/pricing | 2026-05-30 |
| Mistral | https://mistral.ai/pricing | 2026-05-30 |

## Validation procedure

1. Run `python scripts/validate_pricing.py` (requires API keys).
2. The script sends one small request per backtesting model and computes cost via `calculate_cost()`.
3. Manually compare the computed cost against each provider's billing dashboard.
4. If a price is wrong, update `MODEL_PRICING` in `agentcost/pricing/tables.py` and re-run the structural invariant tests.

## Notes

- DeepSeek offers dramatically cheaper cache-hit pricing ($0.0028/MTok for V4 Flash). The table stores cache-miss rates for conservative projections.
- Qwen DashScope charges more for requests > 200K input tokens. The table uses the standard tier.
- Google Gemini pricing varies by context length tier. The table uses the ≤200K tier.
