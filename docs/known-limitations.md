# Known Limitations

Pretia profiles single-turn workflow executions and projects costs at scale. These are the gaps between what it measures and what production costs actually look like.

## 1. Session Context Accumulation

Pretia profiles each run as a single turn. In multi-turn conversational workflows, each subsequent turn appends to the context window, so turn 5 costs more than turn 1 — often 3–5× more for the same response quality. The projection reflects first-turn costs only.

**Mitigation:** For multi-turn workflows, profile with representative mid-session context (e.g., prepend 3–4 turns of history to each input). Alternatively, multiply the projected cost by your expected average session depth.

## 2. Input Distribution Mismatch

Projections assume production inputs match the statistical profile of your profiling inputs. If you profile with short, clean test prompts but production receives long, messy user messages, the projection underestimates. A 3× difference in average input length translates to roughly 2–3× difference in input token costs.

**Mitigation:** Check the input distribution stats in the profiling output (p50/p95 input tokens). Compare against your production traffic's input length distribution. If they differ significantly, re-profile with production-representative inputs or use Langfuse import (`pretia analyze --from-langfuse`).

## 3. Tool Response Size Mismatch

When profiling against test stubs or small databases, tool responses are shorter than production. These responses become context for subsequent LLM steps, so understated tool output compounds through the workflow. A retrieval step returning 500 tokens in testing but 5,000 tokens in production affects every downstream step.

**Mitigation:** Profile against production-representative data sources. If using test stubs, ensure they return responses of similar length to production.

## 4. Provider-Side Caching

DeepSeek's server-side prompt cache prices cache-hit input tokens at 2% of the standard rate — a 50× discount. When profiling runs execute rapidly with the same system prompt, most runs hit cache. The profiled costs reflect cached pricing, not cold-start pricing. Production requests with unique prompts pay full price.

**Mitigation:** Pretia busts the cache by default during profiling (`--cache-mode cold` is the default). If your production workload genuinely benefits from warm caches (stable system prompts, high request volume), use `--allow-cache` to profile with caching enabled.

## 5. Tiered Pricing Boundaries

v1 uses the standard pricing tier for all providers. Several providers charge differently for very large requests: Gemini charges more above 200K input tokens, DashScope charges more above 200K tokens. If your workflow regularly exceeds these boundaries, the projection underestimates by the tier differential (typically 2–4×).

**Mitigation:** Check your p95 input token count in the profiling output. If it approaches a provider's tier boundary, verify the pricing on the provider's pricing page and apply a manual correction for the affected runs.

## 6. Model Drift

Providers update models within version lines without changing the model identifier. These silent updates can shift token counts and response patterns by 5–25%. A model that previously answered in 200 tokens might start answering in 300 after an update. The pricing table and the profiled cost distribution both reflect the model at profiling time.

**Mitigation:** Re-profile periodically. Pretia warns when a profile is more than 30 days old and degrades the confidence tier at 90+ days. Monitor production costs against projections to detect drift early.

## 7. Batch vs. Real-Time Pricing

v1 uses real-time API pricing. Batch APIs (OpenAI Batch API, Anthropic Message Batches) typically cost ~50% less for the same model and tokens. If you deploy via batch processing, the projection overestimates by roughly 2×.

**Mitigation:** If deploying via batch, apply a 0.5× multiplier to projected costs. Pretia does not yet support batch pricing natively — this is planned for a future release.
