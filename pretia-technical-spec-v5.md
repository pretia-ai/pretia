# Pretia — Technical Specification v2
## "Ship slower, ship right, ship with intelligence"
## May 2026

---

## THE PROBLEM

AI agents are deployed cost-blind. Every team building agents today follows the same pattern: prototype a workflow, ship it to production, and discover what it costs weeks later when the invoice arrives. There is no "build step" that checks cost, no CI gate that flags regressions, no projection that says "this will cost $11,700/month at your current traffic" before you merge the PR.

This isn't a minor inconvenience. The numbers are severe:

- **92% of agent deployments exceed their cost budgets** (IDC 2026). Not by a small margin — the median overrun is 2-4x.
- **40%+ of enterprise agent pilots will be canceled by 2027**, with cost cited as the #1 reason (Gartner).
- **Agents consume 5-30x more tokens per task than chatbots** (Gartner, March 2026). A simple support agent processes 3,000-5,000 tokens per ticket. An agentic workflow with retrieval, review loops, and tool calls processes 50,000-200,000.
- **60% of enterprise AI builders create workflows without IT oversight** (Retool 2026). Nobody is reviewing the cost implications.
- **Context growth is the silent killer.** In looping agents, every iteration re-sends the entire conversation plus accumulated tool outputs. By iteration 8, the context might be 10x what it was at iteration 1 — and cost scales linearly with context size. Most teams don't realize this until the bill arrives.

The root cause is simple: **there is no equivalent of a Lighthouse score, a test suite, or a cost estimate for agent workflows.** Every website has a performance budget. Every code change has a test suite. Every cloud deployment has a cost projection. Agent workflows have nothing.

### What exists today and why it's not enough

**Observability tools (Langfuse, Braintrust, LangSmith)** show you what happened AFTER your agent ran in production. They're the equivalent of looking at your credit card statement — useful for understanding the past, useless for predicting the future. They answer "what did we spend?" not "what WILL we spend?"

**FinOps tools (CloudZero, Finout)** attribute cloud bills to teams and products. They see the aggregated OpenAI invoice — "$14K this month" — but can't tell you that Step 4 of your support agent is responsible for 68% of it because it re-reads the full context on every review iteration. They operate at the billing layer, not the workflow layer.

**Gateways (LiteLLM, Cloudflare AI Gateway)** enforce per-key budget limits. When the budget is hit, the agent stops. Hard cutoff. No degradation. No warning before it happens. And no projection of what the budget SHOULD be.

**Nobody** answers the pre-deployment question: "If I merge this PR and this agent handles 1,000 requests per day, what will it cost me per month, and which steps are the biggest cost drivers?"

---

## THE SOLUTION

Pretia is the pre-deployment cost intelligence layer for AI agent workflows. It answers two questions before you deploy:

1. **"How much will this cost at scale?"** — Profile your workflow on sample inputs, project costs at 10x/100x/1000x traffic with distributional statistics (not just averages), and detect cost time-bombs like context growth and stuck loops.

2. **"How do I make it cheaper?"** — Get specific, actionable recommendations: downshift this model (estimated savings: $4,200/mo), add context compaction at this step ($7,800/mo), cap these review iterations ($2,100/mo).

The delivery mechanism is a GitHub Action that comments on every PR touching an agent workflow:

```
⚠ Pretia Report — support-agent workflow

Projected monthly cost increased 388%: $2,400/mo → $11,700/mo at current traffic

Step                Before    After     Change    Flag
1. Classify intent  $180      $180      —
2. Retrieve context $320      $340      +6%
3. Generate draft   $890      $920      +3%
4. Review + iterate $710      $9,200    +1,196%   🔴
5. Format response  $300      $310      +3%

💡 Recommendations:
  MODEL SWAP: Step 4 uses Opus for review. Sonnet scores 97% equivalent
             on this task type (confidence: 0.92). Savings: −$4,200/mo
  ARCHITECTURE: Step 4 re-reads full context every iteration (+1,200 tok/turn).
             Add compaction. Savings: −$7,800/mo
  WORKFLOW: Review loops avg 8.2 iterations. Cap at 3 (quality delta <2%).
             Savings: −$2,100/mo

Total potential savings: −$14,100/mo ($11,700 → $3,200)
```

### What we build

- **An open-source Python SDK** (`pip install pretia`) that instruments LangGraph, OpenAI Agents SDK, Qwen-Agent, and CrewAI workflows. DeepSeek models work through existing collectors via their OpenAI-compatible API. Point it at your agent — it auto-generates diverse test inputs from your system prompt (~$0.02), profiles the workflow, and projects cost distributions. Two commands from install to first report:
  ```bash
  pip install pretia
  pretia profile run my_agent.py
  # Auto-generates 20 inputs → runs workflow → opens HTML report in browser
  # Cost: ~$2. Time: 3 minutes. No JSONL files, no configuration.
  ```
- **A local web UI** (`pretia ui`) that gives a visual experience for the full journey: select your workflow → configure input generation → watch profiling progress live → explore the interactive report with recommendations. Same pipeline as the CLI, visual interface. Runs on `localhost`, no cloud, no auth.
- **An HTML report** generated after every profiling run — visual cost breakdown per step, recommendations with dollar savings, context growth sparklines, overall score. Opens in the browser automatically. Shareable, screenshottable, embeddable in README.
- **Langfuse/Braintrust trace import** for teams with existing observability — analyze production traces for cost patterns without re-executing anything. Zero cost, zero friction.
- **A GitHub Action** that runs cost checks on every PR in four modes: static analysis (free), auto-generated inputs (~$1-3), user-curated samples (~$2-5), or Langfuse trace import (free).
- **A recommendation engine** that starts with heuristics (v1) and evolves into an ML-powered classifier trained on 800K+ public examples + proprietary user feedback (v1.5+).
- **A cost prediction model** (v2) that estimates costs from workflow structure alone — no profiling needed — trained on accumulated user data.
- **A benchmarking engine** (v3) that tells you "your support agent costs 2.3x the median for similar workflows" — powered by cross-company clustering.

### What we don't build

No proxy (use LiteLLM). No routing (use Martian). No tracing (use Langfuse). No compaction (use Morph). No quality evals (use Braintrust). Pretia sits ABOVE the entire stack. It consumes these tools' data, it doesn't replace them.

### Why this, why now, why a solo founder

**Why this:** The pre-deployment cost simulation gap is the only space in the LLM tooling ecosystem that is simultaneously (a) not occupied by any player at $100M+ funding, (b) validated by adjacent tools proving the market exists (Braintrust → CI gates, CloudZero → cost attribution, Langfuse → cost visibility), and (c) buildable as an SDK by a solo developer.

**Why now:** Q1-Q2 2026 is the moment of "agent bill shock." Enterprises launched agents in late 2025, and the first production bills are arriving now. The EU AI Act enforcement starts August 2, 2026 — 10 weeks away — requiring audit trails for high-risk AI systems. The pain is acute and the timing is compressed.

**Why solo:** This is an SDK, not a platform. Distribution is via PyPI/npm/GitHub, not enterprise sales. Langfuse reached 26K GitHub stars starting with 2 people. The Scope founder (YC, $800K raise) is solo. The v1 is 100% buildable by one person in 12 weeks, and the ML layer is scikit-learn + XGBoost on a laptop, not a GPU cluster.

---

## TIMELINE OVERVIEW

| Phase | Weeks | Focus | ML Component |
|-------|-------|-------|-------------|
| **v1.0** | 1-20 | Core SDK (collectors for LangGraph, OAI Agents, Qwen-Agent; pricing for 7 providers) + jackknife+ conformal intervals + CLT-corrected Monte Carlo + 5 pattern detectors + BCa bootstrap + CVaR + stratified analysis + three-layer validation (500+ synthetic distributions + SWE-bench + 14 model-optimized real workflows incl. PDF RAG, long-context, multi-turn sessions) + recommendations + verify loop + CI gate + HTML report + graph view + local web UI | None (stats + heuristics) |
| **v1.5** | 21-24 | Task Complexity Classifier | Logistic regression on RouterBench |
| **v2.0** | 25-34 | Live monitoring + Cost Prediction Model | XGBoost on profiling data |
| **v3.0** | 35-48 | Spend Governance + Workflow Benchmarking | HDBSCAN clustering |

The v1 ships in 20 weeks. The extra time (vs. a CLI-only 12-week version) buys the backtesting suite ($950 to validate predictions on 14 real-world archetypes), implementation prompts with verify loop, graph visualization, HTML report, and local web UI. Every one of these additions addresses a specific risk: backtesting prevents wrong predictions (product-killing), the verify loop creates stickiness, the graph view makes the product demo-ready. ML is layered on top starting at v1.5. This is deliberate: the v1 must stand on its own merits. ML amplifies the moat, it doesn't create it.

---

## PART 1 — CORE ARCHITECTURE (v1.0, weeks 1-20)

### 1.1 Instrumentation Layer

All three major agent frameworks have native token tracking. They expose it differently.

**LangGraph** — LangChain Callback pattern:
```python
callback = UsageMetadataCallbackHandler()
config = {"callbacks": [callback]}
for chunk in graph.astream(inputs, config=config):
    ...
# callback exposes tokens via on_llm_end → response.usage_metadata
# Granularity: per LLM call + per graph node
# Langfuse CallbackHandler captures everything automatically
```

**OpenAI Agents SDK** — Native usage tracking + RunHooks:
```python
result = await Runner.run(agent, "input")
usage = result.context_wrapper.usage
# usage.input_tokens, output_tokens, total_tokens
# PER-REQUEST breakdown:
for req in usage.request_usage_entries:
    print(f"{req.input_tokens} in, {req.output_tokens} out")

# RunHooks for lifecycle events:
class CostHooks(RunHooks):
    async def on_agent_end(self, ctx, agent, output):
        u = ctx.usage
        print(f"{agent.name}: {u.total_tokens} tokens")
# Also: on_tool_start/end, on_handoff
# Langfuse/OpenInference instrumentation supported natively
```

**CrewAI** — LangChain callbacks under the hood (built on LangChain), but exposes agents and tasks as first-class abstractions. Same `CallbackHandler` pattern.

**Qwen-Agent** — Generator-based execution model with NO callback/hooks system. Agents use `Agent.run(messages)` which yields response chunks. Token usage is captured by wrapping the agent's `BaseChatModel.chat()` method with an `_InstrumentedChatModel` proxy. For DashScope-based agents, usage data is extracted from `Message.extra['model_service_info']`.

**DeepSeek** — Uses an OpenAI-compatible API endpoint (`https://api.deepseek.com`). DeepSeek models (V4 Flash, V4 Pro) work through `LangGraphCollector` (via `langchain_openai.ChatOpenAI` with custom `base_url`) and `OpenAIAgentsCollector` (via the `openai` SDK with custom `base_url`) with zero collector changes. Only pricing table and input generator updates were needed.

### 1.2 The Collector Abstraction

```
┌──────────────────────────────────────────────────┐
│                 Pretia SDK                     │
│                                                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ LangGraph│  │  OpenAI   │  │Qwen-Agent│  │   Generic    │  │
│  │ Collector│  │  Agents   │  │ Collector│  │  Collector   │  │
│  │          │  │ Collector │  │          │  │  (manual)    │  │
│  └─────┬────┘  └─────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│        │             │              │               │           │
│        ▼             ▼              ▼               ▼           │
│  ┌────────────────────────────────────────────┐  │
│  │            Unified StepRecord              │  │
│  │                                            │  │
│  │  step_name: str                            │  │
│  │  step_type: llm | tool | retrieval         │  │
│  │  model: str                                │  │
│  │  input_tokens: int                         │  │
│  │  output_tokens: int                        │  │
│  │  context_size: int  ← total prompt tokens  │  │
│  │  tool_definitions_tokens: int              │  │
│  │  system_prompt_hash: str  ← for ML later   │  │
│  │  system_prompt_tokens: int                 │  │
│  │  output_format: str  ← json|text|code      │  │
│  │  is_retry: bool                            │  │
│  │  iteration: int                            │  │
│  │  parent_step: Optional[str]                │  │
│  │  duration_ms: int                          │  │
│  │  timestamp: datetime                       │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │                             │
│                     ▼                             │
│  ┌────────────────────────────────────────────┐  │
│  │       ProfileStore (JSON / SQLite)         │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**QwenAgentCollector** wraps the agent's LLM client (`agent.llm`) with an `_InstrumentedChatModel` proxy. Since Qwen-Agent has no callback hooks, the collector intercepts `BaseChatModel.chat()` calls, capturing token usage from DashScope response metadata and timing from monotonic clocks. Handles streaming (buffers the generator, captures usage from the final chunk) and non-streaming modes. Installed via `pip install pretia[qwen]` (lazy-imported to avoid requiring qwen-agent for core functionality).

**DeepSeek models** require no dedicated collector. They use OpenAI-compatible APIs, so `LangGraphCollector` (via `ChatOpenAI(openai_api_base="https://api.deepseek.com")`) and `OpenAIAgentsCollector` capture tokens normally. The pricing table covers DeepSeek V4 Flash ($0.14/$0.28 per MTok) and V4 Pro ($0.435/$0.87 per MTok), with cache-miss rates used for conservative projections (cache-hit input is 50x cheaper at $0.0028/MTok).

**Key design decision:** StepRecord captures `system_prompt_hash`, `system_prompt_tokens`, and `output_format` from day one. These fields aren't used in v1 (heuristics don't need them), but they're the features the Task Complexity Classifier needs in v1.5. Capturing them now means we don't have to re-instrument later. Cost of carrying extra fields: ~zero.

**GenericCollector for custom pipelines:**
```python
from pretia import GenericCollector

collector = GenericCollector()

@collector.step("classify_intent")
async def classify(input):
    response = await client.messages.create(...)
    return response

# Or context manager:
with collector.step("generate_draft") as step:
    response = await client.messages.create(...)
    step.record(response)  # auto-extracts tokens
```

Trade-off: GenericCollector requires manual step tagging (more friction than LangGraphCollector which discovers steps automatically via callbacks). Both ship in v1. Clearly documented difference.

### 1.3 Input Generation — The Friction Ladder

Asking a developer to curate a 100-line JSONL file is a deal-breaker for adoption. The default experience must be two commands or less. Pretia provides five input modes, ordered from zero friction to maximum precision. The user should never need to start at the bottom.

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT GENERATION MODES                      │
│                                                                │
│  Level 0: estimate     Code analysis only. No execution.       │
│           ──────────   Instant. Free. Widest confidence.       │
│                        $ pretia estimate workflow.py         │
│                                                                │
│  Level 1: single       ONE example input from the user.        │
│           ──────────   1 run + ML priors for variance.         │
│                        $ pretia profile run workflow.py \    │
│                          --input "How do I reset my password?" │
│                                                                │
│  Level 2: auto    ★    LLM generates diverse inputs from       │  ★ DEFAULT
│           ──────────   system prompt + type hints. ~$0.02.     │
│                        $ pretia profile run workflow.py \    │
│                          --auto-generate 20                    │
│                                                                │
│  Level 3: import       Pull real inputs from Langfuse,         │
│           ──────────   Braintrust, or OpenTelemetry traces.    │
│                        $ pretia profile run workflow.py \    │
│                          --from-langfuse --last 100            │
│                                                                │
│  Level 4: manual       User-curated JSONL file.                │
│           ──────────   Maximum precision, maximum friction.    │
│                        $ pretia profile run workflow.py \    │
│                          --inputs samples.jsonl                │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### Level 0 — Static estimate (zero inputs, zero cost, instant)

The v2 Cost Prediction Model parses the workflow code, extracts structural features (steps, models, loops, prompt sizes), and runs the XGBoost predictor. No execution at all. Wide confidence intervals, but a 1-second experience with zero friction. This is the entry point — the user gets a ballpark before committing to anything.

Before the ML model exists (v1), static estimate uses heuristic priors from published token usage data: simple tool-calling agents = 5K-15K tokens/task, multi-agent systems = 200K-1M+, coding agents = 1-3.5M tokens/task (Iternal.ai, Gartner 2026). Combined with the model pricing tables, this gives a rough order-of-magnitude estimate.

```bash
$ pretia estimate workflow.py
# Output:
# ⚡ Static estimate (no execution, wide confidence):
# Workflow: support-agent (5 steps, 1 loop, models: haiku+opus)
# Archetype: support-agent-with-RAG-loop
# Estimated cost per run: $0.15 - $0.80 (p50 - p95)
# At 1,000 runs/day: $4,500 - $24,000/month
# ⚠ Wide range — run `pretia profile` for precise numbers
```

#### Level 1 — Single input (one example, one run)

The user provides ONE representative input as a string. The SDK runs the workflow once, captures the full trace (steps, models, tokens, iterations, context sizes). One real run gives the exact structure and a single cost data point. The projection engine then uses ML priors (from the Cost Prediction Model, or archetype-based priors in v1) to estimate the variance: "based on similar workflows, the p95 is likely 2-3x this p50."

```bash
$ pretia profile run workflow.py --input "How do I reset my password?"
# Executes 1 run → captures trace → projects with priors
# Cost: one agent execution (~$0.05-0.50)
```

#### Level 2 — Auto-generated inputs ★ DEFAULT

The smartest approach and the default experience. The SDK reads the workflow's system prompt + input type hints (Pydantic models, function signatures, docstrings) and calls a cheap LLM (Haiku/mini, ~$0.01-0.02) to generate diverse synthetic inputs.

```python
# Internal: what the auto-generator does
def generate_inputs(workflow, n=20):
    system_prompt = extract_system_prompt(workflow)
    input_schema = extract_input_schema(workflow)  # Pydantic, type hints, or raw

    generation_prompt = f"""
    You are generating test inputs for an AI agent workflow.

    Agent's system prompt (first 500 tokens):
    {system_prompt[:2000]}

    Input format expected:
    {input_schema}

    Generate {n} diverse, representative inputs that cover:
    - Easy cases (simple, short, clear intent)
    - Medium cases (moderate complexity, some ambiguity)
    - Hard cases (long context, edge cases, multi-step reasoning needed)
    - Adversarial cases (confusing input, off-topic, very long)

    Return as a JSON array.
    """

    response = cheap_llm_call(generation_prompt)  # Haiku: ~$0.01
    return parse_json_array(response)
```

The quality of generated inputs depends on how descriptive the system prompt is. Even mediocre synthetic inputs are infinitely better than zero inputs — they'll capture the step structure, model usage, loop behavior, and context growth patterns. The distribution might not perfectly match production, but the patterns are real.

```bash
$ pretia profile run workflow.py --auto-generate 20
# Generates 20 inputs via Haiku → runs workflow on each → full report
# Cost: ~$0.02 (generation) + 20 × ~$0.10 (runs) = ~$2.02
# Time: 2-5 minutes
```

**This is the default.** Running `pretia profile run workflow.py` without flags defaults to `--auto-generate 20`. The first-time user experience is:

```bash
pip install pretia
pretia profile run my_agent.py
```

Two commands. No files. No configuration. ~$2 cost. Full report with recommendations.

#### Level 3 — Import from observability traces

For teams already running Langfuse, Braintrust, or OpenTelemetry in production, the REAL inputs are already captured. Pretia pulls them via the Langfuse Python SDK:

```python
# Internal: what the importer does
from langfuse import get_client

langfuse = get_client()
traces = langfuse.api.observations.get_many(
    fields="core,basic,usage",
    limit=100,
)
inputs = [extract_input(trace) for trace in traces]
```

The user runs:
```bash
$ pretia profile run workflow.py --from-langfuse --last 100
# Pulls 100 most recent production inputs from Langfuse
# Re-runs workflow (or uses Langfuse traces directly for token data)
# Cost: $0 if using traces directly, ~$10 if re-running
```

This is the gold standard for accuracy — you're profiling on actual production traffic. And it's zero-friction for the user because the data already exists.

**Alternative: trace-only mode (no re-execution).** If the user's Langfuse traces already contain per-step token counts (which they do if they use the Langfuse LangChain/OAI Agents integration), Pretia can analyze the traces directly WITHOUT re-running the workflow. Zero cost, zero execution, uses existing data:

```bash
$ pretia analyze --from-langfuse --last 100
# No execution — analyzes existing Langfuse traces
# Computes distributions, detects patterns, generates recommendations
# Cost: $0
```

This is the fastest path to value for teams that already have Langfuse. "Connect your Langfuse, get cost recommendations in 30 seconds."

#### Level 4 — Manual JSONL file

The original approach. User provides a curated file:
```bash
$ pretia profile run workflow.py --inputs samples.jsonl
```

Only makes sense for teams with existing test datasets. Stays as an option but is never the suggested starting point.

#### Input mode selection logic

The CLI auto-suggests the best mode:

```bash
$ pretia profile run workflow.py
# Detected: Langfuse credentials in environment → suggesting --from-langfuse
# Detected: No Langfuse → defaulting to --auto-generate 20
# Use --input "..." for a quick single-run estimate
# Use --inputs file.jsonl if you have a test dataset
```

### 1.4 Projection Engine

**Why linear projection fails:**

1. **Context growth.** In looping agents, context grows per iteration. The average run has 5 iterations, outliers have 15 — and iteration 15 costs 3x iteration 5 due to accumulated context. The mean hides the tails.
2. **Retry amplification.** Under load, rate limits trigger retries. At 100 req/min: 0% retries. At 1000 req/min: ~15%. Linear projection misses this threshold.
3. **Log-normal distribution.** Costs per run follow a log-normal (many cheap runs, few very expensive ones). The mean is pulled up by outliers. p50 and p95 are far more useful.

**Distributional scaling (v1 approach):**

```
Input:  N profiled runs (e.g. 100)
Output: For each target traffic volume (10x, 100x, 1000x):
  - p50, p75, p90, p95, p99 of cost per run
  - Total monthly cost at each percentile
  - Per-step decomposition at each percentile
  - Warnings for detected non-linear patterns
```

**Concretely:**

Step 1 — Collect raw runs, compute per-step stats:
```
Step "classify":  mean=120tok, p50=110, p95=180, model=haiku
Step "retrieve":  mean=2400tok, p50=2100, p95=4800
Step "generate":  mean=8900tok, p50=7200, p95=18000, model=sonnet
Step "review":    mean=12400tok, p50=9000, p95=34000, model=opus
                  ↑ high variance = context growth detected
```

Step 2 — Detect non-linear patterns (five detectors):
- **Context growth (dual correlation):** Run both Pearson and Spearman on context_size vs iteration (minimum 5 data points, p < 0.05 required). Pearson r² > 0.7 → linear growth, use linear+logarithmic model average. Spearman ρ² > 0.7 but Pearson r² < 0.7 → non-linear monotonic growth, fit power-law model (α via log-log regression; sub-linear if α < 1, super-linear if α > 1). The gap between Spearman ρ and Pearson r is diagnostic of non-linearity. Extrapolation capped at observed max iteration.
- **Loop count variance:** if iteration count varies 3-12 (CV > 0.5) → sample iteration count K from distribution, sample K per-iteration costs from flat pool, sum. Composes with context growth when both detected on same step.
- **Token explosion:** if p95/p50 ratio > 3 at n ≥ 30 (or p90/p50 > 4 at n < 30 for stability) → flag as "high variance", give dual projections
- **Step count variance (NEW):** Count active (non-zero-cost) steps per run. CV > 0.3 = WARNING, CV > 0.6 or max > 2× min = DANGER. Catches conditional routing workflows. Triggers whole-run sampling in Monte Carlo (no independent step sampling).
- **Cost bimodality (NEW):** Hartigan's dip test (p < 0.05) or 2-component GMM (BIC delta > 6). When detected, report per-mode statistics: "Mode A (70% of runs): $0.02/run. Mode B (30% of runs): $0.40/run." WARNING severity — Monte Carlo handles it via whole-run resampling.

Step 3 — Project:
- Stable case: `monthly_cost = mean_cost_per_run × daily_volume × 30`
- Non-linear case: CLT-corrected Monte Carlo — sample K run costs (K = min(N, 1000)), compute sample mean μ̂ and variance σ̂², project monthly as N × μ̂ + z × √(N × σ̂²). Fixes a variance inflation bug where the old approach (multiply single sample by N) produced N²×σ² variance instead of N×σ². Non-pattern steps sampled from whole observed runs to preserve inter-step correlation; only flagged steps use pattern-specific models. At n < 30, p95 inflated by (1 + 2/√n) — derived from O(1/√n) convergence rate (Hall 1988), calibrated empirically from synthetic distribution testing (500+ shapes). At n < 20, suppress p95 entirely — report only p50 with range multiplier. ~5 seconds runtime, triggered ONLY when heuristics detect non-linearity. v1.1 upgrades to log-normal KDE smoothed bootstrap with Silverman bandwidth.

**Visibility warnings shipped with every projection:**
- Input distribution stats (token p50/p95/max/CV) — makes "garbage in" assumption visible
- Uniform input flags (all identical iteration counts, or input token CV < 0.1)
- Zero-execution step detection (workflow step never triggered during profiling)
- Stale profile warnings (age > 30 days, step list hash or prompt hash mismatch; >90 days degrades confidence tier)
- Pricing staleness check (>30 days since pricing table update)
- Sample coverage statement ("events occurring less than ~X% of the time may not be represented")

**Not in v1:** rate limit modeling (too many variables), cache hit rate modeling (impossible to estimate offline — but DeepSeek cache-busting is available via `--cache-mode cold`), latency projection (we do cost, not perf).

### 1.5 Recommendation Engine (v1: heuristics only)

Three recommendation types, all rule-based in v1:

**TYPE 1 — MODEL SWAP**

v1 method (heuristics):
1. Classify step by task type via simple rules on system prompt keywords and input/output token ratio:
   - High output/input ratio + creative keywords → generation (needs higher tier)
   - Low output/input ratio + classification keywords → classification (low tier sufficient)
   - Code in output → code generation (test before downshift)
   - JSON schema in prompt → structured extraction (often low tier sufficient)
2. If step uses a model above the estimated minimum tier → recommend downshift
3. Estimate savings: `(current_price - recommended_price) × tokens × volume`

Limitation: cannot guarantee quality is maintained. Recommendation always says: "Consider testing [cheaper model] — estimated savings $X/mo. Validate with an eval before switching."

**TYPE 2 — ARCHITECTURE**

Detect and recommend for:
- **Context growth:** `Δcontext > 500 tokens/iteration` consistently → recommend compaction
- **Re-sent context:** same content hash in prompts of successive steps → recommend caching or pipeline reorganization
- **Oversized tool definitions:** tool_definitions_tokens > 30% of total prompt → recommend tool filtering per step

**TYPE 3 — WORKFLOW**

- **Excessive loops:** cost marginal per iteration + distribution of iteration counts → recommend cap
- **Stuck loop detection:** runs with >2x mean iterations → flag as outliers, compute cost share, recommend circuit breaker

**IMPLEMENTATION PROMPTS — "Copy to Claude Code"**

Every recommendation generates a copy-pasteable prompt tailored for AI coding tools (Claude Code, Codex, Cursor). The prompt is surgical because Pretia has the full context: file path, step name, current model, system prompt content, iteration patterns, context growth data, framework-specific patterns.

**Why this is harder than it looks:** A model swap is a 5-line prompt. But "add context compaction before the review loop" needs to reference LangGraph-specific patterns (state channels, conditional edges, reducer functions), handle the state schema, not break the existing graph topology, and produce code that actually compiles. Each recommendation type has a different complexity tier.

**Tier 1 — Simple (model swap, parameter changes):**

```python
# Model swap — the prompt is almost mechanical
f"""In {file_path}, the {framework} node "{step_name}" currently uses 
model "{current_model}". Change it to "{recommended_model}". 

The step does: {task_description_from_system_prompt_first_200_chars}
Our classifier predicts {recommended_model} achieves {confidence}% 
equivalent quality on this task type.

Update the model parameter. Adjust max_tokens if needed (current: 
{current_max_tokens}). Don't change any other logic.

After making the change, run: pretia profile run {file_path}
to verify the cost reduction."""
```

**Tier 2 — Medium (iteration caps, circuit breakers):**

```python
# Iteration cap — needs framework-specific patterns
f"""In {file_path}, the {framework} node "{step_name}" currently loops 
without a maximum. It averages {mean_iterations} iterations, with 
outliers reaching {max_observed_iterations}. 

Add a maximum iteration cap of {recommended_cap}.

{_framework_specific_loop_cap_instructions(framework, step_name)}

After {recommended_cap} iterations, exit the loop and return the best 
output so far. Log a warning when the cap is hit.

After making the change, run: pretia profile run {file_path}
to verify the optimization."""
```

Where `_framework_specific_loop_cap_instructions` generates different code depending on the framework:

```python
def _framework_specific_loop_cap_instructions(framework, step_name):
    if framework == "langgraph":
        return f"""In your LangGraph StateGraph, the loop is likely controlled 
by a conditional edge after "{step_name}". Add an "iteration_count" field 
to your State TypedDict. Increment it in the {step_name} node. In the 
conditional edge function, check if iteration_count >= {{recommended_cap}} 
and route to the next node instead of looping back.

Example:
  class State(TypedDict):
      messages: list
      iteration_count: int  # ADD THIS
      
  def should_continue(state):
      if state["iteration_count"] >= {{recommended_cap}}:
          return "exit_loop"  # Route to next node
      return "continue_loop"  # Route back to {step_name}"""
    elif framework == "openai_agents":
        return f"""In your OpenAI Agents SDK runner, add a loop counter 
to the agent's context. Check it in the agent's instructions or in a 
tool guard. When the cap is reached, the agent should return its best 
output and stop requesting further tool calls."""
    # ... CrewAI, Generic patterns
```

**Tier 3 — Complex (architecture changes: compaction, step insertion, pipeline restructuring):**

These are the hardest because they modify the graph topology. The prompt must include:
- The full current graph structure (nodes + edges) so the coding AI understands the context
- The exact insertion point for the new node
- How state flows through the new node (input/output types)
- Framework-specific wiring (LangGraph `add_node` + `add_edge`, etc.)

```python
# Context compaction — needs full graph context
f"""In {file_path}, the {framework} workflow has this structure:
{_render_graph_as_text(graph_structure)}

The node "{step_name}" re-sends the full conversation context on each 
loop iteration. Context grows +{delta_tokens} tokens/iteration, reaching 
{max_context_tokens} tokens by iteration {max_observed_iterations}.

ADD a new node "compact_context" between "{preceding_node}" and 
"{step_name}" (inside the loop, before each re-entry):

1. The compact_context node receives state["{state_messages_field}"]
2. It keeps the last 3 messages verbatim
3. It summarizes everything before that into ~500 tokens using {cheap_model}
4. It writes the compacted messages back to state["{state_messages_field}"]
5. Then {step_name} receives the compacted context (~{estimated_compacted_tokens} tokens)

{_framework_specific_node_insertion(framework, "compact_context", preceding_node, step_name)}

IMPORTANT: The compaction must happen INSIDE the loop (before each 
iteration of {step_name}), not outside it. The first iteration can 
skip compaction since the context is still small.

Don't change the {step_name} logic itself — only add the compaction 
step before it.

After making the change, run: pretia profile run {file_path}
to verify the context growth is reduced."""
```

Where `_render_graph_as_text` produces a simple textual representation of the graph:
```
START → classify_intent → retrieve_context → generate_draft → review_and_iterate ↺ → format_response → END
                                                               (loop: avg 8.2 iterations)
```

And `_framework_specific_node_insertion` generates the actual code pattern:
```python
def _framework_specific_node_insertion(framework, new_node, before, after):
    if framework == "langgraph":
        return f"""In your StateGraph builder, add:
  graph.add_node("{new_node}", compact_context_fn)
  
And rewire the edges:
  # Before: ... → {before} → {after} (loop back)
  # After:  ... → {before} → {new_node} → {after} (loop back)
  
Replace the edge from {before} to {after} with:
  graph.add_edge("{before}", "{new_node}")
  graph.add_edge("{new_node}", "{after}")"""
```

**The "render_graph_as_text" function** is reusable — the same graph extractor that feeds the visual graph view (section 1.7) also produces the text representation for the prompt. No additional parsing needed.

### 1.6 Verification loop — "Optimize → Verify → Iterate"

The implementation prompts all end with `run: pretia profile run {file_path} to verify`. This is deliberate — it creates a closed optimization loop:

```
Profile → Recommend → Implement (via prompt) → Re-profile → Verify → Iterate
```

**What verification checks:**

After the user implements a recommendation and re-profiles, Pretia compares the new profile against the old one and produces a verification report:

```
✅ Verification: support_agent.py

Recommendation: "Downshift model at step 4 (Opus → Sonnet)"
  Before: $9,200/mo at step "review_and_iterate"
  After:  $4,800/mo at step "review_and_iterate"
  Actual savings: $4,400/mo (predicted: $4,200/mo) ✓
  
Recommendation: "Cap review iterations to 3"
  Before: avg 8.2 iterations, max 12
  After:  avg 2.8 iterations, max 3
  Actual savings: $2,300/mo (predicted: $2,100/mo) ✓

Recommendation: "Add context compaction" 
  Status: NOT YET IMPLEMENTED
  Step "compact_context" not found in workflow graph.
  
Overall: 2 of 3 recommendations applied.
  Before: $11,700/mo → After: $5,100/mo → Remaining potential: $2,900/mo
  New score: 62/100 (was 28/100)
```

**Implementation:**

```python
# pretia verify compares two profiles
def verify(old_profile, new_profile, recommendations):
    results = []
    for rec in recommendations:
        if rec.type == "model_swap":
            # Check if the model actually changed at the target step
            old_model = old_profile.steps[rec.step].model
            new_model = new_profile.steps[rec.step].model
            applied = (new_model == rec.recommended_model)
            actual_savings = old_profile.steps[rec.step].monthly_cost - new_profile.steps[rec.step].monthly_cost
            
        elif rec.type == "architecture":
            # Check if new node exists in the graph
            applied = rec.new_node_name in new_profile.steps
            actual_savings = compute_step_delta(old_profile, new_profile, rec.step)
            
        elif rec.type == "workflow":
            # Check if iteration count decreased
            old_iters = old_profile.steps[rec.step].iterations.mean
            new_iters = new_profile.steps[rec.step].iterations.mean
            applied = new_iters <= rec.recommended_cap * 1.1  # 10% tolerance
            actual_savings = compute_step_delta(old_profile, new_profile, rec.step)
            
        results.append(VerificationResult(
            recommendation=rec,
            applied=applied,
            predicted_savings=rec.estimated_savings,
            actual_savings=actual_savings,
            prediction_accuracy=actual_savings / rec.estimated_savings if rec.estimated_savings else None,
        ))
    return results
```

**The verify command:**
```bash
pretia verify --baseline .pretia/baseline.json
# Re-profiles the workflow, compares to baseline, shows what improved and what's left
```

**In the UI (Screen 3 — Report):** When a baseline exists, the report automatically shows a "Changes since baseline" section with green checkmarks for applied recommendations and remaining recommendations still to do. The before/after graph updates to show the current state vs. the remaining optimizations.

**Why this matters for the moat:** Every verification result is a training signal. If the user applied a model swap and the actual savings were $4,400 vs. predicted $4,200, that's a data point that validates the classifier. If a compaction recommendation saved less than predicted, that's a signal to adjust the compaction heuristics. Over time, the predictions get more accurate because they're grounded in real implementation outcomes.

**The "optimize until green" loop:** The score (0-100) creates a natural gamification. The user sees 28/100 (red), implements 2 of 3 recommendations, re-profiles, sees 62/100 (amber), implements the last one, re-profiles, sees 89/100 (green). Each cycle takes 5 minutes (implement → `pretia verify`). The product becomes a game you play until your score is green.

### 1.7 GitHub Action

```yaml
# .github/workflows/pretia.yml
name: Pretia
on:
  pull_request:
    paths: ['agents/**', 'workflows/**']

jobs:
  cost-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pretia/action@v1
        with:
          workflow: agents/support_agent.py
          baseline: .pretia/baseline.json
          traffic: 1000/day
          threshold: 50              # % increase that blocks
          mode: static               # static | auto-generate | sample | import
          auto_generate_count: 5     # inputs to generate (for auto-generate mode)
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Four CI modes:**

| Mode | How it works | Cost | Accuracy |
|------|-------------|------|----------|
| `static` (default) | Analyzes code diff, estimates impact from baseline | Free | Low — catches model changes, misses logic changes |
| `auto-generate` | Generates 5 synthetic inputs, runs workflow, compares to baseline | ~$1-3/run | High — real measurements, diverse inputs |
| `sample` | Re-runs workflow on 5-10 inputs from a JSONL file | ~$2-5/run | High — user-curated inputs |
| `import` | Pulls latest traces from Langfuse, analyzes without re-execution | Free | High — real production data, no execution cost |

Default = `static` on every PR. Recommended upgrade: `auto-generate` on PRs that touch agent logic, or `import` if Langfuse is available.

**Smart mode selection:** When `LANGFUSE_PUBLIC_KEY` is in environment, the action auto-suggests `import` mode. Otherwise defaults to `static` with a comment suggesting `auto-generate` for more accurate results.

**Baseline format:**
```json
{
  "version": "1.0",
  "workflow": "agents/support_agent.py",
  "profiled_at": "2026-05-20T14:30:00Z",
  "sample_size": 100,
  "traffic_assumption": "1000/day",
  "steps": {
    "classify_intent": {
      "model": "claude-3-haiku",
      "tokens": {
        "input": { "p50": 340, "p95": 520 },
        "output": { "p50": 45, "p95": 80 }
      },
      "cost_per_run": { "p50": 0.00012, "p95": 0.00019 },
      "iterations": { "mean": 1.0, "max": 1 },
      "system_prompt_hash": "a3f8c2...",
      "system_prompt_tokens": 280,
      "output_format": "json",
      "task_complexity_tier": null
    },
    "review_and_iterate": {
      "model": "claude-opus-4",
      "tokens": {
        "input": { "p50": 9200, "p95": 34000 },
        "output": { "p50": 1800, "p95": 4200 }
      },
      "cost_per_run": { "p50": 0.18, "p95": 0.62 },
      "iterations": { "mean": 5.3, "max": 12 },
      "context_growth_rate": 1200,
      "flags": ["high_variance", "context_growth"],
      "system_prompt_hash": "b7d1e9...",
      "system_prompt_tokens": 1450,
      "output_format": "text",
      "task_complexity_tier": null
    }
  },
  "total_monthly": {
    "p50": 2400, "p75": 4100, "p90": 7800, "p95": 11700
  }
}
```

Note: `task_complexity_tier` is `null` in v1 (filled by heuristics), populated by the ML classifier in v1.5.

### 1.8 User Interface Layer

Pretia has three visual output channels, all sharing the same underlying data from the ProfileStore. The UI is NOT a separate product — it's the rendering layer on top of the same analysis pipeline.

#### HTML Report (ships with v1)

Every `pretia profile run` generates a self-contained HTML file (`.pretia/report.html`) and opens it in the browser automatically. The report is a **single static file** — no server, no JavaScript framework dependency at runtime. Built with Jinja2 templates + inline CSS/JS, compiled at profiling time.

**Report contents:**
- **Score ring:** circular SVG indicator (0-100) color-coded: 0-40 red, 41-70 amber, 71-100 green. Score is calculated as: `100 - (waste_percentage)` where waste = cost reducible by following all recommendations. A score of 28 means 72% of the spend is recoverable. Displayed alongside four metric cards: monthly projected cost (red if recommendations exist), projected cost after fixes (green), cost per run (p50), and number of recommendations.
- **Cost waterfall chart:** per-step horizontal bars showing relative cost share. Each bar width = step cost / total cost. The most expensive step is highlighted in red, others in teal. Shows step name, bar, dollar amount, and percentage. The visual must make the dominant cost driver immediately obvious.
- **Context growth sparklines:** for each looping step, a mini SVG line chart showing how context size grows per iteration (x-axis = iteration, y-axis = token count).
- **Recommendations cards:** each card has a colored type tag (teal for MODEL SWAP, blue for ARCHITECTURE, purple for WORKFLOW), a title, a description, estimated savings in green aligned right, and a confidence indicator (progress bar + percentage or "confirmed" for pattern-based recs). Cards are stacked vertically. Below all cards: a green summary banner showing total savings and before→after cost.
- **Projection table:** traffic (100/day, 1K/day, 10K/day) × percentiles (p50, p75, p90, p95). Tabular numeric formatting.
- **Architecture graph view — before/after:** The most visually impactful feature. Two tabs showing the agent workflow as a DAG:
  - **"Current architecture" tab:** Each step is a node colored by cost (white = healthy, red = expensive). Edges annotated with token counts. The most expensive node is visually dominant. Recommendation badges are pinned directly on the problematic nodes (teal badge = model swap with savings, blue = architecture with savings, purple = workflow with savings). Loop edges are dashed with annotations (+1,200 tok/iter, avg 8.2 loops). The pain points are instantly visible.
  - **"After recommendations" tab:** Same architecture but with all recommendations applied. The red node becomes green. New steps appear (e.g. compaction step inserted before the loop). Model names change (Opus → Sonnet). Loop annotations change (max 3 loops, 8K tok instead of 34K). Green "applied" badges confirm each change. Summary banner: $11,700 → $3,200, -73%.

  **Technical approach for graph extraction:**
  - LangGraph: `graph.get_graph()` returns a Mermaid-compatible graph definition with nodes and edges. Parse to extract structure.
  - OpenAI Agents SDK: extract agents + handoff relationships from the Runner configuration.
  - CrewAI: extract tasks + sequential/parallel relationships from the Crew definition.
  - Generic: fall back to the StepRecord `parent_step` field to reconstruct the DAG.
  - Graph rendering: generate SVG server-side (for HTML report) or render with React (for web UI). Simple top-to-bottom layout — no need for dagre or d3 for DAGs with 3-10 nodes. Position nodes vertically, add loop-back edges with bezier curves for iterations.
  - Cost-to-color mapping: compute cost share per node, map to 3 tiers: <10% share = healthy (white), 10-40% = warning (amber), >40% = expensive (red).
  - "After" graph: apply each recommendation to the graph structure programmatically (insert compaction node, change model name, cap loop annotation, recompute costs) and re-render.
- **Raw data toggle:** expandable section with the full StepRecord JSON for power users.
- **Footer:** "Profiled on N auto-generated inputs · workflow_name.py · framework"

**Technical approach:**
```python
# After profiling completes:
from pretia.report import render_html_report

report_path = render_html_report(
    profile=profile_data,
    projections=projection_results,
    recommendations=recommendation_list,
    output_path=".pretia/report.html"
)
webbrowser.open(f"file://{report_path}")
```

The template is a single Jinja2 file (~300 lines) bundled with the pip package. CSS is inline (no external dependencies). Charts use inline SVG (no Chart.js or D3 needed for the v1 — the sparklines and waterfall are simple enough to generate server-side as SVG paths). The file is fully self-contained and shareable.

#### Local Web UI (ships with v1, Sprint 6)

`pretia ui` launches a local web server on `localhost:7100` with a visual interface for the full workflow.

**Three screens:**

**Screen 1 — Setup:**
- Workflow file input (text field with file path, auto-completes)
- Auto-detection card: framework badge ("LangGraph — auto-detected"), steps count + loop count, models listed, tools listed. This card appears instantly when the workflow path is entered — the backend parses the file and returns the metadata.
- Input mode selector: 2x2 grid of cards. Auto-generate (default, highlighted with green border + "default" badge) / Import from Langfuse / Single input / Upload JSONL. Each card shows a title, description, and cost estimate.
- Profiling settings: slider for input count (5-50, default 20), dropdown for traffic volume (100/day, 1K/day, 10K/day, Custom)
- Green "Run profiling" button, full width. Below: estimated cost and time ("~20 runs, ~$2.00, ~3 min")

**Screen 2 — Live profiling:**
- Progress bar with label: "Run 14 of 20" and "~1 min remaining"
- Three metric cards in a row: "Cost so far" ($5.88), "Avg per run" ($0.42), "Projected/mo" ($12,600 in red if above threshold)
- Live cost table: columns = Step, Avg tokens, Avg cost, Share (mini horizontal bar), Model. Rows fill as data accumulates. The most expensive step row turns red. Share bars use proportional width (dominant step bar fills 80%+)
- Pattern flags: appear below the table as colored pills when detected. Red pill for danger patterns (context growth), amber for warnings (high iteration count)
- "Currently running" card at bottom: shows the current input text in monospace, step progress as dots (filled = complete, pulsing = active, empty = pending), and step names with checkmarks

**Screen 3 — Report:**
- Same layout as the HTML report but interactive
- **Architecture graph view (primary view):** tabbed before/after showing the workflow DAG with cost overlay and recommendations applied. This is the default landing view — the graph is the first thing the user sees after profiling. The before graph has red nodes with recommendation badges. The after graph shows the optimized architecture in green with applied badges. Summary banner shows total savings and percentage.
- Cost waterfall (secondary view): horizontal bars, click a step to expand per-run distribution, context growth chart, token breakdown
- Recommendations list: click to expand reasoning and "how to implement" guidance
- Three action buttons in the header: "Export as HTML" / "Copy PR comment" / "Save baseline"

**REST API endpoints:**

```
GET  /api/detect          → Parse workflow file, return framework + steps + models
POST /api/generate-inputs → Generate synthetic inputs from system prompt
POST /api/profile/start   → Start profiling (returns WebSocket URL for progress)
GET  /api/profile/status  → Current profiling status
GET  /api/report          → Full report data (JSON)
GET  /api/report/html     → Download standalone HTML report
GET  /api/baseline        → Current baseline
POST /api/baseline/save   → Save current profile as baseline
```

**Technical approach:**

```
Backend:  FastAPI (lightweight, async, WebSocket support for live progress)
Frontend: Single-page React app compiled to static bundle
Comms:    WebSocket for live profiling updates, REST for everything else
Bundling: Frontend pre-built and shipped inside the pip package
          (no npm install needed by the user)
```

The React frontend is pre-compiled to a static JS bundle (~200KB gzipped) during the package build process and included in the pip distribution. When the user runs `pretia ui`, FastAPI serves the static bundle + exposes API endpoints. No Node.js, no npm, no build step for the user.

```python
# What `pretia ui` does:
import uvicorn
from pretia.ui.app import create_app

app = create_app()
uvicorn.run(app, host="127.0.0.1", port=7100)
# Opens browser to http://localhost:7100
```

**WebSocket protocol for live profiling:**
```json
// Server → Client messages during profiling:
{"type": "run_start", "run": 1, "total": 20, "input_preview": "How do I..."}
{"type": "step_complete", "run": 1, "step": "classify", "tokens": 340, "cost": 0.00012}
{"type": "step_complete", "run": 1, "step": "generate", "tokens": 8900, "cost": 0.032}
{"type": "pattern_detected", "pattern": "context_growth", "step": "review", "rate": 1200}
{"type": "run_complete", "run": 1, "total_cost": 0.42}
{"type": "profiling_complete", "report_url": "/api/report"}
```

**Why FastAPI and not Streamlit/Gradio:** Streamlit adds ~50MB of dependencies and imposes its own layout system. Gradio is similar. FastAPI + pre-built React bundle is ~5MB total, gives full control over the UX, and the React frontend can be reused for a future cloud version if needed.

#### GitHub PR Comment (ships with v1, Sprint 5)

The GitHub Action produces a formatted PR comment. This is the CI interface — developers see it without leaving their normal workflow. Same data as the HTML report, but condensed to fit a GitHub comment (~30 lines max).

Already specified in section 1.6.

### 1.9 v1.0 Execution Plan (Weeks 1-20)

**Weeks 1-2: Foundations**
- Python package: `pip install pretia`
- `StepRecord` dataclass + `ProfileStore` (JSON)
- `LangGraphCollector`: hook into LangChain callbacks → StepRecords
- `GenericCollector`: decorator + context manager
- CLI: `pretia profile run workflow.py --inputs sample.jsonl`
- Unit tests for collectors

**Weeks 3-4: OpenAI Agents SDK + Stats**
- `OpenAIAgentsCollector`: hook via RunHooks + native usage tracking
- Stats module: distributions per step (p50-p99)
- Pattern detection: context growth, loop variance, high variance
- CLI: `pretia report profile.json --traffic 1000/day`

**Weeks 5-6: Projection Engine**
- Linear projection for stable cases
- Monte Carlo for non-linear cases
- Confidence intervals
- Baseline comparison: `pretia diff baseline.json new_profile.json`

**Weeks 7-8: Recommendation Engine**
- Rule-based recommendations (model swap, architecture, workflow)
- Impact estimation in $/month per recommendation
- CLI: `pretia recommend profile.json`

**Weeks 9-10: GitHub Action**
- Three CI modes (static, dryrun, sample)
- PR comment formatting
- Baseline management: `pretia baseline update`
- Configurable thresholds

**Weeks 11-12: Polish + Launch**
- CrewAI Collector (same callback pattern as LangGraph)
- End-to-end integration tests
- README with demo GIF
- Launch: PyPI + GitHub + HN + LangChain Discord

---

## PART 2 — ML LAYER

### 2.1 Task Complexity Classifier (v1.5, weeks 21-24)

**What it does:** Given a step's (system_prompt, sample_input, sample_output), predicts the minimum model tier sufficient for the task: tier-1 (Haiku/mini), tier-2 (Sonnet/4o), tier-3 (Opus/o3). Replaces the keyword heuristics in model swap recommendations with a trained classifier.

**Why it matters for the moat:** Every model swap recommendation that users follow (or revert) becomes a labeled training example. After 10K swaps observed, the classifier's accuracy becomes proprietary and unreplicable.

#### Data sources

**Primary — RouterBench (free, 405K examples):**
Publicly available benchmark by Martian. 405,000+ inference outcomes across 64 tasks and 11 LLMs. Each record contains: the prompt, the model's response, a quality score, and the inference cost. We derive our label: for each prompt, what's the cheapest model that achieves ≥90% of the best model's quality score? That IS the tier label.

Limitations: RouterBench covers benchmark-style tasks (commonsense QA, math, coding, summarization), not real agent workflow prompts. Good for bootstrapping, needs supplementation with real workflow data.

**Secondary — LLMRouterBench (free, 400K+ examples):**
Jan 2026 benchmark. 400K+ instances across 21 datasets and 33 models including 13 recent flagship models and 5 proprietary. More recent models than RouterBench. $2.7K in API costs to build — but the data is public.

**Tertiary — Self-generated synthetic data (~$150):**
200 system prompts across difficulty levels × 3 model tiers × 5 sample inputs. Run each, score with LLM-as-judge, label tier. Fills the gap between benchmark tasks and real agent workflow prompts.

**Ongoing — User feedback (free, accumulates over time):**
Every time a user follows a model swap recommendation and reports quality maintained → positive label. Every revert → negative label. This is the most valuable data source long-term. Captured automatically via the `task_complexity_tier` field in baselines: if the user changes the model at a step and re-profiles, we see the quality delta.

#### Model architecture

**Simplest viable approach (what we ship first):**

```python
# Step 1: Embed the system prompt
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 22M params, runs on CPU
prompt_embedding = embedder.encode(system_prompt)  # 384-dim vector

# Step 2: Combine with numerical features
features = np.concatenate([
    prompt_embedding,                          # 384 dims
    [system_prompt_tokens,                     # 1 dim
     avg_input_tokens,                         # 1 dim
     avg_output_tokens,                        # 1 dim
     output_input_ratio,                       # 1 dim
     n_constraints_in_prompt,                  # 1 dim (regex count)
     has_json_schema,                          # 1 dim (bool)
     has_few_shot_examples,                    # 1 dim (bool)
     has_chain_of_thought,                     # 1 dim (bool)
     n_tools_available]                        # 1 dim
])  # Total: 393 features

# Step 3: Train classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf.fit(X_train, y_train)  # y ∈ {tier-1, tier-2, tier-3}

# Step 4: Predict with confidence
tier, confidence = clf.predict(features), clf.predict_proba(features).max()
# Only recommend swap if confidence > 0.85
```

**Why logistic regression and not a neural net:**
- 405K RouterBench examples + 1K synthetic = enough for logistic regression on embeddings
- Trains in <30 seconds on CPU
- No GPU needed, no serving infrastructure
- Interpretable: you can inspect which features drive the tier prediction
- Confidence scores are well-calibrated (important for "should I trust this recommendation?")

**Upgrade path:** If accuracy plateaus below 85%, swap logistic regression for XGBoost (still no GPU, trains in minutes, handles feature interactions better). If that plateaus, fine-tune the sentence-transformer on domain-specific prompt pairs — but that's v3 territory.

#### Training pipeline

```
RouterBench (405K) ──┐
                     ├──▶ Label derivation ──▶ (prompt, tier) pairs
LLMRouterBench (400K)┘    "cheapest model
                            scoring ≥90% of
                            best quality"
                                │
Synthetic prompts (1K) ────────┤
                                │
User feedback (ongoing) ───────┤
                                ▼
                     ┌──────────────────┐
                     │  Feature pipeline │
                     │  MiniLM embed +   │
                     │  numerical feats   │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ LogisticRegression│
                     │ or XGBoost       │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Serialized model  │
                     │ (~2MB .pkl file)  │
                     │ Shipped with SDK  │
                     └──────────────────┘
```

The trained model ships AS PART OF THE PIP PACKAGE (~2MB). No API calls, no cloud inference, no latency. `pretia recommend` runs the classifier locally.

#### Integration with v1

The v1 baseline already captures `system_prompt_hash`, `system_prompt_tokens`, and `output_format` per step. In v1.5:
- The `task_complexity_tier` field goes from `null` to a predicted value
- Model swap recommendations go from "this step looks like classification based on keywords" to "our classifier (85% accuracy on 400K+ examples) predicts tier-1 is sufficient, confidence: 0.92"
- Recommendations with confidence < 0.85 fall back to v1 heuristics with a note

#### Cost to bootstrap: ~$150

200 synthetic prompts × 3 model tiers × 5 inputs × ~$0.05/run = $150. Combined with 405K free RouterBench examples. Total training time: <5 minutes on a laptop.

#### Weeks 21-24 execution plan

- Week 21: Download RouterBench + LLMRouterBench. Write label derivation script (cheapest model at ≥90% quality). Generate 200 synthetic prompts, run through 3 tiers, score with LLM-as-judge.
- Week 22: Build feature pipeline (MiniLM embeddings + numerical features). Train logistic regression. Evaluate accuracy on held-out set. If <80%, try XGBoost.
- Week 23: Integrate classifier into `pretia recommend`. Update PR comment format to show classifier confidence. Add feedback loop: detect when users change models post-recommendation.
- Week 24: Ship v1.5. Blog post: "How we predict which model your agent step really needs."

---

### 2.2 Cost Prediction Model (v2.0, weeks 25-34)

**What it does:** Given a NEW workflow's structure (without profiling it on 100 inputs), predicts its cost distribution. The "zero-shot cost estimate."

**Why it matters:** Eliminates the cold start problem. Instead of "profile 100 runs to get a cost estimate," you get "paste your workflow code, get an instant estimate." Accuracy improves as more workflows are profiled across all users.

#### Data sources

**Primary — Your own profiling data (accumulates from v1 users):**
Every `pretia profile` run produces a (workflow_structure, cost_distribution) pair. After 6 months of v1 adoption, target: 200-500 real workflow profiles.

**Secondary — Self-generated synthetic workflows (~$200):**
Build 100 template workflows across 5 archetypes:
- Support agent (RAG + classify + generate + review loop)
- Code review agent (read files + analyze + suggest + iterate)
- Data extraction agent (parse + extract + validate + format)
- Research agent (search + summarize + synthesize, multi-agent)
- Sales/outreach agent (qualify + personalize + draft)

Vary parameters: steps (3-10), models (haiku→opus), loop iterations (1-15), context sizes. Profile each on 20 inputs.

Cost: 100 workflows × 20 inputs × ~$0.10/run = $200.

**Tertiary — Public priors:**
Iternal.ai publishes token consumption per use case archetype (simple agents: 5K-15K tokens, complex multi-agent: 200K-1M+, coding agents: 1-3.5M). Use as Bayesian priors for archetypes where you have few observations.

#### Model architecture

**XGBoost on workflow-level features:**

```python
workflow_features = {
    # Structure
    "n_steps": 5,
    "n_llm_steps": 4,
    "n_tool_steps": 3,
    "has_loop": True,
    "n_loops": 1,
    "max_loop_depth": 2,

    # Models
    "max_model_tier": 3,          # 1=haiku, 2=sonnet, 3=opus
    "avg_model_tier": 2.1,
    "n_distinct_models": 3,

    # Prompts (from static analysis, no execution needed)
    "total_system_prompt_tokens": 4200,
    "avg_system_prompt_tokens": 840,
    "max_system_prompt_tokens": 1800,

    # Tools
    "total_tools_defined": 8,
    "total_tool_definition_tokens": 3200,
    "has_rag": True,
    "has_mcp": False,

    # Archetype embedding (from Task Complexity Classifier)
    "step_tier_distribution": [0.2, 0.6, 0.2],  # % tier-1, tier-2, tier-3
}

# Target: log(cost_per_run_p50), log(cost_per_run_p95)
# Log-transform because costs are log-normally distributed
```

**Why XGBoost:**
- Handles small datasets well (200-500 samples)
- Handles mixed feature types (numerical + categorical)
- Captures non-linear interactions (loops × model tier × context)
- Feature importance is interpretable ("78% of prediction variance comes from max_model_tier and has_loop")
- Trains in seconds, predicts in milliseconds
- No GPU needed

**Minimum data needed:** ~200 workflows for basic predictions. ~500 for reliable confidence intervals. Below 200, fall back to archetype-based lookup tables (the public priors from Iternal.ai).

#### Upgrade path

At 2,000+ workflows: add per-step features (not just workflow-level aggregates). Train a model that predicts cost per step, then sums. More accurate but needs more data.

At 5,000+ workflows: experiment with a small neural network (2-3 hidden layers) that takes workflow graph structure as input. Only worth it if XGBoost accuracy plateaus.

#### Weeks 25-30 execution plan (alongside live monitoring build)

- Week 25-26: Collect all profiling data from v1 users. Generate synthetic workflows ($200). Build feature extraction pipeline from workflow code → feature dict.
- Week 27-28: Train XGBoost. Evaluate with cross-validation. Compare against baseline (archetype lookup table + linear projection). Ship only if XGBoost beats baseline by >15% MAPE.
- Week 29-30: Integrate into CLI: `pretia estimate workflow.py` (instant, no profiling needed). Add to GitHub Action `static` mode: code-only cost estimate.

---

### 2.3 Workflow Benchmarking Engine (v3.0, weeks 35-48)

**What it does:** Groups similar workflows into clusters. Enables: "your support agent costs $0.82/run. The median for similar workflows is $0.35/run. You're at the 89th percentile."

**Why it matters for the moat:** The benchmark data is the ultimate network effect. Every new workflow makes every existing user's benchmark more accurate. A competitor with 50 workflows can't benchmark. You with 5,000 can.

#### Data sources

**Primary — Organic user data:** Every profiled workflow contributes its feature vector + cost distribution. At month 9+, target: 500-1000 workflows.

**Secondary — GitHub scraping (free):**
Parse LangGraph/CrewAI workflow definitions from public repos. Extract structure (nodes, edges, tools, models) WITHOUT executing them. Estimated yield: 500-2000 parseable workflows from GitHub search. No cost data (can't run them), but provides structural diversity for the embedding space.

**Tertiary — Synthetic bootstrap (same $200 as use case 2):** The 100 synthetic workflows from the Cost Prediction Model also seed the clustering.

#### Model architecture

**HDBSCAN on engineered features (no training needed):**

```python
from hdbscan import HDBSCAN

workflow_vectors = []  # Each: ~20 numerical features per workflow
for w in all_workflows:
    vec = [
        w.n_steps, w.n_llm_steps, w.n_tool_steps,
        w.has_loop, w.n_loops, w.max_loop_depth,
        w.max_model_tier, w.avg_model_tier,
        w.total_system_prompt_tokens,
        w.has_rag, w.has_mcp,
        w.step_tier_distribution_tier1,  # from classifier
        w.step_tier_distribution_tier2,
        w.step_tier_distribution_tier3,
        w.total_cost_p50,  # log-transformed
        w.total_cost_p95,
        w.cost_variance_ratio,
        # Use-case tag (one-hot): support, coding, extraction, research, sales
    ]
    workflow_vectors.append(vec)

clusterer = HDBSCAN(min_cluster_size=15, min_samples=5)
labels = clusterer.fit_predict(StandardScaler().fit_transform(workflow_vectors))

# Each cluster = a benchmark group
# "Cluster 3: support agents with RAG + review loop. N=47. Median cost: $0.35/run."
```

**Why HDBSCAN:**
- Density-based: no need to specify number of clusters (k)
- Handles noise (outlier workflows get label -1, not forced into a cluster)
- Minimum cluster size is configurable (we set 15 = need at least 15 similar workflows to form a benchmark group)
- No training — it's unsupervised, runs on the data directly
- Updates incrementally as new workflows arrive (approximate_predict)

**Minimum data needed:** ~15 workflows per cluster to be meaningful. With 5-10 common archetypes, need ~100-500 total workflows. Below that, benchmarks are too noisy to show users.

#### Benchmark output format

```json
{
  "workflow": "agents/support_agent.py",
  "cluster": {
    "id": 3,
    "name": "Support agent with RAG + review loop",
    "n_workflows": 47,
    "median_cost_per_run": 0.35,
    "p25_cost": 0.22,
    "p75_cost": 0.58
  },
  "your_position": {
    "cost_per_run": 0.82,
    "percentile": 89,
    "verdict": "Your workflow costs 2.3x the cluster median.",
    "top_cost_drivers_vs_median": [
      "Step 'review' uses Opus (cluster median: Sonnet) → +$0.28/run",
      "Average 8.2 iterations (cluster median: 4.1) → +$0.15/run"
    ]
  }
}
```

---

## PART 3 — DATA STRATEGY

### 3.1 What we capture from day one (even before ML)

Every `pretia profile` run produces a dataset that feeds ALL future ML features. The v1 ProfileStore saves:

```
Per workflow:
  - Structure: n_steps, step names, step types, DAG edges
  - Models: which model per step
  - Tools: tool definitions, MCP servers
  - Use-case tag: user-provided or inferred

Per step per run:
  - Tokens: input, output, context_size, tool_definitions
  - System prompt: hash + token count + raw text (opt-in)
  - Output format: json/text/code (auto-detected)
  - Iteration number (if in a loop)
  - Is_retry flag
  - Duration

Per workflow per profiling session:
  - N runs
  - Cost distribution (p50-p99)
  - Detected patterns (context_growth, high_variance, stuck_loops)
  - Recommendations generated
  - Recommendations followed (tracked via baseline diffs)
```

**Privacy:** System prompt raw text is opt-in (some users won't share proprietary prompts). The hash + token count + output_format are always captured — enough for the classifier. For benchmarking, only aggregated/anonymized features leave the user's environment unless they opt into cloud benchmarks.

### 3.2 Data flywheel

```
More users adopt v1 (free OSS)
        │
        ▼
More workflows profiled
        │
        ├──▶ Better Task Complexity Classifier (user swap feedback)
        ├──▶ Better Cost Prediction Model (more training examples)
        └──▶ Better Benchmarks (more workflows per cluster)
                │
                ▼
        Better recommendations + benchmarks
                │
                ▼
        More users adopt (word of mouth, "the tool that knows
        what your agent should cost")
                │
                ▼
        Repeat
```

The key insight: each ML model feeds the next. The Task Complexity Classifier produces `step_tier_distribution` features that the Cost Prediction Model and Workflow Benchmarking both consume. Improving the classifier improves everything downstream.

### 3.3 Total bootstrap cost

| Item | Cost | What you get |
|------|------|-------------|
| RouterBench download | $0 | 405K labeled prompt-tier pairs |
| LLMRouterBench download | $0 | 400K labeled prompt-tier pairs |
| Synthetic prompts for classifier | ~$150 | 1K domain-specific training examples |
| Synthetic workflows for cost model | ~$200 | 100 profiled workflows across 5 archetypes |
| GitHub scraping for benchmarking | $0 | 500-2000 workflow structures (no cost data) |
| **Total** | **~$350** | Bootstrap data for all three ML models |

---

## PART 4 — RISKS AND MITIGATIONS

### Technical risks

**Projections are wrong.** The #1 credibility killer. Mitigation: always show intervals (p50-p95), flag high-variance steps, disclaimer "based on N samples." The ML cost predictor (v2) adds a confidence score: "estimate confidence: 72% — run a full profile for higher accuracy."

**Profiling is expensive.** 100 runs × $1/run = $100. Mitigation: three CI modes (static/dryrun/sample). The ML cost predictor (v2) eliminates the need for profiling in many cases — instant estimate from code structure.

**Framework APIs change.** LangGraph changed callback APIs 3x in 2025. OpenAI Agents SDK is v0.14. Mitigation: Collector abstraction isolates the SDK. If LangGraph changes, update one file.

**Classifier generalizes poorly.** RouterBench = benchmark tasks ≠ real agent prompts. Mitigation: synthetic data fills the gap initially. User feedback corrects over time. Confidence threshold (0.85) prevents bad recommendations — low-confidence falls back to heuristics.

**Not enough data for ML.** If v1 adoption is slow, ML models don't have enough training data. Mitigation: all ML features have a heuristic fallback. The product works without ML (v1 is stats-only). ML is an accelerant, not a dependency.

### Strategic risks

**Braintrust adds a cost check.** Their CI quality gate is one feature flag away from a cost gate. Mitigation: our projection engine (distributional + Monte Carlo) and ML-powered recommendations are deeper than a simple cost threshold. And once we have the benchmark data, the network effect is the moat.

**Langfuse adds cost projection.** ClickHouse has unlimited engineering resources. Mitigation: Langfuse is observability-first (post-hoc). Cost projection is pre-deployment. Different product category, different buyer moment. But watch closely.

**"Feature, not product" risk.** Cost profiling could be absorbed into agent frameworks themselves (LangGraph adds `cost_estimate()` natively). Mitigation: the ML layer (classifier + predictor + benchmarks) is the product, not the profiling. Profiling is the wedge.

---

## PART 4B — PREDICTION QUALITY ASSURANCE

This section describes the validation infrastructure that must pass before v1 launches. If the predictions are unreliable, the product is dead. No amount of UI polish or ML sophistication will save it. The backtesting suite is not a "nice to have" — it is the go/no-go gate for launch.

### 4B.1 The validation philosophy

The goal is NOT to be exactly right. It's to be **usefully right within stated bounds.** A projection of "$4,100 – $11,700/mo (moderate confidence)" that lands at $9,200 in production is a success. A projection of "$11,700/mo" that lands at $45,000 is a product-killing failure. The difference is honesty about uncertainty.

Every projection must satisfy three properties:

1. **Calibration:** The p50 estimate should be exceeded roughly 50% of the time. The p95 should be exceeded roughly 5% of the time. If the p95 is exceeded 25% of the time, the engine is systematically overconfident.

2. **Directional accuracy:** If Pretia says step 4 is the most expensive step at 78% of total cost, it should actually be the most expensive step in production. Getting the relative ranking right matters more than the absolute numbers.

3. **Useful range:** The p50-to-p95 spread should be tight enough to be actionable. A projection of "$500 – $500,000/mo" is technically correct but useless. The range should be within 3-5x from bottom to top for moderate-confidence estimates.

### 4B.2 The Backtesting Suite

A set of real agent workflows, each profiled at small sample sizes and validated against large-sample ground truth. Run before launch. Run again after any change to the projection engine.

**14 test workflows (W3/W6/W7/W8/W10 cut; W4/W5 redesigned; W14–W19 added; model-optimized for cost):**

| # | Workflow | Models | Ground truth | Est. cost | Key patterns |
|---|----------|--------|-------------|-----------|-------------|
| W1 | Support agent | Haiku, Sonnet | 200 | $4 | Baseline |
| W2 | Support agent (complex) | Haiku, Sonnet, Opus | 500 | $182 | Loop variance, context growth, Opus validation |
| W4 | Compliance/document review (REDESIGNED) | DeepSeek V4, Qwen 3.6 Plus | 500 | $15 | Self-reflection loops, context growth |
| W5 | Multimodal extraction + structured output (REDESIGNED) | Sonnet 4.6 (vision) | 220 | $18 | Vision tokens, JSON mode |
| W9 | Sales/outreach (OpenAI) | GPT-5.4 | 200 | $4 | OpenAI generation pricing |
| W11 | Support (Qwen) | Qwen-Turbo, Qwen 3.6 Plus | 200 | $1 | Qwen pricing |
| W12 | Extraction (DeepSeek) | DeepSeek V4 Flash | 200 | $2 | DeepSeek pricing, cache |
| W13 | Routing agent | Haiku, Sonnet | 300 | $22 | Step count variance, bimodality |
| W14 | Simple PDF RAG + structured output | OpenAI embed + Sonnet | 300 | $38 | Retrieval variance, cross-provider |
| W15 | Agentic multi-hop PDF RAG | OpenAI embed + Gemini Flash + DeepSeek V4 | 500 | $55 | All 4 cost models simultaneously |
| W16 | Map-reduce PDF analysis | Haiku, Sonnet | 300 | $19 | Fan-out, variable N, parallel |
| W17 | Insurance claims agent | Haiku, OpenAI embed, Sonnet | 500 | $27 | Real-world decision tree, multi-doc RAG |
| W18 | Long-document single-pass (NEW) | DeepSeek V4 | 500 | $9 | Long context (50K–100K tokens), cost scaling |
| W19 | Multi-turn conversation, 8 turns (NEW) | DeepSeek V4 | 500 | $65 | Session accumulation, context growth across turns |

All DeepSeek workflows (W4/W12/W15/W18/W19) profiled in cache-cold mode. Anthropic workflows also cache-busted by default.

Provider coverage: Anthropic (W1/2/5/13/14/16/17), OpenAI generation (W9) + embeddings (W14/15/17), Gemini (W15), Qwen (W4/W11), DeepSeek (W4/W12/W15/W18/W19).

**Shared PDF pipeline (W14–W17):** Page-level classification → text extraction (pdfplumber, $0) or vision model (~$0.005–0.015/page) → semantic chunking → embedding.

**W19 converts a known limitation to a feature:** empirically calibrates a session multiplier (actual_8_turn_cost / single_turn_cost) that feeds the `--session-depth` CLI flag. Instead of documenting "multi-turn costs 2–3× more," ship with a validated correction factor.

**For each workflow, build 2 input datasets** (subsample from ground truth for n=50/100 accuracy curves via 200 random subsamples):

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| Synthetic-50 | 50 inputs | Auto-generated by Pretia (the default experience) | Tests the actual user experience |
| Ground-truth | 200–500 inputs | Curated or sourced from public datasets matching the archetype | Ground truth for calibration |

**Input distribution:** 35% easy / 30% medium / 20% hard / 10% edge / 5% adversarial. Skewed variants (80/15/5) for W2 and W4. W17: 40–50 scenario-based claims. W18: PDFs 30–100 pages. W19: 8-turn conversation scripts.

**Where to get realistic inputs for each archetype:**

| Archetype | Public data source | How to use |
|-----------|--------------------|-----------|
| Support agent | Bitext customer support dataset (HuggingFace, 27 intents, multi-turn), Ubuntu Dialogue Corpus | Extract user messages as inputs |
| Code review | GitHub PR descriptions from popular open-source repos (scrape via GitHub API, filter for PRs with 10-500 lines changed) | Use PR description + diff stats as input |
| Extraction / PDF RAG | CORD-19 papers, SEC EDGAR 10-K filings, Wikipedia infoboxes, public government PDFs | Use document excerpts or full PDFs as input |
| Research agent | Google Natural Questions (NQ) dataset, MS MARCO queries, HotpotQA (multi-hop questions) | Use questions as research queries |
| Sales/outreach | Crunchbase company descriptions (free tier), LinkedIn company profiles (public) | Use company name + description as lead input |
| Insurance claims | Synthetic claims JSON seeded from sample case study, expanded to 40–50 claims across 8 scenario types | Structured claim objects as input |

### 4B.3 The Validation Protocol

For each of the 14 workflows:

**Phase A — Profile at small sample sizes:**
```
pretia profile run W1.py --auto-generate 20 --output w1_synth20.json
pretia profile run W1.py --auto-generate 100 --output w1_synth100.json
pretia profile run W1.py --inputs w1_realistic_500.jsonl --output w1_real500.json
```

**Phase B — Extract projections:**
From each profile, extract the projected cost distribution at 1K/day traffic: p50, p75, p90, p95, p99, plus per-step breakdown.

**Phase C — Compute ground truth:**
The 500-run realistic profile IS the ground truth. Its distribution is the "actual" cost. Compute the actual p50, p75, p90, p95, p99.

**Phase D — Score the projections:**

```python
def score_projection(projected, actual_distribution):
    results = {}
    
    # 1. Calibration: is p50 near actual median?
    results["p50_ratio"] = projected.p50 / actual_distribution.p50
    # Ideal: 0.8 - 1.2 (within 20%)
    
    # 2. P95 coverage: does actual p95 fall below projected p95?
    results["p95_coverage"] = actual_distribution.p95 <= projected.p95
    # Ideal: True (at least 95% of actual runs are below our p95)
    
    # 3. Range usefulness: is p95/p50 ratio < 5x?
    results["range_ratio"] = projected.p95 / projected.p50
    # Ideal: < 5.0 (tight enough to be actionable)
    
    # 4. Directional accuracy: is the most expensive step correct?
    proj_top_step = max(projected.steps, key=lambda s: s.cost)
    actual_top_step = max(actual_distribution.steps, key=lambda s: s.cost)
    results["top_step_correct"] = proj_top_step.name == actual_top_step.name
    # Ideal: True
    
    # 5. Step ranking correlation
    proj_ranking = rank_steps_by_cost(projected)
    actual_ranking = rank_steps_by_cost(actual_distribution)
    results["ranking_correlation"] = spearmanr(proj_ranking, actual_ranking)
    # Ideal: > 0.8
    
    return results
```

**Phase D+ — Bootstrap ground truth CIs:**
Compute bootstrap 90% CIs on all calibration-relevant percentiles (1,000 bootstrap samples from ground truth data). Use the CI as the comparison target, not point estimates. This makes calibration statistically honest and prevents spurious failures from ground truth noise.

**Phase D++ — Directional bias meta-check:**
Track whether projected p95 is above or below ground truth p95 across all workflows. If 8+ out of 10 show the same direction, the engine has a systematic bias that individual workflow calibration wouldn't catch.

**Phase E — Pass/fail criteria (revised per adversarial review):**

| Metric | Pass | Warn | Fail |
|--------|------|------|------|
| p50 ratio (projected/actual) | 0.7 – 2.0× | 2.0 – 3.0× | >3.0× or <0.5× |
| p95 coverage | ≥85% simple / ≥75% complex | 60-75% | <60% |
| Range ratio (p95/p50) | <3× simple / <8× complex | 8-10× | >10× (useless) |
| Top step correct | Yes (30% co-dominant threshold) | — | No |
| Step ranking correlation | >0.7 for 4+ steps; dropped for ≤3 steps | 0.5–0.7 | <0.5 |

Rationale for changes: p50 underestimates are more harmful than overestimates for a budgeting tool (0.7 floor vs 0.5). Complex workflows genuinely have wider distributions (split thresholds). At n=20, steps within 30% of each other are statistically indistinguishable (wider co-dominant band). With 3 steps, only perfect ranking passes r>0.8 — tests noise, not engine quality.

**The launch gate:** All 14 workflows must pass p50 ratio AND top step correct (hard gates — core value propositions). At least 80% of workflows must pass each remaining metric. Tail metrics (p95 coverage, range ratio, ranking correlation) are diagnostic — they inform engine quality but don't individually block launch given the multiple-testing problem (65+ metric-workflow pairs).

### 4B.4 Expected failure modes and mitigations

**Failure: p50 underestimate on loop-heavy workflows (W2, W4, W8, W15).**
Root cause: 20 synthetic inputs don't trigger enough high-iteration runs to capture the tail. The average looks fine but the p95 is way off.
Mitigation: When the projection engine detects a loop step with high variance (CV > 0.5), it should automatically widen the confidence interval and add a warning: "Loop step detected with high variance. Projection may underestimate tail costs. Consider profiling with 50+ samples."

**Failure: directional miss on multi-model workflows (W4, W8).**
Root cause: Two steps use similar models (Sonnet vs Opus), and the relative cost depends on context size which varies by input. With 20 samples, the ranking might flip.
Mitigation: When two steps are within 20% of each other in cost share, flag both as "co-dominant cost drivers" instead of picking one winner.

**Failure: context growth overestimate.**
Root cause: Linear extrapolation of context growth from 3 observed iterations to 12 projected iterations. In reality, agents often plateau in context size (they learn to trim or the conversation converges).
Mitigation: Offer two context growth models — linear (conservative, current default) and logarithmic (assumes plateau after 2x the observed iterations). Report both projections and let the user see the range. If backtesting shows logarithmic is more accurate, make it the default.

**Failure: synthetic inputs are too "clean."**
Root cause: Auto-generated inputs from Haiku are grammatically perfect, moderate length, single-language. Production has typos, 2,000-word rants, code snippets, mixed languages, empty strings.
Mitigation: The input generator prompt explicitly requests diversity tiers including adversarial cases. Add a specific "chaos inputs" tier: empty string, single character, 5,000-word input, input in wrong language, JSON blob, SQL injection attempt. These stress-test the agent's error handling paths which are often the most expensive (retries, fallbacks).

### 4B.5 Confidence system: jackknife+ conformal prediction intervals

Replace the hand-tuned point-score confidence system with **jackknife+ conformal prediction intervals** (Barber et al. 2021, *Annals of Statistics*). Jackknife+ provides finite-sample coverage guarantees: targeting 1−α coverage, you get at least 1−2α for any sample size, any distribution, no model assumptions.

**How it works:** Uses leave-one-out residuals from the bootstrap. For each profiling run, compute the prediction error when that run is held out. The quantile of those residuals defines the prediction interval width. For monthly totals, propagate through CLT: monthly interval = N × point_estimate ± z × √N × per_run_SE.

**Tier derivation from interval width:**

| Interval width relative to p50 | Tier | Label |
|-------------------------------|------|-------|
| < 2× | HIGH | "projected" |
| 2–5× | MODERATE | "estimated" |
| 5–10× | LOW | "estimated (wide range)" |
| > 10× | VERY LOW | "order of magnitude" |

Report output: "Projected monthly cost: $4,100–$11,700 (90% prediction interval, MODERATE confidence)." The interval is ground truth; the tier is derived UX.

**Anti-gaming diagnostic:** The n_eff entropy gate (n_eff = n × H/H_max) remains as a separate warning for profiling uniformity detection, but no longer drives the tier. Jackknife+ intervals widen naturally for low-variance data.

**Stratified profiling analysis:** Tag runs with input complexity tier (easy/medium/hard/adversarial). Compute per-tier distributions. Mix with user-specified weights (`--traffic-mix`) or defaults (35/30/20/10/5). Enables post-profiling reweighting without re-running.

MAPIE Python library implements jackknife+, CV+, and split conformal. A from-scratch implementation is ~200 lines.

### 4B.5b Profiling tiers (revised)

n=20 is dropped entirely. The 36% probability of missing the true p95 is unacceptable, and n=20 doesn't give enough data for jackknife+ to produce useful intervals.

| Tier | Sample size | When to use |
|------|------------|-------------|
| **Standard** (default) | n=50 | Pre-deployment budgeting, cost estimation |
| **Budget grade** | n=100+ | CI regression detection, formal procurement |

### 4B.5c Three-layer validation strategy

1. **Layer 1 — Synthetic distribution testing** (zero API cost, 1–2 days): Generate 500+ synthetic "workflows" with known distributions (log-normal varying σ, bimodal varying separation/mixing, Pareto-tailed varying shape, zero-inflated varying trigger probability, uniform). Sample at n = 20/50/100/300. Feed into projection engine. Validates Monte Carlo math, tail inflation, pattern detection, and confidence calibration. Empirically calibrates the (1 + 2/√n) tail inflation factor. Run BEFORE spending API dollars.

2. **Layer 2 — SWE-bench trajectory analysis** (zero API cost, half day): Download per-instance execution trajectories with token usage from SWE-bench experiments. Group by repository for workflow-like cost distributions. Real-world heavy-tailed shapes without API cost. Limitation: different tasks, not repeated runs of one workflow.

3. **Layer 3 — Real workflow profiling** (~$1,093): The 14 workflows at 200–300 runs. Only layer validating complete pipeline including collector correctness, API response parsing, pricing accuracy, and PDF/vision processing.

| Aspect | Layer 1 | Layer 2 | Layer 3 |
|--------|---------|---------|---------|
| Monte Carlo math | Primary | ✓ | ✓ |
| Distribution shape coverage | 500+ shapes | Real shapes | 14 workflows |
| Data collection / pricing | — | — | Primary |
| End-to-end pipeline | — | — | Primary |

### 4B.6 Post-launch calibration tracking

After launch, every `pretia verify` call produces a (predicted, actual) pair. These accumulate and are used to:

1. **Compute global calibration metrics:** "Our p50 projections have a median error of ±22%." Published on the website monthly.
2. **Detect systematic biases:** If projections consistently underestimate loop-heavy workflows by 40%, adjust the loop projection model.
3. **Train the ML cost prediction model (v2):** The (workflow_structure, projected_cost, actual_cost) triples are training data.
4. **Produce per-archetype accuracy:** "For support agents, our projections are within ±15%. For multi-agent research workflows, ±40%." This helps users know how much to trust their specific projection.

### 4B.7 Budget and timeline

| Item | Cost | When |
|------|------|------|
| Build 14 test workflows (W1-W2, W4-W5, W9, W11-W19) + shared PDF pipeline | ~$0 (your time) | Sprint 3 |
| Curate inputs: PDF corpus, claims data, long docs, conversation scripts | ~$0 (public data) | Sprint 3 |
| Generate synthetic inputs (50 per workflow × 14 workflows) | ~$7 | Sprint 3 |
| Ground truth profiles (200-500 runs × 14 workflows, model-optimized, cache-cold) | **~$467** | Sprint 3 |
| W12 cache-warm comparison (50 runs) + pricing validation | ~$6 | Before/during backtesting |
| Skewed variants (W2, W4, W15) + contingency | ~$230 | During backtesting |
| **Total validation budget** | **~$697** | Pre-launch |

Model optimization saves 36% vs. $1,093 by swapping expensive models for DeepSeek/Qwen where provider-specific validation isn't needed. ~$253 headroom for cost estimation errors.

### 4B.8 The `pretia validate` command

A user-facing command that lets anyone validate the projection engine on their specific workflow:

```bash
pretia validate workflow.py --budget 10
# Runs the 20-vs-100 comparison test:
# 1. Profiles on 20 auto-generated inputs → projection A
# 2. Profiles on 100 auto-generated inputs → projection B  
# 3. Compares A vs B
# 4. Reports: "20-sample projection is within 18% of 100-sample projection. 
#              20 samples is sufficient for this workflow."
# OR: "Warning: 20-sample projection differs by 62% from 100-sample. 
#              This workflow has high variance. Use 50+ samples."
# Budget: ~$10 for both runs combined
```

This builds user trust — they can verify the engine's accuracy on their own data before relying on it for real cost decisions.

| Not in scope | Why | Who does it |
|---|---|---|
| TypeScript SDK | Python first. TS if demand proves out. | — |
| Cloud-hosted SaaS | Local only in v1. Cloud version is a v3+ consideration. | — |
| Live monitoring | v1 is offline profiling. Live monitoring = Phase 2. | Langfuse, Datadog |
| Proxy / Gateway | We sit above gateways, not replace them. | LiteLLM, Cloudflare |
| Model routing | We recommend which model. We don't route requests. | Martian, Unify |
| Context compaction | We detect when it's needed. We don't do it. | Morph |
| Quality evaluation | We do cost. Quality = Braintrust, DeepEval. | Braintrust |
| A/B quality testing | v1 recommends swaps. v2 classifier adds confidence. v3 might auto-test. | — |
| Pydantic AI / Mastra / Vercel AI SDK | Too many frameworks, too few hands. Cover 70-80% with LangGraph + OAI Agents + Qwen-Agent + DeepSeek (via OAI-compat). | — |
| Dynamic pricing tables | Static lookup. User updates config if provider changes prices. | — |

### Known Limitations (Document Prominently)

These are fundamental to the "profile a sample, project to production" methodology. They are not engine bugs. Document them prominently, not buried in footnotes.

1. **Session context accumulation.** Single-turn profiling underestimates multi-turn conversational agent costs by 2–3× for an 8-turn average session. Recommend profiling with mid-session context or use `--session-depth` correction (v2).
2. **Input distribution mismatch.** Projections are conditioned on the input distribution used during profiling. Input distribution statistics (token p50/p95/CV) make this assumption visible.
3. **Tool response size mismatch.** Test stubs or toy databases may produce 10–100× smaller tool responses than production. Every step downstream of a tool call will have understated context size.
4. **Provider-side caching.** Rapid profiling may benefit from prompt cache hits that production doesn't. DeepSeek cache-hit is 50× cheaper ($0.0028/MTok vs $0.14/MTok). Use `--cache-mode cold` for conservative estimates; parse `prompt_cache_hit_tokens` when available.
5. **Tiered pricing boundaries.** v1 uses standard-tier pricing. Qwen requests exceeding 200K input tokens and Gemini requests exceeding 200K context may be underpriced. Warn when observed context exceeds 150K tokens.
6. **Model drift.** Providers update models within version lines. Token counts and costs may shift 5–25% between updates. Profiles older than 30 days may not reflect current behavior.
7. **Batch vs. real-time pricing.** v1 uses real-time API pricing. Users deploying via batch APIs (OpenAI Batch, Anthropic Message Batches) will see ~50% lower actual costs than projected.

### Pre-Backtesting Engineering Checklist (~25–29 days total)

**Statistical methods (A1–A6):**

1. Implement jackknife+ conformal prediction intervals — A1 (2-3 days)
2. Implement BIC-based GMM bimodality detection + MC integration — A2 (1 day)
3. Implement BCa bootstrap (replaces ad-hoc tail inflation) — A3 (half day)
4. Implement CVaR computation from Monte Carlo samples — A4 (half day)
5. Implement stratified profiling analysis with `--traffic-mix` flag — A5 (2 days)
6. Replace CV with robust_cv (MAD/median) in all detectors — A6 (1 day)

**Engine fixes (from v1 adversarial review):**

7. Fix CLT variance inflation in Monte Carlo — §1.1 (half day)
8. Implement dual Pearson+Spearman context growth detection with p-value gating — §2.1 (2 hrs)
9. Implement step count variance detector — §2.4 (1-2 hrs)
10. Cap context growth extrapolation at observed max — §1.4 (1 hour)
11. Implement n_eff entropy gate as diagnostic — §3.5 (1 hr)
12. Build synthetic distribution testing framework (Layer 1) — §5B (1-2 days)
13. Download and parse SWE-bench trajectory data (Layer 2) — §5B (half day)

**Data collection & validation:**

14. Write mock response unit tests for all collectors (DeepSeek cache, Qwen DashScope, vision, parallel, structured output, per-turn) — §4.1 (1 day)
15. Validate pricing table against billing dashboards — §4.2 (2-3 hrs, ~$5)
16. Implement cache-busting for DeepSeek + Anthropic profiling runs — §4.3 (3 hrs)

**Data schema (B1–B4):**

17. Add B1 fields to StepRecord (tool details, cache, model_version, truncation) — (half day)
18. Add B2 fields to RunRecord (active_step_list, loop_exit_reason, complexity_tier) — (half day)
19. Add B3 fields to ProfileSession (environment snapshot, git context) — (half day)
20. Add B4 fields to WorkflowRecord (fingerprint, graph topology, step_model_map) — (1 day)

**Workflow engineering:**

21. Build W14 simple PDF RAG + shared PDF pipeline (1 day)
22. Build W15 agentic multi-hop RAG (DeepSeek generation) (1-2 days)
23. Build W16 map-reduce (half day)
24. Build W17 insurance claims agent (1-2 days)
25. Build W18 long-document single-pass (half day)
26. Build W19 multi-turn conversation, 8 turns with per-turn recording (1 day)
27. Redesign W5 multimodal + structured output (half day)
28. Redesign W4 as compliance review with DeepSeek+Qwen (half day)
29. Build W13 routing workflow (1-2 days)
30. Curate shared PDF corpus (30-50 PDFs, including 30+ page docs for W18) (half day)
31. Prepare skewed-distribution input sets for W2, W4 (2-3 hrs)

**Calibration setup:**

32. Drop n=20 tier; set n=50 default, n=100 budget grade (1 hr)
33. Implement subsampling calibration: 200 subsamples at n=50 and n=100 per workflow (half day)
34. Implement tightened n=50 calibration thresholds (2 hrs)
35. Implement `--session-depth` flag (half day)
36. Implement `--cache-mode cold/warm` flag (half day)

---

## PART 6 — DEVELOPMENT PLAN

### Pre-development (Week 0 — before writing a single line of code)

**Day 1-2: Set up the foundation**
- Register `pretia` on PyPI (claim the name)
- Create GitHub repo with MIT license, README skeleton, contributing guide
- Set up CI (GitHub Actions for tests + linting)
- Set up project structure:
  ```
  pretia/
  ├── pretia/
  │   ├── __init__.py
  │   ├── collectors/
  │   │   ├── base.py          # StepRecord + BaseCollector
  │   │   ├── langgraph.py     # LangGraphCollector
  │   │   ├── openai_agents.py # OpenAIAgentsCollector
  │   │   ├── qwen_agent.py    # QwenAgentCollector (LLM client wrapping)
  │   │   └── generic.py       # GenericCollector (decorator/ctx manager)
  │   ├── projection/
  │   │   ├── stats.py         # Distribution calculations
  │   │   ├── montecarlo.py    # Non-linear projection
  │   │   └── patterns.py      # Context growth, loop variance detection
  │   ├── recommend/
  │   │   ├── heuristics.py    # v1 rule-based recommendations
  │   │   ├── classifier.py    # v1.5 ML classifier (loads .pkl)
  │   │   ├── rules.py         # Recommendation types + formatting
  │   │   ├── prompts.py       # Generate implementation prompts per framework
  │   │   └── verify.py        # Compare old/new profiles, check if recs were applied
  │   ├── ci/
  │   │   ├── baseline.py      # Baseline management
  │   │   ├── diff.py          # Baseline comparison
  │   │   └── report.py        # PR comment formatting
  │   ├── inputs/
  │   │   ├── generator.py     # LLM-powered synthetic input generation
  │   │   ├── importer.py      # Langfuse/Braintrust trace import
  │   │   ├── schema.py        # Extract input schema from workflow code
  │   │   └── selector.py      # Auto-detect best input mode
  │   ├── validation/
  │   │   ├── suite.py         # Backtesting protocol runner
  │   │   ├── scoring.py       # Calibration scoring (p50 ratio, coverage, ranking)
  │   │   ├── validate_cmd.py  # `pretia validate` command (20-vs-100 test)
  │   │   └── confidence.py    # Confidence tier computation
  │   ├── models/               # Shipped ML models (.pkl files)
  │   ├── pricing/
  │   │   └── tables.py        # Static model pricing lookup
  │   ├── store.py             # ProfileStore (JSON/SQLite)
  │   ├── report/
  │   │   ├── template.html.j2 # Jinja2 HTML report template
  │   │   ├── renderer.py      # Render profile data → HTML file
  │   │   ├── charts.py        # Inline SVG chart generators (waterfall, sparklines)
  │   │   └── graph.py         # DAG graph renderer (before/after SVG)
  │   ├── graph/
  │   │   ├── extractor.py     # Extract DAG structure from LangGraph/OAI/CrewAI
  │   │   ├── layout.py        # Position nodes (simple top-to-bottom)
  │   │   ├── colorizer.py     # Map cost shares to node colors
  │   │   └── transform.py     # Apply recommendations → produce "after" graph
  │   ├── ui/
  │   │   ├── app.py           # FastAPI app (REST + WebSocket)
  │   │   ├── static/          # Pre-built React bundle (~200KB)
  │   │   └── ws.py            # WebSocket handler for live profiling
  │   └── cli.py               # Click-based CLI
  ├── ui-frontend/              # React source (built separately, output → pretia/ui/static/)
  │   ├── src/
  │   ├── package.json
  │   └── vite.config.ts
  ├── action/                   # GitHub Action
  │   ├── action.yml
  │   └── entrypoint.sh
  ├── tests/
  ├── pyproject.toml
  └── README.md
  ```

**Day 3-5: Build the test harness FIRST**
- Create 3 minimal agent workflows for testing:
  - A LangGraph support agent (3 steps, no loop) — simplest case
  - A LangGraph review agent (5 steps, 1 loop) — tests context growth
  - An OpenAI Agents SDK tool-calling agent (2 agents, handoff) — tests multi-agent
- These are your integration test fixtures AND your demo workflows
- Write the expected StepRecord output for each (test-driven)
- Estimated cost to run tests: ~$2-5 per full suite

**Day 5: Validate framework APIs haven't changed**
- Run each test workflow manually, confirm callbacks/hooks fire as documented
- LangGraph: `UsageMetadataCallbackHandler` still works
- OpenAI Agents SDK: `result.context_wrapper.usage.request_usage_entries` still exists
- Document any API drift since the docs you read

### v1.0 Development (Weeks 1-20)

#### Sprint 1: Collectors + Input Generator (Weeks 1-2)

**Goal:** `pretia profile run workflow.py` works for LangGraph with auto-generated inputs (no JSONL file needed).

| Day | Task | Output |
|-----|------|--------|
| 1-2 | `StepRecord` dataclass + `ProfileStore` (JSON) | Data layer |
| 3-4 | `LangGraphCollector`: hook into LangChain `UsageMetadataCallbackHandler`, map callbacks → StepRecords. Handle: `on_llm_start` (capture model name, prompt tokens), `on_llm_end` (capture output tokens, total), `on_tool_start/end` (capture tool calls) | LangGraph instrumentation |
| 5-6 | `GenericCollector`: `@collector.step("name")` decorator + `with collector.step("name") as s` context manager. Auto-extract tokens from OpenAI/Anthropic response objects | Manual instrumentation |
| 7-8 | **Input generator**: extract system prompt + input schema from workflow code → generate diverse inputs via Haiku/mini call (~$0.02). Support: `--auto-generate N` (default), `--input "..."` (single), `--inputs file.jsonl` (manual). CLI skeleton (Click): `pretia profile run`, `pretia estimate`, `pretia report` | Input generation + CLI |
| 9-10 | Integration tests with the 3 fixture workflows. Test all input modes. Verify StepRecords match expected output | Test coverage |

**Deliverable:** `pretia profile run workflow.py` auto-generates 20 inputs, profiles the workflow, outputs per-step token counts. Two commands from install to first report.

#### Sprint 2: OpenAI Agents SDK + Stats + Langfuse Import (Weeks 3-4)

**Goal:** Multi-framework support + cost calculations + trace import.

| Day | Task | Output |
|-----|------|--------|
| 1-3 | `OpenAIAgentsCollector`: hook via `RunHooks` (`on_agent_start/end`, `on_tool_start/end`). Use `context_wrapper.usage.request_usage_entries` for per-request breakdown. Map to StepRecords | OAI Agents instrumentation |
| 4-5 | Pricing tables module: static lookup for all major models (Anthropic, OpenAI, Google, DeepSeek, Qwen, Llama, Mistral). Covers DeepSeek V4 generation (Flash + Pro) with cache-miss pricing and Qwen 3.x family (Max, Plus, Turbo). Token count × price = cost. Update mechanism: JSON config file user can override | Cost calculation |
| 6-7 | Stats module: per-step distributions (p50, p75, p90, p95, p99). Detect patterns: context growth (r² correlation), loop variance (coefficient of variation), high variance (p95/p50 > 3) | Statistical analysis |
| 8-9 | **Langfuse trace importer**: `--from-langfuse --last N` pulls traces via Langfuse Python SDK (`langfuse.api.observations.get_many`). Extract per-step token data from existing traces WITHOUT re-executing. Also: `pretia analyze --from-langfuse` mode for zero-cost analysis of production data | Trace import |
| 10 | `pretia report profile.json --traffic 1000/day` → markdown report with per-step breakdown, flags, totals. Auto-detect best input mode (Langfuse available? → suggest import. Otherwise → auto-generate) | Report generation + smart defaults |

**Deliverable:** Can profile LangGraph or OAI Agents workflows via auto-generate, single input, manual file, OR Langfuse import. Compute costs, detect patterns, generate reports.

#### Sprint 3: Projection Engine + Backtesting Suite (Weeks 5-8)

Sprint 3 is expanded to 4 weeks. Weeks 5-6 build the projection engine. Weeks 7-8 build the backtesting suite and run it. The projection engine does NOT ship without passing the backtesting suite. This is the kill/keep gate.

**Goal:** Accurate cost projections with jackknife+ conformal prediction intervals, validated via three-layer strategy (synthetic distributions + SWE-bench + 14 model-optimized real workflows). ~25–29 days of pre-backtesting engineering (see checklist in §4B) must pass before running the ~$697 backtesting suite.

| Day | Task | Output |
|-----|------|--------|
| 1-3 | Linear projector with distributional output. Confidence tier system using effective sample size (n_eff via cost entropy). **Engine fixes:** CLT-corrected Monte Carlo, dual Pearson+Spearman context growth detection with p-value gate, tail inflation factor (1 + 2/√n), context growth cap at observed max, p90/p50 threshold at n < 30 | Basic projection + fixes |
| 4-7 | Monte Carlo projector with CLT correction + 5 pattern detectors (context growth, loop variance, token variance, step count variance, cost bimodality). Non-pattern steps sampled from whole runs. Visibility warnings: input distribution stats, uniform-input flags, zero-execution step detection, stale profile warnings, sample coverage statement, p95 suppression at n < 20. Tiered profiling recommendations (n=20/50/100+) | Non-linear projection + detectors + warnings |
| 8-10 | Baseline management with Mann-Whitney U significance testing + bootstrap 95% CI on cost change. **Synthetic distribution testing framework** (Layer 1): 500+ synthetic workflows across log-normal/bimodal/Pareto/zero-inflated shapes, validate engine at n=20/50/100/300. Calibrate tail inflation factor empirically. **SWE-bench trajectory analysis** (Layer 2): download, parse, integrate as additional test distributions | Baseline + synthetic validation |
| 11-13 | **Build 14 test workflows** (W1-W2, W4-W5, W9, W11-W19). W4 redesigned as compliance review (DeepSeek+Qwen). W5 multimodal. W14-W17: PDF RAG/claims. W18: long-document single-pass (DeepSeek, 50K-100K tokens). W19: 8-turn multi-turn conversation (DeepSeek, per-turn recording). Cache-busting for all DeepSeek + Anthropic runs. Mock collector tests for vision, parallel, structured output, per-turn recording. Pricing validation. PDF corpus curation | Test infrastructure + data validation |
| 14-16 | **Run real backtesting** (Layer 3): 50-sample projections against 200-500 run ground truth for 14 workflows. Subsampling calibration (200 subsamples at n=50 and n=100). BCa bootstrap CIs on ground truth. Tightened n=50 metrics (0.8-1.7× p50, 25% co-dominant). Conformal interval coverage. CVaR accuracy. Directional bias meta-check. Skewed variants. W12 cache-warm comparison. Session multiplier from W19. **Budget: ~$467 + $230 contingency** | Calibration results |
| 17-18 | **Fix failures.** Hard gates: p50 ratio AND top step correct for all 14 workflows. 80% pass rate for remaining metrics. Calibrate `--session-depth` from W19. Calibrate `--cache-mode` discount from W12. Plot calibration curves (accuracy vs sample size) | Validated engine |
| 19-20 | `pretia validate workflow.py` command: 20-vs-100 comparison test. Zero-token step validation. Unrecognized model error handling. Tiered profiling in CLI output | User-facing validation |

**Deliverable:** Projection engine validated across 500+ synthetic distributions + 14 model-optimized real-world archetypes (including PDF RAG, long-context, multi-turn sessions, and insurance claims) with jackknife+ conformal intervals, CLT-corrected Monte Carlo, 5 pattern detectors (GMM bimodality with cost model), BCa bootstrap, CVaR, stratified analysis, and `--session-depth` / `--cache-mode` / `--traffic-mix` CLI flags. Launch gate: PASSED.

#### Sprint 4: Recommendations + Verify Loop (Weeks 9-12)

Sprint 4 is expanded to 4 weeks because the implementation prompts and verification loop are critical for product stickiness — without them, recommendations are just text. With them, Pretia becomes an optimization loop users run until their score is green.

**Goal:** Actionable, dollar-denominated optimization suggestions with copy-paste implementation prompts and a verification cycle.

| Day | Task | Output |
|-----|------|--------|
| 1-3 | Model swap recommender: task type heuristics (keyword scan on system prompt + output/input ratio) → minimum tier estimate → savings calculation. Conservative: only recommend when clear signal (classification, extraction, simple formatting) | Model swap recs |
| 4-5 | Architecture recommender: context growth detection → compaction recommendation with savings. Re-sent context detection (hash first 200 tokens of consecutive prompts) → caching recommendation | Architecture recs |
| 6-7 | Workflow recommender: loop analysis → iteration cap recommendation. Stuck loop detection (>2x mean iterations) → circuit breaker recommendation. Cost-per-iteration-marginal analysis | Workflow recs |
| 8-9 | Recommendation formatting: each rec has type, title, description, estimated savings, confidence level. Aggregate: total potential savings. Score calculation (100 - waste%). Integrate into `pretia report` and `pretia recommend` | Formatted output |
| 10-12 | **Tier 1 prompts (simple):** model swap template — mechanical, fill from StepRecord. **Tier 2 prompts (medium):** iteration cap — framework-specific patterns (LangGraph conditional edges, OAI Agents loop guards). **Tier 3 prompts (complex):** compaction insertion — include full graph structure as text, framework-specific node insertion code, state wiring. Each template includes `_framework_specific_*` helper functions for LangGraph, OAI Agents SDK, CrewAI | Implementation prompts |
| 13-14 | **Graph-as-text renderer:** reuse the graph extractor to produce a text representation of the workflow (e.g. `START → classify → retrieve → generate → review ↺ → format → END`) for inclusion in Tier 3 prompts. The coding AI needs to see the graph to modify it safely | Prompt context |
| 15-16 | **Verify command:** `pretia verify --baseline .pretia/baseline.json` re-profiles the workflow, compares to baseline, checks which recommendations were applied (model changed? new node exists? iteration count decreased?), shows actual vs predicted savings, updates score | Verify loop |
| 17-18 | **Verification report format:** green checkmarks for applied recs with actual savings vs predicted, remaining recs with "not yet applied", new score, and overall before→after summary. Every verify result is logged as a training signal for future classifier versions | Verify report |
| 19-20 | Integration tests: full recommend → implement (manually) → verify cycle on the 3 fixture workflows. Ensure score improves after each recommendation is applied. Test that prompts produce valid code when pasted into Claude Code | End-to-end test |

**Deliverable:** `pretia recommend` produces dollar-denominated recommendations with tiered implementation prompts. `pretia verify` checks what was implemented and shows the improvement. The "optimize until green" loop works end-to-end.

#### Sprint 5: GitHub Action (Weeks 13-14)

**Goal:** Cost checks in CI on every PR.

| Day | Task | Output |
|-----|------|--------|
| 1-2 | `static` mode: parse git diff, detect model changes / step additions / loop modifications in workflow code, estimate impact from baseline. No LLM calls needed | Free CI mode |
| 3-4 | `auto-generate` mode: generate 5 synthetic inputs, run workflow, compare to baseline | Standard CI mode |
| 5-6 | `sample` mode: run workflow on 5-10 inputs from JSONL, measure real costs. `import` mode: pull Langfuse traces, analyze without re-execution | Paid + free CI modes |
| 7-8 | PR comment: format cost report + diff + recommendations as GitHub comment. Include implementation prompts in collapsed `<details>` blocks. Respect threshold config (block merge if cost increase > X%) | GitHub integration |
| 9-10 | `action.yml` + `entrypoint.sh`. Test with a real GitHub repo. Publish to GitHub Marketplace | Shipped Action |

**Deliverable:** Working GitHub Action in four modes, published on Marketplace.

#### Sprint 6: HTML Report + Graph View + Local Web UI (Weeks 15-18)

Sprint 6 is extended to 4 weeks (from 2) to accommodate the graph visualization. This is the visual layer that makes the product demo-ready.

**Goal:** Visual output that makes Pretia feel like a product, not a script.

| Day | Task | Output |
|-----|------|--------|
| 1-2 | **Graph extractor:** parse LangGraph `.get_graph()` to extract nodes + edges + metadata. Map StepRecords to nodes. Cost-to-color: <10% share = white, 10-40% = amber, >40% = red | Graph data |
| 3-4 | **Graph layout + SVG renderer:** simple top-to-bottom node positioning. Bezier loop-back edges for iterations. Node badges for recommendations. Edge annotations for token counts. Generate SVG server-side (Jinja2) | Before graph |
| 5-6 | **"After" graph transformer:** apply each recommendation to the graph structure (insert compaction node, change model name, update loop annotation, recompute costs). Re-render as green SVG. Tabbed before/after with summary banner | After graph |
| 7-8 | HTML report template (Jinja2): score ring, 4 metric cards, cost waterfall (inline SVG bars), recommendation cards, before/after graph tabs, projection table. Single self-contained .html file. `pretia profile run` auto-opens it | HTML report |
| 9-10 | Context growth sparklines, raw data toggle, footer. Polish CSS. Test with demo workflows | Report complete |
| 11-12 | FastAPI backend for `pretia ui`: serve pre-built React frontend, REST endpoints for workflow detection + input config, WebSocket endpoint for live profiling progress | UI backend |
| 13-14 | React frontend Screen 1 (Setup): file selector, framework detection, input mode picker, traffic config, "Run Profiling" button | UI screen 1 |
| 15-16 | React frontend Screen 2 (Live): progress bar, per-step cost accumulating via WebSocket, pattern flags | UI screen 2 |
| 17-18 | React frontend Screen 3 (Report): graph view as primary tab (before/after), interactive waterfall + recommendations. Export buttons | UI screen 3 |
| 19-20 | Pre-compile React to static bundle, include in pip package. End-to-end test: CLI → HTML report → UI all rendering the same data | UI complete |

**Deliverable:** `pretia profile run` opens an HTML report with the before/after architecture graph. `pretia ui` launches a full visual experience with live profiling and interactive graph.

#### Sprint 7: Polish + Launch (Weeks 19-20)

| Day | Task | Output |
|-----|------|--------|
| 1-2 | CrewAI Collector (same LangChain callback pattern — should be quick). Qwen-Agent and DeepSeek compatibility already shipped (Sprint 2b/2c) | Additional frameworks |
| 3-4 | End-to-end integration tests: profile → report → recommend → baseline → diff → CI comment → HTML report → UI | Full pipeline tested |
| 5-6 | README: clear quickstart, architecture diagram, demo GIF (record both CLI and UI experiences). Screenshots of the HTML report for social sharing | Documentation |
| 7-8 | One-page landing website (GitHub Pages). Screenshot gallery showing the report, the UI, the PR comment | Web presence |
| 9 | Publish v1.0 to PyPI. GitHub Action to Marketplace. Tag release | v1.0 shipped |
| 10 | Launch: HN "Show HN", LangChain Discord, r/MachineLearning, Twitter/X, DEV.to blog post | Distribution |

### v1.5 Development (Weeks 21-24)

#### Sprint 8: Task Complexity Classifier

| Day | Task | Output |
|-----|------|--------|
| 1-2 | Download RouterBench (405K) + LLMRouterBench (400K). Write label derivation: for each prompt, cheapest model scoring ≥90% of best → tier label (tier-1/2/3) | 800K labeled examples |
| 3-4 | Generate 200 synthetic agent-style system prompts (40 per archetype: support, code review, extraction, research, sales). Run each through 3 model tiers × 5 inputs. Score with LLM-as-judge. **Cost: ~$150** | 1K domain-specific examples |
| 5-6 | Feature pipeline: `all-MiniLM-L6-v2` embedding (384d) + 9 numerical features (prompt tokens, output/input ratio, has_json_schema, etc.) = 393 features. Train logistic regression. Evaluate on held-out 20%. If accuracy <80%, try XGBoost | Trained classifier |
| 7-8 | Integrate into `pretia recommend`: replace keyword heuristics with classifier prediction + confidence score. Confidence < 0.85 → fall back to heuristics with note. Serialize model as .pkl, ship in pip package (~2MB) | ML-powered recs |
| 9-10 | Feedback capture: when user changes model at a step and re-profiles, log (prompt_hash, old_model, new_model, quality_maintained: bool). This trains future classifier versions | Feedback loop |

**Deliverable:** v1.5 shipped. Model swap recommendations backed by classifier trained on 800K+ examples.

### v2.0 Development (Weeks 25-34)

Two parallel tracks: live monitoring + cost prediction model.

#### Track A: Live Monitoring (Weeks 25-30)

| Week | Task |
|------|------|
| 25-26 | `pretia monitor` daemon: same Collectors but in production mode. Stream StepRecords to local SQLite or cloud store. Real-time cost accumulation per workflow/agent/team |
| 27-28 | Delta dashboard: projection (from baseline) vs reality (from monitor). Show: "Step 4 costs 3.2x more than projected because retry rate is 18% in production vs 2% in profiling" |
| 29-30 | Budget alerts: configurable thresholds per workflow/team. "Agent support-bot has spent $1,200 this week (budget: $1,000). At current rate, will exceed monthly budget by Tuesday." First paid tier: $500/mo for 10 workflows |

#### Track B: Cost Prediction Model (Weeks 25-30)

| Week | Task |
|------|------|
| 25-26 | Collect all profiling data from v1/v1.5 users. Generate 100 synthetic workflows across 5 archetypes ($200). Build feature extraction: workflow code → feature dict (20 features) |
| 27-28 | Train XGBoost on log(cost) targets. Cross-validate. Compare vs baseline (archetype lookup table). Ship only if MAPE improvement >15% |
| 29-30 | `pretia estimate workflow.py` — instant cost estimate from code structure, no profiling. Integrate into `static` CI mode for better accuracy |

#### Weeks 31-34: Integration + Paid Launch

| Week | Task |
|------|------|
| 31-32 | Integrate monitoring + prediction into the Web UI dashboard. Paid tier launch |
| 33-34 | First 10 paying customers. Iterate based on feedback. Blog: "How we predict agent costs before deployment" |

### v3.0 Development (Weeks 35-48)

| Phase | Task |
|-------|------|
| Weeks 35-38 | Spend Governance: budget circuit breakers per-agent (intelligent degradation: downshift model → compact context → halt gracefully). Attribution by team/workflow/client |
| Weeks 39-42 | Audit trail: every agent spend decision logged with identity, scope, context, timestamp. EU AI Act compliance report generator |
| Weeks 43-46 | Workflow Benchmarking: HDBSCAN clustering on accumulated workflow data. Benchmark reports: "your cost vs cluster median." Requires ~500 workflows minimum |
| Weeks 47-48 | Enterprise pilot program: 3-5 companies on Spend Governance tier ($5K+/mo). Iterate on compliance/audit features based on their requirements |

---

## PART 7 — KEY MILESTONES AND SUCCESS METRICS

| Milestone | When | Metric | Why it matters |
|-----------|------|--------|---------------|
| Backtesting suite passes (14/14 workflows) | Week 8 | p50 ratio + top step correct for all; 80% pass on remaining; conformal coverage ≥85% | Predictions are trustworthy |
| v1.0 shipped (with UI + graph) | Week 20 | Package on PyPI + Action on Marketplace + UI | Product exists |
| First 10 design partners | Week 22 | 10 teams using the SDK regularly | Validation |
| 100 GitHub stars | Week 22 | Community signal | Social proof for YC |
| v1.5 shipped (ML classifier) | Week 24 | Classifier accuracy >80% on held-out | ML moat begins |
| 1,000 SDK installs | Week 28 | PyPI download count | Distribution |
| 500 workflows profiled | Week 32 | ProfileStore telemetry | Data moat begins |
| First paid customer | Week 32 | $500 MRR | Revenue |
| 20 paid teams | Week 44 | $10K MRR | Business viability |
| 3 enterprise pilots | Week 48 | $15K+ MRR pipeline | Enterprise traction |
| Benchmark data meaningful | Week 44 | >500 workflows, >5 clusters with 15+ members | Network effect begins |

---

## APPENDIX — COST BUDGET

| Item | Cost | When |
|------|------|------|
| Domain + hosting (GitHub Pages) | ~$12/year | Week 0 |
| Test workflow execution (ongoing) | ~$50/month | Ongoing |
| Synthetic distribution testing (500+ shapes) + SWE-bench trajectory analysis | $0 (engineering time) | Week 7 |
| **Backtesting suite: ground truth profiling (14 workflows × 200-500 runs, model-optimized)** | **~$467** | Week 7-8 |
| Cache-warm comparison + pricing validation + skewed variants + contingency | ~$230 | Week 7-8 |
| Synthetic data for ML classifier | ~$150 | Week 21 |
| Synthetic workflows for cost model | ~$200 | Week 25 |
| **Total pre-revenue investment** | **~$1,550** | Weeks 0-25 |
