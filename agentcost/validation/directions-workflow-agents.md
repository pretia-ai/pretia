# Directions: Workflow Agent Implementation for All 14 Backtesting Workflows

**Purpose:** This file specifies how to build the 14 workflow agent implementations — the orchestration code that takes an input, runs it through the correct steps with the correct models, handles control flow (loops, routing, fan-out, multi-turn), and records cost data as StepRecords. These agents are what gets profiled and backtested.

**Context for Claude Code:** You have the AgentCost codebase (including the projection engine, the collector, the StepRecord schema), the generated system prompts in `prompts/`, `projection-engine-recommendation-addition-2.md` (engine design, W17 architecture), `cross-cutting-robustness.md`, and the technical spec. This file tells you how to build the workflow orchestrators that connect inputs to prompts to API calls to cost data.

**What you are building:** A runnable agent per workflow that the AgentCost collector can invoke. Each agent takes one input (JSON), executes all workflow steps, and returns a list of StepRecords. The harness runs N inputs through an agent and feeds the StepRecords to the projection engine.

---

## Part 1: Architecture

### Three Layers

```
┌─────────────────────────────────────────────────────┐
│  HARNESS (shared)                                   │
│  Loads inputs, runs workflow, collects StepRecords   │
│  python run_workflow.py --workflow W1 --profile profiling --n 50
│  └──────────────────────────────────────────────────│
│  WORKFLOW AGENTS (per-workflow)                      │
│  Orchestration logic: step ordering, loops, routing  │
│  agents/w01_support_simple.py                       │
│  └──────────────────────────────────────────────────│
│  PROVIDERS (shared)                                  │
│  API wrappers: call model, return response + usage   │
│  providers/anthropic.py, openai.py, deepseek.py...  │
└─────────────────────────────────────────────────────┘
```

### StepRecord Schema

Each API call produces one StepRecord. The projection engine consumes these. Use the schema from the codebase — it should include at minimum:

```python
@dataclass
class StepRecord:
    workflow_id: str          # "W1", "W2", etc.
    run_id: str               # Unique per input execution
    step_name: str            # "intake_classify", "research_draft_loop_iter_3"
    step_index: int           # Sequential order within this run
    model: str                # "claude-haiku-4.5", "deepseek-v4", etc.
    provider: str             # "anthropic", "openai", "deepseek", "qwen", "google"
    input_tokens: int         # From API response.usage
    output_tokens: int        # From API response.usage
    cost_usd: float           # Computed: input_tokens × input_price + output_tokens × output_price
    latency_ms: int           # Wall-clock time for this API call
    cache_hit_tokens: int     # From API response (DeepSeek: prompt_cache_hit_tokens)
    finish_reason: str        # "stop", "length", "tool_use", etc.
    metadata: dict            # Workflow-specific: iteration number, routing decision, etc.
```

**Critical:** `input_tokens` and `output_tokens` must come from the API response's `usage` field, not from local tokenization. Local token counts will differ from the provider's count, and cost is billed on the provider's count.

### Provider Layer: LiteLLM

Use **LiteLLM** (`pip install litellm`) as the unified provider interface. It wraps all 5 providers behind one `completion()` call, handles message formatting differences, auth, and retries. This eliminates writing 5 separate provider wrappers.

```python
import litellm
from litellm import completion
import uuid, time

def call_model(model, system_prompt, messages, max_tokens=4096, tools=None) -> ProviderResponse:
    # 1. Cache-bust: replace {{CACHE_BUST_SUFFIX}} with a fresh UUID
    prompt = system_prompt.replace("{{CACHE_BUST_SUFFIX}}", str(uuid.uuid4()))
    
    # 2. Build the messages array (LiteLLM uses OpenAI format for all providers)
    full_messages = [{"role": "system", "content": prompt}] + messages
    
    # 3. Call via LiteLLM — it routes to the correct provider based on model name
    start = time.monotonic()
    response = completion(
        model=model,            # e.g. "anthropic/claude-haiku-4.5", "deepseek/deepseek-v4"
        messages=full_messages,
        max_tokens=max_tokens,
        tools=tools,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    
    # 4. Extract usage — LiteLLM normalizes this across providers
    usage = response.usage
    return ProviderResponse(
        content=response.choices[0].message.content,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_hit_tokens=getattr(usage, 'prompt_cache_hit_tokens', 0),  # DeepSeek-specific
        finish_reason=response.choices[0].finish_reason,
        latency_ms=latency_ms,
        raw_response=response,
    )
```

**Why LiteLLM, not raw SDKs:**
- One interface for Anthropic, OpenAI, DeepSeek, Qwen, and Google. No per-provider message formatting logic.
- Usage data (prompt_tokens, completion_tokens) returned in a consistent format across all providers.
- Built-in retry logic with exponential backoff (configure via `litellm.num_retries`).
- Model name routing: `"anthropic/claude-haiku-4.5"`, `"openai/gpt-5.4-nano"`, `"deepseek/deepseek-v4"`, `"qwen/qwen-3.6-plus"`, `"gemini/gemini-2.5-flash"`.

**Why not LangGraph/LangChain for orchestration:**
The workflow graphs are fixed and known at build time — a loop with a termination condition, a router with 3 paths, a fan-out. Custom Python with `asyncio` for W16 parallel calls is simpler and gives full control over StepRecord creation at every API call boundary. Agent frameworks tend to abstract away the exact things we need to measure (per-call token counts, cache behavior, per-step cost).

**However:** The existing AgentCost codebase was built around LangGraph. If the codebase already has LangGraph patterns, state management, and graph definitions, it may be more consistent to build the workflow agents as LangGraph graphs rather than introducing a parallel custom orchestration system. LangGraph's `StateGraph` maps naturally to these workflow patterns (loops via conditional edges, routing via branching, fan-out via `Send`), and its callback system can capture per-node usage data for StepRecords.

**Let Claude Code decide.** It has access to the codebase and can see which approach integrates better. The non-negotiable requirement either way: every API call must produce a StepRecord with `input_tokens`, `output_tokens`, `cache_hit_tokens`, and `cost_usd` extracted from the provider's usage response — not from local tokenization, not from framework estimates. If LangGraph's callback/event system exposes the raw API response usage, use it. If it abstracts it away, add a hook or wrapper that captures it.

**Provider-specific overrides:** If LiteLLM doesn't expose a provider-specific field you need (e.g., DeepSeek's `prompt_cache_hit_tokens`), access it via `response._raw_response` or `response.model_extra`. Write a thin extraction layer on top of LiteLLM rather than bypassing it.

**LiteLLM model strings for this project:**

| Provider | Model | LiteLLM model string |
|---|---|---|
| Anthropic | Haiku 4.5 | `anthropic/claude-haiku-4.5` |
| Anthropic | Sonnet 4.6 | `anthropic/claude-sonnet-4.6` |
| Anthropic | Opus 4.7 | `anthropic/claude-opus-4.7` |
| OpenAI | GPT-5.4 Nano | `openai/gpt-5.4-nano` |
| OpenAI | GPT-5.4 | `openai/gpt-5.4` |
| OpenAI | text-embedding-3-small | `openai/text-embedding-3-small` |
| DeepSeek | V4 | `deepseek/deepseek-v4` |
| DeepSeek | V4 Flash | `deepseek/deepseek-v4-flash` |
| Qwen | Qwen-Turbo | `qwen/qwen-turbo` |
| Qwen | Qwen 3.6 Plus | `qwen/qwen-3.6-plus` |
| Google | Gemini 2.5 Flash | `gemini/gemini-2.5-flash` |

Verify these model strings against LiteLLM's current model list (`litellm.model_list` or their docs) before building — model string format may have changed.

### Pricing Table

LiteLLM includes a built-in cost tracker (`litellm.completion_cost(response)`) that computes per-call cost from its internal pricing table. Use this as the primary cost source. Cross-check against the pricing table in the AgentCost codebase — if they disagree, the codebase's table takes precedence (it's what the projection engine uses).

```python
from litellm import completion_cost
cost = completion_cost(response)  # Returns USD cost for this call
```

If LiteLLM's pricing is stale or missing for a model, fall back to manual computation:

```python
cost = (input_tokens * price_per_input_token) + (output_tokens * price_per_output_token)
```

Maintain a `pricing_overrides.py` with per-model prices for any models LiteLLM doesn't price correctly. Use current API pricing. If a provider has tiered pricing (cached vs. uncached tokens), apply the correct tier based on `cache_hit_tokens`.

---

## Part 2: Workflow Patterns

The 14 workflows decompose into 7 patterns. Build the patterns as composable abstractions, then instantiate each workflow with pattern-specific configuration.

### Pattern 1: Single-Step

**Workflows:** W1, W5, W11, W12, W18

```
input → [load prompt] → [call model] → [record StepRecord] → done
```

The simplest pattern. One API call. The step loads the system prompt from `prompts/{workflow}/step1_*.txt`, constructs the user message from the input, calls the model, and records the result.

**W1 specifics:** The orchestrator first routes to Haiku or Sonnet based on a simple heuristic (input token count < 80 → Haiku, ≥ 80 → Sonnet). This is the external routing mentioned in the system prompt spec. The routing decision itself doesn't cost anything — it's a code branch, not an API call.

**W5 specifics:** For image inputs, the user message includes the image as a base64-encoded content block (Anthropic vision format). For text inputs, the user message is text only. The prompt is the same either way.

**W11 specifics:** Same prompt and routing as W1, but calls Qwen instead of Anthropic. The routing heuristic is identical.

**W12 specifics:** Calls DeepSeek V4 Flash. The input is the full document text as the user message.

**W18 specifics:** Calls DeepSeek V4. The input is the full document text (30K–100K tokens) as the user message. Set `max_tokens` high enough for the output (2,500 should suffice).

### Pattern 2: Multi-Step Linear

**Workflows:** W9

```
input → [step 1: qualify] → [parse output] → [step 2: draft email] → done
```

Two sequential API calls. Step 1's output (the lead rating and talking points) is included in Step 2's user message alongside the original lead profile.

**W9 specifics:** Both steps use OpenAI models (GPT-5.4 Nano for qualify, GPT-5.4 for draft). Parse Step 1's JSON output to extract `rating` and `key_talking_points`, then include them in Step 2's user message.

### Pattern 3: Self-Assessment Loop

**Workflows:** W2, W4

```
input → [step 1: classify/review] → loop { [step 2: draft/critique] → [check termination] } → [conditional step 3: final review]
```

The loop iterates until a termination condition is met or the maximum iteration count is reached. Each iteration appends its output to the conversation context.

**W2 specifics:**
- Step 1 (Haiku): Classify the input. Parse `complexity` from the JSON output.
- Step 2 (Sonnet): Research + draft loop. Each iteration:
  - Send: system prompt + full conversation history (original question + classification + all prior drafts/assessments)
  - Parse: `confidence` and `should_continue` from the JSON output
  - Terminate when: `confidence >= 0.9` OR iteration count reaches 12
  - Each iteration produces one StepRecord with `metadata.iteration = N`
- Step 3 (Opus): Conditional. Triggered when `complexity == "complex"` AND iteration count ≥ 4. If triggered, send the final draft for review.
- **Context growth:** The conversation history sent to Step 2 grows with each iteration. The step_index for each iteration must reflect this (step_index = 2 for iter 1, step_index = 3 for iter 2, etc.). Input tokens will increase by ~500–1,000 per iteration.

**W4 specifics:**
- Step 1 (DeepSeek V4): Initial compliance review. Full document in user message.
- Step 2 (Qwen 3.6 Plus): Critique. Receives the document + Step 1's findings.
  - Parse: `satisfied` from the JSON output
  - If `satisfied == false`, go to Step 3
  - If `satisfied == true`, done
- Step 3 (DeepSeek V4): Revision. Receives the document + prior findings + critique.
  - After revision, loop back to Step 2 (critique the revision)
  - Terminate when: Step 2 says `satisfied == true` OR total iterations (Step 2 + Step 3 pairs) reaches 8
- **Context growth:** Same pattern as W2. Each critique-revision pair adds to the context.

### Pattern 4: Router

**Workflows:** W13

```
input → [step 1: classify] → branch {
  TIER_1 → [step 2: path A (Haiku)]
  TIER_2 → [step 3: path B (Sonnet)]
  TIER_3 → [step 4: path C (Sonnet + tools)]
}
```

Step 1 classifies the input into a tier. The tier determines which downstream step runs. Only one path executes per input — the other two are skipped.

**W13 specifics:**
- Step 1 (Haiku): Parse `tier` from JSON output.
- Path A (Haiku): Simple response. 2 StepRecords total (classify + respond).
- Path B (Sonnet): Analytical response. 2 StepRecords total.
- Path C (Sonnet + tools): Complex response with tool calls. The tools (web_search, calculator, unit_converter) are simulated — the agent includes the tool schemas in the prompt, the model generates tool call requests, but the agent provides mock tool responses. This adds 1–2 extra round-trips. 3–4 StepRecords total.
- **Cost bimodality:** Path A costs ~$0.003, Path C costs ~$0.08. The routing decision is the cost driver.

**Tool simulation for Path C:** When the model outputs a tool call, the agent generates a plausible mock response (e.g., for `web_search({"query": "USD EUR exchange rate"})`, respond with `{"result": "1 USD = 0.92 EUR as of June 2026"}`). The mock response is included in the next API call as a tool result. This simulates the tool round-trip without actually calling external services. Each tool call adds one StepRecord (for the follow-up API call that receives the tool result).

### Pattern 5: RAG Pipeline

**Workflows:** W14, W15

```
[W14] input → [embed query] → [retrieve] → [step: generate answer] → done
[W15] input → [embed query] → [retrieve] → loop { [assess sufficiency] → [re-retrieve if needed] } → [step: generate answer] → done
```

These workflows include non-LLM steps (embedding, vector search) that don't produce StepRecords, plus LLM steps that do.

**W14 specifics:**
- Embedding (OpenAI text-embedding-3-small): Embed the query. This is an API call that produces a StepRecord, but at embedding pricing (much cheaper per token). Record it with `step_name = "embed_query"`.
- Retrieval: Vector search against the W14 PDF corpus. Not an API call — no StepRecord. But the retrieved chunks determine the context size for the generation step, which is the primary cost driver.
- Generation (Sonnet 4.6): System prompt from `prompts/w14_simple_rag/step4_generate_answer.txt`. User message = query + retrieved chunks. Context size varies from ~500 to ~20,000 tokens depending on how many chunks match.
- **Retrieval simulation:** Since there's no live vector DB, simulate retrieval by: (a) embedding the query with OpenAI, (b) computing cosine similarity against pre-embedded chunks (embedded during PDF generation), (c) returning the top-K chunks where K is determined by the input's tier (easy = 1–2 chunks, hard = 8–15 chunks). The chunk embeddings and text should be pre-computed and stored alongside the PDF corpus.

**W15 specifics:**
- Same embedding and retrieval as W14, but adds the sufficiency loop:
- Sufficiency assessment (Gemini 2.5 Flash): After each retrieval round, assess whether the context is sufficient. Parse `sufficient` from the JSON output.
  - If `sufficient == false` AND hop_number < 4: use the `refined_query` to re-embed and re-retrieve. Append new chunks to accumulated context.
  - If `sufficient == true` OR hop_number == 4: proceed to generation.
- Generation (DeepSeek V4): Same as W14 but with accumulated context from all hops.
- **Context growth:** Each hop adds retrieved chunks to the context. By hop 4, the sufficiency assessor sees all accumulated context (possibly 15,000+ tokens). The generation step also sees the full accumulated context.
- **StepRecords per hop:** Each hop produces: 1 embedding StepRecord + 1 sufficiency assessment StepRecord. The final generation is 1 StepRecord. A 3-hop run produces ~7 StepRecords.

### Pattern 6: Map-Reduce

**Workflows:** W16

```
input (PDF) → [step 1: split] → [step 2: process section 1] + [step 2: process section 2] + ... + [step 2: process section N] → [step 3: aggregate] → done
```

Step 1 determines N. Step 2 runs N times (can be parallelized). Step 3 aggregates all N outputs.

**W16 specifics:**
- Step 1 (Sonnet 4.6): Split the document into sections. Parse the `sections` array from JSON output to determine N and the page ranges.
- Step 2 (Haiku 4.5): Process each section. Each call gets the section text as the user message. Run these in parallel (asyncio or threading) to reduce wall-clock time. Each call produces one StepRecord with `metadata.section_id = K`.
- Step 3 (Sonnet 4.6): Aggregate. User message = all N section analyses concatenated. Input tokens scale with N.
- **Cost driver:** N determines the number of parallel Haiku calls (each ~$0.003–0.01) plus the aggregation cost (scales with N × output_per_section).

### Pattern 7: Pipeline with Overrides

**Workflows:** W17

```
input (claim JSON) → [step 1: intake + override check] → branch {
  short_circuit → done (1 StepRecord)
  proceed → [embed + retrieve policy] → [step 3: evaluate + decide] → branch {
    no_flags → done
    flags → [step 4: conditional routing] → done
  }
}
```

The most complex workflow. Multiple branching points, conditional step execution, function call output.

**W17 specifics:**
- Step 1 (Haiku 4.5): Intake and override check. Parse JSON output for `short_circuit` and `flags`.
  - If `short_circuit == true`: the pipeline ends here. 1 StepRecord. Cheapest outcome.
  - If `short_circuit == false`: proceed to retrieval.
- Retrieval: Embed the claim's diagnosis and procedure codes. Retrieve from the correct provider's policy PDF (use `claim.provider` to select the corpus). Same simulation as W14 (pre-embedded chunks, cosine similarity, top-K retrieval).
- Step 3 (Sonnet 4.6): Evaluate the claim against retrieved policy sections. Parse `action` and `flags` from JSON output. The output includes a function call (e.g., `approve_claim_payment(payment_in_dollars=2200.0)`) as specified in the output schema.
- Step 4 (Haiku 4.5): Conditional routing. Only runs if `flags` is non-empty (high_amount or code_inconsistency). Wraps the evaluation decision with a routing action.
- **Full architecture:** See `projection-engine-recommendation-addition-2.md` for the complete W17 specification. The agent must match it exactly.
- **Cost bimodality:** Short-circuit claims (inactive member, missing docs) cost ~$0.01. Full-pipeline claims with routing cost ~$0.05–0.15. The override rules in Step 1 control which mode each claim falls into.

### Pattern 8: Multi-Turn Conversation

**Workflows:** W19

```
input (8-turn script) → turn 1: [call model with prompt + msg 1] →
  turn 2: [call model with prompt + msg 1 + resp 1 + msg 2] →
  ... →
  turn 8: [call model with prompt + msg 1 + resp 1 + ... + msg 8] → done
```

The same model and prompt are used for all 8 turns. The conversation history accumulates.

**W19 specifics:**
- Model: DeepSeek V4 for all turns.
- System prompt: loaded once from `prompts/w19_multi_turn/step1_respond.txt`.
- Each turn:
  1. Replace `{{CACHE_BUST_SUFFIX}}` with a fresh UUID (unique per turn, not per conversation).
  2. Construct messages array: all prior user messages and assistant responses, plus the new user message from the conversation script.
  3. Call the API.
  4. Append the response to the conversation history.
  5. Record a StepRecord with `metadata.turn_number = N`.
- **Context growth:** Turn 1 input ≈ 1,400 tokens. Turn 8 input ≈ 9,200–11,200 tokens. Each turn's StepRecord should show increasing `input_tokens`. If input tokens don't increase, the history accumulation is broken.
- **8 StepRecords per run.** The per-run cost is the sum of all 8 StepRecords' costs.

---

## Part 3: Harness

### The Runner

```python
# run_workflow.py
def run_workflow(workflow_id, input_data, prompts_dir="prompts/") -> list[StepRecord]:
    agent = load_agent(workflow_id)
    prompts = load_prompts(workflow_id, prompts_dir)
    records = agent.execute(input_data, prompts)
    return records

def run_batch(workflow_id, input_dir, profile, n, prompts_dir="prompts/") -> list[list[StepRecord]]:
    inputs = load_inputs(workflow_id, input_dir, profile, n)
    all_records = []
    for input_data in inputs:
        records = run_workflow(workflow_id, input_data, prompts_dir)
        all_records.append(records)
    return all_records
```

### CLI Interface

```
python run_workflow.py --workflow W1 --profile profiling --n 50 --seed 42 \
    --prompts-dir prompts/ \
    --inputs-dir inputs/generated/profiling/w01/ \
    --output-dir results/profiling/w01/ \
    --pricing-table pricing.json
```

Flags:
- `--workflow`: Which workflow to run (W1–W19)
- `--profile`: `profiling` or `ground_truth`
- `--n`: How many inputs to process
- `--seed`: Random seed (for reproducibility of any stochastic elements)
- `--prompts-dir`: Where the system prompt .txt files live
- `--inputs-dir`: Where the generated input JSON files live
- `--output-dir`: Where to write the StepRecords (one JSON file per run)
- `--pricing-table`: Path to the pricing JSON
- `--dry-run`: Load inputs and prompts, print what would be called, but make no API calls
- `--parallel`: Number of concurrent runs (default 1; useful for workflows without shared state)

### Output Format

Each run produces a JSON file:

```json
{
  "workflow_id": "W1",
  "run_id": "w01_prof_easy_007",
  "input_id": "w01_prof_easy_007",
  "input_tier": "easy",
  "profile": "profiling",
  "total_cost_usd": 0.023,
  "total_input_tokens": 1842,
  "total_output_tokens": 312,
  "step_count": 1,
  "steps": [
    {
      "step_name": "classify_respond",
      "step_index": 0,
      "model": "claude-haiku-4.5",
      "provider": "anthropic",
      "input_tokens": 1842,
      "output_tokens": 312,
      "cost_usd": 0.023,
      "latency_ms": 1250,
      "cache_hit_tokens": 0,
      "finish_reason": "stop",
      "metadata": {"routed_to": "haiku"}
    }
  ],
  "metadata": {
    "input_tier": "easy",
    "is_dirty": false
  }
}
```

### Error Handling

- **API errors (rate limits, timeouts):** Retry with exponential backoff (3 retries, 1s/2s/4s). If all retries fail, record the run as failed with the error in metadata. Do not retry indefinitely.
- **JSON parse failures:** If a model's output doesn't parse as JSON, record the raw output in metadata and mark the step as `parse_error`. The run still produces StepRecords (with the actual tokens consumed), but the downstream steps may fail if they depend on parsed output.
- **Finish reason "length":** Record the step normally but flag `truncated: true` in metadata. This counts as a valid run for cost purposes (the tokens were consumed), but the pilot check will catch it.
- **Context window exceeded:** If the accumulated context exceeds the model's context window (relevant for W2 at high iteration counts, W15 at hop 4, W18 with long documents), record the failure and move on. This should not happen if the input generation respected the token ranges, but it's a safety net.

---

## Part 4: Workflow-Specific Implementation Notes

### W14/W15 — Retrieval Simulation

Since there's no live vector database, simulate retrieval:

1. **Pre-compute embeddings:** During PDF generation, each chunk gets embedded with OpenAI text-embedding-3-small and stored alongside the chunk text in the content manifest.
2. **At runtime:** Embed the query, compute cosine similarity against all chunk embeddings, return top-K chunks.
3. **K selection:** The input's structural descriptor includes `expected_chunk_count`. Use this as K (or K ± 1 for slight variation).
4. **The embedding call IS a StepRecord.** Record it with embedding pricing.
5. **The cosine similarity search is NOT a StepRecord.** It's local computation.

### W16 — Parallel Execution

The N section-processing calls in Step 2 can run in parallel. Use `asyncio.gather()` or `concurrent.futures.ThreadPoolExecutor`. Record wall-clock latency for the parallel batch (not per-call latency) if reporting overall latency, but record per-call latency in each StepRecord.

### W17 — Provider-Specific Retrieval

When retrieving policy documents for W17, use the `claim.provider` field to select the correct provider's policy PDF. Do not mix providers — a United Healthcare claim must retrieve from United Healthcare's policy, not Aetna's.

### W13 — Tool Simulation

For Path C tool calls:
1. The model generates a tool call in its output.
2. The agent parses the tool call (function name + parameters).
3. The agent generates a mock response based on the function:
   - `web_search`: return a plausible 1–2 sentence answer
   - `calculator`: actually compute the expression and return the result
   - `unit_converter`: actually perform the conversion and return the result
4. Send the tool result back to the model in a follow-up API call.
5. The model generates its final response incorporating the tool result.
6. Each round-trip (tool call + tool result + final response) produces separate StepRecords.

### W2 — Opus Conditional

The Opus review step (Step 3) fires when:
- Step 1 classified the input as `complexity: "complex"` AND
- The loop ran for ≥ 4 iterations

Both conditions must be true. If the input is complex but the loop converged in 3 iterations, Opus does not fire. If the input is simple but the loop ran 5 iterations (unlikely but possible), Opus does not fire.

---

## Part 5: Integration with AgentCost

The StepRecords produced by the harness must be consumable by the existing projection engine. Verify:

1. **Schema compatibility:** The StepRecord JSON matches what the collector expects. Compare against the collector's schema in the codebase.
2. **Cost computation consistency:** The pricing table used by the agents matches the one used by the projection engine. If the engine has its own pricing table, use that one, not a separate copy.
3. **Run ID convention:** Use the input ID as the run ID so runs can be traced back to specific inputs (and their tier labels) for stratified analysis.

### Testing the Integration

After building the agents, run a minimal integration test:

1. Run W1 with 5 profiling inputs.
2. Feed the resulting StepRecords into the projection engine.
3. Verify the engine produces a projection (mean, CI, detectors).
4. Check that no StepRecord fields are missing or mistyped.

If this works, the agents are compatible with the engine.

---

## File Manifest

The exact file structure depends on the orchestration choice (custom patterns vs. LangGraph graphs). The structure below is illustrative — adapt to match the codebase conventions.

**If using custom patterns:**
```
agents/
  patterns/
    single_step.py           # Pattern 1: W1, W5, W11, W12, W18
    multi_step_linear.py     # Pattern 2: W9
    self_assessment_loop.py  # Pattern 3: W2, W4
    router.py                # Pattern 4: W13
    rag_pipeline.py          # Pattern 5: W14
    multi_hop_rag.py         # Pattern 5: W15
    map_reduce.py            # Pattern 6: W16
    pipeline_overrides.py    # Pattern 7: W17
    multi_turn.py            # Pattern 8: W19

  workflows/
    w01.py through w19.py    # Per-workflow config wiring pattern to prompts/models/params
```

**If using LangGraph:**
```
agents/
  graphs/
    single_step.py           # StateGraph for W1, W5, W11, W12, W18
    multi_step_linear.py     # StateGraph for W9
    self_assessment_loop.py  # StateGraph with conditional loop edges for W2, W4
    router.py                # StateGraph with branching for W13
    rag_pipeline.py          # StateGraph for W14
    multi_hop_rag.py         # StateGraph with retrieval loop for W15
    map_reduce.py            # StateGraph with Send() for W16 fan-out
    pipeline_overrides.py    # StateGraph with override short-circuits for W17
    multi_turn.py            # StateGraph with turn accumulation for W19

  workflows/
    w01.py through w19.py    # Per-workflow graph compilation with specific config
```

**Shared regardless of orchestration choice:**
```
providers/
  llm.py                     # Thin wrapper around LiteLLM: call_model(), cache-bust, extract usage
  embeddings.py              # LiteLLM embedding calls for W14/W15/W17
  pricing_overrides.py       # Per-model price overrides for models LiteLLM doesn't price correctly

harness/
  run_workflow.py             # CLI entry point
  step_record.py              # StepRecord dataclass
  retrieval_sim.py            # Cosine similarity retrieval simulation for W14/W15/W17
  tool_sim.py                 # Mock tool responses for W13 Path C
```

Each `workflows/wXX.py` file is a thin configuration layer that imports a pattern (or compiles a graph) and wires it to the correct prompts, models, and workflow-specific parameters (routing thresholds, loop conditions, etc.). The patterns/graphs contain the reusable orchestration logic.
