"""Base class for per-workflow input generators.

Provides seeded tier allocation, dirty input injection, style/tone shift,
token length stretch, and CLI scaffolding. Subclasses override
``generate_single`` to produce workflow-specific content.
"""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILING_WEIGHTS: dict[str, float] = {
    "easy": 0.40,
    "medium": 0.35,
    "hard": 0.20,
    "edge": 0.05,
}

GROUND_TRUTH_WEIGHTS: dict[str, float] = {
    "easy": 0.55,
    "medium": 0.25,
    "hard": 0.12,
    "edge": 0.05,
    "extreme": 0.03,
}

_STYLE_ARTIFACTS = [
    lambda rng, t: re.sub(r"\b(\w{4,})\b", lambda m: _typo(rng, m.group()), t, count=rng.randint(1, 3)),
    lambda rng, t: t.lower(),
    lambda rng, t: t.replace(". ", ".. ").replace("?", "??"),
    lambda rng, t: t.replace("I ", "i ").replace("I'm", "im").replace("I've", "ive"),
    lambda rng, t: t.replace("you", "u").replace("your", "ur").replace("are", "r"),
    lambda rng, t: t + "   ",
    lambda rng, t: t.replace(", ", " "),
]


def _typo(rng: random.Random, word: str) -> str:
    """Introduce a single-character typo into a word."""
    if len(word) < 4 or rng.random() > 0.15:
        return word
    idx = rng.randint(1, len(word) - 2)
    ops = ["swap", "drop", "double"]
    op = rng.choice(ops)
    if op == "swap" and idx < len(word) - 1:
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    if op == "drop":
        return word[:idx] + word[idx + 1:]
    return word[:idx] + word[idx] + word[idx] + word[idx + 1:]


@dataclass
class GeneratedInput:
    """One tagged input ready for serialization."""

    id: str
    workflow: str
    profile: str
    tier: str
    token_count: int
    is_dirty: bool
    dirty_type: str | None
    structural_descriptor: dict[str, Any]
    input_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow": self.workflow,
            "profile": self.profile,
            "tier": self.tier,
            "token_count": self.token_count,
            "is_dirty": self.is_dirty,
            "dirty_type": self.dirty_type,
            "structural_descriptor": self.structural_descriptor,
            "input_data": self.input_data,
        }


class BaseInputGenerator(ABC):
    """Abstract base for workflow-specific input generators.

    Subclasses must implement:
    - ``workflow_id`` — e.g. "W1"
    - ``dirty_types`` — list of dirty input type strings applicable to this workflow
    - ``generate_single(tier, profile, rng, idx)`` — produce one ``GeneratedInput``
    """

    workflow_id: str = ""
    dirty_types: list[str] = []

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    @property
    def tier_weights(self) -> dict[str, dict[str, float]]:
        """Override for workflows with non-standard tier weights (e.g. W13)."""
        return {
            "profiling": PROFILING_WEIGHTS,
            "ground_truth": GROUND_TRUTH_WEIGHTS,
        }

    def allocate_tiers(self, profile: str, n: int) -> list[str]:
        """Assign tier labels to n inputs using the configured weights."""
        weights = self.tier_weights[profile]
        tiers: list[str] = []
        remaining = n

        sorted_tiers = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for tier_name, weight in sorted_tiers[:-1]:
            count = round(n * weight)
            tiers.extend([tier_name] * count)
            remaining -= count

        last_tier = sorted_tiers[-1][0]
        tiers.extend([last_tier] * max(0, remaining))

        if len(tiers) > n:
            tiers = tiers[:n]
        while len(tiers) < n:
            tiers.append(sorted_tiers[0][0])

        self.rng.shuffle(tiers)
        return tiers

    def select_dirty_indices(self, n: int, tiers: list[str]) -> dict[int, str]:
        """Select ~5% of inputs to be dirty, spread across tiers."""
        if not self.dirty_types:
            return {}

        dirty_count = max(1, round(n * 0.05))
        dirty_count = min(dirty_count, n)

        tier_indices: dict[str, list[int]] = {}
        for idx, tier in enumerate(tiers):
            tier_indices.setdefault(tier, []).append(idx)

        available_tiers = [t for t in tier_indices if tier_indices[t]]
        dirty_map: dict[int, str] = {}
        attempts = 0
        while len(dirty_map) < dirty_count and attempts < dirty_count * 10:
            attempts += 1
            tier = self.rng.choice(available_tiers)
            candidates = [i for i in tier_indices[tier] if i not in dirty_map]
            if not candidates:
                continue
            idx = self.rng.choice(candidates)
            dirty_type = self.rng.choice(self.dirty_types)
            dirty_map[idx] = dirty_type

        return dirty_map

    def apply_style_shift(self, text: str, profile: str) -> str:
        """Apply tone/style artifacts for ground truth inputs."""
        if profile == "profiling":
            return text
        if self.rng.random() > 0.82:
            return text
        n_artifacts = self.rng.randint(2, 4)
        result = text
        chosen = self.rng.sample(_STYLE_ARTIFACTS, min(n_artifacts, len(_STYLE_ARTIFACTS)))
        for artifact_fn in chosen:
            result = artifact_fn(self.rng, result)
        return result

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: chars / 4."""
        return max(1, len(text) // 4)

    _FILLER_PHRASES = [
        " I appreciate your help with this matter.",
        " This is really important for our workflow.",
        " Our team depends on this for daily operations.",
        " We've been using this product since it launched.",
        " Thank you for looking into this promptly.",
        " Please let me know if you need more details.",
        " I've tried troubleshooting on my own but couldn't resolve it.",
        " Several of my colleagues are experiencing the same issue.",
        " We need this resolved before our quarterly review.",
        " Any guidance you can provide would be greatly appreciated.",
    ]

    def pad_to_token_range(
        self, text: str, target_min: int, target_max: int, rng: random.Random,
    ) -> str:
        """Pad or truncate text to hit a random point within the target token range."""
        target = rng.randint(target_min, target_max)
        current = self.estimate_tokens(text)

        if current < target:
            while self.estimate_tokens(text) < target:
                text += rng.choice(self._FILLER_PHRASES)

        if self.estimate_tokens(text) > target_max:
            text = text[: target_max * 4]

        return text

    def llm_rewrite(
        self,
        template: str,
        instruction: str,
        target_tokens: int,
        rng: random.Random,
    ) -> str:
        """Sync LLM rewrite — only for standalone/non-async usage."""
        import asyncio

        return asyncio.run(
            self.llm_rewrite_async(template, instruction, target_tokens, rng)
        )

    async def llm_rewrite_async(
        self,
        template: str,
        instruction: str,
        target_tokens: int,
        rng: random.Random,
    ) -> str:
        """Call DeepSeek to rewrite a template into a unique variant.

        Async version — used by ``generate_batch_async`` for full concurrency.
        Falls back to the original template if the LLM call fails.
        """
        from inputs.generators._llm import generate_text

        system = (
            "You are a content generator. Rewrite the example text to create a "
            "new, unique variant that preserves the same intent and difficulty level "
            "but uses different wording, details, and structure. "
            "Output ONLY the rewritten text, no explanation.\n"
            "<!-- session: {{CACHE_BUST_SUFFIX}} -->"
        )
        user = (
            f"{instruction}\n\n"
            f"Target length: approximately {target_tokens} tokens "
            f"({target_tokens * 4} characters).\n\n"
            f"Example to rewrite (create a DIFFERENT variant, do not copy):\n"
            f"{template}"
        )
        try:
            resp = await generate_text(
                system, user,
                max_tokens=max(256, target_tokens * 2),
                dry_run=False,
            )
            if resp.content and len(resp.content.strip()) > 10:
                return resp.content.strip()
        except Exception:
            pass
        return template

    @abstractmethod
    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one input for the given tier and profile."""

    def generate_batch(
        self,
        profile: str,
        n: int,
    ) -> list[GeneratedInput]:
        """Generate a full input set with proper tier allocation and dirty injection."""
        self.rng = random.Random(self.seed)

        tiers = self.allocate_tiers(profile, n)
        dirty_map = self.select_dirty_indices(n, tiers)

        inputs: list[GeneratedInput] = []
        for idx in range(n):
            tier = tiers[idx]
            is_dirty = idx in dirty_map
            dirty_type = dirty_map.get(idx)

            inp = self.generate_single(
                tier=tier,
                profile=profile,
                rng=self.rng,
                idx=idx,
                is_dirty=is_dirty,
                dirty_type=dirty_type,
            )
            inputs.append(inp)

        return inputs

    def get_rewritable_text(self, inp: GeneratedInput) -> str | None:
        """Return the text field to rewrite, or None if not rewritable.

        Override in subclasses to specify which input_data field contains
        the primary text content for LLM rewriting.
        """
        for key in ("customer_message", "document_text", "user_query", "query", "content"):
            if key in inp.input_data and isinstance(inp.input_data[key], str):
                return key
        return None

    def get_llm_instruction(self, inp: GeneratedInput) -> str:
        """Return the LLM rewrite instruction for this input.

        Override in subclasses for workflow-specific generation prompts.
        """
        return f"Generate a {inp.tier}-difficulty input for the {self.workflow_id} workflow."

    async def generate_batch_async(
        self,
        profile: str,
        n: int,
    ) -> list[GeneratedInput]:
        """Generate inputs with concurrent LLM rewriting.

        Phase 1: Generate all templates synchronously (instant).
        Phase 2: Fire all LLM rewrites concurrently via asyncio.gather,
        exploiting DeepSeek's concurrency limits (500 Pro, 2500 Flash).
        """
        import asyncio

        # Phase 1: sync template generation (< 1 second)
        inputs = self.generate_batch(profile, n)

        # Phase 2: concurrent LLM rewrite (if not dry_run)
        if getattr(self, "dry_run", True):
            return inputs

        async def _rewrite_one(idx: int, inp: GeneratedInput) -> GeneratedInput:
            text_key = self.get_rewritable_text(inp)
            if text_key is None:
                return inp

            text = inp.input_data[text_key]
            instruction = self.get_llm_instruction(inp)
            rng = random.Random(self.seed + idx)

            ranges = getattr(self, "_token_ranges_lookup", None)
            if ranges and profile in ranges and inp.tier in ranges[profile]:
                tmin, tmax = ranges[profile][inp.tier]
            else:
                tmin, tmax = max(1, inp.token_count // 2), inp.token_count * 2

            target = rng.randint(tmin, tmax)
            rewritten = await self.llm_rewrite_async(
                text, instruction, target, rng,
            )
            rewritten = self.pad_to_token_range(rewritten, tmin, tmax, rng)
            rewritten = self.apply_style_shift(rewritten, profile)

            new_data = dict(inp.input_data)
            new_data[text_key] = rewritten
            if "input" in new_data:
                new_data["input"] = rewritten

            return GeneratedInput(
                id=inp.id,
                workflow=inp.workflow,
                profile=inp.profile,
                tier=inp.tier,
                token_count=self.estimate_tokens(rewritten),
                is_dirty=inp.is_dirty,
                dirty_type=inp.dirty_type,
                structural_descriptor=inp.structural_descriptor,
                input_data=new_data,
            )

        tasks = [_rewrite_one(idx, inp) for idx, inp in enumerate(inputs)]
        return list(await asyncio.gather(*tasks))

    def save_batch(
        self,
        inputs: list[GeneratedInput],
        output_dir: str,
    ) -> Path:
        """Save generated inputs as individual JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for inp in inputs:
            filepath = out / f"{inp.id}.json"
            filepath.write_text(json.dumps(inp.to_dict(), indent=2, default=str))

        return out

    def make_id(self, profile: str, tier: str, idx: int) -> str:
        wf = self.workflow_id.lower().replace("w", "w")
        wf_num = self.workflow_id.upper().replace("W", "")
        prof_short = "prof" if profile == "profiling" else "gt"
        return f"w{wf_num}_{prof_short}_{tier}_{idx:03d}"


def add_cli(generator_cls: type[BaseInputGenerator]) -> None:
    """Add a Click CLI to a generator class and run it."""
    import asyncio
    import time

    import click

    @click.command()
    @click.option("--profile", "-p", required=True, type=click.Choice(["profiling", "ground_truth"]))
    @click.option("--n", type=int, default=None, help="Number of inputs (default: 50 profiling, 500 GT)")
    @click.option("--seed", type=int, default=42, help="Random seed")
    @click.option("--output-dir", "-o", required=True, help="Output directory")
    @click.option("--dry-run", is_flag=True, help="Generate template stubs without LLM calls")
    @click.option("--concurrent/--sequential", default=True, help="Use async concurrency (default: concurrent)")
    def main(profile: str, n: int | None, seed: int, output_dir: str, dry_run: bool, concurrent: bool) -> None:
        if n is None:
            n = 50 if profile == "profiling" else 500

        gen = generator_cls(seed=seed)
        if hasattr(gen, "dry_run"):
            gen.dry_run = dry_run

        start = time.monotonic()
        if concurrent and not dry_run:
            inputs = asyncio.run(gen.generate_batch_async(profile, n))
        else:
            inputs = gen.generate_batch(profile, n)
        elapsed = time.monotonic() - start

        gen.save_batch(inputs, output_dir)
        click.echo(f"Generated {len(inputs)} inputs for {gen.workflow_id} ({profile}) in {output_dir}")

        tier_counts = Counter(i.tier for i in inputs)
        dirty_count = sum(1 for i in inputs if i.is_dirty)
        click.echo(f"  Tiers: {dict(sorted(tier_counts.items()))}")
        click.echo(f"  Dirty: {dirty_count}")
        click.echo(f"  Time: {elapsed:.1f}s")

    main()
