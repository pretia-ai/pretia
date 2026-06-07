"""W11 Support (Qwen) — reuses W1's inputs for cross-provider comparison.

Generates the same inputs as W1 with the workflow field changed to W11.
The directions spec requires identical inputs so W1 (Anthropic) and W11
(Qwen) cost comparisons are valid.
"""

from __future__ import annotations

import random
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli
from inputs.generators.w01_support_simple import W01SupportSimpleGenerator


class W11SupportQwenGenerator(BaseInputGenerator):
    workflow_id = "W11"
    dirty_types = ["typos", "mixed_unicode", "near_empty"]

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._w1 = W01SupportSimpleGenerator(seed=seed)

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        w1_input = self._w1.generate_single(
            tier=tier, profile=profile, rng=rng, idx=idx,
            is_dirty=is_dirty, dirty_type=dirty_type,
        )
        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W11",
            profile=w1_input.profile,
            tier=w1_input.tier,
            token_count=w1_input.token_count,
            is_dirty=w1_input.is_dirty,
            dirty_type=w1_input.dirty_type,
            structural_descriptor=w1_input.structural_descriptor,
            input_data=w1_input.input_data,
        )

    def generate_batch(
        self,
        profile: str,
        n: int,
    ) -> list[GeneratedInput]:
        w1_inputs = self._w1.generate_batch(profile, n)
        return [
            GeneratedInput(
                id=self.make_id(profile, inp.tier, idx),
                workflow="W11",
                profile=inp.profile,
                tier=inp.tier,
                token_count=inp.token_count,
                is_dirty=inp.is_dirty,
                dirty_type=inp.dirty_type,
                structural_descriptor=inp.structural_descriptor,
                input_data=inp.input_data,
            )
            for idx, inp in enumerate(w1_inputs)
        ]


if __name__ == "__main__":
    add_cli(W11SupportQwenGenerator)
