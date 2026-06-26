"""W5 — Multimodal Extraction: single-step Sonnet with vision support.

Accepts text or image inputs. Images are passed as base64 content blocks.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from pretia.collectors.base import StepRecord
from bt_agents import BaseAgent
from bt_agents.patterns.single_step import run_single_step


def _build_messages(input_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build user messages, handling image inputs as vision content blocks."""
    image_path = input_data.get("image_path")
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        media_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                    {"type": "text", "text": input_data.get("input", "Extract all data.")},
                ],
            }
        ]
    return [{"role": "user", "content": input_data.get("input", "")}]


class W05MultimodalExtraction(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_single_step(
            input_text=input_data.get("input", ""),
            system_prompt=prompts["extract"],
            model="claude-sonnet-4-6",
            step_name="extract",
            output_format="json",
            max_tokens=8192,
            messages=_build_messages(input_data),
            dry_run=input_data.get("_dry_run", False),
        )


agent = W05MultimodalExtraction()
