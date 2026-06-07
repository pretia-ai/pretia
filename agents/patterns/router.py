"""Routing pattern: classify input then route to one of N execution paths.

Used by W13. Flow: classify → select route → execute (optionally with tool calls).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentcost.collectors.base import StepRecord

from agents.harness.step_builder import build_llm_step
from agents.providers.llm import call_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RouteConfig:
    """Configuration for one route in a classification-based router."""

    model: str
    prompt_key: str
    step_name: str
    output_format: str
    max_tokens: int
    has_tools: bool = False


async def run_router(
    *,
    input_text: str,
    prompts: dict[str, str],
    classifier_model: str,
    classifier_prompt_key: str,
    classifier_step_name: str,
    classifier_max_tokens: int,
    routes: dict[str, RouteConfig],
    default_route: str,
    tool_schemas: list[dict] | None = None,
    tool_simulator: Callable[[str, dict], str] | None = None,
    max_tool_rounds: int = 3,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Classify input and route to the appropriate execution path.

    Args:
        input_text: Raw user input to classify and process.
        prompts: Map of prompt keys to system prompt text.
        classifier_model: Model for the classification step.
        classifier_prompt_key: Key into *prompts* for the classifier system prompt.
        classifier_step_name: Step name for the classification StepRecord.
        classifier_max_tokens: Max output tokens for classification.
        routes: Map of tier names to RouteConfig for each execution path.
        default_route: Fallback tier name when classification parse fails.
        tool_schemas: Tool definitions for routes that use function calling.
        tool_simulator: Callable(tool_name, arguments_dict) -> result string.
        max_tool_rounds: Maximum tool-call/response round-trips per route.
        dry_run: If True, skip real API calls.

    Returns:
        List of StepRecords: [classify_record, route_records...].
    """
    records: list[StepRecord] = []

    # ------------------------------------------------------------------
    # Step 1: Classify the input
    # ------------------------------------------------------------------
    classifier_prompt = prompts[classifier_prompt_key]
    classify_messages = [{"role": "user", "content": input_text}]

    classify_response = await call_model(
        classifier_model,
        classifier_prompt,
        classify_messages,
        max_tokens=classifier_max_tokens,
        dry_run=dry_run,
    )

    classify_record = build_llm_step(
        step_name=classifier_step_name,
        response=classify_response,
        system_prompt=classifier_prompt,
        output_format="json",
    )
    records.append(classify_record)

    # ------------------------------------------------------------------
    # Step 2: Parse tier from classifier output and select route
    # ------------------------------------------------------------------
    tier = default_route
    try:
        parsed = json.loads(classify_response.content)
        tier = parsed.get("tier", default_route)
        if tier not in routes:
            logger.warning(
                "Classifier returned unknown tier %r, falling back to %r",
                tier,
                default_route,
            )
            tier = default_route
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "Failed to parse classifier output as JSON: %s — using default route %r",
            exc,
            default_route,
        )

    route_cfg = routes[tier]
    route_prompt = prompts[route_cfg.prompt_key]

    # ------------------------------------------------------------------
    # Step 3a: Non-tool route — single LLM call
    # ------------------------------------------------------------------
    if not route_cfg.has_tools:
        route_messages = [{"role": "user", "content": input_text}]
        route_response = await call_model(
            route_cfg.model,
            route_prompt,
            route_messages,
            max_tokens=route_cfg.max_tokens,
            dry_run=dry_run,
        )

        route_record = build_llm_step(
            step_name=route_cfg.step_name,
            response=route_response,
            system_prompt=route_prompt,
            output_format=route_cfg.output_format,
        )
        records.append(route_record)
        return records

    # ------------------------------------------------------------------
    # Step 3b: Tool-enabled route — iterative tool-call loop
    # ------------------------------------------------------------------
    tool_definitions_tokens = _estimate_tool_schema_tokens(tool_schemas)
    conversation: list[dict[str, Any]] = [{"role": "user", "content": input_text}]

    initial_response = await call_model(
        route_cfg.model,
        route_prompt,
        conversation,
        max_tokens=route_cfg.max_tokens,
        tools=tool_schemas,
        dry_run=dry_run,
    )

    initial_record = build_llm_step(
        step_name=route_cfg.step_name,
        response=initial_response,
        system_prompt=route_prompt,
        output_format=route_cfg.output_format,
        tool_definitions_tokens=tool_definitions_tokens,
    )
    records.append(initial_record)

    # Append assistant message to conversation for subsequent rounds
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": initial_response.content}
    if initial_response.tool_calls:
        assistant_msg["tool_calls"] = initial_response.tool_calls
    conversation.append(assistant_msg)

    # Iterate tool call rounds
    current_tool_calls = initial_response.tool_calls
    for round_idx in range(1, max_tool_rounds + 1):
        if not current_tool_calls:
            break

        # Execute each tool call and append results to conversation
        for tc in current_tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            arguments_raw = tc.get("function", {}).get("arguments", "{}")
            tool_call_id = tc.get("id", "")

            try:
                arguments = json.loads(arguments_raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "Failed to parse tool arguments for %s: %s", tool_name, exc
                )
                arguments = {}

            if tool_simulator is not None:
                tool_result = tool_simulator(tool_name, arguments)
            else:
                tool_result = json.dumps({"error": "No tool simulator configured"})

            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result,
            })

        # Call LLM again with tool results
        followup_response = await call_model(
            route_cfg.model,
            route_prompt,
            conversation,
            max_tokens=route_cfg.max_tokens,
            tools=tool_schemas,
            dry_run=dry_run,
        )

        followup_record = build_llm_step(
            step_name=f"{route_cfg.step_name}_tool_round_{round_idx}",
            response=followup_response,
            system_prompt=route_prompt,
            output_format=route_cfg.output_format,
            iteration=round_idx + 1,
            tool_definitions_tokens=tool_definitions_tokens,
        )
        records.append(followup_record)

        # Update conversation with assistant response
        followup_msg: dict[str, Any] = {
            "role": "assistant",
            "content": followup_response.content,
        }
        if followup_response.tool_calls:
            followup_msg["tool_calls"] = followup_response.tool_calls
        conversation.append(followup_msg)

        current_tool_calls = followup_response.tool_calls

    return records


def _estimate_tool_schema_tokens(schemas: list[dict] | None) -> int:
    """Rough token estimate for tool schemas (chars / 4)."""
    if not schemas:
        return 0
    return len(json.dumps(schemas)) // 4
