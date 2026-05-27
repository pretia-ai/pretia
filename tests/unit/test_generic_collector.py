"""Tests for GenericCollector: context manager, decorator, iteration counting, collect()."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from agentcost.collectors.generic import GenericCollector


class MockUsage:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockResponse:
    def __init__(self, model, usage):
        self.model = model
        self.usage = usage


class TestContextManagerExplicit:
    @pytest.mark.asyncio
    async def test_record_llm_call_produces_step_record(self):
        collector = GenericCollector()
        collector.new_run()
        async with collector.step("my_step") as s:
            s.record_llm_call(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        run = collector.end_run()

        assert len(run) == 1
        record = run[0]
        assert record.step_name == "my_step"
        assert record.model == "gpt-4o-mini"
        assert record.input_tokens == 100
        assert record.output_tokens == 50

    @pytest.mark.asyncio
    async def test_no_recording_produces_no_record(self):
        collector = GenericCollector()
        collector.new_run()
        async with collector.step("no_llm"):
            pass
        run = collector.end_run()

        assert len(run) == 0

    @pytest.mark.asyncio
    async def test_timing_captures_duration(self):
        collector = GenericCollector()
        collector.new_run()
        async with collector.step("slow_step") as s:
            await asyncio.sleep(0.05)
            s.record_llm_call(model="gpt-4o", input_tokens=10, output_tokens=5)
        run = collector.end_run()

        assert len(run) == 1
        assert run[0].duration_ms >= 50


class TestDecoratorAutoExtraction:
    @pytest.mark.asyncio
    async def test_openai_style_response(self):
        collector = GenericCollector()
        collector.new_run()

        @collector.step("classify")
        async def classify(text):
            return MockResponse(
                model="gpt-4o",
                usage=MockUsage(prompt_tokens=200, completion_tokens=80),
            )

        await classify("hello")
        run = collector.end_run()

        assert len(run) == 1
        assert run[0].model == "gpt-4o"
        assert run[0].input_tokens == 200
        assert run[0].output_tokens == 80

    @pytest.mark.asyncio
    async def test_anthropic_style_response(self):
        collector = GenericCollector()
        collector.new_run()

        @collector.step("summarize")
        async def summarize(text):
            return MockResponse(
                model="claude-sonnet-4-20250514",
                usage=MockUsage(input_tokens=300, output_tokens=120),
            )

        await summarize("hello")
        run = collector.end_run()

        assert len(run) == 1
        assert run[0].model == "claude-sonnet-4-20250514"
        assert run[0].input_tokens == 300
        assert run[0].output_tokens == 120

    @pytest.mark.asyncio
    async def test_unrecognized_return_produces_no_record(self):
        collector = GenericCollector()
        collector.new_run()

        @collector.step("plain")
        async def plain(text):
            return "just a string"

        result = await plain("hello")
        run = collector.end_run()

        assert result == "just a string"
        assert len(run) == 0


class TestIterationCounting:
    @pytest.mark.asyncio
    async def test_same_step_increments_iteration(self):
        collector = GenericCollector()
        collector.new_run()
        for _ in range(3):
            async with collector.step("classify") as s:
                s.record_llm_call(model="gpt-4o-mini", input_tokens=10, output_tokens=5)
        run = collector.end_run()

        assert len(run) == 3
        assert run[0].iteration == 1
        assert run[1].iteration == 2
        assert run[2].iteration == 3

    @pytest.mark.asyncio
    async def test_iteration_resets_between_runs(self):
        collector = GenericCollector()

        collector.new_run()
        for _ in range(2):
            async with collector.step("classify") as s:
                s.record_llm_call(model="gpt-4o-mini", input_tokens=10, output_tokens=5)
        run1 = collector.end_run()

        collector.new_run()
        async with collector.step("classify") as s:
            s.record_llm_call(model="gpt-4o-mini", input_tokens=10, output_tokens=5)
        run2 = collector.end_run()

        assert run1[0].iteration == 1
        assert run1[1].iteration == 2
        assert run2[0].iteration == 1


class TestCollect:
    @pytest.mark.asyncio
    async def test_collect_runs_workflow_per_input(self):
        collector = GenericCollector()

        async def workflow(inp):
            async with collector.step("echo") as s:
                s.record_llm_call(model="gpt-4o-mini", input_tokens=len(inp), output_tokens=10)

        runs = await collector.collect(workflow, ["short", "a longer input"])

        assert len(runs) == 2
        assert runs[0][0].input_tokens == 5
        assert runs[1][0].input_tokens == 14


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_everything(self):
        collector = GenericCollector()
        collector.new_run()
        async with collector.step("step") as s:
            s.record_llm_call(model="gpt-4o", input_tokens=10, output_tokens=5)
        collector.end_run()

        assert len(collector.all_runs) == 1
        collector.reset()
        assert len(collector.all_runs) == 0


class TestSystemPrompt:
    @pytest.mark.asyncio
    async def test_system_prompt_hashed_and_counted(self):
        collector = GenericCollector()
        collector.new_run()
        prompt = "You are a helpful assistant."
        async with collector.step("chat") as s:
            s.record_llm_call(
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                system_prompt=prompt,
            )
        run = collector.end_run()

        expected_hash = hashlib.sha256(prompt.encode()).hexdigest()
        assert run[0].system_prompt_hash == expected_hash
        assert run[0].system_prompt_tokens == len(prompt) // 4


class TestParentStep:
    @pytest.mark.asyncio
    async def test_parent_step_captured(self):
        collector = GenericCollector()
        collector.new_run()
        async with collector.step("sub_task", parent_step="main_task") as s:
            s.record_llm_call(model="gpt-4o", input_tokens=10, output_tokens=5)
        run = collector.end_run()

        assert run[0].parent_step == "main_task"
