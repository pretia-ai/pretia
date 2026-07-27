"""Tests for collector error surfacing: last_error, callback guards, safe truncation."""

from __future__ import annotations

from pathlib import Path

from pretia.collectors.base import BaseCollector, StepRecord
from pretia.collectors.generic import GenericCollector

_COLLECTORS_DIR = Path(__file__).resolve().parents[2] / "pretia" / "collectors"


class TestLastErrorGenericCollector:
    async def test_last_error_set_on_failure(self):
        """collect() stores the exception in last_error when a workflow raises."""
        collector = GenericCollector()

        async def boom(_inp: str) -> None:
            raise RuntimeError("boom")

        await collector.collect(boom, ["test"])
        assert isinstance(collector.last_error, RuntimeError)
        assert str(collector.last_error) == "boom"

    async def test_last_error_reset_between_calls(self):
        """A successful collect() after a failure clears last_error to None."""
        collector = GenericCollector()

        async def boom(_inp: str) -> None:
            raise RuntimeError("boom")

        await collector.collect(boom, ["test"])
        assert collector.last_error is not None

        async def ok(_inp: str) -> None:
            return None

        await collector.collect(ok, ["test"])
        assert collector.last_error is None


class TestCollectSyncForwardsConcurrency:
    def test_concurrency_forwarded(self):
        """collect_sync passes the concurrency kwarg through to collect()."""
        received: list[int | None] = []

        class Spy(BaseCollector):
            async def collect(self, workflow, inputs, on_run_complete=None, concurrency=None):
                received.append(concurrency)
                return [[]]

        spy = Spy()
        spy.collect_sync(lambda x: x, ["a"], concurrency=3)
        assert received == [3]


class TestOnRunCompleteGuarded:
    async def test_generic_collector_survives_callback_error(self):
        """GenericCollector still returns results when on_run_complete raises."""
        collector = GenericCollector()

        async def workflow(_inp: str) -> None:
            return None

        def bad_callback(_i: int, _t: int, _recs: list[StepRecord]) -> None:
            raise ValueError("callback exploded")

        result = await collector.collect(workflow, ["a", "b"], on_run_complete=bad_callback)
        # collect should return a list per input, not propagate the callback error
        assert len(result) == 2

    def test_openai_agents_has_callback_guard(self):
        """OpenAI Agents collector wraps on_run_complete in a try/except."""
        source = (_COLLECTORS_DIR / "openai_agents.py").read_text()
        assert "on_run_complete" in source
        assert "callback failed" in source

    def test_qwen_agent_has_callback_guard(self):
        """Qwen-Agent collector wraps on_run_complete in a try/except."""
        source = (_COLLECTORS_DIR / "qwen_agent.py").read_text()
        assert "on_run_complete" in source
        assert "callback failed" in source


class TestSafeInputTruncation:
    def test_openai_agents_uses_str_truncation(self):
        """openai_agents.py uses str(inp)[:80] (safe for dicts), not inp[:80]."""
        source = (_COLLECTORS_DIR / "openai_agents.py").read_text()
        assert "str(inp)[:80]" in source

    def test_qwen_agent_uses_str_truncation(self):
        """qwen_agent.py uses str(inp)[:80] (safe for dicts), not inp[:80]."""
        source = (_COLLECTORS_DIR / "qwen_agent.py").read_text()
        assert "str(inp)[:80]" in source
