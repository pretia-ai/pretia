"""Tests for Langfuse trace importer (fully mocked, no real API calls)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the langfuse module before importing our importer
# ---------------------------------------------------------------------------

_mock_langfuse = MagicMock()
_mock_langfuse_api = MagicMock()
_mock_langfuse_api_client = MagicMock()

sys.modules.setdefault("langfuse", _mock_langfuse)
sys.modules.setdefault("langfuse.api", _mock_langfuse_api)
sys.modules.setdefault("langfuse.api.client", _mock_langfuse_api_client)

from agentcost.inputs.importer import (  # noqa: E402
    LangfuseObservation,
    LangfuseTrace,
    _compute_duration_ms,
    _extract_input_text,
    create_langfuse_client,
    extract_inputs,
    fetch_traces,
    traces_to_step_records,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockUsage:
    def __init__(self, input_val=100, output_val=50):
        self.input = input_val
        self.output = output_val
        self.total = input_val + output_val


class MockObservationView:
    def __init__(
        self,
        obs_id="obs-1",
        name="classify",
        obs_type="GENERATION",
        model="gpt-4o",
        usage=None,
        start_time=None,
        end_time=None,
        parent_observation_id=None,
    ):
        self.id = obs_id
        self.name = name
        self.type = obs_type
        self.model = model
        self.usage = usage if usage is not None else MockUsage()
        self.start_time = start_time or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        self.end_time = end_time or datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
        self.parent_observation_id = parent_observation_id


class MockTraceWithDetails:
    def __init__(self, trace_id="trace-1", name="my_agent", input_val="Hello"):
        self.id = trace_id
        self.name = name
        self.observations = ["obs-1"]


class MockTraceWithFullDetails:
    def __init__(
        self,
        trace_id="trace-1",
        name="my_agent",
        input_val="Hello, help me",
        observations=None,
        timestamp=None,
    ):
        self.id = trace_id
        self.name = name
        self.input = input_val
        self.timestamp = timestamp or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        self.observations = observations or []


def _make_observation(**kwargs):
    return LangfuseObservation(
        observation_id=kwargs.get("observation_id", "obs-1"),
        name=kwargs.get("name", "classify"),
        observation_type=kwargs.get("observation_type", "GENERATION"),
        model=kwargs.get("model", "gpt-4o"),
        input_tokens=kwargs.get("input_tokens", 100),
        output_tokens=kwargs.get("output_tokens", 50),
        start_time=kwargs.get("start_time", datetime(2026, 1, 1, tzinfo=UTC)),
        end_time=kwargs.get("end_time", datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)),
        duration_ms=kwargs.get("duration_ms", 1000),
        parent_observation_id=kwargs.get("parent_observation_id"),
    )


def _make_trace(**kwargs):
    return LangfuseTrace(
        trace_id=kwargs.get("trace_id", "trace-1"),
        name=kwargs.get("name", "my_agent"),
        input_text=kwargs.get("input_text", "Hello, help me"),
        timestamp=kwargs.get("timestamp", datetime(2026, 1, 1, tzinfo=UTC)),
        observations=kwargs.get("observations", [_make_observation()]),
        total_input_tokens=kwargs.get("total_input_tokens", 100),
        total_output_tokens=kwargs.get("total_output_tokens", 50),
        total_cost=kwargs.get("total_cost", 0.001),
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_compute_duration_ms(self):
        start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = start + timedelta(seconds=1.5)
        assert _compute_duration_ms(start, end) == 1500

    def test_compute_duration_ms_none(self):
        assert _compute_duration_ms(None, None) == 0
        assert _compute_duration_ms(datetime.now(UTC), None) == 0

    def test_extract_input_text_string(self):
        assert _extract_input_text("Hello world") == "Hello world"

    def test_extract_input_text_none(self):
        assert _extract_input_text(None) is None

    def test_extract_input_text_empty_string(self):
        assert _extract_input_text("  ") is None

    def test_extract_input_text_dict_messages(self):
        data = {"messages": [{"role": "user", "content": "Help me"}]}
        assert _extract_input_text(data) == "Help me"

    def test_extract_input_text_dict_content(self):
        data = {"content": "Direct content"}
        assert _extract_input_text(data) == "Direct content"

    def test_extract_input_text_dict_input(self):
        data = {"input": "Input field"}
        assert _extract_input_text(data) == "Input field"

    def test_extract_input_text_fallback_str(self):
        data = [1, 2, 3]
        result = _extract_input_text(data)
        assert result == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# create_langfuse_client
# ---------------------------------------------------------------------------


class TestCreateClient:
    def test_create_client_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://my-langfuse.com")

        mock_api_class = MagicMock()
        mock_client = MagicMock(LangfuseAPI=mock_api_class)
        with patch.dict(sys.modules, {"langfuse.api.client": mock_client}):
            import importlib

            from agentcost.inputs import importer

            importlib.reload(importer)

            mock_api_class.reset_mock()
            with patch.object(importer, "LangfuseAPI", mock_api_class, create=True):
                pass

        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://my-langfuse.com")
        client = create_langfuse_client()
        assert client is not None

    def test_create_client_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

        with pytest.raises(OSError, match="LANGFUSE_SECRET_KEY"):
            create_langfuse_client()

    def test_create_client_missing_secret_only(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")

        with pytest.raises(OSError, match="LANGFUSE_SECRET_KEY"):
            create_langfuse_client()

    def test_create_client_missing_public_only(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

        with pytest.raises(OSError, match="LANGFUSE_PUBLIC_KEY"):
            create_langfuse_client()


# ---------------------------------------------------------------------------
# fetch_traces
# ---------------------------------------------------------------------------


class TestFetchTraces:
    def test_fetch_traces_basic(self):
        obs1 = MockObservationView(obs_id="obs-1", name="classify", model="gpt-4o")
        obs2 = MockObservationView(obs_id="obs-2", name="respond", model="gpt-4o-mini")

        mock_client = MagicMock()
        trace_summary = MockTraceWithDetails("trace-1", "my_agent")
        traces_response = MagicMock()
        traces_response.data = [trace_summary]
        mock_client.trace.list.return_value = traces_response

        full_trace = MockTraceWithFullDetails(
            "trace-1",
            "my_agent",
            "Hello",
            [obs1, obs2],
        )
        mock_client.trace.get.return_value = full_trace

        result = fetch_traces(mock_client, last_n=3)

        assert len(result) == 1
        assert result[0].trace_id == "trace-1"
        assert result[0].name == "my_agent"
        assert result[0].input_text == "Hello"
        assert len(result[0].observations) == 2
        assert result[0].total_input_tokens == 200
        assert result[0].total_output_tokens == 100

    def test_fetch_traces_with_name_filter(self):
        mock_client = MagicMock()
        traces_response = MagicMock()
        traces_response.data = []
        mock_client.trace.list.return_value = traces_response

        fetch_traces(mock_client, last_n=5, name="my_agent")

        call_kwargs = mock_client.trace.list.call_args
        assert call_kwargs.kwargs.get("name") == "my_agent"

    def test_fetch_traces_empty_result(self):
        mock_client = MagicMock()
        traces_response = MagicMock()
        traces_response.data = []
        mock_client.trace.list.return_value = traces_response

        result = fetch_traces(mock_client)
        assert result == []

    def test_fetch_traces_missing_usage(self):
        obs = MockObservationView(obs_id="obs-1", name="step", usage=None)
        obs.usage = None

        mock_client = MagicMock()
        trace_summary = MockTraceWithDetails("t-1")
        traces_response = MagicMock()
        traces_response.data = [trace_summary]
        mock_client.trace.list.return_value = traces_response
        mock_client.trace.get.return_value = MockTraceWithFullDetails(
            "t-1",
            "agent",
            "input",
            [obs],
        )

        result = fetch_traces(mock_client, last_n=1)

        assert len(result) == 1
        assert result[0].observations[0].input_tokens == 0
        assert result[0].observations[0].output_tokens == 0

    def test_fetch_traces_missing_input_text(self):
        mock_client = MagicMock()
        trace_summary = MockTraceWithDetails("t-1")
        traces_response = MagicMock()
        traces_response.data = [trace_summary]
        mock_client.trace.list.return_value = traces_response
        mock_client.trace.get.return_value = MockTraceWithFullDetails(
            "t-1",
            "agent",
            input_val=None,
            observations=[],
        )

        result = fetch_traces(mock_client, last_n=1)

        assert len(result) == 1
        assert result[0].input_text is None

    def test_fetch_traces_auth_error(self):
        mock_client = MagicMock()
        mock_client.trace.list.side_effect = Exception("401 Unauthorized")

        with pytest.raises(PermissionError, match="authentication failed"):
            fetch_traces(mock_client)

    def test_fetch_traces_connection_error(self):
        mock_client = MagicMock()
        mock_client.trace.list.side_effect = Exception("Connection refused")

        with pytest.raises(ConnectionError, match="Failed to connect"):
            fetch_traces(mock_client)

    def test_fetch_traces_caps_at_100(self):
        mock_client = MagicMock()
        traces_response = MagicMock()
        traces_response.data = []
        mock_client.trace.list.return_value = traces_response

        fetch_traces(mock_client, last_n=500)

        call_kwargs = mock_client.trace.list.call_args
        assert call_kwargs.kwargs.get("limit") == 100


# ---------------------------------------------------------------------------
# traces_to_step_records
# ---------------------------------------------------------------------------


class TestTracesToStepRecords:
    def test_basic_conversion(self):
        obs1 = _make_observation(
            name="classify", model="gpt-4o", input_tokens=100, output_tokens=50
        )
        obs2 = _make_observation(
            observation_id="obs-2",
            name="respond",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
        )
        traces = [_make_trace(observations=[obs1, obs2])]

        runs = traces_to_step_records(traces)

        assert len(runs) == 1
        assert len(runs[0]) == 2
        assert runs[0][0].step_name == "classify"
        assert runs[0][0].model == "gpt-4o"
        assert runs[0][0].input_tokens == 100
        assert runs[0][0].step_type == "llm"
        assert runs[0][1].step_name == "respond"

    def test_two_traces_two_runs(self):
        t1 = _make_trace(trace_id="t-1", observations=[_make_observation()])
        t2 = _make_trace(trace_id="t-2", observations=[_make_observation()])

        runs = traces_to_step_records([t1, t2])
        assert len(runs) == 2

    def test_iteration_counting(self):
        obs_list = [
            _make_observation(observation_id="o-1", name="review"),
            _make_observation(observation_id="o-2", name="review"),
            _make_observation(observation_id="o-3", name="review"),
        ]
        traces = [_make_trace(observations=obs_list)]

        runs = traces_to_step_records(traces)
        iterations = [r.iteration for r in runs[0]]
        assert iterations == [1, 2, 3]

    def test_parent_step_mapping(self):
        parent_obs = _make_observation(observation_id="parent-1", name="orchestrator")
        child_obs = _make_observation(
            observation_id="child-1",
            name="classify",
            parent_observation_id="parent-1",
        )
        traces = [_make_trace(observations=[parent_obs, child_obs])]

        runs = traces_to_step_records(traces)
        assert runs[0][1].parent_step == "orchestrator"

    def test_skips_event_observations(self):
        gen_obs = _make_observation(name="classify", observation_type="GENERATION")
        event_obs = _make_observation(
            observation_id="ev-1",
            name="log_event",
            observation_type="EVENT",
        )
        traces = [_make_trace(observations=[gen_obs, event_obs])]

        runs = traces_to_step_records(traces)
        assert len(runs[0]) == 1
        assert runs[0][0].step_name == "classify"

    def test_span_mapped_to_tool(self):
        span_obs = _make_observation(
            name="search_db",
            observation_type="SPAN",
            model=None,
        )
        traces = [_make_trace(observations=[span_obs])]

        runs = traces_to_step_records(traces)
        assert runs[0][0].step_type == "tool"
        assert runs[0][0].model == "unknown"

    def test_retrieval_detection(self):
        retrieval_obs = _make_observation(
            name="retrieve_documents",
            observation_type="SPAN",
            model=None,
        )
        traces = [_make_trace(observations=[retrieval_obs])]

        runs = traces_to_step_records(traces)
        assert runs[0][0].step_type == "retrieval"

    def test_empty_traces(self):
        runs = traces_to_step_records([])
        assert runs == []

    def test_step_record_validation_passes(self):
        traces = [_make_trace()]
        runs = traces_to_step_records(traces)
        for run in runs:
            for rec in run:
                assert rec.step_type in {"llm", "tool", "retrieval"}
                assert rec.iteration >= 1
                assert rec.input_tokens >= 0


# ---------------------------------------------------------------------------
# extract_inputs
# ---------------------------------------------------------------------------


class TestExtractInputs:
    def test_extract_inputs_basic(self):
        traces = [_make_trace(trace_id=f"t-{i}", input_text=f"input {i}") for i in range(5)]
        result = extract_inputs(traces)
        assert len(result) == 5
        assert result[0] == "input 0"

    def test_extract_inputs_insufficient(self):
        traces = [
            _make_trace(trace_id="t-1", input_text="hello"),
            _make_trace(trace_id="t-2", input_text=None),
            _make_trace(trace_id="t-3", input_text=None),
        ]
        with pytest.raises(ValueError, match="Only 1 of 3"):
            extract_inputs(traces)

    def test_extract_inputs_none_filtered(self):
        traces = [
            _make_trace(trace_id="t-1", input_text="hello"),
            _make_trace(trace_id="t-2", input_text=None),
            _make_trace(trace_id="t-3", input_text="world"),
        ]
        result = extract_inputs(traces)
        assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# Observation duration
# ---------------------------------------------------------------------------


class TestObservationDuration:
    def test_observation_duration_ms(self):
        start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = start + timedelta(seconds=1.5)
        obs = _make_observation(
            start_time=start,
            end_time=end,
            duration_ms=_compute_duration_ms(start, end),
        )
        assert obs.duration_ms == 1500


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_observation_to_dict(self):
        obs = _make_observation()
        d = obs.to_dict()
        json.dumps(d)
        assert d["observation_id"] == "obs-1"
        assert d["name"] == "classify"

    def test_trace_to_dict(self):
        trace = _make_trace()
        d = trace.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["trace_id"] == "trace-1"
