"""Tests for pretia.estimate — static analysis engine and estimate CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from pretia.cli import cli
from pretia.estimate import (
    ModelEstimate,
    _detect_framework,
    _estimate_cost,
    _estimate_tokens,
    _extract_models,
    _extract_system_prompts,
    estimate_workflow,
)

runner = CliRunner()


# ===========================================================================
# Static analysis engine
# ===========================================================================


class TestDetectFramework:
    def test_langgraph_import(self) -> None:
        src = "from langgraph.graph import StateGraph\n"
        assert _detect_framework(src) == "langgraph"

    def test_langgraph_import_alt(self) -> None:
        src = "import langgraph\n"
        assert _detect_framework(src) == "langgraph"

    def test_openai_agents(self) -> None:
        src = "from agents import Runner\n"
        assert _detect_framework(src) == "openai-agents"

    def test_qwen_agent(self) -> None:
        src = "from qwen_agent.agent import Agent\n"
        assert _detect_framework(src) == "qwen-agent"

    def test_generic_fallback(self) -> None:
        src = "import os\nimport sys\n"
        assert _detect_framework(src) == "generic"

    def test_empty_source(self) -> None:
        assert _detect_framework("") == "generic"


class TestExtractModels:
    def test_single_model_kwarg(self) -> None:
        src = 'run_step(model="claude-haiku-4-5", input="hello")\n'
        models = _extract_models(src)
        assert len(models) == 1
        assert models[0]["model_name"] == "claude-haiku-4-5"

    def test_multiple_model_kwargs(self) -> None:
        src = (
            'run_step(model="claude-haiku-4-5", step_name="classify")\n'
            'run_step(model="gpt-4o", step_name="generate")\n'
        )
        models = _extract_models(src)
        assert len(models) == 2
        names = {m["model_name"] for m in models}
        assert names == {"claude-haiku-4-5", "gpt-4o"}

    def test_alternate_model_kwarg(self) -> None:
        src = 'run_step(model="claude-haiku-4-5", alternate_model="claude-sonnet-4-6")\n'
        models = _extract_models(src)
        names = {m["model_name"] for m in models}
        assert "claude-haiku-4-5" in names
        assert "claude-sonnet-4-6" in names

    def test_classifier_model_kwarg(self) -> None:
        src = 'run_router(classifier_model="claude-haiku-4-5")\n'
        models = _extract_models(src)
        assert len(models) == 1
        assert models[0]["model_name"] == "claude-haiku-4-5"

    def test_max_tokens_extraction(self) -> None:
        src = 'run_step(model="claude-haiku-4-5", max_tokens=1024)\n'
        models = _extract_models(src)
        assert models[0]["max_tokens"] == 1024

    def test_step_name_extraction(self) -> None:
        src = 'run_step(model="gpt-4o", step_name="classify_intent")\n'
        models = _extract_models(src)
        assert models[0]["step_name"] == "classify_intent"

    def test_no_models(self) -> None:
        src = "x = 42\nprint('hello')\n"
        assert _extract_models(src) == []

    def test_syntax_error(self) -> None:
        src = "def broken(\n"
        assert _extract_models(src) == []

    def test_keeps_duplicate_model_call_sites(self) -> None:
        src = 'step1(model="claude-haiku-4-5")\nstep2(model="claude-haiku-4-5")\n'
        models = _extract_models(src)
        assert len(models) == 2

    def test_non_string_model_ignored(self) -> None:
        src = "run_step(model=some_variable)\n"
        assert _extract_models(src) == []


class TestExtractSystemPrompts:
    def test_system_prompt_kwarg(self) -> None:
        src = (
            'agent(model="gpt-4o", system_prompt='
            '"You are a helpful assistant that answers questions '
            'about insurance policies.")\n'
        )
        prompts = _extract_system_prompts(src)
        assert len(prompts) == 1
        assert "insurance" in prompts[0]

    def test_system_message_kwarg(self) -> None:
        src = (
            'chat(system_message="You are an expert financial '
            'advisor who helps with tax questions.")\n'
        )
        prompts = _extract_system_prompts(src)
        assert len(prompts) == 1
        assert "financial" in prompts[0]

    def test_instructions_kwarg(self) -> None:
        src = 'Agent(instructions="You are a customer support agent for a SaaS product.")\n'
        prompts = _extract_system_prompts(src)
        assert len(prompts) == 1
        assert "customer support" in prompts[0]

    def test_no_prompts(self) -> None:
        src = "x = 42\nprint('hello')\n"
        assert _extract_system_prompts(src) == []

    def test_short_strings_ignored(self) -> None:
        src = 'agent(system_prompt="short")\n'
        assert _extract_system_prompts(src) == []

    def test_multiple_prompts(self) -> None:
        src = (
            "step1(system_prompt="
            '"You are a classifier that categorizes '
            'customer inquiries into topics.")\n'
            "step2(system_prompt="
            '"You are a response generator that writes '
            'helpful customer replies.")\n'
        )
        prompts = _extract_system_prompts(src)
        assert len(prompts) == 2

    def test_deduplicates(self) -> None:
        src = (
            'step1(system_prompt="You are a helpful AI that answers questions accurately.")\n'
            'step2(system_prompt="You are a helpful AI that answers questions accurately.")\n'
        )
        prompts = _extract_system_prompts(src)
        assert len(prompts) == 1

    def test_syntax_error(self) -> None:
        src = "def broken(\n"
        assert _extract_system_prompts(src) == []

    def test_non_string_value_ignored(self) -> None:
        src = "agent(system_prompt=some_variable)\n"
        assert _extract_system_prompts(src) == []


class TestEstimateTokens:
    def test_basic(self) -> None:
        text = "one two three four five"
        result = _estimate_tokens(text)
        assert result == 7  # ceil(5 * 1.3) = 7

    def test_empty(self) -> None:
        assert _estimate_tokens("") == 0

    def test_single_word(self) -> None:
        assert _estimate_tokens("hello") == 2  # ceil(1 * 1.3) = 2

    def test_long_text(self) -> None:
        text = " ".join(["word"] * 100)
        assert _estimate_tokens(text) == 130  # ceil(100 * 1.3) = 130


class TestEstimateCost:
    def test_known_model(self) -> None:
        models = [
            ModelEstimate(
                model_name="claude-haiku-4-5",
                canonical_name="claude-haiku-4-5",
                step_name=None,
                max_tokens=None,
                input_price_per_m=1.0,
                output_price_per_m=5.0,
            ),
        ]
        cost = _estimate_cost(models)
        assert cost > 0

    def test_unknown_model_skipped(self) -> None:
        models = [
            ModelEstimate(
                model_name="my-custom-model",
                canonical_name=None,
                step_name=None,
                max_tokens=None,
                input_price_per_m=None,
                output_price_per_m=None,
            ),
        ]
        assert _estimate_cost(models) == 0.0

    def test_max_tokens_affects_cost(self) -> None:
        base = ModelEstimate(
            model_name="gpt-4o",
            canonical_name="gpt-4o",
            step_name=None,
            max_tokens=None,
            input_price_per_m=2.5,
            output_price_per_m=10.0,
        )
        with_tokens = ModelEstimate(
            model_name="gpt-4o",
            canonical_name="gpt-4o",
            step_name=None,
            max_tokens=4096,
            input_price_per_m=2.5,
            output_price_per_m=10.0,
        )
        cost_default = _estimate_cost([base])
        cost_with_max = _estimate_cost([with_tokens])
        assert cost_default != cost_with_max

    def test_empty_models(self) -> None:
        assert _estimate_cost([]) == 0.0

    def test_with_system_prompt_tokens_above_default(self) -> None:
        models = [
            ModelEstimate(
                model_name="gpt-4o",
                canonical_name="gpt-4o",
                step_name=None,
                max_tokens=None,
                input_price_per_m=2.5,
                output_price_per_m=10.0,
            ),
        ]
        cost_default = _estimate_cost(models)
        cost_with_sp = _estimate_cost(models, system_prompt_tokens=2000)
        assert cost_with_sp > cost_default

    def test_with_system_prompt_tokens_uses_sp_plus_user_input(self) -> None:
        models = [
            ModelEstimate(
                model_name="gpt-4o",
                canonical_name="gpt-4o",
                step_name=None,
                max_tokens=None,
                input_price_per_m=2.5,
                output_price_per_m=10.0,
            ),
        ]
        cost_default = _estimate_cost(models)
        cost_with_sp = _estimate_cost(models, system_prompt_tokens=200)
        assert cost_with_sp < cost_default


class TestEstimateWorkflow:
    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            estimate_workflow("/nonexistent/path.py")

    def test_simple_workflow(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text(
            "from langgraph.graph import StateGraph\n"
            "def build():\n"
            '    run_step(model="claude-haiku-4-5", max_tokens=1024)\n'
        )
        est = estimate_workflow(str(wf))
        assert est.framework == "langgraph"
        assert len(est.models) == 1
        assert est.models[0].model_name == "claude-haiku-4-5"
        assert est.models[0].canonical_name == "claude-haiku-4-5"
        assert est.estimated_cost_per_run > 0

    def test_no_models_workflow(self, tmp_path: Path) -> None:
        wf = tmp_path / "empty.py"
        wf.write_text("x = 42\n")
        est = estimate_workflow(str(wf))
        assert est.models == []
        assert est.estimated_cost_per_run == 0.0

    def test_unrecognized_model(self, tmp_path: Path) -> None:
        wf = tmp_path / "custom.py"
        wf.write_text('run(model="my-private-llm")\n')
        est = estimate_workflow(str(wf))
        assert len(est.models) == 1
        assert est.models[0].canonical_name is None
        assert est.models[0].input_price_per_m is None

    def test_real_workflow_w01(self) -> None:
        w01 = Path("bt_agents/workflows/w01.py")
        if not w01.exists():
            pytest.skip("bt_agents/workflows/w01.py not available")
        est = estimate_workflow(str(w01))
        assert len(est.models) >= 1
        model_names = {m.model_name for m in est.models}
        assert "claude-haiku-4-5" in model_names

    def test_real_workflow_w13(self) -> None:
        w13 = Path("bt_agents/workflows/w13.py")
        if not w13.exists():
            pytest.skip("bt_agents/workflows/w13.py not available")
        est = estimate_workflow(str(w13))
        assert len(est.models) >= 2

    def test_workflow_estimate_is_frozen(self, tmp_path: Path) -> None:
        wf = tmp_path / "frozen.py"
        wf.write_text('run(model="gpt-4o")\n')
        est = estimate_workflow(str(wf))
        with pytest.raises(AttributeError):
            est.framework = "modified"  # type: ignore[misc]

    def test_system_prompt_extraction_end_to_end(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text(
            "from langgraph.graph import StateGraph\n"
            "agent = Agent(\n"
            '    model="claude-haiku-4-5",\n'
            "    system_prompt="
            '"You are a helpful assistant that classifies '
            'customer inquiries into topics.",\n'
            ")\n"
        )
        est = estimate_workflow(str(wf))
        assert est.estimated_system_prompt_tokens > 0

    def test_no_system_prompt_gives_zero_tokens(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        est = estimate_workflow(str(wf))
        assert est.estimated_system_prompt_tokens == 0


# ===========================================================================
# Estimate CLI command
# ===========================================================================


class TestEstimateCommand:
    def test_help(self) -> None:
        result = runner.invoke(cli, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "WORKFLOW_PATH" in result.output

    def test_basic_estimate(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="claude-haiku-4-5", max_tokens=512)\n')
        result = runner.invoke(cli, ["estimate", str(wf)])
        assert result.exit_code == 0
        assert "claude-haiku-4-5" in result.output
        assert "Static Estimate" in result.output

    def test_no_models_found(self, tmp_path: Path) -> None:
        wf = tmp_path / "empty.py"
        wf.write_text("x = 42\n")
        result = runner.invoke(cli, ["estimate", str(wf)])
        assert result.exit_code == 0
        assert "No models detected" in result.output

    def test_nonexistent_file(self) -> None:
        result = runner.invoke(cli, ["estimate", "nonexistent.py"])
        assert result.exit_code != 0

    def test_with_traffic_flag(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="claude-haiku-4-5")\n')
        result = runner.invoke(cli, ["estimate", str(wf), "--traffic", "5000"])
        assert result.exit_code == 0
        assert "5,000" in result.output

    def test_caveat_message(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        result = runner.invoke(cli, ["estimate", str(wf)])
        assert result.exit_code == 0
        assert "rough estimate" in result.output

    def test_shows_system_prompt_tokens(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text(
            "run(\n"
            '    model="claude-haiku-4-5",\n'
            "    system_prompt="
            '"You are a helpful assistant that answers '
            'questions about insurance policies and claims.",\n'
            ")\n"
        )
        result = runner.invoke(cli, ["estimate", str(wf)])
        assert result.exit_code == 0
        assert "System prompt tokens" in result.output
        assert "extracted from source" in result.output


# ===========================================================================
# CLI confirmation and progress
# ===========================================================================


class TestProfileRunConfirmation:
    def test_yes_flag_in_help(self) -> None:
        result = runner.invoke(cli, ["profile", "run", "--help"])
        assert result.exit_code == 0
        assert "--yes" in result.output

    def test_confirmation_declined(self, tmp_path: Path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")
        result = runner.invoke(
            cli,
            ["profile", "run", str(wf), "--input", "hello"],
            input="n\n",
        )
        assert "Cancelled" in result.output
        assert result.exit_code == 0


class TestInferRunCount:
    def test_single_input(self) -> None:
        from pretia.cli import _infer_run_count

        assert (
            _infer_run_count(
                auto_generate=None,
                single_input=("hello",),
                inputs_file=None,
                from_langfuse=False,
                langfuse_last_n=10,
            )
            == 1
        )

    def test_multiple_inputs(self) -> None:
        from pretia.cli import _infer_run_count

        assert (
            _infer_run_count(
                auto_generate=None,
                single_input=("hello", "world", "test"),
                inputs_file=None,
                from_langfuse=False,
                langfuse_last_n=10,
            )
            == 3
        )

    def test_empty_tuple_falls_through(self) -> None:
        from pretia.cli import _infer_run_count

        assert (
            _infer_run_count(
                auto_generate=None,
                single_input=(),
                inputs_file=None,
                from_langfuse=False,
                langfuse_last_n=10,
            )
            == 50
        )

    def test_auto_generate_explicit(self) -> None:
        from pretia.cli import _infer_run_count

        assert (
            _infer_run_count(
                auto_generate=50,
                single_input=(),
                inputs_file=None,
                from_langfuse=False,
                langfuse_last_n=10,
            )
            == 50
        )

    def test_auto_generate_default(self) -> None:
        from pretia.cli import _infer_run_count

        assert (
            _infer_run_count(
                auto_generate=None,
                single_input=(),
                inputs_file=None,
                from_langfuse=False,
                langfuse_last_n=10,
            )
            == 50
        )

    def test_from_langfuse(self) -> None:
        from pretia.cli import _infer_run_count

        assert (
            _infer_run_count(
                auto_generate=None,
                single_input=(),
                inputs_file=None,
                from_langfuse=True,
                langfuse_last_n=25,
            )
            == 25
        )

    def test_inputs_file(self, tmp_path: Path) -> None:
        from pretia.cli import _infer_run_count

        f = tmp_path / "inputs.txt"
        f.write_text("line1\nline2\nline3\n")
        assert (
            _infer_run_count(
                auto_generate=None,
                single_input=(),
                inputs_file=str(f),
                from_langfuse=False,
                langfuse_last_n=10,
            )
            == 3
        )
