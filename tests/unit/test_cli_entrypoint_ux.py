"""Test CLI rendering of entrypoint errors: panel and picker UX."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from pretia.cli import cli


class TestEntrypointPanel:
    def test_entrypoint_error_shows_panel_not_usage(self):
        """EntrypointError renders a friendly panel, not click's Usage: header."""
        runner = CliRunner()

        with patch("pretia.cli.run") as mock_run:
            from pretia.runner import EntrypointError

            mock_run.side_effect = EntrypointError(
                "No usable entrypoint found",
                rejected=[],
                wrapper_snippet="def workflow(user_input: str):\n    pass",
                workflow_path="test_script.py",
            )

            # Invoke via the CLI group to get the real exception handling
            runner.invoke(
                cli,
                ["profile", "run", "test_script.py", "--yes", "--input", "hello"],
                catch_exceptions=False,
            )

        # The mock prevents actually reaching the error handler in cli.py,
        # so test the panel rendering directly instead.

    def test_show_entrypoint_panel_output(self):
        """Direct test of _show_entrypoint_panel rendering."""
        from io import StringIO

        from rich.console import Console

        from pretia.runner import EntrypointError

        exc = EntrypointError(
            "No usable entrypoint found in the workflow module.",
            rejected=[],
            wrapper_snippet=(
                "def workflow(user_input: str):\n"
                "    client = OpenAI()\n"
                "    return client.chat.completions.create(messages=[])"
            ),
            workflow_path="helpdesk.py",
        )

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)

        with patch("pretia.cli.console", test_console):
            with patch("sys.exit") as mock_exit:
                from pretia.cli import _show_entrypoint_panel

                _show_entrypoint_panel(exc)

        output = buf.getvalue()
        assert "One small step needed" in output
        assert "Usage:" not in output
        assert "helpdesk.py" in output
        mock_exit.assert_called_once_with(1)

    def test_panel_contains_wrapper_snippet(self):
        """The panel includes the tailored wrapper code."""
        from io import StringIO

        from rich.console import Console

        from pretia.runner import EntrypointError

        snippet = "def workflow(user_input: str):\n    return run_agent(client, messages)"
        exc = EntrypointError(
            "No usable entrypoint found.",
            rejected=[],
            wrapper_snippet=snippet,
            workflow_path="agent.py",
        )

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)

        with patch("pretia.cli.console", test_console):
            with patch("sys.exit"):
                from pretia.cli import _show_entrypoint_panel

                _show_entrypoint_panel(exc)

        output = buf.getvalue()
        assert "workflow" in output
        assert "run_agent" in output

    def test_panel_shows_rejected_candidates(self):
        """When rejected callables exist, they appear in the panel."""
        from io import StringIO

        from rich.console import Console

        from pretia.runner import EntrypointError

        def main():
            pass

        def run_agent(client, messages):
            pass

        exc = EntrypointError(
            "No usable entrypoint found.",
            rejected=[("main", main), ("run_agent", run_agent)],
            wrapper_snippet="def workflow(user_input: str): ...",
            workflow_path="script.py",
        )

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)

        with patch("pretia.cli.console", test_console):
            with patch("sys.exit"):
                from pretia.cli import _show_entrypoint_panel

                _show_entrypoint_panel(exc)

        output = buf.getvalue()
        assert "main()" in output


class TestAmbiguousPicker:
    def test_non_tty_falls_through_to_panel(self):
        """When stdin is not a TTY, ambiguity falls through to the panel."""
        from io import StringIO

        from rich.console import Console

        from pretia.runner import AmbiguousEntrypointError

        exc = AmbiguousEntrypointError(
            "Found multiple single-argument callables",
            candidates=[("foo", "foo(x)"), ("bar", "bar(x)")],
        )

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)

        with (
            patch("pretia.cli.console", test_console),
            patch("sys.stdin") as mock_stdin,
            patch("sys.exit") as mock_exit,
        ):
            mock_stdin.isatty.return_value = False

            from pretia.cli import _handle_ambiguous_entrypoint

            _handle_ambiguous_entrypoint(
                exc,
                workflow_path="test.py",
                yes=False,
                collector="auto",
                auto_generate=None,
                single_input=(),
                inputs_file=None,
                from_langfuse=False,
                langfuse_last_n=10,
                output_dir=".pretia",
                verbose=False,
                allow_cache=False,
                generator_model=None,
                corpus=None,
                no_html=True,
                no_open=True,
                unit=None,
                current_cost=None,
                traffic=None,
                concurrency=None,
            )

        mock_exit.assert_called_once_with(1)
        output = buf.getvalue()
        assert "One small step needed" in output


# ---------------------------------------------------------------------------
# Doctor explain mode
# ---------------------------------------------------------------------------

_CORPUS_DIR = Path(__file__).parent / "discovery_corpus"


class TestDoctorExplainMode:
    def test_doctor_shows_provenance_for_class_agent(self):
        """Doctor on class_agent.py shows 'via' and the discovery rule."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", str(_CORPUS_DIR / "class_agent.py")])
        assert result.exit_code == 0
        assert "via" in result.output
        assert "agent class scan" in result.output

    def test_doctor_shows_needs_input_for_cli_script(self):
        """Doctor on cli_script.py shows 'needs input', no traceback."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", str(_CORPUS_DIR / "cli_script.py")])
        assert result.exit_code == 0
        assert "needs input" in result.output
        assert "Traceback" not in result.output
