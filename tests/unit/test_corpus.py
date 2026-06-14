"""Tests for agentcost.inputs.corpus — document corpus loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agentcost.inputs.corpus import load_corpus_context


class TestLoadCorpusFile:
    def test_reads_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "summary.txt"
        f.write_text("This is a pre-written corpus summary about insurance policies.")
        result = load_corpus_context(str(f))
        assert "insurance policies" in result

    def test_reads_markdown_file(self, tmp_path: Path) -> None:
        f = tmp_path / "summary.md"
        f.write_text("# Corpus\n\nDocuments about financial products.")
        result = load_corpus_context(str(f))
        assert "financial products" in result


class TestLoadCorpusDirectory:
    def test_scans_txt_files(self, tmp_path: Path) -> None:
        (tmp_path / "doc1.txt").write_text("First document about customer support workflows.")
        (tmp_path / "doc2.txt").write_text("Second document about billing and invoicing.")
        result = load_corpus_context(str(tmp_path))
        assert "Document: doc1.txt" in result
        assert "Document: doc2.txt" in result
        assert "customer support" in result

    def test_scans_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "guide.md").write_text("# User Guide\n\nHow to configure the system.")
        result = load_corpus_context(str(tmp_path))
        assert "Document: guide.md" in result
        assert "configure" in result

    def test_mixed_file_types(self, tmp_path: Path) -> None:
        (tmp_path / "readme.md").write_text("Product readme content here.")
        (tmp_path / "data.txt").write_text("Raw data file with important info.")
        (tmp_path / "script.py").write_text("print('not a document')")
        result = load_corpus_context(str(tmp_path))
        assert "Document: readme.md" in result
        assert "Document: data.txt" in result
        assert "script.py" not in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = load_corpus_context(str(tmp_path))
        assert result == ""

    def test_word_limit_truncation(self, tmp_path: Path) -> None:
        long_text = " ".join(["word"] * 500)
        (tmp_path / "long.txt").write_text(long_text)
        result = load_corpus_context(str(tmp_path))
        words_in_result = result.split()
        assert len(words_in_result) < 250
        assert "..." in result

    def test_subdirectory_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested document content.")
        result = load_corpus_context(str(tmp_path))
        assert "Document: nested.txt" in result


class TestLoadCorpusErrors:
    def test_nonexistent_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus_context("/nonexistent/path")

    def test_pdf_without_pdfplumber(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake pdf content")
        with patch("agentcost.inputs.corpus.logger") as mock_logger:
            result = load_corpus_context(str(tmp_path))
        assert "doc.pdf" not in result or result == ""
        mock_logger.warning.assert_called()


class TestLoadCorpusCLI:
    def test_corpus_in_help(self) -> None:
        from click.testing import CliRunner

        from agentcost.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["profile", "run", "--help"])
        assert result.exit_code == 0
        assert "--corpus" in result.output
