"""Tests for agentcost.toml configuration loading."""

from __future__ import annotations

import dataclasses

import pytest

from agentcost.config import (
    AgentCostConfig,
    ProfileConfig,
    ProjectConfig,
    ReportConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_project_config_defaults(self):
        cfg = ProjectConfig()
        assert cfg.name is None

    def test_profile_config_defaults(self):
        cfg = ProfileConfig()
        assert cfg.default_volume == (100, 1000, 10000)
        assert cfg.baseline_path == ".agentcost/baseline.json"

    def test_report_config_defaults(self):
        cfg = ReportConfig()
        assert cfg.output_dir == ".agentcost/reports"

    def test_agentcost_config_all_defaults(self):
        cfg = AgentCostConfig()
        assert cfg.project.name is None
        assert cfg.profile.default_volume == (100, 1000, 10000)
        assert cfg.profile.baseline_path == ".agentcost/baseline.json"
        assert cfg.report.output_dir == ".agentcost/reports"

    def test_frozen(self):
        cfg = AgentCostConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.project = ProjectConfig(name="x")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_config — file not found
# ---------------------------------------------------------------------------


class TestLoadConfigMissing:
    def test_file_not_found_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg == AgentCostConfig()

    def test_default_path_not_found_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg == AgentCostConfig()


# ---------------------------------------------------------------------------
# load_config — parsing
# ---------------------------------------------------------------------------


class TestLoadConfigParsing:
    def test_loads_project_name(self, tmp_path):
        toml = tmp_path / "agentcost.toml"
        toml.write_text('[project]\nname = "my-agent"\n')
        cfg = load_config(toml)
        assert cfg.project.name == "my-agent"

    def test_loads_profile_section(self, tmp_path):
        toml = tmp_path / "agentcost.toml"
        toml.write_text(
            '[profile]\ndefault_volume = [500, 5000]\nbaseline_path = "custom/baseline.json"\n'
        )
        cfg = load_config(toml)
        assert cfg.profile.default_volume == (500, 5000)
        assert cfg.profile.baseline_path == "custom/baseline.json"

    def test_loads_report_section(self, tmp_path):
        toml = tmp_path / "agentcost.toml"
        toml.write_text('[report]\noutput_dir = "build/reports"\n')
        cfg = load_config(toml)
        assert cfg.report.output_dir == "build/reports"

    def test_partial_toml_uses_defaults_for_missing_sections(self, tmp_path):
        toml = tmp_path / "agentcost.toml"
        toml.write_text('[project]\nname = "partial"\n')
        cfg = load_config(toml)
        assert cfg.project.name == "partial"
        assert cfg.profile == ProfileConfig()
        assert cfg.report == ReportConfig()

    def test_unknown_keys_ignored(self, tmp_path):
        toml = tmp_path / "agentcost.toml"
        toml.write_text('[project]\nname = "x"\nfoo = "bar"\n\n[unknown_section]\nkey = 42\n')
        cfg = load_config(toml)
        assert cfg.project.name == "x"

    def test_empty_toml_returns_defaults(self, tmp_path):
        toml = tmp_path / "agentcost.toml"
        toml.write_text("")
        cfg = load_config(toml)
        assert cfg == AgentCostConfig()
