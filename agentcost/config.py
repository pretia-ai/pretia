"""Load project configuration from agentcost.toml."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_VOLUMES = (100, 1000, 10000)
_DEFAULT_BASELINE_PATH = ".agentcost/baseline.json"
_DEFAULT_OUTPUT_DIR = ".agentcost/reports"


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """[project] section of agentcost.toml."""

    name: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """[profile] section of agentcost.toml."""

    default_volume: tuple[int, ...] = _DEFAULT_VOLUMES
    baseline_path: str = _DEFAULT_BASELINE_PATH


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """[report] section of agentcost.toml."""

    output_dir: str = _DEFAULT_OUTPUT_DIR


@dataclass(frozen=True, slots=True)
class AgentCostConfig:
    """Root configuration object loaded from agentcost.toml."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def load_config(path: Path | None = None) -> AgentCostConfig:
    """Read agentcost.toml and return a typed config.

    If *path* is None, looks for ``agentcost.toml`` in the current directory.
    Returns default config when the file does not exist.
    """
    if path is None:
        path = Path("agentcost.toml")

    if not path.exists():
        logger.debug("Config file %s not found, using defaults", path)
        return AgentCostConfig()

    with path.open("rb") as f:
        raw = tomllib.load(f)

    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> AgentCostConfig:
    """Convert raw TOML dict to typed AgentCostConfig."""
    project_raw = raw.get("project", {})
    profile_raw = raw.get("profile", {})
    report_raw = raw.get("report", {})

    project = ProjectConfig(
        name=project_raw.get("name"),
    )

    volume = profile_raw.get("default_volume", list(_DEFAULT_VOLUMES))
    profile = ProfileConfig(
        default_volume=tuple(volume),
        baseline_path=profile_raw.get("baseline_path", _DEFAULT_BASELINE_PATH),
    )

    report = ReportConfig(
        output_dir=report_raw.get("output_dir", _DEFAULT_OUTPUT_DIR),
    )

    return AgentCostConfig(project=project, profile=profile, report=report)
