"""Shared dataclasses for the unified validation framework."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CheckStatus(StrEnum):
    PASS = "PASS"  # noqa: S105
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    details: dict[str, Any]
    blocking: bool
    check_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
            "details": self.details,
            "blocking": self.blocking,
        }
        if self.check_id is not None:
            d["check_id"] = self.check_id
        return d


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: int
    name: str
    checks: tuple[CheckResult, ...]
    duration_s: float

    @property
    def passed(self) -> bool:
        return not any(
            c.status == CheckStatus.FAIL and c.blocking for c in self.checks
        )

    @property
    def blocking_failures(self) -> list[str]:
        return [
            c.name for c in self.checks
            if c.status == CheckStatus.FAIL and c.blocking
        ]

    @property
    def warnings(self) -> list[str]:
        return [c.name for c in self.checks if c.status == CheckStatus.WARN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "name": self.name,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "blocking_failures": self.blocking_failures,
            "warnings": self.warnings,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    timestamp: str
    stages: tuple[StageResult, ...]
    total_duration_s: float
    api_cost_usd: float = 0.0

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.stages)

    @property
    def max_passed_stage(self) -> int:
        for s in reversed(self.stages):
            if s.passed:
                return s.stage
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "max_passed_stage": self.max_passed_stage,
            "stages": [s.to_dict() for s in self.stages],
            "total_duration_s": round(self.total_duration_s, 2),
            "api_cost_usd": round(self.api_cost_usd, 6),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
