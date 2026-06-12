"""Track cumulative API spend during backtest execution.

Implements budget checkpoints that can halt execution when spend
exceeds limits or when early failures indicate systemic problems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Workflows considered cheap/linear for the Comparison B gate.
_CHEAP_WORKFLOWS = {
    "W1-support-simple",
    "W9-sales-openai",
    "W11-support-qwen",
    "W12-extraction-deepseek",
}


@dataclass
class BudgetTracker:
    """Accumulate API costs across workflows and comparisons.

    Provides checkpoint gates that detect systemic failures early
    and prevent burning budget on a broken projection engine.
    """

    limit: float
    spent: float = 0.0
    per_workflow: dict[str, float] = field(default_factory=dict)
    per_comparison: dict[str, float] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, workflow: str, comparison: str, cost: float) -> None:
        """Add a cost entry and update all accumulators."""
        self.spent += cost
        self.per_workflow[workflow] = self.per_workflow.get(workflow, 0.0) + cost
        self.per_comparison[comparison] = self.per_comparison.get(comparison, 0.0) + cost
        entry = {
            "workflow": workflow,
            "comparison": comparison,
            "cost": cost,
            "cumulative": self.spent,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.log.append(entry)
        logger.debug(
            "Recorded $%.4f for %s/%s (cumulative $%.4f / $%.2f limit)",
            cost,
            workflow,
            comparison,
            self.spent,
            self.limit,
        )

    def check_limit(self) -> bool:
        """Return True if cumulative spend has reached or exceeded the budget limit."""
        over = self.spent >= self.limit
        if over:
            logger.warning(
                "Budget limit reached: $%.4f spent >= $%.2f limit",
                self.spent,
                self.limit,
            )
        return over

    def check_comparison_a_gate(self, scores: dict[str, Any]) -> tuple[bool, str]:
        """Gate after Comparison A completes for all workflows.

        Count how many workflows failed. If 5 or more failed, the
        projection engine has a systemic problem and further spend
        is unlikely to help.

        Returns (should_stop, message).
        """
        failed = [
            name
            for name, score in scores.items()
            if not (score if isinstance(score, bool) else score.get("passed", True))
        ]
        if len(failed) >= 5:
            msg = (
                f"Comparison A gate: {len(failed)} workflows failed "
                f"({', '.join(sorted(failed))}). "
                f"Systemic engine problem detected — halting execution."
            )
            logger.warning(msg)
            return True, msg

        msg = (
            f"Comparison A gate: {len(failed)} failure(s) out of "
            f"{len(scores)} workflows — within tolerance, continuing."
        )
        logger.info(msg)
        return False, msg

    def check_comparison_b_cheap_gate(self, scores: dict[str, Any]) -> tuple[bool, str]:
        """Gate after Comparison B completes for cheap/linear workflows.

        Check if any linear workflow (W1, W9, W11, W12) failed.
        These are simple pipelines that should pass easily; failure
        indicates a fundamental problem with the projection engine.

        Returns (should_stop, message).
        """
        cheap_scores = {name: score for name, score in scores.items() if name in _CHEAP_WORKFLOWS}
        failed = [
            name
            for name, score in cheap_scores.items()
            if not (score if isinstance(score, bool) else score.get("passed", True))
        ]
        if failed:
            msg = (
                f"Comparison B cheap gate: linear workflow(s) failed "
                f"({', '.join(sorted(failed))}). "
                f"These should pass easily — halting execution."
            )
            logger.warning(msg)
            return True, msg

        msg = (
            f"Comparison B cheap gate: all {len(cheap_scores)} cheap "
            f"workflows passed — continuing."
        )
        logger.info(msg)
        return False, msg

    def summary(self) -> dict[str, Any]:
        """Return a concise summary of budget state."""
        return {
            "total_spent": self.spent,
            "per_workflow": dict(self.per_workflow),
            "per_comparison": dict(self.per_comparison),
            "limit": self.limit,
            "remaining": max(0.0, self.limit - self.spent),
            "log_entries": len(self.log),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a fully serializable snapshot of the tracker state."""
        return {
            "limit": self.limit,
            "spent": self.spent,
            "per_workflow": dict(self.per_workflow),
            "per_comparison": dict(self.per_comparison),
            "log": list(self.log),
        }
