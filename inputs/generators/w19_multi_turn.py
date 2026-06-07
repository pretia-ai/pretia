"""Input generator for W19: Multi-turn CloudOps conversation scripts.

Generate exactly 8-turn conversation scripts with a mix of substantive
CloudOps questions and filler turns.  Profiling averages 5 substantive
turns; ground truth averages 7 (session depth drift).
"""

from __future__ import annotations

import random
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

_TOTAL_TURNS = 8

# ---------------------------------------------------------------------------
# Token-length targets (total across all 8 turns, chars ≈ tokens * 4)
# ---------------------------------------------------------------------------
_TOKEN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "profiling": {
        "easy":   (100, 250),
        "medium": (200, 450),
        "hard":   (350, 700),
        "edge":   (80, 800),
    },
    "ground_truth": {
        "easy":     (150, 400),
        "medium":   (300, 700),
        "hard":     (500, 1_100),
        "edge":     (100, 1_200),
        "extreme":  (800, 1_800),
    },
}

# ---------------------------------------------------------------------------
# Filler messages — short acknowledgments
# ---------------------------------------------------------------------------
_FILLERS = [
    "ok",
    "thanks",
    "got it",
    "makes sense",
    "appreciate it",
    "sure",
    "alright",
    "understood",
    "cool",
    "right",
    "noted",
    "great",
    "perfect",
    "yep",
    "okay thanks",
]

# ---------------------------------------------------------------------------
# CloudOps topics with substantive message variants per turn position
#
# Each topic has:
#   - opener: turn 1 variants (3-5)
#   - follow_ups: turn 2-4 variants (3-5 each)
#   - deep_dives: turn 5-8 variants referencing earlier context (3-5)
#   - cross_refs: variants that cross-reference earlier turns (3-5)
# ---------------------------------------------------------------------------
_TOPICS: list[dict[str, Any]] = [
    {
        "name": "deployment_failure",
        "opener": [
            "Our latest deployment to production failed on the Deploy pipeline — can you walk me through the rollback procedure?",
            "We pushed a release through Deploy but the health checks are failing on 3 out of 8 pods. What should I do?",
            "Deploy is showing a failed status for service auth-gateway. The error mentions image pull failures.",
            "The Deploy pipeline succeeded but the service is throwing 502s. Dashboard shows the new pods aren't ready.",
            "I triggered a canary deploy through Deploy but the error rate spiked to 15%. Should I abort?",
        ],
        "follow_ups": [
            "What does the Deploy rollback actually do — does it redeploy the previous image or revert the config?",
            "I see two previous versions in Deploy history. How do I pick which one to roll back to?",
            "The rollback completed but Dashboard still shows elevated error rates. Is there a cache issue?",
            "Can you check if the environment variables were correctly propagated in the Deploy config?",
            "I need to verify the health check endpoint. Where in Deploy can I see the probe configuration?",
        ],
        "deep_dives": [
            "Going back to the image pull failure you mentioned — could this be related to our Docker registry auth token expiring?",
            "Based on the rollback steps you described, should I also invalidate the CDN cache to avoid serving stale assets?",
            "Can you correlate the deployment timeline with the Guard alerts to see if the security scan flagged anything?",
            "If the canary deployment error rate threshold is 5% but we hit 15%, why didn't the auto-rollback trigger?",
            "Show me how to set up a Deploy pre-flight check so we catch image pull issues before the actual deploy starts.",
        ],
        "cross_refs": [
            "You mentioned earlier that the rollback reverts config — does that include the Scale auto-scaling rules too?",
            "Going back to the 502 errors from turn 1 — could they be caused by the pod readiness probe timeout being too low?",
            "Earlier you said to check environment variables. I found a mismatch. Now how do I redeploy with corrected values?",
        ],
    },
    {
        "name": "scaling_alert",
        "opener": [
            "Scale is showing a critical alert — our API gateway is at 95% CPU utilization. What auto-scaling options do I have?",
            "We're getting throttled on the database connection pool. Scale shows the read replicas maxed out.",
            "Dashboard shows a traffic spike to 3x normal volume. Scale hasn't triggered auto-scaling yet.",
            "The Scale horizontal pod autoscaler keeps oscillating between 4 and 12 replicas. How do I stabilize it?",
            "We need to pre-scale for a product launch tomorrow. How do I configure Scale for anticipated 10x traffic?",
        ],
        "follow_ups": [
            "What's the difference between horizontal and vertical scaling in the Scale configuration?",
            "The auto-scaler is using CPU metrics but our bottleneck is memory. How do I switch the scaling metric?",
            "I set min replicas to 8 but Scale is only showing 6 active. Is there a resource quota blocking it?",
            "Can Scale handle scaling down gracefully during off-peak hours without dropping active connections?",
            "What cooldown period should I set between scale-up and scale-down events?",
        ],
        "deep_dives": [
            "Referring to the CPU alert from earlier — should I also set up Guard to alert on sustained high CPU before it becomes critical?",
            "Based on the replica oscillation issue, would a custom metric from Dashboard be more stable for auto-scaling decisions?",
            "You mentioned resource quotas — how do I check and increase the namespace quota in Scale?",
            "If we pre-scale to 20 replicas, what's the estimated cost impact shown in Dashboard?",
            "Can I configure Scale to use predictive scaling based on historical Dashboard traffic patterns?",
        ],
        "cross_refs": [
            "You mentioned switching to memory-based scaling. Does that work with the JVM-based services we discussed earlier?",
            "Going back to the connection pool issue — if I scale the read replicas, do I also need to update the connection pool config in Deploy?",
            "The oscillation fix you suggested — will that conflict with the pre-scaling configuration for the launch?",
        ],
    },
    {
        "name": "monitoring_gap",
        "opener": [
            "Dashboard isn't showing any metrics for the new payment-service we deployed last week. How do I set up monitoring?",
            "We had an outage last night but Guard didn't fire any alerts. I need to review the alerting configuration.",
            "The Dashboard latency graphs look flat at 0ms which is obviously wrong. Is the metrics pipeline broken?",
            "I need to create a custom Dashboard panel that shows request success rate broken down by customer tier.",
            "Guard is sending too many false-positive alerts for the background job service. How do I tune the thresholds?",
        ],
        "follow_ups": [
            "What metrics format does Dashboard expect — Prometheus, StatsD, or OpenTelemetry?",
            "How do I add custom labels to the metrics so I can filter by environment and region in Dashboard?",
            "The Guard alert rules are using a 1-minute evaluation window. Is that too aggressive for batch workloads?",
            "Can I set up Dashboard to show a comparison view between this week and last week?",
            "How do I create a Guard alert that only fires if the error rate stays above 5% for more than 10 minutes?",
        ],
        "deep_dives": [
            "Referring to the missing payment-service metrics — could the issue be that the service mesh sidecar isn't scraping the /metrics endpoint?",
            "You said to use OpenTelemetry. Does the collector need to be deployed as a sidecar or can I use the centralized collector through Deploy?",
            "Based on the false positive issue, would implementing anomaly-based alerting in Guard work better than static thresholds?",
            "Can I export the Dashboard panels as infrastructure-as-code so they're version controlled alongside the Deploy configs?",
            "If I set up distributed tracing through Dashboard, will it automatically correlate with the Guard alerts?",
        ],
        "cross_refs": [
            "Earlier you mentioned the metrics pipeline might be broken. Could that be related to the Deploy config issue we discussed?",
            "The custom labels you described — will Scale's auto-scaler pick up those labels for its decision logic?",
            "Going back to the false positives — if I tune Guard thresholds, will the on-call rotation in Dashboard still be accurate?",
        ],
    },
    {
        "name": "integration_setup",
        "opener": [
            "I need to integrate our CloudOps Dashboard with PagerDuty for on-call alerting. What's the setup process?",
            "We want to connect Guard's security scanning results to our JIRA board automatically.",
            "How do I set up a webhook from Deploy to our Slack channel so the team gets notified on deployments?",
            "I need to configure SSO for Dashboard using our company's Okta instance.",
            "We want to export all Guard audit logs to our SIEM (Splunk). What format and protocol does Guard use?",
        ],
        "follow_ups": [
            "Does the PagerDuty integration support escalation policies or just basic notifications?",
            "For the JIRA integration, can Guard automatically assign the ticket to the service owner?",
            "The Slack webhook is working but it's posting to the wrong channel. Where do I update the webhook URL?",
            "Can the SSO integration support role-based access control so dev and ops teams see different Dashboard views?",
            "What retention period does Guard enforce for audit logs before they're rotated?",
        ],
        "deep_dives": [
            "Referring to the PagerDuty setup — can I use the integration to trigger automated Scale responses, not just human alerts?",
            "You mentioned service owner assignment in JIRA. How does Guard determine the service owner — is it from the Deploy manifest?",
            "Based on the SSO discussion, if someone's Okta session expires mid-deploy, does Deploy handle the re-authentication gracefully?",
            "For the SIEM export, can I filter which Guard events get sent to Splunk to avoid log volume explosion?",
            "If the Slack webhook fails, does Deploy queue the notifications and retry, or are they lost?",
        ],
        "cross_refs": [
            "Earlier you said Guard pulls service owners from Deploy manifests. What happens if a service isn't in Deploy yet?",
            "The RBAC setup for Dashboard — does that affect what Scale configurations each team can modify?",
            "Going back to the Slack webhook — could the wrong channel issue be because we have a staging and prod Deploy pipeline with the same webhook?",
        ],
    },
    {
        "name": "billing_question",
        "opener": [
            "Our CloudOps bill jumped 40% this month. Dashboard shows normal traffic. Where is the cost spike coming from?",
            "I need to set up cost alerts in Dashboard so we get notified before exceeding our monthly budget.",
            "We're trying to optimize costs — Scale shows we have idle replicas during off-peak hours.",
            "Can I get a cost breakdown by service from Dashboard? I need to charge back to individual teams.",
            "The GPU instances for our ML pipeline are running 24/7 but we only use them for 4 hours daily. How do I optimize?",
        ],
        "follow_ups": [
            "Does Dashboard show cost attribution by namespace or do I need to set up additional tagging?",
            "For the cost alerts, can I set different thresholds for different environments (dev vs prod)?",
            "If Scale down-sizes replicas during off-peak, how long does it take to scale back up when traffic returns?",
            "Can I schedule the GPU instances to auto-start and stop through Scale?",
            "Is there a way to see historical cost trends in Dashboard to predict next month's bill?",
        ],
        "deep_dives": [
            "Referring to the 40% cost spike — could it be related to the Deploy pipeline running more builds than usual due to the CI changes?",
            "You mentioned cost attribution by namespace. Does that include the Scale auto-scaler overhead and Guard scanning costs?",
            "Based on the GPU scheduling discussion, would spot instances through Scale be more cost-effective than scheduled on-demand?",
            "If I set up cost alerts but the spending is driven by Guard security scans, can I exclude those from the alert threshold?",
            "The chargeback reports from Dashboard — can they be auto-emailed to team leads monthly?",
        ],
        "cross_refs": [
            "Earlier you said idle replicas are a cost driver. Is this related to the Scale oscillation issue we discussed?",
            "You mentioned spot instances for GPU. If a spot instance gets preempted, how does Deploy handle the workload migration?",
            "Going back to the cost breakdown — do the Guard compliance scan costs show up under the service namespace or under a shared guard namespace?",
        ],
    },
    {
        "name": "incident_response",
        "opener": [
            "We're in an active incident — Dashboard is showing 100% error rate on the checkout service. Walk me through the runbook.",
            "Guard just flagged a potential security breach — unusual API access patterns from an internal service account.",
            "Our primary database is unreachable. Scale shows the failover hasn't triggered. What do I do?",
            "Dashboard shows a cascading failure spreading from the auth service to downstream services. How do I contain it?",
            "We got a customer report of data corruption. Guard audit logs show unusual write patterns.",
        ],
        "follow_ups": [
            "I've isolated the checkout service through Deploy. Dashboard still shows errors — are there dependent services I'm missing?",
            "For the security breach, should I revoke the service account token immediately or monitor it first to understand the scope?",
            "The database failover requires manual DNS update. Can Deploy automate this?",
            "I've enabled circuit breakers on the auth service. Dashboard shows the error rate dropping but latency is increasing.",
            "Guard is flagging more suspicious write patterns. How do I export the audit trail for the forensics team?",
        ],
        "deep_dives": [
            "Referring to the cascading failure — should I use Scale to increase replicas of healthy services to absorb redirected traffic?",
            "You said to isolate the service via Deploy. Does that also disconnect it from the Guard scanning pipeline?",
            "Based on the security breach scope — can Guard automatically correlate the suspicious patterns with Deploy changes from the same time window?",
            "If the circuit breaker latency increase continues, at what point should I use Scale to add more auth service replicas?",
            "For the data corruption forensics, can I use Dashboard's historical metrics to pinpoint the exact time the corruption started?",
        ],
        "cross_refs": [
            "You mentioned increasing replicas for healthy services. Does that conflict with the cost budget alerts we set up earlier?",
            "Going back to the service account token — if I revoke it, will the Deploy pipeline break since it uses the same account?",
            "Earlier you suggested exporting Guard audit trails. Can I correlate those with the Dashboard latency data from the same timeframe?",
        ],
    },
]

# ---------------------------------------------------------------------------
# Unicode dirty artifacts for W19
# ---------------------------------------------------------------------------
_UNICODE_ARTIFACTS = [
    "​",       # zero-width space
    "‌",       # zero-width non-joiner
    "‍",       # zero-width joiner
    "﻿",       # BOM
    " ",       # non-breaking space
    " ",       # line separator
    " ",       # paragraph separator
    "‪",       # left-to-right embedding
    "‬",       # pop directional formatting
]


def _inject_unicode(text: str, rng: random.Random) -> str:
    """Insert random Unicode artifacts into a message."""
    n_artifacts = rng.randint(2, 6)
    result = list(text)
    for _ in range(n_artifacts):
        pos = rng.randint(0, len(result))
        artifact = rng.choice(_UNICODE_ARTIFACTS)
        result.insert(pos, artifact)
    return "".join(result)


def _make_near_limit(rng: random.Random) -> str:
    """Generate a very long message approaching token limits."""
    topics = [
        "I need a comprehensive analysis of our entire CloudOps infrastructure ",
        "including every service deployed through Deploy, all Scale configurations, ",
        "complete Dashboard metric history for the past 90 days, Guard security scan ",
        "results with remediation timelines, cross-referenced with deployment frequency ",
        "and incident response times. Also include cost trends broken down by team, ",
        "service, environment, and time period. Compare our current setup against ",
        "industry best practices for each component. ",
    ]
    text = ""
    while len(text) < 8000:
        text += rng.choice(topics)
    return text[:rng.randint(8000, 12000)]


class MultiTurnGenerator(BaseInputGenerator):
    """Generate 8-turn CloudOps conversation scripts for W19."""

    workflow_id = "W19"
    dirty_types = ["mixed_unicode", "near_limit_inputs"]

    _token_ranges_lookup = _TOKEN_RANGES

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.dry_run = True

    def get_rewritable_text(self, inp: GeneratedInput) -> str | None:
        return None

    async def generate_batch_async(self, profile: str, n: int) -> list[GeneratedInput]:
        """Generate inputs with concurrent per-turn LLM rewriting."""
        import asyncio

        inputs = self.generate_batch(profile, n)
        if getattr(self, "dry_run", True):
            return inputs

        async def _rewrite_turns(idx: int, inp: GeneratedInput) -> GeneratedInput:
            turns = inp.input_data.get("turns", [])
            new_turns: list[str] = []
            topic = inp.structural_descriptor.get("topic", "deployment")
            for t_idx, turn in enumerate(turns):
                if len(turn) < 15:  # filler turn, skip
                    new_turns.append(turn)
                    continue
                rng = random.Random(self.seed + idx * 100 + t_idx)
                rewritten = await self.llm_rewrite_async(
                    turn,
                    f"Generate turn {t_idx + 1} of an 8-turn CloudOps support "
                    f"conversation about {topic}.",
                    rng.randint(20, 100),
                    rng,
                )
                new_turns.append(rewritten)
            new_data = dict(inp.input_data)
            new_data["turns"] = new_turns
            new_data["input"] = new_turns[0] if new_turns else ""
            if "conversation_script" in new_data:
                new_data["conversation_script"] = [
                    {"turn": i + 1, "user_message": m}
                    for i, m in enumerate(new_turns)
                ]
            return GeneratedInput(
                id=inp.id,
                workflow=inp.workflow,
                profile=inp.profile,
                tier=inp.tier,
                token_count=sum(len(t) // 4 for t in new_turns),
                is_dirty=inp.is_dirty,
                dirty_type=inp.dirty_type,
                structural_descriptor=inp.structural_descriptor,
                input_data=new_data,
            )

        tasks = [_rewrite_turns(idx, inp) for idx, inp in enumerate(inputs)]
        return list(await asyncio.gather(*tasks))

    def _pick_topic(self, rng: random.Random) -> dict[str, Any]:
        return rng.choice(_TOPICS)

    def _build_conversation(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        topic: dict[str, Any],
    ) -> tuple[list[str], int, int, bool]:
        """Build an 8-turn conversation and return (turns, substantive_count, topic_shifts, has_contradiction).

        Session depth drift:
        - Profiling avg 5 substantive turns: randomly place 3 filler in positions 3-7.
        - GT avg 7 substantive turns: randomly place 1 filler in positions 5-7.

        Tier controls the *nature* of substantive turns, not the count:
        - Easy: simple questions, no cross-references
        - Medium: deeper follow-ups
        - Hard: cross-references to earlier turns
        - Edge: contradictions, topic shifts
        """
        turns: list[str] = [""] * _TOTAL_TURNS
        topic_shifts = 0
        has_contradiction = tier == "edge"

        # Start with all 8 positions substantive, then inject filler per profile
        substantive_positions = list(range(8))
        filler_positions: list[int] = []

        if profile == "profiling":
            # Place 3 filler turns in positions 3-7 (avg 5 substantive)
            eligible = list(range(3, 8))
            filler_inject = rng.sample(eligible, 3)
            for fi in filler_inject:
                substantive_positions.remove(fi)
                filler_positions.append(fi)
        else:
            # GT: place 1 filler turn in positions 5-7 (avg 7 substantive)
            eligible = list(range(5, 8))
            filler_inject = rng.sample(eligible, 1)
            for fi in filler_inject:
                substantive_positions.remove(fi)
                filler_positions.append(fi)

        # Fill substantive turns
        # Turn 0 always gets an opener
        if 0 in substantive_positions:
            turns[0] = rng.choice(topic["opener"])

        # Remaining substantive turns: tier controls content complexity
        for pos in substantive_positions:
            if pos == 0:
                continue
            if tier == "easy":
                # Easy: only basic follow-ups, no deep dives
                turns[pos] = rng.choice(topic["follow_ups"])
            elif tier == "medium":
                if pos <= 3:
                    turns[pos] = rng.choice(topic["follow_ups"])
                else:
                    pool = topic["deep_dives"] + topic["follow_ups"]
                    turns[pos] = rng.choice(pool)
            elif tier in ("hard", "extreme"):
                if pos <= 2:
                    turns[pos] = rng.choice(topic["follow_ups"])
                elif pos <= 4:
                    turns[pos] = rng.choice(topic["deep_dives"])
                else:
                    # Late turns cross-reference earlier content
                    turns[pos] = rng.choice(topic["cross_refs"])
            else:
                # edge or unknown
                if pos <= 2:
                    turns[pos] = rng.choice(topic["follow_ups"])
                elif pos <= 4:
                    pool = topic["deep_dives"] + topic["follow_ups"]
                    turns[pos] = rng.choice(pool)
                else:
                    pool = topic["deep_dives"] + topic["cross_refs"]
                    turns[pos] = rng.choice(pool)

        # Edge: inject contradictions and language switching
        if tier == "edge" and has_contradiction:
            # Add a contradiction somewhere in turns 3-6
            contra_pos = rng.randint(3, min(6, _TOTAL_TURNS - 1))
            if contra_pos in substantive_positions:
                turns[contra_pos] = (
                    "Actually, forget what I said earlier about the "
                    + topic["name"].replace("_", " ")
                    + ". I was wrong about the requirements. Let me start over with a different approach."
                )
            # Add a topic shift
            other_topic = rng.choice([t for t in _TOPICS if t["name"] != topic["name"]])
            shift_pos = rng.randint(5, 7)
            turns[shift_pos] = rng.choice(other_topic["opener"])
            topic_shifts = 1

        # Fill filler turns
        for pos in filler_positions:
            turns[pos] = rng.choice(_FILLERS)

        # Ensure no empty turns
        for i in range(_TOTAL_TURNS):
            if not turns[i]:
                turns[i] = rng.choice(_FILLERS)

        substantive_count = len(substantive_positions)
        return turns, substantive_count, topic_shifts, has_contradiction

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one 8-turn conversation script."""
        topic = self._pick_topic(rng)
        turns, substantive_count, topic_shifts, has_contradiction = self._build_conversation(
            tier, profile, rng, topic,
        )

        # Apply dirty input
        if is_dirty:
            if dirty_type == "mixed_unicode":
                # Inject Unicode artifacts into 2-3 random turns
                n_dirty_turns = rng.randint(2, 3)
                dirty_positions = rng.sample(range(_TOTAL_TURNS), n_dirty_turns)
                for pos in dirty_positions:
                    turns[pos] = _inject_unicode(turns[pos], rng)
            elif dirty_type == "near_limit_inputs":
                # Make one turn extremely long
                long_pos = rng.randint(1, _TOTAL_TURNS - 1)
                turns[long_pos] = _make_near_limit(rng)

        # Pad/truncate total turn content to fit the target token range
        ranges = _TOKEN_RANGES.get(profile, _TOKEN_RANGES["profiling"])
        if tier in ranges:
            tmin, tmax = ranges[tier]
            total_text = " ".join(turns)
            total_tokens = self.estimate_tokens(total_text)
            target = rng.randint(tmin, tmax)

            # Identify substantive (non-filler) turn indices for padding
            substantive_indices = [i for i in range(_TOTAL_TURNS) if turns[i] not in _FILLERS]

            if total_tokens < target and substantive_indices:
                # Distribute padding across substantive turns
                deficit = target - total_tokens
                per_turn_extra = max(1, deficit // len(substantive_indices))
                for si in substantive_indices:
                    current = self.estimate_tokens(turns[si])
                    turns[si] = self.pad_to_token_range(
                        turns[si], current, current + per_turn_extra, rng,
                    )
                    # Recheck total
                    total_tokens = sum(self.estimate_tokens(t) for t in turns)
                    if total_tokens >= target:
                        break

            if total_tokens > tmax and substantive_indices:
                # Truncate longest substantive turns
                while total_tokens > tmax:
                    longest_idx = max(substantive_indices, key=lambda i: len(turns[i]))
                    excess_chars = (total_tokens - tmax) * 4
                    turns[longest_idx] = turns[longest_idx][: max(20, len(turns[longest_idx]) - excess_chars)]
                    total_tokens = sum(self.estimate_tokens(t) for t in turns)
                    if len(turns[longest_idx]) <= 20:
                        break

        # Apply style shift to substantive GT turns
        if profile == "ground_truth":
            for i in range(_TOTAL_TURNS):
                if turns[i] not in _FILLERS:
                    turns[i] = self.apply_style_shift(turns[i], profile)

        # Calculate token stats
        user_msg_tokens = [self.estimate_tokens(t) for t in turns]
        avg_user_tokens = sum(user_msg_tokens) / len(user_msg_tokens) if user_msg_tokens else 0

        # Estimate turn-8 context tokens (all turns + simulated assistant responses)
        total_user_tokens = sum(user_msg_tokens)
        # Rough estimate: assistant responses are ~2x user message length on average
        estimated_context = int(total_user_tokens * 3)

        # Build conversation script
        conversation_script = [
            {"turn": i + 1, "user_message": msg}
            for i, msg in enumerate(turns)
        ]

        structural_descriptor: dict[str, Any] = {
            "substantive_turn_count": substantive_count,
            "topic_shifts": topic_shifts,
            "includes_contradiction": has_contradiction,
            "avg_user_message_tokens": round(avg_user_tokens),
            "estimated_turn_8_context_tokens": estimated_context,
        }

        token_count = total_user_tokens

        input_data: dict[str, Any] = {
            "turns": turns,
            "conversation_script": conversation_script,
            "input": turns[0],
        }

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W19",
            profile=profile,
            tier=tier,
            token_count=token_count,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=structural_descriptor,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(MultiTurnGenerator)
