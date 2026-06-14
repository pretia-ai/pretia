#!/bin/bash
set -euo pipefail

# ── Inputs from GitHub Action ───────────────────────────────────────────
WORKFLOW_PATH="${INPUT_WORKFLOW_PATH:?workflow_path input is required}"
BASELINE_PATH="${INPUT_BASELINE_PATH:-.agentcost/baseline.json}"
MODE="${INPUT_MODE:-diff}"
FRAMEWORK="${INPUT_FRAMEWORK:-auto}"
DAILY_VOLUME="${INPUT_DAILY_VOLUME:-1000}"
POST_COMMENT="${INPUT_POST_COMMENT:-true}"
COST_THRESHOLD="${INPUT_COST_THRESHOLD:-}"

RESULT_FILE="/tmp/agentcost_result.json"
COMMENT_MARKER="<!-- agentcost-pr-comment -->"

# ── Run analysis ────────────────────────────────────────────────────────
THRESHOLD_ARG=""
if [ -n "$COST_THRESHOLD" ]; then
    THRESHOLD_ARG="--cost-threshold $COST_THRESHOLD"
fi

python -m agentcost.ci.github \
    --workflow-path "$WORKFLOW_PATH" \
    --baseline-path "$BASELINE_PATH" \
    --mode "$MODE" \
    --framework "$FRAMEWORK" \
    --daily-volume "$DAILY_VOLUME" \
    $THRESHOLD_ARG \
    --output-file "$RESULT_FILE" || ANALYSIS_EXIT=$?

ANALYSIS_EXIT=${ANALYSIS_EXIT:-0}

# ── Set outputs ─────────────────────────────────────────────────────────
if [ -f "$RESULT_FILE" ]; then
    SCORE=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('score', ''))")
    PROJECTED_COST=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('projected_cost', ''))")
    COST_DELTA=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('cost_delta', ''))")
    DELTA_PCT=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('delta_pct', ''))")
    REC_COUNT=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('rec_count', ''))")
    REPORT_PATH=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('report_path', ''))")

    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "score=$SCORE" >> "$GITHUB_OUTPUT"
        echo "projected_cost=$PROJECTED_COST" >> "$GITHUB_OUTPUT"
        echo "cost_delta=$COST_DELTA" >> "$GITHUB_OUTPUT"
        echo "delta_pct=$DELTA_PCT" >> "$GITHUB_OUTPUT"
        echo "rec_count=$REC_COUNT" >> "$GITHUB_OUTPUT"
        echo "report_path=$REPORT_PATH" >> "$GITHUB_OUTPUT"
    fi
fi

# ── Post PR comment ────────────────────────────────────────────────────
if [ "$POST_COMMENT" = "true" ] && [ -n "${GITHUB_TOKEN:-}" ] && [ -f "$RESULT_FILE" ]; then
    PR_NUMBER=""
    if [ -n "${GITHUB_EVENT_PATH:-}" ] && [ -f "${GITHUB_EVENT_PATH}" ]; then
        PR_NUMBER=$(python3 -c "
import json, sys
try:
    e = json.load(open('${GITHUB_EVENT_PATH}'))
    pr = e.get('pull_request', {})
    print(pr.get('number', ''))
except Exception:
    print('')
")
    fi

    if [ -n "$PR_NUMBER" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
        COMMENT_BODY=$(python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
sys.stdout.write(d.get('comment_markdown', ''))
")

        API_URL="https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments"

        # Search for an existing AgentCost comment to update
        EXISTING_ID=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$API_URL?per_page=100" | \
            python3 -c "
import json, sys
comments = json.load(sys.stdin)
for c in comments:
    if '$COMMENT_MARKER' in c.get('body', ''):
        print(c['id'])
        break
" 2>/dev/null || true)

        if [ -n "$EXISTING_ID" ]; then
            # Update existing comment
            curl -s -X PATCH \
                -H "Authorization: token $GITHUB_TOKEN" \
                -H "Accept: application/vnd.github.v3+json" \
                "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/comments/${EXISTING_ID}" \
                -d "$(python3 -c "import json; print(json.dumps({'body': open('$RESULT_FILE').read() and json.load(open('$RESULT_FILE'))['comment_markdown']}))")" \
                > /dev/null
            echo "Updated existing AgentCost PR comment."
        else
            # Post new comment
            curl -s -X POST \
                -H "Authorization: token $GITHUB_TOKEN" \
                -H "Accept: application/vnd.github.v3+json" \
                "$API_URL" \
                -d "$(python3 -c "import json; print(json.dumps({'body': json.load(open('$RESULT_FILE'))['comment_markdown']}))")" \
                > /dev/null
            echo "Posted AgentCost PR comment."
        fi
    fi
fi

exit "$ANALYSIS_EXIT"
