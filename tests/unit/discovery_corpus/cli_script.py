from __future__ import annotations


def check_status(system_name):
    return f"Status: {system_name}"


def lookup_policy(topic):
    return f"Policy: {topic}"


TOOL_DISPATCH = {
    "check_status": check_status,
    "lookup_policy": lookup_policy,
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_status",
            "parameters": {"type": "object"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "parameters": {"type": "object"},
        },
    },
]

SYSTEM_PROMPT = (
    "You are an IT helpdesk assistant. "
    "Your role is to help employees resolve technical issues "
    "and look up policies."
)


def run_agent(client, messages):
    return messages


def main():
    run_agent(None, [])
