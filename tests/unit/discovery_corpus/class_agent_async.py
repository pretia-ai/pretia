from __future__ import annotations


class ITHelpdeskAgent:
    SYSTEM_PROMPT = (
        "You are a helpful IT helpdesk assistant for Acme Corp. "
        "Your role is to diagnose and resolve technical issues."
    )
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "diagnose_issue",
                "parameters": {"type": "object"},
            },
        }
    ]

    def __init__(self):
        self._messages = []
        self._dispatch = {"diagnose_issue": self._diagnose}

    @staticmethod
    def _diagnose(symptoms):
        return f"Diagnosed: {symptoms}"

    async def run(self, msg):
        return f"async: {msg}"
