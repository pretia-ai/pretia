from __future__ import annotations


class SecureAgent:
    def __init__(self, api_key):
        self._key = api_key

    def run(self, msg):
        return f"secure: {msg}"


agent = SecureAgent("sk-test-key")
