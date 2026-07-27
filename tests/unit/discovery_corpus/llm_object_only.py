from __future__ import annotations


class ChatAnthropic:
    def __init__(self):
        self.model = "claude-3"

    def ainvoke(self, x):
        return x


llm = ChatAnthropic()
