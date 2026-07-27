from __future__ import annotations


class Bot:
    def chat(self, msg):
        return msg


Bot.__module__ = "mylib.bots"  # type: ignore[attr-defined]

helpdesk = Bot()
