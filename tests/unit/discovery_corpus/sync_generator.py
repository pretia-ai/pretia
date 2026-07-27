from __future__ import annotations


def chat(msg: str):
    yield f"Thinking about: {msg}"
    yield f"Answer: {msg}"
