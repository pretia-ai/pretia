from __future__ import annotations


class FakeGraph:
    def ainvoke(self, x):
        return x


graph = FakeGraph()
