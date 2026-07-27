from __future__ import annotations


class Pipeline:
    def __call__(self, inp):
        return f"pipeline: {inp}"


app = Pipeline()
