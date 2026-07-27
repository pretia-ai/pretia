from __future__ import annotations


class FakeASGI:
    def __call__(self, scope, receive, send):
        pass


app = FakeASGI()


def process_request(inp):
    return f"processed: {inp}"
