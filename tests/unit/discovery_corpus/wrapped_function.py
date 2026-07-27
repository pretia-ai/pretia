from __future__ import annotations

import functools


def _multi_arg_impl(client, messages, model):
    return f"result from {model}"


@functools.wraps(_multi_arg_impl)
def workflow(user_input):
    return _multi_arg_impl(None, [user_input], "gpt-4")
