from __future__ import annotations

import click


def handle_request(user_input: str) -> str:
    return f"handled: {user_input}"


@click.command()
@click.argument("query")
def main(query: str) -> None:
    result = handle_request(query)
    click.echo(result)
