"""Command-line interface for AgentCost."""

from __future__ import annotations

import click


@click.group()
@click.version_option()
def cli() -> None:
    """Pre-deployment cost intelligence for AI agent workflows."""


if __name__ == "__main__":
    cli()
