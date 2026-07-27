from __future__ import annotations


def load_config(path):
    return {}


load_config.__module__ = "somelib.config"  # type: ignore[attr-defined]


def main():
    load_config("config.yaml")
