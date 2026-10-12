"""Flat YAML configs with ``key=value`` command-line overrides."""

from __future__ import annotations

from pathlib import Path

import yaml


def parse_override(text: str) -> tuple[str, object]:
    if "=" not in text:
        raise ValueError(f"Override {text!r} must look like key=value")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def load_config(path, overrides: list[str] | None = None) -> dict:
    """Read a flat YAML mapping and apply overrides (values parsed as YAML scalars)."""

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    for item in overrides or []:
        key, value = parse_override(item)
        config[key] = value
    # YAML reads 2e-5 (no dot) as a string; coerce numeric-looking strings.
    for key, value in list(config.items()):
        if isinstance(value, str):
            try:
                config[key] = float(value) if any(ch in value.lower() for ch in ".e") else int(value)
            except ValueError:
                pass
    return config


def pick(config: dict, dataclass_type):
    """Build ``dataclass_type`` from the matching keys of ``config``."""

    names = set(dataclass_type.__dataclass_fields__)
    return dataclass_type(**{key: value for key, value in config.items() if key in names})
