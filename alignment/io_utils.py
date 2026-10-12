"""JSON / JSONL helpers with durable writes. Kept free of torch so judging needs no GPU stack."""

from __future__ import annotations

import json
import os
from pathlib import Path

CHECKPOINT_META = "alignment_meta.json"
COMPLETE_MARKER = "TRAINING_COMPLETE.json"


def log_line(*args) -> None:
    """``print`` that flushes, so progress shows up live when a script runs behind a pipe (``!python ...``)."""

    print(*args, flush=True)


def read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    """Atomic JSONL write (temp file + rename)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # newline="\n" keeps files (and their recorded hashes) identical on Windows and Linux.
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path, row: dict) -> None:
    """Append one record and force it to disk so partial progress survives a disconnect."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload, indent: int | None = 2) -> None:
    """Atomic JSON write (temp file + rename)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
