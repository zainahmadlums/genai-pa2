"""Batched deterministic generation on fixed evaluation prompts.

Output entries follow the assignment format::

    {"id": ..., "instruction": ..., "output": ..., "generator": ..., "dataset": ...}

The output file is rewritten (atomically) after every batch, and an existing file is
reused on restart, so a disconnected notebook continues where it stopped.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from alignment.batching import encode_text, get_eos_token_id, get_pad_token_id
from alignment.prompt_format import (
    DEFAULT_TEMPLATE,
    STOP_STRINGS,
    format_generation_prompt,
    trim_at_stop_strings,
)
from alignment.io_utils import log_line, read_json, write_json


@dataclass
class GenerationConfig:
    """Decoding settings. Defaults are plain greedy decoding."""

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    max_prompt_tokens: int = 1024
    batch_size: int = 8


def load_eval_prompts(path, limit: int | None = None) -> list[dict]:
    """Read an evaluation file: a JSON list of records with ``id`` and ``instruction``."""

    payload = read_json(path)
    rows = payload if isinstance(payload, list) else payload.get("prompts", payload.get("data"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list of prompt records")
    records = []
    for index, row in enumerate(rows):
        instruction = row.get("instruction") or row.get("prompt")
        if not instruction:
            raise ValueError(f"Record {index} in {path} has no instruction")
        record = dict(row)
        record["instruction"] = instruction
        record.setdefault("id", f"prompt-{index:04d}")
        records.append(record)
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate ids")
    return records[:limit] if limit else records


def _left_pad(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, ids in enumerate(sequences):
        input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, width - len(ids):] = 1
    return input_ids, attention_mask


def stop_token_ids(tokenizer, template: str | None) -> list[int]:
    """Ids of the template's turn markers that the tokenizer holds as single (special) tokens.

    For ``chatml`` with the Qwen tokenizer these are ``<|im_end|>`` and ``<|im_start|>``.
    They must be handled as token ids: ``decode(skip_special_tokens=True)`` removes them from
    the text, so a text-level search for the marker would never find them. Markers that are
    ordinary text for this tokenizer (the ``alpaca`` headers) are not returned; those are cut
    at text level by ``trim_at_stop_strings``.
    """

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if template is None or not callable(convert):
        return []
    unknown = getattr(tokenizer, "unk_token_id", None)
    ids = []
    for marker in STOP_STRINGS.get(template, []):
        try:
            token_id = convert(marker)
        except Exception:  # noqa: BLE001 - a tokenizer without this vocabulary entry
            continue
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 or token_id == unknown:
            continue
        if encode_text(tokenizer, marker) == [token_id] and token_id not in ids:
            ids.append(token_id)
    return ids


def generation_kwargs(tokenizer, config: GenerationConfig, template: str | None = None) -> dict:
    eos_token_id = get_eos_token_id(tokenizer)
    extra_stops = [token_id for token_id in stop_token_ids(tokenizer, template) if token_id != eos_token_id]
    kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": get_pad_token_id(tokenizer),
        # Generation also stops at a template turn marker that is a single token (chatml).
        "eos_token_id": [eos_token_id, *extra_stops] if extra_stops and eos_token_id is not None else eos_token_id,
        "num_beams": 1,
    }
    if config.do_sample:
        kwargs.update({"temperature": config.temperature, "top_p": config.top_p})
    else:
        # Explicit None silences sampling defaults stored in a model's generation_config.
        kwargs.update({"temperature": None, "top_p": None, "top_k": None})
    if config.repetition_penalty and config.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = config.repetition_penalty
    if config.no_repeat_ngram_size and config.no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
    return kwargs


@torch.no_grad()
def generate_batch(model, tokenizer, instructions: list[str], template: str, config: GenerationConfig,
                   device, autocast=None) -> list[dict]:
    """Generate one response per instruction. Returns dicts with ``output`` and token counts."""

    prompts = [format_generation_prompt(tokenizer, instruction, template) for instruction in instructions]
    encoded = [encode_text(tokenizer, prompt)[-config.max_prompt_tokens:] for prompt in prompts]
    input_ids, attention_mask = _left_pad(encoded, get_pad_token_id(tokenizer))
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    kwargs = generation_kwargs(tokenizer, config, template)
    if autocast is not None:
        with autocast():
            sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    else:
        sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    new_tokens = sequences[:, input_ids.shape[1]:]
    eos_token_id = get_eos_token_id(tokenizer)
    stops = set(stop_token_ids(tokenizer, template))
    if eos_token_id is not None:
        stops.add(eos_token_id)
    results = []
    for row in range(new_tokens.shape[0]):
        ids = new_tokens[row].tolist()
        # Cut at the first EOS or special turn-marker token BEFORE decoding: decoding drops special
        # tokens, after which an invented next turn could no longer be told apart from the answer.
        cut = next((position for position, token_id in enumerate(ids) if token_id in stops), None)
        hit_eos = cut is not None
        if hit_eos:
            ids = ids[:cut]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        results.append({
            "output": trim_at_stop_strings(text, template),
            "new_tokens": len(ids),
            "finished": bool(hit_eos),
        })
    return results


class GenerationCacheMismatch(ValueError):
    """An existing output file was produced with a different checkpoint, template or decoding setup."""


PROVENANCE_VERSION = 1
# batch_size is deliberately not part of the provenance: it changes speed, not the decoding rule.
_PROVENANCE_DECODING_KEYS = ("max_new_tokens", "do_sample", "temperature", "top_p", "repetition_penalty",
                             "no_repeat_ngram_size", "max_prompt_tokens")


def checkpoint_fingerprint(path) -> str | None:
    """Content fingerprint of a local checkpoint / adapter directory, ``None`` for anything else.

    Covers every file's relative name, size and the SHA-256 of its first and last megabyte
    (whole file when it is at most 2 MB), so retraining into the same path changes the
    fingerprint while copying the directory elsewhere does not. Hub model ids are not
    directories; they are identified by their name instead.
    """

    path = Path(path)
    if not path.is_dir():
        return None
    chunk = 1 << 20
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        size = file.stat().st_size
        digest.update(f"{file.relative_to(path).as_posix()}\0{size}\0".encode("utf-8"))
        with open(file, "rb") as handle:
            digest.update(handle.read(chunk))
            if size > 2 * chunk:
                handle.seek(size - chunk)
                digest.update(handle.read(chunk))
            elif size > chunk:
                digest.update(handle.read())
    return digest.hexdigest()[:16]


def generation_provenance(generator: str, dataset: str | None, template: str, config: GenerationConfig,
                          model_identity: dict | None = None) -> dict:
    """What a saved output file is valid for. Compared for equality before any reuse."""

    decoding = asdict(config)
    return {
        "provenance_version": PROVENANCE_VERSION,
        "generator": generator,
        "dataset": dataset,
        "template": template,
        "decoding": {key: decoding[key] for key in _PROVENANCE_DECODING_KEYS},
        "model_identity": dict(model_identity or {}),
    }


def describe_difference(saved: dict, current: dict, prefix: str = "") -> str:
    """Human-readable list of the provenance fields that differ (saved -> current)."""

    parts = []
    for key in sorted(set(saved) | set(current)):
        old, new = saved.get(key), current.get(key)
        if old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            parts.append(describe_difference(old, new, f"{prefix}{key}."))
        else:
            parts.append(f"{prefix}{key}: {old!r} -> {new!r}")
    return "; ".join(part for part in parts if part)


def generate_file(model, tokenizer, records: list[dict], out_path, generator: str, dataset: str | None,
                  template: str = DEFAULT_TEMPLATE, config: GenerationConfig | None = None, device="cpu",
                  autocast=None, overwrite: bool = False, extra_meta: dict | None = None, log=log_line,
                  model_identity: dict | None = None) -> list[dict]:
    """Generate for every record and write ``out_path`` (JSON list in the input order).

    Existing outputs in ``out_path`` are reused only when ``<out>.meta.json`` records the
    same provenance (generator label, dataset label, prompt template, decoding settings and
    ``model_identity``). Any difference, or a missing provenance record, raises
    ``GenerationCacheMismatch`` before anything is generated; replacing the file needs
    ``overwrite=True``. The provenance is written before the first batch, so an output file
    always has a meta file that describes it, also after a crash.
    """

    config = config or GenerationConfig()
    out_path = Path(out_path)
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    provenance = generation_provenance(generator, dataset, template, config, model_identity)
    done: dict[str, dict] = {}
    if out_path.exists() and not overwrite:
        existing = read_json(out_path)
        others = sorted({str(entry.get("generator")) for entry in existing} - {generator})
        if others:
            raise ValueError(
                f"{out_path} already holds outputs from {others}; use a different --output_path or --overwrite"
            )
        saved = read_json(meta_path).get("provenance") if meta_path.exists() else None
        if saved is None:
            raise GenerationCacheMismatch(
                f"{out_path} exists but {meta_path.name} has no provenance record, so it cannot be verified that "
                "these outputs came from this checkpoint, prompt template and decoding settings. "
                "Use a different --output_path, or --overwrite to regenerate."
            )
        if saved != provenance:
            raise GenerationCacheMismatch(
                f"{out_path} was generated with different settings: {describe_difference(saved, provenance)}. "
                "Existing outputs are not reused or relabelled. Use a different --output_path, or --overwrite "
                "to regenerate everything with the current settings."
            )
        wanted = {record["id"]: record["instruction"] for record in records}
        for entry in existing:
            if wanted.get(entry["id"]) == entry["instruction"]:
                done[entry["id"]] = entry
        if done:
            log(f"[generate] reusing {len(done)} existing outputs from {out_path} (same provenance)")
    elif out_path.exists():
        out_path.unlink()  # explicit overwrite: old outputs go before the new provenance is recorded
        log(f"[generate] --overwrite: discarded existing {out_path}")

    pending = [record for record in records if record["id"] not in done]
    # Longest prompts first keeps padding small; results are written in the original order.
    pending.sort(key=lambda record: len(record["instruction"]), reverse=True)
    stats = {"finished_with_eos": 0, "hit_max_new_tokens": 0, "empty_outputs": 0, "new_tokens": 0}
    started = time.time()
    reused = len(done)

    def flush() -> list[dict]:
        entries = [done[record["id"]] for record in records if record["id"] in done]
        write_json(out_path, entries)
        return entries

    def write_meta(status: str) -> None:
        write_json(meta_path, {
            "status": status,
            "generator": generator,
            "dataset": dataset,
            "template": template,
            "decoding": asdict(config),
            "stop_string_trimming": True,
            "provenance": provenance,
            "num_outputs": len(done),
            "reused_outputs": reused,
            "generated_this_run": len(done) - reused,
            "stats_this_run": stats,
            "seconds_this_run": round(time.time() - started, 1),
            **(extra_meta or {}),
        })

    write_meta("in_progress")  # before the first batch: the output file never exists without its provenance
    model.eval()
    for start in range(0, len(pending), config.batch_size):
        chunk = pending[start:start + config.batch_size]
        outputs = generate_batch(model, tokenizer, [record["instruction"] for record in chunk], template, config,
                                 device, autocast)
        for record, result in zip(chunk, outputs):
            entry = {
                "id": record["id"],
                "instruction": record["instruction"],
                "output": result["output"],
                "generator": generator,
                "dataset": dataset or record.get("dataset") or "unknown",
            }
            for key in ("category", "expected_behavior"):
                if key in record:
                    entry[key] = record[key]
            done[record["id"]] = entry
            stats["finished_with_eos"] += int(result["finished"])
            stats["hit_max_new_tokens"] += int(not result["finished"])
            stats["empty_outputs"] += int(not result["output"])
            stats["new_tokens"] += result["new_tokens"]
        flush()
        log(f"[generate] {generator}: {min(start + config.batch_size, len(pending))}/{len(pending)} new "
            f"({time.time() - started:.0f}s)")

    entries = flush()
    write_meta("complete")
    return entries
