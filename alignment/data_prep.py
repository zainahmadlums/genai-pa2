"""Deterministic conversion of public datasets into single-turn SFT / preference JSONL files.

Sources (public Hugging Face datasets, pinned to a commit so reruns give identical files):

* SFT: ``HuggingFaceH4/ultrachat_200k`` split ``train_sft`` (MIT). First user/assistant
  exchange of each conversation.
* HH DPO: ``Anthropic/hh-rlhf`` train files of ``harmless-base``, ``helpful-base``,
  ``helpful-online`` and ``helpful-rejection-sampled`` (MIT). Only single-turn
  transcripts (one ``Human`` turn, one ``Assistant`` turn).
* UltraFeedback DPO: ``argilla/ultrafeedback-binarized-preferences-cleaned`` split
  ``train`` (MIT). Single-turn pairs with ``chosen-rating > rejected-rating``.
* Evaluation: ``tatsu-lab/alpaca_eval`` file ``alpaca_eval.json`` (CC BY-NC 4.0).

Selection is the same for every track: shuffle candidates with the seed, drop duplicate
instructions, drop anything whose instruction appears in a fixed evaluation set, apply
token-length filters, then take the first ``n_val`` rows as validation and the next
``n_train`` rows as training. Train and validation are therefore disjoint by instruction.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from alignment.prompt_format import DEFAULT_TEMPLATE, format_prompt

ULTRACHAT = {
    "repo": "HuggingFaceH4/ultrachat_200k",
    "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "split": "train_sft",
    "license": "mit",
}
HH_RLHF = {
    "repo": "Anthropic/hh-rlhf",
    "revision": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
    "subsets": ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"],
    "file": "train.jsonl.gz",
    "license": "mit",
}
ULTRAFEEDBACK = {
    "repo": "argilla/ultrafeedback-binarized-preferences-cleaned",
    "revision": "770076f077c4c5e298498fa32f804857f46d5134",
    "split": "train",
    "license": "mit",
}
ALPACA_EVAL = {
    "repo": "tatsu-lab/alpaca_eval",
    "revision": "2edc6fad8be6b14ea7230aabfd08188da6b8b814",
    "file": "alpaca_eval.json",
    "license": "cc-by-nc-4.0",
}

_WHITESPACE = re.compile(r"\s+")
_HH_TURN = re.compile(r"\n\n(Human|Assistant):")


def normalize_text(text: str) -> str:
    """Case- and whitespace-insensitive key used for de-duplication and eval exclusion."""

    return _WHITESPACE.sub(" ", text).strip().lower()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_instruction_keys(paths) -> set[str]:
    """Normalised instructions of every fixed evaluation prompt (never used for training)."""

    keys: set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("prompts", payload.get("data", []))
        for row in rows:
            text = row.get("instruction") or row.get("prompt")
            if text:
                keys.add(normalize_text(text))
    return keys


# ---------------------------------------------------------------------------
# Row converters (pure functions; unit tested offline)
# ---------------------------------------------------------------------------


def ultrachat_row_to_example(row: dict) -> dict | None:
    """First user turn -> instruction, first assistant turn -> response."""

    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    first, second = messages[0], messages[1]
    if first.get("role") != "user" or second.get("role") != "assistant":
        return None
    instruction, response = (first.get("content") or "").strip(), (second.get("content") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


def parse_hh_transcript(text: str) -> list[tuple[str, str]]:
    """Split an HH-RLHF transcript into ``(role, text)`` turns."""

    parts = _HH_TURN.split(text)
    turns = []
    # parts = [prefix, role, text, role, text, ...]
    for index in range(1, len(parts) - 1, 2):
        turns.append((parts[index], parts[index + 1].strip()))
    return turns


def hh_row_to_pair(row: dict) -> dict | None:
    """Keep only single-turn rows: one Human turn and one Assistant turn on both sides."""

    chosen_turns = parse_hh_transcript(row.get("chosen") or "")
    rejected_turns = parse_hh_transcript(row.get("rejected") or "")
    if [role for role, _ in chosen_turns] != ["Human", "Assistant"]:
        return None
    if [role for role, _ in rejected_turns] != ["Human", "Assistant"]:
        return None
    instruction = chosen_turns[0][1]
    if not instruction or instruction != rejected_turns[0][1]:
        return None
    chosen, rejected = chosen_turns[1][1], rejected_turns[1][1]
    if not chosen or not rejected or chosen == rejected:
        return None
    return {"instruction": instruction, "chosen": chosen, "rejected": rejected}


def ultrafeedback_row_to_pair(row: dict) -> dict | None:
    """Single-turn UltraFeedback pair with a strictly higher rating for ``chosen``."""

    chosen_messages, rejected_messages = row.get("chosen") or [], row.get("rejected") or []
    if len(chosen_messages) != 2 or len(rejected_messages) != 2:
        return None
    if chosen_messages[-1].get("role") != "assistant" or rejected_messages[-1].get("role") != "assistant":
        return None
    instruction = (row.get("prompt") or chosen_messages[0].get("content") or "").strip()
    chosen = (chosen_messages[-1].get("content") or "").strip()
    rejected = (rejected_messages[-1].get("content") or "").strip()
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    chosen_rating, rejected_rating = row.get("chosen-rating"), row.get("rejected-rating")
    if chosen_rating is not None and rejected_rating is not None and not chosen_rating > rejected_rating:
        return None
    return {"instruction": instruction, "chosen": chosen, "rejected": rejected}


# ---------------------------------------------------------------------------
# Filtering and splitting
# ---------------------------------------------------------------------------


class LengthFilter:
    """Token-length rules applied with the training tokenizer and prompt template.

    * the formatted prompt must have at most ``max_prompt_tokens`` tokens;
    * with ``require_full_fit`` every response must fit in ``max_length`` together with
      the prompt and one EOS token, so no training target is truncated and EOS is always
      supervised;
    * responses shorter than ``min_response_tokens`` are dropped.
    """

    def __init__(self, tokenizer, template: str = DEFAULT_TEMPLATE, max_length: int = 512,
                 max_prompt_tokens: int = 256, require_full_fit: bool = True, min_response_tokens: int = 2):
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.max_prompt_tokens = max_prompt_tokens
        self.require_full_fit = require_full_fit
        self.min_response_tokens = min_response_tokens

    def describe(self) -> dict:
        return {"template": self.template, "max_length": self.max_length, "max_prompt_tokens": self.max_prompt_tokens,
                "require_full_fit": self.require_full_fit, "min_response_tokens": self.min_response_tokens}

    def reason_to_drop(self, example: dict) -> str | None:
        from alignment.batching import encode_text  # same tokenisation path as training

        prompt_tokens = len(encode_text(self.tokenizer, format_prompt(example["instruction"], self.template)))
        if prompt_tokens > self.max_prompt_tokens:
            return "prompt_too_long"
        for field in ("response", "chosen", "rejected"):
            if field not in example:
                continue
            response_tokens = len(encode_text(self.tokenizer, example[field]))
            if response_tokens < self.min_response_tokens:
                return "response_too_short"
            if self.require_full_fit and prompt_tokens + response_tokens + 1 > self.max_length:
                return "response_does_not_fit"
        return None


def select_split(candidates, n_train: int, n_val: int, seed: int, exclude_keys: set[str] | None = None,
                 length_filter: LengthFilter | None = None) -> tuple[list[dict], list[dict], dict]:
    """Shuffle, de-duplicate, filter and split ``candidates`` (a list of example dicts).

    Returns ``(train, val, stats)``. Validation rows are taken first, then training rows.
    Raises ``ValueError`` when fewer than ``n_train + n_val`` rows survive.
    """

    exclude_keys = exclude_keys or set()
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    needed = n_train + n_val
    kept, seen, dropped = [], set(), Counter()
    for index in order:
        if len(kept) >= needed:
            break
        example = candidates[index]
        key = normalize_text(example["instruction"])
        if key in seen:
            dropped["duplicate_instruction"] += 1
            continue
        if key in exclude_keys:
            dropped["in_fixed_eval_set"] += 1
            continue
        if length_filter is not None:
            reason = length_filter.reason_to_drop(example)
            if reason is not None:
                dropped[reason] += 1
                continue
        seen.add(key)
        kept.append(example)
    stats = {"candidates": len(candidates), "kept": len(kept), "requested_train": n_train, "requested_val": n_val,
             "dropped_before_quota": dict(dropped), "seed": seed}
    if len(kept) < needed:
        raise ValueError(
            f"Only {len(kept)} examples survive filtering but {needed} are required. "
            f"Dropped: {dict(dropped)}. Lower the sizes or relax the length filter."
        )
    return kept[n_val:needed], kept[:n_val], stats


def with_ids(rows: list[dict], prefix: str) -> list[dict]:
    """Attach stable ids in output order (ids first for readability)."""

    return [{"id": f"{prefix}-{index:06d}", **row} for index, row in enumerate(rows)]


# ---------------------------------------------------------------------------
# Dataset loaders (need network access; used on Kaggle)
# ---------------------------------------------------------------------------


def load_ultrachat_candidates(max_rows: int | None = None, seed: int = 42) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(ULTRACHAT["repo"], split=ULTRACHAT["split"], revision=ULTRACHAT["revision"])
    indices = list(range(len(dataset)))
    if max_rows is not None and max_rows < len(indices):
        # Deterministic subsample so we do not convert all ~208k conversations.
        indices = sorted(random.Random(seed).sample(indices, max_rows))
        dataset = dataset.select(indices)
    candidates = []
    for row in dataset:
        example = ultrachat_row_to_example(row)
        if example is not None:
            example["source"] = ULTRACHAT["repo"]
            candidates.append(example)
    return candidates


def load_hh_candidates(subsets: list[str] | None = None) -> list[dict]:
    from huggingface_hub import hf_hub_download

    candidates = []
    for subset in subsets or HH_RLHF["subsets"]:
        path = hf_hub_download(HH_RLHF["repo"], f"{subset}/{HH_RLHF['file']}", repo_type="dataset",
                               revision=HH_RLHF["revision"])
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                pair = hh_row_to_pair(json.loads(line))
                if pair is not None:
                    pair["source"] = f"{HH_RLHF['repo']}:{subset}"
                    candidates.append(pair)
    return candidates


def load_ultrafeedback_candidates() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(ULTRAFEEDBACK["repo"], split=ULTRAFEEDBACK["split"], revision=ULTRAFEEDBACK["revision"])
    candidates = []
    for row in dataset:
        pair = ultrafeedback_row_to_pair(row)
        if pair is not None:
            pair["source"] = f"{ULTRAFEEDBACK['repo']}:{row.get('source')}"
            candidates.append(pair)
    return candidates


def balance_hh_candidates(candidates: list[dict], harmless_fraction: float, total: int, seed: int,
                          minimum: int = 0) -> tuple[list[dict], dict]:
    """Subsample HH candidates so about ``harmless_fraction`` come from ``harmless-base``.

    The pool is at most ``total`` rows and is shrunk to the largest size at which the
    requested mix is still possible. If that would leave fewer than ``minimum`` rows, the
    larger side fills the gap and the mix is no longer exact; the returned stats record
    the mix actually used.
    """

    harmless = [row for row in candidates if row["source"].endswith(":harmless-base")]
    helpful = [row for row in candidates if not row["source"].endswith(":harmless-base")]
    rng = random.Random(seed)
    rng.shuffle(harmless)
    rng.shuffle(helpful)
    limits = [total]
    if harmless_fraction > 0:
        limits.append(int(len(harmless) / harmless_fraction))
    if harmless_fraction < 1:
        limits.append(int(len(helpful) / (1 - harmless_fraction)))
    size = max(min(limits), min(total, minimum))
    want_harmless = min(len(harmless), int(round(size * harmless_fraction)))
    want_helpful = min(len(helpful), size - want_harmless)
    want_harmless = min(len(harmless), size - want_helpful)
    pool = harmless[:want_harmless] + helpful[:want_helpful]
    stats = {"harmless_available": len(harmless), "helpful_available": len(helpful),
             "harmless_in_pool": want_harmless, "helpful_in_pool": want_helpful,
             "harmless_fraction_requested": harmless_fraction,
             "harmless_fraction_in_pool": want_harmless / max(1, len(pool))}
    return pool, stats


def alpaca_eval_subset(rows: list[dict], n: int = 300, seed: int = 42) -> list[dict]:
    """Fixed evaluation subset: ``random.Random(seed).sample`` of row indices, kept in file order."""

    if n > len(rows):
        raise ValueError(f"Requested {n} prompts but the source has only {len(rows)}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), n))
    subset = []
    for index in indices:
        row = rows[index]
        subset.append({
            "id": f"alpaca-eval-{index:04d}",
            "instruction": row["instruction"],
            "dataset": "alpaca_eval",
            "source_dataset": row.get("dataset"),
            "source_index": index,
        })
    return subset
