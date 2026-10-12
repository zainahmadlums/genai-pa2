"""Blind pairwise comparison files (candidate vs baseline, e.g. DPO vs SFT).

Each JSONL row holds the judge-visible fields (``instruction``, ``response_a``,
``response_b`` and an optional rubric hint) and a separate ``hidden`` block that maps
positions A/B back to model identities. Position is randomised with a fixed seed and is
balanced: half of the pairs show the candidate as Response A.
"""

from __future__ import annotations

import random

PROFILES = ("general", "safety", "persona")


def _index(entries: list[dict], name: str) -> dict[str, dict]:
    indexed = {}
    for entry in entries:
        if entry["id"] in indexed:
            raise ValueError(f"Duplicate id {entry['id']!r} in {name} outputs")
        indexed[entry["id"]] = entry
    return indexed


def build_pairs(baseline_entries: list[dict], candidate_entries: list[dict], seed: int = 42,
                comparison: str = "candidate_vs_baseline", profile: str = "general",
                include_expected_behavior: bool = True) -> list[dict]:
    """Return pairwise rows in the order of ``baseline_entries``.

    Both files must cover the same prompt ids with identical instructions.
    """

    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    baseline, candidate = _index(baseline_entries, "baseline"), _index(candidate_entries, "candidate")
    missing = sorted(set(baseline) ^ set(candidate))
    if missing:
        raise ValueError(f"{len(missing)} prompt ids are not present in both files, e.g. {missing[:5]}")

    ids = [entry["id"] for entry in baseline_entries]
    candidate_first = [index % 2 == 0 for index in range(len(ids))]
    random.Random(seed).shuffle(candidate_first)

    pairs = []
    for index, (prompt_id, candidate_is_a) in enumerate(zip(ids, candidate_first)):
        base_entry, cand_entry = baseline[prompt_id], candidate[prompt_id]
        if base_entry["instruction"] != cand_entry["instruction"]:
            raise ValueError(f"Instruction mismatch for id {prompt_id!r}")
        first, second = (cand_entry, base_entry) if candidate_is_a else (base_entry, cand_entry)
        row = {
            "pair_id": f"{comparison}:{prompt_id}",
            "prompt_id": prompt_id,
            "index": index,
            "comparison": comparison,
            "profile": profile,
            "dataset": base_entry.get("dataset"),
            "category": base_entry.get("category"),
            "instruction": base_entry["instruction"],
            "response_a": first["output"],
            "response_b": second["output"],
            "hidden": {
                "model_a": first["generator"],
                "model_b": second["generator"],
                "candidate": cand_entry["generator"],
                "baseline": base_entry["generator"],
                "candidate_position": "A" if candidate_is_a else "B",
                "seed": seed,
            },
        }
        if include_expected_behavior and base_entry.get("expected_behavior"):
            row["expected_behavior"] = base_entry["expected_behavior"]
        pairs.append(row)
    return pairs


def judge_view(pair: dict) -> dict:
    """The only fields a judge prompt may use.

    Model identities are never included, and neither is ``pair_id`` because it embeds
    the comparison name; the judge sees neutral item numbers instead.
    """

    view = {
        "instruction": pair["instruction"],
        "response_a": pair["response_a"],
        "response_b": pair["response_b"],
    }
    if pair.get("expected_behavior"):
        view["expected_behavior"] = pair["expected_behavior"]
    return view


def translate_verdict(pair: dict, verdict: str) -> dict:
    """Map a judge verdict (``A``/``B``/``tie``/``unknown``) back to model identities."""

    hidden = pair["hidden"]
    if verdict in ("A", "B"):
        winner_model = hidden["model_a"] if verdict == "A" else hidden["model_b"]
        outcome = "candidate_win" if verdict == hidden["candidate_position"] else "baseline_win"
    elif verdict == "tie":
        winner_model, outcome = None, "tie"
    else:
        winner_model, outcome = None, "unknown"
    return {"outcome": outcome, "winner_model": winner_model}
