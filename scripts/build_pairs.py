"""Build a blind pairwise comparison file (candidate vs baseline) from two generation files.

    python scripts/build_pairs.py \
        --baseline_path outputs/eval/generations/alpaca_sft.json \
        --candidate_path outputs/eval/generations/alpaca_dpo_uf.json \
        --output_path outputs/eval/pairs/dpo_uf_vs_sft_pairs.jsonl \
        --comparison dpo_uf_vs_sft --profile general

Response order is randomised with ``--seed`` (balanced A/B). Model identities are stored
only in each row's ``hidden`` block, which the judge prompt never reads.
Profiles: ``general`` (AlpacaEval), ``safety`` (helpful/harmless set), ``persona``.

Both inputs must come from ``generate.py`` runs that finished and used the same prompt
template, decoding settings and precision; this is read from the ``.meta.json`` file next
to each generation file. Otherwise the script stops and lists the differences, because a
judge would then compare decoding policies instead of models. The checkpoints may differ,
that is the comparison. ``--allow_unverified_inputs`` is an explicit escape hatch for
legacy or hand-made files.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.pairwise import PROFILES, build_pairs  # noqa: E402
from alignment.io_utils import read_json, write_json, write_jsonl  # noqa: E402


def describe_difference(saved: dict, current: dict, prefix: str = "") -> str:
    """Fields that differ between two settings dicts, as ``key: old -> new`` (nested keys dotted)."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline_path", required=True, help="generation JSON of the baseline (e.g. SFT)")
    parser.add_argument("--candidate_path", required=True, help="generation JSON of the candidate (e.g. DPO)")
    parser.add_argument("--output_path", required=True, help="pairwise JSONL to write")
    parser.add_argument("--comparison", required=True, help="name such as dpo_hh_vs_sft")
    parser.add_argument("--profile", default="general", choices=PROFILES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_expected_behavior", action="store_true",
                        help="do not pass the eval set's expected_behavior rubric hint to the judge")
    parser.add_argument("--allow_unverified_inputs", action="store_true",
                        help="pair the files even if their .meta.json files are missing, incomplete or record "
                             "different generation settings (legacy or hand-made files only); the pairs meta "
                             "file then says inputs_verified=false")
    return parser


def generation_settings(path, entries: list[dict]) -> tuple[dict | None, list[str]]:
    """Read ``<generation>.meta.json`` and return ``(comparable settings, problems)``.

    The comparable settings are the ones that must be equal for a fair pair: prompt template,
    decoding settings, precision, sampling seed and output post-processing. Checkpoint and
    adapter identity are deliberately left out: the two sides of a pair are different models.
    """

    path = Path(path)
    meta_path = path.with_name(path.stem + ".meta.json")
    if not meta_path.exists():
        return None, [f"{path.name}: no {meta_path.name} next to it, so its generation settings are unknown"]
    meta = read_json(meta_path)
    provenance = meta.get("provenance")
    if not isinstance(provenance, dict):
        return None, [f"{path.name}: {meta_path.name} has no provenance record (written by an older version)"]
    problems = []
    if meta.get("status") != "complete":
        problems.append(f"{path.name}: generation is not complete (status={meta.get('status')!r}); rerun generate.py")
    if meta.get("num_outputs") != len(entries):
        problems.append(f"{path.name}: holds {len(entries)} outputs but its meta file records {meta.get('num_outputs')}")
    identity = provenance.get("model_identity") or {}
    settings = {
        "template": provenance.get("template"),
        "decoding": provenance.get("decoding"),
        "precision": identity.get("precision"),
        "seed": identity.get("seed"),
        "stop_string_trimming": meta.get("stop_string_trimming"),
    }
    return settings, problems


def check_inputs(baseline_path, baseline: list[dict], candidate_path, candidate: list[dict]) -> tuple[dict | None, list[str]]:
    """Return ``(shared generation settings, problems)``; no problems means the pair is comparable."""

    baseline_settings, problems = generation_settings(baseline_path, baseline)
    candidate_settings, candidate_problems = generation_settings(candidate_path, candidate)
    problems = problems + candidate_problems
    if baseline_settings is not None and candidate_settings is not None and baseline_settings != candidate_settings:
        problems.append("the two files were generated with different settings (baseline -> candidate): "
                        + describe_difference(baseline_settings, candidate_settings))
    shared = baseline_settings if not problems else None
    return shared, problems


def main(argv=None) -> list[dict]:
    args = build_parser().parse_args(argv)
    baseline, candidate = read_json(args.baseline_path), read_json(args.candidate_path)
    settings, problems = check_inputs(args.baseline_path, baseline, args.candidate_path, candidate)
    if problems and not args.allow_unverified_inputs:
        raise SystemExit(
            "[build_pairs] refusing to pair these files because a fair comparison cannot be confirmed:\n  - "
            + "\n  - ".join(problems)
            + "\nRegenerate both sides with identical template and decoding flags (generate.py, --overwrite where "
              "needed). For hand-made or legacy files without metadata only, --allow_unverified_inputs pairs them "
              "anyway and records that the inputs were not verified."
        )
    for problem in problems:
        print(f"[build_pairs] WARNING (--allow_unverified_inputs): {problem}")
    pairs = build_pairs(baseline, candidate, args.seed, args.comparison, args.profile,
                        include_expected_behavior=not args.no_expected_behavior)
    output_path = Path(args.output_path)
    write_jsonl(output_path, pairs)
    positions = Counter(pair["hidden"]["candidate_position"] for pair in pairs)
    hidden = pairs[0]["hidden"] if pairs else {}
    write_json(output_path.with_name(output_path.stem + ".meta.json"), {
        "comparison": args.comparison, "profile": args.profile, "seed": args.seed, "pairs": len(pairs),
        "candidate": hidden.get("candidate"), "baseline": hidden.get("baseline"),
        "candidate_position_counts": dict(positions),
        "baseline_path": str(args.baseline_path), "candidate_path": str(args.candidate_path),
        # True only when both generation files are complete and share template, decoding and precision.
        "inputs_verified": not problems,
        "generation_settings": settings,
        "unverified_reasons": problems,
    })
    print(f"wrote {len(pairs)} pairs to {output_path} (candidate shown as A in {positions['A']}, as B in {positions['B']})")
    return pairs


if __name__ == "__main__":
    main()
