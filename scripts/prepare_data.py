"""Build the single-turn training/validation JSONL files and their provenance records.

Examples:
    python scripts/prepare_data.py --task sft
    python scripts/prepare_data.py --task hh
    python scripts/prepare_data.py --task ultrafeedback
    python scripts/prepare_data.py --task all
    python scripts/prepare_data.py --task hh --n_train 200 --n_val 50 --out_dir outputs/data_smoke

Outputs (default ``outputs/data``): ``sft_{train,val}.jsonl``, ``hh_{train,val}.jsonl``,
``uf_{train,val}.jsonl`` and one ``*_provenance.json`` per task. Existing files are kept
unless ``--force`` is given. Prompts from every file in ``eval_sets/`` are excluded.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment import data_prep  # noqa: E402
from alignment.prompt_format import DEFAULT_TEMPLATE  # noqa: E402
from alignment.train_utils import write_json, write_jsonl  # noqa: E402

DEFAULT_SIZES = {"sft": (30000, 1000), "hh": (15000, 1000), "ultrafeedback": (15000, 1000)}
PREFIX = {"sft": "sft", "hh": "hh", "ultrafeedback": "uf"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["sft", "hh", "ultrafeedback", "all"])
    parser.add_argument("--out_dir", default="outputs/data")
    parser.add_argument("--n_train", type=int, default=None, help="default: 30000 (sft) / 15000 (hh, ultrafeedback)")
    parser.add_argument("--n_val", type=int, default=None, help="default: 1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="tokenizer used for the length filters")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    parser.add_argument("--allow_truncated_responses", action="store_true",
                        help="keep examples whose response does not fully fit in max_length")
    parser.add_argument("--eval_sets_dir", default="eval_sets", help="prompts in these JSON files are excluded")
    parser.add_argument("--hh_harmless_fraction", type=float, default=0.5,
                        help="target share of harmless-base pairs in the HH pool")
    parser.add_argument("--ultrachat_max_rows", type=int, default=0,
                        help="convert only a seeded subsample of this many ultrachat conversations (0 = all)")
    parser.add_argument("--force", action="store_true", help="rebuild files that already exist")
    return parser


def prepare(task: str, args, tokenizer, exclude_keys: set[str]) -> dict:
    out_dir = Path(args.out_dir)
    prefix = PREFIX[task]
    train_path, val_path = out_dir / f"{prefix}_train.jsonl", out_dir / f"{prefix}_val.jsonl"
    provenance_path = out_dir / f"{prefix}_provenance.json"
    if train_path.exists() and val_path.exists() and provenance_path.exists() and not args.force:
        print(f"[prepare:{task}] {train_path} already exists; use --force to rebuild")
        return {}

    n_train = args.n_train if args.n_train is not None else DEFAULT_SIZES[task][0]
    n_val = args.n_val if args.n_val is not None else DEFAULT_SIZES[task][1]
    length_filter = data_prep.LengthFilter(
        tokenizer, template=args.template, max_length=args.max_length, max_prompt_tokens=args.max_prompt_tokens,
        require_full_fit=not args.allow_truncated_responses,
    )
    extra = {}
    if task == "sft":
        source = data_prep.ULTRACHAT
        candidates = data_prep.load_ultrachat_candidates(args.ultrachat_max_rows or None, args.seed)
        extra["conversion"] = "first user turn -> instruction, first assistant turn -> response"
        extra["ultrachat_max_rows"] = args.ultrachat_max_rows
    elif task == "hh":
        source = data_prep.HH_RLHF
        candidates = data_prep.load_hh_candidates()
        extra["single_turn_pairs_available"] = len(candidates)
        candidates, balance = data_prep.balance_hh_candidates(
            candidates, args.hh_harmless_fraction, total=3 * (n_train + n_val), seed=args.seed,
            minimum=int(1.25 * (n_train + n_val)),
        )
        extra["conversion"] = "transcripts with exactly one Human and one Assistant turn; identical Human turn on both sides"
        extra["balance"] = balance
    else:
        source = data_prep.ULTRAFEEDBACK
        candidates = data_prep.load_ultrafeedback_candidates()
        extra["conversion"] = "single-turn pairs with chosen-rating > rejected-rating; last assistant message of each side"

    print(f"[prepare:{task}] {len(candidates)} single-turn candidates from {source['repo']}", flush=True)
    train, val, stats = data_prep.select_split(candidates, n_train, n_val, args.seed, exclude_keys, length_filter)
    train, val = data_prep.with_ids(train, f"{prefix}-train"), data_prep.with_ids(val, f"{prefix}-val")
    overlap = {data_prep.normalize_text(row["instruction"]) for row in train} & {
        data_prep.normalize_text(row["instruction"]) for row in val}
    assert not overlap, "train and validation instructions overlap"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    provenance = {
        "task": task,
        "source": source,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "length_filter": length_filter.describe(),
        "selection": stats,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_sources": dict(Counter(row["source"] for row in train)),
        "val_sources": dict(Counter(row["source"] for row in val)),
        "excluded_eval_prompts": len(exclude_keys),
        "files": {str(train_path): data_prep.sha256_file(train_path), str(val_path): data_prep.sha256_file(val_path)},
        **extra,
    }
    write_json(provenance_path, provenance)
    print(f"[prepare:{task}] wrote {len(train)} train / {len(val)} val rows to {out_dir} "
          f"(dropped before quota: {stats['dropped_before_quota']})")
    return provenance


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eval_files = sorted(Path(args.eval_sets_dir).glob("*.json"))
    exclude_keys = data_prep.load_eval_instruction_keys(eval_files)
    print(f"[prepare] excluding {len(exclude_keys)} fixed evaluation prompts from {len(eval_files)} files")
    tasks = ["sft", "hh", "ultrafeedback"] if args.task == "all" else [args.task]
    for task in tasks:
        prepare(task, args, tokenizer, exclude_keys)


if __name__ == "__main__":
    main()
