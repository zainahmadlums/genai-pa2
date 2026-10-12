"""Create (or verify) the fixed AlpacaEval subset: 300 prompts, seed 42.

The committed file ``eval_sets/alpaca_eval_300_seed42.json`` was produced by this script.
Run it again to verify that the committed subset is reproducible:

    python scripts/make_alpaca_eval_subset.py --verify

Source: ``tatsu-lab/alpaca_eval`` file ``alpaca_eval.json`` at a pinned commit
(805 prompts, CC BY-NC 4.0). Selection: ``sorted(random.Random(42).sample(range(805), 300))``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.data_prep import ALPACA_EVAL, alpaca_eval_subset  # noqa: E402
from alignment.io_utils import read_json, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default=None, help="default: eval_sets/alpaca_eval_<n>_seed<seed>.json")
    parser.add_argument("--source_json", default=None, help="local copy of alpaca_eval.json (skips the download)")
    parser.add_argument("--verify", action="store_true", help="compare with the existing file instead of writing")
    return parser


def load_source(source_json: str | None) -> list[dict]:
    if source_json:
        return read_json(source_json)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ALPACA_EVAL["repo"], ALPACA_EVAL["file"], repo_type="dataset",
                           revision=ALPACA_EVAL["revision"])
    return read_json(path)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output_path or f"eval_sets/alpaca_eval_{args.n}_seed{args.seed}.json")
    subset = alpaca_eval_subset(load_source(args.source_json), args.n, args.seed)
    if args.verify:
        existing = read_json(output_path)
        if existing != subset:
            raise SystemExit(f"MISMATCH: {output_path} differs from the regenerated subset")
        print(f"OK: {output_path} matches the regenerated subset ({len(subset)} prompts)")
        return
    if output_path.exists():
        if read_json(output_path) == subset:
            print(f"{output_path} already up to date ({len(subset)} prompts)")
            return
        raise SystemExit(f"{output_path} exists with different content; remove it first if you intend to replace it")
    write_json(output_path, subset)
    print(f"wrote {len(subset)} prompts to {output_path}")
    print(json.dumps(subset[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
