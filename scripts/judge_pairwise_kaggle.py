"""Judge a saved pairwise JSONL file with a Kaggle-hosted model. No generation happens here.

    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/pairs/dpo_uf_vs_sft_pairs.jsonl \
        --output_path outputs/eval/judged/dpo_uf_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash \
        --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

Files written next to ``--output_path``:
    <name>.jsonl            one judgment per pair, appended after every batch
    <name>.failures.jsonl   pairs with no judgment after all retries
    <name>.attempts.jsonl   every request attempt (status, error, raw reply)
    <name>.summary.json     win / loss / tie / unknown / failure counts and rates

Rerun the same command to continue: with ``--reuse_judgments`` finished pairs are skipped
and earlier failures are retried. ``--list_models`` prints the judge models Kaggle exposes.
``--backend mock`` checks the pipeline offline; its verdicts are not results.

Saved judgments are reused only when they were produced by the same backend, the same
judge model and the same protocol (batch size, rubric text, response clipping). If the
output or attempts file holds anything else (for example mock verdicts, or another judge
model), the run stops before sending a request. Choose another ``--output_path``, or add
``--invalidate_incompatible`` to rename the old files to ``*.stale<N>`` and judge every
pair again. ``--sleep_seconds`` is enforced between all requests of a process, including
the availability check, retries and consecutive comparisons.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judge import (  # noqa: E402
    MIN_SLEEP_SECONDS,
    PACER_STATE_FILE,
    JudgeCacheMismatch,
    JudgeRunner,
    shared_pacer,
)
from alignment.judge_backends import JudgeBackendError, KaggleBenchmarksBackend, make_backend  # noqa: E402
from alignment.io_utils import read_jsonl  # noqa: E402

DEFAULT_JUDGE_MODEL = "google/gemini-3.6-flash"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_path", help="pairwise JSONL from scripts/build_pairs.py")
    parser.add_argument("--output_path", help="judged JSONL to write/append")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--backend", default="auto", choices=["auto", "kaggle_benchmarks", "openai_proxy", "mock"])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0,
                        help=f"minimum wait between request attempts (values below {MIN_SLEEP_SECONDS:g} are raised)")
    parser.add_argument("--max_retries", type=int, default=3, help="attempts per batch")
    parser.add_argument("--reuse_judgments", action="store_true", help="skip pairs already judged in --output_path")
    parser.add_argument("--retry_unknown", action="store_true", help="with --reuse_judgments, re-judge 'unknown' pairs")
    parser.add_argument("--invalidate_incompatible", action="store_true",
                        help="if saved results come from another backend, judge model or protocol (batch size, "
                             "rubric, response clipping), rename them to *.stale<N> and start again; "
                             "without this flag such a run stops with an error before any request")
    parser.add_argument("--failures_path", default=None)
    parser.add_argument("--attempts_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="stop after N batches (for a first check)")
    parser.add_argument("--max_response_chars", type=int, default=6000)
    parser.add_argument("--skip_preflight", action="store_true", help="do not send the one-line availability check")
    parser.add_argument("--list_models", action="store_true", help="print models listed by kaggle_benchmarks and exit")
    return parser


def list_models() -> None:
    print("LLMS_AVAILABLE =", os.environ.get("LLMS_AVAILABLE", "(not set)"))
    print("LLM_DEFAULT =", os.environ.get("LLM_DEFAULT", "(not set)"))
    try:
        for name in KaggleBenchmarksBackend.available_models():
            print(" ", name)
    except Exception as exc:  # noqa: BLE001
        print(f"kaggle_benchmarks unavailable: {type(exc).__name__}: {exc}")


def main(argv=None, *, backend=None, pacer=None) -> dict | None:
    """Command-line entry point. ``backend`` and ``pacer`` are injection points for tests."""

    args = build_parser().parse_args(argv)
    if args.list_models:
        list_models()
        return None
    if not args.pairs_path or not args.output_path:
        raise SystemExit("--pairs_path and --output_path are required")

    pairs = read_jsonl(args.pairs_path)
    if backend is None:
        try:
            backend = make_backend(args.backend, args.judge_model)
        except JudgeBackendError as exc:
            raise SystemExit(f"[judge] {exc}") from exc
    is_mock = backend.name == "mock"
    sleep_seconds = args.sleep_seconds
    if not is_mock and sleep_seconds < MIN_SLEEP_SECONDS:
        print(f"[judge] --sleep_seconds {sleep_seconds:g} raised to the minimum of {MIN_SLEEP_SECONDS:g}")
        sleep_seconds = MIN_SLEEP_SECONDS

    if pacer is None:
        # One pacer per process: the interval also holds between consecutive comparisons run from the
        # same notebook kernel. The state file extends it to separate processes writing to this folder.
        pacer = shared_pacer()
        pacer.state_path = None if is_mock else Path(args.output_path).parent / PACER_STATE_FILE
    runner = JudgeRunner(backend, args.judge_model, args.batch_size, sleep_seconds, args.max_retries,
                         args.max_response_chars, sleep_fn=pacer.sleep_fn, clock=pacer.clock, pacer=pacer)

    on_incompatible = "invalidate" if args.invalidate_incompatible else "error"
    if args.reuse_judgments and on_incompatible == "error":
        # Checked before the preflight so that no judge call is spent on a run that cannot proceed.
        incompatible = runner.incompatible_cache(args.output_path, args.attempts_path)
        if incompatible:
            raise SystemExit(
                f"[judge] {args.output_path} holds results that were not produced by backend={backend.name!r}, "
                f"judge_model={args.judge_model!r} with the current protocol: {'; '.join(incompatible)}.\n"
                "They are never mixed with or relabelled as results of this judge. Use a different --output_path, "
                "or add --invalidate_incompatible to rename the old files to *.stale<N> and judge all pairs again."
            )

    if not is_mock and not args.skip_preflight:
        # One tiny request so an unavailable judge model fails here, not after many retries.
        # It goes through the shared pacer like every other request.
        try:
            reply = pacer.request(sleep_seconds, lambda: backend.call("Reply with the single word OK."))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"[judge] preflight request to {args.judge_model!r} failed: {type(exc).__name__}: {exc}\n"
                "Run with --list_models and pass a supported model through --judge_model."
            ) from exc
        print(f"[judge] preflight reply from {args.judge_model}: {str(reply)[:80]!r}")

    print(f"[judge] {len(pairs)} pairs, backend={backend.name}, model={args.judge_model}, batch={args.batch_size}, "
          f"sleep={sleep_seconds:g}s, attempts={args.max_retries}, protocol_id={runner.protocol_id}")
    try:
        summary = runner.run(pairs, args.output_path, args.failures_path, args.attempts_path, args.summary_path,
                             reuse_judgments=args.reuse_judgments, retry_unknown=args.retry_unknown,
                             max_batches=args.max_batches, on_incompatible=on_incompatible)
    except JudgeCacheMismatch as exc:
        raise SystemExit(f"[judge] {exc}") from exc
    if backend.name == "mock":
        print("[judge] NOTE: mock backend used; these verdicts are a pipeline check, not evaluation results")
    print(f"[judge] {summary['comparison']}: candidate wins {summary['candidate_wins']}, baseline wins "
          f"{summary['baseline_wins']}, ties {summary['ties']}, unknown {summary['unknown']}, failures "
          f"{summary['failures']}, not attempted {summary['not_attempted']} of {summary['total_pairs']}")
    return summary


if __name__ == "__main__":
    main()
