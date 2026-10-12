"""Generate responses for a fixed evaluation file with one model (base, SFT, DPO, or a LoRA adapter).

Always run a small sanity check first and read the printed samples:

    python scripts/generate.py --model_path outputs/sft/best --prompts_path eval_sets/alpaca_eval_300_seed42.json \
        --generator qwen2.5-0.5b-sft --output_path outputs/eval/generations/alpaca_sft.json --smoke 5

Then the full subset (same command without --smoke). Output: a JSON list of
``{id, instruction, output, generator, dataset}`` plus ``<output>.meta.json`` with the
decoding settings and provenance. Rerunning the same command continues an unfinished file.

Existing outputs are reused only if they were produced by the same checkpoint contents
(and adapter), prompt template, decoding settings, precision and generator label. If any
of these changed, the run stops without generating and without touching the file; pass
``--overwrite`` to regenerate everything, or choose another ``--output_path``.
``--batch_size`` may change freely.

Decoding defaults to greedy. Use the same flags for every model in a comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.generation import (  # noqa: E402
    GenerationCacheMismatch,
    GenerationConfig,
    checkpoint_fingerprint,
    generate_file,
    load_eval_prompts,
)
from alignment.prompt_format import CHAT_TEMPLATE, DEFAULT_TEMPLATE, TEMPLATES  # noqa: E402
from alignment.train_utils import AmpSettings, load_checkpoint_meta, resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Hugging Face id or local checkpoint directory")
    parser.add_argument("--adapter_path", default=None, help="optional PEFT/LoRA adapter directory (persona track)")
    parser.add_argument("--prompts_path", required=True, help="JSON list with id and instruction")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator", required=True, help="model label stored in the output, e.g. qwen2.5-0.5b-sft")
    parser.add_argument("--dataset", default=None, help="dataset label; default: taken from the prompt records")
    parser.add_argument("--template", default="auto",
                        choices=["auto", CHAT_TEMPLATE, *sorted(TEMPLATES)],
                        help="auto = template stored with the checkpoint, else the project default")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_sample", action="store_true", help="sampling instead of greedy (not the default)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="sanity check: first N prompts, printed, written to <output>.smoke.json")
    parser.add_argument("--print_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> list[dict]:
    args = build_parser().parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(args.output_path)
    limit, print_samples, overwrite = args.limit, args.print_samples, args.overwrite
    if args.smoke:
        limit, print_samples, overwrite = args.smoke, args.smoke, True
        output_path = output_path.with_name(output_path.stem + ".smoke.json")

    meta = load_checkpoint_meta(args.model_path) if Path(args.model_path).is_dir() else {}
    template = args.template
    if template == "auto":
        template = meta.get("template") or (CHAT_TEMPLATE if args.adapter_path else DEFAULT_TEMPLATE)
    elif meta.get("template") and meta["template"] != template:
        print(f"[generate] WARNING: checkpoint was trained with template {meta['template']!r} but {template!r} was requested")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path or args.model_path)
    # fp32 weights + fp16 autocast: the same numerics as training.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.to(device).eval()
    amp = AmpSettings(device, args.precision)

    records = load_eval_prompts(args.prompts_path, limit)
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
        batch_size=args.batch_size,
    )
    print(f"[generate] model={args.model_path} adapter={args.adapter_path} template={template} device={device} "
          f"prompts={len(records)} -> {output_path}")
    # Identity of what generates the text. Local directories are identified by content (so retraining into
    # the same path is noticed and moving a checkpoint is not); hub models by their id.
    model_identity = {
        "model": checkpoint_fingerprint(args.model_path) or str(args.model_path),
        "adapter": (checkpoint_fingerprint(args.adapter_path) or str(args.adapter_path)) if args.adapter_path else None,
        "precision": amp.effective,
        "seed": args.seed if args.do_sample else None,  # greedy decoding does not depend on the seed
    }
    try:
        entries = generate_file(
            model, tokenizer, records, output_path, args.generator, args.dataset, template, config, device,
            autocast=amp.autocast, overwrite=overwrite, model_identity=model_identity,
            extra_meta={"model_path": str(args.model_path), "adapter_path": args.adapter_path,
                        "prompts_path": str(args.prompts_path), "precision": amp.effective, "seed": args.seed,
                        "checkpoint_meta": meta},
        )
    except GenerationCacheMismatch as exc:
        raise SystemExit(f"[generate] {exc}") from exc
    for entry in entries[:print_samples]:
        print("=" * 80)
        print(f"[{entry['id']}] INSTRUCTION: {entry['instruction']}")
        print(f"OUTPUT ({entry['generator']}): {entry['output']}")
    return entries


if __name__ == "__main__":
    main()
