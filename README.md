# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

Implementation of the assignment in `docs/assignment_manual.pdf` (identical to `PA2.pdf`):

1. **SFT** of `Qwen/Qwen2.5-0.5B` on 30k single-turn UltraChat examples with a hand-written training loop.
2. **Two from-scratch DPO runs** from the same SFT best checkpoint, with a frozen SFT reference:
   Anthropic HH-RLHF (15k pairs) and UltraFeedback (15k pairs).
3. **Persona LoRA** on `HuggingFaceTB/SmolLM2-360M-Instruct` with TRL `DPOTrainer` + PEFT (the only place TRL is used).
4. **Evaluation**: fixed prompt sets, saved generations, blind randomised A/B pair files, and a resumable judge that
   calls a Kaggle-hosted model.

> **No results are included.** Nothing in this repository was trained, generated or judged yet. Training needs
> Kaggle T4 x2 GPUs and judging needs Kaggle's model proxy. To run everything, follow
> **[docs/KAGGLE_RUN_GUIDE.md](docs/KAGGLE_RUN_GUIDE.md)**.

## Layout

```text
submission_adapter.py        grading interface: six thin wrappers (signatures unchanged from the starter)
alignment/
  batching.py                prompt+response+EOS batches, -100 masking, truncation, padding
  losses.py                  response log-probabilities, SFT loss, DPO loss, reward accuracy / margin
  engine.py                  the custom training loop (accumulation, AMP, clipping, schedules, checkpoints, resume)
  sft_trainer.py             SFT task (collate, loss, validation)
  dpo_trainer.py             DPO task (policy + frozen reference, validation metrics)
  prompt_format.py           prompt templates shared by training and generation
  data_prep.py               dataset conversion, filtering, deterministic splits
  generation.py              batched greedy generation with restartable output files
  pairwise.py                blind A/B pair construction and verdict translation
  judge.py, judge_backends.py  rate-limited, resumable judging; Kaggle model adapters
  plotting.py                training curves
scripts/                     command-line entry points (each has --help)
configs/                     sft.yaml, dpo_hh.yaml, dpo_ultrafeedback.yaml, persona_lora.yaml
notebooks/                   01_train_kaggle, 02_generate_kaggle, 03_judge_kaggle
eval_sets/                   fixed evaluation prompts (HH 300, AlpacaEval 300 seed 42, persona 40)
data/persona_lora/           the provided 220 persona preference pairs
tests/                       public adapter tests (unchanged) + supplementary tests
docs/                        Kaggle run guide, design notes, data provenance, report template, the manual
tools/build_notebooks.py     regenerates the notebooks
```

## Local check (CPU, no model or dataset downloads)

```bash
python -m pip install torch pytest pyyaml matplotlib      # locally only; on Kaggle torch is preinstalled
python -m pip install "transformers>=4.56.2,<5"           # optional: enables tests/test_end_to_end_tiny_model.py
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
PYTHONPATH=. python -m pytest -q tests
```

`tests/test_end_to_end_tiny_model.py` builds a random 2-layer Qwen2 model and a character tokenizer in a temporary
directory and runs the real `train_sft.py`, `train_dpo.py`, `generate.py`, `build_pairs.py`,
`judge_pairwise_kaggle.py --backend mock`, `plot_metrics.py` and `summarize_results.py` on it. It is skipped when
`transformers` is not installed. The persona script (TRL + PEFT) has no offline test.

On Windows PowerShell use `$env:PYTHONPATH="."` first, or run `python -m pytest -q tests` from the repository root
(`pyproject.toml` already puts the root on the test path).

## What has and has not been verified

Verified on a CPU development machine (Windows, Python 3.14, torch 2.14.1+cpu, transformers 4.57.6, datasets 4.8.5,
peft 0.21.2, trl 1.14.1):

- `python -m pytest -q tests`: all tests pass, including the unchanged public adapter tests, the end-to-end run of
  the real scripts on a random 2-layer Qwen2 model, and the TRL + LoRA persona script on the same tiny model.
- Ten deliberate single-line bugs (no label shift, unmasked prompt, EOS forced after truncation, masks derived from
  the pad id, flipped DPO sign, wrong beta scaling, ties counted as correct, wrong accumulation divisor, judge
  pacing removed, verdicts not mapped through the hidden position) each make at least one test fail.
- Batching with the real `Qwen/Qwen2.5-0.5B` tokenizer (PAD = EOS = `<|endoftext|>`, id 151643, no BOS).
- `scripts/prepare_data.py --task hh` and `--task ultrafeedback` on the real datasets (counts in
  `docs/DATA_PROVENANCE.md`).
- `--help` of every script and Python syntax of every notebook code cell.

Not verified (needs Kaggle):

- Any GPU run: fp16 autocast with `GradScaler`, the two-GPU policy/reference placement, memory use, run times and
  disk use. No model weights were downloaded; no training on real models was done.
- `scripts/prepare_data.py --task sft` on the full UltraChat download (its converter was checked on 400 real rows).
- The Kaggle judge backends (`kaggle_benchmarks`, model proxy) and the availability of `google/gemini-3.6-flash`.
- The notebooks as a whole (cells compile; they were not executed).

## Pipeline commands (run on Kaggle; see the run guide for the notebook form)

```bash
python scripts/check_env.py --require_gpus 2
python scripts/prepare_data.py --task all                              # outputs/data/*.jsonl + provenance
python scripts/train_sft.py --config configs/sft.yaml                  # outputs/sft/{best,final}
python scripts/train_dpo.py --config configs/dpo_hh.yaml               # outputs/dpo_hh/{best,final}
python scripts/train_dpo.py --config configs/dpo_ultrafeedback.yaml    # outputs/dpo_ultrafeedback/{best,final}
python scripts/train_persona_lora.py --config configs/persona_lora.yaml  # outputs/persona_lora/adapter
python scripts/plot_metrics.py --run_dir outputs/sft outputs/dpo_hh outputs/dpo_ultrafeedback outputs/persona_lora

python scripts/generate.py --model_path outputs/sft/best --prompts_path eval_sets/alpaca_eval_300_seed42.json \
    --generator qwen2.5-0.5b-sft --output_path outputs/eval/generations/alpaca_sft.json --smoke 5
python scripts/build_pairs.py --baseline_path outputs/eval/generations/alpaca_sft.json \
    --candidate_path outputs/eval/generations/alpaca_dpo_uf.json \
    --output_path outputs/eval/pairs/dpo_uf_vs_sft_pairs.jsonl --comparison dpo_uf_vs_sft --profile general
python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/pairs/dpo_uf_vs_sft_pairs.jsonl \
    --output_path outputs/eval/judged/dpo_uf_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash \
    --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
python scripts/summarize_results.py --out_dir outputs/report           # report_tables.md from files on disk
```

Every training script accepts `--smoke` (a 4-step run), `--resume`, `--force`, and `--override key=value ...`.

## Implementation facts the report must state

| Topic | Choice | Where |
|---|---|---|
| Prompt format | `### Instruction:\n{instruction}\n\n### Response:\n` + response + EOS (`<\|endoftext\|>`), no system prompt; the same for SFT, both DPO runs and generation | `alignment/prompt_format.py` |
| SFT loss reduction | mean NLL over all unmasked response tokens (token mean) | `alignment/losses.py` |
| EOS / truncation | EOS appended to the response, then right-truncation to 512 tokens | `alignment/batching.py` |
| DPO loss | `mean(-logsigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r))))` on summed response log-probabilities | `alignment/losses.py` |
| DPO metrics | reward accuracy with strict `r_w > r_l` (0 at the tied start), reward margin `mean(r_w - r_l)` | `alignment/losses.py`, `alignment/dpo_trainer.py` |
| Checkpoint selection | SFT: lowest validation loss. DPO: highest validation reward accuracy, margin breaks ties. Best + final saved | `alignment/engine.py` |
| Precision | fp16 autocast, fp32 trainable weights, `GradScaler`, gradients unscaled before clipping | `alignment/engine.py`, `alignment/train_utils.py` |

More detail and the reasons: [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md). Data sources and filters:
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). Report outline: [docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md).

## Sources and attribution

- Starter files (`submission_adapter.py` scaffold, `tests/test_public_adapter_smoke.py`, `tests/toy_lm.py`,
  `data/persona_lora/*`, `eval_sets/hh_rlhf_eval_300_seed42.json`, `docs/assignment_manual.pdf`) come from
  <https://github.com/ahmad903501/small-model-alignment-starter> (branch `main`). The tests, data and evaluation
  files are byte-identical to the starter.
- `eval_sets/alpaca_eval_300_seed42.json` is a 300-prompt subset of `tatsu-lab/alpaca_eval` (CC BY-NC 4.0).
- DPO follows Rafailov et al., *Direct Preference Optimization* (NeurIPS 2023). The SFT and DPO training code was
  written for this project; no code from TRL or other DPO implementations is used outside the persona track.
- Library interfaces were checked against `trl==1.14.1` and `kaggle-benchmarks==0.6.1` sources.
