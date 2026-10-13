# Saved Kaggle run evidence

These are the downloaded outputs of the completed training, generation and judging runs.
The three notebooks below preserve their executed cells and outputs. The reusable source notebooks remain in
[`notebooks/`](../notebooks/). No model weights or LoRA adapter weights are included in this directory.

## What to open

| Stage | Executed notebook | Supporting evidence |
|---|---|---|
| Training | [Training notebook](Training/genai-pa2-training.ipynb) | Four training runs, completion records, validation metrics, and the 227-test result |
| Generation | [Generation notebook](Generation/genai-pa2-generation.ipynb) | Saved model responses, generation settings and blind A/B pair construction |
| Judging | [Judging notebook](Judging/genai-pa2-judging.ipynb) | Judge execution, comparison summaries and completion checks |

## Downloadable artifacts

| Artifact | Contents |
|---|---|
| [SFT outputs](Training/outputs/sft_run_outputs.zip) | SFT metrics, configuration, completion record, best/final checkpoint metadata and tokenizers; prepared SFT, HH and UltraFeedback train/validation data and provenance |
| [HH DPO outputs](Training/outputs/dpo_hh_run_outputs.zip) | Metrics, configuration, completion record, best/final checkpoint metadata and tokenizers |
| [UltraFeedback DPO outputs](Training/outputs/dpo_ultrafeedback_run_outputs.zip) | Metrics, configuration, completion record, best/final checkpoint metadata and tokenizers |
| [Persona records](Training/outputs/persona_lora/) | Data split, run configuration, metrics, trainer log history and completion record |
| [Training plots and tables](Training/outputs/training_plots_and_report.zip) | Four curve figures, evaluation tables and machine-readable training summaries |
| [Generations and comparison pairs](Generation/outputs/eval_generations_and_pairs.zip) | Generated responses and metadata, blind A/B pairs and metadata, and summary files |
| [Judge outputs](Judging/outputs/judged_outputs.zip) | Per-pair decisions, request-attempt logs, failure logs, comparison summaries and aggregate tables |

Download a ZIP using its GitHub file page's **Download raw file** button, then extract it locally.
The ZIPs retain their original `outputs/...` paths. Extract different stages into separate folders if you want
to preserve each stage's `outputs/report/` files, which share filenames. These generated tables are supporting
evidence, not the final written assignment report.

## Recorded comparison results

Wins and losses below are from the candidate's perspective. Win rate excludes ties:
`wins / (wins + losses)`. The judge was `google/gemini-3.6-flash` through `kaggle_benchmarks`.

| Candidate vs baseline | Pairs | Wins | Losses | Ties | Win rate |
|---|---:|---:|---:|---:|---:|
| HH DPO vs SFT (safety) | 300 | 190 | 78 | 32 | 70.90% |
| UltraFeedback DPO vs SFT (general) | 300 | 103 | 172 | 25 | 37.45% |
| Persona LoRA vs base (style) | 40 | 8 | 31 | 1 | 20.51% |
| Persona LoRA vs base (quality) | 40 | 4 | 32 | 4 | 11.11% |

All four comparisons are complete, with zero unknown decisions, failures or unattempted pairs.
These are results on the saved evaluation sets under the recorded judging protocol, not claims of universal improvement.

## Adapter verification

From the repository root:

```bash
python -m pytest -q tests/test_public_adapter_smoke.py tests/test_batching_edge_cases.py tests/test_losses_math.py
```

On 2026-10-14 this returned **35 passed**. It checks the six required `submission_adapter.py` interfaces,
response-only masking and log-probabilities, padding and truncation, SFT/DPO loss arithmetic, policy gradients,
and frozen reference-model behavior. No adapter changes were required. The private grading tests are unavailable.

All six downloaded ZIPs also passed their archive integrity checks before this evidence was committed.
