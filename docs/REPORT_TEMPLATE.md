# Final report template (8 pages maximum, including figures, tables and references)

Outline following section 14 of the manual. Text in *italics* tells you what to write; `[...]` marks a value to
copy from a file. **Do not submit this template with unfilled placeholders, and do not write a number that is not in
one of your output files.** Suggested page budget in parentheses.

---

## 1. Repository, models and data (about 0.75 page)

- Repository: `[GitHub URL]` (commit `[hash]`).
- Base checkpoint: `Qwen/Qwen2.5-0.5B`. Persona base: `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Seed: `[42]` for data selection, training, pair randomisation and generation.

| Track | Dataset and revision | Train / val rows | Filters |
|---|---|---|---|
| SFT | `[sft_provenance.json: source]` | `[train_rows]` / `[val_rows]` | *single turn = first exchange; prompt <= 256 tokens; full fit in 512; eval prompts excluded* |
| HH DPO | `[hh_provenance.json]` | `[...]` / `[...]` | *single-turn transcripts; harmless/helpful mix `[balance]`* |
| UltraFeedback DPO | `[uf_provenance.json]` | `[...]` / `[...]` | *single turn; chosen rating > rejected rating* |
| Persona LoRA | `data/persona_lora/ramsay_style_preferences.json` | `[200]` / `[20]` | *seeded split* |

- Fixed evaluation sets: `eval_sets/hh_rlhf_eval_300_seed42.json` (300), `eval_sets/alpaca_eval_300_seed42.json`
  (300, seed 42; state the sampling rule from `docs/DATA_PROVENANCE.md`), `eval_sets/persona_eval_prompts.json` (40).
- Prompt format: *quote the template from `alignment/prompt_format.py`; state that it is identical for SFT, DPO and
  generation; the persona track uses the SmolLM2 chat template.*
- SFT loss reduction: *mean over unmasked response tokens.*

## 2. Experimental configuration (about 1 page)

*Paste the "Experimental configuration" table from `outputs/report/report_tables.md` (one row each for SFT, HH DPO,
UltraFeedback DPO, LoRA-DPO) and check it against each `run_config.json`. It must show the values actually used:*
trainable parameters (full fine-tuning, or LoRA rank / alpha / dropout / target modules), optimizer, peak learning
rate, scheduler, warmup, weight decay, gradient clipping, per-device batch size, accumulation, effective batch size,
precision, maximum sequence length, epochs or optimizer steps, seed, DPO beta and the exact reference checkpoint.

Generation settings used for evaluation (from the `*.meta.json` files):

| Setting | Value |
|---|---|
| Decoding | `[greedy]` |
| Temperature / top-p | `[not used with greedy decoding]` |
| Max new tokens | `[256]` |
| Repetition penalty / no-repeat n-gram | `[1.0 / 0]` |
| Length penalty | `[not used]` |
| Chat template | `[project template for Qwen runs; SmolLM2 chat template for persona]` |
| Post-processing | `[cut at the next template turn marker]` |

## 3. Parameter choices and deviations (about 0.75 page)

*One row per change from the guide configuration. Start from `docs/DESIGN_NOTES.md`, section 6, and add anything you
changed while running. Separate resource-driven from performance-driven changes and give the evidence.*

| Setting | Guide value | Final value | Kind (resource / performance) | Evidence and effect |
|---|---|---|---|---|
| `[...]` | `[...]` | `[...]` | `[...]` | `[pilot run, memory limit, validation metric, ...]` |

*LoRA justification (required): rank, target modules, learning rate, batch size, epochs. Use the reasoning in
`docs/DESIGN_NOTES.md`, section 5, plus what you observed.*

## 4. Training behaviour (about 1.25 pages)

- Figure 1: `outputs/plots/sft_curves.png` *(training loss, EMA-smoothed with the stated alpha, and validation loss;
  selected checkpoint marked).*
- Figure 2: `outputs/plots/dpo_hh_curves.png`; Figure 3: `outputs/plots/dpo_ultrafeedback_curves.png` *(DPO loss,
  reward accuracy, reward margin).* Optionally `persona_lora_curves.png`.

| Run | Selected step | Validation metric at selection | Final-step metric | Optimizer steps applied / attempted | Skipped: steps / non-finite micro-batches |
|---|---|---|---|---|---|
| SFT | `[best_step]` | `val_loss = [...]` | `[...]` | `[optimizer_steps]` / `[attempted_steps]` | `[skipped_steps]` / `[skipped_nonfinite]` |
| HH DPO | `[...]` | `reward accuracy = [...]`, `margin = [...]` | `[...]` | `[...]` | `[...]` |
| UltraFeedback DPO | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` |
| LoRA-DPO | `[best_step]` | `eval_loss = [best_metrics]` | `[final_metrics, or "not evaluated at the last step"]` | `[optimizer_steps]` | n/a |

*For LoRA-DPO the submitted adapter is the checkpoint with the lowest validation loss (`best_step` in
`TRAINING_COMPLETE.json`), which need not be the last training step (`final_step`). `final_metrics` is filled only
if an evaluation ran exactly at the last step; do not report the selected checkpoint's metrics as "final".
If a run was resumed, say so and quote its `resume` record from `metrics.jsonl` (records removed, state of the
best checkpoint).*

*Write 4 to 6 sentences: did the losses converge; any instability or overfitting; the step-0 DPO validation
(reward accuracy `[...]`, tie fraction `[...]`) and why accuracy is 0 at a tied start; why the selected checkpoint was
chosen. Do not infer response quality from training loss.*

## 5. DPO-versus-SFT results (about 1.25 pages)

*Paste the "Pairwise judgments" table from `outputs/report/report_tables.md`. Required columns: sample count, wins,
ties, losses, unknown, failures, win rate with and without ties, judge model.*

| Comparison | Eval set | Pairs | Wins | Losses | Ties | Unknown | Failures | Win rate excl. ties (95% CI) | Win rate, ties as half | Judge |
|---|---|---|---|---|---|---|---|---|---|---|
| UltraFeedback DPO vs SFT | AlpacaEval 300 | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` |
| HH DPO vs SFT | helpful/harmless 300 | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` |

*State: same prompts, same decoding, same judge and prompt for both models; A/B order randomised with seed 42;
judge saw only "Response A/B"; request attempts and retries (`request_attempts`, `failed_request_attempts`);
how invalid judgments were handled (unknown = no usable verdict; failure = no reply after 3 attempts; both excluded
from win rates and reported).* *Add the per-category table for the helpful/harmless set.*

**Persona (reported separately; do not combine with the tables above):**

| Comparison | Profile | Pairs | Wins | Losses | Ties | Unknown | Failures | Win rate excl. ties |
|---|---|---|---|---|---|---|---|---|
| Persona LoRA vs SmolLM2-Instruct | style faithfulness | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` |
| Persona LoRA vs SmolLM2-Instruct | usefulness and safety | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` |

## 6. Interpretation and error analysis (about 2 pages)

*Answer explicitly, for each evaluation set: which model was better, by how much, and why you think so. Mark what is
measured and what is interpretation.*

- *Relate the outcome to the preference data, the SFT baseline, beta, training length (469 optimizer steps at
  1e-6), reward accuracy and margin, and visible behaviour.*
- *Is the effect uniform or concentrated? Use the per-category table (refusals, privacy, medical, benign
  helpfulness, ...), the refusal-marker rate and mean output length from the "Generation files" table.*
- *Position bias check: `position_statistics` in each summary (how often A was chosen).*
- *Examples (quote prompt id, both outputs shortened, verdict and judge reason from `*_judged.jsonl`):*
  - *2 where DPO improves on SFT;*
  - *2 where DPO regresses;*
  - *1 where little changes.*
- *Persona: did style become more faithful without loss of usefulness or safety? Use the two persona comparisons and
  the four prompt categories (cooking, coding, general, safety).*
- *Candidate explanations to test against your evidence: preference-data mismatch, noisy pairs, over-optimisation,
  too little training, a strong SFT baseline, decoding sensitivity.*

## 7. Limitations and reproducibility (about 0.75 page)

- *Judge-model bias and a single judge; 300-prompt evaluation sets and the width of the confidence intervals; one
  seed; any failed or repeated runs; compute limits.*
- *The AlpacaEval subset rule is this project's own reading of "300 prompts, seed 42".*
- Paths: configs `configs/*.yaml`; run records `results/<run>/{run_config.json,loop_config.json,metrics.jsonl}`;
  data provenance `results/data/*_provenance.json`; generations `results/eval/generations/*.json`;
  judged outputs `results/eval/judged/*`; tables `results/report/report_tables.md`.
- Commands: `docs/KAGGLE_RUN_GUIDE.md`.

## References

1. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
2. Ouyang et al., *Training language models to follow instructions with human feedback*, NeurIPS 2022.
3. Bai et al., *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*, 2022.
4. Dubois et al., *Length-Controlled AlpacaEval*, 2024.
5. Ding et al., *Enhancing Chat Language Models by Scaling High-quality Instructional Conversations* (UltraChat), 2023.
6. Cui et al., *UltraFeedback: Boosting Language Models with Scaled AI Feedback*, 2023/2024.
7. Hugging Face Transformers, TRL and PEFT documentation; Kaggle `kaggle-benchmarks` documentation.
