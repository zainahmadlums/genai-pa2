# Design notes

Choices made in this implementation, with the reason for each. Section 6 lists every place where the code departs
from, or adds to, the guide configuration in the manual; copy those into the report's "deviations" section.

## 1. Prompt format

```text
### Instruction:
{instruction}

### Response:
{response}<|endoftext|>
```

- `Qwen/Qwen2.5-0.5B` is a base model, not a chat model. Its tokenizer has `eos_token = pad_token = <|endoftext|>`
  and no BOS token. A plain-text template with the model's real EOS needs no new special-token behaviour, and the
  untuned base model can be prompted with exactly the same text for the baseline.
- No system prompt. The same template is used for SFT, both DPO runs and generation. The template name is stored in
  every checkpoint (`alignment_meta.json`); `train_dpo.py` refuses a template that differs from the SFT checkpoint's
  and `generate.py` reads it automatically.
- Prompt and response are tokenised separately and concatenated, so the prompt/response boundary in the labels is exact.
- `chatml` is available as an alternative template (`--override template=chatml` for SFT). It was not chosen because
  the base model was not trained as a chat model.
- The persona track uses SmolLM2-Instruct's own chat template (applied by TRL in training and by
  `tokenizer.apply_chat_template` in generation), including that template's default system prompt.

## 2. Batches and losses

- Sequence = prompt ids + response ids + EOS, then right-truncated to `max_length`. EOS is lost when the response is
  cut, as the manual specifies.
- Labels are `-100` on prompt and padding positions. Masks come from sequence lengths, never from
  `input_ids == pad_id`, because PAD and EOS share one id in Qwen.
- Logits at position `t` are scored against the label at `t + 1`. A response token at position 0 (empty prompt) has
  no context and is not scored.
- **SFT reduction: mean NLL over all unmasked response tokens** of the micro-batch (`token_mean`). With gradient
  accumulation the optimizer step uses the mean of the micro-batch losses in the group.
- DPO uses summed response log-probabilities and `-logsigmoid(beta * (policy_gap - reference_gap))`, averaged over
  pairs. Reference log-probabilities are computed under `torch.no_grad()` with the reference in eval mode and are
  detached again inside the loss.
- The adapter functions pad to `max_length`; the training collators pad to the longest row of the batch. Both call
  the same `build_lm_batch`, and padding does not change any score.
- Vocabulary normalisation (`log_softmax`) is always done in fp32, also when the forward pass ran in fp16.

## 3. Training loop (`alignment/engine.py`)

- Gradient accumulation: each micro-batch loss is divided by the number of micro-batches actually in its group, so
  the short final group (30,000 / 2 = 15,000 micro-batches = 937 full groups of 16 plus one of 8) is still an average.
- Precision: fp16 autocast on CUDA with fp32 weights and `GradScaler`; `scaler.unscale_` runs before
  `clip_grad_norm_`, so the clip threshold applies to true gradient norms. The logged `grad_norm` is the norm before
  clipping.
- Learning rate: linear warmup then cosine decay for SFT (29 warmup steps of 938); constant for DPO.
- Weight decay is applied to weight matrices only (parameters with at least two dimensions), not to biases or
  normalisation weights.
- A micro-batch with a non-finite loss is skipped and counted in `skipped_nonfinite` in `metrics.jsonl`; a non-zero
  value must be reported.
- Attempted and applied optimizer steps are counted separately. `step` is the attempted step (one per accumulation
  group; it fixes the data position, the evaluation cadence and the end of training). An update is not applied when
  the group has no finite loss or its gradients are not finite: under fp16 the `GradScaler` skips it and lowers its
  scale, and without a scaler the loop itself refuses to apply it. Such a step increases `skipped_steps`, does not
  advance the learning-rate schedule, and adds nothing to the averaged gradient norm (a logging window with no
  applied step records `grad_norm: null`, and `loss: null` if it had no finite loss). The run summary reports
  `optimizer_steps` (applied), `attempted_steps` and `skipped_steps`; report a non-zero `skipped_steps`.
- Validation runs at step 0 (logged as `initial`, never selected as best), every `eval_every_steps` optimizer steps
  and at the end. `metrics.jsonl` is appended and fsynced after every record.
- Only `best/` and `final/` are saved (fp32 safetensors plus tokenizer and `alignment_meta.json`), written to a
  temporary directory and renamed.
- `TRAINING_COMPLETE.json` marks a finished run; rerunning a finished stage is a no-op unless `--force` is given.
- Optional mid-run resume state (`--override resume_every_steps=N resume_dir=...`) holds model, optimizer, scheduler,
  scaler and counters. Data order is a seeded permutation per epoch, so a resumed run sees the same batches. A test
  checks that an interrupted and resumed run ends with the same weights as an uninterrupted one.
- Resuming rolls back to the snapshot, which can be older than the last lines of `metrics.jsonl`. Records written
  after the snapshot step are removed (atomic rewrite) before training continues, so no step appears twice, and a
  `{"type": "resume", ...}` record notes the step, how many records were removed and the state of `best/`.
  `best/` may have been replaced after the snapshot; in that case the checkpoint on disk (real weights with their
  own stored validation metrics) is adopted as the current best and marked `adopted_from_disk`. On a GPU the
  re-trained steps are not bit-identical to the lost ones, so such a best checkpoint can come from the earlier
  session's trajectory; the resume record makes that visible.

## 4. DPO specifics (`alignment/dpo_trainer.py`)

- Policy: `outputs/sft/best`, fp32, on `cuda:0`. Reference: a second copy of the same checkpoint on `cuda:1`,
  parameters frozen. With one GPU or on CPU both share the device.
- The reference is loaded in **fp32 and run under the same fp16 autocast as the policy** (`reference_dtype: fp32`).
  The manual allows an fp16 reference; fp32 weights were chosen because cuda:1 has spare memory and the two models
  then go through the same arithmetic, which keeps the step-0 rewards tied. `reference_dtype: fp16` halves the
  reference's memory if needed. Whether the step-0 tie is exact on real GPUs has not been observed yet: read
  `val_tie_fraction` at step 0 in `metrics.jsonl` and report it.
- Chosen and rejected rows go through the model in one forward pass (`single_forward: true`); a test checks that
  this equals two separate passes.
- Reference log-probabilities of the validation set are computed once and cached, since the reference never changes.
- Validation metrics: DPO loss, reward accuracy (strict `r_w > r_l`), reward margin, tie fraction and mean
  log-probabilities. The best checkpoint has the highest validation reward accuracy; reward margin breaks ties.
- Optimizer: AdamW (weight decay 0) by default; `--override optimizer=rmsprop` selects RMSprop.

## 5. Persona LoRA (TRL `DPOTrainer`) - choices to justify in the report

| Setting | Value | Reason |
|---|---|---|
| Rank / alpha | r = 16, alpha = 32 | A style shift needs little capacity; r = 16 trains about 8.7M of 362M parameters (about 2.4%). alpha = 2r is the usual scaling. |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Persona is mostly word choice and tone, which the MLP projections influence as well as attention. |
| Dropout | 0.05 | Only 200 training pairs; light regularisation. |
| Learning rate | 1e-4, cosine, 10% warmup | LoRA adapters start at zero and need a much larger rate than the 1e-6 used for full fine-tuning. |
| Batch | 4 x 2 accumulation = 8 | 25 optimizer steps per epoch on 200 pairs. |
| Epochs | 3 (75 steps) | Enough passes for 200 pairs; the adapter with the lowest validation DPO loss is kept. |
| Beta | 0.1 | Same as the from-scratch runs. |
| Reference | base model with the adapter disabled | TRL's behaviour when `peft_config` is given and no `ref_model`. |

These are starting values chosen from the model architecture and data size, not tuned results. The parameter count
is an estimate from the layer shapes; the script prints the real count. If the smoke outputs show the persona is too
weak or the model stops being useful, change rank, learning rate or epochs and report the comparison.
The script pins `CUDA_VISIBLE_DEVICES=0` so the Hugging Face Trainer does not wrap the model in `DataParallel`, and
sets `fp16=True, bf16=False` because TRL 1.x would otherwise default to bf16.

## 6. Departures from and additions to the guide configuration

| Item | Guide | Here | Kind |
|---|---|---|---|
| `transformers` / `datasets` lower bounds | `>=4.43`, `>=2.18` | `>=4.56.2,<5`, `>=4.7,<5` | required by `trl==1.14.1` (not in the starter requirements) |
| `bitsandbytes` | in starter requirements | removed | unused: no quantisation anywhere |
| SFT prompt cap | not specified | prompts over 256 tokens filtered out | keeps room for the response in 512 tokens |
| Truncated targets | allowed | examples that do not fully fit in 512 tokens are filtered out | every example then supervises EOS |
| HH data mix | not specified | 50% harmless-base, 50% helpful subsets (target) | the track is about safety and refusals |
| DPO weight decay | not specified | 0.0 | manual lists none for DPO |
| DPO reference dtype | fp16 allowed | fp32 weights under fp16 autocast | see section 4 |
| Weight decay scope | not specified | matrices only | standard practice |
| Mid-run resume state | not required | optional, off by default | about 6 GB per save for the 0.5B model |
| Generation post-processing | not specified | output cut at the first `\n### Instruction:` / `\n### Response:` | same rule for every model; a base model otherwise continues with invented turns |
| Judge rubric hint | not specified | the eval set's `expected_behavior` text is shown to the judge | part of the provided evaluation set; `--no_expected_behavior` removes it |

Everything else follows the manual's tables (sizes, batch sizes, accumulation, learning rates, warmup, schedule,
clipping, beta, context length, epochs).

## 7. Evaluation pipeline

- `generate.py`: greedy decoding by default, left padding, 256 new tokens, the checkpoint's template. `--smoke N`
  writes a separate `*.smoke.json` and prints the outputs. Output files are rewritten after every batch.
- Generation reuse: `<output>.meta.json` records a provenance block (generator label, dataset label, prompt template,
  decoding settings except batch size, precision, and a content fingerprint of the checkpoint and adapter
  directories; hub models are identified by id). It is written before the first batch. A rerun reuses saved answers
  only when the whole block is identical; otherwise, or when the block is missing, it stops without generating and
  without touching the files, and names the fields that differ. Replacing outputs needs `--overwrite`. The checkpoint
  fingerprint covers each file's name, size and the first and last megabyte, so retraining into the same path is
  detected while copying or re-attaching the same checkpoint is not flagged; a same-size file that differs only in
  its middle would not be noticed.
- Template turn markers that are single special tokens (`<|im_end|>`, `<|im_start|>` for `chatml` with the Qwen
  tokenizer) are handled as token ids: generation stops at them and the answer is cut at the first one before
  decoding, because decoding with `skip_special_tokens` removes them from the text. The default `alpaca` markers
  are ordinary text and are cut after decoding.
- `build_pairs.py`: balanced, seeded A/B assignment; identities only in `hidden`. It reads the `.meta.json` of both
  generation files and refuses to pair them unless both are complete and share template, decoding settings,
  precision and post-processing (checkpoint identity is allowed to differ). `--allow_unverified_inputs` overrides
  this for legacy or hand-made files and records `inputs_verified: false` in the pairs meta file.
- `judge.py`: the prompt contains only the instruction, the optional expected-behaviour hint and the two responses
  labelled A and B, with neutral item numbers. A stored fingerprint of each pair prevents reusing a judgment after
  its responses changed.
- Judgment reuse: every judgment, attempt and failure record stores the backend, the judge model and a
  `protocol_id` (hash of backend, judge model, batch size, response clipping length, rubric text and protocol
  version). Saved results are reused only when all records carry the current `protocol_id`. Anything else,
  including mock verdicts, another judge model and files written before this tracking existed, stops the run before
  any request is sent. `--invalidate_incompatible` renames the four result files to `*.stale<N>` and judges every
  pair again; results of different judges are never mixed in one file or summary. `--sleep_seconds` and
  `--max_retries` are not part of the protocol. Each summary reports `reused_judgments` and `judged_this_run`.
- Matching replies to pairs: an explicit item number in the judge's reply is always used. A number that appears
  twice, an unreadable number or an entry without a number in an otherwise numbered reply gives that pair no verdict
  from this reply, so it is asked again. List position is used only when no entry has a number and the reply has
  exactly one entry per pair.
- Pacing: one pacer per process spaces the end of each request and the start of the next by at least
  `--sleep_seconds` (minimum 5 for real backends). It covers the availability check, first attempts, retries and
  consecutive comparisons started from the same notebook kernel. The time of the last request is also kept in
  `<judged dir>/.judge_last_request`, so a separate process writing to the same folder waits as well.
- Outcomes: `candidate_win`, `baseline_win`, `tie` (valid verdicts); `unknown` (the judge answered but not with A, B
  or tie after all attempts); `failure` (no judgment after all attempts). Summaries give counts, rates over all pairs
  (these sum to 1), rates over valid judgments, the win rate excluding ties with a 95% Wilson interval, the win rate
  counting ties as half, a two-sided sign test, A/B position statistics and per-category tables.
- Judge backends: `kaggle_benchmarks` (`kbench.llms[name]` or `kbench.kaggle.load_model(name)`, `llm.prompt(text)`
  inside `kbench.chats.new(..., orphan=True)`) and an OpenAI-compatible fallback on `$MODEL_PROXY_URL/openapi`.
  Both were written from the `kaggle-benchmarks` 0.6.1 source and **have not been run against the real service**.

## 8. Disk budget on Kaggle (`/kaggle/working`, 20 GB)

| Item | Size (approx.) |
|---|---|
| One fp32 checkpoint of Qwen2.5-0.5B | 2 GB |
| SFT + two DPO runs, best + final each | 12 GB |
| Temporary copy while a `best/` is replaced | 2 GB |
| Prepared data, metrics, plots, zips without weights | under 0.3 GB |
| Resume state (model + AdamW moments) | 6 GB, twice that while being replaced; keep it in `/kaggle/temp` |

Smoke runs delete their own checkpoints. The sizes are estimates from the parameter count (494M parameters x 4
bytes); check `du -sh outputs/*` in the notebook.
