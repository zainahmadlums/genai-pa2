# Running the project on Kaggle

Three notebooks, one per phase. Each one is imported as it is and run with **Save & Run All**; no cell has to be
edited and there are no manual test or smoke steps.

| Notebook | Settings | Attach as input | Produces |
|---|---|---|---|
| `notebooks/01_train_kaggle.ipynb` | GPU T4 x2, Internet On | nothing | `outputs/{sft,dpo_hh,dpo_ultrafeedback}/{best,final}`, `outputs/persona_lora/adapter`, metrics, plots |
| `notebooks/02_generate_kaggle.ipynb` | GPU T4 x2, Internet On | saved version of 01 | `outputs/eval/generations/*.json`, `outputs/eval/pairs/*.jsonl` |
| `notebooks/03_judge_kaggle.ipynb` | Kaggle benchmark notebook, no GPU | saved version of 02 | `outputs/eval/judged/*` |

Budget about 6.5 to 8.5 GPU hours for notebook 01, 0.5 to 1 hour for notebook 02, and 0.5 to 1 hour (no GPU) for
notebook 03.

**Only the beginning of notebook 01 has been run on Kaggle** (setup, tests, data preparation and the first 100 SFT
optimizer steps; see "First Kaggle run" in section 1). Everything after that point, and notebooks 02 and 03, were
checked offline only (see section 5). Treat the first run of each notebook as the test: send its status and log
before starting the next one.

## The same five steps for every notebook

1. **Upload:** import the `.ipynb` file into a new Kaggle notebook (**File -> Import Notebook**).
2. **Settings:** right panel, **Session options** (accelerator and Internet as in the table).
3. **Attach the previous saved output:** **Add Input -> Notebooks ->** the saved, completed version of the previous
   notebook. Notebook 01 needs no input.
4. **Run:** **Save Version -> Save & Run All (Commit)**. It runs in the background; no open browser tab is needed.
5. **Send the result:** when the version has finished, send its status (Complete or Failed) and its log (version
   page -> **Logs**). The last cell of each notebook prints the summary tables and a line starting with
   `NOTEBOOK 0N FINISHED`. If the version failed, the Python error of the step that stopped it is at the end of
   the log.

Every command in the notebooks is run as an argument list and checked: a failing install, environment check, test,
data step, training, generation or packaging command raises an error and stops the run at that cell. A run that
needs a checkpoint or a pair file that is not there stops as well instead of skipping the step.

## 0. Where the code comes from

Notebooks 01 and 02 clone <https://github.com/zainahmadlums/genai-pa2> (branch `main`) into
`/kaggle/working/small-model-alignment`. The URL is already in the first code cell.

- Kaggle sees what is **pushed** to `main` at the moment the notebook runs. Commit and push before a run; the
  notebook file you upload should come from the same commit.
- The clone uses no credentials, so the repository has to be publicly readable. For a private repository upload
  the project folder (without `.venv` and `outputs`) as a Kaggle dataset, attach it, and set `CODE_DATASET_DIR` in
  the first code cell to the folder that contains `submission_adapter.py`
  (for example `/kaggle/input/<dataset-name>/genaiPA2`). `CODE_DATASET_DIR` takes precedence in all three notebooks.
- Notebook 03 does not clone: it copies the code from the attached output of notebook 02, so it needs neither
  Internet access nor GitHub. It clones only if no attached input contains the code.
- A copy of the code that is already in `/kaggle/working/small-model-alignment` is used as it is and never
  updated. To pick up a newer commit in an interactive session, restart the session or delete that folder.

## 1. Notebook 01: training

1. Kaggle -> **Create -> New Notebook -> File -> Import Notebook** -> upload `notebooks/01_train_kaggle.ipynb`.
2. **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
3. No input.
4. **Save Version -> Save & Run All (Commit)**.
5. Send the status and the log.

What the run does, in order:

- installs the dependencies (PyTorch is never installed or pinned; Kaggle's CUDA build is kept);
- `check_env.py --require_gpus 2` prints the torch, transformers and datasets versions and `nvidia-smi`, and stops
  the run if fewer than two GPUs are visible;
- `pytest tests` once (about 200 tests, 2 to 4 minutes on CPU; it includes the public adapter tests, the training
  math, and runs of the real scripts on a random 2-layer model with the TRL/PEFT versions just installed);
- `prepare_data.py` writes `outputs/data/{sft,hh,uf}_{train,val}.jsonl` and `*_provenance.json`. If it stops
  with "Only N examples survive filtering", add `--allow_truncated_responses` or a smaller `--n_train` to that
  command and report the change;
- trains SFT, DPO on HH-RLHF, DPO on UltraFeedback and the persona LoRA. After each stage the notebook checks that
  the run finished and that its checkpoint directory exists;
- writes plots and tables, then prints disk use and `outputs/report/report_tables.md`.

In the DPO logs look at the line `eval step 0/...`: with identical policy and reference it should show
`val_reward_accuracy=0.0000`, `val_reward_margin=0.0000` and `val_tie_fraction=1.0000`. A tie fraction below 1
with a margin of about 0 means the two GPUs produced slightly different floating-point results; note it in the
report (see `docs/DESIGN_NOTES.md`, section 4).

The commands the notebook runs:

```bash
python scripts/prepare_data.py --task sft          # and --task hh, --task ultrafeedback
python scripts/train_sft.py --config configs/sft.yaml --resume \
    --override resume_every_steps=250 resume_dir=/kaggle/temp/resume/sft
python scripts/train_dpo.py --config configs/dpo_hh.yaml --resume \
    --override sft_checkpoint=<sft>/best resume_every_steps=100 resume_dir=/kaggle/temp/resume/dpo_hh
python scripts/train_dpo.py --config configs/dpo_ultrafeedback.yaml --resume \
    --override sft_checkpoint=<sft>/best resume_every_steps=100 resume_dir=/kaggle/temp/resume/dpo_ultrafeedback
python scripts/train_persona_lora.py --config configs/persona_lora.yaml
python scripts/plot_metrics.py --run_dir <run dirs> --out_dir outputs/plots
python scripts/summarize_results.py --runs <run dirs> --out_dir outputs/report
```

**Optional settings** (section 5 of the notebook; the defaults are the full run):

- `STAGES`: to split the work across sessions, keep only some stages (for example `["sft"]`), commit, then create
  the next version with the remaining stages and attach the earlier version through **Add Input -> Notebooks**.
  `finished()` finds completed runs inside attached inputs, the DPO runs load the SFT checkpoint straight from
  there, and notebook 02 accepts several attached versions. A DPO stage without a finished SFT run stops with an
  error.
- `RUN_SMOKE = True` adds a 4-step smoke run (`--smoke`, checkpoints removed afterwards) before each full run. It
  is off by default and not needed.

**First Kaggle run: out of memory at the first periodic SFT evaluation (fixed in the code).** On T4 x2 the
setup, the tests, data preparation, the SFT evaluation at step 0 and SFT steps 1 to 100 ran; the evaluation at
step 100 then failed with `CUDA out of memory` on GPU 0 inside `SFTTask.evaluate`.

- Cause: the step-0 evaluation runs with only the weights on the GPU. From the first update on, the GPU also
  holds AdamW's two moment tensors per weight and, until this fix, the gradients of the step that had just
  finished. On top of that, validation scored 8 rows at once, and the logits over Qwen's vocabulary of about
  152k tokens are normalised in fp32.
- Fix: the training loop releases the gradients as soon as an optimizer step is finished (the update itself is
  unchanged), and a validation batch is now as large as a training micro-batch: `eval_batch_size: 2` rows for
  SFT and `eval_batch_size: 1` pair for both DPO configs, which had the same 8 rows per forward pass and would
  have failed the same way. Validation therefore needs less memory than the training steps that already ran.
  Validation data, counts, cadence, metrics and checkpoint selection are unchanged; the metrics do not depend
  on the batch size (`tests/test_eval_memory_and_batch_split.py`). Each evaluation takes somewhat longer.
- Restart: push the fix to `main`, open the same Kaggle notebook and choose **Save Version -> Save & Run All
  (Commit)** again. No cell is edited and `01_train_kaggle.ipynb` itself did not change, so it does not have to
  be uploaded again. The new version clones the pushed code and starts from the beginning (every version
  runs in a fresh session, so nothing of the failed one is reused).

**If something stops.**

| Situation | What to do |
|---|---|
| Kernel restarted, session still alive | Run all cells again. Finished stages are skipped; the interrupted stage continues from `/kaggle/temp/resume/<run>` (saved every 250 / 100 optimizer steps). |
| Session ended | Finished stages survive only in a saved version. Attach it as input and rerun; the interrupted stage starts from its beginning. |
| Out of GPU memory in DPO | Add `"reference_dtype=fp16"` to the `--override` arguments of that stage, then `"single_forward=false"`. Report it as a resource-driven change. |
| `skipped_nonfinite` above 0 or NaN losses under fp16 | Rerun the stage with `"precision=fp32"` and `"output_dir=outputs/<run>_fp32"` added to `--override` (slower) and report it. |
| Disk nearly full | The last cell prints the size of every folder under `outputs/`. Delete `outputs/*_smoke` and any leftover `outputs/*/resume`. Do not zip weights. |

## 2. Notebook 02: generation and pair files

1. Import `notebooks/02_generate_kaggle.ipynb` as a new notebook.
2. **Session options**: Accelerator **GPU T4 x2** (one GPU is used), Internet **On**.
3. **Add Input -> Notebooks ->** the saved, completed version(s) of notebook 01. They appear under
   `/kaggle/input/<notebook-slug>/small-model-alignment/outputs/...`; the notebook locates them itself.
4. **Save Version -> Save & Run All (Commit)**.
5. Send the status and the log. The last cell prints the first two answers of every model, the generation table
   (outputs, empty outputs, mean length) and the pair files: check there that the answers are sensible and end.

All four finished runs (`sft`, `dpo_hh`, `dpo_ultrafeedback`, `persona_lora`) are required. If one is missing from
the attached inputs, or its checkpoint directory is missing, the run stops before anything is generated and names
the run.

Files produced:

```text
outputs/eval/generations/alpaca_{base,sft,dpo_uf}.json      300 prompts each (AlpacaEval subset, seed 42)
outputs/eval/generations/hh_{base,sft,dpo_hh}.json          300 prompts each (helpful/harmless set)
outputs/eval/generations/persona_{base,lora}.json           40 prompts each
outputs/eval/generations/*.meta.json                        decoding settings and counts per file
outputs/eval/pairs/dpo_uf_vs_sft_pairs.jsonl                profile general
outputs/eval/pairs/dpo_hh_vs_sft_pairs.jsonl                profile safety
outputs/eval/pairs/persona_vs_base_style_pairs.jsonl        profile persona
outputs/eval/pairs/persona_vs_base_quality_pairs.jsonl      profile safety (usefulness and safety check)
outputs/eval/pairs/sft_vs_base_pairs.jsonl                  optional, own analysis
```

Decoding is greedy with `--max_new_tokens 256`. If outputs repeat, add the same anti-repetition flag for all models
as two more entries of the `GEN_FLAGS` list (for example `"--repetition_penalty", "1.1"`), set `OVERWRITE = True`,
run again, and document it. `RUN_SMOKE = True` first runs every model on 5 prompts (`*.smoke.json`); it is off by
default.

A generation file is tied to what produced it. If you rerun with a different checkpoint, adapter, prompt template,
precision or decoding flag (anything except `--batch_size`), the command stops with a message such as
`decoding.max_new_tokens: 8 -> 256` and leaves the existing file alone. Either set `OVERWRITE = True`
(`--overwrite`) to regenerate that file completely, or write to another `--output_path`. Files produced by an
earlier version of this code have no provenance record and are refused the same way.

`build_pairs.py` applies the matching check when two files are paired: both must be finished (`status: complete`
in their `.meta.json`) and share template, decoding flags and precision. If, say, the SFT file is still the old
greedy one and the DPO file was generated with `--repetition_penalty 1.1`, pairing stops and prints
`decoding.repetition_penalty: 1.0 -> 1.1`; regenerate the older side with `OVERWRITE = True` and pair again.
`--allow_unverified_inputs` exists only for hand-made or legacy files and marks the pairs file as unverified.

## 3. Notebook 03: judging

1. Open <https://www.kaggle.com/benchmarks/tasks/new>. Kaggle creates a notebook with `kaggle_benchmarks`
   preinstalled and model access configured. In it use **File -> Import Notebook** with
   `notebooks/03_judge_kaggle.ipynb`.
2. No accelerator is needed. The code is copied from the attached input, so no Internet access or GitHub URL is
   required for it.
3. **Add Input -> Notebooks ->** the saved, completed version of notebook 02.
4. **Save Version -> Save & Run All (Commit)**.
5. Send the status and the log. The last cells print one line per comparison, the judgment tables and either
   `NOTEBOOK 03 FINISHED. Every pair ...` or `NOTEBOOK 03 FINISHED WITH UNRESOLVED PAIRS: ...`.

What the run does:

- stops before the first request if one of the four required pair files is missing;
- lists the models the notebook exposes. `JUDGE_MODEL = "google/gemini-3.6-flash"` is the default; if it is not
  offered, set `JUDGE_MODEL` to a listed model and name it in the report;
- judges the four required comparisons. Each one starts with a one-line availability request; if the model is
  unavailable the run stops there with the provider's error instead of retrying sixty batches. There is no
  separate sample step: the summary cell prints the first three verdicts and reasons of every comparison;
- pairs that the judge never resolved (after 3 attempts per batch) do **not** fail the run: they are counted in
  every summary and listed by the last cell, so the judgments obtained so far are saved with the version.

Per comparison the judge writes `<name>_judged.jsonl`, `<name>_judged.failures.jsonl`,
`<name>_judged.attempts.jsonl` and `<name>_judged.summary.json`. Running again skips finished pairs and retries
failed ones. To continue in a new version, attach the earlier saved version of notebook 03 as input in addition to
notebook 02: section 2 copies its judgment files back before judging.

Equivalent command line:

```bash
python scripts/judge_pairwise_kaggle.py --list_models
python scripts/judge_pairwise_kaggle.py \
    --pairs_path outputs/eval/pairs/dpo_hh_vs_sft_pairs.jsonl \
    --output_path outputs/eval/judged/dpo_hh_vs_sft_judged.jsonl \
    --judge_model google/gemini-3.6-flash \
    --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

`--sleep_seconds` below 5 is raised to 5 and applies between all requests of the kernel: the availability check,
retries and the next comparison included. `--retry_unknown` re-judges pairs whose verdict was not A, B or tie.
`--backend mock` runs the same pipeline without any model call (its verdicts are not results).

Saved judgments belong to one backend, one judge model and one protocol (batch size, rubric, response clipping).
If you change `JUDGE_MODEL`, `BACKEND` or the batch size after some pairs were judged, or the folder holds mock
verdicts, the run stops before sending anything and lists what it found. Then either

- keep the old results and write the new ones elsewhere (another `--output_path` / `JUDGED_DIR`), or
- set `INVALIDATE_INCOMPATIBLE = True` in section 3 (`--invalidate_incompatible`): the old judgment, failure,
  attempt and summary files are renamed to `*.stale1`, `*.stale2`, ... and every pair is judged again.

Results from different judges are never combined, and a summary always describes only judgments made by the judge
it names (`reused_judgments` and `judged_this_run` show how many came from earlier runs of that same judge).

**Unverified parts of this phase.** The judge adapter was written from the published `kaggle-benchmarks` 0.6.1
source and could not be run outside Kaggle. Not confirmed: that `google/gemini-3.6-flash` is offered to your
account, that the model proxy is available to a **Save & Run All** run of a benchmark notebook (and to a plain,
non-benchmark notebook), and how Kaggle's quota counts these calls. If `kaggle_benchmarks` is importable but a call
fails in an unexpected way, try `BACKEND = "openai_proxy"`, and keep `*_judged.attempts.jsonl`: it stores the raw
error of every attempt.

## 4. Collecting material for the report

After notebook 03, download from the three saved versions:

- `outputs/plots/*_curves.png` and `*_eval_table.csv` (notebook 01);
- `outputs/report/report_tables.md` from notebook 03 (judgment tables) and from notebook 01 (configuration and
  training tables), or run `scripts/summarize_results.py` once with all outputs in place;
- `outputs/data/*_provenance.json`, `outputs/*/run_config.json`, `outputs/*/metrics.jsonl`;
- `outputs/eval/generations/*.json` and `outputs/eval/judged/*` for examples and error analysis.

Copy the small files (run configs, metrics, plots, provenance, generations, judged outputs) into a tracked folder
such as `results/` and commit them, so the report can point at them. `outputs/` itself is git-ignored and the model
weights stay on Kaggle. Then fill in `docs/REPORT_TEMPLATE.md`.

## 5. What was checked offline, and what was not

`tests/test_kaggle_notebooks.py` executes the code cells of all three notebooks with temporary folders in place of
`/kaggle/working` and `/kaggle/input`:

- the notebooks are exactly what `tools/build_notebooks.py` writes (regenerate with
  `python tools/build_notebooks.py`; edit the cells there, not in the `.ipynb` files);
- notebooks 01 and 02 with their commands recorded instead of run: the order and arguments of every command,
  paths with spaces, the required checkpoints, and that the first failing command stops the run;
- notebook 03 for real with `--backend mock`: code copied from the attached input, pair files collected, four
  comparisons judged, tables and zip written.

Not checked: any GPU run, the downloads, the real training and generation commands as launched from a notebook,
the Kaggle judge, and Kaggle's own behaviour (import, **Save & Run All**, attaching outputs, what is kept from a
version that failed). The first full run of each notebook is the first real test of it.
