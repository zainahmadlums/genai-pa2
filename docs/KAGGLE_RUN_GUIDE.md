# Running the project on Kaggle

Three notebooks, one per phase, each restartable:

| Notebook | Needs | Produces |
|---|---|---|
| `notebooks/01_train_kaggle.ipynb` | GPU T4 x2, Internet | `outputs/{sft,dpo_hh,dpo_ultrafeedback}/{best,final}`, `outputs/persona_lora/adapter`, metrics, plots |
| `notebooks/02_generate_kaggle.ipynb` | GPU, Internet, output of 01 | `outputs/eval/generations/*.json`, `outputs/eval/pairs/*.jsonl` |
| `notebooks/03_judge_kaggle.ipynb` | Kaggle benchmark notebook (model access), output of 02 | `outputs/eval/judged/*` |

Budget about 6.5 to 8.5 GPU hours for notebook 01, 0.5 to 1 hour for notebook 02, and 0.5 to 1 hour (no GPU) for
notebook 03. Nothing here has been run on Kaggle yet; treat the first pass as a test and read the smoke outputs.

## 0. Put the code where Kaggle can reach it (once)

**Option A: GitHub (the assignment's expected route).**

```bash
cd genaiPA2
git init
git add .
git commit -m "Small-model alignment: SFT, DPO, persona LoRA, evaluation"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`.gitignore` already excludes `.venv/` and `outputs/`. Then set `REPO_URL` in the first code cell
of each notebook. A private repository needs a token in the URL
(`https://<token>@github.com/<user>/<repo>.git`, stored as a Kaggle secret); a public one needs nothing.

**Option B: upload as a Kaggle dataset.** Zip the project folder without `.venv` and `outputs`,
create a dataset from it (Kaggle -> Datasets -> New Dataset), attach it to the notebook, and set
`CODE_DATASET_DIR` in the first code cell to the folder that contains `submission_adapter.py`
(for example `/kaggle/input/<dataset-name>/genaiPA2`). The first cell also finds an attached copy automatically
when `REPO_URL` is left unchanged.

## 1. Notebook 01: training

1. Kaggle -> **Create -> New Notebook -> File -> Import Notebook** -> upload `notebooks/01_train_kaggle.ipynb`.
2. Right panel, **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
3. Edit `REPO_URL` (or `CODE_DATASET_DIR`) in the first code cell.
4. Run sections 1 to 6 interactively. Expected:
   - `nvidia-smi` lists two Tesla T4 with about 15 GiB each; `check_env.py` prints the torch, transformers and
     datasets versions and exits with an error if fewer than two GPUs are visible;
   - `pytest` passes (about 110 tests, 2 to 4 minutes; two of them train a random 2-layer model on CPU to check
     the real scripts and the TRL/PEFT versions that were just installed);
   - `prepare_data.py` writes `outputs/data/{sft,hh,uf}_{train,val}.jsonl` and `*_provenance.json`. If it stops
     with "Only N examples survive filtering", rerun that task with `--allow_truncated_responses` or smaller
     `--n_train`, and report the change.
5. Run the smoke cells (`--smoke`: 4 optimizer steps, checkpoints removed afterwards). For DPO look at the line
   `eval step 0/...`: with identical policy and reference it should show `val_reward_accuracy=0.0000`,
   `val_reward_margin=0.0000` and `val_tie_fraction=1.0000`. A tie fraction below 1 with a margin of about 0 means
   the two GPUs produced slightly different floating-point results; note it in the report
   (see `docs/DESIGN_NOTES.md`, section 4).
6. Start the full run with **Save Version -> Save & Run All (Commit)**. It runs in the background for the whole
   notebook; the Output tab of the saved version then contains `small-model-alignment/outputs/...` and the zip files.

What the full commands are (already in the notebook):

```bash
python scripts/prepare_data.py --task all
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

**Splitting across sessions.** Set `STAGES` in section 5 (for example `["sft"]`), commit, then create the next
version with the remaining stages and attach the earlier version through **Add Input -> Notebooks**. `finished()`
finds completed runs inside attached inputs, the DPO runs load the SFT checkpoint straight from there, and notebook
02 accepts several attached versions.

**If something stops.**

| Situation | What to do |
|---|---|
| Kernel restarted, session still alive | Run all cells again. Finished stages are skipped; the interrupted stage continues from `/kaggle/temp/resume/<run>` (saved every 250 / 100 optimizer steps). |
| Session ended | Finished stages survive only in a saved version. Attach it as input and rerun; the interrupted stage starts from its beginning. |
| Out of GPU memory in DPO | `--override reference_dtype=fp16`, then `single_forward=false`. Report it as a resource-driven change. |
| `skipped_nonfinite` above 0 or NaN losses under fp16 | Rerun the stage with `--override precision=fp32 output_dir=outputs/<run>_fp32` (slower) and report it. |
| Disk nearly full | `du -sh outputs/*`; delete `outputs/*_smoke` and any leftover `outputs/*/resume`. Do not zip weights. |

## 2. Notebook 02: generation and pair files

1. Import `notebooks/02_generate_kaggle.ipynb` as a new notebook. Accelerator **GPU T4 x2**, Internet **On**.
2. **Add Input -> Notebooks ->** the saved version(s) of notebook 01. They appear under
   `/kaggle/input/<notebook-slug>/small-model-alignment/outputs/...`; the notebook locates them itself.
3. Edit `REPO_URL` in the first code cell (or leave it: the code is then copied from the attached training output).
4. Run section 4 (5 prompts per model) and read the printed outputs: correct checkpoint, sensible answers, answers
   that end. Then run section 5 for the full sets and section 6 for the pair files.
5. Save the version with its output.

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

Decoding is greedy with `--max_new_tokens 256`. If outputs repeat, set the same anti-repetition flag for all models
in `GEN_FLAGS`, set `OVERWRITE = True`, regenerate, rebuild the pairs, and document it.

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
   `notebooks/03_judge_kaggle.ipynb`, or paste its cells. No accelerator is needed.
2. **Add Input -> Notebooks ->** the saved version of notebook 02. The code is copied from that input, so no
   Internet access or GitHub URL is required in this notebook.
3. Run section 3. It prints the models your notebook exposes. Keep `JUDGE_MODEL = "google/gemini-3.6-flash"` if it
   is available, otherwise choose a listed model and name it in the report.
4. Run section 4: one batch of five pairs. A preflight request runs first; if the model is unavailable the cell
   stops with the provider's error instead of retrying sixty batches.
5. Run section 5 for the full comparisons, then section 6 for the summaries. Save the version.

Per comparison the judge writes `<name>_judged.jsonl`, `<name>_judged.failures.jsonl`,
`<name>_judged.attempts.jsonl` and `<name>_judged.summary.json`. Rerunning a cell skips finished pairs and retries
failed ones. To continue in a new session, attach the earlier version of notebook 03 as input: section 2 copies its
judgment files back before judging.

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
verdicts, the cell stops before sending anything and lists what it found. Then either

- keep the old results and write the new ones elsewhere (another `--output_path` / `JUDGED_DIR`), or
- set `INVALIDATE_INCOMPATIBLE = True` in section 3 (`--invalidate_incompatible`): the old judgment, failure,
  attempt and summary files are renamed to `*.stale1`, `*.stale2`, ... and every pair is judged again.

Results from different judges are never combined, and a summary always describes only judgments made by the judge
it names (`reused_judgments` and `judged_this_run` show how many came from earlier runs of that same judge).

**Unverified parts of this phase.** The judge adapter was written from the published `kaggle-benchmarks` 0.6.1
source and could not be run outside Kaggle. Not confirmed: that `google/gemini-3.6-flash` is offered to your
account, that a plain (non-benchmark) notebook exposes the model proxy, and how Kaggle's quota counts these calls.
If `kaggle_benchmarks` is importable but a call fails in an unexpected way, try `BACKEND = "openai_proxy"`, and
keep `*_judged.attempts.jsonl`: it stores the raw error of every attempt.

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
