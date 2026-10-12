"""Regenerate the three Kaggle notebooks in notebooks/ from the cell lists below.

    python tools/build_notebooks.py

The notebooks are stored without outputs. Edit the cells here, not in the .ipynb files.

Every notebook is meant to be run with "Save & Run All": no cell needs editing, commands are run as
argument lists through ``run`` / ``py`` (no shell), and a failing command raises and stops the run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/zainahmadlums/genai-pa2.git"
REPO_BRANCH = "main"
REPO_NAME = "small-model-alignment"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def setup_code(prefer_attached_code: bool) -> str:
    """First code cell of every notebook: the ``run`` / ``py`` helpers and the copy of the code."""

    text = '''
import os, shlex, shutil, subprocess, sys
from pathlib import Path

# ---- Settings: the defaults work as they are, nothing has to be edited ----------------------
REPO_URL = "<<REPO_URL>>"
REPO_BRANCH = "<<REPO_BRANCH>>"
CODE_DATASET_DIR = ""   # optional: use an uploaded copy of the code, e.g. "/kaggle/input/my-pa2-code/genaiPA2"
PREFER_ATTACHED_CODE = <<PREFER_ATTACHED_CODE>>   # True: take the code from an attached notebook output instead of GitHub
# ---------------------------------------------------------------------------------------------

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORK / "<<REPO_NAME>>"
CODE_MARKER = "scripts/judge_pairwise_kaggle.py"   # present only in a complete copy of the code


def run(*argv, **popen_options):
    """Run one command given as separate arguments (no shell, so spaces in paths are safe), output shown live.

    A non-zero exit status raises CalledProcessError, which stops "Run All" at this cell.
    """
    argv = [str(arg) for arg in argv]
    print("$", shlex.join(argv), flush=True)
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", **popen_options) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, argv)


def py(*argv, **popen_options):
    """Run a script or module with this notebook's Python interpreter."""
    run(sys.executable, *argv, **popen_options)


def find_code_in_inputs():
    """A copy of the code inside an attached input (an uploaded dataset or an earlier notebook's output)."""
    matches = sorted(INPUT.glob("**/" + CODE_MARKER))
    return matches[0].parents[1] if matches else None


if (REPO_DIR / CODE_MARKER).exists():
    print("using the code already at", REPO_DIR, "as it is (an existing copy is never updated)")
elif REPO_DIR.exists():
    raise RuntimeError(f"{REPO_DIR} exists but is not a complete copy of the code. Delete it (or restart the "
                       "session) and run this cell again.")
else:
    if CODE_DATASET_DIR:
        source = Path(CODE_DATASET_DIR)
        if not (source / CODE_MARKER).exists():
            raise FileNotFoundError(f"CODE_DATASET_DIR={CODE_DATASET_DIR!r} does not contain {CODE_MARKER}")
    else:
        source = find_code_in_inputs() if PREFER_ATTACHED_CODE else None
    if source is not None:
        shutil.copytree(source, REPO_DIR, ignore=shutil.ignore_patterns("outputs", ".git", ".venv", "__pycache__"))
        print("copied the code from", source)
    else:
        if PREFER_ATTACHED_CODE:
            print("no copy of the code in the attached inputs; cloning it instead (needs Internet On)")
        run("git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR)

os.chdir(REPO_DIR)
os.environ["PYTHONUNBUFFERED"] = "1"   # training / generation logs appear line by line
sys.path.insert(0, str(REPO_DIR))
print("working directory:", os.getcwd())
'''
    for name, value in (("REPO_URL", REPO_URL), ("REPO_BRANCH", REPO_BRANCH), ("REPO_NAME", REPO_NAME),
                        ("PREFER_ATTACHED_CODE", str(prefer_attached_code))):
        text = text.replace(f"<<{name}>>", value)
    return text


INSTALL_CODE = '''
# Keeps Kaggle's CUDA build of PyTorch: torch is never installed or pinned here.
py("-m", "pip", "uninstall", "-y", "-q", "torchao")
py("-m", "pip", "install", "-q", "transformers>=4.56.2,<5", "datasets>=4.7,<5", "huggingface_hub",
   "accelerate>=1.4.0", "tqdm", "pyyaml", "sentencepiece", "safetensors", "matplotlib", "peft>=0.17.0",
   "trl==1.14.1", "pytest")
py("-m", "pip", "install", "-q", "-e", ".", "--no-deps")
'''

FIND_RUNS_CODE = '''
RUNS = {"sft": "best", "dpo_hh": "best", "dpo_ultrafeedback": "best", "persona_lora": "adapter"}   # run -> checkpoint


def finished(run_name):
    """Directory of a finished run: outputs/<run_name> in this session, else inside any attached input."""
    local = Path("outputs") / run_name
    if (local / "TRAINING_COMPLETE.json").exists():
        return local
    matches = sorted(INPUT.glob(f"**/outputs/{run_name}/TRAINING_COMPLETE.json"))
    return matches[-1].parent if matches else None


def checkpoint(run_name):
    """Checkpoint directory of a finished run. Raises when the run did not finish or its weights are missing."""
    run_dir = finished(run_name)
    if run_dir is None:
        raise FileNotFoundError(f"run '{run_name}' is not finished: no outputs/{run_name}/TRAINING_COMPLETE.json in "
                                "this session or in an attached input (Add Input -> Notebooks -> the saved version "
                                "of notebook 01 that completed it)")
    path = run_dir / RUNS[run_name]
    if not path.is_dir():
        raise FileNotFoundError(f"run '{run_name}' is marked finished but its checkpoint {path} is missing")
    return path


for name in RUNS:
    print(f"{name:18s} {finished(name) or 'not finished yet'}")
'''

DISK_USE_CODE = '''
def size_gb(path):
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    return sum(item.stat().st_size for item in files) / 1e9

for path in sorted(Path("outputs").glob("*")) + sorted(WORK.glob("*.zip")):
    print(f"{size_gb(path):8.2f} GB  {path}")
usage = shutil.disk_usage(WORK)
print(f"{usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB at {WORK}")
'''

# =============================================================================================
# 01 - training
# =============================================================================================
train_cells = [
    md(f'''
# 01 - Training (SFT, two from-scratch DPO runs, LoRA persona)

Run this notebook first. Nothing has to be edited.

1. **Import:** Kaggle -> Create -> New Notebook -> File -> Import Notebook -> upload this file.
2. **Settings** (right panel, Session options): Accelerator **GPU T4 x2**, Internet **On**.
3. **Inputs:** none.
4. **Run:** **Save Version -> Save & Run All (Commit)**. It runs in the background; no open browser tab is needed.
5. **Afterwards:** send the status of the version (Complete or Failed) and its log; the last cell prints the
   summary tables. Then go on to notebook 02.

The notebook clones `{REPO_URL}` (branch `{REPO_BRANCH}`), installs the dependencies (PyTorch is left untouched),
checks for two GPUs, runs the offline tests, prepares the data, trains the four runs and writes plots and tables.
A step that fails raises a Python error and stops the run there; the error is the last thing in the log.

Expected time on T4 x2: SFT about 1.5 h, each DPO run 2 to 2.5 h, persona 10 to 20 min (6.5 to 8.5 h in total,
inside the 12 h session limit). To split the work across sessions, set `STAGES` in section 5 and attach the earlier
version's output through **Add Input**: finished stages found there are skipped.

Everything is written under `/kaggle/working/{REPO_NAME}/outputs/<run_name>/`.
'''),
    md("## 1. Get the code"),
    code(setup_code(prefer_attached_code=False)),
    md("## 2. Install dependencies (PyTorch is left untouched)"),
    code(INSTALL_CODE),
    md("## 3. Check the environment: two T4 GPUs and package versions"),
    code('py("scripts/check_env.py", "--require_gpus", "2")'),
    md("## 4. Offline tests (public adapter contract and training math; no downloads)"),
    code('py("-m", "pytest", "-q", "tests", env={**os.environ, "PYTHONPATH": str(REPO_DIR)})'),
    md('''
## 5. Choose stages and look for finished runs

`finished(run)` finds a completed run in this session or in any attached input, so a later session never repeats
a stage that already finished. `checkpoint(run)` returns its weights and raises if the run is not complete.
'''),
    code('''
STAGES = ["sft", "dpo_hh", "dpo_ultrafeedback", "persona_lora"]   # remove entries to run only part of the pipeline
RUN_SMOKE = False   # optional: True adds a 4-step smoke run before each full run (not needed for Save & Run All)
'''),
    code(FIND_RUNS_CODE + '''
unknown = [name for name in STAGES if name not in RUNS]
if unknown:
    raise ValueError(f"unknown entries in STAGES: {unknown}; choose from {list(RUNS)}")


def stage_needed(name):
    return name in STAGES and finished(name) is None
'''),
    md('''
## 6. Prepare data

Builds the single-turn files in `outputs/data/` (30k/1k SFT, 15k/1k HH, 15k/1k UltraFeedback) from pinned dataset
revisions and writes a `*_provenance.json` next to each. Prompts of the fixed evaluation sets are excluded.
Existing files are kept.
'''),
    code('''
py("scripts/make_alpaca_eval_subset.py", "--verify")
for stage, task in (("sft", "sft"), ("dpo_hh", "hh"), ("dpo_ultrafeedback", "ultrafeedback")):
    if stage_needed(stage):
        py("scripts/prepare_data.py", "--task", task)
for path in sorted(Path("outputs/data").glob("*")):
    print(f"{path.stat().st_size:>13,d}  {path}")
'''),
    md('''
## 7. SFT

The resume state goes to `/kaggle/temp`, which does not count against the 20 GB output quota and survives a kernel
restart in the same session.
'''),
    code('''
if stage_needed("sft"):
    if RUN_SMOKE:
        py("scripts/train_sft.py", "--config", "configs/sft.yaml", "--smoke")
    py("scripts/train_sft.py", "--config", "configs/sft.yaml", "--resume",
       "--override", "resume_every_steps=250", "resume_dir=/kaggle/temp/resume/sft")
    py("scripts/package_outputs.py", "--source", "outputs/sft", "outputs/data",
       "--zip_path", WORK / "sft_run_outputs.zip")
    print("SFT best checkpoint:", checkpoint("sft"))   # raises if the run did not finish
'''),
    md('''
## 8. DPO on HH-RLHF

Policy on `cuda:0`, frozen reference on `cuda:1`, both loaded from the SFT best checkpoint.
The step-0 validation must show reward accuracy 0.0 and margin 0.0: policy and reference are identical, so all
rewards tie and the strict comparison counts none as correct.
'''),
    code('''
if stage_needed("dpo_hh"):
    SFT_BEST = checkpoint("sft")   # raises unless SFT finished above or in an attached input
    if RUN_SMOKE:
        py("scripts/train_dpo.py", "--config", "configs/dpo_hh.yaml", "--smoke",
           "--override", f"sft_checkpoint={SFT_BEST}")
    py("scripts/train_dpo.py", "--config", "configs/dpo_hh.yaml", "--resume",
       "--override", f"sft_checkpoint={SFT_BEST}", "resume_every_steps=100", "resume_dir=/kaggle/temp/resume/dpo_hh")
    py("scripts/package_outputs.py", "--source", "outputs/dpo_hh", "--zip_path", WORK / "dpo_hh_run_outputs.zip")
    print("DPO (HH) best checkpoint:", checkpoint("dpo_hh"))
'''),
    md("## 9. DPO on UltraFeedback (same code, same SFT checkpoint, different data)"),
    code('''
if stage_needed("dpo_ultrafeedback"):
    SFT_BEST = checkpoint("sft")
    py("scripts/train_dpo.py", "--config", "configs/dpo_ultrafeedback.yaml", "--resume",
       "--override", f"sft_checkpoint={SFT_BEST}", "resume_every_steps=100",
       "resume_dir=/kaggle/temp/resume/dpo_ultrafeedback")
    py("scripts/package_outputs.py", "--source", "outputs/dpo_ultrafeedback",
       "--zip_path", WORK / "dpo_ultrafeedback_run_outputs.zip")
    print("DPO (UltraFeedback) best checkpoint:", checkpoint("dpo_ultrafeedback"))
'''),
    md("## 10. Persona: TRL DPOTrainer + LoRA on SmolLM2-360M-Instruct"),
    code('''
if stage_needed("persona_lora"):
    if RUN_SMOKE:
        py("scripts/train_persona_lora.py", "--config", "configs/persona_lora.yaml", "--smoke")
    py("scripts/train_persona_lora.py", "--config", "configs/persona_lora.yaml")
    py("scripts/package_outputs.py", "--source", "outputs/persona_lora",
       "--zip_path", WORK / "persona_lora_run_outputs.zip", "--include_weights")
    print("persona adapter:", checkpoint("persona_lora"))
'''),
    md('''
## 11. Training plots and summary tables

Every stage in `STAGES` must be finished with its checkpoint in place by now, otherwise this cell raises. The plots
and tables cover every finished run, including runs found in attached inputs.
'''),
    code('''
for name in STAGES:
    print(f"{name:18s} {checkpoint(name)}")
RUN_DIRS = [finished(name) for name in RUNS if finished(name) is not None]
py("scripts/plot_metrics.py", "--run_dir", *RUN_DIRS, "--out_dir", "outputs/plots")
py("scripts/summarize_results.py", "--runs", *RUN_DIRS, "--out_dir", "outputs/report")
py("scripts/package_outputs.py", "--source", "outputs/plots", "outputs/report",
   "--zip_path", WORK / "training_plots_and_report.zip")
'''),
    code('''
from IPython.display import Image, display
for path in sorted(Path("outputs/plots").glob("*_curves.png")):
    print(path)
    display(Image(filename=str(path)))
'''),
    md('''
## 12. Disk use and summary

The three Qwen runs hold about 12 GB of checkpoints (best + final, fp32); the output limit is 20 GB.
The zip files contain metrics, configs, logs and plots only (no weights), so they are small and easy to download.

With **Save & Run All (Commit)** the outputs are attached to the saved version. Open notebook 02 next and attach
this notebook with **Add Input -> Notebooks**.

**Restarting:** run all cells again. A finished stage is skipped. An interrupted stage continues from
`/kaggle/temp/resume/<run>` while the session is alive and restarts from its beginning in a new session.
'''),
    code(DISK_USE_CODE + '''
print(Path("outputs/report/report_tables.md").read_text(encoding="utf-8"))
print("NOTEBOOK 01 FINISHED. Stages complete:", ", ".join(STAGES))
'''),
]

# =============================================================================================
# 02 - generation
# =============================================================================================
GEN_FLAGS = ["--max_new_tokens", "256", "--batch_size", "16"]
GEN_DIR = "outputs/eval/generations"
PAIRS_DIR = "outputs/eval/pairs"

generate_cells = [
    md(f'''
# 02 - Generation and pairwise files

Run this notebook after notebook 01 has completed. Nothing has to be edited.

1. **Import:** Kaggle -> Create -> New Notebook -> File -> Import Notebook -> upload this file.
2. **Settings** (right panel, Session options): Accelerator **GPU T4 x2** (one GPU is used), Internet **On**.
3. **Inputs:** **Add Input -> Notebooks ->** the saved, completed version of notebook 01 (several versions may be
   attached when the stages were split across sessions).
4. **Run:** **Save Version -> Save & Run All (Commit)**.
5. **Afterwards:** send the status of the version and its log; the last cell prints the first answers of every
   model, the generation table and the pair files. Then go on to notebook 03.

The notebook clones `{REPO_URL}` (branch `{REPO_BRANCH}`), reads the four checkpoints from the attached input and
writes `{GEN_DIR}/*.json` and `{PAIRS_DIR}/*.jsonl`. Decoding is greedy with identical settings for every model.
All four finished runs are required: a missing checkpoint or a failing command raises and stops the run.
Finished generation files are reused on restart.
'''),
    md("## 1. Get the code and install"),
    code(setup_code(prefer_attached_code=False)),
    code(INSTALL_CODE),
    code('py("scripts/check_env.py", "--require_gpus", "1")'),
    md("## 2. Locate the checkpoints from notebook 01 (all four are required)"),
    code(FIND_RUNS_CODE + '''
from alignment.io_utils import read_json

SFT = checkpoint("sft")
DPO_HH = checkpoint("dpo_hh")
DPO_UF = checkpoint("dpo_ultrafeedback")
PERSONA = checkpoint("persona_lora")
# The base model is prompted with the same template the SFT model was trained with.
TEMPLATE = read_json(SFT / "alignment_meta.json")["template"]
print("SFT:", SFT, "\\nDPO_HH:", DPO_HH, "\\nDPO_UF:", DPO_UF, "\\nPERSONA:", PERSONA, "\\ntemplate:", TEMPLATE)
'''),
    md('''
## 3. Decoding settings (the same for every model in a comparison)

Greedy decoding, 256 new tokens. If the outputs loop, add an anti-repetition flag for **all** models as two more
entries of the `GEN_FLAGS` list below, for example `"--repetition_penalty", "1.1"` or
`"--no_repeat_ngram_size", "4"`, set `OVERWRITE = True` to regenerate, and record the change in the report.

A saved generation file is reused only for the same checkpoint contents, template and decoding flags. After any
such change the command stops and names the difference instead of reusing old answers; `OVERWRITE = True`
regenerates the files.
'''),
    code(f'''
OVERWRITE = False
RUN_SMOKE = False   # optional: True first runs every model on 5 prompts and prints the answers
GEN_FLAGS = {json.dumps(GEN_FLAGS)} + (["--overwrite"] if OVERWRITE else [])   # one list entry per argument
GEN_DIR = "{GEN_DIR}"
ALPACA = "eval_sets/alpaca_eval_300_seed42.json"
HH = "eval_sets/hh_rlhf_eval_300_seed42.json"
PERSONA_PROMPTS = "eval_sets/persona_eval_prompts.json"
BASE = "Qwen/Qwen2.5-0.5B"
SMOL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def generate(model, prompts, generator, out_name, *extra, smoke=False):
    py("scripts/generate.py", "--model_path", model, "--prompts_path", prompts, "--generator", generator,
       "--output_path", f"{{GEN_DIR}}/{{out_name}}", *extra, *GEN_FLAGS, *(["--smoke", "5"] if smoke else []))


def generate_all(smoke=False):
    generate(BASE, ALPACA, "qwen2.5-0.5b-base", "alpaca_base.json", "--template", TEMPLATE, smoke=smoke)
    generate(SFT, ALPACA, "qwen2.5-0.5b-sft", "alpaca_sft.json", smoke=smoke)
    generate(BASE, HH, "qwen2.5-0.5b-base", "hh_base.json", "--template", TEMPLATE, smoke=smoke)
    generate(SFT, HH, "qwen2.5-0.5b-sft", "hh_sft.json", smoke=smoke)
    generate(DPO_UF, ALPACA, "qwen2.5-0.5b-dpo-ultrafeedback", "alpaca_dpo_uf.json", smoke=smoke)
    generate(DPO_HH, HH, "qwen2.5-0.5b-dpo-hh", "hh_dpo_hh.json", smoke=smoke)
    generate(SMOL, PERSONA_PROMPTS, "smollm2-360m-instruct-base", "persona_base.json",
             "--template", "chat_template", smoke=smoke)
    generate(SMOL, PERSONA_PROMPTS, "smollm2-360m-instruct-persona-lora", "persona_lora.json",
             "--adapter_path", PERSONA, "--template", "chat_template", smoke=smoke)
'''),
    md('''
## 4. Optional sanity check: 5 prompts per model

Off by default. With `RUN_SMOKE = True` it writes `*.smoke.json` files only and prints the answers.
'''),
    code('''
if RUN_SMOKE:
    generate_all(smoke=True)
'''),
    md('''
## 5. Full generation

AlpacaEval subset (300 prompts, seed 42): base, SFT, UltraFeedback-DPO. Helpful/harmless set (300 prompts): base,
SFT, HH-DPO. Persona prompts (40): original SmolLM2-Instruct and the LoRA persona model.
'''),
    code('generate_all()'),
    md('''
## 6. Pairwise files (written before any judging)

Response order is randomised with seed 42 (balanced A/B); model identities are stored only in each row's `hidden`
block. The persona pair is built twice: once to judge style faithfulness and once to check that usefulness and
safety held.

Pairing stops with an explanation if one of the two generation files is unfinished or if they were generated with
different template, decoding flags or precision: regenerate the affected file (`OVERWRITE = True`) and rerun.
'''),
    code(f'''
PAIRS_DIR = "{PAIRS_DIR}"


def pair(baseline, candidate, comparison, profile):
    py("scripts/build_pairs.py", "--baseline_path", f"{{GEN_DIR}}/{{baseline}}",
       "--candidate_path", f"{{GEN_DIR}}/{{candidate}}", "--output_path", f"{{PAIRS_DIR}}/{{comparison}}_pairs.jsonl",
       "--comparison", comparison, "--profile", profile)


pair("alpaca_sft.json", "alpaca_dpo_uf.json", "dpo_uf_vs_sft", "general")
pair("hh_sft.json", "hh_dpo_hh.json", "dpo_hh_vs_sft", "safety")
pair("persona_base.json", "persona_lora.json", "persona_vs_base_style", "persona")
pair("persona_base.json", "persona_lora.json", "persona_vs_base_quality", "safety")
# Optional, for your own analysis (not a required comparison): SFT against the untuned base model.
pair("alpaca_base.json", "alpaca_sft.json", "sft_vs_base", "general")
'''),
    md("## 7. Package and summary"),
    code('''
py("scripts/summarize_results.py", "--runs", "--eval_dir", "outputs/eval", "--out_dir", "outputs/report")
py("scripts/package_outputs.py", "--source", "outputs/eval", "outputs/report",
   "--zip_path", WORK / "eval_generations_and_pairs.zip")
for path in sorted(Path(GEN_DIR).glob("*")) + sorted(Path(PAIRS_DIR).glob("*")):
    print(f"{path.stat().st_size:>13,d}  {path}")
# The first two answers of every model, to see at a glance that they are sensible and that they end.
for path in sorted(Path(GEN_DIR).glob("*.json")):
    if not path.name.endswith((".meta.json", ".smoke.json")):
        print("=" * 100)
        for entry in read_json(path)[:2]:
            print(f"{path.name} [{entry['id']}] {entry['instruction'][:200]!r}\\n  -> {entry['output'][:500]!r}")
print(Path("outputs/report/report_tables.md").read_text(encoding="utf-8"))
print("NOTEBOOK 02 FINISHED. Pair files:", sorted(path.name for path in Path(PAIRS_DIR).glob("*_pairs.jsonl")))
'''),
    md('''
With **Save & Run All (Commit)** the outputs are attached to the saved version. Open notebook 03 next and attach
this notebook with **Add Input -> Notebooks**.

**Restart:** run all cells again. A finished generation file is reused and an interrupted one continues from its
last completed batch.
'''),
]

# =============================================================================================
# 03 - judging
# =============================================================================================
judge_cells = [
    md('''
# 03 - Pairwise judging with a Kaggle-hosted model

Run this notebook after notebook 02 has completed. Nothing has to be edited.

1. **Create and import:** open https://www.kaggle.com/benchmarks/tasks/new so that `kaggle_benchmarks` and the
   Kaggle model proxy (`MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`) are available, then **File -> Import Notebook ->**
   upload this file.
2. **Settings:** no accelerator is needed. Internet is not needed for the code: it is copied from the attached input.
3. **Inputs:** **Add Input -> Notebooks ->** the saved, completed version of notebook 02.
4. **Run:** **Save Version -> Save & Run All (Commit)**.
5. **Afterwards:** send the status of the version and its log; the last cells print one line per comparison, the
   first verdicts with their reasons, the judgment tables and whether any pair is unresolved.

Nothing is generated here. The notebook reads the saved pairwise JSONL files, judges 5 pairs per request, waits at
least 5 seconds between all request attempts, tries a failing batch up to 3 times, and appends results after every
batch. Each comparison starts with a one-line availability check, so an unavailable judge model raises before any
pair is sent. A missing pair file raises before the first request.
'''),
    md("## 1. Get the code (copied from the attached notebook 02 output; cloned from GitHub only if none is attached)"),
    code(setup_code(prefer_attached_code=True)),
    code('''
# Judging needs only the standard library plus kaggle_benchmarks (or openai for the proxy fallback).
import importlib
for name in ("kaggle_benchmarks", "openai"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "installed"))
    except Exception as exc:
        print(name, "NOT AVAILABLE:", type(exc).__name__, exc)
'''),
    md("## 2. Collect the pairwise files (and earlier judgments when resuming in a new session)"),
    code('''
PAIRS_DIR = Path("outputs/eval/pairs")
JUDGED_DIR = Path("outputs/eval/judged")
PAIRS_DIR.mkdir(parents=True, exist_ok=True)
JUDGED_DIR.mkdir(parents=True, exist_ok=True)

for source in sorted(INPUT.glob("**/outputs/eval/pairs/*_pairs.jsonl")):
    target = PAIRS_DIR / source.name
    if not target.exists():
        shutil.copyfile(source, target)
# Judgments from an earlier version of THIS notebook (attach that version as input to continue it).
for source in sorted(INPUT.glob("**/outputs/eval/judged/*")):
    target = JUDGED_DIR / source.name
    if source.is_file() and not target.exists():
        shutil.copyfile(source, target)

REQUIRED = ["dpo_uf_vs_sft", "dpo_hh_vs_sft", "persona_vs_base_style", "persona_vs_base_quality"]
print("pairs:", sorted(path.name for path in PAIRS_DIR.glob("*_pairs.jsonl")))
print("existing judgment files:", sorted(path.name for path in JUDGED_DIR.glob("*_judged.jsonl")))
missing = [name for name in REQUIRED if not (PAIRS_DIR / f"{name}_pairs.jsonl").exists()]
if missing:
    raise FileNotFoundError(f"no pair file for {missing}: attach the saved, completed version of notebook 02 "
                            "(Add Input -> Notebooks) and run again")
'''),
    md('''
## 3. Choose the judge model

`google/gemini-3.6-flash` is the model suggested by the assignment. The cell lists what this notebook actually
exposes. If the suggested model is not available (the availability check in section 4 raises), set `JUDGE_MODEL` to
a listed model and name it in the report.
'''),
    code('''
sys.path.insert(0, str(REPO_DIR / "scripts"))
import judge_pairwise_kaggle as judge_cli

JUDGE_MODEL = "google/gemini-3.6-flash"
BACKEND = "auto"   # kaggle_benchmarks first, then the OpenAI-compatible model proxy
# Saved judgments are reused only for the same backend, judge model and batch size. If you change one of them
# after judging has started, the run stops before sending anything. Set this to True to rename the old result
# files to *.stale<N> and judge every pair again with the new settings.
INVALIDATE_INCOMPATIBLE = False
judge_cli.main(["--list_models"])
'''),
    code('''
def judge(comparison):
    """Judge one comparison. A missing pair file or a judge that cannot be used raises instead of being skipped."""
    pairs_path = PAIRS_DIR / f"{comparison}_pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} is missing")
    argv = ["--pairs_path", str(pairs_path),
            "--output_path", str(JUDGED_DIR / f"{comparison}_judged.jsonl"),
            "--judge_model", JUDGE_MODEL, "--backend", BACKEND,
            "--batch_size", "5", "--sleep_seconds", "5", "--max_retries", "3", "--reuse_judgments"]
    if INVALIDATE_INCOMPATIBLE:
        argv += ["--invalidate_incompatible"]
    try:
        return judge_cli.main(argv)
    except SystemExit as exc:
        raise RuntimeError(f"judging {comparison} stopped: {exc}") from exc
'''),
    md('''
## 4. Judging

About 60 requests per 300-pair comparison. If a cell stops (quota, disconnect), run it again: finished pairs are
skipped and failed ones are retried.
'''),
    code('judge("dpo_uf_vs_sft")'),
    code('judge("dpo_hh_vs_sft")'),
    code('''
judge("persona_vs_base_style")
judge("persona_vs_base_quality")
'''),
    code('''
# Optional: SFT vs base (own analysis only).
RUN_OPTIONAL = False
if RUN_OPTIONAL:
    judge("sft_vs_base")
'''),
    md("## 5. Summaries"),
    code('''
import json
from alignment.io_utils import read_jsonl

for path in sorted(JUDGED_DIR.glob("*.summary.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(f"{summary['comparison']}: judge={summary['judge_model']} pairs={summary['total_pairs']} "
          f"wins={summary['candidate_wins']} losses={summary['baseline_wins']} ties={summary['ties']} "
          f"unknown={summary['unknown']} failures={summary['failures']} complete={summary['complete']} "
          f"win_rate_excl_ties={summary['candidate_win_rate_excluding_ties']}")
# The first three verdicts and reasons of every comparison, to see that the judge answers sensibly.
for path in sorted(JUDGED_DIR.glob("*_judged.jsonl")):
    for record in read_jsonl(path)[:3]:
        print(f"  {path.name} {record['prompt_id']} | verdict: {record['verdict']} | winner: "
              f"{record['winner_model']} | {record['reason']}")
'''),
    code('''
py("scripts/summarize_results.py", "--runs", "--eval_dir", "outputs/eval", "--out_dir", "outputs/report")
py("scripts/package_outputs.py", "--source", "outputs/eval/judged", "outputs/report",
   "--zip_path", WORK / "judged_outputs.zip")
print(Path("outputs/report/report_tables.md").read_text(encoding="utf-8"))
'''),
    code('''
# Pairs the judge never resolved are reported here, not raised: the judgments obtained so far have to be saved
# with this version so that a new version can reuse them and retry only the unresolved pairs.
unresolved = {}
for comparison in REQUIRED:
    summary = json.loads((JUDGED_DIR / f"{comparison}_judged.summary.json").read_text(encoding="utf-8"))
    if summary["failures"] or summary["not_attempted"]:
        unresolved[comparison] = {"failures": summary["failures"], "not_attempted": summary["not_attempted"]}
if unresolved:
    print("NOTEBOOK 03 FINISHED WITH UNRESOLVED PAIRS:", unresolved)
    print("To retry only those: attach this saved version to a new version of this notebook (Add Input) and run it.")
else:
    print("NOTEBOOK 03 FINISHED. Every pair of the required comparisons has a judgment:", ", ".join(REQUIRED))
'''),
    md('''
With **Save & Run All (Commit)** the saved version keeps `outputs/eval/judged/*`: judgments, `*.failures.jsonl`,
`*.attempts.jsonl` and `*.summary.json`. Unresolved pairs stay listed in the failures file and are counted
separately in every summary.
'''),
]

NOTEBOOKS = {
    "01_train_kaggle.ipynb": train_cells,
    "02_generate_kaggle.ipynb": generate_cells,
    "03_judge_kaggle.ipynb": judge_cells,
}


def render(cells: list[dict]) -> str:
    """The .ipynb text for ``cells``; the same input always gives the same text."""

    return json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n"


def main(out_dir: Path = ROOT / "notebooks") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = out_dir / name
        # newline="\n": the files are byte-identical whether they are generated on Windows or Linux.
        path.write_text(render(cells), encoding="utf-8", newline="\n")
        print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
