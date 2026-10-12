"""Regenerate the three Kaggle notebooks in notebooks/ from the cell lists below.

    python tools/build_notebooks.py

The notebooks are stored without outputs. Edit the cells here, not in the .ipynb files.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


SETUP_CODE = f'''
import os, shutil, subprocess, sys
from pathlib import Path

# ---- EDIT THESE ---------------------------------------------------------------------------
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"   # your GitHub repository
REPO_BRANCH = "main"
CODE_DATASET_DIR = ""   # alternative to GitHub: an uploaded copy, e.g. "/kaggle/input/my-pa2-code/genaiPA2"
# -------------------------------------------------------------------------------------------

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "{REPO_NAME}"

def find_code_in_inputs():
    """A copy of the repository inside an attached input (an uploaded dataset or an earlier notebook's output)."""
    for marker in sorted(Path("/kaggle/input").glob("**/scripts/judge_pairwise_kaggle.py")):
        return marker.parents[1]
    return None

if REPO_DIR.exists():
    print("repository already present:", REPO_DIR)
else:
    source = Path(CODE_DATASET_DIR) if CODE_DATASET_DIR else None
    if source is None and "YOUR_USERNAME" in REPO_URL:
        source = find_code_in_inputs()
        if source is None:
            raise SystemExit("Set REPO_URL to your GitHub repository, or set CODE_DATASET_DIR, or attach an input "
                             "that contains the code.")
    if source is not None:
        shutil.copytree(source, REPO_DIR, ignore=shutil.ignore_patterns("outputs", ".git", ".venv", "__pycache__"))
        print("copied code from", source)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)], check=True)

os.chdir(REPO_DIR)
os.environ["PYTHONUNBUFFERED"] = "1"   # show training / generation logs live in "!python ..." cells
sys.path.insert(0, str(REPO_DIR))
print("working directory:", os.getcwd())
'''

INSTALL_CODE = '''
# Keeps Kaggle's CUDA build of PyTorch: torch is never installed or pinned here.
!python -m pip uninstall -y -q torchao || true
!python -m pip install -q "transformers>=4.56.2,<5" "datasets>=4.7,<5" huggingface_hub "accelerate>=1.4.0" tqdm pyyaml sentencepiece safetensors matplotlib "peft>=0.17.0" "trl==1.14.1" pytest
!python -m pip install -q -e . --no-deps
'''

FIND_RUNS_CODE = '''
from pathlib import Path

def finished(run_name):
    """Directory of a finished run: outputs/<run_name> in this session, else inside any attached input."""
    local = Path("outputs") / run_name
    if (local / "TRAINING_COMPLETE.json").exists():
        return local
    matches = sorted(Path("/kaggle/input").glob(f"**/outputs/{run_name}/TRAINING_COMPLETE.json"))
    return matches[-1].parent if matches else None

for name in ("sft", "dpo_hh", "dpo_ultrafeedback", "persona_lora"):
    print(f"{name:18s} {finished(name) or 'not finished yet'}")
'''

# =============================================================================================
# 01 - training
# =============================================================================================
train_cells = [
    md('''
# 01 - Training (SFT, two from-scratch DPO runs, LoRA persona)

**Kaggle settings before running:** Accelerator **GPU T4 x2**, Internet **On**.

Recommended way to run:

1. Interactively run sections 1 to 5 (setup, tests, data) and the smoke cells of each stage to see that everything works.
2. Then use **Save Version -> Save & Run All (Commit)**. The commit runs the whole notebook in the background
   (no open browser tab needed) and keeps everything under `/kaggle/working` as the version's output.

Expected time on T4 x2: SFT about 1.5 h, each DPO run 2 to 2.5 h, persona 10 to 20 min (6.5 to 8.5 h in total,
inside the 12 h session limit). To split the work across sessions, set `STAGES` in section 6 and attach the earlier
version's output through **Add Input**: finished stages found there are skipped.

Everything is written under `/kaggle/working/small-model-alignment/outputs/<run_name>/`.
'''),
    md("## 1. Get the code"),
    code(SETUP_CODE),
    md("## 2. Install dependencies (PyTorch is left untouched)"),
    code(INSTALL_CODE),
    md("## 3. Check the environment: two T4 GPUs and package versions"),
    code('''
!nvidia-smi
!python scripts/check_env.py --require_gpus 2
'''),
    md("## 4. Offline tests (adapter contract and training math; no downloads)"),
    code('''
!PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
!PYTHONPATH=. python -m pytest -q tests
'''),
    md('''
## 5. Choose stages and look for finished runs

`finished(run)` finds a completed run in this session or in any attached input, so a later session never repeats
a stage that already finished.
'''),
    code('''
STAGES = ["sft", "dpo_hh", "dpo_ultrafeedback", "persona_lora"]   # remove entries to run only part of the pipeline
RUN_SMOKE = True                                                  # 4-step check before each full run
'''),
    code(FIND_RUNS_CODE),
    md('''
## 6. Prepare data

Builds the single-turn files in `outputs/data/` (30k/1k SFT, 15k/1k HH, 15k/1k UltraFeedback) from pinned dataset
revisions and writes a `*_provenance.json` next to each. Prompts of the fixed evaluation sets are excluded.
Existing files are kept.
'''),
    code('''
!python scripts/make_alpaca_eval_subset.py --verify
if "sft" in STAGES and finished("sft") is None:
    !python scripts/prepare_data.py --task sft
if "dpo_hh" in STAGES and finished("dpo_hh") is None:
    !python scripts/prepare_data.py --task hh
if "dpo_ultrafeedback" in STAGES and finished("dpo_ultrafeedback") is None:
    !python scripts/prepare_data.py --task ultrafeedback
!ls -la outputs/data
'''),
    md('''
## 7. SFT

A 4-step smoke run first (its checkpoints are deleted automatically), then the real run. The resume state goes to
`/kaggle/temp`, which does not count against the 20 GB output quota and survives a kernel restart in the same session.
'''),
    code('''
if "sft" in STAGES and finished("sft") is None:
    if RUN_SMOKE:
        !python scripts/train_sft.py --config configs/sft.yaml --smoke
    !python scripts/train_sft.py --config configs/sft.yaml --resume --override resume_every_steps=250 resume_dir=/kaggle/temp/resume/sft
    !python scripts/package_outputs.py --source outputs/sft outputs/data --zip_path /kaggle/working/sft_run_outputs.zip
SFT_BEST = finished("sft") / "best" if finished("sft") else None
print("SFT best checkpoint:", SFT_BEST)
'''),
    md('''
## 8. DPO on HH-RLHF

Policy on `cuda:0`, frozen reference on `cuda:1`, both loaded from the SFT best checkpoint.
The step-0 validation must show reward accuracy 0.0 and margin 0.0: policy and reference are identical, so all
rewards tie and the strict comparison counts none as correct.
'''),
    code('''
if "dpo_hh" in STAGES and finished("dpo_hh") is None:
    assert SFT_BEST is not None, "SFT must be finished first: run it above or attach its output with Add Input"
    if RUN_SMOKE:
        !python scripts/train_dpo.py --config configs/dpo_hh.yaml --smoke --override sft_checkpoint=$SFT_BEST
    !python scripts/train_dpo.py --config configs/dpo_hh.yaml --resume --override sft_checkpoint=$SFT_BEST resume_every_steps=100 resume_dir=/kaggle/temp/resume/dpo_hh
    !python scripts/package_outputs.py --source outputs/dpo_hh --zip_path /kaggle/working/dpo_hh_run_outputs.zip
'''),
    md("## 9. DPO on UltraFeedback (same code, same SFT checkpoint, different data)"),
    code('''
if "dpo_ultrafeedback" in STAGES and finished("dpo_ultrafeedback") is None:
    assert SFT_BEST is not None, "SFT must be finished first: run it above or attach its output with Add Input"
    !python scripts/train_dpo.py --config configs/dpo_ultrafeedback.yaml --resume --override sft_checkpoint=$SFT_BEST resume_every_steps=100 resume_dir=/kaggle/temp/resume/dpo_ultrafeedback
    !python scripts/package_outputs.py --source outputs/dpo_ultrafeedback --zip_path /kaggle/working/dpo_ultrafeedback_run_outputs.zip
'''),
    md("## 10. Persona: TRL DPOTrainer + LoRA on SmolLM2-360M-Instruct"),
    code('''
if "persona_lora" in STAGES and finished("persona_lora") is None:
    if RUN_SMOKE:
        !python scripts/train_persona_lora.py --config configs/persona_lora.yaml --smoke
    !python scripts/train_persona_lora.py --config configs/persona_lora.yaml
    !python scripts/package_outputs.py --source outputs/persona_lora --zip_path /kaggle/working/persona_lora_run_outputs.zip --include_weights
'''),
    md("## 11. Training plots and summary tables"),
    code('''
RUN_DIRS = " ".join(str(finished(name)) for name in ("sft", "dpo_hh", "dpo_ultrafeedback", "persona_lora") if finished(name))
if RUN_DIRS:
    !python scripts/plot_metrics.py --run_dir $RUN_DIRS --out_dir outputs/plots
    !python scripts/summarize_results.py --runs $RUN_DIRS --out_dir outputs/report
    !python scripts/package_outputs.py --source outputs/plots outputs/report --zip_path /kaggle/working/training_plots_and_report.zip
'''),
    code('''
from IPython.display import Image, display
for path in sorted(Path("outputs/plots").glob("*_curves.png")):
    print(path)
    display(Image(filename=str(path)))
'''),
    md('''
## 12. Disk use and saving

The three Qwen runs hold about 12 GB of checkpoints (best + final, fp32); the output limit is 20 GB.
The zip files contain metrics, configs, logs and plots only (no weights), so they are small and easy to download.

- If you ran this through **Save & Run All (Commit)**, the outputs are already attached to the saved version.
- If you ran it interactively, use **Save Version -> Quick Save** and enable *Always save output when creating a
  Quick Save* under Advanced Settings.

Then open notebook 02 and attach this notebook with **Add Input -> Notebooks**.

**Restarting:** run all cells again. A finished stage is skipped. An interrupted stage continues from
`/kaggle/temp/resume/<run>` while the session is alive and restarts from its beginning in a new session.
'''),
    code('''
!du -sh outputs/* /kaggle/working/*.zip 2>/dev/null
!df -h /kaggle/working | tail -1
'''),
]

# =============================================================================================
# 02 - generation
# =============================================================================================
GEN_FLAGS = "--max_new_tokens 256 --batch_size 16"
GEN_DIR = "outputs/eval/generations"


def gen(model: str, prompts: str, generator: str, out: str, extra: str = "", indent: str = "    ") -> str:
    parts = [f"{indent}!python scripts/generate.py --model_path {model} --prompts_path {prompts}",
             f"--generator {generator} --output_path {GEN_DIR}/{out}"]
    if extra:
        parts.append(extra)
    parts.append("$GEN_FLAGS $SMOKE")
    return " ".join(parts)


def generation_block(indent: str = "    ") -> str:
    lines = [
        "if SFT:",
        gen("$BASE", "$ALPACA", "qwen2.5-0.5b-base", "alpaca_base.json", "--template $TEMPLATE"),
        gen("$SFT", "$ALPACA", "qwen2.5-0.5b-sft", "alpaca_sft.json"),
        gen("$BASE", "$HH", "qwen2.5-0.5b-base", "hh_base.json", "--template $TEMPLATE"),
        gen("$SFT", "$HH", "qwen2.5-0.5b-sft", "hh_sft.json"),
        "if DPO_UF:",
        gen("$DPO_UF", "$ALPACA", "qwen2.5-0.5b-dpo-ultrafeedback", "alpaca_dpo_uf.json"),
        "if DPO_HH:",
        gen("$DPO_HH", "$HH", "qwen2.5-0.5b-dpo-hh", "hh_dpo_hh.json"),
        "if PERSONA:",
        gen("$SMOL", "$PERSONA_PROMPTS", "smollm2-360m-instruct-base", "persona_base.json",
            "--template chat_template"),
        gen("$SMOL", "$PERSONA_PROMPTS", "smollm2-360m-instruct-persona-lora", "persona_lora.json",
            "--adapter_path $PERSONA --template chat_template"),
    ]
    return "\n".join(lines)


generate_cells = [
    md('''
# 02 - Generation and pairwise files

**Kaggle settings:** Accelerator **GPU T4 x2** (one GPU is used), Internet **On**.
**Add Input -> Notebooks ->** your saved *01 - Training* notebook version (several versions may be attached).

This notebook only reads checkpoints and writes `outputs/eval/generations/*.json` and
`outputs/eval/pairs/*.jsonl`. Decoding is greedy with identical settings for every model.
Section 4 runs each model on 5 prompts and prints the outputs: read them before starting section 5.
Finished files are reused on restart.
'''),
    md("## 1. Get the code and install"),
    code(SETUP_CODE),
    code(INSTALL_CODE),
    code("!python scripts/check_env.py"),
    md("## 2. Locate the checkpoints from the training notebook"),
    code(FIND_RUNS_CODE + '''
from alignment.io_utils import read_json

def checkpoint(run_name, sub):
    run_dir = finished(run_name)
    return str(run_dir / sub) if run_dir else ""

SFT = checkpoint("sft", "best")
DPO_HH = checkpoint("dpo_hh", "best")
DPO_UF = checkpoint("dpo_ultrafeedback", "best")
PERSONA = checkpoint("persona_lora", "adapter")
# The base model is prompted with the same template the SFT model was trained with.
TEMPLATE = read_json(Path(SFT) / "alignment_meta.json")["template"] if SFT else "alpaca"
print("SFT:", SFT, "\\nDPO_HH:", DPO_HH, "\\nDPO_UF:", DPO_UF, "\\nPERSONA:", PERSONA, "\\ntemplate:", TEMPLATE)
'''),
    md('''
## 3. Decoding settings (the same for every model in a comparison)

Greedy decoding, 256 new tokens. If the sanity outputs loop, add an anti-repetition flag for **all** models, for
example `--repetition_penalty 1.1` or `--no_repeat_ngram_size 4`, set `OVERWRITE` below to regenerate, and record
the change in the report.

A saved generation file is reused only for the same checkpoint contents, template and decoding flags. After any
such change the command stops and names the difference instead of reusing old answers; `OVERWRITE = True`
regenerates the files.
'''),
    code(f'''
OVERWRITE = False
# Every $NAME used in the "!" lines below is a Python variable (IPython expands a line only if all of them exist).
GEN_FLAGS = "{GEN_FLAGS}" + (" --overwrite" if OVERWRITE else "")
ALPACA = "eval_sets/alpaca_eval_300_seed42.json"
HH = "eval_sets/hh_rlhf_eval_300_seed42.json"
PERSONA_PROMPTS = "eval_sets/persona_eval_prompts.json"
BASE = "Qwen/Qwen2.5-0.5B"
SMOL = "HuggingFaceTB/SmolLM2-360M-Instruct"
'''),
    md('''
## 4. Sanity check: 5 prompts per model

Writes `*.smoke.json` files only. Check the checkpoint path, the prompt format, that answers stop (EOS) and that
their length is reasonable.
'''),
    code('SMOKE = "--smoke 5"\n' + generation_block()),
    md('''
## 5. Full generation

AlpacaEval subset (300 prompts, seed 42): base, SFT, UltraFeedback-DPO. Helpful/harmless set (300 prompts): base,
SFT, HH-DPO. Persona prompts (40): original SmolLM2-Instruct and the LoRA persona model.
'''),
    code('SMOKE = ""\n' + generation_block()),
    md('''
## 6. Pairwise files (written before any judging)

Response order is randomised with seed 42 (balanced A/B); model identities are stored only in each row's `hidden`
block. The persona pair is built twice: once to judge style faithfulness and once to check that usefulness and
safety held.

Pairing stops with an explanation if one of the two generation files is unfinished or if they were generated with
different template, decoding flags or precision: regenerate the affected file (`OVERWRITE = True`) and rerun.
'''),
    code(f'''
G = "{GEN_DIR}"
P = "outputs/eval/pairs"

def have(*names):
    return all((Path(G) / name).exists() for name in names)

if have("alpaca_sft.json", "alpaca_dpo_uf.json"):
    !python scripts/build_pairs.py --baseline_path $G/alpaca_sft.json --candidate_path $G/alpaca_dpo_uf.json --output_path $P/dpo_uf_vs_sft_pairs.jsonl --comparison dpo_uf_vs_sft --profile general
if have("hh_sft.json", "hh_dpo_hh.json"):
    !python scripts/build_pairs.py --baseline_path $G/hh_sft.json --candidate_path $G/hh_dpo_hh.json --output_path $P/dpo_hh_vs_sft_pairs.jsonl --comparison dpo_hh_vs_sft --profile safety
if have("persona_base.json", "persona_lora.json"):
    !python scripts/build_pairs.py --baseline_path $G/persona_base.json --candidate_path $G/persona_lora.json --output_path $P/persona_vs_base_style_pairs.jsonl --comparison persona_vs_base_style --profile persona
    !python scripts/build_pairs.py --baseline_path $G/persona_base.json --candidate_path $G/persona_lora.json --output_path $P/persona_vs_base_quality_pairs.jsonl --comparison persona_vs_base_quality --profile safety
if have("alpaca_base.json", "alpaca_sft.json"):
    # Optional, for your own analysis (not the required comparison): SFT against the untuned base model.
    !python scripts/build_pairs.py --baseline_path $G/alpaca_base.json --candidate_path $G/alpaca_sft.json --output_path $P/sft_vs_base_pairs.jsonl --comparison sft_vs_base --profile general
'''),
    md("## 7. Package and save"),
    code('''
!python scripts/summarize_results.py --runs --eval_dir outputs/eval --out_dir outputs/report
!python scripts/package_outputs.py --source outputs/eval outputs/report --zip_path /kaggle/working/eval_generations_and_pairs.zip
!ls -la outputs/eval/generations outputs/eval/pairs
'''),
    md('''
Save the notebook version with its output (**Save Version**; for an interactive run choose Quick Save with
*Always save output*), then attach this notebook to notebook 03.

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

**Create this notebook from https://www.kaggle.com/benchmarks/tasks/new** so that `kaggle_benchmarks` and the Kaggle
model proxy (`MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`) are available, then bring these cells into it
(File -> Import Notebook, or copy and paste). No GPU is needed.
**Add Input -> Notebooks ->** your saved *02 - Generation* notebook version.

Nothing is generated here. The notebook reads saved pairwise JSONL files, judges 5 pairs per request, waits at least
5 seconds between all request attempts, tries a failing batch up to 3 times, and appends results after every batch.
Rerun any cell to continue where it stopped.
'''),
    md("## 1. Get the code (taken from the attached generation output; otherwise from GitHub)"),
    code(SETUP_CODE),
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

for source in sorted(Path("/kaggle/input").glob("**/outputs/eval/pairs/*_pairs.jsonl")):
    target = PAIRS_DIR / source.name
    if not target.exists():
        shutil.copyfile(source, target)
# Judgments from an earlier version of THIS notebook (attach that version as input to continue it).
for source in sorted(Path("/kaggle/input").glob("**/outputs/eval/judged/*")):
    target = JUDGED_DIR / source.name
    if source.is_file() and not target.exists():
        shutil.copyfile(source, target)

print("pairs:", sorted(path.name for path in PAIRS_DIR.glob("*_pairs.jsonl")))
print("existing judgment files:", sorted(path.name for path in JUDGED_DIR.glob("*_judged.jsonl")))
'''),
    md('''
## 3. Choose the judge model

`google/gemini-3.6-flash` is the model suggested by the assignment. The cell lists what this notebook actually
exposes. If the suggested model is not available (the first check in section 4 fails), set `JUDGE_MODEL` to a
listed model and name it in the report.
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
def judge(comparison, max_batches=None):
    pairs_path = PAIRS_DIR / f"{comparison}_pairs.jsonl"
    if not pairs_path.exists():
        print("skipping", comparison, "- no pairs file")
        return None
    argv = ["--pairs_path", str(pairs_path),
            "--output_path", str(JUDGED_DIR / f"{comparison}_judged.jsonl"),
            "--judge_model", JUDGE_MODEL, "--backend", BACKEND,
            "--batch_size", "5", "--sleep_seconds", "5", "--max_retries", "3", "--reuse_judgments"]
    if max_batches:
        argv += ["--max_batches", str(max_batches)]
    if INVALIDATE_INCOMPATIBLE:
        argv += ["--invalidate_incompatible"]
    return judge_cli.main(argv)
'''),
    md("## 4. First check: one batch of 5 pairs. Read the reasons before spending more calls."),
    code('''
judge("dpo_uf_vs_sft", max_batches=1)
from alignment.io_utils import read_jsonl
first_check = JUDGED_DIR / "dpo_uf_vs_sft_judged.jsonl"
if first_check.exists():
    for record in read_jsonl(first_check)[:5]:
        print(record["prompt_id"], "| verdict:", record["verdict"], "| winner:", record["winner_model"], "|", record["reason"])
'''),
    md('''
## 5. Full judging

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
    md("## 6. Summaries"),
    code('''
import json
for path in sorted(JUDGED_DIR.glob("*.summary.json")):
    summary = json.loads(path.read_text())
    print(f"{summary['comparison']}: judge={summary['judge_model']} pairs={summary['total_pairs']} "
          f"wins={summary['candidate_wins']} losses={summary['baseline_wins']} ties={summary['ties']} "
          f"unknown={summary['unknown']} failures={summary['failures']} complete={summary['complete']} "
          f"win_rate_excl_ties={summary['candidate_win_rate_excluding_ties']}")
'''),
    code('''
!python scripts/summarize_results.py --runs --eval_dir outputs/eval --out_dir outputs/report
!python scripts/package_outputs.py --source outputs/eval/judged outputs/report --zip_path /kaggle/working/judged_outputs.zip
print(open("outputs/report/report_tables.md").read())
'''),
    md('''
Save the notebook version with its output to keep `outputs/eval/judged/*`: judgments, `*.failures.jsonl`,
`*.attempts.jsonl` and `*.summary.json`. Unresolved pairs stay listed in the failures file and are counted
separately in every summary.
'''),
]

NOTEBOOKS = {
    "01_train_kaggle.ipynb": train_cells,
    "02_generate_kaggle.ipynb": generate_cells,
    "03_judge_kaggle.ipynb": judge_cells,
}


def main() -> None:
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = out_dir / name
        path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
