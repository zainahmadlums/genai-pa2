"""The Kaggle notebooks run top to bottom with "Save & Run All": they are exactly what
tools/build_notebooks.py writes, their cells are plain Python, every command is checked, and a missing
checkpoint or pair file raises instead of being skipped.

The code cells are executed here with temporary directories (both with a space in their name) standing in
for /kaggle/working and /kaggle/input. For notebooks 01 and 02 the commands are recorded instead of run
(no GPU, no downloads); notebook 03 runs for real with the offline mock judge.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from alignment import judge
from alignment.io_utils import read_json, write_json, write_jsonl
from alignment.judge import RequestPacer
from tests.test_pairwise_and_judge import FakeClock, make_pairs

ROOT = Path(__file__).resolve().parents[1]
TRAIN, GENERATE, JUDGE = NAMES = ["01_train_kaggle.ipynb", "02_generate_kaggle.ipynb", "03_judge_kaggle.ipynb"]
REPO_URL = "https://github.com/zainahmadlums/genai-pa2.git"
CHECKPOINTS = {"sft": "best", "dpo_hh": "best", "dpo_ultrafeedback": "best", "persona_lora": "adapter"}
REQUIRED = ["dpo_uf_vs_sft", "dpo_hh_vs_sft", "persona_vs_base_style", "persona_vs_base_quality"]
CELL_BREAK = "\n# ---- next cell ----\n"


def load_builder():
    spec = importlib.util.spec_from_file_location("pa2_build_notebooks", ROOT / "tools" / "build_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_source(name: str) -> str:
    payload = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return CELL_BREAK.join("".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code")


def finish_run(outputs: Path, run: str) -> Path:
    """What a completed training run leaves behind (no weights)."""

    run_dir = outputs / run
    write_json(run_dir / "TRAINING_COMPLETE.json", {"completed": True})
    write_json(run_dir / CHECKPOINTS[run] / "alignment_meta.json", {"template": "template-from-checkpoint"})
    return run_dir


class FakeProcess:
    def __init__(self, returncode: int):
        self.returncode = returncode
        self.stdout = iter(["recorded, not run\n"])

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class RecordedCommands:
    """Replaces ``subprocess.Popen``: records each command and leaves behind what the real one would."""

    def __init__(self, fail: str | None = None, incomplete: tuple[str, ...] = ()):
        self.argvs: list[list[str]] = []
        self.fail = fail              # a command with this argument exits with status 1
        self.incomplete = incomplete  # training runs that exit with status 0 without finishing

    def __call__(self, argv, **options):
        assert all(isinstance(arg, str) for arg in argv) and "shell" not in options
        argv = list(argv)
        self.argvs.append(argv)
        if self.fail in argv:
            return FakeProcess(1)
        if argv[:2] == ["git", "clone"]:
            write_json(Path(argv[-1]) / "scripts" / "judge_pairwise_kaggle.py", "placeholder")
        elif Path(argv[1]).name.startswith("train_") and "--smoke" not in argv:
            run = Path(argv[argv.index("--config") + 1]).stem
            if run not in self.incomplete:
                finish_run(Path("outputs"), run)
        elif Path(argv[1]).name == "generate.py" and "--smoke" not in argv:
            write_json(argv[argv.index("--output_path") + 1],
                       [{"id": "p0", "instruction": "question", "output": f"answer {len(self.argvs)}"}])
        elif Path(argv[1]).name == "summarize_results.py":
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "report_tables.md").write_text("# tables\n", encoding="utf-8")
        return FakeProcess(0)

    def steps(self) -> list[str]:
        labels = []
        for argv in self.argvs:
            if argv[0] == "git":
                labels.append("git " + argv[1])
            elif argv[1] == "-m":
                labels.append(argv[2])
            else:
                labels.append(Path(argv[1]).stem + (" --smoke" if "--smoke" in argv else ""))
        return labels

    def named(self, step: str) -> list[list[str]]:
        return [argv for argv, label in zip(self.argvs, self.steps()) if label == step]


class Sandbox:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.work, self.inputs = tmp_path / "kaggle working", tmp_path / "kaggle input"
        self.work.mkdir()
        self.inputs.mkdir()
        self.repo = self.work / "small-model-alignment"
        self.monkeypatch = monkeypatch

    def cells(self, name: str, **settings) -> list[str]:
        """Code cells of a notebook with the two Kaggle directories redirected and ``settings`` assigned."""

        source = code_source(name)
        for kaggle_dir, local in (("/kaggle/working", self.work), ("/kaggle/input", self.inputs)):
            literal = f'Path("{kaggle_dir}")'
            assert source.count(literal) == 1, f"{name}: {literal} must be defined exactly once"
            source = source.replace(literal, f"Path({str(local)!r})")
        for key, value in settings.items():
            source, count = re.subn(rf"^{key} = .*$", lambda match: f"{key} = {value!r}", source, flags=re.MULTILINE)
            assert count == 1, f"{name}: expected one top-level assignment of {key}"
        return source.split(CELL_BREAK)

    def execute(self, sources: list[str]) -> dict:
        namespace = {"__name__": "__main__"}
        for index, source in enumerate(sources):
            exec(compile(source, f"<code cell {index}>", "exec"), namespace)
        return namespace

    def record(self, **options) -> RecordedCommands:
        commands = RecordedCommands(**options)
        self.monkeypatch.setattr(subprocess, "Popen", commands)
        return commands

    def attach_code(self, input_name: str) -> Path:
        """An attached input that holds a copy of the code, as a saved notebook output does."""

        repo = self.inputs / input_name / "small-model-alignment"
        for folder in ("scripts", "alignment"):
            shutil.copytree(ROOT / folder, repo / folder, ignore=shutil.ignore_patterns("__pycache__"))
        return repo

    def attach_training_output(self, runs=tuple(CHECKPOINTS)) -> Path:
        outputs = self.inputs / "pa2 training" / "small-model-alignment" / "outputs"
        for run in runs:
            finish_run(outputs, run)
        return outputs

    def attach_generation_output(self, comparisons=tuple(REQUIRED)) -> Path:
        repo = self.attach_code("pa2 generation")
        for comparison in comparisons:
            write_jsonl(repo / "outputs" / "eval" / "pairs" / f"{comparison}_pairs.jsonl",
                        [{**pair, "pair_id": f"{comparison}:{pair['prompt_id']}", "comparison": comparison}
                         for pair in make_pairs(6)])
        return repo


@pytest.fixture
def kaggle(tmp_path, monkeypatch):
    # The first cell changes directory, extends sys.path and sets an environment variable: all undone afterwards.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    if importlib.util.find_spec("IPython") is None:  # only needed to display the plots
        display = types.ModuleType("IPython.display")
        display.Image = display.display = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
        monkeypatch.setitem(sys.modules, "IPython.display", display)
    sys.modules.pop("judge_pairwise_kaggle", None)
    yield Sandbox(tmp_path, monkeypatch)
    sys.modules.pop("judge_pairwise_kaggle", None)


@pytest.fixture
def instant_pacer(monkeypatch):
    """The judge's 5-second pacing runs on a virtual clock."""

    clock = FakeClock()
    monkeypatch.setattr(judge, "_SHARED_PACER",
                        RequestPacer(clock=clock.clock, sleep_fn=clock.sleep, wall_clock=clock.clock))
    return clock


# ---------------------------------------------------------------------------
# Generated files and their form
# ---------------------------------------------------------------------------


def test_notebooks_on_disk_are_exactly_what_the_builder_writes(tmp_path):
    builder = load_builder()
    builder.main(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == NAMES
    for name in NAMES:
        written = (tmp_path / name).read_bytes()
        assert b"\r" not in written  # the same bytes on Windows and Linux
        assert written.decode("utf-8") == builder.render(builder.NOTEBOOKS[name])
        assert written.decode("utf-8") == (ROOT / "notebooks" / name).read_text(encoding="utf-8"), (
            f"notebooks/{name} is out of date: run python tools/build_notebooks.py")


@pytest.mark.parametrize("name", NAMES)
def test_notebooks_need_no_editing_and_contain_only_checked_python(name):
    payload = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    first = payload["cells"][0]
    assert first["cell_type"] == "markdown" and "Save & Run All" in "".join(first["source"])
    source = code_source(name)
    for line in source.splitlines():
        # No "!" or "%" lines: their exit status would be ignored and "Run All" would carry on.
        assert not line.lstrip().startswith(("!", "%")), f"{name}: shell line {line.strip()!r}"
    assert "shell=True" not in source and "os.system" not in source
    assert f'REPO_URL = "{REPO_URL}"' in source and 'REPO_BRANCH = "main"' in source
    assert "YOUR_USERNAME" not in source and "@github.com" not in source  # no placeholder, no credentials
    assert 'CODE_DATASET_DIR = ""' in source
    assert f"PREFER_ATTACHED_CODE = {name == JUDGE}" in source
    if name != JUDGE:
        assert "RUN_SMOKE = False" in source


# ---------------------------------------------------------------------------
# First cell: commands and where the code comes from
# ---------------------------------------------------------------------------


def test_commands_run_without_a_shell_show_their_output_and_raise_on_failure(kaggle, monkeypatch, capsys):
    write_json(kaggle.repo / "scripts" / "judge_pairwise_kaggle.py", "an existing copy of the code")
    with monkeypatch.context() as patch:
        commands = RecordedCommands()
        patch.setattr(subprocess, "Popen", commands)
        namespace = kaggle.execute(kaggle.cells(TRAIN)[:1])
    assert commands.argvs == []  # an existing copy is used as it is: no clone, no pull
    assert "never updated" in capsys.readouterr().out
    assert os.path.samefile(os.getcwd(), kaggle.repo)

    script = kaggle.work / "print arguments.py"
    script.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    namespace["py"](script, "two words", Path("a directory") / "file.json", "--flag")
    assert json.dumps(["two words", str(Path("a directory") / "file.json"), "--flag"]) in capsys.readouterr().out

    with pytest.raises(subprocess.CalledProcessError) as error:
        namespace["py"]("-c", "import sys; print('to stdout'); print('to stderr', file=sys.stderr); sys.exit(3)")
    assert error.value.returncode == 3
    shown = capsys.readouterr().out
    assert "to stdout" in shown and "to stderr" in shown
    with pytest.raises(FileNotFoundError):
        namespace["run"]("no-such-program-pa2")


def test_training_notebook_clones_the_repository_even_when_a_copy_is_attached(kaggle):
    kaggle.attach_code("an earlier output")
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(TRAIN)[:1])
    # The destination contains a space and is still one argument.
    assert commands.argvs == [["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, str(kaggle.repo)]]


def test_judging_notebook_copies_the_code_from_the_attached_output_without_cloning(kaggle):
    attached = kaggle.attach_generation_output()
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(JUDGE)[:1])
    assert commands.argvs == []
    assert (kaggle.repo / "scripts" / "judge_pairwise_kaggle.py").read_bytes() == (
        attached / "scripts" / "judge_pairwise_kaggle.py").read_bytes()
    assert not (kaggle.repo / "outputs").exists()  # results are not copied along with the code


def test_judging_notebook_clones_only_when_no_code_is_attached(kaggle):
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(JUDGE)[:1])
    assert commands.steps() == ["git clone"] and REPO_URL in commands.argvs[0]


@pytest.mark.parametrize("name", NAMES)
def test_code_dataset_dir_overrides_both_github_and_attached_outputs(kaggle, name):
    kaggle.attach_code("a notebook output")
    dataset = kaggle.inputs / "my pa2 code" / "genaiPA2"
    shutil.copytree(ROOT / "scripts", dataset / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (dataset / "from_dataset.txt").write_text("x", encoding="utf-8")
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(name, CODE_DATASET_DIR=str(dataset))[:1])
    assert commands.argvs == [] and (kaggle.repo / "from_dataset.txt").exists()


def test_first_cell_fails_loudly_instead_of_continuing_without_the_code(kaggle):
    commands = kaggle.record(fail="clone")
    with pytest.raises(FileNotFoundError, match="CODE_DATASET_DIR"):
        kaggle.execute(kaggle.cells(TRAIN, CODE_DATASET_DIR=str(kaggle.inputs / "not there"))[:1])
    assert commands.argvs == [] and not kaggle.repo.exists()
    with pytest.raises(subprocess.CalledProcessError):  # git clone failed
        kaggle.execute(kaggle.cells(TRAIN)[:1])
    assert commands.steps() == ["git clone"]
    kaggle.repo.mkdir()  # left over, without the code
    with pytest.raises(RuntimeError, match="not a complete copy"):
        kaggle.execute(kaggle.cells(TRAIN)[:1])
    assert commands.steps() == ["git clone"]


# ---------------------------------------------------------------------------
# Notebook 01
# ---------------------------------------------------------------------------


def test_training_notebook_full_run(kaggle, capsys):
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(TRAIN))
    assert commands.steps() == [
        "git clone", "pip", "pip", "pip", "check_env", "pytest",
        "make_alpaca_eval_subset", "prepare_data", "prepare_data", "prepare_data",
        "train_sft", "package_outputs", "train_dpo", "package_outputs", "train_dpo", "package_outputs",
        "train_persona_lora", "package_outputs", "plot_metrics", "summarize_results", "package_outputs",
    ]  # no smoke runs, the tests run once
    assert all(argv[0] in ("git", sys.executable) for argv in commands.argvs)
    pip_arguments = [arg for argv in commands.named("pip") for arg in argv[3:]]
    assert "trl==1.14.1" in pip_arguments and not any(re.match(r"torch(?!ao)", arg) for arg in pip_arguments)
    assert commands.named("check_env")[0][2:] == ["--require_gpus", "2"]
    assert commands.named("pytest")[0][1:] == ["-m", "pytest", "-q", "tests"]
    assert [argv[argv.index("--task") + 1] for argv in commands.named("prepare_data")] == ["sft", "hh", "ultrafeedback"]
    assert commands.named("train_sft")[0][1:] == [
        "scripts/train_sft.py", "--config", "configs/sft.yaml", "--resume",
        "--override", "resume_every_steps=250", "resume_dir=/kaggle/temp/resume/sft"]
    hh, ultrafeedback = commands.named("train_dpo")
    sft_checkpoint = f"sft_checkpoint={Path('outputs') / 'sft' / 'best'}"
    assert hh[1:] == ["scripts/train_dpo.py", "--config", "configs/dpo_hh.yaml", "--resume", "--override",
                      sft_checkpoint, "resume_every_steps=100", "resume_dir=/kaggle/temp/resume/dpo_hh"]
    assert ultrafeedback[3] == "configs/dpo_ultrafeedback.yaml" and sft_checkpoint in ultrafeedback
    plot = commands.named("plot_metrics")[0]
    run_dirs = [str(Path("outputs") / run) for run in CHECKPOINTS]
    assert plot[plot.index("--run_dir") + 1:plot.index("--out_dir")] == run_dirs  # one argument per run
    summarize = commands.named("summarize_results")[0]
    assert summarize[summarize.index("--runs") + 1:summarize.index("--out_dir")] == run_dirs
    zips = [argv[argv.index("--zip_path") + 1] for argv in commands.named("package_outputs")]
    assert zips == [str(kaggle.work / f"{stem}.zip") for stem in (
        "sft_run_outputs", "dpo_hh_run_outputs", "dpo_ultrafeedback_run_outputs", "persona_lora_run_outputs",
        "training_plots_and_report")]
    assert "NOTEBOOK 01 FINISHED" in capsys.readouterr().out


def test_training_notebook_reuses_a_finished_run_from_an_attached_input(kaggle):
    sft = finish_run(kaggle.inputs / "pa2 training v1" / "small-model-alignment" / "outputs", "sft")
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(TRAIN, STAGES=["dpo_hh"]))
    training = [step for step in commands.steps() if step.startswith(("train_", "prepare_"))]
    assert training == ["prepare_data", "train_dpo"]
    assert f"sft_checkpoint={sft / 'best'}" in commands.named("train_dpo")[0]  # spaces in the path, one argument
    plot = commands.named("plot_metrics")[0]
    assert plot[plot.index("--run_dir") + 1:plot.index("--out_dir")] == [str(sft), str(Path("outputs") / "dpo_hh")]


@pytest.mark.parametrize("stage", ["dpo_hh", "dpo_ultrafeedback"])
def test_training_notebook_requires_the_sft_checkpoint_before_dpo(kaggle, stage):
    commands = kaggle.record()
    with pytest.raises(FileNotFoundError, match="'sft' is not finished"):
        kaggle.execute(kaggle.cells(TRAIN, STAGES=[stage]))
    assert not any(step.startswith("train_") for step in commands.steps())


def test_training_notebook_stops_when_a_run_exits_cleanly_without_finishing(kaggle):
    commands = kaggle.record(incomplete=("sft",))
    with pytest.raises(FileNotFoundError, match="'sft' is not finished"):
        kaggle.execute(kaggle.cells(TRAIN))
    assert commands.steps()[-2:] == ["train_sft", "package_outputs"]  # no DPO on a missing checkpoint


@pytest.mark.parametrize("failing", ["pip", "scripts/check_env.py", "pytest", "scripts/prepare_data.py",
                                     "scripts/train_sft.py", "scripts/train_dpo.py", "scripts/package_outputs.py",
                                     "scripts/plot_metrics.py"])
def test_training_notebook_stops_at_the_first_failing_command(kaggle, failing):
    commands = kaggle.record(fail=failing)
    with pytest.raises(subprocess.CalledProcessError):
        kaggle.execute(kaggle.cells(TRAIN))
    assert failing in commands.argvs[-1] and sum(failing in argv for argv in commands.argvs) == 1


def test_training_smoke_runs_are_opt_in_and_unknown_stages_are_rejected(kaggle):
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(TRAIN, RUN_SMOKE=True))
    assert [step for step in commands.steps() if step.startswith("train_")] == [
        "train_sft --smoke", "train_sft", "train_dpo --smoke", "train_dpo", "train_dpo",
        "train_persona_lora --smoke", "train_persona_lora"]
    with pytest.raises(ValueError, match="unknown entries in STAGES"):
        kaggle.execute(kaggle.cells(TRAIN, STAGES=["sft", "dpo"]))


# ---------------------------------------------------------------------------
# Notebook 02
# ---------------------------------------------------------------------------


def test_generation_notebook_full_run(kaggle, capsys):
    outputs = kaggle.attach_training_output()
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(GENERATE))
    assert commands.steps() == (["git clone", "pip", "pip", "pip", "check_env"] + ["generate"] * 8
                                + ["build_pairs"] * 5 + ["summarize_results", "package_outputs"])
    assert commands.named("check_env")[0][2:] == ["--require_gpus", "1"]
    flags = ["--max_new_tokens", "256", "--batch_size", "16"]
    generations = commands.named("generate")
    assert generations[0][1:] == [
        "scripts/generate.py", "--model_path", "Qwen/Qwen2.5-0.5B",
        "--prompts_path", "eval_sets/alpaca_eval_300_seed42.json", "--generator", "qwen2.5-0.5b-base",
        "--output_path", "outputs/eval/generations/alpaca_base.json", "--template", "template-from-checkpoint", *flags]
    assert generations[1][1:] == [
        "scripts/generate.py", "--model_path", str(outputs / "sft" / "best"),  # spaces in the path, one argument
        "--prompts_path", "eval_sets/alpaca_eval_300_seed42.json", "--generator", "qwen2.5-0.5b-sft",
        "--output_path", "outputs/eval/generations/alpaca_sft.json", *flags]
    assert generations[7][1:] == [
        "scripts/generate.py", "--model_path", "HuggingFaceTB/SmolLM2-360M-Instruct",
        "--prompts_path", "eval_sets/persona_eval_prompts.json", "--generator", "smollm2-360m-instruct-persona-lora",
        "--output_path", "outputs/eval/generations/persona_lora.json",
        "--adapter_path", str(outputs / "persona_lora" / "adapter"), "--template", "chat_template", *flags]
    models = [argv[argv.index("--model_path") + 1] for argv in generations]
    assert models[4:6] == [str(outputs / "dpo_ultrafeedback" / "best"), str(outputs / "dpo_hh" / "best")]
    assert [Path(argv[argv.index("--output_path") + 1]).name for argv in generations] == [
        "alpaca_base.json", "alpaca_sft.json", "hh_base.json", "hh_sft.json", "alpaca_dpo_uf.json", "hh_dpo_hh.json",
        "persona_base.json", "persona_lora.json"]
    assert all(argv[-4:] == flags for argv in generations)  # same decoding for every model, no smoke, no overwrite
    pairs = commands.named("build_pairs")
    assert [(argv[argv.index("--comparison") + 1], argv[argv.index("--profile") + 1]) for argv in pairs] == [
        ("dpo_uf_vs_sft", "general"), ("dpo_hh_vs_sft", "safety"), ("persona_vs_base_style", "persona"),
        ("persona_vs_base_quality", "safety"), ("sft_vs_base", "general")]
    assert pairs[0][1:] == [
        "scripts/build_pairs.py", "--baseline_path", "outputs/eval/generations/alpaca_sft.json",
        "--candidate_path", "outputs/eval/generations/alpaca_dpo_uf.json",
        "--output_path", "outputs/eval/pairs/dpo_uf_vs_sft_pairs.jsonl", "--comparison", "dpo_uf_vs_sft",
        "--profile", "general"]
    shown = capsys.readouterr().out
    assert "persona_lora.json [p0] 'question'" in shown and shown.count("  -> 'answer ") == 8  # first answers shown
    assert "NOTEBOOK 02 FINISHED" in shown


@pytest.mark.parametrize("missing", list(CHECKPOINTS))
def test_generation_notebook_requires_every_finished_run(kaggle, missing):
    kaggle.attach_training_output([run for run in CHECKPOINTS if run != missing])
    commands = kaggle.record()
    with pytest.raises(FileNotFoundError, match=f"'{missing}' is not finished"):
        kaggle.execute(kaggle.cells(GENERATE))
    assert "generate" not in commands.steps()  # nothing is generated for an incomplete set of models


def test_generation_notebook_requires_the_checkpoint_directory_of_a_finished_run(kaggle):
    outputs = kaggle.attach_training_output()
    shutil.rmtree(outputs / "persona_lora" / "adapter")
    commands = kaggle.record()
    with pytest.raises(FileNotFoundError, match="'persona_lora' is marked finished but its checkpoint"):
        kaggle.execute(kaggle.cells(GENERATE))
    assert "generate" not in commands.steps()


@pytest.mark.parametrize("failing", ["scripts/check_env.py", "scripts/generate.py", "scripts/build_pairs.py",
                                     "scripts/summarize_results.py"])
def test_generation_notebook_stops_at_the_first_failing_command(kaggle, failing):
    kaggle.attach_training_output()
    commands = kaggle.record(fail=failing)
    with pytest.raises(subprocess.CalledProcessError):
        kaggle.execute(kaggle.cells(GENERATE))
    assert failing in commands.argvs[-1] and sum(failing in argv for argv in commands.argvs) == 1


def test_generation_smoke_is_opt_in_and_flags_stay_separate_arguments(kaggle):
    kaggle.attach_training_output()
    commands = kaggle.record()
    kaggle.execute(kaggle.cells(GENERATE, RUN_SMOKE=True, OVERWRITE=True))
    generations = commands.named("generate --smoke") + commands.named("generate")
    assert len(generations) == 16 and commands.steps().index("generate") == commands.steps().index("generate --smoke") + 8
    flags = ["--max_new_tokens", "256", "--batch_size", "16", "--overwrite"]
    assert all(argv[-7:] == flags + ["--smoke", "5"] for argv in generations[:8])
    assert all(argv[-5:] == flags for argv in generations[8:])


# ---------------------------------------------------------------------------
# Notebook 03
# ---------------------------------------------------------------------------


def test_judging_notebook_runs_end_to_end_with_the_mock_judge(kaggle, instant_pacer, monkeypatch, capsys):
    kaggle.attach_generation_output()
    started, real_popen = [], subprocess.Popen

    def popen(argv, **options):
        started.append(list(argv))
        return real_popen(argv, **options)

    monkeypatch.setattr(subprocess, "Popen", popen)
    sources = kaggle.cells(JUDGE, BACKEND="mock")
    namespace = kaggle.execute(sources)

    # No clone; the two reporting scripts really ran.
    assert [Path(argv[1]).stem for argv in started] == ["summarize_results", "package_outputs"]
    judged = kaggle.repo / "outputs" / "eval" / "judged"
    for comparison in REQUIRED:
        summary = read_json(judged / f"{comparison}_judged.summary.json")
        assert summary["comparison"] == comparison and summary["backend"] == "mock"
        assert summary["complete"] and summary["valid_judgments"] == summary["total_pairs"] == 6
        assert (summary["batch_size"], summary["sleep_seconds"], summary["max_retries"]) == (5, 5.0, 3)
    assert sorted(path.name for path in judged.glob("*_judged.jsonl")) == sorted(f"{name}_judged.jsonl" for name in REQUIRED)
    assert (kaggle.work / "judged_outputs.zip").stat().st_size > 0
    shown = capsys.readouterr().out
    assert shown.count("| verdict: ") == 12  # three verdicts with their reasons per comparison
    assert "Pairwise judgments" in shown and "NOTEBOOK 03 FINISHED. Every pair" in shown

    # Unresolved pairs are reported by the last cell; the run is not failed, so the judgments are kept.
    summary_path = judged / "dpo_hh_vs_sft_judged.summary.json"
    write_json(summary_path, {**read_json(summary_path), "failures": 2})
    exec(sources[-1], namespace)
    shown = capsys.readouterr().out
    assert "FINISHED WITH UNRESOLVED PAIRS" in shown and "dpo_hh_vs_sft" in shown


@pytest.mark.parametrize("missing", REQUIRED)
def test_judging_notebook_requires_every_pair_file_before_any_judge_call(kaggle, instant_pacer, missing):
    kaggle.attach_generation_output([name for name in REQUIRED if name != missing])
    with pytest.raises(FileNotFoundError, match=missing):
        kaggle.execute(kaggle.cells(JUDGE, BACKEND="mock"))
    assert "judge_pairwise_kaggle" not in sys.modules  # stopped before the judge was even loaded
    assert not list((kaggle.repo / "outputs" / "eval" / "judged").iterdir())


def test_judging_notebook_turns_a_judge_stop_into_a_python_error(kaggle, instant_pacer):
    kaggle.attach_generation_output()
    # Judgments of another judge, as an attached earlier version of notebook 03 would bring them along.
    write_jsonl(kaggle.inputs / "pa2 judging v1" / "outputs" / "eval" / "judged" / "dpo_uf_vs_sft_judged.jsonl",
                [{"pair_id": "x", "protocol_id": "another-judge", "backend": "kaggle_benchmarks", "judge_model": "m"}])
    with pytest.raises(RuntimeError, match="judging dpo_uf_vs_sft stopped"):
        kaggle.execute(kaggle.cells(JUDGE, BACKEND="mock"))
    assert not (kaggle.repo / "outputs" / "eval" / "judged" / "dpo_hh_vs_sft_judged.jsonl").exists()
