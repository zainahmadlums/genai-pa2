"""Generation file format/resume, config loading, command-line entry points and notebooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from alignment.config import load_config, parse_override, pick
from alignment.engine import LoopConfig
from alignment.dpo_trainer import DPOTaskConfig
from alignment.generation import GenerationConfig, generate_file, generation_kwargs, load_eval_prompts
from alignment.sft_trainer import SFTTaskConfig
from alignment.train_utils import read_json
from tests.helpers import TemplateToyTokenizer

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


class EchoGenerator(torch.nn.Module):
    """Fake causal LM: 'generates' the last two prompt tokens, then EOS, then padding."""

    def __init__(self, eos_token_id: int, pad_token_id: int):
        super().__init__()
        self.eos_token_id, self.pad_token_id = eos_token_id, pad_token_id
        self.calls: list[dict] = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append({"shape": tuple(input_ids.shape), "kwargs": kwargs,
                           "left_padded": bool(attention_mask[:, -1].all())})
        tail = input_ids[:, -2:]
        eos = torch.full((input_ids.shape[0], 1), self.eos_token_id)
        pad = torch.full((input_ids.shape[0], 2), self.pad_token_id)
        return torch.cat([input_ids, tail, eos, pad], dim=1)


def _records(count: int) -> list[dict]:
    return [{"id": f"q{index}", "instruction": "word " * (index + 1) + f"end{index}", "dataset": "toy_eval",
             "category": "c"} for index in range(count)]


def test_generation_file_format_order_and_resume(tmp_path):
    tokenizer = TemplateToyTokenizer()
    model = EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    records = _records(5)
    out_path = tmp_path / "gen" / "toy.json"
    config = GenerationConfig(max_new_tokens=8, batch_size=2)
    entries = generate_file(model, tokenizer, records, out_path, "toy-model", None, "alpaca", config, "cpu",
                            log=lambda *_: None)

    assert [entry["id"] for entry in entries] == [record["id"] for record in records]  # input order kept
    for entry, record in zip(entries, records):
        assert set(entry) == {"id", "instruction", "output", "generator", "dataset", "category"}
        assert entry["instruction"] == record["instruction"]
        assert entry["generator"] == "toy-model" and entry["dataset"] == "toy_eval"
        assert entry["output"] == ":"  # echo of the last two prompt tokens ":\n", stripped; EOS and padding removed
    assert read_json(out_path) == entries
    assert len(model.calls) == 3 and all(call["left_padded"] for call in model.calls)
    assert model.calls[0]["kwargs"]["do_sample"] is False and model.calls[0]["kwargs"]["max_new_tokens"] == 8
    meta = read_json(tmp_path / "gen" / "toy.meta.json")
    assert meta["template"] == "alpaca" and meta["decoding"]["repetition_penalty"] == 1.0
    assert meta["stats_this_run"]["finished_with_eos"] == 5 and meta["num_outputs"] == 5

    # Restart with two more prompts: only the new ones are generated.
    more = _records(7)
    resumed_model = EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    entries = generate_file(resumed_model, tokenizer, more, out_path, "toy-model", None, "alpaca", config, "cpu",
                            log=lambda *_: None)
    assert len(entries) == 7 and sum(call["shape"][0] for call in resumed_model.calls) == 2
    assert [entry["id"] for entry in entries] == [record["id"] for record in more]

    # Another model can neither reuse nor silently replace this file.
    other_model = EchoGenerator(tokenizer.eos_token_id, tokenizer.pad_token_id)
    with pytest.raises(ValueError, match="already holds outputs"):
        generate_file(other_model, tokenizer, records, out_path, "other-model", None, "alpaca", config, "cpu",
                      log=lambda *_: None)
    assert other_model.calls == [] and len(read_json(out_path)) == 7
    entries = generate_file(other_model, tokenizer, records, out_path, "other-model", None, "alpaca", config, "cpu",
                            overwrite=True, log=lambda *_: None)
    assert sum(call["shape"][0] for call in other_model.calls) == 5
    assert {entry["generator"] for entry in entries} == {"other-model"}


def test_greedy_and_anti_repetition_generation_kwargs():
    tokenizer = TemplateToyTokenizer()
    greedy = generation_kwargs(tokenizer, GenerationConfig())
    assert greedy["do_sample"] is False and greedy["num_beams"] == 1
    assert "repetition_penalty" not in greedy and "no_repeat_ngram_size" not in greedy
    assert greedy["eos_token_id"] == tokenizer.eos_token_id and greedy["pad_token_id"] == tokenizer.pad_token_id
    tuned = generation_kwargs(tokenizer, GenerationConfig(repetition_penalty=1.1, no_repeat_ngram_size=4))
    assert tuned["repetition_penalty"] == 1.1 and tuned["no_repeat_ngram_size"] == 4 and tuned["do_sample"] is False


def test_eval_prompt_loading_validates_ids(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"id": "a", "instruction": "x"}, {"id": "a", "instruction": "y"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_prompts(path)
    assert len(load_eval_prompts(ROOT / "eval_sets" / "hh_rlhf_eval_300_seed42.json")) == 300
    assert len(load_eval_prompts(ROOT / "eval_sets" / "alpaca_eval_300_seed42.json", limit=5)) == 5


def test_configs_match_the_guide_recipe():
    sft = load_config(ROOT / "configs" / "sft.yaml")
    assert sft["model_name"] == "Qwen/Qwen2.5-0.5B"
    assert (sft["train_size"], sft["val_size"], sft["max_length"]) == (30000, 1000, 512)
    assert (sft["batch_size"], sft["grad_accum_steps"], sft["epochs"]) == (2, 16, 1)
    assert sft["learning_rate"] == pytest.approx(2e-5) and sft["warmup_ratio"] == pytest.approx(0.03)
    assert (sft["schedule"], sft["weight_decay"], sft["max_grad_norm"], sft["precision"]) == ("cosine", 0.1, 1.0, "fp16")
    loop = pick(sft, LoopConfig)
    assert loop.batch_size * loop.grad_accum_steps == 32
    assert pick(sft, SFTTaskConfig).reduction == "token_mean"

    for name in ("dpo_hh.yaml", "dpo_ultrafeedback.yaml"):
        dpo = load_config(ROOT / "configs" / name)
        assert dpo["sft_checkpoint"] == "outputs/sft/best"
        assert (dpo["train_size"], dpo["val_size"], dpo["max_length"], dpo["max_prompt_length"]) == (15000, 1000, 512, 256)
        assert (dpo["batch_size"], dpo["grad_accum_steps"], dpo["epochs"]) == (1, 32, 1)
        assert dpo["learning_rate"] == pytest.approx(1e-6) and dpo["beta"] == pytest.approx(0.1)
        assert (dpo["schedule"], dpo["max_grad_norm"]) == ("constant", 1.0)
        assert "template" not in dpo  # inherited from the SFT checkpoint
        assert pick(dpo, DPOTaskConfig).beta == pytest.approx(0.1)
    assert load_config(ROOT / "configs" / "dpo_hh.yaml")["train_file"] != load_config(
        ROOT / "configs" / "dpo_ultrafeedback.yaml")["train_file"]

    persona = load_config(ROOT / "configs" / "persona_lora.yaml")
    assert persona["model_name"] == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert persona["data_file"] == "data/persona_lora/ramsay_style_preferences.json"


def test_config_overrides_are_typed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("learning_rate: 2e-5\noutput_dir: outputs/x\nname: alpaca\n", encoding="utf-8")
    config = load_config(path, ["learning_rate=1e-6", "max_steps=4", "optimizer=rmsprop", "flag=true"])
    assert config["learning_rate"] == pytest.approx(1e-6) and isinstance(config["learning_rate"], float)
    assert config["max_steps"] == 4 and config["optimizer"] == "rmsprop" and config["flag"] is True
    assert config["output_dir"] == "outputs/x" and config["name"] == "alpaca"
    assert load_config(path)["learning_rate"] == pytest.approx(2e-5)
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")


def test_requirements_do_not_pin_torch():
    lines = [line.strip().lower() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    assert not any(line.startswith("torch") for line in lines)
    assert any(line.startswith("transformers") and "<5" in line for line in lines)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_every_script_has_working_help(script):
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, cwd=ROOT,
                            timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_expected_scripts_exist():
    names = {path.name for path in SCRIPTS}
    assert {"check_env.py", "prepare_data.py", "make_alpaca_eval_subset.py", "train_sft.py", "train_dpo.py",
            "train_persona_lora.py", "generate.py", "build_pairs.py", "judge_pairwise_kaggle.py", "plot_metrics.py",
            "summarize_results.py", "package_outputs.py"} <= names


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_are_valid_and_their_python_cells_compile(notebook):
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4 and payload["cells"]
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        assert cell["outputs"] == [] and cell["execution_count"] is None  # shipped without fabricated outputs
        source = "".join(cell["source"])
        python_lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            # Shell lines (also inside if-blocks) are replaced by an equally indented no-op.
            is_shell = stripped.startswith(("!", "%"))
            python_lines.append(line[: len(line) - len(stripped)] + "pass" if is_shell else line)
        compile("\n".join(python_lines), f"{notebook.name}[cell {index}]", "exec")


def test_notebooks_never_install_torch_and_keep_stages_separate():
    sources = {path.name: "".join("".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"])
               for path in NOTEBOOKS}
    for name, source in sources.items():
        for line in source.splitlines():
            if "pip install" in line:
                assert " torch" not in line.replace("torchao", ""), f"{name} must not install torch"
    assert "train_sft.py" in sources["01_train_kaggle.ipynb"] and "train_dpo.py" in sources["01_train_kaggle.ipynb"]
    assert "generate.py" in sources["02_generate_kaggle.ipynb"] and "train_" not in sources["02_generate_kaggle.ipynb"]
    judge = sources["03_judge_kaggle.ipynb"]
    assert "generate.py" not in judge and "train_" not in judge  # judging never generates or trains
    assert '"--sleep_seconds", "5"' in judge and '"--batch_size", "5"' in judge and "--reuse_judgments" in judge


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_shell_lines_only_use_python_variables_defined_in_the_notebook(notebook):
    """IPython expands ``$NAME`` in a ``!`` line only when every name on that line is a Python variable."""

    import re

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code")
    assigned = set(re.findall(r"^\s*([A-Za-z_]\w*)\s*=", source, flags=re.MULTILINE))
    for line in source.splitlines():
        if line.lstrip().startswith("!"):
            for name in re.findall(r"\$([A-Za-z_]\w*)", line):
                assert name in assigned, f"{notebook.name}: ${name} is not assigned in the notebook: {line.strip()}"
            assert "{" not in line and "}" not in line  # braces would also be expanded


def test_three_separate_kaggle_notebooks_exist():
    assert [path.name for path in NOTEBOOKS] == ["01_train_kaggle.ipynb", "02_generate_kaggle.ipynb",
                                                 "03_judge_kaggle.ipynb"]
