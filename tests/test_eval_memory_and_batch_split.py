"""Regression tests for the out-of-memory failure at the first periodic evaluation on a Kaggle T4:
the gradients of a finished optimizer step are released before validation runs, validation batches are
no larger than a training micro-batch, and validation metrics do not depend on the batch split."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import alignment.engine as engine
from alignment.config import load_config, pick
from alignment.dpo_trainer import DPOTask, DPOTaskConfig
from alignment.engine import LoopConfig, run_training
from alignment.prompt_format import format_prompt
from alignment.sft_trainer import SFTTask, SFTTaskConfig
from tests.helpers import BigramLM, TemplateToyTokenizer, manual_bigram_logprob
from tests.test_training_loops import DPO_ROWS, SFT_ROWS, loop_config, quiet

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
SPLITS = [1, 2, 3, 4, 5, 8]  # 5 validation rows: short final batches of 1 and 2, one batch, oversized batch


def expected_batches(rows: int, batch_size: int) -> list[int]:
    return [min(batch_size, rows - start) for start in range(0, rows, batch_size)]


def record_batches(task) -> list[int]:
    """Record the number of rows of every batch that ``task`` collates."""

    original, sizes = task.collate, []

    def collate(rows):
        sizes.append(len(rows))
        return original(rows)

    task.collate = collate
    return sizes


def response_logprob(model, tokenizer, instruction: str, response: str) -> tuple[float, int]:
    """Summed response log-probability of one row and its number of scored tokens, without batching."""

    prompt_ids = tokenizer.encode(format_prompt(instruction, "alpaca"))
    response_ids = tokenizer.encode(response) + [tokenizer.eos_token_id]
    return float(manual_bigram_logprob(model, prompt_ids, response_ids)), len(response_ids)


# ---------------------------------------------------------------------------
# Gradients are released before a periodic evaluation
# ---------------------------------------------------------------------------


def build_task(kind: str, tokenizer):
    if kind == "sft":
        model = BigramLM(tokenizer.vocab_size, seed=71)
        task = SFTTask(model, tokenizer, SFT_ROWS[8:], SFTTaskConfig(max_length=128, eval_batch_size=1), CPU)
        return model, task, SFT_ROWS[:8]
    model = BigramLM(tokenizer.vocab_size, seed=72)
    config = DPOTaskConfig(beta=0.5, max_length=128, max_prompt_length=None, eval_batch_size=1)
    task = DPOTask(model, BigramLM(tokenizer.vocab_size, seed=73), tokenizer, DPO_ROWS[8:], config, CPU)
    return model, task, DPO_ROWS[:8]


@pytest.mark.parametrize("kind", ["sft", "dpo"])
def test_gradients_are_released_before_every_evaluation_once_optimizer_state_exists(tmp_path, monkeypatch, kind):
    """8 rows, batch 1, accumulation 2 -> 4 optimizer steps, evaluated before and after each of them."""

    tokenizer = TemplateToyTokenizer()
    optimizers, build = [], engine.build_optimizer

    def capture(*args, **kwargs):
        optimizers.append(build(*args, **kwargs))
        return optimizers[-1]

    monkeypatch.setattr(engine, "build_optimizer", capture)
    settings = dict(batch_size=1, grad_accum_steps=2, learning_rate=0.01)

    model, task, rows = build_task(kind, tokenizer)
    parameters = list(model.parameters())
    evaluate, seen = task.evaluate, []

    def recording_evaluate():
        optimizer = optimizers[-1]
        seen.append({
            "held_gradients": sum(parameter.grad is not None for parameter in parameters),
            "adam_moments": sum("exp_avg" in optimizer.state.get(parameter, {}) for parameter in parameters),
        })
        return evaluate()

    task.evaluate = recording_evaluate
    summary = run_training(loop_config(tmp_path / "evaluated", eval_every_steps=1, **settings), model, rows, task,
                           CPU, log=quiet)

    assert summary["optimizer_steps"] == 4 and len(seen) == 5  # initial, periodic at 1, 2, 3 and final at 4
    assert seen[0] == {"held_gradients": 0, "adam_moments": 0}
    # After the first update AdamW holds its moments for every parameter; the gradients must be gone.
    assert seen[1:] == [{"held_gradients": 0, "adam_moments": len(parameters)}] * 4

    # Releasing them early changes nothing about the updates: same weights as a run that never evaluates.
    plain_model, plain_task, _ = build_task(kind, tokenizer)
    run_training(loop_config(tmp_path / "plain", eval_at_start=False, eval_every_steps=0, **settings), plain_model,
                 rows, plain_task, CPU, log=quiet)
    assert torch.equal(model.table.detach(), plain_model.table.detach())
    assert not torch.equal(model.table.detach(), BigramLM(tokenizer.vocab_size, seed=71 if kind == "sft" else 72).table)


# ---------------------------------------------------------------------------
# Validation metrics do not depend on the validation batch size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", SPLITS)
def test_sft_validation_metrics_are_the_same_for_every_batch_split(batch_size):
    tokenizer = TemplateToyTokenizer()
    model = BigramLM(tokenizer.vocab_size, seed=81)
    rows = SFT_ROWS[:5]  # responses of 4, 5 and 7 characters, so batches are padded differently per split
    scored = [response_logprob(model, tokenizer, row["instruction"], row["response"]) for row in rows]
    total_nll, total_tokens = -sum(logp for logp, _ in scored), sum(tokens for _, tokens in scored)

    task = SFTTask(model, tokenizer, rows, SFTTaskConfig(max_length=128, eval_batch_size=batch_size), CPU)
    sizes = record_batches(task)
    metrics = task.evaluate()

    assert sizes == expected_batches(5, batch_size)
    assert metrics["val_rows"] == 5 and metrics["val_response_tokens"] == total_tokens
    assert metrics["val_loss"] == pytest.approx(total_nll / total_tokens, rel=1e-6)  # mean over response tokens
    assert metrics["val_nll_per_example"] == pytest.approx(total_nll / 5, rel=1e-6)
    assert metrics["val_perplexity"] == pytest.approx(math.exp(total_nll / total_tokens), rel=1e-6)


@pytest.mark.parametrize("single_forward", [True, False])
@pytest.mark.parametrize("batch_size", SPLITS)
def test_dpo_validation_metrics_are_the_same_for_every_batch_split(batch_size, single_forward):
    tokenizer = TemplateToyTokenizer()
    policy, reference = BigramLM(tokenizer.vocab_size, seed=91), BigramLM(tokenizer.vocab_size, seed=92)
    rows, beta = DPO_ROWS[:5], 0.5

    def logps(model, side):
        return torch.tensor([response_logprob(model, tokenizer, row["instruction"], row[side])[0] for row in rows],
                            dtype=torch.float64)

    policy_chosen, policy_rejected = logps(policy, "chosen"), logps(policy, "rejected")
    reference_chosen, reference_rejected = logps(reference, "chosen"), logps(reference, "rejected")
    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    assert 0 < int((logits > 0).sum()) < 5  # the reward accuracy is neither 0 nor 1 for these models

    config = DPOTaskConfig(beta=beta, max_length=128, max_prompt_length=None, eval_batch_size=batch_size,
                           single_forward=single_forward)
    task = DPOTask(policy, copy.deepcopy(reference), tokenizer, rows, config, CPU)
    sizes = record_batches(task)
    metrics = task.evaluate()

    assert sizes == expected_batches(5, batch_size)
    assert len(task._reference_val_cache) == len(sizes)  # one cached reference entry per validation batch
    close = dict(rel=1e-5, abs=1e-6)
    assert metrics["val_pairs"] == 5
    assert metrics["val_loss"] == pytest.approx(float(F.softplus(-logits).mean()), **close)
    assert metrics["val_reward_accuracy"] == pytest.approx(float((logits > 0).double().mean()))
    assert metrics["val_reward_margin"] == pytest.approx(float(logits.mean()), **close)
    assert metrics["val_tie_fraction"] == 0.0
    assert metrics["val_policy_chosen_logp"] == pytest.approx(float(policy_chosen.mean()), **close)
    assert metrics["val_policy_rejected_logp"] == pytest.approx(float(policy_rejected.mean()), **close)
    assert metrics["val_reference_chosen_logp"] == pytest.approx(float(reference_chosen.mean()), **close)
    assert metrics["val_reference_rejected_logp"] == pytest.approx(float(reference_rejected.mean()), **close)
    assert task.evaluate() == metrics  # second evaluation reuses the cached reference values


# ---------------------------------------------------------------------------
# Shipped defaults
# ---------------------------------------------------------------------------

# (batch_size, grad_accum_steps, max_length, val_size, eval_every_steps): the training protocol is unchanged.
SHIPPED = [
    ("configs/sft.yaml", SFTTaskConfig, (2, 16, 512, 1000, 100)),
    ("configs/dpo_hh.yaml", DPOTaskConfig, (1, 32, 512, 1000, 100)),
    ("configs/dpo_ultrafeedback.yaml", DPOTaskConfig, (1, 32, 512, 1000, 100)),
]


@pytest.mark.parametrize("path, task_type, protocol", SHIPPED)
def test_shipped_validation_batches_are_no_larger_than_a_training_micro_batch(path, task_type, protocol):
    config = load_config(ROOT / path)
    task_config, loop = pick(config, task_type), pick(config, LoopConfig)
    assert (loop.batch_size, loop.grad_accum_steps, task_config.max_length, config["val_size"],
            loop.eval_every_steps) == protocol
    # With gradients disabled, a validation batch of at most the training micro-batch size needs less
    # memory than a training step on the same device.
    assert task_config.eval_batch_size == loop.batch_size
    assert task_type().eval_batch_size == task_config.eval_batch_size  # same value when a config omits the key


def test_nothing_that_launches_training_overrides_the_validation_batch_size():
    for path in ("scripts/train_sft.py", "scripts/train_dpo.py", "tools/build_notebooks.py",
                 "notebooks/01_train_kaggle.ipynb"):
        assert "eval_batch_size" not in (ROOT / path).read_text(encoding="utf-8"), path
