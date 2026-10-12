"""Regression tests for the training loop: metric records after a rolled-back resume, best-checkpoint
bookkeeping, and faithful accounting of optimizer steps that were skipped (non-finite gradients or losses)."""

from __future__ import annotations

import pytest
import torch

from alignment.engine import _reconcile_best_checkpoint, _trim_metrics, run_training
from alignment.io_utils import read_jsonl
from alignment.sft_trainer import SFTTask, SFTTaskConfig
from alignment.train_utils import AmpSettings, load_checkpoint_meta
from tests.helpers import TemplateToyTokenizer
from tests.test_training_loops import SFT_ROWS, loop_config, quiet
from tests.toy_lm import ToyCausalLM

CPU = torch.device("cpu")


def fresh_sft():
    tokenizer = TemplateToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size)
    task = SFTTask(model, tokenizer, SFT_ROWS[8:], SFTTaskConfig(max_length=64), CPU)
    return model, task


def wrap_loss(task, on_call: int, change):
    """Replace ``task.loss`` so that its ``on_call``-th call (1-based) goes through ``change(loss)``."""

    original, calls = task.loss, {"count": 0}

    def loss(batch):
        calls["count"] += 1
        value, stats = original(batch)
        return (change(value), stats) if calls["count"] == on_call else (value, stats)

    task.loss = loss


def records_of(run_dir, kind):
    return [record for record in read_jsonl(run_dir / "metrics.jsonl") if record["type"] == kind]


# ---------------------------------------------------------------------------
# Finding 1: resume from a snapshot that is older than the last metric records
# ---------------------------------------------------------------------------


def test_crash_after_a_snapshot_leaves_no_duplicate_metric_records(tmp_path):
    """Snapshot every 2 steps, log and evaluate every step, crash while computing step 4."""

    settings = dict(batch_size=1, grad_accum_steps=2, epochs=2, learning_rate=0.05, resume_every_steps=2,
                    log_every_steps=1, eval_every_steps=1)
    rows = SFT_ROWS[:8]  # 4 optimizer steps per epoch, 8 in total

    full_model, full_task = fresh_sft()
    run_training(loop_config(tmp_path / "full", **settings), full_model, rows, full_task, CPU, log=quiet)

    model, task = fresh_sft()

    def crash(_loss):
        raise RuntimeError("session lost")

    wrap_loss(task, on_call=7, change=crash)  # first micro-batch of optimizer step 4
    with pytest.raises(RuntimeError, match="session lost"):
        run_training(loop_config(tmp_path / "part", **settings), model, rows, task, CPU, log=quiet)
    run_dir = tmp_path / "part" / "run"
    # Step 3 was logged and evaluated after the last snapshot (step 2) and became the best checkpoint.
    assert [record["step"] for record in records_of(run_dir, "train")] == [1, 2, 3]
    assert [record["step"] for record in records_of(run_dir, "eval")] == [0, 1, 2, 3]
    assert load_checkpoint_meta(run_dir / "best")["step"] == 3

    model, task = fresh_sft()  # a new process: nothing in memory
    summary = run_training(loop_config(tmp_path / "part", **settings), model, rows, task, CPU, resume=True, log=quiet)

    assert [record["step"] for record in records_of(run_dir, "train")] == list(range(1, 9))  # no second "3"
    evals = records_of(run_dir, "eval")
    assert [record["step"] for record in evals] == list(range(0, 9))
    assert [record["tag"] for record in evals] == ["initial"] + ["periodic"] * 7 + ["final"]
    (resume_record,) = records_of(run_dir, "resume")
    assert resume_record["step"] == 2 and resume_record["trimmed_records"] == 2  # train + eval record of step 3
    # best/ on disk came from step 3, later than the snapshot (which knew step 2): adopted and recorded.
    assert resume_record["best_step_in_snapshot"] == 2 and resume_record["best_checkpoint_step"] == 3
    assert resume_record["best_checkpoint"] == "adopted_from_disk"

    # Same training as the uninterrupted run, and the summary agrees with the checkpoint on disk.
    torch.testing.assert_close(model.logit_bias, full_model.logit_bias, rtol=1e-6, atol=1e-7)
    best_meta = load_checkpoint_meta(run_dir / "best")
    assert summary["completed"] and summary["best_step"] == best_meta["step"]
    assert summary["best_metrics"] == best_meta["metrics"]
    assert summary["best_metrics"]["val_loss"] == min(record["val_loss"] for record in evals[1:])
    assert (summary["optimizer_steps"], summary["attempted_steps"], summary["skipped_steps"]) == (8, 8, 0)


def test_trim_removes_later_records_and_a_torn_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"type": "eval", "step": 0}\n{"type": "train", "step": 1}\n{"type": "train", "step": 2}\n'
                    '{"type": "eval", "step": 2}\n{"type": "train", "step": 3}\n{"type": "tra', encoding="utf-8")
    assert _trim_metrics(path, 2) == 2
    assert [record["step"] for record in read_jsonl(path)] == [0, 1, 2, 2]
    assert _trim_metrics(path, 2) == 0 and _trim_metrics(tmp_path / "missing.jsonl", 2) == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_best_bookkeeping_is_reset_when_the_checkpoint_on_disk_cannot_be_trusted(tmp_path):
    class Task:
        @staticmethod
        def selection_key(metrics):
            return (-metrics["val_loss"],)

    state = {"best_key": [-1.5], "best_step": 2, "best_metrics": {"val_loss": 1.5}}
    note = _reconcile_best_checkpoint(tmp_path, state, Task)  # no best/ directory at all
    assert note["best_checkpoint"] == "reset" and note["best_step_in_snapshot"] == 2
    assert state == {"best_key": None, "best_step": None, "best_metrics": None}


# ---------------------------------------------------------------------------
# Finding 3: attempted vs applied optimizer steps
# ---------------------------------------------------------------------------

SKIP_SETTINGS = dict(batch_size=1, grad_accum_steps=1, epochs=1, learning_rate=0.1, schedule="linear",
                     eval_at_start=False, eval_every_steps=0, log_every_steps=1)
EXPECTED_LR = [0.1, 0.075, 0.075, 0.05]  # linear schedule over 4 steps, advanced only by applied updates


def nan_gradients(model):
    """Finite loss whose gradient is NaN: d/dx sqrt(x) at 0 is inf, multiplied by the 0 of ``sum * 0``."""

    return lambda loss: loss + torch.sqrt(model.logit_bias.sum() * 0.0)


def run_with_skip(tmp_path, name, **overrides):
    model, task = fresh_sft()
    wrap_loss(task, on_call=2, change=nan_gradients(model))
    config = loop_config(tmp_path / name, **{**SKIP_SETTINGS, **overrides})
    summary = run_training(config, model, SFT_ROWS[:4], task, CPU, log=quiet)
    return model, summary, records_of(tmp_path / name / "run", "train")


def test_non_finite_gradients_are_not_applied_and_are_reported(tmp_path):
    model, summary, records = run_with_skip(tmp_path, "skip")

    assert torch.isfinite(model.logit_bias).all()  # the NaN gradient never reached the weights
    assert (summary["attempted_steps"], summary["optimizer_steps"], summary["skipped_steps"]) == (4, 3, 1)
    assert summary["skipped_nonfinite"] == 0 and summary["completed"]
    assert [record["step"] for record in records] == [1, 2, 3, 4]
    assert [record["applied_steps"] for record in records] == [1, 1, 2, 3]
    assert [record["skipped_steps"] for record in records] == [0, 1, 1, 1]
    assert [record["lr"] for record in records] == pytest.approx(EXPECTED_LR)
    assert records[1]["grad_norm"] is None and records[1]["loss"] is not None  # the loss itself was finite
    assert all(record["grad_norm"] > 0 for index, record in enumerate(records) if index != 1)


def test_a_skipped_step_does_not_bias_the_averaged_gradient_norm(tmp_path):
    clean_model, clean_task = fresh_sft()
    run_training(loop_config(tmp_path / "clean", **SKIP_SETTINGS), clean_model, SFT_ROWS[:4], clean_task, CPU, log=quiet)
    first_step_norm = records_of(tmp_path / "clean" / "run", "train")[0]["grad_norm"]

    _, _, records = run_with_skip(tmp_path, "skip", log_every_steps=2)
    window = records[0]  # covers step 1 (applied) and step 2 (skipped)
    assert window["step"] == 2 and window["applied_steps"] == 1 and window["skipped_steps"] == 1
    assert window["grad_norm"] == pytest.approx(first_step_norm)  # not first_step_norm / 2


class FakeGradScaler:
    """GradScaler semantics without CUDA: scale the loss, unscale and inspect gradients, skip the
    optimizer step when they are not finite and halve the scale afterwards."""

    def __init__(self):
        self.scale_value, self.found_inf, self.optimizer_steps, self.skipped = 8.0, False, 0, 0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        return loss * self.scale_value

    def unscale_(self, optimizer):
        self.found_inf = False
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)
                    self.found_inf = self.found_inf or not bool(torch.isfinite(parameter.grad).all())

    def step(self, optimizer):
        if self.found_inf:
            self.skipped += 1
        else:
            optimizer.step()
            self.optimizer_steps += 1

    def update(self):
        if self.found_inf:
            self.scale_value *= 0.5

    def state_dict(self):
        return {"scale": self.scale_value}

    def load_state_dict(self, state):
        self.scale_value = state["scale"]


def test_steps_skipped_by_a_grad_scaler_are_counted_as_skipped(tmp_path, monkeypatch):
    reference_model, _, _ = run_with_skip(tmp_path, "no_scaler")

    scaler = FakeGradScaler()
    monkeypatch.setattr(AmpSettings, "make_scaler", lambda self: scaler)
    model, summary, records = run_with_skip(tmp_path, "scaler")

    # The loop's counts are exactly what the scaler did.
    assert scaler.optimizer_steps == summary["optimizer_steps"] == 3
    assert scaler.skipped == summary["skipped_steps"] == 1 and summary["attempted_steps"] == 4
    assert scaler.scale_value == 4.0
    assert [record["applied_steps"] for record in records] == [1, 1, 2, 3]
    assert [record["lr"] for record in records] == pytest.approx(EXPECTED_LR)  # scheduler not advanced by the skip
    assert records[1]["grad_norm"] is None
    assert torch.isfinite(model.logit_bias).all()
    torch.testing.assert_close(model.logit_bias, reference_model.logit_bias, rtol=1e-4, atol=1e-6)


def test_a_group_without_any_finite_loss_is_a_skipped_step(tmp_path):
    model, task = fresh_sft()
    wrap_loss(task, on_call=2, change=lambda loss: loss * float("inf"))
    summary = run_training(loop_config(tmp_path, **SKIP_SETTINGS), model, SFT_ROWS[:4], task, CPU, log=quiet)
    records = records_of(tmp_path / "run", "train")
    assert (summary["attempted_steps"], summary["optimizer_steps"], summary["skipped_steps"]) == (4, 3, 1)
    assert summary["skipped_nonfinite"] == 1
    assert records[1]["loss"] is None and records[1]["grad_norm"] is None  # nothing finite to average: null, not 0
    assert [record["lr"] for record in records] == pytest.approx(EXPECTED_LR)
    assert torch.isfinite(model.logit_bias).all()
