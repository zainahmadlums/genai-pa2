"""Pairwise files, hidden identities, judge prompt/parsing, pacing, retries, resume and summaries."""

from __future__ import annotations

import json

import pytest

from alignment.judge import (
    JudgeRunner,
    build_judge_prompt,
    normalize_verdict,
    parse_judge_response,
    sign_test_p_value,
    summarize,
    wilson_interval,
)
from alignment.judge_backends import MockBackend, make_backend
from alignment.pairwise import build_pairs, judge_view, translate_verdict
from alignment.io_utils import read_json, read_jsonl

SFT_NAME, DPO_NAME = "qwen2.5-0.5b-sft", "qwen2.5-0.5b-dpo-hh"


def make_entries(generator: str, count: int = 12, suffix: str = "") -> list[dict]:
    return [{"id": f"p{index:02d}", "instruction": f"Instruction {index}", "output": f"{generator} answer {index}{suffix}",
             "generator": generator, "dataset": "hh_rlhf_style_eval", "category": "cat_a" if index % 2 else "cat_b",
             "expected_behavior": "Refuse harmful requests."} for index in range(count)]


def make_pairs(count: int = 12, seed: int = 42) -> list[dict]:
    return build_pairs(make_entries(SFT_NAME, count), make_entries(DPO_NAME, count), seed=seed,
                       comparison="dpo_hh_vs_sft", profile="safety")


class FakeClock:
    """Virtual time: ``sleep`` advances the clock and records every wait."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedBackend:
    """Replays scripted replies. A callable builds the reply from the prompt; an Exception is raised."""

    name = "scripted"

    def __init__(self, script, clock: FakeClock | None = None, call_seconds: float = 0.0):
        self.script = list(script)
        self.prompts: list[str] = []
        self.call_times: list[float] = []
        self.clock, self.call_seconds = clock, call_seconds

    def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.clock is not None:
            self.call_times.append(self.clock.now)
            self.clock.now += self.call_seconds
        step = self.script.pop(0) if self.script else all_winner("A")
        if isinstance(step, BaseException):
            raise step
        return step(prompt) if callable(step) else step


def all_winner(winner: str):
    def reply(prompt: str) -> str:
        count = prompt.count("===== End of item")
        return json.dumps({"judgments": [{"item": n, "winner": winner, "reason": "r"} for n in range(1, count + 1)]})
    return reply


def runner_for(backend, clock: FakeClock, **kwargs) -> JudgeRunner:
    return JudgeRunner(backend, "test/judge-model", sleep_fn=clock.sleep, clock=clock.clock, log=lambda *_: None, **kwargs)


# ---------------------------------------------------------------------------
# Pair construction and identity handling
# ---------------------------------------------------------------------------


def test_pairs_are_balanced_deterministic_and_keep_prompt_order():
    pairs = make_pairs()
    assert [pair["prompt_id"] for pair in pairs] == [f"p{index:02d}" for index in range(12)]
    positions = [pair["hidden"]["candidate_position"] for pair in pairs]
    assert positions.count("A") == positions.count("B") == 6
    assert positions != ["A", "B"] * 6  # shuffled, not alternating
    assert make_pairs() == pairs
    assert [p["hidden"]["candidate_position"] for p in make_pairs(seed=7)] != positions


def test_hidden_mapping_matches_the_displayed_responses():
    for pair in make_pairs():
        hidden = pair["hidden"]
        assert pair["response_a"].startswith(hidden["model_a"])
        assert pair["response_b"].startswith(hidden["model_b"])
        assert {hidden["model_a"], hidden["model_b"]} == {SFT_NAME, DPO_NAME}
        assert hidden["candidate"] == DPO_NAME and hidden["baseline"] == SFT_NAME
        candidate_model = hidden["model_a"] if hidden["candidate_position"] == "A" else hidden["model_b"]
        assert candidate_model == DPO_NAME


def test_verdict_translation_uses_the_stored_identity():
    pairs = make_pairs()
    as_a = next(pair for pair in pairs if pair["hidden"]["candidate_position"] == "A")
    as_b = next(pair for pair in pairs if pair["hidden"]["candidate_position"] == "B")
    assert translate_verdict(as_a, "A") == {"outcome": "candidate_win", "winner_model": DPO_NAME}
    assert translate_verdict(as_a, "B") == {"outcome": "baseline_win", "winner_model": SFT_NAME}
    assert translate_verdict(as_b, "B") == {"outcome": "candidate_win", "winner_model": DPO_NAME}
    assert translate_verdict(as_b, "A") == {"outcome": "baseline_win", "winner_model": SFT_NAME}
    assert translate_verdict(as_a, "tie") == {"outcome": "tie", "winner_model": None}
    assert translate_verdict(as_a, "unknown") == {"outcome": "unknown", "winner_model": None}


def test_pair_building_rejects_mismatched_files():
    sft, dpo = make_entries(SFT_NAME), make_entries(DPO_NAME)
    with pytest.raises(ValueError, match="not present in both"):
        build_pairs(sft, dpo[:-1])
    changed = [dict(entry) for entry in dpo]
    changed[3]["instruction"] = "something else"
    with pytest.raises(ValueError, match="Instruction mismatch"):
        build_pairs(sft, changed)
    with pytest.raises(ValueError, match="Duplicate id"):
        build_pairs(sft + sft[:1], dpo)


def test_judge_prompt_never_reveals_model_identity():
    pairs = make_pairs()
    for pair in pairs:
        pair["response_a"], pair["response_b"] = "First neutral text.", "Second neutral text."
    prompt = build_judge_prompt(pairs[:5], profile="safety")
    lowered = prompt.lower()
    for secret in (SFT_NAME, DPO_NAME, "dpo_hh_vs_sft", "sft", "dpo", "qwen", "checkpoint", "baseline", "candidate"):
        assert secret not in lowered
    assert "pair_id" not in prompt and "p00" not in prompt
    assert prompt.count("[Response A]") == 5 and prompt.count("[Response B]") == 5
    assert "appropriate refusal" in prompt and "Expected behavior" in prompt
    assert set(judge_view(pairs[0])) == {"instruction", "response_a", "response_b", "expected_behavior"}
    general = build_judge_prompt(pairs[:2], profile="general")
    assert "safety-sensitive" not in general and "helpfulness" in general
    assert "celebrity chef" in build_judge_prompt(pairs[:2], profile="persona")
    with pytest.raises(ValueError):
        build_judge_prompt(pairs[:2], profile="other")


def test_empty_and_long_responses_are_rendered_safely():
    pairs = make_pairs(2)
    pairs[0]["response_a"] = "   "
    pairs[1]["response_b"] = "x" * 50
    prompt = build_judge_prompt(pairs, max_response_chars=10)
    assert "(empty response)" in prompt and "x" * 10 + " [truncated]" in prompt


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_accepts_fenced_json_lists_and_positional_items():
    fenced = '```json\n{"judgments": [{"item": 2, "winner": "b", "reason": "x"}, {"item": 1, "winner": "Response A"}]}\n```'
    parsed = parse_judge_response(fenced, 2)
    assert parsed[1]["verdict"] == "A" and parsed[2]["verdict"] == "B"
    wrapped = 'Here you go: [{"winner": "tie"}, {"winner": "A"}, {"winner": "banana"}] hope that helps'
    parsed = parse_judge_response(wrapped, 3)
    assert [parsed[n]["verdict"] for n in (1, 2, 3)] == ["tie", "A", "unknown"]
    partial = parse_judge_response('{"judgments": [{"item": 3, "winner": "A"}, {"item": 9, "winner": "B"}]}', 3)
    assert set(partial) == {3}
    with pytest.raises(ValueError):
        parse_judge_response("I cannot decide.", 2)
    with pytest.raises(ValueError):
        parse_judge_response("", 2)
    # Unnumbered items are only trusted when the count matches.
    assert parse_judge_response('[{"winner": "A"}]', 3) == {}


def test_verdict_normalisation():
    assert [normalize_verdict(v) for v in ("A", " b ", "Response B", "TIE", "draw", "A is better", None, "C")] == [
        "A", "B", "B", "tie", "tie", "unknown", "unknown", "unknown"]


# ---------------------------------------------------------------------------
# Runner: batching, pacing, retries, checkpoints, resume
# ---------------------------------------------------------------------------


def test_runner_batches_of_five_and_waits_five_seconds_between_all_attempts(tmp_path):
    pairs, clock = make_pairs(12), FakeClock()
    script = [all_winner("A"), RuntimeError("503 overloaded"), "not json at all", all_winner("B"), all_winner("tie")]
    backend = ScriptedBackend(script, clock, call_seconds=2.0)
    summary = runner_for(backend, clock, batch_size=5, sleep_seconds=5.0, max_retries=3).run(
        pairs, tmp_path / "judged.jsonl")

    assert [prompt.count("[Response A]") for prompt in backend.prompts] == [5, 5, 5, 5, 2]
    # Every request starts at least 5 s after the previous one ended, retries included.
    gaps = [later - (earlier + 2.0) for earlier, later in zip(backend.call_times, backend.call_times[1:])]
    assert len(gaps) == 4 and all(gap >= 5.0 for gap in gaps)
    assert clock.sleeps == [5.0] * 4

    assert summary["complete"] and summary["failures"] == 0 and summary["unknown"] == 0
    assert summary["request_attempts"] == 5 and summary["failed_request_attempts"] == 2
    assert summary["retried_request_attempts"] == 2
    attempts = read_jsonl(tmp_path / "judged.attempts.jsonl")
    assert [(a["batch"], a["attempt"], a["status"]) for a in attempts] == [
        (1, 1, "ok"), (2, 1, "error"), (2, 2, "error"), (2, 3, "ok"), (3, 1, "ok")]
    assert "503 overloaded" in attempts[1]["error"]

    judged = read_jsonl(tmp_path / "judged.jsonl")
    assert [record["pair_id"] for record in judged] == [pair["pair_id"] for pair in pairs]
    assert [record["attempts"] for record in judged] == [1] * 5 + [3] * 5 + [1] * 2
    for record, pair in zip(judged, pairs):
        assert record["judge_model"] == "test/judge-model"
        assert record["candidate_position"] == pair["hidden"]["candidate_position"]
        if record["verdict"] in ("A", "B"):
            expected = "candidate_win" if record["verdict"] == pair["hidden"]["candidate_position"] else "baseline_win"
            assert record["outcome"] == expected
            assert record["winner_model"] == pair["hidden"][f"model_{record['verdict'].lower()}"]
    assert summary["ties"] == 2
    assert read_json(tmp_path / "judged.summary.json") == summary


def test_slow_requests_do_not_add_unneeded_sleep_but_fast_ones_do(tmp_path):
    pairs, clock = make_pairs(10), FakeClock()
    backend = ScriptedBackend([all_winner("A"), all_winner("A")], clock, call_seconds=0.5)
    runner = runner_for(backend, clock, batch_size=5, sleep_seconds=7.5)
    runner.run(pairs, tmp_path / "judged.jsonl")
    assert clock.sleeps == [7.5]  # measured from the end of the previous attempt
    assert backend.call_times[1] - (backend.call_times[0] + 0.5) == pytest.approx(7.5)


def test_unresolved_batches_go_to_the_failure_file_and_are_retried_on_resume(tmp_path):
    pairs, clock = make_pairs(10), FakeClock()
    output = tmp_path / "judged.jsonl"
    script = [all_winner("A"), RuntimeError("denied"), RuntimeError("denied"), RuntimeError("denied")]
    backend = ScriptedBackend(script, clock)
    summary = runner_for(backend, clock, batch_size=5, max_retries=3).run(pairs, output, reuse_judgments=True)
    assert len(backend.prompts) == 4
    assert summary["failures"] == 5 and summary["valid_judgments"] == 5 and summary["complete"]
    failures = read_jsonl(tmp_path / "judged.failures.jsonl")
    assert [f["pair_id"] for f in failures] == [pair["pair_id"] for pair in pairs[5:]]
    assert failures[0]["attempts"] == 3 and len(failures[0]["errors"]) == 3 and "denied" in failures[0]["errors"][0]
    assert len(read_jsonl(output)) == 5
    rates = summary["rates_over_all_pairs"]
    assert rates["failure"] == 0.5
    assert sum(value for value in rates.values()) == pytest.approx(1.0)

    # Rerun: finished pairs cost no calls, failed pairs are tried again.
    second = ScriptedBackend([all_winner("B")], clock)
    summary = runner_for(second, clock, batch_size=5).run(pairs, output, reuse_judgments=True)
    assert len(second.prompts) == 1 and second.prompts[0].count("[Response A]") == 5
    assert summary["failures"] == 0 and summary["valid_judgments"] == 10
    assert read_jsonl(tmp_path / "judged.failures.jsonl") == []
    assert [r["pair_id"] for r in read_jsonl(output)] == [pair["pair_id"] for pair in pairs]

    # A third run has nothing left to do and makes no request.
    third = ScriptedBackend([], clock)
    runner_for(third, clock).run(pairs, output, reuse_judgments=True)
    assert third.prompts == []


def test_partial_replies_retry_only_the_missing_pairs_and_invalid_winners_become_unknown(tmp_path):
    pairs, clock = make_pairs(5), FakeClock()

    def first(prompt):  # answers items 1-3, item 4 is invalid, item 5 is missing
        return json.dumps({"judgments": [{"item": 1, "winner": "A"}, {"item": 2, "winner": "B"},
                                         {"item": 3, "winner": "tie"}, {"item": 4, "winner": "the left one"}]})

    def second(prompt):  # two pairs left: resolve the second one (originally item 5), item 1 invalid again
        assert prompt.count("[Response A]") == 2
        return json.dumps({"judgments": [{"item": 1, "winner": "???"}, {"item": 2, "winner": "A"}]})

    backend = ScriptedBackend([first, second], clock)
    summary = runner_for(backend, clock, batch_size=5, max_retries=2).run(pairs, tmp_path / "judged.jsonl")
    judged = {record["pair_id"]: record for record in read_jsonl(tmp_path / "judged.jsonl")}
    assert judged[pairs[3]["pair_id"]]["outcome"] == "unknown" and judged[pairs[3]["pair_id"]]["winner_model"] is None
    assert judged[pairs[4]["pair_id"]]["verdict"] == "A" and judged[pairs[4]["pair_id"]]["attempts"] == 2
    assert summary["unknown"] == 1 and summary["failures"] == 0 and summary["valid_judgments"] == 4
    assert read_jsonl(tmp_path / "judged.attempts.jsonl")[0]["status"] == "partial"

    # Unknown pairs are kept on a normal resume and re-judged only when asked.
    again = ScriptedBackend([], clock)
    runner_for(again, clock).run(pairs, tmp_path / "judged.jsonl", reuse_judgments=True)
    assert again.prompts == []
    retry = ScriptedBackend([all_winner("B")], clock)
    summary = runner_for(retry, clock).run(pairs, tmp_path / "judged.jsonl", reuse_judgments=True, retry_unknown=True)
    assert len(retry.prompts) == 1 and summary["unknown"] == 0 and summary["valid_judgments"] == 5


def test_judgments_are_checkpointed_after_every_batch_and_survive_a_crash(tmp_path):
    pairs, clock = make_pairs(12), FakeClock()
    output = tmp_path / "judged.jsonl"

    class Crash(BaseException):
        pass

    backend = ScriptedBackend([all_winner("A"), all_winner("A"), Crash("notebook disconnected")], clock)
    with pytest.raises(Crash):
        runner_for(backend, clock, batch_size=5).run(pairs, output)
    assert len(read_jsonl(output)) == 10  # two finished batches are already on disk
    partial = read_json(tmp_path / "judged.summary.json")
    assert partial["complete"] is False and partial["not_attempted"] == 2

    resumed = ScriptedBackend([all_winner("B")], clock)
    summary = runner_for(resumed, clock, batch_size=5).run(pairs, output, reuse_judgments=True)
    assert len(resumed.prompts) == 1 and resumed.prompts[0].count("[Response A]") == 2
    assert summary["complete"] and summary["valid_judgments"] == 12


def test_reuse_ignores_judgments_for_changed_pairs_and_fresh_runs_back_up_old_files(tmp_path):
    pairs, clock = make_pairs(5), FakeClock()
    output = tmp_path / "judged.jsonl"
    runner_for(ScriptedBackend([all_winner("A")], clock), clock).run(pairs, output)

    changed = [dict(pair) for pair in make_pairs(5)]
    changed[2]["response_a"] = "a regenerated answer"
    backend = ScriptedBackend([all_winner("B")], clock)
    runner_for(backend, clock).run(changed, output, reuse_judgments=True)
    assert len(backend.prompts) == 1 and backend.prompts[0].count("[Response A]") == 1  # only the changed pair

    fresh = ScriptedBackend([all_winner("tie")], clock)
    summary = runner_for(fresh, clock).run(pairs, output, reuse_judgments=False)
    assert len(fresh.prompts) == 1 and summary["ties"] == 5
    assert (tmp_path / "judged.jsonl.bak").exists()


def test_max_batches_limits_a_first_check(tmp_path):
    pairs, clock = make_pairs(12), FakeClock()
    backend = ScriptedBackend([], clock)
    summary = runner_for(backend, clock, batch_size=5).run(pairs, tmp_path / "judged.jsonl", max_batches=1)
    assert len(backend.prompts) == 1 and summary["not_attempted"] == 7 and summary["complete"] is False


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def test_summary_counts_rates_and_denominators():
    pairs = make_pairs(10)
    verdicts = ["win", "win", "win", "win", "loss", "loss", "tie", "unknown", "failure", "pending"]
    judgments, failures = {}, {}
    for pair, verdict in zip(pairs, verdicts):
        position = pair["hidden"]["candidate_position"]
        other = "B" if position == "A" else "A"
        if verdict == "win":
            judgments[pair["pair_id"]] = {"outcome": "candidate_win", "verdict": position}
        elif verdict == "loss":
            judgments[pair["pair_id"]] = {"outcome": "baseline_win", "verdict": other}
        elif verdict in ("tie", "unknown"):
            judgments[pair["pair_id"]] = {"outcome": verdict, "verdict": verdict}
        elif verdict == "failure":
            failures[pair["pair_id"]] = {"pair_id": pair["pair_id"]}
    summary = summarize(pairs, judgments, failures)

    assert (summary["candidate_wins"], summary["baseline_wins"], summary["ties"], summary["unknown"],
            summary["failures"], summary["not_attempted"]) == (4, 2, 1, 1, 1, 1)
    assert summary["total_pairs"] == 10 and summary["valid_judgments"] == 7 and summary["complete"] is False
    assert summary["rates_over_all_pairs"] == {"candidate_win": 0.4, "baseline_win": 0.2, "tie": 0.1, "unknown": 0.1,
                                               "failure": 0.1, "not_attempted": 0.1}
    assert summary["rates_over_valid_judgments"]["candidate_win"] == pytest.approx(4 / 7)
    assert summary["candidate_win_rate_excluding_ties"] == pytest.approx(4 / 6)
    assert summary["candidate_win_rate_ties_as_half"] == pytest.approx(4.5 / 7)
    assert summary["candidate"] == DPO_NAME and summary["baseline"] == SFT_NAME
    by_category = summary["by_category"]
    assert set(by_category) == {"cat_a", "cat_b"}
    assert sum(block["total_pairs"] for block in by_category.values()) == 10
    position = summary["position_statistics"]
    assert position["candidate_shown_as_A"] + position["candidate_shown_as_B"] == 10
    assert position["verdict_A"] + position["verdict_B"] == 6


def test_summary_handles_empty_denominators_and_statistics_helpers():
    pairs = make_pairs(4)
    summary = summarize(pairs, {}, {pair["pair_id"]: {} for pair in pairs})
    assert summary["failures"] == 4 and summary["candidate_win_rate_excluding_ties"] is None
    assert summary["wilson_95ci_excluding_ties"] is None and summary["sign_test_p_value"] is None
    low, high = wilson_interval(8, 10)
    assert 0.44 < low < 0.50 and 0.94 < high < 0.98
    assert sign_test_p_value(5, 5) == 1.0
    assert sign_test_p_value(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_p_value(9, 1) == pytest.approx(2 * 11 / 1024)


def test_mock_backend_runs_the_whole_pipeline_offline(tmp_path):
    pairs = make_pairs(7)
    pairs[0]["response_a"], pairs[0]["response_b"] = "same", "same"
    backend = make_backend("mock", "mock")
    assert isinstance(backend, MockBackend)
    summary = JudgeRunner(backend, "mock", batch_size=5, sleep_seconds=0.0, log=lambda *_: None).run(
        pairs, tmp_path / "judged.jsonl")
    assert backend.calls == 2 and summary["backend"] == "mock"
    assert summary["complete"] and summary["ties"] >= 1 and summary["valid_judgments"] == 7
    # The candidate name is longer, so the length heuristic picks it whenever outputs differ.
    assert summary["candidate_wins"] == 6
