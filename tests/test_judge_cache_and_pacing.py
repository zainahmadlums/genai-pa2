"""Regression tests: judgment reuse is bound to backend/model/protocol, replies are matched by explicit
item number, and one pacer spaces every request of a process (preflight, retries, consecutive comparisons)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from alignment import judge
from alignment.io_utils import read_json, read_jsonl, write_jsonl
from alignment.judge import JudgeCacheMismatch, JudgeRunner, RequestPacer, parse_judge_response
from tests.test_pairwise_and_judge import FakeClock, ScriptedBackend, all_winner, make_pairs

ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL = "google/gemini-3.6-flash"


def load_cli():
    spec = importlib.util.spec_from_file_location("pa2_judge_cli", ROOT / "scripts" / "judge_pairwise_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def named_backend(name: str, script, clock: FakeClock, call_seconds: float = 0.0) -> ScriptedBackend:
    backend = ScriptedBackend(script, clock, call_seconds=call_seconds)
    backend.name = name
    return backend


def make_runner(backend, clock: FakeClock, model: str = REAL_MODEL, **kwargs) -> JudgeRunner:
    return JudgeRunner(backend, model, sleep_fn=clock.sleep, clock=clock.clock, log=lambda *_: None, **kwargs)


# ---------------------------------------------------------------------------
# Finding 3: explicit item numbers are honoured; position is used only for fully un-numbered replies
# ---------------------------------------------------------------------------


def test_mixed_numbered_and_unnumbered_reply_never_assigns_by_position():
    parsed = parse_judge_response('[{"item": 2, "winner": "A"}, {"winner": "B"}]', 2)
    assert set(parsed) == {2} and parsed[2]["verdict"] == "A"  # pair 1 stays unanswered, it does not get "A"
    parsed = parse_judge_response('[{"winner": "B"}, {"item": 1, "winner": "A"}]', 2)
    assert set(parsed) == {1} and parsed[1]["verdict"] == "A"


def test_duplicate_out_of_range_and_invalid_item_numbers():
    duplicated = parse_judge_response(
        '[{"item": 1, "winner": "A"}, {"item": 1, "winner": "B"}, {"item": 2, "winner": "tie"}]', 2)
    assert set(duplicated) == {2} and duplicated[2]["verdict"] == "tie"  # conflicting answers for item 1: none used
    out_of_range = parse_judge_response(
        '[{"item": 0, "winner": "A"}, {"item": 3, "winner": "B"}, {"item": 2, "winner": "B"}]', 2)
    assert set(out_of_range) == {2} and out_of_range[2]["verdict"] == "B"
    # An unreadable number is not "no number": it must not switch the reply to positional matching.
    assert parse_judge_response('[{"item": "first", "winner": "A"}, {"winner": "B"}]', 2) == {}
    assert parse_judge_response('[{"item": true, "winner": "A"}, {"item": 2, "winner": "B"}]', 2).keys() == {2}
    reordered = parse_judge_response('[{"item": "#2", "winner": "A"}, {"item": 1.0, "winner": "B"}]', 2)
    assert {number: item["verdict"] for number, item in reordered.items()} == {1: "B", 2: "A"}


def test_positional_matching_only_for_fully_unnumbered_replies_of_the_right_length():
    positional = parse_judge_response('[{"winner": "A"}, {"winner": "B"}]', 2)
    assert {number: item["verdict"] for number, item in positional.items()} == {1: "A", 2: "B"}
    assert parse_judge_response('[{"winner": "A"}]', 2) == {}
    assert parse_judge_response('[{"winner": "A"}, {"winner": "B"}, {"winner": "A"}]', 2) == {}
    assert parse_judge_response('[{"winner": "A"}, "B"]', 2) == {}


def test_mixed_reply_retries_the_unidentified_pair_instead_of_misassigning_it(tmp_path):
    pairs, clock = make_pairs(2), FakeClock()

    def second(prompt):
        assert prompt.count("[Response A]") == 1  # only the pair that got no identifiable answer is asked again
        return '{"judgments": [{"item": 1, "winner": "tie"}]}'

    backend = ScriptedBackend(['[{"item": 2, "winner": "A"}, {"winner": "B"}]', second], clock)
    summary = make_runner(backend, clock, max_retries=2).run(pairs, tmp_path / "judged.jsonl")
    judged = {record["pair_id"]: record for record in read_jsonl(tmp_path / "judged.jsonl")}
    assert judged[pairs[1]["pair_id"]]["verdict"] == "A" and judged[pairs[1]["pair_id"]]["attempts"] == 1
    assert judged[pairs[0]["pair_id"]]["verdict"] == "tie" and judged[pairs[0]["pair_id"]]["attempts"] == 2
    assert summary["valid_judgments"] == 2 and summary["failures"] == 0


# ---------------------------------------------------------------------------
# Finding 1: reuse is bound to backend, judge model and protocol; provenance is kept
# ---------------------------------------------------------------------------


def test_mock_judgments_are_never_reused_or_relabelled_for_a_real_judge(tmp_path):
    pairs, clock = make_pairs(5), FakeClock()
    output = tmp_path / "judged.jsonl"
    mock = named_backend("mock", [all_winner("A")], clock)
    make_runner(mock, clock, model="mock").run(pairs, output, reuse_judgments=True)
    before = output.read_text(encoding="utf-8")
    assert read_json(tmp_path / "judged.summary.json")["backend"] == "mock"

    real = named_backend("kaggle_benchmarks", [all_winner("B")], clock)
    runner = make_runner(real, clock)
    assert runner.incompatible_cache(output)
    with pytest.raises(JudgeCacheMismatch, match="backend='mock'"):
        runner.run(pairs, output, reuse_judgments=True)
    # Nothing was sent, nothing was rewritten, and the summary still describes the mock run.
    assert real.prompts == [] and output.read_text(encoding="utf-8") == before
    assert read_json(tmp_path / "judged.summary.json")["judge_model"] == "mock"

    summary = runner.run(pairs, output, reuse_judgments=True, on_incompatible="invalidate")
    assert len(real.prompts) == 1 and real.prompts[0].count("[Response A]") == 5  # all pairs judged afresh
    assert (summary["backend"], summary["judge_model"]) == ("kaggle_benchmarks", REAL_MODEL)
    assert summary["reused_judgments"] == 0 and summary["judged_this_run"] == 5
    assert summary["protocol_id"] == runner.protocol_id and summary["protocol"]["batch_size"] == 5
    assert summary["request_attempts"] == 1  # the mock attempt is not counted for the real judge
    records = read_jsonl(output)
    assert {(r["backend"], r["judge_model"], r["protocol_id"]) for r in records} == {
        ("kaggle_benchmarks", REAL_MODEL, runner.protocol_id)}
    # The mock files are set aside, not deleted.
    assert (tmp_path / "judged.jsonl.stale1").read_text(encoding="utf-8") == before
    assert read_json(tmp_path / "judged.summary.json.stale1")["backend"] == "mock"
    assert {r["backend"] for r in read_jsonl(tmp_path / "judged.attempts.jsonl.stale1")} == {"mock"}
    assert {r["backend"] for r in read_jsonl(tmp_path / "judged.attempts.jsonl")} == {"kaggle_benchmarks"}

    # The same judge and protocol again: everything is reused and reported as reused.
    again = named_backend("kaggle_benchmarks", [], clock)
    summary = make_runner(again, clock).run(pairs, output, reuse_judgments=True)
    assert again.prompts == [] and summary["reused_judgments"] == 5 and summary["judged_this_run"] == 0


@pytest.mark.parametrize("change", ["judge_model", "backend", "batch_size", "max_response_chars", "rubric"])
def test_any_protocol_change_blocks_reuse_before_a_request_is_sent(tmp_path, monkeypatch, change):
    pairs, clock = make_pairs(5), FakeClock()
    output = tmp_path / "judged.jsonl"
    make_runner(named_backend("kaggle_benchmarks", [all_winner("A")], clock), clock).run(pairs, output)

    settings = {"backend": "kaggle_benchmarks", "model": REAL_MODEL, "batch_size": 5, "max_response_chars": 6000}
    if change == "judge_model":
        settings["model"] = "google/another-judge"
    elif change == "backend":
        settings["backend"] = "openai_proxy"
    elif change == "rubric":
        monkeypatch.setattr(judge, "RUBRIC_SHA256", "a-different-rubric")
    else:
        settings[change] = settings[change] + 1
    backend = named_backend(settings["backend"], [all_winner("B")], clock)
    runner = make_runner(backend, clock, model=settings["model"], batch_size=settings["batch_size"],
                         max_response_chars=settings["max_response_chars"])
    with pytest.raises(JudgeCacheMismatch):
        runner.run(pairs, output, reuse_judgments=True)
    assert backend.prompts == []
    assert {record["verdict"] for record in read_jsonl(output)} == {"A"}


def test_pacing_and_retry_settings_are_not_part_of_the_protocol(tmp_path):
    pairs, clock = make_pairs(5), FakeClock()
    output = tmp_path / "judged.jsonl"
    make_runner(named_backend("kaggle_benchmarks", [all_winner("A")], clock), clock).run(pairs, output)
    backend = named_backend("kaggle_benchmarks", [], clock)
    summary = make_runner(backend, clock, sleep_seconds=9.0, max_retries=1).run(pairs, output, reuse_judgments=True)
    assert backend.prompts == [] and summary["reused_judgments"] == 5


def test_legacy_judgments_without_protocol_information_are_rejected_explicitly(tmp_path):
    pairs, clock = make_pairs(5), FakeClock()
    output, attempts = tmp_path / "judged.jsonl", tmp_path / "judged.attempts.jsonl"
    make_runner(named_backend("kaggle_benchmarks", [all_winner("A")], clock), clock).run(pairs, output)
    for path in (output, attempts):  # what files written before protocol tracking look like
        write_jsonl(path, [{key: value for key, value in record.items() if key not in ("protocol_id", "backend")}
                           for record in read_jsonl(path)])

    backend = named_backend("kaggle_benchmarks", [all_winner("B")], clock)
    runner = make_runner(backend, clock)
    with pytest.raises(JudgeCacheMismatch, match="legacy record without protocol information"):
        runner.run(pairs, output, reuse_judgments=True)
    assert backend.prompts == []
    summary = runner.run(pairs, output, reuse_judgments=True, on_incompatible="invalidate")
    assert len(backend.prompts) == 1 and summary["judged_this_run"] == 5
    assert (tmp_path / "judged.jsonl.stale1").exists()
    with pytest.raises(ValueError, match="on_incompatible"):
        runner.run(pairs, output, on_incompatible="ignore")


def test_an_incompatible_attempt_log_blocks_reuse_even_without_saved_judgments(tmp_path):
    pairs, clock = make_pairs(5), FakeClock()
    output = tmp_path / "judged.jsonl"
    failing = named_backend("mock", [RuntimeError("x"), RuntimeError("x"), RuntimeError("x")], clock)
    summary = make_runner(failing, clock, model="mock").run(pairs, output)
    assert summary["failures"] == 5 and read_jsonl(output) == []
    backend = named_backend("kaggle_benchmarks", [all_winner("A")], clock)
    with pytest.raises(JudgeCacheMismatch):
        make_runner(backend, clock).run(pairs, output, reuse_judgments=True)
    assert backend.prompts == []


def test_cli_stops_before_the_preflight_on_incompatible_results_and_can_invalidate(tmp_path):
    cli = load_cli()
    pairs_path, output = tmp_path / "pairs.jsonl", tmp_path / "judged" / "x_judged.jsonl"
    write_jsonl(pairs_path, make_pairs(5))
    base = ["--pairs_path", str(pairs_path), "--output_path", str(output), "--reuse_judgments"]
    cli.main([*base, "--backend", "mock", "--judge_model", "mock", "--sleep_seconds", "0"])

    clock = FakeClock()
    pacer = RequestPacer(clock=clock.clock, sleep_fn=clock.sleep, wall_clock=clock.clock)
    backend = named_backend("kaggle_benchmarks", ["OK", all_winner("A")], clock)
    with pytest.raises(SystemExit, match="invalidate_incompatible"):
        cli.main(base, backend=backend, pacer=pacer)
    assert backend.prompts == []  # not even the availability check was sent
    assert read_json(output.with_name("x_judged.summary.json"))["backend"] == "mock"

    summary = cli.main([*base, "--invalidate_incompatible"], backend=backend, pacer=pacer)
    assert len(backend.prompts) == 2 and backend.prompts[0] == "Reply with the single word OK."
    assert summary["backend"] == "kaggle_benchmarks" and summary["judge_model"] == cli.DEFAULT_JUDGE_MODEL
    assert summary["reused_judgments"] == 0 and summary["judged_this_run"] == 5
    assert output.with_name("x_judged.jsonl.stale1").exists()


# ---------------------------------------------------------------------------
# Finding 4: one pacer for the whole process
# ---------------------------------------------------------------------------


def test_interval_holds_across_preflight_retries_and_consecutive_comparisons(tmp_path, monkeypatch):
    """Two comparisons run back to back from one process, as the judging notebook does."""

    cli = load_cli()
    clock = FakeClock()
    # main() is called WITHOUT a pacer argument: it must pick up the process-wide one.
    monkeypatch.setattr(judge, "_SHARED_PACER",
                        RequestPacer(clock=clock.clock, sleep_fn=clock.sleep, wall_clock=clock.clock))
    backend = named_backend("kaggle_benchmarks", [
        "OK", RuntimeError("503"), all_winner("A"),   # comparison 1: preflight, failed attempt, retry
        "OK", all_winner("B"),                        # comparison 2: preflight, first attempt
    ], clock, call_seconds=1.0)

    for name in ("first", "second"):
        pairs_path = tmp_path / f"{name}_pairs.jsonl"
        write_jsonl(pairs_path, [{**pair, "pair_id": f"{name}:{pair['prompt_id']}", "comparison": name}
                                 for pair in make_pairs(5)])
        summary = cli.main(["--pairs_path", str(pairs_path), "--output_path",
                            str(tmp_path / "judged" / f"{name}_judged.jsonl"), "--reuse_judgments",
                            "--sleep_seconds", "5"], backend=backend)
        assert summary["complete"] and summary["valid_judgments"] == 5

    assert len(backend.call_times) == 5
    gaps = [later - (earlier + 1.0) for earlier, later in zip(backend.call_times, backend.call_times[1:])]
    # end of one request -> start of the next, for every consecutive pair of requests, including
    # preflight -> first attempt, attempt -> retry, and last request of comparison 1 -> preflight of comparison 2.
    assert gaps == pytest.approx([5.0, 5.0, 5.0, 5.0])
    assert clock.sleeps == pytest.approx([5.0] * 4)
    assert (tmp_path / "judged" / judge.PACER_STATE_FILE).exists()


def test_values_below_five_seconds_are_raised_for_real_backends(tmp_path):
    cli = load_cli()
    clock = FakeClock()
    pacer = RequestPacer(clock=clock.clock, sleep_fn=clock.sleep, wall_clock=clock.clock)
    backend = named_backend("kaggle_benchmarks", ["OK", all_winner("A")], clock)
    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs_path, make_pairs(5))
    summary = cli.main(["--pairs_path", str(pairs_path), "--output_path", str(tmp_path / "judged.jsonl"),
                        "--sleep_seconds", "0"], backend=backend, pacer=pacer)
    assert summary["sleep_seconds"] == 5.0 and clock.sleeps == [5.0]


def test_pacer_state_file_carries_the_interval_into_a_new_process(tmp_path):
    clock = FakeClock()
    state = tmp_path / "judged" / judge.PACER_STATE_FILE
    first = RequestPacer(clock=clock.clock, sleep_fn=clock.sleep, wall_clock=clock.clock, state_path=state)
    assert first.wait(5.0) == 0.0  # nothing before the very first request
    first.mark()
    clock.now += 2.0
    fresh_process = RequestPacer(clock=clock.clock, sleep_fn=clock.sleep, wall_clock=clock.clock, state_path=state)
    assert fresh_process.wait(5.0) == pytest.approx(3.0) and clock.sleeps == [pytest.approx(3.0)]
    clock.now += 60.0
    assert fresh_process.wait(5.0) == 0.0
    state.write_text("not a number", encoding="utf-8")
    assert RequestPacer(clock=clock.clock, sleep_fn=clock.sleep, wall_clock=clock.clock, state_path=state).wait(5.0) == 0.0
    without_state = RequestPacer(clock=clock.clock, sleep_fn=clock.sleep)
    assert without_state.request(5.0, lambda: "reply") == "reply"
    assert without_state.wait(5.0) == pytest.approx(5.0)


def test_request_marks_the_pacer_even_when_the_call_fails():
    clock = FakeClock()
    pacer = RequestPacer(clock=clock.clock, sleep_fn=clock.sleep)

    def failing():
        raise RuntimeError("denied")

    with pytest.raises(RuntimeError):
        pacer.request(5.0, failing)
    assert pacer.wait(5.0) == pytest.approx(5.0)  # the failed request still counts for pacing


def test_stale_archives_never_overwrite_each_other(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "a.summary.json"
    for round_number in (1, 2):
        first.write_text(json.dumps({"round": round_number}), encoding="utf-8")
        second.write_text(json.dumps({"round": round_number}), encoding="utf-8")
        moved = judge.archive_stale([first, second, tmp_path / "missing.jsonl"])
        assert [path.name for path in moved] == [f"a.jsonl.stale{round_number}", f"a.summary.json.stale{round_number}"]
    assert json.loads((tmp_path / "a.jsonl.stale1").read_text(encoding="utf-8")) == {"round": 1}
    assert not first.exists()
