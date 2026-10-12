"""Dataset conversion, deterministic splitting, and the fixed evaluation files."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from alignment import data_prep
from alignment.prompt_format import TEMPLATES, format_prompt, trim_at_stop_strings
from tests.helpers import TemplateToyTokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_templates():
    assert format_prompt("  Say hi  ", "alpaca") == "### Instruction:\nSay hi\n\n### Response:\n"
    assert format_prompt("Say hi", "chatml") == "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n"
    assert set(TEMPLATES) == {"alpaca", "chatml"}
    with pytest.raises(ValueError):
        format_prompt("x", "unknown")
    assert trim_at_stop_strings(" Hello.\n### Instruction:\nNext question", "alpaca") == "Hello."
    assert trim_at_stop_strings("Hello.", "alpaca") == "Hello."


def test_ultrachat_conversion_takes_the_first_exchange_only():
    row = {"prompt": "p", "messages": [
        {"role": "user", "content": " What is 2+2? "}, {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3+3?"}, {"role": "assistant", "content": "6."}]}
    assert data_prep.ultrachat_row_to_example(row) == {"instruction": "What is 2+2?", "response": "4."}
    assert data_prep.ultrachat_row_to_example({"messages": [{"role": "user", "content": "only one"}]}) is None
    assert data_prep.ultrachat_row_to_example(
        {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]}) is None
    assert data_prep.ultrachat_row_to_example(
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "  "}]}) is None


def test_hh_conversion_keeps_only_single_turn_pairs():
    single = {"chosen": "\n\nHuman: How do I boil an egg?\n\nAssistant: Simmer it for eight minutes.",
              "rejected": "\n\nHuman: How do I boil an egg?\n\nAssistant: No idea."}
    assert data_prep.hh_row_to_pair(single) == {
        "instruction": "How do I boil an egg?", "chosen": "Simmer it for eight minutes.", "rejected": "No idea."}
    assert data_prep.parse_hh_transcript(single["chosen"]) == [
        ("Human", "How do I boil an egg?"), ("Assistant", "Simmer it for eight minutes.")]

    multi = {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: Help?\n\nAssistant: Sure.",
             "rejected": "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: Help?\n\nAssistant: No."}
    assert data_prep.hh_row_to_pair(multi) is None
    different_prompts = {"chosen": "\n\nHuman: A\n\nAssistant: x", "rejected": "\n\nHuman: B\n\nAssistant: y"}
    assert data_prep.hh_row_to_pair(different_prompts) is None
    identical = {"chosen": "\n\nHuman: A\n\nAssistant: x", "rejected": "\n\nHuman: A\n\nAssistant: x"}
    assert data_prep.hh_row_to_pair(identical) is None
    empty_reply = {"chosen": "\n\nHuman: A\n\nAssistant: x", "rejected": "\n\nHuman: A\n\nAssistant:"}
    assert data_prep.hh_row_to_pair(empty_reply) is None


def test_ultrafeedback_conversion_requires_single_turn_and_a_strict_preference():
    def row(chosen_rating, rejected_rating, chosen="Good answer.", rejected="Bad answer."):
        return {"source": "evol_instruct", "prompt": "Explain X.",
                "chosen": [{"role": "user", "content": "Explain X."}, {"role": "assistant", "content": chosen}],
                "rejected": [{"role": "user", "content": "Explain X."}, {"role": "assistant", "content": rejected}],
                "chosen-rating": chosen_rating, "rejected-rating": rejected_rating}

    assert data_prep.ultrafeedback_row_to_pair(row(5.0, 3.0)) == {
        "instruction": "Explain X.", "chosen": "Good answer.", "rejected": "Bad answer."}
    assert data_prep.ultrafeedback_row_to_pair(row(4.0, 4.0)) is None           # rating tie
    assert data_prep.ultrafeedback_row_to_pair(row(5.0, 3.0, rejected="Good answer.")) is None  # identical texts
    multi_turn = row(5.0, 3.0)
    multi_turn["chosen"] = multi_turn["chosen"] + multi_turn["chosen"]
    assert data_prep.ultrafeedback_row_to_pair(multi_turn) is None


def _candidates(count: int) -> list[dict]:
    return [{"instruction": f"Question number {index}", "response": f"Answer {index}"} for index in range(count)]


def test_select_split_is_deterministic_disjoint_and_excludes_eval_prompts():
    candidates = _candidates(60) + [{"instruction": "question   NUMBER 3", "response": "duplicate"}]
    exclude = {data_prep.normalize_text("Question number 7"), data_prep.normalize_text("Question number 8")}
    train, val, stats = data_prep.select_split(candidates, n_train=40, n_val=10, seed=42, exclude_keys=exclude)
    again_train, again_val, _ = data_prep.select_split(candidates, 40, 10, 42, exclude)
    assert train == again_train and val == again_val
    other_train, _, _ = data_prep.select_split(candidates, 40, 10, 43, exclude)
    assert train != other_train

    assert len(train) == 40 and len(val) == 10
    train_keys = {data_prep.normalize_text(row["instruction"]) for row in train}
    val_keys = {data_prep.normalize_text(row["instruction"]) for row in val}
    assert len(train_keys) == 40 and not train_keys & val_keys
    assert not (train_keys | val_keys) & exclude
    assert stats["kept"] == 50 and stats["seed"] == 42

    with pytest.raises(ValueError, match="survive filtering"):
        data_prep.select_split(candidates, n_train=100, n_val=10, seed=42)


def test_duplicates_and_eval_prompts_are_dropped_even_when_everything_is_needed():
    candidates = _candidates(10) + [{"instruction": "QUESTION number 0", "response": "dup"}]
    exclude = {data_prep.normalize_text("Question number 9")}
    train, val, stats = data_prep.select_split(candidates, n_train=7, n_val=2, seed=1, exclude_keys=exclude)
    keys = [data_prep.normalize_text(row["instruction"]) for row in train + val]
    assert len(set(keys)) == 9 and "question number 9" not in keys
    assert stats["kept"] == 9


def test_length_filter_rules():
    tokenizer = TemplateToyTokenizer()
    length_filter = data_prep.LengthFilter(tokenizer, template="alpaca", max_length=60, max_prompt_tokens=40)
    prompt_tokens = len(tokenizer.encode(format_prompt("Hi", "alpaca")))
    assert prompt_tokens == 35  # one token per character with the toy tokenizer
    assert length_filter.reason_to_drop({"instruction": "Hi", "response": "ok then"}) is None
    assert length_filter.reason_to_drop({"instruction": "Hi there my friend", "response": "ok"}) == "prompt_too_long"
    assert length_filter.reason_to_drop({"instruction": "Hi", "response": "x"}) == "response_too_short"
    fits = "y" * (60 - prompt_tokens - 1)
    assert length_filter.reason_to_drop({"instruction": "Hi", "response": fits}) is None
    assert length_filter.reason_to_drop({"instruction": "Hi", "response": fits + "y"}) == "response_does_not_fit"
    assert length_filter.reason_to_drop({"instruction": "Hi", "chosen": "fine", "rejected": fits + "y"}) == "response_does_not_fit"
    relaxed = data_prep.LengthFilter(tokenizer, template="alpaca", max_length=60, max_prompt_tokens=40,
                                     require_full_fit=False)
    assert relaxed.reason_to_drop({"instruction": "Hi", "response": fits + "yyyy"}) is None


def test_hh_balancing_fills_from_the_other_pool_when_one_is_short():
    harmless = [{"instruction": f"h{index}", "source": "Anthropic/hh-rlhf:harmless-base"} for index in range(5)]
    helpful = [{"instruction": f"p{index}", "source": "Anthropic/hh-rlhf:helpful-base"} for index in range(50)]
    # Without a minimum the pool shrinks until the requested 50/50 mix is possible.
    pool, stats = data_prep.balance_hh_candidates(harmless + helpful, 0.5, total=20, seed=42)
    assert len(pool) == 10 and stats["harmless_in_pool"] == 5 and stats["helpful_in_pool"] == 5
    # With a minimum the larger side fills the gap and the real mix is recorded.
    pool, stats = data_prep.balance_hh_candidates(harmless + helpful, 0.5, total=20, seed=42, minimum=20)
    assert len(pool) == 20
    assert stats["harmless_in_pool"] == 5 and stats["helpful_in_pool"] == 15
    assert stats["harmless_fraction_in_pool"] == 0.25
    balanced, stats = data_prep.balance_hh_candidates(harmless * 4 + helpful, 0.5, total=20, seed=42)
    assert stats["harmless_in_pool"] == 10 and stats["helpful_in_pool"] == 10


def test_with_ids_and_eval_key_loading(tmp_path):
    rows = data_prep.with_ids([{"instruction": "a"}, {"instruction": "b"}], "sft-train")
    assert [row["id"] for row in rows] == ["sft-train-000000", "sft-train-000001"]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"id": "1", "instruction": "  Hello   World "}]), encoding="utf-8")
    assert data_prep.load_eval_instruction_keys([path, tmp_path / "missing.json"]) == {"hello world"}


# ---------------------------------------------------------------------------
# Fixed evaluation files shipped with the repository
# ---------------------------------------------------------------------------


def _load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_hh_eval_set_is_the_unmodified_300_prompt_file():
    rows = _load("eval_sets/hh_rlhf_eval_300_seed42.json")
    assert len(rows) == 300 and len({row["id"] for row in rows}) == 300
    categories = Counter(row["category"] for row in rows)
    assert len(categories) == 10 and set(categories.values()) == {30}
    assert data_prep.sha256_file(ROOT / "eval_sets/hh_rlhf_eval_300_seed42.json").startswith("0731105407d06152")


def test_alpaca_eval_subset_is_300_prompts_selected_with_seed_42():
    rows = _load("eval_sets/alpaca_eval_300_seed42.json")
    assert len(rows) == 300 and len({row["id"] for row in rows}) == 300
    assert all(row["dataset"] == "alpaca_eval" and row["instruction"].strip() for row in rows)
    expected_indices = sorted(random.Random(42).sample(range(805), 300))
    assert [row["source_index"] for row in rows] == expected_indices
    assert [row["id"] for row in rows] == [f"alpaca-eval-{index:04d}" for index in expected_indices]


def test_alpaca_subset_function_is_reproducible():
    source = [{"instruction": f"instruction {index}", "dataset": "helpful_base"} for index in range(805)]
    first, second = data_prep.alpaca_eval_subset(source, 300, 42), data_prep.alpaca_eval_subset(source, 300, 42)
    assert first == second and len(first) == 300
    assert data_prep.alpaca_eval_subset(source, 300, 43) != first
    with pytest.raises(ValueError):
        data_prep.alpaca_eval_subset(source[:10], 300, 42)


def test_persona_files_are_consistent_and_disjoint():
    pairs = _load("data/persona_lora/ramsay_style_preferences.json")
    assert len(pairs) == 220 and all(set(row) == {"prompt", "chosen", "rejected"} for row in pairs)
    prompts = _load("eval_sets/persona_eval_prompts.json")
    assert Counter(row["category"] for row in prompts) == {"cooking": 10, "coding": 10, "general": 10, "safety": 10}
    assert len({row["id"] for row in prompts}) == 40
    training_keys = {data_prep.normalize_text(row["prompt"]) for row in pairs}
    assert not training_keys & {data_prep.normalize_text(row["instruction"]) for row in prompts}


def test_fixed_eval_sets_do_not_overlap_each_other():
    keys = [data_prep.load_eval_instruction_keys([ROOT / "eval_sets" / name]) for name in
            ("hh_rlhf_eval_300_seed42.json", "alpaca_eval_300_seed42.json", "persona_eval_prompts.json")]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
