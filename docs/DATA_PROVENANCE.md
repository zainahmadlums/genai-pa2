# Data provenance

All training data is downloaded on Kaggle by `scripts/prepare_data.py` from public Hugging Face datasets pinned to a
commit. The script writes a `*_provenance.json` next to each output with the exact counts, filters, seed and file
hashes of that run; quote those files in the report. The numbers of rows surviving each filter are **not known until
the script has run**.

| Track | Dataset (commit) | Licence | Split / files | Conversion to single turn |
|---|---|---|---|---|
| SFT | `HuggingFaceH4/ultrachat_200k` (`8049631c405a`) | MIT | `train_sft` (207,865 conversations) | first user message -> instruction, first assistant message -> response |
| HH DPO | `Anthropic/hh-rlhf` (`09be8c5bbc57`) | MIT | `train.jsonl.gz` of `harmless-base`, `helpful-base`, `helpful-online`, `helpful-rejection-sampled` | transcripts with exactly one `Human` and one `Assistant` turn on both sides and an identical `Human` turn; assistant texts are `chosen` / `rejected` |
| UltraFeedback DPO | `argilla/ultrafeedback-binarized-preferences-cleaned` (`770076f077c4`) | MIT | `train` (60,917 rows) | two-message conversations with `chosen-rating > rejected-rating`; last assistant message of each side |
| Persona | `data/persona_lora/ramsay_style_preferences.json` (starter) | course material | 220 pairs | 200 train / 20 validation after a seed-42 shuffle; formatted as TRL conversational preference rows |

What was checked while this code was written:

- Column names and types of all three datasets, through the Hugging Face dataset viewer API.
- **HH preparation was run end to end** on a development machine (`prepare_data.py --task hh`, real dataset at the
  pinned commit, real Qwen tokenizer, seed 42): 49,241 single-turn pairs are available (12,189 from `harmless-base`,
  37,052 from the three helpful subsets); the 50/50 pool holds 24,378 pairs; 16,000 were kept after dropping 304
  duplicate instructions, 48 over-long prompts, 46 too-short responses, 33 pairs that do not fit in 512 tokens and
  2 prompts that appear in the fixed evaluation sets. The resulting 15,000 training pairs contain 7,423
  `harmless-base` pairs (49.5%); validation contains 497 of 1,000. A Kaggle run with the same seed and library
  versions should reproduce these counts; confirm them in `outputs/data/hh_provenance.json`.
- **UltraFeedback preparation was run end to end** the same way (`--task ultrafeedback`): 60,846 of the 60,917 rows
  convert to single-turn pairs; reaching 16,000 kept pairs required examining 31,274 shuffled candidates, of which
  8,600 were dropped because a response does not fit in 512 tokens, 5,958 for prompts over 256 tokens, 711 for
  too-short responses, 3 for appearing in the fixed evaluation sets and 2 as duplicates (about 51% pass).
- **UltraChat (SFT) preparation was not run**: the full download (about 730 MB) was not made. Its converter was
  checked on 400 real rows fetched through the dataset viewer API (399 converted). A rough character-based estimate
  on that sample (tokens taken as characters / 3.5) keeps about 35% of rows, which would leave well over the
  31,000 needed out of 207,865; the exact counts appear in `outputs/data/sft_provenance.json` after
  `prepare_data.py --task sft` has run on Kaggle.

Requiring full fit in 512 tokens removes the longer answers, so both the SFT targets and the UltraFeedback pairs
are shorter on average than their source datasets. State this in the report; `--allow_truncated_responses`
keeps the long examples instead (their targets are then cut at 512 tokens and lose EOS).

## Selection procedure (identical for SFT, HH and UltraFeedback)

Implemented in `alignment/data_prep.py::select_split` with seed 42:

1. Shuffle the single-turn candidates with `random.Random(42)`.
2. Walk the shuffled list and drop a candidate when
   - its instruction (lower-cased, whitespace-collapsed) was already kept (duplicate);
   - its instruction equals a prompt in any file under `eval_sets/` (the fixed evaluation prompts never enter training);
   - the formatted prompt is longer than 256 tokens (Qwen tokenizer, project template);
   - a response has fewer than 2 tokens;
   - prompt + response + EOS does not fit in 512 tokens, so no target is truncated and every example supervises
     EOS (disable with `--allow_truncated_responses`).
3. The first 1,000 kept rows are validation, the next 30,000 (SFT) or 15,000 (DPO) are training. Train and
   validation therefore never share an instruction.

HH only: before step 1 the pool is drawn with a target of 50% `harmless-base` pairs and 50% helpful pairs
(`--hh_harmless_fraction`), because the HH track is about safety and refusals. If one side has too few single-turn
pairs the other fills the gap; the provenance file records the actual mix.

The script stops with an error instead of silently producing fewer rows when a dataset cannot supply the requested
count after filtering.

## Fixed evaluation sets

| File | Content | Origin |
|---|---|---|
| `eval_sets/hh_rlhf_eval_300_seed42.json` | 300 synthetic helpful/harmless prompts, 10 categories x 30, with `expected_behavior` | starter repository, unmodified (sha256 `0731105407d06152...`) |
| `eval_sets/alpaca_eval_300_seed42.json` | 300 AlpacaEval instructions | `tatsu-lab/alpaca_eval`, file `alpaca_eval.json` (805 prompts) at commit `2edc6fad8be6`; indices `sorted(random.Random(42).sample(range(805), 300))`; ids `alpaca-eval-<source index>` |
| `eval_sets/persona_eval_prompts.json` | 40 prompts: 10 cooking, 10 coding, 10 general, 10 safety-sensitive | written for this project; none appears in the persona training file (checked by a test) |

The assignment fixes the AlpacaEval subset only as "300 prompts, seed 42" and does not publish the staff sampling
code, so the staff subset may contain different prompts. The selection rule above is deterministic and reproducible
with `python scripts/make_alpaca_eval_subset.py --verify`. AlpacaEval is licensed CC BY-NC 4.0; only instructions
(no reference outputs) are stored here.
