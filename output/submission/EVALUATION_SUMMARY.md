# Evaluation summary

Zain Abdullah Ahmad — 27100215

Repository: https://github.com/zainahmadlums/genai-pa2

These results come from completed Kaggle runs at evidence commit `20cb609f71e9d032b7b5ceee0a4644123c7f5ed9`.
Comparisons use fixed prompts, matched greedy decoding and saved outputs, judged by `google/gemini-3.6-flash`.
Candidate identities are hidden and A/B placement is balanced. Generation allows 256 new tokens, with repetition
penalty 1.0 and no added length penalty. Temperature and top-p are inactive under greedy decoding.

| Candidate vs baseline | Evaluation set / rubric | Pairs | Wins | Losses | Ties | Win rate excluding ties | Ties as half |
|---|---|---:|---:|---:|---:|---:|---:|
| HH DPO vs SFT | Synthetic helpful/harmless | 300 | 190 | 78 | 32 | 70.90% | 68.67% |
| UltraFeedback DPO vs SFT | Fixed AlpacaEval subset | 300 | 103 | 172 | 25 | 37.45% | 38.50% |
| Persona LoRA vs base | Joint persona and quality | 40 | 8 | 31 | 1 | 20.51% | 21.25% |
| Persona LoRA vs base | General quality | 40 | 4 | 32 | 4 | 11.11% | 15.00% |

Wins and losses are from the candidate's perspective. The rates are `W / (W + L)` and `(W + T / 2) / n`.
All 680 judgments completed with zero retries, unknown verdicts, failures or unattempted pairs. Invalid judgments
would be reported separately and excluded from valid-verdict win rates.

The archived comparison `persona_vs_base_style` uses a joint rubric requiring persona fidelity together with
helpfulness, correctness and safety. It does not isolate style. The persona rows use the same 40 prompts under
different rubrics and must not be treated as 80 independent prompts or pooled with other evaluation sets.

HH DPO wins the overall safety-oriented comparison. UltraFeedback DPO and persona LoRA lose their respective
comparisons despite improved validation preference metrics. These are single-seed, single-judge results;
relative wins do not establish absolute correctness or safety. The report provides category results, confidence
intervals, response diagnostics and examples.

Selected checkpoints: SFT step 938, HH DPO step 469, UltraFeedback DPO step 400, persona adapter step 70.

Evidence:

- `runs/Training/outputs/`: training metrics, final configurations and checkpoint-selection records.
- `runs/Generation/outputs/eval_generations_and_pairs.zip`: responses, blind pairs and generation settings.
- `runs/Judging/outputs/judged_outputs.zip`: decisions, comparison summaries and request logs.
- `output/pdf/report_evidence.json`: consolidated report calculations and examples.
- `output/pdf/figures/`: four training figures included in the report.

This AlpacaEval subset and local judging procedure do not constitute an official leaderboard result.
