# Report source

Author: Zain Abdullah Ahmad, 27100215.

The submission PDF is `27100215_PA2_Report.pdf` (eight pages, including references).

To rebuild, open a terminal in this folder and run twice:

```text
pdflatex -interaction=nonstopmode -halt-on-error -jobname=27100215_PA2_Report report.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=27100215_PA2_Report report.tex
```

Keep the `figures/` directory beside `report.tex`. A standard LaTeX installation with the packages named in the preamble is required. The source can also be imported with its figures into a LaTeX editor.

`report_evidence.json` consolidates source metrics, completion records, judge summaries, example pairs, generation settings and derived response statistics. Original artifacts remain under the repository's `runs/` directory at evidence commit `20cb609f71e9d032b7b5ceee0a4644123c7f5ed9`.

Figures use the saved metrics without extra smoothing. Training curves show the logged window means; validation points are unsmoothed. Dashed lines identify the selected checkpoints.

The repeated-four-gram diagnostic lowercases the output, splits on whitespace, constructs all contiguous four-word tuples, and flags outputs with at least 40 words and `1 - len(set(grams)) / len(grams) >= 0.5`. The evidence JSON stores the flagged IDs. Mean output length uses the original output's whitespace-separated word count. These are descriptive diagnostics, not judge labels.

The report follows the seven sections required by the manual. PDF text, page count, numeric summaries, and rendered pages were checked before delivery. No experiment was rerun or retuned for this report.

Prose revised on 16 October 2026 to focus on the final method, consequential configuration, results and interpretation. The PDF remains eight pages. Numerical results and plotted figures are unchanged. Implementation mechanics, the method tutorial and immaterial debugging history were removed; scientific limitations remain. The preceding versions are retained locally under `previous/2026-10-14/` and `previous/2026-10-16-before-report-focus/`.

The final submission review corrected the persona metric label to **joint persona and quality**, added the required LoRA rationale, clarified guide-versus-final choices and small-category uncertainty, and checked response statistics against raw generations. The archived `persona_vs_base_style` identifier is preserved as provenance. Its rubric includes usefulness, correctness and safety, so it is not a style-only measure.

## Reproducing response diagnostics

Run this from the repository root with Python (standard library only):

```python
import json
import statistics
import zipfile

with zipfile.ZipFile("runs/Generation/outputs/eval_generations_and_pairs.zip") as archive:
    for name in sorted(archive.namelist()):
        if "/generations/" not in name or not name.endswith(".json") or name.endswith(".meta.json"):
            continue
        rows = json.loads(archive.read(name))
        repeated = []
        for row in rows:
            words = row["output"].lower().split()
            grams = list(zip(words, words[1:], words[2:], words[3:]))
            if len(words) >= 40 and 1 - len(set(grams)) / len(grams) >= 0.5:
                repeated.append(row["id"])
        mean_words = statistics.mean(len(row["output"].split()) for row in rows)
        print(name, len(rows), mean_words, len(repeated), repeated)
```

Validation response lengths use the saved Qwen tokenizer, without special tokens or appended EOS. The 1,000 UltraFeedback validation pairs average 199.015 chosen tokens and 150.036 rejected tokens; HH averages 58.106 and 59.030. To reproduce, open `runs/Training/outputs/sft_run_outputs.zip`, load `outputs/sft/best/tokenizer.json` with `tokenizers.Tokenizer.from_str`, and average `len(tokenizer.encode(row[side], add_special_tokens=False).ids)` over `chosen` and `rejected` in `outputs/data/uf_val.jsonl` and `outputs/data/hh_val.jsonl`. These descriptive results are stored in `report_evidence.json`.
