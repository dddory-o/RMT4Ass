# Kazakh prompt morphological complexity and LLM answer quality (KazQAD pilot) - Assignment 4

Course: Research Methods and Tools, Astana IT University. Release tag: `assignment-4`.

## What this repository reproduces
Every number, table and figure in the Assignment 4 report (Results, Additional analysis, Appendix B) is produced by one script from two data files.

| Path | Content |
|---|---|
| `data/kazqad_auto_results.csv` | 120 raw model responses (40 KazQAD questions x 3 systems) with passages, reference answers and automatic indicators (output of notebook cell 6) |
| `data/kazqad_final_results.csv` | the same 120 responses with the human scores `meaning_human`, `grammar_human` (0-2, blind scoring, one native-speaker rater) |
| `data/error_coding.csv` | manual coding of the 22 responses with a score below 2 (Appendix B) |
| `analysis.py` | all statistics, baselines, tables and figures |
| `results/` | generated tables (CSV) and `results.json` |
| `figures/` | generated figures |
| `notebook/kazqad_pilot.ipynb` | Colab notebook that collected the data (KazQAD download, selection, API calls, scoring workbook, analysis cells 8-9) |

## Reproduce
```
pip install -r requirements.txt      # tested with Python 3.13, pandas 3.0.5, SciPy 1.18.1, NumPy 2.5.3, Matplotlib 3.11.2
python analysis.py                   # runs in a few seconds, seed 42, 10,000 bootstrap resamples
```
The output tables in `results/` should be identical to those shipped here. The bootstrap uses a fixed seed.

## What cannot be reproduced exactly
The model responses were collected once (GPT-6 through the web chat on 06.11.2026; gemini-3.5-flash and qwen/qwen3.8-27b through their APIs on 07.11.2026). Hosted models change, so re-running the collection may give different responses. Collection needs API keys (never stored in this repository; use Colab Secrets: `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GROQ_API_KEY`).

## Data source and licence
Questions, passages and reference answers come from KazQAD (Yeshpanov et al., 2024; https://github.com/IS2AI/KazQAD), licensed CC BY-SA, so the data files here are shared under CC BY-SA. Code: MIT (see `LICENSE`). No personal data are included.

SHA-256 of the data files:
- kazqad_auto_results.csv `cbb7a466a28c233277a17b28798b613dbc285fdbc6144e890b2348cda7f83983`
- kazqad_final_results.csv `ff393314654e84d59e1b00f000a73ebf5960afd6d1eac0b8c367e5803e26da06`
