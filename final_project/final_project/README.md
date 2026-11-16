# Final project: Kazakh question morphological complexity vs. small-LLM answer quality (KazQAD)

Course: Research Methods and Tools, Astana IT University. Release tag: `final`. The pilot (Assignment 4) is in `../assignment-4_repo/` (tag `assignment-4`).

## Design
500 KazQAD questions (250 lowest / 250 highest mean letters per word; pool of 544 questions with 4-8 words and passage <= 150 words, from the evaluation repository of Maxutov et al.) x 2 open models (Qwen/Qwen3-4B, google/gemma-3-4b-it), greedy decoding, max_new_tokens 96 = 1,000 responses. Primary metric: token F1. 120 responses (60 per model, random) scored blind by one human rater (meaning 0-2, grammar 0-2).

## Reproduce every table and figure (CPU only, about 1 minute, no API keys)
```
pip install -r requirements.txt
python analysis_large.py     # data/*.csv -> results/table1-3_*.csv, figures/fig_results.png
python analysis_human.py     # human sample -> results/human_*.csv
python analysis_extra.py     # proxy descriptives, confounds, regression, quintiles -> results/extra_*.csv
python check_refs.py         # (needs internet) checks the DOIs of the paper's reference list
```
Seed 42, 10,000 bootstrap resamples. Regenerating the responses needs a GPU: run `notebooks/kazqad_open_models.ipynb` in Google Colab (T4). The Hugging Face token must be stored only in Colab Secrets, never in files.

## Which file gives which table of the paper
| Paper | File |
|---|---|
| Table 1 | results/table1_main.csv (column `exact_match_pct` is the reference-containment rate) |
| Table 2 | results/table2_low_vs_high.csv |
| Table 3 | results/extra_quintiles.csv |
| Table 4 | results/extra_adjusted_regression.csv |
| Table 5 | results/human_table_main.csv |
| Section 4.4, 4.6 | results/table3_models.csv, results/extra_between_model.csv, results/human_vs_auto_metrics.csv, results/human_vs_contains_ref.csv |
| Figure 1 | figures/fig_results.png |
| Reference list | references_doi.csv |

## Files
| path | content |
|---|---|
| data/selected_questions.csv | the 500 selected questions, proxy values and reference answers |
| data/responses.csv, data/responses_scored.csv | 1,000 raw responses / with automatic scores |
| data/human_scoring_sample_filled.xlsx | human scores (sheets Rubric, Scoring, Key) |
| notebooks/kazqad_open_models.ipynb | Colab notebook that generated the responses |
| analysis_large.py, analysis_human.py, analysis_extra.py | analysis scripts |

## What cannot be reproduced exactly
The responses were generated once on a Colab T4 GPU. Re-running generation with other library or driver versions may change a few responses. The pilot (three hosted systems) cannot be regenerated, because hosted models change.

## Data source, licence, privacy
Questions, passages and reference answers derive from KazQAD (Yeshpanov et al., 2024; CC BY-SA); the data files here are shared under CC BY-SA. Code: MIT (see `LICENSE`). No personal data and no API keys or tokens are included.

## SHA-256 of the data files
- `data/selected_questions.csv`: `5d5e5f2f6665ed5aa6c04a5ce9cfd52f0a4fcb505a0af6599fb904d4a121db80`
- `data/responses.csv`: `68008a418cc3867a704f74ac68c84b458827140b32fed7679add058a55b4b6a1`
- `data/responses_scored.csv`: `f99cbfa5b393f1a2a2edbe6173f600d52551c296489ab8a800c6c1dba2f2334c`
- `data/human_scoring_sample_filled.xlsx`: `0997c09ad2b845ad265e3e9fe064b5b24a66ee4520a6341a30d161a4ddea9af1`
