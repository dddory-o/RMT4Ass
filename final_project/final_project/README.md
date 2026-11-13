# Final project: Kazakh prompt morphological complexity vs. LLM answer quality (KazQAD)

Pilot (Assignment 4) lives in `../assignment-4_repo/` (tag `assignment-4`). This folder is the final, scaled-up study (tag `final`).

## Design
500 KazQAD questions (250 lowest / 250 highest mean letters-per-word, from the Maxutov et al. pool: 4-8 word questions, passage <=150 words) x 2 open models (Qwen/Qwen3-4B, google/gemma-3-4b-it), greedy decoding, max_new_tokens 96 = 1000 responses. Automatic metric: token F1 (primary). 120 responses (60 per model) were scored blind by a human (meaning 0-2, grammar 0-2).

## Reproduce (CPU only, ~1 min; no API keys needed)
    pip install -r requirements.txt
    python analysis_large.py     # data/*.csv -> results/table1-3_*.csv, figures/fig_results.png
    python analysis_human.py     # human sample -> results/human_*.csv
Seed 42, 10,000 bootstrap resamples. Regenerating the responses themselves needs a GPU: run `notebooks/kazqad_open_models.ipynb` in Colab (T4); HF token only via Colab Secrets.

## Files
| path | content |
|---|---|
| data/selected_questions.csv | 500 selected questions + references |
| data/responses.csv, responses_scored.csv | 1000 raw / scored model responses |
| data/human_scoring_sample_filled.xlsx | human scores (Rubric, Scoring, Key sheets) |
| results/table1-3_*.csv | automatic-metric tables |
| results/human_*.csv | human-score tables |
| figures/fig_results.png | main figure |

Source data: KazQAD questions via github.com/akylbekmaxutov/LLM-eval-using-Kazakh (see that repo for terms; no license file there). Code: MIT (see ../assignment-4_repo/LICENSE).
