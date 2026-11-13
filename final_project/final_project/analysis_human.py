"""Analysis of the human-scored sample (120 responses) against the automatic metrics.
Run from the repository root: python analysis_human.py
Inputs : data/human_scoring_sample_filled.xlsx (sheets Scoring, Key), data/responses_scored.csv
Outputs: results/human_*.csv
"""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(42)
x = pd.read_excel("data/human_scoring_sample_filled.xlsx", sheet_name=None)
s = x["Scoring"].rename(columns={"meaning_human (0-2)": "meaning_human", "grammar_human (0-2)": "grammar_human"})
d = s.merge(x["Key"], on="blind_id")
assert len(d) == 120 and d.meaning_human.isin([0, 1, 2]).all() and d.grammar_human.isin([0, 1, 2]).all()
R = pd.read_csv("data/responses_scored.csv")[["qid", "model", "f1", "recall", "recall_stem", "contains_ref", "resp_words", "proxy"]]
d = d.drop(columns=[c for c in ("f1",) if c in d.columns]).merge(R, on=["qid", "model"], how="left", validate="1:1")
assert d.f1.notna().all()
def ci(v, n=10000):
    v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(n)]; return np.percentile(b, 2.5), np.percentile(b, 97.5)
def rbc(a, b): return 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1

rows = []
for m in list(d.model.unique()) + ["all"]:
    for g in ("low", "high", "all"):
        t = d[((d.model == m) | (m == "all")) & ((d.condition == g) | (g == "all"))]
        lo, hi = ci(t.meaning_human); glo, ghi = ci(t.grammar_human)
        rows.append(dict(model=m, group=g, n=len(t), meaning_mean=t.meaning_human.mean(), meaning_ci=f"[{lo:.2f}, {hi:.2f}]", fully_correct_pct=100 * (t.meaning_human == 2).mean(),
                         grammar_mean=t.grammar_human.mean(), grammar_ci=f"[{glo:.2f}, {ghi:.2f}]", grammar_perfect_pct=100 * (t.grammar_human == 2).mean()))
pd.DataFrame(rows).round(3).to_csv("results/human_table_main.csv", index=False)

tests = []
for m in list(d.model.unique()) + ["all"]:
    t = d[(d.model == m) | (m == "all")]
    for col in ("meaning_human", "grammar_human"):
        a, b = t[t.condition == "low"][col], t[t.condition == "high"][col]; mw = stats.mannwhitneyu(a, b); rho = stats.spearmanr(t.proxy, t[col])
        tests.append(dict(model=m, measure=col, low=a.mean(), high=b.mean(), diff=a.mean() - b.mean(), p=mw.pvalue, r=rbc(a, b), rho=rho.statistic, rho_p=rho.pvalue, n_low=len(a), n_high=len(b)))
pd.DataFrame(tests).round(4).to_csv("results/human_low_vs_high.csv", index=False)

ms = d.model.unique(); a, b = d[d.model == ms[0]], d[d.model == ms[1]]
mod = [dict(measure=c, **{ms[0]: a[c].mean(), ms[1]: b[c].mean()}, p=stats.mannwhitneyu(a[c], b[c]).pvalue) for c in ("meaning_human", "grammar_human")]
pd.DataFrame(mod).round(4).to_csv("results/human_models.csv", index=False)

val = []
for c in ("f1", "recall", "recall_stem", "contains_ref"):
    r = stats.spearmanr(d.meaning_human, d[c]); val.append(dict(metric=c, spearman_rho=r.statistic, p=r.pvalue))
val = pd.DataFrame(val); val.round(4).to_csv("results/human_vs_auto_metrics.csv", index=False)
d["ref_in_resp"] = d.contains_ref.astype(int)
ct = pd.crosstab(d.meaning_human, d.ref_in_resp); ct.to_csv("results/human_vs_contains_ref.csv")
print(pd.read_csv("results/human_table_main.csv").to_string(index=False)); print(pd.read_csv("results/human_low_vs_high.csv").to_string(index=False))
print(pd.read_csv("results/human_models.csv").to_string(index=False)); print(val.round(3).to_string(index=False)); print(ct)
