"""Automatic scoring and tables for the 500-question run (cells 5-7 of notebooks/kazqad_open_models.ipynb, as a script).
Run from the final_project folder: python analysis_large.py
Inputs : data/selected_questions.csv, data/responses.csv     Outputs: results/*.csv, figures/fig_results.png
"""
import os, re, json, ast, numpy as np, pandas as pd
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
OUT_DIR, SEED = "results", 42
RESP_FILE = "data/responses.csv"
def words(t): return re.findall(r"[^\W\d_]+", str(t), flags=re.UNICODE)
sel = pd.read_csv("data/selected_questions.csv"); sel["refs"] = sel["refs"].map(ast.literal_eval)
# CELL 5 - automatic scoring
import unicodedata, math
def norm(t):
    t = unicodedata.normalize("NFC", str(t)).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()
def stem(tok): return tok if len(tok) < 4 else tok[:max(4, math.ceil(0.7 * len(tok)))]

def score(resp, refs):
    """Best over the reference answers: exact-token recall, stem recall (tolerates suffix changes), token F1."""
    rt = norm(resp).split(); rs = [stem(w) for w in rt]; best = dict(recall=0.0, recall_stem=0.0, f1=0.0)
    for ref in refs:
        gt = norm(ref).split()
        if not gt: continue
        common = sum(min(rt.count(w), gt.count(w)) for w in set(gt))
        rec = common / len(gt); prec = common / len(rt) if rt else 0
        gs = [stem(w) for w in gt]
        rec_s = sum(1 for w in gs if w in rs) / len(gs)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        best = dict(recall=max(best["recall"], rec), recall_stem=max(best["recall_stem"], rec_s), f1=max(best["f1"], f1))
    return best

R = pd.read_csv(RESP_FILE).merge(sel[["qid", "condition", "proxy", "q_words", "ctx_words", "question", "refs"]], on="qid")
sc = pd.DataFrame([score(r, refs) for r, refs in zip(R.response, R.refs)])
R = pd.concat([R.reset_index(drop=True), sc], axis=1)
R["resp_words"] = R.response.map(lambda t: len(words(t)))
R["contains_ref"] = [int(any(norm(x) and norm(x) in norm(r) for x in refs)) for r, refs in zip(R.response, R.refs)]
R["cyrillic_ok"] = R.response.map(lambda t: int((lambda L: bool(L) and sum(1 for c in L if "Ѐ" <= c <= "ӿ") / len(L) >= 0.9)([c for c in str(t) if c.isalpha()])))
R["no_info"] = R.response.map(lambda t: int(bool(re.search(r"(ақпарат|мәлімет)[^.]{0,60}(берілмеген|көрсетілмеген|жоқ)|нақты ақпарат|ешқандай", str(t), re.I))))
R.drop(columns="refs").to_csv("data/responses_scored.csv", index=False)
print("scored responses:", len(R), "| models:", R.model.nunique())
print("empty responses:", int((R.resp_words == 0).sum()), "| not Cyrillic:", int((R.cyrillic_ok == 0).sum()))
# CELL 6 - results tables (all in English)
# Why F1: on the 120 pilot responses it had the highest rank correlation with the human meaning score (rho 0.52, against 0.45 for stem recall and 0.37 for the old pilot score)
from scipy import stats
rng = np.random.default_rng(SEED)
def boot_ci(x, n=5000):
    x = np.asarray(x, float); b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
def rbc(a, b): return 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1

METRIC = "f1"      # primary automatic metric: token-level F1 (the metric of Maxutov et al., 2024); alternative: "recall_stem"
print("TABLE 1. Mean", METRIC, "(0-1) by model and complexity group, 95% bootstrap CI")
t1 = []
for m in R.model.unique():
    for g in ("low", "high", "all"):
        x = R[(R.model == m) & ((R.condition == g) | (g == "all"))][METRIC]; lo, hi = boot_ci(x)
        t1.append(dict(model=m, group=g, n=len(x), mean=round(x.mean(), 3), ci=f"[{lo:.3f}, {hi:.3f}]", exact_match_pct=round(100 * R.loc[x.index, "contains_ref"].mean(), 1), words=round(R.loc[x.index, "resp_words"].mean(), 1)))
print(pd.DataFrame(t1).to_string(index=False)); pd.DataFrame(t1).to_csv(f"{OUT_DIR}/table1_main.csv", index=False)

print("\nTABLE 2. Low vs high complexity (per model): difference, 95% CI, Mann-Whitney p, effect size r, Spearman rho (proxy vs score)")
t2 = []
for m in R.model.unique():
    d = R[R.model == m]; a, b = d[d.condition == "low"][METRIC], d[d.condition == "high"][METRIC]
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
    rho = stats.spearmanr(d.proxy, d[METRIC])
    t2.append(dict(model=m, low=round(a.mean(), 3), high=round(b.mean(), 3), diff=round(a.mean() - b.mean(), 3),
                   ci=f"[{np.percentile(diffs, 2.5):.3f}, {np.percentile(diffs, 97.5):.3f}]", p=round(stats.mannwhitneyu(a, b).pvalue, 4),
                   r=round(rbc(a, b), 3), rho=round(rho.statistic, 3), rho_p=round(rho.pvalue, 4)))
print(pd.DataFrame(t2).to_string(index=False)); pd.DataFrame(t2).to_csv(f"{OUT_DIR}/table2_low_vs_high.csv", index=False)

print("\nTABLE 3. Models on the same questions: Friedman test and paired Wilcoxon tests (Holm-corrected)")
P = R.pivot(index="qid", columns="model", values=METRIC).dropna()
ms = list(P.columns)
if len(ms) >= 3: print("Friedman: chi2 = %.2f, p = %.4g" % stats.friedmanchisquare(*[P[m] for m in ms]))
pairs, raw = [], []
for i in range(len(ms)):
    for j in range(i + 1, len(ms)):
        pairs.append((ms[i], ms[j])); raw.append(stats.wilcoxon(P[ms[i]], P[ms[j]]).pvalue)
order = np.argsort(raw); holm = np.empty(len(raw)); run = 0
for k, idx in enumerate(order): run = max(run, (len(raw) - k) * raw[idx]); holm[idx] = min(1, run)
t3 = pd.DataFrame([dict(pair=f"{a} vs {b}", mean_diff=round((P[a] - P[b]).mean(), 3), wilcoxon_p=round(p, 4), holm_p=round(h, 4)) for (a, b), p, h in zip(pairs, raw, holm)])
print(t3.to_string(index=False)); t3.to_csv(f"{OUT_DIR}/table3_models.csv", index=False)

print("\nTABLE 4. Pooled over models, one value per question (avoids counting a question several times)")
Q = R.groupby(["qid", "condition"], as_index=False)[[METRIC, "proxy"]].mean()
a, b = Q[Q.condition == "low"][METRIC], Q[Q.condition == "high"][METRIC]; rho = stats.spearmanr(Q.proxy, Q[METRIC])
print(f"low {a.mean():.3f}  high {b.mean():.3f}  diff {a.mean() - b.mean():.3f}  Mann-Whitney p = {stats.mannwhitneyu(a, b).pvalue:.4g}  r = {rbc(a, b):.3f}  Spearman rho = {rho.statistic:.3f} (p = {rho.pvalue:.4g})")

print("\nTABLE 5. Response behaviour")
print(R.groupby("model").agg(words_mean=("resp_words", "mean"), no_info_statements=("no_info", "sum"), not_cyrillic=("cyrillic_ok", lambda x: int((x == 0).sum())), empty=("resp_words", lambda x: int((x == 0).sum()))).round(1).to_string())
# CELL 7 - charts
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ms = list(R.model.unique()); x = np.arange(len(ms))
for j, g in enumerate(("low", "high")):
    means, errs = [], [[], []]
    for m in ms:
        v = R[(R.model == m) & (R.condition == g)][METRIC]; lo, hi = boot_ci(v)
        means.append(v.mean()); errs[0].append(v.mean() - lo); errs[1].append(hi - v.mean())
    ax[0].bar(x + (j - .5) * .35, means, .35, yerr=errs, capsize=3, label=f"{g} complexity", color=["#4C78A8", "#E45756"][j])
ax[0].set_xticks(x); ax[0].set_xticklabels(ms); ax[0].set_ylabel(f"mean {METRIC} (0-1), 95% CI"); ax[0].set_title("Quality by model and complexity group"); ax[0].legend()
for m in ms:
    d = R[R.model == m].copy(); d["bin"] = pd.qcut(d.proxy, 8, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(p=("proxy", "mean"), s=(METRIC, "mean"))
    ax[1].plot(g.p, g.s, marker="o", label=m)
ax[1].set_xlabel("complexity proxy (mean letters per word in the question)"); ax[1].set_ylabel(f"mean {METRIC}"); ax[1].set_title("Quality against complexity (8 bins)"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/fig_results.png", dpi=200)