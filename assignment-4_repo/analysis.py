"""Assignment 4 analysis: reproduces every table and figure from the two data files.
Usage: python analysis.py   (from the repository root)
Inputs : data/kazqad_auto_results.csv, data/kazqad_final_results.csv
Outputs: results/*.csv, results/results.json, figures/*.png
"""
import json, math, re, unicodedata
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, NBOOT = 42, 10000
rng = np.random.default_rng(SEED)
SHORT = {"A": "GPT-6", "B": "Gemini", "C": "Qwen"}
NAME = {"A": "GPT-6", "B": "gemini-3.5-flash", "C": "qwen/qwen3.8-27b"}
auto = pd.read_csv("data/kazqad_auto_results.csv", encoding="utf-8-sig")
fin = pd.read_csv("data/kazqad_final_results.csv", encoding="utf-8-sig")
assert len(auto) == 120 and len(fin) == 120
d = fin.merge(auto[["qid", "model_key", "response", "ref_answers", "ref_answer_str", "context_clean", "question"]], on=["qid", "model_key"], validate="1:1")
assert d.meaning_human.isin([0, 1, 2]).all() and d.grammar_human.isin([0, 1, 2]).all()
R = {}   # everything that goes into the report

# ---------------------------------------------------------------- helpers
def boot_ci(x, f=np.mean):
    x = np.asarray(x, float)
    b = [f(rng.choice(x, len(x), replace=True)) for _ in range(NBOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
def boot_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(NBOOT)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
def boot_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a); out = []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n); out.append((a[i] - b[i]).mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
def rank_biserial(a, b):   # positive: first group tends to be larger
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1
def mw(a, b):
    if pd.concat([pd.Series(a), pd.Series(b)]).nunique() < 2: return None
    return stats.mannwhitneyu(a, b, alternative="two-sided")
def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1, run)
    return adj

# ---------------------------------------------------------------- 1. main table with bootstrap CIs
rows = []
for k in "ABC":
    for grp in ("low", "high", "all"):
        g = d[(d.model_key == k) & ((d.condition == grp) | (grp == "all"))]
        m, gr, a = g.meaning_human.values, g.grammar_human.values, g.meaning_auto.values
        lo, hi = boot_ci(m); glo, ghi = boot_ci(gr)
        rows.append(dict(system=NAME[k], group=grp, n=len(g), meaning_mean=m.mean(), meaning_sd=m.std(ddof=1), meaning_ci_lo=lo, meaning_ci_hi=hi,
                         fully_correct_pct=100 * (m == 2).mean(), grammar_mean=gr.mean(), grammar_ci_lo=glo, grammar_ci_hi=ghi,
                         grammar_perfect_pct=100 * (gr == 2).mean(), meaning_auto_mean=a.mean(), words_mean=g.resp_words.mean()))
main = pd.DataFrame(rows); main.to_csv("results/table_main.csv", index=False)

# ---------------------------------------------------------------- 2. baselines (automatic metric only)
def norm(t):
    t = unicodedata.normalize("NFC", str(t)).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()
def stem(tok): return None if len(tok) < 4 else tok[:max(4, math.ceil(0.7 * len(tok)))]
def meaning_auto(resp, refs):   # identical to the notebook (cell 6)
    r = " " + norm(resp) + " "; rt = norm(resp).split(); best = 0
    for ref in refs:
        n = norm(ref)
        if not n: continue
        if (" " + n + " ") in r: return 2
        ok = True
        for t in n.split():
            s = stem(t); ok = ok and (any(w == t for w in rt) if s is None else any(w.startswith(s) for w in rt))
        if ok: best = max(best, 1)
    return best
def sentences(ctx): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(ctx).strip()) if s.strip()]
def pick_overlap(ctx, q):
    qs = {(stem(t) or t) for t in norm(q).split()}
    best, bs = None, -1
    for s in sentences(ctx):
        toks = norm(s).split(); sc = sum(1 for t in toks if (stem(t) or t) in qs)
        if sc > bs: best, bs = s, sc
    return best
qs = d.drop_duplicates("qid").set_index("qid")
qs["refs"] = qs.ref_answers.map(lambda x: eval(x) if isinstance(x, str) else x)
bl = []
for qid, r in qs.iterrows():
    first = sentences(r.context_clean)[0]; ov = pick_overlap(r.context_clean, r.question)
    bl.append(dict(qid=qid, condition=r.condition, first=first, overlap=ov,
                   auto_first=meaning_auto(first, r.refs), auto_overlap=meaning_auto(ov, r.refs),
                   words_first=len(first.split()), words_overlap=len(ov.split())))
bl = pd.DataFrame(bl); bl.to_csv("results/baseline_answers.csv", index=False)
base_rows = []
for lab, col, wc in (("Baseline 1: first sentence of the passage", "auto_first", "words_first"), ("Baseline 2: sentence with most question words", "auto_overlap", "words_overlap")):
    for grp in ("low", "high", "all"):
        g = bl[(bl.condition == grp) | (grp == "all")]; lo, hi = boot_ci(g[col])
        base_rows.append(dict(system=lab, group=grp, n=len(g), meaning_auto_mean=g[col].mean(), ci_lo=lo, ci_hi=hi, words_mean=g[wc].mean()))
for k in "ABC":
    for grp in ("low", "high", "all"):
        g = d[(d.model_key == k) & ((d.condition == grp) | (grp == "all"))]; lo, hi = boot_ci(g.meaning_auto)
        base_rows.append(dict(system=NAME[k], group=grp, n=len(g), meaning_auto_mean=g.meaning_auto.mean(), ci_lo=lo, ci_hi=hi, words_mean=g.resp_words.mean()))
pd.DataFrame(base_rows).to_csv("results/table_baselines_auto.csv", index=False)
pv = d.pivot(index="qid", columns="model_key", values="meaning_auto"); pv["B1"] = bl.set_index("qid").auto_first; pv["B2"] = bl.set_index("qid").auto_overlap
R["baseline_tests"] = {f"{x}_vs_{y}": dict(p=float(stats.wilcoxon(pv[x], pv[y]).pvalue), diff=float((pv[x] - pv[y]).mean())) for x, y in [("B2", "A"), ("B2", "B"), ("B2", "C"), ("B1", "A")]}
R["baseline_copy_share"] = {k: float(((bl[c].map(norm)).str.len() > 0).mean()) for k, c in (("first", "first"), ("overlap", "overlap"))}

# ---------------------------------------------------------------- 3. statistics
# 3a low vs high within system (human), with effect size and bootstrap CI of the difference
tests = []
for k in "ABC":
    g = d[d.model_key == k]
    for col in ("meaning_human", "grammar_human", "meaning_auto"):
        a, b = g[g.condition == "low"][col], g[g.condition == "high"][col]; t = mw(a, b)
        tests.append(dict(system=NAME[k], measure=col, low_mean=a.mean(), high_mean=b.mean(), diff_low_minus_high=a.mean() - b.mean(),
                          diff_ci_lo=boot_diff(a, b)[0], diff_ci_hi=boot_diff(a, b)[1], U=None if t is None else t.statistic, p=None if t is None else t.pvalue,
                          rank_biserial=None if t is None else rank_biserial(a, b)))
pd.DataFrame(tests).to_csv("results/table_low_vs_high.csv", index=False)
# 3b question-level (one value per question = mean over the three systems): avoids counting one question three times
ql = d.groupby(["qid", "condition"], as_index=False)[["meaning_human", "grammar_human", "meaning_auto", "complexity_proxy"]].mean()
qrows = []
for col in ("meaning_human", "grammar_human", "meaning_auto"):
    a, b = ql[ql.condition == "low"][col], ql[ql.condition == "high"][col]; t = mw(a, b)
    rho = stats.spearmanr(ql.complexity_proxy, ql[col])
    qrows.append(dict(measure=col, low_mean=a.mean(), high_mean=b.mean(), diff=a.mean() - b.mean(), ci_lo=boot_diff(a, b)[0], ci_hi=boot_diff(a, b)[1],
                      U=t.statistic, p=t.pvalue, rank_biserial=rank_biserial(a, b), rho=rho.statistic, rho_p=rho.pvalue))
pd.DataFrame(qrows).to_csv("results/table_question_level.csv", index=False)
# 3c between systems on the same 40 questions
P = {c: d.pivot(index="qid", columns="model_key", values=c) for c in ("meaning_human", "grammar_human")}
sysrows = []
for c, p in P.items():
    fr = stats.friedmanchisquare(p.A, p.B, p.C)
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]; raw = []
    for x, y in pairs:
        w = stats.wilcoxon(p[x], p[y], zero_method="wilcox"); raw.append(w.pvalue)
    adj = holm(np.array(raw))
    for (x, y), r_, a_ in zip(pairs, raw, adj):
        lo, hi = boot_paired(p[x], p[y])
        sysrows.append(dict(measure=c, pair=f"{NAME[x]} vs {NAME[y]}", mean_diff=(p[x] - p[y]).mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=r_, holm_p=a_, friedman_chi2=fr.statistic, friedman_p=fr.pvalue))
pd.DataFrame(sysrows).to_csv("results/table_systems.csv", index=False)
# McNemar (exact) on "fully correct" (meaning_human == 2)
bin_ = (P["meaning_human"] == 2).astype(int); mc = []
for x, y in [("A", "B"), ("A", "C"), ("B", "C")]:
    b_ = int(((bin_[x] == 1) & (bin_[y] == 0)).sum()); c_ = int(((bin_[x] == 0) & (bin_[y] == 1)).sum())
    p_ = 1.0 if b_ + c_ == 0 else stats.binomtest(b_, b_ + c_, 0.5).pvalue
    mc.append(dict(pair=f"{NAME[x]} vs {NAME[y]}", x_only_correct=b_, y_only_correct=c_, p=p_))
mc = pd.DataFrame(mc); mc["holm_p"] = holm(mc.p.values); mc.to_csv("results/table_mcnemar.csv", index=False)
# 3d human vs automatic
ct = pd.crosstab(d.meaning_human, d.meaning_auto).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0); ct.to_csv("results/table_human_vs_auto.csv")
sp = stats.spearmanr(d.meaning_human, d.meaning_auto)
R["agree"] = dict(exact=float((d.meaning_human == d.meaning_auto).mean()), rho=float(sp.statistic), p=float(sp.pvalue),
                  by_system={NAME[k]: float((d[d.model_key == k].meaning_human == d[d.model_key == k].meaning_auto).mean()) for k in "ABC"})

# ---------------------------------------------------------------- 4. sensitivity: drop the two items whose reference does not match the passage (Q13, Q29)
sens = []
for lab, mask in (("all 40 questions", d.qid.notna()), ("without Q13 and Q29", ~d.qid.isin(["Q13", "Q29"]))):
    s = d[mask]; qq = s.groupby(["qid", "condition"], as_index=False).meaning_human.mean()
    a, b = qq[qq.condition == "low"].meaning_human, qq[qq.condition == "high"].meaning_human; t = mw(a, b)
    row = dict(subset=lab, n_questions=qq.qid.nunique(), low_mean=a.mean(), high_mean=b.mean(), diff=a.mean() - b.mean(), p=t.pvalue, rank_biserial=rank_biserial(a, b))
    for k in "ABC":
        g = s[s.model_key == k]; aa, bb = g[g.condition == "low"].meaning_human, g[g.condition == "high"].meaning_human
        row[f"diff_{k}"] = aa.mean() - bb.mean(); row[f"p_{k}"] = mw(aa, bb).pvalue
    sens.append(row)
pd.DataFrame(sens).to_csv("results/table_sensitivity.csv", index=False)
# alternative split: tertiles of the proxy instead of the extreme-group design (rank correlation per system already in 3b); response length by group
len_rows = []
for k in "ABC":
    g = d[d.model_key == k]; a, b = g[g.condition == "low"].resp_words, g[g.condition == "high"].resp_words; t = mw(a, b)
    len_rows.append(dict(system=NAME[k], low_words=a.mean(), high_words=b.mean(), p=t.pvalue, median_all=g.resp_words.median(), iqr_lo=g.resp_words.quantile(.25), iqr_hi=g.resp_words.quantile(.75)))
pd.DataFrame(len_rows).to_csv("results/table_efficiency_length.csv", index=False)

# ---------------------------------------------------------------- 5. failures for the error analysis (all responses with a score below 2)
fail = d[(d.meaning_human < 2) | (d.grammar_human < 2)].copy()
fail["system"] = fail.model_key.map(NAME)
fail[["blind_id", "qid", "system", "condition", "question", "ref_answer_str", "response", "meaning_human", "grammar_human", "meaning_auto"]].sort_values(["qid", "system"]).to_csv("results/failures_to_code.csv", index=False)
R["n_failures"] = int(len(fail)); R["n_meaning_lt2"] = int((d.meaning_human < 2).sum()); R["n_grammar_lt2"] = int((d.grammar_human < 2).sum())

# error coding (manual, data/error_coding.csv): counts must match the failures
ec = pd.read_csv("data/error_coding.csv")
assert set(ec.blind_id) == set(fail.blind_id)
assert (ec.dimension == "meaning").sum() == R["n_meaning_lt2"] and (ec.dimension == "grammar").sum() == R["n_grammar_lt2"]
et = ec.merge(d[["blind_id", "model_key", "condition"]], on="blind_id")
tab = et.groupby("error_type").agg(count=("blind_id", "size"), GPT6=("model_key", lambda x: (x == "A").sum()), Gemini=("model_key", lambda x: (x == "B").sum()),
                                   Qwen=("model_key", lambda x: (x == "C").sum()), low=("condition", lambda x: (x == "low").sum()), high=("condition", lambda x: (x == "high").sum())).reset_index()
tab["pct_of_error_flags"] = 100 * tab["count"] / len(ec); tab.to_csv("results/table_errors.csv", index=False)

# ---------------------------------------------------------------- 6. figures
col = {"low": "#4C78A8", "high": "#E45756"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a_, (c, t) in zip(ax, (("meaning", "Meaning (human, 0-2)"), ("grammar", "Grammar (human, 0-2)"))):
    for j, grp in enumerate(("low", "high")):
        s = main[main.group == grp]; x = np.arange(3) + (j - .5) * .35
        y = s[f"{c}_mean"].values; err = [y - s[f"{c}_ci_lo"].values, s[f"{c}_ci_hi"].values - y]
        a_.bar(x, y, .35, yerr=err, capsize=3, color=col[grp], label=f"{grp} complexity (n=20)")
    a_.set_xticks(range(3)); a_.set_xticklabels([NAME[k] for k in "ABC"], fontsize=9); a_.set_title(t); a_.set_ylim(0, 2.15); a_.set_ylabel("mean score, 95% bootstrap CI")
ax[0].legend(loc="lower left", fontsize=8); plt.tight_layout(); plt.savefig("figures/fig1_means_ci.png", dpi=200); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a_, c, t in zip(ax, ("meaning_human", "grammar_human"), ("Meaning score distribution", "Grammar score distribution")):
    lab, left = [], {}
    for k in "ABC":
        for grp in ("low", "high"):
            g = d[(d.model_key == k) & (d.condition == grp)][c]; lab.append(f"{SHORT[k]}\n{grp}")
    cnt = {s: [int(((d.model_key == k) & (d.condition == grp) & (d[c] == s)).sum()) for k in "ABC" for grp in ("low", "high")] for s in (0, 1, 2)}
    bottom = np.zeros(6)
    for s_, colr in ((2, "#59A14F"), (1, "#F2C14E"), (0, "#E15759")):
        a_.bar(range(6), cnt[s_], bottom=bottom, color=colr, label=f"score {s_}"); bottom += np.array(cnt[s_])
    a_.set_xticks(range(6)); a_.set_xticklabels(lab, fontsize=7.5); a_.set_title(t); a_.set_ylabel("number of responses")
ax[0].legend(fontsize=8, loc="lower left"); plt.tight_layout(); plt.savefig("figures/fig2_score_distribution.png", dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(4.6, 3.8)); im = ax.imshow(ct.values, cmap="Blues")
for i in range(3):
    for j in range(3): ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", color="white" if ct.values[i, j] > 40 else "black")
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xlabel("automatic meaning score"); ax.set_ylabel("human meaning score"); ax.set_title("Human vs automatic (n=120)")
plt.tight_layout(); plt.savefig("figures/fig3_human_vs_auto.png", dpi=200); plt.close()

json.dump(R, open("results/results.json", "w"), indent=1, default=float)
print("done; failures to code:", len(fail))
