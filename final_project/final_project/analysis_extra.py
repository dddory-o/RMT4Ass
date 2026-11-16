"""Extra analyses for the final paper: complexity-proxy descriptives, confound check, adjusted effect.
Run from final_project/: python analysis_extra.py   Inputs: data/responses_scored.csv  Outputs: results/extra_*.csv
"""
import numpy as np, pandas as pd
from scipy import stats
SEED, NBOOT = 42, 10000
R = pd.read_csv("data/responses_scored.csv")
rng = np.random.default_rng(SEED)

# 1) proxy and confounders by group (one row per question)
Q = R.drop_duplicates("qid")[["qid", "condition", "proxy", "q_words", "ctx_words"]]
d = Q.groupby("condition").agg(n=("qid", "size"), proxy_mean=("proxy", "mean"), proxy_min=("proxy", "min"), proxy_max=("proxy", "max"),
                               q_words_mean=("q_words", "mean"), ctx_words_mean=("ctx_words", "mean")).round(3)
d.to_csv("results/extra_group_descriptives.csv"); print(d.to_string(), "\n")
c = []
for v in ["q_words", "ctx_words"]:
    rho, p = stats.spearmanr(Q["proxy"], Q[v]); c.append(dict(variable=v, rho_with_proxy=round(rho, 3), p=round(p, 4)))
pd.DataFrame(c).to_csv("results/extra_proxy_confounds.csv", index=False); print(pd.DataFrame(c).to_string(), "\n")

# 2) adjusted effect: f1 ~ z(proxy) + z(q_words) + z(ctx_words), OLS, bootstrap CI over questions
def z(x): return (x - x.mean()) / x.std(ddof=0)
rows = []
for m, g in R.groupby("model"):
    g = g.reset_index(drop=True)
    X = np.column_stack([np.ones(len(g)), z(g.proxy), z(g.q_words), z(g.ctx_words)])
    y = g.f1.values
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(g), len(g))
        bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    for k, nm in enumerate(["intercept", "proxy", "q_words", "ctx_words"]):
        rows.append(dict(model=m, term=nm, beta=round(b[k], 4), ci=f"[{lo[k]:.4f}, {hi[k]:.4f}]"))
A = pd.DataFrame(rows); A.to_csv("results/extra_adjusted_regression.csv", index=False); print(A.to_string(), "\n")

# 3) f1 by proxy quintile (dose-response), both models pooled by qid mean
P = R.groupby(["qid"]).agg(proxy=("proxy", "first"), f1=("f1", "mean")).reset_index()
P["quintile"] = pd.qcut(P.proxy, 5, labels=[1, 2, 3, 4, 5])
T = P.groupby("quintile", observed=True).agg(n=("qid", "size"), proxy_mean=("proxy", "mean"), f1_mean=("f1", "mean")).round(3)
T.to_csv("results/extra_quintiles.csv"); print(T.to_string(), "\n")

# 4) item agreement between models (are the same questions hard?)
W = R.pivot(index="qid", columns="model", values="f1")
rho, p = stats.spearmanr(W.iloc[:, 0], W.iloc[:, 1]); print("between-model rho of item f1:", round(rho, 3), "p=", round(p, 4))
pd.DataFrame([dict(rho=round(rho, 3), p=round(p, 4))]).to_csv("results/extra_between_model.csv", index=False)
