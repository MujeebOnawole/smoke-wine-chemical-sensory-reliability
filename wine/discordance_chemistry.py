"""Round 6b (2026-10-26): two follow-up analyses on the pulled per-wine predictions, no new model fits.

1. Chemistry of guaiacol-discordant wines beyond total phenol load: non-guaiacol free phenols, the ratio of
   non-guaiacol phenols to guaiacol, glycoside share, and a logistic model of wine-level discordance on
   log guaiacol + cultivar + log non-guaiacol phenols (association only).
2. Prospective disagreement signal |ensemble - guaiacol regression| per prediction: does it flag guaiacol
   discordance, large guaiacol errors and large ensemble errors, and how does it rank ensemble error
   (AURC) against consensus SD? Wine-cluster bootstrap, 2,000 resamples, seed 42, as in the main analysis.

Outputs: LOT/tables_round6/round6b_*.csv and .md, printed digest.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WA = os.path.join(ROOT, "results", "wine", "wine_assessment")
WINE = os.path.join(ROOT, "data", "awri_wine_smoke_wide_labelled.csv")
TOUT = os.path.join(ROOT, "results", "wine")
SETTINGS = [("cv", "Cultivar-stratified CV"), ("cultivar_holdout", "Cultivar hold-out"),
            ("grouped_site", "Sites held out"), ("grouped_region", "Regions held out"),
            ("external_trainB_testA", "Train B, test A"), ("external_trainA_testB", "Train A, test B")]
COVERAGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_BOOT, SEED, LARGE = 2000, 42, 1.5

wine = pd.read_csv(WINE).set_index("wine_uid")
pred = pd.read_csv(os.path.join(WA, "per_wine_predictions.csv"))
dw = pd.read_csv(os.path.join(WA, "discordance_wines.csv"))
PHEN = ["wine_guaiacol", "wine_4-methylguaiacol", "wine_m-cresol", "wine_o-cresol", "wine_p-cresol", "wine_syringol", "wine_4-methylsyringol"]
GLY = ["wine_CrRG", "wine_GuRG", "wine_MGuRG", "wine_MSyGG", "wine_PhRG", "wine_SyGG"]
wine["phenol_sum"] = wine[PHEN].sum(axis=1)
wine["nongua_phenols"] = wine[PHEN[1:]].sum(axis=1)
wine["ratio_nongua_gua"] = wine["nongua_phenols"] / wine["wine_guaiacol"]
wine["glyc_sum"] = wine[GLY].sum(axis=1)
wine["glyc_share"] = wine["glyc_sum"] / (wine["glyc_sum"] + wine["phenol_sum"])


def mwu(a, b):
    return mannwhitneyu(a, b, alternative="two-sided").pvalue


# ------------------------------------------------------------------ 1. chemistry of discordant wines
rows = []
for s, name in SETTINGS:
    d = dw[(dw.reference == "b_guaiacol") & (dw.feature_set == "both") & (dw.setting == s)].set_index("wine_uid")
    w = wine.join(d[["discordant_share", "mean_ref_resid"]], how="inner")
    w["disc"] = (w.discordant_share >= 0.5).astype(int)
    a, b = w[w.disc == 1], w[w.disc == 0]
    row = {"Evaluation": name, "Discordant wines": len(a), "Concordant wines": len(b)}
    for col, lab in [("phenol_sum", "total phenols"), ("nongua_phenols", "non-guaiacol phenols"), ("wine_guaiacol", "guaiacol"),
                     ("ratio_nongua_gua", "non-guaiacol / guaiacol ratio"), ("glyc_share", "glycoside share")]:
        row[f"median {lab}, discordant"] = round(a[col].median(), 2)
        row[f"median {lab}, concordant"] = round(b[col].median(), 2)
        row[f"MWU p, {lab}"] = round(mwu(a[col], b[col]), 4)
    # direction split: over-predicted (resid > 0) vs under-predicted discordant wines
    over, under = a[a.mean_ref_resid > 0], a[a.mean_ref_resid < 0]
    row["over-predicted wines"] = len(over)
    row["under-predicted wines"] = len(under)
    row["median ratio, over-predicted"] = round(over.ratio_nongua_gua.median(), 2) if len(over) else np.nan
    row["median ratio, under-predicted"] = round(under.ratio_nongua_gua.median(), 2) if len(under) else np.nan
    row["median glycoside share, over"] = round(over.glyc_share.median(), 2) if len(over) else np.nan
    row["median glycoside share, under"] = round(under.glyc_share.median(), 2) if len(under) else np.nan
    # adjusted association (association only): logistic disc ~ log1p guaiacol + cultivar + log1p non-guaiacol phenols
    X = pd.DataFrame({"log_gua": np.log1p(w.wine_guaiacol), "log_nongua": np.log1p(w.nongua_phenols),
                      "PIN": (w.cultivar_code == "PIN").astype(float), "SHI": (w.cultivar_code == "SHI").astype(float)})
    try:
        fit = sm.Logit(w.disc, sm.add_constant(X)).fit(disp=0, maxiter=200)
        for k in ["log_gua", "log_nongua", "PIN", "SHI"]:
            row[f"logit coef {k}"] = round(fit.params[k], 2)
            row[f"logit p {k}"] = round(fit.pvalues[k], 3)
        row["logit note"] = ""
    except Exception as e:  # separation or convergence
        row["logit note"] = f"not fitted: {type(e).__name__}"
    rows.append(row)
chem = pd.DataFrame(rows)
chem.to_csv(os.path.join(TOUT, "round6b_discordance_chemistry.csv"), index=False)
with open(os.path.join(TOUT, "round6b_discordance_chemistry.md"), "w") as f:
    f.write("Round 6b. Chemistry of wine-level guaiacol discordance (discordant in at least half of the repeats), grape and wine chemistry. Association only.\n\n")
    f.write(chem.T.to_markdown() + "\n")
print(chem.T.to_string())

# ------------------------------------------------------------------ 2. prospective disagreement signal
def aurc(err, score):
    order = np.argsort(score, kind="stable")
    out = []
    for c in COVERAGES:
        k = max(1, int(round(c * len(err))))
        out.append(err[order[:k]].mean())
    return np.trapezoid(out, COVERAGES) / (COVERAGES[-1] - COVERAGES[0])


def safe_auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def boot_diff(d, fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    wines = d.wine_uid.unique()
    groups = {w: g for w, g in d.groupby("wine_uid")}
    vals = []
    for _ in range(n):
        samp = pd.concat([groups[w] for w in rng.choice(wines, len(wines))], ignore_index=True)
        vals.append(fn(samp))
    return np.nanpercentile(vals, [2.5, 97.5])


rows = []
for s, name in SETTINGS:
    for fs in ["both", "grape"]:
        d = pred[(pred.setting == s) & (pred.feature_set == fs)].copy()
        d["gap"] = (d.ensemble - d.b_guaiacol).abs()
        d["ens_err"] = (d.ensemble - d.smoke_flavour).abs()
        d["gua_err"] = (d.b_guaiacol - d.smoke_flavour).abs()
        d["gua_disc"] = (d.gua_err >= LARGE).astype(int)
        d["ens_large"] = (d.ens_err >= LARGE).astype(int)
        row = {"Evaluation": name, "Features": {"both": "grape + wine", "grape": "grape"}[fs], "n": len(d),
               "AUROC gap flags guaiacol discordance": round(safe_auc(d.gua_disc, d.gap), 2),
               "AUROC consensus SD flags guaiacol discordance": round(safe_auc(d.gua_disc, d.consensus_std), 2),
               "Spearman gap vs guaiacol error": round(spearmanr(d.gap, d.gua_err).correlation, 2),
               "AUROC gap flags large ensemble error": round(safe_auc(d.ens_large, d.gap), 2),
               "AUROC consensus SD flags large ensemble error": round(safe_auc(d.ens_large, d.consensus_std), 2),
               "AURC ensemble error, ranked by gap": round(aurc(d.ens_err.values, d.gap.values), 3),
               "AURC ensemble error, ranked by consensus SD": round(aurc(d.ens_err.values, d.consensus_std.values), 3)}
        lo, hi = boot_diff(d, lambda x: aurc(x.ens_err.values, x.gap.values) - aurc(x.ens_err.values, x.consensus_std.values))
        row["AURC gap minus consensus (95% CI)"] = f"{lo:+.3f} to {hi:+.3f}"
        # screening use: share of guaiacol-discordant predictions caught when the 30% largest gaps are flagged
        k = int(round(0.3 * len(d)))
        flagged = d.sort_values("gap", ascending=False).head(k)
        row["Share of discordant predictions caught by top-30% gap"] = round(flagged.gua_disc.sum() / max(1, d.gua_disc.sum()), 2)
        row["Precision of top-30% gap for discordance"] = round(flagged.gua_disc.mean(), 2)
        rows.append(row)
gap = pd.DataFrame(rows)
gap.to_csv(os.path.join(TOUT, "round6b_prediction_gap.csv"), index=False)
with open(os.path.join(TOUT, "round6b_prediction_gap.md"), "w") as f:
    f.write("Round 6b. Prospective disagreement signal |ensemble prediction - guaiacol regression| per prediction. No sensory outcome is used to compute the signal.\n\n")
    f.write(gap.to_markdown(index=False) + "\n")
print(gap.to_string())


# ------------------------------------------------------------------ 3. Parker four-phenol sum baseline (guaiacol + o-, m-, p-cresol)
# Training set of a split = all wines not in that (setting, repeat, fold) test set; every wine is tested once per repeat
# in every setting, so this reconstructs the split exactly. OLS of rating on log1p(sum of the four phenols).
FOUR = {"both": ["wine_guaiacol", "wine_o-cresol", "wine_m-cresol", "wine_p-cresol"],
        "wine": ["wine_guaiacol", "wine_o-cresol", "wine_m-cresol", "wine_p-cresol"],
        "grape": ["grape_guaiacol", "grape_o-cresol", "grape_m-cresol", "grape_p-cresol"]}
feat = {fs: np.log1p(wine[cols].sum(axis=1)) for fs, cols in FOUR.items()}
rows, per = [], []
for s, name in SETTINGS:
    for fs in ["both", "grape"]:
        d = pred[(pred.setting == s) & (pred.feature_set == fs)].copy()
        d["b_four"] = np.nan
        for (rep, fold), g in d.groupby(["repeat", "fold"]):
            test = g.wine_uid.values
            train = [w for w in wine.index if w not in set(test)]
            x_tr, y_tr = feat[fs].loc[train].values, wine.loc[train, "smoke_flavour"].values
            b1, b0 = np.polyfit(x_tr, y_tr, 1)
            d.loc[g.index, "b_four"] = b0 + b1 * feat[fs].loc[test].values
        # sanity: under cv the test sets of one repeat partition the 63 wines
        assert d.groupby("repeat").wine_uid.nunique().eq(63).all() or s.startswith("external")
        d["four_err"] = (d.b_four - d.smoke_flavour).abs()
        d["ens_err"] = (d.ensemble - d.smoke_flavour).abs()
        lo, hi = boot_diff(d, lambda x: x.ens_err.mean() - x.four_err.mean())
        ss_res = ((d.b_four - d.smoke_flavour) ** 2).sum(); ss_tot = ((d.smoke_flavour - d.smoke_flavour.mean()) ** 2).sum()
        rows.append({"Evaluation": name, "Features": {"both": "grape + wine", "grape": "grape"}[fs], "n": len(d),
                     "MAE four-phenol sum": round(d.four_err.mean(), 3), "R2 four-phenol sum": round(1 - ss_res / ss_tot, 2),
                     "MAE ensemble": round(d.ens_err.mean(), 3), "ens minus four (95% CI)": f"{lo:+.3f} to {hi:+.3f}",
                     "verdict": "ensemble lower" if hi < 0 else ("four-phenol lower" if lo > 0 else "no difference"),
                     "large_err_rate four": round((d.four_err >= LARGE).mean(), 2)})
        per.append(d[["setting", "feature_set", "repeat", "fold", "wine_uid", "b_four"]])
four = pd.DataFrame(rows)
four.to_csv(os.path.join(TOUT, "round6b_four_phenol_baseline.csv"), index=False)
pd.concat(per).to_csv(os.path.join(TOUT, "round6b_four_phenol_predictions.csv"), index=False)
with open(os.path.join(TOUT, "round6b_four_phenol_baseline.md"), "w") as f:
    f.write("Round 6b. Parker four-phenol sum (guaiacol + o-, m-, p-cresol; grape analogue for grape-only) as an OLS baseline, refitted inside every split.\n\n")
    f.write(four.to_markdown(index=False) + "\n")
print(four.to_string())
