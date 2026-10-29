"""
Wine smoke-flavour assessment: baselines, chemical-sensory discordance, uncertainty-guided
screening and stage (grape vs wine) comparison on the 63 AWRI wines.

All criteria below were fixed before the script was run on the full settings.

Evaluations (identical splits and seeds to wine_trust_test.py / wine_trust_grouped.py, so
the ensemble numbers reproduce those runs):
  cv                     5 x 5 stratified (cultivar) CV, RepeatedStratifiedKFold(random_state=42)
  external_trainB_testA  train Study B (40), test Study A (23)
  external_trainA_testB  train Study A, test Study B
  grouped_site           whole vineyards (A) / regions (B) held out, 5 x 5
  grouped_region         whole regions held out, Study A pooled as Adelaide Hills, 5 x 5
  cultivar_holdout       leave one cultivar out (3 folds); chemistry only, no cultivar indicator,
                         because the held-out cultivar's indicator is constant in training
Feature sets: both, wine, grape (log1p concentrations + cultivar one-hot except cultivar_holdout).

Models fitted on every training set:
  ensemble        five tuned regressors (Lasso, PLS, SVR, RF, GBR), mean prediction (existing)
  lasso, pls, svr, rf, gbr   the members alone
  b_guaiacol      OLS on log1p guaiacol (wine guaiacol when wine features are present)
  b_best_single   OLS on the single analyte most correlated with the target in training (Tan et al. 2024)
  b_phenol_index  OLS on the mean standardised log1p free volatile phenol (not glycoside) concentration
  b_marker_index  OLS on the mean standardised log1p concentration of all analytes
  b_cultivar_mean training mean of the same cultivar (global training mean under cultivar_holdout)

Uncertainty rankings for the ensemble prediction (higher = less reliable):
  consensus_std   SD of the five member predictions (the paper's consensus signal)
  attr_disagree   1 - mean pairwise Pearson r of member attribution vectors (the paper's agreement signal)
  rf_tree_std     SD of the random forest's per-tree predictions (single-model baseline)
  knn_distance    mean Euclidean distance to the 3 nearest training wines in standardised chemistry space
  pred_extreme    |ensemble prediction - training mean target|
  random          uniform noise, seeded
  oracle          the absolute error itself (upper bound)

Analyses written by --analyse (also runnable alone on a pulled per_wine_predictions.csv):
  1 baseline_comparison.csv  MAE, R2, large-error rate per model; paired |err| difference versus
                             the ensemble with a wine-cluster bootstrap CI
  2 discordance_*.csv        reference model = b_phenol_index. A wine is discordant in a prediction when
                             |reference residual| >= DISCORD_UNITS (1.5 rating units, about the 95th
                             percentile of the ensemble's CV error). Per-wine discordance share, sign,
                             cultivar / study / glycoside-share / Dunnett associations, ensemble error on
                             discordant wines, and AUROC of each uncertainty ranking for flagging them
  3 risk_coverage.csv        MAE of retained predictions at coverage 0.5 to 1.0 for every ranking,
                             AURC per ranking with paired bootstrap CI against consensus_std, and the
                             share of large errors (>= LARGE_ERR units) caught when 20 / 30 % are withheld
  4 triage.csv               binary smoky (Dunnett) classification from the ensemble score. Threshold =
                             midpoint of the training means of smoky and non-smoky wines (labels only,
                             no test outcome). FN / FP rates with and without abstention on the 20 / 30 %
                             least reliable predictions, plus AUROC of the score. Sensitivity thresholds
                             3, 4 and 5 are reported as exploratory
  5 stage_comparison.csv     paired |err| difference between feature sets within the same split
Bootstrap: wines are resampled as clusters (all rows of a wine move together), 2,000 resamples, seed 42.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from wine_trust_grouped import group_folds
from wine_trust_test import (TARGET, attributions, feature_matrix, fit_members, pairwise_agreement,
                             predict_members, reference_thresholds)

warnings.filterwarnings("ignore")

LABEL = "smoky_dunnett"
DISCORD_REFERENCES = ["b_guaiacol", "b_phenol_index"]  # primary: guaiacol alone (routinely measured marker)
DISCORD_UNITS = 1.5
LARGE_ERR = 1.5
COVERAGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
WITHHOLD = [0.2, 0.3]
EXPLORATORY_THRESHOLDS = [3.0, 4.0, 5.0]
RANKINGS = ["consensus_std", "attr_disagree", "rf_tree_std", "knn_distance", "pred_extreme", "random", "oracle"]
BASELINES = ["b_guaiacol", "b_best_single", "b_phenol_index", "b_marker_index", "b_cultivar_mean"]
MEMBERS = ["lasso", "pls", "svr", "rf", "gbr"]
MODELS = ["ensemble"] + MEMBERS + BASELINES


def is_glycoside(name):
    return name.endswith(("RG", "GG"))


# ----------------------------------------------------------------------------- fitting
def ols_on(cols_tr, y_tr, cols_te):
    m = LinearRegression().fit(cols_tr, y_tr)
    return m.predict(cols_te)


def index_feature(Xtr, Xte, idx):
    sc = StandardScaler().fit(Xtr[:, idx])
    return sc.transform(Xtr[:, idx]).mean(axis=1, keepdims=True), sc.transform(Xte[:, idx]).mean(axis=1, keepdims=True)


def baselines(X, y, tr, te, chem, chem_idx, cultivar, holdout_cultivar):
    out = {}
    Xtr, Xte, ytr = X[tr], X[te], y[tr]
    names = list(chem)
    # guaiacol itself (exact name, so 4-methylguaiacol is never matched): wine_ when present, otherwise grape_
    g = [c for c in names if c in ("wine_guaiacol",)] or [c for c in names if c in ("grape_guaiacol",)]
    if len(g) != 1:
        raise ValueError(f"expected exactly one guaiacol column, found {g}")
    gi = [chem_idx[names.index(g[0])]]
    out["b_guaiacol"] = ols_on(Xtr[:, gi], ytr, Xte[:, gi])
    # best single analyte, selected on training
    cors = [abs(pearsonr(Xtr[:, j], ytr)[0]) if np.std(Xtr[:, j]) > 0 else 0 for j in chem_idx]
    bj = [chem_idx[int(np.argmax(cors))]]
    out["b_best_single"] = ols_on(Xtr[:, bj], ytr, Xte[:, bj])
    out["best_single_name"] = names[int(np.argmax(cors))]
    # indices
    ph = [chem_idx[k] for k, c in enumerate(names) if not is_glycoside(c)]
    ftr, fte = index_feature(Xtr, Xte, ph)
    out["b_phenol_index"] = ols_on(ftr, ytr, fte)
    ftr, fte = index_feature(Xtr, Xte, chem_idx)
    out["b_marker_index"] = ols_on(ftr, ytr, fte)
    # cultivar mean
    if holdout_cultivar:
        out["b_cultivar_mean"] = np.full(len(te), ytr.mean())
    else:
        means = pd.Series(ytr).groupby(cultivar[tr]).mean()
        out["b_cultivar_mean"] = np.array([means.get(c, ytr.mean()) for c in cultivar[te]])
    return out


def knn_distance(Xtr, Xte, chem_idx, k=3):
    sc = StandardScaler().fit(Xtr[:, chem_idx])
    A, B = sc.transform(Xtr[:, chem_idx]), sc.transform(Xte[:, chem_idx])
    d = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    return np.sort(d, axis=1)[:, :k].mean(axis=1)


def run_split(df, X, y, tr, te, chem, chem_idx, seed, n_jobs, holdout_cultivar=False):
    fitted = fit_members(X[tr], y[tr], seed, n_jobs)
    tau_c, tau_a = reference_thresholds(X[tr], y[tr], fitted, chem_idx, seed)
    baseline = np.median(X[tr], axis=0)
    preds = predict_members(fitted, X[te])
    agr = pairwise_agreement(attributions(fitted, X[te], chem_idx, baseline))
    std = preds.std(axis=1)
    high_c, high_a = std <= tau_c, agr >= tau_a
    rows = pd.DataFrame({"idx": te, "wine_uid": df.wine_uid.values[te], "study": df.study.values[te],
                         "cultivar_code": df.cultivar_code.values[te], TARGET: y[te], LABEL: df[LABEL].values[te]})
    for k, n in enumerate(fitted):
        rows[n] = preds[:, k]
    rows["ensemble"] = preds.mean(axis=1)
    rows["consensus_std"] = std
    rows["attr_disagree"] = 1.0 - agr
    rows["scenario"] = np.where(high_c & high_a, "A", np.where(high_c, "B", np.where(high_a, "C", "D")))
    rf = fitted["rf"]
    rows["rf_tree_std"] = np.column_stack([t.predict(X[te]) for t in rf.estimators_]).std(axis=1)
    rows["knn_distance"] = knn_distance(X[tr], X[te], chem_idx)
    rows["pred_extreme"] = np.abs(rows["ensemble"] - y[tr].mean())
    rows["random"] = np.random.default_rng(seed).uniform(size=len(te))
    rows["oracle"] = np.abs(rows["ensemble"] - y[te])
    b = baselines(X, y, tr, te, chem, chem_idx, df.cultivar_code.values, holdout_cultivar)
    for n in BASELINES:
        rows[n] = b[n]
    rows["best_single_name"] = b["best_single_name"]
    # triage threshold from training labels only
    ytr, ltr = y[tr], df[LABEL].values[tr]
    rows["triage_threshold"] = 0.5 * (ytr[ltr == 1].mean() + ytr[ltr == 0].mean()) if (ltr == 1).any() and (ltr == 0).any() else np.nan
    rows["n_train"] = len(tr)
    return rows


def compute(df, args):
    y = df[TARGET].to_numpy(float)
    parts = []
    for fs in args.feature_sets:
        Xdf, chem = feature_matrix(df, fs)
        X = Xdf.to_numpy(float)
        chem_idx = [Xdf.columns.get_loc(c) for c in chem]
        Xchem = Xdf[chem].to_numpy(float)
        chem_idx_only = list(range(len(chem)))
        print(f"\n=== feature set {fs} ({len(chem)} analytes) ===", flush=True)

        def add(setting, rows, rep, fold):
            rows["setting"], rows["feature_set"], rows["repeat"], rows["fold"] = setting, fs, rep, fold
            parts.append(rows)

        cv = RepeatedStratifiedKFold(n_splits=args.n_folds, n_repeats=args.n_repeats, random_state=args.seed)
        for k, (tr, te) in enumerate(cv.split(X, df.cultivar_code)):
            add("cv", run_split(df, X, y, tr, te, chem, chem_idx, args.seed + k, args.n_jobs), k // args.n_folds, k % args.n_folds)
        print("  cv done", flush=True)

        for a, b in [("B", "A"), ("A", "B")]:
            tr, te = np.where(df.study == a)[0], np.where(df.study == b)[0]
            add(f"external_train{a}_test{b}", run_split(df, X, y, tr, te, chem, chem_idx, args.seed, args.n_jobs), 0, 0)
        print("  external done", flush=True)

        for gcol in ["group_site", "group_region"]:
            groups = df[gcol].to_numpy()
            for rep in range(args.n_repeats):
                fold = group_folds(groups, args.n_folds, np.random.default_rng(args.seed + rep))
                for k in range(args.n_folds):
                    te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
                    if len(te) == 0:
                        continue
                    add(f"grouped_{gcol[6:]}", run_split(df, X, y, tr, te, chem, chem_idx, args.seed + rep * 100 + k, args.n_jobs), rep, k)
            print(f"  {gcol} done", flush=True)

        for k, cult in enumerate(sorted(df.cultivar_code.unique())):
            te, tr = np.where(df.cultivar_code == cult)[0], np.where(df.cultivar_code != cult)[0]
            add("cultivar_holdout", run_split(df, Xchem, y, tr, te, chem, chem_idx_only, args.seed, args.n_jobs, holdout_cultivar=True), 0, k)
        print("  cultivar holdout done", flush=True)

    per_wine = pd.concat(parts, ignore_index=True)
    analytes = [c for c in df.columns if c.startswith("wine_") and c != "wine_uid"]
    glyc = [c for c in analytes if is_glycoside(c[5:])]
    phen = [c for c in analytes if not is_glycoside(c[5:])]
    share = (df[glyc].sum(axis=1) / (df[glyc].sum(axis=1) + df[phen].sum(axis=1))).rename("wine_glycoside_share")
    per_wine = per_wine.merge(pd.concat([df.wine_uid, share, df[phen].sum(axis=1).rename("wine_phenol_sum")], axis=1), on="wine_uid")
    return per_wine


# ----------------------------------------------------------------------------- analyses
def cluster_boot(d, stat, n_boot, seed):
    """Bootstrap over wines: resample wine_uids with replacement, keep all their rows."""
    rng = np.random.default_rng(seed)
    uids = d.wine_uid.unique()
    groups = {u: g for u, g in d.groupby("wine_uid")}
    vals = []
    for _ in range(n_boot):
        samp = pd.concat([groups[u] for u in rng.choice(uids, len(uids))], ignore_index=True)
        v = stat(samp)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return tuple(np.percentile(vals, [2.5, 97.5]))


def settings_iter(pw):
    for (s, fs), d in pw.groupby(["setting", "feature_set"]):
        yield s, fs, d


def analyse_baselines(pw, n_boot, seed):
    rows = []
    for s, fs, d in settings_iter(pw):
        y = d[TARGET].to_numpy()
        e_ens = np.abs(d["ensemble"] - y)
        for m in MODELS:
            e = np.abs(d[m] - y)
            r = {"setting": s, "feature_set": fs, "model": m, "n_pred": len(d), "mae": e.mean(),
                 "r2": r2_score(y, d[m]), "large_err_rate": (e >= LARGE_ERR).mean(),
                 "mae_diff_ens_minus_model": e_ens.mean() - e.mean()}
            if m != "ensemble":
                lo, hi = cluster_boot(d, lambda b: (np.abs(b["ensemble"] - b[TARGET]) - np.abs(b[m] - b[TARGET])).mean(), n_boot, seed)
                r["mae_diff_ci_lo"], r["mae_diff_ci_hi"] = lo, hi
            rows.append(r)
        rows.append({"setting": s, "feature_set": fs, "model": "best_single_name_mode", "n_pred": len(d),
                     "note": d.best_single_name.mode().iloc[0]})
    return pd.DataFrame(rows)


def analyse_discordance(pw, n_boot, seed):
    per_wine_rows, test_rows = [], []
    for ref in DISCORD_REFERENCES:
        for s, fs, d in settings_iter(pw):
            g, t = _discordance_one(d.copy(), ref, s, fs, n_boot, seed)
            per_wine_rows.append(g)
            test_rows.append(t)
    return pd.concat(per_wine_rows, ignore_index=True), pd.DataFrame(test_rows)


def _discordance_one(d, ref, s, fs, n_boot, seed):
    d["ref_resid"] = d[ref] - d[TARGET]  # positive: chemistry predicts smokier than panel
    d["discordant"] = (d.ref_resid.abs() >= DISCORD_UNITS).astype(int)
    d["ens_err"], d["ref_err"] = np.abs(d["ensemble"] - d[TARGET]), d.ref_resid.abs()
    g = d.groupby(["wine_uid", "study", "cultivar_code"]).agg(
        n_pred=("discordant", "size"), discordant_share=("discordant", "mean"),
        mean_ref_resid=("ref_resid", "mean"), mean_ens_err=("ens_err", "mean"), mean_ref_err=("ref_err", "mean"),
        smoke_flavour=(TARGET, "first"), smoky_dunnett=(LABEL, "first"),
        glycoside_share=("wine_glycoside_share", "first"), phenol_sum=("wine_phenol_sum", "first"),
        mean_consensus_std=("consensus_std", "mean")).reset_index()
    g["setting"], g["feature_set"], g["reference"] = s, fs, ref

    disc = d.discordant.values == 1
    t = {"reference": ref, "setting": s, "feature_set": fs, "n_pred": len(d), "n_discordant_pred": int(disc.sum()),
         "n_wines_ever_discordant": int((g.discordant_share > 0).sum()),
         "share_overpredicted": float((d.ref_resid[disc] > 0).mean()) if disc.any() else np.nan,
         "ref_mae_discordant": d.ref_err[disc].mean() if disc.any() else np.nan,
         "ens_mae_discordant": d.ens_err[disc].mean() if disc.any() else np.nan,
         "ens_mae_concordant": d.ens_err[~disc].mean() if (~disc).any() else np.nan}
    if disc.sum() >= 3:
        lo, hi = cluster_boot(d[disc], lambda b: (b.ref_err - b.ens_err).mean(), n_boot, seed)
        t["ref_minus_ens_mae_discordant_ci"] = (lo, hi)
        for rk in RANKINGS:
            if rk != "oracle":
                t[f"auroc_flag_discordant_{rk}"] = roc_auc_score(disc, d[rk]) if 0 < disc.mean() < 1 else np.nan
    # associations at the wine level (share > 0.5 counts as discordant wine) using one row per wine
    w = g.copy()
    w["disc_wine"] = (w.discordant_share > 0.5).astype(int)
    if w.disc_wine.sum() >= 2 and (w.disc_wine == 0).sum() >= 2:
        for col in ["cultivar_code", "study"]:
            tab = pd.crosstab(w[col], w.disc_wine)
            t[f"{col}_counts"] = tab.to_dict()
            if tab.shape == (2, 2):
                t[f"{col}_fisher_p"] = fisher_exact(tab.values)[1]
        for col in ["glycoside_share", "phenol_sum", "smoke_flavour"]:
            a, b = w.loc[w.disc_wine == 1, col], w.loc[w.disc_wine == 0, col]
            t[f"{col}_mwu_p"] = mannwhitneyu(a, b).pvalue
            t[f"{col}_median_disc_vs_conc"] = (a.median(), b.median())
        tab = pd.crosstab(w.smoky_dunnett, w.disc_wine)
        t["dunnett_counts"] = tab.to_dict()
    return g, t


def risk_curve(d, rank_col, coverages):
    """MAE of retained predictions when the highest-ranked (least reliable) are withheld, within each repeat."""
    out = []
    for cov in coverages:
        errs = []
        for _, g in d.groupby("repeat"):
            n_keep = max(1, int(round(cov * len(g))))
            keep = g.sort_values(rank_col, kind="stable").iloc[:n_keep]
            errs.append(np.abs(keep["ensemble"] - keep[TARGET]).to_numpy())
        out.append(np.concatenate(errs).mean())
    return np.array(out)


def analyse_risk_coverage(pw, n_boot, seed):
    curve_rows, aurc_rows = [], []
    for s, fs, d in settings_iter(pw):
        base = risk_curve(d, "consensus_std", COVERAGES)
        for rk in RANKINGS:
            c = risk_curve(d, rk, COVERAGES)
            for cov, v in zip(COVERAGES, c):
                curve_rows.append({"setting": s, "feature_set": fs, "ranking": rk, "coverage": cov, "mae_retained": v})
            r = {"setting": s, "feature_set": fs, "ranking": rk, "aurc": c.mean(), "aurc_minus_consensus": c.mean() - base.mean()}
            if rk != "consensus_std":
                lo, hi = cluster_boot(d, lambda b: risk_curve(b, rk, COVERAGES).mean() - risk_curve(b, "consensus_std", COVERAGES).mean(), n_boot, seed)
                r["aurc_diff_ci_lo"], r["aurc_diff_ci_hi"] = lo, hi
            # large errors caught when withholding
            err = np.abs(d["ensemble"] - d[TARGET])
            large = err >= LARGE_ERR
            r["n_large_err"] = int(large.sum())
            for w in WITHHOLD:
                caught = 0
                for _, g in d.groupby("repeat"):
                    n_w = int(round(w * len(g)))
                    held = g.sort_values(rk, kind="stable").iloc[len(g) - n_w:] if n_w > 0 else g.iloc[:0]
                    caught += int((np.abs(held["ensemble"] - held[TARGET]) >= LARGE_ERR).sum())
                r[f"large_err_caught_withhold{int(w * 100)}"] = caught / large.sum() if large.sum() else np.nan
            aurc_rows.append(r)
    return pd.DataFrame(curve_rows), pd.DataFrame(aurc_rows)


def triage_metrics(d, thr_col_or_val, abstain_frac, rank_col="consensus_std"):
    """Classify every prediction, then split into retained and abstained by the ranking.

    Rates are over retained predictions; the *_abstained counts say how many of the
    classification errors the abstention removed (errors avoided), and n_pos_abstained
    how many smoky wines were withheld along with them.
    """
    thr = d[thr_col_or_val] if isinstance(thr_col_or_val, str) else thr_col_or_val
    pred_pos = d["ensemble"] >= thr
    pos = d[LABEL] == 1
    fn, fp = (~pred_pos & pos), (pred_pos & ~pos)
    keep = pd.Series(True, index=d.index)
    for _, g in d.groupby("repeat"):
        n_a = int(round(abstain_frac * len(g)))
        if n_a > 0:
            keep.loc[g.sort_values(rank_col, kind="stable").index[len(g) - n_a:]] = False
    return {"n_retained": int(keep.sum()), "n_pos_retained": int((pos & keep).sum()),
            "fn_rate": float((fn & keep).sum() / (pos & keep).sum()) if (pos & keep).any() else np.nan,
            "fp_rate": float((fp & keep).sum() / (~pos & keep).sum()) if (~pos & keep).any() else np.nan,
            "n_fn": int((fn & keep).sum()), "n_fp": int((fp & keep).sum()),
            "n_fn_abstained": int((fn & ~keep).sum()), "n_fp_abstained": int((fp & ~keep).sum()),
            "n_pos_abstained": int((pos & ~keep).sum()),
            "accuracy": float((pred_pos == pos)[keep].mean()),
            "n_errors_total": int((fn | fp).sum())}


def analyse_triage(pw):
    rows = []
    for s, fs, d in settings_iter(pw):
        pos = d[LABEL] == 1
        auc = roc_auc_score(pos, d["ensemble"]) if 0 < pos.mean() < 1 else np.nan
        for thr_name, thr in [("training_midpoint", "triage_threshold")] + [(f"exploratory_{t}", t) for t in EXPLORATORY_THRESHOLDS]:
            for ab in [0.0] + WITHHOLD:
                for rk in ["consensus_std", "rf_tree_std", "knn_distance", "random"]:
                    if ab == 0.0 and rk != "consensus_std":
                        continue
                    m = triage_metrics(d, thr, ab, rk)
                    rows.append({"setting": s, "feature_set": fs, "threshold": thr_name,
                                 "threshold_value_mean": d["triage_threshold"].mean() if thr_name == "training_midpoint" else thr,
                                 "abstain_frac": ab, "ranking": rk if ab > 0 else "none", "score_auroc": auc, **m})
    return pd.DataFrame(rows)


def analyse_stage(pw, n_boot, seed):
    rows = []
    for s, d in pw.groupby("setting"):
        fsets = sorted(d.feature_set.unique())
        key = ["wine_uid", "repeat", "fold"]
        for i, a in enumerate(fsets):
            for b in fsets[i + 1:]:
                m = d[d.feature_set == a].merge(d[d.feature_set == b], on=key, suffixes=("_a", "_b"))
                if m.empty:
                    continue
                m["diff"] = np.abs(m[f"ensemble_a"] - m[f"{TARGET}_a"]) - np.abs(m[f"ensemble_b"] - m[f"{TARGET}_b"])
                m["pred_gap"] = np.abs(m["ensemble_a"] - m["ensemble_b"])
                lo, hi = cluster_boot(m, lambda x: x["diff"].mean(), n_boot, seed)
                rows.append({"setting": s, "feature_a": a, "feature_b": b, "n_pairs": len(m),
                             "mae_a": np.abs(m["ensemble_a"] - m[f"{TARGET}_a"]).mean(),
                             "mae_b": np.abs(m["ensemble_b"] - m[f"{TARGET}_b"]).mean(),
                             "mae_a_minus_b": m["diff"].mean(), "ci_lo": lo, "ci_hi": hi,
                             "share_pred_gap_over_1": (m.pred_gap > 1).mean(),
                             "spearman_gap_vs_err_a": spearmanr(m.pred_gap, np.abs(m["ensemble_a"] - m[f"{TARGET}_a"]))[0]})
    return pd.DataFrame(rows)


def analyse(per_wine, out_dir, n_boot, seed):
    pw = per_wine
    analyse_baselines(pw, n_boot, seed).to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)
    print("baselines done", flush=True)
    dw, dt = analyse_discordance(pw, n_boot, seed)
    dw.to_csv(os.path.join(out_dir, "discordance_wines.csv"), index=False)
    dt.to_csv(os.path.join(out_dir, "discordance_tests.csv"), index=False)
    print("discordance done", flush=True)
    rc, au = analyse_risk_coverage(pw, n_boot, seed)
    rc.to_csv(os.path.join(out_dir, "risk_coverage.csv"), index=False)
    au.to_csv(os.path.join(out_dir, "risk_coverage_aurc.csv"), index=False)
    print("risk coverage done", flush=True)
    analyse_triage(pw).to_csv(os.path.join(out_dir, "triage.csv"), index=False)
    print("triage done", flush=True)
    analyse_stage(pw, n_boot, seed).to_csv(os.path.join(out_dir, "stage_comparison.csv"), index=False)
    print("stage done", flush=True)

    # console digest
    bc = pd.read_csv(os.path.join(out_dir, "baseline_comparison.csv"))
    print("\nMAE by model (rows: setting / feature set):")
    print(bc[bc.model.isin(MODELS)].pivot_table(index=["setting", "feature_set"], columns="model", values="mae").round(2).to_string())
    print("\nAURC by ranking (lower is better):")
    print(au.pivot_table(index=["setting", "feature_set"], columns="ranking", values="aurc").round(3).to_string())
    print("\nDiscordance by reference model:")
    cols = [c for c in dt.columns if c in ("reference", "setting", "feature_set", "n_pred", "n_discordant_pred", "n_wines_ever_discordant",
                                           "share_overpredicted", "ref_mae_discordant", "ens_mae_discordant",
                                           "auroc_flag_discordant_consensus_std", "auroc_flag_discordant_knn_distance")]
    print(dt[cols].round(2).to_string(index=False))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(here, "..", "data", "awri_wine_smoke_wide_labelled.csv"))
    ap.add_argument("--out_dir", default=os.path.join(here, "..", "results", "wine", "wine_assessment"))
    ap.add_argument("--feature_sets", nargs="+", default=["both", "wine", "grape"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyse_only", action="store_true", help="skip fitting; read per_wine_predictions.csv from out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pw_path = os.path.join(args.out_dir, "per_wine_predictions.csv")

    if args.analyse_only:
        per_wine = pd.read_csv(pw_path)
    else:
        df = pd.read_csv(args.data)
        assert LABEL in df.columns, f"{LABEL} missing: run add_wine_significance.py first"
        per_wine = compute(df, args)
        per_wine.to_csv(pw_path, index=False)
        with open(os.path.join(args.out_dir, "run_params.json"), "w") as f:
            json.dump({**vars(args), "DISCORD_UNITS": DISCORD_UNITS, "LARGE_ERR": LARGE_ERR,
                       "COVERAGES": COVERAGES, "WITHHOLD": WITHHOLD, "EXPLORATORY_THRESHOLDS": EXPLORATORY_THRESHOLDS}, f, indent=2)
        print(f"\nwrote {pw_path}: {per_wine.shape}", flush=True)
    analyse(per_wine, args.out_dir, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
