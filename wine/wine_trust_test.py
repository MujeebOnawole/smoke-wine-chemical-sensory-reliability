"""
Wine-level trust test: does ensemble consensus plus explanation agreement
separate reliable from unreliable smoke-flavour predictions?

Data: data/processed/awri_wine_smoke_wide.csv (63 wines, AWRI Studies A and B),
built by extract_awri_wine_tables.py. Target: panel smoke flavour (0-10).

Ensemble: five deliberately different regressors (Lasso, PLS, SVR, random
forest, gradient boosting), each tuned by inner 5-fold CV inside every outer
training set (nested; no test wine is ever used for tuning).

Trust signals, computed per test wine:
- consensus  : standard deviation of the five member predictions (low = agree)
- agreement  : mean pairwise Pearson correlation of the members' compound-
               masking attributions. Attribution of analyte j for model m is
               f_m(x) - f_m(x with feature j set to its training-set median),
               the tabular analogue of substructure masking. Cross-ensemble
               correlation is used instead of a single-best-model sign rule,
               which tracks prediction extremeness rather than reliability.
Both thresholds are the medians of the same signals on the outer-training
wines, measured from inner out-of-fold predictions, so the test wines never
set their own thresholds.

Scenarios (as in the taste framework): A = high consensus + high agreement,
B = high consensus + low agreement, C = low consensus + high agreement,
D = both low.

Evaluations:
- cv       : repeated K-fold (default 5 x 5), stratified by cultivar, pooled
             out-of-fold results per repeat.
- external : train on Study B, test on Study A (primary external test), and
             train on A, test on B (secondary; A has only 23 wines).
Feature sets: wine chemistry, grape chemistry, or both; log1p concentrations
plus cultivar one-hot in every set.
"""

import argparse
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "smoke_flavour"
SCENARIOS = ["A", "B", "C", "D"]


def build_members(seed, n_jobs):
    inner = KFold(5, shuffle=True, random_state=seed)
    return {
        "lasso": make_pipeline(StandardScaler(), LassoCV(cv=inner, max_iter=50000, random_state=seed)),
        "pls": GridSearchCV(make_pipeline(StandardScaler(), PLSRegression()),
                            {"plsregression__n_components": [1, 2, 3, 4, 5]},
                            cv=inner, scoring="neg_mean_squared_error"),
        "svr": GridSearchCV(make_pipeline(StandardScaler(), SVR()),
                            {"svr__C": [0.3, 1, 3, 10, 30], "svr__gamma": ["scale", 0.01, 0.03, 0.1],
                             "svr__epsilon": [0.1, 0.3]},
                            cv=inner, scoring="neg_mean_squared_error", n_jobs=n_jobs),
        "rf": GridSearchCV(RandomForestRegressor(n_estimators=500, random_state=seed),
                           {"min_samples_leaf": [1, 3, 5], "max_features": [0.33, 0.66, 1.0]},
                           cv=inner, scoring="neg_mean_squared_error", n_jobs=n_jobs),
        "gbr": GridSearchCV(GradientBoostingRegressor(learning_rate=0.05, subsample=0.8, random_state=seed),
                            {"n_estimators": [100, 300], "max_depth": [2, 3]},
                            cv=inner, scoring="neg_mean_squared_error", n_jobs=n_jobs),
    }


def feature_matrix(df, feature_set):
    prefixes = {"wine": ["wine_"], "grape": ["grape_"], "both": ["wine_", "grape_"]}[feature_set]
    chem = [c for c in df.columns if c != "wine_uid" and any(c.startswith(p) for p in prefixes)]
    X = np.log1p(df[chem].astype(float))
    cultivar = pd.get_dummies(df["cultivar_code"], prefix="cv").astype(float)
    return pd.concat([X, cultivar], axis=1), chem


def fit_members(X, y, seed, n_jobs):
    fitted = {}
    for name, est in build_members(seed, n_jobs).items():
        est = clone(est).fit(X, y)
        fitted[name] = est.best_estimator_ if hasattr(est, "best_estimator_") else est
    return fitted


def predict_members(fitted, X):
    return np.column_stack([np.ravel(m.predict(X)) for m in fitted.values()])


def attributions(fitted, X, chem_idx, baseline):
    """(n_samples, n_models, n_chem): prediction drop when each analyte is masked."""
    base = predict_members(fitted, X)
    out = np.zeros((X.shape[0], base.shape[1], len(chem_idx)))
    for k, j in enumerate(chem_idx):
        Xm = X.copy()
        Xm[:, j] = baseline[j]
        out[:, :, k] = base - predict_members(fitted, Xm)
    return out


def pairwise_agreement(attr):
    """Mean pairwise Pearson r of attribution vectors across models, per sample."""
    scores = []
    for a in attr:
        rs = []
        for i, k in itertools.combinations(range(a.shape[0]), 2):
            if np.std(a[i]) < 1e-12 or np.std(a[k]) < 1e-12:
                rs.append(0.0)  # a model that ignores every analyte cannot agree
            else:
                rs.append(pearsonr(a[i], a[k])[0])
        scores.append(float(np.mean(rs)))
    return np.array(scores)


def trust_signals(fitted, X, chem_idx, baseline):
    preds = predict_members(fitted, X)
    return preds, preds.std(axis=1), pairwise_agreement(attributions(fitted, X, chem_idx, baseline))


def reference_thresholds(X, y, fitted, chem_idx, seed):
    """Median consensus and agreement on training wines, from inner out-of-fold fits."""
    stds, agrs = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        inner = {n: clone(m).fit(X[tr], y[tr]) for n, m in fitted.items()}
        baseline = np.median(X[tr], axis=0)
        _, stds[va], agrs[va] = trust_signals(inner, X[va], chem_idx, baseline)
    return float(np.median(stds)), float(np.median(agrs))


def run_split(X, y, tr, te, chem_idx, seed, n_jobs, chem_names=None):
    fitted = fit_members(X[tr], y[tr], seed, n_jobs)
    tau_c, tau_a = reference_thresholds(X[tr], y[tr], fitted, chem_idx, seed)
    baseline = np.median(X[tr], axis=0)
    preds, std, agr = trust_signals(fitted, X[te], chem_idx, baseline)
    high_c, high_a = std <= tau_c, agr >= tau_a
    scenario = np.where(high_c & high_a, "A", np.where(high_c, "B", np.where(high_a, "C", "D")))
    rows = pd.DataFrame(preds, columns=[f"pred_{n}" for n in fitted])
    rows["idx"] = te
    rows["pred_ensemble"] = preds.mean(axis=1)
    rows["consensus_std"] = std
    rows["attr_agreement"] = agr
    rows["tau_consensus"] = tau_c
    rows["tau_agreement"] = tau_a
    rows["scenario"] = scenario
    # Per-analyte attribution (mean over the five members), saved for the odour-activity check
    attr = attributions(fitted, X[te], chem_idx, baseline).mean(axis=1)
    names = chem_names if chem_names is not None else [f"j{j}" for j in chem_idx]
    for k, name in enumerate(names):
        rows[f"attr_{name}"] = attr[:, k]
    return rows


def summarise(res, y_col=TARGET):
    y, p = res[y_col].to_numpy(), res["pred_ensemble"].to_numpy()
    err = np.abs(y - p)
    out = {
        "n": int(len(res)),
        "ensemble_r2": float(r2_score(y, p)),
        "ensemble_rmse": float(np.sqrt(mean_squared_error(y, p))),
        "ensemble_mae": float(mean_absolute_error(y, p)),
        "ensemble_pearson": float(pearsonr(y, p)[0]),
    }
    for c in [c for c in res.columns if c.startswith("pred_") and c != "pred_ensemble"]:
        out[f"{c[5:]}_rmse"] = float(np.sqrt(mean_squared_error(y, res[c])))
    for s in SCENARIOS:
        m = res["scenario"].to_numpy() == s
        out[f"scenario_{s}_frac"] = float(m.mean())
        out[f"scenario_{s}_mae"] = float(err[m].mean()) if m.any() else None
    ab = np.isin(res["scenario"], ["A", "B"])
    out["mae_AB"] = float(err[ab].mean()) if ab.any() else None
    out["mae_CD"] = float(err[~ab].mean()) if (~ab).any() else None
    out["mae_ratio_CD_over_AB"] = (out["mae_CD"] / out["mae_AB"]) if out["mae_AB"] and out["mae_CD"] else None
    out["spearman_abs_err_vs_consensus_std"] = float(spearmanr(err, res["consensus_std"])[0])
    out["spearman_abs_err_vs_agreement"] = float(spearmanr(err, res["attr_agreement"])[0])
    return out


def bootstrap_ratio(res, n_boot, seed):
    rng = np.random.default_rng(seed)
    err = np.abs(res[TARGET].to_numpy() - res["pred_ensemble"].to_numpy())
    ab = np.isin(res["scenario"].to_numpy(), ["A", "B"])
    ratios = []
    for _ in range(n_boot):
        i = rng.integers(0, len(err), len(err))
        if ab[i].any() and (~ab[i]).any():
            ratios.append(err[i][~ab[i]].mean() / err[i][ab[i]].mean())
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(here, "..", "data", "awri_wine_smoke_wide_labelled.csv"))
    ap.add_argument("--out_dir", default=os.path.join(here, "..", "results", "wine", "wine_trust"))
    ap.add_argument("--feature_sets", nargs="+", default=["both", "wine", "grape"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    y = df[TARGET].to_numpy(float)
    summary_rows = []

    for fs in args.feature_sets:
        Xdf, chem = feature_matrix(df, fs)
        X = Xdf.to_numpy(float)
        chem_idx = [Xdf.columns.get_loc(c) for c in chem]
        print(f"\n=== feature set: {fs} ({len(chem)} analytes) ===", flush=True)

        # Repeated stratified CV
        cv = RepeatedStratifiedKFold(n_splits=args.n_folds, n_repeats=args.n_repeats, random_state=args.seed)
        parts = []
        for k, (tr, te) in enumerate(cv.split(X, df["cultivar_code"])):
            r = run_split(X, y, tr, te, chem_idx, args.seed + k, args.n_jobs, chem)
            r["repeat"], r["fold"] = k // args.n_folds, k % args.n_folds
            parts.append(r)
            print(f"  cv split {k + 1}/{args.n_folds * args.n_repeats} done", flush=True)
        oof = pd.concat(parts, ignore_index=True)
        oof = oof.merge(df[["wine_uid", "study", "cultivar_code", TARGET]], left_on="idx", right_index=True)
        oof.to_csv(os.path.join(args.out_dir, f"cv_oof_{fs}.csv"), index=False)
        per_rep = pd.DataFrame([summarise(g) for _, g in oof.groupby("repeat")])
        agg = per_rep.mean(numeric_only=True).add_suffix("_mean").to_dict()
        agg.update(per_rep.std(numeric_only=True).add_suffix("_sd").to_dict())
        summary_rows.append({"evaluation": "cv_5x5", "feature_set": fs, **agg})

        # External study transfer
        for train_s, test_s in [("B", "A"), ("A", "B")]:
            tr = np.where(df["study"] == train_s)[0]
            te = np.where(df["study"] == test_s)[0]
            r = run_split(X, y, tr, te, chem_idx, args.seed, args.n_jobs, chem)
            r = r.merge(df[["wine_uid", "study", "cultivar_code", TARGET]], left_on="idx", right_index=True)
            r.to_csv(os.path.join(args.out_dir, f"external_train{train_s}_test{test_s}_{fs}.csv"), index=False)
            s = summarise(r)
            s["mae_ratio_ci95"] = bootstrap_ratio(r, args.n_boot, args.seed)
            summary_rows.append({"evaluation": f"external_train{train_s}_test{test_s}", "feature_set": fs, **s})
            print(f"  external train {train_s} -> test {test_s}: R2={s['ensemble_r2']:.2f} "
                  f"C+D/A+B MAE ratio={s['mae_ratio_CD_over_AB']}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.out_dir, "wine_trust_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "run_params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    cols = ["evaluation", "feature_set"] + [c for c in summary.columns if c.startswith(
        ("ensemble_r2", "ensemble_mae", "scenario_A_frac", "mae_ratio", "spearman"))]
    print("\n" + summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
