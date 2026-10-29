"""
Grouped cross-validation for the wine-level trust test.

Reuses run_split/summarise/feature_matrix from wine_trust_test.py (same ensemble,
same nested tuning, same thresholds) and only changes how wines are split: all wines
from one site are held out together, so no test wine shares a vineyard (Study A) or
region (Study B) with a training wine.

Groupings (columns from add_wine_groups.py)
  group_site   : Study A vineyard letter, Study B region (primary)
  group_region : region only, Study A pooled into Adelaide Hills (strict sensitivity)

Group-to-fold assignment is reshuffled each repeat (largest groups placed first into
the currently smallest fold), so repeats differ. Output per grouping and feature set:
pooled out-of-fold predictions per repeat, summary mean/sd over repeats, and the
C+D / A+B error ratio with a bootstrap interval over wines in the pooled first repeat.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from wine_trust_test import TARGET, bootstrap_ratio, feature_matrix, run_split, summarise


def group_folds(groups, n_folds, rng):
    """Assign whole groups to folds; shuffle ties, place big groups first into the smallest fold."""
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_of, load = {}, np.zeros(n_folds)
    for i in order:
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += counts[i]
    return np.array([fold_of[g] for g in groups])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(here, "..", "data", "awri_wine_smoke_wide_labelled.csv"))
    ap.add_argument("--out_dir", default=os.path.join(here, "..", "results", "wine", "wine_trust_grouped"))
    ap.add_argument("--group_cols", nargs="+", default=["group_site", "group_region"])
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
    rows = []
    for gcol in args.group_cols:
        groups = df[gcol].to_numpy()
        print(f"\n##### grouping {gcol}: {len(np.unique(groups))} groups", flush=True)
        for fs in args.feature_sets:
            Xdf, chem = feature_matrix(df, fs)
            X = Xdf.to_numpy(float)
            chem_idx = [Xdf.columns.get_loc(c) for c in chem]
            parts = []
            for rep in range(args.n_repeats):
                fold = group_folds(groups, args.n_folds, np.random.default_rng(args.seed + rep))
                for k in range(args.n_folds):
                    te = np.where(fold == k)[0]
                    tr = np.where(fold != k)[0]
                    if len(te) == 0:
                        continue
                    r = run_split(X, y, tr, te, chem_idx, args.seed + rep * 100 + k, args.n_jobs, chem)
                    r["repeat"], r["fold"] = rep, k
                    parts.append(r)
                print(f"  {gcol} {fs} repeat {rep + 1}/{args.n_repeats} done", flush=True)
            oof = pd.concat(parts, ignore_index=True)
            oof = oof.merge(df[["wine_uid", "study", "cultivar_code", TARGET, gcol]],
                            left_on="idx", right_index=True)
            oof.to_csv(os.path.join(args.out_dir, f"oof_{gcol}_{fs}.csv"), index=False)
            per_rep = pd.DataFrame([summarise(g) for _, g in oof.groupby("repeat")])
            agg = per_rep.mean(numeric_only=True).add_suffix("_mean").to_dict()
            agg.update(per_rep.std(numeric_only=True).add_suffix("_sd").to_dict())
            agg["mae_ratio_ci95_repeat0"] = bootstrap_ratio(oof[oof.repeat == 0], args.n_boot, args.seed)
            rows.append({"evaluation": f"grouped_cv_{gcol}", "feature_set": fs, **agg})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "wine_trust_grouped_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "run_params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    cols = ["evaluation", "feature_set"] + [c for c in summary.columns if c.startswith(
        ("ensemble_r2_mean", "ensemble_mae_mean", "scenario_A_frac_mean", "mae_ratio_CD_over_AB_mean",
         "spearman")) and "_sd" not in c]
    print("\n" + summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
