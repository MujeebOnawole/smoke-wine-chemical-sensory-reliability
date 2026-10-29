"""Marker-versus-alternative gap as a flag for guaiacol discordance: control alternatives.

For every out-of-sample prediction, the gap is |alternative prediction - guaiacol OLS prediction|. A
prediction is guaiacol-discordant when |guaiacol OLS residual| >= 1.5 rating units. The table reports,
per evaluation setting and feature set, how well the gap built from each alternative flags discordant
predictions (AUROC, precision of the 30 percent largest gaps) and the MAE of that alternative. The
alternatives are the five-model ensemble, Lasso alone, the training-selected best single analyte and the
free volatile phenol index. If a simpler alternative flags discordance as well as the ensemble, the signal
is the disagreement itself, not the ensemble (Table S14 of the manuscript).

Reads per_wine_predictions.csv written by wine_assessment.py; no new model fits.
"""
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WA = os.path.join(ROOT, "results", "wine", "wine_assessment")
TOUT = os.path.join(ROOT, "results", "wine")
SETTINGS = [("cv", "Cultivar-stratified CV"), ("cultivar_holdout", "Cultivar hold-out"),
            ("grouped_site", "Sites held out"), ("grouped_region", "Regions held out"),
            ("external_trainB_testA", "Train B, test A"), ("external_trainA_testB", "Train A, test B")]
ALTERNATIVES = [("ensemble", "ensemble"), ("lasso", "Lasso"), ("b_best_single", "best single analyte"),
                ("b_phenol_index", "phenol index")]
LARGE, TOP = 1.5, 0.30


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")


def main():
    pred = pd.read_csv(os.path.join(WA, "per_wine_predictions.csv"))
    rows = []
    for setting, name in SETTINGS:
        for fs in ["both", "grape"]:
            d = pred[(pred.setting == setting) & (pred.feature_set == fs)].copy()
            disc = ((d.b_guaiacol - d.smoke_flavour).abs() >= LARGE).astype(int)
            row = {"Evaluation": name, "Features": {"both": "grape + wine", "grape": "grape"}[fs],
                   "n": len(d), "Discordant": int(disc.sum())}
            k = int(round(TOP * len(d)))
            for col, label in ALTERNATIVES:
                gap = (d[col] - d.b_guaiacol).abs()
                flagged = disc[gap.sort_values(ascending=False, kind="stable").index[:k]]  # ties keep row order
                row[f"AUROC, {label} gap"] = round(safe_auc(disc, gap), 2)
                row[f"Precision top-30%, {label} gap"] = round(flagged.mean(), 2)
                row[f"MAE of {label}"] = round((d[col] - d.smoke_flavour).abs().mean(), 2)
            rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(TOUT, exist_ok=True)
    out.to_csv(os.path.join(TOUT, "round6c_gap_controls.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
