"""Odour-activity-value (OAV) check of wine attributions.

For each wine, compares the per-analyte attribution saved by wine_trust_test.py with the
OAV of the four phenols that have an odour detection threshold in Parker et al. (2023)
(wine matrix values in ug/L: guaiacol 23, o-cresol 62, m-cresol 20, p-cresol 64).
Per wine, the Spearman correlation between attribution and OAV is computed over these
four analytes (wine_ features only), then summarised by scenario. Four points per wine
make each correlation coarse, so the scenario summary (mean rho, share rho > 0) is the
reported quantity, not any single wine. Descriptive only; fixed before looking at results.

Usage: python wine_oav_check.py --data awri_wine_smoke_wide.csv --oof <csv> [<csv> ...] --out oav_check.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THRESHOLD_UG_L = {"guaiacol": 23.0, "o-cresol": 62.0, "m-cresol": 20.0, "p-cresol": 64.0}


def per_wine_rho(res, wide):
    wine = wide.set_index("wine_uid")
    cols = [a for a in THRESHOLD_UG_L if f"attr_wine_{a}" in res.columns]
    rows = []
    for _, r in res.iterrows():
        oav = np.array([wine.loc[r["wine_uid"], f"wine_{a}"] / THRESHOLD_UG_L[a] for a in cols], float)
        att = np.array([r[f"attr_wine_{a}"] for a in cols], float)
        if np.std(oav) < 1e-12 or np.std(att) < 1e-12:
            rho = np.nan
        else:
            rho = spearmanr(att, oav)[0]
        top_oav = cols[int(np.argmax(oav))]
        top_att = cols[int(np.argmax(att))]
        rows.append({"wine_uid": r["wine_uid"], "scenario": r["scenario"], "rho": rho,
                     "top_oav": top_oav, "top_attr": top_att, "top_match": top_oav == top_att,
                     "max_oav": float(oav.max())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--oof", nargs="+", required=True, help="CSVs from wine_trust_test.py or wine_trust_grouped.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wide = pd.read_csv(a.data)
    out = []
    for path in a.oof:
        res = pd.read_csv(path)
        if not any(c.startswith("attr_wine_") for c in res.columns):
            print(f"skip {path}: no per-analyte attribution columns (feature set without wine analytes)")
            continue
        d = per_wine_rho(res, wide)
        for scen, g in [("all", d)] + list(d.groupby("scenario")):
            out.append({"file": os.path.basename(path), "scenario": scen, "n": len(g),
                        "mean_rho": g["rho"].mean(), "frac_rho_pos": (g["rho"] > 0).mean(),
                        "frac_top_match": g["top_match"].mean(), "frac_wines_any_oav_ge_1": (g["max_oav"] >= 1).mean()})
    pd.DataFrame(out).to_csv(a.out, index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
