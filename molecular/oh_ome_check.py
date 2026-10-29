"""Known-SAR matched-pair test: does the attribution recover the loss of smoke character when a phenolic OH is capped as OCH3?

Local CPU only (reads pulled result csv files, a few seconds). Run, for example
    python oh_ome_check.py --data_dir ../Results/results_staging

Framing in the paper (revised): the test asks whether the attribution tracked the label change, without assuming that
methoxy capping must reduce smoke character. The rationale below is the original one, kept as written when the criteria were fixed.

Criteria fixed here BEFORE any Scenario A versus B reading (2026-10-04)

Ground truth. Capping an aromatic OH as an aromatic OCH3 removes the hydrogen-bond donor that phenolic and
smoky odour is associated with, so the OCH3 member of a pair is expected to be less smoke-related than the OH
member. The data do not always agree (partners often keep the label), so only pairs whose labels agree with
the chemistry are scored as cliffs. The analogue in the activity-cliff framework (Onawole et al. 2026,
J Cheminformatics 18:81) is scaffold sensitivity and substructure specificity.

Pairs. Molecule M has exactly one phenolic OH (c[OX2H]) and no aromatic OCH3 (c[OX2][CH3]). Partner N is M with
that OH replaced by OCH3 and must be a modelled molecule with exactly one aromatic OCH3. Both members need
attribution rows, and the OH row of M and the OCH3 row of N must exist. Labels are those of the model being
evaluated (combined, smoky-only or phenolic-only).
  cliff pair    : y(M) = 1 and y(N) = 0
  control pairs : y(M) = y(N) (both 0 or both 1), used to check whether any effect is label-blind

Metrics, each a yes/no per pair, reported as share with an exact (Clopper-Pearson) 95 percent interval
  H1 substituent contrast : attribution of the OH in M is higher than the attribution of the OCH3 in N
  H2 substructure specificity : attribution of the OCH3 in N is negative (group present only in the non-smoke member)
  H3 scaffold sensitivity : mean ring-system attribution of M is higher than that of N
The attribution is Smoke_related_attribution of the ring_substituent row with fragment O (OH) or CO (OCH3).

Primary model: random split, combined label. The other three models are secondary.
Primary readout: H1 to H3 on cliff pairs, with the control-pair rate shown beside each so a label-blind effect is visible.
A versus B: the OH member (M) is assigned a Scenario A or B by the ensemble criterion (same threshold as
explanation_eval.py, median over training molecules), and cliff pairs are compared with Fisher's exact test and a
bootstrap interval on the risk difference. Expected to be underpowered, so it is descriptive.
Partitions: all pairs (design analysis, stated as whole-dataset) and test-partition pairs, where n permits.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explanation_eval import ab_compare, load  # noqa: E402
from smoke_report import add_ensemble_scenarios  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SMOKE = "Smoke_related_attribution"
PHENOL_OH = Chem.MolFromSmarts("c[OX2H1]")
AR_OME = Chem.MolFromSmarts("c[OX2][CH3]")
CAP = AllChem.ReactionFromSmarts("[c:1][OX2H1]>>[c:1]O[CH3]")
TAGS = [("smoke_odour_random", "label_combined"), ("smoke_odour_scaffold", "label_combined"),
        ("smoke_odour_random_smoky", "label_smoky"), ("smoke_odour_random_phenolic", "label_phenolic")]


def exact_ci(k, n):
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def find_pairs(pred):
    mols = {r.COMPOUND_ID: Chem.MolFromSmiles(r.SMILES) for r in pred.itertuples()}
    can = {Chem.MolToSmiles(m): cid for cid, m in mols.items() if m is not None}
    pairs = []
    for cid, m in mols.items():
        if m is None or len(m.GetSubstructMatches(PHENOL_OH)) != 1 or m.HasSubstructMatch(AR_OME):
            continue
        for prod in CAP.RunReactants((m,)):
            try:
                Chem.SanitizeMol(prod[0])
            except Exception:
                continue
            s = Chem.MolToSmiles(prod[0])
            n = can.get(s)
            if n is not None and len(mols[n].GetSubstructMatches(AR_OME)) == 1:
                pairs.append((cid, n))
                break
    return sorted(set(pairs))


def frag_attr(g, frag):
    x = g[(g["substructure_kind"] == "ring_substituent") & (g["scaffold_smiles"] == frag)][SMOKE]
    return float(x.max()) if len(x) else np.nan


def ring_attr(g):
    x = g.loc[g["substructure_kind"] == "ring_system", SMOKE]
    return float(x.mean()) if len(x) else np.nan


def run_tag(data_dir, tag, label_col, labels):
    pred, attr = load(data_dir, tag, "full")
    tau = float(pred.loc[pred["group"] == "training", "xai_ensemble_agreement"].median())
    pred = add_ensemble_scenarios(pred, tau).set_index("COMPOUND_ID")
    by = {c: g for c, g in attr.groupby("COMPOUND_ID")}
    y = labels.set_index("COMPOUND_ID")[label_col] if label_col != "label_combined" else pred["TARGET"]
    rows = []
    for m_id, n_id in find_pairs(pred.reset_index()):
        if m_id not in by or n_id not in by or m_id not in y.index or n_id not in y.index:
            continue
        oh, ome = frag_attr(by[m_id], "O"), frag_attr(by[n_id], "CO")
        if np.isnan(oh) or np.isnan(ome):
            continue
        ym, yn = int(y[m_id]), int(y[n_id])
        rows.append({"oh_id": m_id, "ome_id": n_id, "y_oh": ym, "y_ome": yn,
                     "kind": "cliff" if (ym, yn) == (1, 0) else ("control" if ym == yn else "reverse"),
                     "partition_oh": pred.loc[m_id, "group"], "partition_ome": pred.loc[n_id, "group"],
                     "oh_attr": oh, "ome_attr": ome, "ring_oh": ring_attr(by[m_id]), "ring_ome": ring_attr(by[n_id]),
                     "H1_oh_gt_ome": oh > ome, "H2_ome_negative": ome < 0,
                     "H3_ring_higher_in_oh": ring_attr(by[m_id]) > ring_attr(by[n_id]),
                     "scenario_ens_oh": pred.loc[m_id, "scenario_ensemble"],
                     "pred_oh": int(pred.loc[m_id, "predicted_class"]), "pred_ome": int(pred.loc[n_id, "predicted_class"])})
    return pd.DataFrame(rows), tau


def summarise(df, tag):
    out = []
    for part, d in [("all", df), ("test", df[(df.partition_oh == "test") | (df.partition_ome == "test")])]:
        for kind in ["cliff", "control", "reverse"]:
            g = d[d.kind == kind]
            for both_ok in [False, True]:
                gg = g[(g.pred_oh == g.y_oh) & (g.pred_ome == g.y_ome)] if both_ok else g
                row = {"model": tag, "partition": part, "kind": kind, "both_predicted_correctly": both_ok, "n": len(gg)}
                for h in ["H1_oh_gt_ome", "H2_ome_negative", "H3_ring_higher_in_oh"]:
                    k = int(gg[h].sum()) if len(gg) else 0
                    lo, hi = exact_ci(k, len(gg))
                    row[h] = k / len(gg) if len(gg) else np.nan
                    row[h + "_lo"], row[h + "_hi"] = lo, hi
                out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()
    out_dir = os.path.join(args.data_dir, "oh_ome")
    os.makedirs(out_dir, exist_ok=True)
    labels = pd.read_csv(os.path.join(args.data_dir, "smoke_odour_random_smoky_split.csv"))[
        ["COMPOUND_ID", "label_smoky", "label_phenolic"]]
    summ, ab = [], []
    for tag, lab in TAGS:
        df, tau = run_tag(args.data_dir, tag, lab, labels)
        df.to_csv(os.path.join(out_dir, f"{tag}_pairs.csv"), index=False)
        summ.append(summarise(df, tag))
        cl = df[df.kind == "cliff"]
        for h in ["H1_oh_gt_ome", "H2_ome_negative", "H3_ring_higher_in_oh"]:
            a = ab_compare(cl.assign(hit=cl[h]), "hit", "scenario_ens_oh")
            a.update({"model": tag, "hypothesis": h})
            ab.append(a)
        print(tag, "pairs", len(df), df.kind.value_counts().to_dict(), "tau", round(tau, 3))
    s = pd.concat(summ)
    s.to_csv(os.path.join(out_dir, "oh_ome_summary.csv"), index=False)
    pd.DataFrame(ab).to_csv(os.path.join(out_dir, "oh_ome_A_vs_B.csv"), index=False)
    with open(os.path.join(out_dir, "run_params.json"), "w") as f:
        json.dump({"script": "oh_ome_check.py", "primary_model": "smoke_odour_random", "cap_smarts": "[c:1][OX2H1]>>[c:1]O[CH3]",
                   "attribution_fragments": {"OH": "O", "OCH3": "CO"}}, f, indent=2)
    pd.set_option("display.width", 220)
    print(s[(s.partition == "all") & (~s.both_predicted_correctly)].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
