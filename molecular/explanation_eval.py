"""
Explanation evaluation for the smoke-odour RGCN (does a Scenario A explanation make
chemical sense, and does it do so more often than a Scenario B explanation?).

Why this script exists: A vs B differ in explanation agreement, not in accuracy, so
accuracy is the wrong instrument for that claim. The checks below follow the four-tier
logic of the XAI evaluation framework (scaffold recognition, context sensitivity,
model consistency, model independence), with ground truth taken from independent
chemistry rather than from the model.

CRITERIA ARE FIXED HERE, BEFORE ANY A-vs-B COMPARISON IS READ. Do not edit a definition
after seeing results; add a new, dated check instead.

Ground truth used (independent of the model)
  - Phenolic OH is the established smoke-odour motif (guaiacol, cresols, phenol, syringol:
    Parker 2023, Tan 2024, Lim 2025). Expected: its attribution promotes Smoke_related.
  - Within a matched pair (same ring system, Tanimoto >= 0.5, one smoke-related and one
    not), the difference in label must be explained by a substructure that is present
    only in the non-smoke member and has a negative Smoke_related attribution.

Checks
  E1 motif_sign       phenolic OH attribution > 0, among molecules predicted smoke-related
                      whose only OH groups are phenolic (so the 'O' substituent row is the phenol OH).
                      Also OH rank (1 = most smoke-promoting row of that molecule).
  E2 pair_specificity matched pairs; fraction where a substituent / functional-group label
                      present only in the non-smoke member has negative Smoke_related attribution
                      (substructure specificity). Fraction where the ring-system attribution is
                      higher in the smoke member (scaffold sensitivity). Top-3 most similar
                      non-smoke partners per smoke molecule, to stop benzene dominating.
  E3 consistency      mean attribution over all rows has the sign of the predicted class.
                      NOTE: the dual (best-model) Scenario A/B split is defined by this same sign
                      rule, so E3 under scenario="scenario" is circular (A = 100%, B = 0%).
                      Interpret E3 only under scenario_ensemble.
  E4 independence     ensemble attribution agreement (already per molecule), summarised per scenario.
  E1b oh_discrimination  among ALL phenol-only-OH molecules (not just predicted smoky), AUC of the OH
                      attribution for separating smoke-related from other-odour phenols. An explanation
                      that is positive for every phenol (AUC near 0.5) is a shortcut, not an explanation
                      of why a molecule is smoky. This is the smoke analogue of showing that the same
                      scaffold gets different contributions when the outcome differs.
  E6 descriptor_contrast  (needs --contrast_tags) same molecules scored by models trained for different
                      labels (e.g. smoky-only vs phenolic-only). Pre-specified expectation: for
                      phenol-only-OH molecules the OH attribution is higher in the model whose label the
                      molecule carries. Mean OH attribution per label group (smoky, phenolic) per model.
                      Mirrors the pathogen-selective fragment analysis: same molecule, different outcome
                      models, different fragment contributions.
  E5 blinded sheet    random sample of A and B explanations in shuffled order, scenario hidden,
                      with a separate key file, for a chemist to rate 'makes sense / does not'.

A vs B is compared on E1 and E3 with a Fisher exact test and a bootstrap interval on the
risk difference. Both scenario definitions are reported (dual best-model criterion and
cross-ensemble agreement).

Partition: XAI design-trend checks use the whole dataset by default and the output says
so; every table is also produced for the test partition alone. Reliability claims still
come only from the test partition (smoke_report.py).

Known property of the attributions (verify before interpreting): ring-system rows saturate
at about +-1.0 (masking the ring collapses the prediction), so E1 and E2 look at the
substituent and functional-group rows, where the chemistry is.

Usage (cheap, pandas + RDKit; run as a small batch job on Bunya or on pulled results):
  python explanation_eval.py --data_dir <dir containing <tag>_xai_full ...> --tag smoke_odour_random
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import fisher_exact

from smoke_report import add_ensemble_scenarios

RDLogger.DisableLog("rdApp.*")

PHENOL_OH = Chem.MolFromSmarts("c[OX2H]")
ANY_OH = Chem.MolFromSmarts("[OX2H]")
SMOKE = "Smoke_related_attribution"
SCEN_DEFS = ["scenario", "scenario_ensemble"]


# ---------------------------------------------------------------- helpers
def load(data_dir, tag, kind):
    d = os.path.join(data_dir, f"{tag}_xai_{kind}")
    pred = pd.read_csv(os.path.join(d, "smoke_odour_predictions.csv"))
    pred = pred[pred["error"].isna()].copy()
    attr = pd.read_csv(os.path.join(d, "smoke_odour_attributions.csv"))
    return pred, attr


def phenol_only_oh(smiles):
    """True if the molecule has a phenolic OH and every OH group is phenolic."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    n_phenol = len(m.GetSubstructMatches(PHENOL_OH))
    return n_phenol > 0 and n_phenol == len(m.GetSubstructMatches(ANY_OH))


def risk_diff_ci(a_hit, b_hit, n_boot=5000, seed=0):
    a_hit, b_hit = np.asarray(a_hit, float), np.asarray(b_hit, float)
    if len(a_hit) == 0 or len(b_hit) == 0:
        return None
    rng = np.random.default_rng(seed)
    d = [rng.choice(a_hit, len(a_hit)).mean() - rng.choice(b_hit, len(b_hit)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(a_hit.mean() - b_hit.mean()), float(lo), float(hi)


def ab_compare(df, hit_col, scen_col):
    """Compare hit rate of Scenario A vs Scenario B explanations."""
    a, b = df[df[scen_col] == "A"][hit_col], df[df[scen_col] == "B"][hit_col]
    out = {"scenario_def": scen_col, "metric": hit_col, "n_A": len(a), "n_B": len(b),
           "rate_A": float(a.mean()) if len(a) else None, "rate_B": float(b.mean()) if len(b) else None}
    if len(a) and len(b):
        table = [[int(a.sum()), int((~a.astype(bool)).sum())], [int(b.sum()), int((~b.astype(bool)).sum())]]
        out["fisher_p"] = float(fisher_exact(table)[1])
        ci = risk_diff_ci(a.to_numpy(), b.to_numpy())
        out["risk_diff_A_minus_B"], out["ci95_lo"], out["ci95_hi"] = ci
    return out


# ---------------------------------------------------------------- E1 motif sign
def e1_motif(pred, attr):
    sub = attr[(attr["substructure_kind"] == "ring_substituent") & (attr["scaffold_smiles"] == "O")]
    oh = sub.groupby("COMPOUND_ID")[SMOKE].max().rename("oh_attr")
    # rank of the OH row among all non-ring-system rows of the molecule (1 = most smoke-promoting)
    rest = attr[attr["substructure_kind"] != "ring_system"].copy()
    rest["rank"] = rest.groupby("COMPOUND_ID")[SMOKE].rank(ascending=False, method="min")
    ohr = rest[(rest["substructure_kind"] == "ring_substituent") & (rest["scaffold_smiles"] == "O")] \
        .groupby("COMPOUND_ID")["rank"].min().rename("oh_rank")
    n_rows = rest.groupby("COMPOUND_ID").size().rename("n_rows")
    d = pred[pred["predicted_class"] == 1].copy()
    d = d[d["SMILES"].map(phenol_only_oh)]
    d = d.join(oh, on="COMPOUND_ID").join(ohr, on="COMPOUND_ID").join(n_rows, on="COMPOUND_ID")
    d = d[d["oh_attr"].notna()].copy()
    d["motif_sign_ok"] = d["oh_attr"] > 0
    d["oh_rank_top1"] = d["oh_rank"] == 1
    return d


# ---------------------------------------------------------------- E2 matched pairs
def ring_key(g):
    return "|".join(sorted(g.loc[g["substructure_kind"] == "ring_system", "scaffold_smiles"]))


def e2_pairs(pred, attr, top_k=3, min_sim=0.5):
    keys = attr.groupby("COMPOUND_ID")[["substructure_kind", "scaffold_smiles"]].apply(ring_key)
    p = pred.set_index("COMPOUND_ID")
    ids = [i for i in p.index if i in keys.index and keys[i] != ""]
    fps = {}
    for i in ids:
        m = Chem.MolFromSmiles(p.loc[i, "SMILES"])
        if m is not None:
            fps[i] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    by_attr = {cid: g for cid, g in attr.groupby("COMPOUND_ID")}
    rows = []
    df = pd.DataFrame({"id": list(fps), "key": [keys[i] for i in fps], "y": [p.loc[i, "TARGET"] for i in fps]})
    for key, g in df.groupby("key"):
        act, ina = g[g.y == 1].id.tolist(), g[g.y == 0].id.tolist()
        if not act or not ina:
            continue
        ina_fps = [fps[i] for i in ina]
        for a in act:
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[a], ina_fps))
            for j in np.argsort(-sims)[:top_k]:
                if sims[j] < min_sim:
                    break
                b = ina[j]
                rows.append(_pair_row(a, b, float(sims[j]), by_attr, p))
    return pd.DataFrame(rows)


def _labels(g):
    return g[g["substructure_kind"] != "ring_system"].groupby("scaffold_smiles")[SMOKE].mean()


def _pair_row(a, b, sim, by_attr, p):
    ga, gb = by_attr[a], by_attr[b]
    la, lb = _labels(ga), _labels(gb)
    only_b = lb[~lb.index.isin(la.index)]
    ra = ga.loc[ga["substructure_kind"] == "ring_system", SMOKE].mean()
    rb = gb.loc[gb["substructure_kind"] == "ring_system", SMOKE].mean()
    return {"smoke_id": a, "other_id": b, "tanimoto": sim,
            "n_only_in_nonsmoke": len(only_b),
            "substructure_specific": bool(len(only_b) and (only_b < 0).any()),
            "scaffold_sensitive": bool(ra > rb),
            "both_correct": bool(p.loc[a, "predicted_class"] == 1 and p.loc[b, "predicted_class"] == 0),
            "both_scenarioA_ens": bool(p.loc[a, "scenario_ensemble"] == "A" and p.loc[b, "scenario_ensemble"] == "A")}


# ---------------------------------------------------------------- E3 consistency
def e3_consistency(pred, attr):
    mean_attr = attr.groupby("COMPOUND_ID")[SMOKE].mean().rename("mean_attr")
    d = pred.join(mean_attr, on="COMPOUND_ID")
    d = d[d["mean_attr"].notna()].copy()
    d["consistent"] = np.where(d["predicted_class"] == 1, d["mean_attr"] > 0, d["mean_attr"] < 0)
    return d


# ---------------------------------------------------------------- E5 blinded sheet
def blinded_sheet(d, attr, out_dir, n_per=30, seed=7):
    rng = np.random.default_rng(seed)
    picks = []
    for s in ["A", "B"]:
        g = d[d["scenario_ensemble"] == s]
        picks.append(g.iloc[rng.permutation(len(g))[:n_per]].assign(true_scenario=s))
    s = pd.concat(picks)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s["blind_id"] = [f"M{i + 1:03d}" for i in range(len(s))]
    by = {cid: g for cid, g in attr.groupby("COMPOUND_ID")}

    def fmt(cid, top):
        g = by.get(cid)
        if g is None:
            return ""
        g = g.sort_values(SMOKE, ascending=not top).head(3)
        return "; ".join(f"{r.scaffold_smiles} ({r[SMOKE]:+.2f})" for _, r in g.iterrows())

    sheet = pd.DataFrame({
        "blind_id": s["blind_id"], "SMILES": s["SMILES"], "prob_smoke": s["prob_Smoke_related"].round(3),
        "top_promoting": [fmt(c, True) for c in s["COMPOUND_ID"]],
        "top_opposing": [fmt(c, False) for c in s["COMPOUND_ID"]],
        "rating_makes_sense_1_to_5": "", "comment": ""})
    sheet.to_csv(os.path.join(out_dir, "blinded_sheet_FOR_RATER.csv"), index=False)
    s[["blind_id", "COMPOUND_ID", "true_scenario", "scenario", "xai_ensemble_agreement"]] \
        .to_csv(os.path.join(out_dir, "blinded_key_DO_NOT_OPEN_BEFORE_RATING.csv"), index=False)
    # rdkit.Chem.Draw needs libXrender, which the Bunya container lacks. Draw only if available;
    # otherwise render locally with render_blinded_sheet.py from blinded_sheet_FOR_RATER.csv.
    try:
        from rdkit.Chem import Draw
        mols = [Chem.MolFromSmiles(x) for x in s["SMILES"]]
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 240), legends=s["blind_id"].tolist())
        img.save(os.path.join(out_dir, "blinded_molecules.png"))
    except ImportError as e:
        print(f"NOTE: molecule grid not drawn here ({e}); run render_blinded_sheet.py locally on the sheet")



# ---------------------------------------------------------------- E1b OH discrimination
def e1b_discrimination(pred, attr):
    from sklearn.metrics import roc_auc_score
    sub = attr[(attr["substructure_kind"] == "ring_substituent") & (attr["scaffold_smiles"] == "O")]
    oh = sub.groupby("COMPOUND_ID")[SMOKE].max().rename("oh_attr")
    d = pred.join(oh, on="COMPOUND_ID")
    d = d[d["oh_attr"].notna() & d["SMILES"].map(phenol_only_oh)]
    rows = []
    for part, g in [("all", d), ("test", d[d["group"] == "test"])]:
        ok = g["TARGET"].nunique() == 2
        rows.append({"partition": part, "n_phenols": len(g), "n_smoke_related": int((g["TARGET"] == 1).sum()),
                     "mean_oh_attr_smoke": float(g.loc[g.TARGET == 1, "oh_attr"].mean()) if (g.TARGET == 1).any() else None,
                     "mean_oh_attr_other": float(g.loc[g.TARGET == 0, "oh_attr"].mean()) if (g.TARGET == 0).any() else None,
                     "auc_oh_attr_vs_label": float(roc_auc_score(g["TARGET"], g["oh_attr"])) if ok else None,
                     "auc_prob_vs_label": float(roc_auc_score(g["TARGET"], g["prob_Smoke_related"])) if ok else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E6 descriptor contrast
def e6_contrast(data_dir, tags):
    """OH attribution per label group (smoky, phenolic) in each model, for the same phenol-only-OH molecules."""
    frames = {}
    for t in tags:
        pred, attr = load(data_dir, t, "full")
        sub = attr[(attr["substructure_kind"] == "ring_substituent") & (attr["scaffold_smiles"] == "O")]
        frames[t] = sub.groupby("COMPOUND_ID")[SMOKE].max().rename(t)
    base_split = pd.read_csv(os.path.join(data_dir, f"{tags[0]}_split.csv"))
    smi_col = "PROCESSED_SMILES" if "PROCESSED_SMILES" in base_split else "SMILES"
    d = base_split.set_index("COMPOUND_ID")[[smi_col, "label_smoky", "label_phenolic"]]
    d = d[d[smi_col].map(phenol_only_oh)]
    for t, s in frames.items():
        d = d.join(s, how="left")
    d = d.dropna(subset=list(frames))
    rows = []
    for (sm, ph), g in d.groupby(["label_smoky", "label_phenolic"]):
        row = {"label_smoky": int(sm), "label_phenolic": int(ph), "n": len(g)}
        for t in frames:
            row[f"mean_OH_attr[{t}]"] = float(g[t].mean())
        rows.append(row)
    return pd.DataFrame(rows), d


# ---------------------------------------------------------------- markers
def markers_table(data_dir, tag):
    pred, attr = load(data_dir, tag, "markers")
    marker_csv = os.path.join(data_dir, "smoke_odour_markers_external.csv")
    names = pd.read_csv(marker_csv).set_index("COMPOUND_ID")["MARKER_NAME"] if os.path.exists(marker_csv) else None
    e1 = e1_motif(pred, attr)
    if names is not None:
        e1["marker"] = e1["COMPOUND_ID"].map(names)
    keep = [c for c in ["marker", "COMPOUND_ID", "SMILES", "prob_Smoke_related", "scenario", "scenario_ensemble",
                        "xai_ensemble_agreement", "oh_attr", "oh_rank", "n_rows", "motif_sign_ok"] if c in e1.columns]
    return e1[keep]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", required=True, help="e.g. smoke_odour_random")
    ap.add_argument("--contrast_tags", nargs="*", default=[],
                    help="tags of models trained for other labels (E6), e.g. smoke_odour_random_smoky smoke_odour_random_phenolic")
    ap.add_argument("--n_blind", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = os.path.join(args.data_dir, f"{args.tag}_explanation_eval")
    os.makedirs(out, exist_ok=True)

    pred, attr = load(args.data_dir, args.tag, "full")
    tau = float(pred.loc[pred["group"] == "training", "xai_ensemble_agreement"].median())
    pred = add_ensemble_scenarios(pred, tau)
    report = {"tag": args.tag, "ensemble_agreement_threshold_from_training": tau,
              "partition_note": "design-trend checks on all partitions; every table repeated for test only"}

    # E1
    e1 = e1_motif(pred, attr)
    e1.to_csv(os.path.join(out, "e1_phenol_oh_per_molecule.csv"), index=False)
    rows = []
    for part, d in [("all", e1), ("test", e1[e1["group"] == "test"])]:
        for sd in SCEN_DEFS:
            for s in ["A", "B", "C", "D"]:
                g = d[d[sd] == s]
                rows.append({"partition": part, "scenario_def": sd, "scenario": s, "n": len(g),
                             "motif_sign_ok": float(g["motif_sign_ok"].mean()) if len(g) else None,
                             "oh_rank_top1": float(g["oh_rank_top1"].mean()) if len(g) else None,
                             "median_oh_attr": float(g["oh_attr"].median()) if len(g) else None})
    pd.DataFrame(rows).to_csv(os.path.join(out, "e1_by_scenario.csv"), index=False)

    # E3
    e3 = e3_consistency(pred, attr)
    e3.to_csv(os.path.join(out, "e3_consistency_per_molecule.csv"), index=False)
    rows = []
    for part, d in [("all", e3), ("test", e3[e3["group"] == "test"])]:
        for sd in SCEN_DEFS:
            for s in ["A", "B", "C", "D"]:
                g = d[d[sd] == s]
                rows.append({"partition": part, "scenario_def": sd, "scenario": s, "n": len(g),
                             "consistent": float(g["consistent"].mean()) if len(g) else None,
                             "mean_ensemble_agreement": float(g["xai_ensemble_agreement"].mean()) if len(g) else None})
    pd.DataFrame(rows).to_csv(os.path.join(out, "e3_e4_by_scenario.csv"), index=False)

    # A vs B
    comps = []
    for part, d1, d3 in [("all", e1, e3), ("test", e1[e1["group"] == "test"], e3[e3["group"] == "test"])]:
        for sd in SCEN_DEFS:
            comps.append({"partition": part, **ab_compare(d1, "motif_sign_ok", sd)})
            comps.append({"partition": part, **ab_compare(d1, "oh_rank_top1", sd)})
            comps.append({"partition": part, **ab_compare(d3, "consistent", sd)})
    pd.DataFrame(comps).to_csv(os.path.join(out, "ab_comparison.csv"), index=False)

    # E2
    pairs = e2_pairs(pred, attr)
    pairs.to_csv(os.path.join(out, "e2_matched_pairs.csv"), index=False)
    if len(pairs):
        report["e2"] = {
            "n_pairs": int(len(pairs)),
            "n_unique_smoke_molecules": int(pairs["smoke_id"].nunique()),
            "substructure_specificity": float(pairs["substructure_specific"].mean()),
            "scaffold_sensitivity": float(pairs["scaffold_sensitive"].mean()),
            "substructure_specificity_both_correct": float(pairs.loc[pairs.both_correct, "substructure_specific"].mean())
            if pairs.both_correct.any() else None,
            "n_both_correct": int(pairs.both_correct.sum()),
            "substructure_specificity_both_scenarioA_ens": float(pairs.loc[pairs.both_scenarioA_ens, "substructure_specific"].mean())
            if pairs.both_scenarioA_ens.any() else None,
            "n_both_scenarioA_ens": int(pairs.both_scenarioA_ens.sum()),
            "note": "pairs from the whole dataset; ring-system attribution saturates, treat scaffold_sensitivity with caution"}

    # E1b
    e1b = e1b_discrimination(pred, attr)
    e1b.to_csv(os.path.join(out, "e1b_oh_discrimination.csv"), index=False)
    report["e1b"] = e1b.to_dict(orient="records")

    # E6
    if args.contrast_tags:
        c_tab, c_det = e6_contrast(args.data_dir, args.contrast_tags)
        c_tab.to_csv(os.path.join(out, "e6_descriptor_contrast.csv"), index=False)
        c_det.to_csv(os.path.join(out, "e6_descriptor_contrast_per_molecule.csv"))
        report["e6"] = c_tab.to_dict(orient="records")

    # E5
    cand = pred[(pred["predicted_class"] == 1) & pred["scenario_ensemble"].isin(["A", "B"])]
    blinded_sheet(cand, attr, out, n_per=args.n_blind, seed=args.seed)

    # Markers (external, held out)
    try:
        markers_table(args.data_dir, args.tag).to_csv(os.path.join(out, "markers_motif.csv"), index=False)
    except FileNotFoundError:
        report["markers"] = "xai_markers outputs not found"

    # Saturation diagnostic
    ring = attr[attr["substructure_kind"] == "ring_system"][SMOKE].abs()
    report["ring_attribution_saturated_fraction"] = float((ring >= 0.99).mean())

    with open(os.path.join(out, "explanation_eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nA vs B:\n", pd.DataFrame(comps).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
