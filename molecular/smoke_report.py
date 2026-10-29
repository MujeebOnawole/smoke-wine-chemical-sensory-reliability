"""
Summarise one smoke-odour run (one split, one label variant) after predict.py.

Reporting rule: reliability numbers (scenario coverage and accuracy) come from
the held-out TEST partition; the train/valid/test breakdown is written for the
SI; substructure design trends use the whole dataset and are labelled as such.

Two explanation-agreement definitions are reported side by side:
- dual   : the published taste criterion (best model; >= 70% of substructures
           positive for the predicted class, or mean attribution > 0)
- ensemble: mean pairwise Pearson r of the five members' attribution vectors
           (the measure used in the wine-level test). HIGH if >= the median over
           TRAINING molecules, so the threshold never sees test labels or test
           molecules. Molecules with < 2 substructures count as not agreed.

Phenol specificity: the ten held-out wine markers are all simple phenols, so a
model that learned "aromatic OH = smoky" would recall every one. The report
therefore gives the false-positive rate among test phenols labelled "other
odour" (e.g. vanillin, eugenol, thymol), next to the rate among non-phenols.

Reads (under --data_dir, TAG = smoke_odour_<split>[_<variant>]):
  <TAG>_classification_final_eval/{full_results.json, decision_threshold.json, individual_model_metrics.csv}
  <TAG>_xai_full/smoke_odour_{predictions,attributions}.csv
  <TAG>_xai_markers/smoke_odour_{predictions,attributions}.csv
  smoke_odour_markers_external.csv
Writes <TAG>_report/ with CSV tables and report.json, and prints them.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
PHENOL = Chem.MolFromSmarts("c[OX2H]")


def run_tag(split, variant):
    return f"smoke_odour_{split}" if variant == "combined" else f"smoke_odour_{split}_{variant}"


def add_ensemble_scenarios(pred, tau):
    agreed = pred["xai_ensemble_agreement"].fillna(-np.inf) >= tau
    high = pred["ensemble_consensus"] == "HIGH"
    pred["xai_ensemble_agreed"] = agreed
    pred["scenario_ensemble"] = np.where(high & agreed, "A", np.where(high, "B", np.where(agreed, "C", "D")))
    return pred


def scenario_table(pred, col):
    rows = []
    for grp, g in pred.groupby("group"):
        for sc in ["A", "B", "C", "D", "all"]:
            s = g if sc == "all" else g[g[col] == sc]
            pos = s["TARGET"] == 1
            rows.append({
                "agreement": col, "group": grp, "scenario": sc, "n": len(s),
                "coverage": len(s) / len(g) if len(g) else np.nan,
                "accuracy": s["correct"].mean() if len(s) else np.nan,
                "n_smoke_related": int(pos.sum()),
                "recall_smoke_related": (s.loc[pos, "predicted_class"] == 1).mean() if pos.any() else np.nan,
            })
    return pd.DataFrame(rows)


def ab_vs_cd(t, col):
    ab = t[col].isin(["A", "B"])
    return {"accuracy_AB": float(t.loc[ab, "correct"].mean()) if ab.any() else None,
            "accuracy_CD": float(t.loc[~ab, "correct"].mean()) if (~ab).any() else None,
            "coverage_A": float((t[col] == "A").mean())}


def has_phenol(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return bool(m is not None and m.HasSubstructMatch(PHENOL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", choices=["random", "scaffold"], required=True)
    ap.add_argument("--variant", choices=["combined", "smoky", "phenolic"], default="combined")
    ap.add_argument("--top_k", type=int, default=25)
    args = ap.parse_args()

    tag = run_tag(args.split, args.variant)
    eval_dir = os.path.join(args.data_dir, f"{tag}_classification_final_eval")
    out = os.path.join(args.data_dir, f"{tag}_report")
    os.makedirs(out, exist_ok=True)
    report = {"split": args.split, "label_variant": args.variant}

    # 1. Test-set performance (ensemble + all 25 CV models)
    with open(os.path.join(eval_dir, "full_results.json")) as f:
        ens = json.load(f)["ensemble_results"]
    with open(os.path.join(eval_dir, "decision_threshold.json")) as f:
        thr = json.load(f)
    at_f1 = ens.get("metrics_at_f1", {})
    report["test_ensemble"] = {
        "auc": ens.get("auc"), "average_precision": ens.get("average_precision"),
        "threshold_valid_tuned": thr["threshold"],
        **{k: at_f1.get(k) for k in ["matthews_corrcoef", "f1", "precision", "recall",
                                     "specificity", "balanced_accuracy"]},
    }
    indiv = pd.read_csv(os.path.join(eval_dir, "individual_model_metrics.csv"))
    report["test_individual_models"] = {
        c: {"mean": float(indiv[c].mean()), "sd": float(indiv[c].std()), "n": int(indiv[c].notna().sum())}
        for c in ["auc", "average_precision", "matthews_corrcoef_raw"] if c in indiv.columns}

    # 2. Scenarios by partition, both agreement definitions (test = reliability evidence)
    pred = pd.read_csv(os.path.join(args.data_dir, f"{tag}_xai_full", "smoke_odour_predictions.csv"))
    pred = pred[pred["error"].isna()].copy()
    tau = float(pred.loc[pred["group"] == "training", "xai_ensemble_agreement"].median())
    report["ensemble_agreement_threshold_from_training"] = tau
    pred = add_ensemble_scenarios(pred, tau)
    sc = pd.concat([scenario_table(pred, "scenario"), scenario_table(pred, "scenario_ensemble")])
    sc.to_csv(os.path.join(out, "scenarios_by_partition.csv"), index=False)
    t = pred[pred["group"] == "test"]
    report["test_dual_criterion"] = ab_vs_cd(t, "scenario")
    report["test_ensemble_agreement"] = ab_vs_cd(t, "scenario_ensemble")
    report["test_xai_coverage_dual"] = float(t["xai_is_agreed"].notna().mean())
    report["test_xai_coverage_ensemble"] = float(t["xai_ensemble_agreement"].notna().mean())

    # 3. Phenol specificity on the test partition
    t = t.copy()
    t["has_phenol"] = t["SMILES"].map(has_phenol)
    spec = {}
    for name, m in [("phenol", t["has_phenol"]), ("non_phenol", ~t["has_phenol"])]:
        neg, pos = t[m & (t["TARGET"] == 0)], t[m & (t["TARGET"] == 1)]
        spec[name] = {"n_negative": len(neg),
                      "false_positive_rate": float((neg["predicted_class"] == 1).mean()) if len(neg) else None,
                      "n_positive": len(pos),
                      "recall": float((pos["predicted_class"] == 1).mean()) if len(pos) else None}
    report["test_phenol_specificity"] = spec
    t.loc[t["has_phenol"], ["COMPOUND_ID", "SMILES", "TARGET", "predicted_class", "prob_Smoke_related",
                            "scenario", "scenario_ensemble"]] \
        .sort_values("prob_Smoke_related", ascending=False) \
        .to_csv(os.path.join(out, "test_phenols.csv"), index=False)

    # 4. Held-out wine smoke markers
    mk_pred = pd.read_csv(os.path.join(args.data_dir, f"{tag}_xai_markers", "smoke_odour_predictions.csv"))
    mk_pred = add_ensemble_scenarios(mk_pred, tau)
    mk = pd.read_csv(os.path.join(args.data_dir, "smoke_odour_markers_external.csv"))
    mk_pred = mk_pred.merge(mk[["COMPOUND_ID", "MARKER_NAME", "NN_TANIMOTO_TO_REMAINING"]], on="COMPOUND_ID")
    cols = ["MARKER_NAME", "prob_Smoke_related", "predicted_class_name", "ensemble_agreement_ratio",
            "xai_is_agreed", "scenario", "xai_ensemble_agreement", "scenario_ensemble", "NN_TANIMOTO_TO_REMAINING"]
    mk_pred[cols].to_csv(os.path.join(out, "markers_predictions.csv"), index=False)
    report["markers_recalled"] = int((mk_pred["predicted_class"] == 1).sum())
    report["markers_total"] = int(len(mk_pred))
    mk_attr = pd.read_csv(os.path.join(args.data_dir, f"{tag}_xai_markers", "smoke_odour_attributions.csv"))
    mk_attr = mk_attr.merge(mk[["COMPOUND_ID", "MARKER_NAME"]], on="COMPOUND_ID")
    mk_attr[["MARKER_NAME", "substructure_kind", "scaffold_smiles", "Smoke_related_attribution"]] \
        .sort_values(["MARKER_NAME", "Smoke_related_attribution"], ascending=[True, False]) \
        .to_csv(os.path.join(out, "markers_attributions.csv"), index=False)

    # 5. Design insight across the WHOLE dataset: substructures ranked by mean attribution
    attr = pd.read_csv(os.path.join(args.data_dir, f"{tag}_xai_full", "smoke_odour_attributions.csv"))
    agg = (attr.groupby(["substructure_kind", "scaffold_smiles"])["Smoke_related_attribution"]
           .agg(["mean", "count"]).reset_index())
    agg = agg[agg["count"] >= 5].sort_values("mean", ascending=False)
    agg.to_csv(os.path.join(out, "wholedataset_substructure_attribution.csv"), index=False)

    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    pd.set_option("display.width", 220)
    print(json.dumps({k: v for k, v in report.items() if k.startswith(("test_", "ensemble_", "label"))},
                     indent=2, default=float))
    print("\nTEST partition scenarios (reliability evidence):")
    print(sc[sc["group"] == "test"].round(3).to_string(index=False))
    print(f"\nHeld-out wine smoke markers: {report['markers_recalled']}/{report['markers_total']} "
          f"predicted smoke-related")
    print(mk_pred[cols].round(3).to_string(index=False))
    print(f"\nTop {args.top_k} smoke-promoting substructures (WHOLE dataset, count >= 5; design insight):")
    print(agg.head(args.top_k).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
