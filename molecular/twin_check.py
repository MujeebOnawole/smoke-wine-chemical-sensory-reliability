"""Near-duplicate check for the random and scaffold splits (local CPU, about 1 minute).

For each test molecule, the maximum Morgan Tanimoto (radius 2, 2048 bits) to any training molecule.
Molecules at 0.99 or above are near-identical in 2D fingerprint terms (typically stereoisomers).
Reports discrimination of the 5-model ensemble probabilities on test subsets with and without such a
twin. Writes results/molecular/twin_check.csv (Table S4).
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "molecular")
DATA = os.path.join(HERE, "..", "data")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
rows = []
for tag, split in [("random", "smoke_odour_random_split.csv"), ("scaffold", "smoke_odour_scaffold_split.csv")]:
    sp = pd.read_csv(os.path.join(DATA, split))
    p = pd.read_csv(os.path.join(RES, f"smoke_odour_{tag}_xai_full", "smoke_odour_predictions.csv"))[["COMPOUND_ID", "prob_Smoke_related"]]
    d = sp.merge(p, on="COMPOUND_ID")
    fps = {c: gen.GetFingerprint(Chem.MolFromSmiles(s)) for c, s in zip(d.COMPOUND_ID, d.PROCESSED_SMILES)}
    trf = [fps[c] for c in d[d.group == "training"].COMPOUND_ID]
    te = d[d.group == "test"].copy()
    te["nn"] = [max(DataStructs.BulkTanimotoSimilarity(fps[c], trf)) for c in te.COMPOUND_ID]
    for lab, m in [("All test molecules", te.nn >= 0), ("With training twin (>= 0.99)", te.nn >= 0.99),
                   ("Without twin (< 0.99)", te.nn < 0.99), ("Nearest training neighbour < 0.80", te.nn < 0.80)]:
        g = te[m]
        ok = g.TARGET.nunique() == 2
        rows.append({"split": tag, "subset": lab, "n": len(g), "positives": int(g.TARGET.sum()),
                     "auroc": roc_auc_score(g.TARGET, g.prob_Smoke_related) if ok else np.nan,
                     "average_precision": average_precision_score(g.TARGET, g.prob_Smoke_related) if ok else np.nan})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "twin_check.csv"), index=False)
print(out.round(3).to_string())
