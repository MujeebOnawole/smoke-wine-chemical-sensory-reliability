"""
Convert the 19-label flavour splits into the taste-pipeline input format for
the binary smoke-odour task.

TARGET = 1 if a compound carries the smoky or the phenolic odour label
(descriptors: smoky, smoke, burnt, charred, tarry, ashy, tobacco, phenolic,
medicinal, creosote, guaiacol, iodine), else 0. The ten wine smoke markers were
already removed from the splits by prepare_smoke_taint_data.py and are written
here, in the same format, as an external test file.

Inputs  (data/processed/): smoke_taint_{random,scaffold}_split.csv, smoke_markers_external.csv
Outputs (../bunya_smoke_odour/processed_data/):
    smoke_odour_random_split.csv, smoke_odour_scaffold_split.csv, smoke_odour_markers_external.csv,
    smoke_odour_random_{smoky,phenolic}_split.csv (label-variant ablations, TARGET = that label only)
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data" / "processed"
OUT = BASE.parent / "bunya_smoke_odour" / "processed_data"
KEEP = ["PROCESSED_SMILES", "COMPOUND_ID", "TARGET", "group", "label_smoky", "label_phenolic"]


def to_binary(df):
    df = df.copy()
    df["TARGET"] = ((df["label_smoky"] == 1) | (df["label_phenolic"] == 1)).astype(int)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["random", "scaffold"]:
        df = to_binary(pd.read_csv(SRC / f"smoke_taint_{split}_split.csv"))
        df[KEEP].to_csv(OUT / f"smoke_odour_{split}_split.csv", index=False)
        print(f"{split}: {len(df)} compounds")
        print(pd.crosstab(df["group"], df["TARGET"], margins=True).to_string(), "\n")

    # Label-variant ablations (random split only; same group assignment as the main file)
    base = pd.read_csv(SRC / "smoke_taint_random_split.csv")
    for variant, col in [("smoky", "label_smoky"), ("phenolic", "label_phenolic")]:
        v = base.copy()
        v["TARGET"] = v[col].astype(int)
        v[KEEP].to_csv(OUT / f"smoke_odour_random_{variant}_split.csv", index=False)
        print(f"random/{variant}:")
        print(pd.crosstab(v["group"], v["TARGET"], margins=True).to_string(), "\n")

    markers = to_binary(pd.read_csv(SRC / "smoke_markers_external.csv"))
    markers["group"] = "external_marker"
    markers[KEEP + ["MARKER_NAME", "NN_TANIMOTO_TO_REMAINING", "NN_SMILES"]].to_csv(
        OUT / "smoke_odour_markers_external.csv", index=False)
    print(f"markers: {len(markers)} (all TARGET=1: {bool((markers['TARGET'] == 1).all())})")


if __name__ == "__main__":
    main()
