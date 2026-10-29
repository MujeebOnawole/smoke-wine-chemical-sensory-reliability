"""
Add the source studies' binary sensory classification to the wine table.

`smoky_dunnett` = 1 when the wine's smoke flavour rating was significantly higher
than its cultivar control by Dunnett's test in the source study, else 0.

Study B (Parker et al. 2023): parsed from the asterisks in the Smoke Flavour
column of SI Table S6 (ajgw4964850-sup-0001-f1.docx). The extraction script
strips footnote marks, so the flag is recovered here from the docx directly.

Study A (Jiang et al. 2022): the SI Table S4 gives smoke aroma only. Smoke
flavour significance is stated in the main text (p. 12): Chardonnay vineyard J;
Pinot Noir vineyards C, E, G and L; Shiraz vineyards C, F, G and K. Hard-coded
below with that provenance.

Input : Taste_SmokeTaint/data/processed/awri_wine_smoke_wide_grouped.csv
Output: awri_wine_smoke_wide_labelled.csv (same rows, one new column) written to
        Taste_SmokeTaint/data/processed/ and bunya_smoke_odour/processed_data/.
"""

from pathlib import Path

import docx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # Smoke_project/
PARKER_SI = ROOT / "ajgw4964850-sup-0001-f1.docx"
SRC = ROOT / "Taste_SmokeTaint" / "data" / "processed" / "awri_wine_smoke_wide_grouped.csv"
OUTS = [
    ROOT / "Taste_SmokeTaint" / "data" / "processed" / "awri_wine_smoke_wide_labelled.csv",
    ROOT / "bunya_smoke_odour" / "processed_data" / "awri_wine_smoke_wide_labelled.csv",
]

# Jiang et al. 2022, p. 12, smoke flavour significantly higher than control (Dunnett, P = 0.05)
STUDY_A_SMOKY = {"CHA-J", "PIN-C", "PIN-E", "PIN-G", "PIN-L", "SHI-C", "SHI-F", "SHI-G", "SHI-K"}


def study_b_flags():
    d = docx.Document(PARKER_SI)
    table = next(t for t in d.tables
                 if [c.text.strip() for c in t.rows[0].cells][:4] == ["Cultivar", "Wine code", "Smoke Aroma", "Smoke Flavour"])
    flags = {}
    for r in table.rows[1:]:
        cells = [c.text.strip() for c in r.cells]
        code = cells[1].rstrip("‡†*∗ ")
        flags[code] = int("*" in cells[3] or "∗" in cells[3])
    return flags


def main():
    df = pd.read_csv(SRC)
    b = study_b_flags()
    assert len(b) == 40, f"expected 40 Study B wines in Table S6, got {len(b)}"
    flag = []
    for _, r in df.iterrows():
        if r.study == "B":
            flag.append(b[r.sample_id])
        else:
            flag.append(int(r.sample_id in STUDY_A_SMOKY))
    df["smoky_dunnett"] = flag
    assert set(STUDY_A_SMOKY) <= set(df.loc[df.study == "A", "sample_id"]), "Study A codes do not match"
    print(df.groupby(["study", "cultivar_code"]).smoky_dunnett.agg(["sum", "count"]))
    print("total smoky:", df.smoky_dunnett.sum(), "of", len(df))
    print("smoke flavour, smoky vs not:",
          df.groupby("smoky_dunnett").smoke_flavour.agg(["min", "median", "max"]).round(2).to_dict("index"))
    for out in OUTS:
        df.to_csv(out, index=False)
        print("wrote", out)


if __name__ == "__main__":
    main()
