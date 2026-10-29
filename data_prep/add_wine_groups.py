"""
Add CV grouping columns to the AWRI wine table.

Study A (23 wines, Jiang et al. 2022): vineyard letter A-L is the suffix of sample_id
(e.g. PIN-G). Wines from one vineyard share one fire exposure across cultivars.
Study B (40 wines, Parker et al. 2023): the only site information is the Region column
of Table S1 (ajgw4964850-sup-0001-f1.docx), 12 regions, no vineyard ID.

Columns added
  group_site   : A_<vineyard letter> for Study A, B_<region> for Study B (primary grouping)
  group_region : region only; all Study A wines are Adelaide Hills (single Cudlee Creek
                 fire), so this pools Study A with Study B Adelaide Hills (strict sensitivity grouping)

Writes awri_wine_smoke_wide_grouped.csv next to the input and into bunya_smoke_odour/processed_data.
"""
import os
import sys

import docx
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
smoke = os.path.abspath(os.path.join(here, "..", ".."))
src = os.path.join(here, "..", "data", "processed", "awri_wine_smoke_wide.csv")
parker_docx = os.path.join(smoke, "ajgw4964850-sup-0001-f1.docx")
outs = [
    os.path.join(here, "..", "data", "processed", "awri_wine_smoke_wide_grouped.csv"),
    os.path.join(smoke, "bunya_smoke_odour", "processed_data", "awri_wine_smoke_wide_grouped.csv"),
]

df = pd.read_csv(src)
t = docx.Document(parker_docx).tables[0]
region = {r.cells[0].text.strip(): r.cells[1].text.strip() for r in t.rows[1:]}

def site(row):
    if row["study"] == "A":
        return "A_" + row["sample_id"].split("-")[-1]
    return "B_" + region[row["sample_id"]].replace(" ", "_")

def reg(row):
    if row["study"] == "A":
        return "Adelaide_Hills"
    return region[row["sample_id"]].replace(" ", "_")

missing = [s for s in df.loc[df.study == "B", "sample_id"] if s not in region]
if missing:
    sys.exit(f"Study B wines without a region: {missing}")
df["group_site"] = df.apply(site, axis=1)
df["group_region"] = df.apply(reg, axis=1)
for o in outs:
    df.to_csv(o, index=False)
    print("wrote", os.path.normpath(o))
print(df.groupby("study").group_site.nunique().to_dict(), "site groups per study")
print("region groups:", df.group_region.nunique())
print(df.group_site.value_counts().to_string())
