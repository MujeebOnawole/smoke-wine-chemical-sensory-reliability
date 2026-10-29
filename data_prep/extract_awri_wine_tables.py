"""
Extract per-wine smoke-marker chemistry and sensory smoke ratings from the two
AWRI studies into one tidy dataset.

Sources (files in Smoke_project/):
- Study A: Jiang et al. 2022, AJGWR 9820204 (Adelaide Hills, pre-veraison smoke).
    Main PDF Table 3 (harvest grapes, ug/kg, mean of n=3) and Table 4 (wines,
    ug/L, smoke flavour); SI docx Table S4 (smoke aroma).
- Study B: Parker et al. 2023, AJGWR 4964850 (NSW/SA/VIC bushfire smoke).
    SI docx Table S3 (grapes, ug/kg), Table S4 (wines, ug/L), Table S6 (smoke
    aroma and smoke flavour).

Both studies used the same AWRI laboratory methods and the same trained-panel
0-10 smoke rating protocol, so they can be pooled; `study` is kept as a column.

Values below the limit of quantitation are set to half the LoQ, the convention
Parker et al. used for their PLS models; `below_loq` flags them in the long file.
LoQ: 1 ug/L or ug/kg, except syringol and 4-methylsyringol at 2.

Outputs (data/processed/):
- awri_wine_smoke_long.csv : one row per sample x matrix x analyte, with raw text
- awri_wine_smoke_wide.csv : one row per wine, grape_* and wine_* columns + ratings
"""

import re
from pathlib import Path

import pandas as pd  # import before fitz: PyMuPDF otherwise loads an older libstdc++ first
import docx
import fitz

BASE = Path(__file__).resolve().parent.parent
SRC = BASE.parent
OUT = BASE / "data" / "processed"

JIANG_PDF = next(SRC.glob("Australian Journal of Grape and Wine Research - 2022 - Jiang*.pdf"))
JIANG_SI = SRC / "ajgw9820204-sup-0001-f1.docx"
PARKER_SI = SRC / "ajgw4964850-sup-0001-f1.docx"

CULTIVAR_CODE = {"Chardonnay": "CHA", "Pinot noir": "PIN", "Pinot Noir": "PIN", "Shiraz": "SHI"}
LOQ = {"syringol": 2.0, "4-methylsyringol": 2.0}
FOOTNOTE_MARKS = "‡†∗*§"

# Header text -> canonical analyte name
ANALYTE_ALIASES = {
    "SyGG": "SyGG", "MSyGG": "MSyGG", "PhRG": "PhRG", "GuRG": "GuRG", "CrRG": "CrRG", "MGuRG": "MGuRG",
    "MGu": "4-methylguaiacol", "4-Methyl-guaiacol": "4-methylguaiacol", "4 – Methylguaiacol": "4-methylguaiacol",
    "Gu": "guaiacol", "Guaiacol": "guaiacol",
    "o-cresol": "o-cresol", "m-cresol": "m-cresol", "p-cresol": "p-cresol",
    "Sy": "syringol", "Syringol": "syringol",
    "MSy": "4-methylsyringol", "4-Methyl-syringol": "4-methylsyringol",
}


def clean_label(text):
    return text.strip().strip(FOOTNOTE_MARKS).strip()


def parse_value(raw, analyte):
    """Return (numeric, below_loq). Takes the mean from 'mean ± sd' strings."""
    raw = raw.strip()
    if raw.startswith("<LoQ") or raw.startswith("<LOQ"):
        return LOQ.get(analyte, 1.0) / 2, True
    num = re.match(r"[-+]?\d*\.?\d+", raw)
    if num is None:
        raise ValueError(f"Unparseable value {raw!r} for {analyte}")
    return float(num.group()), False


def canonical_analyte(header):
    name = re.sub(r"\s*\((µg|μg)/(kg|L)\)", "", header).strip()
    if name not in ANALYTE_ALIASES:
        raise KeyError(f"Unknown analyte header {header!r}")
    return ANALYTE_ALIASES[name]


# --------------------------------------------------------------------------
# Study B: Parker et al. 2023 supplementary docx
# --------------------------------------------------------------------------
def parker_concentrations(table, matrix):
    headers = [c.text.strip() for c in table.rows[0].cells]
    analytes = [canonical_analyte(h) for h in headers[1:]]
    rows = []
    for r in table.rows[1:]:
        cells = [c.text.strip() for c in r.cells]
        sample = clean_label(cells[0])
        for analyte, raw in zip(analytes, cells[1:]):
            value, below = parse_value(raw, analyte)
            rows.append(dict(study="B", source="Parker2023_SI", sample_id=sample,
                             cultivar_code=sample[:3], matrix=matrix, analyte=analyte,
                             raw_value=raw, value=value, below_loq=below))
    return rows


def parker_sensory(table):
    rows = []
    for r in table.rows[1:]:
        cells = [c.text.strip() for c in r.cells]
        sample = clean_label(cells[1])
        rows.append(dict(sample_id=sample, smoke_aroma=float(clean_label(cells[2])),
                         smoke_flavour=float(clean_label(cells[3]))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Study A: Jiang et al. 2022 main-text tables (PDF) and SI Table S4
# --------------------------------------------------------------------------
def jiang_tokens(start_marker, end_marker):
    text = "\n".join(page.get_text() for page in fitz.open(JIANG_PDF))
    block = text[text.index(start_marker):]
    block = block[:block.index(end_marker)]
    return [t.strip() for t in block.split("\n") if t.strip()]


def parse_jiang_table(tokens, analytes, n_leading_skip, has_flavour):
    """Walk the token stream: cultivar header, then vineyard rows of fixed width."""
    width = n_leading_skip + len(analytes) + (1 if has_flavour else 0)
    rows, cultivar, i = [], None, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in CULTIVAR_CODE:
            cultivar = tok
            i += 1
            continue
        if cultivar and tok == "Non-smoke":
            # 99th-percentile reference row, not a wine. Its width varies (the
            # Shiraz row in Table 4 has no "—" flavour cell), so skip to the
            # next vineyard label instead of a fixed number of tokens.
            i += 1
            while i < len(tokens) and not re.fullmatch(r"[A-L][‡∗]?", tokens[i]):
                i += 1
            continue
        if cultivar and re.fullmatch(r"[A-L][‡∗]?", tok):
            rows.append((cultivar, clean_label(tok), tokens[i + 1:i + 1 + width]))
            i += 1 + width
            continue
        i += 1
    return rows


def jiang_grapes():
    tokens = jiang_tokens("Table 3: Phenolic analytes", "Apart from “non-smoke” values")
    analytes = ["SyGG", "MSyGG", "PhRG", "GuRG", "CrRG", "MGuRG",
                "4-methylguaiacol", "guaiacol", "o-cresol", "p-cresol", "m-cresol"]
    out = []
    for cultivar, vineyard, values in parse_jiang_table(tokens, analytes, n_leading_skip=1, has_flavour=False):
        code = CULTIVAR_CODE[cultivar]
        for analyte, raw in zip(analytes, values[1:]):  # values[0] is berry weight
            value, below = parse_value(raw, analyte)
            out.append(dict(study="A", source="Jiang2022_Table3", sample_id=f"{code}-{vineyard}",
                            cultivar_code=code, matrix="grape", analyte=analyte,
                            raw_value=raw, value=value, below_loq=below))
    return out


def jiang_wines():
    tokens = jiang_tokens("Table 4: Phenolic analytes", "Volatile phenols and phenolic in wine results")
    analytes = ["SyGG", "MSyGG", "PhRG", "GuRG", "CrRG", "MGuRG", "4-methylguaiacol", "guaiacol",
                "o-cresol", "p-cresol", "m-cresol", "syringol", "4-methylsyringol"]
    conc, flavour = [], []
    for cultivar, vineyard, values in parse_jiang_table(tokens, analytes, n_leading_skip=0, has_flavour=True):
        code = CULTIVAR_CODE[cultivar]
        sample = f"{code}-{vineyard}"
        for analyte, raw in zip(analytes, values[:-1]):
            value, below = parse_value(raw, analyte)
            conc.append(dict(study="A", source="Jiang2022_Table4", sample_id=sample,
                             cultivar_code=code, matrix="wine", analyte=analyte,
                             raw_value=raw, value=value, below_loq=below))
        flavour.append(dict(sample_id=sample, smoke_flavour=float(clean_label(values[-1]))))
    return conc, pd.DataFrame(flavour)


def jiang_aroma():
    table = docx.Document(JIANG_SI).tables[3]  # Table S4: mean sensory ratings
    rows, cultivar = [], None
    for r in table.rows[1:]:
        cells = [c.text.strip() for c in r.cells]
        cultivar = cells[0] or cultivar
        rows.append(dict(sample_id=f"{CULTIVAR_CODE[cultivar]}-{clean_label(cells[1])}",
                         smoke_aroma=float(clean_label(cells[2]))))
    return pd.DataFrame(rows)


def main():
    parker_tables = docx.Document(PARKER_SI).tables
    long_rows = (parker_concentrations(parker_tables[2], "grape")
                 + parker_concentrations(parker_tables[3], "wine")
                 + jiang_grapes())
    jiang_conc, jiang_flavour = jiang_wines()
    long_rows += jiang_conc
    long_df = pd.DataFrame(long_rows)

    sensory = pd.concat([
        parker_sensory(parker_tables[5]).assign(study="B"),
        jiang_flavour.merge(jiang_aroma(), on="sample_id", how="left").assign(study="A"),
    ], ignore_index=True)

    wide = long_df.pivot_table(index=["study", "sample_id", "cultivar_code"],
                               columns=["matrix", "analyte"], values="value", aggfunc="first")
    wide.columns = [f"{m}_{a}" for m, a in wide.columns]
    wide = wide.reset_index()
    # Keep only samples that became a sensory-rated wine
    wide = sensory.merge(wide, on=["study", "sample_id"], how="left")
    wide.insert(0, "wine_uid", wide["study"] + "_" + wide["sample_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT / "awri_wine_smoke_long.csv", index=False)
    wide.to_csv(OUT / "awri_wine_smoke_wide.csv", index=False)

    print(f"Long rows: {len(long_df)}")
    print(f"Wines: {len(wide)}  by study/cultivar:")
    print(wide.groupby(["study", "cultivar_code"]).size().to_string())
    print(f"Wines missing grape data: {wide.filter(like='grape_').isna().all(axis=1).sum()}")
    print(f"Wines missing smoke aroma: {wide['smoke_aroma'].isna().sum()}")


if __name__ == "__main__":
    main()
