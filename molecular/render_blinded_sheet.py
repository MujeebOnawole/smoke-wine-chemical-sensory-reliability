"""Render the blinded rating sheet to a molecule grid (run locally; needs RDKit drawing, not available in the Bunya container).
Usage: python render_blinded_sheet.py <path to blinded_sheet_FOR_RATER.csv>
Writes blinded_molecules.png next to the sheet. Scenario labels are never read."""
import os
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

sheet = sys.argv[1]
df = pd.read_csv(sheet)
mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 240), legends=df["blind_id"].tolist())
out = os.path.join(os.path.dirname(os.path.abspath(sheet)), "blinded_molecules.png")
img.save(out)
print("wrote", out)
