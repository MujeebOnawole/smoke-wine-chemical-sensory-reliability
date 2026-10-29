"""
Consolidate all flavor/odor datasets into a unified training set.

Sources:
1. FooDB (already processed) - 2,362 compounds, 116 fine-grained flavor labels
2. FlavorDB (via Pyrfume) - 25,595 compounds, text-based flavor/odor labels
3. Leffingwell (via Pyrfume) - 3,522 compounds, 113 binary odor labels
4. GoodScents (via Pyrfume) - 4,565 compounds, text descriptors
5. FlavorNet (via Pyrfume) - 716 compounds, text descriptors

Output:
- consolidated_flavor_dataset.csv: unified multi-label binary encoding
- consolidation_report.txt: detailed merge statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available. SMILES canonicalization will be skipped.")

BASE = Path(__file__).resolve().parent.parent
PYRFUME = BASE / "QAAFI AI project _data" / "Taste_SmokeTaint" / "data" / "raw" / "pyrfume"
FOODB_PROCESSED = BASE / "QAAFI AI project _data" / "Taste_SmokeTaint" / "data" / "processed"
OUT = BASE / "QAAFI AI project _data" / "Taste_SmokeTaint" / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def canonicalize_smiles(smi):
    """Canonicalize SMILES using RDKit. Returns None if invalid."""
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except:
        pass
    return None


def parse_semicolon_descriptors(text):
    """Parse semicolon-separated descriptor string into a set of lowercased terms."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return set()
    return {t.strip().lower() for t in text.split(';') if t.strip()}


# ---- Standardized flavor vocabulary ----
# Map diverse descriptors to a unified set of ~20 categories
DESCRIPTOR_MAP = {
    # Smoky (our primary target)
    'smoky': 'smoky', 'smoke': 'smoky', 'smoked': 'smoky',
    'tobacco': 'smoky', 'burnt': 'smoky', 'charred': 'smoky',
    'campfire': 'smoky', 'ashy': 'smoky', 'tarry': 'smoky',

    # Phenolic (smoke-related)
    'phenolic': 'phenolic', 'phenol': 'phenolic', 'creosote': 'phenolic',
    'medicinal': 'phenolic', 'guaiacol': 'phenolic', 'iodine': 'phenolic',

    # Fruity
    'fruity': 'fruity', 'apple': 'fruity', 'pear': 'fruity', 'peach': 'fruity',
    'apricot': 'fruity', 'cherry': 'fruity', 'plum': 'fruity', 'grape': 'fruity',
    'berry': 'fruity', 'banana': 'fruity', 'melon': 'fruity', 'tropical': 'fruity',
    'pineapple': 'fruity', 'strawberry': 'fruity', 'raspberry': 'fruity',
    'blackberry': 'fruity', 'blueberry': 'fruity', 'grapefruit': 'fruity',
    'orange': 'fruity', 'lemon': 'fruity', 'lime': 'fruity', 'citrus': 'fruity',
    'black currant': 'fruity', 'passion fruit': 'fruity', 'mango': 'fruity',
    'fig': 'fruity', 'ripe': 'fruity', 'overripe': 'fruity',

    # Floral
    'floral': 'floral', 'rose': 'floral', 'jasmine': 'floral', 'violet': 'floral',
    'lilac': 'floral', 'lavender': 'floral', 'geranium': 'floral',
    'muguet': 'floral', 'lily': 'floral', 'carnation': 'floral',
    'chamomile': 'floral', 'chrysanthemum': 'floral', 'honeysuckle': 'floral',
    'hyacinth': 'floral', 'magnolia': 'floral', 'orchid': 'floral',
    'orris': 'floral', 'narcissus': 'floral', 'tuberose': 'floral',

    # Sweet
    'sweet': 'sweet', 'honey': 'sweet', 'caramel': 'sweet', 'vanilla': 'sweet',
    'sugar': 'sweet', 'candy': 'sweet', 'syrup': 'sweet', 'molasses': 'sweet',
    'butterscotch': 'sweet', 'toffee': 'sweet', 'chocolate': 'sweet',
    'cocoa': 'sweet', 'caramellic': 'sweet', 'cotton candy': 'sweet',
    'saccharine': 'sweet', 'sugary': 'sweet',

    # Green/Herbal
    'green': 'green', 'grassy': 'green', 'herbal': 'green', 'leafy': 'green',
    'hay': 'green', 'tea': 'green', 'fresh': 'green', 'cucumber': 'green',
    'herbaceous': 'green', 'vegetable': 'green', 'celery': 'green',
    'basil': 'green', 'thyme': 'green', 'oregano': 'green', 'mint': 'green',
    'minty': 'green', 'eucalyptus': 'green',

    # Woody
    'woody': 'woody', 'cedar': 'woody', 'pine': 'woody', 'sandalwood': 'woody',
    'oak': 'woody', 'bark': 'woody', 'resinous': 'woody', 'balsamic': 'woody',
    'amber': 'woody', 'mossy': 'woody', 'earthy': 'woody', 'mushroom': 'woody',
    'musty': 'woody', 'damp': 'woody', 'forest': 'woody',

    # Spicy
    'spicy': 'spicy', 'cinnamon': 'spicy', 'clove': 'spicy', 'pepper': 'spicy',
    'peppery': 'spicy', 'ginger': 'spicy', 'nutmeg': 'spicy', 'anise': 'spicy',
    'anisic': 'spicy', 'cardamom': 'spicy', 'camphoreous': 'spicy',
    'camphor': 'spicy', 'warm': 'spicy', 'pungent': 'spicy', 'sharp': 'spicy',

    # Nutty
    'nutty': 'nutty', 'almond': 'nutty', 'hazelnut': 'nutty', 'walnut': 'nutty',
    'peanut': 'nutty', 'coconut': 'nutty', 'marzipan': 'nutty',
    'roasted': 'nutty', 'toasted': 'nutty', 'popcorn': 'nutty',

    # Fatty/Oily
    'fatty': 'fatty', 'oily': 'fatty', 'waxy': 'fatty', 'buttery': 'fatty',
    'creamy': 'fatty', 'soapy': 'fatty', 'dairy': 'fatty', 'milky': 'fatty',
    'cheesy': 'fatty', 'lard': 'fatty',

    # Savory/Meaty
    'savory': 'savory', 'meaty': 'savory', 'beefy': 'savory', 'brothy': 'savory',
    'chicken': 'savory', 'fishy': 'savory', 'umami': 'savory',
    'bouillon': 'savory',

    # Sulfurous
    'sulfurous': 'sulfurous', 'sulfur': 'sulfurous', 'onion': 'sulfurous',
    'garlic': 'sulfurous', 'cabbage': 'sulfurous', 'radish': 'sulfurous',
    'horseradish': 'sulfurous', 'mustard': 'sulfurous',
    'alliaceous': 'sulfurous', 'egg': 'sulfurous',

    # Fermented/Alcoholic
    'fermented': 'fermented', 'winey': 'fermented', 'wine': 'fermented',
    'alcoholic': 'fermented', 'rum': 'fermented', 'brandy': 'fermented',
    'cognac': 'fermented', 'beer': 'fermented', 'vinegar': 'fermented',
    'sour': 'fermented', 'acidic': 'fermented', 'yeast': 'fermented',
    'malty': 'fermented',

    # Chemical/Solvent
    'chemical': 'chemical', 'solvent': 'chemical', 'ethereal': 'chemical',
    'gasoline': 'chemical', 'plastic': 'chemical', 'paint': 'chemical',
    'aldehydic': 'chemical', 'ketonic': 'chemical', 'acetic': 'chemical',

    # Leather/Animal
    'leather': 'animal', 'leathery': 'animal', 'animal': 'animal',
    'musk': 'animal', 'musky': 'animal', 'fecal': 'animal', 'civet': 'animal',
    'castoreum': 'animal', 'ambergris': 'animal',

    # Bitter
    'bitter': 'bitter',

    # Cooling
    'cooling': 'cooling', 'cool': 'cooling', 'menthol': 'cooling', 'icy': 'cooling',

    # Powdery
    'powdery': 'powdery', 'dusty': 'powdery', 'dry': 'powdery',

    # Metallic
    'metallic': 'metallic', 'blood': 'metallic', 'iron': 'metallic',
}

# The unified label set
UNIFIED_LABELS = sorted(set(DESCRIPTOR_MAP.values()))
print(f"Unified label set: {len(UNIFIED_LABELS)} categories")
print(f"  {UNIFIED_LABELS}")


def descriptors_to_labels(desc_set):
    """Map a set of raw descriptors to unified label set."""
    labels = set()
    for desc in desc_set:
        desc_lower = desc.lower().strip()
        if desc_lower in DESCRIPTOR_MAP:
            labels.add(DESCRIPTOR_MAP[desc_lower])
        # Try partial matching for compound descriptors like "maraschino cherry"
        else:
            for key, val in DESCRIPTOR_MAP.items():
                if key in desc_lower or desc_lower in key:
                    labels.add(val)
                    break
    return labels


# ====================================================================
# 1. Load FooDB (already processed)
# ====================================================================
print("\n--- Loading FooDB ---")
foodb_long = pd.read_csv(FOODB_PROCESSED / "foodb_compound_flavor_long.csv")
foodb_ref = pd.read_csv(FOODB_PROCESSED / "foodb_flavor_reference.csv")

# Get unique SMILES with their flavor descriptors
foodb_flavors = foodb_long.groupby('smiles')['flavor_name'].apply(
    lambda x: set(x.dropna())
).reset_index()

# Map to unified labels
foodb_flavors['unified_labels'] = foodb_flavors['flavor_name'].apply(descriptors_to_labels)
foodb_flavors['source'] = 'foodb'

# Canonicalize SMILES
if HAS_RDKIT:
    foodb_flavors['canonical_smiles'] = foodb_flavors['smiles'].apply(canonicalize_smiles)
    foodb_flavors = foodb_flavors[foodb_flavors['canonical_smiles'].notna()].copy()
else:
    foodb_flavors['canonical_smiles'] = foodb_flavors['smiles']

print(f"FooDB: {len(foodb_flavors)} compounds")


# ====================================================================
# 2. Load FlavorDB (via Pyrfume)
# ====================================================================
print("\n--- Loading FlavorDB ---")
fdb_mol = pd.read_csv(PYRFUME / "flavordb_molecules.csv")
fdb_beh = pd.read_csv(PYRFUME / "flavordb_behavior.csv")

# Merge molecules with behavior
fdb = fdb_mol.merge(fdb_beh, left_on='CID', right_on='Stimulus', how='inner')

# Parse flavor and odor percepts
fdb['all_descriptors'] = fdb.apply(lambda r:
    parse_semicolon_descriptors(r.get('Odor Percepts', '')) |
    parse_semicolon_descriptors(r.get('Flavor Percepts', '')),
    axis=1)

# Filter out compounds with no descriptors
fdb = fdb[fdb['all_descriptors'].apply(len) > 0].copy()

# Map to unified labels
fdb['unified_labels'] = fdb['all_descriptors'].apply(descriptors_to_labels)
fdb = fdb[fdb['unified_labels'].apply(len) > 0].copy()

# Canonicalize SMILES
if HAS_RDKIT:
    fdb['canonical_smiles'] = fdb['IsomericSMILES'].apply(canonicalize_smiles)
    fdb = fdb[fdb['canonical_smiles'].notna()].copy()
else:
    fdb['canonical_smiles'] = fdb['IsomericSMILES']

fdb['source'] = 'flavordb'
print(f"FlavorDB: {len(fdb)} compounds with mapped labels")


# ====================================================================
# 3. Load Leffingwell (already multi-label binary)
# ====================================================================
print("\n--- Loading Leffingwell ---")
leff_mol = pd.read_csv(PYRFUME / "leffingwell_molecules.csv")
leff_beh = pd.read_csv(PYRFUME / "leffingwell_behavior.csv")

# Leffingwell behavior is already binary - get column names as descriptors
leff_desc_cols = [c for c in leff_beh.columns if c != 'Stimulus']

# Convert binary to descriptor sets
def binary_to_desc_set(row):
    return {col for col in leff_desc_cols if row.get(col, 0) == 1}

leff_beh['all_descriptors'] = leff_beh.apply(binary_to_desc_set, axis=1)
leff = leff_mol.merge(leff_beh[['Stimulus', 'all_descriptors']], left_on='CID', right_on='Stimulus', how='inner')

# Filter and map
leff = leff[leff['all_descriptors'].apply(len) > 0].copy()
leff['unified_labels'] = leff['all_descriptors'].apply(descriptors_to_labels)
leff = leff[leff['unified_labels'].apply(len) > 0].copy()

if HAS_RDKIT:
    leff['canonical_smiles'] = leff['IsomericSMILES'].apply(canonicalize_smiles)
    leff = leff[leff['canonical_smiles'].notna()].copy()
else:
    leff['canonical_smiles'] = leff['IsomericSMILES']

leff['source'] = 'leffingwell'
print(f"Leffingwell: {len(leff)} compounds with mapped labels")


# ====================================================================
# 4. Load GoodScents
# ====================================================================
print("\n--- Loading GoodScents ---")
gs_mol = pd.read_csv(PYRFUME / "goodscents_molecules.csv")
gs_beh = pd.read_csv(PYRFUME / "goodscents_behavior.csv")
gs_stim = pd.read_csv(PYRFUME / "goodscents_stimuli.csv")

# GoodScents uses CAS as Stimulus; stimuli table maps CAS -> CID
gs_beh['all_descriptors'] = gs_beh['Descriptors'].apply(parse_semicolon_descriptors)
# Merge behavior with stimuli to get CID
gs_beh_with_cid = gs_beh.merge(gs_stim[['Stimulus', 'CID']], on='Stimulus', how='inner')
# Now merge with molecules on CID
gs = gs_mol.merge(gs_beh_with_cid[['CID', 'all_descriptors']], on='CID', how='inner')
gs = gs[gs['all_descriptors'].apply(len) > 0].copy()
gs['unified_labels'] = gs['all_descriptors'].apply(descriptors_to_labels)
gs = gs[gs['unified_labels'].apply(len) > 0].copy()

if HAS_RDKIT:
    gs['canonical_smiles'] = gs['IsomericSMILES'].apply(canonicalize_smiles)
    gs = gs[gs['canonical_smiles'].notna()].copy()
else:
    gs['canonical_smiles'] = gs['IsomericSMILES']

gs['source'] = 'goodscents'
print(f"GoodScents: {len(gs)} compounds with mapped labels")


# ====================================================================
# 5. Load FlavorNet
# ====================================================================
print("\n--- Loading FlavorNet ---")
fn_mol = pd.read_csv(PYRFUME / "flavornet_molecules.csv")
fn_beh = pd.read_csv(PYRFUME / "flavornet_behavior.csv")
fn_stim = pd.read_csv(PYRFUME / "flavornet_stimuli.csv")

fn_beh['all_descriptors'] = fn_beh['Descriptors'].apply(parse_semicolon_descriptors)
# Merge behavior with stimuli to get CID
fn_beh_with_cid = fn_beh.merge(fn_stim[['Stimulus', 'CID']], on='Stimulus', how='inner')
fn = fn_mol.merge(fn_beh_with_cid[['CID', 'all_descriptors']], on='CID', how='inner')
fn = fn[fn['all_descriptors'].apply(len) > 0].copy()
fn['unified_labels'] = fn['all_descriptors'].apply(descriptors_to_labels)
fn = fn[fn['unified_labels'].apply(len) > 0].copy()

if HAS_RDKIT:
    fn['canonical_smiles'] = fn['IsomericSMILES'].apply(canonicalize_smiles)
    fn = fn[fn['canonical_smiles'].notna()].copy()
else:
    fn['canonical_smiles'] = fn['IsomericSMILES']

fn['source'] = 'flavornet'
print(f"FlavorNet: {len(fn)} compounds with mapped labels")


# ====================================================================
# 6. Consolidate all datasets
# ====================================================================
print("\n" + "="*60)
print("CONSOLIDATING ALL DATASETS")
print("="*60)

# Collect all into unified format: canonical_smiles, unified_labels, source
all_data = []

for df, src in [(foodb_flavors, 'foodb'), (fdb, 'flavordb'), (leff, 'leffingwell'),
                (gs, 'goodscents'), (fn, 'flavornet')]:
    for _, row in df.iterrows():
        all_data.append({
            'canonical_smiles': row['canonical_smiles'],
            'unified_labels': row['unified_labels'],
            'source': row['source'],
        })

all_df = pd.DataFrame(all_data)
print(f"\nTotal entries before dedup: {len(all_df)}")
print(f"Unique SMILES before dedup: {all_df['canonical_smiles'].nunique()}")

# Group by canonical_smiles: merge labels from all sources
grouped = all_df.groupby('canonical_smiles').agg(
    unified_labels=('unified_labels', lambda x: set().union(*x)),
    sources=('source', lambda x: set(x)),
    n_sources=('source', 'nunique'),
).reset_index()

print(f"Unique compounds after dedup: {len(grouped)}")
print(f"\nSource overlap:")
print(f"  In 1 source: {(grouped['n_sources'] == 1).sum()}")
print(f"  In 2 sources: {(grouped['n_sources'] == 2).sum()}")
print(f"  In 3+ sources: {(grouped['n_sources'] >= 3).sum()}")

# Create multi-label binary encoding
for label in UNIFIED_LABELS:
    grouped[f'label_{label}'] = grouped['unified_labels'].apply(lambda x: int(label in x))

# Count labels per compound
grouped['n_labels'] = grouped[[f'label_{l}' for l in UNIFIED_LABELS]].sum(axis=1)
grouped['sources_str'] = grouped['sources'].apply(lambda x: ','.join(sorted(x)))

# Drop intermediate columns
output = grouped.drop(columns=['unified_labels', 'sources']).copy()

# Sort by number of labels (richest annotations first)
output = output.sort_values('n_labels', ascending=False).reset_index(drop=True)

# ====================================================================
# 7. Summary statistics
# ====================================================================
label_cols = [f'label_{l}' for l in UNIFIED_LABELS]

print(f"\n--- Label Distribution ---")
for label in UNIFIED_LABELS:
    col = f'label_{label}'
    n = output[col].sum()
    pct = n / len(output) * 100
    print(f"  {label:15s}: {n:6d} compounds ({pct:5.1f}%)")

print(f"\n--- Key: Smoky compounds ---")
smoky = output[output['label_smoky'] == 1]
print(f"  Total smoky compounds: {len(smoky)}")
print(f"  Sources: {smoky['sources_str'].value_counts().to_dict()}")

print(f"\n--- Key: Phenolic compounds ---")
phenolic = output[output['label_phenolic'] == 1]
print(f"  Total phenolic compounds: {len(phenolic)}")

print(f"\n--- Labels per compound ---")
print(output['n_labels'].describe().to_string())

# ====================================================================
# 8. Save outputs
# ====================================================================
output.to_csv(OUT / "consolidated_flavor_dataset.csv", index=False)
print(f"\nSaved: {OUT / 'consolidated_flavor_dataset.csv'}")
print(f"  {len(output)} compounds, {len(UNIFIED_LABELS)} label columns")

# Also save a smoky-focused binary dataset for the primary task
smoky_binary = output[['canonical_smiles', 'label_smoky', 'label_phenolic',
                        'n_labels', 'sources_str']].copy()
# Add all labels as context features
for col in label_cols:
    if col not in smoky_binary.columns:
        smoky_binary[col] = output[col]
smoky_binary.to_csv(OUT / "smoky_binary_dataset.csv", index=False)
print(f"Saved: {OUT / 'smoky_binary_dataset.csv'}")

# Report
report_lines = [
    "FLAVOR DATASET CONSOLIDATION REPORT",
    "=" * 60,
    f"Date: 2026-02-21",
    "",
    "Sources:",
    f"  FooDB:       {len(foodb_flavors):>6,} compounds",
    f"  FlavorDB:    {len(fdb):>6,} compounds",
    f"  Leffingwell: {len(leff):>6,} compounds",
    f"  GoodScents:  {len(gs):>6,} compounds",
    f"  FlavorNet:   {len(fn):>6,} compounds",
    "",
    f"After deduplication by canonical SMILES: {len(output):>6,} unique compounds",
    f"Unified label categories: {len(UNIFIED_LABELS)}",
    f"Labels: {UNIFIED_LABELS}",
    "",
    "Label distribution:",
]
for label in UNIFIED_LABELS:
    col = f'label_{label}'
    n = output[col].sum()
    pct = n / len(output) * 100
    report_lines.append(f"  {label:15s}: {n:6d} ({pct:5.1f}%)")

report_lines.extend([
    "",
    f"Smoky compounds: {len(smoky)}",
    f"Phenolic compounds: {len(phenolic)}",
    "",
    "Output files:",
    "  1. consolidated_flavor_dataset.csv - Full multi-label dataset",
    "  2. smoky_binary_dataset.csv - Smoky-focused binary classification",
])

report_text = "\n".join(report_lines)
(OUT / "consolidation_report.txt").write_text(report_text)
print(f"Saved: {OUT / 'consolidation_report.txt'}")
