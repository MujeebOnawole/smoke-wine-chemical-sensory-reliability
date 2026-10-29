# prepare_smoke_taint_data.py
# Converts consolidated_flavor_dataset.csv into model-ready CSV files
# with dual split strategy (random + scaffold-based) for 19-label
# multi-label flavor classification.

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 19 flavor label columns
LABEL_COLUMNS = [
    'label_animal', 'label_bitter', 'label_chemical', 'label_cooling',
    'label_fatty', 'label_fermented', 'label_floral', 'label_fruity',
    'label_green', 'label_metallic', 'label_nutty', 'label_phenolic',
    'label_powdery', 'label_savory', 'label_smoky', 'label_spicy',
    'label_sulfurous', 'label_sweet', 'label_woody'
]

CLASS_NAMES = [
    'animal', 'bitter', 'chemical', 'cooling', 'fatty', 'fermented',
    'floral', 'fruity', 'green', 'metallic', 'nutty', 'phenolic',
    'powdery', 'savory', 'smoky', 'spicy', 'sulfurous', 'sweet', 'woody'
]

# Volatile phenols measured as wine smoke-taint markers. The ten compounds are
# the union of those quantified by Tan et al. (2024, JAFC; free and total forms
# of all ten) and Parker et al. (2023, AJGWR; the first seven). They are removed
# from the training/valid/test splits and written to a separate external file,
# so any score the model gives them is a genuine out-of-training prediction.
SMOKE_MARKERS = {
    'guaiacol': 'COc1ccccc1O',
    '4-methylguaiacol': 'COc1cc(C)ccc1O',
    'syringol': 'COc1cccc(OC)c1O',
    '4-methylsyringol': 'COc1cc(C)cc(OC)c1O',
    'o-cresol': 'Cc1ccccc1O',
    'm-cresol': 'Cc1cccc(O)c1',
    'p-cresol': 'Cc1ccc(O)cc1',
    'phenol': 'Oc1ccccc1',
    '4-ethylguaiacol': 'CCc1ccc(O)c(OC)c1',
    '4-ethylphenol': 'CCc1ccc(O)cc1',
}


def hold_out_smoke_markers(df):
    """Split off the wine smoke markers; return (remaining_df, markers_df).

    Also records each marker's nearest-neighbour Tanimoto similarity
    (Morgan radius 2, 2048 bits) to the remaining compounds, which shows how
    far the external test is from interpolation.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    canon_to_name = {Chem.MolToSmiles(Chem.MolFromSmiles(s)): n
                     for n, s in SMOKE_MARKERS.items()}
    is_marker = df['PROCESSED_SMILES'].isin(canon_to_name)
    markers = df[is_marker].copy()
    remaining = df[~is_marker].copy()
    markers['MARKER_NAME'] = markers['PROCESSED_SMILES'].map(canon_to_name)

    missing = set(SMOKE_MARKERS) - set(markers['MARKER_NAME'])
    if missing:
        logger.warning(f"Smoke markers not found in dataset: {sorted(missing)}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ref_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in remaining['PROCESSED_SMILES']]
    ref_smiles = remaining['PROCESSED_SMILES'].tolist()
    nn_sim, nn_smiles = [], []
    for smi in markers['PROCESSED_SMILES']:
        sims = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(smi)), ref_fps)
        i = int(np.argmax(sims))
        nn_sim.append(round(sims[i], 3))
        nn_smiles.append(ref_smiles[i])
    markers['NN_TANIMOTO_TO_REMAINING'] = nn_sim
    markers['NN_SMILES'] = nn_smiles

    logger.info(f"Held out {len(markers)} wine smoke markers:")
    for _, r in markers.iterrows():
        logger.info(f"  {r['MARKER_NAME']:18s} smoky={r['label_smoky']} phenolic={r['label_phenolic']} "
                    f"NN Tanimoto={r['NN_TANIMOTO_TO_REMAINING']}")
    return remaining, markers


def validate_and_canonicalize_smiles(smiles: str):
    """Validate SMILES via RDKit and return canonical form, or None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def generate_scaffold(smiles: str, generic: bool = False):
    """Generate Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return scaffold
    except Exception:
        return ''


def generate_detailed_scaffold(smiles: str):
    """Generate a more specific scaffold using the full Murcko decomposition.

    Falls back to canonical SMILES itself for molecules whose base scaffold
    is too generic (e.g. bare benzene) or empty (no rings).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )

        if not scaffold:
            return smiles

        scaffold_mol = Chem.MolFromSmiles(scaffold)
        if scaffold_mol is None:
            return smiles

        n_heavy = scaffold_mol.GetNumHeavyAtoms()
        if n_heavy <= 6:
            return smiles

        return scaffold
    except Exception:
        return smiles


def scaffold_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Scaffold split for multi-label classification.

    Uses fill-ratio-priority greedy allocation. Since multi-label data
    cannot be stratified by a single class, we use a label-frequency-weighted
    deviation approach: the composite score considers how well the per-label
    distributions in each split match the overall dataset distribution.

    Algorithm:
    - Scaffolds too large for valid/test are forced into training.
    - Remaining scaffolds are sorted largest-first and greedily assigned
      using a composite score: 20 * fill_ratio^2 + label_deviation.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.RandomState(seed)

    df = df.copy()
    logger.info("Generating Murcko scaffolds...")
    df['scaffold'] = df['PROCESSED_SMILES'].apply(generate_scaffold)

    total = len(df)

    # Compute overall label frequencies
    label_fracs = {}
    label_counts_total = {}
    for label_col in LABEL_COLUMNS:
        count = df[label_col].sum()
        label_counts_total[label_col] = count
        label_fracs[label_col] = count / total

    logger.info("Target label frequencies:")
    for label_col, name in zip(LABEL_COLUMNS, CLASS_NAMES):
        logger.info(f"  {name}: {label_counts_total[label_col]} ({label_fracs[label_col]*100:.1f}%)")

    split_names = ['training', 'valid', 'test']
    split_fracs = {'training': train_frac, 'valid': val_frac, 'test': test_frac}
    split_targets = {s: total * split_fracs[s] for s in split_names}
    split_label_targets = {
        s: {lc: split_fracs[s] * label_counts_total[lc] for lc in LABEL_COLUMNS}
        for s in split_names
    }

    # Group by scaffold
    scaffold_groups_iter = df.groupby('scaffold')
    scaffolds = []
    for name, group in scaffold_groups_iter:
        lc = {col: group[col].sum() for col in LABEL_COLUMNS}
        scaffolds.append((name, group, lc))

    rng.shuffle(scaffolds)

    # Scaffolds larger than the valid/test target MUST go to training
    max_minor_size = int(total * max(val_frac, test_frac))
    large = [s for s in scaffolds if len(s[1]) > max_minor_size]
    small = [s for s in scaffolds if len(s[1]) <= max_minor_size]
    small.sort(key=lambda x: len(x[1]), reverse=True)

    logger.info(f"Scaffolds: {len(scaffolds)} total, {len(large)} forced to training "
                f"({sum(len(s[1]) for s in large)} mol), "
                f"{len(small)} for greedy allocation")

    split_idx = {s: [] for s in split_names}
    split_sizes = {s: 0 for s in split_names}
    split_lc = {s: {col: 0 for col in LABEL_COLUMNS} for s in split_names}

    # Phase 1: Force oversized scaffolds into training
    for _, group, lc in large:
        split_idx['training'].extend(group.index.tolist())
        split_sizes['training'] += len(group)
        for col in LABEL_COLUMNS:
            split_lc['training'][col] += lc.get(col, 0)

    logger.info(f"After forcing large scaffolds: training = {split_sizes['training']} molecules")

    # Phase 2: Fill-ratio-priority greedy with label-balance tiebreaking
    for _, group, scaffold_lc in small:
        indices = group.index.tolist()
        n = len(indices)

        best_split = None
        best_score = float('inf')

        for s in split_names:
            target = split_targets[s]
            if target == 0:
                continue

            fill_after = (split_sizes[s] + n) / target

            # Hard block: don't exceed 110% while others are below 95%
            if fill_after > 1.10:
                others_under = any(
                    split_sizes[o] / max(split_targets[o], 1) < 0.95
                    for o in split_names if o != s
                )
                if others_under:
                    continue

            # Label deviation after adding this scaffold
            label_dev = 0.0
            for col in LABEL_COLUMNS:
                current = split_lc[s][col] + scaffold_lc.get(col, 0)
                lt = split_label_targets[s][col]
                if lt > 0:
                    label_dev += ((current - lt) / lt) ** 2

            # Composite score: fill ratio^2 (weighted 20x) + label deviation
            score = 20.0 * (fill_after ** 2) + label_dev

            if score < best_score:
                best_score = score
                best_split = s

        # Fallback: assign to the least-filled split
        if best_split is None:
            best_split = min(split_names,
                             key=lambda s: split_sizes[s] / max(split_targets[s], 1))

        split_idx[best_split].extend(indices)
        split_sizes[best_split] += n
        for col in LABEL_COLUMNS:
            split_lc[best_split][col] += scaffold_lc.get(col, 0)

    df.loc[split_idx['training'], 'group'] = 'training'
    df.loc[split_idx['valid'], 'group'] = 'valid'
    df.loc[split_idx['test'], 'group'] = 'test'

    # Log achieved distribution vs targets
    for s in split_names:
        n_s = split_sizes[s]
        logger.info(f"  {s}: {n_s} ({n_s/total*100:.1f}%)")
        for col, name in zip(LABEL_COLUMNS, CLASS_NAMES):
            achieved = split_lc[s][col]
            pct = achieved / max(n_s, 1) * 100
            target_pct = label_fracs[col] * 100
            logger.info(f"    {name}: {achieved} ({pct:.1f}% vs target {target_pct:.1f}%)")

    df = df.drop(columns=['scaffold'])
    return df


def random_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Random split for multi-label classification.
    Stratifies on label_smoky, the primary target and a minority label, so
    each partition receives a proportional share of smoky positives. For
    multi-label data, perfect stratification across all 19 labels is not
    feasible with standard methods.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    stratify_col = 'label_smoky'

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_frac, stratify=df[stratify_col], random_state=seed
    )

    # Second split: train vs val (val_frac relative to remaining)
    relative_val_frac = val_frac / (train_frac + val_frac)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_frac,
        stratify=train_val_df[stratify_col], random_state=seed
    )

    df = df.copy()
    df['group'] = ''
    df.loc[train_df.index, 'group'] = 'training'
    df.loc[val_df.index, 'group'] = 'valid'
    df.loc[test_df.index, 'group'] = 'test'

    return df


def prepare_smoke_taint_data(input_path: str, output_dir: str, seed: int = 42):
    """
    Main pipeline: consolidated_flavor_dataset.csv -> model-ready CSVs
    with dual split strategy for 19-label multi-label flavor classification.

    Args:
        input_path: Path to consolidated_flavor_dataset.csv
        output_dir: Directory to save processed files
        seed: Random seed for reproducibility
    """
    logger.info(f"Reading consolidated_flavor_dataset.csv from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Raw dataset: {len(df)} rows, columns: {list(df.columns)}")

    # --- Step 1: Validate label columns ---
    logger.info("Validating label columns...")
    missing_labels = [col for col in LABEL_COLUMNS if col not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}")

    # Ensure all label columns are binary integers
    for col in LABEL_COLUMNS:
        if not set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Label column {col} contains non-binary values: {df[col].unique()}")
        df[col] = df[col].astype(int)

    logger.info("Label distribution:")
    for col, name in zip(LABEL_COLUMNS, CLASS_NAMES):
        pos_count = df[col].sum()
        pct = pos_count / len(df) * 100
        logger.info(f"  {name}: {pos_count} ({pct:.1f}%)")

    # --- Step 2: Validate and canonicalize SMILES ---
    logger.info("Validating and canonicalizing SMILES...")
    smiles_col = 'canonical_smiles'
    if smiles_col not in df.columns:
        # Try alternative column names
        for alt in ['SMILES', 'smiles', 'Canonical_SMILES']:
            if alt in df.columns:
                smiles_col = alt
                break
        else:
            raise ValueError(f"No SMILES column found. Available: {list(df.columns)}")

    df['PROCESSED_SMILES'] = df[smiles_col].apply(validate_and_canonicalize_smiles)

    invalid_count = df['PROCESSED_SMILES'].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Dropping {invalid_count} rows with invalid SMILES")
        df = df.dropna(subset=['PROCESSED_SMILES'])
    logger.info(f"After SMILES validation: {len(df)} rows")

    # --- Step 3: Drop duplicate SMILES (keep first) ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['PROCESSED_SMILES'], keep='first')
    logger.info(f"After deduplication: {len(df)} rows (removed {before_dedup - len(df)} duplicates)")

    # --- Step 4: Generate COMPOUND_ID ---
    df = df.reset_index(drop=True)
    df['COMPOUND_ID'] = [f'FLAVOR_{i:05d}' for i in range(1, len(df) + 1)]

    # Final label distribution
    logger.info("Final label distribution:")
    for col, name in zip(LABEL_COLUMNS, CLASS_NAMES):
        pos_count = df[col].sum()
        pct = pos_count / len(df) * 100
        logger.info(f"  {name}: {pos_count} ({pct:.1f}%)")

    # Multi-label statistics
    n_labels_per_sample = df[LABEL_COLUMNS].sum(axis=1)
    logger.info(f"\nLabels per sample: mean={n_labels_per_sample.mean():.2f}, "
                f"median={n_labels_per_sample.median():.0f}, "
                f"min={n_labels_per_sample.min()}, max={n_labels_per_sample.max()}")

    # --- Step 5: Hold out wine smoke markers as an external test set ---
    os.makedirs(output_dir, exist_ok=True)
    df, markers = hold_out_smoke_markers(df)
    marker_path = os.path.join(output_dir, 'smoke_markers_external.csv')
    markers[['PROCESSED_SMILES', 'COMPOUND_ID', 'MARKER_NAME'] + LABEL_COLUMNS +
            ['NN_TANIMOTO_TO_REMAINING', 'NN_SMILES']].to_csv(marker_path, index=False)
    logger.info(f"Saved: {marker_path}")
    df = df.reset_index(drop=True)

    # --- Step 6: Generate splits ---

    # Select output columns
    output_cols = ['PROCESSED_SMILES', 'COMPOUND_ID'] + LABEL_COLUMNS + ['group']

    # Split A: Random (stratified on label_smoky)
    logger.info("\n--- Generating Random Split ---")
    df_random = random_split(df.copy(), seed=seed)
    random_path = os.path.join(output_dir, 'smoke_taint_random_split.csv')
    df_random[output_cols].to_csv(random_path, index=False)
    _log_split_stats(df_random, 'Random')
    logger.info(f"Saved: {random_path}")

    # Split B: Scaffold Split
    logger.info("\n--- Generating Scaffold Split ---")
    df_scaffold = scaffold_split(df.copy(), seed=seed)
    scaffold_path = os.path.join(output_dir, 'smoke_taint_scaffold_split.csv')
    df_scaffold[output_cols].to_csv(scaffold_path, index=False)
    _log_split_stats(df_scaffold, 'Scaffold')
    logger.info(f"Saved: {scaffold_path}")

    # Verify scaffold split has no leakage
    _verify_scaffold_no_leakage(df_scaffold)

    logger.info("\nData preparation complete!")
    return random_path, scaffold_path


def _log_split_stats(df, split_name):
    """Log split statistics including per-label distribution."""
    logger.info(f"\n{split_name} Split Statistics:")
    for group in ['training', 'valid', 'test']:
        subset = df[df['group'] == group]
        logger.info(f"  {group}: {len(subset)} ({len(subset)/len(df)*100:.1f}%)")
        for col, name in zip(LABEL_COLUMNS, CLASS_NAMES):
            pos_count = subset[col].sum()
            pct = pos_count / len(subset) * 100 if len(subset) > 0 else 0
            logger.info(f"    {name}: {pos_count} ({pct:.1f}%)")


def _verify_scaffold_no_leakage(df):
    """Verify no scaffold appears in more than one split."""
    logger.info("Verifying scaffold split: no leakage check...")
    df_check = df.copy()
    df_check['scaffold'] = df_check['PROCESSED_SMILES'].apply(generate_scaffold)

    train_scaffolds = set(df_check[df_check['group'] == 'training']['scaffold'].unique())
    val_scaffolds = set(df_check[df_check['group'] == 'valid']['scaffold'].unique())
    test_scaffolds = set(df_check[df_check['group'] == 'test']['scaffold'].unique())

    train_val_overlap = train_scaffolds & val_scaffolds
    train_test_overlap = train_scaffolds & test_scaffolds
    val_test_overlap = val_scaffolds & test_scaffolds

    if train_val_overlap or train_test_overlap or val_test_overlap:
        logger.warning(f"Scaffold leakage detected!")
        logger.warning(f"  Train-Val overlap: {len(train_val_overlap)} scaffolds")
        logger.warning(f"  Train-Test overlap: {len(train_test_overlap)} scaffolds")
        logger.warning(f"  Val-Test overlap: {len(val_test_overlap)} scaffolds")
    else:
        logger.info("  No scaffold leakage detected. Splits are clean.")

    logger.info(f"  Unique scaffolds - Train: {len(train_scaffolds)}, "
               f"Val: {len(val_scaffolds)}, Test: {len(test_scaffolds)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare consolidated flavor data for multi-label RGCN model"
    )
    parser.add_argument(
        '--input', type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data', 'processed', 'consolidated_flavor_dataset.csv'),
        help='Path to consolidated_flavor_dataset.csv'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data', 'processed'),
        help='Output directory for processed files'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    prepare_smoke_taint_data(args.input, args.output_dir, args.seed)
