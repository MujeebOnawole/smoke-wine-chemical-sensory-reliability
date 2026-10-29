# predict.py
# Full XAI pipeline: ensemble prediction + best-model attribution + dual-criteria agreement + scenario classification
# Produces publication-ready output with proper A-D scenarios and scaffold-level design rules.
# (Replaces the original predict.py which used PredictionManager with known multiclass bugs.)

import os
import sys
import json
import time
import pickle
import signal
import argparse
import traceback
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from torch_geometric.data import Batch, Data
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFragmentToSmiles

from model import BaseGNN
from build_data import construct_mol_graph_from_smiles
from config import Configuration
from logger import get_logger

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================
AGREEMENT_THRESHOLD = 0.70   # >=70% fragments align with prediction
CONSENSUS_THRESHOLD = 0.80   # >=80% models predict same class (4 of 5)
CLASS_NAMES = ['Other_odour', 'Smoke_related']
TOP_K_SCAFFOLDS = 20         # Max scaffolds to report in design rules
CHECKPOINT_INTERVAL_SEC = 300  # Save checkpoint every 5 minutes
MAX_SCAFFOLDS_PER_MOL = 40   # Cap scaffolds per molecule
ATTRIBUTION_BATCH_SIZE = 5   # Masked graphs per forward pass
MAX_RINGS_COMBINATORIAL = 10 # Beyond this, skip full combinatorial decomposition (2^10 = 1024 combos max)

# Acyclic functional groups masked for XAI, alongside ring systems and ring
# substituents. Only atoms outside rings are taken from each match.
ODOUR_FG_SMARTS = {
    'carboxylic_acid': '[CX3](=O)[OX2H1]',
    'ester': '[CX3](=O)[OX2][#6]',
    'aldehyde': '[CX3H1](=O)[#6]',
    'ketone': '[#6][CX3](=O)[#6]',
    'alcohol': '[CX4][OX2H]',
    'ether': '[#6][OX2][#6]',
    'thiol': '[#6][SX2H]',
    'sulfide': '[#6][SX2][#6]',
    'amine': '[NX3;!$(N-C=O);!$(N-S=O)]',
    'alkene': '[CX3]=[CX3]',
    'nitrile': '[CX2]#N',
}


def _logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Class-probability matrix (batch, num_classes) from model logits.

    The pipeline trains binary tasks with a single BCE logit; softmax over a
    width-1 tensor returns 1.0 everywhere, so expand sigmoid(p) to [1-p, p].
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if logits.shape[-1] == 1:
        p = torch.sigmoid(logits)
        return torch.cat([1.0 - p, p], dim=-1)
    return torch.softmax(logits, dim=-1)


# =============================================================================
# Model Loading
# =============================================================================
def load_ensemble_models(config: Configuration, device: str) -> Tuple[List[Dict], Dict]:
    """Load top-5 ensemble models from best_models.json (Step 4 output).

    Returns:
        (models_list, best_model_info)
        models_list: list of dicts with keys 'model', 'cv', 'fold', 'hyperparams', 'metrics'
        best_model_info: the entry with the highest validation-partition MCC (never a test metric)
    """
    best_models_file = os.path.join(
        config.output_dir,
        f"{config.task_name}_{config.task_type}_final_eval",
        "best_models",
        "best_models.json",
    )

    with open(best_models_file, "r") as f:
        json_data = json.load(f)

    if "models" not in json_data:
        raise ValueError("best_models.json does not contain 'models' key")

    best_models_meta = json_data["models"]
    logger.info(f"Loading top {len(best_models_meta)} models from {best_models_file}")

    models_list = []
    for model_info in best_models_meta:
        cv_num = model_info["cv"]
        fold = model_info["fold"]
        metrics = model_info.get("metrics", {})
        mcc_val = metrics.get("mcc", metrics.get("matthews_corrcoef", None))
        auc_val = metrics.get("auc", None)
        metric_str = f"MCC: {mcc_val}, AUC: {auc_val}"

        checkpoint_path = os.path.join(
            config.output_dir,
            f"{config.task_name}_{config.task_type}_cv_results",
            f"cv{cv_num}",
            "checkpoints",
            f"{config.task_name}_{config.task_type}_cv{cv_num}_fold{fold}_best.ckpt",
        )

        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hyperparams = checkpoint["hyperparameters"]

        model = BaseGNN(
            config=config,
            rgcn_hidden_feats=hyperparams["rgcn_hidden_feats"],
            ffn_hidden_feats=hyperparams["ffn_hidden_feats"],
            ffn_dropout=hyperparams["ffn_dropout"],
            rgcn_dropout=hyperparams["rgcn_dropout"],
            classification=config.classification,
            num_classes=config.num_classes if config.classification else None,
        )

        # Filter incompatible keys (old bias terms)
        model_keys = set(model.state_dict().keys())
        filtered_state = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_keys and "graph_conv_layer.bias" not in k
        }
        model.load_state_dict(filtered_state, strict=False)
        model = model.to(device)
        model.eval()

        models_list.append({
            "model": model,
            "cv": cv_num,
            "fold": fold,
            "hyperparams": hyperparams,
            "metrics": metrics,
        })
        logger.info(f"  Loaded CV{cv_num} Fold{fold} ({metric_str})")

    if not models_list:
        raise ValueError("No models were successfully loaded")

    # Select the single attribution member by MCC on the VALIDATION partition. The 'matthews_corrcoef'
    # stored in best_models.json is a test-partition metric and must never be used for selection.
    # The validation MCC is read from best_models.json when present, otherwise from
    # full_results.json of the same final_eval run (keyed by CV repeat and fold). If neither is
    # available the run stops, so there is no silent fallback to a test metric.
    valid_lookup = {}
    full_path = os.path.join(
        config.output_dir, f"{config.task_name}_{config.task_type}_final_eval", "full_results.json")
    if os.path.exists(full_path):
        with open(full_path, "r") as fh:
            for r in json.load(fh).get("individual_results", []):
                if "valid_matthews_corrcoef" in r:
                    valid_lookup[(int(r["cv"]), int(r["fold"]))] = float(r["valid_matthews_corrcoef"])

    def _valid_mcc(m):
        met = m.get("metrics", {})
        v = met.get("valid_matthews_corrcoef")
        if v is None or v != v:  # missing or NaN
            v = valid_lookup.get((int(m["cv"]), int(m["fold"])))
        if v is None:
            raise ValueError(
                f"No validation MCC for CV{m['cv']} Fold{m['fold']}; refusing to select the attribution "
                f"member with a test-partition metric. Expected in best_models.json or {full_path}")
        return float(v)

    best_model_info = max(models_list, key=_valid_mcc)
    best_mcc = _valid_mcc(best_model_info)
    best_model_info["selection"] = {"criterion": "validation_matthews_corrcoef", "value": best_mcc,
                                    "candidates": {f"CV{m['cv']}_Fold{m['fold']}": _valid_mcc(m) for m in models_list}}
    logger.info(f"Successfully loaded {len(models_list)} models")
    logger.info(f"Best model for attribution (validation MCC): CV{best_model_info['cv']} "
                f"Fold{best_model_info['fold']} (valid MCC={best_mcc:.4f})")
    return models_list, best_model_info


# =============================================================================
# Ensemble Prediction (all models)
# =============================================================================
def get_ensemble_prediction(
    models: List[Dict], graph: Data, config: Configuration, device: str
) -> Optional[Dict]:
    """Run all ensemble models on one graph, return aggregated prediction.

    Returns dict with keys:
        mean_probs, std_probs, predicted_class, class_name, confidence,
        agreement_ratio, individual_predictions
    """
    predictions = []
    for minfo in models:
        pred = _predict_single(minfo["model"], graph, config, device)
        if pred is not None:
            predictions.append(pred)

    if not predictions:
        return None

    num_classes = config.num_classes
    pred_matrix = np.array(predictions)  # (n_models, num_classes)
    mean_probs = np.mean(pred_matrix, axis=0)
    std_probs = np.std(pred_matrix, axis=0)
    threshold = getattr(config, "decision_threshold", None)
    if pred_matrix.shape[1] == 2 and threshold is not None:
        # Binary: threshold tuned on the validation set in final_eval.py
        predicted_class = int(mean_probs[1] >= threshold)
        model_classes = (pred_matrix[:, 1] >= threshold).astype(int)
    else:
        predicted_class = int(np.argmax(mean_probs))
        model_classes = np.argmax(pred_matrix, axis=1)
    confidence = float(mean_probs[predicted_class])

    # Agreement: fraction of models predicting same class as ensemble
    agreement_ratio = float(np.mean(model_classes == predicted_class))

    class_names = getattr(config, "class_names", CLASS_NAMES)
    class_name = class_names[predicted_class] if class_names else f"class_{predicted_class}"

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "predicted_class": predicted_class,
        "class_name": class_name,
        "confidence": confidence,
        "agreement_ratio": agreement_ratio,
        "individual_predictions": pred_matrix.tolist(),
        "n_models": len(predictions),
    }


def _predict_single(model, graph: Data, config: Configuration, device: str) -> Optional[np.ndarray]:
    """Run a single model, return softmax probability vector (num_classes,)."""
    try:
        with torch.no_grad():
            graph_device = next(model.parameters()).device
            graph_copy = Data(
                x=graph.x.to(graph_device),
                edge_index=graph.edge_index.to(graph_device),
                edge_attr=graph.edge_attr.to(graph_device) if hasattr(graph, "edge_attr") and graph.edge_attr is not None else None,
                edge_type=graph.edge_type.to(graph_device) if hasattr(graph, "edge_type") and graph.edge_type is not None else None,
                batch=None,
            )
            if hasattr(graph, "smask"):
                graph_copy.smask = graph.smask.to(graph_device)

            model_batch = Batch.from_data_list([graph_copy])
            model.eval()
            pred, _ = model(model_batch)

            probs = _logits_to_probs(pred).squeeze(0).cpu().numpy()
            return probs
    except Exception as e:
        logger.debug(f"Single model prediction error: {e}")
        return None


# =============================================================================
# Scaffold Attribution (best model only)
# =============================================================================
def compute_scaffold_attributions(
    best_model, smiles: str, scaffold_atoms_dict: Dict[int, List[int]],
    config: Configuration, device: str, kinds: Optional[Dict[int, str]] = None
) -> Dict[int, Dict]:
    """Compute per-scaffold attribution using the BEST MODEL ONLY.

    For each scaffold, raw_diff = softmax(original) - softmax(masked).
    Then adaptive tanh scaling normalizes across scaffolds within each molecule,
    so cross-molecule aggregation (design rules) gives equal weight to every molecule.

    Args:
        best_model: single BaseGNN model
        smiles: SMILES string
        scaffold_atoms_dict: {scaffold_idx: [atom_indices]}
        config: Configuration
        device: torch device string

    Returns:
        {scaffold_idx: {
            'smiles': str,
            'atoms': list,
            'attribution': np.array(num_classes)
        }}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # Build original graph (no masking)
    original_graph = construct_mol_graph_from_smiles(smiles, smask=[])
    if original_graph is None:
        return {}

    # Get original prediction
    best_model.eval()
    with torch.no_grad():
        orig_batch = Batch.from_data_list([original_graph]).to(device)
        orig_logits, _ = best_model(orig_batch)
        orig_probs = _logits_to_probs(orig_logits).squeeze(0)  # (num_classes,)

    results = {}
    # Collect all masked graphs first
    scaffold_indices = []
    masked_graphs = []

    for sc_idx, atoms in scaffold_atoms_dict.items():
        atoms = list(map(int, atoms))
        # Get scaffold SMILES
        try:
            scf_smi = MolFragmentToSmiles(mol, atomsToUse=atoms, kekuleSmiles=False)
            if ":" in scf_smi or "*" in scf_smi:
                continue
        except Exception:
            continue

        masked_graph = construct_mol_graph_from_smiles(smiles, smask=atoms)
        if masked_graph is None:
            continue

        scaffold_indices.append((sc_idx, atoms, scf_smi))
        masked_graphs.append(masked_graph)

    if not masked_graphs:
        return {}

    # Process masked graphs in batches
    with torch.no_grad():
        all_masked_probs = []
        for i in range(0, len(masked_graphs), ATTRIBUTION_BATCH_SIZE):
            batch_graphs = masked_graphs[i : i + ATTRIBUTION_BATCH_SIZE]
            masked_batch = Batch.from_data_list([g.to(device) for g in batch_graphs])
            masked_logits, _ = best_model(masked_batch)
            masked_probs = _logits_to_probs(masked_logits).cpu().numpy()  # (batch, num_classes)
            all_masked_probs.append(masked_probs)

        all_masked_probs = np.concatenate(all_masked_probs, axis=0)  # (n_scaffolds, num_classes)
        orig_probs_np = orig_probs.cpu().numpy()

    # Collect raw diffs first
    raw_diffs = []
    for j in range(len(scaffold_indices)):
        raw_diffs.append(orig_probs_np - all_masked_probs[j])  # (num_classes,)

    # Adaptive tanh scaling — normalizes per-molecule so cross-molecule
    # aggregation (design rules) gives equal weight to every molecule.
    # alpha = 1/max_abs_diff maps the largest shift to tanh(1) ≈ 0.76,
    # then scaling_factor = 1/0.76 rescales so the peak ≈ 1.0.
    diff_matrix = np.abs(np.stack(raw_diffs))  # (n_scaffolds, num_classes)
    max_diff = diff_matrix.max() + 1e-6
    alpha = 1.0 / max_diff
    scaling_factor = 1.0 / 0.76

    for j, (sc_idx, atoms, scf_smi) in enumerate(scaffold_indices):
        attribution = np.tanh(alpha * raw_diffs[j]) * scaling_factor  # (num_classes,)
        results[sc_idx] = {
            "smiles": scf_smi,
            "atoms": atoms,
            "attribution": attribution,
            "kind": (kinds or {}).get(sc_idx, "ring_system"),
        }

    return results


def ensemble_attribution_agreement(
    models: List[Dict], smiles: str, scaffold_atoms_dict: Dict[int, List[int]],
    config: Configuration, device: str, class_idx: int, kinds: Optional[Dict[int, str]] = None
) -> Tuple[Optional[float], int]:
    """Mean pairwise Pearson r between the ensemble members' attribution vectors.

    Each member's attribution for `class_idx` over the same substructures forms
    one vector; agreement is the mean correlation over all member pairs. This is
    the same cross-ensemble measure used in the wine-level trust test, reported
    alongside the published best-model dual criterion. Needs >= 2 substructures;
    a member with a constant vector contributes r = 0 for its pairs.
    """
    per_model = []
    for minfo in models:
        attrs = compute_scaffold_attributions(minfo["model"], smiles, scaffold_atoms_dict, config, device, kinds=kinds)
        per_model.append({k: float(v["attribution"][class_idx]) for k, v in attrs.items()})
    keys = sorted(set.intersection(*(set(m) for m in per_model))) if per_model else []
    if len(per_model) < 2 or len(keys) < 2:
        return None, len(keys)
    mat = np.array([[m[k] for k in keys] for m in per_model])
    rs = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            if mat[i].std() < 1e-12 or mat[j].std() < 1e-12:
                rs.append(0.0)
            else:
                rs.append(float(np.corrcoef(mat[i], mat[j])[0, 1]))
    return float(np.mean(rs)), len(keys)


# =============================================================================
# XAI Agreement (dual criteria)
# =============================================================================
def check_xai_agreement_classification(
    scaffold_attributions: Dict[int, Dict], predicted_class: int
) -> Tuple[float, float, bool, str]:
    """Check whether scaffold attributions agree with the ensemble prediction.

    Dual criteria (either sufficient):
      Criterion 1 (ratio): >=70% of scaffolds have attr[predicted_class] > 0
      Criterion 2 (mean):  mean attr[predicted_class] across all scaffolds > 0

    Returns:
        (agreement_ratio, mean_attr, is_agreed, criterion_used)
        criterion_used in {'ratio', 'mean', 'both', 'none'}
    """
    if not scaffold_attributions:
        return 0.0, 0.0, False, "none"

    attrs_for_pred = []
    for sc_data in scaffold_attributions.values():
        attr = sc_data["attribution"]
        attrs_for_pred.append(float(attr[predicted_class]))

    n_total = len(attrs_for_pred)
    n_positive = sum(1 for a in attrs_for_pred if a > 0.0)
    agreement_ratio = n_positive / n_total
    mean_attr = float(np.mean(attrs_for_pred))

    crit_ratio = agreement_ratio >= AGREEMENT_THRESHOLD
    crit_mean = mean_attr > 0.0

    if crit_ratio and crit_mean:
        criterion_used = "both"
    elif crit_ratio:
        criterion_used = "ratio"
    elif crit_mean:
        criterion_used = "mean"
    else:
        criterion_used = "none"

    is_agreed = crit_ratio or crit_mean
    return agreement_ratio, mean_attr, is_agreed, criterion_used


# =============================================================================
# Scenario Classification (2x2 matrix)
# =============================================================================
def classify_scenario(
    ensemble_agreement_ratio: float, xai_is_agreed: bool
) -> Tuple[str, str]:
    """Classify into scenario A/B/C/D based on consensus x agreement.

    HIGH consensus = ensemble_agreement_ratio >= CONSENSUS_THRESHOLD
    HIGH reliability = xai_is_agreed is True
    """
    high_consensus = ensemble_agreement_ratio >= CONSENSUS_THRESHOLD

    if high_consensus and xai_is_agreed:
        return "A", "HIGH Consensus + HIGH Reliability: Trust prediction and explanation for SAR"
    elif high_consensus and not xai_is_agreed:
        return "B", "HIGH Consensus + LOW Reliability: Trust prediction, validate mechanism experimentally"
    elif not high_consensus and xai_is_agreed:
        return "C", "LOW Consensus + HIGH Reliability: Investigate prediction uncertainty, trust mechanism"
    else:
        return "D", "LOW Consensus + LOW Reliability: Full experimental validation required"


# =============================================================================
# Substituent Extraction (structural info only, no fake attribution)
# =============================================================================
def _extract_substituents(smiles: str, core_atoms: List[int]) -> List[Dict[str, str]]:
    """Extract substituent SMILES and aromatic/aliphatic context.
    No attribution values — just structural information."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    core_set = set(core_atoms)
    all_idx = set(range(mol.GetNumAtoms()))
    non_core = all_idx - core_set

    # Build adjacency
    nbrs = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}

    # Find attachment points
    attachment_points = {}
    for ca in core_set:
        for nb in nbrs.get(ca, []):
            if nb in non_core and nb not in attachment_points:
                attachment_points[nb] = ca

    visited = set()
    substituents = []

    for start in attachment_points:
        if start in visited:
            continue
        # BFS to collect fragment
        fragment = set()
        queue = [start]
        while queue:
            atom = queue.pop(0)
            if atom in fragment:
                continue
            fragment.add(atom)
            visited.add(atom)
            for nb in nbrs.get(atom, []):
                if nb in non_core and nb not in fragment:
                    queue.append(nb)

        attach_atom = attachment_points[start]
        is_aromatic = mol.GetAtomWithIdx(attach_atom).GetIsAromatic()
        context = "aromatic" if is_aromatic else "aliphatic"

        try:
            frag_smi = MolFragmentToSmiles(mol, atomsToUse=list(fragment), kekuleSmiles=True)
            # Skip suspicious ring fragments
            if frag_smi in ("CCC", "CC", "C", "c"):
                if any(mol.GetAtomWithIdx(idx).IsInRing() for idx in fragment):
                    continue
            substituents.append({"smiles": frag_smi, "context": context})
        except Exception:
            continue

    return substituents


# =============================================================================
# Murcko Scaffold Extraction (ring-combination approach)
# =============================================================================
def _get_murcko_scaffolds(smiles: str) -> Dict[int, List[int]]:
    """Extract Murcko scaffolds as dict of {idx: [atom_indices]}.
    Uses the ring-combination approach with MAX_SCAFFOLDS cap.

    For molecules with >MAX_RINGS_COMBINATORIAL rings, falls back to
    individual rings + full scaffold only (avoids 2^N combinatorial explosion
    that causes OOM on large natural products).
    """
    import itertools
    from collections import deque

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    n_rings = mol.GetRingInfo().NumRings()
    if n_rings == 0:
        return {}

    # Build adjacency
    graph = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}
    ring_sets = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if not ring_sets:
        return {}

    def shortest_path(src: set, dst: set):
        visited = set(src)
        queue = deque([[i] for i in src])
        while queue:
            path = queue.popleft()
            last = path[-1]
            if last in dst:
                return path
            for nb in graph.get(last, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(path + [nb])
        return []

    N = len(ring_sets)

    if N > MAX_RINGS_COMBINATORIAL:
        # Fallback for large molecules: individual rings + full scaffold only
        # This avoids 2^N combinatorial explosion (e.g. 2^15 = 32768 combos)
        logger.info(f"Large molecule ({N} rings) — using fallback scaffold extraction for {smiles[:60]}...")
        all_scaffolds = []

        # Each individual ring
        for ring in ring_sets:
            all_scaffolds.append(ring)

        # Full scaffold (all rings connected)
        full = set().union(*ring_sets)
        base = ring_sets[0]
        for i in range(1, N):
            full |= set(shortest_path(base, ring_sets[i]))
        all_scaffolds.append(full)

        # Pairwise ring combinations (N*(N-1)/2 — manageable even for N=30)
        for i, j in itertools.combinations(range(N), 2):
            atoms = ring_sets[i] | ring_sets[j]
            atoms |= set(shortest_path(ring_sets[i], ring_sets[j]))
            all_scaffolds.append(atoms)
    else:
        # Standard combinatorial approach (safe for N <= MAX_RINGS_COMBINATORIAL)
        all_scaffolds = []
        for k in range(1, N + 1):
            for combo in itertools.combinations(range(N), k):
                atoms = set().union(*(ring_sets[i] for i in combo))
                if len(combo) > 1:
                    base = ring_sets[combo[0]]
                    for idx in combo[1:]:
                        atoms |= set(shortest_path(base, ring_sets[idx]))
                all_scaffolds.append(atoms)

    # Deduplicate
    unique = []
    for s in all_scaffolds:
        if not any(s == u for u in unique):
            unique.append(s)

    scaffolds = {i: sorted(s) for i, s in enumerate(unique)}

    # Cap to MAX_SCAFFOLDS_PER_MOL largest
    if len(scaffolds) > MAX_SCAFFOLDS_PER_MOL:
        sorted_idxs = sorted(scaffolds.keys(), key=lambda i: len(scaffolds[i]), reverse=True)[
            :MAX_SCAFFOLDS_PER_MOL
        ]
        scaffolds = {i: scaffolds[i] for i in sorted_idxs}

    return scaffolds


def _get_substructures(smiles: str) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Substructures to mask: Murcko ring systems, each acyclic branch attached
    to a ring (e.g. OH, OCH3, ethyl, CHO on a benzene ring), and acyclic
    functional groups. Returns ({idx: atoms}, {idx: kind}).

    Ring-only masking cannot separate guaiacol from phenol or explain acyclic
    odorants; the substituent and functional-group entries cover both cases.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, {}
    ring_atoms = {a for ring in mol.GetRingInfo().AtomRings() for a in ring}

    entries = [(sorted(a), "ring_system") for a in _get_murcko_scaffolds(smiles).values()]

    # Ring substituents: connected components of non-ring atoms touching a ring
    seen = set()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in ring_atoms or idx in seen:
            continue
        comp, stack, touches_ring = set(), [idx], False
        while stack:
            a = stack.pop()
            if a in comp:
                continue
            comp.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j in ring_atoms:
                    touches_ring = True
                elif j not in comp:
                    stack.append(j)
        seen |= comp
        if touches_ring and ring_atoms:
            entries.append((sorted(comp), "ring_substituent"))

    # Acyclic functional groups
    for name, smarts in ODOUR_FG_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(patt):
            atoms = sorted(a for a in match if a not in ring_atoms)
            if atoms:
                entries.append((atoms, f"fg_{name}"))

    atoms_dict, kinds, unique = {}, {}, []
    for atoms, kind in entries:
        if atoms in unique or len(atoms) >= mol.GetNumAtoms():
            continue  # skip duplicates and whole-molecule masks
        unique.append(atoms)
        atoms_dict[len(atoms_dict)] = atoms
        kinds[len(kinds)] = kind
        if len(atoms_dict) >= MAX_SCAFFOLDS_PER_MOL:
            break
    return atoms_dict, kinds


# =============================================================================
# ComprehensiveTasteAnalyzer
# =============================================================================
class ComprehensiveTasteAnalyzer:
    """Orchestrates ensemble prediction, scaffold attribution, agreement checking,
    and scenario classification for each molecule in the dataset."""

    def __init__(
        self,
        models: List[Dict],
        best_model_info: Dict,
        config: Configuration,
        device: str,
        top_k: int = TOP_K_SCAFFOLDS,
    ):
        self.models = models
        self.best_model = best_model_info["model"]
        self.config = config
        self.device = device
        self.top_k = top_k
        self.should_terminate = False

        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("Termination signal received. Will exit after current molecule.")
        self.should_terminate = True

    # ------------------------------------------------------------------
    # Single molecule analysis
    # ------------------------------------------------------------------
    def analyze_molecule(
        self, smiles: str, compound_id: str, target_label: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full analysis pipeline for one molecule.

        Steps:
        1. Ensemble prediction (all models)
        2. Murcko scaffold decomposition
        3. Scaffold attribution (best model only)
        4. Substituent extraction (structural info only)
        5. Dual-criteria agreement check
        6. Scenario classification
        """
        result: Dict[str, Any] = {
            "COMPOUND_ID": compound_id,
            "SMILES": smiles,
            "TARGET": target_label,
            "group": group,
        }

        # --- 1. Ensemble prediction ---
        graph = construct_mol_graph_from_smiles(smiles, smask=[])
        if graph is None:
            result["error"] = "Failed to construct graph"
            return result

        ens = get_ensemble_prediction(self.models, graph, self.config, self.device)
        if ens is None:
            result["error"] = "No valid ensemble predictions"
            return result

        predicted_class = ens["predicted_class"]
        result["predicted_class"] = predicted_class
        result["predicted_class_name"] = ens["class_name"]
        result["confidence"] = ens["confidence"]

        class_names = getattr(self.config, "class_names", CLASS_NAMES)
        for i, cn in enumerate(class_names):
            result[f"prob_{cn}"] = float(ens["mean_probs"][i])
        result["prediction_std"] = float(np.max(ens["std_probs"]))
        result["n_models"] = ens["n_models"]
        result["ensemble_agreement_ratio"] = ens["agreement_ratio"]
        result["ensemble_consensus"] = "HIGH" if ens["agreement_ratio"] >= CONSENSUS_THRESHOLD else "LOW"
        result["individual_predictions"] = str(ens["individual_predictions"])

        # --- 2-6. Scaffold XAI (wrapped for robustness against OOM / problematic molecules) ---
        def _no_xai_result(reason: str):
            """Populate result with no-XAI defaults when scaffold analysis fails or is skipped."""
            result["n_scaffolds"] = 0
            result["xai_agreement_ratio"] = None
            result["xai_mean_attr"] = None
            result["xai_is_agreed"] = None
            result["xai_criterion_used"] = "none"
            result["xai_skip_reason"] = reason
            result["xai_ensemble_agreement"] = None
            result["xai_n_substructures_common"] = 0
            result["scenario"] = classify_scenario(ens["agreement_ratio"], False)[0]
            result["scenario_description"] = classify_scenario(ens["agreement_ratio"], False)[1]
            result["scaffold_attributions"] = {}
            result["substituent_info"] = {}

        try:
            # --- 2. Murcko scaffold decomposition ---
            scaffolds, kinds = _get_substructures(smiles)
            n_scaffolds = len(scaffolds)
            result["n_scaffolds"] = n_scaffolds

            # If no scaffolds (no rings), skip XAI
            if n_scaffolds == 0:
                _no_xai_result("No maskable substructures")
                if target_label is not None:
                    result["correct"] = bool(predicted_class == target_label)
                return result

            # --- 3. Scaffold attribution (best model only) ---
            scaffold_attrs = compute_scaffold_attributions(
                self.best_model, smiles, scaffolds, self.config, self.device, kinds=kinds
            )

            # --- 4. Substituent extraction (no fake attribution) ---
            substituent_info = {}
            for sc_idx, sc_data in scaffold_attrs.items():
                subs = _extract_substituents(smiles, sc_data["atoms"])
                substituent_info[sc_idx] = subs

            # --- 5. Dual-criteria agreement check ---
            agr_ratio, mean_attr, is_agreed, crit = check_xai_agreement_classification(
                scaffold_attrs, predicted_class
            )
            result["xai_agreement_ratio"] = agr_ratio
            result["xai_mean_attr"] = mean_attr
            result["xai_is_agreed"] = is_agreed
            result["xai_criterion_used"] = crit

            # --- 5b. Cross-ensemble attribution agreement (all 5 models) ---
            ens_agr, n_common = ensemble_attribution_agreement(
                self.models, smiles, scaffolds, self.config, self.device, predicted_class, kinds=kinds
            )
            result["xai_ensemble_agreement"] = ens_agr
            result["xai_n_substructures_common"] = n_common

            # --- 6. Scenario classification ---
            scenario, desc = classify_scenario(ens["agreement_ratio"], is_agreed)
            result["scenario"] = scenario
            result["scenario_description"] = desc

            # Store attributions and substituents for later CSV writing
            result["scaffold_attributions"] = scaffold_attrs
            result["substituent_info"] = substituent_info

        except (MemoryError, RuntimeError) as e:
            # Catch OOM from both Python (MemoryError) and CUDA (RuntimeError: CUDA out of memory)
            logger.warning(f"Memory error during XAI for {compound_id} ({smiles[:60]}...): {e}")
            _no_xai_result(f"MemoryError: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception as e:
            logger.warning(f"XAI failed for {compound_id} ({smiles[:60]}...): {e}")
            _no_xai_result(f"Error: {e}")

        # Correctness
        if target_label is not None:
            result["correct"] = bool(predicted_class == target_label)

        return result

    # ------------------------------------------------------------------
    # Dataset processing
    # ------------------------------------------------------------------
    def analyze_dataset(
        self,
        input_csv: str,
        output_dir: str,
        batch_size: int = 32,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        resume: bool = False,
    ) -> List[Dict]:
        """Process all molecules from input CSV."""
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")

        # Read input
        if input_csv.lower().endswith(".parquet"):
            df = pd.read_parquet(input_csv)
        else:
            df = pd.read_csv(input_csv)
        df.columns = df.columns.str.strip()

        # Handle SMILES column name
        if "SMILES" not in df.columns and "PROCESSED_SMILES" in df.columns:
            df.rename(columns={"PROCESSED_SMILES": "SMILES"}, inplace=True)
            logger.info("Renamed 'PROCESSED_SMILES' -> 'SMILES'")

        required = ["SMILES", "COMPOUND_ID"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing: {missing}")

        df = df.dropna(subset=["SMILES", "COMPOUND_ID"])
        total = len(df)

        # Slice
        if end_idx is None or end_idx >= total:
            end_idx = total - 1
        start_idx = max(0, start_idx)
        df = df.iloc[start_idx : end_idx + 1].reset_index(drop=True)

        has_target = self.config.labels_name in df.columns
        has_group = "group" in df.columns

        logger.info(f"Processing {len(df)} molecules (indices {start_idx}-{end_idx} of {total})")
        logger.info(f"Target labels available: {has_target}")

        # Resume from checkpoint
        all_results = []
        processed_ids = set()
        if resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "rb") as f:
                    ckpt = pickle.load(f)
                all_results = ckpt.get("results", [])
                processed_ids = {r["COMPOUND_ID"] for r in all_results if "COMPOUND_ID" in r}
                logger.info(f"Resumed checkpoint with {len(all_results)} molecules already processed")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}. Starting fresh.")
                all_results = []
                processed_ids = set()

        global_start = time.time()
        last_save = global_start
        n_processed = len(all_results)
        n_total = len(df)
        n_errors = 0

        for i in range(n_total):
            if self.should_terminate:
                logger.info("Termination requested. Saving checkpoint...")
                break

            row = df.iloc[i]
            compound_id = str(row["COMPOUND_ID"])
            smiles = str(row["SMILES"])

            # Skip already processed
            if compound_id in processed_ids:
                continue

            target = int(row[self.config.labels_name]) if has_target and pd.notna(row.get(self.config.labels_name)) else None
            group = str(row["group"]) if has_group and pd.notna(row.get("group")) else None

            try:
                result = self.analyze_molecule(smiles, compound_id, target_label=target, group=group)
                all_results.append(result)
                n_processed += 1
            except Exception as e:
                logger.error(f"Error processing {compound_id} ({smiles}): {e}")
                logger.debug(traceback.format_exc())
                all_results.append({
                    "COMPOUND_ID": compound_id,
                    "SMILES": smiles,
                    "error": str(e),
                })
                n_errors += 1
                n_processed += 1

            # Progress logging
            if n_processed % batch_size == 0 or n_processed == n_total:
                elapsed = time.time() - global_start
                rate = n_processed / elapsed if elapsed > 0 else 0
                remaining = (n_total - n_processed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {n_processed}/{n_total} ({n_processed / n_total * 100:.1f}%) "
                    f"- {rate:.2f} mol/s - ETA: {_format_time(remaining)} - Errors: {n_errors}"
                )

            # Checkpoint
            now = time.time()
            if (now - last_save) >= CHECKPOINT_INTERVAL_SEC:
                _save_checkpoint(checkpoint_path, all_results, input_csv, n_total, start_idx, end_idx)
                last_save = now

            # Memory cleanup
            if n_processed % (batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final checkpoint
        _save_checkpoint(checkpoint_path, all_results, input_csv, n_total, start_idx, end_idx)

        # Save all outputs
        self._save_results(all_results, output_dir)

        total_time = time.time() - global_start
        logger.info("=" * 60)
        logger.info(f"Processing complete: {n_processed} molecules in {_format_time(total_time)}")
        logger.info(f"Errors: {n_errors}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("=" * 60)

        return all_results

    # ------------------------------------------------------------------
    # Save results to 4 output files
    # ------------------------------------------------------------------
    def _save_results(self, results: List[Dict], output_dir: str):
        """Produce the 4 output files: predictions CSV, attributions CSV, summary JSON, design rules JSON."""
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]

        logger.info(f"Saving results: {len(valid_results)} valid, {len(error_results)} errors")

        # --- 1. Predictions CSV ---
        pred_rows = []
        for r in results:
            row = {
                "COMPOUND_ID": r.get("COMPOUND_ID"),
                "SMILES": r.get("SMILES"),
                "TARGET": r.get("TARGET"),
                "group": r.get("group"),
                "predicted_class": r.get("predicted_class"),
                "predicted_class_name": r.get("predicted_class_name"),
                "confidence": r.get("confidence"),
            }
            class_names = getattr(self.config, "class_names", CLASS_NAMES)
            for cn in class_names:
                row[f"prob_{cn}"] = r.get(f"prob_{cn}")
            row["prediction_std"] = r.get("prediction_std")
            row["n_models"] = r.get("n_models")
            row["ensemble_agreement_ratio"] = r.get("ensemble_agreement_ratio")
            row["ensemble_consensus"] = r.get("ensemble_consensus")
            row["xai_agreement_ratio"] = r.get("xai_agreement_ratio")
            row["xai_mean_attr"] = r.get("xai_mean_attr")
            row["xai_is_agreed"] = r.get("xai_is_agreed")
            row["xai_criterion_used"] = r.get("xai_criterion_used")
            row["xai_ensemble_agreement"] = r.get("xai_ensemble_agreement")
            row["xai_n_substructures_common"] = r.get("xai_n_substructures_common")
            row["scenario"] = r.get("scenario")
            row["scenario_description"] = r.get("scenario_description")
            row["n_scaffolds"] = r.get("n_scaffolds")
            row["correct"] = r.get("correct")
            row["error"] = r.get("error")
            pred_rows.append(row)

        pred_df = pd.DataFrame(pred_rows)
        pred_path = os.path.join(output_dir, "smoke_odour_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions: {pred_path} ({len(pred_df)} rows, {len(pred_df.columns)} cols)")

        # --- 2. Attributions CSV ---
        attr_rows = []
        for r in valid_results:
            compound_id = r["COMPOUND_ID"]
            predicted_class = r.get("predicted_class")
            scaffold_attrs = r.get("scaffold_attributions", {})
            substituent_info = r.get("substituent_info", {})
            class_names = getattr(self.config, "class_names", CLASS_NAMES)

            for sc_idx, sc_data in scaffold_attrs.items():
                attribution = sc_data["attribution"]
                promotes_idx = int(np.argmax(attribution))
                promotes_class = class_names[promotes_idx] if class_names else f"class_{promotes_idx}"
                is_consistent = bool(attribution[predicted_class] > 0.0) if predicted_class is not None else None

                # Substituent info for this scaffold
                subs = substituent_info.get(sc_idx, [])
                sub_smiles = ",".join(s["smiles"] for s in subs) if subs else ""
                sub_contexts = ",".join(s["context"] for s in subs) if subs else ""

                row = {
                    "COMPOUND_ID": compound_id,
                    "scaffold_index": sc_idx,
                    "scaffold_smiles": sc_data["smiles"],
                    "substructure_kind": sc_data.get("kind", "ring_system"),
                }
                for ci, cn in enumerate(class_names):
                    row[f"{cn}_attribution"] = float(attribution[ci])
                row["promotes_class"] = promotes_class
                row["is_consistent"] = is_consistent
                row["substituent_smiles"] = sub_smiles
                row["substituent_contexts"] = sub_contexts
                attr_rows.append(row)

        if attr_rows:
            attr_df = pd.DataFrame(attr_rows)
            attr_path = os.path.join(output_dir, "smoke_odour_attributions.csv")
            attr_df.to_csv(attr_path, index=False)
            logger.info(f"Saved attributions: {attr_path} ({len(attr_df)} rows)")
        else:
            logger.warning("No attribution rows to save")

        # --- 3. Summary JSON ---
        summary = self._build_summary(valid_results, error_results)
        summary_path = os.path.join(output_dir, "smoke_odour_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary: {summary_path}")

        # --- 4. Design Rules JSON ---
        design_rules = aggregate_design_rules(valid_results, getattr(self.config, "class_names", CLASS_NAMES), self.top_k)
        rules_path = os.path.join(output_dir, "smoke_odour_design_rules.json")
        with open(rules_path, "w") as f:
            json.dump(design_rules, f, indent=2, default=str)
        logger.info(f"Saved design rules: {rules_path}")

        # Print scenario distribution
        scenario_counts = defaultdict(int)
        for r in valid_results:
            scenario_counts[r.get("scenario", "unknown")] += 1
        logger.info("Scenario distribution:")
        for s in ["A", "B", "C", "D"]:
            cnt = scenario_counts.get(s, 0)
            pct = cnt / len(valid_results) * 100 if valid_results else 0
            logger.info(f"  Scenario {s}: {cnt} ({pct:.1f}%)")

    def _build_summary(self, valid_results: List[Dict], error_results: List[Dict]) -> Dict:
        """Build summary statistics JSON."""
        n_valid = len(valid_results)
        if n_valid == 0:
            return {"n_molecules": 0, "error": "No valid results"}

        # Scenario distribution
        scenario_dist = defaultdict(int)
        for r in valid_results:
            scenario_dist[r.get("scenario", "unknown")] += 1

        # Criterion distribution
        crit_dist = defaultdict(int)
        for r in valid_results:
            crit_dist[r.get("xai_criterion_used", "unknown")] += 1

        # Class distribution
        class_dist = defaultdict(int)
        for r in valid_results:
            class_dist[r.get("predicted_class_name", "unknown")] += 1

        # Per-class scenario A percentage
        class_names = getattr(self.config, "class_names", CLASS_NAMES)
        per_class_a = {}
        for cn in class_names:
            cls_results = [r for r in valid_results if r.get("predicted_class_name") == cn]
            if cls_results:
                n_a = sum(1 for r in cls_results if r.get("scenario") == "A")
                per_class_a[cn] = round(n_a / len(cls_results) * 100, 1)
            else:
                per_class_a[cn] = 0.0

        # Accuracy by scenario (if target labels available)
        accuracy_by_scenario = {}
        for s in ["A", "B", "C", "D"]:
            s_results = [r for r in valid_results if r.get("scenario") == s and r.get("correct") is not None]
            if s_results:
                accuracy_by_scenario[s] = round(
                    sum(1 for r in s_results if r["correct"]) / len(s_results), 4
                )
            else:
                accuracy_by_scenario[s] = None

        scenario_a_pct = round(scenario_dist.get("A", 0) / n_valid * 100, 1) if n_valid > 0 else 0

        return {
            "n_molecules": n_valid + len(error_results),
            "n_valid": n_valid,
            "n_errors": len(error_results),
            "n_models": valid_results[0].get("n_models", 0) if valid_results else 0,
            "xai_model": "best_model_only",
            "consensus_threshold": CONSENSUS_THRESHOLD,
            "agreement_threshold": AGREEMENT_THRESHOLD,
            "scenario_distribution": dict(scenario_dist),
            "scenario_A_pct": scenario_a_pct,
            "per_class_scenario_A": per_class_a,
            "accuracy_by_scenario": accuracy_by_scenario,
            "criterion_distribution": dict(crit_dist),
            "class_distribution": dict(class_dist),
        }


# =============================================================================
# Design Rules Aggregation
# =============================================================================
def aggregate_design_rules(
    results: List[Dict], class_names: List[str], top_k: int = TOP_K_SCAFFOLDS
) -> Dict:
    """Aggregate scaffold attributions across all molecules for SAR insights.

    Groups scaffolds by SMILES, computes:
      - mean attribution per class
      - occurrence count
      - scenario A percentage (trustworthiness)
    """
    scaffold_stats: Dict[str, Dict] = defaultdict(lambda: {
        "count": 0,
        "attr_sums": np.zeros(len(class_names)),
        "scenario_a_count": 0,
    })

    for r in results:
        scaffold_attrs = r.get("scaffold_attributions", {})
        scenario = r.get("scenario")

        for sc_data in scaffold_attrs.values():
            smi = sc_data["smiles"]
            attr = sc_data["attribution"]
            stats = scaffold_stats[smi]
            stats["count"] += 1
            stats["attr_sums"] += attr
            if scenario == "A":
                stats["scenario_a_count"] += 1

    if not scaffold_stats:
        empty = {f"{cn.lower()}_promoting_scaffolds": [] for cn in class_names}
        empty["most_trustworthy_scaffolds"] = []
        return empty

    # Build per-scaffold summary
    scaffold_summaries = []
    for smi, stats in scaffold_stats.items():
        count = stats["count"]
        mean_attr = stats["attr_sums"] / count
        scenario_a_pct = round(stats["scenario_a_count"] / count * 100, 1) if count > 0 else 0.0
        dominant_idx = int(np.argmax(mean_attr))
        dominant_class = class_names[dominant_idx]

        entry = {
            "smiles": smi,
            "count": count,
            "scenario_A_pct": scenario_a_pct,
            "dominant_class": dominant_class,
        }
        for ci, cn in enumerate(class_names):
            entry[f"mean_attr_{cn}"] = round(float(mean_attr[ci]), 6)

        scaffold_summaries.append(entry)

    # Sort for each category
    result = {}
    for ci, cn in enumerate(class_names):
        key = f"{cn.lower()}_promoting_scaffolds"
        sorted_list = sorted(
            scaffold_summaries,
            key=lambda x: x.get(f"mean_attr_{cn}", 0),
            reverse=True,
        )[:top_k]
        result[key] = [
            {
                "smiles": s["smiles"],
                "mean_attr": s[f"mean_attr_{cn}"],
                "count": s["count"],
                "scenario_A_pct": s["scenario_A_pct"],
            }
            for s in sorted_list
        ]

    # Most trustworthy scaffolds (highest Scenario A %)
    # Require minimum count to avoid noise
    min_count = max(5, len(results) // 100)  # At least 1% occurrence or 5
    trustworthy = [s for s in scaffold_summaries if s["count"] >= min_count]
    trustworthy.sort(key=lambda x: x["scenario_A_pct"], reverse=True)
    result["most_trustworthy_scaffolds"] = [
        {
            "smiles": s["smiles"],
            "scenario_A_pct": s["scenario_A_pct"],
            "dominant_class": s["dominant_class"],
            "count": s["count"],
        }
        for s in trustworthy[:top_k]
    ]

    return result


# =============================================================================
# Utilities
# =============================================================================
def _save_checkpoint(path: str, results: List[Dict], input_csv: str,
                     total: int, start_idx: int, end_idx: int):
    """Save checkpoint pickle (strip heavy data first)."""
    # Strip scaffold_attributions and substituent_info to keep checkpoint small
    light_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k not in ("scaffold_attributions", "substituent_info")}
        light_results.append(r_copy)

    state = {
        "results": results,  # Keep full for resume
        "metadata": {
            "input_csv": input_csv,
            "timestamp": datetime.now().isoformat(),
            "total_molecules": total,
            "start_idx": start_idx,
            "end_idx": end_idx,
        },
    }
    try:
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved checkpoint ({len(results)} results) to {path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


# =============================================================================
# Main
# =============================================================================
def main():
    config = Configuration()

    parser = argparse.ArgumentParser(
        description="Comprehensive XAI prediction pipeline (smoke-related odour, binary)."
    )
    default_input = config.origin_data_path
    parser.add_argument("--input_csv", type=str, default=default_input, help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32, help="Progress logging interval")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers (unused, kept for compat)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Device")
    parser.add_argument("--ensemble", type=str, choices=["best5"], default="best5", help="Ensemble type")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (inclusive)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--top_k", type=int, default=TOP_K_SCAFFOLDS, help="Top K scaffolds in design rules")
    args = parser.parse_args()

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    if args.device == "cuda":
        try:
            if torch.cuda.device_count() == 0:
                raise RuntimeError("No CUDA devices")
        except Exception as e:
            logger.warning(f"CUDA not usable ({e}); falling back to CPU")
            args.device = "cpu"

    logger.info("=" * 60)
    logger.info("COMPREHENSIVE SMOKE-ODOUR XAI PIPELINE")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # Read data to set num_classes
    if args.input_csv.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input_csv)
    else:
        df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip()

    if config.classification:
        # The class count is fixed by the trained model, not inferred from the
        # input file: an input holding only positives (e.g. the held-out smoke
        # markers) would otherwise be read as a one-class task.
        config.num_classes = len(config.class_names)
        logger.info(f"num_classes={config.num_classes} from class_names {config.class_names}")
        threshold_path = os.path.join(
            config.output_dir, f"{config.task_name}_{config.task_type}_final_eval", "decision_threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                config.decision_threshold = float(json.load(f)["threshold"])
            logger.info(f"Decision threshold (validation-tuned): {config.decision_threshold:.4f}")
        else:
            config.decision_threshold = 0.5
            logger.warning(f"{threshold_path} missing; using 0.5")

    config.prediction_mode = True

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_csv), "smoke_odour_xai")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Input:  {args.input_csv}")
    logger.info(f"Output: {args.output_dir}")

    # Load models
    try:
        models, best_model_info = load_ensemble_models(config, args.device)
    except Exception as e:
        if args.device == "cuda":
            logger.warning(f"Model loading failed on CUDA: {e}. Retrying on CPU...")
            args.device = "cpu"
            models, best_model_info = load_ensemble_models(config, args.device)
        else:
            raise

    # Run analysis
    analyzer = ComprehensiveTasteAnalyzer(models, best_model_info, config, args.device, top_k=args.top_k)
    analyzer.analyze_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resume=args.resume,
    )

    logger.info("Comprehensive XAI pipeline completed successfully.")


if __name__ == "__main__":
    main()
