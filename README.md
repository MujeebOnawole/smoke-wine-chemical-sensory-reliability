# From chemical markers to sensory perception

**How far can a chemical marker be trusted to stand in for what people taste, and can a model say when it cannot?**

Smoke exposure of grapes is routinely assessed by measuring volatile phenols, with guaiacol as the
headline marker, and reading the concentration as a proxy for smoke flavour in the finished wine. This
repository holds the code and result tables behind a study that treats that step as a reliability
problem. On 63 smoke-affected Australian wines with published chemistry and trained-panel ratings, it
asks three questions in turn.

1. **How often does the marker disagree with the panel?** Guaiacol alone predicts panel smoke flavour
   with a mean absolute error of 1.25 rating units under cultivar-stratified cross-validation, and 26 of
   the 63 wines miss by 1.5 units or more in at least one fold. The discordant wines are not random. They
   carry more guaiacol and a lower ratio of other volatile phenols to guaiacol than concordant wines.
2. **Does a multivariate model fix it, and what is the fair comparison?** A five-regressor ensemble on
   the full phenol panel brings the error on guaiacol-discordant predictions from 2.21 down to 0.76
   units. But with wine chemistry available it does not beat Lasso alone or the single best analyte
   chosen on the training data (4-methylguaiacol), so the gain is in the analyte panel, not the
   learner. With grape chemistry only, the ensemble is the strongest option.
3. **Can the prediction carry its own reliability label?** Ensemble consensus (the spread of the five
   members) ranks error better than random in four of six evaluation settings, matches a simple
   distance-to-training-set rule, and beats attribution disagreement, which is worse than random here.
   The most useful screening signal turns out to be the disagreement between the marker regression and
   any accurate alternative. That gap flags guaiacol-discordant predictions with AUROC 0.85 to 0.96 when
   wine chemistry is available, and Lasso does this as well as the ensemble.

The wines are the case study. The reliability questions, and the finding that an explanation criterion
borrowed from one domain can fail in another, apply to any pipeline that maps a chemical measurement to
a sensory or biological judgement. The reliability framework (consensus plus explanation agreement,
Scenarios A to D) was published for molecular taste prediction (Onawole et al., 2026) and is applied
here, not proposed.

## What the evaluations established

All numbers are out-of-sample. Held-out evaluations include cultivar-stratified 5 x 5 cross-validation,
whole vineyards and whole regions held out, leave-one-cultivar-out, and transfer in both directions
between the two source studies.

| | |
|---|---|
| Guaiacol alone, cultivar-stratified CV | MAE 1.25, R² 0.35 |
| Guaiacol alone, leave-one-cultivar-out | MAE 1.85, R² below zero |
| Wines guaiacol-discordant in at least one CV fold | 26 of 63 (35 of 63 under cultivar hold-out) |
| Ensemble vs guaiacol on discordant predictions (CV) | MAE 0.76 vs 2.21 |
| Ensemble vs best single analyte, wine chemistry (CV) | MAE 0.62 vs 0.66, not significant |
| Ensemble vs best single analyte, grape chemistry (CV) | MAE 0.74 vs 1.14 |
| Marker-vs-alternative gap flags discordance (wine chemistry) | AUROC 0.85 (CV) to 0.96 (cultivar hold-out) |
| Smoky vs not-smoky triage at the training midpoint | accuracy 0.92 (CV), 0.94 (cultivar hold-out), 0.90 and 0.83 (study transfer) |

Two further results shape how the numbers should be read.

**Abstention has a cost that accuracy curves hide.** Withholding the 20 to 30 percent least-consensual
predictions removes smoky wines from the answered set faster than it removes errors, so a consensus
gate tuned for error reduction quietly under-reports the positive class.

**Explanation agreement did not transfer.** Agreement between the attribution vectors of the ensemble
members is a useful reliability axis for molecular graphs, where each prediction has its own structure
to explain. For 63 wines sharing the same dozen analytes it ranks error worse than random. The
molecular half of the study makes the complementary point. A graph neural network trained to recognise
smoke-related odorants (6,538 molecules, 527 positives, external test on the 10 wine smoke-taint
phenols) reaches a held-out AUROC of 0.87 on a random split and 0.79 on a scaffold split, and its
Scenario A explanations are more sign-consistent than Scenario B ones by about 11 points. Yet when the
model is trained on the combined smoky-or-phenolic label it learns a phenol shortcut, and the
attribution on the phenolic OH no longer discriminates smoke-related from other phenols (AUROC 0.50
against 0.96 for the probability itself). An explanation can be internally consistent and still not
mean what the reader assumes.

## Repository layout

```
wine/           wine-level analyses (the primary study): ensemble, baselines, discordance,
                uncertainty rankings, triage, stage comparison, figures
molecular/      RGCN smoke-odour classifier with substructure-masking attribution and the
                Scenario A to D framework (flat modules, run from inside the folder)
data_prep/      scripts that built the curated inputs from the source publications and databases
data/           curated model inputs (wine table, odorant splits, external markers)
results/        reference outputs of every stage, so each table in the paper can be traced,
                plus data_checks/ for counts that support preprocessing decisions
```

### Wine level, `wine/`

| script | what it does |
|---|---|
| `wine_trust_test.py` | five tuned regressors (Lasso, PLS, SVR, RF, GBR), nested CV and study transfer, consensus and attribution-agreement scenarios |
| `wine_trust_grouped.py` | the same with whole vineyards or regions held out |
| `wine_assessment.py` | the assessment the paper rests on: fair baselines on identical splits, discordance against a guaiacol and a phenol-index reference, seven uncertainty rankings with risk-coverage curves, triage on the source studies' significance labels, grape versus wine chemistry, wine-cluster bootstrap intervals |
| `discordance_chemistry.py` | chemistry of discordant wines and the prospective marker-versus-ensemble gap (reads the pulled predictions, no new fits) |
| `gap_controls.py` | the gap built from Lasso, the best single analyte and the phenol index as controls for the ensemble gap |
| `wine_oav_check.py` | attribution ranking against odour activity values |
| `figures.py` | main-text wine figures and Tables 1 to 3 from `results/` |

Every evaluation setting, model, baseline and ranking is defined in the module docstring of
`wine_assessment.py`, and those definitions were fixed before the full run.

### Molecular level, `molecular/`

```bash
cd molecular
export SPLIT_TYPE=random          # or scaffold
export LABEL_VARIANT=combined     # or smoky, phenolic (label ablations on the random split)
python prep_data.py               # molecular graphs and substructure masks
python hyper.py                   # Optuna search, 20 trials
python stat_val.py                # 5 repeats x 5 folds, 25 models
python final_eval.py              # all models on test, validation-tuned threshold, 5-model ensemble
python predict.py --input_csv ../data/smoke_odour_markers_external.csv --output_dir ../data/<tag>_xai_markers
python predict.py --input_csv ../data/<tag>_split.csv --output_dir ../data/<tag>_xai_full
python smoke_report.py --data_dir ../data --split random --variant combined
python explanation_eval.py --data_dir ../data --tag smoke_odour_random
```

`<tag>` is `smoke_odour_<split>` with `_smoky` or `_phenolic` appended for the label ablations. Run
outputs are written next to the inputs in `data/`, as they were on the cluster, and are not tracked.
`oh_ome_check.py` is the matched-pair test of whether attribution recovers the loss of smoke character
when a phenolic OH is capped as a methyl ether. `twin_check.py` reports test performance with and
without near-duplicate training molecules. Training ran on GPU nodes of an HPC cluster; scheduler
scripts are site-specific and are not included. Trained checkpoints (100 models, 147 MB) are available
from the author on request.

### Data preparation, `data_prep/`

These scripts document provenance. They ran in the authors' working tree against the raw downloads
and the publishers' supporting-information files, and their input paths reflect that tree.
`extract_awri_wine_tables.py` reads the per-wine chemistry and panel ratings from the two source
publications, `add_wine_groups.py` and `add_wine_significance.py` add the hold-out groups and the
studies' own smoky-versus-control significance calls, `consolidate_flavor_datasets.py` and
`prepare_smoke_taint_data.py` merge the odorant databases and define the smoke-related label, and
`make_smoke_odour_splits.py` writes the random and scaffold partitions. `flavordb_sweet_tag_check.py` recomputes, from the raw Pyrfume FlavorDB behaviour table, the counts behind one
preprocessing decision. FlavorDB carries the flavour percept "sweet" on 22,596 of its 25,595 compounds, and
21,899 of those have no other flavour or odour descriptor, so the tag behaves as a default rather than an
annotation. The flavour percept "sweet" was therefore dropped before consolidation, the odour percept "sweet"
(73 compounds) was kept, and compounds left without any descriptor were removed. The counts are in
`results/data_checks/flavordb_sweet_tag_counts.csv`.

## Data

`data/` carries the curated inputs.

- `awri_wine_smoke_wide_labelled.csv`: 63 wines (Shiraz 24, Chardonnay 20, Pinot Noir 19) from two
  Australian Wine Research Institute studies, with grape and wine volatile phenol and glycoside
  concentrations, panel smoke aroma and flavour ratings, vineyard and region groups, and the binary
  significance label (20 smoky). Values were transcribed from the published tables and supporting
  information of the two studies (Jiang et al., 2022; Parker et al., 2023); please cite those papers
  when using the wine table.
- `smoke_odour_*_split.csv`: 6,538 odorants with SMILES, the smoke-related label (527 positives) and the
  partition. The two ablation files relabel the same random partition as smoky-only (360) or
  phenolic-only (227). Odour descriptors were consolidated from public flavour and odour databases (GoodScents,
  Leffingwell, FlavorNet, FlavorDB and FooDB); the merged intermediate tables are not redistributed.
- `smoke_odour_markers_external.csv`: the 10 wine smoke-taint phenols removed from all splits and used
  as an external test.

## Results

`results/wine/wine_assessment/` holds the per-prediction table every wine number derives from
(`per_wine_predictions.csv`, 63 wines x settings x feature sets x repeats) together with the baseline
comparison, discordance tests, risk-coverage curves and triage tables. `results/wine/wine_trust/` and
`wine_trust_grouped/` are the earlier scenario-based runs. `results/molecular/` holds, for each of the
four RGCN runs, the final evaluation, the report tables (scenario coverage by partition, marker
predictions, whole-dataset substructure attribution), the full-dataset predictions and substructure
attributions, and the explanation evaluation. `results/tables/` are the main-text tables as CSV.

## Installation

```bash
git clone https://github.com/MujeebOnawole/smoke-wine-chemical-sensory-reliability.git
cd smoke-wine-chemical-sensory-reliability
pip install -r requirements.txt
```

The wine analyses need only pandas, numpy, scikit-learn, scipy, statsmodels and matplotlib. The
molecular pipeline additionally needs RDKit, PyTorch, PyTorch Geometric and Optuna; install torch and
torch_geometric as a matched pair for your CUDA version first.

## Citation

A manuscript describing this work is in preparation. Please check back for a citation.

## Licence

MIT. See `LICENSE`.
