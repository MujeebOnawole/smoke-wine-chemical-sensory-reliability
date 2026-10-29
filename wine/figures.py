"""Round 6 (2026-10-26) wine-first figures and tables for the manuscript revamp.

Inputs: bunya_smoke_odour/Results/wine_assessment/ (job 30627082, corrected guaiacol baseline) and the labelled
wine table. Outputs: figures in LOF/figures_round6/, tables (csv + markdown) in LOT/tables_round6/.
Style follows build_figures.py (Okabe-Ito palette, 300 dpi, white background). No heavy compute.

Usage: python build_figures_wine.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WA = os.path.join(ROOT, "results", "wine", "wine_assessment")
WINE = os.path.join(ROOT, "data", "awri_wine_smoke_wide_labelled.csv")
OUT = os.path.join(ROOT, "figures")  # not tracked
TOUT = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TOUT, exist_ok=True)

SCEN_COL = {"A": "#0072B2", "B": "#56B4E9", "C": "#E69F00", "D": "#D55E00"}
CULT_COL = {"CHA": "#009E73", "PIN": "#CC79A7", "SHI": "#D55E00"}
CULT_NAME = {"CHA": "Chardonnay", "PIN": "Pinot Noir", "SHI": "Shiraz"}
STUDY_MK = {"A": "o", "B": "s"}
GREY, DARK = "#6E6E6E", "#222222"
SETTINGS = [("cv", "Cultivar-stratified CV"), ("cultivar_holdout", "Cultivar hold-out"),
            ("grouped_site", "Sites held out"), ("grouped_region", "Regions held out"),
            ("external_trainB_testA", "Train B, test A"), ("external_trainA_testB", "Train A, test B")]
SET_SHORT = dict(SETTINGS)
MODELS = [("b_guaiacol", "Guaiacol alone", "#D55E00"), ("b_best_single", "Best single analyte", "#E69F00"),
          ("b_phenol_index", "Phenol index", "#F0E442"), ("b_marker_index", "Marker index", "#CC79A7"),
          ("b_cultivar_mean", "Cultivar mean", "#999999"), ("lasso", "Lasso alone", "#56B4E9"),
          ("ensemble", "Five-model ensemble", "#0072B2")]
RANKS = [("consensus_std", "Consensus SD", "#0072B2"), ("knn_distance", "kNN distance", "#009E73"),
         ("rf_tree_std", "RF tree SD", "#CC79A7"), ("attr_disagree", "Attribution disagreement", "#D55E00"),
         ("random", "Random", "#999999"), ("oracle", "Oracle", "#222222")]

plt.rcParams.update({"font.size": 10, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def rd(name):
    return pd.read_csv(os.path.join(WA, name))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def panel_label(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=15, fontweight="bold")


def write_table(df, name, caption):
    df.to_csv(os.path.join(TOUT, name + ".csv"), index=False)
    with open(os.path.join(TOUT, name + ".md"), "w") as f:
        f.write(caption + "\n\n" + df.to_markdown(index=False) + "\n")
    print("table", name, df.shape)


wine = pd.read_csv(WINE)
pred = rd("per_wine_predictions.csv")
base = rd("baseline_comparison.csv")
dtest = rd("discordance_tests.csv")
dwine = rd("discordance_wines.csv")
rc = rd("risk_coverage.csv")
aurc = rd("risk_coverage_aurc.csv")
tri = rd("triage.csv")
stage = rd("stage_comparison.csv")

# ============================================================ Figure 1: guaiacol discordance
cvb = pred[(pred.setting == "cv") & (pred.feature_set == "both")]
disc_cv = dwine[(dwine.reference == "b_guaiacol") & (dwine.feature_set == "both") & (dwine.setting == "cv")].set_index("wine_uid")
w = wine.set_index("wine_uid")
w["disc_share"] = disc_cv["discordant_share"]
w["ref_resid"] = disc_cv["mean_ref_resid"]
w["ens_err"] = disc_cv["mean_ens_err"]

fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax = axs[0]
x = np.log1p(w["wine_guaiacol"])
# display-only OLS on all 63 wines (the analysis refits inside every split)
b1, b0 = np.polyfit(x, w["smoke_flavour"], 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, b0 + b1 * xs, color=GREY, lw=1.5, label="Guaiacol OLS (all 63 wines, display only)")
ax.fill_between(xs, b0 + b1 * xs - 1.5, b0 + b1 * xs + 1.5, color=GREY, alpha=0.12, lw=0, label="Residual within 1.5 rating units")
for cult, col in CULT_COL.items():
    for st, mk in STUDY_MK.items():
        m = (w.cultivar_code == cult) & (w.study == st)
        if not m.any():
            continue
        disc = w.loc[m, "disc_share"] >= 0.5
        ax.scatter(x[m & ~disc.reindex(w.index, fill_value=False)], w.loc[m & ~disc.reindex(w.index, fill_value=False), "smoke_flavour"],
                   s=55, marker=mk, color=col, edgecolor="white", lw=0.8, zorder=3)
        ax.scatter(x[m & disc.reindex(w.index, fill_value=False)], w.loc[m & disc.reindex(w.index, fill_value=False), "smoke_flavour"],
                   s=95, marker=mk, color=col, edgecolor=DARK, lw=1.6, zorder=4)
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], marker="o", ls="", color=c, markersize=8, label=CULT_NAME[k]) for k, c in CULT_COL.items()]
handles += [Line2D([], [], marker="o", ls="", color="white", markeredgecolor=DARK, markersize=8, label="Study A (circles)"),
            Line2D([], [], marker="s", ls="", color="white", markeredgecolor=DARK, markersize=8, label="Study B (squares)"),
            Line2D([], [], marker="o", ls="", color="white", markeredgecolor=DARK, markeredgewidth=1.8, markersize=10,
                   label="Guaiacol-discordant in CV (dark outline)")]
h0, l0 = ax.get_legend_handles_labels()
ax.legend(h0 + handles, l0 + [h.get_label() for h in handles], frameon=False, fontsize=8.5, loc="upper left")
ticks = [0, 1, 2, 5, 10, 20, 50, 100]
ax.set_xticks(np.log1p(ticks))
ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("Guaiacol in wine (µg/L, log scale)")
ax.set_ylabel("Panel smoke flavour (0 to 10)")
ax.set_title("Guaiacol alone misranks many wines", fontsize=12)
panel_label(ax, "A")

ax = axs[1]
g = dtest[(dtest.reference == "b_guaiacol") & (dtest.feature_set == "both")].set_index("setting")
order = ["cv", "cultivar_holdout", "grouped_site", "grouped_region", "external_trainB_testA", "external_trainA_testB"]
ys = np.arange(len(order))
ax.barh(ys + 0.2, [g.loc[s, "ref_mae_discordant"] for s in order], 0.38, color="#D55E00", edgecolor="white", label="Guaiacol OLS, discordant wines")
ax.barh(ys - 0.2, [g.loc[s, "ens_mae_discordant"] for s in order], 0.38, color="#0072B2", edgecolor="white", label="Ensemble, discordant wines")
ax.scatter([g.loc[s, "ens_mae_concordant"] for s in order], ys - 0.2, marker="|", s=160, color=DARK, zorder=4, label="Ensemble, concordant wines")
ax.set_yticks(ys)
ax.set_yticklabels([f"{SET_SHORT[s]}\n({int(g.loc[s, 'n_discordant_pred'])} of {int(g.loc[s, 'n_pred'])} predictions)" for s in order], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Mean absolute error (rating units)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_title("The ensemble recovers most discordant wines", fontsize=12)
panel_label(ax, "B")

ax = axs[2]
PHEN = ["wine_guaiacol", "wine_4-methylguaiacol", "wine_m-cresol", "wine_o-cresol", "wine_p-cresol", "wine_syringol", "wine_4-methylsyringol"]
w["ratio"] = w[PHEN[1:]].sum(axis=1) / w["wine_guaiacol"]
groups = [("Concordant", w[w.disc_share < 0.5]), ("Over-predicted\nby guaiacol", w[(w.disc_share >= 0.5) & (w.ref_resid > 0)]),
          ("Under-predicted\nby guaiacol", w[(w.disc_share >= 0.5) & (w.ref_resid < 0)])]
for i, (lab, sub) in enumerate(groups):
    vals = sub["ratio"]
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
    ax.scatter(i + jitter, vals, s=45, color=[CULT_COL[c] for c in sub.cultivar_code], edgecolor="white", zorder=3)
    ax.hlines(vals.median(), i - 0.3, i + 0.3, color=DARK, lw=2, zorder=4)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{lab}\n(n = {len(sub)})" for lab, sub in groups], fontsize=8.5)
ax.set_yscale("log")
ax.set_ylabel("Other free phenols / guaiacol in wine")
ax.set_title("Discordant wines differ in phenol balance", fontsize=12)
panel_label(ax, "C")
fig.tight_layout()
save(fig, "Figure1_guaiacol_discordance.png")

# ============================================================ Figure 2: baselines
fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
four = pd.read_csv(os.path.join(TOUT, "round6b_four_phenol_baseline.csv"))
FOURMAE = {(r.Evaluation, r.Features): r["MAE four-phenol sum"] for _, r in four.iterrows()}
MODELS2 = MODELS[:2] + [("b_four", "Four-phenol sum", "#009E73")] + MODELS[2:]
for ax, fs, ttl, lab in [(axs[0], "both", "Grape and wine chemistry", "A"), (axs[1], "grape", "Grape chemistry only", "B")]:
    b = base[base.feature_set == fs].set_index(["setting", "model"])
    ys = np.arange(len(order))
    for j, (m, name, col) in enumerate(MODELS2):
        off = (j - 3.5) * 0.1
        fsname = {"both": "grape + wine", "grape": "grape"}[fs]
        vals = [FOURMAE[(SET_SHORT[s], fsname)] if m == "b_four" else b.loc[(s, m), "mae"] for s in order]
        ax.scatter(vals, ys + off, s=60 if m != "ensemble" else 90, color=col, edgecolor=DARK if m == "ensemble" else "white",
                   lw=1.2 if m == "ensemble" else 0.6, zorder=4, label=name, marker="D" if m == "ensemble" else "o")
    ax.set_yticks(ys)
    ax.set_yticklabels([SET_SHORT[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error (rating units)")
    ax.set_xlim(0.4, 2.1)
    ax.set_title(ttl, fontsize=12)
    panel_label(ax, lab)
axs[0].legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(1.05, -0.14), ncol=4)
fig.tight_layout()
save(fig, "Figure2_baselines.png")

# ============================================================ Figure 3: uncertainty rankings and triage
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
panels = [("cv", axs[0, 0]), ("cultivar_holdout", axs[0, 1]), ("grouped_region", axs[0, 2]),
          ("external_trainB_testA", axs[1, 0]), ("external_trainA_testB", axs[1, 1])]
for s, ax in panels:
    r = rc[(rc.setting == s) & (rc.feature_set == "both")]
    for rk, name, col in RANKS:
        q = r[r.ranking == rk].sort_values("coverage")
        ax.plot(q.coverage, q.mae_retained, marker="o", ms=4, lw=2 if rk == "consensus_std" else 1.4,
                ls="--" if rk in ("random", "oracle") else "-", color=col, label=name)
    ax.set_xlabel("Coverage (share of wines retained)")
    ax.set_ylabel("MAE of retained wines")
    ax.set_title(SET_SHORT[s], fontsize=12)
    ax.set_xlim(0.47, 1.03)
axs[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
for ax, lab in zip(axs.flat, "ABCDEF"):
    panel_label(ax, lab)
ax = axs[1, 2]
t = tri[(tri.threshold == "training_midpoint") & (tri.feature_set == "both")]
xs = np.arange(len(order))
for off, frac, col in [(-0.25, 0.0, "#999999"), (0.0, 0.2, "#56B4E9"), (0.25, 0.3, "#0072B2")]:
    q = t[(t.abstain_frac == frac) & (t.ranking == ("none" if frac == 0 else "consensus_std"))].set_index("setting")
    acc = [q.loc[s, "accuracy"] for s in order]
    ax.bar(xs + off, acc, 0.24, color=col, edgecolor="white", label=f"Abstain {int(frac * 100)}% by consensus SD" if frac else "No abstention")
    if frac:
        for xx, s in zip(xs + off, order):
            n_pos_ab, n_pos = int(q.loc[s, "n_pos_abstained"]), int(q.loc[s, "n_pos_retained"]) + int(q.loc[s, "n_pos_abstained"])
            ax.text(xx, q.loc[s, "accuracy"] + 0.01, f"{n_pos_ab}/{n_pos}", ha="center", fontsize=6.5, color=DARK, rotation=90, va="bottom")
ax.set_xticks(xs)
ax.set_xticklabels([SET_SHORT[s].replace(", ", ",\n").replace(" held out", "\nheld out").replace("-stratified ", "\n") for s in order], fontsize=8)
ax.set_ylim(0.6, 1.08)
ax.set_ylabel("Accuracy of smoky versus not-smoky call")
ax.set_title("Triage on the Dunnett label (smoky wines withheld / smoky wines)", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
fig.tight_layout()
save(fig, "Figure3_uncertainty_rankings_triage.png")

# ============================================================ Figure 4: transfer scatters and stage comparison
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, s, lab in [(axs[0], "external_trainB_testA", "A"), (axs[1], "external_trainA_testB", "B")]:
    q = pred[(pred.setting == s) & (pred.feature_set == "both")]
    for sc, col in SCEN_COL.items():
        qq = q[q.scenario == sc]
        ax.scatter(qq.smoke_flavour, qq.ensemble, s=60, color=col, edgecolor="white", label=f"Scenario {sc} (n = {len(qq)})", zorder=3)
    gq = q.copy()
    ax.scatter(gq.smoke_flavour, gq.b_guaiacol, s=28, marker="x", color=GREY, label="Guaiacol OLS", zorder=2)
    lim = [0, 9]
    ax.plot(lim, lim, color=DARK, lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed smoke flavour")
    ax.set_ylabel("Predicted smoke flavour")
    mae_e = (q.ensemble - q.smoke_flavour).abs().mean()
    mae_g = (q.b_guaiacol - q.smoke_flavour).abs().mean()
    ax.set_title(f"{SET_SHORT[s]} (ensemble MAE {mae_e:.2f}, guaiacol {mae_g:.2f})", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    panel_label(ax, lab)
ax = axs[2]
ys = np.arange(len(order))
for off, fs, col, name in [(-0.25, "grape", "#E69F00", "Grape chemistry"), (0.0, "wine", "#56B4E9", "Wine chemistry"), (0.25, "both", "#0072B2", "Grape and wine")]:
    b = base[(base.feature_set == fs) & (base.model == "ensemble")].set_index("setting")
    ax.barh(ys + off, [b.loc[s, "mae"] for s in order], 0.24, color=col, edgecolor="white", label=name)
st = stage[(stage.feature_a == "both") & (stage.feature_b == "grape")].set_index("setting")
for y, s in zip(ys, order):
    r = st.loc[s]
    ax.text(1.95, y, f"both minus grape {r.mae_a_minus_b:+.2f} ({r.ci_lo:+.2f} to {r.ci_hi:+.2f})", fontsize=7.5, va="center", color=DARK)
ax.set_yticks(ys)
ax.set_yticklabels([SET_SHORT[s] for s in order])
ax.invert_yaxis()
ax.set_xlim(0, 2.9)
ax.set_xlabel("Ensemble mean absolute error (rating units)")
ax.set_title("Assessment stage", fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
panel_label(ax, "C")
fig.tight_layout()
save(fig, "Figure4_transfer_and_stage.png")

# ============================================================ Tables
def verdict(r):
    if r.model == "ensemble":
        return ""
    if r.mae_diff_ci_hi < 0:
        return "ensemble lower"
    if r.mae_diff_ci_lo > 0:
        return "baseline lower"
    return "no difference"


# Table 1: baselines, MAE with verdict, both and grape feature sets
rows = []
for s in order:
    for fs in ["both", "grape"]:
        b = base[(base.setting == s) & (base.feature_set == fs)].set_index("model")
        row = {"Evaluation": SET_SHORT[s], "Features": {"both": "grape + wine", "grape": "grape"}[fs], "n": int(b.loc["ensemble", "n_pred"])}
        for m, name, _ in MODELS:
            r = b.loc[m]
            v = verdict(r.rename({"model": "model"}).to_frame().T.assign(model=m).iloc[0])
            mark = {"ensemble lower": "*", "baseline lower": "†", "no difference": "", "": ""}[v]
            row[name] = f"{r.mae:.2f}{mark}"
            if m == "b_best_single":
                fr = four[(four.Evaluation == SET_SHORT[s]) & (four.Features == {"both": "grape + wine", "grape": "grape"}[fs])].iloc[0]
                fmark = {"ensemble lower": "*", "four-phenol lower": "†", "no difference": ""}[fr["verdict"]]
                row["Four-phenol sum"] = f"{fr['MAE four-phenol sum']:.2f}{fmark}"
        row["Best single analyte (mode)"] = b.loc["best_single_name_mode", "note"].replace("wine_", "").replace("grape_", "grape ") if "best_single_name_mode" in b.index else ""
        row["Ensemble R2"] = f"{b.loc['ensemble', 'r2']:.2f}"
        rows.append(row)
t1 = pd.DataFrame(rows)
write_table(t1, "Table1_baselines", "Table 2. Mean absolute error of the ensemble and fair baselines by evaluation setting. * ensemble significantly lower than the baseline, † baseline significantly lower (wine-cluster bootstrap of the paired difference, 2,000 resamples, 95% interval excludes 0).")

# Table 2: discordance summary (guaiacol reference, both and grape)
rows = []
for s in order:
    for fs in ["both", "grape"]:
        r = dtest[(dtest.reference == "b_guaiacol") & (dtest.setting == s) & (dtest.feature_set == fs)].iloc[0]
        ci = r.ref_minus_ens_mae_discordant_ci
        rows.append({"Evaluation": SET_SHORT[s], "Features": {"both": "grape + wine", "grape": "grape"}[fs],
                     "Predictions": int(r.n_pred), "Discordant predictions": int(r.n_discordant_pred),
                     "Wines ever discordant": int(r.n_wines_ever_discordant),
                     "Share over-predicted by guaiacol": f"{r.share_overpredicted:.2f}",
                     "MAE guaiacol, discordant": f"{r.ref_mae_discordant:.2f}", "MAE ensemble, discordant": f"{r.ens_mae_discordant:.2f}",
                     "MAE ensemble, concordant": f"{r.ens_mae_concordant:.2f}",
                     "Guaiacol minus ensemble MAE, discordant (95% CI)": ci if isinstance(ci, str) and ci != "nan" else "",
                     "AUROC flag, consensus SD": f"{r.auroc_flag_discordant_consensus_std:.2f}",
                     "AUROC flag, kNN distance": f"{r.auroc_flag_discordant_knn_distance:.2f}",
                     "AUROC flag, attribution disagreement": f"{r.auroc_flag_discordant_attr_disagree:.2f}",
                     "Phenol sum MWU p": f"{r.phenol_sum_mwu_p:.3f}"})
t2 = pd.DataFrame(rows)
gap = pd.read_csv(os.path.join(TOUT, "round6b_prediction_gap.csv"))
gap["Features"] = gap["Features"]
t2 = t2.merge(gap[["Evaluation", "Features", "AUROC gap flags guaiacol discordance"]].rename(columns={"AUROC gap flags guaiacol discordance": "AUROC flag, ensemble-guaiacol gap"}), on=["Evaluation", "Features"], how="left")
t2["AUROC flag, ensemble-guaiacol gap"] = t2["AUROC flag, ensemble-guaiacol gap"].map(lambda v: f"{v:.2f}")
for c in ["Guaiacol minus ensemble MAE, discordant (95% CI)"]:
    t2[c] = t2[c].map(lambda v: "" if not isinstance(v, str) else v.replace("(", "").replace(")", "").replace(", ", " to "))
    t2[c] = t2[c].map(lambda v: " to ".join(f"{float(x):.2f}" for x in v.split(" to ")) if v else "")
write_table(t2, "Table2_discordance", "Table 1. Guaiacol-sensory discordance (|residual of the training-fitted guaiacol regression| at least 1.5 rating units) and what the ensemble does with those wines.")

# Table 3: AURC by ranking (both feature set) with CI vs consensus
rows = []
for s in order:
    a = aurc[(aurc.setting == s) & (aurc.feature_set == "both")].set_index("ranking")
    row = {"Evaluation": SET_SHORT[s], "Large errors (>1.5)": int(a.loc["consensus_std", "n_large_err"])}
    for rk, name, _ in RANKS:
        r = a.loc[rk]
        if rk == "consensus_std":
            row[name] = f"{r.aurc:.3f}"
        else:
            sig = "*" if r.aurc_diff_ci_hi < 0 else ("†" if r.aurc_diff_ci_lo > 0 else "")
            row[name] = f"{r.aurc:.3f}{sig}"
    g = gap[(gap.Evaluation == SET_SHORT[s]) & (gap.Features == "grape + wine")].iloc[0]
    ci = g["AURC gap minus consensus (95% CI)"]
    lo = float(ci.split(" to ")[0])
    row["Ensemble-guaiacol gap"] = f"{g['AURC ensemble error, ranked by gap']:.3f}" + ("†" if lo > 0 else "")
    row["Large errors caught by consensus SD, withhold 30%"] = f"{a.loc['consensus_std', 'large_err_caught_withhold30']:.2f}"
    rows.append(row)
t3 = pd.DataFrame(rows)
write_table(t3, "Table3_aurc", "Table 3. Area under the risk-coverage curve (lower is better) for ranking wine predictions by each uncertainty signal, grape and wine chemistry. * significantly lower than consensus SD, † significantly higher (wine-cluster bootstrap, 95% interval).")

# Table S: triage
rows = []
for s in order:
    t = tri[(tri.threshold == "training_midpoint") & (tri.feature_set == "both") & (tri.setting == s)]
    for frac, rk in [(0.0, "none"), (0.2, "consensus_std"), (0.3, "consensus_std"), (0.3, "knn_distance"), (0.3, "random")]:
        r = t[(t.abstain_frac == frac) & (t.ranking == rk)].iloc[0]
        rows.append({"Evaluation": SET_SHORT[s], "Abstention": f"{int(frac * 100)}%", "Ranking": dict((k, n) for k, n, _ in RANKS).get(rk, "none"),
                     "Threshold (mean)": f"{r.threshold_value_mean:.2f}", "AUROC of score": f"{r.score_auroc:.2f}",
                     "Retained": int(r.n_retained), "Smoky retained": int(r.n_pos_retained), "Smoky withheld": int(r.n_pos_abstained),
                     "False negatives": int(r.n_fn), "False positives": int(r.n_fp), "Errors withheld": int(r.n_fn_abstained + r.n_fp_abstained),
                     "Accuracy": f"{r.accuracy:.3f}"})
write_table(pd.DataFrame(rows), "TableS_triage", "Table S9. Triage of wines as smoky or not smoky (Dunnett label) at the training-midpoint threshold, with and without abstention.")

# Table S: stage comparison
st = stage.copy()
st["Evaluation"] = st.setting.map(SET_SHORT)
st = st[["Evaluation", "feature_a", "feature_b", "n_pairs", "mae_a", "mae_b", "mae_a_minus_b", "ci_lo", "ci_hi", "share_pred_gap_over_1"]]
st.columns = ["Evaluation", "Feature set A", "Feature set B", "Predictions", "MAE A", "MAE B", "MAE A minus B", "CI low", "CI high", "Share of predictions differing by >1 unit"]
for c in ["MAE A", "MAE B", "MAE A minus B", "CI low", "CI high", "Share of predictions differing by >1 unit"]:
    st[c] = st[c].map(lambda v: f"{v:.2f}")
write_table(st, "TableS_stage", "Table S10. Assessment-stage comparison of ensemble error between feature sets (wine-cluster bootstrap, 95% interval).")
print("done")
