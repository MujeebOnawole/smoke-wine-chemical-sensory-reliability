# statistical_testing.py

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from logger import get_logger

logger = get_logger(__name__)


class StatisticalTesting:
    """Statistical diagnostics for cross-validation results."""

    def __init__(self):
        self._logger = get_logger(__name__)

    def create_single_model_diagnostic_plots(
        self,
        cv_results: List[Dict],
        metric: str,
        save_path: str,
    ) -> Dict:
        """Generate a 2x2 diagnostic panel for a single model's CV results.

        Parameters
        ----------
        cv_results : list of dict
            Each dict must contain ``cv``, ``fold``, and the key given by
            *metric* (e.g. ``auc``, ``rmse``, ``f1``).
        metric : str
            Name of the metric column to analyse.
        save_path : str
            Full path for the output PNG.

        Returns
        -------
        dict
            Summary statistics: mean, std, ci_lower, ci_upper,
            normality_p_value.
        """
        try:
            df = pd.DataFrame(cv_results)

            if metric not in df.columns:
                self._logger.warning(
                    f"Metric '{metric}' not found in cv_results columns: "
                    f"{list(df.columns)}"
                )
                return {}

            values = pd.to_numeric(df[metric], errors='coerce').dropna().values

            if len(values) < 3:
                self._logger.warning(
                    f"Not enough valid values ({len(values)}) for diagnostic plots"
                )
                return {}

            # ---- summary stats ------------------------------------------------
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1))
            n = len(values)
            se = stats.sem(values)
            ci = stats.t.interval(0.95, n - 1, loc=mean_val, scale=se)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])

            _, normality_p = stats.shapiro(values)
            normality_p = float(normality_p)

            summary = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'normality_p_value': normality_p,
            }

            # ---- figure -------------------------------------------------------
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Top-left: histogram + KDE
            ax = axes[0, 0]
            ax.hist(values, bins='auto', density=True, alpha=0.6,
                    color='steelblue', edgecolor='black')
            try:
                from scipy.stats import gaussian_kde
                kde_x = np.linspace(values.min(), values.max(), 200)
                kde = gaussian_kde(values)
                ax.plot(kde_x, kde(kde_x), color='darkblue', linewidth=2)
            except Exception:
                pass
            ax.set_title(f'{metric.upper()} Distribution')
            ax.set_xlabel(metric.upper())
            ax.set_ylabel('Density')

            # Top-right: box plot grouped by CV run
            ax = axes[0, 1]
            if 'cv' in df.columns:
                cv_groups = df.groupby('cv')[metric].apply(
                    lambda s: pd.to_numeric(s, errors='coerce').dropna().tolist()
                )
                box_data = [v for v in cv_groups if len(v) > 0]
                box_labels = [f'CV {k}' for k, v in cv_groups.items() if len(v) > 0]
                if box_data:
                    ax.boxplot(box_data, labels=box_labels)
            else:
                ax.boxplot(values)
            ax.set_title(f'{metric.upper()} by CV Run')
            ax.set_ylabel(metric.upper())

            # Bottom-left: Q-Q plot
            ax = axes[1, 0]
            stats.probplot(values, dist='norm', plot=ax)
            ax.set_title('Q-Q Plot (Normality Check)')

            # Bottom-right: CV x Fold heatmap
            ax = axes[1, 1]
            if 'cv' in df.columns and 'fold' in df.columns:
                pivot = df.pivot_table(
                    values=metric, index='cv', columns='fold', aggfunc='first'
                )
                pivot = pivot.apply(pd.to_numeric, errors='coerce')
                try:
                    import seaborn as sns
                    sns.heatmap(
                        pivot, annot=True, fmt='.4f', cmap='YlOrRd',
                        ax=ax, linewidths=0.5
                    )
                except ImportError:
                    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
                    plt.colorbar(im, ax=ax)
                    for i in range(pivot.shape[0]):
                        for j in range(pivot.shape[1]):
                            val = pivot.values[i, j]
                            if not np.isnan(val):
                                ax.text(j, i, f'{val:.4f}', ha='center',
                                        va='center', fontsize=8)
                ax.set_title(f'{metric.upper()} Heatmap (CV x Fold)')
                ax.set_xlabel('Fold')
                ax.set_ylabel('CV Run')
            else:
                ax.text(0.5, 0.5, 'CV/Fold info\nnot available',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_title(f'{metric.upper()} Heatmap')

            fig.suptitle(
                f'Single-Model Diagnostic: {metric.upper()}  '
                f'(mean={mean_val:.4f}, std={std_val:.4f})',
                fontsize=14, y=1.02,
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            self._logger.info(f"Single-model diagnostic plots saved to {save_path}")
            return summary

        except Exception as e:
            self._logger.error(
                f"Error creating single model diagnostic plots: {e}"
            )
            plt.close('all')
            return {}


# ---------------------------------------------------------------------------
# Standalone function
# ---------------------------------------------------------------------------

def create_diagnostic_plots(
    cv_results: List[Dict],
    metric: str,
    save_path: str,
    methods: Optional[List[str]] = None,
) -> None:
    """Generate a 2x2 diagnostic panel for one or more methods.

    Parameters
    ----------
    cv_results : list of dict
        Each dict should contain the key given by *metric* and optionally
        ``cv``, ``fold``, ``method``.
    metric : str
        Metric column to visualise (e.g. ``'auc'``, ``'rmse'``, ``'f1'``).
    save_path : str
        Full path for the output PNG.
    methods : list of str, optional
        Method labels. If the dicts do not contain a ``method`` key the
        first entry in *methods* is used as a blanket label.
    """
    try:
        df = pd.DataFrame(cv_results)

        if metric not in df.columns:
            logger.warning(
                f"Metric '{metric}' not found in cv_results columns: "
                f"{list(df.columns)}"
            )
            return

        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=[metric])

        if df.empty:
            logger.warning("No valid data for diagnostic plots after filtering")
            return

        # Assign method labels
        if 'method' not in df.columns:
            label = methods[0] if methods else 'Model'
            df['method'] = label

        method_list = methods if methods else df['method'].unique().tolist()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # -- Top-left: box plot per method ------------------------------------
        ax = axes[0, 0]
        plot_data = [
            df.loc[df['method'] == m, metric].values for m in method_list
        ]
        plot_data = [d for d in plot_data if len(d) > 0]
        plot_labels = [m for m, d in zip(method_list,
                       [df.loc[df['method'] == m, metric].values
                        for m in method_list]) if len(d) > 0]
        if plot_data:
            bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            colours = plt.cm.Set2(np.linspace(0, 1, len(plot_data)))
            for patch, colour in zip(bp['boxes'], colours):
                patch.set_facecolor(colour)
        ax.set_title(f'{metric.upper()} Box Plot')
        ax.set_ylabel(metric.upper())

        # -- Top-right: violin plot per method --------------------------------
        ax = axes[0, 1]
        try:
            import seaborn as sns
            sns.violinplot(
                data=df[df['method'].isin(method_list)],
                x='method', y=metric, ax=ax, inner='box', cut=0,
            )
        except ImportError:
            if plot_data:
                parts = ax.violinplot(plot_data, showmeans=True,
                                      showmedians=True)
                ax.set_xticks(range(1, len(plot_labels) + 1))
                ax.set_xticklabels(plot_labels)
        ax.set_title(f'{metric.upper()} Violin Plot')
        ax.set_ylabel(metric.upper())

        # -- Bottom-left: fold-wise bar chart ---------------------------------
        ax = axes[1, 0]
        if 'fold' in df.columns:
            fold_stats = (
                df[df['method'].isin(method_list)]
                .groupby('fold')[metric]
                .agg(['mean', 'std'])
                .reset_index()
            )
            fold_stats['std'] = fold_stats['std'].fillna(0)
            x_pos = np.arange(len(fold_stats))
            ax.bar(x_pos, fold_stats['mean'], yerr=fold_stats['std'],
                   capsize=4, color='steelblue', edgecolor='black', alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'Fold {int(f)}' for f in fold_stats['fold']])
            ax.set_title(f'{metric.upper()} by Fold (mean +/- std)')
            ax.set_ylabel(metric.upper())
        else:
            ax.text(0.5, 0.5, 'Fold info not available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12)
            ax.set_title(f'{metric.upper()} by Fold')

        # -- Bottom-right: summary statistics table ---------------------------
        ax = axes[1, 1]
        ax.axis('off')
        table_rows = []
        for m in method_list:
            vals = df.loc[df['method'] == m, metric].values
            if len(vals) == 0:
                continue
            n = len(vals)
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if n > 1 else 0.0
            med_v = np.median(vals)
            min_v = np.min(vals)
            max_v = np.max(vals)

            if n >= 3:
                se = stats.sem(vals)
                ci = stats.t.interval(0.95, n - 1, loc=mean_v, scale=se)
                ci_str = f'[{ci[0]:.4f}, {ci[1]:.4f}]'
            else:
                ci_str = 'N/A'

            table_rows.append([
                m, f'{n}', f'{mean_v:.4f}', f'{std_v:.4f}',
                f'{med_v:.4f}', f'{min_v:.4f}', f'{max_v:.4f}', ci_str,
            ])

        col_labels = ['Method', 'N', 'Mean', 'Std', 'Median',
                       'Min', 'Max', '95% CI']
        if table_rows:
            table = ax.table(
                cellText=table_rows, colLabels=col_labels,
                loc='center', cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.4)
        ax.set_title('Summary Statistics', pad=20)

        fig.suptitle(
            f'Diagnostic Plots: {metric.upper()}',
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Diagnostic plots saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating diagnostic plots: {e}")
        plt.close('all')
