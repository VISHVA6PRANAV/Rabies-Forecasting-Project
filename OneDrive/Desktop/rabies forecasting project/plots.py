
# --- Plot helpers ---
import os
import matplotlib.pyplot as plt
import seaborn as sns

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _save_current_fig(fig, out_path):
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, bbox_inches="tight", dpi=200)

def plot_metrics(plot_df, chart_type):
    fig, ax = plt.subplots(figsize=(8, 4 + 0.5 * max(1, len(plot_df))))
    sns.set_style("whitegrid")
    if plot_df.empty:
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
        return fig
    if chart_type == "Bar":
        plot_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Metric")
    elif chart_type == "Horizontal Bar":
        plot_df.plot(kind="barh", ax=ax)
        ax.set_xlabel("Score")
        ax.set_xlim(0, 1.05)
        ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plot_df.T.plot(ax=ax, marker="o")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig

def save_all_plots(plot_df, out_dir, save_comparisons=True, save_per_metric=True, save_per_algorithm=True):
    '''
    Creates an organized set of plots:

    out_dir/
      comparisons/
        comparison_bar.png
        comparison_horizontal_bar.png
        comparison_line.png
      per_metric/
        accuracy_bar.png
        precision_bar.png
        recall_bar.png
        f1_score_bar.png
      per_algorithm/
        <AlgorithmName>/
          metrics_line.png
    '''
    _ensure_dir(out_dir)
    sns.set_style("whitegrid")

    # A) Comparisons across algorithms
    if save_comparisons:
        cmp_dir = os.path.join(out_dir, "comparisons")
        _ensure_dir(cmp_dir)
        for ctype, fname in [("Bar", "comparison_bar.png"),
                             ("Horizontal Bar", "comparison_horizontal_bar.png"),
                             ("Line", "comparison_line.png")]:
            fig = plot_metrics(plot_df, ctype)
            _save_current_fig(fig, os.path.join(cmp_dir, fname))
            plt.close(fig)

    # B) Per metric: bar chart of all algorithms for that metric
    if save_per_metric:
        pm_dir = os.path.join(out_dir, "per_metric")
        _ensure_dir(pm_dir)
        for metric in plot_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4 + 0.5 * max(1, len(plot_df))))
            plot_df[metric].sort_values(ascending=False).plot(kind="bar", ax=ax)
            ax.set_ylabel(metric.title())
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{metric.title()} by Algorithm")
            _save_current_fig(fig, os.path.join(pm_dir, f"{metric}_bar.png"))
            plt.close(fig)

    # C) Per algorithm: line chart across metrics
    if save_per_algorithm:
        pa_dir = os.path.join(out_dir, "per_algorithm")
        _ensure_dir(pa_dir)
        for alg, row in plot_df.iterrows():
            alg_dir = os.path.join(pa_dir, _sanitize(alg))
            _ensure_dir(alg_dir)
            fig, ax = plt.subplots(figsize=(8, 4))
            row.plot(ax=ax, marker="o")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title(f"Metrics for {alg}")
            _save_current_fig(fig, os.path.join(alg_dir, "metrics_line.png"))
            plt.close(fig)

def _sanitize(name: str) -> str:
    # safe folder name
    return "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in name).strip().replace(" ", "_")
