
"""
Batch generator to produce organized plots for any metrics CSV.
Usage (example):
    python generator.py /path/to/metrics.csv
Creates outputs/<file-stem>/<timestamp>/ with comparisons, per_metric, per_algorithm, and a selected_metrics.csv.
"""
import sys, os, time, pathlib, pandas as pd
from utils import validate_and_prepare, prepare_plot_df, ensure_dir
from plots import save_all_plots

def generate(csv_path, selected_algs=None, selected_metrics=None):
    df = pd.read_csv(csv_path)
    df, err = validate_and_prepare(df)
    if err:
        raise ValueError(err)
    plot_df = prepare_plot_df(df, selected_algs, selected_metrics)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("outputs", pathlib.Path(csv_path).stem, timestamp)
    ensure_dir(out_dir)
    save_all_plots(plot_df, out_dir)
    plot_df.reset_index().to_csv(os.path.join(out_dir, "selected_metrics.csv"), index=False)
    return out_dir

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generator.py <metrics.csv>")
        sys.exit(2)
    out = generate(sys.argv[1])
    print("Generated:", out)
