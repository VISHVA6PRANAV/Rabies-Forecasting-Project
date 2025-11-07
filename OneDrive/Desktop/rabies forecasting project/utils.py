
# --- Utility helpers ---
import os
import pandas as pd

REQUIRED_COLS = {"algorithm", "accuracy", "precision", "recall", "f1_score"}

def example_dataframe():
    return pd.DataFrame({
        "algorithm": ["Logistic Regression", "Random Forest", "GCN"],
        "accuracy": [0.78, 0.86, 0.82],
        "precision": [0.60, 0.81, 0.75],
        "recall": [0.55, 0.79, 0.72],
        "f1_score": [0.57, 0.80, 0.73]
    })

def validate_and_prepare(df):
    cols_lower = {c.lower(): c for c in df.columns}
    if not REQUIRED_COLS.issubset(set(cols_lower.keys())):
        return None, "CSV must contain columns: algorithm, accuracy, precision, recall, f1_score"
    # Preserve only the required columns (any order in file is fine)
    use_cols = ["algorithm", "accuracy", "precision", "recall", "f1_score"]
    df = df[[cols_lower[c] for c in use_cols]]
    df.columns = use_cols
    for m in ["accuracy", "precision", "recall", "f1_score"]:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0).clip(0, 1)
    df["algorithm"] = df["algorithm"].astype(str)
    return df, None

def metrics_list():
    return ["accuracy", "precision", "recall", "f1_score"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def prepare_plot_df(df, selected_algs=None, selected_metrics=None):
    if selected_algs is None:
        selected_algs = df["algorithm"].unique().tolist()
    if selected_metrics is None:
        selected_metrics = metrics_list()
    plot_df = df[df["algorithm"].isin(selected_algs)].set_index("algorithm")
    plot_df = plot_df[selected_metrics]
    return plot_df
