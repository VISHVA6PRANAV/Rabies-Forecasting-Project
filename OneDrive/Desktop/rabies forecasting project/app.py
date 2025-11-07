
# --- Streamlit app entrypoint ---
import io, os, time, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from utils import example_dataframe, validate_and_prepare, metrics_list, ensure_dir, prepare_plot_df
from plots import plot_metrics, save_all_plots

st.set_page_config(page_title="Model Metrics Explorer", layout="wide")

st.sidebar.header("Input / Options")
uploaded = st.sidebar.file_uploader("Upload metrics CSV (cols: algorithm, accuracy, precision, recall, f1_score)", type=["csv"])
if uploaded:
    # keep a copy on disk so outputs live next to the upload
    raw_bytes = uploaded.getvalue()
    fname = pathlib.Path(uploaded.name)
    ensure_dir("uploads")
    upload_path = f"uploads/{fname.stem}.csv"
    with open(upload_path, "wb") as f:
        f.write(raw_bytes)

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        df = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8")))
else:
    use_example = st.sidebar.checkbox("Use example data", value=True)
    df = example_dataframe() if use_example else st.sidebar.text_input("No data loaded. Upload a CSV.", "")

if isinstance(df, str):
    st.stop()

df, errors = validate_and_prepare(df)
if errors:
    st.error(errors)
    st.stop()

metrics = metrics_list()

st.sidebar.header("Visualization")
selected_algs = st.sidebar.multiselect("Select algorithm(s)", options=df["algorithm"].unique(), default=df["algorithm"].unique().tolist())
selected_metrics = st.sidebar.multiselect("Select metric(s)", options=metrics, default=metrics)
chart_type = st.sidebar.radio("Chart type", ("Bar", "Horizontal Bar", "Line"))
show_table = st.sidebar.checkbox("Show numeric table", value=True)
download_btn = st.sidebar.button("Download selected metrics CSV")

plot_df = prepare_plot_df(df, selected_algs, selected_metrics)

st.title("Model Metrics Explorer")
st.markdown("Choose algorithms and metrics from the sidebar. The app shows charts and a table for comparison.")

st.subheader("Metric Comparison")
fig = plot_metrics(plot_df, chart_type)
st.pyplot(fig)

if show_table:
    st.subheader("Metrics Table")
    st.dataframe(plot_df.round(4))

# --- Auto-save organized plots when a CSV is uploaded ---
if uploaded:
    # Make a structured output folder: outputs/<file-stem>/<timestamp>/
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("outputs", pathlib.Path(uploaded.name).stem, timestamp)
    ensure_dir(out_dir)
    # Save the current comparison figure
    fig_path = os.path.join(out_dir, f"comparison_{chart_type.lower().replace(' ', '_')}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=200)
    # Save a full organized set of plots
    save_all_plots(plot_df, out_dir)
    # Save selected metrics CSV
    plot_df.reset_index().to_csv(os.path.join(out_dir, "selected_metrics.csv"), index=False)
    # Show success message
    st.success(f"Saved organized plots and CSV to: {out_dir}")

st.subheader("Per-algorithm details")
if len(selected_algs) == 1:
    alg = selected_algs[0]
    st.markdown(f"**Selected algorithm:** {alg}")
    uploaded_img = st.file_uploader("Upload confusion matrix / feature importance image (optional)", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        st.image(uploaded_img, use_column_width=True)
else:
    st.info("Choose exactly one algorithm in the sidebar to enable image upload for per-algorithm plots.")

if download_btn and not plot_df.empty:
    to_download = plot_df.reset_index()
    csv_bytes = to_download.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="selected_metrics.csv", mime="text/csv")

st.markdown("---")
st.markdown("Tips: Provide a CSV with columns `algorithm, accuracy, precision, recall, f1_score`. Scores should be in [0,1].")
