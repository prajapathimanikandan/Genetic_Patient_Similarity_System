import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from src.analysis import run_analysis, guess_target

st.set_page_config(page_title="Genetic Data Explorer", page_icon="🧬", layout="wide")
st.title("🧬 Genetic Data Compression & Pattern Visualization")
st.caption("Explore high-dimensional genetic and clinical data with PCA and LDA.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload your Kaggle CSV file", type=["csv"])
    default_path = Path("data/genetic_disorders.csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("Uploaded dataset loaded.")
    elif default_path.exists():
        df = pd.read_csv(default_path)
        st.success(f"Loaded {default_path}")
    else:
        st.info("Upload a CSV or place it at data/genetic_disorders.csv")
        st.markdown("[Open Kaggle dataset](https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes/data)")
        st.stop()

    st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")
    suggested = guess_target(df)
    target_options = ["(No target / unsupervised)"] + list(df.columns)
    default_index = target_options.index(suggested) if suggested in target_options else 0
    target_choice = st.selectbox("Class / target column for LDA (optional)", target_options, index=default_index)
    target_col = None if target_choice.startswith("(") else target_choice
    max_components = st.slider("PCA components to calculate", 2, min(30, max(2, min(len(df)-1, len(df.columns)*3))), 2)
    variance_threshold = st.slider("Cumulative variance target", 0.80, 0.99, 0.95, 0.01)

if len(df) < 3:
    st.error("Please upload at least 3 rows.")
    st.stop()

try:
    result = run_analysis(df, target_col=target_col, n_components=max_components, variance_threshold=variance_threshold)
except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.info("Check that the CSV contains usable columns and enough non-missing values.")
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Original features", result["original_feature_count"])
c2.metric("PCA components", result["pca_component_count"])
c3.metric("Rows analyzed", f'{result["row_count"]:,}')
c4.metric("Variance retained", f'{result["variance_retained"]:.1%}')

with st.expander("Preview and data quality", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    quality = pd.DataFrame({
        "Missing values": df.isna().sum(),
        "Missing (%)": (df.isna().mean()*100).round(2),
        "Data type": df.dtypes.astype(str)
    })
    st.dataframe(quality, use_container_width=True)

st.subheader("Explained variance")
variance_df = result["variance_df"]
fig_var = px.bar(variance_df, x="Component", y="Explained variance (%)",
                 title="Variance explained by each principal component")
fig_var.update_layout(xaxis_title="Principal component", yaxis_title="Explained variance (%)")
st.plotly_chart(fig_var, use_container_width=True)
fig_cum = px.line(variance_df, x="Component", y="Cumulative variance (%)",
                  markers=True, title="Cumulative explained variance")
st.plotly_chart(fig_cum, use_container_width=True)
st.info(f'{result["recommended_components"]} component(s) retain at least {variance_threshold:.0%} of the variance (within the calculated components).')

st.subheader("2D PCA visualization")
pca_plot = result["pca_plot"]
if target_col:
    fig_pca = px.scatter(pca_plot, x="PC1", y="PC2", color="Class",
                         hover_data=["Row"], title="Patients projected onto the first two principal components")
else:
    fig_pca = px.scatter(pca_plot, x="PC1", y="PC2", hover_data=["Row"],
                         title="Rows projected onto the first two principal components")
st.plotly_chart(fig_pca, use_container_width=True)

st.subheader("LDA visualization")
if result["lda_plot"] is not None:
    lda_plot = result["lda_plot"]
    if "LD2" in lda_plot.columns:
        fig_lda = px.scatter(lda_plot, x="LD1", y="LD2", color="Class",
                             hover_data=["Row"], title="LDA class-discriminant projection")
    else:
        fig_lda = px.strip(lda_plot, x="LD1", y="Class", color="Class",
                           hover_data=["Row"], title="LDA one-dimensional class projection")
    st.plotly_chart(fig_lda, use_container_width=True)
    st.caption("LDA is supervised: it uses the selected class labels to find directions that separate classes.")
else:
    st.warning(result["lda_message"])

st.subheader("PCA vs LDA comparison")
comparison = pd.DataFrame([
    {"Method": "PCA", "Uses class labels": "No", "Objective": "Preserve overall variance",
     "Dimensions shown": "2", "Best use": "Exploration and compression"},
    {"Method": "LDA", "Uses class labels": "Yes", "Objective": "Separate known classes",
     "Dimensions shown": str(result["lda_dimensions"]), "Best use": "Class-discriminative visualization"}
])
st.table(comparison)
st.markdown("""
**Interpretation:** PCA may preserve variation caused by age, measurement scale, batch effects, or other factors,
even when those directions do not distinguish genetic-disorder classes. LDA explicitly uses the known class labels,
so it may show clearer class separation. That separation is not proof of clinical usefulness: it depends on label
quality, class balance, sample size, and validation on unseen data.
""")
st.download_button("Download PCA coordinates", result["pca_plot"].to_csv(index=False).encode("utf-8"),
                   "pca_coordinates.csv", "text/csv")
if result["lda_plot"] is not None:
    st.download_button("Download LDA coordinates", result["lda_plot"].to_csv(index=False).encode("utf-8"),
                       "lda_coordinates.csv", "text/csv")

st.caption("Research/education prototype only. Do not use these visualizations for diagnosis or patient-level clinical decisions.")
