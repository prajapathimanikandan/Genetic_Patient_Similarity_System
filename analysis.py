import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_HINTS = [
    "genetic_disorder", "genetic disorder", "disorder", "disorder_subclass",
    "class", "target", "diagnosis", "label", "status"
]


def guess_target(df):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for hint in TARGET_HINTS:
        for low, original in lowered.items():
            if hint == low or hint in low:
                return original
    return None


def _is_identifier_column(column_name):
    """Identify columns that are likely row identifiers rather than useful features."""
    name = str(column_name).strip().lower()
    identifier_names = {
        "id", "patient_id", "patient id", "record_id", "record id",
        "case_id", "case id", "uuid", "index", "patient_name",
        "patient name", "name"
    }
    return name in identifier_names or name.endswith("_id") or name.endswith(" id")


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    if categorical_cols:
        # Keep the encoded matrix bounded in size. Rare values and columns with
        # many unique categories are grouped into an infrequent-category bucket.
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=20,
                max_categories=20,
                sparse_output=False,
                dtype=np.float32
            ))
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False
    )


def run_analysis(df, target_col=None, n_components=2, variance_threshold=0.95):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    data = df.copy()
    data = data.dropna(axis=1, how="all")

    if target_col not in data.columns:
        target_col = None

    y = data[target_col].copy() if target_col else None
    X = data.drop(columns=[target_col]).copy() if target_col else data.copy()

    # Remove blank columns, constant columns and obvious record identifiers.
    X = X.drop(
        columns=[c for c in X.columns if str(c).strip().lower().startswith("unnamed")],
        errors="ignore"
    )
    X = X.drop(
        columns=[c for c in X.columns if _is_identifier_column(c)],
        errors="ignore"
    )
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # Drop categorical columns that are almost unique for every row. These are
    # usually names, codes or free-text identifiers and create huge one-hot matrices.
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    high_cardinality = []
    for col in categorical_cols:
        unique_count = X[col].nunique(dropna=True)
        if unique_count > max(100, int(0.5 * len(X))):
            high_cardinality.append(col)
    if high_cardinality:
        X = X.drop(columns=high_cardinality)

    if X.shape[1] == 0:
        raise ValueError(
            "No useful feature columns remain. Check whether the CSV contains "
            "actual measurement or clinical columns."
        )

    if len(X) < 3:
        raise ValueError("At least three usable rows are required.")

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    # Use float32 to approximately halve memory usage compared with float64.
    transformed = np.asarray(transformed, dtype=np.float32, order="C")

    if transformed.ndim != 2 or transformed.shape[1] < 2:
        raise ValueError(
            "At least two encoded feature dimensions are required for a 2D PCA plot."
        )

    max_possible = min(transformed.shape[0] - 1, transformed.shape[1])
    count = max(2, min(int(n_components), max_possible))

    # Randomized PCA is efficient for a large feature matrix when only a few
    # principal components are requested.
    pca = PCA(
        n_components=count,
        svd_solver="randomized",
        random_state=42
    )
    scores = pca.fit_transform(transformed)

    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    recommended = int(np.searchsorted(cumulative, variance_threshold) + 1)
    recommended = min(recommended, count)

    variance_df = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(count)],
        "Explained variance (%)": ratios * 100,
        "Cumulative variance (%)": cumulative * 100
    })

    pca_plot = pd.DataFrame({
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "Row": X.index.astype(str)
    })

    if y is not None:
        pca_plot["Class"] = y.fillna("Missing").astype(str).values

    lda_plot = None
    lda_message = "Choose a target column to enable supervised LDA."
    lda_dimensions = 0

    if y is not None:
        valid = y.notna()
        labels = y.loc[valid].astype(str)
        classes = labels.nunique()

        if classes < 2:
            lda_message = "LDA needs at least two distinct non-missing classes."
        elif int(valid.sum()) <= classes:
            lda_message = "LDA needs more labeled rows than distinct classes."
        else:
            lda_dimensions = min(2, classes - 1)
            try:
                lda = LinearDiscriminantAnalysis(n_components=lda_dimensions)
                lda_scores = lda.fit_transform(
                    transformed[valid.to_numpy()],
                    labels
                )

                lda_data = {
                    "LD1": lda_scores[:, 0],
                    "Row": X.index[valid].astype(str),
                    "Class": labels.values
                }
                if lda_scores.shape[1] > 1:
                    lda_data["LD2"] = lda_scores[:, 1]

                lda_plot = pd.DataFrame(lda_data)
                lda_message = ""
            except Exception as exc:
                lda_message = f"LDA could not be fitted: {exc}"
                lda_plot = None

    return {
        "original_feature_count": int(X.shape[1]),
        "pca_component_count": int(count),
        "row_count": int(len(X)),
        "variance_retained": float(cumulative[-1]),
        "recommended_components": int(recommended),
        "variance_df": variance_df,
        "pca_plot": pca_plot,
        "lda_plot": lda_plot,
        "lda_message": lda_message,
        "lda_dimensions": lda_dimensions
    }
