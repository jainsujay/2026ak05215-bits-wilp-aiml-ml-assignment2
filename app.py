from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes",
    "Support Vector Machine",
]
SCALED_MODELS = {"Logistic Regression", "KNN", "Support Vector Machine"}

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")
st.title("M.Tech AIML - Machine Learning Assignment 2")
st.markdown("### Interactive Classification Model Deployment")
st.caption("Breast Cancer Wisconsin (Diagnostic): 0 = malignant, 1 = benign. "
           "Binary precision, recall, F1 and AUC use 1 (benign) as the positive class.")

st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Classification Model", MODEL_NAMES)
uploaded_file = st.sidebar.file_uploader("Upload Test CSV Data", type=["csv"])

@st.cache_resource
def load_artifacts(model_name):
    """Load already-trained model and scaler once per selected model."""
    filename = MODEL_DIR / (model_name.lower().replace(" ", "_") + ".pkl")
    return joblib.load(filename), joblib.load(MODEL_DIR / "scaler.pkl")


def validate_test_frame(test_df, scaler):
    """Validate targets and feature columns; return data in TRAINING order.

    StandardScaler was fitted on a DataFrame so feature_names_in_ records
    the exact 30 names and their order, including for unscaled classifiers.
    """
    if "target" not in test_df.columns:
        raise ValueError("Uploaded CSV must include the 'target' column.")
    if test_df.columns.duplicated().any():
        raise ValueError("Uploaded CSV contains duplicate column names.")

    expected_features = list(scaler.feature_names_in_)
    incoming_features = [c for c in test_df.columns if c != "target"]
    missing = sorted(set(expected_features) - set(incoming_features))
    extra = sorted(set(incoming_features) - set(expected_features))
    if missing or extra:
        raise ValueError(f"Feature mismatch. Missing: {missing}; unexpected: {extra}")

    # Reorder columns. An identical set in a different order can be dangerous.
    X = test_df[expected_features].copy()
    try:
        X = X.apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(test_df["target"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All features and target values must be numeric.") from exc
    if X.empty:
        raise ValueError("The uploaded CSV contains no test rows.")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("The uploaded CSV contains missing feature or target values.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features must not contain infinity or invalid numeric values.")
    if not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 (malignant) and 1 (benign).")
    return X, y.astype(int)


def evaluate_model(model, scaler, model_name, X, y):
    """Predict and compute metrics; NEVER call .fit() on uploaded test data."""
    processed = scaler.transform(X) if model_name in SCALED_MODELS else X
    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)[:, 1]
    # ROC AUC is undefined unless test data contains both classes.
    auc = roc_auc_score(y, probabilities) if y.nunique() == 2 else np.nan
    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return predictions, metrics

if uploaded_file is None:
    st.info("Please upload a test dataset in CSV format using the sidebar. "
        "The dataset must contain all the required input features "
        "and a 'target' column containing the actual class labels.")
else:
    try:
        test_df = pd.read_csv(uploaded_file)
        selected_model, scaler = load_artifacts(model_choice)
        X_test, y_test = validate_test_frame(test_df, scaler)

        st.subheader("Uploaded Test Data Preview")
        st.dataframe(test_df.head(), use_container_width=True)
        st.caption(f"Uploaded {len(test_df)} test rows with {X_test.shape[1]} feature columns.")

        preds, scores = evaluate_model(selected_model, scaler, model_choice, X_test, y_test)
        st.subheader(f"Evaluation Metrics for: {model_choice}")
        metric_cols = st.columns(6)
        for col, (label, value) in zip(metric_cols, scores.items()):
            col.metric(label, "N/A" if np.isnan(value) else f"{value:.4f}")
        if np.isnan(scores["AUC"]):
            st.warning("AUC is undefined because the uploaded CSV contains only one target class.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, preds, labels=[0, 1])
            cm_df = pd.DataFrame(cm,
                index=["Actual Malignant (0)", "Actual Benign (1)"],
                columns=["Predicted Malignant (0)", "Predicted Benign (1)"])
            st.dataframe(cm_df, use_container_width=True)
        with col_b:
            st.subheader("Classification Report")
            report = classification_report(
                y_test, preds, labels=[0, 1],
                target_names=["Malignant (0)", "Benign (1)"],
                output_dict=True, zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        st.divider()
        st.subheader("All Five Models — Current Uploaded Test CSV")
        st.caption("Each saved model is evaluated on THIS uploaded CSV, without retraining.")
        comparison_rows = []
        for name in MODEL_NAMES:
            other_model, _ = load_artifacts(name)
            _, current_scores = evaluate_model(other_model, scaler, name, X_test, y_test)
            comparison_rows.append({"ML Model Name": name, **current_scores})
        st.dataframe(pd.DataFrame(comparison_rows).round(4),
                     hide_index=True, use_container_width=True)

        # Optional reference table uses the ORIGINAL test_data.csv from training.
        # It is deliberately separate to avoid misrepresenting uploaded-data results.
        original_metrics_path = BASE_DIR / "metrics.csv"
        if original_metrics_path.exists():
            with st.expander("Original training-run test-set metrics (reference only)"):
                st.dataframe(pd.read_csv(original_metrics_path),
                             hide_index=True, use_container_width=True)

    except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
        st.error(f"Cannot evaluate uploaded data: {exc}")
    except Exception as exc:
        st.error(f"Model evaluation failed: {exc}")
        st.caption("Verify compatible dependency versions and regenerate models if needed.")
