#!/usr/bin/env python
# coding: utf-8

# In[10]:


from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)


# In[14]:


# Artifacts are stored alongside this script.
BASE_DIR = Path("train_model.ipynb").resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


# In[16]:


#1. Load Dataset (Meets criteria: 30 features >= 12, 569 instances >= 500)
# IMPORTANT: original target codes are 0=malignant, 1=benign.
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")


# In[18]:


# 2. Hold out 20% of records. stratify preserves class proportions.
# A fixed seed makes your split and results reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# In[20]:


# 3. Standardize ONLY on training examples to prevent data leakage.
# This matters for distance- and margin-based classifiers.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")


# In[22]:


# 4. Save ONLY the held-out test records, with the correct target column.
# Streamlit will load saved estimators, NOT train on this uploaded CSV.
test_data = X_test.copy()
test_data["target"] = y_test.to_numpy()
test_data.to_csv(BASE_DIR / "test_data.csv", index=False)


# In[24]:


# 5. Train the same five classifier types as in your original code.
# Keeping hyperparameters unchanged helps preserve your experimental setup.
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}
SCALED_MODELS = {"Logistic Regression", "KNN", "Support Vector Machine"}
results = []

for name, model in models.items():
    # LR, KNN and SVM use the fitted scaler; Tree and NB use raw 30 features.
    if name in SCALED_MODELS:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

    # Names match the Streamlit dropdown exactly.
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, MODEL_DIR / filename)

    # All six metrics use positive label 1 (BENIGN).
    # AUC uses probability scores (not hard 0/1 predictions).
    scores = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "AUC": round(roc_auc_score(y_test, probabilities), 4),
        "Precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "F1": round(f1_score(y_test, predictions, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, predictions), 4),
    }
    results.append(scores)

    print(f"\n=== {name} ===")
    print("Confusion matrix (rows=actual; columns=predicted; class order 0,1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=["malignant (0)", "benign (1)"],
                                zero_division=0))


# In[26]:


# 6. Save summary for README and side-by-side UI comparison.
metrics = pd.DataFrame(results)
metrics.to_csv(BASE_DIR / "metrics.csv", index=False)
print("\n--- MODEL COMPARISON (POSITIVE CLASS: BENIGN) ---")
print(metrics.to_string(index=False))
print(f"\nSaved five models and scaler in {MODEL_DIR}")
print(f"Saved {len(test_data)} test records to {BASE_DIR / 'test_data.csv'}")
print(f"Saved evaluation table to {BASE_DIR / 'metrics.csv'}")


# In[ ]:




