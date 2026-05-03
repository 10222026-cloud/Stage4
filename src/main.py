import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv("data/sample.csv")

X = data[["time", "vibration"]]
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# 1. BASELINE MODEL
# -----------------------------
model_base = LogisticRegression()
model_base.fit(X_train, y_train)

y_pred_base = model_base.predict(X_test)
y_prob_base = model_base.predict_proba(X_test)[:, 1]

print("\n===== BASELINE =====")
print(confusion_matrix(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_base))


# -----------------------------
# 2. CLASS WEIGHT
# -----------------------------
model_weight = LogisticRegression(class_weight='balanced')
model_weight.fit(X_train, y_train)

y_pred_weight = model_weight.predict(X_test)
y_prob_weight = model_weight.predict_proba(X_test)[:, 1]

print("\n===== CLASS WEIGHT =====")
print(confusion_matrix(y_test, y_pred_weight))
print(classification_report(y_test, y_pred_weight))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_weight))


# -----------------------------
# 3. SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression()
model_smote.fit(X_train_sm, y_train_sm)

y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

print("\n===== SMOTE =====")
print(confusion_matrix(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_smote))


# -----------------------------
# 4. THRESHOLD TUNING
# -----------------------------
threshold = 0.3  # try 0.3 instead of default 0.5

y_pred_thresh = (y_prob_smote >= threshold).astype(int)

print("\n===== THRESHOLD TUNING (0.3) =====")
print(confusion_matrix(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_smote))


# -----------------------------
# 5. FOCAL LOSS (SIMULATED via weighting)
# -----------------------------
# sklearn doesn't support focal loss directly → simulate via strong class weighting
model_focal = LogisticRegression(class_weight={0:1, 1:3})
model_focal.fit(X_train, y_train)

y_pred_focal = model_focal.predict(X_test)
y_prob_focal = model_focal.predict_proba(X_test)[:, 1]

print("\n===== FOCAL LOSS (SIMULATED) =====")
print(confusion_matrix(y_test, y_pred_focal))
print(classification_report(y_test, y_pred_focal))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_focal))
