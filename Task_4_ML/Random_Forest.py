import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

import warnings
warnings.filterwarnings("ignore")

# =========================
# Paths
# =========================
FEATURES_DIR = "Task_4_v2/split_style_bow_glove_outputs/three_style_features"

SAVE_DIR = "Task_4_ML"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASETS = {
    ("bow", "original"): (
        FEATURES_DIR + "/bow_train_original.csv",
        FEATURES_DIR + "/bow_test_original.csv"
    ),
    ("bow", "style_b"): (
        FEATURES_DIR + "/bow_train_style_b.csv",
        FEATURES_DIR + "/bow_test_style_b.csv"
    ),
    ("bow", "style_c"): (
        FEATURES_DIR + "/bow_train_style_c.csv",
        FEATURES_DIR + "/bow_test_style_c.csv"
    ),
    ("glove", "original"): (
        FEATURES_DIR + "/glove_train_original.csv",
        FEATURES_DIR + "/glove_test_original.csv"
    ),
    ("glove", "style_b"): (
        FEATURES_DIR + "/glove_train_style_b.csv",
        FEATURES_DIR + "/glove_test_style_b.csv"
    ),
    ("glove", "style_c"): (
        FEATURES_DIR + "/glove_train_style_c.csv",
        FEATURES_DIR + "/glove_test_style_c.csv"
    ),
}

# =========================
# Tracking best model
# =========================
best_f1 = -1
best_model = None
best_encoder = None
best_info = None

results = []

# =========================
# Training loop
# =========================
for (rep, style), (train_path, test_path) in DATASETS.items():

    print("\n" + "="*60)
    print(f"{rep.upper()} | {style}")
    print("="*60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ["row_id", "ground_truth"]]

    X_train = train_df[feature_cols].values
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)[feature_cols].values

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["ground_truth"])
    y_test = le.transform(test_df["ground_truth"])

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-macro: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results.append({
        "rep": rep,
        "style": style,
        "model": "RandomForest",
        "accuracy": acc,
        "f1_macro": f1
    })

    # =========================
    # BEST MODEL CHECK
    # =========================
    if f1 > best_f1:
        best_f1 = f1
        best_model = rf
        best_encoder = le
        best_info = (rep, style)

# =========================
# SAVE BEST MODEL (.pkl)
# =========================
model_name = f"RandomForest_{best_info[0]}_{best_info[1]}.pkl"
model_path = os.path.join(SAVE_DIR, model_name)

joblib.dump(best_model, model_path)

# also save encoder (important)
encoder_path = os.path.join(SAVE_DIR, f"RandomForest_{best_info[0]}_{best_info[1]}.pkl")
joblib.dump(best_encoder, encoder_path)

print("\n" + "="*60)
print("BEST MODEL SAVED")
print("="*60)
print("Model:", model_name)
print("F1-score:", best_f1)

# =========================
# RESULTS TABLE
# =========================
results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)

print("\n FINAL RESULTS")
print(results_df)

# =========================
# PLOT
# =========================
plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="rep", y="f1_macro", hue="style")
plt.title("Random Forest Performance Across All Datasets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()