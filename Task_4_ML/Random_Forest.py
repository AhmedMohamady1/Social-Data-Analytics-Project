import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Paths & Configuration
# ==========================================
FEATURES_DIR = "Task_4_v2/split_style_bow_glove_outputs/three_style_features"
SAVE_DIR = "Task_4_ML/Task_4_ML_Production"
os.makedirs(SAVE_DIR, exist_ok=True)

DATASETS = {
    ("bow", "original"): (f"{FEATURES_DIR}/bow_train_original.csv", f"{FEATURES_DIR}/bow_test_original.csv"),
    ("bow", "style_b"): (f"{FEATURES_DIR}/bow_train_style_b.csv", f"{FEATURES_DIR}/bow_test_style_b.csv"),
    ("bow", "style_c"): (f"{FEATURES_DIR}/bow_train_style_c.csv", f"{FEATURES_DIR}/bow_test_style_c.csv"),
    ("glove", "original"): (f"{FEATURES_DIR}/glove_train_original.csv", f"{FEATURES_DIR}/glove_test_original.csv"),
    ("glove", "style_b"): (f"{FEATURES_DIR}/glove_train_style_b.csv", f"{FEATURES_DIR}/glove_test_style_b.csv"),
    ("glove", "style_c"): (f"{FEATURES_DIR}/glove_train_style_c.csv", f"{FEATURES_DIR}/glove_test_style_c.csv"),
}

# ==========================================
# 2. Tracking Best Model
# ==========================================
best_f1 = -1
best_model = None
best_encoder = None
best_selector = None
best_feature_cols = None
best_info = None
best_class_weights = None
results = []

# ==========================================
# 3. Training Loop
# ==========================================
for (rep, style), (train_path, test_path) in DATASETS.items():
    if not os.path.exists(train_path):
        continue

    print(f"\n[Processing] {rep.upper()} | {style}")

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ["row_id", "ground_truth"]]

    X_train = train_df[feature_cols].values
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)[feature_cols].values

    # Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["ground_truth"])
    y_test = le.transform(test_df["ground_truth"])

    # ==========================================
    # Feature Selection (ONLY BOW)
    # ==========================================
    selector = None
    if rep == "bow":
        k = min(500, X_train.shape[1])
        selector = SelectKBest(chi2, k=k)

        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        print(f"-> Feature Selection: Kept {X_train.shape[1]} features")

    # ==========================================
    # CLASS WEIGHTS (Penalty for Positive = 50)
    # SAFE mapping using LabelEncoder
    # ==========================================
    class_weights = {
        le.transform(["negative"])[0]: 1,
        le.transform(["neutral"])[0]: 1,
        le.transform(["positive"])[0]: 50
    }

    # ==========================================
    # Random Forest Model
    # ==========================================
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    # Train
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # ==========================================
    # Evaluation
    # ==========================================
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results.append({
        "rep": rep,
        "style": style,
        "accuracy": acc,
        "f1_macro": f1
    })

    # ==========================================
    # Track Best Model
    # ==========================================
    if f1 > best_f1:
        best_f1 = f1
        best_model = rf
        best_encoder = le
        best_selector = selector
        best_feature_cols = feature_cols
        best_info = (rep, style)
        best_class_weights = class_weights

# ==========================================
# 4. SAVE BEST MODEL ONLY
# ==========================================
if best_model is not None:
    filename = f"random_forest_{best_info[0]}_{best_info[1]}_p50.pkl"
    path = os.path.join(SAVE_DIR, filename)

    save_obj = {
        "model": best_model,
        "label_encoder": best_encoder,
        "selector": best_selector,
        "feature_columns": best_feature_cols,
        "class_weights": best_class_weights,
        "representation": best_info[0],
        "style": best_info[1]
    }

    joblib.dump(save_obj, path)

    print("\n" + "="*50)
    print("BEST MODEL SAVED SUCCESSFULLY")
    print(f"Model: Random Forest")
    print(f"Representation: {best_info[0]}")
    print(f"Style: {best_info[1]}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"Saved At: {path}")
    print("="*50)

# ==========================================
# 5. RESULTS TABLE
# ==========================================
results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)
print("\nFINAL RESULTS:")
print(results_df)

# Visualization
plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="rep", y="f1_macro", hue="style")
plt.title("Random Forest Performance (Final Production Model)")
plt.show()