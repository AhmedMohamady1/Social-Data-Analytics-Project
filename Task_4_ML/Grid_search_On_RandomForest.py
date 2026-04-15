# ==========================================
# 0. IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2

import warnings
warnings.filterwarnings("ignore")


# ==========================================
# 1. PATHS & CONFIG
# ==========================================
FEATURES_DIR = "Task_4_v2/split_style_bow_glove_outputs/three_style_features"
# SAVE_DIR = "Task_4_ML/GridSearch_RF"
# os.makedirs(SAVE_DIR, exist_ok=True)

DATASETS = {
    ("bow", "original"): (f"{FEATURES_DIR}/bow_train_original.csv", f"{FEATURES_DIR}/bow_test_original.csv"),
    ("bow", "style_b"): (f"{FEATURES_DIR}/bow_train_style_b.csv", f"{FEATURES_DIR}/bow_test_style_b.csv"),
    ("bow", "style_c"): (f"{FEATURES_DIR}/bow_train_style_c.csv", f"{FEATURES_DIR}/bow_test_style_c.csv"),
    ("glove", "original"): (f"{FEATURES_DIR}/glove_train_original.csv", f"{FEATURES_DIR}/glove_test_original.csv"),
    ("glove", "style_b"): (f"{FEATURES_DIR}/glove_train_style_b.csv", f"{FEATURES_DIR}/glove_test_style_b.csv"),
    ("glove", "style_c"): (f"{FEATURES_DIR}/glove_train_style_c.csv", f"{FEATURES_DIR}/glove_test_style_c.csv"),
}


# ==========================================
# 2. TRACK BEST MODEL
# ==========================================
best_f1 = -1
best_model = None
best_info = None
results = []


# ==========================================
# 3. GRID SEARCH LOOP
# ==========================================
for (rep, style), (train_path, test_path) in DATASETS.items():

    if not os.path.exists(train_path):
        continue

    print(f"\n==============================")
    print(f"[PROCESSING] {rep.upper()} | {style}")
    print(f"==============================")

    # --------------------------
    # Load data
    # --------------------------
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ["row_id", "ground_truth"]]

    X_train = train_df[feature_cols].values
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)[feature_cols].values

    # --------------------------
    # Label Encoding
    # --------------------------
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["ground_truth"])
    y_test = le.transform(test_df["ground_truth"])

    # --------------------------
    # Feature Selection (BoW only)
    # --------------------------
    selector = None
    if rep == "bow":
        k = min(500, X_train.shape[1])
        selector = SelectKBest(chi2, k=k)

        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        print(f"Kept features: {X_train.shape[1]}")

    # --------------------------
    # Class Weights
    # --------------------------
    classes = list(le.classes_)

    class_weights = {i: 1 for i in range(len(classes))}
    if "positive" in classes:
        class_weights[le.transform(["positive"])[0]] = 50


    # ==========================================
    # 4. MODEL + GRID SEARCH
    # ==========================================
    rf = RandomForestClassifier(
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_

    # --------------------------
    # Predict
    # --------------------------
    y_pred = best_estimator.predict(X_test)

    # --------------------------
    # Evaluation
    # --------------------------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\nBest Params:", grid.best_params_)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-macro: {f1:.4f}")

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results.append({
        "rep": rep,
        "style": style,
        "accuracy": acc,
        "f1_macro": f1
    })

    # --------------------------
    # Track best model
    # --------------------------
    if f1 > best_f1:
        best_f1 = f1
        best_model = best_estimator
        best_info = (rep, style)
        best_le = le
        best_selector = selector
        best_features = feature_cols
        best_params = grid.best_params_


# ==========================================
# 5. SAVE BEST MODEL
# ==========================================
# if best_model is not None:
#     save_path = os.path.join(
#         SAVE_DIR,
#         f"rf_gridsearch_{best_info[0]}_{best_info[1]}.pkl"
#     )

#     joblib.dump({
#         "model": best_model,
#         "label_encoder": best_le,
#         "selector": best_selector,
#         "feature_columns": best_features,
#         "best_params": best_params,
#         "representation": best_info[0],
#         "style": best_info[1]
#     }, save_path)

    print("\n==============================")
    print("BEST MODEL SAVED")
    print("Representation:", best_info[0])
    print("Style:", best_info[1])
    print("F1-macro:", best_f1)
    # print("Saved to:", save_path)
    print("==============================")



# ==========================================
# 6. RESULTS SUMMARY
# ==========================================
results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)

print("\nFINAL RESULTS:")
print(results_df)