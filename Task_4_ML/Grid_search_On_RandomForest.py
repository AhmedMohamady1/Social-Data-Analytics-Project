import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
FEATURES_DIR = "Task_4_v2/split_style_bow_glove_outputs/three_style_features"

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

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["gini", "entropy"]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []

for (rep, style), (train_path, test_path) in DATASETS.items():

    print("\n" + "="*60)
    print(f"{rep.upper()} | {style}")
    print("="*60)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ["row_id", "ground_truth"]]

    # IMPORTANT: align features (fix previous bug)
    X_train = train_df[feature_cols]
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)[feature_cols]

    # Labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["ground_truth"])
    y_test = le.transform(test_df["ground_truth"])

    # Model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Train
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("BEST PARAMS:", grid.best_params_)
    print("CV BEST F1:", grid.best_score_)
    print("TEST ACC:", acc)
    print("TEST F1:", f1)

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    results.append({
        "rep": rep,
        "style": style,
        "accuracy": acc,
        "f1_macro": f1,
        "best_cv_f1": grid.best_score_
    })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1_macro", ascending=False)

print("\n FINAL GRIDSEARCH RESULTS")
print(results_df)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="rep", y="f1_macro", hue="style")
plt.title("Random Forest (GridSearch) Performance")
plt.show()