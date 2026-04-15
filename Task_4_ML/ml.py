# Imports & Setup
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE

# Global Configuration
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

OUTPUT_DIR = Path("Machine_Learning_Outputs")
PLOT_DIR = OUTPUT_DIR / "EDA_Visuals"
REPORT_DIR = OUTPUT_DIR / "Results"

for d in [PLOT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = Path("Task_4_v2/split_style_bow_glove_outputs/three_style_features")

# Load Dataset Paths (TRAIN/TEST ONLY)
DATASETS = {
    ("bow", "original"): (
        FEATURES_DIR / "bow_train_original.csv",
        FEATURES_DIR / "bow_test_original.csv"
    ),
    ("bow", "style_b"): (
        FEATURES_DIR / "bow_train_style_b.csv",
        FEATURES_DIR / "bow_test_style_b.csv"
    ),
    ("bow", "style_c"): (
        FEATURES_DIR / "bow_train_style_c.csv",
        FEATURES_DIR / "bow_test_style_c.csv"
    ),
    ("glove", "original"): (
        FEATURES_DIR / "glove_train_original.csv",
        FEATURES_DIR / "glove_test_original.csv"
    ),
    ("glove", "style_b"): (
        FEATURES_DIR / "glove_train_style_b.csv",
        FEATURES_DIR / "glove_test_style_b.csv"
    ),
    ("glove", "style_c"): (
        FEATURES_DIR / "glove_train_style_c.csv",
        FEATURES_DIR / "glove_test_style_c.csv"
    ),
}


# Helper Function (Sampling)
def get_sampler(rep, y_train):
    if rep == "bow":
        return RandomOverSampler(random_state=42)
    else:
        counts = np.bincount(y_train)
        min_class = np.min(counts[counts > 0])
        k_val = 1
        return BorderlineSMOTE(k_neighbors=k_val, random_state=42)
    

# Model Training Pipeline
results = []

for (rep, style), (train_path, test_path) in DATASETS.items():

    if not train_path.exists() or not test_path.exists():
        print(f"Missing: {rep}-{style}")
        continue

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ["row_id", "ground_truth"]]

    X_train = train_df[feature_cols].values
    X_test = test_df.reindex(columns=feature_cols, fill_value=0).values

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["ground_truth"])
    y_test = le.transform(test_df["ground_truth"])

    print("\n" + "="*70)
    print(f"{rep.upper()} | {style}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print("="*70)

    for model_name in ["DecisionTree", "NaiveBayes"]:

        steps = []

        # Scaling
        if rep == "bow":
            steps.append(("scaler", MinMaxScaler()))
            steps.append(("select", SelectKBest(chi2, k=int(0.2 * X_train.shape[1]))))
        else:
            steps.append(("scaler", StandardScaler()))

        # Sampling (train only)
        steps.append(("sampler", get_sampler(rep, y_train)))

        # Model
        if model_name == "DecisionTree":
            clf = DecisionTreeClassifier(class_weight={0:1, 1:1, 2:50}, random_state=42)
        else:
            clf = ComplementNB() if rep == "bow" else GaussianNB()

        steps.append(("clf", clf))

        model = ImbPipeline(steps)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_,
                    yticklabels=le.classes_)
        plt.title(f"{rep}-{style}-{model_name}")
        plt.savefig(PLOT_DIR / f"cm_{rep}_{style}_{model_name}.png")
        plt.close()

        # Save results
        results.append({
            "rep": rep,
            "style": style,
            "model": model_name,
            "accuracy": acc,
            "f1_macro": f1
        })

# Train Evaluation
y_train_pred = model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average="macro")

print("TRAIN accuracy:", train_acc)
print("TRAIN f1:", train_f1)



# Final Report
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1_macro", ascending=False)

results_df.to_csv(REPORT_DIR / "final_results.csv", index=False)

print("\n FINAL RESULTS")
print(results_df)

# Insight Visualization
plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="rep", y="f1_macro", hue="model")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(PLOT_DIR / "model_comparison.png")
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ===== Load best data =====
train_path, test_path = DATASETS[("glove", "style_b")]

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["row_id", "ground_truth"]).values
y_train = LabelEncoder().fit_transform(train_df["ground_truth"])

# ===== Pipeline =====
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# ===== Hyperparameter Grid =====
param_grid = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [3, 5, 10, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 5]
}

# ===== Grid Search =====
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

# ===== Best Model =====
print("\nBEST PARAMS:", grid.best_params_)
print("BEST CV F1:", grid.best_score_)




best_model = grid.best_estimator_

X_test = test_df.drop(columns=["row_id", "ground_truth"]).values
y_test = LabelEncoder().fit(train_df["ground_truth"]).transform(test_df["ground_truth"])

y_pred = best_model.predict(X_test)

print("TEST ACC:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))