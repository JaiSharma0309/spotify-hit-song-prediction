## @file hit_song.py
## @brief Predicts Spotify hit songs using Logistic Regression and SVM (RBF).
##
## This script loads an enriched Spotify audio-feature dataset, defines a hit
## threshold using track popularity, and trains two classification models
## (Logistic Regression and SVM with RBF kernel) to predict whether a track
## becomes a hit. It performs:
##  - preprocessing (numeric scaling and one-hot encoding),
##  - train/validation split,
##  - hyperparameter tuning using GridSearchCV,
##  - evaluation on a held-out test set,
##  - model comparison,
##  - interpretation using logistic regression coefficients,
##  - interpretation using permutation importance for SVM.
##
## The script prints:
##  - dataset structure,
##  - class balance,
##  - cross-validation scores,
##  - final test accuracies,
##  - best hyperparameters,
##  - feature effects and SVM feature importances.
##
## @author Jai Sharma
## @date 2025

import warnings
warnings.filterwarnings("ignore")  # keep terminal output clean

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt




## ============================================================================
## 1. Load dataset (all genres)
## ============================================================================

## @brief Load the enriched Spotify dataset from CSV.
## @details This assumes a file named "spotify_enriched.csv" is present in the
##          same directory as this script. Any auxiliary columns such as an
##          index or original "target" column are dropped if present.

df = pd.read_csv("spotify_enriched.csv")

# Drop junk columns if present
for col in ["Unnamed: 0", "target"]:
    if col in df.columns:
        df = df.drop(columns=[col])

print("Total tracks in dataset:", df.shape[0])

# Make sure we have track_popularity
df = df.dropna(subset=["track_popularity"]).copy()


## ============================================================================
## 2. Define hit label from popularity
## ============================================================================

## @brief Convert track popularity into a binary hit label.
## @details A track is labeled as a "hit" if its Spotify popularity is greater
##          than or equal to the 75th percentile (top 25% of songs).
##          The resulting label is stored in df["hit"].

hit_threshold = df["track_popularity"].quantile(0.75)
print("Hit threshold (track_popularity):", hit_threshold)

df["hit"] = (df["track_popularity"] >= hit_threshold).astype(int)

print("\nClass balance (0 = non-hit, 1 = hit):")
print(df["hit"].value_counts())
print("Name:", df["hit"].value_counts().name, ", dtype:", df["hit"].dtype)


## ============================================================================
## 3. Feature engineering and column selection
## ============================================================================

## @brief Construct numeric and categorical feature sets for modeling.
## @details
##  - Numeric features will be imputed and scaled.
##  - Categorical features will be imputed and one-hot encoded.
##  - Metadata columns not suitable for prediction are dropped.

# Release year from release_date
df["release_year"] = pd.to_datetime(
    df["release_date"], errors="coerce"
).dt.year

numeric_feats = [
    "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "liveness", "loudness", "speechiness",
    "tempo", "valence", "artist_popularity", "release_year"
]

categorical_feats = [
    "key", "mode", "time_signature", "album_type", "explicit"
]

# Remove columns we don't want to use as predictors
cols_to_drop = [
    "song_title", "artist", "spotify_id",
    "matched_title", "matched_artist",
    "release_date", "track_popularity", "genres"
]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])


## ============================================================================
## 4. Train–test split (golden rule: test set only once)
## ============================================================================

## @brief Split the data into training and test sets.
## @details Uses stratified sampling on the "hit" label to maintain class
##          balance in both splits.

X = df[numeric_feats + categorical_feats]
y = df["hit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)


## ============================================================================
## 5. Preprocessing pipeline (ColumnTransformer)
## ============================================================================

## @brief Build preprocessing steps for numeric and categorical features.
## @details
##  - Numeric features: median imputation + standard scaling
##  - Categorical features: most-frequent imputation + one-hot encoding
##  - Combined using ColumnTransformer and later wrapped in a sklearn Pipeline.

numeric_processor = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_processor = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", drop="if_binary")
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_processor, numeric_feats),
        ("cat", categorical_processor, categorical_feats),
    ]
)


## ============================================================================
## 6. Baseline: DummyClassifier
## ============================================================================

## @brief Compute a simple baseline using DummyClassifier.
## @details The baseline always predicts the most frequent class in the
##          training set. Its cross-validated accuracy is used as a sanity
##          check to ensure our real models are learning something useful.

dummy = DummyClassifier(strategy="most_frequent")
dummy_scores = cross_validate(dummy, X_train, y_train, cv=5)
print("\nDummy mean val accuracy:", np.mean(dummy_scores["test_score"]))


## ============================================================================
## 7. Logistic Regression model with GridSearchCV
## ============================================================================

## @brief Train and tune a Logistic Regression classifier.
## @details
##  - Wrapped in a Pipeline with the preprocessor.
##  - Hyperparameter C is selected using GridSearchCV with 5-fold CV.
##  - The best model is evaluated on the held-out test set.

log_reg_pipe = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000)
)

log_param_grid = {
    "logisticregression__C": [0.01, 0.1, 1, 10]
}

log_grid = GridSearchCV(
    log_reg_pipe,
    param_grid=log_param_grid,
    cv=5,
    n_jobs=-1
)

log_grid.fit(X_train, y_train)
log_best = log_grid.best_estimator_

log_cv = log_grid.best_score_
log_test_acc = log_best.score(X_test, y_test)

print("\n[LogReg] Best params:", log_grid.best_params_)
print("[LogReg] Best CV score:", log_cv)
print("[LogReg] Final test accuracy:", log_test_acc)


## ============================================================================
## 8. SVM with RBF kernel and GridSearchCV
## ============================================================================

## @brief Train and tune an SVM classifier with RBF kernel.
## @details
##  - Uses the same preprocessor as Logistic Regression.
##  - Hyperparameters C and gamma are tuned via GridSearchCV.
##  - The tuned SVM is compared against Logistic Regression on the test set.

svm_pipe = make_pipeline(
    preprocessor,
    SVC(kernel="rbf")
)

svm_param_grid = {
    "svc__C": [1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1]
}

svm_grid = GridSearchCV(
    svm_pipe,
    param_grid=svm_param_grid,
    cv=5,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)
svm_best = svm_grid.best_estimator_

svm_cv = svm_grid.best_score_
svm_test_acc = svm_best.score(X_test, y_test)

print("\n[SVM] Best params:", svm_grid.best_params_)
print("[SVM] Best CV score:", svm_cv)
print("[SVM] Final test accuracy:", svm_test_acc)


## ============================================================================
## 9. Test-set comparison summary
## ============================================================================

## @brief Print a compact comparison of baseline, Logistic Regression,
##        and SVM RBF on the held-out test set.

print("\n=== Model comparison on test set ===")
print(f"Dummy baseline:   {np.mean(dummy_scores['test_score']):.3f} (CV)")
print(f"LogReg test acc:  {log_test_acc:.3f}")
print(f"SVM RBF test acc: {svm_test_acc:.3f}")


## ============================================================================
## 10. Extract transformed feature names from ColumnTransformer
## ============================================================================

## @brief Recover feature names after preprocessing.
## @details
##  ColumnTransformer expands categorical features into many one-hot columns.
##  This step retrieves the final expanded feature names so we can interpret
##  model coefficients and permutation importance in a human-readable way.

ct = log_best.named_steps["columntransformer"]

try:
    full_feature_names = ct.get_feature_names_out()
except AttributeError:
    # Fallback: numeric + generic cat names if get_feature_names_out
    # is not available (older sklearn). Here we at least label numeric
    # features clearly.
    num_names = [f"num__{f}" for f in numeric_feats]
    full_feature_names = np.array(num_names)


## ============================================================================
## 11. Logistic Regression coefficient analysis
## ============================================================================

## @brief Inspect the learned weights from Logistic Regression.
## @details
##  - Coefficients are in the scaled / encoded feature space.
##  - Positive coefficients increase hit probability.
##  - Negative coefficients decrease hit probability.

print("\n=== Logistic Regression: feature effects (scaled space) ===")

log_model = log_best.named_steps["logisticregression"]
coefs = log_model.coef_[0]

coef_df = pd.DataFrame({
    "feature": full_feature_names,
    "coef": coefs
}).sort_values("coef", ascending=False)

print("\nTop 15 features increasing hit probability:")
print(coef_df.head(15).to_string(index=False))

print("\nTop 15 features decreasing hit probability:")
print(coef_df.tail(15).to_string(index=False))


## ============================================================================
## 12. SVM RBF permutation feature importance
## ============================================================================

## @brief Estimate feature importance for SVM using permutation importance.
## @details
##  - SVM with RBF kernel does not expose coefficients.
##  - permutation_importance measures how much shuffling each feature
##    harms test accuracy.
##  - The output is a ranked list of the most important features.

print("\n=== SVM RBF: permutation feature importance (test set) ===")

perm = permutation_importance(
    svm_best, X_test, y_test,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

# Ensure matching lengths between feature names and importance array
n_feat = min(len(full_feature_names), perm.importances_mean.shape[0])
names_for_perm = full_feature_names[:n_feat]
importances_for_perm = perm.importances_mean[:n_feat]

perm_df = pd.DataFrame({
    "feature": names_for_perm,
    "importance": importances_for_perm
}).sort_values("importance", ascending=False)

print("\nTop 20 features by permutation importance (SVM RBF):")
print(perm_df.head(20).to_string(index=False))


## ============================================================================
## 13. Visualization: violin plots, permutation bar chart, ROC curves
## ============================================================================

if __name__ == "__main__":
    # Create output directory for figures
    os.makedirs("figures", exist_ok=True)

    # --------------------------
    # 13.1 Violin plots by hit
    # --------------------------
    key_numeric = [
        col for col in ["danceability", "energy", "valence", "tempo"]
        if col in df.columns
    ]

    for col in key_numeric:
        fig, ax = plt.subplots(figsize=(6, 4))

        non_hit = df[df["hit"] == 0][col].dropna()
        hit = df[df["hit"] == 1][col].dropna()

        ax.violinplot(
            [non_hit.values, hit.values],
            positions=[0, 1],
            showmeans=True
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-hit", "Hit"])
        ax.set_ylabel(col)
        ax.set_title(f"{col} distribution by hit label")
        fig.tight_layout()
        fig.savefig(f"figures/violin_{col}.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------
    # 13.2 Permutation importance bar chart (SVM RBF)
    # -----------------------------------------------
    top_perm = perm_df.head(15).iloc[::-1]  # reverse for barh (largest at top)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_perm["feature"], top_perm["importance"])
    ax.set_xlabel("Mean decrease in accuracy")
    ax.set_title("Top 15 features by permutation importance (SVM RBF)")
    fig.tight_layout()
    fig.savefig("figures/perm_importance_svm.png", dpi=150)
    plt.close(fig)

    # ----------------------------
    # 13.3 ROC curves (LogReg vs SVM)
    # ----------------------------
    # Logistic Regression uses predicted probabilities
    y_score_log = log_best.predict_proba(X_test)[:, 1]

    # SVM uses decision_function (scores); that works fine for ROC
    if hasattr(svm_best.named_steps["svc"], "decision_function"):
        y_score_svm = svm_best.decision_function(X_test)
    else:
        y_score_svm = svm_best.predict_proba(X_test)[:, 1]

    fpr_log, tpr_log, _ = roc_curve(y_test, y_score_log)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)

    auc_log = auc(fpr_log, tpr_log)
    auc_svm = auc(fpr_svm, tpr_svm)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_log, tpr_log, label=f"LogReg (AUC = {auc_log:.3f})")
    ax.plot(fpr_svm, tpr_svm, label=f"SVM RBF (AUC = {auc_svm:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves on test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/roc_logreg_vs_svm.png", dpi=150)
    plt.close(fig)
