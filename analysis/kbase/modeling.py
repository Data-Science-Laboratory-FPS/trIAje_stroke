# ===============================================================
# Generic binary classification pipeline using FLAML
# Cohort-agnostic (no stroke-specific references)
# ===============================================================

import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


from typing import Optional, List


def run_binary_automl_model(
    cohort_name: str,
    target_column: str,
    demand_code: Optional[int] = None,
    time_budget: int = 600,
    test_size: float = 0.2,
    seed: int = 42,
    min_age: Optional[int] = None,
    drop_columns: Optional[List[str]] = None,
    n_splits_cv: int = 5,
    optimize_beta: int = 2,
    plot_feature_importance: bool = True,
):
    
    """
    Runs a complete binary classification pipeline using FLAML AutoML.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already loaded).
    cohort_name : str
        Name of the cohort (used for logging only).
    target_column : str
        Binary target column.
    time_budget : int
        FLAML training budget in seconds.
    test_size : float
        Proportion of test split.
    seed : int
        Random seed.
    min_age : Optional[int]
        Optional age filter (>= min_age).
    cohort_filter : pd.Series | None
        Optional boolean mask to define the cohort.
    drop_columns : Optional[List[str]]
        Columns to drop before modelling.
    n_splits_cv : int
        Number of CV folds.
    optimize_beta : int
        Beta for F-beta threshold optimisation (beta > 1 favours recall).
    plot_feature_importance : bool
        Whether to compute and plot permutation importance.

    Returns
    -------
    dict
        Dictionary with model, metrics, threshold and feature importance.
    """

    print(f"=== Running the model for {cohort_name} cohort ===")

    # -----------------------------
    # Data loading & column selection
    # -----------------------------
    import os
    import pyarrow.parquet as pq
    import kbase.preprocessing as dp
    from kbase.config import settings

    modelling_cols = [
        "age",
        "sex",
        "demand_type_1",
        "p1_real",
        "day_week",
        "time_of_day",
        "month",
        "season",
        "year",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
    ]

    df = pq.read_table(
        os.path.join(settings.source_tables_path, settings.triaje_table_cleaned_path)
    ).to_pandas()

    selected_cols = [c for c in modelling_cols if c in df.columns]
    df_model = df[selected_cols].copy()


    # -----------------------------
    # Basic validation
    # -----------------------------
    if target_column not in df_model.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # -----------------------------
    # Cohort filtering
    # -----------------------------
    if demand_code is not None:
        if "demand_type_1" not in df_model.columns:
            raise ValueError("'demand_type_1' column not found but demand_code was provided")
        df_model = df_model[df_model["demand_type_1"] == demand_code].copy()

    if min_age is not None:
        if "age" not in df_model.columns:
            raise ValueError("'age' column not found but min_age was provided")
        df_model = df_model[df_model["age"] >= min_age].copy()

    if drop_columns is not None:
        df_model = df_model.drop(columns=drop_columns, errors="ignore")

    print(f"Final cohort size: {len(df_model)}")
    print(df_model[target_column].value_counts(dropna=False))

    # -----------------------------
    # Train / test split
    # -----------------------------
    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    # -----------------------------
    # Sample weights (class imbalance)
    # -----------------------------
    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # -----------------------------
    # AutoML training
    # -----------------------------
    print(f"\n--- Starting AutoML (Budget: {time_budget}s) ---")

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        sample_weight=sample_weight,
        time_budget=time_budget,
        metric="auto",
        task="classification",
        eval_method="cv",
        n_splits=n_splits_cv,
        seed=seed,
        verbose=1,
        log_training_metric=True,
    )

    print("\n--- Training completed ---")
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best CV score:  {1 - automl.best_loss:.4f}")

    # -----------------------------
    # Threshold optimisation (F-beta)
    # -----------------------------
    print(f"\n--- Optimising decision threshold (F{optimize_beta}) ---")

    y_train_prob = automl.predict_proba(X_train)[:, 1]
    y_test_prob = automl.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.05, 0.95, 0.01)
    fbeta_scores = [
        fbeta_score(y_train, (y_train_prob >= t).astype(int), beta=optimize_beta)
        for t in thresholds
    ]

    best_threshold = thresholds[np.argmax(fbeta_scores)]
    print(f"Optimal threshold: {best_threshold:.3f}")

    # -----------------------------
    # Test set evaluation
    # -----------------------------
    print("\n--- Test set results ---")

    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_test_prob),
        "pr_auc": average_precision_score(y_test, y_test_prob),
        "f1": f1_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("-" * 40)
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    print("-" * 40)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_test_pred))

    # -----------------------------
    # Permutation importance
    # -----------------------------
    importance_df = None

    if plot_feature_importance:
        print("\n--- Permutation feature importance ---")
        perm_imp = permutation_importance(
            automl,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=5,
            random_state=seed,
            n_jobs=-1,
        )

        importance_df = (
            pd.DataFrame({
                "feature": X.columns,
                "importance": perm_imp.importances_mean,
            })
            .sort_values("importance", ascending=False)
            .head(15)
        )

        print(importance_df.to_string(index=False))

        plt.figure(figsize=(10, 6))
        plt.barh(
            importance_df["feature"][::-1],
            importance_df["importance"][::-1],
        )
        plt.title("Top features (permutation importance)")
        plt.xlabel("Importance (Δ PR-AUC)")
        plt.tight_layout()
        plt.show()

    return {
        "automl": automl,
        "threshold": best_threshold,
        "metrics": metrics,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "feature_importance": importance_df,
    }
