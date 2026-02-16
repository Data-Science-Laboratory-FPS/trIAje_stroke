# ===============================================================
# Generic binary classification pipeline using FLAML
# Cohort-agnostic (no stroke-specific references)
# ===============================================================

import pandas as pd
import numpy as np
import seaborn as sns
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
import os
import pyarrow.parquet as pq
from typing import Optional, List, Union

import kbase.preprocessing as dp 
from kbase.config import settings

protocol_features = {
    # 16: Dyspnea / Respiratory distress (Asfixia)
    16: [
        # 1. Asfixia start
        'q1_a', 'q1_b',
        # 2. Previous diseases
        'q2_a', 'q2_b', 'q2_c', 'q2_d', 'q2_e', 'q2_f', 'q2_g', 'q2_h',
        # 3. Speech difficulty
        'q3_a', 'q3_b', 'q3_c',
        # 4. Associated symptoms
        'q4_a', 'q4_b', 'q4_c', 'q4_d', 'q4_e', 'q4_f', 'q4_g', 
        'q4_h', 'q4_i', 'q4_j', 'q4_k', 'q4_l', 'q4_m'
    ],
    
    # 23: Chest Pain (Dolor Torácico)
    23: [
        # 1. Heart history
        'q1_a', 'q1_b', 'q1_c', 'q1_d',
        # 2. Location
        'q2_a', 'q2_b',
        # 3. Type of pain
        'q3_a', 'q3_b', 'q3_c', 'q3_d',
        # 4. Relation to movement/rest
        'q4_a', 'q4_b', 'q4_c', 
        # 5. Associated symptoms
        'q5_a', 'q5_b', 'q5_c', 'q5_d', 'q5_e', 'q5_f',
        # 6. Change/Improvement with meds
        'q6_a', 'q6_b', 'q6_c', 'q6_d'
    ],

    # 54: Stroke / ACV
    54: [
        # 1. Gaze / Eyes
        'q1_a', 'q1_b',
        # 2. Breathing
        'q2_a', 'q2_b',
        # 3. Main complaint (Face, Arm, Speech, etc.)
        'q3_a', 'q3_b', 'q3_c', 'q3_d', 'q3_e', 'q3_f',
        # 4. First time occurrence
        'q4_a', 'q4_b',
        # 5. Time of evolution (Numerical/String)
        'q5_a', 
        # 6. Lifestyle / Autonomy
        'q6_a', 'q6_b'
    ],

    # 36 & 58: Cardiac Arrest / Unconsciousness
    'cardiac_arrest': [
        # 1. Can speak?
        'q1_a', 'q1_b',
        # 2. Eyes/Gaze
        'q2_a', 'q2_b',
        # 3. Breathing quality
        'q3_a', 'q3_b', 'q3_d', # Nota: Tu protocolo salta de b a d
        # 4. Reaction to stimulus
        'q4_a', 'q4_b',
        # 5. Previous diseases
        'q5_a', 'q5_b', 'q5_c', 'q5_d', 'q5_e', 'q5_f',
        # 6. Time of evolution
        'q6_a',
        # 7. Previous episodes
        'q7_a', 'q7_b'
    ]
}

def run_binary_automl_model(
    cohort_name: str,
    target_column: str,
    demand_code: Optional[Union[int, List[int]]] = None,
    triage_value: Optional[int] = None,
    time_budget: int = 600,
    test_size: float = 0.2,
    seed: int = 42,
    min_age: Optional[int] = None,
    optimize_metric: Optional[str] = None,
    n_splits_cv: int = 5,
    optimize_beta: int = 2,
    plot_feature_importance: bool = True,
    use_permutation_importance: bool = False,
) -> dict:

    
    """
    Runs a complete binary classification pipeline using FLAML AutoML.

    Parameters
    ----------
    cohort_name : str
        Name of the cohort (used for logging only).
    target_column : str
        Binary target column.
    demand_code : Optional[Union[int, List[int]]]
        If provided, filters df_model to demand_type_1.
        Can be a single integer or a list of integers (e.g., [36, 58]).
    triage_value : Optional[int]
        If provided, filters to triage == triage_value.
        If triage_value == 0, q1-q7 are removed from the feature set.
    time_budget : int
        FLAML training budget in seconds.
    test_size : float
        Proportion of test split.
    seed : int
        Random seed.
    min_age : Optional[int]
        Optional age filter (>= min_age). If None, no age filtering is applied.
    optimize_metric : Optional[str]
        Metric to optimize during training.
    n_splits_cv : int
        Number of CV folds.
    optimize_beta : int
        Beta for F-beta threshold optimization (beta > 1 favors recall).
    plot_feature_importance : bool
        Whether to compute and plot feature importance.
    use_permutation_importance : bool
        If True, use permutation importance (slower but more accurate).
        If False, use built-in feature importance from the model (faster).

    Returns
    -------
    dict
        Dictionary with model, metrics, threshold and feature importance.
    """

    print(f"=== Running the model for {cohort_name} cohort ===")

    if triage_value is None:
        print("Triage subset: ALL patients (no triage-based filtering)")
    elif triage_value == 1:
        print("Triage subset: WITH structured triage information")
    elif triage_value == 0:
        print("Triage subset: WITHOUT structured triage information")
    else:
        raise ValueError("triage_value must be None, 0, or 1")

    # -----------------------------
    # Data loading & column selection
    # -----------------------------

    # Load preprocessed cleaned table
    df = pq.read_table(
        os.path.join(settings.source_tables_path, settings.triaje_table_cleaned_path)
    ).to_pandas()

    # Select base columns 
    base_cols = [
        "age",
        "sex",
        "demand_type_1",
        "p1_real",
        "day_week",
        "time_of_day",
        "month",
        "season",
        "year",
        "triage"
    ]

    # Find all columns generated during One-Hot Encoding that start with 'q'
    triage_questions_cols = [col for col in df.columns if col.startswith('q')]
    modelling_cols = base_cols + triage_questions_cols

    # Select columns for final modeling
    selected_cols = [c for c in modelling_cols if c in df.columns]
    df_model = df[selected_cols].copy()

    df_model = dp.transform_column_dtypes(df_model)

    # -----------------------------
    # Basic validation
    # -----------------------------
    if target_column not in df_model.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    if "triage" not in df_model.columns:
        raise ValueError("Required column 'triage' not found in dataframe")

    # -----------------------------
    # Cohort & triage filtering
    # -----------------------------
    if demand_code is not None:
        if "demand_type_1" not in df_model.columns:
            raise ValueError("'demand_type_1' column not found but demand_code was provided")
        
        # Handle both single integer and list of integers
        if isinstance(demand_code, (list, tuple)):
            df_model = df_model[df_model["demand_type_1"].isin(demand_code)].copy()
            print(f"Filtering by demand_type_1 in {demand_code}")
        else:
            df_model = df_model[df_model["demand_type_1"] == demand_code].copy()
            print(f"Filtering by demand_type_1 == {demand_code}")

    if triage_value is not None:
        df_model = df_model[df_model["triage"] == triage_value].copy()

        # If no triage questions were asked, remove triage questions/responses from the feature set
        if triage_value == 0:
            df_model = df_model.drop(columns=triage_questions_cols)

    if min_age is not None:
        if "age" not in df_model.columns:
            raise ValueError("'age' column not found but min_age was provided")
        df_model = df_model[df_model["age"] >= min_age].copy()
        print(f"Filtering by age >= {min_age}")

    # Drop triage columns not present in the specific demand type
    if demand_code is not None and protocol_features is not None:
            # Determine the correct key for protocol_features
            rule_key = None
            if isinstance(demand_code, (list, tuple)):
                if set(demand_code) == {36, 58}:
                    rule_key = 'cardiac_arrest'
                # Check if it's a single-item list that matches a protocol key
                elif len(demand_code) == 1 and demand_code[0] in protocol_features:
                    rule_key = demand_code[0]
            elif demand_code in protocol_features:
                rule_key = demand_code
    
            # Apply pruning if a rule was found
            if rule_key is not None:
                allowed_cols = protocol_features[rule_key]
                current_q_cols = [c for c in df_model.columns if c.startswith('q')]
                cols_to_drop = [c for c in current_q_cols if c not in allowed_cols]
                
                if cols_to_drop:
                    df_model = df_model.drop(columns=cols_to_drop)
                    print(f"Pruning: Dropped {len(cols_to_drop)} columns for protocol {rule_key}")
                    
    # Drop columns after cohort filtering
    cols_to_drop = ["demand_type_1", "triage"]
    cols_to_drop = [c for c in cols_to_drop if c in df_model.columns]
    if cols_to_drop:
        df_model = df_model.drop(columns=cols_to_drop)

    print(f"\nFinal cohort size: {len(df_model)}")
    print("Outcome: ", df_model[target_column].value_counts(dropna=False), "\n")
    print(df_model.info())

    if optimize_metric is None:
        print("\nTraining metric to optimize: FLAML default (auto)")
    else:
        print(f"\nTraining metric to optimize: {optimize_metric}\n")

    # -----------------------------
    # Train / test split
    # -----------------------------

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    # Ensure categorical consistency
    # This prevents the 'categorical_feature do not match' error in LightGBM/FLAML
    categorical_cols = X.select_dtypes(include=['category']).columns
    
    for col in categorical_cols:
        # 1. Convert to string and then back to category to reset the label dictionary
        # 2. This ensures that X_train and X_test share the exact same internal mapping
        #    even if one split is missing a specific category value.
        X[col] = X[col].astype(str).astype('category')
        
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
        metric=optimize_metric,
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
    # Threshold optimization (decision threshold)
    # -----------------------------
    print(f"\n--- Optimizing decision threshold ---")

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

    # -----------------------------
    # Confusion Matrix Plot
    # -----------------------------
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    labels = ['Negative', 'Positive']

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.title('Classification Pipeline Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Metrics computation (clinical interpretation)
    # -----------------------------
    # Confusion matrix components
    tn, fp, fn, tp = conf_matrix.ravel()

    # Core rates
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    precision = precision_score(y_test, y_test_pred) if (tp + fp) > 0 else np.nan
    recall = recall_score(y_test, y_test_pred) if (tp + fn) > 0 else np.nan

    # Triage-oriented metrics
    overtriage = 1 - precision if not np.isnan(precision) else np.nan
    undertriage = 1 - npv if not np.isnan(npv) else np.nan

    # Error rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    # Likelihood ratios
    lr_pos = recall / (1 - specificity) if specificity < 1 else np.nan
    lr_neg = (1 - recall) / specificity if specificity > 0 else np.nan

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc_score(y_test, y_test_prob),
        "pr_auc": average_precision_score(y_test, y_test_prob),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1_score(y_test, y_test_pred),
        "overtriage": overtriage,
        "undertriage": undertriage,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "lr_positive": lr_pos,
        "lr_negative": lr_neg,
    }

    # -----------------------------
    # Print metrics in a nice format
    # -----------------------------
    print("\n" + "="*60)
    print(f"{'CLASSIFICATION REPORT':^60}")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=labels))

    print("\n" + "="*60)
    print(f"{'PERFORMANCE METRICS (%)':^60}")
    print("="*60)
    
    print(f"\n{'Classification Metrics:':<30}")
    print(f"  Accuracy:                    {metrics['accuracy']*100:.2f}%")
    print(f"  ROC-AUC:                     {metrics['roc_auc']*100:.2f}%")
    print(f"  PR-AUC:                      {metrics['pr_auc']*100:.2f}%")
    print(f"  F1 Score:                    {metrics['f1']*100:.2f}%")
    
    print(f"\n{'Positive Class Performance:':<30}")
    print(f"  Precision (PPV):             {metrics['precision']*100:.2f}%")
    print(f"  Recall (Sensitivity):        {metrics['recall']*100:.2f}%")
    
    print(f"\n{'Negative Class Performance:':<30}")
    print(f"  Specificity:                 {metrics['specificity']*100:.2f}%")
    print(f"  NPV:                         {metrics['npv']*100:.2f}%")
    
    print(f"\n{'Triage-Specific Metrics:':<30}")
    print(f"  Overtriage Rate:             {metrics['overtriage']*100:.2f}%")
    print(f"  Undertriage Rate:            {metrics['undertriage']*100:.2f}%")
    
    print(f"\n{'Error Rates:':<30}")
    print(f"  False Positive Rate:         {metrics['false_positive_rate']*100:.2f}%")
    print(f"  False Negative Rate:         {metrics['false_negative_rate']*100:.2f}%")
    
    print(f"\n{'Likelihood Ratios (Abs):':<30}")
    print(f"  LR+:                         {metrics['lr_positive']:.4f}")
    print(f"  LR-:                         {metrics['lr_negative']:.4f}")
    
    print("\n" + "="*60 + "\n")

    # -----------------------------
    # Feature importance
    # -----------------------------
    importance_df = None
    
    if plot_feature_importance:
        if use_permutation_importance:
            # Option 1: Permutation Importance (slower, model-agnostic, more reliable)
            print("--- Computing permutation importance (this may take a while) ---")
            
            try:
                # Create a proper wrapper with fit method
                class ClassifierWrapper:
                    def __init__(self, model):
                        self.model = model
                        self._estimator_type = "classifier"
                        self.classes_ = np.array([0, 1])
                    
                    def fit(self, X, y):
                        # Dummy fit method (model is already trained)
                        return self
                    
                    def predict(self, X):
                        return self.model.predict(X)
                    
                    def predict_proba(self, X):
                        return self.model.predict_proba(X)
                
                wrapped_model = ClassifierWrapper(automl)
                
                perm_imp = permutation_importance(
                    wrapped_model,
                    X_test,
                    y_test,
                    scoring="roc_auc",
                    n_repeats=5,
                    random_state=seed,
                    n_jobs=-1,
                )
                
                importance_df = (
                    pd.DataFrame({
                        "feature": X.columns,
                        "importance": perm_imp.importances_mean,
                        "importance_std": perm_imp.importances_std,
                    })
                    .sort_values("importance", ascending=False)
                )
                
                print(f"Permutation importance computed successfully.")
                
            except Exception as e:
                print(f"Warning: Permutation importance failed with error: {e}")
                print("Falling back to built-in feature importance...")
                use_permutation_importance = False
        
        if not use_permutation_importance:
            # Option 2: Built-in Feature Importance (fast, model-specific)
            print("--- Computing built-in feature importance ---")
            
            try:
                # Get the underlying model from FLAML
                best_model = automl.model
                
                # Get feature names (accounting for dropped features)
                feature_names = X_train.columns.tolist()
                
                # Try to get feature importance from the model
                if hasattr(best_model, 'feature_importances_'):
                    # Tree-based models (LightGBM, XGBoost, RandomForest)
                    importances = best_model.feature_importances_
                    
                    # Ensure the length matches
                    if len(importances) == len(feature_names):
                        importance_df = (
                            pd.DataFrame({
                                "feature": feature_names,
                                "importance": importances,
                            })
                            .sort_values("importance", ascending=False)
                        )
                        print(f"Built-in feature importance computed successfully.")
                    else:
                        print(f"Warning: Importance length ({len(importances)}) doesn't match features ({len(feature_names)})")
                        importance_df = None
                    
                elif hasattr(best_model, 'coef_'):
                    # Linear models (LogisticRegression, etc.)
                    importances = np.abs(best_model.coef_[0])
                    
                    if len(importances) == len(feature_names):
                        importance_df = (
                            pd.DataFrame({
                                "feature": feature_names,
                                "importance": importances,
                            })
                            .sort_values("importance", ascending=False)
                        )
                        print(f"Built-in feature importance computed successfully.")
                    else:
                        print(f"Warning: Coefficient length ({len(importances)}) doesn't match features ({len(feature_names)})")
                        importance_df = None
                    
                else:
                    print(f"Warning: Model type '{type(best_model).__name__}' does not have built-in feature importance.")
                    importance_df = None
                
            except Exception as e:
                print(f"Warning: Could not extract feature importance: {e}")
                importance_df = None
        
        # Plotting
        if importance_df is not None and len(importance_df) > 0:
            top_k = min(15, len(importance_df))
            plot_df = importance_df.head(top_k)
            
            plt.figure(figsize=(10, 6))
            plt.barh(plot_df["feature"][::-1], plot_df["importance"][::-1])
            
            if use_permutation_importance:
                plt.title("Top Features (Permutation Importance, ROC-AUC)")
                plt.xlabel("Mean decrease in ROC-AUC")
            else:
                plt.title(f"Top Features (Built-in Importance - {automl.best_estimator})")
                plt.xlabel("Feature Importance")
            
            plt.tight_layout()
            plt.show()
            
            # Print top features
            print(f"\nTop {top_k} most important features:")
            for idx, row in plot_df.iterrows():
                print(f"  {row['feature']:<20} {row['importance']:.6f}")
    
    # -----------------------------
    # Build summary DataFrame (one-row, ready to concatenate)
    # -----------------------------
    summary_df = (
        pd.DataFrame(metrics, index=[0])
        .assign(
            cohort=cohort_name,
            demand_code=str(demand_code) if demand_code is not None else None,  # Convert to string for consistency
            triage_value=triage_value,
            optimize_metric=optimize_metric,
            optimize_beta=optimize_beta,
            threshold=best_threshold,
            n_samples=len(df_model),
            n_train=len(X_train),
            n_test=len(X_test),
        )
    )

    return {
        "automl": automl,
        "threshold": best_threshold,
        "metrics": metrics,
        "metrics_df": summary_df,
        "feature_importance": importance_df,
    }