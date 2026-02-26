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
    classification_report, average_precision_score, 
    log_loss
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
import pyarrow.parquet as pq
from typing import Optional, List, Union
import shap
import warnings

import kbase.preprocessing as dp 
import kbase.eda as eda 
from kbase.config import settings

# This ensures .info() shows up to 50 columns by default
pd.set_option('display.max_info_columns', 50)
# This ensures that when you print a DataFrame, you see 50 rows
pd.set_option('display.max_rows', 50)

protocol_questions = {
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

def triage_print(triage_value, cohort_name, demand_code):
    """ Prints the initial configuration of the model run """
    print(f"=== Running the model for {cohort_name} cohort - code {demand_code} ===")

    if triage_value is None:
        print("Triage subset: ALL patients (no triage-based filtering)")
    elif triage_value == 1:
        print("Triage subset: WITH structured triage information")
    elif triage_value == 0:
        print("Triage subset: WITHOUT structured triage information")
    else:
        raise ValueError("triage_value must be None, 0, or 1")

def data_load_col_selection(target_column, triage_value, include_embeddings)-> pd.DataFrame:
    """Loads data and performs initial feature selection based on settings"""
    # Load preprocessed cleaned table
    df = pq.read_table(
        os.path.join(settings.source_tables_path, settings.triaje_table_cleaned_path)
    ).to_pandas()

    # Select base columns 
    base_cols = [
        "age",
        "sex",
        "demand_type_1",
        target_column,
        "day_week",
        "time_of_day",
        "month",
        "season",
        "location_patient",
        "triage"
    ]

    # Initialize modelling columns with base columns
    modelling_cols = base_cols.copy()
    # Add triage questions (One-Hot Encoded columns)
    if triage_value != 0:
        modelling_cols += [col for col in df.columns if col.startswith('q')]
    # Include NLP text embeddings if flag is set to True
    if include_embeddings:
        embedding_cols = [col for col in df.columns if col.startswith('emb_')]
        modelling_cols += embedding_cols
        print(f"Feature Set: Including {len(embedding_cols)} text embedding dimensions.")
            
    # Select columns for final modeling
    selected_cols = [c for c in modelling_cols if c in df.columns]
    df_model = df[selected_cols].copy()
    # Transform data types
    df_model = dp.transform_column_dtypes(df_model)

    return df_model

def col_validation(df, target_column):
    """Validates that essential columns are present in the dataframe"""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    if "triage" not in df.columns:
        raise ValueError("Required column 'triage' not found in dataframe")

def data_filtering(df, target_column, demand_code, 
                   triage_value, min_age, export_table, protocol_features) -> pd.DataFrame:
    """Applies cohort, triage, age, and protocol-specific filters"""

    # Filter by valid values in target_column
    if target_column is not None:
        df = df.dropna(subset=[target_column])
        if target_column not in df.columns:
            raise ValueError(f"{target_column} column not found but demand_code was provided")
        
    # Filter by demand_code
    if demand_code is not None:
        if "demand_type_1" not in df.columns:
            raise ValueError("'demand_type_1' column not found but demand_code was provided")
        
        # Handle both single integer and list of integers
        if isinstance(demand_code, (list, tuple)):
            df = df[df["demand_type_1"].isin(demand_code)].copy()
            print(f"Filtering by demand_type_1 in {demand_code}")
        else:
            df = df[df["demand_type_1"] == demand_code].copy()
            print(f"Filtering by demand_type_1 == {demand_code}")

    # Filter by triage patients
    if triage_value is not None:
        df = df[df["triage"] == triage_value].copy()

    # Filter by min age
    if min_age is not None:
        if "age" not in df.columns:
            raise ValueError("'age' column not found but min_age was provided")
        df = df[df["age"] >= min_age].copy()
        print(f"Filtering by age >= {min_age}")

    # Drop triage columns not present in the specific demand type
    if demand_code is not None and protocol_features is not None:
        # Determine the correct key for protocol_features
        rule_key = None
        # Normalize demand_code to a list for consistent comparison
        codes_list = demand_code if isinstance(demand_code, (list, tuple)) else [demand_code]

        # Case 1: Cardiac Arrest (Multiple codes)
        if set(codes_list) == {36, 58}:
            rule_key = 'cardiac_arrest'
        # Case 2: Single code protocols (16, 23, 54)
        elif len(codes_list) == 1 and codes_list[0] in protocol_features:
            rule_key = codes_list[0]

        # Apply pruning if a rule was found
        if rule_key is not None:
            allowed_cols = protocol_features[rule_key]
            current_q_cols = [c for c in df.columns if c.startswith('q')]
            cols_to_drop = [c for c in current_q_cols if c not in allowed_cols]
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                print(f"Pruning: Dropped {len(cols_to_drop)} columns for protocol {rule_key}")
        # Integrated warning if no protocol matches the demand_code
        else:
            print(f"⚠️ Warning: No specific protocol found for demand_code {demand_code}")

    # Clean column names: remove 'q' prefix and underscores for readability
    df.columns = [col.replace('q', '', 1).replace('_', '') 
                  if col.startswith('q') else col for col in df.columns]
                   
    # Drop columns after cohort filtering
    cols_to_drop = ["demand_type_1"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Export of preprocessed, ready for modeling, table
    if export_table:    
        rule_to_path = {
            'cardiac_arrest': settings.cardiacarrest_table_modeling,
            16: settings.dyspnea_table_modeling,
            23: settings.chestpain_table_modeling,
            54: settings.stroke_table_modeling,
        }
        export_path = rule_to_path.get(rule_key)
        
        if export_path is None:
            raise ValueError(f"Unknown export path: {rule_key}")
        
        df.to_parquet(
            os.path.join(settings.source_tables_path, export_path),
            index=False
        )

        # Check if the export file was created correctly
        if os.path.exists(export_path):
            file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
            print(f"✅ Export successful!")
            print(f"--- Path: {export_path}")
            print(f"--- Size: {file_size_mb:.2f} MB")
        else:
            print(f"❌ Export failed: File not found at {export_path}")

    print(f"\nFinal cohort size for {target_column}: {len(df)}")
    print("Outcome distribution: ", df[target_column].value_counts(dropna=False), "\n")
    # Show info for only the first 50 columns
    df.iloc[:, :50].info()

    return df

def train_test_split_weights(df, target_column, test_size, seed):
    """Splits data into train/test sets and computes sample weights for imbalance"""
    X = df.drop(columns=[target_column])
    y = df[target_column]

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

    # Sample weights (class imbalance)
    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    return X_train, X_test, y_train, y_test, sample_weight

def run_automl_training(X_train, y_train, sample_weight, time_budget, 
                        optimize_metric, n_splits_cv, seed, task):
    """Executes the FLAML AutoML optimization process"""
    print(f"\n--- Starting AutoML ({task.upper()}) | Budget: {time_budget}s ---")
    if optimize_metric is None:
        print("Training metric: FLAML default")
    else:
        print(f"Training metric: {optimize_metric}")
        
    automl = AutoML()
    automl.fit(
        X_train=X_train, 
        y_train=y_train, 
        sample_weight=sample_weight,
        time_budget=time_budget, 
        metric=optimize_metric, 
        task=task,
        eval_method="cv", 
        n_splits=n_splits_cv, 
        seed=seed, 
        verbose=1,
        log_training_metric=True,
    )
    
    print("\n--- Training completed ---")
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best CV score:  {1 - automl.best_loss:.4f}")
    
    return automl

def optimize_threshold(y_train, y_train_prob, optimize_beta):
    """Finds the optimal decision threshold based on F-beta score"""
    print(f"\n--- Optimizing decision threshold (Beta={optimize_beta}) ---")
    thresholds = np.arange(0.05, 0.95, 0.01)
    fbeta_scores = [
        fbeta_score(y_train, (y_train_prob >= t).astype(int), beta=optimize_beta)
        for t in thresholds
    ]
    best_threshold = thresholds[np.argmax(fbeta_scores)]
    print(f"Optimal threshold found: {best_threshold:.3f}")
    
    return best_threshold

def plot_feature_importances(automl, X_train, feature_importance,
                             X_test, y_test, seed, task="classification"):
    """
    Computes and plots feature importance using permutation or built-in methods.

    feature_importance values:
        0 → no importance computed
        1 → permutation only (fallback to built-in if it fails)
        2 → both permutation and built-in

    Permutation importance: shuffles each feature independently and measures
    how much the chosen scoring metric drops. A large drop = high importance.
    Importances are metric decrements (can be negative if the feature adds noise).

    Built-in importance: model-internal scores (e.g. impurity reduction in trees).
    These are relative scores with no direct metric interpretation.
    """
    # Return early if no importance requested
    if feature_importance == 0:
        return None, None

    importance_df_permutation = None
    importance_df_builtin = None

    # Exclude text embedding columns — only clinical features are evaluated
    # for computational efficiency and clinical interpretability
    non_emb_cols = [col for col in X_train.columns if not col.startswith('emb')]

    # Average Precision (PR-AUC) mirrors the 'ap' training metric used in FLAML,
    # ensuring consistency between training objective and importance evaluation
    perm_scoring = "average_precision"

    # -------------------------------------------------------------------------
    # PERMUTATION IMPORTANCE (feature_importance == 1 or 2)
    # -------------------------------------------------------------------------
    use_permutation_importance = False

    if feature_importance >= 1:
        print(f"--- Computing permutation importance ({perm_scoring}) for {len(non_emb_cols)} clinical features ---")

        try:
            class ClassifierWrapper:
                """
                Wraps FLAML AutoML for sklearn's permutation_importance API.
                Reconstructs the full feature matrix on each prediction call
                by merging permuted clinical columns with the original embeddings.
                """
                def __init__(self, model, X_full):
                    self.model = model
                    self.X_full = X_full
                    self._estimator_type = "classifier"
                    self.classes_ = np.array(model.classes_)

                def fit(self, X, y):
                    return self

                def _reconstruct(self, X_clinical):
                    # Re-attach original embedding columns using index alignment,
                    # then restore original column order expected by the model
                    emb_cols = [c for c in self.X_full.columns if c.startswith('emb')]
                    X_emb = self.X_full.loc[X_clinical.index, emb_cols]
                    return pd.concat([X_clinical, X_emb], axis=1)[self.X_full.columns]

                def predict(self, X):
                    return self.model.predict(self._reconstruct(X))

                def predict_proba(self, X):
                    proba = self.model.predict_proba(self._reconstruct(X))
                    # Ensure 2D output for both binary and multiclass
                    if proba.ndim == 1:
                        proba = np.column_stack([1 - proba, proba])
                    return proba

            # Full X_test passed so the wrapper can recover embeddings internally
            wrapped_model = ClassifierWrapper(automl, X_test)

            # Only clinical columns are permuted; embeddings are reconstructed internally
            perm_imp = permutation_importance(
                wrapped_model,
                X_test[non_emb_cols],
                y_test,
                scoring=perm_scoring,
                n_repeats=5,
                random_state=seed,
                n_jobs=-1,
            )

            importance_df_permutation = (
                pd.DataFrame({
                    "feature": non_emb_cols,
                    "importance": perm_imp.importances_mean,
                    "importance_std": perm_imp.importances_std,
                })
                .sort_values("importance", ascending=False)
            )

            use_permutation_importance = True
            print("Permutation importance computed successfully.")

        except Exception as e:
            print(f"Warning: Permutation importance failed with error: {e}")
            # If mode is 1 (permutation only), fallback to built-in
            if feature_importance == 1:
                print("Falling back to built-in feature importance...")
                feature_importance = 2  # Trigger built-in block below

    # -------------------------------------------------------------------------
    # BUILT-IN IMPORTANCE (feature_importance == 2, or fallback from mode 1)
    # -------------------------------------------------------------------------
    if feature_importance == 2:
        print("--- Computing built-in feature importance ---")

        try:
            best_model = automl.model
            feature_names = automl.feature_names_in_.tolist()

            if hasattr(best_model, 'feature_importances_'):
                # Tree-based models: cumulative impurity reduction across all splits
                importances = best_model.feature_importances_

                if len(importances) == len(feature_names):
                    importance_df_builtin = (
                        pd.DataFrame({
                            "feature": feature_names,
                            "importance": importances,
                        })
                        .sort_values("importance", ascending=False)
                    )
                    print("Built-in feature importance computed successfully.")
                else:
                    print(f"Warning: Importance length ({len(importances)}) doesn't match features ({len(feature_names)})")

            elif hasattr(best_model, 'coef_'):
                # Linear models: absolute coefficients as proxy for importance.
                # For multiclass, coef_ is 2D (n_classes x n_features) →
                # average across classes to get a single importance per feature
                coef = best_model.coef_
                importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef[0])

                if len(importances) == len(feature_names):
                    importance_df_builtin = (
                        pd.DataFrame({
                            "feature": feature_names,
                            "importance": importances,
                        })
                        .sort_values("importance", ascending=False)
                    )
                    print("Built-in feature importance computed successfully.")
                else:
                    print(f"Warning: Coefficient length ({len(importances)}) doesn't match features ({len(feature_names)})")

            else:
                print(f"Warning: Model type '{type(best_model).__name__}' does not support built-in feature importance.")

            # Filter out embedding features — not individually interpretable
            if importance_df_builtin is not None:
                importance_df_builtin = (
                    importance_df_builtin[~importance_df_builtin["feature"].str.startswith('emb')]
                    .sort_values("importance", ascending=False)
                )
                print("Built-in importance filtered to show clinical features only.")

        except Exception as e:
            print(f"Warning: Could not extract built-in feature importance: {e}")

    # -------------------------------------------------------------------------
    # PLOTTING — one plot per computed importance type
    # -------------------------------------------------------------------------
    def _plot_and_print(importance_df, is_permutation):
        top_k = min(15, len(importance_df))
        plot_df = importance_df.head(top_k)

        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["feature"][::-1], plot_df["importance"][::-1])

        if is_permutation:
            plt.title(f"Top Features (Permutation Importance — {perm_scoring})")
            plt.xlabel(f"Mean decrease in {perm_scoring} (percentage points)")
        else:
            plt.title(f"Top Features (Built-in Importance — {automl.best_estimator})")
            plt.xlabel("Feature Importance (relative)")

        plt.tight_layout()
        plt.show()

        print(f"\nTop {top_k} most important features:")
        if is_permutation:
            print(f"Metric: Mean decrease in {perm_scoring} (percentage points)\n")
        for _, row in plot_df.iterrows():
            if is_permutation:
                print(f"  {row['feature']:<20} = {row['importance']*100:.4f} pp")
            else:
                print(f"  {row['feature']:<20} {row['importance']:.4f}")

    if importance_df_permutation is not None:
        _plot_and_print(importance_df_permutation, is_permutation=True)

    if importance_df_builtin is not None:
        _plot_and_print(importance_df_builtin, is_permutation=False)

    return importance_df_permutation, importance_df_builtin

import warnings

def plot_shap_interpretation(automl, X_test, task="classification", max_display=15, seed=42):
    """
    Computes SHAP values and plots a summary excluding embedding features.
    SHAP provides directional impact per feature (positive = pushes toward positive class).

    For binary classification: single summary plot.
    For multiclass: one summary plot per class.
    """
    print("\n--- Initializing SHAP Explainer ---")

    def _prepare_for_shap(X):
        """Convert categorical columns to numeric codes for SHAP compatibility.
        Required for XGBoost, LightGBM and any model that stored categorical
        metadata during training — avoids 'categorical_feature do not match' errors."""
        X_shap = X.copy()
        cat_cols = X_shap.select_dtypes(include='category').columns
        for col in cat_cols:
            X_shap[col] = X_shap[col].cat.codes
        return X_shap

    try:
        # Clinical features only — embeddings excluded for interpretability
        non_emb_cols = [col for col in X_test.columns if not col.startswith('emb')]

        # Extract the underlying fitted estimator from FLAML
        model = automl.model.estimator

        # Subsample X_test for compute efficiency
        sample_size = min(500, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=seed)

        # Convert categoricals to numeric codes — applied to all model types
        # to avoid categorical mismatch errors between training and SHAP input
        X_sample_shap = _prepare_for_shap(X_sample)

        # Precompute non-embedding column indices once for slicing shap_values arrays
        all_cols = X_test.columns.tolist()
        non_emb_idx = [all_cols.index(c) for c in non_emb_cols]

        # X used for plotting — converted + clinical columns only
        X_plot = X_sample_shap[non_emb_cols]

        # Use TreeExplainer for tree-based models (exact, fast)
        # Fallback to generic Explainer for linear or other model types
        tree_model_types = (
            "LGBMClassifier", "XGBClassifier", "RandomForestClassifier",
            "ExtraTreesClassifier", "GradientBoostingClassifier"
        )

        if type(model).__name__ in tree_model_types:
            print(f"Using TreeExplainer for {type(model).__name__}...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_shap)
        else:
            print(f"Using generic Explainer for {type(model).__name__}...")
            background = X_sample_shap.sample(min(100, len(X_sample_shap)), random_state=seed)
            explainer = shap.Explainer(model, background)
            shap_out = explainer(X_sample_shap)
            shap_values = shap_out.values

        print(f"SHAP values computed on {sample_size} samples.")

        # --- Plotting ---
        if task == "multiclass":
            # TreeExplainer returns a list of 2D arrays (one per class)
            # Generic Explainer returns a 3D array (n_samples, n_features, n_classes)
            # Normalize to list of 2D arrays for consistent handling
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

            class_names = [str(c) for c in automl.classes_]

            for i, class_name in enumerate(class_names):
                print(f"\nGenerating SHAP Summary Plot — Class: {class_name}")
                shap_class_filtered = shap_values[i][:, non_emb_idx]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
                    shap.summary_plot(
                        shap_class_filtered,
                        X_plot,
                        max_display=max_display,
                        show=True,
                    )

        else:
            # Binary classification:
            # - TreeExplainer returns list [neg_class, pos_class] or single 2D array
            # - Generic Explainer returns 2D array directly
            # Always use positive class (index 1) for interpretability
            if isinstance(shap_values, list):
                # RandomForest / tree models return list of arrays — take positive class
                shap_values_binary = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values_binary = shap_values[:, :, 1]
            else:
                # Already a 2D array (XGBoost, generic explainer)
                shap_values_binary = shap_values

            shap_filtered = shap_values_binary[:, non_emb_idx]

            print("\nGenerating SHAP Summary Plot — Positive class")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="shap")
                shap.summary_plot(
                    shap_filtered,
                    X_plot,
                    max_display=max_display,
                    show=True,
                )

        print("SHAP interpretation completed successfully.")
        return shap_values

    except Exception as e:
        print(f"Warning: SHAP interpretation failed: {e}")
        return None
    
def run_binary_automl_model(
    cohort_name: str,
    target_column: str,
    demand_code: Optional[Union[int, List[int]]] = None,
    triage_value: Optional[int] = None,
    include_embeddings: bool = True,
    export_table: bool = False, 
    time_budget: int = 600,
    test_size: float = 0.2,
    seed: int = 42,
    min_age: Optional[int] = None,
    optimize_metric: Optional[str] = None,
    n_splits_cv: int = 5,
    optimize_beta: int = 2,
    feature_importance: int = 0,
    plot_shap: bool = True
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
        If provided, filters df to demand_type_1.
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

    # -----------------------------
    # Print type of triage-specific cohort
    # -----------------------------
    triage_print(triage_value, cohort_name, demand_code)

    # -----------------------------
    # Data loading & column selection
    # -----------------------------
    df = data_load_col_selection(target_column, triage_value, include_embeddings)

    # -----------------------------
    # Basic validation
    # -----------------------------
    col_validation(df, target_column)

    # -----------------------------
    # Cohort & triage filtering
    # -----------------------------
    df_filtered = data_filtering(df, target_column, demand_code, triage_value, min_age,
                        export_table, protocol_features=protocol_questions)

    # -----------------------------
    # Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test, sample_weight = train_test_split_weights(
        df_filtered, target_column, test_size, seed)

    # -----------------------------
    # AutoML training
    # -----------------------------
    automl = run_automl_training(X_train, y_train, sample_weight, time_budget, 
                                 optimize_metric, n_splits_cv, seed, task = "classification")

    # -----------------------------
    # Threshold optimization (decision threshold)
    # -----------------------------
    y_train_prob = automl.predict_proba(X_train)[:, 1]
    y_test_prob = automl.predict_proba(X_test)[:, 1]
    best_threshold = optimize_threshold(y_train, y_train_prob, optimize_beta)
    
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

    # Plot metrics
    # Create an evaluation dataframe by mapping back to the filtered data
    plot_df = df.loc[y_test.index].copy() 
    plot_df['target_real'] = y_test
    plot_df['target_pred'] = y_test_pred
    eda.evaluate_diagnostic_performance(plot_df, 'target_real', 'target_pred')

    # -----------------------------
    # Feature importance
    # -----------------------------
    importance_df_permutation, importance_df_builtin = plot_feature_importances(
        automl, X_train, feature_importance, X_test, y_test, seed, task="classification")

    # -----------------------------
    # SHAP plot
    # -----------------------------
    shap_values = None
    if plot_shap:
        shap_values = plot_shap_interpretation(
            automl, X_test, task="classification", seed=seed
        )
    
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
            n_samples=len(df_filtered),
            n_train=len(X_train),
            n_test=len(X_test),
        )
    )

    display(summary_df)

# ===============================================================
# Generic multiclass classification pipeline using FLAML
# Optimized for P1-36, P1-58 and Non-Emergency prediction
# ===============================================================

def run_multiclass_automl_model(
    cohort_name: str,
    target_column: str,
    demand_code: Optional[Union[int, List[int]]] = [36, 58],
    triage_value: Optional[int] = None,
    include_embeddings: bool = True,
    export_table: bool = False, 
    time_budget: int = 600,
    test_size: float = 0.2,
    seed: int = 42,
    min_age: Optional[int] = None,
    optimize_metric: str = 'macro_f1',
    n_splits_cv: int = 5,
    plot_feature_importance: bool = True,
    use_permutation_importance: bool = False
) -> dict:
    """
    Runs a complete multiclass classification pipeline using FLAML AutoML.

    Parameters
    ----------
    cohort_name : str
        Name of the cohort.
    target_column : str
        Multiclass target column (e.g., 0: None, 1: P1-36, 2: P1-58).
    demand_code : Optional[Union[int, List[int]]]
        Filtering codes, default is [36, 58].
    triage_value : Optional[int]
        Filter by triage availability.
    include_embeddings : bool
        Whether to include text embeddings.
    time_budget : int
        FLAML training budget in seconds.
    test_size : float
        Proportion of test split.
    seed : int
        Random seed.
    min_age : Optional[int]
        Optional age filter.
    optimize_metric : str
        Metric to optimize (macro_f1 is recommended for multiclass).
    n_splits_cv : int
        Number of CV folds.
    plot_feature_importance : bool
        Whether to compute and plot feature importance.
    use_permutation_importance : bool
        If True, use permutation importance.

    Returns
    -------
    dict
        Dictionary with model, metrics, and summary results.
    """

    # -----------------------------
    # Print type of triage-specific cohort
    # -----------------------------
    triage_print(triage_value, cohort_name, demand_code)

    # -----------------------------
    # Data loading & column selection
    # -----------------------------
    df = data_load_col_selection(triage_value, include_embeddings)

    # -----------------------------
    # Create multiclass variable
    # -----------------------------
    df.loc[(df["demand_type_1"] == 36) & (df[target_column] == 1), target_column] = 1
    df.loc[(df["demand_type_1"] == 58) & (df[target_column] == 1), target_column] = 2
    
    # -----------------------------
    # Basic validation
    # -----------------------------
    col_validation(df, target_column)
    
    # -----------------------------
    # Cohort & triage filtering
    # -----------------------------
    df = data_filtering(df, target_column, demand_code, triage_value, min_age,
                        export_table, protocol_features=protocol_questions)

    # -----------------------------
    # Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test, sample_weight = train_test_split_weights(
        df, target_column, test_size, seed)

    # -----------------------------
    # AutoML training (task set to multiclass)
    # -----------------------------
    automl = run_automl_training(
        X_train, y_train, sample_weight, time_budget, 
        optimize_metric, n_splits_cv, seed, task="multiclass"
    )

    # -----------------------------
    # Test set evaluation
    # -----------------------------
    print("\n--- Test set results (Multiclass) ---")
    
    # Multiclass prediction uses the highest probability class (argmax)
    y_test_pred = automl.predict(X_test)
    y_test_prob = automl.predict_proba(X_test)

    # -----------------------------
    # Confusion Matrix Plot
    # -----------------------------
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    label_mapping = {
    0: "P0",
    1: "P1-Unconscious",
    2: "P1-Cardiac Arrest"
    }
    class_labels = sorted(df[target_column].unique())
    labels_str = [label_mapping.get(l, str(l)) for l in class_labels]

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Purples', fmt='d',
                xticklabels=labels_str, yticklabels=labels_str)
    plt.title(f'Multiclass Confusion Matrix - {cohort_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Metrics computation
    # -----------------------------
    accuracy = (y_test_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_test_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
    logloss_val = log_loss(y_test, y_test_prob)
    
    # ROC-AUC OvR (One-vs-Rest) is the standard for multiclass
    roc_auc_ovr = roc_auc_score(y_test, y_test_prob, multi_class='ovr', average='weighted')

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "log_loss": logloss_val,
        "roc_auc_ovr": roc_auc_ovr
    }

    # -----------------------------
    # Print metrics 
    # -----------------------------
    print("\n" + "="*60)
    print(f"{'MULTICLASS CLASSIFICATION REPORT':^60}")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=labels_str))

    print("\n" + "="*60)
    print(f"{'PERFORMANCE METRICS (%)':^60}")
    print("="*60)
    
    print(f"\n{'Overall Performance:':<30}")
    print(f"  Accuracy:                    {metrics['accuracy']*100:.2f}%")
    print(f"  ROC-AUC (OvR):               {metrics['roc_auc_ovr']*100:.2f}%")
    print(f"  Macro F1 Score:              {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1 Score:           {metrics['weighted_f1']*100:.2f}%")
    
    print(f"\n{'Model Error:':<30}")
    print(f"  Log Loss:                    {metrics['log_loss']:.4f}")
    
    print("="*60 + "\n")

    # -----------------------------
    # Feature importance
    # -----------------------------
    importance_df = None
    if plot_feature_importance:
        importance_df = plot_feature_importances(automl, X_train, use_permutation_importance, 
                                                 X_test, y_test, seed, task="multiclass")

    # -----------------------------
    # Build summary DataFrame
    # -----------------------------
    summary_df = (
        pd.DataFrame(metrics, index=[0])
        .assign(
            cohort=cohort_name,
            n_samples=len(df),
            best_model=automl.best_estimator
        )
    )

    display(summary_df)
