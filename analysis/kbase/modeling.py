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
    precision_score, recall_score, accuracy_score, confusion_matrix,
    classification_report, average_precision_score, 
    log_loss, precision_recall_curve, roc_curve
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
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

def time_print():
    now = datetime.now() 
    print(f"Current Date and Time: {now.strftime('%d/%m/%Y %H:%M:%S')}\n")

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

def data_load_col_selection(target_column, triage_value, 
                            include_lr, include_history,
                            include_medication, include_embeddings)-> pd.DataFrame:
    """Loads data and performs initial feature selection based on settings"""
    # Load preprocessed cleaned table
    df = pq.read_table(
        os.path.join(settings.source_tables_path, settings.triaje_table_cleaned_path)
    ).to_pandas()

    # Select base columns
    base_cols = [
        "demandpk",
        "age",
        "sex",
        "demand_type_1",
        target_column,
        "day_week_monday", "day_week_tuesday", "day_week_wednesday", "day_week_thursday",
        "day_week_friday", "day_week_saturday", "day_week_sunday",
        "time_of_day_early_morning", "time_of_day_morning", "time_of_day_afternoon", "time_of_day_night",
        "month_january", "month_february", "month_march", "month_april",
        "month_may", "month_june", "month_july", "month_august",
        "month_september", "month_october", "month_november", "month_december",
        "season_spring", "season_summer", "season_autumn", "season_winter",
        "year",
        "location_patient_home", "location_patient_public_road", "location_patient_other",
        "alert_receiver_112", "alert_receiver_user", "alert_receiver_pol_fg",
        "alert_receiver_hs", "alert_receiver_tele", "alert_receiver_others",
        "province_almeria", "province_cadiz", "province_cordoba", "province_granada",
        "province_huelva", "province_jaen", "province_malaga", "province_sevilla",
        "incident_latitude",
        "incident_longitude",
        "triage",
    ]

    # Initialize modelling columns with base columns
    modelling_cols = base_cols.copy()
    # Add triage questions (One-Hot Encoded columns)
    if triage_value != 0:
        modelling_cols += [col for col in df.columns if col.startswith('q')]
    # Add literal reason columns (One-Hot Encoded columns)
    if include_lr == True:
        modelling_cols += [col for col in df.columns if col.startswith('lr_')]
    # Add past history columns (One-Hot Encoded columns)
    if include_history == True:
        modelling_cols += [col for col in df.columns if col.startswith('hist_')]
    # Add medication columns (One-Hot Encoded columns)
    if include_medication == True:
        modelling_cols += [col for col in df.columns if col.startswith('atc_')]
    # Include NLP text embeddings if flag is set to True
    if include_embeddings:
        embedding_cols = [col for col in df.columns if col.startswith('emb_')]
        modelling_cols += embedding_cols
        print(f"Feature Set: Including {len(embedding_cols)} text embedding dimensions.")
            
    # Select base columns for modeling
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
                   triage_value, min_age, years, export_table, protocol_features) -> pd.DataFrame:
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

    # Filter out medication columns by threshold-specific available medication columns
    df = dp.analyze_atc_columns(
        df,
        threshold=1.0,
        drop_columns=True,
    )

    # Filter by triage patients
    if triage_value is not None:
        df = df[df["triage"] == triage_value].copy()

    # Filter by min age
    if min_age is not None:
        if "age" not in df.columns:
            raise ValueError("'age' column not found but min_age was provided")
        df = df[df["age"] >= min_age].copy()
        print(f"Filtering by age >= {min_age}")

    # Filter by specific years
    if years is not None:
        if "year" not in df.columns:
            raise ValueError("'year' column not found but years was provided")
        df = df[df["year"].isin(years)].copy()
        print(f"Filtering specific year cohort: {years}")

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
        # Integrated warning if no protocol matches the demand_code
        else:
            print(f"⚠️ Warning: No specific protocol found for demand_code {demand_code}")

    # Clean column names: remove 'q' prefix and underscores for readability
    df.columns = [col.replace('q', '', 1).replace('_', '') 
                  if col.startswith('q') else col for col in df.columns]
                   
    # Export of preprocessed, ready for modeling, table
    if export_table:    
        rule_to_path = {
            'cardiac_arrest': settings.cardiacarrest_table_modeling,
            16: settings.dyspnea_table_modeling,
            23: settings.chestpain_table_modeling,
            54: settings.stroke_table_cleaned_path,
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

    # Drop columns after cohort filtering and data export
    cols_to_drop = ["demandpk", "demand_type_1", "triage", "year", "literal_reason", "hcdm_id"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # --- Column summary by group ---
    triage_cols  = [c for c in df.columns if c[0].isdigit()]
    lr_cols      = [c for c in df.columns if c.startswith("lr_")]
    hist_cols    = [c for c in df.columns if c.startswith("hist_")]
    atc_cols     = [c for c in df.columns if c.startswith("atc")]
    other_cols   = [c for c in df.columns if c not in triage_cols + lr_cols + hist_cols + atc_cols]

    print("\n" + "="*40)
    print("FEATURE SUMMARY BY GROUP")
    print("="*40)
    print(f"  Triage questions  (digit prefix): {len(triage_cols):>4}")
    print(f"  Literal reason    (lr_):           {len(lr_cols):>4}")
    print(f"  Past history      (hist_):         {len(hist_cols):>4}")
    print(f"  Medication        (atc):           {len(atc_cols):>4}")
    print(f"  Other:                             {len(other_cols):>4}")
    print(f"  {'─'*30}")
    print(f"  Total features:                    {len(df.columns):>4}")
    print("="*40 + "\n")

    print(f"\nFinal cohort size for {target_column}: {len(df)}")
    print("Outcome distribution: ", df[target_column].value_counts(dropna=False), "\n")
    # Show info for only the first 50 columns
    df.iloc[:, :50].info()

    print(f"\n" + "="*40)
    print(f"Number of features for modeling: {len(df.columns)}")
    print("="*40 + "\n")

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

import pandas as pd
from flaml import AutoML

def run_automl_training(X_train, y_train, sample_weight, time_budget, 
                        optimize_metric, n_splits_cv, seed, task):
    """Executes the FLAML AutoML optimization process and prints a benchmarking table"""
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

    # ---------------------------------------------------------
    # Benchmarking: Best Per Estimator Analysis
    # ---------------------------------------------------------
    try:
        history_list = []
        
        # Method A: Try accessing via best_loss_per_estimator (most stable)
        if hasattr(automl, 'best_loss_per_estimator') and automl.best_loss_per_estimator:
            for learner, loss in automl.best_loss_per_estimator.items():
                if loss < 1.0: # Only include learners that were actually tested
                    history_list.append({
                        'Estimator': learner,
                        'CV_Score': 1 - loss
                    })
        
        # Method B: Fallback to _search_states if Method A provided no results
        if not history_list and hasattr(automl, '_search_states'):
            for learner_id, state in automl._search_states.items():
                if hasattr(state, 'best_loss') and state.best_loss < 1.0:
                    history_list.append({
                        'Estimator': learner_id,
                        'CV_Score': 1 - state.best_loss
                    })

        if history_list:
            benchmark_df = pd.DataFrame(history_list)
            # Sort by best score
            top_models = benchmark_df.sort_values(by='CV_Score', ascending=False).reset_index(drop=True)
            top_models.index += 1 # Start ranking at 1

            print("\n--- Model Benchmarking (Best Results per Estimator Type) ---")
            print(top_models.to_string())
        else:
            print("\n⚠️ No detailed benchmarking history available for this run.")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate benchmarking table: {e}")
    
    return automl

def set_fb_threshold(y_train, y_train_prob, optimize_beta):
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

def compute_youden_threshold(
    y_train,
    y_train_prob,
    y_test,
    y_test_prob,
    sensitivity_range: Optional[tuple] = (0.80, 0.90),
):
    """
    Finds the optimal decision threshold by maximising the Youden index J = Se + Sp - 1.

    Geometrically, J is the maximum vertical distance between the ROC curve and
    the no-discrimination diagonal. The threshold that maximises J provides the
    best symmetric trade-off between sensitivity and specificity.

    Note: Youden assumes symmetric misclassification costs. In emergency triage it
    serves as an exploratory starting point before shifting the threshold toward
    higher sensitivity by clinical imperative.

    Optimisation is performed on TRAIN to avoid threshold over-fitting on TEST.
    The ROC curve plotted uses TEST for an unbiased visual assessment.

    Parameters
    ----------
    y_train, y_train_prob : train labels and predicted probabilities
    y_test,  y_test_prob  : test labels and predicted probabilities
    sensitivity_range : tuple of float or None
        If provided, restricts the search to ROC points whose sensitivity falls
        within [se_min, se_max] (values in [0, 1], e.g. (0.70, 0.90)).
        Within that band, the point that maximises J is selected.
        If no ROC point falls in the range, falls back to the global Youden optimum
        with a warning.

    Returns
    -------
    float
        Optimal threshold (maximises J on train set, optionally within sensitivity_range).
    """
    print("\n--- Optimising decision threshold (Youden Index) ---")

    # Compute ROC on TRAIN — threshold selection must not touch TEST
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_prob)

    # J = Se + Sp - 1  ≡  tpr - fpr
    youden_j = tpr_train - fpr_train

    if sensitivity_range is not None:
        se_min, se_max = sensitivity_range
        print(f"  Sensitivity range : [{se_min:.2%}, {se_max:.2%}]")
        mask = (tpr_train >= se_min) & (tpr_train <= se_max)
        if mask.any():
            candidates = np.where(mask)[0]
            best_idx = int(candidates[np.argmax(youden_j[candidates])])
        else:
            print(f"  WARNING: no ROC point found in sensitivity range "
                  f"[{se_min:.2%}, {se_max:.2%}]. Falling back to global Youden optimum.")
            best_idx = int(np.argmax(youden_j))
    else:
        best_idx = int(np.argmax(youden_j))

    best_thr  = float(thresholds_train[best_idx])
    best_j    = float(youden_j[best_idx])
    best_se   = float(tpr_train[best_idx])
    best_sp   = float(1.0 - fpr_train[best_idx])

    print(f"  Optimal threshold : {best_thr:.4f}")
    print(f"  Youden J          : {best_j:.4f}")
    print(f"  Sensitivity (Se)  : {best_se:.4f}")
    print(f"  Specificity (Sp)  : {best_sp:.4f}")

    # --- ROC plot on TEST with Youden point (from train optimisation) ---
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    auc_test = roc_auc_score(y_test, y_test_prob)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr_test, tpr_test, color='steelblue', lw=2,
            label=f'ROC curve — test  (AUC = {auc_test:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='No discrimination')

    # Sensitivity range band
    if sensitivity_range is not None:
        se_min, se_max = sensitivity_range
        ax.axhspan(se_min, se_max, color='gold', alpha=0.15,
                   label=f'Sensitivity range [{se_min:.0%}, {se_max:.0%}]')

    # Youden point
    ax.scatter(1.0 - best_sp, best_se, color='crimson', zorder=5, s=120,
               label=(f'Selected point  J = {best_j:.3f}\n'
                      f'Se = {best_se:.3f}   Sp = {best_sp:.3f}\n'
                      f'Threshold = {best_thr:.4f}'))

    # Vertical segment from diagonal to selected point (visual magnitude of J)
    ax.vlines(x=1.0 - best_sp,
              ymin=1.0 - best_sp, ymax=best_se,
              colors='crimson', linestyles='dashed', lw=1.5, alpha=0.7,
              label=f'J = {best_j:.3f}  (vertical distance to diagonal)')

    ax.set_xlabel('1 − Specificity  (FPR)', fontsize=11)
    ax.set_ylabel('Sensitivity  (TPR)', fontsize=11)
    title = 'ROC Curve — Youden Index Threshold'
    if sensitivity_range is not None:
        title += f'  (Se range [{se_min:.0%}, {se_max:.0%}])'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    plt.tight_layout()
    plt.show()

    return best_thr

def optimize_clinical_threshold(
    y_train,
    y_train_prob,
    y_test,
    y_test_prob,
    max_undertriage: float = 0.10,
    max_overtriage: float = 0.50,
) -> float:
    """
    Finds the optimal decision threshold using a clinically-grounded strategy.

    Strategy:
        Maximize F1 subject to two simultaneous constraints:
            · Undertriage < max_undertriage  → FN / (FN + TP)
            · Overtriage  < max_overtriage   → FP / (FP + TN)
        Search is performed on TRAIN to avoid optimistic threshold selection.
        Fallback to undertriage-only constraint if no threshold meets both.

    Parameters
    ----------
    y_train : array-like
        True labels for the training set.
    y_train_prob : array-like
        Predicted probabilities for the positive class (train set).
    y_test : array-like
        True labels for the test set (used only for visualization).
    y_test_prob : array-like
        Predicted probabilities for the positive class (test set).
    max_undertriage : float
        Maximum allowed undertriage rate (default 10%).
    max_overtriage : float
        Maximum allowed overtriage rate (default 50%).

    Returns
    -------
    float
        Optimal clinical threshold.
    """
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, roc_auc_score,
        average_precision_score, accuracy_score
    )

    print("\n--- Optimizing clinical threshold ---")
    print(f"    Constraints: undertriage < {max_undertriage:.0%} | overtriage < {max_overtriage:.0%}")

    # Fine-grained threshold grid on TRAIN — 1000 points for uniform resolution
    thresholds_grid = np.linspace(0.01, 0.99, 1000)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    undertriage_arr = np.array([
        ((y_train_prob < thr) & (y_train == 1)).sum() / n_pos if n_pos > 0 else 0
        for thr in thresholds_grid
    ])
    overtriage_arr = np.array([
        ((y_train_prob >= thr) & (y_train == 0)).sum() / n_neg if n_neg > 0 else 0
        for thr in thresholds_grid
    ])

    f1_arr = np.array([
        f1_score(y_train, (y_train_prob >= thr).astype(int), zero_division=0)
        for thr in thresholds_grid
    ])
    accuracy_arr = np.array([
        accuracy_score(y_train, (y_train_prob >= thr).astype(int))
        for thr in thresholds_grid
    ])
    recall_arr = np.array([
        recall_score(y_train, (y_train_prob >= thr).astype(int), zero_division=0)
        for thr in thresholds_grid
    ])

    # Maximize F1 within the safe zone (both constraints met simultaneously)
    valid_mask    = (undertriage_arr < max_undertriage) & (overtriage_arr < max_overtriage)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        best_idx           = valid_indices[np.argmax(f1_arr[valid_indices])]
        clinical_threshold = float(thresholds_grid[best_idx])
        print(f"\n  ✅ Optimal threshold : {clinical_threshold:.4f}")
        print(f"     F1               : {f1_arr[best_idx]:.4f}")
        print(f"     Accuracy         : {accuracy_arr[best_idx]:.4f}")
        print(f"     Recall           : {recall_arr[best_idx]:.4f}")
        print(f"     Undertriage      : {undertriage_arr[best_idx]:.2%}  (limit: {max_undertriage:.0%})")
        print(f"     Overtriage       : {overtriage_arr[best_idx]:.2%}  (limit: {max_overtriage:.0%})")
    else:
        # Fallback: undertriage constraint only (clinically more critical)
        print("\n  ⚠️  No threshold meets both constraints simultaneously.")
        print("      → Fallback: undertriage constraint only")
        under_only = np.where(undertriage_arr < max_undertriage)[0]

        if len(under_only) > 0:
            best_idx           = under_only[np.argmax(f1_arr[under_only])]
            clinical_threshold = float(thresholds_grid[best_idx])
            print(f"      → Fallback threshold : {clinical_threshold:.4f}")
            print(f"         Resulting overtriage: {overtriage_arr[best_idx]:.2%} (exceeds {max_overtriage:.0%})")
        else:
            best_idx           = int(np.argmin(undertriage_arr))
            clinical_threshold = float(thresholds_grid[best_idx])
            print(f"      → Minimum undertriage threshold: {clinical_threshold:.4f}")

    # -------------------------------------------------------------------------
    # Visualization — 2x2 layout (threshold plot + ROC top, PR bottom center)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))

    # Top-left: threshold optimization curves
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(thresholds_grid, f1_arr,          'g-',  label='F1-Score',             linewidth=2)
    ax1.plot(thresholds_grid, accuracy_arr,    'c-',  label='Accuracy',             linewidth=1.5, alpha=0.8)
    ax1.plot(thresholds_grid, recall_arr,      'b--', label='Recall (Sensitivity)', linewidth=1.5, alpha=0.8)
    ax1.plot(thresholds_grid, undertriage_arr, 'r-',  label='Undertriage (FN/P)',   linewidth=2)
    ax1.plot(thresholds_grid, overtriage_arr,  'm-',  label='Overtriage (FP/N)',    linewidth=2)

    ax1.axvline(clinical_threshold, color='black', linestyle='--', linewidth=1.5,
                label=f'Optimal threshold = {clinical_threshold:.4f}')
    ax1.axhline(max_undertriage, color='red',     linestyle=':', alpha=0.6,
                label=f'Undertriage limit = {max_undertriage:.0%}')
    ax1.axhline(max_overtriage,  color='magenta', linestyle=':', alpha=0.6,
                label=f'Overtriage limit = {max_overtriage:.0%}')

    if len(valid_indices) > 0:
        thr_lo = thresholds_grid[valid_indices[0]]
        thr_hi = thresholds_grid[valid_indices[-1]]
        ax1.axvspan(thr_lo, thr_hi, alpha=0.08, color='green',
                    label='Safe zone (both constraints met)')

    ax1.set_title('Threshold Optimization\n'
                  f'(max F1 | undertriage < {max_undertriage:.0%} & overtriage < {max_overtriage:.0%})')
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Rate / Metric')
    ax1.legend(fontsize=7.5, loc='center right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    ax1.grid(alpha=0.3)

    # Top-right: ROC Curve (TEST)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_test_prob)
    roc_auc_val = roc_auc_score(y_test, y_test_prob)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(fpr_curve, tpr_curve, 'navy', linewidth=2,
             label=f'ROC-AUC = {roc_auc_val:.4f}')
    ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', label='Random (0.5)')

    test_under = ((y_test_prob < clinical_threshold) & (y_test == 1)).sum() / y_test.sum()
    test_spec  = ((y_test_prob < clinical_threshold) & (y_test == 0)).sum() / (y_test == 0).sum()
    ax2.scatter([1 - test_spec], [1 - test_under], color='black', zorder=5, s=80,
                label=f'Threshold = {clinical_threshold:.4f}')

    ax2.set_title('ROC Curve (Test Set)')
    ax2.set_xlabel('1 - Specificity (FPR)')
    ax2.set_ylabel('Sensitivity (TPR)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Bottom-center: PR Curve (TEST) — spans both bottom columns
    ax3 = fig.add_subplot(2, 2, 3)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc_val = average_precision_score(y_test, y_test_prob)
    baseline   = float(y_test.mean())

    ax3.plot(rec_curve, prec_curve, 'darkorange', linewidth=2,
             label=f'PR-AUC = {pr_auc_val:.4f}')
    ax3.axhline(baseline, color='gray', linestyle='--',
                label=f'Baseline (random) = {baseline:.2f}')

    test_pred_opt = (y_test_prob >= clinical_threshold).astype(int)
    opt_prec = precision_score(y_test, test_pred_opt, zero_division=0)
    opt_rec  = recall_score(y_test, test_pred_opt, zero_division=0)
    ax3.scatter([opt_rec], [opt_prec], color='black', zorder=5, s=80,
                label=f'Threshold = {clinical_threshold:.4f}  |  P={opt_prec:.3f}  R={opt_rec:.3f}')

    ax3.set_title('Precision-Recall Curve (Test Set)')
    ax3.set_xlabel('Recall (Sensitivity)')
    ax3.set_ylabel('Precision (PPV)')
    ax3.legend(fontsize=9)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.05])
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return clinical_threshold

def plot_calibration(
    y_train, y_train_prob,
    y_test,  y_test_prob,
    threshold: float,
    n_bins: int = 15,
    save_path: str = None,
) -> None:
    """
    Two-panel calibration figure:
      Left  — predicted probability distributions by class (train vs test).
      Right — reliability diagram (calibration curve) for train and test.

    Parameters
    ----------
    n_bins    : number of bins for the calibration curve (quantile strategy).
    save_path : if provided, saves the figure to that path (e.g. 'calibration.png').
    """
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: probability distributions by class ---
    ax = axes[0]
    ax.hist(y_train_prob[y_train == 0], bins=80, alpha=0.5, color='steelblue',
            density=True, label='Negative (train)')
    ax.hist(y_train_prob[y_train == 1], bins=80, alpha=0.5, color='red',
            density=True, label='Positive (train)')
    ax.hist(y_test_prob[y_test == 0],   bins=80, alpha=0.3, color='navy',
            density=True, label='Negative (test)', linestyle='--')
    ax.hist(y_test_prob[y_test == 1],   bins=80, alpha=0.3, color='darkred',
            density=True, label='Positive (test)', linestyle='--')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold:.4f}')
    ax.set_title('Predicted probability distribution\n(train vs test)')
    ax.set_xlabel('Predicted probability P(=1)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Right: reliability diagram ---
    ax2 = axes[1]
    for probs, labels, name, color in [
        (y_train_prob, y_train, 'Train', 'steelblue'),
        (y_test_prob,  y_test,  'Test',  'red'),
    ]:
        fraction_pos, mean_pred = calibration_curve(labels, probs,
                                                    n_bins=n_bins,
                                                    strategy='quantile')
        ax2.plot(mean_pred, fraction_pos, 's-', label=name,
                 color=color, linewidth=2)
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--',
             label='Perfect calibration')
    ax2.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                label=f'Threshold = {threshold:.4f}')
    ax2.set_title('Calibration curve\n(are predicted probabilities reliable?)')
    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Fraction of positives')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def print_test_metrics_with_ci(
    y_test,
    y_test_prob,
    threshold: float,
    n_bootstrap: int = 1000,
    ci_seed:     int = 42,
) -> dict:
    """
    Plots the confusion matrix, computes all test-set metrics, prints them
    with 95% confidence intervals, and returns the metrics dict.

    Clopper–Pearson (exact binomial) for binary proportions:
        Accuracy, Precision, Recall, Specificity, NPV

    Stratified bootstrap for composite metrics:
        F1, MCC, Youden Index, Balanced Accuracy
    """
    from scipy.stats import beta as _beta
    from sklearn.metrics import matthews_corrcoef

    y_test_pred = (y_test_prob >= threshold).astype(int)

    # --- Confusion matrix plot -----------------------------------------------
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

    # --- Metrics computation --------------------------------------------------
    tn, fp, fn, tp = conf_matrix.ravel()
    n_test = len(y_test)

    prec  = precision_score(y_test, y_test_pred, zero_division=0)
    rec   = recall_score(y_test, y_test_pred, zero_division=0)
    spec  = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv   = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc   = (tp + tn) / n_test

    f1      = f1_score(y_test, y_test_pred, zero_division=0)
    mcc     = matthews_corrcoef(y_test, y_test_pred)
    youden  = rec + spec - 1   if not np.isnan(spec) else np.nan
    bal_acc = (rec + spec) / 2 if not np.isnan(spec) else np.nan

    overtriage  = 1 - prec if not np.isnan(prec) else np.nan
    undertriage = 1 - npv  if not np.isnan(npv)  else np.nan

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    lr_pos = rec / (1 - spec) if (not np.isnan(spec) and spec < 1) else np.nan
    lr_neg = (1 - rec) / spec  if (not np.isnan(spec) and spec > 0) else np.nan

    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc  = average_precision_score(y_test, y_test_prob)

    # --- Clopper–Pearson CIs -------------------------------------------------
    def _cp(x, n, alpha=0.05):
        if n == 0:
            return (np.nan, np.nan)
        lo = _beta.ppf(alpha / 2,     x,     n - x + 1) if x > 0 else 0.0
        hi = _beta.ppf(1 - alpha / 2, x + 1, n - x)     if x < n else 1.0
        return lo, hi

    ci_cp = {
        "acc":  _cp(int(tp + tn), int(n_test)),
        "prec": _cp(int(tp), int(tp + fp)),
        "rec":  _cp(int(tp), int(tp + fn)),
        "spec": _cp(int(tn), int(tn + fp)),
        "npv":  _cp(int(tn), int(tn + fn)),
    }

    # --- Stratified bootstrap CIs --------------------------------------------
    def _safe_div(a, b):
        return a / b if b != 0 else np.nan

    rng     = np.random.default_rng(ci_seed)
    y_te_s  = pd.Series(y_test.values if hasattr(y_test, "values") else y_test).reset_index(drop=True)
    y_pr_s  = pd.Series(y_test_pred).reset_index(drop=True)
    pos_idx = np.where(y_te_s == 1)[0]
    neg_idx = np.where(y_te_s == 0)[0]

    boot = {"f1": [], "mcc": [], "youden": [], "bal_acc": []}

    for _ in range(n_bootstrap):
        pos_s = rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) else np.array([], dtype=int)
        neg_s = rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) else np.array([], dtype=int)
        idx   = np.concatenate([pos_s, neg_s])
        yt    = y_te_s.iloc[idx]
        yp    = y_pr_s.iloc[idx]

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        prec_b = _safe_div(tp_b, tp_b + fp_b)
        rec_b  = _safe_div(tp_b, tp_b + fn_b)
        spec_b = _safe_div(tn_b, tn_b + fp_b)

        f1_b = (
            _safe_div(2 * prec_b * rec_b, prec_b + rec_b)
            if not (np.isnan(prec_b) or np.isnan(rec_b)) else np.nan
        )
        denom_mcc = np.sqrt((tp_b+fp_b) * (tp_b+fn_b) * (tn_b+fp_b) * (tn_b+fn_b))
        mcc_b     = _safe_div(tp_b * tn_b - fp_b * fn_b, denom_mcc)
        youden_b  = (rec_b + spec_b - 1) if not (np.isnan(rec_b) or np.isnan(spec_b)) else np.nan
        bal_b     = _safe_div(rec_b + spec_b, 2)

        for key, val in [("f1", f1_b), ("mcc", mcc_b), ("youden", youden_b), ("bal_acc", bal_b)]:
            if not np.isnan(val):
                boot[key].append(val)

    ci_boot = {
        k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
        for k, v in boot.items()
    }

    # --- Print ---------------------------------------------------------------
    print("\n--- Test set results ---")

    def _pct(val, ci):
        lo, hi = ci
        s = f"{val*100:.2f}%"
        if not (np.isnan(lo) or np.isnan(hi)):
            s += f"   [95% CI: {lo*100:.2f}% – {hi*100:.2f}%]"
        return s

    def _raw(val, ci):
        lo, hi = ci
        s = f"{val:+.4f}"
        if not (np.isnan(lo) or np.isnan(hi)):
            s += f"   [95% CI: {lo:+.4f} – {hi:+.4f}]"
        return s

    no_ci = (np.nan, np.nan)

    print("\n" + "="*60)
    print(f"{'CLASSIFICATION REPORT':^60}")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=labels))

    print("\n" + "="*70)
    print(f"{'PERFORMANCE METRICS WITH 95% CI':^70}")
    print("="*70)

    print(f"\n  Binary metrics  (Clopper–Pearson CI):")
    print(f"  Accuracy:              {_pct(acc,  ci_cp['acc'])}")
    print(f"  ROC-AUC:               {roc_auc*100:.2f}%")
    print(f"  PR-AUC:                {pr_auc*100:.2f}%")
    print(f"  Precision (PPV):       {_pct(prec, ci_cp['prec'])}")
    print(f"  Recall (Sensitivity):  {_pct(rec,  ci_cp['rec'])}")
    print(f"  Specificity:           {_pct(spec, ci_cp['spec'])}")
    print(f"  NPV:                   {_pct(npv,  ci_cp['npv'])}")

    print(f"\n  Composite metrics  (Stratified Bootstrap CI):")
    print(f"  F1 Score:              {_pct(f1,      ci_boot['f1'])}")
    print(f"  MCC:                   {_raw(mcc,     ci_boot['mcc'])}")
    print(f"  Youden Index:          {_raw(youden,  ci_boot['youden'])}")
    print(f"  Balanced Accuracy:     {_pct(bal_acc, ci_boot['bal_acc'])}")

    print(f"\n  Triage-specific metrics:")
    print(f"  Overtriage Rate:       {overtriage*100:.2f}%")
    print(f"  Undertriage Rate:      {undertriage*100:.2f}%")

    print(f"\n  Error rates:")
    print(f"  False Positive Rate:   {fpr*100:.2f}%")
    print(f"  False Negative Rate:   {fnr*100:.2f}%")

    print(f"\n  Likelihood ratios:")
    print(f"  LR+:                   {lr_pos:.4f}")
    print(f"  LR-:                   {lr_neg:.4f}")

    print("\n" + "="*70 + "\n")

    return {
        "accuracy":            acc,
        "roc_auc":             roc_auc,
        "pr_auc":              pr_auc,
        "precision":           prec,
        "recall":              rec,
        "specificity":         spec,
        "npv":                 npv,
        "f1":                  f1,
        "mcc":                 mcc,
        "youden_index":        youden,
        "balanced_accuracy":   bal_acc,
        "overtriage":          overtriage,
        "undertriage":         undertriage,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "lr_positive":         lr_pos,
        "lr_negative":         lr_neg,
    }

def print_train_test_comparison(
    automl,
    X_train, y_train,
    X_test,  y_test,
    threshold: float,
    pr_auc_diff_threshold: float = 0.05,
    f1_diff_threshold: float     = 0.05,
    overfit_gap_threshold: float = 0.05,
) -> pd.DataFrame:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, accuracy_score,
    )

    # Prevalence computed automatically from y_train
    prevalence = float(y_train.mean())

    # --- Probabilities and predictions ---
    y_train_prob = automl.predict_proba(X_train)[:, 1]
    y_test_prob  = automl.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_test_pred  = (y_test_prob  >= threshold).astype(int)

    # --- Compute metrics helper ---
    def _metrics(y_true, y_pred, y_prob):
        return {
            "PR-AUC":      average_precision_score(y_true, y_prob),
            "ROC-AUC":     roc_auc_score(y_true, y_prob),
            "F1":          f1_score(y_true, y_pred, zero_division=0),
            "Accuracy":    accuracy_score(y_true, y_pred),
            "Precision":   precision_score(y_true, y_pred, zero_division=0),
            "Recall":      recall_score(y_true, y_pred, zero_division=0),
            "Specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        }

    train_m = _metrics(y_train, y_train_pred, y_train_prob)
    test_m  = _metrics(y_test,  y_test_pred,  y_test_prob)

    # -------------------------------------------------------------------------
    # Baselines: value a random classifier would achieve
    # -------------------------------------------------------------------------
    baselines = {
        "PR-AUC":      (prevalence,          f"{prevalence*100:.1f}%  (prevalence)"),
        "ROC-AUC":     (0.50,                "50.0%  (random)"),
        "F1":          (0.0,                 "~0.0%  (imbalanced)"),
        "Accuracy":    (1 - prevalence,      f"{(1-prevalence)*100:.1f}%  (majority class)"),
        "Precision":   (prevalence,          f"{prevalence*100:.1f}%  (prevalence)"),
        "Recall":      (0.50,                "50.0%  (floor)"),
        "Specificity": (0.50,                "50.0%  (floor)"),
    }

    # -------------------------------------------------------------------------
    # Prevalence-adjusted underfitting thresholds
    # -------------------------------------------------------------------------
    underfit_thresholds = {
        "PR-AUC":      (min(prevalence * 2, 0.50),
                        f"prevalence × 2 = {min(prevalence*2, 0.50)*100:.1f}%"),
        "ROC-AUC":     (0.60,
                        "random + 10pp = 60.0%"),
        "F1":          (max(0.30, prevalence * 1.5),
                        f"max(0.30, prev×1.5) = {max(0.30, prevalence*1.5)*100:.1f}%"),
        "Accuracy":    ((1 - prevalence) + 0.05,
                        f"majority + 5pp = {((1-prevalence)+0.05)*100:.1f}%"),
        "Precision":   (prevalence * 1.5,
                        f"prevalence × 1.5 = {prevalence*1.5*100:.1f}%"),
        "Recall":      (0.50,
                        "clinical floor = 50.0%"),
        "Specificity": (0.50,
                        "technical floor = 50.0%"),
    }

    # -------------------------------------------------------------------------
    # Per-metric gap thresholds for overfitting detection
    # -------------------------------------------------------------------------
    diff_thresholds = {
        "PR-AUC":      pr_auc_diff_threshold,
        "ROC-AUC":     overfit_gap_threshold,
        "F1":          f1_diff_threshold,
        "Accuracy":    overfit_gap_threshold,
        "Precision":   overfit_gap_threshold,
        "Recall":      overfit_gap_threshold,
        "Specificity": overfit_gap_threshold,
    }

    rows = []
    metric_order = ["PR-AUC", "ROC-AUC", "F1", "Accuracy", "Precision", "Recall", "Specificity"]

    for metric in metric_order:
        train_val              = train_m[metric]
        test_val               = test_m[metric]
        gap                    = train_val - test_val
        diff_thr               = diff_thresholds[metric]
        under_thr, under_label = underfit_thresholds[metric]
        baseline_val, baseline_label = baselines[metric]

        is_underfit = train_val < under_thr
        is_overfit  = gap > diff_thr

        rows.append({
            "Metric":         metric,
            "Baseline":       baseline_val,
            "Baseline_label": baseline_label,
            "Train":          train_val,
            "Underfit_thr":   under_thr,
            "Underfit_label": under_label,
            "Underfitting":   "🔴 Yes" if is_underfit else "✅ No",
            "Test":           test_val,
            "Gap (Tr-Te)":    gap,
            "Max Gap":        diff_thr,
            "Overfitting":    "🔴 Yes" if is_overfit  else "✅ No",
        })

    comparison_df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Pretty print
    # -------------------------------------------------------------------------
    sep = "─" * 125

    print("\n" + sep)
    print(f"{'TRAIN vs TEST PERFORMANCE COMPARISON':^125}")
    print(f"{'(threshold applied: ' + str(round(threshold, 4)) + ')':^125}")
    print(sep)
    print(f"  Dataset context  :  prevalence = {prevalence*100:.1f}%  "
          f"|  PR-AUC random baseline = {prevalence*100:.1f}%  "
          f"|  Accuracy random baseline = {(1-prevalence)*100:.1f}%")
    print(f"  Overfitting gap  :  default threshold = {overfit_gap_threshold*100:.1f}pp  "
          f"(PR-AUC = {pr_auc_diff_threshold*100:.1f}pp, F1 = {f1_diff_threshold*100:.1f}pp)")
    print(f"  Underfitting     :  thresholds adjusted per metric based on dataset prevalence.")
    print(sep)

    print(
        f"  {'Metric':<14}"
        f"  {'Baseline':<26}"
        f"  {'Train':>8}"
        f"  {'Underfit threshold':<26}"
        f"  {'Underfit':^10}"
        f"  {'Test':>8}"
        f"  {'Gap':>8}"
        f"  {'Max Gap':>8}"
        f"  {'Overfit':^10}"
    )
    print(sep)

    for _, row in comparison_df.iterrows():
        print(
            f"  {row['Metric']:<14}"
            f"  {row['Baseline_label']:<26}"
            f"  {row['Train']*100:>7.2f}%"
            f"  {row['Underfit_label']:<26}"
            f"  {row['Underfitting']:^10}"
            f"  {row['Test']*100:>7.2f}%"
            f"  {row['Gap (Tr-Te)']*100:>+7.2f}pp"
            f"  {row['Max Gap']*100:>7.1f}pp"
            f"  {row['Overfitting']:^10}"
        )

    print(sep)
    n_overfit  = comparison_df["Overfitting"].str.contains("Yes").sum()
    n_underfit = comparison_df["Underfitting"].str.contains("Yes").sum()
    print(f"\n  Underfitting: {n_underfit} metric(s) flagged  |  "
          f"Overfitting: {n_overfit} metric(s) flagged")
    print(sep + "\n")

    return comparison_df

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

def plot_shap_interpretation(automl, X_test, task="classification", max_display=15, seed=42):
    """
    Computes SHAP values and plots a summary excluding embedding features.
    SHAP provides directional impact per feature (positive = pushes toward positive class).

    For binary classification: single summary plot.
    For multiclass: one summary plot per class.
    """

    print("\n--- Initializing SHAP Explainer ---")
    _start_time = time.time()

    def _prepare_for_shap(X):
        """
        Convert categorical columns to plain int32 for SHAP compatibility.
        LightGBM stores categorical metadata during training and validates
        that input dtypes match — cat.codes returns int8/int16 which can
        still trigger the mismatch error. Casting to int32 avoids this.
        """
        X_shap = X.copy()
        cat_cols = X_shap.select_dtypes(include='category').columns
        for col in cat_cols:
            X_shap[col] = X_shap[col].cat.codes.astype('int32')
        return X_shap

    try:
        # Clinical features only — embeddings excluded for interpretability
        non_emb_cols = [col for col in X_test.columns if not col.startswith('emb')]

        # Extract the underlying fitted estimator from FLAML
        model = automl.model.estimator

        # Subsample X_test for compute efficiency
        sample_size = min(500, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=seed)

        # Convert categoricals to int32 — applied to all model types
        X_sample_shap = _prepare_for_shap(X_sample)

        # Align features to what the model was actually trained on.
        # FLAML may internally select a feature subset, so X_test can have
        # more columns than the underlying estimator expects.
        if hasattr(model, 'feature_name_'):          # LightGBM
            train_features = model.feature_name_
        elif hasattr(model, 'feature_names_in_'):    # XGBoost / sklearn
            train_features = list(model.feature_names_in_)
        else:
            train_features = X_sample_shap.columns.tolist()

        X_sample_shap = X_sample_shap[train_features]

        # Precompute non-embedding column indices once for slicing shap_values arrays
        non_emb_cols = [c for c in train_features if not c.startswith('emb')]
        non_emb_idx  = [train_features.index(c) for c in non_emb_cols]

        # X used for plotting — converted + clinical columns only
        X_plot = X_sample_shap[non_emb_cols]

        tree_model_types = (
            "LGBMClassifier", "XGBClassifier", "RandomForestClassifier",
            "ExtraTreesClassifier", "GradientBoostingClassifier"
        )

        if type(model).__name__ in tree_model_types:
            print(f"Using TreeExplainer for {type(model).__name__}...")

            if type(model).__name__ == "LGBMClassifier":
                # LightGBM stores categorical feature indices in booster_.pandas_categorical
                # and validates them on every prediction call, including those made by SHAP.
                # Converting to int32 is not enough — LightGBM checks its own internal
                # metadata, not the dtype of the input.
                # Fix: temporarily patch pandas_categorical to empty, compute SHAP values,
                # then restore the original metadata unconditionally via finally.
                #
                # booster_ is the underlying Booster object exposed by LGBMClassifier.
                # We probe for pandas_categorical defensively in case the attribute
                # does not exist in older LightGBM versions.
                booster = model.booster_
                original_categorical = getattr(booster, 'pandas_categorical', None)

                if original_categorical is not None:
                    booster.pandas_categorical = []

                try:
                    explainer   = shap.TreeExplainer(
                        model,
                        feature_perturbation="tree_path_dependent"
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="LightGBM binary classifier.*TreeExplainer",
                            category=UserWarning
                        )
                        shap_values = explainer.shap_values(X_sample_shap)
                finally:
                    # Restore original metadata even if SHAP raises an exception
                    if original_categorical is not None:
                        booster.pandas_categorical = original_categorical
            else:
                explainer   = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                shap_values = explainer.shap_values(X_sample_shap)

        else:
            print(f"Using generic Explainer for {type(model).__name__}...")
            background  = X_sample_shap.sample(min(100, len(X_sample_shap)), random_state=seed)
            explainer   = shap.Explainer(model, background)
            shap_out    = explainer(X_sample_shap)
            shap_values = shap_out.values

        print(f"SHAP values computed on {sample_size} samples.")

        # --- Plotting ---
        if task == "multiclass":
            # TreeExplainer returns list of 2D arrays (one per class)
            # Generic Explainer returns 3D array (n_samples, n_features, n_classes)
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
            # Binary: use positive class (index 1)
            # TreeExplainer → list [neg, pos] or single 2D array
            # Generic Explainer → 2D array directly
            if isinstance(shap_values, list):
                shap_values_binary = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values_binary = shap_values[:, :, 1]
            else:
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
        elapsed = time.time() - _start_time
        print(f"Total SHAP time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return shap_values

    except Exception as e:
        print(f"Warning: SHAP interpretation failed: {e}")
        return None
    
def run_binary_automl_model(
    cohort_name: str,
    target_column: str,
    demand_code: Optional[Union[int, List[int]]] = None,
    triage_value: Optional[int] = None,
    include_lr: bool = True,
    include_history: bool = True,
    include_medication: bool = True,
    include_embeddings: bool = False,
    export_table: bool = False, 
    time_budget: int = 600,
    test_size: float = 0.2,
    seed: int = 42,
    min_age: Optional[int] = None,
    years: Optional[List[int]] = None,
    sex_group: bool = False,
    optimize_metric: Optional[str] = None,
    n_splits_cv: int = 5,
    optimize_beta: int = 1,
    use_clinical_threshold: bool = False,
    use_youden: bool = False,
    max_undertriage: float = 0.10,
    max_overtriage: float = 0.50,
    feature_importance: int = 0,
    plot_shap: bool = True,
    plot_calibration_curve: bool = True,
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
    # Print time
    # -----------------------------
    time_print()

    # -----------------------------
    # Print type of triage-specific cohort
    # -----------------------------
    triage_print(triage_value, cohort_name, demand_code)

    # -----------------------------
    # Data loading & column selection
    # -----------------------------
    df = data_load_col_selection(target_column, triage_value, 
                                 include_lr, include_history, include_medication, include_embeddings)

    # -----------------------------
    # Basic validation
    # -----------------------------
    col_validation(df, target_column)

    # -----------------------------
    # Cohort & triage filtering
    # -----------------------------
    df_filtered = data_filtering(df, target_column, demand_code, triage_value, min_age,
                        years, export_table, protocol_features=protocol_questions)

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
    
    if use_clinical_threshold:
        best_threshold = optimize_clinical_threshold(
            y_train, y_train_prob,
            y_test,  y_test_prob,
            max_undertriage=max_undertriage,
            max_overtriage=max_overtriage,
        )
    elif use_youden:
        best_threshold = compute_youden_threshold(
            y_train, y_train_prob,
            y_test,  y_test_prob,
        )
    else:
        best_threshold = set_fb_threshold(y_train, y_train_prob, optimize_beta)
    
    # -----------------------------
    # Test set evaluation
    # -----------------------------
    metrics = print_test_metrics_with_ci(y_test, y_test_prob, threshold=best_threshold)
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    # Calibration plot
    if plot_calibration_curve:
        plot_calibration(y_train, y_train_prob, y_test, y_test_prob, threshold=best_threshold)

    # Train vs Test comparison — overfitting / underfitting diagnosis
    comparison_df = print_train_test_comparison(
        automl, X_train, y_train, X_test, y_test,
        threshold=best_threshold,
        pr_auc_diff_threshold=0.05,
        f1_diff_threshold=0.05,
        overfit_gap_threshold=0.05,   
    )

    # Plot test metrics
    # Create an evaluation dataframe by mapping back to the filtered data
    plot_df = df.loc[y_test.index].copy() 
    plot_df['target_real'] = y_test
    plot_df['target_pred'] = y_test_pred
    eda.evaluate_diagnostic_performance(plot_df, 'target_real', 'target_pred', sex_group = False)

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
