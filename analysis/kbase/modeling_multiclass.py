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

    global _current_demand_code, _current_time_budget
    _current_demand_code = demand_code
    _current_time_budget = time_budget

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
    X_train, X_test, y_train, y_test, year_train, year_test, sample_weight = train_test_split_weights(
        df, target_column, test_size, seed)

    # -----------------------------
    # AutoML training (task set to multiclass)
    # -----------------------------
    automl = run_automl_training(
        X_train, y_train, year_train, sample_weight, time_budget, 
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