"""Fixed hyperparameter configurations for reproducible model comparisons.

This module centralizes project-level fixed model configurations. These
configurations are intended for reproducible comparisons between feature sets,
so that observed performance differences are attributable to the features
rather than to FLAML search variability.

Naming convention:
    FIXED_<ESTIMATOR>_<COHORT>_<FEATURE_SET>_<OUTCOME>_<SOURCE>

Future configurations for other cohorts, outcomes, or feature sets (for
example legacy ``com_`` columns) should follow the same convention and be added
to ``AVAILABLE_CONFIGS`` with a readable key.
"""

# Stroke, current hist_ columns, outcome p1_real_emerg_bps.
# Source: FLAML xgboost configuration selected with a 1800-second search budget
# on 2026-07-23 at 12:18. This was the most heavily regularized of five
# observed configurations and had a test ROC-AUC within the observed 0.17pp
# band while producing the smallest train-test gap.
FIXED_XGBOOST_STROKE_HIST_P1_REAL_EMERG_BPS_20260723 = {
    "colsample_bylevel": 0.8645517069844122,
    "colsample_bytree": 0.9171509896058663,
    "learning_rate": 0.023777042000665518,
    "max_leaves": 9,
    "min_child_weight": 0.8420281826538534,
    "n_estimators": 620,
    "reg_alpha": 0.0009765625,
    "reg_lambda": 32.603752855056904,
    "subsample": 0.8628114965887879,
}

# Backward-friendly short alias for the current reference configuration.
FIXED_XGBOOST_CONFIG = FIXED_XGBOOST_STROKE_HIST_P1_REAL_EMERG_BPS_20260723

AVAILABLE_CONFIGS = {
    "stroke_hist_p1_real_emerg_bps_xgboost_20260723": (
        FIXED_XGBOOST_STROKE_HIST_P1_REAL_EMERG_BPS_20260723
    ),
}
