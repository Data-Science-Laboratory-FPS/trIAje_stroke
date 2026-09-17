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
# Source: FLAML lgbm configuration selected on 2026-08-21.
FIXED_LGBM_STROKE_HIST_P1_REAL_EMERG_BPS_20260821 = {
    "learning_rate": 0.08368102183669972,
    "num_leaves": 10,
    "n_estimators": 78,
    "min_child_samples": 15,
    "colsample_bytree": 0.8829297501914448,
    "reg_alpha": 0.0048790014346145345,
    "reg_lambda": 0.018013859492223706,
    "log_max_bin": 7,
}

# Backward-friendly short alias for the current reference configuration.
FIXED_LGBM_CONFIG = FIXED_LGBM_STROKE_HIST_P1_REAL_EMERG_BPS_20260821

AVAILABLE_CONFIGS = {
    "stroke_hist_p1_real_emerg_bps_lgbm_20260821": (
        FIXED_LGBM_STROKE_HIST_P1_REAL_EMERG_BPS_20260821
    ),
}
