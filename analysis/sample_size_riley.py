"""Minimum sample size for binary prediction-model development.

This script applies the three Riley et al. criteria implemented by
``pmsampsize`` and verifies the package results against the published closed
form equations. The C statistic is converted to Cox-Snell R-squared by the
package's implementation of Riley, Van Calster, and Collins, which uses a
fixed-seed simulation of 1,000,000 observations.

Run from the analysis directory with::

    python sample_size_riley.py
"""

import inspect
import io
import json
import math
import os
import platform
from contextlib import redirect_stdout
from importlib.metadata import metadata, version
from pathlib import Path
from typing import Dict

from pmsampsize.pmsampsize import cstat2rsq, pmsampsize
from tabulate import tabulate


OUTCOME = "p1_real_emerg_bps"
DEMAND_CODE = 54
COHORT_N = 46_163
COHORT_EVENTS = 14_039
PREVALENCE = COHORT_EVENTS / COHORT_N
CANDIDATE_PARAMETERS = 186
TARGET_SHRINKAGE = 0.9
INTERCEPT_MARGIN = 0.05
TRAINING_N = 36_930
TRAINING_EVENTS = 11_232
TEST_N = 9_233
SEED = 123_456
SIMULATION_N = 1_000_000

# Verified in figures/04_stroke/tables/cv_fold_metrics.docx: mean ROC-AUC
# 0.6707, rounded to 0.671 for the primary calculation.
CV_C_STATISTIC_EXACT = 0.6707
CV_C_STATISTIC = 0.671

# Verified in the final fixed-configuration model output: test ROC-AUC
# 0.671544, rounded to 0.672 for the sensitivity analysis.
TEST_C_STATISTIC_EXACT = 0.671544
TEST_C_STATISTIC = 0.672

OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "sample_size_riley.json"


def verify_pipeline_inputs() -> Dict[str, object]:
    """Recreate the project's train/test matrix and verify all supplied counts."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/triaje-matplotlib")

    # The pipeline functions are intentionally reused without modifying them.
    # Their verbose diagnostic output is suppressed here so that this script
    # prints only the final sample-size table.
    with redirect_stdout(io.StringIO()):
        from kbase import modeling as ml

        data = ml.data_load_col_selection(
            target_column=OUTCOME,
            triage_value=None,
            include_lr=True,
            include_hist=True,
            include_com=False,
            include_medication=True,
            include_embeddings=False,
        )
        ml.col_validation(data, OUTCOME)
        filtered = ml.data_filtering(
            df=data,
            target_column=OUTCOME,
            demand_code=DEMAND_CODE,
            triage_value=None,
            min_age=None,
            years=None,
            export_table=False,
            protocol_features=ml.protocol_questions,
        )
        X_train, X_test, y_train, y_test, _, _, _ = ml.train_test_split_weights(
            filtered,
            target_column=OUTCOME,
            test_size=0.2,
            seed=42,
        )

    observed = {
        "outcome_type": "binary",
        "outcome_column": OUTCOME,
        "cohort_n": int(len(filtered)),
        "cohort_events": int(filtered[OUTCOME].sum()),
        "cohort_prevalence": float(filtered[OUTCOME].mean()),
        "training_n": int(len(X_train)),
        "training_events": int(y_train.sum()),
        "test_n": int(len(X_test)),
        "test_events": int(y_test.sum()),
        "candidate_parameters": int(X_train.shape[1]),
        "feature_families": {
            "triage_questions": sum(column[0].isdigit() for column in X_train.columns),
            "literal_reason": sum(column.startswith("lr_") for column in X_train.columns),
            "clinical_history": sum(column.startswith("hist_") for column in X_train.columns),
            "legacy_history": sum(column.startswith("com_") for column in X_train.columns),
            "medication": sum(column.startswith("atc") for column in X_train.columns),
        },
    }
    classified = sum(observed["feature_families"].values())
    observed["feature_families"]["other"] = int(X_train.shape[1] - classified)

    expected = {
        "cohort_n": COHORT_N,
        "cohort_events": COHORT_EVENTS,
        "training_n": TRAINING_N,
        "training_events": TRAINING_EVENTS,
        "test_n": TEST_N,
        "candidate_parameters": CANDIDATE_PARAMETERS,
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            raise AssertionError(
                f"Pipeline verification failed for {key}: "
                f"observed {observed[key]}, expected {expected_value}."
            )
    if set(filtered[OUTCOME].dropna().unique()) != {0, 1}:
        raise AssertionError("The outcome is not binary after cohort filtering.")

    observed["verified"] = True
    return observed


def manual_binary_criteria(r2_cs: float) -> Dict[str, object]:
    """Calculate the three binary-outcome criteria from Riley et al."""
    p = PREVALENCE
    max_r2_cs = 1 - math.exp(2 * (p * math.log(p) + (1 - p) * math.log(1 - p)))
    if r2_cs >= max_r2_cs:
        raise ValueError("Cox-Snell R-squared must be below its theoretical maximum.")

    n_shrinkage = math.ceil(
        CANDIDATE_PARAMETERS
        / ((TARGET_SHRINKAGE - 1) * math.log(1 - r2_cs / TARGET_SHRINKAGE))
    )
    criterion_2_shrinkage = r2_cs / (r2_cs + INTERCEPT_MARGIN * max_r2_cs)
    n_r2_difference = math.ceil(
        CANDIDATE_PARAMETERS
        / (
            (criterion_2_shrinkage - 1)
            * math.log(1 - r2_cs / criterion_2_shrinkage)
        )
    )
    n_intercept = math.ceil(
        (1.96 / INTERCEPT_MARGIN) ** 2 * p * (1 - p)
    )

    return {
        "r2_cs_max": max_r2_cs,
        "r2_nagelkerke": r2_cs / max_r2_cs,
        "criterion_1_shrinkage": {
            "target": TARGET_SHRINKAGE,
            "minimum_n": n_shrinkage,
        },
        "criterion_2_r2_difference": {
            "maximum_difference": INTERCEPT_MARGIN,
            "implied_shrinkage": criterion_2_shrinkage,
            "minimum_n": n_r2_difference,
        },
        "criterion_3_intercept_precision": {
            "margin": INTERCEPT_MARGIN,
            "minimum_n": n_intercept,
        },
        "minimum_n": max(n_shrinkage, n_r2_difference, n_intercept),
    }


def run_scenario(name: str, c_statistic: float) -> Dict[str, object]:
    """Run pmsampsize and independently verify its three criteria."""
    conversion = cstat2rsq(
        cstatistic=c_statistic,
        prevalence=PREVALENCE,
        seed=SEED,
        noprint=True,
    )
    raw_r2_cs = float(conversion["R2_coxsnell"])
    package_result = pmsampsize(
        type="b",
        cstatistic=c_statistic,
        parameters=CANDIDATE_PARAMETERS,
        prevalence=PREVALENCE,
        shrinkage=TARGET_SHRINKAGE,
        seed=SEED,
        noprint=True,
    )
    r2_cs_used = float(package_result["rsquared"])
    manual = manual_binary_criteria(r2_cs_used)

    package_criteria = [
        int(package_result["results_table"][index][1]) for index in range(3)
    ]
    manual_criteria = [
        manual["criterion_1_shrinkage"]["minimum_n"],
        manual["criterion_2_r2_difference"]["minimum_n"],
        manual["criterion_3_intercept_precision"]["minimum_n"],
    ]
    if package_criteria != manual_criteria:
        raise AssertionError(
            f"Manual criteria {manual_criteria} do not match pmsampsize "
            f"criteria {package_criteria}."
        )
    if int(package_result["sample_size"]) != manual["minimum_n"]:
        raise AssertionError("Manual and pmsampsize final sample sizes differ.")

    minimum_n = int(package_result["sample_size"])
    required_events = math.ceil(minimum_n * PREVALENCE)
    required_epp = minimum_n * PREVALENCE / CANDIDATE_PARAMETERS
    observed_epp = TRAINING_EVENTS / CANDIDATE_PARAMETERS
    training_meets = TRAINING_N >= minimum_n and TRAINING_EVENTS >= required_events

    return {
        "name": name,
        "c_statistic": c_statistic,
        "simulation_seed": SEED,
        "simulation_n": SIMULATION_N,
        "r2_cs_simulated_unrounded": raw_r2_cs,
        "r2_cs_used_by_pmsampsize": r2_cs_used,
        "r2_cs_max": manual["r2_cs_max"],
        "r2_nagelkerke": manual["r2_nagelkerke"],
        "criteria": {
            "shrinkage_at_least_0.9": manual["criterion_1_shrinkage"],
            "nagelkerke_r2_difference_at_most_0.05": manual[
                "criterion_2_r2_difference"
            ],
            "intercept_margin_plus_minus_0.05": manual[
                "criterion_3_intercept_precision"
            ],
        },
        "minimum_n": minimum_n,
        "required_events": required_events,
        "required_events_per_candidate_parameter": required_epp,
        "observed_events_per_candidate_parameter": observed_epp,
        "training_meets_minimum": training_meets,
        "training_n_minus_minimum": TRAINING_N - minimum_n,
        "training_events_minus_required": TRAINING_EVENTS - required_events,
        "manual_formula_verification": True,
    }


def verify_documented_example() -> Dict[str, object]:
    """Reproduce the package's documented binary Cox-Snell R-squared example."""
    result = pmsampsize(
        type="b",
        csrsquared=0.288,
        parameters=24,
        prevalence=0.174,
        shrinkage=0.9,
        noprint=True,
    )
    observed_criteria = [int(result["results_table"][i][1]) for i in range(3)]
    expected_criteria = [623, 662, 221]
    if observed_criteria != expected_criteria or int(result["sample_size"]) != 662:
        raise AssertionError("The documented pmsampsize binary example was not reproduced.")
    return {
        "inputs": {
            "r2_cs": 0.288,
            "candidate_parameters": 24,
            "prevalence": 0.174,
        },
        "criterion_minimum_n": observed_criteria,
        "final_minimum_n": int(result["sample_size"]),
        "verified": True,
    }


def main() -> None:
    signature = inspect.signature(pmsampsize)
    if "cstatistic" not in signature.parameters:
        raise RuntimeError("Installed pmsampsize does not support cstatistic.")

    verified_inputs = verify_pipeline_inputs()
    primary = run_scenario("Primary: cross-validation mean ROC-AUC", CV_C_STATISTIC)
    sensitivity = run_scenario("Sensitivity: test ROC-AUC", TEST_C_STATISTIC)
    documented_example = verify_documented_example()

    methods_sentence = (
        "No formal sample size calculation was performed, as all eligible calls "
        "were included. Following Riley et al. [REF], with 186 candidate "
        "parameters, an outcome prevalence of 0.304 and an anticipated "
        f"Cox–Snell R² of {primary['r2_cs_used_by_pmsampsize']:.4f} "
        f"(corresponding to a C statistic of {primary['c_statistic']:.3f}), at "
        f"least {primary['minimum_n']:,} observations "
        f"({primary['required_events']:,} events) were required to limit "
        "overfitting (target shrinkage 0.9), fewer than the 36,930 calls "
        "(11,232 events) in the training set. As these criteria were developed "
        "for regression models, machine learning methods may require larger samples."
    )

    package_metadata = metadata("pmsampsize")
    output = {
        "analysis": "Riley minimum sample size for binary prediction-model development",
        "references": [
            {
                "citation": "Riley et al. BMJ 2020;368:m441",
                "doi": "10.1136/bmj.m441",
            },
            {
                "citation": (
                    "Riley, Van Calster, and Collins. Statistics in Medicine "
                    "2021;40:859-864"
                ),
                "doi": "10.1002/sim.8806",
            },
        ],
        "software": {
            "python_runtime": platform.python_version(),
            "pmsampsize_version": version("pmsampsize"),
            "pmsampsize_requires_python": package_metadata.get("Requires-Python"),
            "supports_cstatistic": True,
        },
        "verified_inputs": {
            **verified_inputs,
            "prevalence_used": PREVALENCE,
            "prevalence_reported": round(PREVALENCE, 3),
            "target_shrinkage": TARGET_SHRINKAGE,
            "cv_roc_auc_exact": CV_C_STATISTIC_EXACT,
            "cv_roc_auc_used": CV_C_STATISTIC,
            "test_roc_auc_exact": TEST_C_STATISTIC_EXACT,
            "test_roc_auc_used": TEST_C_STATISTIC,
        },
        "primary": primary,
        "sensitivity": sensitivity,
        "documented_example_verification": documented_example,
        "methods_sentence": methods_sentence,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for scenario in (primary, sensitivity):
        criteria = scenario["criteria"]
        rows.append(
            [
                scenario["name"],
                f"{scenario['c_statistic']:.3f}",
                f"{scenario['r2_cs_used_by_pmsampsize']:.4f}",
                criteria["shrinkage_at_least_0.9"]["minimum_n"],
                criteria["nagelkerke_r2_difference_at_most_0.05"]["minimum_n"],
                criteria["intercept_margin_plus_minus_0.05"]["minimum_n"],
                scenario["minimum_n"],
                scenario["required_events"],
                f"{scenario['required_events_per_candidate_parameter']:.2f}",
                "Yes" if scenario["training_meets_minimum"] else "No",
            ]
        )
    print(
        tabulate(
            rows,
            headers=[
                "Scenario",
                "C",
                "R²_CS",
                "n: shrinkage",
                "n: ΔR²",
                "n: intercept",
                "Final n",
                "Events",
                "Required EPP",
                "Training meets",
            ],
            tablefmt="github",
        )
    )
    print(f"\nObserved EPP: {TRAINING_EVENTS / CANDIDATE_PARAMETERS:.2f}")
    print(f"Result written to: {OUTPUT_PATH}")
    print(f"\nMethods sentence:\n{methods_sentence}")


if __name__ == "__main__":
    main()
