import importlib.util

import numpy as np
import pandas as pd
import pytest

from kbase.paired_comparison import (
    _gee_paired_error_difference,
    _prediction_error_rate,
    calculate_categorical_nri,
    classify_disparity_change,
    compare_model_vs_triage_common,
    compare_paired_proportion,
    generate_outcome_year_bootstrap_indices,
    get_common_comparison_subset,
    matched_operating_point_analysis,
    newcombe_independent_interval,
    newcombe_paired_interval,
    run_paired_inference,
)


def test_newcombe_paired_known_example_and_equal_discordance():
    low, high, difference, phi = newcombe_paired_interval(36, 12, 2, 19)
    assert difference == pytest.approx(0.1449, abs=1e-4)
    assert phi == pytest.approx(0.6057, abs=1e-3)
    assert -1 <= low <= difference <= high <= 1

    model = np.array([1] * 8 + [0] * 2)
    triage = np.array([1] * 6 + [0, 0] + [1, 1])
    result = compare_paired_proportion(model, triage, "equal discordance")
    assert result["b"] == result["c"]
    assert result["difference"] == 0
    assert result["p_value"] == 1
    large_equal = compare_paired_proportion(
        np.array([1] * 30 + [0] * 30),
        np.array([0] * 30 + [1] * 30),
        "large equal discordance",
    )
    assert large_equal["test_reported"] == "Continuity-corrected McNemar chi-square"
    assert large_equal["p_value"] == 1


@pytest.mark.parametrize("b,c,expected_test", [
    (3, 2, "Exact binomial McNemar"),
    (30, 10, "Continuity-corrected McNemar chi-square"),
])
def test_mcnemar_selection_and_optional_statsmodels_agreement(b, c, expected_test):
    a, d = 40, 25
    model = np.array([1] * a + [1] * b + [0] * c + [0] * d)
    triage = np.array([1] * a + [0] * b + [1] * c + [0] * d)
    result = compare_paired_proportion(model, triage, "test")
    assert result["test_reported"] == expected_test
    if importlib.util.find_spec("statsmodels") is not None:
        from statsmodels.stats.contingency_tables import mcnemar

        table = [[a, b], [c, d]]
        if b + c < 25:
            reference = mcnemar(table, exact=True)
            assert result["exact_p_value"] == pytest.approx(reference.pvalue)
        else:
            reference = mcnemar(table, exact=False, correction=True)
            assert result["chi2_corrected_p_value"] == pytest.approx(reference.pvalue)


def _identical_frame():
    return pd.DataFrame({
        "target_real": [0, 0, 1, 1, 0, 1, 0, 1],
        "target_pred_model": [0, 1, 1, 0, 0, 1, 0, 1],
        "p1_assigned": [0, 1, 1, 0, 0, 1, 0, 1],
        "target_prob_model": [0.1, 0.7, 0.8, 0.3, 0.2, 0.9, 0.1, 0.8],
        "year": [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
        "sex": [0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_identical_systems_have_zero_differences_nri_and_sex_dd():
    frame = _identical_frame()
    result = run_paired_inference(frame, n_bootstrap=50, seed=42)
    for key in ("sensitivity", "specificity", "undertriage", "overtriage"):
        assert result[key]["difference"] == pytest.approx(0)
    assert result["nri"]["nri_overall"] == pytest.approx(0)
    for row in result["sex_gaps"]["rows"]:
        assert row["difference_in_differences"] == pytest.approx(0)


def test_nri_identities_on_fixed_synthetic_data():
    rng = np.random.default_rng(123)
    n = 5000
    y = rng.binomial(1, 0.3, n)
    triage = rng.binomial(1, np.where(y == 1, 0.45, 0.25))
    model = rng.binomial(1, np.where(y == 1, 0.70, 0.40))
    result = calculate_categorical_nri(y, model, triage)
    event = y == 1
    nonevent = y == 0
    delta_sensitivity = model[event].mean() - triage[event].mean()
    delta_specificity = (1 - model[nonevent]).mean() - (1 - triage[nonevent]).mean()
    assert result["nri_events"] == pytest.approx(delta_sensitivity, abs=1e-12)
    assert result["nri_nonevents"] == pytest.approx(delta_specificity, abs=1e-12)
    assert result["nri_overall"] == pytest.approx(
        delta_sensitivity + delta_specificity, abs=1e-12
    )


@pytest.mark.skipif(
    importlib.util.find_spec("statsmodels") is None,
    reason="statsmodels is not installed",
)
def test_gee_identity_coefficient_matches_manual_difference_and_identical_p():
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0] * 20)
    model = np.array([0, 0, 0, 1, 1, 0, 0, 0] * 20)
    triage = np.array([0, 0, 1, 1, 0, 0, 0, 0] * 20)
    manual = (
        _prediction_error_rate(y, model, "undertriage")[0]
        - _prediction_error_rate(y, triage, "undertriage")[0]
    )
    result = _gee_paired_error_difference(y, model, triage, "undertriage")
    assert result["difference"] == pytest.approx(manual, abs=1e-8)
    identical = _gee_paired_error_difference(y, model, model, "undertriage")
    assert identical["p_value"] == pytest.approx(1, abs=1e-6)


def test_bootstrap_is_reproducible_and_preserves_stratum_sizes():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    year = np.array([2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021])
    first = generate_outcome_year_bootstrap_indices(y, year, 20, 42)
    second = generate_outcome_year_bootstrap_indices(y, year, 20, 42)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    original_cells = pd.Series(list(zip(y, year))).value_counts().to_dict()
    for positions in first:
        replicate_cells = pd.Series(list(zip(y[positions], year[positions]))).value_counts().to_dict()
        assert replicate_cells == original_cells


def test_newcombe_independent_extremes_stay_in_parameter_space():
    low, high = newcombe_independent_interval(20, 20, 0, 20)
    assert -1 <= low <= high <= 1


def test_matched_thresholds_depend_only_on_training_data():
    frame = _identical_frame()
    y_train = pd.Series([0, 0, 0, 1, 1, 1], index=np.arange(100, 106))
    probabilities = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.9])
    triage = pd.Series([0, 0, 1, 1, 1, 0], index=y_train.index)
    years = pd.Series([2020, 2020, 2021, 2020, 2021, 2021], index=y_train.index)
    first = matched_operating_point_analysis(
        frame, "oof", y_train, probabilities, triage, years, 0.5
    )
    changed_test = frame.copy()
    changed_test["target_prob_model"] = 1 - changed_test["target_prob_model"]
    second = matched_operating_point_analysis(
        changed_test, "oof", y_train, probabilities, triage, years, 0.5
    )
    assert first["tau_spec"] == second["tau_spec"]
    assert first["tau_sens"] == second["tau_sens"]
    assert len(first["table"]) == 4
    assert set(first["table"]["Metric"]) == {"Sensitivity", "Specificity"}


def test_disparity_change_classification():
    assert classify_disparity_change(0.02, -0.03) == "reverses"
    assert classify_disparity_change(-0.07, -0.10) == "widens"


def test_common_subset_is_the_same_for_all_inference_sections():
    frame = _identical_frame()
    frame.loc[1, "p1_assigned"] = np.nan
    common, report = get_common_comparison_subset(frame)
    result = compare_model_vs_triage_common(
        frame, export_tables=False, run_inference=True, n_bootstrap=30
    )
    inference = result["paired_inference"]
    assert result["report"]["n_common"] == report["n_common"] == len(common)
    assert inference["sensitivity"]["n"] + inference["specificity"]["n"] == len(common)
    assert len(inference["bootstrap_indices"][0]) == len(common)
    assert inference["sex_gaps"]["n_men"] + inference["sex_gaps"]["n_women"] == len(common)
