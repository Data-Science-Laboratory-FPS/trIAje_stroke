#!/usr/bin/env python3
"""Stroke-general benchmark with AP screening/HPO and recall-constrained OOF thresholding."""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "experiments" / "stroke_03_constrained_runs" / "general"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))

STABLE_MODEL_ORDER = [
    "XGBoost",
    "Random Forest",
    "Logistic Regression",
    "Elastic Net Logistic",
    "Extra Trees",
    "LightGBM",
    "HistGradientBoosting",
]

MODEL_ORDER_INDEX = {name: index for index, name in enumerate(STABLE_MODEL_ORDER)}


def model_order_key(name: str) -> int:
    return MODEL_ORDER_INDEX.get(name, len(MODEL_ORDER_INDEX))


def ordered_names(names):
    return sorted(names, key=model_order_key)


def setup_env() -> None:
    from run_stroke_general_ref03 import configure_environment

    configure_environment()


def build_stable_models(pos_weight: float):
    from scipy.stats import loguniform, randint, uniform
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    def pipe(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    return {
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            {"model__C": loguniform(1e-4, 10.0)},
        ),
        "Elastic Net Logistic": (
            pipe(LogisticRegression(max_iter=600, tol=1e-3, class_weight="balanced", solver="saga", penalty="elasticnet", l1_ratio=0.5, random_state=42)),
            {"model__C": loguniform(1e-4, 5.0), "model__l1_ratio": uniform(0.0, 1.0)},
        ),
        "Random Forest": (
            pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1)),
            {"model__n_estimators": randint(100, 500), "model__max_depth": [None, 5, 10, 15, 20], "model__min_samples_leaf": randint(1, 20), "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]},
        ),
        "XGBoost": (
            pipe(xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42, verbosity=0, eval_metric="logloss", n_jobs=1)),
            {"model__n_estimators": randint(100, 600), "model__learning_rate": loguniform(0.01, 0.3), "model__max_depth": randint(3, 10), "model__min_child_weight": randint(1, 10), "model__subsample": uniform(0.5, 0.5), "model__colsample_bytree": uniform(0.5, 0.5), "model__gamma": uniform(0, 1), "model__scale_pos_weight": [1.0, pos_weight * 0.5, pos_weight, pos_weight * 1.5]},
        ),
        "LightGBM": (
            pipe(__import__("lightgbm").LGBMClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42, verbosity=-1, n_jobs=1)),
            {"model__n_estimators": randint(100, 700), "model__learning_rate": loguniform(0.01, 0.25), "model__num_leaves": randint(15, 128), "model__max_depth": [-1, 4, 6, 8, 12], "model__min_child_samples": randint(10, 80), "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.6, 0.4), "model__reg_lambda": loguniform(1e-4, 10.0), "model__scale_pos_weight": [1.0, pos_weight * 0.5, pos_weight, pos_weight * 1.5]},
        ),
        "Extra Trees": (
            pipe(ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1)),
            {"model__n_estimators": randint(100, 500), "model__max_depth": [None, 5, 10, 15, 20], "model__min_samples_leaf": randint(1, 20), "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]},
        ),
        "HistGradientBoosting": (
            pipe(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, l2_regularization=0.1, class_weight="balanced", random_state=42)),
            {"model__learning_rate": loguniform(0.01, 0.3), "model__max_iter": randint(100, 500), "model__max_leaf_nodes": randint(15, 100), "model__min_samples_leaf": randint(5, 50), "model__l2_regularization": loguniform(1e-5, 1.0)},
        ),
    }


def atomic_write(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def safe_clone(estimator):
    try:
        return clone(estimator)
    except RuntimeError:
        return copy.deepcopy(estimator)


def fbeta(precision: float, recall: float, beta: float) -> float:
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom else 0.0


def point_at_threshold(y_true, probabilities, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    pred = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "ppv": float(precision),
        "npv": float(npv),
        "f1": float(fbeta(precision, recall, 1.0)),
        "f2": float(fbeta(precision, recall, 2.0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def sweep_points(y_true, probabilities) -> list[dict]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_y = y_true[order]
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    cumulative_positives = np.cumsum(sorted_y)
    points = []
    for end in np.r_[np.flatnonzero(np.diff(sorted_probabilities)) + 1, len(sorted_probabilities)]:
        tp = int(cumulative_positives[end - 1])
        fp = int(end - tp)
        fn = positives - tp
        tn = negatives - fp
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        npv = tn / (tn + fn) if tn + fn else 0.0
        points.append({
            "threshold": float(sorted_probabilities[end - 1]),
            "recall": float(recall),
            "specificity": float(specificity),
            "precision": float(precision),
            "ppv": float(precision),
            "npv": float(npv),
            "f1": float(fbeta(precision, recall, 1.0)),
            "f2": float(fbeta(precision, recall, 2.0)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })
    all_positive = point_at_threshold(y_true, probabilities, np.nextafter(float(probabilities.min()), -np.inf))
    if not points or all_positive["threshold"] != points[-1]["threshold"]:
        points.append(all_positive)
    return points


def best_fbeta_point(y_true, probabilities, beta: float) -> dict:
    key = "f2" if beta == 2.0 else "f1"
    return max(sweep_points(y_true, probabilities), key=lambda p: (p[key], p["recall"], p["specificity"]))


def max_spec_at_recall_point(y_true, probabilities, min_recall: float) -> dict:
    points = sweep_points(y_true, probabilities)
    for point in points:
        point["feasible"] = point["recall"] >= min_recall
        point["min_recall"] = min_recall
    feasible = [point for point in points if point["feasible"]]
    if feasible:
        return max(feasible, key=lambda p: (p["specificity"], p["recall"], -p["fp"]))
    return max(points, key=lambda p: (p["recall"], p["specificity"], -p["fp"]))


def probability_metrics(y_true, probabilities) -> dict:
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def all_operating_points(y_true, probabilities) -> dict:
    primary = max_spec_at_recall_point(y_true, probabilities, 0.80)
    return {
        "primary_recall_0.80": primary,
        "recall_0.80": primary,
        "recall_0.85": max_spec_at_recall_point(y_true, probabilities, 0.85),
        "recall_0.90": max_spec_at_recall_point(y_true, probabilities, 0.90),
        "f1": best_fbeta_point(y_true, probabilities, 1.0),
        "f2": best_fbeta_point(y_true, probabilities, 2.0),
    }


def metrics_at_operating_point(y_true, probabilities, point: dict) -> dict:
    return {
        **point_at_threshold(y_true, probabilities, point["threshold"]),
        **probability_metrics(y_true, probabilities),
    }


def calibration_bins(y_true, probabilities, n_bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probabilities >= edges[i]) & (probabilities <= edges[i + 1])
        else:
            mask = (probabilities >= edges[i]) & (probabilities < edges[i + 1])
        if not np.any(mask):
            continue
        bins.append({
            "bin_low": float(edges[i]),
            "bin_high": float(edges[i + 1]),
            "n": int(mask.sum()),
            "mean_predicted": float(probabilities[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })
    return bins


def write_worker_progress(progress_path: Path, worker_key: str, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = progress_path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
            current[worker_key] = payload
            atomic_write(progress_path, current)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def fit_predict_fold(estimator, X, y, fit_idx, validation_idx, fold_number, progress_path, phase, model_name):
    pid = os.getpid()
    worker_key = f"pid-{pid}-fold-{fold_number}"
    started = time.time()
    write_worker_progress(progress_path, worker_key, {
        "pid": pid,
        "fold": fold_number,
        "phase": phase,
        "model": model_name,
        "status": "running",
        "started": started,
        "elapsed": 0.0,
    })
    fold_estimator = safe_clone(estimator)
    fold_estimator.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    probabilities = fold_estimator.predict_proba(X.iloc[validation_idx])[:, 1]
    write_worker_progress(progress_path, worker_key, {
        "pid": pid,
        "fold": fold_number,
        "phase": phase,
        "model": model_name,
        "status": "done",
        "started": started,
        "elapsed": time.time() - started,
    })
    return validation_idx, probabilities


def oof_predictions(estimator, X, y, n_splits: int, n_jobs: int, progress_path: Path, phase: str, model_name: str):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_splits = list(cv.split(X, y))
    atomic_write(progress_path, {})
    results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=1, pre_dispatch=n_jobs)(
        delayed(fit_predict_fold)(estimator, X, y, fit_idx, val_idx, fold, progress_path, phase, model_name)
        for fold, (fit_idx, val_idx) in enumerate(fold_splits, start=1)
    )
    probabilities = np.zeros(len(y), dtype=float)
    for validation_idx, fold_probabilities in results:
        probabilities[validation_idx] = fold_probabilities
    return probabilities


def candidate_dicts(distributions: dict, n_iter: int) -> list[dict]:
    if n_iter <= 0:
        return [{}]
    return list(ParameterSampler(distributions, n_iter=n_iter, random_state=42))


def update_state(path: Path, state: dict, **updates) -> None:
    state.update(updates)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(path, state)


def start_refit_heartbeat(state_path: Path, state: dict, model_name: str, best_row: dict, estimate_seconds: float):
    stop_event = threading.Event()
    started = time.time()

    def heartbeat():
        while not stop_event.wait(15):
            elapsed = time.time() - started
            remaining = max(0.0, estimate_seconds - elapsed) if estimate_seconds else None
            state["refit_progress"] = {
                "model": model_name,
                "status": "running",
                "elapsed_seconds": elapsed,
                "eta_seconds": remaining,
                "best_hpo_candidate": best_row.get("candidate"),
                "best_hpo_average_precision": best_row.get("average_precision"),
                "best_hpo_log_loss": best_row.get("log_loss"),
                "params": best_row.get("params", {}),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            update_state(state_path, state)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    return stop_event


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-folds", type=int, default=3)
    parser.add_argument("--hpo-folds", type=int, default=5)
    parser.add_argument("--oof-folds", type=int, default=10)
    parser.add_argument("--hpo-candidates", type=int, default=12)
    parser.add_argument("--survivors", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_env()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "training.log"
    state_path = RUN_DIR / "state.json"
    split_path = RUN_DIR / "split.npz"
    result_path = RUN_DIR / "results.json"
    progress_path = RUN_DIR / "parallel_progress.json"

    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        started = datetime.now(timezone.utc).isoformat()
        print(f"\nstarted={started} objective=max_oof_specificity_at_recall_0.80 resume={args.resume}", flush=True)
        try:
            from run_stroke_general_ref03 import prepare_data

            X, y, _year = prepare_data()
            if split_path.exists() and args.resume:
                split = np.load(split_path)
                train_idx, test_idx = split["train_idx"], split["test_idx"]
            else:
                train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.20, random_state=42, stratify=y)
                np.savez(split_path, train_idx=train_idx, test_idx=test_idx)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pos_weight = float((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
            all_specs = build_stable_models(pos_weight)
            specs = {name: all_specs[name] for name in STABLE_MODEL_ORDER if name in all_specs}

            if args.resume and state_path.exists():
                state = json.loads(state_path.read_text())
                state.update({
                    "status": "running",
                    "resumed_at": started,
                    "screening_folds": args.screening_folds,
                    "hpo_folds": args.hpo_folds,
                    "oof_folds": args.oof_folds,
                    "hpo_candidates": args.hpo_candidates,
                    "survivors": args.survivors,
                    "model_total": len(specs),
                    "model_order": STABLE_MODEL_ORDER,
                })
                state.setdefault("completed_models", {})
                state.setdefault("models", {})
                state.setdefault("failed_models", {})
            else:
                state = {
                    "status": "running",
                    "started": started,
                    "objective": "max_oof_specificity_at_recall_0.80",
                    "strategy": "3-fold AP screening -> 5-fold AP HPO -> 10-fold OOF specificity threshold at recall >= 0.80",
                    "primary_objective": "MAXIMISE OOF SPECIFICITY SUBJECT TO RECALL >= 0.80",
                    "screening_folds": args.screening_folds,
                    "hpo_folds": args.hpo_folds,
                    "oof_folds": args.oof_folds,
                    "hpo_candidates": args.hpo_candidates,
                    "survivors": args.survivors,
                    "model_total": len(specs),
                    "model_order": STABLE_MODEL_ORDER,
                    "completed_models": {},
                    "models": {},
                    "failed_models": {},
                }
            update_state(state_path, state, phase="loading data", active_model=None)

            screening = []
            for model_name, (base_estimator, _distributions) in specs.items():
                existing_screen = state["models"].get(model_name, {}).get("screening")
                if existing_screen:
                    screening.append((model_name, existing_screen))
                    continue
                print(f"\nPHASE 1 SCREEN {model_name} folds={args.screening_folds}", flush=True)
                update_state(
                    state_path,
                    state,
                    phase="PHASE 1 - fast model screening (Average Precision)",
                    active_model=model_name,
                    active_candidate=0,
                    active_total=0,
                    active_fold=0,
                    active_total_folds=args.screening_folds,
                )
                started_model = time.time()
                try:
                    probabilities = oof_predictions(
                        base_estimator, X_train, y_train, args.screening_folds, args.n_jobs,
                        progress_path, "screening", model_name,
                    )
                    row = {
                        **probability_metrics(y_train, probabilities),
                        "elapsed_seconds": time.time() - started_model,
                    }
                    state["models"][model_name] = {"screening": row, "status": "screened"}
                    screening.append((model_name, row))
                    update_state(state_path, state, active_fold=args.screening_folds)
                    print(json.dumps({"model": model_name, "screening_ap": row["average_precision"], "screening_auroc": row["auroc"]}), flush=True)
                except Exception:
                    tb = traceback.format_exc()
                    print(f"SCREENING FAILED {model_name}:\n{tb}", flush=True)
                    state["failed_models"][model_name] = {"phase": "screening", "error": tb.strip().splitlines()[-1]}
                    update_state(state_path, state)

            ranked_screening = sorted(screening, key=lambda item: item[1]["average_precision"], reverse=True)
            selected_survivors = [name for name, _row in ranked_screening[: args.survivors]]
            survivors = ordered_names(selected_survivors)
            resume_active = state.get("active_model") if args.resume else None
            if resume_active in survivors and resume_active not in state["completed_models"]:
                survivors = [resume_active] + [name for name in survivors if name != resume_active]
            state["screening_ranking"] = [{"model": name, **row} for name, row in ranked_screening]
            state["survivor_models"] = survivors
            state["survivor_selection"] = "top screening Average Precision, executed in requested model order"
            update_state(state_path, state, phase="PHASE 1 complete")
            print(f"SURVIVORS {survivors}", flush=True)

            for model_name in survivors:
                if model_name in state["completed_models"]:
                    continue
                base_estimator, distributions = specs[model_name]
                print(f"\nPHASE 2 HPO {model_name} candidates={args.hpo_candidates} folds={args.hpo_folds}", flush=True)
                update_state(
                    state_path,
                    state,
                    phase="PHASE 2 - AP hyperparameter tuning",
                    active_model=model_name,
                    active_candidate=0,
                    active_total=args.hpo_candidates,
                    active_fold=0,
                    active_total_folds=args.hpo_folds,
                )
                model_started = time.time()
                model_state = state["models"].setdefault(model_name, {})
                try:
                    hpo_rows = model_state.get("hpo_history", []) if args.resume else []
                    best_row = model_state.get("best_hpo") if args.resume else None
                    if best_row and len(hpo_rows) >= args.hpo_candidates:
                        print(json.dumps({"model": model_name, "resume": "using_existing_hpo", "hpo_candidates": len(hpo_rows), "best_hpo_ap": best_row["average_precision"]}), flush=True)
                    else:
                        hpo_rows = []
                        best_row = None
                        for index, params in enumerate(candidate_dicts(distributions, args.hpo_candidates), start=1):
                            candidate_started = time.time()
                            update_state(state_path, state, active_candidate=index, active_fold=0)
                            estimator = safe_clone(base_estimator).set_params(**params)
                            probabilities = oof_predictions(
                                estimator, X_train, y_train, args.hpo_folds, args.n_jobs,
                                progress_path, "hpo", model_name,
                            )
                            row = {
                                "candidate": index,
                                "params": params,
                                **probability_metrics(y_train, probabilities),
                                "elapsed_seconds": time.time() - candidate_started,
                            }
                            hpo_rows.append(row)
                            if best_row is None or row["average_precision"] > best_row["average_precision"]:
                                best_row = row
                            model_state.update({"hpo_history": hpo_rows, "best_hpo": best_row, "status": "hpo"})
                            update_state(state_path, state, active_fold=args.hpo_folds)
                            print(json.dumps({"model": model_name, "candidate": index, "hpo_ap": row["average_precision"], "hpo_auroc": row["auroc"]}), flush=True)

                    best_estimator = safe_clone(base_estimator).set_params(**best_row["params"])
                    update_state(
                        state_path,
                        state,
                        phase="PHASE 3 - 10-fold OOF operating points",
                        active_candidate=1,
                        active_total=1,
                        active_fold=0,
                        active_total_folds=args.oof_folds,
                    )
                    oof_prob = oof_predictions(
                        best_estimator, X_train, y_train, args.oof_folds, args.n_jobs,
                        progress_path, "oof", model_name,
                    )
                    update_state(
                        state_path,
                        state,
                        phase="PHASE 3 - OOF threshold sweep (max specificity at recall >= 0.80)",
                        active_fold=args.oof_folds,
                    )
                    atomic_write(progress_path, {})
                    oof_points = all_operating_points(y_train, oof_prob)
                    primary_point = oof_points["primary_recall_0.80"]
                    oof_metrics = {**primary_point, **probability_metrics(y_train, oof_prob)}
                    update_state(
                        state_path,
                        state,
                        phase="FINAL - refit on 80% train",
                        active_fold=args.oof_folds,
                    )
                    hpo_times = [row.get("elapsed_seconds", 0.0) for row in hpo_rows if row.get("elapsed_seconds")]
                    refit_estimate = max(hpo_times) / max(1, args.hpo_folds) if hpo_times else 0.0
                    refit_stop = start_refit_heartbeat(state_path, state, model_name, best_row, refit_estimate)
                    try:
                        final_estimator = safe_clone(base_estimator).set_params(**best_row["params"]).fit(X_train, y_train)
                    finally:
                        refit_stop.set()
                    state["refit_progress"] = {
                        "model": model_name,
                        "status": "done",
                        "elapsed_seconds": state.get("refit_progress", {}).get("elapsed_seconds"),
                        "eta_seconds": 0.0,
                        "best_hpo_candidate": best_row.get("candidate"),
                        "best_hpo_average_precision": best_row.get("average_precision"),
                        "best_hpo_log_loss": best_row.get("log_loss"),
                        "params": best_row.get("params", {}),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    update_state(
                        state_path,
                        state,
                        phase="FINAL - held-out 20% evaluation",
                        active_fold=args.oof_folds,
                    )
                    test_prob = final_estimator.predict_proba(X_test)[:, 1]
                    test_metrics = metrics_at_operating_point(y_test, test_prob, primary_point)
                    test_operating_points = {
                        name: metrics_at_operating_point(y_test, test_prob, point)
                        for name, point in oof_points.items()
                    }
                    final = {
                        "best_hpo": best_row,
                        "selected_threshold": primary_point["threshold"],
                        "threshold_source": "train_10fold_oof_specificity_at_recall_0.80",
                        "primary_operating_point": "recall_0.80",
                        "oof_metrics": oof_metrics,
                        "oof_operating_points": oof_points,
                        "test_metrics": test_metrics,
                        "test_operating_points": test_operating_points,
                        "calibration": {
                            "oof_bins": calibration_bins(y_train, oof_prob),
                            "test_bins": calibration_bins(y_test, test_prob),
                        },
                        "elapsed_seconds": time.time() - model_started,
                    }
                    model_state.update({"status": "completed", "oof": oof_metrics, "final": final})
                    state["completed_models"][model_name] = final
                    joblib.dump(final_estimator, RUN_DIR / f"{model_name.lower().replace(' ', '_')}.joblib")
                    update_state(state_path, state, active_fold=args.oof_folds)
                    print(json.dumps({"model": model_name, "oof_specificity_at_recall_0.80": oof_metrics["specificity"], "oof_recall": oof_metrics["recall"], "test_specificity": test_metrics["specificity"], "test_recall": test_metrics["recall"], "secondary_oof_f2": oof_points["f2"]["f2"]}), flush=True)
                except Exception:
                    tb = traceback.format_exc()
                    print(f"MODEL FAILED {model_name}:\n{tb}", flush=True)
                    state["failed_models"][model_name] = {"phase": state.get("phase"), "error": tb.strip().splitlines()[-1]}
                    update_state(state_path, state)

            ranked = sorted(
                state["completed_models"].items(),
                key=lambda item: (item[1]["oof_metrics"]["feasible"], item[1]["oof_metrics"]["specificity"], item[1]["oof_metrics"]["average_precision"], -item[1]["oof_metrics"]["fp"]),
                reverse=True,
            )
            payload = {
                "status": "ok",
                "objective": state["objective"],
                "strategy": state["strategy"],
                "finished": datetime.now(timezone.utc).isoformat(),
                "models": dict(ranked),
                "winner": ranked[0][0] if ranked else None,
            }
            atomic_write(result_path, payload)
            update_state(state_path, state, status="ok", active_model=None, phase="finished", winner=payload["winner"])
            print(json.dumps(payload, indent=2, default=str), flush=True)
            return 0
        except Exception:
            print(traceback.format_exc(), flush=True)
            if state_path.exists():
                state = json.loads(state_path.read_text())
                update_state(state_path, state, status="interrupted_or_error")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
