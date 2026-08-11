#!/usr/bin/env python3
"""Reproducible stroke-general benchmark following the ARAI 03 protocol.

The workflow is deliberately sklearn-based: one 80/20 stratified split, six
class-imbalance-aware classifiers, randomized CV tuning, OOF threshold
selection, and one final held-out test evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
# Keep the earlier ROC-AUC run intact; this run is explicitly F1-optimised.
OUT_DIR = ROOT / "experiments" / "stroke_03_ref_runs" / "general_f1"
DATA_ROOT = Path("/media/datos/datos_compartidos/trIAje/data")
sys.path.insert(0, str(ROOT / "analysis"))


def configure_environment() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-triaje")
    os.environ.setdefault("PYTHONPATH", str(ROOT / "analysis"))
    os.environ.setdefault("SOURCE_TABLES_PATH", str(DATA_ROOT / "df_modelling"))
    os.environ.setdefault("EDATOSCASO_LOAD_PATH", "../tabla_madre/master_table.parquet")
    os.environ.setdefault("MASTER_TABLE_PATH", "../tabla_madre/master_table.parquet")
    os.environ.setdefault("MASTER_TABLE_4DEMANDS_PATH", "../tabla_madre/master_table_4demands.parquet")
    os.environ.setdefault("TABLE_EMBEDDING_SBERT_PATH", "embeddings/sbert.parquet")
    os.environ.setdefault("TABLE_EMBEDDING_RIGOBERT_PATH", "embeddings/rigobert.parquet")
    os.environ.setdefault("TRIAJE_TABLE_CLEANED_PATH", "df_stroke_preprocessed_2026-07-23.parquet")
    os.environ.setdefault("STROKE_TABLE_CLEANED_PATH", "df_stroke_preprocessed_2026-07-23.parquet")
    os.environ.setdefault("CARDIACARREST_TABLE_MODELING", "df_cardiacarrest_modeling.parquet")
    os.environ.setdefault("CHESTPAIN_TABLE_MODELING", "df_chestpain_modeling.parquet")
    os.environ.setdefault("DYSPNEA_TABLE_MODELING", "df_dyspnea_modeling.parquet")
    os.environ.setdefault("RUN_EMBEDDING_SBERT", "0")
    os.environ.setdefault("RUN_EMBEDDING_RIGOBERT", "0")
    os.environ.setdefault("TEST_PIPELINE", "all")
    os.environ["PYTHONPATH"] = str(ROOT / "analysis") + os.pathsep + os.environ.get("PYTHONPATH", "")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class Progress:
    def __init__(self, total: int, log_path: Path | None = None):
        self.total = total
        self.completed = 0
        self.current = "preparing"
        self.started = time.time()
        self.current_started = self.started
        self.durations: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.log_path = log_path

    def start(self):
        self.thread.start()

    def begin(self, name: str):
        self.current = name
        self.current_started = time.time()
        self.render()

    def done(self, duration: float):
        self.durations.append(duration)
        self.completed += 1
        self.current = "waiting"
        self.render()

    def render(self):
        elapsed = time.time() - self.started
        eta = 0
        fit_rate = 0
        fits = 0
        if self.log_path and self.log_path.exists():
            log = self.log_path.read_text(encoding="utf-8", errors="replace")
            markers = list(re.finditer(r"\n\[[^\]]+\] RandomizedSearchCV n_iter=\d+", log))
            if markers:
                segment = log[markers[-1].start():]
                fits = min(200, len(re.findall(r"\[CV\] END", segment)))
                active_elapsed = max(1, elapsed - sum(self.durations))
                fit_rate = fits / active_elapsed if fits else 0
        if fits and fit_rate:
            current_remaining = (200 - fits) / fit_rate
            future_models = max(0, self.total - self.completed - 1)
            future_per_model = 200 / fit_rate
            eta = current_remaining + future_models * future_per_model
            eta_text = f"ETA {format_seconds(eta)} ({fit_rate:.2f} fits/s)"
        elif self.completed:
            eta = (sum(self.durations) / len(self.durations)) * (self.total - self.completed)
            eta_text = f"ETA {format_seconds(eta)}"
        else:
            eta_text = "ETA calculando"
        width = 28
        filled = int(width * self.completed / self.total)
        bar = "#" * filled + "." * (width - filled)
        print(
            f"\n[PROGRESS] [{bar}] {self.completed}/{self.total} models | "
            f"current={self.current} | elapsed {format_seconds(elapsed)} | {eta_text} | CV {fits}/200",
            flush=True,
        )

    def _heartbeat(self):
        while not self.stop_event.wait(15):
            self.render()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.render()


def format_seconds(value: float) -> str:
    value = max(0, int(value))
    return f"{value // 3600:02d}:{(value % 3600) // 60:02d}:{value % 60:02d}"


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def oof_f1_threshold(y_true, probabilities) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true), np.asarray(probabilities))
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.where(precision[:-1] + recall[:-1] > 0, precision[:-1] + recall[:-1], 1)
    return float(thresholds[int(np.argmax(f1))])


def metrics_at_threshold(y_true, probabilities, threshold) -> dict:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss, log_loss,
        confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    )

    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def build_models(pos_weight: float):
    from scipy.stats import loguniform, randint, uniform
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from pytabkit import FTT_D_Classifier, RealMLP_TD_Classifier, RealTabR_D_Classifier, TabM_D_Classifier

    def pipe(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    specs = {
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            {"model__C": loguniform(1e-4, 10.0)},
        ),
        "Elastic Net Logistic": (
            pipe(LogisticRegression(max_iter=600, tol=1e-3, class_weight="balanced", solver="saga", penalty="elasticnet", random_state=42)),
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
        "CatBoost": (
            pipe(CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, loss_function="Logloss", eval_metric="Logloss", class_weights=(1.0, pos_weight), random_seed=42, verbose=False, thread_count=1, allow_writing_files=False)),
            {"model__iterations": randint(150, 800), "model__learning_rate": loguniform(0.01, 0.25), "model__depth": randint(4, 11), "model__l2_leaf_reg": loguniform(1e-2, 20.0), "model__random_strength": uniform(0.0, 2.0), "model__border_count": [32, 64, 128, 254]},
        ),
        "RealMLP": (
            RealMLP_TD_Classifier(device="cpu", random_state=42, n_cv=1, n_refit=0, n_repeats=1, val_fraction=0.2, n_threads=1, verbosity=0, use_early_stopping=True),
            {"n_epochs": [64, 128, 256], "batch_size": [256, 512], "hidden_width": [128, 256, 512], "n_hidden_layers": [2, 3, 4], "p_drop": [0.0, 0.1, 0.2], "lr": loguniform(1e-4, 3e-3)},
        ),
        "TabM": (
            TabM_D_Classifier(device="cpu", random_state=42, n_cv=1, n_refit=0, n_repeats=1, val_fraction=0.2, n_threads=1, verbosity=0, patience=16),
            {"tabm_k": [16, 32, 64], "n_epochs": [64, 128, 256], "batch_size": [256, 512], "d_block": [128, 256], "n_blocks": [2, 3, 4], "dropout": [0.0, 0.1, 0.2], "lr": loguniform(1e-4, 3e-3)},
        ),
        "RealTabR": (
            RealTabR_D_Classifier(device="cpu", random_state=42, n_cv=1, n_refit=0, n_repeats=1, val_fraction=0.2, n_threads=1, verbosity=0),
            {"n_epochs": [64, 128, 256], "batch_size": [128, 256], "d_main": [128, 256], "encoder_n_blocks": [1, 2, 3], "predictor_n_blocks": [1, 2, 3], "dropout0": [0.0, 0.1, 0.2], "patience": [16, 32]},
        ),
        "FT-Transformer": (
            FTT_D_Classifier(device="cpu", random_state=42, n_cv=1, n_refit=0, n_repeats=1, val_fraction=0.2, n_threads=1, verbosity=0),
            {"max_epochs": [64, 128, 256], "batch_size": [128, 256], "module_d_token": [64, 96, 128], "module_n_layers": [2, 3, 4], "module_n_heads": [4, 8], "lr": loguniform(1e-4, 3e-3), "module_attention_dropout": [0.0, 0.1, 0.2], "module_ffn_dropout": [0.0, 0.1, 0.2]},
        ),
    }
    return specs


def prepare_data():
    from kbase import modeling as ml

    df = ml.data_load_col_selection(
        "p1_real_emerg_bps", None, True, True, True, True, False,
    )
    ml.col_validation(df, "p1_real_emerg_bps")
    filtered = ml.data_filtering(
        df, "p1_real_emerg_bps", 54, None, None, None, False, ml.protocol_questions,
    )

    # data_load_col_selection() only picks up triage-question columns whose
    # names start with "q", but this table's stroke (demand_type_1==54)
    # triage columns are stored bare (e.g. "1a", not "q1_a"), so they are
    # silently dropped upstream. Re-attach them here by name from the raw
    # source table, aligned on the preserved row index.
    bare_triage_cols = sorted({
        col.replace("q", "", 1).replace("_", "")
        for col in ml.protocol_questions.get(54, [])
    })
    source_path = os.path.join(ml.settings.source_tables_path, ml.settings.triaje_table_cleaned_path)
    available = [c for c in bare_triage_cols if c in pq.read_schema(source_path).names]
    if available:
        triage_df = pq.read_table(source_path, columns=available).to_pandas()
        triage_df = triage_df.loc[filtered.index]
        filtered = pd.concat([filtered, triage_df], axis=1)

    y = filtered["p1_real_emerg_bps"].astype(int)
    year = filtered["year"].copy()
    X = filtered.drop(columns=["p1_real_emerg_bps", "year"])
    X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object", "string"]).columns, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    return X, y, year


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure_environment()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "training.log"
    state_path = OUT_DIR / "state.json"
    split_path = OUT_DIR / "split.npz"
    final_path = OUT_DIR / "results.json"

    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(log, sys.__stdout__)), contextlib.redirect_stderr(Tee(log, sys.__stderr__)):
        started = datetime.now(timezone.utc).isoformat()
        print(f"\nstarted={started} n_iter={args.n_iter} resume={args.resume}", flush=True)
        progress = Progress(6, OUT_DIR / "training.log")
        progress.start()
        try:
            X, y, year = prepare_data()
            from sklearn.model_selection import StratifiedKFold, train_test_split

            if split_path.exists() and args.resume:
                split = np.load(split_path, allow_pickle=False)
                train_idx, test_idx = split["train_idx"], split["test_idx"]
                print("Loaded existing fixed train/test split.", flush=True)
            else:
                train_idx, test_idx = train_test_split(
                    np.arange(len(y)), test_size=0.20, random_state=42, stratify=y,
                )
                np.savez(split_path, train_idx=train_idx, test_idx=test_idx)
                print("Created fixed 80/20 stratified train/test split.", flush=True)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            print(f"Rows={len(y):,} train={len(y_train):,} test={len(y_test):,} features={X.shape[1]}", flush=True)
            pos_weight = float((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
            print(f"Outcome train: {y_train.value_counts().to_dict()} scale_pos_weight={pos_weight:.4f}", flush=True)

            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() and args.resume else {
                "status": "running", "started": started, "protocol": "03ref", "completed_models": {},
            }
            specs = build_models(pos_weight)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for name, (estimator, params) in specs.items():
                if name in state.get("completed_models", {}):
                    progress.done(float(state["completed_models"][name].get("elapsed_seconds", 0)))
                    print(f"Skipping completed model: {name}", flush=True)
                    continue
                progress.begin(name)
                model_started = time.time()
                print(f"\n[{name}] RandomizedSearchCV n_iter={args.n_iter} refit=f1 diagnostics=roc_auc,log_loss cv=10", flush=True)
                from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
                search = RandomizedSearchCV(
                    estimator, params, n_iter=args.n_iter,
                    scoring={"f1": "f1", "roc_auc": "roc_auc", "log_loss": "neg_log_loss"},
                    refit="f1", cv=cv, random_state=42, n_jobs=1, verbose=2,
                )
                search.fit(X_train, y_train)
                probabilities_oof = cross_val_predict(
                    search.best_estimator_, X_train, y_train, cv=cv,
                    method="predict_proba", n_jobs=1,
                )[:, 1]
                threshold = oof_f1_threshold(y_train, probabilities_oof)
                test_probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
                result = {
                    "best_cv_f1": float(search.best_score_),
                    "best_cv_auroc": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
                    "best_params": search.best_params_,
                    "oof_threshold": threshold,
                    "oof_metrics": metrics_at_threshold(y_train, probabilities_oof, threshold),
                    "test_metrics": metrics_at_threshold(y_test, test_probabilities, threshold),
                    "elapsed_seconds": time.time() - model_started,
                    "search_history": [
                        {
                            "iteration": int(i + 1),
                            "cv_f1": float(f1),
                            "cv_auroc": float(roc_auc),
                            "cv_log_loss": float(-loss),
                        }
                        for i, (f1, roc_auc, loss) in enumerate(
                            zip(search.cv_results_["mean_test_f1"], search.cv_results_["mean_test_roc_auc"], search.cv_results_["mean_test_log_loss"])
                        )
                    ],
                }
                joblib.dump(search.best_estimator_, OUT_DIR / f"{name.lower().replace(' ', '_')}.joblib")
                np.savez_compressed(OUT_DIR / f"{name.lower().replace(' ', '_')}_predictions.npz", oof=probabilities_oof, test=test_probabilities)
                state.setdefault("completed_models", {})[name] = result
                atomic_json(state_path, state)
                print(json.dumps(result, indent=2, default=str), flush=True)
                progress.done(result["elapsed_seconds"])

            ranking = sorted(
                ((name, result) for name, result in state["completed_models"].items()),
                key=lambda item: item[1]["test_metrics"]["auroc"], reverse=True,
            )
            payload = {"status": "ok", "finished": datetime.now(timezone.utc).isoformat(), "protocol": "03ref", "population": "general", "models": dict(ranking), "winner_by_test_auroc": ranking[0][0] if ranking else None}
            atomic_json(final_path, payload)
            state["status"] = "ok"
            atomic_json(state_path, state)
            print("\nFINAL RESULTS", flush=True)
            print(json.dumps(payload, indent=2, default=str), flush=True)
            return 0
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["status"] = "interrupted_or_error"
                state["error"] = repr(exc)
                atomic_json(state_path, state)
            return 1
        finally:
            progress.close()


if __name__ == "__main__":
    raise SystemExit(main())
