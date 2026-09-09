#!/usr/bin/env python3
"""Small ARAI-03-like benchmark on the trIAje stroke cohort.

It intentionally mirrors the reference workflow: fixed 80/20 split, explicit
model benchmark, stratified OOF probabilities for threshold selection, and a
single held-out test evaluation.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, loguniform


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/datos/datos_compartidos/trIAje/data/df_modelling/df_stroke_preprocessed_2026-07-23.parquet")
OUT = ROOT / "experiments" / "arai03_like_stroke_metrics.json"


def f1_threshold(y, p):
    thresholds = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 200)))
    scores = [f1_score(y, p >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main():
    df = pd.read_parquet(DATA)
    df = df.loc[(df["demand_type_1"] == 54) & (df["age"] >= 18)].copy()
    df = df.dropna(subset=["p1_real_emerg"])
    y = df.pop("p1_real_emerg").astype(int)
    drop = [c for c in df.columns if c in {"demand_type_1", "year", "p1_assigned"} or c.endswith("_date")]
    X = df.drop(columns=drop, errors="ignore")
    X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
            {"logisticregression__C": loguniform(1e-3, 10.0)},
        ),
        "Random Forest": (
            RandomForestClassifier(n_jobs=4, class_weight="balanced", random_state=42),
            {"n_estimators": randint(100, 300), "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", "log2", 0.5]},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_jobs=4, class_weight="balanced", random_state=42),
            {"n_estimators": randint(100, 300), "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", "log2", 0.5]},
        ),
    }
    fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    for name, (base, param_distributions) in models.items():
        tuning = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=5,
            scoring="f1",
            cv=fold,
            random_state=42,
            n_jobs=1,
            refit=True,
        )
        tuning.fit(X_train, y_train)
        base = tuning.best_estimator_
        oof = np.zeros(len(X_train))
        for tr, va in fold.split(X_train, y_train):
            m = clone(base)
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            oof[va] = m.predict_proba(X_train.iloc[va])[:, 1]
        threshold = f1_threshold(y_train.to_numpy(), oof)
        model = clone(base).fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = test_prob >= threshold
        rows.append({
            "model": name,
            "tuning_scoring": "f1",
            "best_cv_f1": tuning.best_score_,
            "n": len(df), "features": X.shape[1],
            "threshold_oof_f1": threshold,
            "test_auroc": roc_auc_score(y_test, test_prob),
            "test_average_precision": average_precision_score(y_test, test_prob),
            "test_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_f1": f1_score(y_test, test_pred, zero_division=0),
        })
    payload = {"strategy": "arai03_like", "target": "p1_real_emerg", "demand_code": 54, "rows": rows}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
