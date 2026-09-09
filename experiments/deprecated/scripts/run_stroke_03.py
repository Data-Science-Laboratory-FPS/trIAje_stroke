#!/usr/bin/env python3
"""Run the stroke 03 modelling experiment and persist metrics.

This is the current trIAje 03_04 strategy with the target-column typo fixed:
the pipeline produces ``p1_real_emerg``, not ``p1_real``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "stroke_03_runs"
DATA_ROOT = Path("/media/datos/datos_compartidos/trIAje/data")
sys.path.insert(0, str(ROOT / "analysis"))


def configure_environment() -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["general", "with_triage", "without_triage"], required=True)
    parser.add_argument("--time-budget", type=int, default=600)
    parser.add_argument("--fixed-config", action="store_true")
    args = parser.parse_args()

    configure_environment()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{args.name}.log"
    result_path = OUT_DIR / f"{args.name}.json"

    import contextlib
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        started = datetime.now(timezone.utc).isoformat()
        print(f"started={started}")
        print(f"name={args.name} time_budget={args.time_budget}")
        try:
            from kbase import modeling as ml

            triage_value = {
                "general": None,
                "with_triage": 1,
                "without_triage": 0,
            }[args.name]
            fixed_config = None
            if args.fixed_config:
                from kbase.model_hyperparameters import FIXED_XGBOOST_CONFIG
                fixed_config = FIXED_XGBOOST_CONFIG
            result = ml.run_binary_automl_model(
                cohort_name="Stroke",
                target_column="p1_real_emerg",
                demand_code=54,
                triage_value=triage_value,
                min_age=18,
                time_budget=-1 if fixed_config is not None else args.time_budget,
                fixed_config=fixed_config,
                fixed_estimator="xgboost",
                optimize_metric="ap",
                optimize_beta=1,
                train_threshold="oof",
                feature_importance=0,
                plot_shap=False,
                n_interpretability_features=0,
                plot_calibration_curve=False,
                plot_sa_roc_curve=False,
                export_cv_fold_metrics=True,
            )
            summary = result["summary"].iloc[0].to_dict()
            metrics = result["metrics"]
            payload = {
                "status": "ok",
                "started": started,
                "finished": datetime.now(timezone.utc).isoformat(),
                "name": args.name,
                "target": "p1_real_emerg",
                "demand_code": 54,
                "triage_value": triage_value,
                "threshold": result["threshold"],
                "metrics": metrics,
                "summary": summary,
            }
            result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            print(json.dumps(payload, indent=2, default=str))
            return 0
        except Exception as exc:
            payload = {
                "status": "error",
                "started": started,
                "finished": datetime.now(timezone.utc).isoformat(),
                "name": args.name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            traceback.print_exc(file=sys.stdout)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
