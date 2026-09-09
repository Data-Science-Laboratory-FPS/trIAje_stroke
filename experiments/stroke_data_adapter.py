#!/usr/bin/env python3
"""Project-specific data adapter for the stroke benchmark pipeline.

For another project, keep the pipeline script unchanged as far as possible and
replace this module with the target project's data loading logic. The pipeline
expects ``prepare_data()`` to return ``X, y, year`` where X is numeric,
row-aligned with y, and free of target/leakage columns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/datos/datos_compartidos/trIAje/data")
sys.path.insert(0, str(ROOT / "analysis"))


def configure_environment() -> None:
    """Set the environment variables expected by the trIAje analysis package."""
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


def prepare_data():
    """Load and encode the corrected stroke cohort used by the active run."""
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
