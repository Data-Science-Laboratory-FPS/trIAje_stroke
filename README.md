# trIAje

trIAje is a clinical research project that develops AI-assisted triage support models for emergency medical dispatch calls. Using historical dispatch records from an Emergency Medical Service (EMS), the project builds and evaluates machine-learning classifiers that predict whether a call should be assigned the highest dispatch priority (P1) for four high-impact emergency presentations ("chief complaints").

## The four clinical pathways

| Pathway | Demand code(s) | Cohort name | Classification task | AutoML entry point | Notebooks |
|---|---|---|---|---|---|
| Cardiac arrest / Unconsciousness | 36, 58 | `"Unconscious/Cardiac arrest"` | Multiclass (no-P1 / P1-36 / P1-58) | `run_multiclass_automl_model` | `03_01`, `04_01` |
| Non-traumatic chest pain | 23 | `"Non-traumatic Chest Pain"` | Binary (P1 vs. non-P1) | `run_binary_automl_model` | `03_02`, `04_02` |
| Dyspnea / Respiratory distress | 16 | `"Dyspnea"` | Binary (P1 vs. non-P1) | `run_binary_automl_model` | `03_03`, `04_03` |
| Stroke / Cerebrovascular accident | 54 | `"Stroke"` | Binary (P1 vs. non-P1) | `run_binary_automl_model` | `01_04`, `02_04`, `03_04`, `04_04` |

All four pathways are derived from the **same master dataset** by filtering on `demand_type_1`, and are processed through structurally parallel "modeling" (`03_*`) and "post-hoc EDA" (`04_*`) notebooks. The ground-truth label is `p1_real_emerg` in all cases (for cardiac arrest, it is further split by `demand_type_1` into the 36/58 multiclass labels). The stroke pathway additionally has two extra notebooks (`01_04`, `02_04`) for pathway-specific QC and descriptive (Table 1) reporting.

## Repository structure

```
trIAje/
├── CLAUDE.md                  # Guidance for AI coding assistants working in this repo
├── README.md                  # This file
├── trIAje.Rproj, .RData, .Rhistory, .Rproj.user/   # Local R session (gitignored, not part of the tracked pipeline)
├── export_tables/              # Gitignored CSV exports for downstream/statistical reporting
│   ├── edatoscaso.csv
│   ├── triaje_table.csv
│   └── icd_unique/              # Per-pathway unique ICD code lists (produced by stats/stats.ipynb)
└── analysis/
    ├── .env                     # Local configuration (gitignored); see "Setup" below
    ├── kbase/                   # Shared Python package imported by all notebooks
    │   ├── config.py            # Settings (pydantic-settings), reads analysis/.env
    │   ├── preprocessing.py      # Column renaming, dtype handling, embeddings, ATC/ICD/missing-value utilities
    │   ├── eda.py                # Clinical-performance evaluation (confusion-matrix metrics, plots)
    │   └── modeling.py           # FLAML AutoML pipelines (binary + multiclass), SHAP, calibration, figure export
    ├── demands/                 # Reserved for future complaint-specific sub-modules (currently empty)
    ├── figures/                 # Output figures, written by kbase.modeling.save_figure()
    │   └── 04_stroke/
    │       └── tables/          # Table 1 outputs (.docx / .jpg) from 02_04_tables_stroke.ipynb
    ├── 00_01_grouping_duplicates.ipynb   # Stage 00: deduplicate raw EMS records
    ├── 00_02_EDA-QC.ipynb                # Stage 00: data-quality checks
    ├── 00_03_EDA_general.ipynb           # Stage 00: filter to the 4 target pathways
    ├── 01_dataset.ipynb                  # Stage 01: build the master preprocessed table
    ├── 01_04_dataset_stroke.ipynb        # Stroke pathway: feature QC / correlation analysis
    ├── 02_04_tables_stroke.ipynb         # Stroke pathway: descriptive Table 1 (docx/jpg)
    ├── 03_01_modelling_cardiacarrest.ipynb
    ├── 03_02_modelling_chestpain.ipynb
    ├── 03_03_modelling_dyspnea.ipynb
    ├── 03_04_modelling_stroke.ipynb
    ├── 04_01_eda_cardiacarrest.ipynb
    ├── 04_02_eda_chestpain.ipynb
    ├── 04_03_eda_dyspnea.ipynb
    ├── 04_04_eda_stroke.ipynb
    └── stats/
        └── stats.ipynb           # ICD export + ad-hoc statistical QC
```

## Analysis workflow

The pipeline is a numbered sequence of notebooks. Stages `00` and `01` build a single shared master dataset; stages `03` and `04` then run **in parallel, once per pathway**, against that shared dataset.

```
00_01_grouping_duplicates ──► tabla_madre/master_table.parquet
        │
00_02_EDA-QC  (data-quality checks, no persisted output)
        │
00_03_EDA_general ──► tabla_madre/master_table_4demands.parquet
        │              (filtered to demand codes 16, 23, 36, 54, 58)
        │
01_dataset ──► df_modelling/df_triaje_preprocessed_<date>.parquet
        │      (master cleaned/feature-engineered table, all 4 pathways)
        │
        ├──► 01_04_dataset_stroke   (stroke-only QC / correlation analysis)
        ├──► 02_04_tables_stroke    (stroke-only descriptive Table 1)
        │
        ├──► 03_01_modelling_cardiacarrest ──► 04_01_eda_cardiacarrest
        ├──► 03_02_modelling_chestpain     ──► 04_02_eda_chestpain
        ├──► 03_03_modelling_dyspnea       ──► 04_03_eda_dyspnea
        └──► 03_04_modelling_stroke        ──► 04_04_eda_stroke

stats/stats.ipynb ──► export_tables/icd_unique/*.csv  (per-pathway unique ICD codes)
```

### Stage 00 — Data ingestion, deduplication, and QC

- **`00_01_grouping_duplicates.ipynb`** loads the raw EDATOSCASO export (`EDATOSCASO_source/...parquet`), groups and deduplicates records by patient, and writes `tabla_madre/master_table.parquet`.
- **`00_02_EDA-QC.ipynb`** performs exploratory data-quality checks on the master table; it does not persist any output.
- **`00_03_EDA_general.ipynb`** filters the master table to the four target demand codes (16, 23, 36, 54, 58) and writes `tabla_madre/master_table_4demands.parquet`.

### Stage 01 — Master dataset construction (`01_dataset.ipynb`)

This is the central feature-engineering notebook. It loads `master_table_4demands.parquet` and produces `df_modelling/df_triaje_preprocessed_<date>.parquet`, the cleaned table consumed by every downstream notebook (`01_04`, `02_04`, all `03_*`, all `04_*`, and `stats/stats.ipynb`).

The notebook applies the following transformations, **shared across all four pathways**:

- Column renaming, reordering, and dtype normalization (`kbase.preprocessing.rename_columns`, `reorder_dataframe`, `transform_column_dtypes`)
- Temporal features (day of week, time of day, month, season)
- Geographic features (province dummies, incident coordinates)
- Triage questions: loads `preguntas_triaje/preguntas_respuesta_detalle_<date>.parquet` and maps Q1–Q7 to one-hot columns `q1_a` … `q7_b` (`-1` = not asked / no triage)
- Medication: loads `medicacion_asistencia_antecedentes/medicacion_<date>.parquet`, maps `nombre_grupo_atc` to one-hot `atc_group_*` columns (237 columns), and creates the `has_med` flag
- Past history and active problems (comorbidities): loads `medicacion_asistencia_antecedentes/problemas_one_hot_encoding_<date>.parquet` and `medicacion_asistencia_antecedentes/antecedentes_stroke_one_hot_encoding_<date>.parquet`, and unifies them into a single `com_*` comorbidity taxonomy (37 columns, e.g. `com_1_1_hypertension`, `com_5_1_cerebrovascular`, `com_8_1_fall_risk`), plus the `has_history` and `has_problems` flags

#### Stroke-specific preprocessing currently embedded in `01_dataset.ipynb`

In addition to the shared steps above, `01_dataset.ipynb` currently contains **three blocks of logic that apply only to stroke (`demand_type_1 == 54`)**, interleaved with the pathway-agnostic pipeline:

1. **Literal-reason ("motivo literal") one-hot encoding.** Loads a stroke-specific table (`motivo_literal_ictus/df_stroke_motivo_literal_onehot_<date>.parquet`) containing 26 clinically curated features extracted from the free-text dispatch reason (e.g. `lr_suspected_stroke`, `lr_focal_facial_weakness`, `lr_focal_arm_weakness`, `lr_onset_hyperacute`, `lr_history_prior_stroke`). Columns are merged and renamed to the `lr_*` prefix via `lr_mergeand_audit()` and `translate_lr_columns()`.
2. **Past-history (antecedentes) one-hot encoding for stroke.** The `antecedentes_stroke_one_hot_encoding_<date>.parquet` source feeds into the unified `com_*` comorbidity columns described above; this source table is currently stroke-specific even though the resulting `com_*` columns are present in the schema for all pathways.
3. **`p1_real_emerg` / `p1_real_bps` ground-truth correction rules.** A block of special-case logic, keyed on `demand_type_1 == 54`, that adjusts the ground-truth priority labels for stroke based on ICD codes.

As a consequence, the `lr_*` (literal-reason) feature block is currently populated only for stroke calls. For the other three pathways these 26 columns exist in the schema (so the table has a uniform shape) but are not informative.

#### Planned refactor: complaint-specific preprocessing functions

The current implementation embeds the stroke-specific logic described above directly in `01_dataset.ipynb`, interleaved with the pathway-agnostic steps. **The intended future architecture is to extract pathway-specific logic — literal-reason feature extraction, ground-truth correction rules, and similar steps — into dedicated, reusable preprocessing functions, one per chief complaint**, likely organized under `kbase/` or the currently empty `analysis/demands/` package. Under that design:

- `01_dataset.ipynb` would perform only the pathway-agnostic feature engineering, and
- each pathway's specific rules (e.g. stroke literal-reason features, stroke ground-truth correction) would be applied through an explicit, named function call rather than being interleaved inline.

**This refactor has not been implemented yet.** The description above reflects the current state of the code, not the target architecture.

### Stroke pathway extras (`01_04`, `02_04`)

- **`01_04_dataset_stroke.ipynb`** loads `df_triaje_preprocessed_<date>.parquet` filtered to `demand_type_1 == 54`, plus the reduced modeling table (`stroke_table_cleaned_path`, produced by `03_04_modelling_stroke.ipynb` with `export_table=True`). It runs ATC medication completeness analysis (`kbase.preprocessing.analyze_atc_columns`) and correlation/Jaccard-similarity analyses for general features, past history, literal reason, and medication groups. It is a pure QC notebook and does not export data.
- **`02_04_tables_stroke.ipynb`** ("Table 1 – Stroke") generates descriptive-statistics tables (`.docx` + `.jpg`) comparing patient and call characteristics across periods (COVID / post-COVID) and outcome definitions (`p1_real_emerg` / `p1_assigned`), using both `df_triaje_preprocessed` and the stroke modeling table. Outputs are written to `analysis/figures/04_stroke/tables/`.

### Stage 03 — AutoML modeling per pathway

Each `03_*` notebook imports `kbase.modeling` and calls either `run_binary_automl_model()` (chest pain, dyspnea, stroke) or `run_multiclass_automl_model()` (cardiac arrest / unconsciousness), filtering the master table by `demand_code` and training a FLAML AutoML model on `target_column = "p1_real_emerg"`.

Common features of these pipelines:

- Configurable feature groups via `include_lr`, `include_history`, `include_com`, `include_medication`, `include_embeddings` (binary pipeline only)
- Optional triage-only / no-triage subsets (`triage_value = 1` / `0`)
- Optional year filtering, age filtering, and outcome-stratified cross-validation
- Threshold optimization (F-beta, Youden's J, or clinical overtriage/undertriage targets)
- SHAP and permutation-importance plots, calibration curves, and confusion matrices, saved via `kbase.modeling.save_figure()` to `analysis/figures/0{1-4}_<pathway>/`
- `export_table=True` writes a reduced, modeling-ready parquet table for the pathway (see "Inputs and outputs" below)

Each `03_*` notebook is structured as an **iterative experiment log** — multiple cells exploring different `time_budget`, `optimize_beta`, `years`, and feature-set configurations — rather than a single linear pipeline. The cells with `export_table=True` are the ones that persist a table for downstream use.

### Stage 04 — Post-hoc EDA per pathway

Each `04_*` notebook calls `kbase.eda.analyze_emergency_performance(...)`, which reloads `df_triaje_preprocessed_<date>.parquet`, filters to the pathway's demand codes, and computes confusion-matrix-based clinical performance metrics (accuracy, recall, precision, specificity, F1, overtriage, undertriage) comparing the ground-truth label (`p1_real_emerg`) against either the system's original triage assignment (`p1_assigned`, used in `04_04_eda_stroke`) or a model prediction column (`p1_predicted`, used in `04_01`–`04_03`), optionally stratified by triage availability and by sex.

### `stats/` — ICD export and statistical reporting

- **`stats/stats.ipynb`** exports, for each pathway, the set of unique ICD codes (`JUICIOCLINICO1/2/3`) found in `df_triaje` to `export_tables/icd_unique/<pathway>.csv`. The remainder of the notebook contains ad-hoc, interactive QC snippets (duplicate-record analysis, resolution-code breakdowns) that are not part of the automated pipeline (see "Known issues" below).
- **`export_tables/`** also contains `edatoscaso.csv` and `triaje_table.csv`, which are consumed by a local R project (`trIAje.Rproj`, `.RData`, `.Rproj.user/`) for additional statistical reporting. These R artifacts are gitignored and are not part of the version-controlled pipeline.

## `kbase/` shared library

| Module | Role |
|---|---|
| `config.py` | `Settings` (pydantic-settings) singleton `settings`, reading paths and pipeline flags from `analysis/.env` |
| `preprocessing.py` | Column renaming/reordering/dtype dictionaries; utilities for ATC analysis, ICD filtering, text-embedding merges, missing-value analysis |
| `eda.py` | `evaluate_diagnostic_performance` / `analyze_emergency_performance` — confusion-matrix-based clinical metrics and plots |
| `modeling.py` | `run_binary_automl_model`, `run_multiclass_automl_model`, FLAML AutoML training, threshold optimization (F-beta, Youden), SHAP / permutation feature importance, calibration plots, `protocol_questions` (per-pathway triage question sets), `save_figure()` |

## Inputs and outputs

| Notebook | Reads | Writes |
|---|---|---|
| `00_01_grouping_duplicates` | `EDATOSCASO_source/*.parquet` | `tabla_madre/master_table.parquet` |
| `00_02_EDA-QC` | `master_table.parquet` | — |
| `00_03_EDA_general` | `master_table.parquet` | `tabla_madre/master_table_4demands.parquet` |
| `01_dataset` | `master_table_4demands.parquet`, `preguntas_triaje/*`, `motivo_literal_ictus/*` (stroke), `medicacion_asistencia_antecedentes/*` | `df_modelling/df_triaje_preprocessed_<date>.parquet` |
| `01_04_dataset_stroke` | `df_triaje_preprocessed_<date>.parquet`, `stroke_table_cleaned_path` | — |
| `02_04_tables_stroke` | `df_triaje_preprocessed_<date>.parquet`, `stroke_table_cleaned_path` | `figures/04_stroke/tables/*.docx`, `*.jpg` |
| `03_01_modelling_cardiacarrest` | `df_triaje_preprocessed_<date>.parquet` | `cardiacarrest_table_modeling` (when `export_table=True`); figures under `figures/01_cardiac_arrest/` |
| `03_02_modelling_chestpain` | `df_triaje_preprocessed_<date>.parquet` | `chestpain_table_modeling` (when `export_table=True`); figures under `figures/02_chestpain/` |
| `03_03_modelling_dyspnea` | `df_triaje_preprocessed_<date>.parquet` | `dyspnea_table_modeling` (when `export_table=True`); figures under `figures/03_dyspnea/` |
| `03_04_modelling_stroke` | `df_triaje_preprocessed_<date>.parquet` | `stroke_table_cleaned_path` (when `export_table=True`); figures under `figures/04_stroke/` |
| `04_0X_eda_*` | `df_triaje_preprocessed_<date>.parquet` | metric plots (displayed inline only) |
| `stats/stats.ipynb` | `df_triaje` (in-memory) | `export_tables/icd_unique/*.csv` |

All `*.parquet`, `*.csv`, `*.xlsx`, `*.docx`, `*.jpg`, and `*.png` files, and the `data/` directory, are gitignored; actual data files live under `SOURCE_TABLES_PATH`.

## Setup

1. Clone the repository and ensure access to the shared data directory referenced by `SOURCE_TABLES_PATH` (e.g. `/opt/datos_compartidos/trIAje/data`).
2. Create `analysis/.env` (gitignored) with the required settings, for example:

   ```
   SOURCE_TABLES_PATH = '/opt/datos_compartidos/trIAje/data'
   EDATOSCASO_LOAD_PATH = 'EDATOSCASO_source/dataset_edatoscaso_llamadas_antecedentes.parquet'
   MASTER_TABLE_PATH = 'tabla_madre/master_table.parquet'
   MASTER_TABLE_4DEMANDS_PATH = 'tabla_madre/master_table_4demands.parquet'
   TABLE_EMBEDDING_SBERT_PATH = 'df_modelling/literal_reason_embedding_sbert_<date>.parquet'
   TABLE_EMBEDDING_RIGOBERT_PATH = 'df_modelling/literal_reason_embedding_rigobert_<date>.parquet'
   TRIAJE_TABLE_CLEANED_PATH = 'df_modelling/df_triaje_preprocessed_<date>.parquet'
   stroke_table_cleaned_path = 'df_modelling/df_stroke_preprocessed_<date>.parquet'
   cardiacarrest_table_modeling = 'df_modelling/df_cardiacarrest_modeling_<date>.parquet'
   chestpain_table_modeling = 'df_modelling/df_chestpain_modeling_<date>.parquet'
   dyspnea_table_modeling = 'df_modelling/df_dyspnea_modeling_<date>.parquet'
   ```

   Optional flags (see `kbase/config.py`): `run_embedding_sbert`, `run_embedding_rigobert` (`0` = load cached embeddings, `1` = recompute), and `test_pipeline` (`"all"` for full data, or `"test"` for a 10,000-row subset).

3. Install dependencies: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `flaml`, `shap`, `matplotlib`, `seaborn`, `scipy`, `python-docx`, `pydantic-settings`, `jupyter`.
4. Run notebooks **from the `analysis/` directory** so that `kbase` resolves as a local package:

   ```bash
   cd analysis
   jupyter notebook
   ```

   To execute a single notebook non-interactively:

   ```bash
   jupyter nbconvert --to notebook --execute <notebook>.ipynb
   ```

5. Recommended execution order: `00_01` → `00_02` → `00_03` → `01_dataset`, then the per-pathway `03_*` / `04_*` notebooks (and, for stroke, `01_04` / `02_04`).

## Known issues and TODO

- **`01_04_dataset_stroke.ipynb`**: `hist_cols = [c for c in df_stroke_reduced.columns if c.startswith("hist_")]` now evaluates to an empty list, since the antecedentes/active-problems columns were unified into `com_*`. The "Past history" correlation/Jaccard sections silently produce empty-group output instead of an error.
- **`03_01_modelling_cardiacarrest.ipynb`**: `run_multiclass_automl_model()` calls `data_load_col_selection(triage_value, include_embeddings)` with only 2 positional arguments, while the function signature requires 7 (`target_column, triage_value, include_lr, include_history, include_com, include_medication, include_embeddings`). Running this notebook currently raises a `TypeError`.
- **`03_01_modelling_cardiacarrest.ipynb`**: the "Subset: With Triage" and "Subset: Without Triage" sections both use `triage_value = 1`; the latter likely should use `triage_value = 0`, as in the analogous dyspnea notebook.
- **`kbase.modeling.data_load_col_selection`**'s `include_history` flag is effectively a no-op: it selects columns starting with `hist_`, but no such columns exist in the current preprocessed table (they were unified into `com_*`, controlled by `include_com`).
- **`04_01_eda_cardiacarrest.ipynb`**: calls `eda.analyze_emergency_performance("Unconscious/Cardiac arrest")` with a single argument, but `analyze_emergency_performance` requires `(demand_type, y_real_col, y_pred_col, sex_group)` with no defaults. Running this notebook currently raises a `TypeError`.
- **`04_02_eda_chestpain.ipynb`** and **`04_03_eda_dyspnea.ipynb`**: both pass `y_pred_col='p1_predicted'`, but `df_triaje_preprocessed_<date>.parquet` does not contain a `p1_predicted` column (only `p1_assigned`, `p1_real_emerg`, `p1_real_bps`). Running these notebooks currently raises a `KeyError`. Of the four `04_*` notebooks, only `04_04_eda_stroke.ipynb` (which uses `p1_assigned`) runs as written.
- **`stats/stats.ipynb`**: the "Environment" cell is fully commented out, and the notebook references `df_triaje`, `df_edt_filtered`, and `df_edt_duplicates_demandapk`, which are not defined anywhere in the notebook — it currently only runs interactively after another notebook's variables are already in the kernel. It also contains Spanish variable names and comments, which do not comply with the English-only language policy in `CLAUDE.md`.
- **`export_tables/edatoscaso.csv`** and **`export_tables/triaje_table.csv`** are not produced by any tracked notebook or script; only `icd_unique/*.csv` is generated by `stats/stats.ipynb`. The origin of these two files should be documented or reproduced in code.
- **`analysis/figures/`** currently only contains `04_stroke/`, even though `kbase.modeling._FIGURES_DIR_MAP` defines output folders for all four pathways (`01_cardiac_arrest`, `02_chestpain`, `03_dyspnea`, `04_stroke`); the other three pathways have not yet been run with figure export.
- **`analysis/demands/`** is currently empty; it is reserved for the planned complaint-specific preprocessing refactor described above.
- The local R project (`trIAje.Rproj`, `.RData`, `.Rproj.user/`, `.Rhistory`) is gitignored and outside the reproducible, version-controlled pipeline.
