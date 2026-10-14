# trIAje: stroke priority modeling

trIAje is a clinical research project that develops and evaluates machine-learning support for emergency medical dispatch of suspected stroke. The analysis focuses on calls classified as stroke (`demand_type_1 == 54`) and predicts whether each call requires the highest dispatch priority (P1).

The main modeling outcome is `p1_real_emerg_bps`, which combines the priority derived from emergency-service and hospital diagnostic information. The original dispatch assignment, `p1_assigned`, is retained as the clinical benchmark.

This README documents only the stroke workflow. Other files may remain in the repository for now, but they are outside the scope of this project and are intentionally omitted here.

## Stroke pipeline

```text
Raw EMS records
    │
    ▼
00_01_grouping_duplicates.ipynb
    │  Deduplicated master table
    ├──────────────► 00_02_EDA-QC.ipynb
    │                 Read-only quality control
    ▼
00_03_EDA_general.ipynb
    │  Intermediate selected cohort
    ▼
01_dataset.ipynb
    │  Stroke-only cleaned and feature-engineered table
    ├──────────────► 04_04_eda_stroke.ipynb
    │                 Clinical performance of the dispatch assignment
    ▼
03_04_modelling_stroke.ipynb
    │  Reduced stroke modeling table and model figures
    ├──────────────► 01_04_dataset_stroke.ipynb
    │                 Feature completeness and correlation checks
    └──────────────► 02_04_tables_stroke.ipynb
                      Descriptive Table 1 outputs
```

Recommended execution order:

1. `00_01_grouping_duplicates.ipynb`
2. `00_02_EDA-QC.ipynb`
3. `00_03_EDA_general.ipynb`
4. `01_dataset.ipynb`
5. `03_04_modelling_stroke.ipynb`
6. `01_04_dataset_stroke.ipynb`
7. `02_04_tables_stroke.ipynb`
8. `04_04_eda_stroke.ipynb`

`00_02_EDA-QC.ipynb` is a diagnostic step and does not create a new dataset. `04_04_eda_stroke.ipynb` reads the cleaned table produced by `01_dataset.ipynb`, so it can be run before or after the modeling notebook.

## Pipeline stages

### 1. Deduplicate the source records

[`analysis/00_01_grouping_duplicates.ipynb`](analysis/00_01_grouping_duplicates.ipynb) loads the raw EMS export configured by `edatoscaso_load_path`, resolves repeated records, removes rows without usable identifiers, creates `PATIENTID`, normalizes data types, and writes the deduplicated master table configured by `master_table_path`.

### 2. Audit the master table

[`analysis/00_02_EDA-QC.ipynb`](analysis/00_02_EDA-QC.ipynb) checks identifiers, duplicate patients, and missingness in the master table. This notebook is read-only and is intended to detect data-quality problems before cohort construction.

### 3. Build the intermediate cohort

[`analysis/00_03_EDA_general.ipynb`](analysis/00_03_EDA_general.ipynb) reads the deduplicated master table, selects the configured emergency-call cohort that contains the stroke records, reorders the columns, and writes the intermediate table configured by `master_table_4demands_path`.

The output setting retains a legacy name. The definitive stroke filter is applied at the beginning of the next notebook.

### 4. Construct the stroke dataset

[`analysis/01_dataset.ipynb`](analysis/01_dataset.ipynb) is the central preprocessing notebook. It renames the source columns and immediately restricts the cohort to stroke calls (`demand_type_1 == 54`). It then:

- cleans identifiers, dates, demographic variables, locations, and dispatch attributes;
- joins the stroke triage questions;
- joins emergency and hospital diagnostic codes;
- derives literal-reason features from the call text;
- joins past-history, active-problem, and medication features;
- creates `p1_assigned`, `p1_real_emerg`, `p1_real_bps`, and the combined `p1_real_emerg_bps` outcome;
- derives temporal and categorical features;
- applies the final exclusion criteria;
- optionally creates or loads text embeddings; and
- exports the cleaned stroke table configured by `triaje_table_cleaned_path`.

The notebook also produces the stroke cohort-flow diagram under `analysis/figures/04_stroke/consort_diagram/`.

### 5. Train and reproduce the stroke model

[`analysis/03_04_modelling_stroke.ipynb`](analysis/03_04_modelling_stroke.ipynb) calls `kbase.modeling.run_binary_automl_model()` for demand code `54` and outcome `p1_real_emerg_bps`. The notebook is organized into four explicit sections:

- **AutoML time-budget comparison:** equivalent searches with budgets from 10 to 60 minutes;
- **Pipeline smoke test:** a short run that validates the complete workflow;
- **Time-budget and iteration-limit comparison:** records the iterations completed in 30 minutes and replays the search with that fixed iteration limit; and
- **Final fixed model:** reproduces the selected LightGBM configuration, calculates feature importance, saves figures, and exports the reduced modeling table configured by `stroke_table_cleaned_path`.

All runs use average precision for model selection and estimate the classification threshold from out-of-fold predictions with Youden's J statistic. Because the experiment cells use `export_table=True`, run only the experiment whose exported table you want to retain. The final fixed-model section is the reproducible endpoint of the modeling workflow.

Model figures are written to `analysis/figures/04_stroke/`.

### 6. Check the modeling features

[`analysis/01_04_dataset_stroke.ipynb`](analysis/01_04_dataset_stroke.ipynb) loads both the cleaned stroke dataset and the reduced table exported by the modeling step. It audits medication completeness and examines correlation and Jaccard similarity within the general, past-history, literal-reason, and medication feature groups. It does not persist a new dataset.

### 7. Generate descriptive tables

[`analysis/02_04_tables_stroke.ipynb`](analysis/02_04_tables_stroke.ipynb) creates descriptive Table 1 outputs for the complete study period and the configured time periods, stratified by real or assigned P1 status. Word and image versions are saved under `analysis/figures/04_stroke/tables/`.

### 8. Evaluate the original dispatch assignment

[`analysis/04_04_eda_stroke.ipynb`](analysis/04_04_eda_stroke.ipynb) compares `p1_assigned` with:

- `p1_real_emerg_bps`, the combined emergency and hospital reference outcome; and
- `p1_real_emerg`, the emergency-service reference outcome.

It reports clinical classification metrics for the complete stroke cohort and stratifies the analysis by sex and age when requested.

## Relevant repository structure

```text
trIAje/
├── README.md
└── analysis/
    ├── .env                              # Local paths and execution flags; not tracked
    ├── 00_01_grouping_duplicates.ipynb  # Source deduplication
    ├── 00_02_EDA-QC.ipynb               # Master-table quality control
    ├── 00_03_EDA_general.ipynb          # Intermediate cohort selection
    ├── 01_dataset.ipynb                 # Stroke dataset construction
    ├── 01_04_dataset_stroke.ipynb       # Stroke feature-quality checks
    ├── 02_04_tables_stroke.ipynb        # Stroke descriptive tables
    ├── 03_04_modelling_stroke.ipynb     # Stroke AutoML and final fixed model
    ├── 04_04_eda_stroke.ipynb           # Clinical-performance analysis
    ├── figures/
    │   └── 04_stroke/                   # Stroke tables and figures
    └── kbase/
        ├── config.py                    # Environment-based configuration
        ├── eda.py                       # Clinical-performance metrics and plots
        ├── labels.py                    # Human-readable feature labels
        ├── modeling.py                  # AutoML, thresholds, explainability, and export
        └── preprocessing.py             # Cleaning and feature-engineering utilities
```

## Main inputs and outputs

| Notebook | Main input | Main output |
|---|---|---|
| `00_01_grouping_duplicates` | Raw EMS records and source history data | `master_table_path` |
| `00_02_EDA-QC` | `master_table_path` | Quality-control results displayed in the notebook |
| `00_03_EDA_general` | `master_table_path` | `master_table_4demands_path` (legacy setting name) |
| `01_dataset` | Intermediate cohort plus triage, diagnostic, history, medication, and text-derived data | `triaje_table_cleaned_path` |
| `03_04_modelling_stroke` | `triaje_table_cleaned_path` | `stroke_table_cleaned_path` and `figures/04_stroke/` |
| `01_04_dataset_stroke` | Cleaned and reduced stroke tables | Quality-control results displayed in the notebook |
| `02_04_tables_stroke` | `stroke_table_cleaned_path` | `figures/04_stroke/tables/*.docx` and `*.jpg` |
| `04_04_eda_stroke` | `triaje_table_cleaned_path` | Clinical-performance results displayed in the notebook |

Patient-level data and generated analysis artifacts are not stored in Git. Access to the configured source-data directory is required to execute the workflow.

## Configuration

The notebooks read local paths and execution flags from `analysis/.env` through `analysis/kbase/config.py`. The stroke workflow uses the following settings:

```dotenv
TRIAJE_PROJECT_PATH=/path/to/trIAje/analysis
SOURCE_TABLES_PATH=/path/to/source/data
EDATOSCASO_LOAD_PATH=relative/path/to/raw_records.parquet
MASTER_TABLE_PATH=relative/path/to/master_table.parquet
MASTER_TABLE_4DEMANDS_PATH=relative/path/to/intermediate_cohort.parquet
TRIAJE_TABLE_CLEANED_PATH=relative/path/to/stroke_preprocessed.parquet
STROKE_TABLE_CLEANED_PATH=relative/path/to/stroke_modeling.parquet
TABLE_EMBEDDING_SBERT_PATH=relative/path/to/sbert_embeddings.parquet
TABLE_EMBEDDING_RIGOBERT_PATH=relative/path/to/rigobert_embeddings.parquet
RUN_EMBEDDING_SBERT=0
RUN_EMBEDDING_RIGOBERT=0
TEST_PIPELINE=all
```

Paths other than `TRIAJE_PROJECT_PATH` and `SOURCE_TABLES_PATH` are resolved relative to `SOURCE_TABLES_PATH`. Use `TEST_PIPELINE=test` for the development subset defined in `kbase/preprocessing.py`; use `TEST_PIPELINE=all` for the complete cohort. Set an embedding flag to `1` only when the corresponding embeddings must be recomputed.

Keep `.env` local and do not commit credentials, protected health information, or absolute paths tied to a private environment.

## Running the analysis

Run Jupyter from the `analysis/` directory so that the local `kbase` package resolves correctly:

```bash
cd analysis
jupyter lab
```

To execute one notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace 03_04_modelling_stroke.ipynb
```

Before running the full pipeline, ensure that the Python environment provides Jupyter, pandas, NumPy, PyArrow, scikit-learn, FLAML, SHAP, Matplotlib, seaborn, SciPy, pydantic-settings, sentence-transformers, transformers, PyTorch, and python-docx.
