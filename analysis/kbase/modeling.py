# ===============================================================
# Generic binary classification pipeline using FLAML
# Cohort-agnostic (no stroke-specific references)
# ===============================================================

import pandas as pd
import numpy as np
import seaborn as sns
from flaml import AutoML
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, accuracy_score, confusion_matrix,
    classification_report, average_precision_score,
    log_loss, precision_recall_curve, roc_curve, brier_score_loss
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
import time
import pyarrow.parquet as pq
from typing import Optional, List, Union
import shap
import warnings
try:
    from IPython.display import display
except ImportError:
    display = print

import kbase.preprocessing as dp
import kbase.eda as eda
from kbase.paired_comparison import (
    compare_model_vs_triage_common,
    get_common_comparison_subset,
)
from kbase.config import settings
from kbase.labels import get_labels_map, get_display_label, resolve_demand_key
from kbase.model_hyperparameters import (
    AVAILABLE_CONFIGS,
    FIXED_LGBM_CONFIG,
    FIXED_LGBM_STROKE_HIST_P1_REAL_EMERG_BPS_20260821,
)

_FIGURES_DIR_MAP = {
    16: 'figures/03_dyspnea',
    23: 'figures/02_chestpain',
    54: 'figures/04_stroke',
    'cardiac_arrest': 'figures/01_cardiac_arrest',
}

_OUTCOME_COLUMNS = [
    "p1_assigned",
    "p1_real_emerg",
    "p1_real_bps",
    "p1_real_emerg_bps",
]

TELEPHONIC_TRIAGE_LABEL = "Telephonic triage system"
ML_MODEL_LABEL = "ML model"

SUBGROUP_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "subgroup_pastel",
    [
        (0.00, "#EF9496"),
        (0.30, "#F5BE7D"),
        (0.50, "#CEC76E"),
        (0.75, "#B2C764"),
        (1.00, "#8DC05A"),
    ],
)
SUBGROUP_HEATMAP_CMAP.set_bad("#F2F2F2")


class ModelingResult(dict):
    """Dictionary-like pipeline result with a concise notebook representation.

    All detailed DataFrames and analysis objects remain available by key.  The
    compact representation prevents notebooks from rendering every exported
    interpretability and comparison table after a bare function call.
    """

    @staticmethod
    def _format_metric(value, decimals=4):
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    def __repr__(self):
        metrics = self.get("metrics") or {}
        summary = self.get("summary")
        report = self.get("common_subset_report") or {}

        n_test = None
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            n_test = summary.iloc[0].get("n_test")

        test_text = "n/a" if pd.isna(n_test) else f"{int(n_test):,}"
        common_text = "n/a"
        if report.get("n_common") is not None:
            common_text = (
                f"{int(report['n_common']):,} "
                f"(excluded: {int(report.get('n_excluded', 0)):,})"
            )

        return (
            "ModelingResult(\n"
            f"  test N: {test_text}\n"
            f"  ROC-AUC: {self._format_metric(metrics.get('roc_auc'))}\n"
            f"  PR-AUC: {self._format_metric(metrics.get('pr_auc'))}\n"
            f"  threshold: {self._format_metric(self.get('threshold'))}\n"
            f"  paired comparison N: {common_text}\n"
            "  Detailed results remain available by key; tables and figures were exported.\n"
            ")"
        )

    def _repr_html_(self):
        import html
        return "<pre>{}</pre>".format(html.escape(self.__repr__()))

    def _repr_pretty_(self, printer, cycle):
        printer.text(self.__repr__())


# Set by run_*_automl_model at the start of each pipeline run.
# All internal plot functions read from here via save_figure().
_current_demand_code = None
_current_time_budget = None

_ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _get_figures_dir(demand_code):
    """Returns the absolute figures directory path for a given demand_code."""
    key = resolve_demand_key(demand_code)
    rel = _FIGURES_DIR_MAP.get(key)
    return os.path.join(_ANALYSIS_DIR, rel) if rel is not None else None

def _resolve_figure_path(filename, figures_dir=None):
    folder = figures_dir if figures_dir is not None else _get_figures_dir(_current_demand_code)
    if folder is None:
        return
    os.makedirs(folder, exist_ok=True)
    if _current_time_budget is not None:
        stem, ext = os.path.splitext(filename)
        filename = f"{stem}_{int(_current_time_budget // 60)}min{ext}"
    return os.path.join(folder, filename)

def save_figure(fig, filename, dpi=300, figures_dir=None):
    """Saves a matplotlib figure to the demand-specific figures directory with high quality."""
    path = _resolve_figure_path(filename, figures_dir=figures_dir)
    if path is None:
        print(f"[save_figure] WARNING: no folder for demand_code={_current_demand_code!r}, skipping '{filename}'")
        return
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved: {path}")

def save_table(df, filename, title=None, footnote=None):
    """Saves a dataframe as a Word table in the demand-specific tables directory."""
    folder = _get_figures_dir(_current_demand_code)
    if folder is None:
        print(f"[save_table] WARNING: no folder for demand_code={_current_demand_code!r}, skipping '{filename}'")
        return None
    os.makedirs(folder, exist_ok=True)
    if _current_time_budget is not None:
        stem, ext = os.path.splitext(filename)
        filename = f"{stem}_{int(_current_time_budget // 60)}min{ext}"

    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.section import WD_ORIENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    HDR_BG = "2D2D2D"
    HDR_FG = RGBColor(0xFF, 0xFF, 0xFF)
    ALT_BG = "EFEFEF"
    EVEN_BG = "FFFFFF"

    def _set_cell_bg(cell, hex_color):
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), hex_color)
        tcPr.append(shd)

    display_df = df.copy()
    for col in display_df.select_dtypes(include=[np.number]).columns:
        display_df[col] = display_df[col].map(
            lambda v: "" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v)
        )

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width

    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(10)

    if title:
        title_para = doc.add_paragraph()
        title_run = title_para.add_run(title)
        title_run.bold = True
        title_run.font.size = Pt(11)
        doc.add_paragraph()

    cols = list(display_df.columns)
    table = doc.add_table(rows=len(display_df) + 1, cols=len(cols))
    table.style = 'Table Grid'

    for j, col_name in enumerate(cols):
        cell = table.rows[0].cells[j]
        cell.text = str(col_name)
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = HDR_FG
        run.font.size = Pt(8)
        _set_cell_bg(cell, HDR_BG)

    for i, row_vals in enumerate(display_df.itertuples(index=False)):
        bg = ALT_BG if i % 2 == 1 else EVEN_BG
        for j, val in enumerate(row_vals):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val)
            run = cell.paragraphs[0].runs[0]
            run.font.size = Pt(8)
            _set_cell_bg(cell, bg)

    if footnote:
        doc.add_paragraph()
        note_para = doc.add_paragraph()
        note_run = note_para.add_run(footnote)
        note_run.italic = True
        note_run.font.size = Pt(8)

    tables_dir = os.path.join(folder, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, filename)
    doc.save(path)
    print(f"Table saved: {path}")
    return path

# This ensures .info() shows up to 50 columns by default
pd.set_option('display.max_info_columns', 50)
# This ensures that when you print a DataFrame, you see 50 rows
pd.set_option('display.max_rows', 50)

protocol_questions = {
    # 16: Dyspnea / Respiratory distress (Asfixia)
    16: [
        # 1. Asfixia start
        'q1_a', 'q1_b',
        # 2. Previous diseases
        'q2_a', 'q2_b', 'q2_c', 'q2_d', 'q2_e', 'q2_f', 'q2_g', 'q2_h',
        # 3. Speech difficulty
        'q3_a', 'q3_b', 'q3_c',
        # 4. Associated symptoms
        'q4_a', 'q4_b', 'q4_c', 'q4_d', 'q4_e', 'q4_f', 'q4_g', 
        'q4_h', 'q4_i', 'q4_j', 'q4_k', 'q4_l', 'q4_m'
    ],
    
    # 23: Chest Pain (Dolor Torácico)
    23: [
        # 1. Heart history
        'q1_a', 'q1_b', 'q1_c', 'q1_d',
        # 2. Location
        'q2_a', 'q2_b',
        # 3. Type of pain
        'q3_a', 'q3_b', 'q3_c', 'q3_d',
        # 4. Relation to movement/rest
        'q4_a', 'q4_b', 'q4_c', 
        # 5. Associated symptoms
        'q5_a', 'q5_b', 'q5_c', 'q5_d', 'q5_e', 'q5_f',
        # 6. Change/Improvement with meds
        'q6_a', 'q6_b', 'q6_c', 'q6_d'
    ],

    # 54: Stroke / ACV
    54: [
        # 1. Gaze / Eyes
        'q1_a', 'q1_b',
        # 2. Breathing
        'q2_a', 'q2_b',
        # 3. Main complaint (Face, Arm, Speech, etc.)
        'q3_a', 'q3_b', 'q3_c', 'q3_d', 'q3_e', 'q3_f',
        # 4. First time occurrence
        'q4_a', 'q4_b',
        # 5. Time of evolution (Numerical/String)
        'q5_a', 
        # 6. Lifestyle / Autonomy
        'q6_a', 'q6_b'
    ],

    # 36 & 58: Cardiac Arrest / Unconsciousness
    'cardiac_arrest': [
        # 1. Can speak?
        'q1_a', 'q1_b',
        # 2. Eyes/Gaze
        'q2_a', 'q2_b',
        # 3. Breathing quality
        'q3_a', 'q3_b', 'q3_d', # Nota: Tu protocolo salta de b a d
        # 4. Reaction to stimulus
        'q4_a', 'q4_b',
        # 5. Previous diseases
        'q5_a', 'q5_b', 'q5_c', 'q5_d', 'q5_e', 'q5_f',
        # 6. Time of evolution
        'q6_a',
        # 7. Previous episodes
        'q7_a', 'q7_b'
    ]
}

def time_print():
    now = datetime.now() 
    print(f"Current Date and Time: {now.strftime('%d/%m/%Y %H:%M:%S')}\n")

def _make_outcome_year_strata(y, year):
    """Builds combined outcome-year strata for balanced splitting."""
    # Keep outcome and year aligned even when callers pass Series with custom indexes.
    y_s = pd.Series(y)
    if hasattr(year, "index") and y_s.index.equals(year.index):
        year_s = pd.Series(year)
    else:
        year_s = pd.Series(np.asarray(year), index=y_s.index)
    return y_s.astype(str) + "__year_" + year_s.astype(str)


def _validate_strata_counts(strata, min_count, context):
    """Validates that every stratum has enough samples for the requested split."""
    # scikit-learn stratification requires enough samples in every stratum.
    counts = pd.Series(strata).value_counts()
    rare = counts[counts < min_count]
    if not rare.empty:
        raise ValueError(
            f"{context} requires at least {min_count} samples per outcome-year stratum. "
            f"Rare strata: {rare.to_dict()}"
        )

class OutcomeYearStratifiedKFold:
    """K-fold splitter that stratifies by the stored outcome-year strata."""

    def __init__(self, strata, n_splits=5, random_state=42):
        # Store the precomputed outcome-year strata used later by FLAML CV.
        self.strata = pd.Series(strata)
        self.n_splits = n_splits
        self.random_state = random_state
        _validate_strata_counts(
            self.strata,
            min_count=n_splits,
            context=f"{n_splits}-fold cross-validation",
        )

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of CV folds."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yields train/test indices while preserving outcome-year strata per fold."""
        if hasattr(X, "index"):
            strata = self.strata.loc[X.index].reset_index(drop=True)
        else:
            strata = self.strata.iloc[:len(X)].reset_index(drop=True)

        rng = np.random.default_rng(self.random_state)
        fold_test_indices = [[] for _ in range(self.n_splits)]

        # Split each outcome-year cell across folds, rotating fold assignment to balance sizes.
        for stratum_number, (_, stratum) in enumerate(strata.groupby(strata, sort=False)):
            indices = stratum.index.to_numpy()
            rng.shuffle(indices)
            for fold_idx, chunk in enumerate(np.array_split(indices, self.n_splits)):
                rotated_fold_idx = (fold_idx + stratum_number) % self.n_splits
                fold_test_indices[rotated_fold_idx].extend(chunk.tolist())

        all_indices = np.arange(len(strata))
        for test_indices in fold_test_indices:
            test_indices = np.array(sorted(test_indices), dtype=int)
            train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=False)
            yield train_indices, test_indices

def triage_print(triage_value, cohort_name, demand_code):
    """ Prints the initial configuration of the model run """
    print(f"=== Running the model for {cohort_name} cohort - code {demand_code} ===")

    if triage_value is None:
        print("Triage subset: ALL patients (no triage-based filtering)")
    elif triage_value == 1:
        print("Triage subset: WITH structured triage information")
    elif triage_value == 0:
        print("Triage subset: WITHOUT structured triage information")
    else:
        raise ValueError("triage_value must be None, 0, or 1")

def data_load_col_selection(target_column, triage_value,
                            include_lr, include_hist, include_com,
                            include_medication, include_embeddings)-> pd.DataFrame:
    """Loads data and performs initial feature selection based on settings"""
    # Load preprocessed cleaned table
    df = pq.read_table(
        os.path.join(settings.source_tables_path, settings.triaje_table_cleaned_path)
    ).to_pandas()

    # Select base columns
    ## Some columns are later excluded for modeling, but are included in the exported
    ## table for statistics analysis
    base_cols = [
        # --- Patient demographics ---
        "age",
        "sex",

        # --- Target ---
        target_column,

        # --- Temporal: day of week ---
        "day_week_monday", "day_week_tuesday", "day_week_wednesday", "day_week_thursday",
        "day_week_friday", "day_week_saturday", "day_week_sunday",

        # --- Temporal: time of day ---
        "time_of_day_early_morning", "time_of_day_morning", "time_of_day_afternoon", "time_of_day_night",

        # --- Temporal: month (seasons excluded — redundant with months, lower resolution) ---
        "month_january", "month_february", "month_march", "month_april",
        "month_may", "month_june", "month_july", "month_august",
        "month_september", "month_october", "month_november", "month_december",

        # --- Call context: patient location ---
        "location_patient_home", "location_patient_public_road", "location_patient_other",

        # --- Call context: entity receiving the alert ---
        "alert_receiver_112", "alert_receiver_user", "alert_receiver_pol_fg",
        "alert_receiver_hs", "alert_receiver_tele", "alert_receiver_others",

        # --- Geographic: province and coordinates ---
        "province_almeria", "province_cadiz", "province_cordoba", "province_granada",
        "province_huelva", "province_jaen", "province_malaga", "province_sevilla",
        # "incident_latitude",
        # "incident_longitude",

        # --- Excluded before modeling (kept for export / statistics only) ---
        "demandpk",
        "demand_date",
        "demand_type_1",
        "has_icd_emerg",
        "has_icd_bps",
        "triage",
        "year",
        "has_history",
        "has_history_old",
        "has_problems_old",
        "has_com",
        "has_med",
        "literal_reason", 
        "hcdm_id"
    ]
    base_cols += _OUTCOME_COLUMNS

    # Initialize modelling columns with base columns
    modelling_cols = base_cols.copy()
    # Add age groups (One-Hot Encoded columns, e.g. age_15_24, age_75_plus)
    modelling_cols += [
        col for col in df.columns
        if col.startswith('age_') and col.split('_')[1].isdigit()
    ]
    # Add triage questions (One-Hot Encoded columns)
    if triage_value != 0:
        modelling_cols += [col for col in df.columns if col.startswith('q')]
    # Add literal reason columns (One-Hot Encoded columns)
    if include_lr == True:
        modelling_cols += [col for col in df.columns if col.startswith('lr_')]
    # Add medication columns (One-Hot Encoded columns)
    if include_medication == True:
        modelling_cols += [col for col in df.columns if col.startswith('atc_')]
    # Add past history and active problem columns
    # sourced from antecedentes-problemas-motivoliteral (one-hot encoded columns)
    if include_hist == True:
        modelling_cols += [col for col in df.columns if col.startswith('hist_')]
    # Add legacy past history and active problem columns
    # sourced from old antecedentes/problemas tables unified into com_* columns.
    if include_com == True:
        modelling_cols += [col for col in df.columns if col.startswith('com_')]
    # Include NLP text embeddings if flag is set to True
    if include_embeddings:
        embedding_cols = [col for col in df.columns if col.startswith('emb_')]
        modelling_cols += embedding_cols
        print(f"Feature Set: Including {len(embedding_cols)} text embedding dimensions.")
            
    # Select base columns for modeling, preserving order while preventing
    # duplicate labels from repeated feature-family inclusion.
    selected_cols = list(dict.fromkeys(c for c in modelling_cols if c in df.columns))
    df_model = df[selected_cols].copy()
    # Transform data types
    df_model = dp.transform_column_dtypes(df_model)

    return df_model

def col_validation(df, target_column):
    """Validates that essential columns are present in the dataframe"""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    if "triage" not in df.columns:
        raise ValueError("Required column 'triage' not found in dataframe")

def data_filtering(df, target_column, demand_code, 
                   triage_value, min_age, years, export_table, protocol_features) -> pd.DataFrame:
    """Applies cohort, triage, age, and protocol-specific filters"""

    # Filter by valid values in target_column
    if target_column is not None:
        df = df.dropna(subset=[target_column])
        if target_column not in df.columns:
            raise ValueError(f"{target_column} column not found but demand_code was provided")
        
    # Filter by demand_code
    if demand_code is not None:
        if "demand_type_1" not in df.columns:
            raise ValueError("'demand_type_1' column not found but demand_code was provided")
        
        # Handle both single integer and list of integers
        if isinstance(demand_code, (list, tuple)):
            df = df[df["demand_type_1"].isin(demand_code)].copy()
            print(f"Filtering by demand_type_1 in {demand_code}")
        else:
            df = df[df["demand_type_1"] == demand_code].copy()
            print(f"Filtering by demand_type_1 == {demand_code}")

    # Filter out medication columns by threshold-specific available medication columns and
    # group and remove similar coloumns
    df = dp.analyze_atc_columns(
        df,
        threshold=1.0,
        drop_columns=True,
    )

    # Group all atc_, hist_ and com_ columns together (including the grouped
    # categories created above, which pandas appends at the end).
    atc_cols = [c for c in df.columns if c.startswith('atc_')]
    hist_cols = [c for c in df.columns if c.startswith('hist_')]
    com_cols = [c for c in df.columns if c.startswith('com_')]
    other_cols = [c for c in df.columns if c not in atc_cols and c not in hist_cols and c not in com_cols]
    df = df[other_cols + atc_cols + hist_cols + com_cols]

    # Filter by triage patients
    if triage_value is not None:
        df = df[df["triage"] == triage_value].copy()

    # Filter by min age
    if min_age is not None:
        if "age" not in df.columns:
            raise ValueError("'age' column not found but min_age was provided")
        df = df[df["age"] >= min_age].copy()
        print(f"Filtering by age >= {min_age}")

    # Filter by specific years
    if years is not None:
        if "year" not in df.columns:
            raise ValueError("'year' column not found but years was provided")
        df = df[df["year"].isin(years)].copy()
        print(f"Filtering specific year cohort: {years}")

    # Drop triage columns not present in the specific demand type
    if demand_code is not None and protocol_features is not None:
        # Determine the correct key for protocol_features
        rule_key = None
        # Normalize demand_code to a list for consistent comparison
        codes_list = demand_code if isinstance(demand_code, (list, tuple)) else [demand_code]

        # Case 1: Cardiac Arrest (Multiple codes)
        if set(codes_list) == {36, 58}:
            rule_key = 'cardiac_arrest'
        # Case 2: Single code protocols (16, 23, 54)
        elif len(codes_list) == 1 and codes_list[0] in protocol_features:
            rule_key = codes_list[0]

        # Apply pruning if a rule was found
        if rule_key is not None:
            allowed_cols = protocol_features[rule_key]
            current_q_cols = [c for c in df.columns if c.startswith('q')]
            cols_to_drop = [c for c in current_q_cols if c not in allowed_cols]
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        # Integrated warning if no protocol matches the demand_code
        else:
            print(f"⚠️ Warning: No specific protocol found for demand_code {demand_code}")

    # Clean column names: remove 'q' prefix and underscores for readability
    df.columns = [col.replace('q', '', 1).replace('_', '') 
                  if col.startswith('q') else col for col in df.columns]
                   
    # Export of preprocessed, ready for modeling, table
    if export_table:    
        rule_to_path = {
            'cardiac_arrest': settings.cardiacarrest_table_modeling,
            16: settings.dyspnea_table_modeling,
            23: settings.chestpain_table_modeling,
            54: settings.stroke_table_cleaned_path,
        }
        export_path = rule_to_path.get(rule_key)
        
        if export_path is None:
            raise ValueError(f"Unknown export path: {rule_key}")
        
        full_export_path = os.path.join(settings.source_tables_path, export_path)
        if not df.columns.is_unique:
            duplicated_cols = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate columns before export: {duplicated_cols}")

        df.to_parquet(full_export_path, index=False)

        # Check if the export file was created correctly
        if os.path.exists(full_export_path):
            file_size_mb = os.path.getsize(full_export_path) / (1024 * 1024)
            print(f"✅ Export successful!")
            print(f"--- Path: {full_export_path}")
            print(f"--- Size: {file_size_mb:.2f} MB")
        else:
            print(f"❌ Export failed: File not found at {full_export_path}")

    # Drop columns after cohort filtering and data export
    # for preparation to modeling.
    # Continuous 'age' remains a model feature; the WHO age-group indicators
    # (age_15_24 ... age_75_plus) are reserved for subgroup fairness evaluation
    # and are therefore excluded from the modeling feature set here.
    age_group_cols = [
        c for c in df.columns
        if c.startswith('age_') and c.split('_')[1].isdigit()
    ]
    non_target_outcomes = [c for c in _OUTCOME_COLUMNS if c != target_column]
    cols_to_drop = ["demandpk", "demand_date", "demand_type_1",
                    "has_history", "has_history_old", "has_problems_old", "has_com", "has_med",
                    "has_icd_emerg", "has_icd_bps",
                    "triage", "literal_reason", "hcdm_id"] + non_target_outcomes + age_group_cols
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    print(f"\nFinal cohort size for {target_column}: {len(df)}")
    print("Outcome distribution: ", df[target_column].value_counts(dropna=False), "\n")

    print(f"\n" + "="*40)
    print(f"Columns retained before train/test split: {len(df.columns)}")
    print(f"Not used as model features: {target_column} is the outcome; year is used with {target_column} for stratification")
    print("="*40 + "\n")

    return df


def print_feature_summary(X_train):
    """Prints the feature groups that are actually passed to the model."""
    triage_cols  = [c for c in X_train.columns if c[0].isdigit()]
    lr_cols      = [c for c in X_train.columns if c.startswith("lr_")]
    hist_cols    = [c for c in X_train.columns if c.startswith("hist_")]
    com_cols     = [c for c in X_train.columns if c.startswith("com_")]
    atc_cols     = [c for c in X_train.columns if c.startswith("atc")]
    other_cols   = [c for c in X_train.columns if c not in triage_cols + lr_cols + hist_cols + com_cols + atc_cols]

    print("\n" + "="*40)
    print("FEATURE SUMMARY BY GROUP")
    print("="*40)
    print(f"  Triage questions  (digit prefix): {len(triage_cols):>4}")
    print(f"  Literal reason    (lr_):           {len(lr_cols):>4}")
    print(f"  Clinical history  (hist_ new):      {len(hist_cols):>4}")
    print(f"  Legacy history    (com_ old):       {len(com_cols):>4}")
    print(f"  Medication        (atc):           {len(atc_cols):>4}")
    print(f"  Other:                             {len(other_cols):>4}")
    print(f"  {'-'*30}")
    print(f"  Total model features:              {X_train.shape[1]:>4}")
    print("="*40 + "\n")

    X_train.info(verbose=True)

def train_test_split_weights(df, target_column, test_size, seed):
    """Splits data into train/test sets and computes sample weights for imbalance"""
    if "year" not in df.columns:
        raise ValueError("'year' column is required for outcome-year stratified splitting")

    # Keep year only as splitting metadata: it is not used as a model feature.
    year = df["year"]
    X = df.drop(columns=[target_column, "year"])
    y = df[target_column]

    # Stratify by the joint outcome-year cell to preserve both prevalence and year mix.
    strata = _make_outcome_year_strata(y, year)
    _validate_strata_counts(
        strata,
        min_count=2,
        context="Train/test outcome-year stratified split",
    )

    # Ensure categorical consistency
    # This prevents the 'categorical_feature do not match' error in LightGBM/FLAML
    categorical_cols = X.select_dtypes(include=['category']).columns
    
    for col in categorical_cols:
        # 1. Convert to string and then back to category to reset the label dictionary
        # 2. This ensures that X_train and X_test share the exact same internal mapping
        #    even if one split is missing a specific category value.
        X[col] = X[col].astype(str).astype('category')
        
    X_train, X_test, y_train, y_test, year_train, year_test = train_test_split(
        X,
        y,
        year,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Model features used for training: {X_train.shape[1]}")
    print_feature_summary(X_train)

    # Sample weights (class imbalance)
    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    return X_train, X_test, y_train, y_test, year_train, year_test, sample_weight

import pandas as pd
from flaml import AutoML

def run_automl_training(
    X_train,
    y_train,
    year_train,
    sample_weight,
    time_budget,
    optimize_metric,
    n_splits_cv,
    seed,
    task,
    max_iter=None,
    fixed_config=None,
    fixed_estimator="lgbm",
):
    """Executes FLAML AutoML under time, iteration, or fixed-configuration mode."""
    time_budget_set = time_budget is not None and time_budget != -1
    max_iter_set = max_iter is not None
    fixed_config_set = fixed_config is not None

    if fixed_config_set:
        if time_budget_set or max_iter_set:
            raise ValueError(
                "fixed_config is mutually exclusive with time_budget and max_iter. "
                "Pass time_budget=-1 and omit max_iter when fixed_config is supplied."
            )
        budget_label = f"Fixed configuration | estimator: {fixed_estimator}"
        fit_time_budget = -1
    else:
        if time_budget_set and max_iter_set:
            raise ValueError(
                "Supply either time_budget or max_iter, not both. Use "
                "time_budget=-1 together with max_iter for an iteration-bounded search."
            )
        if not time_budget_set and not max_iter_set:
            raise ValueError(
                "A search budget is required: pass time_budget in seconds, "
                "time_budget=-1 with max_iter for an iteration-bounded search, "
                "or fixed_config to evaluate a single configuration."
            )
        if time_budget_set and time_budget <= 0:
            raise ValueError(
                "time_budget must be a positive number of seconds, or -1 with max_iter."
            )
        if max_iter_set and max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        if max_iter_set:
            budget_label = f"Iteration-bounded | max_iter: {max_iter}"
            fit_time_budget = -1
        else:
            budget_label = f"Time-bounded | budget: {time_budget}s"
            fit_time_budget = time_budget

    print(f"\n--- Starting AutoML ({task.upper()}) | {budget_label} ---")
    print(f"Training metric: {optimize_metric if optimize_metric else 'FLAML default'}")
        
    cv_strata = _make_outcome_year_strata(y_train, year_train)
    cv_splitter = OutcomeYearStratifiedKFold(
        cv_strata,
        n_splits=n_splits_cv,
        random_state=seed,
    )

    automl = AutoML()
    fit_kwargs = dict(
        X_train=X_train,
        y_train=y_train,
        sample_weight=sample_weight,
        time_budget=fit_time_budget,
        metric=optimize_metric,
        task=task,
        eval_method="cv",
        n_splits=n_splits_cv,
        split_type=cv_splitter,
        seed=seed,
        verbose=1,
        log_training_metric=True,
    )
    if fixed_config_set:
        # FLAML 2.3.6 records but does not score a single dict starting point
        # when max_iter=1, leaving best_loss=inf. Supplying the same starting
        # point twice with max_iter=2 evaluates that fixed configuration while
        # preventing exploration of any different hyperparameter set.
        fit_kwargs["starting_points"] = {fixed_estimator: [fixed_config, fixed_config]}
        fit_kwargs["max_iter"] = 2
        fit_kwargs["estimator_list"] = [fixed_estimator]
    elif max_iter_set:
        fit_kwargs["max_iter"] = max_iter

    start_time = time.time()
    automl.fit(**fit_kwargs)
    elapsed = time.time() - start_time
    n_iter_done = getattr(automl, "_track_iter", None)
    if n_iter_done is not None:
        n_iter_done += 1

    automl.search_iterations_ = n_iter_done
    automl.search_elapsed_time_ = elapsed
    automl.search_mode_ = (
        "fixed_config" if fixed_config_set
        else "max_iter" if max_iter_set
        else "time_budget"
    )
    automl.search_budget_ = (
        "fixed" if fixed_config_set
        else max_iter if max_iter_set
        else time_budget
    )

    print("\n--- Training completed ---")
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best CV score:  {1 - automl.best_loss:.4f}")
    print(f"Elapsed time:   {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Total FLAML iterations: {n_iter_done}")
    if fixed_config_set:
        print("Search mode:    fixed configuration (no search; fully reproducible)")
        selected = getattr(automl, "best_config", None)
        if selected is not None:
            mismatches = {
                key: (value, selected.get(key))
                for key, value in fixed_config.items()
                if key in selected and selected[key] != value
            }
            if mismatches:
                print("WARNING: the configuration used differs from the one supplied:")
                for key, (requested, actual) in mismatches.items():
                    print(f"  {key}: requested {requested}, used {actual}")
            else:
                print("Configuration verified: hyperparameters match those supplied.")
    elif max_iter_set:
        print("Search mode:    iteration-bounded (fixed max_iter and seed)")
    else:
        print("Search mode:    time-bounded (number of configurations depends on runtime)")

    # ---------------------------------------------------------
    # Benchmarking: Best Per Estimator Analysis
    # ---------------------------------------------------------
    try:
        history_list = []
        
        # Method A: Try accessing via best_loss_per_estimator (most stable)
        if hasattr(automl, 'best_loss_per_estimator') and automl.best_loss_per_estimator:
            for learner, loss in automl.best_loss_per_estimator.items():
                if loss < 1.0: # Only include learners that were actually tested
                    history_list.append({
                        'Estimator': learner,
                        'CV_Score': 1 - loss
                    })
        
        # Method B: Fallback to _search_states if Method A provided no results
        if not history_list and hasattr(automl, '_search_states'):
            for learner_id, state in automl._search_states.items():
                if hasattr(state, 'best_loss') and state.best_loss < 1.0:
                    history_list.append({
                        'Estimator': learner_id,
                        'CV_Score': 1 - state.best_loss
                    })

        if history_list:
            benchmark_df = pd.DataFrame(history_list)
            # Sort by best score
            top_models = benchmark_df.sort_values(by='CV_Score', ascending=False).reset_index(drop=True)
            top_models.index += 1 # Start ranking at 1

            print("\n--- Model Benchmarking (Best Results per Estimator Type) ---")
            print(top_models.to_string())
        else:
            print("\n⚠️ No detailed benchmarking history available for this run.")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate benchmarking table: {e}")
    
    return automl

def print_selected_model_hyperparameters(automl):
    """Prints the hyperparameters for the final selected AutoML model."""
    print("\n--- Final selected model hyperparameters ---")
    print(f"Best estimator: {automl.best_estimator}")

    best_config = getattr(automl, "best_config", None)
    if best_config:
        print("FLAML best_config:")
        for param, value in sorted(best_config.items()):
            print(f"  {param}: {value}")

    final_model = getattr(automl, "model", None)
    estimator = getattr(final_model, "estimator", final_model)
    get_params = getattr(estimator, "get_params", None)

    if callable(get_params):
        print("Estimator get_params():")
        for param, value in sorted(get_params(deep=True).items()):
            print(f"  {param}: {value}")
    elif not best_config:
        print("No hyperparameter details available for the selected model.")

def export_final_model_cv_fold_metrics(
    automl,
    X,
    y,
    year,
    n_splits_cv,
    seed,
    task="classification",
    threshold=None,
    filename="cv_fold_metrics.docx",
):
    """Replays CV with the final selected model and exports fold-level metrics."""
    from sklearn.metrics import matthews_corrcoef

    print("\n--- Exporting final-model cross-validation fold metrics ---")

    final_model = getattr(automl, "model", None)
    estimator = getattr(final_model, "estimator", final_model)
    if estimator is None:
        print("No final estimator available; skipping CV fold metrics export.")
        return pd.DataFrame()

    # FLAML normally converts categorical columns to numeric codes internally
    # before fitting the underlying estimator. Cloning and fitting the raw
    # estimator directly here bypasses that step, so categorical columns
    # (e.g. "sex") must be converted ourselves — otherwise XGBoost raises
    # ValueError on category dtype columns (requires enable_categorical=True).
    X = X.copy()
    cat_cols = X.select_dtypes(include="category").columns
    for col in cat_cols:
        X[col] = X[col].cat.codes.astype("int32")

    cv_strata = _make_outcome_year_strata(y, year)
    cv_splitter = OutcomeYearStratifiedKFold(
        cv_strata,
        n_splits=n_splits_cv,
        random_state=seed,
    )

    def _take(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return np.asarray(data)[indices]

    def _safe_metric(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, ValueError):
            return np.nan

    rows = []
    for fold_number, (train_idx, valid_idx) in enumerate(cv_splitter.split(X, y), start=1):
        fold_estimator = clone(estimator)
        X_fold_train = _take(X, train_idx)
        y_fold_train = _take(y, train_idx)
        X_fold_valid = _take(X, valid_idx)
        y_fold_valid = _take(y, valid_idx)

        fold_sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_fold_train,
        )

        try:
            fold_estimator.fit(X_fold_train, y_fold_train, sample_weight=fold_sample_weight)
        except TypeError:
            fold_estimator.fit(X_fold_train, y_fold_train)

        y_valid_pred = fold_estimator.predict(X_fold_valid)
        y_valid_prob = (
            fold_estimator.predict_proba(X_fold_valid)
            if hasattr(fold_estimator, "predict_proba") else None
        )

        row = {
            "fold": fold_number,
            "estimator": automl.best_estimator,
            "n_train": len(train_idx),
            "n_validation": len(valid_idx),
            "validation_prevalence": float(np.mean(y_fold_valid)),
        }

        if task == "multiclass":
            row.update({
                "accuracy": accuracy_score(y_fold_valid, y_valid_pred),
                "macro_f1": f1_score(y_fold_valid, y_valid_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_fold_valid, y_valid_pred, average="weighted", zero_division=0),
            })
            if y_valid_prob is not None:
                row["log_loss"] = _safe_metric(log_loss, y_fold_valid, y_valid_prob)
                row["roc_auc_ovr_weighted"] = _safe_metric(
                    roc_auc_score,
                    y_fold_valid,
                    y_valid_prob,
                    multi_class="ovr",
                    average="weighted",
                )
        else:
            if threshold is None:
                raise ValueError("threshold must be provided for binary CV fold metrics.")
            y_valid_score = y_valid_prob[:, 1] if y_valid_prob is not None else y_valid_pred
            y_valid_pred_threshold = (y_valid_score >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_fold_valid,
                y_valid_pred_threshold,
                labels=[0, 1],
            ).ravel()

            precision = precision_score(y_fold_valid, y_valid_pred_threshold, zero_division=0)
            recall = recall_score(y_fold_valid, y_valid_pred_threshold, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

            row.update({
                "threshold": threshold,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": accuracy_score(y_fold_valid, y_valid_pred_threshold),
                "roc_auc": _safe_metric(roc_auc_score, y_fold_valid, y_valid_score),
                "pr_auc": _safe_metric(average_precision_score, y_fold_valid, y_valid_score),
                "log_loss": _safe_metric(log_loss, y_fold_valid, y_valid_prob),
                "brier": _safe_metric(brier_score_loss, y_fold_valid, y_valid_score),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "npv": npv,
                "f1": f1_score(y_fold_valid, y_valid_pred_threshold, zero_division=0),
                "f2": fbeta_score(y_fold_valid, y_valid_pred_threshold, beta=2, zero_division=0),
                "mcc": matthews_corrcoef(y_fold_valid, y_valid_pred_threshold),
                "youden_index": recall + specificity - 1 if not np.isnan(specificity) else np.nan,
                "balanced_accuracy": (recall + specificity) / 2 if not np.isnan(specificity) else np.nan,
                "overtriage": 1 - precision,
                "undertriage": 1 - npv if not np.isnan(npv) else np.nan,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
            })

        rows.append(row)

    cv_metrics_df = pd.DataFrame(rows)
    numeric_cols = cv_metrics_df.select_dtypes(include=[np.number]).columns.drop("fold", errors="ignore")
    summary_df = pd.DataFrame([
        {"fold": "mean", **cv_metrics_df[numeric_cols].mean(numeric_only=True).to_dict()},
        {"fold": "std", **cv_metrics_df[numeric_cols].std(numeric_only=True).to_dict()},
    ])
    cv_metrics_df = pd.concat([cv_metrics_df, summary_df], ignore_index=True, sort=False)

    docx_metrics = [
        col for col in cv_metrics_df.columns
        if col not in {"fold", "estimator"}
    ]
    docx_rows = []
    for metric in docx_metrics:
        row = {"metric": metric}
        for _, values in cv_metrics_df.iterrows():
            fold_label = str(values["fold"])
            col_name = f"fold_{fold_label}" if fold_label.isdigit() else fold_label
            row[col_name] = values[metric]
        docx_rows.append(row)
    cv_metrics_docx_df = pd.DataFrame(docx_rows)

    save_table(
        cv_metrics_docx_df,
        filename,
        title="Final selected model cross-validation fold metrics",
        footnote=(
            "Metrics are computed by replaying cross-validation on the training set "
            "with the final selected estimator and the same outcome-year splitter. "
            f"Threshold-dependent binary metrics use threshold = {threshold:.4f}. "
            "The Word table is transposed for readability: rows are metrics and "
            "columns are validation folds plus mean and standard deviation."
        ),
    )
    print(cv_metrics_df.to_string(index=False))
    return cv_metrics_df


def compute_oof_probabilities(
    automl,
    X_train,
    y_train,
    year_train,
    n_splits_cv,
    seed,
):
    """Computes out-of-fold positive-class probabilities on the training set."""
    print("\n--- Computing out-of-fold probabilities for threshold selection ---")

    final_model = getattr(automl, "model", None)
    estimator = getattr(final_model, "estimator", final_model)
    if estimator is None:
        raise RuntimeError(
            "No final estimator available; cannot compute out-of-fold probabilities."
        )

    # FLAML encodes categorical columns before fitting underlying estimators.
    # Cloning the raw estimator bypasses that preprocessing, so replay it here.
    X = X_train.copy()
    cat_cols = X.select_dtypes(include="category").columns
    for col in cat_cols:
        X[col] = X[col].cat.codes.astype("int32")

    cv_strata = _make_outcome_year_strata(y_train, year_train)
    cv_splitter = OutcomeYearStratifiedKFold(
        cv_strata,
        n_splits=n_splits_cv,
        random_state=seed,
    )

    def _take(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return np.asarray(data)[indices]

    y_array = np.asarray(y_train)
    oof_prob = np.full(len(X), np.nan, dtype=float)

    for fold_number, (train_idx, valid_idx) in enumerate(cv_splitter.split(X, y_train), start=1):
        fold_estimator = clone(estimator)
        X_fold_train = _take(X, train_idx)
        y_fold_train = y_array[train_idx]
        X_fold_valid = _take(X, valid_idx)

        fold_sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_fold_train,
        )

        try:
            fold_estimator.fit(X_fold_train, y_fold_train, sample_weight=fold_sample_weight)
        except TypeError:
            fold_estimator.fit(X_fold_train, y_fold_train)

        if not hasattr(fold_estimator, "predict_proba"):
            raise RuntimeError(
                f"Estimator '{type(fold_estimator).__name__}' has no predict_proba method."
            )

        oof_prob[valid_idx] = fold_estimator.predict_proba(X_fold_valid)[:, 1]
        print(f"  Fold {fold_number}/{n_splits_cv}: {len(valid_idx):,} validation samples scored")

    if np.isnan(oof_prob).any():
        n_missing = int(np.isnan(oof_prob).sum())
        raise RuntimeError(
            f"{n_missing} training samples received no out-of-fold prediction; "
            "check the cross-validation splitter."
        )

    print(f"Out-of-fold probabilities computed for {len(oof_prob):,} training samples.")
    return oof_prob


def set_fb_threshold(y_train, y_train_prob, optimize_beta):
    """Finds the optimal decision threshold based on F-beta score"""
    print(f"\n--- Optimizing decision threshold (Beta={optimize_beta}) ---")
    thresholds = np.arange(0.05, 0.95, 0.01)
    fbeta_scores = [
        fbeta_score(y_train, (y_train_prob >= t).astype(int), beta=optimize_beta)
        for t in thresholds
    ]
    best_threshold = thresholds[np.argmax(fbeta_scores)]
    print(f"Optimal threshold found: {best_threshold:.3f}")
    
    return best_threshold

def metrics_at_threshold(y_true, y_prob, threshold):
    """Returns binary classification metrics for a given decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = 1.0 - specificity
    youden_j = (sensitivity + specificity - 1.0
                if not (np.isnan(sensitivity) or np.isnan(specificity))
                else np.nan)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "youden_j": youden_j,
    }


def compute_youden_threshold(
    y_train,
    y_train_prob,
    y_test,
    y_test_prob,
    sensitivity_range: Optional[tuple] = (0.80, 0.90),
):
    """
    Finds the optimal decision threshold by maximising the Youden index J = Se + Sp - 1
    within a target sensitivity range on TRAIN.

    Geometrically, J is the maximum vertical distance between the ROC curve and
    the no-discrimination diagonal. The threshold that maximises J provides the
    best symmetric trade-off between sensitivity and specificity.

    Note: Youden assumes symmetric misclassification costs. In emergency triage it
    serves as an exploratory starting point before shifting the threshold toward
    higher sensitivity by clinical imperative.

    Optimisation is performed on TRAIN to avoid threshold over-fitting on TEST.
    The ROC plot shows the TEST curve with the selected threshold evaluated on TEST.

    Parameters
    ----------
    y_train, y_train_prob : train labels and predicted probabilities
    y_test,  y_test_prob  : test labels and predicted probabilities
    sensitivity_range : tuple of float or None
        If provided, restricts the search to TRAIN ROC points whose sensitivity
        falls within [se_min, se_max] (values in [0, 1], e.g. (0.70, 0.90)).
        Within that band, the point that maximises J is selected.
        If no ROC point falls in the range, falls back to the global Youden optimum
        with a warning.

    Returns
    -------
    float
        Optimal threshold (maximises Youden J on TRAIN, optionally within
        sensitivity_range).
    """
    print("\n--- Optimising decision threshold (Youden Index) ---")

    # Compute ROC on TRAIN — threshold selection must not touch TEST.
    # drop_intermediate=False keeps every threshold so no operating point is skipped.
    fpr_train, tpr_train, thresholds_train = roc_curve(
        y_train, y_train_prob, drop_intermediate=False
    )

    # J = Se + Sp - 1  ≡  tpr - fpr
    youden_j_train = tpr_train - fpr_train

    if sensitivity_range is not None:
        se_min, se_max = sensitivity_range
        print(f"  Sensitivity range : [{se_min:.2%}, {se_max:.2%}]")
        mask = (tpr_train >= se_min) & (tpr_train <= se_max)
        if mask.any():
            candidates = np.where(mask)[0]
            best_idx = int(candidates[np.argmax(youden_j_train[candidates])])
        else:
            print(f"  WARNING: no TRAIN ROC point found in sensitivity range "
                  f"[{se_min:.2%}, {se_max:.2%}]. Falling back to global Youden optimum.")
            best_idx = int(np.argmax(youden_j_train))
    else:
        best_idx = int(np.argmax(youden_j_train))

    best_thr = float(thresholds_train[best_idx])

    # Evaluate the selected threshold on TRAIN and TEST
    m_train = metrics_at_threshold(y_train, y_train_prob, best_thr)
    m_test  = metrics_at_threshold(y_test,  y_test_prob,  best_thr)

    print(f"  Threshold selected on TRAIN : {best_thr:.4f}")
    print(f"  TRAIN metrics at threshold  : "
          f"Se = {m_train['sensitivity']:.4f}  "
          f"Sp = {m_train['specificity']:.4f}  "
          f"J  = {m_train['youden_j']:.4f}")
    print(f"  TEST  metrics at threshold  : "
          f"Se = {m_test['sensitivity']:.4f}  "
          f"Sp = {m_test['specificity']:.4f}  "
          f"J  = {m_test['youden_j']:.4f}")

    # --- ROC plot on TEST; operating point evaluated on TEST ---
    fpr_test_curve, tpr_test_curve, _ = roc_curve(y_test, y_test_prob)
    auc_test = roc_auc_score(y_test, y_test_prob)

    test_fpr = m_test["fpr"]
    test_se  = m_test["sensitivity"]
    test_sp  = m_test["specificity"]
    test_j   = m_test["youden_j"]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr_test_curve, tpr_test_curve, color='steelblue', lw=2,
            label=f'ROC curve — test  (AUC = {auc_test:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='No discrimination')

    # Target sensitivity range (for visual reference in TEST space)
    if sensitivity_range is not None:
        ax.axhspan(se_min, se_max, color='gold', alpha=0.15,
                   label=f'Target Se range [{se_min:.0%}, {se_max:.0%}]')

    # Selected threshold evaluated on TEST
    ax.scatter(test_fpr, test_se, color='crimson', zorder=5, s=120,
               label=(f'Threshold = {best_thr:.4f}  (selected on TRAIN)\n'
                      f'J$_{{test}}$ = {test_j:.3f}   '
                      f'Se$_{{test}}$ = {test_se:.3f}   '
                      f'Sp$_{{test}}$ = {test_sp:.3f}'))

    # Vertical segment from the diagonal (y = FPR_test) to the operating point:
    # length equals J_test, the Youden index evaluated on TEST
    ax.vlines(x=test_fpr,
              ymin=test_fpr, ymax=test_se,
              colors='crimson', linestyles='dashed', lw=1.5, alpha=0.7,
              label=f'J$_{{test}}$ = {test_j:.3f}  (vertical distance to diagonal)')

    ax.set_xlabel('1 − Specificity  (FPR)', fontsize=11)
    ax.set_ylabel('Sensitivity  (TPR)', fontsize=11)
    title = 'ROC Curve — Youden Index Threshold\n(selected on TRAIN, operating point evaluated on TEST)'
    if sensitivity_range is not None:
        title += f'\nTarget Se range [{se_min:.0%}, {se_max:.0%}]'
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    plt.tight_layout()
    save_figure(fig, 'roc_curve_youden.png')
    plt.show()

    return best_thr

def optimize_clinical_threshold(
    y_train,
    y_train_prob,
    y_test,
    y_test_prob,
    max_undertriage: float = 0.10,
    max_overtriage: float = 0.50,
) -> float:
    """
    Finds the optimal decision threshold using a clinically-grounded strategy.

    Strategy:
        Maximize F1 subject to two simultaneous constraints:
            · Undertriage < max_undertriage  → FN / (FN + TP)
            · Overtriage  < max_overtriage   → FP / (FP + TN)
        Search is performed on TRAIN to avoid optimistic threshold selection.
        Fallback to undertriage-only constraint if no threshold meets both.

    Parameters
    ----------
    y_train : array-like
        True labels for the training set.
    y_train_prob : array-like
        Predicted probabilities for the positive class (train set).
    y_test : array-like
        True labels for the test set (used only for visualization).
    y_test_prob : array-like
        Predicted probabilities for the positive class (test set).
    max_undertriage : float
        Maximum allowed undertriage rate (default 10%).
    max_overtriage : float
        Maximum allowed overtriage rate (default 50%).

    Returns
    -------
    float
        Optimal clinical threshold.
    """
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, roc_auc_score,
        average_precision_score, accuracy_score
    )

    print("\n--- Optimizing clinical threshold ---")
    print(f"    Constraints: undertriage < {max_undertriage:.0%} | overtriage < {max_overtriage:.0%}")

    # Fine-grained threshold grid on TRAIN — 1000 points for uniform resolution
    thresholds_grid = np.linspace(0.01, 0.99, 1000)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    undertriage_arr = np.array([
        ((y_train_prob < thr) & (y_train == 1)).sum() / n_pos if n_pos > 0 else 0
        for thr in thresholds_grid
    ])
    overtriage_arr = np.array([
        ((y_train_prob >= thr) & (y_train == 0)).sum() / n_neg if n_neg > 0 else 0
        for thr in thresholds_grid
    ])

    f1_arr = np.array([
        f1_score(y_train, (y_train_prob >= thr).astype(int), zero_division=0)
        for thr in thresholds_grid
    ])
    f2_arr = np.array([
        fbeta_score(y_train, (y_train_prob >= thr).astype(int), beta=2, zero_division=0)
        for thr in thresholds_grid
    ])
    accuracy_arr = np.array([
        accuracy_score(y_train, (y_train_prob >= thr).astype(int))
        for thr in thresholds_grid
    ])
    recall_arr = np.array([
        recall_score(y_train, (y_train_prob >= thr).astype(int), zero_division=0)
        for thr in thresholds_grid
    ])

    # Maximize F1 within the safe zone (both constraints met simultaneously)
    valid_mask    = (undertriage_arr < max_undertriage) & (overtriage_arr < max_overtriage)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        best_idx           = valid_indices[np.argmax(f1_arr[valid_indices])]
        clinical_threshold = float(thresholds_grid[best_idx])
        print(f"\n  ✅ Optimal threshold : {clinical_threshold:.4f}")
        print(f"     F1               : {f1_arr[best_idx]:.4f}")
        print(f"     F2               : {f2_arr[best_idx]:.4f}")
        print(f"     Accuracy         : {accuracy_arr[best_idx]:.4f}")
        print(f"     Recall           : {recall_arr[best_idx]:.4f}")
        print(f"     Undertriage      : {undertriage_arr[best_idx]:.2%}  (limit: {max_undertriage:.0%})")
        print(f"     Overtriage       : {overtriage_arr[best_idx]:.2%}  (limit: {max_overtriage:.0%})")
    else:
        # Fallback: undertriage constraint only (clinically more critical)
        print("\n  ⚠️  No threshold meets both constraints simultaneously.")
        print("      → Fallback: undertriage constraint only")
        under_only = np.where(undertriage_arr < max_undertriage)[0]

        if len(under_only) > 0:
            best_idx           = under_only[np.argmax(f1_arr[under_only])]
            clinical_threshold = float(thresholds_grid[best_idx])
            print(f"      → Fallback threshold : {clinical_threshold:.4f}")
            print(f"         Resulting overtriage: {overtriage_arr[best_idx]:.2%} (exceeds {max_overtriage:.0%})")
        else:
            best_idx           = int(np.argmin(undertriage_arr))
            clinical_threshold = float(thresholds_grid[best_idx])
            print(f"      → Minimum undertriage threshold: {clinical_threshold:.4f}")

    # -------------------------------------------------------------------------
    # Visualization — 2x2 layout (threshold plot + ROC top, PR bottom center)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))

    # Top-left: threshold optimization curves
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(thresholds_grid, f1_arr,          'g-',  label='F1-Score',             linewidth=2)
    ax1.plot(thresholds_grid, f2_arr,          color='darkgreen', linestyle='-.', label='F2-Score', linewidth=1.5, alpha=0.8)
    ax1.plot(thresholds_grid, accuracy_arr,    'c-',  label='Accuracy',             linewidth=1.5, alpha=0.8)
    ax1.plot(thresholds_grid, recall_arr,      'b--', label='Recall (Sensitivity)', linewidth=1.5, alpha=0.8)
    ax1.plot(thresholds_grid, undertriage_arr, 'r-',  label='Undertriage (FN/P)',   linewidth=2)
    ax1.plot(thresholds_grid, overtriage_arr,  'm-',  label='Overtriage (FP/N)',    linewidth=2)

    ax1.axvline(clinical_threshold, color='black', linestyle='--', linewidth=1.5,
                label=f'Optimal threshold = {clinical_threshold:.4f}')
    ax1.axhline(max_undertriage, color='red',     linestyle=':', alpha=0.6,
                label=f'Undertriage limit = {max_undertriage:.0%}')
    ax1.axhline(max_overtriage,  color='magenta', linestyle=':', alpha=0.6,
                label=f'Overtriage limit = {max_overtriage:.0%}')

    if len(valid_indices) > 0:
        thr_lo = thresholds_grid[valid_indices[0]]
        thr_hi = thresholds_grid[valid_indices[-1]]
        ax1.axvspan(thr_lo, thr_hi, alpha=0.08, color='green',
                    label='Safe zone (both constraints met)')

    ax1.set_title('Threshold Optimization\n'
                  f'(max F1 | undertriage < {max_undertriage:.0%} & overtriage < {max_overtriage:.0%})')
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Rate / Metric')
    ax1.legend(fontsize=7.5, loc='center right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    ax1.grid(alpha=0.3)

    # Top-right: ROC Curve (TEST)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_test_prob)
    roc_auc_val = roc_auc_score(y_test, y_test_prob)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(fpr_curve, tpr_curve, 'navy', linewidth=2,
             label=f'ROC-AUC = {roc_auc_val:.4f}')
    ax2.plot([0, 1], [0, 1], 'gray', linestyle='--', label='Random (0.5)')

    test_under = ((y_test_prob < clinical_threshold) & (y_test == 1)).sum() / y_test.sum()
    test_spec  = ((y_test_prob < clinical_threshold) & (y_test == 0)).sum() / (y_test == 0).sum()
    ax2.scatter([1 - test_spec], [1 - test_under], color='black', zorder=5, s=80,
                label=f'Threshold = {clinical_threshold:.4f}')

    ax2.set_title('ROC Curve (Test Set)')
    ax2.set_xlabel('1 - Specificity (FPR)')
    ax2.set_ylabel('Sensitivity (TPR)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Bottom-center: PR Curve (TEST) — spans both bottom columns
    ax3 = fig.add_subplot(2, 2, 3)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc_val = average_precision_score(y_test, y_test_prob)
    baseline   = float(y_test.mean())

    ax3.plot(rec_curve, prec_curve, 'darkorange', linewidth=2,
             label=f'PR-AUC = {pr_auc_val:.4f}')
    ax3.axhline(baseline, color='gray', linestyle='--',
                label=f'Baseline (random) = {baseline:.2f}')

    test_pred_opt = (y_test_prob >= clinical_threshold).astype(int)
    opt_prec = precision_score(y_test, test_pred_opt, zero_division=0)
    opt_rec  = recall_score(y_test, test_pred_opt, zero_division=0)
    ax3.scatter([opt_rec], [opt_prec], color='black', zorder=5, s=80,
                label=f'Threshold = {clinical_threshold:.4f}  |  P={opt_prec:.3f}  R={opt_rec:.3f}')

    ax3.set_title('Precision-Recall Curve (Test Set)')
    ax3.set_xlabel('Recall (Sensitivity)')
    ax3.set_ylabel('Precision (PPV)')
    ax3.legend(fontsize=9)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.05])
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return clinical_threshold

def _fit_calibration_logreg(y, p, eps=1e-6, max_iter=100, tol=1e-10):
    """
    Cox calibration regression: fits y ~ logit(p) by unregularized logistic
    regression (Newton-Raphson / IRLS) and returns the calibration intercept
    (calibration-in-the-large, ideal=0) and slope (ideal=1), each with a 95%
    Wald CI derived from the observed Fisher information at convergence.

    Implemented manually (no statsmodels dependency) since this only needs a
    2-parameter (intercept + slope) unregularized logistic fit.
    """
    p_clipped = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped))
    X = np.column_stack([np.ones_like(logit_p), logit_p])
    y = np.asarray(y, dtype=float)

    beta = np.zeros(2)
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        xtwx = X.T @ (X * w[:, None])
        grad = X.T @ (y - mu)
        try:
            delta = np.linalg.solve(xtwx, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(xtwx, grad, rcond=None)[0]
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break

    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
    w = np.clip(mu * (1 - mu), 1e-12, None)
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(cov))

    z = 1.959963984540054  # 97.5th percentile of N(0, 1)
    intercept, slope = beta
    se_intercept, se_slope = se
    return {
        "intercept": intercept,
        "intercept_ci_lo": intercept - z * se_intercept,
        "intercept_ci_hi": intercept + z * se_intercept,
        "slope": slope,
        "slope_ci_lo": slope - z * se_slope,
        "slope_ci_hi": slope + z * se_slope,
    }

def plot_calibration(
    y_train, y_train_prob,
    y_test,  y_test_prob,
    threshold: float,
    n_bins: int = 15,
) -> None:
    """
    Two-panel calibration figure:
      Left  — predicted probability distributions by class (train vs test).
      Right — reliability diagram (calibration curve) for train and test.

    Parameters
    ----------
    n_bins : number of bins for the calibration curve (quantile strategy).
    """
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: probability distributions by class ---
    ax = axes[0]
    ax.hist(y_train_prob[y_train == 0], bins=80, alpha=0.5, color='steelblue',
            density=True, label='Negative (train)')
    ax.hist(y_train_prob[y_train == 1], bins=80, alpha=0.5, color='red',
            density=True, label='Positive (train)')
    ax.hist(y_test_prob[y_test == 0],   bins=80, alpha=0.3, color='navy',
            density=True, label='Negative (test)', linestyle='--')
    ax.hist(y_test_prob[y_test == 1],   bins=80, alpha=0.3, color='darkred',
            density=True, label='Positive (test)', linestyle='--')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold:.4f}')
    ax.set_title('Predicted probability distribution\n(train vs test)')
    ax.set_xlabel('Predicted probability P(=1)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Right: reliability diagram ---
    ax2 = axes[1]
    for probs, labels, name, color in [
        (y_train_prob, y_train, 'Train', 'steelblue'),
        (y_test_prob,  y_test,  'Test',  'red'),
    ]:
        fraction_pos, mean_pred = calibration_curve(labels, probs,
                                                    n_bins=n_bins,
                                                    strategy='quantile')
        ax2.plot(mean_pred, fraction_pos, 's-', label=name,
                 color=color, linewidth=2)
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--',
             label='Perfect calibration')
    ax2.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                label=f'Threshold = {threshold:.4f}')
    ax2.set_title('Calibration curve\n(are predicted probabilities reliable?)')
    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Fraction of positives')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)

    # --- Quantitative calibration metrics: Brier score + Cox calibration ---
    # regression (intercept = calibration-in-the-large, ideal 0; slope =
    # calibration slope, ideal 1), fit on logit(predicted probability).
    brier_train = brier_score_loss(y_train, y_train_prob)
    brier_test = brier_score_loss(y_test, y_test_prob)
    calib_train = _fit_calibration_logreg(y_train, y_train_prob)
    calib_test = _fit_calibration_logreg(y_test, y_test_prob)

    ax2.annotate(
        f"Test set (N={len(y_test)})\n"
        f"Brier = {brier_test:.4f}\n"
        f"Intercept = {calib_test['intercept']:.3f} "
        f"[{calib_test['intercept_ci_lo']:.3f}, {calib_test['intercept_ci_hi']:.3f}]\n"
        f"Slope = {calib_test['slope']:.3f} "
        f"[{calib_test['slope_ci_lo']:.3f}, {calib_test['slope_ci_hi']:.3f}]",
        xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9),
    )

    plt.tight_layout()
    save_figure(fig, 'calibration_curve.png')
    plt.show()

    overestimation = calib_test['slope'] < 1 and calib_test['intercept'] < 0
    print("\n=== CALIBRATION METRICS (logit-scale Cox regression) ===")
    print(f"Brier score  -> train: {brier_train:.4f}  |  test: {brier_test:.4f}  "
          f"(diff = {brier_test - brier_train:+.4f}; higher test Brier suggests "
          f"calibration overfit)")
    print("-" * 60)
    print(f"Test set (N={len(y_test)}):")
    print(f"  Calibration intercept (calibration-in-the-large, ideal = 0): "
          f"{calib_test['intercept']:.4f}  "
          f"[95% CI {calib_test['intercept_ci_lo']:.4f}, {calib_test['intercept_ci_hi']:.4f}]")
    print(f"  Calibration slope     (ideal = 1):                           "
          f"{calib_test['slope']:.4f}  "
          f"[95% CI {calib_test['slope_ci_lo']:.4f}, {calib_test['slope_ci_hi']:.4f}]")
    print(f"Train set (N={len(y_train)}, reference for overfitting check):")
    print(f"  Calibration intercept: {calib_train['intercept']:.4f}  "
          f"[95% CI {calib_train['intercept_ci_lo']:.4f}, {calib_train['intercept_ci_hi']:.4f}]")
    print(f"  Calibration slope:     {calib_train['slope']:.4f}  "
          f"[95% CI {calib_train['slope_ci_lo']:.4f}, {calib_train['slope_ci_hi']:.4f}]")
    print("-" * 60)
    if overestimation:
        print("Interpretation: test slope < 1 with intercept < 0 -> the model "
              "systematically OVERESTIMATES P(P1) (predicted probabilities are "
              "too extreme/high relative to observed frequencies), consistent "
              "with the reliability curve falling below the diagonal.")
    elif calib_test['slope'] < 1:
        print("Interpretation: test slope < 1 -> predicted probabilities are "
              "too extreme (overconfident) at the tails, but the intercept "
              "does not indicate a clear systematic over/under-estimation.")
    else:
        print("Interpretation: slope >= 1 and/or intercept >= 0 -> no clear "
              "systematic overestimation pattern detected on the test set.")


# ===============================================================
# SA-ROC (Safety-Aware ROC) framework
#
# Reference:
#   Kim, Y.-T. et al. "Defining operational safety in clinical artificial
#   intelligence systems." npj Digital Medicine 9, 281 (2026).
#   https://www.nature.com/articles/s41746-026-02450-7
#   Official code: https://github.com/MGH-LMIC/SA-ROC
#
# Standard accuracy metrics (AUC, etc.) describe how well a model
# *discriminates*, but not *when it is safe to act on its output
# autonomously*. SA-ROC partitions the predicted-probability axis into
# three operational zones, given two clinician-defined reliability targets:
#
#   - Rule-in Safe Zone  : scores high enough that PPV >= alpha_pos -> autonomous escalate
#   - Rule-out Safe Zone : scores low enough that  NPV >= alpha_neg -> autonomous de-prioritize
#   - Gray Zone          : everything in between -> mandatory human review
#
# It also computes the Gray Zone Area (gamma_area), summarizing the model's
# "cost of indecision": how much of the ROC space cannot be automated at the
# requested reliability.
#
# Caveats:
#   - PPV/NPV are prevalence-dependent: the thresholds found here are only
#     valid if the evaluation set's prevalence matches the deployment
#     prevalence, and if the predicted probabilities are reasonably calibrated.
#   - SA-ROC partitions by predicted probability only; it does not capture
#     time-sensitive factors (e.g. a stroke thrombolysis/thrombectomy window),
#     which must be handled separately in the clinical workflow.
#   - Confirm what y_true == 1 means (confirmed diagnosis vs. a triage/operational
#     decision) before interpreting PPV/NPV as diagnostic reliabilities.
# ===============================================================

class SafetyTargetWarning(UserWarning):
    """Emitted when the model cannot satisfy a requested SA-ROC safety target."""
    pass


def _sa_roc_fpr_tpr_at(y_true, y_prob, tau):
    """
    Computes (FPR, TPR) at an exact score threshold `tau` (a sample is
    predicted positive iff score >= tau).

    Computed directly from the labels and scores at the exact tau, rather
    than snapping tau onto sklearn's roc_curve threshold grid, to avoid the
    rounding error that a nearest-threshold lookup introduces in the score
    tails - exactly where high-alpha policies operate.
    """
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    pred_pos = (y_prob >= tau)
    tpr = (pred_pos & (y_true == 1)).sum() / n_pos if n_pos > 0 else 0.0
    fpr = (pred_pos & (y_true == 0)).sum() / n_neg if n_neg > 0 else 0.0
    return fpr, tpr


def evaluate_sa_roc(y_true, y_prob, alpha_pos=0.85, alpha_neg=0.95):
    """
    Applies the SA-ROC framework and partitions predictions into the Rule-in
    Safe, Gray, and Rule-out Safe zones.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Ground-truth binary labels (1 = positive class).
    y_prob : array-like in [0, 1]
        Predicted probability of the positive class.
    alpha_pos : float, default 0.85
        Rule-in safety level (alpha+): minimum acceptable PPV to trust a
        positive (rule-in) action.
    alpha_neg : float, default 0.95
        Rule-out safety level (alpha-): minimum acceptable NPV to trust a
        negative (rule-out) action.

    Returns
    -------
    tau_safe_pos : float
        Lower bound (infimum) of the Rule-in Safe Zone.
    tau_safe_neg : float
        Upper bound (supremum) of the Rule-out Safe Zone.
    pct_gray : float
        Fraction of the cohort that falls in the Gray Zone.
    gamma_area : float
        Gray Zone Area in ROC space (cost of indecision).
    zone_stats : dict
        Per-zone counts, P1/non-P1 breakdown, and achieved NPV (Rule-out) /
        PPV (Rule-in) with 95% Clopper-Pearson CIs. See body for keys.

    Warns
    -----
    SafetyTargetWarning
        If a requested safety target cannot be met, or if the resulting safe
        zones overlap (degenerate model). The corresponding zone is set empty.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Empirical linear scan over every unique observed score (no binning gaps).
    thresholds = np.sort(np.unique(y_prob))

    tau_safe_pos = None  # infimum threshold for the Rule-in Safe Zone
    tau_safe_neg = None  # supremum threshold for the Rule-out Safe Zone

    # Rule-in threshold: smallest t with PPV(score >= t) >= alpha_pos.
    # Low->high scan + first hit = infimum.
    for t in thresholds:
        pred_pos = (y_prob >= t)
        if pred_pos.sum() > 0:
            ppv = (y_true[pred_pos] == 1).sum() / pred_pos.sum()
            if ppv >= alpha_pos:
                tau_safe_pos = t
                break

    # Rule-out threshold: largest t with NPV(score < t) >= alpha_neg.
    # High->low scan + first hit = supremum. Strict "<" matches rule-out
    # (rule-in uses ">=").
    for t in reversed(thresholds):
        pred_neg = (y_prob < t)
        if pred_neg.sum() > 0:
            npv = (y_true[pred_neg] == 0).sum() / pred_neg.sum()
            if npv >= alpha_neg:
                tau_safe_neg = t
                break

    # Warn and fall back to an empty zone if a safety target is unreachable.
    # An empty Safe Zone is a clinical-safety failure, so it must never be
    # silent, but plotting still proceeds.
    if tau_safe_pos is None:
        warnings.warn(
            f"Rule-in target not achievable: no score threshold reaches "
            f"PPV >= {alpha_pos:.0%}. Rule-in Safe Zone set EMPTY.",
            SafetyTargetWarning, stacklevel=2,
        )
        tau_safe_pos = 1.0
    if tau_safe_neg is None:
        warnings.warn(
            f"Rule-out target not achievable: no score threshold reaches "
            f"NPV >= {alpha_neg:.0%}. Rule-out Safe Zone set EMPTY.",
            SafetyTargetWarning, stacklevel=2,
        )
        tau_safe_neg = 0.0

    # Degenerate case: thresholds cross -> zones would overlap. Warn and
    # collapse the overlap by clamping rule-out down to rule-in, so the Gray
    # Zone is empty rather than negative.
    if tau_safe_pos < tau_safe_neg:
        warnings.warn(
            f"Safe Zones overlap (tau_rule_in={tau_safe_pos:.4f} < "
            f"tau_rule_out={tau_safe_neg:.4f}); model too weak to satisfy both "
            f"targets at these alpha levels. Collapsing overlap (empty Gray Zone).",
            SafetyTargetWarning, stacklevel=2,
        )
        tau_safe_neg = tau_safe_pos

    # Assign each sample to a zone and compute cohort distribution.
    # Boundary operators: rule-out is "<", rule-in is ">=".
    idx_rule_out = (y_prob < tau_safe_neg)
    idx_rule_in = (y_prob >= tau_safe_pos)
    idx_gray = ~(idx_rule_out | idx_rule_in)

    n_total = len(y_true)
    n_rule_out = int(idx_rule_out.sum())
    n_rule_in = int(idx_rule_in.sum())
    n_gray = int(idx_gray.sum())
    pct_rule_out = n_rule_out / n_total
    pct_rule_in = n_rule_in / n_total
    pct_gray = n_gray / n_total

    # Gray Zone Area (gamma_area) in ROC space:
    #     gamma_area = FPR(tau_safe_neg) * (1 - TPR(tau_safe_pos))
    fpr_tau_neg, _ = _sa_roc_fpr_tpr_at(y_true, y_prob, tau_safe_neg)
    _, tpr_tau_pos = _sa_roc_fpr_tpr_at(y_true, y_prob, tau_safe_pos)
    gamma_area = fpr_tau_neg * (1 - tpr_tau_pos)

    # Per-zone P1 (positive) vs non-P1 (negative) breakdown.
    n_p1_rule_out = int((y_true[idx_rule_out] == 1).sum())
    n_nonp1_rule_out = n_rule_out - n_p1_rule_out
    n_p1_rule_in = int((y_true[idx_rule_in] == 1).sum())
    n_nonp1_rule_in = n_rule_in - n_p1_rule_in
    n_p1_gray = int((y_true[idx_gray] == 1).sum())
    n_nonp1_gray = n_gray - n_p1_gray

    # Achieved NPV (Rule-out) / PPV (Rule-in) with 95% Clopper-Pearson CIs.
    from scipy.stats import beta as _beta

    def _clopper_pearson(x, n, alpha=0.05):
        if n == 0:
            return (np.nan, np.nan)
        lo = _beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0
        hi = _beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0
        return lo, hi

    npv_rule_out = n_nonp1_rule_out / n_rule_out if n_rule_out > 0 else np.nan
    npv_ci = _clopper_pearson(n_nonp1_rule_out, n_rule_out)
    ppv_rule_in = n_p1_rule_in / n_rule_in if n_rule_in > 0 else np.nan
    ppv_ci = _clopper_pearson(n_p1_rule_in, n_rule_in)

    npv_target_met = (not np.isnan(npv_rule_out)) and (npv_rule_out >= alpha_neg)
    ppv_target_met = (not np.isnan(ppv_rule_in)) and (ppv_rule_in >= alpha_pos)

    zone_stats = {
        "n_total": n_total,
        "rule_out": {
            "n": n_rule_out, "pct": pct_rule_out,
            "n_p1": n_p1_rule_out, "n_nonp1": n_nonp1_rule_out,
            "npv": npv_rule_out, "npv_ci_lo": npv_ci[0], "npv_ci_hi": npv_ci[1],
            "target_met": npv_target_met,
        },
        "gray": {
            "n": n_gray, "pct": pct_gray,
            "n_p1": n_p1_gray, "n_nonp1": n_nonp1_gray,
        },
        "rule_in": {
            "n": n_rule_in, "pct": pct_rule_in,
            "n_p1": n_p1_rule_in, "n_nonp1": n_nonp1_rule_in,
            "ppv": ppv_rule_in, "ppv_ci_lo": ppv_ci[0], "ppv_ci_hi": ppv_ci[1],
            "target_met": ppv_target_met,
        },
    }

    print("=== OPERATIONAL SAFETY REPORT (SA-ROC) ===")
    print(f"Clinical target -> desired PPV (Rule-in): {alpha_pos:.0%} | desired NPV (Rule-out): {alpha_neg:.0%}")
    print("-" * 60)
    print(f"[Rule-out Safe] (NPV >= {alpha_neg:.0%}) : P(positive) <  {tau_safe_neg:.4f} | covers {pct_rule_out:.1%} of cases")
    print(f"    n = {n_rule_out:,} ({pct_rule_out:.1%} of N={n_total:,})  |  P1 = {n_p1_rule_out:,}  |  non-P1 = {n_nonp1_rule_out:,}")
    print(f"    Achieved NPV = {npv_rule_out:.1%}  [95% CI {npv_ci[0]:.1%}-{npv_ci[1]:.1%}]  "
          f"-> target {'MET' if npv_target_met else 'NOT MET'} (>= {alpha_neg:.0%})")
    print(f"[Rule-in Safe]  (PPV >= {alpha_pos:.0%}) : P(positive) >= {tau_safe_pos:.4f} | covers {pct_rule_in:.1%} of cases")
    print(f"    n = {n_rule_in:,} ({pct_rule_in:.1%} of N={n_total:,})  |  P1 = {n_p1_rule_in:,}  |  non-P1 = {n_nonp1_rule_in:,}")
    print(f"    Achieved PPV = {ppv_rule_in:.1%}  [95% CI {ppv_ci[0]:.1%}-{ppv_ci[1]:.1%}]  "
          f"-> target {'MET' if ppv_target_met else 'NOT MET'} (>= {alpha_pos:.0%})")
    print(f"[GRAY ZONE]     (mandatory human review)          | covers {pct_gray:.1%} of cases")
    print(f"    n = {n_gray:,} ({pct_gray:.1%} of N={n_total:,})  |  P1 = {n_p1_gray:,}  |  non-P1 = {n_nonp1_gray:,}")
    print("-" * 60)
    print(f"Gray Zone Area (Gamma_Area): {gamma_area:.4f} (model's cost of indecision)")

    return tau_safe_pos, tau_safe_neg, pct_gray, gamma_area, zone_stats


def plot_sa_roc(y_test, y_test_prob, tau_safe_pos, tau_safe_neg, zone_stats=None):
    """
    Plots the test-set predicted-probability distributions by class, overlaid
    with the Rule-out Safe / Gray / Rule-in Safe zones from SA-ROC.

    If `zone_stats` (as returned by evaluate_sa_roc) is provided, each zone is
    annotated with its N (%) and achieved NPV/PPV with 95% CI.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_test_prob[y_test == 0], bins=50, alpha=0.5, color='steelblue',
            density=True, label='Negative (actual)')
    ax.hist(y_test_prob[y_test == 1], bins=50, alpha=0.5, color='darkred',
            density=True, label='Positive (actual)')

    # Shade the three operational zones.
    ax.axvspan(0, tau_safe_neg, color='blue', alpha=0.1,
               label=f'Rule-out Safe (< {tau_safe_neg:.3f})')
    ax.axvspan(tau_safe_neg, tau_safe_pos, color='gray', alpha=0.2,
               label='Gray Zone (human review)')
    ax.axvspan(tau_safe_pos, 1, color='red', alpha=0.1,
               label=f'Rule-in Safe (>= {tau_safe_pos:.3f})')

    # Mark the two safety thresholds.
    ax.axvline(tau_safe_neg, color='blue', linestyle='--', linewidth=2)
    ax.axvline(tau_safe_pos, color='red', linestyle='--', linewidth=2)

    # Annotate each zone with its N (%) and achieved NPV/PPV [95% CI].
    # Placed at mid-height so they don't overlap the top legend or the
    # histogram peaks (which sit near the baseline).
    if zone_stats is not None:
        y_mid = ax.get_ylim()[1] * 0.5
        ro, gr, ri = zone_stats["rule_out"], zone_stats["gray"], zone_stats["rule_in"]

        ax.annotate(
            f"n={ro['n']:,} ({ro['pct']:.1%})\n"
            f"P1={ro['n_p1']:,} / non-P1={ro['n_nonp1']:,}\n"
            f"NPV={ro['npv']:.1%} [{ro['npv_ci_lo']:.1%}-{ro['npv_ci_hi']:.1%}]",
            xy=(tau_safe_neg / 2, y_mid), ha='center', va='center', fontsize=8.5,
            color='navy', bbox=dict(boxstyle='round', fc='white', ec='blue', alpha=0.85),
        )
        ax.annotate(
            f"n={gr['n']:,} ({gr['pct']:.1%})\n"
            f"P1={gr['n_p1']:,} / non-P1={gr['n_nonp1']:,}",
            xy=((tau_safe_neg + tau_safe_pos) / 2, y_mid), ha='center', va='center', fontsize=8.5,
            color='dimgray', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85),
        )
        ax.annotate(
            f"n={ri['n']:,} ({ri['pct']:.1%})\n"
            f"P1={ri['n_p1']:,} / non-P1={ri['n_nonp1']:,}\n"
            f"PPV={ri['ppv']:.1%} [{ri['ppv_ci_lo']:.1%}-{ri['ppv_ci_hi']:.1%}]",
            xy=((tau_safe_pos + 1) / 2, y_mid), ha='center', va='center', fontsize=8.5,
            color='darkred', bbox=dict(boxstyle='round', fc='white', ec='red', alpha=0.85),
        )

    ax.set_title('SA-ROC framework: triage zones and clinical flow')
    ax.set_xlabel('Predicted probability of positive class')
    ax.set_ylabel('Patient density')
    ax.legend(loc='upper center')
    ax.set_xlim(0, 1)

    save_figure(fig, 'sa_roc_zones.png')
    plt.show()


def _normalized_confusion_matrix(cm, normalize):
    """Returns percentages for a confusion matrix under the requested denominator."""
    valid_normalizers = {"row", "all", "col"}
    if normalize not in valid_normalizers:
        raise ValueError(
            f"normalize must be one of {sorted(valid_normalizers)}; got {normalize!r}."
        )

    cm_float = cm.astype(float)
    if normalize == "row":
        denominator = cm_float.sum(axis=1, keepdims=True)
    elif normalize == "col":
        denominator = cm_float.sum(axis=0, keepdims=True)
    else:
        denominator = cm_float.sum()

    return np.divide(
        cm_float,
        denominator,
        out=np.zeros_like(cm_float, dtype=float),
        where=denominator != 0,
    )


def _confusion_matrix_annotations(cm, normalize="row"):
    cm_norm = _normalized_confusion_matrix(cm, normalize)
    return np.array([
        [f"{cm[i, j]:d}\n({cm_norm[i, j] * 100:.1f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])


def plot_confusion_comparison(
    y_true, y_pred_triage, y_pred_model,
    figures_dir=None, filename='confusion_comparison.png',
    normalize='row',
):
    """Draws side-by-side test-set confusion matrices for triage and ML model."""
    labels = ['Negative', 'Positive']
    cm_triage = confusion_matrix(y_true, y_pred_triage, labels=[0, 1])
    cm_model = confusion_matrix(y_true, y_pred_model, labels=[0, 1])
    annot_triage = _confusion_matrix_annotations(cm_triage, normalize=normalize)
    annot_model = _confusion_matrix_annotations(cm_model, normalize=normalize)
    vmax = max(cm_triage.max(), cm_model.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    sns.heatmap(
        cm_triage,
        annot=annot_triage,
        cmap='Blues',
        fmt='',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=vmax,
        cbar=False,
        annot_kws={'fontsize': 11},
        ax=axes[0],
    )
    sns.heatmap(
        cm_model,
        annot=annot_model,
        cmap='Blues',
        fmt='',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=vmax,
        cbar=True,
        annot_kws={'fontsize': 11},
        ax=axes[1],
    )

    axes[0].set_title(TELEPHONIC_TRIAGE_LABEL)
    axes[1].set_title(ML_MODEL_LABEL)
    axes[0].set_xlabel('Predicted')
    axes[1].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[1].set_ylabel('')
    fig.suptitle('Confusion matrix comparison (test set)')
    plt.tight_layout()
    save_figure(fig, filename, figures_dir=figures_dir)
    plt.show()


def _get_general_metrics_row(df):
    if "Subgroup" in df.columns and (df["Subgroup"] == "General").any():
        return df.loc[df["Subgroup"] == "General"].iloc[0]
    return df.iloc[0]


def _metric_value_with_ci(row, metric):
    value = pd.to_numeric(row.get(f"{metric} (%)"), errors="coerce")
    lo = pd.to_numeric(row.get(f"{metric} CI lo (%)"), errors="coerce")
    hi = pd.to_numeric(row.get(f"{metric} CI hi (%)"), errors="coerce")
    return float(value), float(lo), float(hi)


def _general_row_n(row):
    if "N" not in row.index:
        return np.nan
    return pd.to_numeric(str(row["N"]).replace(",", ""), errors="coerce")


def plot_metrics_comparison(
    triage_overall_df, model_overall_df,
    figures_dir=None, filename='metrics_comparison.png',
    metric_order=('Accuracy', 'Recall', 'Precision', 'Specificity',
                  'F1-Score', 'F2-Score', 'Overtriage', 'Undertriage'),
    error_metrics=('Overtriage', 'Undertriage'),
):
    """Grouped bar chart comparing triage vs ML diagnostic performance with 95% CIs."""
    triage_row = _get_general_metrics_row(triage_overall_df)
    model_row = _get_general_metrics_row(model_overall_df)

    triage_values, triage_lo, triage_hi = [], [], []
    model_values, model_lo, model_hi = [], [], []
    for metric in metric_order:
        value, lo, hi = _metric_value_with_ci(triage_row, metric)
        triage_values.append(value)
        triage_lo.append(lo)
        triage_hi.append(hi)

        value, lo, hi = _metric_value_with_ci(model_row, metric)
        model_values.append(value)
        model_lo.append(lo)
        model_hi.append(hi)

    triage_values = np.array(triage_values, dtype=float)
    model_values = np.array(model_values, dtype=float)
    triage_yerr = np.nan_to_num(
        np.vstack([
            triage_values - np.array(triage_lo, dtype=float),
            np.array(triage_hi, dtype=float) - triage_values,
        ]),
        nan=0.0,
    )
    model_yerr = np.nan_to_num(
        np.vstack([
            model_values - np.array(model_lo, dtype=float),
            np.array(model_hi, dtype=float) - model_values,
        ]),
        nan=0.0,
    )

    x = np.arange(len(metric_order))
    width = 0.36
    blues = plt.get_cmap('Blues')
    triage_color = blues(0.55)
    model_color = blues(0.85)

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, metric in enumerate(metric_order):
        if metric in error_metrics:
            ax.axvspan(idx - 0.5, idx + 0.5, color='gray', alpha=0.08, zorder=0)

    triage_bars = ax.bar(
        x - width / 2,
        triage_values,
        width,
        yerr=triage_yerr,
        capsize=3,
        label=TELEPHONIC_TRIAGE_LABEL,
        color=triage_color,
        edgecolor='white',
        linewidth=0.6,
        zorder=2,
    )
    model_bars = ax.bar(
        x + width / 2,
        model_values,
        width,
        yerr=model_yerr,
        capsize=3,
        label=ML_MODEL_LABEL,
        color=model_color,
        edgecolor='white',
        linewidth=0.6,
        zorder=2,
    )

    for bars in (triage_bars, model_bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    triage_n = _general_row_n(triage_row)
    model_n = _general_row_n(model_row)
    note = "Overtriage and Undertriage: lower is better; all other metrics: higher is better."
    if not pd.isna(triage_n) and not pd.isna(model_n):
        if int(triage_n) != int(model_n):
            raise ValueError(
                "Model-vs-triage metrics must use the same common test subset; "
                f"got triage N={int(triage_n):,} and model N={int(model_n):,}."
            )
        note += f" Common test subset, N = {int(model_n):,}."

    ax.set_title('Diagnostic performance comparison (test set)', fontsize=13)
    ax.set_ylabel('Value (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_order, rotation=30, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.25, zorder=1)
    ax.legend(loc='upper right')
    fig.text(0.01, 0.01, note, ha='left', va='bottom', fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, filename, figures_dir=figures_dir)
    plt.show()


def plot_subgroup_fairness_heatmap(
    age_df,
    sex_df=None,
    figures_dir=None,
    filename='subgroup_fairness_heatmap.png',
    metric_order=('Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score',
                  'Specificity', 'NPV', 'Overtriage', 'Undertriage'),
    lower_is_better=('Overtriage', 'Undertriage'),
    subgroup_order=None,
    sex_subgroup_order=('Men', 'Women'),
    sex_display_labels=None,
    color_mode='signed_ci',
    small_n_threshold=1000,
    color_clip=0.70,
    title='Diagnostic performance by subgroup (test set)',
    age_block_title='Age group',
    sex_block_title='Sex',
    font_family=None,
    value_fontsize=13,
    ci_fontsize=11,
    subgroup_fontsize=11,
    legend_fontsize=11,
):
    """Plot diagnostic performance for age and sex subgroups versus General.

    ``value_fontsize`` and ``ci_fontsize`` control the point estimate and
    confidence-interval annotations inside each heatmap cell.
    ``subgroup_fontsize`` controls the column labels, while
    ``legend_fontsize`` controls the colour-bar labels.
    """
    import matplotlib.font_manager as fm
    from matplotlib.colors import TwoSlopeNorm

    valid_color_modes = {"signed_ci", "signed_pp", "general_green"}
    if color_mode not in valid_color_modes:
        raise ValueError(
            f"color_mode must be one of {sorted(valid_color_modes)}; got {color_mode!r}."
        )

    def _first_row(df, subgroup):
        if df is None or "Subgroup" not in df.columns:
            return None
        rows = df.loc[df["Subgroup"] == subgroup]
        return rows.iloc[0] if not rows.empty else None

    general_row = _first_row(age_df, "General")
    if general_row is None:
        print("[plot_subgroup_fairness_heatmap] WARNING: General row is required; skipping figure.")
        return None

    general_values = {m: _metric_value_with_ci(general_row, m)[0] for m in metric_order}

    def _build_block(df, subgroups, block_name):
        missing = [s for s in subgroups if _first_row(df, s) is None]
        if missing:
            print(f"[plot_subgroup_fairness_heatmap] WARNING: missing {block_name} "
                  f"subgroup row(s): {missing}.")

        shape = (len(metric_order), len(subgroups))
        vals, lo_v, hi_v = (np.full(shape, np.nan) for _ in range(3))
        cols = np.full(shape, np.nan)
        ns = []
        for j, subgroup in enumerate(subgroups):
            row = _first_row(df, subgroup)
            ns.append(_general_row_n(row) if row is not None else np.nan)
            if row is None:
                continue
            for i, metric in enumerate(metric_order):
                value, lo, hi = _metric_value_with_ci(row, metric)
                vals[i, j], lo_v[i, j], hi_v[i, j] = value, lo, hi
                if np.isnan(value):
                    continue
                if subgroup == "General":
                    cols[i, j] = 0.0
                    continue
                delta = value - general_values[metric]
                oriented = -delta if metric in lower_is_better else delta
                if color_mode == "signed_ci":
                    half_width = (hi - lo) / 2
                    scale = max(half_width, 1e-6) if not np.isnan(half_width) else 1e-6
                    cols[i, j] = oriented / scale
                else:
                    cols[i, j] = oriented / 15
        return vals, lo_v, hi_v, np.clip(cols, -color_clip, color_clip), ns

    if subgroup_order is None:
        subgroup_order = [s for s in pd.unique(age_df["Subgroup"]) if s != "General"]
    else:
        subgroup_order = [s for s in subgroup_order if s != "General"]

    blocks = [(None, ["General"], age_df, {})]
    if subgroup_order:
        blocks.append((age_block_title, list(subgroup_order), age_df, {}))
    else:
        print("[plot_subgroup_fairness_heatmap] WARNING: no age subgroup rows found; "
              "plotting the available blocks only.")

    if sex_df is not None:
        sex_present = [s for s in sex_subgroup_order if _first_row(sex_df, s) is not None]
        if not sex_present:
            found = list(pd.unique(sex_df["Subgroup"])) if "Subgroup" in sex_df.columns else []
            print("[plot_subgroup_fairness_heatmap] WARNING: none of the sex subgroups "
                  f"{list(sex_subgroup_order)} found in sex_df (found: {found}); "
                  "plotting without the sex block.")
        else:
            sex_general = _first_row(sex_df, "General")
            if sex_general is not None:
                n_age, n_sex = _general_row_n(general_row), _general_row_n(sex_general)
                if not (pd.isna(n_age) or pd.isna(n_sex)) and int(n_age) != int(n_sex):
                    print(f"[plot_subgroup_fairness_heatmap] WARNING: General N differs "
                          f"(age_df N={int(n_age):,}; sex_df N={int(n_sex):,}).")
            blocks.append((sex_block_title, list(sex_subgroup_order), sex_df,
                           sex_display_labels or {}))

    block_data = [
        (name, subgroups, labels, _build_block(df, subgroups, name or "General"))
        for name, subgroups, df, labels in blocks
    ]

    rc = {}
    if font_family is not None:
        available = {f.name for f in fm.fontManager.ttflist}
        if font_family in available:
            rc["font.family"] = font_family
        else:
            print(f"[plot_subgroup_fairness_heatmap] WARNING: font '{font_family}' not installed; "
                  "using matplotlib default.")

    n_metrics = len(metric_order)
    n_cols_total = sum(len(block[1]) for block in block_data)
    gap = 0.45
    width_ratios = []
    for k, (_, subgroups, _, _) in enumerate(block_data):
        if k > 0:
            width_ratios.append(gap)
        width_ratios.append(len(subgroups))
    width_ratios += [gap, 0.22]

    fig_width = max(9, (n_cols_total + gap * len(block_data)) * 1.85 + 2.5)
    fig_height = max(6, n_metrics * 0.62 + 2.4)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    with plt.rc_context(rc), sns.axes_style("white"):
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(
            1, len(width_ratios), width_ratios=width_ratios,
            left=0.10, right=0.95, bottom=0.03,
            top=0.85 if title else 0.89, wspace=0.0,
        )

        im = None
        for k, (block_name, subgroups, display_labels, data) in enumerate(block_data):
            vals, lo_v, hi_v, cols, ns = data
            ax = fig.add_subplot(gs[0, 2 * k])
            im = ax.imshow(
                np.ma.masked_invalid(cols), cmap=SUBGROUP_HEATMAP_CMAP,
                norm=norm, aspect='auto',
            )
            for i in range(n_metrics):
                for j in range(len(subgroups)):
                    if np.isnan(vals[i, j]):
                        ax.text(
                            j, i, "NA", ha='center', va='center',
                            fontsize=ci_fontsize,
                        )
                        continue
                    ax.text(j, i - 0.13, f"{vals[i, j]:.1f}%", ha='center', va='center',
                            fontsize=value_fontsize,
                            fontweight='bold' if subgroups[j] == "General" else 'normal')
                    ax.text(j, i + 0.20, f"[{lo_v[i, j]:.1f}–{hi_v[i, j]:.1f}]",
                            ha='center', va='center', fontsize=ci_fontsize,
                            color='#333333')

            col_labels = []
            for subgroup, n in zip(subgroups, ns):
                label = display_labels.get(subgroup, subgroup)
                if pd.isna(n):
                    col_labels.append(f"{label}\nN=NA")
                    continue
                small = (small_n_threshold is not None and subgroup != "General"
                         and int(n) < small_n_threshold)
                col_labels.append(f"{label}{'*' if small else ''}\nN={int(n):,}")

            ax.set_xticks(np.arange(len(subgroups)))
            ax.set_xticklabels(
                col_labels,
                fontsize=subgroup_fontsize,
                fontstyle='italic',
            )
            ax.tick_params(axis='x', top=True, bottom=False,
                           labeltop=True, labelbottom=False, length=0, pad=4)
            ax.set_yticks(np.arange(n_metrics))
            ax.set_yticklabels(
                metric_order if k == 0 else [],
                fontsize=10,
                fontweight='bold' if k == 0 else 'normal',
                fontstyle='italic',
            )
            ax.tick_params(axis='y', length=0)
            ax.vlines(np.arange(-0.5, len(subgroups) + 0.5, 1), -0.5, n_metrics - 0.5,
                      colors='white', linewidth=2, clip_on=False)
            ax.hlines(np.arange(-0.5, n_metrics + 0.5, 1), -0.5, len(subgroups) - 0.5,
                      colors='white', linewidth=2, clip_on=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            if block_name:
                ax.set_title(
                    block_name,
                    fontsize=11,
                    fontweight='bold',
                    fontstyle='italic',
                    pad=6,
                )

        cax = fig.add_subplot(gs[0, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Worse', 'Same', 'Better'])
        cbar.ax.tick_params(length=0, labelsize=legend_fontsize)
        cbar.set_label(
            'Deviation vs General (oriented)', fontsize=legend_fontsize
        )
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.965)
        save_figure(fig, filename, figures_dir=figures_dir)
        plt.show()
    return fig


def combine_exported_roc_pr_curves(
    roc_filename='roc_curve_youden.png',
    pr_filename='pr_curve.png',
    output_filename='roc_pr_curves.png',
    figures_dir=None,
    padding=24,
):
    """Combines the already-exported ROC and PR PNGs without redrawing them."""
    try:
        from PIL import Image
    except ImportError:
        print("[combine_exported_roc_pr_curves] WARNING: Pillow not available; skipping combined figure.")
        return None

    roc_path = _resolve_figure_path(roc_filename, figures_dir=figures_dir)
    pr_path = _resolve_figure_path(pr_filename, figures_dir=figures_dir)
    output_path = _resolve_figure_path(output_filename, figures_dir=figures_dir)
    if roc_path is None or pr_path is None or output_path is None:
        print(
            "[combine_exported_roc_pr_curves] WARNING: no figures directory; "
            f"skipping '{output_filename}'."
        )
        return None
    missing = [path for path in (roc_path, pr_path) if not os.path.exists(path)]
    if missing:
        print(
            "[combine_exported_roc_pr_curves] WARNING: missing source figure(s); "
            f"skipping '{output_filename}': {missing}"
        )
        return None

    with Image.open(roc_path) as roc_img, Image.open(pr_path) as pr_img:
        roc_img = roc_img.convert("RGBA")
        pr_img = pr_img.convert("RGBA")
        width = roc_img.width + padding + pr_img.width
        height = max(roc_img.height, pr_img.height)
        combined = Image.new("RGBA", (width, height), "white")
        roc_y = (height - roc_img.height) // 2
        pr_y = (height - pr_img.height) // 2
        combined.paste(roc_img, (0, roc_y), roc_img)
        combined.paste(pr_img, (roc_img.width + padding, pr_y), pr_img)
        combined.convert("RGB").save(output_path)

    print(f"Figure saved: {output_path}")
    return output_path


def print_test_metrics_with_ci(
    y_test,
    y_test_prob,
    y_test_year,
    threshold: float,
    n_bootstrap: int = 1000,
    ci_seed:     int = 42,
) -> dict:
    """
    Plots the confusion matrix, computes all test-set metrics, prints them
    with 95% confidence intervals, and returns the metrics dict.

    Clopper–Pearson (exact binomial) for binary proportions:
        Accuracy, Precision, Recall, Specificity, NPV

    Outcome-year stratified bootstrap for discrimination and composite metrics:
        ROC-AUC, PR-AUC, F1, F2, MCC, Youden Index, Balanced Accuracy
    """
    from scipy.stats import beta as _beta
    from sklearn.metrics import matthews_corrcoef

    y_test_pred = (y_test_prob >= threshold).astype(int)

    # --- Confusion matrix plot -----------------------------------------------
    conf_matrix = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    labels = ['Negative', 'Positive']
    annot = _confusion_matrix_annotations(conf_matrix, normalize="row")

    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=annot, cmap='Blues', fmt='',
                xticklabels=labels, yticklabels=labels,
                cbar=True, annot_kws={'fontsize': 11})
    plt.title('Classification Pipeline Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    save_figure(fig_cm, 'confusion_matrix.png')
    plt.show()

    # --- Metrics computation --------------------------------------------------
    tn, fp, fn, tp = conf_matrix.ravel()
    n_test = len(y_test)

    prec  = precision_score(y_test, y_test_pred, zero_division=0)
    rec   = recall_score(y_test, y_test_pred, zero_division=0)
    spec  = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv   = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc   = (tp + tn) / n_test

    f1      = f1_score(y_test, y_test_pred, zero_division=0)
    f2      = fbeta_score(y_test, y_test_pred, beta=2, zero_division=0)
    mcc     = matthews_corrcoef(y_test, y_test_pred)
    youden  = rec + spec - 1   if not np.isnan(spec) else np.nan
    bal_acc = (rec + spec) / 2 if not np.isnan(spec) else np.nan

    overtriage  = 1 - prec if not np.isnan(prec) else np.nan
    undertriage = 1 - npv  if not np.isnan(npv)  else np.nan

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    lr_pos = rec / (1 - spec) if (not np.isnan(spec) and spec < 1) else np.nan
    lr_neg = (1 - rec) / spec  if (not np.isnan(spec) and spec > 0) else np.nan

    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc  = average_precision_score(y_test, y_test_prob)

    # --- Precision-Recall curve plot -----------------------------------------
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_test_prob)
    baseline = float(y_test.mean())

    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    ax_pr.plot(rec_curve, prec_curve, color='darkorange', lw=2,
               label=f'PR curve  (PR-AUC = {pr_auc:.4f})')
    ax_pr.axhline(baseline, color='gray', linestyle='--',
                   label=f'Baseline (prevalence) = {baseline:.3f}')
    ax_pr.scatter([rec], [prec], color='black', zorder=5, s=80,
                  label=f'Threshold = {threshold:.4f}  |  P={prec:.3f}  R={rec:.3f}')
    ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax_pr.set_ylabel('Precision (PPV)', fontsize=11)
    ax_pr.set_title('Precision-Recall Curve (Test Set)', fontsize=13)
    ax_pr.legend(loc='best', fontsize=9)
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig_pr, 'pr_curve.png')
    plt.show()
    combine_exported_roc_pr_curves()

    # --- Clopper–Pearson CIs -------------------------------------------------
    def _cp(x, n, alpha=0.05):
        if n == 0:
            return (np.nan, np.nan)
        lo = _beta.ppf(alpha / 2,     x,     n - x + 1) if x > 0 else 0.0
        hi = _beta.ppf(1 - alpha / 2, x + 1, n - x)     if x < n else 1.0
        return lo, hi

    ci_cp = {
        "acc":  _cp(int(tp + tn), int(n_test)),
        "prec": _cp(int(tp), int(tp + fp)),
        "rec":  _cp(int(tp), int(tp + fn)),
        "spec": _cp(int(tn), int(tn + fp)),
        "npv":  _cp(int(tn), int(tn + fn)),
    }

    # --- Stratified bootstrap CIs --------------------------------------------
    def _safe_div(a, b):
        return a / b if b != 0 else np.nan

    rng     = np.random.default_rng(ci_seed)
    y_te_s  = pd.Series(y_test.values if hasattr(y_test, "values") else y_test).reset_index(drop=True)
    y_pr_s  = pd.Series(y_test_pred).reset_index(drop=True)
    y_prob_s = pd.Series(
        y_test_prob.values if hasattr(y_test_prob, "values") else y_test_prob
    ).reset_index(drop=True)
    year_s  = pd.Series(y_test_year.values if hasattr(y_test_year, "values") else y_test_year).reset_index(drop=True)
    boot_strata = _make_outcome_year_strata(y_te_s, year_s)
    strata_indices = [
        group.index.to_numpy()
        for _, group in boot_strata.groupby(boot_strata, sort=False)
    ]

    boot = {
        "roc_auc": [],
        "pr_auc": [],
        "f1": [],
        "f2": [],
        "mcc": [],
        "youden": [],
        "bal_acc": [],
    }

    for _ in range(n_bootstrap):
        idx   = np.concatenate([
            rng.choice(stratum_idx, size=len(stratum_idx), replace=True)
            for stratum_idx in strata_indices
        ])
        yt    = y_te_s.iloc[idx]
        yp    = y_pr_s.iloc[idx]
        yprob = y_prob_s.iloc[idx]

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        prec_b = _safe_div(tp_b, tp_b + fp_b)
        rec_b  = _safe_div(tp_b, tp_b + fn_b)
        spec_b = _safe_div(tn_b, tn_b + fp_b)

        f1_b = (
            _safe_div(2 * prec_b * rec_b, prec_b + rec_b)
            if not (np.isnan(prec_b) or np.isnan(rec_b)) else np.nan
        )
        f2_b = (
            _safe_div(5 * prec_b * rec_b, 4 * prec_b + rec_b)
            if not (np.isnan(prec_b) or np.isnan(rec_b)) else np.nan
        )
        denom_mcc = np.sqrt((tp_b+fp_b) * (tp_b+fn_b) * (tn_b+fp_b) * (tn_b+fn_b))
        mcc_b     = _safe_div(tp_b * tn_b - fp_b * fn_b, denom_mcc)
        youden_b  = (rec_b + spec_b - 1) if not (np.isnan(rec_b) or np.isnan(spec_b)) else np.nan
        bal_b     = _safe_div(rec_b + spec_b, 2)

        bootstrap_values = (
            ("roc_auc", roc_auc_score(yt, yprob)),
            ("pr_auc", average_precision_score(yt, yprob)),
            ("f1", f1_b),
            ("f2", f2_b),
            ("mcc", mcc_b),
            ("youden", youden_b),
            ("bal_acc", bal_b),
        )
        for key, val in bootstrap_values:
            if not np.isnan(val):
                boot[key].append(val)

    ci_boot = {
        k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
        for k, v in boot.items()
    }

    # --- Print ---------------------------------------------------------------
    print("\n--- Test set results ---")

    def _pct(val, ci):
        lo, hi = ci
        s = f"{val*100:.2f}%"
        if not (np.isnan(lo) or np.isnan(hi)):
            s += f"   [95% CI: {lo*100:.2f}% – {hi*100:.2f}%]"
        return s

    def _raw(val, ci):
        lo, hi = ci
        s = f"{val:+.4f}"
        if not (np.isnan(lo) or np.isnan(hi)):
            s += f"   [95% CI: {lo:+.4f} – {hi:+.4f}]"
        return s

    no_ci = (np.nan, np.nan)

    print("\n" + "="*60)
    print(f"{'CLASSIFICATION REPORT':^60}")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=labels))

    print("\n" + "="*70)
    print(f"{'PERFORMANCE METRICS WITH 95% CI':^70}")
    print("="*70)

    print(f"\n  Discrimination metrics  (Outcome-year Stratified Bootstrap CI):")
    print(f"  ROC-AUC:               {_pct(roc_auc, ci_boot['roc_auc'])}")
    print(f"  PR-AUC:                {_pct(pr_auc,  ci_boot['pr_auc'])}")

    print(f"\n  Binary metrics  (Clopper–Pearson CI):")
    print(f"  Accuracy:              {_pct(acc,  ci_cp['acc'])}")
    print(f"  Precision (PPV):       {_pct(prec, ci_cp['prec'])}")
    print(f"  Recall (Sensitivity):  {_pct(rec,  ci_cp['rec'])}")
    print(f"  Specificity:           {_pct(spec, ci_cp['spec'])}")
    print(f"  NPV:                   {_pct(npv,  ci_cp['npv'])}")

    print(f"\n  Composite metrics  (Outcome-year Stratified Bootstrap CI):")
    print(f"  F1 Score:              {_pct(f1,      ci_boot['f1'])}")
    print(f"  F2 Score:              {_pct(f2,      ci_boot['f2'])}")
    print(f"  MCC:                   {_raw(mcc,     ci_boot['mcc'])}")
    print(f"  Youden Index:          {_raw(youden,  ci_boot['youden'])}")
    print(f"  Balanced Accuracy:     {_pct(bal_acc, ci_boot['bal_acc'])}")

    print(f"\n  Triage-specific metrics:")
    print(f"  Overtriage Rate:       {overtriage*100:.2f}%")
    print(f"  Undertriage Rate:      {undertriage*100:.2f}%")

    print(f"\n  Error rates:")
    print(f"  False Positive Rate:   {fpr*100:.2f}%")
    print(f"  False Negative Rate:   {fnr*100:.2f}%")

    print(f"\n  Likelihood ratios:")
    print(f"  LR+:                   {lr_pos:.4f}")
    print(f"  LR-:                   {lr_neg:.4f}")

    print("\n" + "="*70 + "\n")

    return {
        "accuracy":            acc,
        "roc_auc":             roc_auc,
        "roc_auc_ci_lower":    ci_boot["roc_auc"][0],
        "roc_auc_ci_upper":    ci_boot["roc_auc"][1],
        "pr_auc":              pr_auc,
        "pr_auc_ci_lower":     ci_boot["pr_auc"][0],
        "pr_auc_ci_upper":     ci_boot["pr_auc"][1],
        "precision":           prec,
        "recall":              rec,
        "specificity":         spec,
        "npv":                 npv,
        "f1":                  f1,
        "f2":                  f2,
        "mcc":                 mcc,
        "youden_index":        youden,
        "balanced_accuracy":   bal_acc,
        "overtriage":          overtriage,
        "undertriage":         undertriage,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "lr_positive":         lr_pos,
        "lr_negative":         lr_neg,
    }

def print_train_test_comparison(
    automl,
    X_train, y_train,
    X_test,  y_test,
    threshold: float,
    pr_auc_diff_threshold: float = 0.05,
    f1_diff_threshold: float     = 0.05,
    f2_diff_threshold: float     = 0.05,
    overfit_gap_threshold: float = 0.05,
) -> pd.DataFrame:
    from sklearn.metrics import (
        f1_score, fbeta_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, accuracy_score,
    )

    # Prevalence computed automatically from y_train
    prevalence = float(y_train.mean())

    # --- Probabilities and predictions ---
    y_train_prob = automl.predict_proba(X_train)[:, 1]
    y_test_prob  = automl.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_test_pred  = (y_test_prob  >= threshold).astype(int)

    # --- Compute metrics helper ---
    def _metrics(y_true, y_pred, y_prob):
        return {
            "PR-AUC":      average_precision_score(y_true, y_prob),
            "ROC-AUC":     roc_auc_score(y_true, y_prob),
            "F1":          f1_score(y_true, y_pred, zero_division=0),
            "F2":          fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            "Accuracy":    accuracy_score(y_true, y_pred),
            "Precision":   precision_score(y_true, y_pred, zero_division=0),
            "Recall":      recall_score(y_true, y_pred, zero_division=0),
            "Specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        }

    train_m = _metrics(y_train, y_train_pred, y_train_prob)
    test_m  = _metrics(y_test,  y_test_pred,  y_test_prob)

    # -------------------------------------------------------------------------
    # Baselines: value a random classifier would achieve
    # -------------------------------------------------------------------------
    baselines = {
        "PR-AUC":      (prevalence,          f"{prevalence*100:.1f}%  (prevalence)"),
        "ROC-AUC":     (0.50,                "50.0%  (random)"),
        "F1":          (0.0,                 "~0.0%  (imbalanced)"),
        "F2":          (0.0,                 "~0.0%  (imbalanced)"),
        "Accuracy":    (1 - prevalence,      f"{(1-prevalence)*100:.1f}%  (majority class)"),
        "Precision":   (prevalence,          f"{prevalence*100:.1f}%  (prevalence)"),
        "Recall":      (0.50,                "50.0%  (floor)"),
        "Specificity": (0.50,                "50.0%  (floor)"),
    }

    # -------------------------------------------------------------------------
    # Prevalence-adjusted underfitting thresholds
    # -------------------------------------------------------------------------
    underfit_thresholds = {
        "PR-AUC":      (min(prevalence * 2, 0.50),
                        f"prevalence × 2 = {min(prevalence*2, 0.50)*100:.1f}%"),
        "ROC-AUC":     (0.60,
                        "random + 10pp = 60.0%"),
        "F1":          (max(0.30, prevalence * 1.5),
                        f"max(0.30, prev×1.5) = {max(0.30, prevalence*1.5)*100:.1f}%"),
        "F2":          (max(0.30, prevalence * 1.5),
                        f"max(0.30, prev×1.5) = {max(0.30, prevalence*1.5)*100:.1f}%"),
        "Accuracy":    ((1 - prevalence) + 0.05,
                        f"majority + 5pp = {((1-prevalence)+0.05)*100:.1f}%"),
        "Precision":   (prevalence * 1.5,
                        f"prevalence × 1.5 = {prevalence*1.5*100:.1f}%"),
        "Recall":      (0.50,
                        "clinical floor = 50.0%"),
        "Specificity": (0.50,
                        "technical floor = 50.0%"),
    }

    # -------------------------------------------------------------------------
    # Per-metric gap thresholds for overfitting detection
    # -------------------------------------------------------------------------
    diff_thresholds = {
        "PR-AUC":      pr_auc_diff_threshold,
        "ROC-AUC":     overfit_gap_threshold,
        "F1":          f1_diff_threshold,
        "F2":          f2_diff_threshold,
        "Accuracy":    overfit_gap_threshold,
        "Precision":   overfit_gap_threshold,
        "Recall":      overfit_gap_threshold,
        "Specificity": overfit_gap_threshold,
    }

    rows = []
    metric_order = ["PR-AUC", "ROC-AUC", "F1", "F2", "Accuracy", "Precision", "Recall", "Specificity"]

    for metric in metric_order:
        train_val              = train_m[metric]
        test_val               = test_m[metric]
        gap                    = train_val - test_val
        diff_thr               = diff_thresholds[metric]
        under_thr, under_label = underfit_thresholds[metric]
        baseline_val, baseline_label = baselines[metric]

        is_underfit = train_val < under_thr
        is_overfit  = gap > diff_thr

        rows.append({
            "Metric":         metric,
            "Baseline":       baseline_val,
            "Baseline_label": baseline_label,
            "Train":          train_val,
            "Underfit_thr":   under_thr,
            "Underfit_label": under_label,
            "Underfitting":   "🔴 Yes" if is_underfit else "✅ No",
            "Test":           test_val,
            "Gap (Tr-Te)":    gap,
            "Max Gap":        diff_thr,
            "Overfitting":    "🔴 Yes" if is_overfit  else "✅ No",
        })

    comparison_df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Pretty print
    # -------------------------------------------------------------------------
    sep = "─" * 125

    print("\n" + sep)
    print(f"{'TRAIN vs TEST PERFORMANCE COMPARISON':^125}")
    print(f"{'(threshold applied: ' + str(round(threshold, 4)) + ')':^125}")
    print(sep)
    print(f"  Dataset context  :  prevalence = {prevalence*100:.1f}%  "
          f"|  PR-AUC random baseline = {prevalence*100:.1f}%  "
          f"|  Accuracy random baseline = {(1-prevalence)*100:.1f}%")
    print(f"  Overfitting gap  :  default threshold = {overfit_gap_threshold*100:.1f}pp  "
          f"(PR-AUC = {pr_auc_diff_threshold*100:.1f}pp, F1 = {f1_diff_threshold*100:.1f}pp, "
          f"F2 = {f2_diff_threshold*100:.1f}pp)")
    print(f"  Underfitting     :  thresholds adjusted per metric based on dataset prevalence.")
    print(sep)

    print(
        f"  {'Metric':<14}"
        f"  {'Baseline':<26}"
        f"  {'Train':>8}"
        f"  {'Underfit threshold':<26}"
        f"  {'Underfit':^10}"
        f"  {'Test':>8}"
        f"  {'Gap':>8}"
        f"  {'Max Gap':>8}"
        f"  {'Overfit':^10}"
    )
    print(sep)

    for _, row in comparison_df.iterrows():
        print(
            f"  {row['Metric']:<14}"
            f"  {row['Baseline_label']:<26}"
            f"  {row['Train']*100:>7.2f}%"
            f"  {row['Underfit_label']:<26}"
            f"  {row['Underfitting']:^10}"
            f"  {row['Test']*100:>7.2f}%"
            f"  {row['Gap (Tr-Te)']*100:>+7.2f}pp"
            f"  {row['Max Gap']*100:>7.1f}pp"
            f"  {row['Overfitting']:^10}"
        )

    print(sep)
    n_overfit  = comparison_df["Overfitting"].str.contains("Yes").sum()
    n_underfit = comparison_df["Underfitting"].str.contains("Yes").sum()
    print(f"\n  Underfitting: {n_underfit} metric(s) flagged  |  "
          f"Overfitting: {n_overfit} metric(s) flagged")
    print(sep + "\n")

    return comparison_df

def plot_feature_importances(automl, X_train, feature_importance,
                             X_test, y_test, seed, task="classification",
                             builtin_importance_type="gain"):
    """
    Computes and plots feature importance using permutation or built-in methods.

    feature_importance values:
        0 → no importance computed
        1 → permutation only (fallback to built-in if it fails)
        2 → both permutation and built-in

    Permutation importance: shuffles each feature independently and measures
    how much the chosen scoring metric drops. A large drop = high importance.
    Importances are metric decrements (can be negative if the feature adds noise).

    Built-in importance: for LightGBM, gain as a percentage of total gain; for
    Random Forest and ExtraTrees, impurity reduction. These are relative scores
    with no direct metric interpretation.
    """
    # Return early if no importance requested
    if feature_importance == 0:
        print("\n--- Feature importance skipped (feature_importance=0) ---")
        return None, None

    importance_df_permutation = None
    importance_df_builtin = None

    # Exclude text embedding columns — only clinical features are evaluated
    # for computational efficiency and clinical interpretability
    non_emb_cols = [col for col in X_train.columns if not col.startswith('emb')]

    # Average Precision (PR-AUC) mirrors the 'ap' training metric used in FLAML,
    # ensuring consistency between training objective and importance evaluation
    perm_scoring = "average_precision"

    # -------------------------------------------------------------------------
    # PERMUTATION IMPORTANCE (feature_importance == 1 or 2)
    # -------------------------------------------------------------------------
    use_permutation_importance = False

    if feature_importance >= 1:
        print(f"--- Computing permutation importance ({perm_scoring}) for {len(non_emb_cols)} clinical features ---")

        try:
            class ClassifierWrapper:
                """
                Wraps FLAML AutoML for sklearn's permutation_importance API.
                Reconstructs the full feature matrix on each prediction call
                by merging permuted clinical columns with the original embeddings.
                """
                def __init__(self, model, X_full):
                    self.model = model
                    self.X_full = X_full
                    self._estimator_type = "classifier"
                    self.classes_ = np.array(model.classes_)

                def fit(self, X, y):
                    return self

                def _reconstruct(self, X_clinical):
                    # Re-attach original embedding columns using index alignment,
                    # then restore original column order expected by the model
                    emb_cols = [c for c in self.X_full.columns if c.startswith('emb')]
                    X_emb = self.X_full.loc[X_clinical.index, emb_cols]
                    return pd.concat([X_clinical, X_emb], axis=1)[self.X_full.columns]

                def predict(self, X):
                    return self.model.predict(self._reconstruct(X))

                def predict_proba(self, X):
                    proba = self.model.predict_proba(self._reconstruct(X))
                    # Ensure 2D output for both binary and multiclass
                    if proba.ndim == 1:
                        proba = np.column_stack([1 - proba, proba])
                    return proba

            # Full X_test passed so the wrapper can recover embeddings internally
            wrapped_model = ClassifierWrapper(automl, X_test)

            # Only clinical columns are permuted; embeddings are reconstructed internally
            perm_imp = permutation_importance(
                wrapped_model,
                X_test[non_emb_cols],
                y_test,
                scoring=perm_scoring,
                n_repeats=5,
                random_state=seed,
                # Keep this sequential: joblib has to pickle the local wrapper
                # and FLAML AutoML object when n_jobs != 1, which can fail.
                n_jobs=1,
            )

            importance_df_permutation = (
                pd.DataFrame({
                    "feature": non_emb_cols,
                    "importance": perm_imp.importances_mean,
                    "importance_std": perm_imp.importances_std,
                })
                .sort_values("importance", ascending=False)
            )

            use_permutation_importance = True
            print("Permutation importance computed successfully.")

        except Exception as e:
            print(f"Warning: Permutation importance failed with error: {e}")
            # If mode is 1 (permutation only), fallback to built-in
            if feature_importance == 1:
                print("Falling back to built-in feature importance...")
                feature_importance = 2  # Trigger built-in block below

    # -------------------------------------------------------------------------
    # BUILT-IN IMPORTANCE (feature_importance == 2, or fallback from mode 1)
    # -------------------------------------------------------------------------
    if feature_importance == 2:
        print("--- Computing built-in feature importance ---")

        try:
            best_model_wrapper = getattr(automl, "model", None)
            best_model = getattr(best_model_wrapper, "estimator", best_model_wrapper)

            if best_model is None:
                raise RuntimeError("No final estimator available.")

            if hasattr(best_model, "feature_name_"):
                feature_names = list(best_model.feature_name_)
            elif hasattr(best_model, "feature_names_in_"):
                feature_names = list(best_model.feature_names_in_)
            elif getattr(automl, "feature_names_in_", None) is not None:
                feature_names = list(automl.feature_names_in_)
            else:
                feature_names = list(X_train.columns)

            if hasattr(best_model, "booster_") or hasattr(best_model, "feature_importances_"):
                if hasattr(best_model, "booster_"):
                    # LightGBM: feature_importances_ defaults to split counts, which favour
                    # continuous features (more candidate split points). Gain is the total
                    # loss reduction contributed by each feature; reported as % of total gain.
                    importances = np.asarray(
                        best_model.booster_.feature_importance(
                            importance_type=builtin_importance_type
                        ),
                        dtype=float,
                    )
                    feature_names = list(best_model.booster_.feature_name())
                    if builtin_importance_type == "gain" and importances.sum() > 0:
                        importances = 100 * importances / importances.sum()
                else:
                    # sklearn tree ensembles (RF, ExtraTrees): impurity-based importance
                    importances = np.asarray(best_model.feature_importances_, dtype=float)

                if len(importances) == len(feature_names):
                    importance_df_builtin = (
                        pd.DataFrame({
                            "feature": feature_names,
                            "importance": importances,
                        })
                        .sort_values("importance", ascending=False)
                    )
                    print("Built-in feature importance computed successfully.")
                else:
                    print(f"Warning: Importance length ({len(importances)}) doesn't match features ({len(feature_names)})")

            elif hasattr(best_model, 'coef_'):
                # Linear models: absolute coefficients as proxy for importance.
                # For multiclass, coef_ is 2D (n_classes x n_features) →
                # average across classes to get a single importance per feature
                coef = np.asarray(best_model.coef_, dtype=float)
                importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef.ravel())

                if len(importances) == len(feature_names):
                    importance_df_builtin = (
                        pd.DataFrame({
                            "feature": feature_names,
                            "importance": importances,
                        })
                        .sort_values("importance", ascending=False)
                    )
                    print("Built-in feature importance computed successfully.")
                else:
                    print(f"Warning: Coefficient length ({len(importances)}) doesn't match features ({len(feature_names)})")

            else:
                print(f"Warning: Model type '{type(best_model).__name__}' does not support built-in feature importance.")

            # Filter out embedding features — not individually interpretable
            if importance_df_builtin is not None:
                importance_df_builtin = (
                    importance_df_builtin[~importance_df_builtin["feature"].str.startswith('emb')]
                    .sort_values("importance", ascending=False)
                )
                print("Built-in importance filtered to show clinical features only.")

        except Exception as e:
            print(f"Warning: Could not extract built-in feature importance: {e}")

    # -------------------------------------------------------------------------
    # PLOTTING — one plot per computed importance type
    # -------------------------------------------------------------------------
    labels_map = get_labels_map(demand_code=_current_demand_code, columns=X_train.columns)

    def _plot_and_print(importance_df, is_permutation):
        top_k = min(15, len(importance_df))
        plot_df = importance_df.head(top_k)
        feature_labels = plot_df["feature"].map(lambda c: get_display_label(c, labels_map))

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))

        if is_permutation:
            # importances are AP fractions -> convert to percentage points
            values = plot_df["importance"] * 100
            errors = plot_df["importance_std"] * 100
        else:
            values = plot_df["importance"]
            errors = None

        ax_imp.barh(
            feature_labels[::-1], values[::-1],
            xerr=None if errors is None else errors[::-1], capsize=2,
        )

        if is_permutation:
            ax_imp.set_title(f"Top Features (Permutation Importance — {perm_scoring})")
            ax_imp.set_xlabel("Mean decrease in average precision (percentage points)")
            filename = 'permutation_importance.png'
        else:
            ax_imp.set_title(f"Top Features (Built-in Importance — {automl.best_estimator}, "
                             f"{builtin_importance_type})")
            ax_imp.set_xlabel("Share of total gain (%)" if builtin_importance_type == "gain"
                              else "Number of splits")
            filename = 'builtin_importance.png'

        plt.tight_layout()
        save_figure(fig_imp, filename)
        plt.show()

        print(f"\nTop {top_k} most important features:")
        if is_permutation:
            print(f"Metric: Mean decrease in {perm_scoring} (percentage points)\n")
        for _, row in plot_df.iterrows():
            if is_permutation:
                print(f"  {row['feature']:<20} = {row['importance']*100:.4f} pp")
            else:
                print(f"  {row['feature']:<20} {row['importance']:.4f}")

    if importance_df_permutation is not None:
        _plot_and_print(importance_df_permutation, is_permutation=True)

    if importance_df_builtin is not None:
        _plot_and_print(importance_df_builtin, is_permutation=False)

    return importance_df_permutation, importance_df_builtin


def report_effective_features(automl, X_train, top_k=15):
    """Reports how many supplied features the final model effectively uses."""
    print("\n--- Effective feature usage of the final model ---")

    best_model = getattr(automl, "model", None)
    estimator = getattr(best_model, "estimator", best_model)
    if estimator is None:
        print("No final estimator available; skipping effective feature report.")
        return None

    if hasattr(estimator, "feature_name_"):
        feature_names = list(estimator.feature_name_)
    elif hasattr(estimator, "feature_names_in_"):
        feature_names = list(estimator.feature_names_in_)
    elif getattr(automl, "feature_names_in_", None) is not None:
        feature_names = list(automl.feature_names_in_)
    else:
        feature_names = list(X_train.columns)

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        basis = "split-based importance (tree ensemble)"
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef.ravel())
        basis = "absolute coefficient (linear model)"
    else:
        print(
            f"Model type '{type(estimator).__name__}' exposes no importance "
            "attribute; effective feature count not available."
        )
        return None

    if len(importances) != len(feature_names):
        print(
            f"Warning: importance length ({len(importances)}) does not match "
            f"feature count ({len(feature_names)}); skipping report."
        )
        return None

    used_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .loc[lambda d: d["importance"] > 0]
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    def _family(col):
        if col and col[0].isdigit():
            return "Triage questions"
        for prefix, name in [
            ("lr_", "Literal reason"),
            ("hist_", "Clinical history (new)"),
            ("com_", "Legacy history (old)"),
            ("atc", "Medication"),
            ("emb", "Text embeddings"),
        ]:
            if col.startswith(prefix):
                return name
        return "Other"

    n_supplied = len(feature_names)
    n_used = len(used_df)
    supplied_by_family = pd.Series([_family(c) for c in feature_names]).value_counts()
    used_by_family = used_df["feature"].map(_family).value_counts()

    print(f"  Importance basis          : {basis}")
    print(f"  Features supplied         : {X_train.shape[1]}")
    print(f"  Features tracked          : {n_supplied}")
    print(f"  Features used (non-zero)  : {n_used}  ({n_used / n_supplied:.1%})")
    print(f"  Features never used       : {n_supplied - n_used}")
    print("\n  Used / supplied by feature family:")
    for family in supplied_by_family.index:
        used_n = int(used_by_family.get(family, 0))
        supplied_n = int(supplied_by_family[family])
        print(f"    {family:<28} {used_n:>4} / {supplied_n:<4} ({used_n / supplied_n:.0%})")

    print(f"\n  Top {min(top_k, n_used)} features by importance:")
    for _, row in used_df.head(top_k).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:.4f}")

    return used_df


def _compute_shap_values(automl, X_test, task="classification", seed=42, sample_size=None):
    """
    Computes SHAP values for the fitted estimator, aligning features to what
    the underlying model was actually trained on and excluding embedding
    columns. Shared by plot_shap_interpretation and plot_marginal_effects so
    SHAP is computed at most once per run.

    Returns a dict with:
      - shap_values: raw SHAP output (list / 2D / 3D ndarray, depending on model and task)
      - train_features: feature names used by the underlying estimator
      - non_emb_cols: clinical (non-embedding) feature names, in train_features order
      - non_emb_idx: indices of non_emb_cols within train_features
      - X_sample_shap: SHAP-ready subsample (categoricals as int32), aligned to train_features
      - X_plot: X_sample_shap restricted to non-embedding columns, with display labels
      - labels_map: display-label mapping for non_emb_cols

    Returns None if SHAP computation fails (caller prints a warning).
    """

    def _prepare_for_shap(X):
        """
        Convert categorical columns to plain int32 for SHAP compatibility.
        LightGBM stores categorical metadata during training and validates
        that input dtypes match — cat.codes returns int8/int16 which can
        still trigger the mismatch error. Casting to int32 avoids this.
        """
        X_shap = X.copy()
        cat_cols = X_shap.select_dtypes(include='category').columns
        for col in cat_cols:
            X_shap[col] = X_shap[col].cat.codes.astype('int32')
        return X_shap

    try:
        # Extract the underlying fitted estimator from FLAML
        model = automl.model.estimator

        # sample_size=None -> full test set (TreeExplainer is fast for small ensembles)
        n_shap = len(X_test) if sample_size is None else min(sample_size, len(X_test))
        X_sample = X_test if n_shap == len(X_test) else X_test.sample(n_shap, random_state=seed)

        # Convert categoricals to int32 — applied to all model types
        X_sample_shap = _prepare_for_shap(X_sample)

        # Align features to what the model was actually trained on.
        # FLAML may internally select a feature subset, so X_test can have
        # more columns than the underlying estimator expects.
        if hasattr(model, 'feature_name_'):          # LightGBM
            train_features = model.feature_name_
        elif hasattr(model, 'feature_names_in_'):    # XGBoost / sklearn
            train_features = list(model.feature_names_in_)
        else:
            train_features = X_sample_shap.columns.tolist()

        X_sample_shap = X_sample_shap[train_features]

        # Precompute non-embedding column indices once for slicing shap_values arrays
        non_emb_cols = [c for c in train_features if not c.startswith('emb')]
        non_emb_idx  = [train_features.index(c) for c in non_emb_cols]

        # X used for plotting — converted + clinical columns only
        X_plot = X_sample_shap[non_emb_cols]

        # Rename columns to human-readable display labels for plotting
        labels_map = get_labels_map(demand_code=_current_demand_code, columns=non_emb_cols)
        X_plot = X_plot.rename(columns=lambda c: get_display_label(c, labels_map))

        tree_model_types = (
            "LGBMClassifier", "XGBClassifier", "RandomForestClassifier",
            "ExtraTreesClassifier", "GradientBoostingClassifier"
        )

        if type(model).__name__ in tree_model_types:
            print(f"Using TreeExplainer for {type(model).__name__}...")

            if type(model).__name__ == "LGBMClassifier":
                # LightGBM stores categorical feature indices in booster_.pandas_categorical
                # and validates them on every prediction call, including those made by SHAP.
                # Converting to int32 is not enough — LightGBM checks its own internal
                # metadata, not the dtype of the input.
                # Fix: temporarily patch pandas_categorical to empty, compute SHAP values,
                # then restore the original metadata unconditionally via finally.
                #
                # booster_ is the underlying Booster object exposed by LGBMClassifier.
                # We probe for pandas_categorical defensively in case the attribute
                # does not exist in older LightGBM versions.
                booster = model.booster_
                original_categorical = getattr(booster, 'pandas_categorical', None)

                if original_categorical is not None:
                    booster.pandas_categorical = []

                try:
                    explainer   = shap.TreeExplainer(
                        model,
                        feature_perturbation="tree_path_dependent"
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="LightGBM binary classifier.*TreeExplainer",
                            category=UserWarning
                        )
                        shap_values = explainer.shap_values(X_sample_shap)
                finally:
                    # Restore original metadata even if SHAP raises an exception
                    if original_categorical is not None:
                        booster.pandas_categorical = original_categorical
            else:
                explainer   = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                shap_values = explainer.shap_values(X_sample_shap)

        else:
            print(f"Using generic Explainer for {type(model).__name__}...")
            background  = X_sample_shap.sample(min(100, len(X_sample_shap)), random_state=seed)
            explainer   = shap.Explainer(model, background)
            shap_out    = explainer(X_sample_shap)
            shap_values = shap_out.values

        print(f"SHAP values computed on {n_shap} samples.")

        return {
            "shap_values": shap_values,
            "train_features": train_features,
            "non_emb_cols": non_emb_cols,
            "non_emb_idx": non_emb_idx,
            "X_sample_shap": X_sample_shap,
            "X_plot": X_plot,
            "labels_map": labels_map,
        }

    except Exception as e:
        print(f"Warning: SHAP computation failed: {e}")
        return None

def plot_shap_interpretation(automl, X_test, task="classification", max_display=15, seed=42, shap_data=None):
    """
    Computes SHAP values and plots a summary excluding embedding features.
    SHAP provides directional impact per feature (positive = pushes toward positive class).

    For binary classification: single summary plot.
    For multiclass: one summary plot per class.

    Parameters
    ----------
    shap_data : Optional[dict]
        Precomputed output of _compute_shap_values. If None, it is computed
        internally.
    """

    print("\n--- Initializing SHAP Explainer ---")
    _start_time = time.time()

    if shap_data is None:
        shap_data = _compute_shap_values(automl, X_test, task=task, seed=seed)
    if shap_data is None:
        return None

    shap_values  = shap_data["shap_values"]
    non_emb_idx  = shap_data["non_emb_idx"]
    X_plot       = shap_data["X_plot"]

    try:
        # --- Plotting ---
        if task == "multiclass":
            # TreeExplainer returns list of 2D arrays (one per class)
            # Generic Explainer returns 3D array (n_samples, n_features, n_classes)
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

            class_names = [str(c) for c in automl.classes_]

            for i, class_name in enumerate(class_names):
                print(f"\nGenerating SHAP Summary Plot — Class: {class_name}")
                shap_class_filtered = shap_values[i][:, non_emb_idx]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
                    shap.summary_plot(
                        shap_class_filtered,
                        X_plot,
                        max_display=max_display,
                        show=False,
                    )
                save_figure(plt.gcf(), f'shap_class_{class_name}.png')
                plt.show()
        else:
            # Binary: use positive class (index 1)
            # TreeExplainer → list [neg, pos] or single 2D array
            # Generic Explainer → 2D array directly
            if isinstance(shap_values, list):
                shap_values_binary = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values_binary = shap_values[:, :, 1]
            else:
                shap_values_binary = shap_values

            shap_filtered = shap_values_binary[:, non_emb_idx]

            print("\nGenerating SHAP Summary Plot — Positive class")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="shap")
                shap.summary_plot(
                    shap_filtered,
                    X_plot,
                    max_display=max_display,
                    show=False,
                )
            plt.gca().set_xlabel("SHAP value (log-odds contribution towards P1)")
            save_figure(plt.gcf(), 'shap_importance.png')
            plt.show()

        print("SHAP interpretation completed successfully.")
        elapsed = time.time() - _start_time
        print(f"Total SHAP time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return shap_values

    except Exception as e:
        print(f"Warning: SHAP interpretation failed: {e}")
        return None

def _rank_shap_features(shap_values, non_emb_cols, non_emb_idx, task="classification"):
    """
    Ranks clinical (non-embedding) features by mean(|SHAP value|), descending.

    For binary classification, uses the positive-class SHAP matrix (handles
    TreeExplainer list output, 3D ndarray, or plain 2D ndarray). For
    multiclass, averages |SHAP| across classes.

    Returns
    -------
    pd.DataFrame
        Columns: 'feature', 'mean_abs_shap', sorted descending.
    """
    if task == "multiclass":
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            class_arrays = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        else:
            class_arrays = shap_values
        per_class_abs = [np.abs(arr[:, non_emb_idx]) for arr in class_arrays]
        mean_abs_shap = np.mean([arr.mean(axis=0) for arr in per_class_abs], axis=0)
    else:
        if isinstance(shap_values, list):
            shap_values_binary = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_binary = shap_values[:, :, 1]
        else:
            shap_values_binary = shap_values

        shap_filtered = shap_values_binary[:, non_emb_idx]
        mean_abs_shap = np.abs(shap_filtered).mean(axis=0)

    return (
        pd.DataFrame({"feature": non_emb_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _summarise_shap(shap_data, top_k=15):
    """Summarises SHAP for all features and prints the top-ranked features."""
    ranking = _rank_shap_features(
        shap_data["shap_values"], shap_data["non_emb_cols"],
        shap_data["non_emb_idx"], task="classification",
    )
    ranking["rank_shap"] = np.arange(1, len(ranking) + 1)
    sv = shap_data["shap_values"]
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]

    X = shap_data["X_sample_shap"]
    cols = list(shap_data["train_features"])
    rows = []
    for _, ranked_feature in ranking.iterrows():
        feat = ranked_feature["feature"]
        s = sv[:, cols.index(feat)]
        x = pd.to_numeric(X[feat], errors="coerce").to_numpy(dtype=float)
        observed = x[~np.isnan(x)]
        is_binary = observed.size > 0 and set(np.unique(observed)).issubset({0.0, 1.0})
        present, absent = (x == 1), (x == 0)

        levels = ""
        unique_levels = np.unique(observed)
        if not is_binary and 0 < len(unique_levels) <= 10:
            level_parts = []
            for level in unique_levels:
                level_mask = x == level
                level_label = str(int(level)) if float(level).is_integer() else str(level)
                level_parts.append(
                    f"{level_label}: n={int(level_mask.sum())}, mean={s[level_mask].mean():.3f}"
                )
            levels = "; ".join(level_parts)

        rows.append({
            "feature": feat,
            "mean_abs_shap": ranked_feature["mean_abs_shap"],
            "n_present": int(present.sum()) if is_binary else np.nan,
            "mean_shap_present": s[present].mean() if is_binary and present.any() else np.nan,
            "mean_shap_absent": s[absent].mean() if is_binary and absent.any() else np.nan,
            "rank_shap": int(ranked_feature["rank_shap"]),
            "levels": levels,
        })

    out = pd.DataFrame(rows)
    top_shap = out.head(min(top_k, len(out)))
    print(f"\nTop {len(top_shap)} most important features:")
    print(f"Metric: Mean absolute SHAP value (log-odds scale, n = {len(X):,})\n")
    for _, row in top_shap.iterrows():
        print(f"  {row['feature']:<20} {row['mean_abs_shap']:.4f}")
    return out


def export_interpretability_table(
    importance_perm,
    importance_builtin,
    effective_features_df,
    shap_summary_df,
    demand_code,
):
    """Exports combined permutation, gain, split-count, and SHAP results."""
    perm = importance_perm[["feature", "importance", "importance_std"]].copy()
    perm["perm_rank"] = perm["importance"].rank(method="min", ascending=False)
    perm = perm.rename(columns={
        "importance": "perm_importance",
        "importance_std": "perm_importance_std",
    })

    gain = importance_builtin[["feature", "importance"]].copy()
    gain["gain_rank"] = gain["importance"].rank(method="min", ascending=False)
    gain = gain.rename(columns={"importance": "gain"})

    splits = effective_features_df[["feature", "importance"]].copy()
    splits = splits.rename(columns={"importance": "splits"})

    shap_cols = [
        "feature", "mean_abs_shap", "rank_shap", "n_present",
        "mean_shap_present", "mean_shap_absent", "levels",
    ]
    merged = (
        shap_summary_df[shap_cols]
        .merge(perm, on="feature", how="outer")
        .merge(gain, on="feature", how="outer")
        .merge(splits, on="feature", how="left")
    )
    merged["splits"] = merged["splits"].fillna(0)

    labels_map = get_labels_map(
        demand_code=demand_code,
        columns=merged["feature"].dropna().tolist(),
    )

    def _family(feature):
        if feature and feature[0].isdigit():
            return "Triage questions"
        if feature.startswith("lr_"):
            return "Literal reason"
        if feature.startswith(("hist_", "com_")):
            return "Clinical history"
        if feature.startswith("atc"):
            return "Medication"
        return "Other"

    merged["Feature"] = merged["feature"].map(
        lambda feature: get_display_label(feature, labels_map)
    )
    merged["Family"] = merged["feature"].map(_family)
    merged = merged.loc[merged["splits"] > 0].sort_values(
        "rank_shap", na_position="last"
    ).reset_index(drop=True)

    raw_df = pd.DataFrame({
        "Feature": merged["Feature"],
        "Family": merged["Family"],
        "Permutation, pp (SD)": merged.apply(
            lambda row: (
                "" if pd.isna(row["perm_importance"])
                else f'{row["perm_importance"] * 100} ({row["perm_importance_std"] * 100})'
            ),
            axis=1,
        ),
        "Perm. rank": merged["perm_rank"],
        "Gain (%)": merged["gain"],
        "Gain rank": merged["gain_rank"],
        "Splits": merged["splits"],
        "Mean |SHAP|": merged["mean_abs_shap"],
        "SHAP rank": merged["rank_shap"],
        "N present": merged["n_present"],
        "Mean SHAP if present": merged["mean_shap_present"],
        "Mean SHAP if absent": merged["mean_shap_absent"],
        "SHAP by level": merged["levels"].fillna(""),
    })

    def _format_rank(value):
        return "" if pd.isna(value) else str(int(value))

    def _format_decimal(value, decimals):
        return "" if pd.isna(value) else f"{value:.{decimals}f}"

    display_df = raw_df.copy()
    display_df["Permutation, pp (SD)"] = merged.apply(
        lambda row: (
            "" if pd.isna(row["perm_importance"])
            else f'{row["perm_importance"] * 100:.2f} '
                 f'({row["perm_importance_std"] * 100:.2f})'
        ),
        axis=1,
    )
    display_df["Perm. rank"] = raw_df["Perm. rank"].map(_format_rank)
    display_df["Gain (%)"] = raw_df["Gain (%)"].map(lambda value: _format_decimal(value, 2))
    display_df["Gain rank"] = raw_df["Gain rank"].map(_format_rank)
    display_df["Splits"] = raw_df["Splits"].map(_format_rank)
    display_df["Mean |SHAP|"] = raw_df["Mean |SHAP|"].map(
        lambda value: _format_decimal(value, 3)
    )
    display_df["SHAP rank"] = raw_df["SHAP rank"].map(_format_rank)
    for column in ["N present", "Mean SHAP if present", "Mean SHAP if absent"]:
        display_df[column] = raw_df[column].map(lambda value: _format_decimal(value, 3))

    title = "Feature contributions of the final LightGBM model (test set)"
    footnote = (
        "Permutation: mean (SD) decrease in test-set average precision over five repeats, "
        "in percentage points. Gain: share of total LightGBM gain. Splits: number of tree "
        "splits using the feature. SHAP: mean absolute SHAP value on the log-odds scale "
        "(n = 9,233 test calls); mean SHAP is shown when a binary feature is present or "
        "absent, or per level for multi-level triage items. Ranks are computed over all "
        "186 features; features never used in a split are omitted."
    )
    save_table(
        display_df,
        "interpretability_table.docx",
        title=title,
        footnote=footnote,
    )

    tables_dir = os.path.join(_get_figures_dir(demand_code), "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "interpretability_table.csv")
    raw_df.to_csv(csv_path, index=False)
    print(f"Table saved: {csv_path}")

    return display_df


def _grid_dims(n: int):
    """Return (n_rows, n_cols) for n panels: max 3 cols, max 3 rows per column."""
    import math
    if n <= 3:
        return 1, n
    for n_cols in range(2, 4):
        n_rows = math.ceil(n / n_cols)
        if n_rows <= 3:
            return n_rows, n_cols
    return math.ceil(n / 3), 3


def plot_marginal_effects(
    automl,
    X_test,
    y_test,
    y_test_year,
    shap_data=None,
    n_features=5,
    task="classification",
    seed=42,
    n_bootstrap=200,
    ci_seed=42,
    n_grid_points=25,
    cat_max_unique=10,
):
    """
    Plots the marginal predicted probability of the positive class against
    the value of each of the top-N most important features (ranked by
    mean(|SHAP value|)), with a 95% bootstrap confidence interval.

    The trained model is held fixed: bootstrap resampling is applied only to
    a subsample of the test set (no retraining), following the outcome-year
    stratified bootstrap pattern used in print_test_metrics_with_ci.

    Continuous features (non-category dtype with > cat_max_unique distinct
    values) are swept across their observed 1st-99th percentile range and
    plotted as a median curve with a shaded 95% CI band.

    Categorical/binary features (category dtype, or <= cat_max_unique
    distinct values) are plotted as median points with 95% CI error bars,
    one per observed category.

    Parameters
    ----------
    automl : flaml.AutoML
        Fitted AutoML object. predict_proba is called on the full feature
        set; FLAML handles its own feature alignment/encoding internally.
    X_test, y_test, y_test_year : array-like
        Test split features, outcome and year (used for the stratified
        bootstrap resampling).
    shap_data : Optional[dict]
        Precomputed output of _compute_shap_values. If None, computed
        internally.
    n_features : int
        Number of top SHAP features to plot.
    n_bootstrap : int
        Number of bootstrap resamples of the test subsample.
    n_grid_points : int
        Number of grid points swept for continuous features.
    cat_max_unique : int
        Features with at most this many distinct values (or category dtype)
        are treated as categorical/binary.

    Returns
    -------
    Optional[pd.DataFrame]
        Top-N features with columns 'feature' and 'mean_abs_shap'. None if
        SHAP computation failed.
    """
    print("\n--- Generating interpretability (marginal effect) plots ---")
    _start_time = time.time()

    if shap_data is None:
        shap_data = _compute_shap_values(automl, X_test, task=task, seed=seed)
    if shap_data is None:
        print("Warning: marginal effect plots skipped (SHAP computation failed).")
        return None

    ranking = _rank_shap_features(
        shap_data["shap_values"], shap_data["non_emb_cols"], shap_data["non_emb_idx"], task=task
    )
    top_features_df = ranking.head(n_features).reset_index(drop=True)
    top_features = top_features_df["feature"].tolist()

    labels_map = get_labels_map(demand_code=_current_demand_code, columns=top_features)

    # Subsample test set for computational efficiency (consistent with SHAP)
    sample_size = min(500, len(X_test))
    sample_pos = np.random.RandomState(seed).choice(len(X_test), size=sample_size, replace=False)

    X_sample    = X_test.iloc[sample_pos].reset_index(drop=True)
    y_sample    = pd.Series(np.asarray(y_test)[sample_pos]).reset_index(drop=True)
    year_sample = pd.Series(np.asarray(y_test_year)[sample_pos]).reset_index(drop=True)

    # Outcome-year stratified bootstrap indices (fixed model, resample test subsample only)
    rng = np.random.default_rng(ci_seed)
    strata = _make_outcome_year_strata(y_sample, year_sample)
    strata_indices = [
        group.index.to_numpy()
        for _, group in strata.groupby(strata, sort=False)
    ]

    boot_idx_list = [
        np.concatenate([
            rng.choice(stratum_idx, size=len(stratum_idx), replace=True)
            for stratum_idx in strata_indices
        ])
        for _ in range(n_bootstrap)
    ]

    def _draw_panel(ax, is_cat, gvals, med, lo, hi, label):
        n_v = len(gvals)
        if is_cat:
            x_pos = np.arange(n_v)
            yerr  = np.vstack([med - lo, hi - med])
            ax.errorbar(x_pos, med, yerr=yerr, fmt='o', color='steelblue',
                        capsize=4, markersize=6, label='Median [95% CI]')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(v) for v in gvals])
            ax.set_xlim(-0.5, n_v - 0.5)
        else:
            ax.plot(gvals, med, color='steelblue', lw=2, label='Median')
            ax.fill_between(gvals, lo, hi, color='steelblue', alpha=0.25,
                            label='95% CI (bootstrap)')
        ax.set_xlabel(label)
        ax.set_ylabel('Predicted probability of positive class')
        ax.set_title(f'Marginal effect: {label}')
        ax.set_ylim(0, 1)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

    _results = []

    for feature in top_features:
        display_label = get_display_label(feature, labels_map)
        col = X_sample[feature]
        original_dtype = col.dtype

        is_categorical = (
            isinstance(original_dtype, pd.CategoricalDtype)
            or col.nunique(dropna=True) <= cat_max_unique
        )

        if is_categorical:
            if isinstance(original_dtype, pd.CategoricalDtype):
                grid_values = list(original_dtype.categories)
            else:
                grid_values = sorted(col.dropna().unique().tolist())
        else:
            lo, hi = np.nanpercentile(col.astype(float), [1, 99])
            grid_values = np.linspace(lo, hi, n_grid_points)
            if pd.api.types.is_integer_dtype(original_dtype):
                grid_values = np.unique(np.round(grid_values).astype(np.int64))

        n_vals = len(grid_values)

        def _assign_feature(df_, values_array, _feature=feature, _dtype=original_dtype):
            if isinstance(_dtype, pd.CategoricalDtype):
                df_[_feature] = pd.Categorical(values_array, dtype=_dtype)
            else:
                df_[_feature] = pd.array(values_array, dtype=_dtype)
            return df_

        boot_means = np.empty((n_vals, n_bootstrap))
        for b, boot_idx in enumerate(boot_idx_list):
            X_boot = X_sample.iloc[boot_idx]

            # Tile the resampled block once per grid value, then make a single
            # predict_proba call covering all grid points at once.
            X_tiled = pd.concat([X_boot] * n_vals, ignore_index=True)
            values_array = np.repeat(grid_values, len(boot_idx))
            X_tiled = _assign_feature(X_tiled, values_array)

            probs = automl.predict_proba(X_tiled)[:, 1]
            probs = probs.reshape(n_vals, len(boot_idx))
            boot_means[:, b] = probs.mean(axis=1)

        medians = np.median(boot_means, axis=1)
        ci_lo   = np.percentile(boot_means, 2.5, axis=1)
        ci_hi   = np.percentile(boot_means, 97.5, axis=1)

        _results.append((display_label, is_categorical, grid_values, medians, ci_lo, ci_hi))

        # Individual plot
        fig_w = max(3.5, n_vals * 1.4 + 1.0) if is_categorical else 8
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        _draw_panel(ax, is_categorical, grid_values, medians, ci_lo, ci_hi, display_label)
        plt.tight_layout()
        save_figure(fig, f'marginal_{feature}.png')
        plt.show()

    # Combined grid figure
    if _results:
        n_plots = len(_results)
        n_rows, n_cols = _grid_dims(n_plots)
        fig_grid, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5, n_rows * 4.5),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        for i, (lbl, is_cat, gvals, med, lo, hi) in enumerate(_results):
            _draw_panel(axes_flat[i], is_cat, gvals, med, lo, hi, lbl)
        for j in range(n_plots, n_rows * n_cols):
            axes_flat[j].set_visible(False)
        plt.tight_layout()
        save_figure(fig_grid, 'marginal_combined.png')
        plt.show()

    print(f"\nTop {len(top_features)} features by mean(|SHAP value|):")
    for _, row in top_features_df.iterrows():
        print(f"  {row['feature']:<20} {row['mean_abs_shap']:.4f}")

    elapsed = time.time() - _start_time
    print(f"Total marginal-effects time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return top_features_df

def run_binary_automl_model(
    cohort_name: str,
    target_column: str,
    demand_code: Optional[Union[int, List[int]]] = None,
    triage_value: Optional[int] = None,
    include_lr: bool = True,
    include_hist: bool = True,
    include_com: bool = True,
    include_medication: bool = True,
    include_embeddings: bool = False,
    export_table: bool = False, 
    time_budget: Optional[int] = 600,
    max_iter: Optional[int] = None,
    fixed_config: Optional[dict] = None,
    fixed_estimator: str = "lgbm",
    test_size: float = 0.2,
    seed: int = 42,
    min_age: Optional[int] = None,
    years: Optional[List[int]] = None,
    sex_group: bool = False,
    optimize_metric: Optional[str] = None,
    n_splits_cv: int = 5,
    optimize_beta: int = 1,
    train_threshold: str = "oof",
    use_clinical_threshold: bool = False,
    use_youden: bool = False,
    max_undertriage: float = 0.10,
    max_overtriage: float = 0.50,
    feature_importance: int = 0,
    builtin_importance_type: str = "gain",
    plot_shap: bool = True,
    shap_sample_size: Optional[int] = None,
    n_interpretability_features: int = 0,
    plot_calibration_curve: bool = True,
    plot_sa_roc_curve: bool = True,
    sa_roc_alpha_pos: float = 0.60,
    sa_roc_alpha_neg: float = 0.90,
    export_cv_fold_metrics: bool = True
) -> dict:

    
    """
    Runs a complete binary classification pipeline using FLAML AutoML.

    Parameters
    ----------
    cohort_name : str
        Name of the cohort (used for logging only).
    target_column : str
        Binary target column.
    demand_code : Optional[Union[int, List[int]]]
        If provided, filters df to demand_type_1.
        Can be a single integer or a list of integers (e.g., [36, 58]).
    triage_value : Optional[int]
        If provided, filters to triage == triage_value.
        If triage_value == 0, q1-q7 are removed from the feature set.
    time_budget : Optional[int]
        FLAML search budget in seconds. Mutually exclusive with max_iter.
        Use time_budget=-1 together with max_iter for an iteration-bounded search.
    max_iter : Optional[int]
        Number of FLAML configurations to evaluate. Mutually exclusive with a
        positive time_budget.
    fixed_config : Optional[dict]
        Explicit hyperparameters to use instead of searching. When supplied,
        FLAML evaluates this single configuration and no search is performed,
        making the execution reproducible and reducing runtime. Use
        FIXED_LGBM_CONFIG for the current stroke hist_ reference
        configuration.
    fixed_estimator : str
        Learner the fixed configuration belongs to. Defaults to "lgbm".
    test_size : float
        Proportion of test split.
    seed : int
        Random seed.
    min_age : Optional[int]
        Optional age filter (>= min_age). If None, no age filtering is applied.
    optimize_metric : Optional[str]
        Metric to optimize during training.
    n_splits_cv : int
        Number of CV folds.
    optimize_beta : int
        Beta for F-beta threshold optimization (beta > 1 favors recall).
    plot_feature_importance : bool
        Whether to compute and plot feature importance.
    use_permutation_importance : bool
        If True, use permutation importance (slower but more accurate).
        If False, use built-in feature importance from the model (faster).
    export_cv_fold_metrics : bool
        Whether to replay CV for the final selected model and export fold metrics.
    train_threshold : str
        Training probabilities used to select the decision threshold.
        Use "oof" for out-of-fold probabilities, or "in-sample" for
        resubstitution probabilities from the final model refitted on all train.

    Returns
    -------
    dict
        Dictionary with model, metrics, threshold and feature importance.
    """
    valid_train_thresholds = {"oof", "in-sample"}
    if train_threshold not in valid_train_thresholds:
        raise ValueError(
            "train_threshold must be one of "
            f"{sorted(valid_train_thresholds)}; got {train_threshold!r}."
        )

    # Register the active demand_code and time_budget so save_figure() knows which folder/suffix to use
    global _current_demand_code, _current_time_budget
    _current_demand_code = demand_code
    _current_time_budget = time_budget if max_iter is None and fixed_config is None else None

    # -----------------------------
    # Print time
    # -----------------------------
    time_print()

    # -----------------------------
    # Print type of triage-specific cohort
    # -----------------------------
    triage_print(triage_value, cohort_name, demand_code)

    # -----------------------------
    # Data loading & column selection
    # -----------------------------
    df = data_load_col_selection(target_column, triage_value,
                                 include_lr, include_hist, include_com, include_medication, include_embeddings)

    # -----------------------------
    # Basic validation
    # -----------------------------
    col_validation(df, target_column)

    # -----------------------------
    # Cohort & triage filtering
    # -----------------------------
    df_filtered = data_filtering(df, target_column, demand_code, triage_value, min_age,
                        years, export_table, protocol_features=protocol_questions)
    subgroup_reference_df = df.loc[df_filtered.index].copy()

    # -----------------------------
    # Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test, year_train, year_test, sample_weight = train_test_split_weights(
        df_filtered, target_column, test_size, seed)

    # -----------------------------
    # AutoML training
    # -----------------------------
    automl = run_automl_training(
        X_train,
        y_train,
        year_train,
        sample_weight,
        time_budget,
        optimize_metric,
        n_splits_cv,
        seed,
        task="classification",
        max_iter=max_iter,
        fixed_config=fixed_config,
        fixed_estimator=fixed_estimator,
    )

    # -----------------------------
    # Threshold optimization (decision threshold)
    # -----------------------------
    # In-sample probabilities are retained for calibration and diagnostic plots.
    y_train_prob = automl.predict_proba(X_train)[:, 1]
    y_test_prob = automl.predict_proba(X_test)[:, 1]
    print_selected_model_hyperparameters(automl)

    if train_threshold == "oof":
        y_train_threshold_prob = compute_oof_probabilities(
            automl,
            X_train,
            y_train,
            year_train,
            n_splits_cv,
            seed,
        )
        threshold_basis = "out-of-fold"
    else:
        y_train_threshold_prob = y_train_prob
        threshold_basis = "in-sample (resubstitution)"

    print(f"\nDecision threshold selected on {threshold_basis} training probabilities.")
    
    if use_clinical_threshold:
        best_threshold = optimize_clinical_threshold(
            y_train, y_train_threshold_prob,
            y_test,  y_test_prob,
            max_undertriage=max_undertriage,
            max_overtriage=max_overtriage,
        )
    elif use_youden:
        best_threshold = compute_youden_threshold(
            y_train, y_train_threshold_prob,
            y_test,  y_test_prob,
        )
    else:
        best_threshold = set_fb_threshold(y_train, y_train_threshold_prob, optimize_beta)

    if train_threshold == "oof":
        m_insample = metrics_at_threshold(y_train, y_train_prob, best_threshold)
        m_oof = metrics_at_threshold(y_train, y_train_threshold_prob, best_threshold)
        print("\n--- Threshold behaviour on training data ---")
        print(f"  In-sample   : Se = {m_insample['sensitivity']:.4f}  Sp = {m_insample['specificity']:.4f}")
        print(f"  Out-of-fold : Se = {m_oof['sensitivity']:.4f}  Sp = {m_oof['specificity']:.4f}")
        print(
            "  Sensitivity optimism (in-sample - OOF): "
            f"{(m_insample['sensitivity'] - m_oof['sensitivity']) * 100:+.2f}pp"
        )

    cv_fold_metrics = (
        export_final_model_cv_fold_metrics(
            automl,
            X_train,
            y_train,
            year_train,
            n_splits_cv,
            seed,
            task="classification",
            threshold=best_threshold,
        )
        if export_cv_fold_metrics else pd.DataFrame()
    )
    
    # -----------------------------
    # Test set evaluation
    # -----------------------------
    metrics = print_test_metrics_with_ci(y_test, y_test_prob, year_test, threshold=best_threshold)
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    # Calibration plot
    if plot_calibration_curve:
        plot_calibration(y_train, y_train_prob, y_test, y_test_prob, threshold=best_threshold)

    # Operational safety zones (SA-ROC)
    if plot_sa_roc_curve:
        tau_safe_pos, tau_safe_neg, pct_gray, gamma_area, zone_stats = evaluate_sa_roc(
            y_test, y_test_prob, alpha_pos=sa_roc_alpha_pos, alpha_neg=sa_roc_alpha_neg
        )
        plot_sa_roc(y_test, y_test_prob, tau_safe_pos, tau_safe_neg, zone_stats=zone_stats)

    # Train vs Test comparison — overfitting / underfitting diagnosis
    comparison_df = print_train_test_comparison(
        automl, X_train, y_train, X_test, y_test,
        threshold=best_threshold,
        pr_auc_diff_threshold=0.05,
        f1_diff_threshold=0.05,
        f2_diff_threshold=0.05,
        overfit_gap_threshold=0.05,
    )

    # Plot test metrics
    # Create an evaluation dataframe by mapping back to the filtered data
    plot_df = df.loc[y_test.index].copy() 
    plot_df['target_real'] = y_test
    plot_df['target_pred_model'] = y_test_pred
    plot_df['target_prob_model'] = y_test_prob
    plot_df['target_pred'] = plot_df['target_pred_model']
    model_overall_df = eda.evaluate_diagnostic_performance(
        plot_df, 'target_real', 'target_pred_model',
        sex_group=False,
        figures_dir=_get_figures_dir(demand_code),
        reference_df=subgroup_reference_df,
        export_tables=True,
        evaluation_label="Predicted by the model",
        filename_suffix="_model",
        bootstrap_year_col="year",
    )
    # Fairness analysis by sex
    sex_overall_df = eda.evaluate_diagnostic_performance(
        plot_df, 'target_real', 'target_pred_model',
        sex_group=True,
        figures_dir=_get_figures_dir(demand_code),
        reference_df=subgroup_reference_df,
        export_tables=True,
        evaluation_label="Predicted by the model",
        filename_suffix="_model",
        bootstrap_year_col="year",
    )
    # Fairness analysis by age groups
    age_overall_df = eda.evaluate_diagnostic_performance(
        plot_df, 'target_real', 'target_pred_model',
        age_group=True,
        figures_dir=_get_figures_dir(demand_code),
        reference_df=subgroup_reference_df,
        export_tables=True,
        evaluation_label="Predicted by the model",
        filename_suffix="_model",
        bootstrap_year_col="year",
    )
    plot_subgroup_fairness_heatmap(
        age_df=age_overall_df,
        sex_df=sex_overall_df,
        figures_dir=_get_figures_dir(demand_code),
        filename='subgroup_fairness_heatmap_model.png',
        color_mode='signed_ci',
    )
    model_vs_triage = None
    common_subset_report = None
    model_common_df = None
    triage_overall_df = None
    if "p1_assigned" in plot_df.columns:
        triage_plot_df, common_subset_report = get_common_comparison_subset(
            plot_df,
            y_col="target_real",
            model_col="target_pred_model",
            triage_col="p1_assigned",
        )

        print("\n--- Telephonic triage system diagnostic performance on test set ---")
        print(f"Rows evaluated: {len(triage_plot_df):,} / {len(plot_df):,} test rows")
        if len(triage_plot_df) != len(plot_df):
            print(
                "WARNING: p1_assigned has missing values; confusion comparison "
                f"uses the common subset for both matrices "
                f"(model test N={len(plot_df):,}, common N={len(triage_plot_df):,})."
            )
        plot_confusion_comparison(
            y_true=triage_plot_df['target_real'].astype(int).values,
            y_pred_triage=triage_plot_df['p1_assigned'].astype(int).values,
            y_pred_model=triage_plot_df['target_pred_model'].astype(int).values,
            figures_dir=_get_figures_dir(demand_code),
            filename='confusion_comparison.png',
            normalize='row',
        )
        model_common_df = eda.evaluate_diagnostic_performance(
            triage_plot_df, "target_real", "target_pred_model",
            sex_group=False,
            figures_dir=_get_figures_dir(demand_code),
            reference_df=subgroup_reference_df,
            export_tables=True,
            evaluation_label="Predicted by the model (common subset)",
            filename_suffix="_model_common",
            bootstrap_year_col="year",
        )
        triage_overall_df = eda.evaluate_diagnostic_performance(
            triage_plot_df, "target_real", "p1_assigned",
            sex_group=False,
            figures_dir=_get_figures_dir(demand_code),
            reference_df=subgroup_reference_df,
            export_tables=True,
            evaluation_label="Assigned by the telephonic triage system",
            filename_suffix="_telephonic_triage",
            bootstrap_year_col="year",
        )
        plot_metrics_comparison(
            triage_overall_df=triage_overall_df,
            model_overall_df=model_common_df,
            figures_dir=_get_figures_dir(demand_code),
            filename='metrics_comparison.png',
        )
        model_vs_triage = compare_model_vs_triage_common(
            triage_plot_df,
            figures_dir=_get_figures_dir(demand_code),
            export_tables=True,
            run_inference=True,
            n_bootstrap=1000,
            seed=42,
            model_full_test_df=plot_df,
            train_threshold=train_threshold,
            y_train=y_train,
            y_train_threshold_prob=y_train_threshold_prob,
            train_triage=df.loc[y_train.index, "p1_assigned"],
            year_train=year_train,
            primary_threshold=best_threshold,
        )
    else:
        print("\nWARNING: p1_assigned not found; skipping telephonic triage system evaluation.")

    # -----------------------------
    # Feature importance
    # -----------------------------
    importance_df_permutation, importance_df_builtin = plot_feature_importances(
        automl, X_train, feature_importance, X_test, y_test, seed,
        task="classification", builtin_importance_type=builtin_importance_type)
    effective_features_df = report_effective_features(automl, X_train)

    # -----------------------------
    # SHAP plot & interpretability (marginal effect) plots
    # -----------------------------
    shap_data = None
    shap_values = None
    shap_summary = None
    interpretability_table = None
    if plot_shap or n_interpretability_features > 0:
        shap_data = _compute_shap_values(
            automl, X_test, task="classification", seed=seed,
            sample_size=shap_sample_size,
        )
        if shap_data is not None:
            shap_summary = _summarise_shap(shap_data)

    if all(result is not None for result in [
        importance_df_permutation,
        importance_df_builtin,
        effective_features_df,
        shap_summary,
    ]):
        interpretability_table = export_interpretability_table(
            importance_df_permutation,
            importance_df_builtin,
            effective_features_df,
            shap_summary,
            demand_code,
        )

    if plot_shap:
        shap_values = plot_shap_interpretation(
            automl, X_test, task="classification", seed=seed, shap_data=shap_data
        )

    top_interpretability_features = None
    if n_interpretability_features > 0:
        top_interpretability_features = plot_marginal_effects(
            automl, X_test, y_test, year_test,
            shap_data=shap_data,
            n_features=n_interpretability_features,
            task="classification",
            seed=seed,
        )

    # -----------------------------
    # Build summary DataFrame (one-row, ready to concatenate)
    # -----------------------------
    summary_df = (
        pd.DataFrame(metrics, index=[0])
        .assign(
            cohort=cohort_name,
            demand_code=str(demand_code) if demand_code is not None else None,  # Convert to string for consistency
            triage_value=triage_value,
            optimize_metric=optimize_metric,
            optimize_beta=optimize_beta,
            threshold=best_threshold,
            train_threshold=train_threshold,
            search_mode=getattr(
                automl,
                "search_mode_",
                (
                    "fixed_config" if fixed_config is not None
                    else "max_iter" if max_iter is not None
                    else "time_budget"
                ),
            ),
            search_budget=getattr(
                automl,
                "search_budget_",
                (
                    "fixed" if fixed_config is not None
                    else max_iter if max_iter is not None
                    else time_budget
                ),
            ),
            search_iterations=getattr(automl, "search_iterations_", None),
            search_elapsed_time=getattr(automl, "search_elapsed_time_", None),
            n_samples=len(df_filtered),
            n_train=len(X_train),
            n_test=len(X_test),
        )
    )

    display(summary_df)

    return ModelingResult({
        "automl": automl,
        "metrics": metrics,
        "threshold": best_threshold,
        "search_iterations": getattr(automl, "search_iterations_", None),
        "search_elapsed_time": getattr(automl, "search_elapsed_time_", None),
        "importance_permutation": importance_df_permutation,
        "importance_builtin": importance_df_builtin,
        "effective_features": effective_features_df,
        "shap_values": shap_values,
        "shap_summary": shap_summary,
        "interpretability_table": interpretability_table,
        "top_interpretability_features": top_interpretability_features,
        "cv_fold_metrics": cv_fold_metrics,
        "summary": summary_df,
        "model_overall_performance": model_overall_df,
        "model_common_performance": model_common_df,
        "triage_performance": triage_overall_df,
        "model_vs_triage": model_vs_triage,
        "paired_inference": (
            model_vs_triage.get("paired_inference")
            if model_vs_triage is not None else None
        ),
        "common_subset_report": common_subset_report,
    })
