import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, accuracy_score
from IPython.display import display

from kbase.config import settings

def evaluate_diagnostic_performance(
    df, y_real_col, y_pred_col,
    sex_group=False, age_group=False, triage_group=False,
    figures_dir=None, reference_df=None,
    n_bootstrap=1000, ci_seed=42,
    export_tables=False,
):
    """
    Calculates clinical metrics with 95% confidence intervals and generates plots.

    Clopper-Pearson exact binomial CIs are used for proportion-based metrics:
        Accuracy, Precision, Recall, Specificity, NPV, Overtriage, Undertriage.

    Outcome-stratified bootstrap CIs (n_bootstrap resamples, percentile method)
    are used for composite metrics:
        F1-Score, F2-Score.

    If triage_group=True, results are stratified by triage status
    (General, Triage 1, No Triage 0).
    If sex_group=True, it performs a cross-stratification: each group above is
    subdivided by Men and Women.
    If age_group=True, it performs a cross-stratification: each group above is
    subdivided by WHO age groups (derived from the age_0_14 ... age_75_plus
    indicators), keeping only groups with N > 1000.
    The N > 1000 threshold is evaluated on `reference_df` if provided (e.g. the
    full dataset, before any train/test split), otherwise on `df` itself.
    sex_group and age_group are mutually exclusive subgroup dimensions; if both
    are True, age_group takes precedence.
    If sex_group, age_group, and triage_group are all False, a single overall
    result ("General") is computed with no group breakdown.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples for F1/F2 confidence intervals.
    ci_seed : int
        Random seed for reproducibility of bootstrap sampling.
    export_tables : bool
        When True and figures_dir is provided, exports the formatted performance
        table to a Word (.docx) file under figures_dir/tables/.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score
    from scipy.stats import beta as _beta

    df = df.copy()

    # --- BLOCK: Dynamic Triage Column Creation ---
    if triage_group and 'triage' not in df.columns:
        question_cols = [c for c in df.columns if c[0].isdigit() or (c.startswith('q') and len(c) > 1 and c[1].isdigit())]

        if question_cols:
            is_no_triage = df[question_cols].apply(lambda x: (x == -1).all(), axis=1)
            df['triage'] = np.where(is_no_triage, 0, 1)
            print(f"✔ 'triage' column created based on {len(question_cols)} question columns.")
        else:
            df['triage'] = 1
            print("⚠️ No question columns found. Defaulting to triage=1.")

    # --- BLOCK: Subgroup Setup (sex_group / age_group are mutually exclusive) ---
    subgroup_col = None
    subgroup_values = []

    if sex_group and 'sex' in df.columns:
        # Direct mapping for category types (0: Men, 1: Women)
        df['sex_label'] = df['sex'].map({0: 'Men', 1: 'Women'})
        subgroup_col = 'sex_label'
        subgroup_values = ['Men', 'Women']

    if age_group:
        # WHO age-group indicators -> single categorical label column
        age_group_cols = {
            "age_0_14":    "Children",
            "age_15_24":   "Youth",
            "age_25_44":   "Young Adults",
            "age_45_59":   "Middle-aged Adults",
            "age_60_74":   "Elderly",
            "age_75_plus": "Seniors",
        }
        present_cols = [c for c in age_group_cols if c in df.columns]
        if present_cols:
            df['age_group_label'] = df[present_cols].idxmax(axis=1).map(age_group_cols)

            # Only keep age groups with more than 1000 cases in the reference dataset
            # (defaults to the full stroke cohort, so the threshold doesn't depend on the
            # train/test split size)
            if reference_df is not None:
                count_source = reference_df
            else:
                count_source = df
            count_labels = count_source[present_cols].idxmax(axis=1).map(age_group_cols)
            counts = count_labels.value_counts()
            age_group_order = [age_group_cols[c] for c in present_cols]
            subgroup_col = 'age_group_label'
            subgroup_values = [g for g in age_group_order if counts.get(g, 0) > 1000]

    # --- CI helpers ----------------------------------------------------------
    def _cp(x, n, alpha=0.05):
        """Clopper-Pearson exact binomial 95% CI. Returns (lo, hi) as proportions."""
        if n == 0:
            return (np.nan, np.nan)
        lo = float(_beta.ppf(alpha / 2,     x,     n - x + 1)) if x > 0 else 0.0
        hi = float(_beta.ppf(1 - alpha / 2, x + 1, n - x))     if x < n else 1.0
        return lo, hi

    # Single RNG shared across all subgroup calls for reproducibility.
    rng = np.random.default_rng(ci_seed)

    # --- BLOCK: Internal Metric Calculation Helper ---
    def calculate_metrics(data, group_name, subgroup_name="General"):
        if len(data) == 0:
            return None

        y_real = data[y_real_col]
        y_pred = data[y_pred_col]

        cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        n = int(tn + fp + fn + tp)

        # --- Point estimates -------------------------------------------------
        accuracy    = (tp + tn) / n * 100
        precision   = (tp / (tp + fp) if (tp + fp) > 0 else 0) * 100
        recall      = (tp / (tp + fn) if (tp + fn) > 0 else 0) * 100
        specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0) * 100
        npv         = (tn / (tn + fn) if (tn + fn) > 0 else 0) * 100
        f1          = f1_score(y_real, y_pred, zero_division=0) * 100
        f2          = fbeta_score(y_real, y_pred, beta=2, zero_division=0) * 100
        overtriage  = 100 - precision
        undertriage = 100 - npv

        # --- Clopper-Pearson CIs (exact binomial, no resampling) -------------
        # Overtriage = FP/(TP+FP) and Undertriage = FN/(TN+FN) are also
        # direct binomial proportions, so CP applies directly to them.
        acc_ci   = tuple(v * 100 for v in _cp(int(tp + tn), n))
        prec_ci  = tuple(v * 100 for v in _cp(int(tp),      int(tp + fp)))
        rec_ci   = tuple(v * 100 for v in _cp(int(tp),      int(tp + fn)))
        spec_ci  = tuple(v * 100 for v in _cp(int(tn),      int(tn + fp)))
        npv_ci   = tuple(v * 100 for v in _cp(int(tn),      int(tn + fn)))
        over_ci  = tuple(v * 100 for v in _cp(int(fp),      int(tp + fp)))
        under_ci = tuple(v * 100 for v in _cp(int(fn),      int(tn + fn)))

        # --- Outcome-stratified bootstrap CIs for F1 and F2 -----------------
        # Simple outcome (0/1) stratification — preserves class balance in each
        # resample without requiring a year column. More robust than outcome-year
        # stratification for small or heterogeneous subgroups.
        y_real_arr = np.asarray(y_real)
        y_pred_arr = np.asarray(y_pred)
        neg_idx = np.where(y_real_arr == 0)[0]
        pos_idx = np.where(y_real_arr == 1)[0]

        boot_f1, boot_f2 = [], []
        n_valid = 0

        if len(neg_idx) >= 2 and len(pos_idx) >= 2:
            for _ in range(n_bootstrap):
                idx = np.concatenate([
                    rng.choice(neg_idx, size=len(neg_idx), replace=True),
                    rng.choice(pos_idx, size=len(pos_idx), replace=True),
                ])
                yt = y_real_arr[idx]
                yp = y_pred_arr[idx]

                tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
                prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else np.nan
                rec_b  = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else np.nan

                if not (np.isnan(prec_b) or np.isnan(rec_b)):
                    denom_f1 = prec_b + rec_b
                    f1_b = 2 * prec_b * rec_b / denom_f1 if denom_f1 > 0 else np.nan
                    denom_f2 = 4 * prec_b + rec_b
                    f2_b = 5 * prec_b * rec_b / denom_f2 if denom_f2 > 0 else np.nan
                else:
                    f1_b = f2_b = np.nan

                if not np.isnan(f1_b):
                    boot_f1.append(f1_b)
                    n_valid += 1
                if not np.isnan(f2_b):
                    boot_f2.append(f2_b)

            if n_valid < n_bootstrap // 2:
                print(
                    f"Warning [{group_name} / {subgroup_name}]: only {n_valid}/{n_bootstrap} "
                    f"valid bootstrap replications — F1/F2 CI may be unreliable."
                )

        def _boot_ci(boot_list):
            if len(boot_list) >= 10:
                return (
                    float(np.percentile(boot_list, 2.5))  * 100,
                    float(np.percentile(boot_list, 97.5)) * 100,
                )
            return (np.nan, np.nan)

        f1_ci = _boot_ci(boot_f1)
        f2_ci = _boot_ci(boot_f2)

        def _r(v):
            return round(float(v), 2) if not np.isnan(float(v)) else np.nan

        return {
            "Triage Group":   group_name,
            "Subgroup":       subgroup_name,
            "Display Group":  f"{group_name} ({subgroup_name})" if subgroup_name != "General" else group_name,
            "Accuracy (%)":          _r(accuracy),
            "Accuracy CI lo (%)":    _r(acc_ci[0]),   "Accuracy CI hi (%)":    _r(acc_ci[1]),
            "Precision (%)":         _r(precision),
            "Precision CI lo (%)":   _r(prec_ci[0]),  "Precision CI hi (%)":   _r(prec_ci[1]),
            "Recall (%)":            _r(recall),
            "Recall CI lo (%)":      _r(rec_ci[0]),   "Recall CI hi (%)":      _r(rec_ci[1]),
            "F1-Score (%)":          _r(f1),
            "F1-Score CI lo (%)":    _r(f1_ci[0]),    "F1-Score CI hi (%)":    _r(f1_ci[1]),
            "F2-Score (%)":          _r(f2),
            "F2-Score CI lo (%)":    _r(f2_ci[0]),    "F2-Score CI hi (%)":    _r(f2_ci[1]),
            "Overtriage (%)":        _r(overtriage),
            "Overtriage CI lo (%)":  _r(over_ci[0]),  "Overtriage CI hi (%)":  _r(over_ci[1]),
            "Undertriage (%)":       _r(undertriage),
            "Undertriage CI lo (%)": _r(under_ci[0]), "Undertriage CI hi (%)": _r(under_ci[1]),
            "Specificity (%)":       _r(specificity),
            "Specificity CI lo (%)": _r(spec_ci[0]),  "Specificity CI hi (%)": _r(spec_ci[1]),
            "NPV (%)":               _r(npv),
            "NPV CI lo (%)":         _r(npv_ci[0]),   "NPV CI hi (%)":         _r(npv_ci[1]),
            "N": n,
        }

    # --- BLOCK: Cross-Stratified Grouping Logic ---
    results = []

    if triage_group:
        unique_triage_values = df['triage'].unique()

        if len(unique_triage_values) > 1:
            res_total = calculate_metrics(df, "General", "General")
            if res_total: results.append(res_total)

            if subgroup_col:
                for s_val in subgroup_values:
                    sub_df = df[df[subgroup_col] == s_val]
                    if len(sub_df) > 0:
                        res_sub = calculate_metrics(sub_df, "General", s_val)
                        if res_sub: results.append(res_sub)

        for t_val in [1, 0]:
            t_df = df[df['triage'] == t_val]
            if len(t_df) == 0: continue

            label = "Triage (1)" if t_val == 1 else "No Triage (0)"
            res_triage = calculate_metrics(t_df, label, "General")
            if res_triage: results.append(res_triage)

            if subgroup_col:
                for s_val in subgroup_values:
                    sub_df = t_df[t_df[subgroup_col] == s_val]
                    if len(sub_df) > 0:
                        res_sub = calculate_metrics(sub_df, label, s_val)
                        if res_sub: results.append(res_sub)
    else:
        # No triage stratification: a single overall ("General") result,
        # optionally subdivided by sex or age group.
        res_total = calculate_metrics(df, "General", "General")
        if res_total: results.append(res_total)

        if subgroup_col:
            for s_val in subgroup_values:
                sub_df = df[df[subgroup_col] == s_val]
                if len(sub_df) > 0:
                    res_sub = calculate_metrics(sub_df, "General", s_val)
                    if res_sub: results.append(res_sub)

    results_df = pd.DataFrame(results)

    # --- Formatted display: each metric shown as "estimate (lo–hi)" ----------
    _METRIC_COLS = [
        "Accuracy", "Precision", "Recall", "F1-Score", "F2-Score",
        "Overtriage", "Undertriage", "Specificity", "NPV",
    ]

    def _fmt(row, metric):
        est = row[f"{metric} (%)"]
        lo  = row.get(f"{metric} CI lo (%)", np.nan)
        hi  = row.get(f"{metric} CI hi (%)", np.nan)
        if pd.isna(lo) or pd.isna(hi):
            return f"{est:.2f}"
        return f"{est:.2f} ({lo:.2f}–{hi:.2f})"

    display_rows = []
    for _, row in results_df.iterrows():
        d = {
            "Triage Group": row["Triage Group"],
            "Subgroup":     row["Subgroup"],
        }
        for m in _METRIC_COLS:
            d[f"{m} (%)"] = _fmt(row, m)
        d["N"] = f"{int(row['N']):,}"
        display_rows.append(d)

    from IPython.display import display as _display
    _display(pd.DataFrame(display_rows))

    # --- Word export ---------------------------------------------------------
    if export_tables and figures_dir:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.section import WD_ORIENT
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement

        # Grayscale palette: dark header, alternating white / light-grey rows
        HDR_BG  = "2D2D2D"
        HDR_FG  = RGBColor(0xFF, 0xFF, 0xFF)
        ALT_BG  = "EFEFEF"
        EVEN_BG = "FFFFFF"

        def _set_cell_bg(cell, hex_color):
            tc   = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd  = OxmlElement('w:shd')
            shd.set(qn('w:val'),   'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'),  hex_color)
            tcPr.append(shd)

        if age_group:
            table_title = "Diagnostic performance by WHO age group (95% CI)"
            fname       = "performance_age_subgroups.docx"
        elif sex_group:
            table_title = "Diagnostic performance by sex subgroup (95% CI)"
            fname       = "performance_sex_subgroups.docx"
        else:
            table_title = "Overall diagnostic performance (95% CI)"
            fname       = "performance_overall.docx"

        footnote = (
            "Accuracy, Precision, Recall, Specificity, NPV, Overtriage and Undertriage: "
            "Clopper-Pearson exact 95% CI. "
            f"F1-Score and F2-Score: outcome-stratified bootstrap 95% CI ({n_bootstrap} resamples). "
            "Values shown as estimate (95% CI lower–upper), all in %."
        )

        display_df = pd.DataFrame(display_rows)
        cols       = list(display_df.columns)
        n_data_rows = len(display_df)

        doc = Document()
        # Landscape page so the wide table fits without wrapping
        section = doc.sections[0]
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width

        style = doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(10)

        title_para = doc.add_paragraph()
        title_run  = title_para.add_run(table_title)
        title_run.bold = True
        title_run.font.size = Pt(11)
        doc.add_paragraph()

        table = doc.add_table(rows=n_data_rows + 1, cols=len(cols))
        table.style = 'Table Grid'

        # Header row
        for j, col_name in enumerate(cols):
            cell = table.rows[0].cells[j]
            cell.text = col_name
            run  = cell.paragraphs[0].runs[0]
            run.bold = True
            run.font.color.rgb = HDR_FG
            run.font.size = Pt(8)
            _set_cell_bg(cell, HDR_BG)

        # Data rows
        for i, row_vals in enumerate(display_df.itertuples(index=False)):
            bg = ALT_BG if i % 2 == 1 else EVEN_BG
            for j, val in enumerate(row_vals):
                cell = table.rows[i + 1].cells[j]
                cell.text = str(val)
                run  = cell.paragraphs[0].runs[0]
                run.font.size = Pt(8)
                _set_cell_bg(cell, bg)

        doc.add_paragraph()
        note_para = doc.add_paragraph()
        note_run  = note_para.add_run(footnote)
        note_run.italic = True
        note_run.font.size = Pt(8)

        tables_dir = os.path.join(figures_dir, 'tables')
        os.makedirs(tables_dir, exist_ok=True)
        word_path = os.path.join(tables_dir, fname)
        doc.save(word_path)
        print(f"Table saved: {word_path}")

    # --- BLOCK: Visualization (2x4 Grid) ---
    # Reordered metrics as requested:
    # Row 1: Accuracy, Recall, Precision, Specificity
    # Row 2: F1-Score, F2-Score, Overtriage, Undertriage
    metrics_to_plot = [
        "Accuracy (%)", "Recall (%)", "Precision (%)", "Specificity (%)",
        "F1-Score (%)", "F2-Score (%)", "Overtriage (%)", "Undertriage (%)"
    ]
    metric_y_axis_max = 100

    # 2 rows, 4 columns = 8 slots
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")

    c_blue = "#4A90E2"
    c_teal = "#50E3C2"
    c_orange = "#F5A623"

    if sex_group:
        palette_dict = {"General": c_blue, "Men": c_teal, "Women": c_orange}
        hue_col = "Subgroup"
        legend_title = "Sex Subgroup"
        label_size = 16
        fmt = ".1f"
    elif age_group:
        age_palette = sns.color_palette("Set2", n_colors=len(subgroup_values))
        palette_dict = {"General": c_blue, **dict(zip(subgroup_values, age_palette))}
        hue_col = "Subgroup"
        legend_title = "Age Subgroup"
        label_size = 16
        fmt = ".1f"
    elif triage_group:
        palette_dict = {"General": c_blue, "Triage (1)": c_teal, "No Triage (0)": c_orange}
        hue_col = "Triage Group"
        legend_title = None
        label_size = 18
        fmt = ".2f"
    else:
        palette_dict = None
        hue_col = None
        legend_title = None
        label_size = 18
        fmt = ".2f"

    for i in range(len(axes)):
        if i >= len(metrics_to_plot):
            axes[i].axis('off')
            continue

        metric = metrics_to_plot[i]

        if hue_col is not None:
            sns.barplot(
                data=results_df,
                x="Triage Group",
                y=metric,
                hue=hue_col,
                ax=axes[i],
                palette=palette_dict,
                legend=((sex_group or age_group) and i == 0)
            )
        else:
            sns.barplot(
                data=results_df,
                x="Triage Group",
                y=metric,
                ax=axes[i],
                color=c_blue,
                legend=False,
            )

        axes[i].set_title(f"{metric}", fontweight='bold', fontsize=20)
        axes[i].set_ylim(0, metric_y_axis_max)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Value (%)", fontsize=15)
        axes[i].tick_params(axis='both', labelsize=14)

        if not triage_group:
            # Single "General" category: the x-tick label is redundant
            # (sex_group=True / age_group=True already clarifies via the legend).
            axes[i].set_xticks([])

        if (sex_group or age_group) and i == 0:
            axes[i].legend(title=legend_title, loc='upper right', frameon=True,
                           fontsize=14, title_fontsize=15)

        for p in axes[i].patches:
            height = p.get_height()
            if height > 0:
                axes[i].annotate(f'{height:{fmt}}%',
                                (p.get_x() + p.get_width() / 2., height),
                                ha = 'center', va = 'center',
                                xytext = (0, 10),
                                textcoords = 'offset points',
                                fontsize=label_size,
                                fontweight='bold')

    plt.tight_layout()
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        suffix = '_age' if age_group else '_sex' if sex_group else ''
        path = os.path.join(figures_dir, f'metrics_barplots_2x4{suffix}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {path}")
    plt.show()

    # --- BLOCK: Combined Single-Plot Visualization (sex_group=False and age_group=False only) ---
    # Aggregates all 8 metrics into a single, publication-ready bar chart.
    if not sex_group and not age_group:
        metric_labels = [m.replace(" (%)", "") for m in metrics_to_plot]
        error_metrics = {"Overtriage", "Undertriage"}

        melted = results_df.melt(
            id_vars=["Triage Group"],
            value_vars=metrics_to_plot,
            var_name="Metric",
            value_name="Value (%)",
        )
        melted["Metric"] = melted["Metric"].str.replace(" (%)", "", regex=False)

        fig2, ax2 = plt.subplots(figsize=(16, 8))

        if triage_group:
            sns.barplot(
                data=melted, x="Metric", y="Value (%)", hue="Triage Group",
                order=metric_labels, hue_order=["General", "Triage (1)", "No Triage (0)"],
                palette=palette_dict, ax=ax2,
            )
            ax2.legend(title="Patient group", loc="upper right", frameon=True,
                       fontsize=12, title_fontsize=13)
            title = "Summary of diagnostic performance metrics, by patient group"
        else:
            c_perf, c_error = "#4A90E2", "#5CB85C"
            bar_palette = {m: (c_error if m in error_metrics else c_perf) for m in metric_labels}
            sns.barplot(
                data=melted, x="Metric", y="Value (%)", hue="Metric",
                order=metric_labels, palette=bar_palette, dodge=False,
                legend=False, ax=ax2,
            )
            legend_handles = [
                Patch(facecolor=c_perf, edgecolor="black",
                      label="Diagnostic performance metric (higher is better)"),
                Patch(facecolor=c_error, edgecolor="black",
                      label="Triage error rate (lower is better)"),
            ]
            ax2.legend(handles=legend_handles, title="Metric type", loc="upper right",
                       frameon=True, fontsize=12, title_fontsize=13)
            title = "Summary of diagnostic performance metrics"

        for p in ax2.patches:
            height = p.get_height()
            if height > 0:
                ax2.annotate(f'{height:.2f}%',
                              (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='center', xytext=(0, 10),
                              textcoords='offset points', fontsize=13, fontweight='bold')

        ax2.set_title(title, fontweight='bold', fontsize=20)
        ax2.set_xlabel("")
        ax2.set_ylabel("Value (%)", fontsize=15)
        ax2.set_ylim(0, metric_y_axis_max)
        ax2.tick_params(axis='both', labelsize=14)
        plt.setp(ax2.get_xticklabels(), rotation=0, ha='center')

        plt.tight_layout()
        if figures_dir:
            os.makedirs(figures_dir, exist_ok=True)
            path2 = os.path.join(figures_dir, 'metrics_barplots_combined.png')
            fig2.savefig(path2, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {path2}")
        plt.show()

    return results_df

def analyze_emergency_performance(demand_type, y_real_col, y_pred_col, sex_group=False, age_group=False):
    """
    High-level function to load data and call evaluation.
    This function no longer returns specific objects to avoid redundant printing.
    """
    # Load the complete preprocessed triage table and filter by demand below.
    # Demand-specific exports may already be target-filtered, which would
    # undercount outcomes such as p1_real_emerg_bps.
    df = pq.read_table(
        os.path.join(settings.source_tables_path, settings.triaje_table_cleaned_path)
    ).to_pandas()

    # Mapping and filtering
    demand_mapping = {
        "Unconscious/Cardiac arrest": [36, 58],
        "Non-traumatic Chest Pain": [23],
        "Dyspnea": [16],
        "Stroke": [54]
    }

    target_codes = demand_mapping.get(demand_type)
    if target_codes is None:
        print(f"Error: Demand type '{demand_type}' not recognized.")
        return

    df = df[df['demand_type_1'].isin(target_codes)]
    reference_df = df.copy()
    # Filter by valid values in target_column
    df = df.dropna(subset=[y_real_col])

    print(f"\nNumber of rows in {demand_type} is: {len(df):,}\n")

    # Preprocessing
    df = df.copy()
    if y_pred_col in df.columns and df[y_pred_col].dtype.name == 'category':
        df[y_pred_col] = pd.to_numeric(df[y_pred_col], errors='coerce')

    df[y_pred_col] = df[y_pred_col].astype('int8')
    df[y_real_col] = df[y_real_col].astype('int8')

    # Execute evaluation and plotting
    evaluate_diagnostic_performance(
        df, y_real_col, y_pred_col,
        sex_group=sex_group,
        age_group=age_group,
        reference_df=reference_df,
    )

# Example usage in a cell:
# analyze_emergency_performance("Non-traumatic Chest Pain")
