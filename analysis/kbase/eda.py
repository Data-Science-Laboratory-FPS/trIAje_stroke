import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, accuracy_score
from IPython.display import display

from kbase.config import settings

def evaluate_diagnostic_performance(df, y_real_col, y_pred_col, sex_group=False, triage_group=False, figures_dir=None):
    """
    Calculates clinical metrics and generates plots.
    If triage_group=True, results are stratified by triage status (General, Triage 1, No Triage 0).
    If sex_group=True, it performs a cross-stratification: each group above is subdivided by Men and Women.
    If both are False, a single overall result ("General") is computed, with no group breakdown.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score

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

    if sex_group and 'sex' in df.columns:
        # Direct mapping for category types (0: Men, 1: Women)
        df['sex_label'] = df['sex'].map({0: 'Men', 1: 'Women'})

    # --- BLOCK: Internal Metric Calculation Helper ---
    def calculate_metrics(data, group_name, subgroup_name="General"):
        if len(data) == 0:
            return None
        
        y_real = data[y_real_col]
        y_pred = data[y_pred_col]
        
        cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = accuracy_score(y_real, y_pred) * 100
        precision = (tp / (tp + fp) if (tp + fp) > 0 else 0) * 100
        recall = (tp / (tp + fn) if (tp + fn) > 0 else 0) * 100
        specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0) * 100
        npv = (tn / (tn + fn) if (tn + fn) > 0 else 0) * 100
        f1 = f1_score(y_real, y_pred, zero_division=0) * 100
        f2 = fbeta_score(y_real, y_pred, beta=2, zero_division=0) * 100

        overtriage = 100 - precision
        undertriage = 100 - npv
        
        return {
            "Triage Group": group_name,
            "Sex": subgroup_name,
            "Display Group": f"{group_name} ({subgroup_name})" if subgroup_name != "General" else group_name,
            "Accuracy (%)": round(accuracy, 2),
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1-Score (%)": round(f1, 2),
            "F2-Score (%)": round(f2, 2),
            "Overtriage (%)": round(overtriage, 2),
            "Undertriage (%)": round(undertriage, 2),
            "Specificity (%)": round(specificity, 2),
            "NPV (%)": round(npv, 2),
            "N": len(data)
        }

    # --- BLOCK: Cross-Stratified Grouping Logic ---
    results = []

    if triage_group:
        unique_triage_values = df['triage'].unique()

        if len(unique_triage_values) > 1:
            res_total = calculate_metrics(df, "General", "General")
            if res_total: results.append(res_total)

            if sex_group and 'sex_label' in df.columns:
                for s_val in ['Men', 'Women']:
                    sex_df = df[df['sex_label'] == s_val]
                    if len(sex_df) > 0:
                        res_sex = calculate_metrics(sex_df, "General", s_val)
                        if res_sex: results.append(res_sex)

        for t_val in [1, 0]:
            t_df = df[df['triage'] == t_val]
            if len(t_df) == 0: continue

            label = "Triage (1)" if t_val == 1 else "No Triage (0)"
            res_triage = calculate_metrics(t_df, label, "General")
            if res_triage: results.append(res_triage)

            if sex_group and 'sex_label' in df.columns:
                for s_val in ['Men', 'Women']:
                    sex_df = t_df[t_df['sex_label'] == s_val]
                    if len(sex_df) > 0:
                        res_sex = calculate_metrics(sex_df, label, s_val)
                        if res_sex: results.append(res_sex)
    else:
        # No triage stratification: a single overall ("General") result,
        # optionally subdivided by sex.
        res_total = calculate_metrics(df, "General", "General")
        if res_total: results.append(res_total)

        if sex_group and 'sex_label' in df.columns:
            for s_val in ['Men', 'Women']:
                sex_df = df[df['sex_label'] == s_val]
                if len(sex_df) > 0:
                    res_sex = calculate_metrics(sex_df, "General", s_val)
                    if res_sex: results.append(res_sex)

    results_df = pd.DataFrame(results)
    
    from IPython.display import display
    display(results_df.drop(columns=["Display Group"]))
    
    # --- BLOCK: Visualization (2x4 Grid) ---
    # Reordered metrics as requested:
    # Row 1: Accuracy, Recall, Precision, Specificity
    # Row 2: F1-Score, F2-Score, Overtriage, Undertriage
    metrics_to_plot = [
        "Accuracy (%)", "Recall (%)", "Precision (%)", "Specificity (%)",
        "F1-Score (%)", "F2-Score (%)", "Overtriage (%)", "Undertriage (%)"
    ]
    
    # 2 rows, 4 columns = 8 slots
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")
    
    c_blue = "#4A90E2"
    c_teal = "#50E3C2"
    c_orange = "#F5A623"

    if sex_group:
        palette_dict = {"General": c_blue, "Men": c_teal, "Women": c_orange}
        hue_col = "Sex"
        label_size = 16
        fmt = ".1f"
    elif triage_group:
        palette_dict = {"General": c_blue, "Triage (1)": c_teal, "No Triage (0)": c_orange}
        hue_col = "Triage Group"
        label_size = 18
        fmt = ".2f"
    else:
        palette_dict = None
        hue_col = None
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
                legend=(sex_group and i == 0)
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
        axes[i].set_ylim(0, 120)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Value (%)", fontsize=15)
        axes[i].tick_params(axis='both', labelsize=14)

        if not triage_group:
            # Single "General" category: the x-tick label is redundant
            # (sex_group=True already clarifies via the legend).
            axes[i].set_xticks([])

        if sex_group and i == 0:
            axes[i].legend(title="Sex Subgroup", loc='upper right', frameon=True,
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
        path = os.path.join(figures_dir, 'metrics_barplots.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {path}")
    plt.show()

    # --- BLOCK: Combined Single-Plot Visualization (sex_group=False only) ---
    # Aggregates all 8 metrics into a single, publication-ready bar chart.
    if not sex_group:
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
        ax2.set_ylim(0, 120)
        ax2.tick_params(axis='both', labelsize=14)
        plt.setp(ax2.get_xticklabels(), rotation=0, ha='center')

        plt.tight_layout()
        if figures_dir:
            os.makedirs(figures_dir, exist_ok=True)
            path2 = os.path.join(figures_dir, 'metrics_barplot_combined.png')
            fig2.savefig(path2, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {path2}")
        plt.show()

    return results_df

def analyze_emergency_performance(demand_type, y_real_col, y_pred_col, sex_group):
    """
    High-level function to load data and call evaluation.
    This function no longer returns specific objects to avoid redundant printing.
    """
    # Load data
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
    evaluate_diagnostic_performance(df, y_real_col, y_pred_col, sex_group)

# Example usage in a cell:
# analyze_emergency_performance("Non-traumatic Chest Pain")