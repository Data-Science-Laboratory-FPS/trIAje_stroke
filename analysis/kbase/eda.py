import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from IPython.display import display

from kbase.config import settings

def evaluate_diagnostic_performance(df, y_real_col, y_pred_col):
    """
    Core function to calculate metrics and plot results. 
    It displays the results table and plots directly.
    """
    
    # --- BLOCK: Internal Metric Calculation Helper ---
    def calculate_metrics(data, label):
        if len(data) == 0:
            return None
        
        y_real = data[y_real_col]
        y_pred = data[y_pred_col]
        
        # Confusion matrix components
        cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Metrics calculation as percentages
        accuracy = accuracy_score(y_real, y_pred) * 100
        precision = (tp / (tp + fp) if (tp + fp) > 0 else 0) * 100
        recall = (tp / (tp + fn) if (tp + fn) > 0 else 0) * 100
        specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0) * 100
        npv = (tn / (tn + fn) if (tn + fn) > 0 else 0) * 100
        f1 = f1_score(y_real, y_pred) * 100
        
        # Triage specific errors
        overtriage = 100 - precision
        undertriage = 100 - npv
        
        return {
            "Group": label,
            "Accuracy (%)": round(accuracy, 2),
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1-Score (%)": round(f1, 2),
            "Overtriage (%)": round(overtriage, 2),
            "Undertriage (%)": round(undertriage, 2),
            "Specificity (%)": round(specificity, 2),
            "NPV (%)": round(npv, 2),
            "N": len(data)
        }

    # --- BLOCK: Group Segmentation ---
    results = []
    
    # General Performance
    res_gen = calculate_metrics(df, "General")
    if res_gen: results.append(res_gen)
    
    # Triage Performance (triage == 1)
    # The column name is now hardcoded as 'triage'
    if 'triage' in df.columns:
        res_triage = calculate_metrics(df[df['triage'] == 1], "Triage (1)")
        if res_triage: results.append(res_triage)
        
        # No Triage Performance (triage == 0)
        res_no_triage = calculate_metrics(df[df['triage'] == 0], "No Triage (0)")
        if res_no_triage: results.append(res_no_triage)
    
    results_df = pd.DataFrame(results)
    
    # Display the results table in a pretty format for Jupyter
    display(results_df)
    
    # --- BLOCK: Visualization (2x3 Grid) ---
    metrics_to_plot = [
        "Accuracy (%)", "Precision (%)", "Recall (%)", 
        "F1-Score (%)", "Overtriage (%)", "Undertriage (%)"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")
    palette = ["#4A90E2", "#50E3C2", "#F5A623"]

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(
            data=results_df, 
            x="Group", 
            y=metric, 
            ax=axes[i], 
            palette=palette,
            hue="Group",
            legend=False
        )
        axes[i].set_title(f"{metric}", fontweight='bold', fontsize=12)
        axes[i].set_ylim(0, 110)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Value (%)")
        
        # Annotation labels
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.2f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 8), 
                            textcoords = 'offset points',
                            fontsize=10)

    plt.tight_layout()
    plt.show()

def analyze_emergency_performance(demand_type):
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
    df = df.dropna(subset=['p1_real_emerg'])

    print(f"\nNumber of rows in {demand_type} is: {len(df):,}\n")
    
    # Preprocessing
    df = df.copy()
    if 'p1_predicted' in df.columns and df['p1_predicted'].dtype.name == 'category':
        df['p1_predicted'] = pd.to_numeric(df['p1_predicted'], errors='coerce')
        
    df['p1_predicted'] = df['p1_predicted'].astype('int8')
    df['p1_real_emerg'] = df['p1_real_emerg'].astype('int8')
    
    # Execute evaluation and plotting
    evaluate_diagnostic_performance(df, 'p1_real_emerg', 'p1_predicted')

# Example usage in a cell:
# analyze_emergency_performance("Non-traumatic Chest Pain")