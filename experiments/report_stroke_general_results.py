#!/usr/bin/env python3
"""Print the completed stroke-general benchmark table from its checkpoint."""

import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="general_f1")
    args = parser.parse_args()
    run = Path(__file__).resolve().parents[1] / "experiments" / "stroke_03_ref_runs" / args.run_dir
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    rows = state.get("completed_models", {})
    print("\nRESULTS FINAL - STROKE GENERAL - 03REF - OPTIMISE F1")
    print("=" * 190)
    print(
        f"{'Model':<24} {'CV-F1':>9} {'Test AUROC':>11} {'Test AP':>9} "
        f"{'LogLoss':>8} {'Brier':>8} {'Thr':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} "
        f"{'Spec':>8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}"
    )
    print("-" * 190)
    ranked = sorted(rows.items(), key=lambda item: item[1].get("test_metrics", {}).get("f1", -1), reverse=True)
    for name, row in ranked:
        m = row.get("test_metrics", {})
        print(
            f"{name:<24} {row.get('best_cv_f1', 0):>9.4f} "
            f"{m.get('auroc', 0):>11.4f} {m.get('average_precision', 0):>9.4f} "
            f"{m.get('log_loss', 0):>8.4f} {m.get('brier', 0):>8.4f} {row.get('oof_threshold', 0):>8.4f} "
            f"{m.get('f1', 0):>8.4f} {m.get('precision', 0):>8.4f} "
            f"{m.get('recall', 0):>8.4f} {m.get('specificity', 0):>8.4f} "
            f"{m.get('tp', 0):>6} {m.get('tn', 0):>6} {m.get('fp', 0):>6} {m.get('fn', 0):>6}"
        )
    if ranked:
        print("-" * 190)
        print(f"Winner by held-out test F1: {ranked[0][0]}")
    print(f"Status: {state.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
