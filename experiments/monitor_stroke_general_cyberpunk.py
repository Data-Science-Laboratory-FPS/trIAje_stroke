#!/usr/bin/env python3
"""ANSI cyberpunk monitor for the resumable stroke-general benchmark."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "stroke_03_ref_runs" / "general"
LOG = RUN / "training.log"
STATE = RUN / "state.json"
TOTAL_MODELS = 6
TOTAL_FITS = 200

RESET = "\033[0m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
DIM = "\033[2m"
WHITE = "\033[97m"


def clear():
    print("\033[2J\033[H", end="")


def fmt(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def bar(value, total, width=42):
    value = max(0, min(total, value))
    filled = int(width * value / max(total, 1))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def read_status():
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        state = {"completed_models": {}, "status": "starting"}
    try:
        log = LOG.read_text(encoding="utf-8", errors="replace")
    except Exception:
        log = ""

    markers = list(re.finditer(r"\n\[([^\]]+)\] RandomizedSearchCV n_iter=(\d+)", log))
    current = markers[-1].group(1) if markers else "preparing data"
    n_iter = int(markers[-1].group(2)) if markers else 20
    segment = log[markers[-1].start():] if markers else ""
    fits = len(re.findall(r"\[CV\] END", segment))
    phase = "model tuning" if markers else "data preparation"
    if markers and fits >= n_iter * 10:
        phase = "OOF threshold / test metrics"
    events = [line.strip() for line in log.splitlines() if line.strip()]
    last_event = events[-1] if events else "waiting for log"
    return state, current, phase, min(fits, TOTAL_FITS), n_iter, last_event


def render():
    state, current, phase, fits, n_iter, last_event = read_status()
    completed = state.get("completed_models", {})
    total_elapsed = sum(float(row.get("elapsed_seconds", 0)) for row in completed.values())
    try:
        started_at = datetime.fromisoformat(state.get("started", "")).timestamp()
        live_elapsed = max(0, time.time() - started_at)
    except (TypeError, ValueError):
        live_elapsed = total_elapsed
    active_elapsed = max(1, live_elapsed - total_elapsed)
    fit_rate = fits / active_elapsed if fits else 0
    current_remaining = (TOTAL_FITS - fits) / fit_rate if fit_rate else 0
    future_models = max(0, TOTAL_MODELS - len(completed) - 1)
    fallback_per_model = total_elapsed / len(completed) if completed else 0
    future_per_model = TOTAL_FITS / fit_rate if fit_rate else fallback_per_model
    eta = current_remaining + future_models * future_per_model
    status = state.get("status", "running")
    status_color = GREEN if status == "ok" else YELLOW if status == "running" else RED

    clear()
    print(f"{CYAN}{'=' * 78}{RESET}")
    print(f"{MAGENTA}   T R I A J E   //   S T R O K E   G E N E R A L   //   0 3 R E F{RESET}")
    print(f"{CYAN}{'=' * 78}{RESET}")
    print(f"  {WHITE}STATUS{RESET} {status_color}{status.upper():<14}{RESET}   {DIM}protocol: sklearn + 10-fold CV + OOF threshold{RESET}")
    print()
    print(f"  {MAGENTA}GLOBAL{RESET}  {CYAN}{bar(len(completed), TOTAL_MODELS)}{RESET}  {len(completed)}/{TOTAL_MODELS} models")
    print(f"           live elapsed {YELLOW}{fmt(live_elapsed)}{RESET}   ETA {YELLOW}{fmt(eta) if eta else 'calculating'}{RESET}")
    print()
    print(f"  {MAGENTA}ACTIVE{RESET}  {WHITE}{current}{RESET}")
    print(f"           phase: {CYAN}{phase}{RESET}")
    print(f"           {CYAN}{bar(fits, TOTAL_FITS, 34)}{RESET}  {fits}/{TOTAL_FITS} CV fits  |  {fit_rate:.2f} fits/s  |  active left {fmt(current_remaining)}")
    print(f"           future estimate: {fmt(future_per_model)} per model  |  ETA uses current fit rate")
    print()
    print(f"  {MAGENTA}MODEL METRICS{RESET}")
    print(f"  {'MODEL':<24} {'CV-AUROC':>9} {'TEST-AUROC':>11} {'TEST-AP':>9} {'F1':>8} {'RECALL':>8}")
    print(f"  {'-' * 72}")
    for name, row in completed.items():
        test = row.get("test_metrics", {})
        print(
            f"  {GREEN}{name:<24}{RESET} "
            f"{row.get('best_cv_auroc', 0):>9.4f} "
            f"{test.get('auroc', 0):>11.4f} "
            f"{test.get('average_precision', 0):>9.4f} "
            f"{test.get('f1', 0):>8.4f} "
            f"{test.get('recall', 0):>8.4f}"
        )
        print(
            f"  {DIM}  precision={test.get('precision', 0):.4f} "
            f"specificity={test.get('specificity', 0):.4f} "
            f"TP={test.get('tp', 0)} TN={test.get('tn', 0)} "
            f"FP={test.get('fp', 0)} FN={test.get('fn', 0)} "
            f"threshold={row.get('oof_threshold', 0):.4f}{RESET}"
        )
    print()
    print(f"  {MAGENTA}LAST EVENT{RESET} {DIM}{last_event[-180:]}{RESET}")
    print(f"  {BLUE}refresh: 5s | Ctrl-C to close monitor; training continues independently{RESET}")


def main():
    try:
        while True:
            render()
            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\n{CYAN}monitor closed; training was not stopped{RESET}")


if __name__ == "__main__":
    main()
