# experiments/

Benchmark scripts and run artifacts for the stroke (ictus, `demand_type_1==54`)
predictive model, exploring whether a nested-CV train/test strategy can beat
the reference paper (`ref/Paper modelo predictivo - ictus.md`).

## Scripts

| Script | Purpose |
|---|---|
| `run_stroke_general_constrained.py` | **Current/active strategy.** AP screening → AP HPO → recall-constrained OOF thresholding. See below. |
| `run_stroke_general_ref03.py` | Shared data-loading (`prepare_data()`) + a simpler ARAI-03-style benchmark: one 80/20 split, 6 class-imbalance-aware classifiers, randomized CV tuning, OOF threshold, single test evaluation. `run_stroke_general_constrained.py` imports `prepare_data()` from here. |
| `arai03_like_stroke.py` | Small standalone benchmark mirroring the reference workflow (fixed 80/20 split, explicit model list, stratified OOF thresholding). |
| `run_stroke_03.py` | Runs the original 03_04 notebook modelling strategy (`ml.run_binary_automl_model` / FLAML AutoML) and persists metrics as JSON. |
| `report_stroke_general_results.py` | Prints the completed benchmark table from a run's checkpoint (`results.json`). |
| `monitor_stroke_general_cyberpunk.py` | Terminal live monitor (ANSI colors) for a run in progress — reads `training.log` / `state.json` every 5s. |

## Run directories

| Directory | Produced by | Status |
|---|---|---|
| `stroke_03_runs/` | `run_stroke_03.py` | `status: "error"` — `ModuleNotFoundError: seaborn` |
| `stroke_03_ref_runs/general/`, `general_f1/` | `run_stroke_general_ref03.py` | Left mid-run (`status: "running"`, only Logistic Regression + Elastic Net completed) |
| `stroke_03_constrained_runs/general/` | `run_stroke_general_constrained.py` | **Active** — current results live here (see Cohort fix below) |
| `stroke_03_constrained_runs/general_backup_20260811_101418/` | manual backup | Snapshot of the constrained run *before* the cohort fix (N=42,062, winner XGBoost) — kept for comparison, not reproducible going forward |

## Train/test strategy (`run_stroke_general_constrained.py`)

Nested cross-validation, not a plain train/test split — the held-out test set
is touched exactly once, at the very end.

1. **Holdout split (frozen once).** `train_test_split(test_size=0.20, random_state=42, stratify=y)` on the prepared cohort. Stratified on the target so P1 prevalence matches between train/test. Saved to `split.npz` so `--resume` reuses the same split. The 20% test set is never touched again until step 5.
2. **Phase 1 — screening (3-fold, on the 80% train only).** `StratifiedKFold(3, shuffle=True, random_state=42)` OOF Average Precision for all 7 candidate models (XGBoost, Random Forest, Logistic Regression, Elastic Net, Extra Trees, LightGBM, HistGradientBoosting). Top `--survivors` (default 4) by AP move on.
3. **Phase 2 — HPO (5-fold, survivors only).** `StratifiedKFold(5, shuffle=True, random_state=42)`. `ParameterSampler` random search, `--hpo-candidates` (default 12) draws per model, each scored by OOF Average Precision.
4. **Phase 3 — threshold selection (10-fold OOF, on train only).** With the best hyperparameters found, `StratifiedKFold(10)` produces OOF probabilities on the 80% train. The decision threshold is swept to **maximise specificity subject to recall ≥ 0.80** (`max_oof_specificity_at_recall_0.80` — the primary operating point; other points like best-F2 are also recorded). Threshold is fixed here, using train data only.
5. **Final refit + test evaluation.** The winning model+threshold is refit on the *entire* 80% train, then evaluated **once** on the untouched 20% test set — those are the reported test metrics.

Ranking of survivors: feasibility (recall ≥ 0.80 achieved) → specificity → Average Precision → fewer false positives (`main()` in `run_stroke_general_constrained.py:610-614`).

Everything (model selection, HPO, threshold choice) happens strictly inside
the 80% train split; the 20% test set is only used for the final, single,
honest read-out — no leakage of the threshold into test.

## Cohort fix (2026-08-11)

The constrained run originally reproduced the paper's recall (~79.5% vs
79.6%) but with much lower specificity/AUROC (34.0% vs 42.6%, 0.627 vs 0.670).
Root cause: the cohort didn't match the paper's.

- **Wrong outcome column.** Paper/current ground truth uses `p1_real_emerg_bps`, not `p1_real_emerg`. `p1_real_emerg` has 4,091 rows with unresolved (NaN) outcome that `data_filtering()` silently drops; `p1_real_emerg_bps` is fully resolved for all 46,189 stroke demands.
- **No age filter.** The paper cohort has no `age >= 18` filter; the experiment scripts had one.
- **Missing triage columns.** `data_load_col_selection()` in `kbase/modeling.py` selects triage-question columns via `col.startswith('q')`, but in every available `df_stroke_preprocessed_*` snapshot the stroke (demand 54) triage columns are stored bare (`1a`, `1b`, `2a`, ... `6b`, not `q1_a`...). The filter silently drops all 14 of them. This is a bug in shared code (`kbase/modeling.py`), also affecting protocols 16, 23, and cardiac_arrest — **not yet fixed there**, only patched locally.

Fix applied, scoped to `experiments/run_stroke_general_ref03.py::prepare_data()`
only (not in `kbase/modeling.py`):
- `target_column="p1_real_emerg_bps"`, `min_age=None`
- Re-attach the 14 bare triage columns by name, read directly from the source parquet and aligned on the filtered row index

Verified result: N=46,189, class distribution 32,144 / 14,045 — identical to
Álvaro's reference numbers. Feature count is 181 vs his 180 (one dummy-encoding
artifact from `sex`, no information lost).

Re-run with the corrected cohort: winner **LightGBM** (previously XGBoost on
the uncorrected N=42,062 cohort — see the `general_backup_20260811_101418/`
snapshot for those old numbers).

## Live dashboard

Two pieces, both process this run's live JSON/log files:

1. **HTTP server** (`triaje-stroke-dashboard.service`, transient systemd user unit): `python3 -m http.server 8789 --directory experiments/stroke_03_constrained_runs/general`, serving `dashboard.html` at `http://10.243.128.221:8789/dashboard.html`.
2. **`dashboard.html`** — self-contained HTML/JS/canvas page, polls every 1s:
   - `state.json` — phase, active model/candidate/fold, screening ranking, survivors, per-model HPO history, completed models with OOF/test metrics, `winner`
   - `parallel_progress.json` — per-CV-fold worker heartbeats (written by `write_worker_progress()` in the training script) — driven by `n_jobs`
   - `runtime_status.json` — PID/CPU%/RSS of the training process, written by an ad-hoc `systemd-run --user` watcher loop (not a persisted script) polling `/proc` every 5s; stopped once the run finishes
   - `training.log` — tail of stdout/stderr for the "latest log lines" panel

   On completion (`state.json.status in {"ok","error"}` or `phase=="finished"`):
   status pill flips to `FINISHED`/`FAILED`, elapsed time freezes at
   `updated_at - started` instead of counting up against "now", the CV-worker
   panel switches to a static "RUN FINISHED" card, and the 1s poll loop calls
   `clearInterval` so it stops fetching. The results table ranks models with
   🏆/🥈/🥉 for the top 3 (by the same feasibility → specificity → AP → FP
   ordering as `main()`), plain numbers below.

The training job itself also runs as a transient systemd user unit
(`systemd-run --user ... python3 experiments/run_stroke_general_constrained.py`)
so it survives SSH disconnects.
