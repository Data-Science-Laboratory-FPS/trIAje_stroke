# Dashboard del pipeline

El dashboard es un HTML estático con JavaScript. No entrena modelos ni modifica
archivos. Solo lee por HTTP los JSON y logs escritos por el pipeline dentro del
directorio servido.

## Archivo reutilizable

Plantilla:

```text
experiments/dashboard_template.html
```

Para usarla en un run:

```bash
cp experiments/dashboard_template.html experiments/<run_dir>/dashboard.html
python3 -m http.server 8789 --directory experiments/<run_dir>
```

Abrir:

```text
http://localhost:8789/dashboard.html
```

## Archivos que consume

| Archivo | Obligatorio | Escrito por | Qué muestra |
|---|---:|---|---|
| `state.json` | Sí | `run_stroke_general_constrained.py` | Estado global, fase, modelo activo, ranking, supervivientes, HPO, modelos completados y ganador. |
| `parallel_progress.json` | No | Workers de CV | Estado por fold/proceso durante screening, HPO y OOF. |
| `runtime_status.json` | No | Watcher externo | PID, CPU, memoria RSS y estado del proceso Python. |
| `training.log` | No | Redirección stdout/stderr | Últimas líneas de log. |
| `results.json` | No | Final del pipeline | Resultado final persistido; la tabla suele salir de `state.json` durante y al final del run. |

Si faltan archivos opcionales, el dashboard debe seguir cargando, pero algunos
paneles aparecerán vacíos o como estado no disponible.

## Contrato mínimo de `state.json`

Campos principales usados por el dashboard:

```json
{
  "status": "running | ok | error | interrupted_or_error",
  "started": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "phase": "texto de fase actual",
  "active_model": "LightGBM",
  "active_candidate": 4,
  "active_total": 12,
  "active_fold": 3,
  "active_total_folds": 5,
  "screening_ranking": [],
  "survivor_models": [],
  "completed_models": {},
  "failed_models": {},
  "winner": "LightGBM"
}
```

Cada entrada de `completed_models` debe seguir la estructura que produce el
pipeline:

```json
{
  "LightGBM": {
    "selected_threshold": 0.2617,
    "threshold_source": "train_10fold_oof_specificity_at_recall_0.80",
    "oof_metrics": {
      "recall": 0.80,
      "specificity": 0.43,
      "average_precision": 0.45,
      "fp": 14453
    },
    "test_metrics": {
      "recall": 0.79,
      "specificity": 0.43,
      "average_precision": 0.45,
      "auroc": 0.67,
      "tp": 2231,
      "tn": 2778,
      "fp": 3651,
      "fn": 578
    }
  }
}
```

## Contrato mínimo de `parallel_progress.json`

Formato:

```json
{
  "pid-123-fold-1": {
    "pid": 123,
    "fold": 1,
    "phase": "screening | hpo | oof",
    "model": "LightGBM",
    "status": "running | done | failed",
    "started": 1780000000.0,
    "elapsed": 12.5
  }
}
```

El pipeline lo escribe con lock de fichero para evitar corrupciones cuando
varios workers actualizan a la vez.

## Watcher de runtime opcional

El dashboard puede leer `runtime_status.json`, pero el pipeline no lo necesita.
Para otro proyecto se puede generar con cualquier proceso externo que escriba:

```json
{
  "pid": 12345,
  "cpu_percent": 350.0,
  "rss_mb": 2048.5,
  "status": "running",
  "updated_at": "ISO-8601 timestamp"
}
```

Si no se quiere monitorizar CPU/memoria, no hace falta crear este archivo.

## Ciclo de vida

1. Al inicio, el dashboard intenta leer `state.json`.
2. Mientras `status` sea de ejecución, refresca cada segundo.
3. Los paneles de workers salen de `parallel_progress.json`.
4. El log muestra el tail de `training.log`.
5. Cuando `state.json.status` es `ok` o `error`, o `phase == "finished"`, el
   dashboard congela el estado y deja de hacer polling.

## Archivos necesarios para portarlo

Mínimo:

```text
dashboard.html
state.json
```

Recomendado:

```text
dashboard.html
state.json
parallel_progress.json
training.log
results.json
runtime_status.json
```

Para usarlo con otro pipeline, mantener los nombres de archivo y el contrato de
campos anteriores, o adaptar el JavaScript del HTML.
