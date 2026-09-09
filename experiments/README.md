# experiments/

Este directorio queda organizado alrededor de una única estrategia activa y
reutilizable: validación anidada con selección de modelo, ajuste de
hiperparámetros y selección de umbral dentro de train, dejando el test como
evaluación ciega final.

## Qué usar

Archivos necesarios para correr el pipeline en este proyecto:

| Archivo/directorio | Papel |
|---|---|
| `run_stroke_general_constrained.py` | Motor del pipeline: CV anidada, HPO, selección de umbral, refit final, métricas y checkpoints. |
| `stroke_data_adapter.py` | Adaptador específico del dataset stroke. Define `configure_environment()` y `prepare_data()`. Este es el archivo que se cambia al portar a otro proyecto. |
| `dashboard_template.html` | Plantilla reutilizable del dashboard. Copiarla como `dashboard.html` dentro del `RUN_DIR` que vaya a servirse. |
| `requirements_stroke_challengers.txt` | Dependencias específicas de estos experimentos. |
| `stroke_03_constrained_runs/general/` | Run activo actual: `results.json`, `state.json`, `split.npz`, modelos `.joblib`, logs y dashboard servido. |

Todo lo que no forma parte de esta ruta activa está bajo `deprecated/`.

## Documentación

| Documento | Contenido |
|---|---|
| `docs/METHODOLOGY.md` | Explicación completa de la metodología, fases, criterios de ranking y garantía anti-leakage. |
| `docs/DASHBOARD.md` | Cómo funciona el dashboard, qué JSON consume, cómo servirlo y qué archivos son obligatorios/opcionales. |
| `docs/PORTING.md` | Lista concreta de cambios para reutilizar el pipeline en otro dataset/proyecto. |
| `deprecated/README.md` | Inventario de scripts y runs antiguos, con motivo de deprecación. |

## Resultado actual de stroke

Run activo: `stroke_03_constrained_runs/general/`

- Ganador: `LightGBM`
- Modelo serializado: `stroke_03_constrained_runs/general/lightgbm.joblib`
- Resultados: `stroke_03_constrained_runs/general/results.json`
- Umbral elegido en train OOF: `0.2617584594472823`
- Métricas de test con ese umbral: recall `0.7942`, specificity `0.4321`, PPV `0.3793`, NPV `0.8278`, F1 `0.5134`, F2 `0.6517`, Average Precision `0.4511`, AUROC `0.6723`
- Matriz de test: `TP=2231`, `TN=2778`, `FP=3651`, `FN=578`

## Ejecución

Desde la raíz del repo:

```bash
cd /home/marmengol/projects/trIAje_code
python3 experiments/run_stroke_general_constrained.py --resume
```

Opciones principales:

```bash
python3 experiments/run_stroke_general_constrained.py \
  --screening-folds 3 \
  --hpo-folds 5 \
  --oof-folds 10 \
  --hpo-candidates 12 \
  --survivors 4 \
  --n-jobs 4 \
  --run-dir experiments/stroke_03_constrained_runs/general \
  --resume
```

Si falta `dashboard.html` en el directorio del run, el script lo copia desde
`dashboard_template.html`. Para un nuevo proyecto, usar siempre un `--run-dir`
nuevo para no mezclar splits, checkpoints ni resultados.

## Dashboard

Para servir el dashboard del run actual:

```bash
cd /home/marmengol/projects/trIAje_code
python3 -m http.server 8789 --directory experiments/stroke_03_constrained_runs/general
```

Abrir:

```text
http://localhost:8789/dashboard.html
```

El dashboard no ejecuta entrenamiento. Solo lee archivos estáticos/JSON del
directorio servido. Detalles en `docs/DASHBOARD.md`.

## Estructura actual

```text
experiments/
  README.md
  dashboard_template.html
  docs/
    DASHBOARD.md
    METHODOLOGY.md
    PORTING.md
  run_stroke_general_constrained.py
  stroke_data_adapter.py
  requirements_stroke_challengers.txt
  stroke_03_constrained_runs/
    general/
      dashboard.html
      results.json
      state.json
      split.npz
      *.joblib
      *.log
  deprecated/
    README.md
    scripts/
    runs/
    constrained_backups/
```
